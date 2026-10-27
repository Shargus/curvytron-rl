"""Self-play PPO for Curvytron on the fast numba simulator (curvysim).

Each environment runs one round with 2..P players. A share of rounds is pure
self-play (every player is the current bot and produces training data). In the
other rounds the current bot controls a few players and each remaining player is
a past version of the bot (pool snapshot, sampled with prioritized fictitious
self-play), the lookahead heuristic, or a human-like player: the heuristic or a
snapshot with human reaction times, steering noise and confusion after maluses
(see curvysim/env.py). The bot observes which opponents are human-like, so it can
learn a different style against humans (e.g. using enemy maluses more).

League (AlphaStar style): some rounds put one or more exploiters in the game, policies that
train only against the current main agent; their frozen copies join the pool. Some rounds
start as a trap: the learner is enclosed by a trail whose only ways out are holes. Rewards
add potential-based territory shaping (share of the board reached first).

Examples:
    # laptop / CPU pilot
    uv run python train_selfplay.py --run-name pilot --hidden 256,256 --num-envs 256 --threads 6
    # GPU server: many envs, bigger net, CNNs over the local and whole-board maps
    uv run python train_selfplay.py --run-name big --device cuda --num-envs 1024 --threads 32 \\
        --map-size 48 --global-map --hidden 1024,512 --minibatch 32768 --total-steps 3e9
"""
import argparse
import json
import os
import random
import signal
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tensorboard.compat.proto.summary_pb2 import Summary

from curvysim import BatchEnv, OBS_DIM
from curvysim.core import B_AFFECT
from curvysim.obs import O_TERR
from curvysim.policy import ActorCritic, save_policy, load_policy
from curvysim.render import render, encode_gif

SELF, HEURISTIC = -1, -2   # slot controllers: main agent, heuristic; pool snapshots use their id (>= 0)
EXPLOITER = -3             # exploiter k controls its slots with code EXPLOITER - k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', default='selfplay')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--max-players', type=int, default=13)
    p.add_argument('--n-probs', default='0.12,0.06,0.09,0.09,0.09,0.09,0.08,0.08,0.08,0.08,0.07,0.07',
                   help='relative probabilities of 2,3,..,P players per round')
    p.add_argument('--bonus-modes', default='0.3,0.4,0.3', help='P(no bonus), P(all), P(random subset)')
    p.add_argument('--frame-skip', type=int, default=3, help='server frames per decision (3 = 50 ms)')
    p.add_argument('--delay', default='0,8', help='round-trip latency range in frames (1 frame = 16.7 ms)')
    p.add_argument('--kill-reward', default='0,1',
                   help='range of the per-kill bonus ("aggression"), sampled per player and round and '
                        'observed by the bot, so the play style can be chosen when playing')
    p.add_argument('--eval-aggression', type=float, default=0.2,
                   help='aggression the bot is given in evaluations and videos')
    p.add_argument('--map-size', type=int, default=0, help='egocentric map cells per side (0 = no CNN)')
    p.add_argument('--global-map', action='store_true', help='add the whole-board map (CNN input)')
    p.add_argument('--total-steps', type=float, default=2e8, help='learner transitions')
    p.add_argument('--rollout', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.998, help='0.998 at 20 decisions/s ~ 25 s horizon')
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--ent', type=float, default=0.01)
    p.add_argument('--ent-final', type=float, default=0.002)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--minibatch', type=int, default=8192)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--rollout-on-cpu', action='store_true',
                   help='keep the rollout in CPU RAM and send it per minibatch: slower updates, but GPU '
                        'memory then depends on the minibatch size only (for GPUs with ~20 GB)')
    p.add_argument('--hidden', default='512,512,256')
    p.add_argument('--p-selfplay', type=float, default=0.35,
                   help='share of rounds in which every player is the current bot')
    p.add_argument('--max-learner-share', type=float, default=0.34,
                   help='in mixed rounds, the current bot controls 1 up to this share of the players')
    p.add_argument('--p-human', type=float, default=0.45,
                   help='in mixed rounds, share of the other players that are human-like')
    p.add_argument('--p-heuristic', type=float, default=0.2,
                   help='in mixed rounds, share of the other players that are the heuristic '
                        '(the rest are past versions of the bot)')
    p.add_argument('--human-base-heuristic', type=float, default=0.3,
                   help='share of human-like players driven by the heuristic (the rest by past versions)')
    p.add_argument('--human-delay', default='9,18', help='reaction delay of human-like players, frames')
    p.add_argument('--human-noise', type=float, default=0.05,
                   help='per-decision probability that a human-like player steers randomly')
    p.add_argument('--human-confusion', type=float, default=1.0,
                   help='how much maluses confuse human-like players (0 = not at all, 1 = default)')
    p.add_argument('--snapshot-every', type=int, default=25, help='updates between pool snapshots')
    p.add_argument('--pool-size', type=int, default=50)
    p.add_argument('--active-opponents', type=int, default=4)
    p.add_argument('--exploiters', type=int, default=2, help='league exploiters trained alongside (0 = off)')
    p.add_argument('--p-exploit', type=float, default=0.2,
                   help='share of rounds with an exploiter (--p-selfplay applies to the rest)')
    p.add_argument('--exploiter-kill-reward', default='0.6,1.2', help='aggression range of exploiters')
    p.add_argument('--exploiter-target', type=float, default=0.65,
                   help='placement score vs the main agent at which an exploiter joins the pool and restarts')
    p.add_argument('--exploiter-min-age', type=int, default=50, help='updates before an exploiter can graduate')
    p.add_argument('--exploiter-max-age', type=int, default=400, help='updates before an exploiter restarts anyway')
    p.add_argument('--p-trap', type=float, default=0.15, help='share of rounds that start with the learner trapped')
    p.add_argument('--trap-own', type=float, default=0.5, help='share of traps made of the learner\'s own trail')
    p.add_argument('--territory-coef', type=float, default=0.5,
                   help='potential-based shaping on the share of board reached first (0 = off)')
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-rounds', type=int, default=300)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--checkpoint-every', type=int, default=25,
                   help='updates between full checkpoints (state.pt); a run with one resumes from it')
    p.add_argument('--log-every', type=float, default=60.0, help='seconds between progress reports')
    p.add_argument('--video-every', type=int, default=None,
                   help='updates between sample videos (default: --eval-every, 0 = off)')
    p.add_argument('--video-players', type=int, default=6)
    p.add_argument('--video-opponent', default='mixed', choices=['mixed', 'self', 'heuristic', 'human'],
                   help='who controls the other avatars in the video (mixed: like a mixed training round, '
                        'with an exploiter; human: the bot with human-like handicaps)')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', default=None,
                   help='policy to initialise a new run from (ignored when the run has a state.pt)')
    return p.parse_args()


def to_inputs(model, env, sel):
    """Gather the rows selected by a boolean mask [E, P] as model input tensors."""
    dev = next(model.parameters()).device
    obs = torch.from_numpy(env.obs[sel]).to(dev)
    maps = torch.from_numpy(env.maps[sel]).to(dev) if model.map_size > 0 else None
    gmaps = torch.from_numpy(env.gmaps[sel]).to(dev) if model.global_map else None
    return obs, maps, gmaps


class DeviceInputs:
    """Observation arrays of every slot on the policy device, uploaded once per rollout step;
    each policy then selects its rows there. Gathering rows on the CPU instead costs about as
    much as simulating. On CUDA the env writes its observations straight into pinned memory,
    so the upload is one asynchronous DMA copy per array."""

    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.cuda = device.type == 'cuda'
        if self.cuda:
            pin = lambda a: None if a is None else torch.from_numpy(a).pin_memory().numpy()
            env.obs, env.maps, env.gmaps = pin(env.obs), pin(env.maps), pin(env.gmaps)
        self.refresh()

    def refresh(self):
        """Upload the current observations; call it after anything changed them."""
        up = lambda a: None if a is None else torch.from_numpy(a).to(self.device, non_blocking=self.cuda)
        self.obs, self.maps, self.gmaps = up(self.env.obs), up(self.env.maps), up(self.env.gmaps)

    def wait(self):
        """Block until the upload is done: the env must not overwrite pinned arrays still in flight."""
        if self.cuda:
            torch.cuda.current_stream(self.device).synchronize()

    def rows(self, model, mask):
        m = torch.from_numpy(mask).to(self.device)
        return (self.obs[m], self.maps[m] if model.map_size > 0 else None,
                self.gmaps[m] if model.global_map else None)


@torch.no_grad()
def policy_moves(model, env, sel, greedy=False):
    if not sel.any():
        return np.zeros(0, np.int64)
    return model.act(*to_inputs(model, env, sel), greedy=greedy).cpu().numpy() - 1


# bonus types grouped by who they affect (for logs): enemy maluses, self bonuses, game/all
BONUS_GROUPS = {'enemy': np.array(B_AFFECT) == 1, 'self': np.array(B_AFFECT) == 0,
                'game': (np.array(B_AFFECT) == 2) | (np.array(B_AFFECT) == 3)}


@torch.no_grad()
def evaluate(model, opponent, rounds, seed, n_players=2, bonus_modes=(0.3, 0.4, 0.3), max_envs=512,
             delay=(0, 8), frame_skip=3, aggression=0.2, human_args=None, stats=None):
    """Results of ``model`` (one random slot) vs ``opponent`` (all other slots).

    opponent: an ActorCritic, 'heuristic', or 'human' (the heuristic with human-like handicaps,
    see BatchEnv). Returns (mean return, win rate), both from placement only (the kill bonus is
    not counted). Every player gets ``aggression``. If ``stats`` is a dict, it receives the mean
    number of bonuses per round the model picked up, by group.
    Up to ``max_envs`` rounds run at once and every env counts the same number of rounds, its
    first ones: counting rounds in the order they end would favour the short ones.
    """
    probs = [0.0] * (n_players - 1)
    probs[-1] = 1.0
    map_size = max(model.map_size, getattr(opponent, 'map_size', 0))
    global_map = model.global_map or getattr(opponent, 'global_map', False)
    num_envs = min(rounds, max_envs)
    per_env = -(-rounds // num_envs)
    env = BatchEnv(num_envs, max_players=n_players, n_probs=probs, bonus_modes=bonus_modes,
                   seed=seed, map_size=map_size, global_map=global_map, delay=delay,
                   frame_skip=frame_skip, kill_reward=(aggression, aggression), **(human_args or {}))
    env.reset()
    ret = np.zeros(num_envs)
    done_rounds = np.zeros(num_envs, np.int64)
    results = []
    picks = []
    rng = np.random.default_rng(seed)
    # the server updates avatars in reverse join order, which matters in head-to-head
    # encounters, so the evaluated policy plays from a random slot
    my_slot = rng.integers(0, n_players, num_envs)
    slots = np.arange(env.P)[None, :]

    def setup(envs):
        if opponent == 'human':
            for e in envs:
                for p in range(n_players):
                    if p != my_slot[e]:
                        env.make_human(e, p)
            env.refresh(envs)   # the human-like flags are observed

    setup(range(num_envs))
    while (done_rounds < per_env).any():
        alive = env.alive
        actions = np.zeros((env.E, env.P), np.int64)
        me = alive & (slots == my_slot[:, None])
        opp = alive & ~me
        actions[me] = policy_moves(model, env, me)
        if isinstance(opponent, str):
            actions[opp] = env.heuristic(opp)[opp]
        else:
            actions[opp] = policy_moves(opponent, env, opp)
        _, rew, done, reset = env.step(actions)
        ret += rew[np.arange(env.E), my_slot]
        for e in np.nonzero(done[np.arange(env.E), my_slot])[0]:
            if done_rounds[e] < per_env:
                picks.append(env.pickups[e, my_slot[e]].copy())
        for e in np.nonzero(reset)[0]:
            if done_rounds[e] < per_env:
                results.append(ret[e])
                done_rounds[e] += 1
            ret[e] = 0.0
            my_slot[e] = rng.integers(0, n_players)
        setup(np.nonzero(reset)[0])
    results = np.array(results)
    if stats is not None and picks:
        picks = np.array(picks)
        for g, mask in BONUS_GROUPS.items():
            stats[g] = float(picks[:, mask].sum(1).mean())
    return float(results.mean()), float((results > 0.99).mean())   # win = last survivor


@torch.no_grad()
def record_video(model, n_players, opponent, seed, frame_skip, delay, size_px=400, stride=2, max_seconds=90,
                 aggression=0.2, human_args=None, mix=None):
    """Play one round (all bonuses on) and return (frames, info). The model controls a random
    slot, drawn with a white ring; the others are the model too ('self'), the heuristic, the
    model with human-like handicaps ('human'), or composed like a mixed training round ('mixed':
    ``mix`` gives the exploiter model or None, the past versions and the shares of human-like,
    heuristic and past players). Each head shows its player kind (see render.PLAYER_KINDS)."""
    probs = [0.0] * (n_players - 1)
    probs[-1] = 1.0
    env = BatchEnv(1, max_players=n_players, n_probs=probs, bonus_modes=(0, 1, 0), seed=seed,
                   map_size=model.map_size, global_map=model.global_map, delay=delay, frame_skip=frame_skip,
                   kill_reward=(aggression, aggression), **(human_args or {}))
    env.reset()
    rng = np.random.default_rng(seed)
    me = rng.integers(0, n_players)
    # who drives each slot: a model or 'heuristic', and the letter shown in its head
    drivers = [model] * n_players
    labels = ['B'] * n_players
    if opponent in ('heuristic', 'human'):
        for p in range(n_players):
            if p != me:
                drivers[p] = 'heuristic' if opponent == 'heuristic' else model
                labels[p] = 'L' if opponent == 'heuristic' else 'H'
    elif opponent == 'mixed':
        others = [p for p in rng.permutation(n_players) if p != me]
        # the bot takes 1 up to max_learner_share of the slots, as in training (``me`` is one of them)
        n_learn = int(rng.integers(1, max(1, int(mix['max_learner_share'] * n_players)) + 1))
        rest = others[n_learn - 1:]
        if mix['exploiter'] is not None and rest:
            drivers[rest[0]], labels[rest[0]] = mix['exploiter'], 'E'
            rest = rest[1:]
        past = mix['past']
        for p in rest:
            u = rng.random()
            snapshot = past[rng.integers(len(past))] if past else 'heuristic'
            if u < mix['p_human']:
                drivers[p] = 'heuristic' if rng.random() < mix['human_base_heuristic'] else snapshot
                labels[p] = 'H'
            elif u < mix['p_human'] + mix['p_heuristic']:
                drivers[p], labels[p] = 'heuristic', 'L'
            else:
                drivers[p], labels[p] = snapshot, 'P' if past else 'L'
    for p in range(n_players):
        if labels[p] == 'H':
            env.make_human(0, p)
    env.refresh([0])   # human-like flags are observed
    groups = {}        # driver -> its slots
    for p, d in enumerate(drivers):
        groups.setdefault(id(d), (d, []))[1].append(p)
    slots = np.arange(n_players)[None, :]
    frames = [render(env.S, 0, size_px, me, with_legend=True, labels=labels)]
    ret = 0.0
    kills = 0
    picks = np.zeros(len(B_AFFECT), np.int64)
    step = 0
    max_steps = int(max_seconds * 60 / frame_skip)
    while step < max_steps:
        alive = env.alive
        actions = np.zeros((1, n_players), np.int64)
        for d, ps in groups.values():
            sel = alive & np.isin(slots, ps)
            if sel.any():
                actions[sel] = env.heuristic(sel)[sel] if isinstance(d, str) else policy_moves(d, env, sel)
        _, rew, done, reset = env.step(actions)
        ret += rew[0, me]
        kills += env.kills[0, me]
        if done[0, me]:
            picks = env.pickups[0, me].copy()
        step += 1
        if reset[0]:
            break
        if step % stride == 0:
            frames.append(render(env.S, 0, size_px, me, with_legend=True, labels=labels))
    frames += [frames[-1]] * 10   # hold the last frame for a moment
    placement = int(round((ret + 1) / 2 * (n_players - 1)))   # points: 0 = first out, n-1 = survivor
    info = {'seconds': step * frame_skip / 60, 'rank': n_players - placement, 'fps': 60 / (frame_skip * stride),
            'kills': int(kills), 'bonuses': int(picks.sum()), 'players': ' '.join(sorted(labels))}
    return frames, info


class OpponentPool:
    """Frozen snapshots with prioritized fictitious self-play sampling."""

    def __init__(self, size, n_active, make_model):
        self.size = size
        self.n_active = n_active
        self.make_model = make_model
        self.entries = []      # dicts: id, name, state, wins (EMA of learner score in [0, 1])
        self.active = {}       # id -> model
        self.next_id = 0

    def add(self, model, name='main'):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.entries.append({'id': self.next_id, 'name': name, 'state': state, 'win': 0.5})
        self.next_id += 1
        if len(self.entries) > self.size:   # keep the very first snapshot as an anchor
            self.entries.pop(1)
        self.refresh()

    def weights(self):
        # PFSP "hard" weighting: opponents the learner beats less often get more games
        w = np.array([(1.0 - e['win']) ** 2 + 0.05 for e in self.entries])
        return w / w.sum()

    def refresh(self, ids=None):
        """Load the active opponents: the latest snapshot plus PFSP samples, or the given ids."""
        self.active.clear()
        if ids is not None:
            chosen = {i for i, e in enumerate(self.entries) if e['id'] in ids}
        else:
            k = min(self.n_active, len(self.entries))
            chosen = {len(self.entries) - 1}   # always play the latest snapshot
            w = self.weights()
            while len(chosen) < k:
                chosen.add(int(np.random.choice(len(self.entries), p=w)))
        for i in chosen:
            entry = self.entries[i]
            m = self.make_model()
            m.load_state_dict(entry['state'])
            m.eval()
            self.active[entry['id']] = m

    def state(self):
        return {'entries': self.entries, 'next_id': self.next_id, 'active': list(self.active)}

    def load(self, state):
        self.entries = state['entries']
        self.next_id = state['next_id']
        self.refresh(set(state['active']))

    def record(self, opp_id, learner_score):
        for e in self.entries:
            if e['id'] == opp_id:
                e['win'] = 0.98 * e['win'] + 0.02 * float(learner_score)

    def summary(self, k=3):
        hardest = sorted(self.entries, key=lambda e: e['win'])[:k]
        n_expl = sum(e['name'] != 'main' for e in self.entries)
        return (f"{len(self.entries)} snapshots ({n_expl} from exploiters), hardest for the learner "
                f"(placement score, 0.5 = even): "
                + ', '.join(f"#{e['id']} {e['name']} {e['win']:.2f}" for e in hardest))


class Learner:
    """A policy trained with PPO on the rows of the slots it controls (``key`` in ``slot_ctrl``).

    Observations are stored only for its rows (row-major order of the [E, P] mask), which
    matches ``valid.reshape(-1)`` when concatenated over t."""

    def __init__(self, key, model, lr, T, E, P, store_dev):
        self.key = key
        self.model = model
        self.store_dev = store_dev   # where the rollout inputs are kept until the update
        self.optim = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.obs = [None] * T
        self.map = [None] * T
        self.gmap = [None] * T
        self.act = np.zeros((T, E, P), np.int64)
        self.logp = np.zeros((T, E, P), np.float32)
        self.val = np.zeros((T, E, P), np.float32)
        self.rew = np.zeros((T, E, P), np.float32)
        self.done = np.zeros((T, E, P), np.bool_)
        self.valid = np.zeros((T, E, P), np.bool_)

    def step(self, inputs, mask, t):
        """Sample moves for the rows in ``mask`` (inputs: DeviceInputs) and record them at time ``t``."""
        self.valid[t] = mask
        x = inputs.rows(self.model, mask)
        store = lambda v: None if v is None else v.to(self.store_dev)
        self.obs[t], self.map[t], self.gmap[t] = store(x[0]), store(x[1]), store(x[2])
        if not mask.any():
            return np.zeros(0, np.int64)
        with torch.no_grad():
            logits, value = self.model(*x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            a, logp, value = a.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
        self.act[t][mask] = a
        self.logp[t][mask] = logp
        self.val[t][mask] = value
        return a - 1

    def update(self, inputs, mask, args, ent_coef, device):
        """GAE over the rollout (bootstrapped on the rows in ``mask``), then PPO epochs.
        Returns (transitions used, diagnostics dict or None if there were too few)."""
        model = self.model
        T, E, P = self.valid.shape
        next_val = np.zeros((E, P), np.float32)
        if mask.any():
            with torch.no_grad():
                next_val[mask] = model.value(*inputs.rows(model, mask)).cpu().numpy()
        adv = np.zeros((T, E, P), np.float32)
        lastgae = np.zeros((E, P), np.float32)
        for t in reversed(range(T)):   # per agent stream; gaps after a death are skipped
            v = self.valid[t]
            nonterm = 1.0 - self.done[t]
            delta = self.rew[t] + args.gamma * next_val * nonterm - self.val[t]
            g = delta + args.gamma * args.lam * nonterm * lastgae
            lastgae = np.where(v, g, lastgae)
            next_val = np.where(v, self.val[t], next_val)
            adv[t] = np.where(v, g, 0.0)
        valid = self.valid.reshape(-1)
        N = int(valid.sum())
        if N < 256:
            return 0, None
        # inputs stay where the rollout kept them: on the GPU, or with --rollout-on-cpu in CPU RAM,
        # sent per minibatch so that GPU memory depends on the minibatch size only
        cat = lambda xs: torch.cat(xs) if xs[0] is not None else None
        b_obs, b_map, b_gmap = cat(self.obs), cat(self.map), cat(self.gmap)
        self.obs, self.map, self.gmap = [None] * T, [None] * T, [None] * T   # free the per-step copies
        big_dev = b_obs.device
        b_act = torch.from_numpy(self.act.reshape(-1)[valid]).to(device)
        b_logp = torch.from_numpy(self.logp.reshape(-1)[valid]).to(device)
        b_adv = torch.from_numpy(adv.reshape(-1)[valid]).to(device)
        b_ret = b_adv + torch.from_numpy(self.val.reshape(-1)[valid]).to(device)
        with torch.no_grad():
            b_val = b_ret - b_adv
            explained_var = float(1 - (b_ret - b_val).var() / (b_ret.var() + 1e-8))

        model.train()
        clipfracs, pg_losses, v_losses, ents, kls = [], [], [], [], []
        # equal-sized minibatches: slicing by --minibatch leaves a remainder of N % minibatch rows,
        # and a remainder of 1 makes the advantage std NaN (this killed a run after 3794 updates)
        n_mb = -(-N // args.minibatch)
        for _ in range(args.epochs):
            for big_idx in torch.randperm(N).tensor_split(n_mb):
                big_idx = big_idx.to(big_dev)
                idx = big_idx.to(device)
                logits, value = model(b_obs[big_idx].to(device, non_blocking=True),
                                      b_map[big_idx].to(device, non_blocking=True) if b_map is not None else None,
                                      b_gmap[big_idx].to(device, non_blocking=True) if b_gmap is not None else None)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                adv_mb = b_adv[idx]
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                pg = -torch.min(ratio * adv_mb, ratio.clamp(1 - args.clip, 1 + args.clip) * adv_mb).mean()
                vl = F.mse_loss(value, b_ret[idx])
                ent = dist.entropy().mean()
                loss = pg + args.vf_coef * vl - ent_coef * ent
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                self.optim.step()
                with torch.no_grad():
                    kls.append(((ratio - 1) - (logp - b_logp[idx])).mean().item())
                    clipfracs.append(((ratio - 1).abs() > args.clip).float().mean().item())
                pg_losses.append(pg.item())
                v_losses.append(vl.item())
                ents.append(ent.item())
        model.eval()
        return N, {'pg': np.mean(pg_losses), 'ent': np.mean(ents), 'kl': np.mean(kls), 'clip': np.mean(clipfracs),
                   'vl': np.mean(v_losses), 'ev': explained_var}


class Exploiter(Learner):
    """League exploiter (AlphaStar style): trained only in rounds against the current main agent,
    starting from a copy of it. Once it beats the main agent clearly (or gets too old) a frozen
    copy joins the opponent pool, so the main agent keeps playing against its weaknesses, and the
    exploiter restarts from the latest main agent."""

    def __init__(self, k, model, lr, T, E, P, store_dev):
        super().__init__(EXPLOITER - k, model, lr, T, E, P, store_dev)
        self.k = k
        self.generation = 0

    def restart(self, main_model, lr):
        self.model.load_state_dict(main_model.state_dict())
        self.model.eval()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.age = 0          # updates since the restart
        self.score = 0.5      # EMA of its placement score in [0, 1] in rounds with the main agent
        self.rounds = 0
        self.generation += 1

    def state(self):
        return {'model': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                'optim': self.optim.state_dict(), 'generation': self.generation, 'age': self.age,
                'score': self.score, 'rounds': self.rounds}

    def load(self, state):
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-5)
        self.optim.load_state_dict(state['optim'])
        for k in ('generation', 'age', 'score', 'rounds'):
            setattr(self, k, state[k])


class RunLog:
    """Prints to stdout and appends to runs/<run>/train.log."""

    def __init__(self, path):
        self.f = open(path, 'a', buffering=1)

    def __call__(self, text=''):
        print(text, flush=True)
        self.f.write(text + '\n')


def fmt_time(seconds):
    if not np.isfinite(seconds):
        return '?'
    seconds = int(seconds)
    h, m = divmod(seconds // 60, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{seconds % 60:02d}s'


def fmt_num(x):
    return f'{x / 1e9:.2f}B' if x >= 1e9 else (f'{x / 1e6:.1f}M' if x >= 1e6 else f'{x / 1e3:.1f}k')


def episode_report(episodes, frame_skip):
    """Summary lines over recent main-agent rounds. Each episode is a dict with mode ('self',
    'mixed' or 'exploit'), human (human-like opponents present), ret, len (decisions), n (players),
    cause (-1 survived, 0 wall, 1 own, 2 other), kills, aggr and pickups (per bonus type)."""
    if not episodes:
        return ['  rounds   none finished yet']
    eps = list(episodes)
    arr = lambda key, cond=lambda e: True: np.array([e[key] for e in eps if cond(e)], np.float64)
    def res(cond):
        r = arr('ret', cond)
        if len(r) == 0:
            return 'n/a'
        return f"survives {np.mean(r > 0.99) * 100:4.1f}%  ret {np.mean(r):+.3f}  ({len(r)} rounds)"
    lines = []
    secs = arr('len') * frame_skip / 60.0
    lines.append(f"  rounds   last {len(eps)} learner rounds: avg survival {np.mean(secs):.1f}s of game time, "
                 f"avg {np.mean(arr('n')):.1f} players")
    lines.append(f"  self-play rounds           {res(lambda e: e['mode'] == 'self')}")
    lines.append(f"  mixed, with human-like     {res(lambda e: e['mode'] == 'mixed' and e['human'])}")
    lines.append(f"  mixed, bots/heuristic only {res(lambda e: e['mode'] == 'mixed' and not e['human'])}")
    lines.append(f"  with an exploiter          {res(lambda e: e['mode'] == 'exploit')}")
    lines.append('  by size  ' + ' | '.join(
        f"{lo}-{hi}p ret {np.mean(r):+.3f}" if len(r := arr('ret', lambda e, lo=lo, hi=hi: lo <= e['n'] <= hi))
        else f"{lo}-{hi}p n/a" for lo, hi in ((2, 2), (3, 6), (7, 16))))
    c = arr('cause')
    died = c[c >= 0]
    if len(died):
        lines.append(f"  deaths   wall {np.mean(died == 0) * 100:4.1f}% | own trail {np.mean(died == 1) * 100:4.1f}% | "
                     f"others' trails {np.mean(died == 2) * 100:4.1f}%   (of {len(died)} deaths)")
    k = arr('kills')
    a = arr('aggr')
    if len(k):
        lo, hi = (a <= np.median(a)), (a > np.median(a))
        lines.append(f"  kills    {np.mean(k):.2f} per round (opponents crashing into the bot's trail) | "
                     f"low aggression {np.mean(k[lo]):.2f} vs high {np.mean(k[hi]) if hi.any() else float('nan'):.2f}")
    picks = np.array([e['pickups'] for e in eps])
    hum = np.array([e['mode'] == 'mixed' and e['human'] for e in eps])
    if len(picks):
        def per_round(mask, sel):
            return f"{picks[sel][:, mask].sum(1).mean():.2f}" if sel.any() else 'n/a'
        g = BONUS_GROUPS
        lines.append(f"  bonuses  picked per round: enemy maluses {per_round(g['enemy'], hum)} with human-like "
                     f"opponents vs {per_round(g['enemy'], ~hum)} without | own bonuses "
                     f"{per_round(g['self'], np.ones(len(eps), bool))} | game-wide {per_round(g['game'], np.ones(len(eps), bool))}")
    return lines


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = os.path.join('runs', args.run_name)
    os.makedirs(os.path.join(run_dir, 'pool'), exist_ok=True)
    # a run with a full checkpoint resumes from it (e.g. after a preemption requeued the job)
    state_path = os.path.join(run_dir, 'state.pt')
    ckpt = torch.load(state_path, map_location='cpu', weights_only=False) if os.path.exists(state_path) else None
    if ckpt is None:
        with open(os.path.join(run_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    # purge_step drops the tensorboard points logged after the checkpoint by the interrupted job
    writer = SummaryWriter(run_dir, purge_step=ckpt['total_steps'] if ckpt else None)
    log = RunLog(os.path.join(run_dir, 'train.log'))

    hidden = tuple(int(h) for h in args.hidden.split(','))
    device = torch.device(args.device)
    if device.type == 'cuda':
        # TF32 matmuls and convolutions: several times faster on Ampere, ample precision for PPO
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    model = ActorCritic(OBS_DIM, hidden, map_size=args.map_size, global_map=args.global_map)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
    elif args.resume:
        model.load_state_dict(load_policy(args.resume)[0].state_dict())
    model.to(device)
    make_model = lambda: ActorCritic(**model.config).to(device)

    n_probs = [float(v) for v in args.n_probs.split(',')]
    n_probs = n_probs[: args.max_players - 1]
    bonus_modes = [float(v) for v in args.bonus_modes.split(',')]
    delay = tuple(int(v) for v in args.delay.split(','))
    kill_reward = tuple(float(v) for v in args.kill_reward.split(','))
    expl_kill = tuple(float(v) for v in args.exploiter_kill_reward.split(','))
    human_delay = tuple(int(v) for v in args.human_delay.split(','))
    human_args = {'human_delay': human_delay, 'human_noise': args.human_noise,
                  'human_confusion': args.human_confusion}
    env = BatchEnv(args.num_envs, max_players=args.max_players, n_probs=n_probs,
                   bonus_modes=bonus_modes, frame_skip=args.frame_skip, delay=delay, kill_reward=kill_reward,
                   seed=args.seed + 7919 * (ckpt['update'] if ckpt else 0),   # no replay of the same rounds
                   map_size=args.map_size, global_map=args.global_map, **human_args)
    E, P, T, M = env.E, env.P, args.rollout, args.map_size

    big_dev = torch.device('cpu') if args.rollout_on_cpu else device
    main_l = Learner(SELF, model, args.lr, T, E, P, big_dev)
    exploiters = [Exploiter(k, make_model(), args.lr, T, E, P, big_dev) for k in range(args.exploiters)]
    for x in exploiters:
        x.restart(model, args.lr)
    learners = [main_l] + exploiters
    if ckpt is not None:
        main_l.optim.load_state_dict(ckpt['optim'])
        for x, xs in zip(exploiters, ckpt['exploiters']):
            x.load(xs)

    pool = OpponentPool(args.pool_size, args.active_opponents, make_model)
    # controller of every player slot: SELF (the main agent being trained), an exploiter
    # (EXPLOITER - k), HEURISTIC, or a pool id; human-like players (env.S.human) are driven by
    # the heuristic or a pool snapshot
    slot_ctrl = np.full((E, P), SELF, np.int64)
    mode = ['self'] * E                         # round kind: 'self', 'mixed' or 'exploit'
    trap_slot = np.full(E, -1, np.int64)        # trapped main-agent slot, -1 if the round is no trap
    trap_ring = np.zeros((E, 3))                # ring centre x, y and outer radius
    trap_out = np.zeros(E, np.bool_)            # the trapped player got out of the ring

    def assign(envs):
        keys = list(pool.active.keys())
        changed, traps = [], []
        for e in envs:
            n = int(env.S.n[e])
            slot_ctrl[e] = SELF
            if exploiters and random.random() < args.p_exploit:
                # one exploiter takes one slot (or up to a third of a big round), the main agent the others
                mode[e] = 'exploit'
                x = random.choice(exploiters)
                for p in random.sample(range(n), 1 if n < 6 else random.randint(1, n // 3)):
                    slot_ctrl[e, p] = x.key
                    env.S.aggr[e, p] = random.uniform(*expl_kill)   # observed: refresh below
                changed.append(e)
            elif random.random() < args.p_selfplay:
                mode[e] = 'self'
            else:
                # mixed: the main agent takes 1 up to --max-learner-share of the slots, the others are
                # human-like players, the heuristic or past versions
                mode[e] = 'mixed'
                slots = list(range(n))
                random.shuffle(slots)
                n_learn = random.randint(1, max(1, int(args.max_learner_share * n)))
                for p in slots[n_learn:]:
                    snapshot = random.choice(keys) if keys else HEURISTIC
                    u = random.random()
                    if u < args.p_human:
                        slot_ctrl[e, p] = HEURISTIC if random.random() < args.human_base_heuristic else snapshot
                        env.make_human(e, p)   # observed by the others: refresh below
                    elif u < args.p_human + args.p_heuristic:
                        slot_ctrl[e, p] = HEURISTIC
                    else:
                        slot_ctrl[e, p] = snapshot
                changed.append(e)
            trap_slot[e] = -1
            trap_out[e] = False
            if random.random() < args.p_trap:
                trap_slot[e] = random.choice([p for p in range(n) if slot_ctrl[e, p] == SELF])
                traps.append(e)
        if traps:
            trap_ring[traps] = env.trap(traps, trap_slot[traps], args.trap_own)
        env.refresh(sorted(set(changed) | set(traps)))

    n_params = sum(p.numel() for p in model.parameters())
    if device.type == 'cuda':
        gpu_total = torch.cuda.get_device_properties(device).total_memory / 1e9
        dev_name = f'{torch.cuda.get_device_name(device)} ({gpu_total:.1f} GB)'
    else:
        dev_name = f'cpu ({args.threads} threads)'
    log('=' * 100)
    log(f"Curvytron self-play PPO | run '{args.run_name}' | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  device      {dev_name}")
    log(f"  model       {n_params / 1e6:.2f}M parameters | features {OBS_DIM} | hidden {hidden}"
        f" | local map {f'{M}x{M}' if M else 'off'} | whole-board map {'on' if args.global_map else 'off'}")
    log(f"  envs        {E} parallel rounds, 2-{P} players, decision every {args.frame_skip} frames "
        f"({args.frame_skip * 1000 / 60:.0f} ms), latency {delay[0]}-{delay[1]} frames")
    log(f"  reward      placement (+1 survivor ... -1 first out) + kill bonus {kill_reward[0]}-{kill_reward[1]} "
        f"per kill (sampled per player, observed) + territory shaping x{args.territory_coef}; "
        f"evaluations use aggression {args.eval_aggression}")
    p_rest = 1 - (args.p_exploit if exploiters else 0.0)
    log(f"  opponents   rounds: exploiter {1 - p_rest:.0%} | self-play {args.p_selfplay * p_rest:.0%} | mixed "
        f"{(1 - args.p_selfplay) * p_rest:.0%}; in mixed rounds the bot controls 1 to {args.max_learner_share:.0%} "
        f"of the players, the others are human-like {args.p_human:.0%} | heuristic {args.p_heuristic:.0%} | "
        f"past versions {max(0.0, 1 - args.p_human - args.p_heuristic):.0%}")
    log(f"  human-like  reaction {human_delay[0]}-{human_delay[1]} frames, steering noise {args.human_noise:.0%}, "
        f"confusion after maluses x{args.human_confusion}; pool snapshot every {args.snapshot_every} updates")
    if exploiters:
        log(f"  league      {len(exploiters)} exploiters (aggression {expl_kill[0]}-{expl_kill[1]}), each restarts "
            f"from the main agent after beating it with placement score {args.exploiter_target} "
            f"(at least {args.exploiter_min_age} updates) or after {args.exploiter_max_age} updates")
    log(f"  traps       {args.p_trap:.0%} of rounds start with the learner enclosed by a trail with holes "
        f"({args.trap_own:.0%} its own trail)")
    log(f"  ppo         rollout {T} | gamma {args.gamma} | lambda {args.lam} | lr {args.lr} | "
        f"minibatch {args.minibatch} x {args.epochs} epochs | target {fmt_num(args.total_steps)} steps")
    log(f"  output      {run_dir}  (tensorboard --logdir runs)")
    if ckpt is not None:
        changed = sorted(k for k, v in vars(args).items() if ckpt['args'].get(k) != v)
        log(f"  resumed     from {state_path}: update {ckpt['update']}, {fmt_num(ckpt['total_steps'])} steps"
            + (f" | changed arguments: {', '.join(changed)}" if changed else '')
            + (f" | --resume {args.resume} ignored" if args.resume else ''))
    log('  compiling the simulator on first use (~1-2 min) ...')
    log('=' * 100)

    if ckpt is not None:
        pool.load(ckpt['pool'])
    else:
        pool.add(model)
    inputs = DeviceInputs(env, device)   # pins env.obs/maps/gmaps: before anything writes them
    env.reset()
    assign(range(E))
    n_round = env.S.n.copy()   # players in each env's current round

    ep_ret = np.zeros((E, P))
    ep_len = np.zeros((E, P))
    ep_kills = np.zeros((E, P))
    episodes = deque(maxlen=5000)   # recent main-agent rounds, for logs
    trap_eps = deque(maxlen=2000)   # recent trap rounds of the main agent: escaped, ret
    graduations = 0                 # exploiter snapshots added to the pool
    rounds_done = 0
    ppo_hist = deque(maxlen=20)     # recent per-update PPO diagnostics

    video_every = args.eval_every if args.video_every is None else args.video_every
    os.makedirs(os.path.join(run_dir, 'videos'), exist_ok=True)

    total_steps = 0
    update = 0
    elapsed0 = 0.0      # training time of the previous jobs of this run
    gpu_peak_all = 0.0
    best_score = -np.inf
    last_ev = None
    if ckpt is not None:
        total_steps, update, elapsed0 = ckpt['total_steps'], ckpt['update'], ckpt['elapsed']
        gpu_peak_all, best_score, last_ev = ckpt['gpu_peak_all'], ckpt['best_score'], ckpt['last_ev']
        graduations, rounds_done = ckpt['graduations'], ckpt['rounds_done']
        episodes.extend(ckpt['episodes'])
        trap_eps.extend(ckpt['trap_eps'])
        ppo_hist.extend(ckpt['ppo_hist'])
        random.setstate(ckpt['rng']['python'])
        np.random.set_state(ckpt['rng']['numpy'])
        torch.set_rng_state(ckpt['rng']['torch'])
        if device.type == 'cuda' and ckpt['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
        del ckpt   # the pool snapshots now live in the pool
    steps0 = total_steps
    t_start = time.time()
    t_last_log = t_start
    steps_last_log = total_steps
    t_roll_sum = t_train_sum = 0.0

    def save_state():
        """Everything needed to go on as if the run had never stopped (except the rounds in
        progress), written atomically: a kill while saving leaves the previous file intact."""
        t0 = time.time()
        tmp = state_path + '.tmp'
        torch.save({
            'args': vars(args), 'total_steps': total_steps, 'update': update,
            'elapsed': elapsed0 + time.time() - t_start, 'gpu_peak_all': gpu_peak_all,
            'best_score': best_score, 'last_ev': last_ev, 'graduations': graduations, 'rounds_done': rounds_done,
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optim': main_l.optim.state_dict(),
            'exploiters': [x.state() for x in exploiters],
            'pool': pool.state(),
            'episodes': list(episodes), 'trap_eps': list(trap_eps), 'ppo_hist': list(ppo_hist),
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None},
        }, tmp)
        os.replace(tmp, state_path)
        return time.time() - t0

    # SLURM sends SIGTERM before killing the job (preemption, scancel, time limit): stop at the next
    # safe point, i.e. between two decisions of the rollout or after the update, and checkpoint
    stop = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.append(signum))

    while total_steps < args.total_steps:
        frac = min(total_steps / args.total_steps, 1.0)
        lr_now = args.lr * (1.0 - 0.9 * frac)
        for L in learners:
            for g in L.optim.param_groups:
                g['lr'] = lr_now
        ent_coef = args.ent + (args.ent_final - args.ent) * frac
        t_roll = time.time()

        model.eval()
        for t in range(T):
            if stop:   # the weights are still those of the last update: checkpoint and leave
                break
            alive = env.alive
            inputs.refresh()
            # has the trapped player left its ring?
            te = np.nonzero(trap_slot >= 0)[0]
            if len(te):
                ts = trap_slot[te]
                d2 = (env.S.x[te, ts] - trap_ring[te, 0]) ** 2 + (env.S.y[te, ts] - trap_ring[te, 1]) ** 2
                trap_out[te] |= alive[te, ts] & (d2 > (trap_ring[te, 2] + 2.0) ** 2)
            actions = np.zeros((E, P), np.int64)
            masks = []
            for L in learners:
                mask = alive & (slot_ctrl == L.key)
                actions[mask] = L.step(inputs, mask, t)
                masks.append(mask)
            learner = np.logical_or.reduce(masks)
            heur = alive & (slot_ctrl == HEURISTIC)
            if heur.any():
                actions[heur] = env.heuristic(heur, eps=0.3)[heur]
            for k, m in pool.active.items():
                sel = alive & (slot_ctrl == k)
                if sel.any():
                    with torch.no_grad():
                        actions[sel] = m.act(*inputs.rows(m, sel)).cpu().numpy() - 1
            aggr = env.S.aggr.copy()     # this round's aggression and human-like players
            human = env.S.human.copy()   # (copied: the env may reset during the step)
            phi = env.obs[:, :, O_TERR].astype(np.float64)
            inputs.wait()
            _, rew, done, reset = env.step(actions)
            # learn from placement + kill bonus + potential-based territory shaping (phi = 0 when done)
            phi_next = np.where(done, 0.0, env.obs[:, :, O_TERR])
            shaped = rew + env.kill_rew + args.territory_coef * (args.gamma * phi_next - phi)
            for L in learners:
                L.rew[t] = shaped
                L.done[t] = done
            # episode stats for learner agents (placement only, plus kill counts)
            ep_ret += np.where(learner, rew, 0.0)
            ep_len += learner
            ep_kills += np.where(learner, env.kills, 0)
            for e, p in zip(*np.nonzero(done & learner)):
                score = (ep_ret[e, p] + 1.0) / 2.0   # placement in [0, 1]
                if slot_ctrl[e, p] == SELF:
                    episodes.append({'mode': mode[e], 'human': bool(human[e].any()),
                                     'ret': ep_ret[e, p], 'len': ep_len[e, p],
                                     'n': int(n_round[e]), 'cause': int(env.cause[e, p]),
                                     'kills': ep_kills[e, p], 'aggr': aggr[e, p],
                                     'pickups': env.pickups[e, p].copy()})
                    for c in set(int(c) for c in slot_ctrl[e, :n_round[e]] if c >= 0):
                        pool.record(c, score)
                    if trap_slot[e] == p:
                        trap_eps.append({'out': bool(trap_out[e]), 'ret': ep_ret[e, p]})
                else:
                    x = exploiters[EXPLOITER - int(slot_ctrl[e, p])]
                    x.score = 0.99 * x.score + 0.01 * score
                    x.rounds += 1
                ep_kills[e, p] = 0.0
                ep_ret[e, p] = 0.0
                ep_len[e, p] = 0.0
            if reset.any():
                rounds_done += int(reset.sum())
                assign(np.nonzero(reset)[0])
                n_round[reset] = env.S.n[reset]
        t_roll = time.time() - t_roll
        if stop:
            break

        t_train = time.time()
        alive = env.alive
        inputs.refresh()
        N, d = main_l.update(inputs, alive & (slot_ctrl == SELF), args, ent_coef, device)
        total_steps += N
        for x in exploiters:
            x.update(inputs, alive & (slot_ctrl == x.key), args, ent_coef, device)
            x.age += 1
            graduated = (x.age >= args.exploiter_min_age and x.rounds >= 200
                         and x.score >= args.exploiter_target)
            if graduated or x.age >= args.exploiter_max_age:
                pool.add(x.model, name=f'expl{x.k}.{x.generation}')
                graduations += 1
                writer.add_scalar(f'league/exploiter{x.k}_final_score', x.score, total_steps)
                log(f"  league   exploiter {x.k} gen {x.generation} "
                    f"{'beat the main agent' if graduated else 'timed out'} after {x.age} updates "
                    f"(placement score {x.score:.2f} in {x.rounds} rounds): added to the pool, restarting")
                x.restart(model, args.lr)
        t_train = time.time() - t_train
        update += 1
        t_roll_sum += t_roll
        t_train_sum += t_train
        if d is not None:
            ppo_hist.append(d)

        if update % args.snapshot_every == 0:
            pool.add(model)

        now = time.time()
        sps = (total_steps - steps0) / (now - t_start)
        writer.add_scalar('perf/sps', sps, total_steps)
        if d is not None:
            for k, name in (('pg', 'policy'), ('vl', 'value'), ('ent', 'entropy'), ('kl', 'approx_kl'),
                            ('clip', 'clipfrac'), ('ev', 'explained_variance')):
                writer.add_scalar(f'loss/{name}', d[k], total_steps)
        for x in exploiters:
            writer.add_scalar(f'league/exploiter{x.k}_score', x.score, total_steps)
        if trap_eps:
            recent_traps = list(trap_eps)[-500:]
            writer.add_scalar('train/trap_escape', np.mean([e['out'] for e in recent_traps]), total_steps)
            writer.add_scalar('train/trap_ret', np.mean([e['ret'] for e in recent_traps]), total_steps)
        if episodes:
            recent = list(episodes)[-500:]
            for name, cond in (('selfplay', lambda e: e['mode'] == 'self'),
                               ('with_human', lambda e: e['mode'] == 'mixed' and e['human']),
                               ('mixed_bots', lambda e: e['mode'] == 'mixed' and not e['human']),
                               ('exploit', lambda e: e['mode'] == 'exploit')):
                sel = [e for e in recent if cond(e)]
                if sel:
                    writer.add_scalar(f'train/ret_{name}', np.mean([e['ret'] for e in sel]), total_steps)
                    for g, mask in BONUS_GROUPS.items():
                        writer.add_scalar(f'bonuses/{g}_per_round_{name}',
                                          np.mean([e['pickups'][mask].sum() for e in sel]), total_steps)
            died = [e['cause'] for e in recent if e['cause'] >= 0]
            if died:
                for i, name in enumerate(('wall', 'own', 'other')):
                    writer.add_scalar(f'train/death_{name}', np.mean(np.array(died) == i), total_steps)

        if update == 1 or now - t_last_log >= args.log_every:
            recent_sps = (total_steps - steps_last_log) / max(now - t_last_log, 1e-6)
            eta = (args.total_steps - total_steps) / max(recent_sps, 1e-6)
            h = {k: np.mean([d[k] for d in ppo_hist]) for k in ppo_hist[0]}
            log(f"[{fmt_time(elapsed0 + now - t_start)} elapsed | update {update} | {fmt_num(total_steps)} / "
                f"{fmt_num(args.total_steps)} steps ({100 * total_steps / args.total_steps:.1f}%) | "
                f"ETA {fmt_time(eta)}]")
            log(f"  speed    {fmt_num(recent_sps)} steps/s now, {fmt_num(sps)} avg | time split: simulate+act "
                f"{100 * t_roll_sum / max(t_roll_sum + t_train_sum, 1e-9):.0f}%, learn "
                f"{100 * t_train_sum / max(t_roll_sum + t_train_sum, 1e-9):.0f}% | {rounds_done} rounds played")
            if device.type == 'cuda':
                # peaks since the previous report; 'reserved' is what PyTorch holds (what nvidia-smi shows, minus
                # the CUDA context), 'allocated' what tensors actually use
                peak = torch.cuda.max_memory_allocated(device) / 1e9
                peak_res = torch.cuda.max_memory_reserved(device) / 1e9
                gpu_peak_all = max(gpu_peak_all, peak_res)
                writer.add_scalar('perf/gpu_peak_reserved_gb', peak_res, total_steps)
                log(f"  gpu mem  peak {peak:.1f} GB allocated, {peak_res:.1f} GB reserved of {gpu_total:.1f} GB "
                    f"({100 * peak_res / gpu_total:.0f}%) | highest reserved so far {gpu_peak_all:.1f} GB")
                torch.cuda.reset_peak_memory_stats(device)
            for line in episode_report(episodes, args.frame_skip):
                log(line)
            log(f"  ppo      entropy {h['ent']:.3f} (max 1.099 = random) | kl {h['kl']:.4f} | clipped "
                f"{100 * h['clip']:.1f}% | value loss {h['vl']:.4f} | explained var {h['ev']:.2f} | "
                f"lr {lr_now:.2e} | entropy coef {ent_coef:.4f}")
            log(f"  pool     {pool.summary()}")
            if exploiters:
                log("  league   " + ' | '.join(
                    f"exploiter {x.k} gen {x.generation}: age {x.age}, placement vs main {x.score:.2f} "
                    f"({x.rounds} rounds)" for x in exploiters) + f" | {graduations} added to the pool so far")
            if trap_eps:
                tr = list(trap_eps)
                log(f"  traps    last {len(tr)} trap rounds: got out of the ring {np.mean([e['out'] for e in tr]) * 100:4.1f}%"
                    f" | ret {np.mean([e['ret'] for e in tr]):+.3f}")
            t_last_log = now
            steps_last_log = total_steps
            t_roll_sum = t_train_sum = 0.0

        if update % args.save_every == 0:
            save_policy(os.path.join(run_dir, 'model_latest.pt'), model,
                        {'steps': total_steps, 'update': update, 'args': vars(args)})
            for x in exploiters:
                save_policy(os.path.join(run_dir, f'exploiter{x.k}_latest.pt'), x.model,
                            {'steps': total_steps, 'update': update, 'generation': x.generation, 'args': vars(args)})
        if update % args.eval_every == 0:
            model.eval()
            ev = {}
            picks = {}
            for n_eval in (2, 4, 8):
                picks[n_eval] = {}
                ev[f'ret_heur_{n_eval}p'], ev[f'win_heur_{n_eval}p'] = evaluate(
                    model, 'heuristic', args.eval_rounds, seed=update + n_eval, n_players=n_eval, delay=delay,
                    frame_skip=args.frame_skip, aggression=args.eval_aggression, stats=picks[n_eval])
            picks['human'] = {}
            ev['ret_human_6p'], ev['win_human_6p'] = evaluate(
                model, 'human', args.eval_rounds, seed=update + 6, n_players=6, delay=delay,
                frame_skip=args.frame_skip, aggression=args.eval_aggression, human_args=human_args,
                stats=picks['human'])
            ev['enemy_maluses_human_6p'] = picks['human'].get('enemy', 0.0)
            ev['enemy_maluses_heur_4p'] = picks[4].get('enemy', 0.0)
            ref = make_model()
            mains = [e for e in pool.entries if e['name'] == 'main']
            ref.load_state_dict(mains[max(0, len(mains) - 9)]['state'])
            ref.eval()
            ev['ret_past_6p'], ev['win_past_6p'] = evaluate(
                model, ref, args.eval_rounds, seed=update + 1, n_players=6, delay=delay,
                frame_skip=args.frame_skip, aggression=args.eval_aggression)
            # one number to track: mean round return vs the heuristic (2, 4, 8 players) and vs
            # human-like players (6 players)
            ev['score'] = float(np.mean([ev[f'ret_heur_{n}p'] for n in (2, 4, 8)] + [ev['ret_human_6p']]))
            for k, v in ev.items():
                writer.add_scalar(f'eval/{k}', v, total_steps)
            def delta(k, scale=1.0, digits=1):
                return f" ({(ev[k] - last_ev[k]) * scale:+.{digits}f})" if last_ev else ''
            log(f"  ---- EVALUATION after {fmt_num(total_steps)} steps ({args.eval_rounds} rounds each, "
                f"in parentheses: change since last evaluation) ----")
            for n in (2, 4, 8):
                log(f"  vs heuristic, {n} players: survives {100 * ev[f'win_heur_{n}p']:5.1f}%"
                    f"{delta(f'win_heur_{n}p', 100)} | ret {ev[f'ret_heur_{n}p']:+.3f}"
                    f"{delta(f'ret_heur_{n}p', digits=3)}")
            log(f"  vs human-like, 6 players: survives {100 * ev['win_human_6p']:5.1f}%"
                f"{delta('win_human_6p', 100)} | ret {ev['ret_human_6p']:+.3f}{delta('ret_human_6p', digits=3)}"
                f"   (the heuristic with human reactions, noise and confusion after maluses)")
            log("  (ret: +1 = always last alive, 0 = average placement, -1 = always first out)")
            log(f"  bonuses picked per round: enemy maluses {ev['enemy_maluses_human_6p']:.2f} vs human-like (6p)"
                f" vs {ev['enemy_maluses_heur_4p']:.2f} vs the heuristic (4p) | own bonuses "
                f"{picks['human'].get('self', 0.0):.2f} | game-wide {picks['human'].get('game', 0.0):.2f} (vs human-like)")
            log(f"  vs itself {8 * args.snapshot_every} updates ago, 6 players: survives "
                f"{100 * ev['win_past_6p']:5.1f}% | ret {ev['ret_past_6p']:+.3f}   (> 0 means it improved)")
            path = os.path.join(run_dir, 'pool', f'model_{update:05d}.pt')
            save_policy(path, model, {'steps': total_steps, 'update': update, 'eval': ev, 'args': vars(args)})
            if ev['score'] > best_score:
                best_score = ev['score']
                save_policy(os.path.join(run_dir, 'model_best.pt'), model,
                            {'steps': total_steps, 'update': update, 'eval': ev, 'args': vars(args)})
                log(f"  score {ev['score']:+.3f} -> new best, saved model_best.pt")
            else:
                log(f"  score {ev['score']:+.3f} (best so far {best_score:+.3f})")
            last_ev = ev

        if video_every and update % video_every == 0:
            model.eval()
            mix = {'exploiter': random.choice(exploiters).model if exploiters else None,
                   'past': list(pool.active.values()), 'max_learner_share': args.max_learner_share,
                   'p_human': args.p_human, 'p_heuristic': args.p_heuristic,
                   'human_base_heuristic': args.human_base_heuristic}
            frames, info = record_video(model, args.video_players, args.video_opponent, seed=update,
                                        frame_skip=args.frame_skip, delay=delay, aggression=args.eval_aggression,
                                        human_args=human_args, mix=mix)
            gif = encode_gif(frames, info['fps'])
            path = os.path.join(run_dir, 'videos', f'round_{update:05d}.gif')
            with open(path, 'wb') as f:
                f.write(gif)
            h, w = frames[0].shape[:2]
            img = Summary.Image(height=h, width=w, colorspace=3, encoded_image_string=gif)
            writer._get_file_writer().add_summary(
                Summary(value=[Summary.Value(tag='video/sample_round', image=img)]), total_steps)
            log(f"  video    {path}: {args.video_players} players ({args.video_opponent}: {info['players']}), bot ringed "
                f"in white finished #{info['rank']} of {args.video_players} after {info['seconds']:.0f}s "
                f"with {info['kills']} kill(s) and {info['bonuses']} bonus(es) "
                f"| tensorboard: Images tab, video/sample_round")

        if stop:
            break
        if update % args.checkpoint_every == 0:
            secs = save_state()
            log(f"  ckpt     full checkpoint {state_path} saved in {secs:.1f}s (update {update})")

    if stop:
        secs = save_state()
        log(f"stopped by signal {stop[0]} at update {update}: full checkpoint saved in {secs:.1f}s, "
            f"the run resumes from it when restarted")
        writer.close()
        return
    save_state()   # a requeue after the end finds the run finished
    save_policy(os.path.join(run_dir, 'model_final.pt'), model,
                {'steps': total_steps, 'update': update, 'args': vars(args)})
    log('=' * 100)
    best = (f"best evaluation score {best_score:+.3f} -> {run_dir}/model_best.pt"
            if np.isfinite(best_score) else 'no evaluation ran')
    log(f"done: {fmt_num(total_steps)} steps in {fmt_time(elapsed0 + time.time() - t_start)}, {best}, "
        f"last model {run_dir}/model_final.pt")
    writer.close()


if __name__ == '__main__':
    main()
