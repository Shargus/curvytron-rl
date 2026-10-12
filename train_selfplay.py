"""Self-play PPO for Curvytron on the fast numba simulator (curvysim).

Each environment runs one round with 2..P players. A share of rounds is pure
self-play (every player is the current bot and produces training data). In the
other rounds the current bot controls a few players and each remaining player is
a past version of the bot (pool snapshot, sampled with prioritized fictitious
self-play), the lookahead heuristic, or a human-like player: the heuristic or a
snapshot with human reaction times, steering noise and confusion after maluses
(see curvysim/env.py). The bot observes which opponents are human-like, so it can
learn a different style against humans (e.g. using enemy maluses more).

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
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tensorboard.compat.proto.summary_pb2 import Summary

from curvysim import BatchEnv, OBS_DIM
from curvysim.core import B_AFFECT
from curvysim.policy import ActorCritic, save_policy, load_policy
from curvysim.render import render, encode_gif

LEARNER, HEURISTIC = -1, -2


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
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-rounds', type=int, default=300)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--log-every', type=float, default=60.0, help='seconds between progress reports')
    p.add_argument('--video-every', type=int, default=None,
                   help='updates between sample videos (default: --eval-every, 0 = off)')
    p.add_argument('--video-players', type=int, default=6)
    p.add_argument('--video-opponent', default='human', choices=['self', 'heuristic', 'human'],
                   help='who controls the other avatars in the video (human: the bot with human-like handicaps)')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', default=None, help='checkpoint to initialise from')
    return p.parse_args()


def to_inputs(model, env, sel):
    """Gather the rows selected by a boolean mask [E, P] as model input tensors."""
    dev = next(model.parameters()).device
    obs = torch.from_numpy(env.obs[sel]).to(dev)
    maps = torch.from_numpy(env.maps[sel]).to(dev) if model.map_size > 0 else None
    gmaps = torch.from_numpy(env.gmaps[sel]).to(dev) if model.global_map else None
    return obs, maps, gmaps


@torch.no_grad()
def policy_moves(model, env, sel, greedy=False):
    if not sel.any():
        return np.zeros(0, np.int64)
    return model.act(*to_inputs(model, env, sel), greedy=greedy).cpu().numpy() - 1


# bonus types grouped by who they affect (for logs): enemy maluses, self bonuses, game/all
BONUS_GROUPS = {'enemy': np.array(B_AFFECT) == 1, 'self': np.array(B_AFFECT) == 0,
                'game': (np.array(B_AFFECT) == 2) | (np.array(B_AFFECT) == 3)}


@torch.no_grad()
def evaluate(model, opponent, rounds, seed, n_players=2, bonus_modes=(0.3, 0.4, 0.3), num_envs=128,
             delay=(0, 8), frame_skip=3, aggression=0.2, human_args=None, stats=None):
    """Results of ``model`` (one random slot) vs ``opponent`` (all other slots).

    opponent: an ActorCritic, 'heuristic', or 'human' (the heuristic with human-like handicaps,
    see BatchEnv). Returns (mean return, win rate), both from placement only (the kill bonus is
    not counted). Every player gets ``aggression``. If ``stats`` is a dict, it receives the mean
    number of bonuses per round the model picked up, by group.
    """
    probs = [0.0] * (n_players - 1)
    probs[-1] = 1.0
    map_size = max(model.map_size, getattr(opponent, 'map_size', 0))
    global_map = model.global_map or getattr(opponent, 'global_map', False)
    env = BatchEnv(num_envs, max_players=n_players, n_probs=probs, bonus_modes=bonus_modes,
                   seed=seed, map_size=map_size, global_map=global_map, delay=delay,
                   frame_skip=frame_skip, kill_reward=(aggression, aggression), **(human_args or {}))
    env.reset()
    ret = np.zeros(num_envs)
    results = []
    picks = []
    rng = np.random.default_rng(seed)
    # the server updates avatars in reverse join order, which matters in head-to-head
    # encounters, so the evaluated policy plays from a random slot
    my_slot = rng.integers(0, n_players, num_envs)
    slots = np.arange(env.P)[None, :]

    def setup(e):
        if opponent == 'human':
            for p in range(n_players):
                if p != my_slot[e]:
                    env.make_human(e, p)

    for e in range(num_envs):
        setup(e)
    while len(results) < rounds:
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
            picks.append(env.pickups[e, my_slot[e]].copy())
        for e in np.nonzero(reset)[0]:
            results.append(ret[e])
            ret[e] = 0.0
            my_slot[e] = rng.integers(0, n_players)
            setup(e)
    results = np.array(results[:rounds])
    if stats is not None and picks:
        picks = np.array(picks)
        for g, mask in BONUS_GROUPS.items():
            stats[g] = float(picks[:, mask].sum(1).mean())
    return float(results.mean()), float((results > 0.99).mean())   # win = last survivor


@torch.no_grad()
def record_video(model, n_players, opponent, seed, frame_skip, delay, size_px=240, stride=2, max_seconds=90,
                 aggression=0.2, human_args=None):
    """Play one round (all bonuses on) and return (frames, info). The model controls a random
    slot, drawn with a white ring; the others are the model too ('self'), the heuristic, or the
    model with human-like handicaps ('human')."""
    probs = [0.0] * (n_players - 1)
    probs[-1] = 1.0
    env = BatchEnv(1, max_players=n_players, n_probs=probs, bonus_modes=(0, 1, 0), seed=seed,
                   map_size=model.map_size, global_map=model.global_map, delay=delay, frame_skip=frame_skip,
                   kill_reward=(aggression, aggression), **(human_args or {}))
    env.reset()
    me = np.random.default_rng(seed).integers(0, n_players)
    if opponent == 'human':
        for p in range(n_players):
            if p != me:
                env.make_human(0, p)
    slots = np.arange(n_players)[None, :]
    frames = [render(env.S, 0, size_px, me)]
    ret = 0.0
    kills = 0
    picks = np.zeros(len(B_AFFECT), np.int64)
    step = 0
    max_steps = int(max_seconds * 60 / frame_skip)
    while step < max_steps:
        alive = env.alive
        mine = alive & (slots == me)
        others = alive & ~mine
        actions = np.zeros((1, n_players), np.int64)
        if opponent in ('self', 'human'):
            actions[alive] = policy_moves(model, env, alive)
        else:
            actions[mine] = policy_moves(model, env, mine)
            actions[others] = env.heuristic(others)[others]
        _, rew, done, reset = env.step(actions)
        ret += rew[0, me]
        kills += env.kills[0, me]
        if done[0, me]:
            picks = env.pickups[0, me].copy()
        step += 1
        if reset[0]:
            break
        if step % stride == 0:
            frames.append(render(env.S, 0, size_px, me))
    frames += [frames[-1]] * 10   # hold the last frame for a moment
    placement = int(round((ret + 1) / 2 * (n_players - 1)))   # points: 0 = first out, n-1 = survivor
    info = {'seconds': step * frame_skip / 60, 'rank': n_players - placement, 'fps': 60 / (frame_skip * stride),
            'kills': int(kills), 'bonuses': int(picks.sum())}
    return frames, info


class OpponentPool:
    """Frozen snapshots with prioritized fictitious self-play sampling."""

    def __init__(self, size, n_active, make_model):
        self.size = size
        self.n_active = n_active
        self.make_model = make_model
        self.entries = []      # dicts: id, state, wins (EMA of learner score in [0, 1])
        self.active = {}       # id -> model
        self.next_id = 0

    def add(self, model):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.entries.append({'id': self.next_id, 'state': state, 'win': 0.5})
        self.next_id += 1
        if len(self.entries) > self.size:   # keep the very first snapshot as an anchor
            self.entries.pop(1)
        self.refresh()

    def weights(self):
        # PFSP "hard" weighting: opponents the learner beats less often get more games
        w = np.array([(1.0 - e['win']) ** 2 + 0.05 for e in self.entries])
        return w / w.sum()

    def refresh(self):
        self.active.clear()
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

    def record(self, opp_id, learner_score):
        for e in self.entries:
            if e['id'] == opp_id:
                e['win'] = 0.98 * e['win'] + 0.02 * float(learner_score)

    def summary(self, k=3):
        hardest = sorted(self.entries, key=lambda e: e['win'])[:k]
        return (f"{len(self.entries)} snapshots, hardest for the learner (placement score, 0.5 = even): "
                + ', '.join(f"#{e['id']} {e['win']:.2f}" for e in hardest))


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
    """Summary lines over recent learner rounds. Each episode is a dict with mode ('self' or
    'mixed'), human/heur/pool (opponent kinds present), ret, len (decisions), n (players), cause
    (-1 survived, 0 wall, 1 own, 2 other), kills, aggr and pickups (per bonus type)."""
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
    lines.append(f"  self-play rounds          {res(lambda e: e['mode'] == 'self')}")
    lines.append(f"  mixed, with human-like    {res(lambda e: e['mode'] == 'mixed' and e['human'])}")
    lines.append(f"  mixed, bots/heuristic only {res(lambda e: e['mode'] == 'mixed' and not e['human'])}")
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
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(run_dir)
    log = RunLog(os.path.join(run_dir, 'train.log'))

    hidden = tuple(int(h) for h in args.hidden.split(','))
    device = torch.device(args.device)
    model = ActorCritic(OBS_DIM, hidden, map_size=args.map_size, global_map=args.global_map)
    if args.resume:
        model.load_state_dict(load_policy(args.resume)[0].state_dict())
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    make_model = lambda: ActorCritic(**model.config).to(device)

    n_probs = [float(v) for v in args.n_probs.split(',')]
    n_probs = n_probs[: args.max_players - 1]
    bonus_modes = [float(v) for v in args.bonus_modes.split(',')]
    delay = tuple(int(v) for v in args.delay.split(','))
    kill_reward = tuple(float(v) for v in args.kill_reward.split(','))
    human_delay = tuple(int(v) for v in args.human_delay.split(','))
    human_args = {'human_delay': human_delay, 'human_noise': args.human_noise,
                  'human_confusion': args.human_confusion}
    env = BatchEnv(args.num_envs, max_players=args.max_players, n_probs=n_probs,
                   bonus_modes=bonus_modes, frame_skip=args.frame_skip, delay=delay, kill_reward=kill_reward,
                   seed=args.seed, map_size=args.map_size, global_map=args.global_map, **human_args)
    E, P, T, M = env.E, env.P, args.rollout, args.map_size

    pool = OpponentPool(args.pool_size, args.active_opponents, make_model)
    # controller of every player slot: LEARNER (the bot being trained), HEURISTIC, or a pool id;
    # human-like players (env.S.human) are driven by the heuristic or a pool snapshot
    ctrl = np.full((E, P), LEARNER, np.int64)
    mode = ['self'] * E

    def assign(envs):
        keys = list(pool.active.keys())
        for e in envs:
            n = int(env.S.n[e])
            ctrl[e, :] = LEARNER
            if random.random() < args.p_selfplay or n < 2:
                mode[e] = 'self'
                continue
            mode[e] = 'mixed'
            slots = list(range(n))
            random.shuffle(slots)
            n_learn = random.randint(1, max(1, int(args.max_learner_share * n)))
            for p in slots[n_learn:]:
                snapshot = random.choice(keys) if keys else HEURISTIC
                u = random.random()
                if u < args.p_human:
                    ctrl[e, p] = HEURISTIC if random.random() < args.human_base_heuristic else snapshot
                    env.make_human(e, p)
                elif u < args.p_human + args.p_heuristic:
                    ctrl[e, p] = HEURISTIC
                else:
                    ctrl[e, p] = snapshot

    def learner_mask(alive):
        return alive & (ctrl == LEARNER)

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
        f"per kill (sampled per player, observed); evaluations use aggression {args.eval_aggression}")
    log(f"  opponents   {args.p_selfplay:.0%} self-play rounds; mixed rounds: the bot controls 1 to "
        f"{args.max_learner_share:.0%} of the players, the others are human-like {args.p_human:.0%} | heuristic "
        f"{args.p_heuristic:.0%} | past versions {max(0.0, 1 - args.p_human - args.p_heuristic):.0%}")
    log(f"  human-like  reaction {human_delay[0]}-{human_delay[1]} frames, steering noise {args.human_noise:.0%}, "
        f"confusion after maluses x{args.human_confusion}; pool snapshot every {args.snapshot_every} updates")
    log(f"  ppo         rollout {T} | gamma {args.gamma} | lambda {args.lam} | lr {args.lr} | "
        f"minibatch {args.minibatch} x {args.epochs} epochs | target {fmt_num(args.total_steps)} steps")
    log(f"  output      {run_dir}  (tensorboard --logdir runs)")
    log('  compiling the simulator on first use (~1-2 min) ...')
    log('=' * 100)

    pool.add(model)
    env.reset()
    assign(range(E))
    n_round = env.S.n.copy()   # players in each env's current round

    # observations are stored only for learner rows (row-major order of the [E, P] mask),
    # which matches buf_valid.reshape(-1) when concatenated over t
    buf_obs = [None] * T
    buf_map = [None] * T
    buf_gmap = [None] * T
    buf_act = np.zeros((T, E, P), np.int64)
    buf_logp = np.zeros((T, E, P), np.float32)
    buf_val = np.zeros((T, E, P), np.float32)
    buf_rew = np.zeros((T, E, P), np.float32)
    buf_done = np.zeros((T, E, P), np.bool_)
    buf_valid = np.zeros((T, E, P), np.bool_)

    ep_ret = np.zeros((E, P))
    ep_len = np.zeros((E, P))
    ep_kills = np.zeros((E, P))
    episodes = deque(maxlen=5000)   # recent learner rounds, for logs
    rounds_done = 0
    ppo_hist = deque(maxlen=20)     # recent per-update PPO diagnostics

    video_every = args.eval_every if args.video_every is None else args.video_every
    os.makedirs(os.path.join(run_dir, 'videos'), exist_ok=True)

    total_steps = 0
    update = 0
    t_start = time.time()
    t_last_log = t_start
    steps_last_log = 0
    t_roll_sum = t_train_sum = 0.0
    gpu_peak_all = 0.0
    best_score = -np.inf
    last_ev = None

    while total_steps < args.total_steps:
        frac = min(total_steps / args.total_steps, 1.0)
        for g in optim.param_groups:
            g['lr'] = args.lr * (1.0 - 0.9 * frac)
        ent_coef = args.ent + (args.ent_final - args.ent) * frac
        t_roll = time.time()

        model.eval()
        for t in range(T):
            alive = env.alive
            learner = learner_mask(alive)
            others = alive & ~learner
            actions = np.zeros((E, P), np.int64)
            with torch.no_grad():
                logits, value = model(*to_inputs(model, env, learner))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
                a, logp, value = a.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
            actions[learner] = a - 1
            buf_obs[t] = env.obs[learner]
            if M > 0:
                buf_map[t] = env.maps[learner]
            if args.global_map:
                buf_gmap[t] = env.gmaps[learner]
            buf_act[t][learner] = a
            buf_logp[t][learner] = logp
            buf_val[t][learner] = value
            buf_valid[t] = learner
            # opponents
            heur = others & (ctrl == HEURISTIC)
            if heur.any():
                actions[heur] = env.heuristic(heur, eps=0.3)[heur]
            for k, m in pool.active.items():
                sel = others & (ctrl == k)
                if sel.any():
                    actions[sel] = policy_moves(m, env, sel)
            aggr = env.S.aggr.copy()     # this round's aggression and human-like players
            human = env.S.human.copy()   # (copied: the env may reset during the step)
            _, rew, done, reset = env.step(actions)
            buf_rew[t] = rew + env.kill_rew   # learn from placement + kill bonus
            buf_done[t] = done
            # episode stats for learner agents (placement only, plus kill counts)
            ep_ret += np.where(learner, rew, 0.0)
            ep_len += learner
            ep_kills += np.where(learner, env.kills, 0)
            for e, p in zip(*np.nonzero(done & learner)):
                others_ctrl = ctrl[e, :n_round[e]]
                episodes.append({'mode': mode[e], 'human': bool(human[e].any()),
                                 'ret': ep_ret[e, p], 'len': ep_len[e, p],
                                 'n': int(n_round[e]), 'cause': int(env.cause[e, p]),
                                 'kills': ep_kills[e, p], 'aggr': aggr[e, p],
                                 'pickups': env.pickups[e, p].copy()})
                for c in set(int(c) for c in others_ctrl if c >= 0):
                    pool.record(c, (ep_ret[e, p] + 1.0) / 2.0)   # placement in [0, 1]
                ep_kills[e, p] = 0.0
                ep_ret[e, p] = 0.0
                ep_len[e, p] = 0.0
            if reset.any():
                rounds_done += int(reset.sum())
                assign(np.nonzero(reset)[0])
                n_round[reset] = env.S.n[reset]
        t_roll = time.time() - t_roll

        # bootstrap values and GAE (per agent stream; gaps after a death are skipped)
        learner = learner_mask(env.alive)
        next_val = np.zeros((E, P), np.float32)
        if learner.any():
            with torch.no_grad():
                next_val[learner] = model.value(*to_inputs(model, env, learner)).cpu().numpy()
        adv = np.zeros((T, E, P), np.float32)
        lastgae = np.zeros((E, P), np.float32)
        for t in reversed(range(T)):
            v = buf_valid[t]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * next_val * nonterm - buf_val[t]
            g = delta + args.gamma * args.lam * nonterm * lastgae
            lastgae = np.where(v, g, lastgae)
            next_val = np.where(v, buf_val[t], next_val)
            adv[t] = np.where(v, g, 0.0)
        valid = buf_valid.reshape(-1)
        # with --rollout-on-cpu the large inputs stay in CPU RAM and are sent per minibatch,
        # so GPU memory depends on the minibatch size only
        big_dev = torch.device('cpu') if args.rollout_on_cpu else device
        b_obs = torch.from_numpy(np.concatenate(buf_obs)).to(big_dev)
        b_map = torch.from_numpy(np.concatenate(buf_map)).to(big_dev) if M > 0 else None
        b_gmap = torch.from_numpy(np.concatenate(buf_gmap)).to(big_dev) if args.global_map else None
        b_act = torch.from_numpy(buf_act.reshape(-1)[valid]).to(device)
        b_logp = torch.from_numpy(buf_logp.reshape(-1)[valid]).to(device)
        b_adv = torch.from_numpy(adv.reshape(-1)[valid]).to(device)
        b_ret = b_adv + torch.from_numpy(buf_val.reshape(-1)[valid]).to(device)
        N = b_obs.shape[0]
        total_steps += N
        with torch.no_grad():
            b_val = b_ret - b_adv
            explained_var = float(1 - (b_ret - b_val).var() / (b_ret.var() + 1e-8))

        t_train = time.time()
        model.train()
        clipfracs, pg_losses, v_losses, ents, kls = [], [], [], [], []
        for _ in range(args.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, args.minibatch):
                big_idx = perm[s:s + args.minibatch].to(big_dev)
                idx = big_idx.to(device)
                logits, value = model(b_obs[big_idx].to(device, non_blocking=True),
                                      b_map[big_idx].to(device, non_blocking=True) if M > 0 else None,
                                      b_gmap[big_idx].to(device, non_blocking=True) if args.global_map else None)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                adv_mb = b_adv[idx]
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                pg = -torch.min(ratio * adv_mb, ratio.clamp(1 - args.clip, 1 + args.clip) * adv_mb).mean()
                vl = F.mse_loss(value, b_ret[idx])
                ent = dist.entropy().mean()
                loss = pg + args.vf_coef * vl - ent_coef * ent
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                with torch.no_grad():
                    kls.append(((ratio - 1) - (logp - b_logp[idx])).mean().item())
                    clipfracs.append(((ratio - 1).abs() > args.clip).float().mean().item())
                pg_losses.append(pg.item())
                v_losses.append(vl.item())
                ents.append(ent.item())
        t_train = time.time() - t_train
        update += 1
        t_roll_sum += t_roll
        t_train_sum += t_train
        ppo_hist.append({'ent': np.mean(ents), 'kl': np.mean(kls), 'clip': np.mean(clipfracs),
                         'vl': np.mean(v_losses), 'ev': explained_var})

        if update % args.snapshot_every == 0:
            pool.add(model)

        now = time.time()
        sps = total_steps / (now - t_start)
        writer.add_scalar('perf/sps', sps, total_steps)
        writer.add_scalar('loss/policy', np.mean(pg_losses), total_steps)
        writer.add_scalar('loss/value', np.mean(v_losses), total_steps)
        writer.add_scalar('loss/entropy', np.mean(ents), total_steps)
        writer.add_scalar('loss/approx_kl', np.mean(kls), total_steps)
        writer.add_scalar('loss/clipfrac', np.mean(clipfracs), total_steps)
        writer.add_scalar('loss/explained_variance', explained_var, total_steps)
        if episodes:
            recent = list(episodes)[-500:]
            for name, cond in (('selfplay', lambda e: e['mode'] == 'self'),
                               ('with_human', lambda e: e['mode'] == 'mixed' and e['human']),
                               ('mixed_bots', lambda e: e['mode'] == 'mixed' and not e['human'])):
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
            lr_now = optim.param_groups[0]['lr']
            log(f"[{fmt_time(now - t_start)} elapsed | update {update} | {fmt_num(total_steps)} / "
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
            t_last_log = now
            steps_last_log = total_steps
            t_roll_sum = t_train_sum = 0.0

        if update % args.save_every == 0:
            save_policy(os.path.join(run_dir, 'model_latest.pt'), model,
                        {'steps': total_steps, 'update': update, 'args': vars(args)})
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
            ref.load_state_dict(pool.entries[max(0, len(pool.entries) - 9)]['state'])
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
            frames, info = record_video(model, args.video_players, args.video_opponent, seed=update,
                                        frame_skip=args.frame_skip, delay=delay, aggression=args.eval_aggression,
                                        human_args=human_args)
            gif = encode_gif(frames, info['fps'])
            path = os.path.join(run_dir, 'videos', f'round_{update:05d}.gif')
            with open(path, 'wb') as f:
                f.write(gif)
            h, w = frames[0].shape[:2]
            img = Summary.Image(height=h, width=w, colorspace=3, encoded_image_string=gif)
            writer._get_file_writer().add_summary(
                Summary(value=[Summary.Value(tag='video/sample_round', image=img)]), total_steps)
            log(f"  video    {path}: {args.video_players} players (others: {args.video_opponent}), bot ringed "
                f"in white finished #{info['rank']} of {args.video_players} after {info['seconds']:.0f}s "
                f"with {info['kills']} kill(s) and {info['bonuses']} bonus(es) "
                f"| tensorboard: Images tab, video/sample_round")

    save_policy(os.path.join(run_dir, 'model_final.pt'), model,
                {'steps': total_steps, 'update': update, 'args': vars(args)})
    log('=' * 100)
    best = (f"best evaluation score {best_score:+.3f} -> {run_dir}/model_best.pt"
            if np.isfinite(best_score) else 'no evaluation ran')
    log(f"done: {fmt_num(total_steps)} steps in {fmt_time(time.time() - t_start)}, {best}, "
        f"last model {run_dir}/model_final.pt")
    writer.close()


if __name__ == '__main__':
    main()
