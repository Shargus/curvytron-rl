"""Self-play PPO for Curvytron on the fast numba simulator (curvysim).

Each environment runs one round with 2..P players. Per round, the opponents of
the learner are either the current policy (pure self-play: every slot produces
training data), a frozen snapshot from the opponent pool, or the lookahead
heuristic. Pool opponents are sampled with prioritized fictitious self-play
(snapshots the learner struggles against are picked more often).

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

from curvysim import BatchEnv, OBS_DIM
from curvysim.policy import ActorCritic, save_policy, load_policy

SELF, HEURISTIC = -1, -2


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
    p.add_argument('--hidden', default='512,512,256')
    p.add_argument('--p-self', type=float, default=0.6)
    p.add_argument('--p-heuristic', type=float, default=0.1)
    p.add_argument('--snapshot-every', type=int, default=25, help='updates between pool snapshots')
    p.add_argument('--pool-size', type=int, default=50)
    p.add_argument('--active-opponents', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-rounds', type=int, default=300)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--log-every', type=float, default=60.0, help='seconds between progress reports')
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


@torch.no_grad()
def evaluate(model, opponent, rounds, seed, n_players=2, bonus_modes=(0.3, 0.4, 0.3), num_envs=128,
             delay=(0, 8), frame_skip=3):
    """Results of ``model`` (one random slot) vs ``opponent`` (all other slots).

    opponent: an ActorCritic or the string 'heuristic'. Returns (mean return, win rate).
    """
    probs = [0.0] * (n_players - 1)
    probs[-1] = 1.0
    map_size = max(model.map_size, getattr(opponent, 'map_size', 0))
    global_map = model.global_map or getattr(opponent, 'global_map', False)
    env = BatchEnv(num_envs, max_players=n_players, n_probs=probs, bonus_modes=bonus_modes,
                   seed=seed, map_size=map_size, global_map=global_map, delay=delay,
                   frame_skip=frame_skip)
    env.reset()
    ret = np.zeros(num_envs)
    results = []
    rng = np.random.default_rng(seed)
    # the server updates avatars in reverse join order, which matters in head-to-head
    # encounters, so the evaluated policy plays from a random slot
    my_slot = rng.integers(0, n_players, num_envs)
    slots = np.arange(env.P)[None, :]
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
        for e in np.nonzero(reset)[0]:
            results.append(ret[e])
            ret[e] = 0.0
            my_slot[e] = rng.integers(0, n_players)
    results = np.array(results[:rounds])
    return float(results.mean()), float((results > 0.99).mean())   # win = last survivor


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
    """Summary lines over recent learner rounds. Each episode is a dict with
    ctrl, ret, len (decisions), n (players) and cause (-1 survived, 0 wall, 1 own, 2 other)."""
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
    lines.append(f"  vs self      {res(lambda e: e['ctrl'] == SELF)}")
    lines.append(f"  vs heuristic {res(lambda e: e['ctrl'] == HEURISTIC)}")
    lines.append(f"  vs pool      {res(lambda e: e['ctrl'] >= 0)}")
    lines.append('  by size  ' + ' | '.join(
        f"{lo}-{hi}p ret {np.mean(r):+.3f}" if len(r := arr('ret', lambda e, lo=lo, hi=hi: lo <= e['n'] <= hi))
        else f"{lo}-{hi}p n/a" for lo, hi in ((2, 2), (3, 6), (7, 16))))
    c = arr('cause')
    died = c[c >= 0]
    if len(died):
        lines.append(f"  deaths   wall {np.mean(died == 0) * 100:4.1f}% | own trail {np.mean(died == 1) * 100:4.1f}% | "
                     f"others' trails {np.mean(died == 2) * 100:4.1f}%   (of {len(died)} deaths)")
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
    env = BatchEnv(args.num_envs, max_players=args.max_players, n_probs=n_probs,
                   bonus_modes=bonus_modes, frame_skip=args.frame_skip,
                   delay=delay, seed=args.seed, map_size=args.map_size, global_map=args.global_map)
    E, P, T, M = env.E, env.P, args.rollout, args.map_size

    pool = OpponentPool(args.pool_size, args.active_opponents, make_model)
    ctrl = np.full(E, SELF, np.int64)
    learner_slot = np.zeros(E, np.int64)
    slot_idx = np.arange(P)[None, :]

    def assign(envs):
        keys = list(pool.active.keys())
        for e in envs:
            u = random.random()
            if u < args.p_self:
                c = SELF
            elif u < args.p_self + args.p_heuristic:
                c = HEURISTIC
            else:
                c = random.choice(keys) if keys else SELF
            ctrl[e] = c
            learner_slot[e] = random.randrange(int(env.S.n[e]))

    def learner_mask(alive):
        return alive & ((ctrl[:, None] == SELF) | (slot_idx == learner_slot[:, None]))

    n_params = sum(p.numel() for p in model.parameters())
    dev_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else f'cpu ({args.threads} threads)'
    log('=' * 100)
    log(f"Curvytron self-play PPO | run '{args.run_name}' | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  device      {dev_name}")
    log(f"  model       {n_params / 1e6:.2f}M parameters | features {OBS_DIM} | hidden {hidden}"
        f" | local map {f'{M}x{M}' if M else 'off'} | whole-board map {'on' if args.global_map else 'off'}")
    log(f"  envs        {E} parallel rounds, 2-{P} players, decision every {args.frame_skip} frames "
        f"({args.frame_skip * 1000 / 60:.0f} ms), latency {delay[0]}-{delay[1]} frames")
    log(f"  opponents   self {args.p_self:.0%} | heuristic {args.p_heuristic:.0%} | "
        f"pool {max(0.0, 1 - args.p_self - args.p_heuristic):.0%} (snapshot every {args.snapshot_every} updates)")
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
    episodes = deque(maxlen=5000)   # recent learner rounds, for logs
    rounds_done = 0
    ppo_hist = deque(maxlen=20)     # recent per-update PPO diagnostics

    total_steps = 0
    update = 0
    t_start = time.time()
    t_last_log = t_start
    steps_last_log = 0
    t_roll_sum = t_train_sum = 0.0
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
            heur = others & (ctrl[:, None] == HEURISTIC)
            if heur.any():
                actions[heur] = env.heuristic(heur, eps=0.3)[heur]
            for k, m in pool.active.items():
                sel = others & (ctrl[:, None] == k)
                if sel.any():
                    actions[sel] = policy_moves(m, env, sel)
            _, rew, done, reset = env.step(actions)
            buf_rew[t] = rew
            buf_done[t] = done
            # episode stats for learner agents
            ep_ret += np.where(learner, rew, 0.0)
            ep_len += learner
            for e, p in zip(*np.nonzero(done & learner)):
                c = int(ctrl[e])
                episodes.append({'ctrl': c, 'ret': ep_ret[e, p], 'len': ep_len[e, p],
                                 'n': int(n_round[e]), 'cause': int(env.cause[e, p])})
                if c >= 0:
                    pool.record(c, (ep_ret[e, p] + 1.0) / 2.0)   # placement in [0, 1]
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
        b_obs = torch.from_numpy(np.concatenate(buf_obs)).to(device)
        b_map = torch.from_numpy(np.concatenate(buf_map)).to(device) if M > 0 else None
        b_gmap = torch.from_numpy(np.concatenate(buf_gmap)).to(device) if args.global_map else None
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
            perm = torch.randperm(N, device=device)
            for s in range(0, N, args.minibatch):
                idx = perm[s:s + args.minibatch]
                logits, value = model(b_obs[idx], b_map[idx] if M > 0 else None,
                                      b_gmap[idx] if args.global_map else None)
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
            for name, cond in (('self', lambda e: e['ctrl'] == SELF), ('heur', lambda e: e['ctrl'] == HEURISTIC),
                               ('pool', lambda e: e['ctrl'] >= 0)):
                r = [e['ret'] for e in recent if cond(e)]
                if r:
                    writer.add_scalar(f'train/ret_{name}', np.mean(r), total_steps)
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
            for n_eval in (2, 4, 8):
                ev[f'ret_heur_{n_eval}p'], ev[f'win_heur_{n_eval}p'] = evaluate(
                    model, 'heuristic', args.eval_rounds, seed=update + n_eval, n_players=n_eval, delay=delay,
                    frame_skip=args.frame_skip)
            ref = make_model()
            ref.load_state_dict(pool.entries[max(0, len(pool.entries) - 9)]['state'])
            ref.eval()
            ev['ret_past_6p'], ev['win_past_6p'] = evaluate(
                model, ref, args.eval_rounds, seed=update + 1, n_players=6, delay=delay,
                frame_skip=args.frame_skip)
            # one number to track: mean round return vs the heuristic over 2, 4 and 8 players
            ev['score'] = float(np.mean([ev[f'ret_heur_{n}p'] for n in (2, 4, 8)]))
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
            log("  (ret: +1 = always last alive, 0 = average placement, -1 = always first out)")
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

    save_policy(os.path.join(run_dir, 'model_final.pt'), model,
                {'steps': total_steps, 'update': update, 'args': vars(args)})
    log('=' * 100)
    log(f"done: {fmt_num(total_steps)} steps in {fmt_time(time.time() - t_start)}, best evaluation score "
        f"{best_score:+.3f} -> {run_dir}/model_best.pt, last model {run_dir}/model_final.pt")
    writer.close()


if __name__ == '__main__':
    main()
