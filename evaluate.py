"""Evaluate a trained policy in the simulator.

Examples:
    uv run python evaluate.py --model runs/pilot/model_latest.pt                      # vs heuristic
    uv run python evaluate.py --model runs/big/model_latest.pt --opponent runs/pilot/model_latest.pt
    uv run python evaluate.py --model runs/big/model_latest.pt --players 4 --bonus-modes 0,1,0
"""
import argparse

import torch

from curvysim.policy import load_policy
from train_selfplay import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--opponent', default='heuristic', help="'heuristic' or a checkpoint path")
    ap.add_argument('--players', type=int, default=2)
    ap.add_argument('--rounds', type=int, default=1000)
    ap.add_argument('--bonus-modes', default='0.3,0.4,0.3', help='P(no bonus), P(all), P(random subset)')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--delay', default='0,8', help='round-trip latency range in frames')
    args = ap.parse_args()

    model, ckpt = load_policy(args.model)
    frame_skip = ckpt.get('args', {}).get('frame_skip', 3)
    opponent = args.opponent if args.opponent == 'heuristic' else load_policy(args.opponent)[0]
    modes = tuple(float(v) for v in args.bonus_modes.split(','))
    ret, win = evaluate(model, opponent, args.rounds, seed=args.seed, n_players=args.players,
                        bonus_modes=modes, delay=tuple(int(v) for v in args.delay.split(',')),
                        frame_skip=frame_skip)
    print(f'{args.model} vs {args.opponent} ({args.players} players, {args.rounds} rounds): '
          f'survives the round {win:.3f}, mean round return {ret:+.3f} (+1 = always last alive, '
          f'-1 = always first out, 0 = average placement)')


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
