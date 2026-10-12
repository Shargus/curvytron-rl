"""Batched multi-agent Curvytron environment built on the numba simulator.

Each of the ``E`` environments runs one round at a time with ``n`` players
(sampled per round). Every player slot is an agent; agents are indexed
``[e, p]`` with ``p < P``. A round ends when at most one avatar is alive, and the
environment then auto-resets with a freshly sampled configuration.

Human-like players (``S.human``) get handicapped controls: a longer reaction delay
(set by the caller), occasional steering noise and, after a malus changes their
controls (inverted controls, 90 degree turns, extra speed or size, borderless
board), a few seconds of confusion during which they often press the wrong way.

Reward follows Curvytron round scoring: an avatar dying k-th gets k points and
the survivor gets n-1. The per-agent return is ``2 * points / (n - 1) - 1``
(so +1 / -1 in a duel), paid incrementally as opponents die and settled when
the agent's round ends.
"""
import math

import numba as nb
import numpy as np

from . import core
from .core import (NB, FRAME_MS, MAX_PLAYERS, make_state, reset_env, step_frame, potential_score,
                   queue_move, apply_queued, borderless)

# confusion of human-like players when a malus changes their controls:
# (flip = press the opposite direction instead of a random one, error probability right after the
#  change (fading linearly to 0), duration in ms); scaled by ``human_confusion``
CONF_INVERSE = (True, 0.45, 2000.0)     # inverted controls start or end
CONF_STRAIGHT = (False, 0.3, 1500.0)    # 90 degree turns start or end
CONF_FAST = (False, 0.25, 1500.0)       # sudden speed-up
CONF_BIG = (False, 0.15, 1000.0)        # becomes bigger
CONF_BORDERLESS = (False, 0.2, 1500.0)  # walls appear / disappear


@nb.njit(cache=True)
def _confuse(S, e, p, flip, p0, dur, scale):
    """Start a confusion period unless a stronger one is still going on."""
    p0 *= scale
    t = S.t[e]
    current = 0.0
    if t < S.conf_end[e, p]:
        current = S.conf_p0[e, p] * (S.conf_end[e, p] - t) / S.conf_dur[e, p]
    if p0 >= current:
        S.conf_flip[e, p] = flip
        S.conf_p0[e, p] = p0
        S.conf_end[e, p] = t + dur
        S.conf_dur[e, p] = dur


@nb.njit(cache=True)
def _humanize(S, e, p, move, noise):
    """Steering errors of a human-like player: random noise, plus confusion after a malus."""
    if np.random.random() < noise:
        move = np.random.randint(-1, 2)
    t = S.t[e]
    if t < S.conf_end[e, p]:
        pr = S.conf_p0[e, p] * (S.conf_end[e, p] - t) / S.conf_dur[e, p]
        if np.random.random() < pr:
            if S.conf_flip[e, p]:
                move = -move if move != 0 else (1 if np.random.random() < 0.5 else -1)
            else:
                move = np.random.randint(-1, 2)
    return move
from .obs import OBS_DIM, MAP_CHANNELS, GLOBAL_CHANNELS, compute_obs, compute_maps, compute_globals, heuristic_actions


@nb.njit(cache=True)
def _sample_config(n_cdf, bonus_mode_cdf, bonus_p, enabled):
    u = np.random.random()
    n = 2
    for i in range(n_cdf.shape[0]):
        if u < n_cdf[i]:
            n = i + 2
            break
    u = np.random.random()
    if u < bonus_mode_cdf[0]:          # no bonuses
        for t in range(NB):
            enabled[t] = False
    elif u < bonus_mode_cdf[1]:        # all bonuses (server default)
        for t in range(NB):
            enabled[t] = True
    else:                              # random subset
        for t in range(NB):
            enabled[t] = np.random.random() < bonus_p
    return n


@nb.njit(cache=True)
def _new_round(S, e, n_cdf, bonus_mode_cdf, bonus_p, delay_lo, delay_hi, kill_lo, kill_hi, prev_ps, P):
    enabled = np.zeros(NB, np.bool_)
    n = min(_sample_config(n_cdf, bonus_mode_cdf, bonus_p, enabled), P)
    reset_env(S, e, n, enabled)
    for p in range(P):
        prev_ps[e, p] = 0.0
        # every player has its own network latency (round trip, in frames)
        S.delay[e, p] = np.random.randint(delay_lo, delay_hi + 1)
        # aggression: this player's bonus for each opponent killed on its trail (also observed)
        S.aggr[e, p] = kill_lo + np.random.random() * (kill_hi - kill_lo)


@nb.njit(parallel=True, cache=True)
def _reset_all(S, n_cdf, bonus_mode_cdf, bonus_p, delay_lo, delay_hi, kill_lo, kill_hi, prev_ps, P):
    E = S.size.shape[0]
    for e in nb.prange(E):
        _new_round(S, e, n_cdf, bonus_mode_cdf, bonus_p, delay_lo, delay_hi, kill_lo, kill_hi, prev_ps, P)


@nb.njit(parallel=True, cache=True)
def _step_all(S, actions, frame_skip, delay_lo, delay_hi, kill_lo, kill_hi, jitter, max_t, n_cdf,
              bonus_mode_cdf, bonus_p, prev_ps, rew, kill_rew, kills, done, cause, pickups, reset_flag, P,
              human_noise, human_confusion):
    E = S.size.shape[0]
    for e in nb.prange(E):
        n = S.n[e]
        alive0 = np.zeros(P, np.bool_)
        inv0 = np.zeros(P, np.bool_)
        straight0 = np.zeros(P, np.bool_)
        vel0 = np.zeros(P)
        rad0 = np.zeros(P)
        for p in range(n):
            alive0[p] = S.alive[e, p]
            inv0[p] = S.inv[e, p]
            straight0[p] = S.straight[e, p]
            vel0[p] = S.vel[e, p]
            rad0[p] = S.rad[e, p]
        bless0 = borderless(S, e)
        for p in range(n):
            if S.alive[e, p]:
                move = actions[e, p]
                if S.human[e, p]:
                    move = _humanize(S, e, p, move, human_noise)
                # latency jitter: occasionally one frame late
                queue_move(S, e, p, move, 1 if np.random.random() < 0.2 else 0)
        for f in range(frame_skip):
            apply_queued(S, e)
            dt = FRAME_MS + (np.random.random() * 2.0 - 1.0) * jitter
            step_frame(S, e, dt)
            if S.over[e]:
                break
        # human-like players get confused when a malus changes how they steer
        bless1 = borderless(S, e)
        for p in range(n):
            if not (S.human[e, p] and S.alive[e, p]):
                continue
            if S.inv[e, p] != inv0[p]:
                _confuse(S, e, p, CONF_INVERSE[0], CONF_INVERSE[1], CONF_INVERSE[2], human_confusion)
            if S.straight[e, p] != straight0[p]:
                _confuse(S, e, p, CONF_STRAIGHT[0], CONF_STRAIGHT[1], CONF_STRAIGHT[2], human_confusion)
            if S.vel[e, p] > vel0[p] + 1.0:
                _confuse(S, e, p, CONF_FAST[0], CONF_FAST[1], CONF_FAST[2], human_confusion)
            if S.rad[e, p] > rad0[p] + 1e-6:
                _confuse(S, e, p, CONF_BIG[0], CONF_BIG[1], CONF_BIG[2], human_confusion)
            if bless1 != bless0:
                _confuse(S, e, p, CONF_BORDERLESS[0], CONF_BORDERLESS[1], CONF_BORDERLESS[2], human_confusion)
        if not S.over[e] and S.t[e] >= max_t:
            S.over[e] = True          # timeout: draw, nobody gets the survivor bonus
            for p in range(n):
                if S.alive[e, p]:
                    S.score[e, p] = S.deaths[e]
        scale = 2.0 / max(n - 1, 1)
        for p in range(P):
            kill_rew[e, p] = 0.0
            kills[e, p] = 0
        for p in range(n):   # kill bonus to the owner of the trail hit, if still in its round
            if alive0[p] and not S.alive[e, p]:
                q = S.killer[e, p]
                if q >= 0 and q != p and alive0[q]:
                    kill_rew[e, q] += S.aggr[e, q]
                    kills[e, q] += 1
        for p in range(P):
            rew[e, p] = 0.0
            done[e, p] = False
            cause[e, p] = -1
            if p < n and alive0[p]:
                ps = potential_score(S, e, p)
                rew[e, p] = (ps - prev_ps[e, p]) * scale
                prev_ps[e, p] = ps
                if S.over[e] or not S.alive[e, p]:
                    done[e, p] = True
                    cause[e, p] = S.cause[e, p]   # -1: survived the round (or timeout)
                    for t in range(NB):           # bonuses picked up during the round
                        pickups[e, p, t] = S.pickups[e, p, t]
                    rew[e, p] -= 1.0
        reset_flag[e] = S.over[e]
        if S.over[e]:
            _new_round(S, e, n_cdf, bonus_mode_cdf, bonus_p, delay_lo, delay_hi, kill_lo, kill_hi, prev_ps, P)


class BatchEnv:
    """Vectorised self-play environment.

    Args:
        num_envs: number of parallel rounds.
        max_players: player slots per env (P). Rounds use 2..P players.
        n_probs: probabilities for 2, 3, ..., P players per round.
        bonus_modes: probabilities of (no bonus, all bonuses, random subset).
        frame_skip: server frames (1/60 s) per decision.
        delay: (lo, hi) round-trip latency in frames, sampled per player and round. A move
            chosen from the current state reaches the server that many frames later.
        kill_reward: (lo, hi) range of the per-kill bonus ("aggression"), sampled per player
            and round and observed by that player. Paid separately in ``self.kill_rew``.
        human_delay: (lo, hi) reaction delay in frames of human-like players (see make_human).
        human_noise: per-decision probability that a human-like player steers randomly.
        human_confusion: scale of the confusion after a malus (0 = none, 1 = default table).
        jitter_ms: uniform jitter on the frame duration (the server uses wall-clock steps).
        map_size: if > 0, also build an egocentric occupancy map of map_size x map_size cells
            (``self.maps``, uint8 [E, P, MAP_CHANNELS, M, M]) for a CNN policy.
        global_map: also build the whole-board map (``self.gmaps``, uint8
            [E, P, GLOBAL_CHANNELS, GRID, GRID]).
    """

    def __init__(self, num_envs, max_players=12, n_probs=None, bonus_modes=(0.3, 0.4, 0.3),
                 bonus_p=0.5, frame_skip=2, delay=(0, 8), kill_reward=(0.0, 1.0),
                 human_delay=(9, 18), human_noise=0.05, human_confusion=1.0, jitter_ms=1.0, max_round_s=150.0, seed=0,
                 map_size=0, global_map=False):
        assert 2 <= max_players <= MAX_PLAYERS
        self.E = num_envs
        self.P = max_players
        if n_probs is None:
            n_probs = np.ones(max_players - 1)
        n_probs = np.asarray(n_probs, np.float64)
        n_probs = n_probs / n_probs.sum()
        self.n_cdf = np.cumsum(n_probs)
        self.n_cdf[-1] = 1.0 + 1e-9
        bm = np.asarray(bonus_modes, np.float64)
        bm = bm / bm.sum()
        self.bonus_mode_cdf = np.cumsum(bm)
        self.bonus_p = float(bonus_p)
        self.frame_skip = int(frame_skip)
        self.delay_lo, self.delay_hi = int(delay[0]), int(delay[1])
        assert 0 <= self.delay_lo <= self.delay_hi < core.RING - 2
        self.kill_lo, self.kill_hi = float(kill_reward[0]), float(kill_reward[1])
        self.human_delay = (int(human_delay[0]), int(human_delay[1]))
        assert 0 <= self.human_delay[0] <= self.human_delay[1] < core.RING - 2
        self.human_noise = float(human_noise)
        self.human_confusion = float(human_confusion)
        self.jitter = float(jitter_ms)
        self.max_t = float(max_round_s) * 1000.0
        self.S = make_state(num_envs, max_players)
        self.prev_ps = np.zeros((num_envs, max_players))
        self.rew = np.zeros((num_envs, max_players), np.float32)
        self.done = np.zeros((num_envs, max_players), np.bool_)
        self.kill_rew = np.zeros((num_envs, max_players), np.float32)   # kill bonus earned this step
        self.kills = np.zeros((num_envs, max_players), np.int32)        # kills made this step
        self.pickups = np.zeros((num_envs, max_players, NB), np.int32)  # bonuses of agents done this step
        self.cause = np.full((num_envs, max_players), -1, np.int8)   # death cause of agents done this step
        self.reset_flag = np.zeros(num_envs, np.bool_)
        self.obs = np.zeros((num_envs, max_players, OBS_DIM), np.float32)
        self.map_size = int(map_size)
        self.maps = (np.zeros((num_envs, max_players, MAP_CHANNELS, map_size, map_size), np.uint8)
                     if map_size > 0 else None)
        self.gmaps = (np.zeros((num_envs, max_players, GLOBAL_CHANNELS, core.GRID, core.GRID), np.uint8)
                      if global_map else None)
        core.seed(seed)

    @property
    def alive(self):
        """Agents that act in the next step: present and alive."""
        n = self.S.n[:, None]
        return self.S.alive[:, : self.P] & (np.arange(self.P)[None, :] < n)

    def reset(self):
        _reset_all(self.S, self.n_cdf, self.bonus_mode_cdf, self.bonus_p, self.delay_lo, self.delay_hi,
                   self.kill_lo, self.kill_hi, self.prev_ps, self.P)
        return self.observe()

    def observe(self):
        alive = self.alive
        compute_obs(self.S, self.obs, alive)
        if self.maps is not None:
            compute_maps(self.S, self.maps, alive)
        if self.gmaps is not None:
            compute_globals(self.S, self.gmaps, alive)
        return self.obs

    def step(self, actions):
        """actions: int array [E, P] of moves in {-1, 0, 1}. Returns (obs, rew, done, reset_flag).

        ``rew`` is the placement reward only; the kill bonus of the same step is in ``self.kill_rew``."""
        actions = np.ascontiguousarray(actions, dtype=np.int64)
        _step_all(self.S, actions, self.frame_skip, self.delay_lo, self.delay_hi, self.kill_lo, self.kill_hi,
                  self.jitter, self.max_t, self.n_cdf, self.bonus_mode_cdf, self.bonus_p, self.prev_ps,
                  self.rew, self.kill_rew, self.kills, self.done, self.cause, self.pickups, self.reset_flag,
                  self.P, self.human_noise, self.human_confusion)
        return self.observe(), self.rew, self.done, self.reset_flag

    def heuristic(self, mask, eps=0.3):
        out = np.zeros((self.E, self.P), np.int64)
        heuristic_actions(self.S, out, mask, eps)
        return out

    def make_human(self, e, p, human=True):
        """Mark slot p of env e as human-like (or bot-like) for the current round; human-like
        players also get a human reaction delay. Call after a reset, before stepping."""
        self.S.human[e, p] = human
        if human:
            self.S.delay[e, p] = np.random.randint(self.human_delay[0], self.human_delay[1] + 1)
