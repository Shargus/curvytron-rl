"""Play Curvytron on a real server with a policy trained in curvysim.

The bot mirrors the server state into a one-environment ``SimState`` from the
websocket events (positions, angles, properties, deaths, bonuses), rebuilds
trail bodies exactly like the server does, and feeds the very same
observation function used in training to the policy.

Example (local server started with ``cd curvytron && node bin/curvytron.js``):
    uv run python curvybot.py --model runs/selfplay/model_latest.pt --room botroom
Then open http://127.0.0.1:8080, join room "botroom" and click ready.
"""
import argparse
import json
import math
import threading
import time
from collections import deque

import numpy as np
import torch

import client
from curvysim import core
from curvysim.core import BONUS_NAMES, B_BORDERLESS, B_ESTRAIGHT, MAX_FX, FRAME_MS, NO_MOVE, HIST, RING, board_size
from curvysim.obs import OBS_DIM, MAP_CHANNELS, GLOBAL_CHANNELS, write_obs, write_map, write_global
from curvysim.policy import load_policy

BONUS_INDEX = {name: i for i, name in enumerate(BONUS_NAMES)}


class Mirror:
    """Client-side reconstruction of one round of the server game state."""

    def __init__(self):
        self.S = core.make_state(1, core.MAX_PLAYERS)
        self.slot = {}       # avatar id -> slot
        self.bonus_slot = {}  # bonus id -> slot in S.btype
        self.fx_slot = {}    # bonus id -> slot in S.ftype
        self.angle_seen = set()
        self.present = set()
        self.t0 = None

    # -- helpers -------------------------------------------------------------
    def now(self):
        return 0.0 if self.t0 is None else (time.time() - self.t0) * 1000.0

    def new_game(self, avatar_ids, me):
        ids = sorted(avatar_ids)
        if len(ids) > core.MAX_PLAYERS:   # only mirror MAX_PLAYERS avatars, always including ours
            ids = sorted([me] + [a for a in ids if a != me][: core.MAX_PLAYERS - 1])
        self.slot = {aid: i for i, aid in enumerate(ids)}
        self.present = set(avatar_ids)

    def new_round(self):
        S = self.S
        n = len(self.slot)
        core.reset_env(S, 0, n, np.zeros(core.NB, np.bool_))
        # the server sizes the board for the avatars still present (Game.onStop)
        S.size[0] = min(board_size(max(len(self.present), 1)), core.MAX_SIZE)
        S.next_pop[0] = 1e18   # bonuses come from the server
        for aid, p in self.slot.items():
            S.alive[0, p] = aid in self.present
        self.bonus_slot.clear()
        self.fx_slot.clear()
        self.angle_seen.clear()
        self.t0 = None

    def refresh_flags(self):
        """Derive straight-angle state of each avatar from its active effects."""
        S = self.S
        for p in range(S.n[0]):
            bit = 1 << p
            st = False
            for f in range(MAX_FX):
                if S.ftype[0, f] == B_ESTRAIGHT and S.fmask[0, f] & bit:
                    st = True
            S.straight[0, p] = st

    # -- server events -------------------------------------------------------
    def on_event(self, head, body):
        S = self.S
        if head == 'game:start':
            self.t0 = time.time()
        elif head == 'position':
            aid, x, y = body
            p = self.slot.get(aid)
            if p is None:
                return
            x /= 100.0
            y /= 100.0
            if aid not in self.angle_seen:
                S.turn[0, p] = 0
            self.angle_seen.discard(aid)
            if S.pm_active[0, p]:
                S.since_toggle[0, p] += math.hypot(x - S.x[0, p], y - S.y[0, p])
            S.x[0, p] = x
            S.y[0, p] = y
            if S.alive[0, p] and S.printing[0, p]:
                if not S.has_tl[0, p] or math.hypot(S.tl_x[0, p] - x, S.tl_y[0, p] - y) > S.rad[0, p]:
                    core.add_body(S, 0, p, x, y)
        elif head == 'angle':
            aid, a = body
            p = self.slot.get(aid)
            if p is None:
                return
            a /= 100.0
            if self.t0 is not None:
                d = (a - S.ang[0, p] + math.pi) % (2 * math.pi) - math.pi
                S.turn[0, p] = 0 if abs(d) < 1e-9 else (1 if d > 0 else -1)
                self.angle_seen.add(aid)
            S.ang[0, p] = a
        elif head == 'property':
            aid, prop, value = body
            p = self.slot.get(aid)
            if p is None:
                return
            if prop == 'printing':
                value = bool(value)
                if value != S.printing[0, p]:
                    S.printing[0, p] = value
                    core.add_body(S, 0, p, S.x[0, p], S.y[0, p])
                    if not value:
                        S.has_tl[0, p] = False
                if value:
                    S.pm_active[0, p] = True
                S.since_toggle[0, p] = 0.0
            elif prop == 'radius':
                S.rad[0, p] = float(value)
            elif prop == 'velocity':
                S.vel[0, p] = float(value)
            elif prop == 'inverse':
                S.inv[0, p] = bool(value)
            elif prop == 'invincible':
                S.invinc[0, p] = bool(value)
        elif head == 'die':
            p = self.slot.get(body[0])
            if p is not None and S.alive[0, p]:
                S.alive[0, p] = False
                S.deaths[0] += 1
                core.add_body(S, 0, p, S.x[0, p], S.y[0, p])
                S.pm_active[0, p] = False
        elif head == 'bonus:pop':
            bid, x, y, name = body
            free = np.nonzero(S.btype[0] < 0)[0]
            if name in BONUS_INDEX and len(free):
                b = free[0]
                S.bx[0, b] = x / 100.0
                S.by[0, b] = y / 100.0
                S.btype[0, b] = BONUS_INDEX[name]
                self.bonus_slot[bid] = b
        elif head == 'bonus:clear':
            b = self.bonus_slot.pop(body, None)
            if b is not None:
                S.btype[0, b] = -1
        elif head == 'bonus:stack':
            aid, method, bid, name, duration = body
            p = self.slot.get(aid)
            if p is None or name not in BONUS_INDEX:
                return
            f = self.fx_slot.get(bid)
            if method == 'add':
                if f is None:
                    free = np.nonzero(S.ftype[0] < 0)[0]
                    if not len(free):
                        return
                    f = free[0]
                    self.fx_slot[bid] = f
                    S.ftype[0, f] = BONUS_INDEX[name]
                    S.fmask[0, f] = 0
                    S.fend[0, f] = self.now() + float(duration)
                S.fmask[0, f] |= 1 << p
            elif f is not None:
                S.fmask[0, f] &= ~(1 << p)
                if S.fmask[0, f] == 0:
                    S.ftype[0, f] = -1
                    del self.fx_slot[bid]
            self.refresh_flags()
        elif head == 'borderless':
            for f in range(MAX_FX):
                if S.ftype[0, f] == B_BORDERLESS:
                    S.ftype[0, f] = -1
            if body:
                f = int(np.nonzero(S.ftype[0] < 0)[0][0])
                S.ftype[0, f] = B_BORDERLESS
                S.fmask[0, f] = 0
                S.fend[0, f] = self.now() + core.B_DURATION[B_BORDERLESS]
        elif head == 'clear':
            core.clear_world(S, 0)
        elif head == 'game:leave':
            self.present.discard(body)
            p = self.slot.get(body)
            if p is not None and S.alive[0, p]:
                S.alive[0, p] = False

    def set_latency(self, p, rtt_ms, sent, decisions):
        """Rebuild the latency model of the simulator for our avatar: delay in frames, the
        moves already sent that the server has not applied yet (as of the state we see,
        which is ~rtt/2 old, while our moves need ~rtt/2 to arrive), and the last decisions."""
        S = self.S
        now = time.time()
        d = int(min(max(round(rtt_ms / FRAME_MS), 0), RING - 4))
        S.delay[0, p] = d
        S.frames[0] = 0
        S.sched[0, p, :] = NO_MOVE
        applied = 0
        for ts, m in sent:
            age = (now - ts) * 1000.0
            if age >= rtt_ms:
                applied = m
            else:
                k = int(min(max((rtt_ms - age) / FRAME_MS, 0), max(d - 1, 0)))
                S.sched[0, p, k] = m
        S.move[0, p] = applied
        S.hist[0, p, :] = 0
        for k, m in enumerate(reversed(decisions)):
            if k >= HIST:
                break
            S.hist[0, p, k] = m


class CurvyBot(client.CurvytronClient):
    def __init__(self, model, decide_every=2, greedy=False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.decide_every = decide_every
        self.greedy = greedy
        self.mirror = Mirror()
        self.in_round = False
        self.frames = 0
        self.obs = np.zeros(OBS_DIM, np.float32)
        m = model.map_size
        self.map = np.zeros((MAP_CHANNELS, m, m), np.uint8) if m > 0 else None
        self.gmap = (np.zeros((GLOBAL_CHANNELS, core.GRID, core.GRID), np.uint8)
                     if model.global_map else None)
        self.game_over = threading.Event()
        self.lock = threading.Lock()
        self.wins = 0
        self.rounds = 0
        self.rtt_ms = 50.0          # updated from the server's 'latency' events
        self.extra_delay_ms = 0.0   # processing / decision-interval allowance
        self.sent = deque(maxlen=64)
        self.decisions = deque(maxlen=HIST)

    def run(self):
        try:
            super().run()
        except OSError:
            if self.alive.is_set():   # errors after join() are just the socket closing
                raise

    def _process_recvd(self, recvd):
        moved = False
        with self.lock:
            for message in json.loads(recvd):
                head = message[0]
                body = message[1] if len(message) > 1 else None
                if head == 'room:game:start':
                    super()._parse_message(message)
                    self.mirror.new_game(list(self.game.players.keys()), self.player_id)
                elif head == 'round:new':
                    super()._parse_message(message)
                    self.mirror.new_round()
                    self.in_round = False
                    self.frames = 0
                    self.sent.clear()
                    self.decisions.clear()
                    self.last_action = 0
                elif head == 'latency':
                    self.rtt_ms = 0.7 * self.rtt_ms + 0.3 * float(body)
                elif head == 'game:leave':
                    self.mirror.on_event(head, body)
                elif head == 'game:start':
                    super()._parse_message(message)
                    self.mirror.on_event(head, body)
                    self.in_round = True
                elif head == 'position':
                    self.mirror.on_event(head, body)
                    if body[0] == self.player_id:
                        moved = True
                elif head in ('angle', 'property', 'die', 'bonus:pop', 'bonus:clear',
                              'bonus:stack', 'borderless', 'clear'):
                    if head == 'die':
                        super()._parse_message(message)
                    self.mirror.on_event(head, body)
                elif head == 'round:end':
                    self.in_round = False
                    self.rounds += 1
                    self.wins += int(body == self.player_id)
                    print(f'round {self.rounds}: {"won" if body == self.player_id else "lost"} '
                          f'after {self.mirror.now() / 1000:.1f}s ({self.wins}/{self.rounds} won, '
                          f'latency {self.rtt_ms:.0f} ms)', flush=True)
                elif head == 'end':
                    super()._parse_message(message)
                    self.in_round = False
                    self.game_over.set()
                else:
                    super()._parse_message(message)
            if moved and self.in_round:
                self.frames += 1
                if self.frames % self.decide_every == 0:
                    self.decide()

    def decide(self):
        S = self.mirror.S
        p = self.mirror.slot.get(self.player_id)
        if p is None or not S.alive[0, p]:
            return
        S.t[0] = self.mirror.now()
        self.mirror.set_latency(p, self.rtt_ms + self.extra_delay_ms, self.sent, self.decisions)
        write_obs(S, 0, p, self.obs)
        maps = None
        if self.map is not None:
            write_map(S, 0, p, self.map)
            maps = torch.from_numpy(self.map)[None]
        gmaps = None
        if self.gmap is not None:
            write_global(S, 0, p, self.gmap)
            gmaps = torch.from_numpy(self.gmap)[None]
        with torch.no_grad():
            a = self.model.act(torch.from_numpy(self.obs)[None], maps, gmaps, greedy=self.greedy).item()
        move = a - 1
        self.decisions.append(move)
        if move != self.last_action:
            self._send_message(self.PLAYER_MOVE, {'action': move})
            self.sent.append((time.time(), move))
            self.last_action = move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--server', default='127.0.0.1:8080')
    ap.add_argument('--room', default='botroom')
    ap.add_argument('--name', default='rlbot')
    ap.add_argument('--color', default='#00ffcc')
    ap.add_argument('--bonuses', default='all', help="preset name (all, none, speed, ...) when creating the room")
    ap.add_argument('--max-score', type=int, default=10)
    ap.add_argument('--decide-every', type=int, default=None,
                    help='server frames per decision (default: the frame skip used in training)')
    ap.add_argument('--greedy', action='store_true', help='argmax actions instead of sampling')
    ap.add_argument('--games', type=int, default=1, help='games to play before exiting (0 = forever)')
    ap.add_argument('--wait-players', type=int, default=2,
                    help='only get ready once the room has this many players')
    args = ap.parse_args()

    torch.set_num_threads(1)
    model, ckpt = load_policy(args.model)
    decide_every = args.decide_every or ckpt.get('args', {}).get('frame_skip', 3)
    bot = CurvyBot(model, decide_every=decide_every, greedy=args.greedy,
                   name=args.name, color=args.color)
    # compile the numba observation code before the game starts (first call takes ~1 min)
    warm = Mirror()
    warm.new_game([0, 1], 0)
    warm.new_round()
    warm.set_latency(0, 50.0, [], [])
    write_obs(warm.S, 0, 0, bot.obs)
    if bot.map is not None:
        write_map(warm.S, 0, 0, bot.map)
    if bot.gmap is not None:
        write_global(warm.S, 0, 0, bot.gmap)
    bot.start()
    bot.connect_to_server(args.server)
    time.sleep(1)
    bot.join_room(args.room, on_bonuses=None if args.bonuses == 'none' else args.bonuses,
                  max_score=args.max_score)
    games = 0
    try:
        while args.games == 0 or games < args.games:
            bot.game_over.clear()
            while len(bot.game.players) < args.wait_players:
                time.sleep(0.2)
            bot.send_ready()
            print(f'ready in room "{args.room}" on {args.server}, waiting for the game...', flush=True)
            bot.game_over.wait()
            games += 1
            time.sleep(2)
    finally:
        bot.join(1)


if __name__ == '__main__':
    main()
