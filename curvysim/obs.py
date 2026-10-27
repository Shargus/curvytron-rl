"""Egocentric observation features and a search-based heuristic policy.

Both work on a ``SimState`` (see core.py). The deployment client rebuilds a
``SimState`` for a single env from websocket messages, so the exact same code
produces observations online.

Observation layout (all roughly in [-1, 2]):
    rays      N_RAYS x 3   : distance/RAY_MAX, min(distance, 8)/8, hit-is-wall
    opponents K_OPP x 13   : nearest alive opponents (relative, egocentric)
    bonuses   K_BON x F_BON: every bonus on the map, nearest first: straight-line distance and
                           direction, grid path around trails (reachable, length, reached
                           before every opponent) and one-hot type
    self      N_SELF       : own state, effects, timers, latency, aggression, last moves sent
    lookahead N_LOOK       : survival of each move under a short tree search (2 horizons)
    territory N_TERR       : share of the board reachable first, now and after each move
    humans    N_HUMAN      : which of the nearest opponents are human-like, share of human-like
                           opponents alive (last, so older models can be extended with zero weights)
"""
import math

import numba as nb
import numpy as np

from .core import (
    NB, B_DURATION, B_BORDERLESS, BASE_VEL, BASE_R, PRINT_DELAY, RASTER_RES, RASTER_G,
    MAX_FX, BONUS_CAP, FRAME_MS, TRAIL_LATENCY, HASH_CELL, HASH_G, MAX_SIZE, MAX_PLAYERS,
    RING, NO_MOVE, HIST, GRID, B_AFFECT, AFF_SELF, AFF_ENEMY, BONUS_R, base_angular_velocity, _clampi,
)

N_RAYS = 64
RAY_MAX = 64.0
RAY_NEAR = 8.0
K_OPP = 7
F_OPP = 13
K_BON = BONUS_CAP   # every bonus on the map (the server caps them at 20), nearest first
F_BON = 7 + NB      # valid, distance, direction (2), path: reachable, length, first; type one-hot
N_SELF = 23 + NB + 2 + 3 * HIST
N_LOOK = 12        # lookahead search features (see lookahead_features)
N_TERR = 12        # territory features (see territory_features)
N_HUMAN = K_OPP + 1   # opponents flagged human-like (same order as the opponent block) + share
OBS_DIM = N_RAYS * 3 + K_OPP * F_OPP + K_BON * F_BON + N_SELF + N_LOOK + N_TERR + N_HUMAN
O_TERR = OBS_DIM - N_HUMAN - N_TERR   # territory block: [0] share of free board reached first
OWN_SKIP = TRAIL_LATENCY + 2   # newest own bodies ignored by the rays
HEAD_MARGIN = 2.0              # heuristic: clearance kept from opponents' heads


@nb.njit(cache=True)
def _cast_ray(S, e, p, ox, oy, dx, dy, bless):
    """March along the observation raster. Returns (distance, hit_wall)."""
    size = S.size[e]
    own_min = S.bc[e, p] - OWN_SKIP
    tmax = RAY_MAX
    wall = False
    if not bless:
        tw = 1e9
        if dx > 1e-9:
            tw = min(tw, (size - ox) / dx)
        elif dx < -1e-9:
            tw = min(tw, -ox / dx)
        if dy > 1e-9:
            tw = min(tw, (size - oy) / dy)
        elif dy < -1e-9:
            tw = min(tw, -oy / dy)
        if tw < tmax:
            tmax = max(tw, 0.0)
            wall = True
    t = 0.0
    step = RASTER_RES
    gmax = int(math.ceil(size / RASTER_RES)) + 1
    while t < tmax:
        x = ox + dx * t
        y = oy + dy * t
        if bless:
            x = x % size
            y = y % size
        i = int(x / RASTER_RES)
        j = int(y / RASTER_RES)
        if 0 <= i <= gmax and 0 <= j <= gmax:
            k = S.raster[e, i, j]
            if k >= 0 and k < S.npts[e]:
                if not (S.pown[e, k] == p and S.pnum[e, k] >= own_min):
                    return t, False
        t += step
    return tmax, wall


@nb.njit(cache=True)
def write_obs(S, e, p, out):
    size = S.size[e]
    x = S.x[e, p]
    y = S.y[e, p]
    a = S.ang[e, p]
    ca = math.cos(a)
    sa = math.sin(a)
    bless = False
    bless_rem = 0.0
    for f in range(MAX_FX):
        if S.ftype[e, f] == B_BORDERLESS:
            bless = True
            bless_rem = max(bless_rem, (S.fend[e, f] - S.t[e]) / B_DURATION[B_BORDERLESS])
    o = 0
    # --- rays
    for r in range(N_RAYS):
        th = a + 2.0 * math.pi * r / N_RAYS
        d, wall = _cast_ray(S, e, p, x, y, math.cos(th), math.sin(th), bless)
        out[o] = d / RAY_MAX
        out[o + 1] = min(d, RAY_NEAR) / RAY_NEAR
        out[o + 2] = 1.0 if wall else 0.0
        o += 3
    # --- opponents, nearest first
    n = S.n[e]
    dists = np.full(n, 1e9)
    for q in range(n):
        if q != p and S.alive[e, q]:
            dists[q] = math.hypot(S.x[e, q] - x, S.y[e, q] - y)
    order = np.argsort(dists)
    for k in range(K_OPP):
        if k < n and dists[order[k]] < 1e8:
            q = order[k]
            rx = S.x[e, q] - x
            ry = S.y[e, q] - y
            d = max(dists[q], 1e-6)
            ex = (rx * ca + ry * sa) / d
            ey = (-rx * sa + ry * ca) / d
            da = S.ang[e, q] - a
            out[o] = 1.0
            out[o + 1] = min(d / RAY_MAX, 2.0)
            out[o + 2] = ex
            out[o + 3] = ey
            out[o + 4] = math.cos(da)
            out[o + 5] = math.sin(da)
            out[o + 6] = S.vel[e, q] / BASE_VEL
            out[o + 7] = S.rad[e, q] / BASE_R
            out[o + 8] = 1.0 if S.printing[e, q] else 0.0
            out[o + 9] = 1.0 if S.invinc[e, q] else 0.0
            out[o + 10] = 1.0 if S.inv[e, q] else 0.0
            out[o + 11] = 1.0 if S.straight[e, q] else 0.0
            out[o + 12] = S.turn[e, q]
        else:
            for i in range(F_OPP):
                out[o + i] = 0.0
        o += F_OPP
    # --- bonuses, nearest first (path features are filled in by territory_features)
    o_bon = o
    bd = np.full(BONUS_CAP, 1e9)
    for b in range(BONUS_CAP):
        if S.btype[e, b] >= 0:
            bd[b] = math.hypot(S.bx[e, b] - x, S.by[e, b] - y)
    border = np.argsort(bd)
    for k in range(K_BON):
        b = border[k]
        for i in range(F_BON):
            out[o + i] = 0.0
        if bd[b] < 1e8:
            rx = S.bx[e, b] - x
            ry = S.by[e, b] - y
            d = max(bd[b], 1e-6)
            out[o] = 1.0
            out[o + 1] = min(d / RAY_MAX, 2.0)
            out[o + 2] = (rx * ca + ry * sa) / d
            out[o + 3] = (-rx * sa + ry * ca) / d
            out[o + 7 + S.btype[e, b]] = 1.0
        o += F_BON
    # --- self
    n_alive = 0
    for q in range(n):
        if S.alive[e, q]:
            n_alive += 1
    out[o + 0] = x / size
    out[o + 1] = y / size
    out[o + 2] = ca
    out[o + 3] = sa
    out[o + 4] = size / MAX_SIZE
    out[o + 5] = S.vel[e, p] / BASE_VEL
    out[o + 6] = S.rad[e, p] / BASE_R
    out[o + 7] = 1.0 if S.inv[e, p] else 0.0
    out[o + 8] = 1.0 if S.invinc[e, p] else 0.0
    out[o + 9] = 1.0 if S.straight[e, p] else 0.0
    out[o + 10] = 1.0 if S.printing[e, p] else 0.0
    out[o + 11] = 1.0 if S.t[e] >= PRINT_DELAY else 0.0
    out[o + 12] = max(0.0, PRINT_DELAY - S.t[e]) / PRINT_DELAY
    out[o + 13] = S.turn[e, p]
    out[o + 14] = 1.0 if S.move[e, p] == -1 else 0.0
    out[o + 15] = 1.0 if S.move[e, p] == 0 else 0.0
    out[o + 16] = 1.0 if S.move[e, p] == 1 else 0.0
    out[o + 17] = 1.0 if bless else 0.0
    out[o + 18] = bless_rem
    out[o + 19] = n_alive / MAX_PLAYERS
    out[o + 20] = n / MAX_PLAYERS
    out[o + 21] = min(S.since_toggle[e, p] / 60.0, 1.5)
    out[o + 22] = 1.0 if S.pm_active[e, p] else 0.0
    o += 23
    bit = 1 << p
    for t in range(NB):
        out[o + t] = 0.0
    for f in range(MAX_FX):
        t = S.ftype[e, f]
        if t >= 0 and (S.fmask[e, f] & bit) != 0 and B_DURATION[t] > 0:
            rem = (S.fend[e, f] - S.t[e]) / B_DURATION[t]
            if rem > out[o + t]:
                out[o + t] = rem
    o += NB
    out[o] = S.delay[e, p] / 15.0
    out[o + 1] = S.aggr[e, p]   # kill bonus this player plays for ("hunting vs surviving")
    o += 2
    for k in range(HIST):
        for m in range(3):
            out[o + 3 * k + m] = 1.0 if S.hist[e, p, k] == m - 1 else 0.0
    o += 3 * HIST
    lookahead_features(S, e, p, out, o)
    o += N_LOOK
    territory_features(S, e, p, out, o, o_bon, border, bd)
    o += N_TERR
    n_opp = 0
    n_hum = 0
    for q in range(n):
        if q != p and S.alive[e, q]:
            n_opp += 1
            if S.human[e, q]:
                n_hum += 1
    for k in range(K_OPP):
        out[o + k] = 0.0
        if k < n and dists[order[k]] < 1e8 and S.human[e, order[k]]:
            out[o + k] = 1.0
    out[o + K_OPP] = n_hum / max(n_opp, 1)


@nb.njit(parallel=True, cache=True)
def compute_obs(S, out, mask):
    """out[E, P, OBS_DIM]; only rows with mask[e, p] are written."""
    E = out.shape[0]
    P = out.shape[1]
    for e in nb.prange(E):
        for p in range(P):
            if mask[e, p]:
                write_obs(S, e, p, out[e, p])


# --------------------------------------------------------------------------- #
# Heuristic: short tree search over held actions against the static world
# --------------------------------------------------------------------------- #
@nb.njit(cache=True)
def _blocked(S, e, p, x, y, r, own_min, bless):
    size = S.size[e]
    if not bless:
        if x - r < 0 or x + r > size or y - r < 0 or y + r > size:
            return True
    else:
        x = x % size
        y = y % size
    for q in range(S.n[e]):   # keep clear of opponents' heads
        if q != p and S.alive[e, q]:
            dx = S.x[e, q] - x
            dy = S.y[e, q] - y
            rs = r + S.rad[e, q] + HEAD_MARGIN
            if dx * dx + dy * dy < rs * rs:
                return True
    if S.invinc[e, p]:
        return False
    R = r + S.pr_max[e]
    i0 = _clampi(int(math.floor((x - R) / HASH_CELL)), 0, HASH_G - 1)
    i1 = _clampi(int(math.floor((x + R) / HASH_CELL)), 0, HASH_G - 1)
    j0 = _clampi(int(math.floor((y - R) / HASH_CELL)), 0, HASH_G - 1)
    j1 = _clampi(int(math.floor((y + R) / HASH_CELL)), 0, HASH_G - 1)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for c in range(S.hcnt[e, i, j]):
                k = S.hidx[e, i, j, c]
                if S.pown[e, k] == p and S.pnum[e, k] >= own_min:
                    continue
                dx = S.px[e, k] - x
                dy = S.py[e, k] - y
                rs = r + S.pr[e, k]
                if dx * dx + dy * dy < rs * rs:
                    return True
    return False


@nb.njit(cache=True)
def _rollout(S, e, p, x, y, a, turn, frames, dtf, own_min, bless):
    """Hold ``turn`` for ``frames`` sub-steps of ``dtf`` ms. Returns (survived, x, y, a).

    In straight-angle mode ``turn`` is a single 90 degree turn at the first sub-step."""
    v = S.vel[e, p] / 1000.0 * dtf
    w = base_angular_velocity(S.vel[e, p]) * dtf
    r = S.rad[e, p]
    for i in range(frames):
        if S.straight[e, p]:
            if i == 0:
                a += turn * math.pi / 2.0
        else:
            a += turn * w
        x += math.cos(a) * v
        y += math.sin(a) * v
        if _blocked(S, e, p, x, y, r, own_min, bless):
            return i, x, y, a
    return frames, x, y, a


@nb.njit(cache=True)
def _is_borderless(S, e):
    for f in range(MAX_FX):
        if S.ftype[e, f] == B_BORDERLESS:
            return True
    return False


@nb.njit(cache=True)
def _pending(S, e, p, own_min, bless):
    """Advance the avatar frame by frame through the moves already sent but not yet
    applied by the server (latency). Returns (alive, x, y, a, turn, move)."""
    x = S.x[e, p]
    y = S.y[e, p]
    a = S.ang[e, p]
    turn = S.turn[e, p]
    move = S.move[e, p]
    inv = -1 if S.inv[e, p] else 1
    v = S.vel[e, p] / 1000.0 * FRAME_MS
    w = base_angular_velocity(S.vel[e, p]) * FRAME_MS
    r = S.rad[e, p]
    for k in range(S.delay[e, p]):
        m = S.sched[e, p, (S.frames[e] + k) % RING]
        if m != NO_MOVE and m != move:
            move = m
            turn = m * inv
        if turn != 0:
            if S.straight[e, p]:
                a += turn * math.pi / 2.0
                turn = 0
            else:
                a += turn * w
        x += math.cos(a) * v
        y += math.sin(a) * v
        if _blocked(S, e, p, x, y, r, own_min, bless):
            return False, x, y, a, turn, move
    return True, x, y, a, turn, move


@nb.njit(cache=True)
def _branch_turn(S, e, p, m, turn, move):
    """Angular direction obtained by sending move ``m`` after the pending moves."""
    inv = -1 if S.inv[e, p] else 1
    if m == move:          # nothing is sent, the server keeps the current rotation
        return 0 if S.straight[e, p] else turn
    return m * inv


@nb.njit(cache=True)
def _search(S, e, p, h1, h2, out3, out3b):
    """Survival (in sub-steps) of each move m in (-1, 0, 1): out3[m] holding m for h1,
    out3b[m] = out3[m] + best follow-up survival over h2. Returns False if the avatar
    cannot survive the moves already in flight."""
    bless = _is_borderless(S, e)
    own_min = S.bc[e, p] - OWN_SKIP
    dtf = FRAME_MS * 2.0
    inv = -1 if S.inv[e, p] else 1
    ok, x0, y0, a0, turn0, move0 = _pending(S, e, p, own_min, bless)
    if not ok:
        for mi in range(3):
            out3[mi] = 0
            out3b[mi] = 0
        return False
    for mi in range(3):
        turn = _branch_turn(S, e, p, mi - 1, turn0, move0)
        s1, x1, y1, a1 = _rollout(S, e, p, x0, y0, a0, turn, h1, dtf, own_min, bless)
        best = 0
        if s1 == h1:
            for m2 in range(-1, 2):
                s2, _, _, _ = _rollout(S, e, p, x1, y1, a1, m2 * inv, h2, dtf, own_min, bless)
                if s2 > best:
                    best = s2
        out3[mi] = s1
        out3b[mi] = s1 + best
    return True


@nb.njit(cache=True)
def heuristic_action(S, e, p, eps):
    """Pick move in {-1, 0, 1} maximising survival over a 2-level lookahead."""
    s1 = np.zeros(3, np.int64)
    s2 = np.zeros(3, np.int64)
    _search(S, e, p, 14, 30, s1, s2)
    best = -1.0
    best_m = 0
    for mi in range(3):
        score = s2[mi] + np.random.random() * eps
        if mi == 1:
            score += 0.5   # mild preference for going straight
        if score > best:
            best = score
            best_m = mi - 1
    return best_m


@nb.njit(parallel=True, cache=True)
def heuristic_actions(S, out, mask, eps):
    E = out.shape[0]
    P = out.shape[1]
    for e in nb.prange(E):
        for p in range(P):
            if mask[e, p]:
                out[e, p] = heuristic_action(S, e, p, eps)


# --------------------------------------------------------------------------- #
# Egocentric occupancy map (optional CNN input)
# --------------------------------------------------------------------------- #
MAP_CHANNELS = 3   # 0: any trail, 1: own trail, 2: wall / outside the board
MAP_RES = 1.0      # board units per map cell


@nb.njit(cache=True)
def write_map(S, e, p, out):
    """out[MAP_CHANNELS, M, M] uint8, heading points up (row 0 is ahead), centred on the head."""
    M = out.shape[1]
    size = S.size[e]
    x = S.x[e, p]
    y = S.y[e, p]
    ca = math.cos(S.ang[e, p])
    sa = math.sin(S.ang[e, p])
    bless = False
    for f in range(MAX_FX):
        if S.ftype[e, f] == B_BORDERLESS:
            bless = True
    own_min = S.bc[e, p] - OWN_SKIP
    gmax = int(math.ceil(size / RASTER_RES)) + 1
    for i in range(M):
        fwd = (M / 2.0 - i - 0.5) * MAP_RES
        for j in range(M):
            right = (j - M / 2.0 + 0.5) * MAP_RES
            any_t = 0
            own_t = 0
            wall = 0
            # 2x2 sub-samples per cell so that thin trails are not aliased away
            for si in range(2):
                for sj in range(2):
                    ff = fwd + (si - 0.5) * 0.5 * MAP_RES
                    rr = right + (sj - 0.5) * 0.5 * MAP_RES
                    wx = x + ff * ca - rr * sa
                    wy = y + ff * sa + rr * ca
                    if bless:
                        wx = wx % size
                        wy = wy % size
                    elif wx < 0 or wy < 0 or wx > size or wy > size:
                        wall = 255
                        continue
                    gi = int(wx / RASTER_RES)
                    gj = int(wy / RASTER_RES)
                    if gi > gmax or gj > gmax:
                        continue
                    k = S.raster[e, gi, gj]
                    if k >= 0 and k < S.npts[e]:
                        if S.pown[e, k] == p:
                            if S.pnum[e, k] < own_min:
                                any_t = 255
                                own_t = 255
                        else:
                            any_t = 255
            out[0, i, j] = any_t
            out[1, i, j] = own_t
            out[2, i, j] = wall


@nb.njit(parallel=True, cache=True)
def compute_maps(S, out, mask):
    """out[E, P, MAP_CHANNELS, M, M] uint8; only rows with mask[e, p] are written."""
    E = out.shape[0]
    P = out.shape[1]
    for e in nb.prange(E):
        for p in range(P):
            if mask[e, p]:
                write_map(S, e, p, out[e, p])


@nb.njit(cache=True)
def lookahead_features(S, e, p, out, o):
    """For each move in (-1, 0, 1), sent now (i.e. after the moves already in flight),
    and two horizons (~1 s and ~2 s of holding it): the fraction of the horizon survived
    holding the move, and survived holding it then the best follow-up.
    Obstacles are the current trails, walls and opponents' heads (static world)."""
    s1 = np.zeros(3, np.int64)
    s2 = np.zeros(3, np.int64)
    for hz in range(2):
        h1 = 14 if hz == 0 else 30
        h2 = 30 if hz == 0 else 60
        _search(S, e, p, h1, h2, s1, s2)
        for mi in range(3):
            out[o + hz * 6 + mi * 2] = s1[mi] / h1
            out[o + hz * 6 + mi * 2 + 1] = s2[mi] / (h1 + h2)


# --------------------------------------------------------------------------- #
# Territory (Voronoi) features on the coarse whole-board grid
# --------------------------------------------------------------------------- #
INF = 1 << 30


@nb.njit(cache=True)
def _cell(S, e, x, y):
    cs = S.size[e] / GRID
    return _clampi(int(math.floor(x / cs)), 0, GRID - 1), _clampi(int(math.floor(y / cs)), 0, GRID - 1)


@nb.njit(cache=True)
def _bfs(S, e, src_i, src_j, nsrc, dist, queue, bless):
    """Multi-source BFS (8-neighbourhood) over free cells of S.gown; sources are always free.

    Diagonal steps let paths through trail holes that a trail crossing the grid diagonally
    leaves open only at a cell corner. A continuous trail always blocks a 4-connected chain
    of cells, which no 8-connected path can cross, so solid trails stay closed."""
    for i in range(GRID):
        for j in range(GRID):
            dist[i, j] = INF
    tail = 0
    for s in range(nsrc):
        i = src_i[s]
        j = src_j[s]
        if dist[i, j] != 0:
            dist[i, j] = 0
            queue[tail] = i * GRID + j
            tail += 1
    head = 0
    while head < tail:
        c = queue[head]
        head += 1
        i = c // GRID
        j = c % GRID
        d = dist[i, j] + 1
        for k in range(9):
            if k == 4:
                continue
            ni = i + k // 3 - 1
            nj = j + k % 3 - 1
            if bless:
                ni %= GRID
                nj %= GRID
            elif ni < 0 or nj < 0 or ni >= GRID or nj >= GRID:
                continue
            if S.gown[e, ni, nj] < 0 and dist[ni, nj] > d:
                dist[ni, nj] = d
                queue[tail] = ni * GRID + nj
                tail += 1


@nb.njit(cache=True)
def _territory(S, e, x, y, d_opp, dist, queue, bless, n_free):
    """(share of free cells reached strictly before every opponent, share reachable)."""
    si = np.empty(1, np.int64)
    sj = np.empty(1, np.int64)
    si[0], sj[0] = _cell(S, e, x, y)
    _bfs(S, e, si, sj, 1, dist, queue, bless)
    mine = 0
    reach = 0
    for i in range(GRID):
        for j in range(GRID):
            if dist[i, j] < INF:
                reach += 1
                if dist[i, j] < d_opp[i, j]:
                    mine += 1
    return mine / n_free, reach / n_free


@nb.njit(cache=True)
def _bonus_paths(S, e, p, out, o_bon, border, bd, dist, d_opp, bless):
    """Per bonus (same order as the bonus block): reachable through free cells, path length
    (board units / RAY_MAX) and whether we get there before every opponent. A bonus counts as
    reached anywhere within catch range of its centre."""
    cs = S.size[e] / GRID
    reach_r = BONUS_R + S.rad[e, p]
    rc = int(math.ceil(reach_r / cs))
    for k in range(K_BON):
        b = border[k]
        if bd[b] >= 1e8:
            continue
        ci, cj = _cell(S, e, S.bx[e, b], S.by[e, b])
        best_me = INF
        best_opp = INF
        for di in range(-rc, rc + 1):
            for dj in range(-rc, rc + 1):
                if (di * di + dj * dj) * cs * cs > (reach_r + cs) ** 2:
                    continue
                i = ci + di
                j = cj + dj
                if bless:
                    i %= GRID
                    j %= GRID
                elif i < 0 or j < 0 or i >= GRID or j >= GRID:
                    continue
                best_me = min(best_me, dist[i, j])
                best_opp = min(best_opp, d_opp[i, j])
        f = o_bon + k * F_BON
        if best_me < INF:
            out[f + 4] = 1.0
            out[f + 5] = min(best_me * cs / RAY_MAX, 3.0)
            out[f + 6] = 1.0 if best_me < best_opp else 0.0


@nb.njit(cache=True)
def territory_features(S, e, p, out, o, o_bon, border, bd):
    """Voronoi-style territory on the coarse grid (all heads assumed equally fast):
    now, and for each move held for ~0.8 s after the moves already in flight:
    [survives the hold, share of free board reached first, share reachable].
    Also fills the path features of the bonus block starting at ``o_bon``."""
    bless = _is_borderless(S, e)
    n_free = 0
    for i in range(GRID):
        for j in range(GRID):
            if S.gown[e, i, j] < 0:
                n_free += 1
    n_free = max(n_free, 1)
    dist = np.empty((GRID, GRID), np.int64)
    d_opp = np.empty((GRID, GRID), np.int64)
    queue = np.empty(GRID * GRID + S.n[e], np.int64)
    si = np.empty(S.n[e], np.int64)
    sj = np.empty(S.n[e], np.int64)
    k = 0
    for q in range(S.n[e]):
        if q != p and S.alive[e, q]:
            si[k], sj[k] = _cell(S, e, S.x[e, q], S.y[e, q])
            k += 1
    _bfs(S, e, si, sj, k, d_opp, queue, bless)
    mine, reach = _territory(S, e, S.x[e, p], S.y[e, p], d_opp, dist, queue, bless, n_free)
    _bonus_paths(S, e, p, out, o_bon, border, bd, dist, d_opp, bless)
    out[o] = mine
    out[o + 1] = reach
    out[o + 2] = n_free / (GRID * GRID)
    own_min = S.bc[e, p] - OWN_SKIP
    ok, x0, y0, a0, turn0, move0 = _pending(S, e, p, own_min, bless)
    for mi in range(3):
        b = o + 3 + 3 * mi
        out[b] = 0.0
        out[b + 1] = 0.0
        out[b + 2] = 0.0
        if not ok:
            continue
        turn = _branch_turn(S, e, p, mi - 1, turn0, move0)
        h = 24
        s1, x1, y1, _ = _rollout(S, e, p, x0, y0, a0, turn, h, FRAME_MS * 2.0, own_min, bless)
        if s1 < h:
            continue
        if bless:
            x1 %= S.size[e]
            y1 %= S.size[e]
        m1, r1 = _territory(S, e, x1, y1, d_opp, dist, queue, bless, n_free)
        out[b] = 1.0
        out[b + 1] = m1
        out[b + 2] = r1


# --------------------------------------------------------------------------- #
# Whole-board map (optional CNN input), board-aligned
# --------------------------------------------------------------------------- #
GLOBAL_CHANNELS = 4   # trails (other 255, own 128), own head + heading, opponents' heads + headings, bonuses


@nb.njit(cache=True)
def _stamp(out, c, x, y, cs, rad, value, size, bless):
    i0 = int(math.floor((x - rad) / cs))
    i1 = int(math.floor((x + rad) / cs))
    j0 = int(math.floor((y - rad) / cs))
    j1 = int(math.floor((y + rad) / cs))
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            ii = i
            jj = j
            if bless:
                ii %= GRID
                jj %= GRID
            elif i < 0 or j < 0 or i >= GRID or j >= GRID:
                continue
            if out[c, ii, jj] < value:
                out[c, ii, jj] = value


@nb.njit(cache=True)
def write_global(S, e, p, out):
    """out[GLOBAL_CHANNELS, GRID, GRID] uint8 covering the whole board (index [x cell, y cell])."""
    size = S.size[e]
    cs = size / GRID
    bless = _is_borderless(S, e)
    for i in range(GRID):
        for j in range(GRID):
            o = S.gown[e, i, j]
            out[0, i, j] = 0 if o < 0 else (128 if o == p else 255)
            out[1, i, j] = 0
            out[2, i, j] = 0
            out[3, i, j] = 0
    for q in range(S.n[e]):
        if not S.alive[e, q]:
            continue
        c = 1 if q == p else 2
        x = S.x[e, q]
        y = S.y[e, q]
        ca = math.cos(S.ang[e, q])
        sa = math.sin(S.ang[e, q])
        for k in range(1, 9):   # heading: a fading line ~12 units ahead
            d = k * 1.5
            _stamp(out, c, x + ca * d, y + sa * d, cs, 0.0, 200 - 15 * k, size, bless)
        _stamp(out, c, x, y, cs, cs, 255, size, bless)
    for b in range(BONUS_CAP):
        t = S.btype[e, b]
        if t >= 0:
            aff = B_AFFECT[t]
            v = 255 if aff == AFF_SELF else (170 if aff == AFF_ENEMY else 85)
            _stamp(out, 3, S.bx[e, b], S.by[e, b], cs, BONUS_R, v, size, bless)


@nb.njit(parallel=True, cache=True)
def compute_globals(S, out, mask):
    E = out.shape[0]
    P = out.shape[1]
    for e in nb.prange(E):
        for p in range(P):
            if mask[e, p]:
                write_global(S, e, p, out[e, p])
