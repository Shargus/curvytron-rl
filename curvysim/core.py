"""Fast, batched re-implementation of the Curvytron server physics (numba).

The rules mirror the node server in ``curvytron/src/server`` (Game.js, Avatar.js,
BaseAvatar.js, World.js, PrintManager.js, BonusManager.js, BonusStack.js and the
Bonus/*.js classes). Units are the server's: board units, milliseconds, radians.

The whole state of ``E`` independent rounds lives in a ``SimState`` namedtuple of
numpy arrays so that numba kernels can run over all environments in parallel.
Each environment is one *round* (spawn -> at most one survivor), which is the
natural RL episode; scores between rounds are independent in Curvytron.
"""
from collections import namedtuple
import math

import numba as nb
import numpy as np

# --------------------------------------------------------------------------- #
# Constants (server values)
# --------------------------------------------------------------------------- #
BONUS_NAMES = (
    'BonusSelfSmall', 'BonusSelfSlow', 'BonusSelfFast', 'BonusSelfMaster',
    'BonusEnemySlow', 'BonusEnemyFast', 'BonusEnemyBig', 'BonusEnemyInverse',
    'BonusEnemyStraightAngle', 'BonusGameBorderless', 'BonusAllColor', 'BonusGameClear',
)
NB = len(BONUS_NAMES)
B_SMALL, B_SLOW, B_FAST, B_MASTER, B_ESLOW, B_EFAST, B_EBIG, B_EINV, B_ESTRAIGHT, \
    B_BORDERLESS, B_COLOR, B_CLEAR = range(NB)
AFF_SELF, AFF_ENEMY, AFF_GAME, AFF_ALL = 0, 1, 2, 3
B_AFFECT = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 2], np.int64)
B_DURATION = np.array([7500, 5000, 4000, 7500, 5000, 6000, 7500, 5000, 5000, 10000, 7500, 0], np.float64)
B_PROB = np.array([1, 1, 1, 1, 1, 1, 1, 0.8, 0.6, 0.8, 1, 1], np.float64)
B_DVEL = np.array([0, -8, 12, 0, -8, 12, 0, 0, 0, 0, 0, 0], np.float64)
B_DRAD = np.array([-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], np.float64)

FRAME_MS = 1000.0 / 60.0
BASE_VEL = 16.0           # units / s
BASE_AV = 2.8 / 1000.0    # rad / ms
BASE_R = 0.6
TRAIL_LATENCY = 3
PRINT_DIST = 60.0
HOLE_DIST = 5.0
PRINT_DELAY = 3000.0      # ms after round start before trails are printed
BONUS_R = 3.0
BONUS_CAP = 20
BONUS_POP_MS = 3000.0
BONUS_MARGIN = 0.01
SPAWN_MARGIN = 0.05
SPAWN_ANGLE_MARGIN = 0.3

MAX_PLAYERS = 16
MAX_POINTS = 32768        # trail bodies per round
HASH_CELL = 2.0
HASH_CAP = 32
RASTER_RES = 0.5          # observation raster: board units per cell
MAX_FX = 48
RING = 32                # action queue length (frames) used to model latency
NO_MOVE = 127            # empty slot in the action queue
HIST = 8                 # number of past decisions kept for observations
GRID = 64                # coarse whole-board grid (cells per side, scaled to the board)


def board_size(n_players):
    sq = 80.0 * 80.0
    return float(math.floor(math.sqrt(sq + (n_players - 1) * sq / 5.0) + 0.5))


MAX_SIZE = board_size(MAX_PLAYERS)
HASH_G = int(math.ceil(MAX_SIZE / HASH_CELL)) + 2
RASTER_G = int(math.ceil(MAX_SIZE / RASTER_RES)) + 2

SimState = namedtuple('SimState', [
    # per env
    'size', 't', 'n', 'deaths', 'over', 'next_pop', 'print_started', 'npts', 'pr_max',
    'bon_enabled', 'frames', 'overflow',
    # per player
    'x', 'y', 'ang', 'vel', 'rad', 'inv', 'invinc', 'straight', 'turn', 'move',
    'alive', 'printing', 'tl_x', 'tl_y', 'has_tl', 'bc',
    'pm_active', 'pm_dist', 'pm_lx', 'pm_ly', 'since_toggle', 'score', 'death_t', 'cause', 'killer',
    'aggr',
    # trail bodies
    'px', 'py', 'pr', 'pown', 'pnum',
    'hcnt', 'hidx', 'raster', 'gown',
    # bonuses on the map and active effects
    'bx', 'by', 'btype',
    'ftype', 'fmask', 'fend',
    # latency model: per player delay (frames), queued moves, last decisions
    'delay', 'sched', 'hist',
    # human-like players (handicapped controls, see env.py) and bonus pickups per round
    'human', 'conf_flip', 'conf_p0', 'conf_end', 'conf_dur', 'pickups',
])


def make_state(E, P=MAX_PLAYERS):
    f64, i32, i8, b = np.float64, np.int32, np.int8, np.bool_
    z = lambda shape, dt: np.zeros(shape, dt)
    return SimState(
        size=z(E, f64), t=z(E, f64), n=z(E, i32), deaths=z(E, i32), over=z(E, b),
        next_pop=z(E, f64), print_started=z(E, b), npts=z(E, i32), pr_max=z(E, f64),
        bon_enabled=z((E, NB), b), frames=z(E, i32), overflow=z(E, i32),
        x=z((E, P), f64), y=z((E, P), f64), ang=z((E, P), f64), vel=z((E, P), f64),
        rad=z((E, P), f64), inv=z((E, P), b), invinc=z((E, P), b), straight=z((E, P), b),
        turn=z((E, P), i32), move=z((E, P), i32), alive=z((E, P), b), printing=z((E, P), b),
        tl_x=z((E, P), f64), tl_y=z((E, P), f64), has_tl=z((E, P), b), bc=z((E, P), i32),
        pm_active=z((E, P), b), pm_dist=z((E, P), f64), pm_lx=z((E, P), f64),
        pm_ly=z((E, P), f64), since_toggle=z((E, P), f64), score=z((E, P), f64),
        death_t=z((E, P), f64), cause=np.full((E, P), -1, i8), killer=np.full((E, P), -1, i8),
        aggr=z((E, P), f64),
        px=z((E, MAX_POINTS), f64), py=z((E, MAX_POINTS), f64), pr=z((E, MAX_POINTS), f64),
        pown=z((E, MAX_POINTS), i8), pnum=z((E, MAX_POINTS), i32),
        hcnt=z((E, HASH_G, HASH_G), i32), hidx=z((E, HASH_G, HASH_G, HASH_CAP), i32),
        raster=np.full((E, RASTER_G, RASTER_G), -1, i32),
        gown=np.full((E, GRID, GRID), -1, i8),
        bx=z((E, BONUS_CAP), f64), by=z((E, BONUS_CAP), f64), btype=np.full((E, BONUS_CAP), -1, i32),
        ftype=np.full((E, MAX_FX), -1, i32), fmask=z((E, MAX_FX), i32), fend=z((E, MAX_FX), f64),
        delay=z((E, P), i32), sched=np.full((E, P, RING), NO_MOVE, i8), hist=z((E, P, HIST), i8),
        human=z((E, P), b), conf_flip=z((E, P), b), conf_p0=z((E, P), f64), conf_end=z((E, P), f64),
        conf_dur=z((E, P), f64), pickups=z((E, P, NB), i32),
    )


# --------------------------------------------------------------------------- #
# World / trail bodies
# --------------------------------------------------------------------------- #
@nb.njit(cache=True)
def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@nb.njit(cache=True)
def clear_world(S, e):
    """World.clear(): drop every trail body (round start and BonusGameClear)."""
    S.npts[e] = 0
    S.pr_max[e] = 0.0
    g = int(math.ceil(S.size[e] / HASH_CELL)) + 2
    for i in range(g):
        for j in range(g):
            S.hcnt[e, i, j] = 0
    r = int(math.ceil(S.size[e] / RASTER_RES)) + 2
    for i in range(r):
        for j in range(r):
            S.raster[e, i, j] = -1
    for i in range(GRID):
        for j in range(GRID):
            S.gown[e, i, j] = -1


@nb.njit(cache=True)
def add_body(S, e, p, x, y):
    """Avatar.addPoint -> Game.onPoint -> World.addBody(new AvatarBody)."""
    S.tl_x[e, p] = x
    S.tl_y[e, p] = y
    S.has_tl[e, p] = True
    num = S.bc[e, p]
    S.bc[e, p] = num + 1
    k = S.npts[e]
    if k >= MAX_POINTS:
        S.overflow[e] += 1
        return
    r = S.rad[e, p]
    S.px[e, k] = x
    S.py[e, k] = y
    S.pr[e, k] = r
    S.pown[e, k] = p
    S.pnum[e, k] = num
    S.npts[e] = k + 1
    if r > S.pr_max[e]:
        S.pr_max[e] = r
    hx = _clampi(int(math.floor(x / HASH_CELL)), 0, HASH_G - 1)
    hy = _clampi(int(math.floor(y / HASH_CELL)), 0, HASH_G - 1)
    c = S.hcnt[e, hx, hy]
    if c < HASH_CAP:
        S.hidx[e, hx, hy, c] = k
        S.hcnt[e, hx, hy] = c + 1
    else:
        S.overflow[e] += 1
    # observation raster: stamp the disc of the body
    i0 = _clampi(int(math.floor((x - r) / RASTER_RES)), 0, RASTER_G - 1)
    i1 = _clampi(int(math.floor((x + r) / RASTER_RES)), 0, RASTER_G - 1)
    j0 = _clampi(int(math.floor((y - r) / RASTER_RES)), 0, RASTER_G - 1)
    j1 = _clampi(int(math.floor((y + r) / RASTER_RES)), 0, RASTER_G - 1)
    rr = (r + 0.5 * RASTER_RES) ** 2
    for i in range(i0, i1 + 1):
        cx = (i + 0.5) * RASTER_RES - x
        for j in range(j0, j1 + 1):
            cy = (j + 0.5) * RASTER_RES - y
            if cx * cx + cy * cy <= rr:
                S.raster[e, i, j] = k
    # coarse whole-board grid: owner of the latest body touching each cell. Only cells the disc
    # really overlaps (not its bounding box), so that trail holes stay open on the grid
    cs = S.size[e] / GRID
    i0 = _clampi(int(math.floor((x - r) / cs)), 0, GRID - 1)
    i1 = _clampi(int(math.floor((x + r) / cs)), 0, GRID - 1)
    j0 = _clampi(int(math.floor((y - r) / cs)), 0, GRID - 1)
    j1 = _clampi(int(math.floor((y + r) / cs)), 0, GRID - 1)
    for i in range(i0, i1 + 1):
        dx = max(i * cs - x, 0.0, x - (i + 1) * cs)
        for j in range(j0, j1 + 1):
            dy = max(j * cs - y, 0.0, y - (j + 1) * cs)
            if dx * dx + dy * dy < r * r:
                S.gown[e, i, j] = p


@nb.njit(cache=True)
def query_body(S, e, x, y, r, owner, num_ref):
    """World.getBody(): index of a trail body touching circle (x, y, r), or -1.

    Own bodies only count when ``num_ref - num > TRAIL_LATENCY`` (AvatarBody.match).
    Pass owner=-1 to test against every body (spawn/bonus placement).
    """
    R = r + S.pr_max[e]
    i0 = _clampi(int(math.floor((x - R) / HASH_CELL)), 0, HASH_G - 1)
    i1 = _clampi(int(math.floor((x + R) / HASH_CELL)), 0, HASH_G - 1)
    j0 = _clampi(int(math.floor((y - R) / HASH_CELL)), 0, HASH_G - 1)
    j1 = _clampi(int(math.floor((y + R) / HASH_CELL)), 0, HASH_G - 1)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for c in range(S.hcnt[e, i, j]):
                k = S.hidx[e, i, j, c]
                dx = S.px[e, k] - x
                dy = S.py[e, k] - y
                rs = r + S.pr[e, k]
                if dx * dx + dy * dy < rs * rs:
                    if owner >= 0 and S.pown[e, k] == owner and num_ref - S.pnum[e, k] <= TRAIL_LATENCY:
                        continue
                    return k
    return -1


@nb.njit(cache=True)
def random_point(size, margin):
    return margin + np.random.random() * (size - 2.0 * margin)


@nb.njit(cache=True)
def direction_valid(size, angle, x, y, margin):
    quarter = math.pi / 2.0
    for i in range(4):
        lo = quarter * i
        hi = quarter * (i + 1)
        if angle >= lo and angle < hi:
            b0 = i
            b1 = i + 1 if i < 3 else 0
            d0 = (size - x) if b0 == 0 else ((size - y) if b0 == 1 else (x if b0 == 2 else y))
            d1 = (size - x) if b1 == 0 else ((size - y) if b1 == 1 else (x if b1 == 2 else y))
            if d0 / math.cos(angle - lo) < margin:
                return False
            if d1 / math.cos(hi - angle) < margin:
                return False
            return True
    return False


# --------------------------------------------------------------------------- #
# Avatar mechanics
# --------------------------------------------------------------------------- #
@nb.njit(cache=True)
def base_angular_velocity(vel):
    ratio = vel / BASE_VEL
    return ratio * BASE_AV + math.log(1.0 / ratio) / 1000.0


@nb.njit(cache=True)
def set_printing(S, e, p, printing):
    if S.printing[e, p] != printing:
        S.printing[e, p] = printing
        add_body(S, e, p, S.x[e, p], S.y[e, p])
        if not printing:
            S.has_tl[e, p] = False
    S.since_toggle[e, p] = 0.0


@nb.njit(cache=True)
def pm_random_distance(printing):
    if printing:
        return PRINT_DIST * (0.3 + np.random.random() * 0.7)
    return HOLE_DIST * (0.8 + np.random.random() * 0.5)


@nb.njit(cache=True)
def pm_start(S, e, p):
    if not S.pm_active[e, p]:
        S.pm_active[e, p] = True
        S.pm_lx[e, p] = S.x[e, p]
        S.pm_ly[e, p] = S.y[e, p]
        set_printing(S, e, p, True)
        S.pm_dist[e, p] = pm_random_distance(S.printing[e, p])


@nb.njit(cache=True)
def pm_stop(S, e, p):
    if S.pm_active[e, p]:
        S.pm_active[e, p] = False
        set_printing(S, e, p, False)
        S.pm_dist[e, p] = 0.0
        S.pm_lx[e, p] = 0.0
        S.pm_ly[e, p] = 0.0


@nb.njit(cache=True)
def pm_test(S, e, p):
    if S.pm_active[e, p]:
        dx = S.x[e, p] - S.pm_lx[e, p]
        dy = S.y[e, p] - S.pm_ly[e, p]
        d = math.sqrt(dx * dx + dy * dy)
        S.pm_dist[e, p] -= d
        S.since_toggle[e, p] += d
        S.pm_lx[e, p] = S.x[e, p]
        S.pm_ly[e, p] = S.y[e, p]
        if S.pm_dist[e, p] <= 0.0:
            set_printing(S, e, p, not S.printing[e, p])
            S.pm_dist[e, p] = pm_random_distance(S.printing[e, p])


@nb.njit(cache=True)
def set_move(S, e, p, move):
    """GameController.onMove -> avatar.updateAngularVelocity(move); sent only on change."""
    if S.move[e, p] != move:
        S.move[e, p] = move
        S.turn[e, p] = -move if S.inv[e, p] else move


CAUSE_WALL, CAUSE_OWN, CAUSE_OTHER = 0, 1, 2


@nb.njit(cache=True)
def kill(S, e, p, score, cause, killer):
    S.alive[e, p] = False
    S.cause[e, p] = cause
    S.killer[e, p] = killer   # owner of the trail that was hit (-1 for walls)
    S.score[e, p] = score
    S.death_t[e, p] = S.t[e]
    S.deaths[e] += 1
    add_body(S, e, p, S.x[e, p], S.y[e, p])   # BaseAvatar.die -> addPoint
    pm_stop(S, e, p)
    bit = 1 << p
    for f in range(MAX_FX):
        if S.ftype[e, f] >= 0:
            S.fmask[e, f] &= ~bit


# --------------------------------------------------------------------------- #
# Bonuses
# --------------------------------------------------------------------------- #
@nb.njit(cache=True)
def resolve_player(S, e, p, changed):
    """BonusStack.resolve for avatar p after effect ``changed`` was added/removed."""
    bit = 1 << p
    dvel = 0.0
    drad = 0.0
    ninv = 0
    nmaster = 0
    nstraight = 0
    for f in range(MAX_FX):
        t = S.ftype[e, f]
        if t >= 0 and (S.fmask[e, f] & bit) != 0:
            dvel += B_DVEL[t]
            drad += B_DRAD[t]
            if t == B_EINV:
                ninv += 1
            elif t == B_MASTER:
                nmaster += 1
            elif t == B_ESTRAIGHT:
                nstraight += 1
    if changed == B_ESTRAIGHT or nstraight > 0:
        st = nstraight > 0
        if st and not S.straight[e, p]:
            S.turn[e, p] = 0
        S.straight[e, p] = st
    vel = max(BASE_VEL + dvel, BASE_VEL / 2.0)
    S.vel[e, p] = vel
    S.rad[e, p] = max(BASE_R * 2.0 ** drad, BASE_R / 8.0)
    S.inv[e, p] = (ninv % 2) != 0
    S.invinc[e, p] = nmaster > 0
    if changed == B_MASTER or nmaster > 0:
        if 1 - nmaster > 0:
            pm_start(S, e, p)
        else:
            pm_stop(S, e, p)


@nb.njit(cache=True)
def borderless(S, e):
    for f in range(MAX_FX):
        if S.ftype[e, f] == B_BORDERLESS:
            return True
    return False


@nb.njit(cache=True)
def add_effect(S, e, btype, mask):
    for f in range(MAX_FX):
        if S.ftype[e, f] < 0:
            S.ftype[e, f] = btype
            S.fmask[e, f] = mask
            S.fend[e, f] = S.t[e] + B_DURATION[btype]
            break
    for p in range(S.n[e]):
        if (mask >> p) & 1:
            resolve_player(S, e, p, btype)


@nb.njit(cache=True)
def expire_effects(S, e):
    for f in range(MAX_FX):
        t = S.ftype[e, f]
        if t >= 0 and S.fend[e, f] <= S.t[e]:
            mask = S.fmask[e, f]
            S.ftype[e, f] = -1
            S.fmask[e, f] = 0
            for p in range(S.n[e]):
                if (mask >> p) & 1 and S.alive[e, p]:
                    resolve_player(S, e, p, t)


@nb.njit(cache=True)
def catch_bonus(S, e, p):
    """BonusManager.testCatch + Bonus.applyTo."""
    x = S.x[e, p]
    y = S.y[e, p]
    rs = S.rad[e, p] + BONUS_R
    for b in range(BONUS_CAP - 1, -1, -1):
        t = S.btype[e, b]
        if t < 0:
            continue
        dx = S.bx[e, b] - x
        dy = S.by[e, b] - y
        if dx * dx + dy * dy < rs * rs:
            S.btype[e, b] = -1
            S.pickups[e, p, t] += 1
            aff = B_AFFECT[t]
            if t == B_CLEAR:
                clear_world(S, e)
            elif t == B_COLOR:
                pass  # cosmetic only
            elif aff == AFF_GAME:
                add_effect(S, e, t, 0)
            else:
                mask = 0
                for q in range(S.n[e]):
                    if not S.alive[e, q]:
                        continue
                    if (aff == AFF_SELF and q == p) or (aff == AFF_ENEMY and q != p) or aff == AFF_ALL:
                        mask |= 1 << q
                add_effect(S, e, t, mask)
            return


@nb.njit(cache=True)
def pop_bonus(S, e):
    S.next_pop[e] = S.t[e] + BONUS_POP_MS * (1.0 + np.random.random())
    count = 0
    slot = -1
    for b in range(BONUS_CAP):
        if S.btype[e, b] >= 0:
            count += 1
        elif slot < 0:
            slot = b
    if count >= BONUS_CAP or slot < 0:
        return
    # weighted choice (BonusManager.getRandomBonusType)
    probs = np.zeros(NB)
    total = 0.0
    n_alive = 0
    for q in range(S.n[e]):
        if S.alive[e, q]:
            n_alive += 1
    for t in range(NB):
        if not S.bon_enabled[e, t]:
            continue
        pr = B_PROB[t]
        if t == B_CLEAR:
            ratio = 1.0 - n_alive / S.n[e]
            if ratio >= 0.5:
                pr = math.floor((1.0 - ratio) * 10.0 + 0.5) / 10.0
        if pr > 0:
            probs[t] = pr
            total += pr
    if total <= 0:
        return
    v = np.random.random() * total
    btype = -1
    acc = 0.0
    for t in range(NB):
        acc += probs[t]
        if probs[t] > 0 and v < acc:
            btype = t
            break
    if btype < 0:
        return
    size = S.size[e]
    margin = BONUS_R + BONUS_MARGIN * size
    for _ in range(200):
        x = random_point(size, margin)
        y = random_point(size, margin)
        if query_body(S, e, x, y, margin, -1, 0) >= 0:
            continue
        ok = True
        for b in range(BONUS_CAP):
            if S.btype[e, b] >= 0:
                dx = S.bx[e, b] - x
                dy = S.by[e, b] - y
                if dx * dx + dy * dy < (margin + BONUS_R) ** 2:
                    ok = False
                    break
        if ok:
            S.bx[e, slot] = x
            S.by[e, slot] = y
            S.btype[e, slot] = btype
            return


# --------------------------------------------------------------------------- #
# Round reset and frame update
# --------------------------------------------------------------------------- #
@nb.njit(cache=True)
def reset_env(S, e, n, enabled):
    S.n[e] = n
    sq = 80.0 * 80.0
    S.size[e] = math.floor(math.sqrt(sq + (n - 1) * sq / 5.0) + 0.5)
    S.t[e] = 0.0
    S.frames[e] = 0
    S.deaths[e] = 0
    S.over[e] = False
    S.print_started[e] = False
    S.overflow[e] = 0
    for t in range(NB):
        S.bon_enabled[e, t] = enabled[t]
    clear_world(S, e)
    for b in range(BONUS_CAP):
        S.btype[e, b] = -1
    for f in range(MAX_FX):
        S.ftype[e, f] = -1
        S.fmask[e, f] = 0
    S.next_pop[e] = BONUS_POP_MS * (1.0 + np.random.random())
    size = S.size[e]
    for p in range(S.x.shape[1]):
        S.alive[e, p] = p < n
        S.vel[e, p] = BASE_VEL
        S.rad[e, p] = BASE_R
        S.inv[e, p] = False
        S.invinc[e, p] = False
        S.straight[e, p] = False
        S.turn[e, p] = 0
        S.move[e, p] = 0
        S.printing[e, p] = False
        S.has_tl[e, p] = False
        S.bc[e, p] = 0
        S.pm_active[e, p] = False
        S.pm_dist[e, p] = 0.0
        S.since_toggle[e, p] = 0.0
        S.score[e, p] = 0.0
        S.death_t[e, p] = 0.0
        S.cause[e, p] = -1
        S.killer[e, p] = -1
        S.human[e, p] = False
        S.conf_flip[e, p] = False
        S.conf_p0[e, p] = 0.0
        S.conf_end[e, p] = 0.0
        S.conf_dur[e, p] = 1.0
        for t in range(NB):
            S.pickups[e, p, t] = 0
        for k in range(RING):
            S.sched[e, p, k] = NO_MOVE
        for k in range(HIST):
            S.hist[e, p, k] = 0
        if p < n:
            margin = BASE_R + SPAWN_MARGIN * size
            S.x[e, p] = random_point(size, margin)
            S.y[e, p] = random_point(size, margin)
            amargin = SPAWN_ANGLE_MARGIN * size
            a = np.random.random() * 2 * math.pi
            for _ in range(1000):
                if direction_valid(size, a, S.x[e, p], S.y[e, p], amargin):
                    break
                a = np.random.random() * 2 * math.pi
            S.ang[e, p] = a
        else:
            S.x[e, p] = -100.0
            S.y[e, p] = -100.0


@nb.njit(cache=True)
def step_frame(S, e, dt):
    """One Game.update(step) plus the timers that fire around it."""
    S.t[e] += dt
    S.frames[e] += 1
    expire_effects(S, e)
    if not S.print_started[e] and S.t[e] >= PRINT_DELAY:
        S.print_started[e] = True
        for p in range(S.n[e]):
            if S.alive[e, p]:
                pm_start(S, e, p)
    if S.t[e] >= S.next_pop[e]:
        has_bonus = False
        for t in range(NB):
            has_bonus |= S.bon_enabled[e, t]
        if has_bonus:
            pop_bonus(S, e)
        else:
            S.next_pop[e] = 1e18

    score = S.deaths[e]
    size = S.size[e]
    bless = borderless(S, e)
    death = False
    for p in range(S.n[e] - 1, -1, -1):
        if not S.alive[e, p]:
            continue
        # updateAngle
        if S.turn[e, p] != 0:
            if S.straight[e, p]:
                S.ang[e, p] += S.turn[e, p] * (math.pi / 2.0)
                S.turn[e, p] = 0
            else:
                S.ang[e, p] += S.turn[e, p] * base_angular_velocity(S.vel[e, p]) * dt
        # updatePosition
        v = S.vel[e, p] / 1000.0
        S.x[e, p] += math.cos(S.ang[e, p]) * v * dt
        S.y[e, p] += math.sin(S.ang[e, p]) * v * dt
        num_ref = S.bc[e, p]
        r = S.rad[e, p]
        if S.printing[e, p]:
            if not S.has_tl[e, p]:
                add_body(S, e, p, S.x[e, p], S.y[e, p])
            else:
                dx = S.tl_x[e, p] - S.x[e, p]
                dy = S.tl_y[e, p] - S.y[e, p]
                if math.sqrt(dx * dx + dy * dy) > r:
                    add_body(S, e, p, S.x[e, p], S.y[e, p])
        # borders
        x = S.x[e, p]
        y = S.y[e, p]
        m = 0.0 if bless else r
        border = -1
        if x - m < 0:
            border = 0
        elif x + m > size:
            border = 1
        elif y - m < 0:
            border = 2
        elif y + m > size:
            border = 3
        if border >= 0:
            if bless:
                if border == 0:
                    S.x[e, p] = size
                elif border == 1:
                    S.x[e, p] = 0.0
                elif border == 2:
                    S.y[e, p] = size
                else:
                    S.y[e, p] = 0.0
            else:
                kill(S, e, p, score, CAUSE_WALL, -1)
                death = True
        elif not S.invinc[e, p]:
            k = query_body(S, e, x, y, r, p, num_ref)
            if k >= 0:
                kill(S, e, p, score, CAUSE_OWN if S.pown[e, k] == p else CAUSE_OTHER, S.pown[e, k])
                death = True
        if S.alive[e, p]:
            pm_test(S, e, p)
            catch_bonus(S, e, p)

    if death:
        n_alive = 0
        last = -1
        for p in range(S.n[e]):
            if S.alive[e, p]:
                n_alive += 1
                last = p
        if n_alive <= 1:
            S.over[e] = True
            if last >= 0:
                S.score[e, last] = max(S.n[e] - 1, 1)


@nb.njit(cache=True)
def potential_score(S, e, p):
    """Score the player is guaranteed to have so far (Curvytron round scoring)."""
    if S.alive[e, p] and not S.over[e]:
        return float(S.deaths[e])
    return S.score[e, p]


@nb.njit(cache=True)
def seed(s):
    np.random.seed(s)


@nb.njit(cache=True)
def queue_move(S, e, p, move, extra):
    """Send ``move`` now; the server applies it ``delay + extra`` frames later."""
    k = (S.frames[e] + S.delay[e, p] + extra) % RING
    S.sched[e, p, k] = move
    for i in range(HIST - 1, 0, -1):
        S.hist[e, p, i] = S.hist[e, p, i - 1]
    S.hist[e, p, 0] = move


@nb.njit(cache=True)
def apply_queued(S, e):
    """Apply the moves that reach the server at the coming frame."""
    k = S.frames[e] % RING
    for p in range(S.n[e]):
        m = S.sched[e, p, k]
        if m != NO_MOVE:
            S.sched[e, p, k] = NO_MOVE
            if S.alive[e, p]:
                set_move(S, e, p, m)
