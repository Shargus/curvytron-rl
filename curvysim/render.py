"""Render simulator states to RGB frames and animated GIFs (for training videos)."""
import io
import math
import os

import numba as nb
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import (BONUS_NAMES, BONUS_CAP, BONUS_R, MAX_FX, RASTER_RES, NB,
                   B_SMALL, B_SLOW, B_FAST, B_MASTER, B_ESLOW, B_EFAST, B_EBIG, B_EINV, B_ESTRAIGHT,
                   B_BORDERLESS, B_COLOR, B_CLEAR)

# one colour per player slot (16 max), close to the game's palette
PALETTE = np.array([
    [255, 70, 70], [80, 170, 255], [255, 210, 60], [90, 220, 110], [220, 100, 255], [255, 140, 40],
    [60, 230, 220], [255, 120, 190], [170, 200, 60], [150, 140, 255], [230, 160, 120], [120, 160, 150],
    [200, 60, 120], [80, 120, 220], [210, 210, 210], [140, 90, 60],
], np.uint8)
BG = np.array([34, 34, 34], np.uint8)
ICON_MIN_R = 12     # bonus icons are drawn at least this big (pixels), even if the catch area is smaller
# the game's own bonus icons (Curvytron web/images/bonus.png, MIT, see assets/LICENSE-curvytron.txt):
# 3 x 4 cells, row by row in this order (client BonusManager.spritePosition)
SPRITE_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'bonus.png')
SPRITE_ORDER = ('BonusSelfFast', 'BonusEnemyFast', 'BonusSelfSlow', 'BonusEnemySlow', 'BonusGameBorderless',
                'BonusSelfMaster', 'BonusEnemyBig', 'BonusAllColor', 'BonusEnemyInverse', 'BonusSelfSmall',
                'BonusGameClear', 'BonusEnemyStraightAngle')
BONUS_LABELS = {    # legend text per bonus type
    B_SMALL: 'small (you)', B_SLOW: 'slow (you)', B_FAST: 'fast (you)', B_MASTER: 'invincible (you)',
    B_ESLOW: 'slow (others)', B_EFAST: 'fast (others)', B_EBIG: 'big (others)', B_EINV: 'inverted (others)',
    B_ESTRAIGHT: 'right angles (others)', B_BORDERLESS: 'no borders', B_COLOR: 'swap colours',
    B_CLEAR: 'clear trails',
}
# letters drawn in the heads (``labels`` of render) and their legend entries
PLAYER_KINDS = (('B', 'bot being trained'), ('E', 'exploiter'), ('L', 'lookahead heuristic'),
                ('P', 'past version'), ('H', 'human-like'))
LABEL_R = 5.5       # head radius (pixels) with a letter inside: just enough for it to be legible


@nb.njit(cache=True)
def _disc(img, cx, cy, r, color):
    H = img.shape[0]
    W = img.shape[1]
    for i in range(max(0, int(cy - r - 1)), min(H, int(cy + r + 2))):
        for j in range(max(0, int(cx - r - 1)), min(W, int(cx + r + 2))):
            if (j + 0.5 - cx) ** 2 + (i + 0.5 - cy) ** 2 <= r * r:
                img[i, j, 0] = color[0]
                img[i, j, 1] = color[1]
                img[i, j, 2] = color[2]


@nb.njit(cache=True)
def _ring(img, cx, cy, r0, r1, color):
    H = img.shape[0]
    W = img.shape[1]
    for i in range(max(0, int(cy - r1 - 1)), min(H, int(cy + r1 + 2))):
        for j in range(max(0, int(cx - r1 - 1)), min(W, int(cx + r1 + 2))):
            d2 = (j + 0.5 - cx) ** 2 + (i + 0.5 - cy) ** 2
            if r0 * r0 <= d2 <= r1 * r1:
                img[i, j, 0] = color[0]
                img[i, j, 1] = color[1]
                img[i, j, 2] = color[2]


@nb.njit(cache=True)
def _draw_board(S, e, img, palette, bg):
    """Trails and border of env ``e`` into img[H, W, 3] (square)."""
    H = img.shape[0]
    size = S.size[e]
    scale = H / size
    bless = False
    for f in range(MAX_FX):
        if S.ftype[e, f] == B_BORDERLESS:
            bless = True
    for i in range(H):
        wy = (i + 0.5) / scale
        gj = int(wy / RASTER_RES)
        for j in range(H):
            wx = (j + 0.5) / scale
            gi = int(wx / RASTER_RES)
            k = S.raster[e, gi, gj]
            if k >= 0 and k < S.npts[e]:
                c = palette[S.pown[e, k] % palette.shape[0]]
            else:
                c = bg
            img[i, j, 0] = c[0]
            img[i, j, 1] = c[1]
            img[i, j, 2] = c[2]
    # border: grey, or dashed blue when borderless
    for t in range(H):
        on = (not bless) or (t // 4) % 2 == 0
        if on:
            for a, b in ((0, t), (H - 1, t), (t, 0), (t, H - 1)):
                img[a, b, 0] = 70 if bless else 120
                img[a, b, 1] = 110 if bless else 120
                img[a, b, 2] = 255 if bless else 120


@nb.njit(cache=True)
def _draw_heads(S, e, img, highlight, palette, r_min):
    """Heads (radius at least ``r_min`` pixels) and headings; ``highlight`` = slot ringed in white
    (-1: none)."""
    scale = img.shape[0] / S.size[e]
    white = np.array([255, 255, 255], np.uint8)
    black = np.array([0, 0, 0], np.uint8)
    for p in range(S.n[e]):
        if not S.alive[e, p]:
            continue
        cx = S.x[e, p] * scale
        cy = S.y[e, p] * scale
        r = max(max(S.rad[e, p] * scale, 1.0) + 2.0, r_min)
        col = palette[p % palette.shape[0]]
        # heading: a short line ahead of the head
        for s in range(1, 6):
            d = r + s * 1.2
            _disc(img, cx + math.cos(S.ang[e, p]) * d, cy + math.sin(S.ang[e, p]) * d, 0.9,
                  white if p == highlight else col)
        _ring(img, cx, cy, r, r + 1.5, white if p == highlight else black)
        if p == highlight:
            _ring(img, cx, cy, r + 1.5, r + 3.0, black)
        _disc(img, cx, cy, r, col)


_icons = {}
_sprite = []


def bonus_icon(t, r):
    """RGBA icon of bonus type ``t``, ``2r + 1`` pixels wide: the game's own sprite (green:
    affects the player catching it, red: its opponents, grey: everybody)."""
    key = (t, r)
    if key in _icons:
        return _icons[key]
    if not _sprite:
        _sprite.append(Image.open(SPRITE_PATH).convert('RGBA'))
    sheet = _sprite[0]
    w, h = sheet.width // 3, sheet.height // 4
    i = SPRITE_ORDER.index(BONUS_NAMES[t])
    cell = sheet.crop(((i % 3) * w, (i // 3) * h, (i % 3 + 1) * w, (i // 3 + 1) * h))
    icon = np.asarray(cell.resize((2 * r + 1, 2 * r + 1), Image.LANCZOS))
    _icons[key] = icon
    return icon


def _paste(img, icon, cx, cy):
    """Alpha-blend an RGBA icon centred at (cx, cy) into img, clipped to its bounds."""
    h = icon.shape[0]
    x0 = int(round(cx)) - h // 2
    y0 = int(round(cy)) - h // 2
    ix0, iy0 = max(x0, 0), max(y0, 0)
    ix1, iy1 = min(x0 + h, img.shape[1]), min(y0 + h, img.shape[0])
    if ix0 >= ix1 or iy0 >= iy1:
        return
    sub = icon[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    dst = img[iy0:iy1, ix0:ix1]
    dst[:] = (sub[:, :, :3] * a + dst * (1.0 - a)).astype(np.uint8)


_legends = {}


def _letter(g, cx, cy, letter, color, font):
    """A head letter, black on light player colours and white on dark ones."""
    light = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2] > 140
    g.text((cx, cy), letter, fill=(0, 0, 0) if light else (255, 255, 255), font=font, anchor='mm')


def legend(height, kinds=False):
    """Side panel naming every bonus icon (and, with ``kinds``, the head letters), for frames
    ``height`` pixels tall."""
    key = (height, kinds)
    if key in _legends:
        return _legends[key]
    rows = NB + (len(PLAYER_KINDS) + 1 if kinds else 0)
    row = height / rows
    r = max(int(row * 0.36), 5)
    font = ImageFont.load_default(size=max(int(row * 0.45), 9))
    im = Image.new('RGB', (int(height * 0.42), height), tuple(int(v) for v in BG))
    g = ImageDraw.Draw(im)
    g.line([(0, 0), (0, height)], fill=(90, 90, 90), width=1)
    for t in range(NB):
        cy = int((t + 0.5) * row)
        im.paste(Image.fromarray(bonus_icon(t, r)), (8, cy - r), Image.fromarray(bonus_icon(t, r)))
        g.text((8 + 2 * r + 8, cy), BONUS_LABELS[t], fill=(220, 220, 220), font=font, anchor='lm')
    if kinds:
        head_font = ImageFont.load_default(size=max(int(r * 1.3), 8))
        grey = (200, 200, 200)
        for i, (letter, name) in enumerate(PLAYER_KINDS):
            cy = int((NB + 1 + i + 0.5) * row)
            g.ellipse([8 + r - r * 0.8, cy - r * 0.8, 8 + r + r * 0.8, cy + r * 0.8], fill=grey, outline=(0, 0, 0))
            _letter(g, 8 + r, cy, letter, grey, head_font)
            g.text((8 + 2 * r + 8, cy), name, fill=(220, 220, 220), font=font, anchor='lm')
    _legends[key] = np.asarray(im)
    return _legends[key]


_head_font = []


def render(S, e=0, size_px=256, highlight=-1, with_legend=False, labels=None):
    """RGB frame of env ``e``: trails, bonus icons (see ``bonus_icon``), heads; ``highlight`` = slot
    ringed in white. ``labels`` (one letter per slot, see PLAYER_KINDS) are written in the heads,
    which are then a little bigger. ``with_legend`` adds a side panel naming the icons and letters."""
    img = np.zeros((size_px, size_px, 3), np.uint8)
    _draw_board(S, e, img, PALETTE, BG)
    scale = size_px / S.size[e]
    r = max(int(round(BONUS_R * scale)), ICON_MIN_R)
    for b in range(BONUS_CAP):
        t = S.btype[e, b]
        if t >= 0:
            _paste(img, bonus_icon(int(t), r), S.bx[e, b] * scale, S.by[e, b] * scale)
    _draw_heads(S, e, img, highlight, PALETTE, LABEL_R if labels is not None else 0.0)
    if labels is not None:
        if not _head_font:
            _head_font.append(ImageFont.load_default(size=9))
        im = Image.fromarray(img)
        g = ImageDraw.Draw(im)
        for p in range(S.n[e]):
            if S.alive[e, p] and labels[p]:
                _letter(g, S.x[e, p] * scale, S.y[e, p] * scale, labels[p], PALETTE[p % len(PALETTE)], _head_font[0])
        img = np.asarray(im).copy()
    if with_legend:
        img = np.concatenate([img, legend(size_px, kinds=labels is not None)], axis=1)
    return img


def encode_gif(frames, fps):
    """frames: list/array of [H, W, 3] uint8 -> GIF bytes (loops forever)."""
    images = [Image.fromarray(f).convert('P', palette=Image.ADAPTIVE, colors=64) for f in frames]
    buf = io.BytesIO()
    images[0].save(buf, format='GIF', save_all=True, append_images=images[1:],
                   duration=int(round(1000 / fps)), loop=0, optimize=False)
    return buf.getvalue()
