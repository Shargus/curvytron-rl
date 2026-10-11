"""Render simulator states to RGB frames and animated GIFs (for training videos)."""
import io
import math

import numba as nb
import numpy as np
from PIL import Image

from .core import B_AFFECT, AFF_SELF, AFF_ENEMY, BONUS_CAP, BONUS_R, MAX_FX, B_BORDERLESS, RASTER_RES

# one colour per player slot (16 max), close to the game's palette
PALETTE = np.array([
    [255, 70, 70], [80, 170, 255], [255, 210, 60], [90, 220, 110], [220, 100, 255], [255, 140, 40],
    [60, 230, 220], [255, 120, 190], [170, 200, 60], [150, 140, 255], [230, 160, 120], [120, 160, 150],
    [200, 60, 120], [80, 120, 220], [210, 210, 210], [140, 90, 60],
], np.uint8)
BG = np.array([34, 34, 34], np.uint8)
BONUS_COLORS = np.array([[60, 200, 90], [230, 60, 60], [70, 120, 255]], np.uint8)  # self / enemy / game+all


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
def render_frame(S, e, img, highlight, palette, bg, bonus_colors):
    """Draw env ``e`` into img[H, W, 3] (square). ``highlight`` = slot ringed in white (-1: none)."""
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
    for b in range(BONUS_CAP):
        t = S.btype[e, b]
        if t >= 0:
            aff = B_AFFECT[t]
            ci = 0 if aff == AFF_SELF else (1 if aff == AFF_ENEMY else 2)
            _ring(img, S.bx[e, b] * scale, S.by[e, b] * scale, BONUS_R * scale - 1.5, BONUS_R * scale,
                  bonus_colors[ci])
    white = np.array([255, 255, 255], np.uint8)
    black = np.array([0, 0, 0], np.uint8)
    for p in range(S.n[e]):
        if not S.alive[e, p]:
            continue
        cx = S.x[e, p] * scale
        cy = S.y[e, p] * scale
        r = max(S.rad[e, p] * scale, 1.0) + 2.0
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


def render(S, e=0, size_px=256, highlight=-1):
    img = np.zeros((size_px, size_px, 3), np.uint8)
    render_frame(S, e, img, highlight, PALETTE, BG, BONUS_COLORS)
    return img


def encode_gif(frames, fps):
    """frames: list/array of [H, W, 3] uint8 -> GIF bytes (loops forever)."""
    images = [Image.fromarray(f).convert('P', palette=Image.ADAPTIVE, colors=64) for f in frames]
    buf = io.BytesIO()
    images[0].save(buf, format='GIF', save_all=True, append_images=images[1:],
                   duration=int(round(1000 / fps)), loop=0, optimize=False)
    return buf.getvalue()
