"""World-anchored distance dots for the AEB debug view.

The scene is ego-locked, so this lattice is the only thing on screen that moves
when ego does. See `core/aeb/README.md` section 11 for why a replay needs it.

Dots are blitted from pre-rendered sprites at ``_PHASES`` sub-pixel offsets per
axis: an AA ellipse per dot cost about 4 us each at 175% scaling, and a sprite
snapped to whole pixels made slow motion step.
"""

from __future__ import annotations

import math
from typing import Callable

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QColor, QBrush, QPixmap

STEP_MINOR_M = 10.0
STEP_MAJOR_M = 100.0
MINOR_CLR = QColor(108, 108, 134, 190)
MAJOR_CLR = QColor(165, 165, 205, 235)
_MINOR_R = 1.3
_MAJOR_R = 3.4
_MAX_DOTS = 12000
_PHASES = 4

_Project = Callable[[float, float], tuple[float, float]]
_sprites: dict = {}


def draw_ground_markers(
    p: QPainter, project: _Project, unproject: _Project, w: float, h: float,
    *, minor_clr: QColor = MINOR_CLR, major_clr: QColor = MAJOR_CLR,
    minor_r: float = _MINOR_R, major_r: float = _MAJOR_R,
) -> None:
    """A dot per 10 m of world grid, a larger one per 100 m, culled to the window.

    The style overrides exist for `tools/clip_export`, which draws the same lattice darker.
    """
    xs, zs = zip(*(unproject(sx, sy) for sx, sy in
                   ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h))))

    def span(step: float) -> tuple[int, int, int, int]:
        return (math.floor(min(xs) / step), math.ceil(max(xs) / step),
                math.floor(min(zs) / step), math.ceil(max(zs) / step))

    step = STEP_MINOR_M
    i0, i1, j0, j1 = span(step)
    if (i1 - i0 + 1) * (j1 - j0 + 1) > _MAX_DOTS:
        # Absurd window size: keep the coarse lattice rather than dropping the cue.
        step = STEP_MAJOR_M
        i0, i1, j0, j1 = span(step)
    major_every = max(1, round(STEP_MAJOR_M / step))

    minor: list[tuple[float, float]] = []
    major: list[tuple[float, float]] = []
    for i in range(i0, i1 + 1):
        wx = i * step
        for j in range(j0, j1 + 1):
            sx, sy = project(wx, j * step)
            if not (-4.0 <= sx <= w + 4.0 and -4.0 <= sy <= h + 4.0):
                continue
            is_major = i % major_every == 0 and j % major_every == 0
            (major if is_major else minor).append((sx, sy))

    # Sprites are placed on whole device pixels, so draw in device coordinates.
    dev = p.deviceTransform()
    dpr = math.hypot(dev.m11(), dev.m12())
    a, b, c, d, tx, ty = dev.m11(), dev.m12(), dev.m21(), dev.m22(), dev.dx(), dev.dy()
    p.save()
    p.setWorldTransform(dev.inverted()[0] * p.worldTransform())
    for pts, clr, r in ((minor, minor_clr, minor_r), (major, major_clr, major_r)):
        c0, sprites = _dot_sprites(clr, r, dpr)
        for sx, sy in pts:
            dx = a * sx + c * sy + tx - c0
            dy = b * sx + d * sy + ty - c0
            ix = math.floor(dx)
            iy = math.floor(dy)
            p.drawPixmap(ix, iy, sprites[int((dx - ix) * _PHASES)][int((dy - iy) * _PHASES)])
    p.restore()


def _dot_sprites(clr: QColor, r: float, dpr: float) -> tuple[int, list[list[QPixmap]]]:
    """Dot sprites per sub-pixel phase in device pixels, and the sprite origin to subtract."""
    key = (clr.rgba(), r, dpr)
    hit = _sprites.get(key)
    if hit is not None:
        return hit
    rd = r * dpr
    c0 = math.ceil(rd) + 1
    n = 2 * c0 + 1
    grid = []
    for fx in range(_PHASES):
        col = []
        for fy in range(_PHASES):
            pm = QPixmap(n, n)
            pm.fill(Qt.transparent)
            sp = QPainter(pm)
            sp.setRenderHint(QPainter.Antialiasing, True)
            sp.setPen(Qt.NoPen)
            sp.setBrush(QBrush(clr))
            sp.drawEllipse(QPointF(c0 + (fx + 0.5) / _PHASES, c0 + (fy + 0.5) / _PHASES), rd, rd)
            sp.end()
            col.append(pm)
        grid.append(col)
    if len(_sprites) > 32:
        _sprites.clear()
    _sprites[key] = (c0, grid)
    return _sprites[key]
