"""World-anchored distance dots for the AEB debug view.

The scene is ego-locked, so this lattice is the only thing on screen that moves
when ego does. See `core/aeb/README.md` section 11 for why a replay needs it.
"""

from __future__ import annotations

import math
from typing import Callable

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QColor, QBrush

STEP_MINOR_M = 10.0
STEP_MAJOR_M = 100.0
MINOR_CLR = QColor(108, 108, 134, 190)
MAJOR_CLR = QColor(165, 165, 205, 235)
_MINOR_R = 1.3
_MAJOR_R = 3.4
_MAX_DOTS = 12000

_Project = Callable[[float, float], tuple[float, float]]


def draw_ground_markers(
    p: QPainter, project: _Project, unproject: _Project, w: float, h: float,
) -> None:
    """A dot per 10 m of world grid, a larger one per 100 m, culled to the window."""
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

    minor: list[QPointF] = []
    major: list[QPointF] = []
    for i in range(i0, i1 + 1):
        wx = i * step
        for j in range(j0, j1 + 1):
            sx, sy = project(wx, j * step)
            if not (-4.0 <= sx <= w + 4.0 and -4.0 <= sy <= h + 4.0):
                continue
            is_major = i % major_every == 0 and j % major_every == 0
            (major if is_major else minor).append(QPointF(sx, sy))

    p.setPen(Qt.NoPen)
    for pts, clr, r in ((minor, MINOR_CLR, _MINOR_R), (major, MAJOR_CLR, _MAJOR_R)):
        p.setBrush(QBrush(clr))
        for pt in pts:
            p.drawEllipse(pt, r, r)
