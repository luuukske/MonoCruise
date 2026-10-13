"""Lanes that TruckersMP built a new road on top of and left under the ground.

A mod can place an intersection over existing roads without removing them: both stay
in the sector and the old road sits a metre or two below the new surface. A truck
needs about 4 m to pass under another lane, so a lane that lies within 3.5 m below
another over a stretch of road cannot be driven. Only lanes under something the mod
added count; the base game is never filtered. See core/map_thread/README.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Sequence

_STEP_M = 3.0
_LATERAL_M = 2.5
_BELOW_MIN_M = 0.8
_BELOW_MAX_M = 3.5
_COVER_MIN_M = 30.0
_COVER_FRACTION = 0.3


def _samples(lane) -> list[tuple[float, float, float]]:
    pts, ys = lane.pts, lane.ys
    out: list[tuple[float, float, float]] = []
    for i in range(len(pts) - 1):
        (x0, z0), (x1, z1) = pts[i], pts[i + 1]
        n = max(1, int(math.hypot(x1 - x0, z1 - z0) / _STEP_M))
        for k in range(n):
            f = k / n
            out.append((x0 + f * (x1 - x0), z0 + f * (z1 - z0), ys[i] + f * (ys[i + 1] - ys[i])))
    out.append((pts[-1][0], pts[-1][1], ys[-1]))
    return out


def _usable(lane) -> bool:
    return len(lane.pts) >= 2 and len(lane.ys) == len(lane.pts)


def drop_buried(lanes: Sequence, added_uids: Collection[int]) -> tuple:
    """``lanes`` without those lying 0.8 to 3.5 m under a lane of an added item for 30 m or more."""
    uppers = [lane for lane in lanes if lane.uid in added_uids and _usable(lane)]
    if not uppers:
        return tuple(lanes)
    grid: dict[tuple[int, int], list[tuple[float, float, float, int]]] = defaultdict(list)
    for lane in uppers:
        for x, z, y in _samples(lane):
            grid[(int(x // _LATERAL_M), int(z // _LATERAL_M))].append((x, z, y, lane.uid))
    return tuple(lane for lane in lanes if not (_usable(lane) and _buried(lane, grid)))


def _under(grid, uid: int, x: float, z: float, y: float) -> bool:
    cx, cz = int(x // _LATERAL_M), int(z // _LATERAL_M)
    for gx in (cx - 1, cx, cx + 1):
        for gz in (cz - 1, cz, cz + 1):
            for qx, qz, qy, other in grid.get((gx, gz), ()):
                if (
                    other != uid
                    and (qx - x) ** 2 + (qz - z) ** 2 <= _LATERAL_M ** 2
                    and _BELOW_MIN_M <= qy - y <= _BELOW_MAX_M
                ):
                    return True
    return False


def _buried(lane, grid) -> bool:
    samples = _samples(lane)
    covered = sum(1 for x, z, y in samples if _under(grid, lane.uid, x, z, y))
    return covered * _STEP_M >= _COVER_MIN_M and covered >= _COVER_FRACTION * len(samples)
