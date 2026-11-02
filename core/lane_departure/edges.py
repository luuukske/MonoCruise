"""What lies past each line of ego's lane, from the map: the next lane, an oncoming lane, or the road's edge.

Pure logic on a ``LaneWindow`` (core/road_layout/game_map.py). The map is a prior: no window means ``unknown``.
See core/lane_departure/README.md, "Lane keeping assist".
"""

from __future__ import annotations

import math

LANE = "lane"
ONCOMING = "oncoming"
EDGE = "edge"
UNKNOWN = "unknown"

_LANE_STEP_M = 4.5
_MATCH_M = 1.5
_PARALLEL = 0.9


def point_at(pts, s: float) -> tuple[float, float, float, float] | None:
    """``(x, z, ux, uz)`` ``s`` metres along the polyline, ``u`` its unit tangent; clamped to the ends."""
    walked = 0.0
    last = None
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        hop = math.hypot(bx - ax, bz - az)
        if hop < 1e-9:
            continue
        last = (ax, az, bx, bz, hop, walked)
        if s <= walked + hop:
            break
        walked += hop
    if last is None:
        return None
    ax, az, bx, bz, hop, start = last
    u = min(1.0, max(0.0, (s - start) / hop))
    ux, uz = (bx - ax) / hop, (bz - az) / hop
    return ax + u * (bx - ax), az + u * (bz - az), ux, uz


def _nearest(pts, x: float, z: float) -> tuple[float, float, float] | None:
    """``(distance, tangent x, tangent z)`` of the closest span of a lane, tangent in point order."""
    best = None
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        sx, sz = bx - ax, bz - az
        seg2 = sx * sx + sz * sz
        if seg2 < 1e-9:
            continue
        u = min(1.0, max(0.0, ((x - ax) * sx + (z - az) * sz) / seg2))
        dist = math.hypot(x - ax - u * sx, z - az - u * sz)
        if best is None or dist < best[0]:
            seg = math.sqrt(seg2)
            best = (dist, sx / seg, sz / seg)
    return best


def beyond(window, pts, s: float, sign: float) -> str:
    """What lies one lane to the right (``sign`` +1) or left (-1) of the route at ``s`` metres along it.

    A parallel lane there whose traffic runs ego's way is ``lane``, the other way ``oncoming``; none is ``edge``.
    """
    lanes = getattr(window, "lanes", ())
    index = getattr(window, "index", None)
    here = point_at(pts, s)
    if not lanes or index is None or here is None:
        return UNKNOWN
    cx, cz, ux, uz = here
    px, pz = cx - sign * uz * _LANE_STEP_M, cz + sign * ux * _LANE_STEP_M
    found = None
    for i in index.around(px, pz, _LANE_STEP_M):
        lane = lanes[i]
        hit = _nearest(getattr(lane, "pts", ()), px, pz)
        if hit is None or hit[0] > _MATCH_M:
            continue
        along = hit[1] * ux + hit[2] * uz
        if abs(along) >= _PARALLEL and (found is None or hit[0] < found[0]):
            found = (hit[0], along, int(getattr(lane, "travel", 0)))
    if found is None:
        return EDGE
    if not found[2]:
        return UNKNOWN
    return LANE if found[1] * found[2] > 0.0 else ONCOMING
