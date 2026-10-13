"""Ego's fused map path placed in the world, plus lane queries, built once per frame.

The path leaves ego on the consumer's own curvature (AEB: ``EgoPathModel``;
ACC: its steer/history blend) and follows the map lane further out
(``map_path.py``). None whenever the map worker has nothing fresh.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from core.road_layout.map_path import MapPath, build_map_path, world_points

logger = logging.getLogger(__name__)

_ROAD_KAPPA_SPAN_M = 20.0
_LANE_QUERY_M = 8.0


@dataclass(frozen=True)
class PathContext:
    """Fused ego path in the frame of the consumer's arc start, plus the lanes around ego."""

    path: MapPath
    ox: float
    oz: float
    fwd_x: float
    fwd_z: float
    kappa_ego: float
    kappa_road: float
    route: object
    window: object

    def local(self, wx: float, wz: float) -> tuple[float, float]:
        dx = wx - self.ox
        dz = wz - self.oz
        return dx * self.fwd_x + dz * self.fwd_z, dx * (-self.fwd_z) + dz * self.fwd_x

    def path_coords(self, wx: float, wz: float) -> tuple[float, float] | None:
        """``(s, metres right of the fused path)``; None off either end."""
        return self.path.coords(*self.local(wx, wz))

    def arc_lateral(self, s: float) -> float:
        """Lateral of the steer arc at arc length ``s``, in the same frame."""
        k = self.kappa_ego
        if abs(k) < 1e-6:
            return 0.0
        return -(1.0 - math.cos(k * s)) / k

    def blend(self, wx: float, wz: float, lateral: float) -> tuple[float, float]:
        """``lateral`` pulled toward the offset from the map path by its trust there: ``(lat, w)``."""
        hit = self.path_coords(wx, wz)
        if hit is None:
            return lateral, 0.0
        w = self.path.trust_at(hit[0])
        return w * hit[1] + (1.0 - w) * lateral, w

    def lane_gap(self, wx: float, wz: float) -> float:
        """Distance to the nearest lane centreline of any direction; inf when none is close."""
        index = getattr(self.window, "index", None)
        lanes = getattr(self.window, "lanes", ())
        if index is None:
            return math.inf
        best = math.inf
        for i in index.around(wx, wz, _LANE_QUERY_M):
            pts = lanes[i].pts
            for k in range(len(pts) - 1):
                d = _segment_gap(wx, wz, pts[k], pts[k + 1])
                if d < best:
                    best = d
        return best

    def world_path(self) -> tuple[tuple[float, float], ...]:
        return world_points(self.path, self.ox, self.oz, self.fwd_x, self.fwd_z)


def build_path_context(
    snap,
    ego_x: float,
    ego_z: float,
    fwd_x: float,
    fwd_z: float,
    kappa: float,
    speed: float,
    origin_m: float,
) -> PathContext | None:
    """Context for this frame, or None when the snapshot has no route."""
    route = getattr(snap, "route", None)
    window = getattr(snap, "window", None)
    if route is None or window is None:
        return None
    path = build_map_path(route, ego_x, ego_z, fwd_x, fwd_z, kappa, speed, origin_m=origin_m)
    if path is None:
        return None
    ox = ego_x + origin_m * fwd_x
    oz = ego_z + origin_m * fwd_z
    return PathContext(
        path, ox, oz, fwd_x, fwd_z, kappa, _road_kappa(route.pts), route, window,
    )


def _road_kappa(pts) -> float:
    """Signed curvature of the lane over its first stretch ahead; positive turns left."""
    h0 = _heading_at(pts, 0.0)
    h1 = _heading_at(pts, _ROAD_KAPPA_SPAN_M)
    if h0 is None or h1 is None:
        return 0.0
    turn = math.atan2(h0[0] * h1[1] - h0[1] * h1[0], h0[0] * h1[0] + h0[1] * h1[1])
    # World XZ with Z south: a positive cross turns right of travel, the arc sign is left.
    return -turn / _ROAD_KAPPA_SPAN_M


def _heading_at(pts, s: float):
    left = s
    for i in range(1, len(pts)):
        hop = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if hop < 1e-9:
            continue
        if hop >= left or i == len(pts) - 1:
            return (pts[i][0] - pts[i - 1][0]) / hop, (pts[i][1] - pts[i - 1][1]) / hop
        left -= hop
    return None


def _segment_gap(x: float, z: float, a, b) -> float:
    sx, sz = b[0] - a[0], b[1] - a[1]
    length2 = sx * sx + sz * sz
    if length2 < 1e-12:
        return math.hypot(x - a[0], z - a[1])
    t = max(0.0, min(1.0, ((x - a[0]) * sx + (z - a[1]) * sz) / length2))
    return math.hypot(x - a[0] - t * sx, z - a[1] - t * sz)
