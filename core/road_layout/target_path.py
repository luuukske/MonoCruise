"""Lane-following paths for other vehicles, the way ``map_path`` builds ego's.

Each tracked vehicle gets its own ``EgoRouter`` (lane hysteresis, routes, branch
divergence), fed the vehicle's own positions and heading. The predicted path
leaves the vehicle on its measured curvature and follows its lane further out,
at the offset it holds now. AI traffic drives the nav lanes almost exactly;
TruckersMP players do not, so consumers weigh the two differently.
"""

from __future__ import annotations

import math

from core.road_layout.ego_route import EgoRouter, LaneRoute
from core.road_layout.map_path import DEFAULT_PARAMS, MapPath, PathParams, build_map_path


class TargetRouter:
    """One lane router per vehicle id; ``prune`` drops vehicles no longer seen."""

    def __init__(self) -> None:
        self._routers: dict[int, EgoRouter] = {}

    def route(self, vid: int, window, x: float, y: float, z: float, fwd_x: float, fwd_z: float) -> LaneRoute | None:
        router = self._routers.get(vid)
        if router is None:
            router = EgoRouter()
            self._routers[vid] = router
        return router.step(window, x, y, z, fwd_x, fwd_z)

    def prune(self, live_ids) -> None:
        keep = set(live_ids)
        for vid in [v for v in self._routers if v not in keep]:
            del self._routers[vid]


def target_path(
    route: LaneRoute | None,
    x: float,
    z: float,
    yaw_rad: float,
    kappa: float,
    speed: float,
    params: PathParams = DEFAULT_PARAMS,
) -> MapPath | None:
    """Fused path for a vehicle at ``(x, z)`` with heading ``yaw_rad`` (arc convention)."""
    if route is None:
        return None
    fwd_x, fwd_z = -math.sin(yaw_rad), -math.cos(yaw_rad)
    return build_map_path(route, x, z, fwd_x, fwd_z, kappa, speed, params)
