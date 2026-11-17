"""Target follow: a mover on its own map lane is judged on the road path's prediction of it.

The last stage: every branch of the mover's lane, joined by position over a short ramp,
must stay out of ego's corridor. No map, no verdict. See ``core/aeb/README.md``,
"Target follow".
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.aeb.calibration import AEBCalibration
from core.aeb.filters import _PASS, FilterResult, _suppress
from core.aeb.lane_frame import Lane, _any_body_in_ego_lane, project_to_ego_arc
from core.radar.bent_path import bend_arc

# Spacing of the body samples along a predicted path.
_SAMPLE_M = 2.0
# Widest body half width the box test allows for (a wide load is about 2.5 m across).
_MAX_HALF_W = 2.0

if TYPE_CHECKING:
    from core.aeb.filters import FilterContext


class TargetFollowFilter:
    name = "TargetFollowFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal
        self._plain = None

    def apply(self, ctx: "FilterContext") -> FilterResult:
        cal = self._cal
        follow = ctx.follow_fn
        if follow is None or not cal.target_follow_enabled:
            return _PASS
        # A body AEB already brakes for is never let go on the map's word.
        if ctx.v.id in ctx.latched_threat_ids or getattr(ctx.v, "crash_confirmed", False):
            return _PASS
        # Oncoming traffic stays with the oncoming stages and their measured-closing rescue.
        if ctx.head_on or ctx.follow_s < cal.target_follow_hold_s:
            return _PASS
        # A body predicted along the map meets ego predicted along it too, when ego follows its
        # lane: a lead on a bend never parts from ego's plain arc otherwise (README).
        # The plain arc as a path, so distances read off the arc itself: the lane primitive's
        # straight-line guard hid ego curving into the next lane at 90 km/h (c0686b57).
        egos = [self._plain_path(ctx.ego_arc)]
        road = getattr(ctx, "ego_road_arc", None)
        if road is not None and road is not ctx.ego_arc:
            egos.append(road)
        ego_follow = getattr(ctx, "ego_follow_fn", None)
        if ego_follow is not None:
            egos.extend(ego_follow())
        # A body already in ego's lane is not one that keeps to another.
        if ctx.lane == Lane.EGO or any(
                _any_body_in_ego_lane(ego, ctx.all_target_arcs, cal.lane_half_width) for ego in egos):
            return _PASS
        branches = follow()
        if not branches:
            return _PASS
        # Where, not when: a crosser whose lane crosses ego's path is kept however the timing
        # reads, as turn completion and lane keep judge (README).
        reach = [(ego, _box(ego, ego.fwd_len + ctx.ego_hw + _MAX_HALF_W + cal.corridor_margin
                            + cal.target_follow_clear_m)) for ego in egos]
        for arcs in branches:
            for ego, box in reach:
                if _enters_corridor(ego, box, arcs, ctx.ego_hw, cal):
                    return _PASS
        return _suppress(self.name)

    def _plain_path(self, arc):
        """The plain arc as a path, built once per frame: several bodies ask in one frame."""
        if getattr(arc, "bent", False):
            return arc
        if self._plain is None or self._plain[0] is not arc:
            self._plain = (arc, bend_arc(arc, (0.0,), (arc.curvature,), keep_flat=True))
        return self._plain[1]


def _box(path, grow: float):
    """Bounding box of a path's polyline grown by ``grow``, or None for a plain arc."""
    xs, zs = getattr(path, "_xs", None), getattr(path, "_zs", None)
    if not xs:
        return None
    return min(xs) - grow, max(xs) + grow, min(zs) - grow, max(zs) + grow


def _enters_corridor(ego, ego_box, arcs, ego_hw: float, cal: AEBCalibration) -> bool:
    """Any body of ``arcs`` along its whole path within ego's corridor anywhere ego's front reaches.

    Ego's body runs past its path point ahead: ending the reach at the arc's end missed a
    crosser just in front of a slow ego's bumper (5118b3f6).
    """
    lo, hi = 0.0, ego.arc_length + ego.fwd_len
    for arc in arcs:
        box = _box(arc, max(arc.fwd_len, arc.back_len))
        # Boxes that cannot meet: no sample of this body can be inside the corridor.
        if (ego_box is not None and box is not None
                and (box[0] > ego_box[1] or box[1] < ego_box[0]
                     or box[2] > ego_box[3] or box[3] < ego_box[2])):
            continue
        bar = ego_hw + arc.half_width + cal.corridor_margin + cal.target_follow_clear_m
        n = max(2, int(arc.arc_length / _SAMPLE_M) + 1)
        for k in range(n + 1):
            d = arc.arc_length * k / n
            x, z = arc.position_at_dist(d)
            h = arc.heading_at_dist(d)
            fx, fz = -math.sin(h), -math.cos(h)
            for off in (arc.fwd_len, 0.0, -arc.back_len):
                s, d_abs = project_to_ego_arc(ego, x + off * fx, z + off * fz)
                if lo <= s <= hi and d_abs < bar:
                    return True
    return False
