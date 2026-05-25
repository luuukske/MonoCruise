"""Named AEB filter pipeline — one class per suppression stage."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.radar.traffic import (
    ArcPath, Vehicle,
    build_arc, arc_arc_collision, _accel_to_arc_params,
)
from core.radar.ego_path import ego_curvature_from_history
from core.aeb.calibration import AEBCalibration
from core.aeb.lane_frame import Lane, project_to_ego_arc, classify


class OneEuroFilter:
    """Speed-adaptive low-pass — Casiez et al., "1€ Filter", CHI 2012.

    Cutoff frequency rises with |dx/dt|: heavy smoothing when the signal is
    quiet, near-passthrough when it changes fast.  Tradeoff knobs are
    ``min_cutoff`` (smooth-floor) and ``beta`` (how aggressively cutoff
    follows the derivative).

    ``beta_scale`` adds a magnitude-dependent attenuation of beta:
    ``beta_eff = beta / (1 + beta_scale * |x_prev|)``.  Useful when input
    noise scales with signal magnitude (e.g. yaw_rate/v amplification on
    curvature signals in turns).  Zero disables the scaling.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "beta_scale",
                 "_x_prev", "_dx_prev", "_t_prev")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0,
                 beta_scale: float = 0.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.beta_scale = beta_scale
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def step(self, x: float, t: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0.0:
            return self._x_prev
        dx_raw = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx_raw + (1.0 - a_d) * self._dx_prev
        beta_eff = self.beta / (1.0 + self.beta_scale * abs(self._x_prev))
        cutoff = self.min_cutoff + beta_eff * abs(dx_hat)
        a_x = self._alpha(cutoff, dt)
        x_hat = a_x * x + (1.0 - a_x) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class VehicleCurvatureBlender:
    """Per-vehicle One-Euro state for the blended target-curvature signal.

    Owned by the long-lived caller (``AEBThread``).  Each vehicle id gets its
    own filter so transient-rate adaptation is independent per target.  Stale
    entries are dropped by :meth:`prune` once per frame to bound memory.
    """

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal
        self._filters: dict[int, OneEuroFilter] = {}

    def _get(self, vid: int) -> OneEuroFilter:
        f = self._filters.get(vid)
        if f is None:
            f = OneEuroFilter(
                self._cal.aeb_kappa_one_euro_min_cutoff,
                self._cal.aeb_kappa_one_euro_beta,
                self._cal.aeb_kappa_one_euro_d_cutoff,
                self._cal.aeb_kappa_one_euro_beta_turn_scale,
            )
            self._filters[vid] = f
        return f

    def step(self, vid: int, raw_kappa: float, now: float) -> float:
        return self._get(vid).step(raw_kappa, now)

    def prune(self, active_vids: set[int]) -> None:
        for vid in list(self._filters.keys()):
            if vid not in active_vids:
                del self._filters[vid]


def _vehicle_curvature_blend(
    v: Vehicle,
    abs_v_speed: float,
    cal: AEBCalibration,
    blender: VehicleCurvatureBlender | None = None,
    now: float | None = None,
) -> float:
    """Blend short position-fit and yaw-rate signals for a target vehicle's path.

    AEB-local two-source path prediction — smooth (position fit on the last
    ``cal.aeb_pos_history_len`` history samples) blended with responsive
    (single-frame yaw rate from ``angular_velocity``).  Either side fills in
    when the other is unavailable.

    When ``blender`` is supplied, the blended value is fed through a
    per-vehicle One-Euro filter (see :class:`VehicleCurvatureBlender`).  The
    raw blend is returned when ``blender`` is ``None`` — used by test paths
    that don't carry filter state across frames.
    """
    pos_hist = list(v._position_history)[-cal.aeb_pos_history_len:]
    pos_kappa = ego_curvature_from_history(pos_hist) if len(pos_hist) >= 3 else None
    yaw_kappa = math.radians(v.angular_velocity) / abs_v_speed if abs_v_speed > 0.5 else None
    if pos_kappa is not None and yaw_kappa is not None:
        raw = cal.aeb_yaw_blend * yaw_kappa + (1.0 - cal.aeb_yaw_blend) * pos_kappa
    elif pos_kappa is not None:
        raw = pos_kappa
    elif yaw_kappa is not None:
        raw = yaw_kappa
    else:
        return 0.0
    if blender is None:
        return raw
    return blender.step(v.id, raw, now if now is not None else time.monotonic())

if TYPE_CHECKING:
    pass


@dataclass
class FilterResult:
    suppressed: bool
    reason: str | None = None


_PASS = FilterResult(suppressed=False)


def _pass(reason: str | None = None) -> FilterResult:
    return FilterResult(suppressed=False, reason=reason)


def _suppress(reason: str) -> FilterResult:
    return FilterResult(suppressed=True, reason=reason)


# ---- helpers moved from thread.py ----

def _cross_zone_padding(ego_yaw_rad: float, v_yaw_rad: float, v_speed_ms: float,
                        cal: AEBCalibration) -> float:
    cross_factor = abs(math.sin(ego_yaw_rad - v_yaw_rad))
    return cross_factor * (cal.cross_zone_base + cal.cross_zone_speed * v_speed_ms)


def _apply_cross_zone(arc: ArcPath, padding: float) -> list[ArcPath]:
    if padding < 0.1:
        return [arc]
    front = build_arc(
        arc.start_x + padding * arc.fwd_x,
        arc.start_z + padding * arc.fwd_z,
        arc.yaw_rad, arc.speed, arc.curvature, arc.half_width, arc.horizon,
        decel=arc.decel, accel=arc.accel,
    )
    rear = build_arc(
        arc.start_x - padding * arc.fwd_x,
        arc.start_z - padding * arc.fwd_z,
        arc.yaw_rad, arc.speed, arc.curvature, arc.half_width, arc.horizon,
        decel=arc.decel, accel=arc.accel,
    )
    return [arc, front, rear]


def _earliest_hit(
    ego_arc: ArcPath,
    check_arcs: list[ArcPath],
    margin: float,
    n_samples: int,
    min_lateral_gap: float = 0.0,
) -> tuple[float, float, float] | None:
    best: tuple[float, float, float] | None = None
    for ca in check_arcs:
        h = arc_arc_collision(ego_arc, ca, margin, n_samples, min_lateral_gap)
        if h is not None and (best is None or h[0] < best[0]):
            best = h
    return best


def _world_to_ego_forward(dx: float, dz: float, ego_yaw_rad: float) -> float:
    return dx * math.sin(ego_yaw_rad) + dz * math.cos(ego_yaw_rad)


def _is_approaching(a: ArcPath, b: ArcPath, t: float, dt: float = 0.1) -> bool:
    ax0, az0 = a.position_at_time(t)
    bx0, bz0 = b.position_at_time(t)
    ax1, az1 = a.position_at_time(t + dt)
    bx1, bz1 = b.position_at_time(t + dt)
    d0_sq = (ax0 - bx0) ** 2 + (az0 - bz0) ** 2
    d1_sq = (ax1 - bx1) ** 2 + (az1 - bz1) ** 2
    return d1_sq < d0_sq


def _dampen_turning_curvature(
    v_curvature: float,
    fwd_dot: float,
    ego_fwd_x: float, ego_fwd_z: float,
    veh_fwd_x: float, veh_fwd_z: float,
    abs_v_speed: float,
    arc_length: float,
    cal: AEBCalibration,
) -> float:
    if (abs(v_curvature) <= cal.turning_diverge_kappa
            or abs_v_speed <= 0.5
            or fwd_dot <= -0.5
            or fwd_dot >= 0.7):
        return v_curvature
    theta_max = abs(v_curvature) * arc_length
    theta_to_anti = math.acos(max(-1.0, min(1.0, -fwd_dot)))
    if theta_max <= theta_to_anti:
        return v_curvature
    cross = veh_fwd_x * (-ego_fwd_z) - veh_fwd_z * (-ego_fwd_x)
    if cross * v_curvature >= 0.0:
        return v_curvature
    return v_curvature / cal.turn_complete_curvature_scale


def _build_vehicle_collision_data(
    v: Vehicle,
    dynamic_horizon: float,
    ego_yaw_rad: float,
    ego_fwd_x: float,
    ego_fwd_z: float,
    cal: AEBCalibration,
    blender: VehicleCurvatureBlender | None = None,
    now: float | None = None,
) -> tuple[list[ArcPath], float, list[list[ArcPath]],
           float, float, float, float, float]:
    v_hw = v.size.width / 2.0
    v_hw_coll = max(v_hw - 0.1, 0.3)
    abs_v_speed = abs(v.speed)
    v_curvature = _vehicle_curvature_blend(v, abs_v_speed, cal, blender, now)
    v_yaw_rad = v._smooth_yaw if v._smooth_yaw is not None else math.radians(v.rotation.euler()[1])
    veh_fwd_x = -math.sin(v_yaw_rad)
    veh_fwd_z = -math.cos(v_yaw_rad)
    fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
    head_on = fwd_dot < -0.5
    target_override_decel = cal.full_brake_decel if head_on else 0.0
    arc_curvature = _dampen_turning_curvature(
        v_curvature, fwd_dot,
        ego_fwd_x, ego_fwd_z, veh_fwd_x, veh_fwd_z,
        abs_v_speed, abs_v_speed * dynamic_horizon,
        cal,
    )
    target_decel, target_accel = _accel_to_arc_params(v.accel_for_arc(), target_override_decel)
    veh_arc_coll = v.get_arc(
        dynamic_horizon,
        half_width=v_hw_coll,
        decel=target_override_decel,
        arc_start_pctg=cal.arc_start_pctg,
        curvature_override=arc_curvature,
    )
    tr_hw_colls: list[float] = []
    trailer_arcs_coll: list[ArcPath] = []
    for tr in v.trailers:
        tr_hw = tr.size.width / 2.0
        tr_hw_colls.append(max(tr_hw - 0.1, 0.3))
        tr_pos = tr.position
        _, tr_yaw_deg, _ = tr.rotation.euler()
        tr_yaw_rad = math.radians(tr_yaw_deg)
        tr_is_rev_c = v.speed < -1e-3
        tr_effective_p_c = (1.0 - cal.arc_start_pctg) if tr_is_rev_c else cal.arc_start_pctg
        tr_fwd_x_c = -math.sin(tr_yaw_rad)
        tr_fwd_z_c = -math.cos(tr_yaw_rad)
        tr_body_offset_c = (tr_effective_p_c - 0.5) * tr.size.length
        trailer_arcs_coll.append(
            build_arc(
                tr_pos.x + tr_body_offset_c * tr_fwd_x_c,
                tr_pos.z + tr_body_offset_c * tr_fwd_z_c,
                tr_yaw_rad,
                v.speed,
                arc_curvature,
                tr_hw_colls[-1],
                dynamic_horizon,
                decel=target_decel,
                accel=target_accel,
            )
        )
    all_target_arcs = [veh_arc_coll] + trailer_arcs_coll
    cross_padding = _cross_zone_padding(ego_yaw_rad, v_yaw_rad, abs_v_speed, cal)
    cross_arcs_list = [
        _apply_cross_zone(bt, cross_padding) for bt in all_target_arcs
    ]
    return (all_target_arcs, cross_padding, cross_arcs_list,
            v_yaw_rad, abs_v_speed, veh_fwd_x, veh_fwd_z, v_curvature)


# ---- FilterContext ----

@dataclass
class FilterContext:
    v: Vehicle
    ego_arc: ArcPath
    ego_braked_arc: ArcPath | None
    ego_evasion_left: ArcPath | None
    ego_evasion_right: ArcPath | None
    ego_x: float
    ego_y: float
    ego_z: float
    ego_yaw_rad: float
    ego_speed: float
    ego_pitch_rad: float
    ego_curvature: float
    ego_fwd_x: float
    ego_fwd_z: float
    ego_hw: float
    dynamic_horizon: float
    stopping_buffer: float
    tmp_traffic_session: bool
    ref_kmh_for_filter: float
    cal: AEBCalibration

    # Fields populated lazily by pipeline stages
    dx: float = 0.0
    dz: float = 0.0
    dist_sq: float = 0.0
    dist: float = 0.0
    v_yaw_rad: float = 0.0
    abs_v_speed: float = 0.0
    veh_fwd_x: float = 0.0
    veh_fwd_z: float = 0.0
    v_curvature: float = 0.0
    fwd_dot: float = 0.0
    head_on: bool = False
    near_head_on: bool = False
    co_directional: bool = False
    cross_padding: float = 0.0
    all_target_arcs: list = field(default_factory=list)
    precomputed_cross_arcs: list | None = None
    lane: Lane = Lane.EGO

    # Set of vehicle ids latched by a prior engagement on this target. Used
    # to bypass the TMP rel-speed pre-filter so a target does not vanish
    # from the pipeline as ego brakes and closing speed collapses.
    latched_threat_ids: set = field(default_factory=set)

    # Populated during collision evaluation (set by the pipeline caller)
    unbraked_hit: tuple | None = None
    lateral_gap: float = 0.0


# ---- Filter stages ----

class RangeFilter:
    name = "RangeFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._max_range_sq = cal.max_range ** 2

    def apply(self, ctx: FilterContext) -> FilterResult:
        vx, vz = ctx.v.position.x, ctx.v.position.z
        ctx.dx = vx - ctx.ego_x
        ctx.dz = vz - ctx.ego_z
        ctx.dist_sq = ctx.dx * ctx.dx + ctx.dz * ctx.dz
        if ctx.dist_sq > self._max_range_sq:
            return _suppress("RangeFilter")
        ctx.dist = math.sqrt(ctx.dist_sq)
        return _PASS


class ElevationFilter:
    name = "ElevationFilter"

    def __init__(self, cal: AEBCalibration | None = None) -> None:
        pass

    def apply(self, ctx: FilterContext) -> FilterResult:
        rz = _world_to_ego_forward(ctx.dx, ctx.dz, ctx.ego_yaw_rad)
        expected_y = ctx.ego_y + rz * math.tan(ctx.ego_pitch_rad)
        if abs(ctx.v.position.y - expected_y) > ctx.cal.elevation_margin:
            return _suppress("ElevationFilter")
        return _PASS


def _vehicle_yaw_rad(v: "Vehicle") -> float:
    if v._smooth_yaw is not None:
        return v._smooth_yaw
    _, yaw_deg, _ = v.rotation.euler()
    return math.radians(yaw_deg)


class TmpRelSpeedFilter:
    name = "TmpRelSpeedFilter"

    def apply(self, ctx: FilterContext) -> FilterResult:
        if not ctx.tmp_traffic_session:
            return _PASS
        # Latched targets bypass the rel-speed gate so they don't drop out
        # of the pipeline once ego matches their speed under braking.
        if ctx.v.id in ctx.latched_threat_ids:
            return _PASS
        v_yaw_rad = _vehicle_yaw_rad(ctx.v)
        vf_x = -math.sin(v_yaw_rad)
        vf_z = -math.cos(v_yaw_rad)
        dvx = ctx.ego_speed * ctx.ego_fwd_x - ctx.v.speed * vf_x
        dvz = ctx.ego_speed * ctx.ego_fwd_z - ctx.v.speed * vf_z
        rel_kmh = 3.6 * math.hypot(dvx, dvz)
        cal = ctx.cal
        if ctx.ref_kmh_for_filter > cal.tmp_filter_split_kmh:
            if rel_kmh <= cal.tmp_filter_rel_above_kmh:
                return _suppress("TmpRelSpeedFilter")
        else:
            if rel_kmh <= cal.tmp_filter_rel_below_kmh:
                return _suppress("TmpRelSpeedFilter")
        return _PASS


class LaneClassifier:
    """Sets ctx.lane and populates arc geometry fields; not a suppression stage."""
    name = "LaneClassifier"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        cal = self._cal
        # Geometry fields (v_yaw_rad, veh_fwd_x/z, abs_v_speed, v_curvature)
        # are populated upstream by _build_vehicle_collision_data so the
        # per-vehicle One-Euro blender steps exactly once per frame.  Do not
        # recompute curvature here — that would double-step the filter.
        ctx.fwd_dot = ctx.ego_fwd_x * ctx.veh_fwd_x + ctx.ego_fwd_z * ctx.veh_fwd_z
        ctx.head_on = ctx.fwd_dot < cal.head_on_dot
        ctx.near_head_on = ctx.fwd_dot < cal.near_head_on_dot
        ctx.co_directional = ctx.fwd_dot > cal.co_directional_dot

        # Lane classification via arc projection
        _, d_abs = project_to_ego_arc(ctx.ego_arc, ctx.v.position.x, ctx.v.position.z)
        ctx.lane = classify(d_abs, cal)

        ctx.lateral_gap = cal.lane_separation if ctx.near_head_on else 0.0
        return _PASS


class OppositeLaneFilter:
    """Suppress oncoming vehicles that are in their own lane (not ego's).

    Collapses Fix A + Fix B + oncoming evasion filter + same_curve heuristic.
    Uses lane_frame Lane classification instead of cross-product lateral_offset.
    """
    name = "OppositeLaneFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        if not ctx.head_on or ctx.abs_v_speed <= 1.0:
            return _PASS
        cal = self._cal

        # Determine own_lane using arc-projected lane instead of cross-product.
        # OPPOSITE_OR_OUTER = vehicle is in its own lane (or outer), not in ego's lane.
        own_lane = ctx.lane in (Lane.OPPOSITE_OR_OUTER, Lane.OFF_ROAD)

        if own_lane:
            # Bodies already physically separated: direct suppress (no evasion arc needed).
            v_hw_coll = max(ctx.v.size.width / 2.0 - 0.1, 0.3)
            _, d_abs = project_to_ego_arc(ctx.ego_arc, ctx.v.position.x, ctx.v.position.z)
            if d_abs >= ctx.ego_hw + v_hw_coll:
                return _suppress("OppositeLaneFilter")

        for arc_idx, base_target_arc in enumerate(ctx.all_target_arcs):
            # Determine effective cross padding (Fix A equivalent)
            if (ctx.near_head_on and own_lane):
                effective_padding = ctx.cross_padding * cal.near_head_on_cross_scale
                cross_arcs = _apply_cross_zone(base_target_arc, effective_padding)
            else:
                cross_arcs = (ctx.precomputed_cross_arcs[arc_idx]
                              if ctx.precomputed_cross_arcs else
                              _apply_cross_zone(base_target_arc, ctx.cross_padding))

            unbraked_hit = _earliest_hit(
                ctx.ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                ctx.lateral_gap,
            )
            if unbraked_hit is None:
                continue

            # Oncoming evasion filter logic (Fix B equivalent)
            delta_kappa_t = min(
                cal.evasion_g_oncoming / (ctx.abs_v_speed * ctx.abs_v_speed),
                cal.evasion_max_dkappa,
            )
            if own_lane:
                delta_kappa_t = min(
                    delta_kappa_t * cal.opposite_lane_kappa_scale,
                    cal.evasion_max_dkappa * cal.opposite_lane_kappa_scale,
                )
            # Fix B: road-following expansion
            if own_lane and abs(ctx.ego_curvature) >= cal.turning_diverge_kappa:
                delta_kappa_t = max(
                    delta_kappa_t,
                    min(abs(ctx.ego_curvature), cal.shared_turn_max_kappa),
                )
            evasion_decel = 0.0 if own_lane else base_target_arc.decel
            tgt_left = build_arc(
                base_target_arc.start_x, base_target_arc.start_z,
                base_target_arc.yaw_rad, ctx.v.speed,
                base_target_arc.curvature + delta_kappa_t,
                base_target_arc.half_width, base_target_arc.horizon,
                decel=evasion_decel,
            )
            tgt_right = build_arc(
                base_target_arc.start_x, base_target_arc.start_z,
                base_target_arc.yaw_rad, ctx.v.speed,
                base_target_arc.curvature - delta_kappa_t,
                base_target_arc.half_width, base_target_arc.horizon,
                decel=evasion_decel,
            )
            left_clears = arc_arc_collision(
                ctx.ego_arc, tgt_left, cal.corridor_margin, cal.collision_samples,
            ) is None
            right_clears = arc_arc_collision(
                ctx.ego_arc, tgt_right, cal.corridor_margin, cal.collision_samples,
            ) is None
            if left_clears or right_clears:
                return _suppress("OppositeLaneFilter")
        return _PASS


class OppositeLaneFilterMirrored:
    """Mirror of OppositeLaneFilter Fix B: ego mid-corner, target straight-approaching.

    When ego is in a bend and an oncoming target has low measured curvature
    (still on the straight approach to the same curve from the other side),
    the target's predicted arc chords across ego's curved corridor in world
    frame. ``OppositeLaneFilter`` Fix B expands target evasion arcs by ego's
    curvature only when the target is already in OPPOSITE_OR_OUTER/OFF_ROAD —
    but a straight-approaching target projects onto the ego arc as Lane.EGO,
    so that gate fails and nothing suppresses.

    This stage handles that case: build target evasion arcs offset by
    ``min(|ego_curvature|, shared_turn_max_kappa)`` (the implied road
    curvature the target will follow). If either side clears, suppress.
    """
    name = "OppositeLaneFilterMirrored"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        if not ctx.head_on or ctx.abs_v_speed <= 1.0:
            return _PASS
        cal = self._cal
        if abs(ctx.ego_curvature) < cal.turning_diverge_kappa:
            return _PASS
        if abs(ctx.v_curvature) >= cal.turning_diverge_kappa:
            return _PASS

        delta_kappa_t = min(abs(ctx.ego_curvature), cal.shared_turn_max_kappa)

        for arc_idx, base_target_arc in enumerate(ctx.all_target_arcs):
            cross_arcs = (ctx.precomputed_cross_arcs[arc_idx]
                          if ctx.precomputed_cross_arcs else
                          _apply_cross_zone(base_target_arc, ctx.cross_padding))
            unbraked_hit = _earliest_hit(
                ctx.ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                ctx.lateral_gap,
            )
            if unbraked_hit is None:
                continue

            tgt_left = build_arc(
                base_target_arc.start_x, base_target_arc.start_z,
                base_target_arc.yaw_rad, ctx.v.speed,
                base_target_arc.curvature + delta_kappa_t,
                base_target_arc.half_width, base_target_arc.horizon,
                decel=0.0,
            )
            tgt_right = build_arc(
                base_target_arc.start_x, base_target_arc.start_z,
                base_target_arc.yaw_rad, ctx.v.speed,
                base_target_arc.curvature - delta_kappa_t,
                base_target_arc.half_width, base_target_arc.horizon,
                decel=0.0,
            )
            left_clears = arc_arc_collision(
                ctx.ego_arc, tgt_left, cal.corridor_margin, cal.collision_samples,
            ) is None
            right_clears = arc_arc_collision(
                ctx.ego_arc, tgt_right, cal.corridor_margin, cal.collision_samples,
            ) is None
            if left_clears or right_clears:
                return _suppress("OppositeLaneFilterMirrored")
        return _PASS


class CoDirectionalDivergeFilter:
    """Suppress co-directional vehicles already diverging from ego (Fix C)."""
    name = "CoDirectionalDivergeFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        if not ctx.co_directional:
            return _PASS
        cal = self._cal
        for arc_idx, base_target_arc in enumerate(ctx.all_target_arcs):
            if base_target_arc.speed <= 0.5:
                continue
            cross_arcs = (ctx.precomputed_cross_arcs[arc_idx]
                          if ctx.precomputed_cross_arcs else
                          _apply_cross_zone(base_target_arc, ctx.cross_padding))
            unbraked_hit = _earliest_hit(
                ctx.ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                ctx.lateral_gap,
            )
            if unbraked_hit is None:
                continue

            # Fix C: outer-lane same-turn extended lookahead
            co_diverge_dt = cal.co_dir_diverge_lookahead_s
            g_lat = ctx.lane in (Lane.ADJACENT, Lane.OPPOSITE_OR_OUTER)
            g_ego_k = abs(ctx.ego_curvature) >= cal.turning_diverge_kappa
            g_veh_k = abs(ctx.v_curvature) >= cal.turning_diverge_kappa
            g_sign = ctx.ego_curvature * ctx.v_curvature > 0
            if g_lat and g_ego_k and g_veh_k and g_sign:
                co_diverge_dt = ctx.dynamic_horizon * cal.co_same_turn_lookahead_scale
            if not _is_approaching(ctx.ego_arc, base_target_arc,
                                   unbraked_hit[0], dt=co_diverge_dt):
                return _suppress("CoDirectionalDivergeFilter")
        return _PASS


class TurningCrossTrafficFilter:
    """Suppress cross-traffic whose arc is diverging at the hit point (Fix D absorbed)."""
    name = "TurningCrossTrafficFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        if ctx.head_on or ctx.co_directional:
            return _PASS
        cal = self._cal
        for arc_idx, base_target_arc in enumerate(ctx.all_target_arcs):
            if base_target_arc.speed <= 0.5:
                continue
            cross_arcs = (ctx.precomputed_cross_arcs[arc_idx]
                          if ctx.precomputed_cross_arcs else
                          _apply_cross_zone(base_target_arc, ctx.cross_padding))
            unbraked_hit = _earliest_hit(
                ctx.ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                ctx.lateral_gap,
            )
            if unbraked_hit is None:
                continue
            g_veh_k = abs(base_target_arc.curvature) > cal.turning_diverge_kappa
            if g_veh_k and not _is_approaching(ctx.ego_arc, base_target_arc, unbraked_hit[0]):
                return _suppress("TurningCrossTrafficFilter")
        return _PASS


class TmpCrossTrafficFilter:
    """Suppress TMP vehicles whose projected arc terminates outside ego's lane.

    TMP (multiplayer) vehicle data has higher uncertainty than AI vehicles —
    network jitter, position smoothing, and inconsistent yaw/curvature
    snapshots produce phantom arc-projection collisions during routine
    intersection maneuvers (e.g. a TMP vehicle making a side-road right turn
    appears to cut through ego's lane in the per-frame snapshot).

    A TMP vehicle whose extrapolated arc lands laterally outside ego's lane
    (OPPOSITE_OR_OUTER or OFF_ROAD) is mid-maneuver and will be clear of
    ego's path by the time ego arrives. Genuine threats — head-on or
    co-directional targets continuing into ego's lane — keep their projected
    arc inside Lane.EGO and pass through this filter to the standard
    pipeline. Co-directional in-lane vehicles are skipped here so that
    legitimate same-lane following / overtake handling stays with the
    dedicated stages.

    Non-TMP targets bypass entirely — AI vehicles follow deterministic
    traffic rules, so their snapshot-projected arc is reliable.
    """
    name = "TmpCrossTrafficFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        if not ctx.v.is_tmp:
            return _PASS
        if ctx.co_directional:
            return _PASS
        if ctx.abs_v_speed < 1.0:
            return _PASS
        cal = self._cal
        any_hit = False
        for arc_idx, base_target_arc in enumerate(ctx.all_target_arcs):
            cross_arcs = (ctx.precomputed_cross_arcs[arc_idx]
                          if ctx.precomputed_cross_arcs else
                          _apply_cross_zone(base_target_arc, ctx.cross_padding))
            ghost_hit = _earliest_hit(
                ctx.ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                ctx.lateral_gap,
            )
            if ghost_hit is None:
                continue
            any_hit = True
            # Use a non-braking arc to project the full-horizon end position.
            # The standard base arc may be truncated by target-side brake
            # modeling for near-head-on targets, masking where the cross-traffic
            # actually sweeps to.
            sweep_arc = build_arc(
                base_target_arc.start_x, base_target_arc.start_z,
                base_target_arc.yaw_rad, base_target_arc.speed,
                base_target_arc.curvature, base_target_arc.half_width,
                base_target_arc.horizon, decel=0.0,
            )
            end_x, end_z = sweep_arc.position_at_time(sweep_arc.horizon)
            _, end_d_abs = project_to_ego_arc(ctx.ego_arc, end_x, end_z)
            if classify(end_d_abs, cal) == Lane.EGO:
                # Arc ends inside ego's lane — real threat, do not suppress.
                return _PASS
        if any_hit:
            return _suppress("TmpCrossTrafficFilter")
        return _PASS


class SweepPassFilter:
    """Suppress stationary cross-traffic ego turns through."""
    name = "SweepPassFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        cal = self._cal
        if ctx.abs_v_speed >= cal.sweep_pass_max_target_speed:
            return _PASS
        if abs(ctx.ego_curvature) <= cal.turning_diverge_kappa:
            return _PASS

        vx, vz = ctx.v.position.x, ctx.v.position.z
        for arc_idx, base_target_arc in enumerate(ctx.all_target_arcs):
            cross_arcs = (ctx.precomputed_cross_arcs[arc_idx]
                          if ctx.precomputed_cross_arcs else
                          _apply_cross_zone(base_target_arc, ctx.cross_padding))
            unbraked_hit = _earliest_hit(
                ctx.ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                ctx.lateral_gap,
            )
            if unbraked_hit is None:
                continue
            sp_dist = ctx.ego_arc._dist_at_time(unbraked_hit[0])
            sp_ex, sp_ez = ctx.ego_arc.position_at_dist(sp_dist)
            sp_yaw = ctx.ego_arc.heading_at_dist(sp_dist)
            sp_fwd_x = -math.sin(sp_yaw)
            sp_fwd_z = -math.cos(sp_yaw)
            if (vx - sp_ex) * sp_fwd_x + (vz - sp_ez) * sp_fwd_z <= 0.0:
                return _suppress("SweepPassFilter")
        return _PASS


class CornerEntryStationaryFilter:
    """Suppress stationary vehicles whose pose implies a curved road continuation at corner entry."""
    name = "CornerEntryStationaryFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        cal = self._cal
        if ctx.abs_v_speed >= cal.sweep_pass_max_target_speed:
            return _PASS
        if abs(ctx.ego_curvature) >= cal.turning_diverge_kappa:
            return _PASS
        if ctx.dist <= cal.corner_entry_min_distance:
            return _PASS

        # Symmetric: abs(fwd_dot) folds oncoming and co-directional into [0, π/2].
        road_bend = math.acos(max(0.0, min(1.0, abs(ctx.fwd_dot))))
        if road_bend < cal.corner_entry_min_road_bend:
            return _PASS

        implied_kappa = road_bend / ctx.dist
        if implied_kappa <= cal.turning_diverge_kappa:
            return _PASS

        if ctx.lane != Lane.EGO:
            return _suppress("CornerEntryStationaryFilter")

        # In-lane: require geometric consistency with a curved road continuation.
        lat_signed = -ctx.dx * ctx.ego_fwd_z + ctx.dz * ctx.ego_fwd_x
        if abs(lat_signed) < cal.corner_entry_min_lateral:
            return _PASS

        cross = ctx.ego_fwd_x * ctx.veh_fwd_z - ctx.ego_fwd_z * ctx.veh_fwd_x
        if ctx.fwd_dot < 0.0:
            cross = -cross
        if cross * lat_signed <= 0.0:
            return _PASS

        expected_lat = ctx.dist * math.sin(0.5 * road_bend)
        if abs(expected_lat - abs(lat_signed)) > cal.corner_entry_lateral_tol:
            return _PASS

        return _suppress("CornerEntryStationaryFilter")


class CornerEntryStationaryFilterMirrored:
    """Mirror of CornerEntryStationaryFilter: ego mid-corner, target at the entry from the other side.

    Original fires when ego is straight (entering a corner) and a stationary
    target's pose implies a curved continuation. This stage handles the
    inverse: ego is already in the bend, and a stationary target sits at the
    entry of the same curve from the opposite approach. Their pose's road
    implication is the same curved continuation; they aren't blocking ego's
    straight-line path.

    Mode A (target out-of-lane via arc-projected classification) only. Mode B
    in-lane chord-offset geometry doesn't mirror cleanly — when ego is the
    one on the curve, the target's lateral offset in ego frame collapses
    toward zero and the ``|lat_signed| >= corner_entry_min_lateral`` precondition
    cannot be satisfied.
    """
    name = "CornerEntryStationaryFilterMirrored"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        cal = self._cal
        if ctx.abs_v_speed >= cal.sweep_pass_max_target_speed:
            return _PASS
        if abs(ctx.ego_curvature) < cal.turning_diverge_kappa:
            return _PASS
        if ctx.dist <= cal.corner_entry_min_distance:
            return _PASS

        road_bend = math.acos(max(0.0, min(1.0, abs(ctx.fwd_dot))))
        if road_bend < cal.corner_entry_min_road_bend:
            return _PASS

        implied_kappa = road_bend / ctx.dist
        if implied_kappa <= cal.turning_diverge_kappa:
            return _PASS

        if ctx.lane != Lane.EGO:
            return _suppress("CornerEntryStationaryFilterMirrored")
        return _PASS


class EgoEvasionFilter:
    """Suppress vehicles ego could steer around within 0.1 g (non-head-on, non-co-dir moving)."""
    name = "EgoEvasionFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        if ctx.head_on:
            return _PASS
        if (ctx.co_directional
                and any(a.speed > 0.5 for a in ctx.all_target_arcs)
                and ctx.lane == Lane.EGO):
            return _PASS
        if ctx.ego_evasion_left is None or ctx.ego_evasion_right is None:
            return _PASS
        cal = self._cal

        for arc_idx, base_target_arc in enumerate(ctx.all_target_arcs):
            # Use Fix A effective padding for near-head-on own-lane vehicles
            if ctx.near_head_on and ctx.lane in (Lane.OPPOSITE_OR_OUTER, Lane.OFF_ROAD):
                effective_padding = ctx.cross_padding * cal.near_head_on_cross_scale
                cross_arcs = _apply_cross_zone(base_target_arc, effective_padding)
            else:
                cross_arcs = (ctx.precomputed_cross_arcs[arc_idx]
                              if ctx.precomputed_cross_arcs else
                              _apply_cross_zone(base_target_arc, ctx.cross_padding))

            unbraked_hit = _earliest_hit(
                ctx.ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                ctx.lateral_gap,
            )
            if unbraked_hit is None:
                continue

            left_hit = _earliest_hit(
                ctx.ego_evasion_left, cross_arcs, 0.0, cal.collision_samples,
            )
            right_hit = _earliest_hit(
                ctx.ego_evasion_right, cross_arcs, 0.0, cal.collision_samples,
            )
            if left_hit is None or right_hit is None:
                return _suppress("EgoEvasionFilter")
        return _PASS


def build_pipeline(cal: AEBCalibration) -> list:
    """Return the ordered list of filter stage instances."""
    return [
        RangeFilter(cal),
        ElevationFilter(cal),
        TmpRelSpeedFilter(),
        LaneClassifier(cal),
        OppositeLaneFilter(cal),
        OppositeLaneFilterMirrored(cal),
        CoDirectionalDivergeFilter(cal),
        TurningCrossTrafficFilter(cal),
        TmpCrossTrafficFilter(cal),
        SweepPassFilter(cal),
        CornerEntryStationaryFilter(cal),
        CornerEntryStationaryFilterMirrored(cal),
        EgoEvasionFilter(cal),
    ]
