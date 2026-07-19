"""Named AEB filter pipeline: one class per suppression stage."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.radar.traffic import (
    ArcPath, Vehicle,
    build_arc, arc_arc_collision, _accel_to_arc_params,
    capsule_extents, pair_body_dist_sq,
)
from core.radar.ego_path import ego_curvature_from_history
from core.aeb.calibration import AEBCalibration
from core.aeb.lane_frame import Lane, project_to_ego_arc, classify


class OneEuroFilter:
    """Speed-adaptive low-pass: Casiez et al., "1€ Filter", CHI 2012.

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

    AEB-local two-source path prediction: smooth (position fit on the last
    ``cal.aeb_pos_history_len`` history samples) blended with responsive
    (single-frame yaw rate from ``angular_velocity``).  Either side fills in
    when the other is unavailable.

    When ``blender`` is supplied, the blended value is fed through a
    per-vehicle One-Euro filter (see :class:`VehicleCurvatureBlender`).  The
    raw blend is returned when ``blender`` is ``None``: used by test paths
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
    """Return [arc]. Legacy ghost-arc comb subsumed by ArcPath capsule body
    extents (see traffic.py::_sampled_collision). Kept as a pass-through so the
    padding plumbing retires without editing every call site."""
    return [arc]


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


def _is_approaching(a: ArcPath, b: ArcPath, t: float, dt: float = 0.1,
                    dip_samples: int = 1, dip_active: bool = False) -> bool:
    """True when the two paths still converge across the window [t, t+dt].

    Endpoint comparison alone misreads a fast pass-through: t is the corridor
    contact time, so above ~2*(contact distance)/dt of closing speed the t+dt
    sample lies beyond the target and center distance grows again, which reads
    as "diverging" exactly when the collision is most certain. With
    dip_active, any window sample dipping below body contact (sum of half
    widths, no corridor margin) is a predicted body overlap and counts as
    approaching, regardless of the endpoint comparison.

    Callers gate dip_active on the target being in Lane.EGO: for an in-lane
    target a predicted body overlap is a real rear-end/crossing course, while
    for an out-of-lane target the same overlap is usually a constant-curvature
    extrapolation artifact (e.g. overtaking a slower outer-lane vehicle in a
    shared turn: both arcs cross even though the real vehicles hold their
    lanes), which is exactly what this filter exists to suppress.

    Distances are body-to-body (capsule segment distance) when the arcs carry
    body extents, matching the collision test: measuring reference-point
    distance instead reads a passing/overtaking body as still "approaching"
    because the capsule contact time is when the near bodies touch, long before
    the reference points reach closest approach.
    """
    d0_sq = pair_body_dist_sq(a, b, t)
    contact_sq = (a.half_width + b.half_width) ** 2
    di_sq = d0_sq
    for i in range(1, dip_samples + 1):
        ti = t + dt * i / dip_samples
        di_sq = pair_body_dist_sq(a, b, ti)
        if dip_active and di_sq < contact_sq:
            return True
    return di_sq < d0_sq


# Fractions along the rigid body capsule (rear -> front) sampled by
# _any_body_in_ego_lane. Five points bracket a long trailer whose rear corner
# sits in ego's lane while its reference centre rides the outer lane.
_BODY_LANE_SAMPLES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _any_body_in_ego_lane(
    ego_arc: ArcPath, target_arcs: list[ArcPath], lane_half_width: float,
) -> bool:
    """True if any target body's centreline currently lies within ego's lane.

    Samples each target arc's rigid body capsule at t=0 (rear through front
    reference points) and projects to the ego arc. A long trailer swung into
    ego's lane registers via its rear sample even while the tractor reference
    rides the outer lane of a shared curve (crash clip 434f0401); an
    adjacent-lane overtake body, whose centreline stays beyond
    ``lane_half_width``, does not. The centreline (no body half-width) is used
    so a corridor-grazing outer-lane body is not miscounted as in-lane.
    """
    for arc in target_arcs:
        fx, fz = arc.fwd_x, arc.fwd_z
        back = -arc.back_len
        span = arc.fwd_len + arc.back_len
        for frac in _BODY_LANE_SAMPLES:
            s = back + span * frac
            px = arc.start_x + s * fx
            pz = arc.start_z + s * fz
            _, d_abs = project_to_ego_arc(ego_arc, px, pz)
            if d_abs <= lane_half_width:
                return True
    return False


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
        body_capsule=True,
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
        tr_half_l_c = tr.size.length * 0.5
        tr_cap_fwd_c, tr_cap_back_c = capsule_extents(
            tr_half_l_c, tr_half_l_c, tr_body_offset_c,
        )
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
                fwd_len=tr_cap_fwd_c,
                back_len=tr_cap_back_c,
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

    # Ids flagged by the follow-threat tracker (AEBThread): co-directional
    # in-lane targets with sustained closing range AND sustained own
    # deceleration. Bypass TmpRelSpeedFilter and are exempt from
    # EgoEvasionFilter / CoDirectionalDivergeFilter suppression: single-frame
    # jitter must not eject a lead that behavior says is genuinely braking
    # in ego's path (crash clips ddc0cdf7 / 0fe85c88, 2026-07-10).
    follow_threat_ids: set = field(default_factory=set)

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
        # Follow-threat targets bypass too: sustained closing + sustained own
        # deceleration is realistic follow-traffic behavior, which the
        # rel-speed floor (a no-collision-zone guard) must not silence.
        if ctx.v.id in ctx.follow_threat_ids:
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
        # recompute curvature here: that would double-step the filter.
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
                fwd_len=base_target_arc.fwd_len, back_len=base_target_arc.back_len,
            )
            tgt_right = build_arc(
                base_target_arc.start_x, base_target_arc.start_z,
                base_target_arc.yaw_rad, ctx.v.speed,
                base_target_arc.curvature - delta_kappa_t,
                base_target_arc.half_width, base_target_arc.horizon,
                decel=evasion_decel,
                fwd_len=base_target_arc.fwd_len, back_len=base_target_arc.back_len,
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
                fwd_len=base_target_arc.fwd_len, back_len=base_target_arc.back_len,
            )
            tgt_right = build_arc(
                base_target_arc.start_x, base_target_arc.start_z,
                base_target_arc.yaw_rad, ctx.v.speed,
                base_target_arc.curvature - delta_kappa_t,
                base_target_arc.half_width, base_target_arc.horizon,
                decel=0.0,
                fwd_len=base_target_arc.fwd_len, back_len=base_target_arc.back_len,
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
        # Follow-threat targets are exempt: the diverge check reads arc
        # extrapolation, and a jitter frame can call a genuinely braking
        # in-lane lead "diverging" moments before contact.
        if ctx.v.id in ctx.follow_threat_ids:
            return _PASS
        cal = self._cal
        # A rig with any body (tractor or trailer) physically in ego's lane now
        # is a real rear-end course, not a constant-curvature extrapolation
        # artifact. The lane primitive keys off the tractor reference point only,
        # so a long trailer swung into ego's lane while its tractor rides the
        # outer lane of a shared curve reads as EGO-lane-clear and the outer-lane
        # same-turn extended lookahead extrapolates the whole rig away as
        # "diverging" (crash clip 434f0401). Drop the extended lookahead and arm
        # the in-lane dip rescue for such a target.
        in_lane_body = _any_body_in_ego_lane(
            ctx.ego_arc, ctx.all_target_arcs, cal.lane_half_width)
        # Suppress only when every colliding body of the rig is diverging: a
        # tractor pulling into the outer lane genuinely diverges, but if its
        # trailer is still closing in ego's lane the rig is a real rear-end
        # course. Vetoing on the first diverging arc (the cab, evaluated first)
        # would drop the approaching trailer with it (crash clip 434f0401).
        any_hit = False
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
            any_hit = True

            # Fix C: outer-lane same-turn extended lookahead
            co_diverge_dt = cal.co_dir_diverge_lookahead_s
            g_lat = ctx.lane in (Lane.ADJACENT, Lane.OPPOSITE_OR_OUTER)
            g_ego_k = abs(ctx.ego_curvature) >= cal.turning_diverge_kappa
            g_veh_k = abs(ctx.v_curvature) >= cal.turning_diverge_kappa
            g_sign = ctx.ego_curvature * ctx.v_curvature > 0
            if g_lat and g_ego_k and g_veh_k and g_sign and not in_lane_body:
                co_diverge_dt = ctx.dynamic_horizon * cal.co_same_turn_lookahead_scale
            if _is_approaching(ctx.ego_arc, base_target_arc,
                               unbraked_hit[0], dt=co_diverge_dt,
                               dip_samples=cal.diverge_dip_samples,
                               dip_active=ctx.lane == Lane.EGO or in_lane_body):
                return _PASS
        if any_hit:
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
            if g_veh_k and not _is_approaching(ctx.ego_arc, base_target_arc, unbraked_hit[0],
                                               dip_samples=cal.diverge_dip_samples,
                                               dip_active=ctx.lane == Lane.EGO):
                return _suppress("TurningCrossTrafficFilter")
        return _PASS


class TmpCrossTrafficFilter:
    """Suppress TMP cross-traffic whose jittered snapshot arc only grazes ego's lane.

    TMP (multiplayer) vehicle data has higher uncertainty than AI vehicles:
    network jitter, position smoothing, and inconsistent yaw/curvature
    snapshots project a target's arc through ego's lane during routine
    intersection maneuvers even though the actual MP target sweeps past. The
    canonical phantom is mid-turn (e.g. a TMP vehicle making a side-road right
    turn appears to cut through ego's lane in the per-frame snapshot), but a
    straight snapshot can also produce a body-graze hit: a long vehicle crossing
    ahead clips ego's corridor with its 10 m tail while its centre passes metres
    clear. The old filter answered "where does the target END UP at the 3 s
    horizon" (suppress unless the endpoint lands in Lane.EGO), which suppressed
    the mid-turn phantom AND the straight grazer, but also suppressed a genuine
    straight perpendicular crosser on a dead-center collision course at every
    range (its endpoint always lands tens of metres past ego's lane: corpus FN
    clip ffd29f9e). The question is "does the target occupy ego's lane WHEN ego
    arrives", not "where does it end up".

    Split by trust in the snapshot's motion (`ctx.v_curvature`):

    - Straight snapshot (`|v_curvature| < turning_diverge_kappa`): the motion is
      trustworthy, so decide on the geometry directly. Take the closest approach
      of the two reference centres over the horizon (ego arc vs the non-braked
      sweep arc). If they genuinely meet
      (`d_min <= cal.tmp_cross_center_hit_dist`) it is a real T-bone: pass
      (brake). If the centres miss (the hit is only the long body grazing as it
      clears): suppress.
    - Turning snapshot (`|v_curvature| >= turning_diverge_kappa`): the jittered
      curvature makes the predicted centre path unreliable, so keep the
      full-horizon endpoint-lane test. If the sweep arc ends in OPPOSITE_OR_OUTER
      / OFF_ROAD the target sweeps clear: suppress. If it ends in Lane.EGO it is
      a real continuing threat: pass.

    Applies to TMP vehicles (`v.is_tmp=True`) that are not co-directional and
    have non-trivial speed. Co-directional in-lane vehicles are skipped so that
    legitimate same-lane following / overtake handling stays with the dedicated
    stages.

    Uses a freshly-built non-braking arc from the undamped `ctx.v_curvature`
    (rather than `base_target_arc`) for both branches: the standard arc may be
    truncated by target-side full-brake modeling at near-head-on angles, and its
    Fix D over-rotation damping straightens the arc of a target genuinely
    sweeping through a corner. Either one distorts the projected centre path and
    endpoint and masks where the cross-traffic actually sweeps to.

    Non-TMP targets bypass entirely: AI vehicles follow deterministic traffic
    rules, so their snapshot-projected arc is reliable.
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
        straight = abs(ctx.v_curvature) < cal.turning_diverge_kappa
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
            sweep_arc = build_arc(
                base_target_arc.start_x, base_target_arc.start_z,
                base_target_arc.yaw_rad, base_target_arc.speed,
                ctx.v_curvature, base_target_arc.half_width,
                base_target_arc.horizon, decel=0.0,
            )
            if straight:
                # Trustworthy straight snapshot: a genuine collision brings the
                # two reference centres together; a body-only graze (long tail
                # sweeping clear) keeps them apart. Pass the real T-bone, let
                # the grazer fall through to suppression below.
                if self._center_min_dist(ctx.ego_arc, sweep_arc) <= cal.tmp_cross_center_hit_dist:
                    return _PASS
            else:
                # Jitter-prone turning snapshot: the endpoint-lane heuristic
                # decides whether the swept arc clears ego's lane.
                end_x, end_z = sweep_arc.position_at_time(sweep_arc.horizon)
                _, end_d_abs = project_to_ego_arc(ctx.ego_arc, end_x, end_z)
                if classify(end_d_abs, cal) == Lane.EGO:
                    return _PASS
        if any_hit:
            return _suppress("TmpCrossTrafficFilter")
        return _PASS

    def _center_min_dist(self, ego_arc: ArcPath, target_arc: ArcPath) -> float:
        """Closest approach of the two reference centres over the shared horizon."""
        horizon = min(ego_arc.horizon, target_arc.horizon)
        n = self._cal.collision_samples
        best = float("inf")
        for k in range(n + 1):
            t = horizon * k / n
            ex, ez = ego_arc.position_at_time(t)
            tx, tz = target_arc.position_at_time(t)
            d = math.hypot(ex - tx, ez - tz)
            if d < best:
                best = d
        return best


class OutOfLaneParallelFilter:
    """Suppress co-directional / stationary traffic that stays in its own lane.

    The ArcPath capsule collision body registers a grazing corridor overlap for
    a long vehicle driving or parked alongside ego, because the two long bodies
    are within the corridor sum (body half-widths + margin) laterally over the
    whole overlap. The point model only saw a fleeting reference-point crossing
    at closest approach, which the diverge / evasion stages suppressed on timing
    that the capsule contact time no longer matches.

    A target whose center never enters ego's lane over the horizon
    (arc-projected offset stays above ``lane_half_width``) is lane-keeping
    adjacent traffic or a roadside object ego passes, not a collision course.
    A genuine cut-in or crossing brings its center into ``Lane.EGO`` and is not
    suppressed here; head-on own-lane oncoming is handled by
    ``OppositeLaneFilter``; follow / latched threats are exempt.
    """
    name = "OutOfLaneParallelFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: FilterContext) -> FilterResult:
        if ctx.head_on:
            return _PASS
        cal = self._cal
        stationary = ctx.abs_v_speed < cal.sweep_pass_max_target_speed
        if not (ctx.co_directional or stationary):
            return _PASS
        if ctx.v.id in ctx.follow_threat_ids or ctx.v.id in ctx.latched_threat_ids:
            return _PASS
        if ctx.lane == Lane.EGO:
            return _PASS
        # A trailer swung into ego's lane while its tractor reference rides the
        # outer lane keeps the tractor-based lane out of EGO and its trailer
        # centre-trajectory scan out of lane, yet its rear body sits in ego's
        # path (crash clip 434f0401). The body-length in-lane test catches it.
        if _any_body_in_ego_lane(ctx.ego_arc, ctx.all_target_arcs, cal.lane_half_width):
            return _PASS
        # Rear overtaker: faster co-dir traffic from behind in another lane.
        # Capsule + ego-arc wiggle manufactures corridor hits on these; braking
        # cannot help (see braking_worsens). Suppress before the predicted-
        # center scan, which leaks on the same bent ego arc that creates the
        # phantom (passing FP clips f0b2ace6 / 02642609).
        if (ctx.co_directional and not stationary
                and (ctx.v.speed * ctx.fwd_dot) > ctx.ego_speed
                and ctx.dx * ctx.ego_fwd_x + ctx.dz * ctx.ego_fwd_z < 0.0):
            return _suppress("OutOfLaneParallelFilter")
        # Predicted center must stay out of ego's lane across the horizon.
        n = max(1, cal.out_of_lane_scan_samples)
        for base_target_arc in ctx.all_target_arcs:
            horizon = base_target_arc.horizon
            for k in range(n + 1):
                t = horizon * k / n
                px, pz = base_target_arc.position_at_time(t)
                _, d_abs = project_to_ego_arc(ctx.ego_arc, px, pz)
                if d_abs <= cal.lane_half_width:
                    return _PASS
        return _suppress("OutOfLaneParallelFilter")


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
    in-lane chord-offset geometry doesn't mirror cleanly: when ego is the
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
        cal = self._cal
        # In-lane co-directional moving target: a rear-end lead, never a
        # "driver steers around it" call. The tractor-based lane misses a rig
        # whose trailer is swung into ego's lane while the cab rides the outer
        # lane (crash clip 434f0401), so also bypass on the body-length test.
        if (ctx.co_directional
                and any(a.speed > 0.5 for a in ctx.all_target_arcs)
                and (ctx.lane == Lane.EGO
                     or _any_body_in_ego_lane(
                         ctx.ego_arc, ctx.all_target_arcs, cal.lane_half_width))):
            return _PASS
        # Follow-threat targets are exempt: the in-lane bypass above can lose
        # them to a single jittered lane/heading frame, and "ego could steer
        # around it" is never a safe call against a lead that behavior says
        # is braking in ego's path.
        if ctx.v.id in ctx.follow_threat_ids:
            return _PASS
        if ctx.ego_evasion_left is None or ctx.ego_evasion_right is None:
            return _PASS

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
        OutOfLaneParallelFilter(cal),
        TmpCrossTrafficFilter(cal),
        SweepPassFilter(cal),
        CornerEntryStationaryFilter(cal),
        CornerEntryStationaryFilterMirrored(cal),
        EgoEvasionFilter(cal),
    ]

