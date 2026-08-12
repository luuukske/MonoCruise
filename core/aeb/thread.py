"""AEB thread: arc collision, filter pipeline, continuous decel. Registry: aeb_thread. See core/aeb/README.md."""

from __future__ import annotations

import copy
import enum
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from core.radar.traffic import (
    Vehicle,
    ArcPath, build_arc, arc_arc_collision, _accel_to_arc_params,
    capsule_extents, pair_body_dist_sq,
)
from core.aeb.calibration import AEBCalibration, DEFAULT as _CAL_DEFAULT
from core.aeb.confirm import OccupancyConfirm
from core.aeb.lane_frame import project_to_ego_arc, classify, Lane
from core.aeb.capture import get_recorder, note_intervention
from core.aeb.clip_schema import AEBTickRecord, AEBWarmState, ConsumedContext, LiveAEB
from core.aeb.filters import (
    FilterContext, FilterResult,
    _build_vehicle_collision_data, _world_to_ego_forward, _cross_zone_padding,
    _apply_cross_zone, _earliest_hit, _is_approaching, _dampen_turning_curvature,
    _any_body_in_ego_lane, _vehicle_curvature_blend, VehicleCurvatureBlender,
    build_pipeline, oncoming_closing_into,
)

logger = logging.getLogger(__name__)

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

_AEB_SOUND_PATH = str(Path(__file__).resolve().parent / "AEB_warning.wav")
# Extra seamless-loop plays after stop_warning() (avoids a single short blip).
_AEB_WARNING_STOP_EXTRA_REPLAYS = 1

# Constants

_INF: float = 1e9
_GRAVITY_MS2: float = 9.81

# Brake-capacity floor used when sending_thread has not yet published an
# estimate. Slope-corrected per tick before use; never read as a flat ceiling.
_FULL_BRAKE_DECEL_FALLBACK: float = 7.8
# TMP-only: |v_ego − v_target| (km/h) vs latched ref ego speed: see _latched_filter_ego_kmh.
_TMP_FILTER_EGO_SPLIT_KMH: float = 40.0
_TMP_FILTER_REL_ABOVE_SPLIT_KMH: float = 15.0
_TMP_FILTER_REL_AT_OR_BELOW_SPLIT_KMH: float = 40.0
# Low on purpose: a light dab still carries braking force and shows the driver
# saw the hazard. AEB engagement is the emergency override if it is not enough.
_USER_BRAKE_LATCH_THRESHOLD: float = 0.03

_STOP_BUFFER_FIXED: float = 1.6
_ARC_START_PCTG: float = 0.2
_RISK_CONFIRM_DURATION: float = 0.05
_RISK_CONFIRM_DURATION_ONCOMING: float = _RISK_CONFIRM_DURATION * 2.0

_REAR_DOT_THRESHOLD: float = -0.5
_OVERTAKE_SPEED_MARGIN: float = 2.0

_ELEVATION_MARGIN_M: float = 5.0

_CROSS_SAFE_ZONE_BASE: float = 2.0
_CROSS_SAFE_ZONE_SPEED: float = 0.3

_EVASION_G_THRESHOLD: float = 0.08 * 9.81
_LATERAL_LANE_SEPARATION: float = 3.9
# fwd_dot for lateral-gap activation; looser than head_on (-0.7). See README head-on lateral-gap.
_NEAR_HEAD_ON_DOT: float = -0.5
_TURNING_DIVERGE_CURVATURE: float = 0.007
_EVASION_G_THRESHOLD_ONCOMING: float = 0.13 * 9.81
_EVASION_FILTER_MAX_DELTA_KAPPA: float = 0.008
# Oncoming own-lane lateral offset scales evasion delta_kappa (cross product lat).
_OPPOSITE_LANE_OFFSET: float = 2.0
_OPPOSITE_LANE_KAPPA_SCALE: float = 2.0
# Same-curve shared road compresses cross-product lat; lower own-lane threshold then.
_SAME_CURVE_OWN_LANE_LAT: float = 1.0
_CO_DIR_DIVERGE_LOOKAHEAD_S: float = 0.25
# Co-dir same-turn extended lookahead fraction (README CoDirectionalDivergeFilter).
_CO_SAME_TURN_LOOKAHEAD_SCALE: float = 0.5
# Sweep-pass suppression: stationary cross-traffic ego turns through.
_SWEEP_PASS_MAX_TARGET_SPEED: float = 1.0    # m/s

# Legacy module-level mirrors of cal fields (loop uses AEBCalibration); kept for local refs.
_NEAR_HEAD_ON_CROSS_SCALE: float = 0.3
_NEAR_HEAD_ON_LATERAL_MIN: float = 3.0
_SHARED_TURN_MAX_KAPPA: float = 0.05
_TURN_COMPLETE_CURVATURE_SCALE: float = 3.0

_TRAILER_TRACTOR_RADIUS_M: float = 30.0
_TRAILER_SWAP_SPEED_THRESHOLD_MS: float = 0.5
_TRAILER_TRACTOR_HEADING_DOT: float = 0.9

# Debug shadow_near TN sampler; not in AEBCalibration (recorder rate-limits).
_SHADOW_MIN_SPEED_MS: float = 2.0
_SHADOW_MAX_RANGE_M: float = 80.0
_SHADOW_TN_FILTER_REASONS: frozenset[str] = frozenset({
    "OppositeLaneFilter",
    "OppositeLaneFilterMirrored",
    "EgoEvasionFilter",
    "CornerEntryStationaryFilter",
    "CornerEntryStationaryFilterMirrored",
    "CoDirectionalDivergeFilter",
    "TurningCrossTrafficFilter",
    "TmpCrossTrafficFilter",
    "SweepPassFilter",
})

# Debug auto_crash capture (main_pedal speed-drop mirror); not in cal.
_CRASH_MIN_SPEED_KMH: float = 40.0
_CRASH_SPEED_DROP_KMH: float = 5.0

# Sustained-brake popup: fires once per continuous AEB_brake span, after it
# has held this long, so a single-frame flicker never pops up a notice, and
# only once ego_speed shows the brake actually stopped the truck.
_BRAKE_POPUP_MIN_DURATION_S: float = 0.5
_BRAKE_POPUP_STOPPED_SPEED_MS: float = 0.3


def _find_tractor_for_trailer(trailer: Vehicle, vehicles: list[Vehicle]) -> Vehicle | None:
    """Nearest same-heading non-trailer within radius; TMP and convoy AI slots."""
    _, tr_yaw_deg, _ = trailer.rotation.euler()
    tr_yaw = math.radians(tr_yaw_deg)
    tr_fx = -math.sin(tr_yaw)
    tr_fz = -math.cos(tr_yaw)

    best: Vehicle | None = None
    best_d_sq = _TRAILER_TRACTOR_RADIUS_M * _TRAILER_TRACTOR_RADIUS_M
    for other in vehicles:
        if other.id == trailer.id:
            continue
        if other.is_trailer:
            continue
        dx = other.position.x - trailer.position.x
        dz = other.position.z - trailer.position.z
        d_sq = dx * dx + dz * dz
        if d_sq >= best_d_sq:
            continue
        _, o_yaw_deg, _ = other.rotation.euler()
        o_yaw = math.radians(o_yaw_deg)
        o_fx = -math.sin(o_yaw)
        o_fz = -math.cos(o_yaw)
        if tr_fx * o_fx + tr_fz * o_fz < _TRAILER_TRACTOR_HEADING_DOT:
            continue
        best_d_sq = d_sq
        best = other
    return best


def _swap_trailer_kinematics(vehicles: list[Vehicle]) -> list[Vehicle]:
    """Shallow-copy trailer kinematics from nearest tractor; see README TMP trailer swap."""
    out: list[Vehicle] = []
    for v in vehicles:
        if v.is_trailer and abs(v.speed) < _TRAILER_SWAP_SPEED_THRESHOLD_MS:
            tractor = _find_tractor_for_trailer(v, vehicles)
            if tractor is not None:
                eff = copy.copy(v)
                eff.speed = tractor.speed
                eff.acceleration = tractor.acceleration
                eff._debug_kinematics_swapped = True
                out.append(eff)
                continue
        out.append(v)
    return out


def _tmp_collision_threat(ref_ego_kmh: float, rel_speed_kmh: float) -> bool:
    """TMP session only: True if target should participate in arc collision / TTB."""
    if ref_ego_kmh > _TMP_FILTER_EGO_SPLIT_KMH:
        return rel_speed_kmh > _TMP_FILTER_REL_ABOVE_SPLIT_KMH
    return rel_speed_kmh > _TMP_FILTER_REL_AT_OR_BELOW_SPLIT_KMH


class AEBState(enum.IntEnum):
    STANDBY = 0
    WARN = 1
    BRAKE = 2


@dataclass
class AEBSnapshot:
    ego_x: float = 0.0
    ego_z: float = 0.0
    ego_yaw: float = 0.0
    ego_speed: float = 0.0
    ego_half_w: float = 1.265
    ego_half_l: float = 3.333
    ego_arc: ArcPath | None = None
    ego_braked_arc: ArcPath | None = None
    ego_has_trailer: bool = False

    vehicles: list = field(default_factory=list)
    vehicle_arcs: dict = field(default_factory=dict)
    colliding_ids: set = field(default_factory=set)
    suppressed_ids: set = field(default_factory=set)
    braking_worsens_ids: set = field(default_factory=set)
    evasion_filtered_ids: set = field(default_factory=set)
    oncoming_evasion_filtered_ids: set = field(default_factory=set)
    # Targets whose measured LOS drift vetoed engagement entry this tick
    # (debug/eval visibility; they still warn and still show as colliding).
    los_vetoed_ids: set = field(default_factory=set)
    # Superset: LOS veto plus the extrapolation vetoes (README engagement vetoes).
    engage_vetoed_ids: set = field(default_factory=set)

    aeb_state: AEBState = AEBState.STANDBY
    time_to_collision: float = _INF
    time_to_brake: float = _INF
    hit_x: float = 0.0
    hit_z: float = 0.0

    evasion_left_arc: ArcPath | None = None
    evasion_right_arc: ArcPath | None = None

    suppression_reasons: dict = field(default_factory=dict)
    tmp_traffic_session: bool = False


def _vehicle_in_ego_trajectory(
    snap: AEBSnapshot, x: float, z: float, v_hw: float,
) -> bool:
    """True when (x, z) lies ahead inside ego's predicted corridor."""
    arc = snap.ego_arc
    if arc is None:
        return False
    s, d_abs = project_to_ego_arc(arc, x, z)
    if s < 0.0 or s > arc.arc_length:
        return False
    corridor = arc.half_width + v_hw + _CAL_DEFAULT.corridor_margin
    return d_abs <= corridor


def _should_sample_shadow_tn(snap: AEBSnapshot) -> bool:
    """True when a nearby vehicle was rejected by a spatial AEB filter stage."""
    filtered_ids = (
        snap.suppressed_ids
        | snap.evasion_filtered_ids
        | snap.oncoming_evasion_filtered_ids
    )
    if not filtered_ids:
        return False

    veh_by_id = {v["vid"]: v for v in snap.vehicles if "vid" in v}
    max_r_sq = _SHADOW_MAX_RANGE_M * _SHADOW_MAX_RANGE_M

    for vid in filtered_ids:
        reasons = snap.suppression_reasons.get(vid, [])
        if not any(r.reason in _SHADOW_TN_FILTER_REASONS for r in reasons):
            continue
        veh = veh_by_id.get(vid)
        if veh is None:
            continue
        x = float(veh["x"])
        z = float(veh["z"])
        v_hw = float(veh.get("half_w", 0.0))
        dx = x - snap.ego_x
        dz = z - snap.ego_z
        if dx * dx + dz * dz > max_r_sq:
            continue
        if _vehicle_in_ego_trajectory(snap, x, z, v_hw):
            return True
    return False


@dataclass
class AEBData(ThreadData):
    AEB_warn: bool = False
    AEB_brake: bool = False
    time_to_brake: float = _INF
    em_stop_requested: bool = False
    AEB_target_decel_ms2: float = 0.0
    AEB_ff_decel_ms2: float = 0.0
    AEB_required_decel_ms2: float = 0.0
    AEB_effective_max_decel_ms2: float = 0.0
    AEB_realized_decel_ms2: float = 0.0
    snapshot: AEBSnapshot = field(default_factory=AEBSnapshot)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


def _cross_zone_padding(
    ego_yaw_rad: float,
    v_yaw_rad: float,
    v_speed_ms: float,
    cal: AEBCalibration = _CAL_DEFAULT,
) -> float:
    """Perpendicular-target ghost-arc padding (peaks at 90° yaw diff)."""
    cross_factor = abs(math.sin(ego_yaw_rad - v_yaw_rad))
    return cross_factor * (cal.cross_zone_base + cal.cross_zone_speed * v_speed_ms)


def _apply_cross_zone(arc: ArcPath, padding: float) -> list[ArcPath]:
    """Ghost-arc comb retired; capsule extents cover body (pass-through for call sites)."""
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
    """Ego-space forward component: rz > 0 = in front of ego."""
    return dx * math.sin(ego_yaw_rad) + dz * math.cos(ego_yaw_rad)


def _response_dist(v: float, cal, response_s: float | None = None) -> float:
    """Uncapped v * response_s; rejected cap/tiering in README §7."""
    if response_s is None:
        response_s = cal.stop_buffer_response_s
    return abs(v) * response_s


def _response_s_for_load(cal, has_trailer: bool) -> float:
    """Brake build-up pad for the current load class (trailer air brakes are slower)."""
    return (
        cal.stop_buffer_response_trailer_s if has_trailer
        else cal.stop_buffer_response_s
    )


def _codir_required_cap(
    target_arc: ArcPath,
    fwd_dot: float,
    unbraked_ttc: float,
    ego_travel: float,
    v_closing: float,
    ego_speed: float,
    cal,
    response_s: float | None = None,
) -> float:
    """Co-directional required decel cap (moving vs stopped lead); min with relative frame."""
    if fwd_dot < cal.co_directional_dot:
        return _INF
    abs_ego = abs(ego_speed)
    gap_est = max(ego_travel - target_arc._dist_at_time(unbraked_ttc), 0.0)
    a_t = target_arc.decel if target_arc.decel > 1e-3 else 0.0
    d_move = max(
        gap_est - cal.stop_buffer - _response_dist(v_closing, cal, response_s),
        1e-3,
    )
    r_move = a_t + (v_closing * v_closing) / (2.0 * d_move)
    r_stop = 0.0
    if a_t > 0.0 and target_arc.speed > 1e-3:
        s_stop = (target_arc.speed * target_arc.speed) / (2.0 * a_t)
        d_stop = max(
            gap_est + s_stop - cal.stop_buffer - _response_dist(abs_ego, cal, response_s),
            1e-3,
        )
        r_stop = (abs_ego * abs_ego) / (2.0 * d_stop)
    return max(r_move, r_stop)


def _required_decel_two_frame(
    closing_distance: float,
    v_closing: float,
    ego_travel: float,
    ego_speed: float,
    cal,
    codir_cap: float = _INF,
    response_s: float | None = None,
) -> float:
    """Required decel: relative frame or ego fallback; optional co-dir cap (README continuous-decel)."""
    d_rel = (
        closing_distance - cal.stop_buffer - _response_dist(v_closing, cal, response_s)
    )
    abs_ego = abs(ego_speed)
    if d_rel > 1e-3:
        required = (v_closing * v_closing) / (2.0 * d_rel)
    else:
        d_ego = max(
            ego_travel - cal.stop_buffer - _response_dist(abs_ego, cal, response_s),
            1e-3,
        )
        required = (abs_ego * abs_ego) / (2.0 * d_ego)
    return min(required, codir_cap)


def _is_approaching(a: ArcPath, b: ArcPath, t: float, dt: float = 0.1) -> bool:
    ax0, az0 = a.position_at_time(t)
    bx0, bz0 = b.position_at_time(t)
    ax1, az1 = a.position_at_time(t + dt)
    bx1, bz1 = b.position_at_time(t + dt)
    d0_sq = (ax0 - bx0) ** 2 + (az0 - bz0) ** 2
    d1_sq = (ax1 - bx1) ** 2 + (az1 - bz1) ** 2
    return d1_sq < d0_sq


def _los_predicted_miss(
    track, now_mono: float, cal: AEBCalibration,
) -> float | None:
    """CBDR miss distance from LOS track; None fails open (README LOS veto)."""
    cutoff = now_mono - cal.los_veto_window_s
    ts: list[float] = []
    rng: list[float] = []
    brg: list[float] = []
    for t_s, vx, vz, ex, ez in track:
        if t_s < cutoff:
            continue
        dx = vx - ex
        dz = vz - ez
        ts.append(t_s)
        rng.append(math.hypot(dx, dz))
        b = math.atan2(dz, dx)
        if brg:
            prev = brg[-1]
            while b - prev > math.pi:
                b -= 2.0 * math.pi
            while b - prev < -math.pi:
                b += 2.0 * math.pi
        brg.append(b)
    if len(ts) < cal.los_veto_min_samples:
        return None
    t_mean = sum(ts) / len(ts)
    denom = sum((t_s - t_mean) ** 2 for t_s in ts)
    if denom < 1e-9:
        return None
    r_mean = sum(rng) / len(rng)
    b_mean = sum(brg) / len(brg)
    r_dot = sum((t_s - t_mean) * (r - r_mean) for t_s, r in zip(ts, rng)) / denom
    omega = sum((t_s - t_mean) * (b - b_mean) for t_s, b in zip(ts, brg)) / denom
    r_now = rng[-1]
    v_rel = math.hypot(r_dot, r_now * omega)
    if v_rel < 0.5:
        return None
    return abs(omega) * r_now * r_now / v_rel


def _los_veto_bar(
    fwd_dot: float,
    abs_v_speed: float,
    v_curvature: float,
    cal: AEBCalibration,
) -> tuple[float, float]:
    """(min_range_m, miss_threshold_m) for the CBDR veto; see README LOS veto."""
    if fwd_dot > cal.head_on_dot:
        return cal.los_veto_min_range_m, cal.los_veto_miss_dist_m
    manoeuvring = (
        abs_v_speed >= cal.los_veto_headon_min_speed_ms
        and abs(v_curvature) >= cal.los_veto_headon_max_kappa
    )
    if manoeuvring:
        return cal.los_veto_min_range_m, cal.los_veto_miss_dist_m
    return cal.los_veto_headon_min_range_m, cal.los_veto_headon_miss_dist_m


def _extrapolation_veto(
    ctx,
    unbraked_ttc: float,
    v_target_along_ego: float,
    in_lane_body: bool,
    cal: AEBCalibration,
    d_miss: float | None = None,
) -> bool:
    """Engagement-entry veto for extrapolation-fragile hits (README engagement vetoes)."""
    if not cal.extrap_veto_enabled:
        return False
    if ctx.lane == Lane.EGO or in_lane_body:
        return False
    if ctx.co_directional:
        # Matched-speed neighbour: any contact is lateral, and brakes do not steer.
        # A faster target is an overtaker, a separate class owned by braking_worsens.
        axial = ctx.ego_speed - v_target_along_ego
        if not 0.0 <= axial < cal.codir_adjacent_veto_axial_ms:
            return False
        # Measurably converging tracks are a real side contact, not lane-keeping.
        return d_miss is not None and d_miss >= cal.codir_adjacent_veto_miss_m
    # Crossing/oncoming reached only by holding the current steer: real turns are
    # transient, so a far hit off a bent ego arc is extrapolation, not evidence.
    # Turn-into-path closing (shrinking miss + collapsed |lat|) is the exception.
    if oncoming_closing_into(ctx, cal):
        return False
    return (abs(ctx.ego_curvature) >= cal.turn_veto_min_kappa
            and unbraked_ttc >= cal.turn_veto_min_ttc_s)


def _dampen_turning_curvature(
    v_curvature: float,
    fwd_dot: float,
    ego_fwd_x: float, ego_fwd_z: float,
    veh_fwd_x: float, veh_fwd_z: float,
    abs_v_speed: float,
    arc_length: float,
    cal: AEBCalibration = _CAL_DEFAULT,
) -> float:
    """Fix D: dampen arc_curvature when constant-curvature would over-rotate past alignment."""
    if (abs(v_curvature) <= cal.turning_diverge_kappa
            or abs_v_speed <= 0.5
            or fwd_dot <= cal.near_head_on_dot
            or fwd_dot >= cal.co_directional_dot):
        return v_curvature
    theta_max = abs(v_curvature) * arc_length
    theta_to_anti = math.acos(max(-1.0, min(1.0, -fwd_dot)))
    if theta_max <= theta_to_anti:
        return v_curvature
    # Fix D direction guard: dampen only when rotating toward anti-parallel.
    cross = veh_fwd_x * (-ego_fwd_z) - veh_fwd_z * (-ego_fwd_x)
    if cross * v_curvature >= 0.0:
        return v_curvature  # rotating away: genuine cross-arc threat
    return v_curvature / cal.turn_complete_curvature_scale


def _ls_slope(samples, idx: int) -> float:
    """LS slope of samples[i][idx] vs time; robust to TMP jitter."""
    n = len(samples)
    if n < 2:
        return 0.0
    t_mean = sum(s[0] for s in samples) / n
    v_mean = sum(s[idx] for s in samples) / n
    num = 0.0
    den = 0.0
    for s in samples:
        dt = s[0] - t_mean
        num += dt * (s[idx] - v_mean)
        den += dt * dt
    return num / den if den > 1e-9 else 0.0


def _follow_threat_arc_decel(
    vehicle_id: int,
    follow_tracks: dict[int, deque],
    follow_threat_ids: set[int],
    cal: AEBCalibration,
) -> float | None:
    """Estimated lead braking decel for follow-threat collision arcs."""
    if vehicle_id not in follow_threat_ids:
        return None
    trk = follow_tracks.get(vehicle_id)
    if not trk or len(trk) < 2:
        return None
    own_decel = -_ls_slope(trk, 2)
    if own_decel < cal.follow_threat_min_decel_ms2:
        return None
    return own_decel


def _build_vehicle_collision_data(
    v: Vehicle,
    dynamic_horizon: float,
    ego_yaw_rad: float,
    ego_fwd_x: float,
    ego_fwd_z: float,
    cal: AEBCalibration = _CAL_DEFAULT,
    blender: VehicleCurvatureBlender | None = None,
    now: float | None = None,
    follow_decel_ms2: float | None = None,
) -> tuple[list[ArcPath], float, list[list[ArcPath]],
           float, float, float, float, float]:
    """Collision arcs + geometry tuple for one vehicle (filters harness + thread)."""
    v_hw = v.size.width / 2.0
    v_hw_coll = max(v_hw - 0.1, 0.3)
    abs_v_speed = abs(v.speed)
    v_curvature = _vehicle_curvature_blend(v, abs_v_speed, cal, blender, now)
    v_yaw_rad = v._smooth_yaw if v._smooth_yaw is not None else math.radians(v.rotation.euler()[1])
    veh_fwd_x = -math.sin(v_yaw_rad)
    veh_fwd_z = -math.cos(v_yaw_rad)
    fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
    head_on = fwd_dot < cal.near_head_on_dot
    target_override_decel = cal.full_brake_decel if head_on else 0.0
    arc_decel = target_override_decel
    if follow_decel_ms2 is not None and follow_decel_ms2 > 0.0:
        arc_decel = max(arc_decel, follow_decel_ms2)
    # Fix D on arc_curvature only; v_curvature unchanged for filters.
    arc_curvature = _dampen_turning_curvature(
        v_curvature, fwd_dot,
        ego_fwd_x, ego_fwd_z, veh_fwd_x, veh_fwd_z,
        abs_v_speed, abs_v_speed * dynamic_horizon,
        cal,
    )
    # For trailer arcs built with build_arc() directly. get_arc() calls
    # _accel_to_arc_params internally so veh_arc_coll only needs arc_decel.
    target_decel, target_accel = _accel_to_arc_params(v.accel_for_arc(), arc_decel)
    veh_arc_coll = v.get_arc(
        dynamic_horizon,
        half_width=v_hw_coll,
        decel=arc_decel,
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
        tr_effective_p_c = (
            (1.0 - cal.arc_start_pctg) if tr_is_rev_c else cal.arc_start_pctg
        )
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


class _SoundState(enum.IntEnum):
    STOPPED = 0
    RUNNING = 1
    SHUTTING_DOWN = 2


class _AEBSoundHandler:
    """Pygame AEB warning loop with non-blocking stop and extra replay tail."""

    def __init__(
        self,
        sound_file_path: str,
        stop_extra_replays: int = _AEB_WARNING_STOP_EXTRA_REPLAYS,
    ) -> None:
        self._sound = None
        self._state = _SoundState.STOPPED
        self._sound_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_extra_replays = max(0, int(stop_extra_replays))
        self._replays_remaining = 0

        if not _PYGAME_AVAILABLE:
            logger.warning("pygame not available: AEB sound disabled")
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=256)
                pygame.mixer.init()
            self._sound = pygame.mixer.Sound(sound_file_path)
            self._sound.set_volume(0.8)
        except Exception as exc:
            logger.error(
                "AEB sound init failed for %s (%s): sound disabled",
                sound_file_path, exc,
            )
            self._sound = None

    def start_warning(self) -> None:
        """Start loop; cancels in-flight shutdown and resumes."""
        if self._sound is None:
            return
        with self._lock:
            if self._state == _SoundState.RUNNING:
                return
            if self._state == _SoundState.SHUTTING_DOWN:
                self._state = _SoundState.RUNNING
                self._replays_remaining = 0
                logger.debug("AEB sound: shutdown cancelled, resuming loop")
                return
            self._state = _SoundState.RUNNING
            self._sound_thread = threading.Thread(
                target=self._sound_loop_manager, daemon=True
            )
            self._sound_thread.start()
            logger.debug("AEB sound: warning loop started")

    def stop_warning(self) -> None:
        """Schedule extra replays then finish current clip (non-blocking)."""
        if self._sound is None:
            return
        with self._lock:
            if self._state == _SoundState.RUNNING:
                self._state = _SoundState.SHUTTING_DOWN
                self._replays_remaining = self._stop_extra_replays
                logger.debug(
                    "AEB sound: stop requested: %d extra replay(s) then finishing",
                    self._replays_remaining,
                )

    def _sound_loop_manager(self) -> None:
        sound_length = self._sound.get_length()
        overlap_time = 0.15
        sleep_duration = max(0.0, sound_length - overlap_time)

        last_channel = self._sound.play()

        while True:
            time.sleep(sleep_duration)
            with self._lock:
                if self._state == _SoundState.RUNNING:
                    last_channel = self._sound.play()
                elif self._state == _SoundState.SHUTTING_DOWN:
                    if self._replays_remaining > 0:
                        self._replays_remaining -= 1
                        last_channel = self._sound.play()
                    else:
                        logger.debug("AEB sound: extra replays done: letting current sound finish")
                        break

        if last_channel:
            while last_channel.get_busy():
                time.sleep(0.01)

        with self._lock:
            self._state = _SoundState.STOPPED
        logger.debug("AEB sound: finished playing naturally, thread closing")

    def cleanup(self) -> None:
        """Block until all sound activity is finished, then quit the mixer."""
        self.stop_warning()
        if self._sound_thread and self._sound_thread.is_alive():
            self._sound_thread.join()
        if _PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.quit()
        logger.debug("AEB sound: cleanup complete")


class AEBThread(BaseThread):
    loop_interval = 1 / 30
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="aeb_thread")
        self.data = AEBData()
        self._prev_state: AEBState = AEBState.STANDBY
        self._state_hold_until: float = 0.0
        # Sustained AEB_brake span tracking, for the intervention popup.
        self._brake_span_start_mono: float | None = None
        self._brake_popup_fired: bool = False
        # Separate edge tracker for clip-capture triggers (debug only).
        self._capture_prev_state: AEBState = AEBState.STANDBY
        self._prev_ego_speed_capture_ms: float | None = None
        self._last_snapshot: AEBSnapshot | None = None
        # Per-target OccupancyConfirm for risk aggregates (README confirm windows).
        self._risk_confirm: dict[int, OccupancyConfirm] = {}
        self._radar_visualizer = None
        self._radar_vis_last_vehicle_time: float = -1.0
        self._latched_filter_ego_kmh: float | None = None
        self._sound_handler = _AEBSoundHandler(_AEB_SOUND_PATH)
        self._cal: AEBCalibration = _CAL_DEFAULT
        self._pipeline = build_pipeline(self._cal)
        # One-Euro per target kappa; stepped once per vehicle per frame (README).
        self._curvature_blender = VehicleCurvatureBlender(self._cal)
        # LOS tracks for engagement-entry CBDR veto (README LOS veto).
        self._los_tracks: dict[int, deque] = {}
        # Per-target (t, d_miss) for turn-into-path closing rate (README).
        self._d_miss_hist: dict[int, deque] = {}
        # Follow-threat tracks + hold; see README follow-threat section.
        self._follow_tracks: dict[int, deque] = {}
        self._follow_hold_until: dict[int, float] = {}
        self._follow_threat_ids: set[int] = set()
        # Continuous-decel state
        self._engaged: bool = False
        # Engage confirm: geometry-graded window_s each frame (README tiered entry).
        self._engage_confirm = OccupancyConfirm(
            self._cal.aeb_confirm_occupancy,
            self._cal.aeb_confirm_max_gap_frames,
            self._cal.aeb_engage_confirm_oblique_s,
        )
        # Oblique warn occupancy; shorter window than oblique engage (README).
        self._warn_confirm = OccupancyConfirm(
            self._cal.aeb_confirm_occupancy,
            self._cal.aeb_confirm_max_gap_frames,
            self._cal.aeb_warn_confirm_oblique_s,
        )
        # Longer warn occupancy for fully engage-vetoed out-of-lane sets (README).
        self._warn_vetoed_confirm = OccupancyConfirm(
            self._cal.aeb_confirm_occupancy,
            self._cal.aeb_confirm_max_gap_frames,
            self._cal.aeb_warn_confirm_vetoed_s,
        )
        # Evidence-class warn occupancy: all-oncoming and all-wide-lateral sets.
        self._warn_oncoming_confirm = OccupancyConfirm(
            self._cal.aeb_confirm_occupancy,
            self._cal.aeb_confirm_max_gap_frames,
            self._cal.aeb_warn_confirm_oncoming_s,
        )
        self._warn_wide_lat_confirm = OccupancyConfirm(
            self._cal.aeb_confirm_occupancy,
            self._cal.aeb_confirm_max_gap_frames,
            self._cal.aeb_warn_confirm_wide_lat_s,
        )
        # Raw-warn floor under the certain-geometry instant bypass.
        self._warn_instant_confirm = OccupancyConfirm(
            self._cal.aeb_confirm_occupancy,
            self._cal.aeb_confirm_max_gap_frames,
            self._cal.aeb_warn_instant_min_s,
        )
        # vid -> last mono time the target sat a full lane off the ego arc.
        self._warn_wide_seen: dict[int, float] = {}
        self._published_target_ms2: float = 0.0
        self._last_target_change_mono: float = 0.0
        self._prev_loop_mono: float | None = None
        # Latched threats: TMP rel-speed bypass + headway hold (README latched-threat).
        self._latched_threat_ids: set[int] = set()
        # Scope stamp per latched id; out-of-scope grace before release.
        self._latched_scope_ok_mono: dict[int, float] = {}
        # Rate-limit for gap/collision box debug (Settings.debug only).
        self._gap_debug_last_mono: float = 0.0
        self._gap_debug_was_engaged: bool = False
        # Clip replay overrides _now, readers, and _aeb_active_fn (clip_eval).
        self._now = time.monotonic
        self._aeb_active_fn = None

    def _read_user_braking(self) -> bool:
        """Suppress warn when user/CC already braking unless near-full AEB demand."""
        # Physical pedal only. opdbrakeval is deliberately excluded: it is below
        # brakeval whenever the pedal is pressed, and its coast-down floor would
        # read ordinary OPD coasting as a danger response.
        try:
            pt = registry.get_thread("main_pedal_thread")
            if pt is not None and pt.is_alive():
                with pt.data._lock:
                    driver_brake = float(getattr(pt.data, "brakeval", 0.0))
                if driver_brake > _USER_BRAKE_LATCH_THRESHOLD:
                    return True
        except (KeyError, AttributeError):
            pass

        # Program brake from CC/ACC/limiter. mapper_command_brake is AEB-free by
        # construction: AEB's FF and slam merge into abackward after the mapper.
        try:
            st = registry.get_thread("sending_thread")
            if st is None or not st.is_alive():
                return False
            with st.data._lock:
                return (
                    float(getattr(st.data, "mapper_command_brake", 0.0))
                    > _USER_BRAKE_LATCH_THRESHOLD
                )
        except (KeyError, AttributeError):
            return False

    def _update_follow_threats(
        self,
        vehicles_eff: list,
        ego_arc,
        ego_x: float,
        ego_z: float,
        ego_fwd_x: float,
        ego_fwd_z: float,
        now_mono: float,
        cal: AEBCalibration,
    ) -> None:
        """Follow-threat ids: kinematic hold + lane/converge gate (README follow-threat)."""
        active: set[int] = set()
        for v in vehicles_eff:
            dx = v.position.x - ego_x
            dz = v.position.z - ego_z
            dist = math.hypot(dx, dz)
            if dist > cal.follow_threat_max_range_m:
                continue
            _, d_abs = project_to_ego_arc(
                ego_arc, v.position.x, v.position.z,
            )
            trk = self._follow_tracks.get(v.id)
            if trk is None:
                trk = self._follow_tracks[v.id] = deque()
            trk.append((now_mono, dist, abs(v.speed), d_abs))
            cutoff = now_mono - cal.follow_threat_window_s
            while trk and trk[0][0] < cutoff:
                trk.popleft()
            if (len(trk) >= cal.follow_threat_min_samples
                    and trk[-1][0] - trk[0][0] >= cal.follow_threat_min_span_s):
                v_yaw = (
                    v._smooth_yaw
                    if v._smooth_yaw is not None
                    else math.radians(v.rotation.euler()[1])
                )
                fwd_dot = (ego_fwd_x * -math.sin(v_yaw)
                           + ego_fwd_z * -math.cos(v_yaw))
                if fwd_dot >= cal.co_directional_dot:
                    closing = -_ls_slope(trk, 1)
                    own_decel = -_ls_slope(trk, 2)
                    if (closing >= cal.follow_threat_min_closing_ms
                            and own_decel >= cal.follow_threat_min_decel_ms2):
                        self._follow_hold_until[v.id] = (
                            now_mono + cal.follow_threat_hold_s
                        )
            hold = self._follow_hold_until.get(v.id)
            if hold is not None and hold > now_mono:
                in_ego = classify(d_abs, cal) == Lane.EGO
                lat_converge = -_ls_slope(trk, 3)
                if (in_ego
                        or lat_converge >= cal.follow_threat_min_lat_converge_ms):
                    active.add(v.id)
        self._follow_threat_ids = active

    def _read_acc_lead_id(self) -> int | None:
        """ACC lead id for debug visualizer only; never raises."""
        try:
            acc = registry.get_thread("acc_thread")
            if acc is None or not acc.is_alive():
                return None
            with acc.data._lock:
                if not bool(getattr(acc.data, "has_lead", False)):
                    return None
                return int(getattr(acc.data, "lead_id", -1))
        except (KeyError, AttributeError):
            return None

    def _read_max_brake_ms2(self) -> float:
        """Max brake from sending_thread; fallback _FULL_BRAKE_DECEL if unavailable."""
        try:
            st = registry.get_thread("sending_thread")
            if st is not None and st.is_alive():
                with st.data._lock:
                    v = float(st.data.max_brake_ms2)
                if v > 1.0:
                    return v
        except (KeyError, AttributeError):
            pass
        return _FULL_BRAKE_DECEL_FALLBACK

    def _read_pedals_for_capture(self) -> tuple[float, float, float]:
        """(brakeval, gasval, opdbrakeval) for the consumed context. Never raises."""
        try:
            pt = registry.get_thread("main_pedal_thread")
            if pt is None or not pt.is_alive():
                return 0.0, 0.0, 0.0
            with pt.data._lock:
                return (
                    float(getattr(pt.data, "brakeval", 0.0)),
                    float(getattr(pt.data, "gasval", 0.0)),
                    float(getattr(pt.data, "opdbrakeval", 0.0)),
                )
        except (KeyError, AttributeError):
            return 0.0, 0.0, 0.0

    def _read_program_brake_for_capture(self) -> float:
        """Mapper brake command for the consumed context. Never raises."""
        try:
            st = registry.get_thread("sending_thread")
            if st is None or not st.is_alive():
                return 0.0
            with st.data._lock:
                return float(getattr(st.data, "mapper_command_brake", 0.0))
        except (KeyError, AttributeError):
            return 0.0

    def _build_warm_state(self) -> AEBWarmState:
        """Snapshot the discrete engagement state for the clip window start (plan 5)."""
        warn_hold = None
        brake_hold = None
        if self._prev_state == AEBState.BRAKE:
            brake_hold = self._state_hold_until
        elif self._prev_state == AEBState.WARN:
            warn_hold = self._state_hold_until
        return AEBWarmState(
            engaged=bool(self._engaged),
            latched_threat_ids=sorted(self._latched_threat_ids),
            latched_scope_ok_mono=dict(self._latched_scope_ok_mono),
            latched_filter_ego_kmh=self._latched_filter_ego_kmh,
            warn_hold_until_mono=warn_hold,
            brake_hold_until_mono=brake_hold,
            target_decel_ms2=float(self._published_target_ms2),
        )

    def _capture_aeb_tick(
        self, now_mono: float, radar_t_mono: float, aeb_active: bool,
        snap: "AEBSnapshot", max_brake_ms2: float, required_decel: float,
        effective_max_decel: float, target_published: float, ff_decel: float,
        aeb_warn: bool, aeb_brake: bool, new_state: "AEBState",
        tmp_traffic_session: bool,
    ) -> None:
        """Debug capture tick + engagement trigger; never raises into the loop."""
        recorder = get_recorder()
        if recorder is None:
            return
        try:
            # Nothing competes with the road during the event that caused the
            # clip, so upload notifications are held until it ends.
            note_intervention(bool(aeb_warn) or bool(aeb_brake) or bool(self._engaged))
            brakeval, gasval, opdbrakeval = self._read_pedals_for_capture()
            reasons = {
                str(vid): [r.reason for r in results if getattr(r, "reason", None)]
                for vid, results in snap.suppression_reasons.items()
                if results
            }
            tick = AEBTickRecord(
                t_mono=now_mono,
                radar_t_mono=radar_t_mono,
                consumed=ConsumedContext(
                    max_brake_ms2=float(max_brake_ms2),
                    brakeval=brakeval,
                    gasval=gasval,
                    opdbrakeval=opdbrakeval,
                    program_brake=self._read_program_brake_for_capture(),
                    aeb_enabled=bool(aeb_active),
                ),
                live_aeb=LiveAEB(
                    aeb_warn=bool(aeb_warn),
                    aeb_brake=bool(aeb_brake),
                    engaged=bool(self._engaged),
                    target_decel_ms2=float(target_published),
                    ff_decel_ms2=float(ff_decel),
                    required_decel_ms2=float(required_decel),
                    effective_max_decel_ms2=float(effective_max_decel),
                    time_to_brake=float(snap.time_to_brake),
                    time_to_collision=float(snap.time_to_collision),
                    colliding_ids=sorted(snap.colliding_ids),
                    suppressed_ids=sorted(snap.suppressed_ids),
                    braking_worsens_ids=sorted(snap.braking_worsens_ids),
                    suppression_reasons=reasons,
                ),
            )
            recorder.push_aeb_tick(tick, self._build_warm_state())

            # Auto trigger on WARN/BRAKE entry; a WARN to BRAKE escalation re-fires
            # so brake_reached is tagged (folds into the same clip window, plan 3.1).
            prev = self._capture_prev_state
            entered = new_state != AEBState.STANDBY and prev == AEBState.STANDBY
            escalated = new_state == AEBState.BRAKE and prev == AEBState.WARN
            if entered or escalated:
                recorder.trigger(
                    "auto_engagement",
                    session_kind="TMP" if tmp_traffic_session else "SP",
                    brake_reached=(new_state == AEBState.BRAKE),
                    calibration=self._cal,
                )

            # shadow_near TN sampler when filters reject a candidate (debug; recorder throttles).
            if (aeb_active
                    and new_state == AEBState.STANDBY
                    and not self._engaged
                    and not self._read_user_braking()
                    and snap.ego_speed > _SHADOW_MIN_SPEED_MS
                    and _should_sample_shadow_tn(snap)):
                recorder.trigger(
                    "shadow_near",
                    session_kind="TMP" if tmp_traffic_session else "SP",
                    calibration=self._cal,
                )

            # Crash capture: sudden ego speed drop while above the minimum
            # speed floor (debug clip corpus only; leaves labels untagged).
            prev_capture_ms = self._prev_ego_speed_capture_ms
            if (aeb_active
                    and prev_capture_ms is not None
                    and prev_capture_ms * 3.6 >= _CRASH_MIN_SPEED_KMH
                    and (prev_capture_ms - snap.ego_speed) * 3.6
                    >= _CRASH_SPEED_DROP_KMH):
                recorder.trigger(
                    "auto_crash",
                    session_kind="TMP" if tmp_traffic_session else "SP",
                    calibration=self._cal,
                )
            self._prev_ego_speed_capture_ms = snap.ego_speed

            self._capture_prev_state = new_state
        except Exception:
            logger.debug("AEB clip capture failed", exc_info=True)

    def setup(self) -> None:
        if Settings.debug:
            self._try_start_radar_visualizer()
        logger.debug("AEB setup complete")

    def _try_start_radar_visualizer(self) -> None:
        """Start the Flask/SocketIO radar visualizer (debug-only)."""
        if self._radar_visualizer is not None:
            return
        try:
            # Lazy import so missing optional deps (flask/socketio) don't kill AEB.
            from .radar_visualizer import RadarVisualizer  # type: ignore
        except Exception as exc:
            logger.warning("RadarVisualizer import failed: %s", exc)
            return

        try:
            visualizer = RadarVisualizer(port=5000)
        except Exception as exc:
            logger.warning("RadarVisualizer init failed: %s", exc)
            return

        self._radar_visualizer = visualizer

        def _run() -> None:
            try:
                visualizer.start()
            except Exception as exc:
                logger.warning("RadarVisualizer failed to start: %s", exc)

        threading.Thread(target=_run, daemon=True).start()
        logger.info("RadarVisualizer running on http://127.0.0.1:5000")

    def loop(self) -> None:
        if not self.running:
            return

        aeb_active = (
            self._aeb_active_fn() if self._aeb_active_fn is not None
            else Settings.AEB_enabled
        )
        cal = self._cal

        snapshot = self._read_radar_snapshot()
        if snapshot is None:
            return
        (vehicles, ego_x, ego_y, ego_z, ego_yaw_rad, ego_speed, ego_pitch_deg,
         steer, ego_has_trailer, _ego_curvature_from_history, tmp_traffic_session,
         paused, radar_t_mono) = snapshot

        vehicles_eff = _swap_trailer_kinematics(vehicles)

        # Brake build-up pad, keyed on load: a trailer's air brakes take about
        # three times longer to reach full decel than a solo tractor.
        load_response_s = _response_s_for_load(cal, ego_has_trailer)

        if paused and self._last_snapshot is not None:
            with self.data._lock:
                self.data.snapshot = self._last_snapshot
            return

        now_mono = self._now()

        # Yaw-rate proxy: see core/aeb/README.md §1. Do NOT use RadarData.ego_curvature.
        if ego_speed > 0.5:
            yaw_rate_rad_s = math.radians(steer * ego_speed * cal.yaw_rate_steer_gain)
            ego_curvature = yaw_rate_rad_s / ego_speed
        else:
            ego_curvature = 0.0

        ego_hw: float = cal.ego_half_width
        ego_half_l: float = cal.ego_half_length

        _max_brake_live = self._read_max_brake_ms2()
        effective_decel = cal.ego_decel_frac * _max_brake_live

        t_stop = ego_speed / effective_decel
        dynamic_horizon = min(max(cal.arc_horizon_min, t_stop * 2.0), cal.arc_horizon_max)

        _ego_fwd_x = -math.sin(ego_yaw_rad)
        _ego_fwd_z = -math.cos(ego_yaw_rad)
        _ego_body_offset = (cal.arc_start_pctg - 0.5) * (2.0 * ego_half_l)
        ego_front_x = ego_x + _ego_body_offset * _ego_fwd_x
        ego_front_z = ego_z + _ego_body_offset * _ego_fwd_z
        # Ego body centered on ego_x with half-length ego_half_l; the arc
        # reference sits _ego_body_offset behind center, so the capsule extents
        # are asymmetric about the reference (front reaches farther than rear).
        ego_cap_fwd, ego_cap_back = capsule_extents(
            ego_half_l, ego_half_l, _ego_body_offset,
        )

        ego_arc = build_arc(
            ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
            ego_curvature, ego_hw, dynamic_horizon,
            fwd_len=ego_cap_fwd, back_len=ego_cap_back,
            parallel_margin_scale=cal.capsule_parallel_margin_scale,
        )

        run_collision = aeb_active

        ego_braked_arc: ArcPath | None = None
        if run_collision:
            ego_braked_arc = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                ego_curvature, ego_hw, dynamic_horizon,
                decel=effective_decel,
                fwd_len=ego_cap_fwd, back_len=ego_cap_back,
                parallel_margin_scale=cal.capsule_parallel_margin_scale,
            )

        ego_evasion_left: ArcPath | None = None
        ego_evasion_right: ArcPath | None = None
        if run_collision and ego_speed > 1.0:
            delta_kappa = min(
                cal.evasion_g / (ego_speed * ego_speed),
                cal.evasion_max_dkappa,
            )
            left_kappa = ego_curvature + delta_kappa
            if ego_curvature < 0 and left_kappa < 0:
                left_kappa = left_kappa / 1.3
            right_kappa = ego_curvature - delta_kappa
            if ego_curvature > 0 and right_kappa > 0:
                right_kappa = right_kappa / 1.3
            ego_evasion_left = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                left_kappa, ego_hw, dynamic_horizon,
                fwd_len=ego_cap_fwd, back_len=ego_cap_back,
            )
            ego_evasion_right = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                right_kappa, ego_hw, dynamic_horizon,
                fwd_len=ego_cap_fwd, back_len=ego_cap_back,
            )

        ego_fwd_x = ego_arc.fwd_x
        ego_fwd_z = ego_arc.fwd_z

        ego_kmh_now = ego_speed * 3.6
        ref_kmh_for_filter = (
            self._latched_filter_ego_kmh
            if self._latched_filter_ego_kmh is not None
            else ego_kmh_now
        )
        if self._radar_visualizer is not None:
            lead_id = self._read_acc_lead_id()
            lead_v = None
            if lead_id is not None and lead_id >= 0:
                for v in vehicles:
                    if v.id == lead_id:
                        lead_v = v
                        break
            is_tracked = lead_v is not None
            if lead_v is None and vehicles:
                lead_v = min(
                    vehicles,
                    key=lambda v: (v.position.x - ego_x) ** 2 + (v.position.z - ego_z) ** 2,
                )
            if lead_v is not None:
                if lead_v.time != self._radar_vis_last_vehicle_time:
                    self._radar_vis_last_vehicle_time = lead_v.time
                    r_spd, c_spd, e_spd, a_spd, f_acc = lead_v.radar_speed_accel()
                    self._radar_visualizer.push_data(
                        r_spd, c_spd, e_spd, a_spd, f_acc, is_tracked=is_tracked,
                    )
            else:
                self._radar_vis_last_vehicle_time = -1.0
                self._radar_visualizer.clear()

        colliding_ids: set[int] = set()
        suppressed_ids: set[int] = set()
        braking_worsens_ids: set[int] = set()
        evasion_filtered_ids: set[int] = set()
        oncoming_evasion_filtered_ids: set[int] = set()
        suppression_reasons: dict[int, list[FilterResult]] = {}
        best_ttb: float = _INF
        best_unbraked_ttc: float = _INF
        best_hit_x: float = 0.0
        best_hit_z: float = 0.0
        best_closing_distance: float = _INF
        best_v_closing: float = 0.0
        best_ego_travel: float = _INF
        best_codir_cap: float = _INF
        best_threat_vid: int | None = None
        best_hit_dist: float = _INF
        # Engage aggregates exclude LOS-vetoed ids; warn/disarm use full set (README).
        best_ttb_engage: float = _INF
        best_closing_distance_engage: float = _INF
        best_v_closing_engage: float = 0.0
        best_ego_travel_engage: float = _INF
        best_codir_cap_engage: float = _INF
        # Ego travel to hit uses capsule tip offset from arc reference.
        ego_front_to_surface = ego_cap_fwd + ego_hw
        # certain_geom / nearcertain_geom drive tiered confirm (README tiered entry).
        certain_geom_ids: set[int] = set()
        nearcertain_geom_ids: set[int] = set()
        # Colliding ids whose pose sits in ego's lane band, for the warn gate.
        ego_lane_colliding_ids: set[int] = set()
        # Warn-gate evidence classes: oncoming, and a full lane off the ego arc.
        oncoming_colliding_ids: set[int] = set()
        wide_lat_colliding_ids: set[int] = set()
        wide_lat_checked_ids: set[int] = set()
        nearest_colliding_range: float = _INF
        los_vetoed_ids: set[int] = set()
        # Superset of los_vetoed_ids: every target barred from engagement entry.
        engage_vetoed_ids: set[int] = set()
        los_veto_memo: dict[int, bool] = {}
        los_miss_memo: dict[int, float | None] = {}

        def los_miss(veh: Vehicle) -> float | None:
            """Measured CBDR miss for one vehicle, computed once per frame."""
            if veh.id not in los_miss_memo:
                los_miss_memo[veh.id] = _los_predicted_miss(
                    self._los_tracks.get(veh.id, ()), now_mono, cal,
                )
            return los_miss_memo[veh.id]

        def los_miss_rate(veh: Vehicle) -> float | None:
            """d(d_miss)/dt over the LOS window; None until span >= 0.2 s."""
            dm = los_miss(veh)
            if dm is None:
                return None
            hist = self._d_miss_hist.get(veh.id)
            if hist is None:
                hist = self._d_miss_hist[veh.id] = deque()
            hist.append((now_mono, dm))
            cutoff = now_mono - cal.los_veto_window_s
            while hist and hist[0][0] < cutoff:
                hist.popleft()
            if len(hist) < 2:
                return None
            t0, m0 = hist[0]
            t1, m1 = hist[-1]
            dt = t1 - t0
            if dt < 0.2:
                return None
            return (m1 - m0) / dt
        vehicle_dicts: list[dict] = []
        vehicle_arcs: dict[int, list[ArcPath]] = {}
        newly_risky: set[int] = set()

        ego_pitch_rad = math.radians(ego_pitch_deg)

        # Follow-threat flags must be current before the precompute prefilter
        # below reads them: a braking lead can cross under the rel-speed floor
        # on the same frame it needs the bypass.
        if run_collision:
            self._update_follow_threats(
                vehicles_eff, ego_arc, ego_x, ego_z,
                ego_fwd_x, ego_fwd_z, now_mono, cal,
            )

        # Precompute per-vehicle collision arcs for vehicles passing range/elevation/TMP.
        # Use vehicles_eff so TMP trailer-as-vehicles inherit tractor speed/acceleration.
        vehicle_collision_data: dict[int, tuple] = {}
        max_range_sq = cal.max_range ** 2
        if run_collision:
            for v in vehicles_eff:
                vx, vz = v.position.x, v.position.z
                dx = vx - ego_x
                dz = vz - ego_z
                if dx * dx + dz * dz > max_range_sq:
                    continue
                rz = _world_to_ego_forward(dx, dz, ego_yaw_rad)
                expected_y = ego_y + rz * math.tan(ego_pitch_rad)
                if abs(v.position.y - expected_y) > cal.elevation_margin:
                    continue
                if (tmp_traffic_session
                        and v.id not in self._latched_threat_ids
                        and v.id not in self._follow_threat_ids):
                    _, v_yaw_deg_pc, _ = v.rotation.euler()
                    v_yaw_rad_pc = math.radians(v_yaw_deg_pc)
                    vf_x = -math.sin(v_yaw_rad_pc)
                    vf_z = -math.cos(v_yaw_rad_pc)
                    dvx_pc = ego_speed * ego_fwd_x - v.speed * vf_x
                    dvz_pc = ego_speed * ego_fwd_z - v.speed * vf_z
                    rel_kmh_pc = 3.6 * math.hypot(dvx_pc, dvz_pc)
                    if not _tmp_collision_threat(ref_kmh_for_filter, rel_kmh_pc):
                        continue
                follow_decel = _follow_threat_arc_decel(
                    v.id, self._follow_tracks, self._follow_threat_ids, cal,
                )
                (all_t, cross_pad, cross_list,
                 pc_yaw, pc_aspd, pc_fx, pc_fz, pc_curv,
                 ) = _build_vehicle_collision_data(
                    v, dynamic_horizon, ego_yaw_rad, ego_fwd_x, ego_fwd_z, cal,
                    self._curvature_blender, now_mono,
                    follow_decel_ms2=follow_decel,
                )
                dist_sq = dx * dx + dz * dz
                vehicle_collision_data[v.id] = (
                    all_t, cross_pad, cross_list,
                    dx, dz, dist_sq,
                    pc_yaw, pc_aspd, pc_fx, pc_fz, pc_curv,
                )

        for v in vehicles_eff:
            vx, vz = v.position.x, v.position.z

            # Feed the LOS track regardless of pipeline outcome so veto
            # evidence exists before a target ever produces a hit.
            trk = self._los_tracks.get(v.id)
            if trk is None:
                trk = self._los_tracks[v.id] = deque()
            trk.append((now_mono, vx, vz, ego_x, ego_z))
            los_cutoff = now_mono - cal.los_veto_window_s
            while trk and trk[0][0] < los_cutoff:
                trk.popleft()

            pc = vehicle_collision_data.get(v.id)
            if pc is not None:
                (all_target_arcs, cross_padding, precomputed_cross_arcs,
                 dx, dz, dist_sq,
                 v_yaw_rad, abs_v_speed, veh_fwd_x, veh_fwd_z, v_curvature) = pc
                dist = math.sqrt(dist_sq)
                v_hw = v.size.width / 2.0
                arc_curvature = (
                    all_target_arcs[0].curvature if all_target_arcs else v_curvature
                )
            else:
                dx = vx - ego_x
                dz = vz - ego_z
                dist_sq = dx * dx + dz * dz
                if dist_sq > max_range_sq:
                    continue
                rz = _world_to_ego_forward(dx, dz, ego_yaw_rad)
                expected_y = ego_y + rz * math.tan(ego_pitch_rad)
                if abs(v.position.y - expected_y) > cal.elevation_margin:
                    continue
                dist = math.sqrt(dist_sq)
                v_yaw_rad = (
                    v._smooth_yaw
                    if v._smooth_yaw is not None
                    else math.radians(v.rotation.euler()[1])
                )
                v_hw = v.size.width / 2.0
                abs_v_speed = abs(v.speed)
                v_curvature = _vehicle_curvature_blend(
                    v, abs_v_speed, cal, self._curvature_blender, now_mono,
                )
                veh_fwd_x = -math.sin(v_yaw_rad)
                veh_fwd_z = -math.cos(v_yaw_rad)
                fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
                arc_curvature = _dampen_turning_curvature(
                    v_curvature, fwd_dot,
                    ego_fwd_x, ego_fwd_z, veh_fwd_x, veh_fwd_z,
                    abs_v_speed, abs_v_speed * dynamic_horizon,
                    cal,
                )
                precomputed_cross_arcs = None
                all_target_arcs = []
                cross_padding = 0.0

            veh_arc = v.get_arc(
                dynamic_horizon,
                arc_start_pctg=cal.arc_start_pctg,
                curvature_override=arc_curvature,
            )
            trailer_dicts = []
            trailer_arcs: list[ArcPath] = []
            for tr in v.trailers:
                tr_arc_pos = tr.position
                tr_dict_pos = tr.position
                _, tr_yaw_deg, _ = tr.rotation.euler()
                tr_yaw_rad = math.radians(tr_yaw_deg)
                tr_hw = tr.size.width / 2.0
                tr_is_rev = v.speed < -1e-3
                tr_effective_p = (1.0 - cal.arc_start_pctg) if tr_is_rev else cal.arc_start_pctg
                tr_fwd_x_l = -math.sin(tr_yaw_rad)
                tr_fwd_z_l = -math.cos(tr_yaw_rad)
                tr_body_offset = (tr_effective_p - 0.5) * tr.size.length
                tr_arc = build_arc(
                    tr_arc_pos.x + tr_body_offset * tr_fwd_x_l,
                    tr_arc_pos.z + tr_body_offset * tr_fwd_z_l,
                    tr_yaw_rad,
                    v.speed, arc_curvature, tr_hw, dynamic_horizon,
                )
                trailer_arcs.append(tr_arc)
                trailer_dicts.append({
                    "x": tr_dict_pos.x, "z": tr_dict_pos.z,
                    "yaw": tr_yaw_rad,
                    "half_w": tr_hw,
                    "length": tr.size.length,
                    "is_tmp": tr.is_tmp,
                    "speed_kmh": abs(v.speed) * 3.6,
                })

            veh_dict = {
                "vid": v.id,
                "x": vx, "z": vz,
                "yaw": v_yaw_rad,
                "half_w": v_hw,
                "length": v.size.length,
                "is_tmp": v.is_tmp,
                "is_trailer": getattr(v, "is_trailer", False),
                "kinematics_swapped": getattr(v, "_debug_kinematics_swapped", False),
                "speed_kmh": abs(v.speed) * 3.6,
                "trailers": trailer_dicts,
            }

            vehicle_arcs[v.id] = [veh_arc] + trailer_arcs

            if not run_collision:
                vehicle_dicts.append(veh_dict)
                continue

            # Build FilterContext and run pipeline
            ctx = FilterContext(
                v=v,
                ego_arc=ego_arc,
                ego_braked_arc=ego_braked_arc,
                ego_evasion_left=ego_evasion_left,
                ego_evasion_right=ego_evasion_right,
                ego_x=ego_x, ego_y=ego_y, ego_z=ego_z,
                ego_yaw_rad=ego_yaw_rad,
                ego_speed=ego_speed,
                ego_pitch_rad=ego_pitch_rad,
                ego_curvature=ego_curvature,
                ego_fwd_x=ego_fwd_x,
                ego_fwd_z=ego_fwd_z,
                ego_hw=ego_hw,
                dynamic_horizon=dynamic_horizon,
                tmp_traffic_session=tmp_traffic_session,
                ref_kmh_for_filter=ref_kmh_for_filter,
                cal=cal,
                dx=dx, dz=dz,
                dist_sq=dist_sq, dist=dist,
                v_yaw_rad=v_yaw_rad,
                abs_v_speed=abs_v_speed,
                veh_fwd_x=veh_fwd_x,
                veh_fwd_z=veh_fwd_z,
                v_curvature=v_curvature,
                all_target_arcs=all_target_arcs,
                precomputed_cross_arcs=precomputed_cross_arcs,
                cross_padding=cross_padding,
                latched_threat_ids=self._latched_threat_ids,
                follow_threat_ids=self._follow_threat_ids,
                d_miss=los_miss(v),
                d_miss_rate=los_miss_rate(v),
            )

            suppression_reasons[v.id] = []
            for stage in self._pipeline:
                res = stage.apply(ctx)
                if res.suppressed:
                    suppression_reasons[v.id].append(res)
                    reason = res.reason or ""
                    if reason in ("OppositeLaneFilter", "EgoEvasionFilter",
                                  "OppositeLaneFilterMirrored"):
                        if ctx.head_on:
                            oncoming_evasion_filtered_ids.add(v.id)
                        else:
                            evasion_filtered_ids.add(v.id)
                    elif reason in ("CornerEntryStationaryFilter",
                                  "CornerEntryStationaryFilterMirrored"):
                        oncoming_evasion_filtered_ids.add(v.id)
                    else:
                        suppressed_ids.add(v.id)
                    vehicle_dicts.append(veh_dict)
                    break
            else:
                # No stage suppressed: evaluate collision
                if not all_target_arcs:
                    vehicle_dicts.append(veh_dict)
                    continue

                fwd_dot = ctx.fwd_dot
                head_on = ctx.head_on
                near_head_on = ctx.near_head_on
                lateral_gap = ctx.lateral_gap

                # Re-derive cross_arcs respecting Fix A via lane classification
                own_lane_for_fix_a = ctx.lane in (Lane.OPPOSITE_OR_OUTER, Lane.OFF_ROAD)
                fix_a_active = ctx.near_head_on and own_lane_for_fix_a
                if fix_a_active:
                    effective_cross_padding = cross_padding * cal.near_head_on_cross_scale
                else:
                    effective_cross_padding = cross_padding

                found_hit = False
                for arc_idx, base_target_arc in enumerate(all_target_arcs):
                    if fix_a_active or precomputed_cross_arcs is None:
                        cross_arcs = _apply_cross_zone(base_target_arc, effective_cross_padding)
                    else:
                        cross_arcs = precomputed_cross_arcs[arc_idx]

                    unbraked_hit = _earliest_hit(
                        ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                        lateral_gap,
                    )
                    if unbraked_hit is None:
                        continue

                    unbraked_ttc = unbraked_hit[0]
                    colliding_ids.add(v.id)
                    if ctx.lane == Lane.EGO:
                        ego_lane_colliding_ids.add(v.id)
                    aligned = abs(ctx.fwd_dot) >= cal.aeb_certain_fwd_dot
                    # Pose fixes a lane only inside the range an unseen bend
                    # cannot span (README lane confidence range).
                    d_miss_v = ctx.d_miss
                    lane_trusted = (
                        ctx.co_directional or dist <= cal.lane_confidence_range_m
                    )
                    if ctx.lane == Lane.EGO and aligned and lane_trusted:
                        certain_geom_ids.add(v.id)
                    elif head_on and oncoming_closing_into(ctx, cal):
                        # Turn-into-path: shrinking CBDR miss while ego turns.
                        certain_geom_ids.add(v.id)
                    # Co-directional and crash_confirmed skip the oblique window:
                    # neither is the extrapolation-fragile class it exists for.
                    if ((ctx.lane == Lane.EGO or aligned) and lane_trusted
                            or ctx.co_directional
                            or getattr(v, "crash_confirmed", False)
                            or v.id in certain_geom_ids):
                        nearcertain_geom_ids.add(v.id)
                    # Warn-persistence classes: evidence quality, not threat
                    # level, so they add latency only (README warn persistence).
                    if head_on or near_head_on:
                        oncoming_colliding_ids.add(v.id)
                    if v.id not in wide_lat_checked_ids:
                        wide_lat_checked_ids.add(v.id)
                        _, warn_d_abs = project_to_ego_arc(
                            ego_arc, ego_x + dx, ego_z + dz,
                        )
                        if warn_d_abs > cal.aeb_warn_wide_lat_m:
                            self._warn_wide_seen[v.id] = now_mono
                        # Sticky: closing across the bar must not hand back the
                        # instant warn this gate was already refusing.
                        seen = self._warn_wide_seen.get(v.id)
                        if (seen is not None
                                and now_mono - seen <= cal.aeb_warn_wide_lat_sticky_s):
                            wide_lat_colliding_ids.add(v.id)
                    nearest_colliding_range = min(nearest_colliding_range, dist)
                    newly_risky.add(v.id)
                    rc = self._risk_confirm.get(v.id)
                    if rc is None:
                        rc = OccupancyConfirm(
                            cal.aeb_confirm_occupancy,
                            cal.aeb_confirm_max_gap_frames,
                            cal.risk_confirm_s,
                        )
                        self._risk_confirm[v.id] = rc
                    rc.window_s = (
                        cal.risk_confirm_oncoming_s if head_on else cal.risk_confirm_s
                    )
                    rc.observe(now_mono, True)
                    if not rc.confirmed(now_mono):
                        continue

                    if unbraked_ttc < best_unbraked_ttc:
                        best_unbraked_ttc = unbraked_ttc
                        best_hit_x = unbraked_hit[1]
                        best_hit_z = unbraked_hit[2]

                    braked_hit = _earliest_hit(
                        ego_braked_arc, cross_arcs,
                        cal.corridor_margin,
                        cal.collision_samples,
                        lateral_gap,
                    )

                    # Closing-speed comparison: vector magnitude of the
                    # closing_speed = world-frame |v_ego - v_target| (README braking_worsens).
                    v_ego_x = ego_speed * ego_fwd_x
                    v_ego_z = ego_speed * ego_fwd_z
                    v_t_x = v.speed * veh_fwd_x
                    v_t_z = v.speed * veh_fwd_z
                    closing_unbraked = math.hypot(v_ego_x - v_t_x, v_ego_z - v_t_z)
                    v_target_along_ego = v_t_x * ego_fwd_x + v_t_z * ego_fwd_z

                    braking_worsens = False
                    if braked_hit is not None:
                        # Faster along ego axis: braking worsens imminent rear-end.
                        if v_target_along_ego > ego_speed:
                            braking_worsens = True
                        else:
                            t_braked = braked_hit[0]
                            v_ego_braked = max(0.0, ego_speed - effective_decel * t_braked)
                            v_egb_x = v_ego_braked * ego_fwd_x
                            v_egb_z = v_ego_braked * ego_fwd_z
                            closing_braked = math.hypot(v_egb_x - v_t_x, v_egb_z - v_t_z)
                            if closing_braked > closing_unbraked + cal.brake_worsens_hysteresis_ms:
                                braking_worsens = True
                    elif (v_target_along_ego > ego_speed
                            and dx * ego_fwd_x + dz * ego_fwd_z < 0.0):
                        # Rear overtaker with no braked hit: adjacent pass, not frontal threat.
                        braking_worsens = True

                    if braking_worsens:
                        braking_worsens_ids.add(v.id)
                        found_hit = True
                        continue

                    ttb = unbraked_ttc

                    if ttb < best_ttb:
                        best_ttb = ttb
                        best_threat_vid = v.id
                        best_hit_x = unbraked_hit[1]
                        best_hit_z = unbraked_hit[2]
                        # required_decel gap: min(v_rel*ttc, ego travel to hit); see README.
                        hit_dist = math.hypot(
                            unbraked_hit[1] - ego_front_x,
                            unbraked_hit[2] - ego_front_z,
                        )
                        best_hit_dist = hit_dist
                        best_closing_distance = min(
                            closing_unbraked * unbraked_ttc, hit_dist,
                        )
                        best_v_closing = closing_unbraked
                        best_ego_travel = max(hit_dist - ego_front_to_surface, 0.0)
                        best_codir_cap = _codir_required_cap(
                            base_target_arc, fwd_dot, unbraked_ttc,
                            best_ego_travel, closing_unbraked, ego_speed, cal,
                            response_s=load_response_s,
                        )

                    # LOS-rate veto (engagement entry only): once per vehicle
                    # per frame, memoized across its arcs.
                    vetoed = los_veto_memo.get(v.id)
                    if vetoed is None:
                        vetoed = False
                        min_range, miss_bar = _los_veto_bar(
                            fwd_dot, abs_v_speed, ctx.v_curvature, cal,
                        )
                        # crash_confirmed skips LOS veto (README LOS veto).
                        # Turn-into-path: CBDR miss is still large while closing.
                        if (cal.los_veto_enabled
                                and dist >= min_range
                                and not getattr(v, "crash_confirmed", False)
                                and not oncoming_closing_into(ctx, cal)):
                            vetoed = d_miss_v is not None and d_miss_v > miss_bar
                        los_veto_memo[v.id] = vetoed
                        if vetoed:
                            los_vetoed_ids.add(v.id)
                    if not vetoed and _extrapolation_veto(
                            ctx, unbraked_ttc, v_target_along_ego,
                            _any_body_in_ego_lane(
                                ego_arc, all_target_arcs, cal.lane_half_width),
                            cal, d_miss_v):
                        vetoed = True
                    if vetoed:
                        engage_vetoed_ids.add(v.id)
                    if not vetoed and ttb < best_ttb_engage:
                        best_ttb_engage = ttb
                        # Same tighter-of-two-frames distance as
                        # best_closing_distance above.
                        hit_dist_e = math.hypot(
                            unbraked_hit[1] - ego_front_x,
                            unbraked_hit[2] - ego_front_z,
                        )
                        best_closing_distance_engage = min(
                            closing_unbraked * unbraked_ttc, hit_dist_e,
                        )
                        best_v_closing_engage = closing_unbraked
                        best_ego_travel_engage = max(
                            hit_dist_e - ego_front_to_surface, 0.0,
                        )
                        best_codir_cap_engage = _codir_required_cap(
                            base_target_arc, fwd_dot, unbraked_ttc,
                            best_ego_travel_engage, closing_unbraked,
                            ego_speed, cal, response_s=load_response_s,
                        )
                    found_hit = True

                vehicle_dicts.append(veh_dict)

        for vid in list(self._risk_confirm.keys()):
            if vid in newly_risky:
                continue
            rc = self._risk_confirm[vid]
            rc.observe(now_mono, False)
            if not rc.tracking:
                del self._risk_confirm[vid]
        _active_vids = {v.id for v in vehicles_eff}
        self._curvature_blender.prune(_active_vids)
        for vid in list(self._los_tracks.keys()):
            if vid not in _active_vids:
                del self._los_tracks[vid]
        for vid in list(self._d_miss_hist.keys()):
            if vid not in _active_vids:
                del self._d_miss_hist[vid]
        for vid in list(self._follow_tracks.keys()):
            if vid not in _active_vids:
                del self._follow_tracks[vid]
                self._follow_hold_until.pop(vid, None)

        time_to_brake = best_ttb if (run_collision and best_ttb < _INF) else _INF
        display_ttc = best_unbraked_ttc

        slope_accel = _GRAVITY_MS2 * math.sin(ego_pitch_rad)
        downhill_offset = max(-slope_accel, 0.0)
        capacity_estimate = _max_brake_live
        effective_max_decel = max(
            0.1, cal.ego_decel_frac * capacity_estimate - downhill_offset,
        )
        # Entry base: ego_decel_frac is a margin on the command, and folding it
        # in here hedged twice (README engagement thresholds).
        capability_decel = max(0.1, capacity_estimate - downhill_offset)

        required_decel = 0.0
        if run_collision and best_closing_distance < _INF and best_v_closing > 0.0:
            required_decel = _required_decel_two_frame(
                best_closing_distance, best_v_closing, best_ego_travel,
                ego_speed, cal, best_codir_cap, response_s=load_response_s,
            )

        # Slope modifies a threat-derived demand, it never sources one: gravity
        # alone must not warn or feed FF (README warn/FF contract).
        threat_present = required_decel > 0.0
        effective_required = required_decel + downhill_offset

        # The engage fraction is a hedge against uncertain geometry. Aligned
        # in-lane traffic needs no hedge (README geometry-graded engage).
        certain_engage = any(
            vid in certain_geom_ids and vid not in engage_vetoed_ids
            for vid in colliding_ids
        )
        engage_threshold = capability_decel * (
            cal.aeb_engage_frac_certain if certain_engage else cal.aeb_engage_frac
        )
        disarm_threshold = cal.aeb_disarm_frac * effective_max_decel
        warn_threshold = cal.aeb_warn_frac * effective_max_decel

        ego_kmh_abs = abs(ego_speed) * 3.6
        aeb_speed_ok = ego_kmh_abs >= cal.aeb_min_engage_speed_kmh
        # Warn/FF and new engagements require ego above the speed floor so
        # cross-traffic driving into a stopped truck does not trigger AEB.
        # An event already latched while moving may continue to completion.
        aeb_outputs_ok = aeb_speed_ok or self._engaged

        # brake_ttb + response window: TTB slam headroom for actuator lag (README).
        brake_ttb_active = (
            run_collision
            and time_to_brake < cal.brake_ttb + cal.brake_response_window_s
        )

        # Geometry latch while colliding unbraked ttc inside disarm_hold_ttc_s.
        geom_threat_latched = (
            run_collision
            and best_unbraked_ttc < cal.disarm_hold_ttc_s
            and bool(colliding_ids)
        )

        # Latched headway hold + scope release (README latched-threat).
        active_vid_set = {v.id for v in vehicles_eff}
        ego_v_safe = max(ego_speed, 0.5)
        latched_headway_min = _INF
        for vid in list(self._latched_threat_ids):
            if vid not in active_vid_set:
                self._latched_threat_ids.discard(vid)
                self._latched_scope_ok_mono.pop(vid, None)
                continue
            pc = vehicle_collision_data.get(vid)
            if pc is None:
                self._latched_threat_ids.discard(vid)
                self._latched_scope_ok_mono.pop(vid, None)
                continue
            # Latched scope: forward in-lane or still colliding (README scope release).
            s_along, d_abs = project_to_ego_arc(
                ego_arc, ego_x + pc[3], ego_z + pc[4],
            )
            in_scope = (
                vid in colliding_ids
                or (s_along > 0.0 and d_abs <= cal.lane_half_width)
            )
            if in_scope:
                self._latched_scope_ok_mono[vid] = now_mono
            else:
                last_ok = self._latched_scope_ok_mono.get(vid)
                if last_ok is None:
                    self._latched_scope_ok_mono[vid] = last_ok = now_mono
                if now_mono - last_ok > cal.latched_scope_release_s:
                    self._latched_threat_ids.discard(vid)
                    self._latched_scope_ok_mono.pop(vid, None)
                    continue
            dist_vid = math.sqrt(pc[5])
            gap = max(dist_vid - cal.stop_buffer, 0.0)
            hw = gap / ego_v_safe
            if hw > cal.latched_release_headway_s:
                self._latched_threat_ids.discard(vid)
                self._latched_scope_ok_mono.pop(vid, None)
                continue
            if hw < latched_headway_min:
                latched_headway_min = hw

        latched_distance_threat = (
            run_collision
            and bool(self._latched_threat_ids)
            and latched_headway_min < cal.latched_min_headway_s
        )

        # Engagement entry evaluates only LOS-eligible targets. When no veto
        # fired this frame these equal the full aggregates and behaviour is
        # identical to the unvetoed pipeline.
        required_decel_engage = 0.0
        if (run_collision and best_closing_distance_engage < _INF
                and best_v_closing_engage > 0.0):
            required_decel_engage = _required_decel_two_frame(
                best_closing_distance_engage, best_v_closing_engage,
                best_ego_travel_engage, ego_speed, cal,
                best_codir_cap_engage, response_s=load_response_s,
            )
        effective_required_engage = required_decel_engage + downhill_offset
        brake_ttb_engage_active = (
            run_collision
            and best_ttb_engage < cal.brake_ttb + cal.brake_response_window_s
        )

        if self._engaged:
            if (effective_required < disarm_threshold
                    and not brake_ttb_active
                    and not geom_threat_latched
                    and not latched_distance_threat):
                self._engaged = False
            self._engage_confirm.reset()
        else:
            # New engagements are gated by |ego_speed|: below the threshold
            # the truck is essentially crawling, the user has authority
            qualified = (run_collision
                         and aeb_speed_ok
                         and (effective_required_engage >= engage_threshold
                              or brake_ttb_engage_active))
            # Tiered engage confirm window + occupancy observe (README tiered entry).
            self._engage_confirm.window_s = (
                cal.aeb_engage_confirm_s
                if any(vid in nearcertain_geom_ids for vid in colliding_ids)
                else cal.aeb_engage_confirm_oblique_s
            )
            self._engage_confirm.observe(now_mono, qualified)
            if qualified:
                # Instant engage paths: TTB slam, certain geom, latched (README).
                certain = (
                    brake_ttb_engage_active
                    or any(vid in certain_geom_ids and vid not in engage_vetoed_ids
                           for vid in colliding_ids)
                    or any(vid in self._latched_threat_ids
                           for vid in colliding_ids)
                )
                if certain or self._engage_confirm.confirmed(now_mono):
                    self._engaged = True

        # Promote every currently-colliding target into the latched set so
        # subsequent frames keep them in the pipeline and in the hold check.
        # Newly latched ids start their scope grace at promotion time.
        if self._engaged and run_collision and colliding_ids:
            self._latched_threat_ids.update(colliding_ids)
            for vid in colliding_ids:
                self._latched_scope_ok_mono.setdefault(vid, now_mono)

        if self._engaged:
            if brake_ttb_active:
                target_raw = effective_max_decel
            else:
                target_raw = max(0.0, min(effective_required, effective_max_decel))
                if latched_distance_threat:
                    target_raw = max(
                        target_raw,
                        cal.latched_min_decel_frac * effective_max_decel,
                    )
        else:
            target_raw = 0.0

        if self._prev_loop_mono is None:
            dt_loop = self.loop_interval
        else:
            dt_loop = max(1e-3, min(0.5, now_mono - self._prev_loop_mono))
        self._prev_loop_mono = now_mono

        delta = target_raw - self._published_target_ms2
        time_since_change = now_mono - self._last_target_change_mono
        if self._published_target_ms2 <= 1e-6 and target_raw > 0.0:
            # Engagement edge: step straight to the requirement. Ramping from zero
            # costs the whole build-up window and leaves metres of gap unspent.
            target_published = target_raw
            self._last_target_change_mono = now_mono
        elif (abs(delta) < cal.aeb_target_deadband_ms2
                and time_since_change < cal.aeb_target_refresh_min_s):
            target_published = self._published_target_ms2
        else:
            rate = (
                cal.aeb_target_rate_engaged_ms3 if self._engaged
                else cal.aeb_target_rate_ms3
            )
            slew_limit = rate * dt_loop
            step = max(-slew_limit, min(slew_limit, delta))
            target_published = self._published_target_ms2 + step
            target_published = max(0.0, target_published)
            if abs(target_published - self._published_target_ms2) > 1e-6:
                self._last_target_change_mono = now_mono
        self._published_target_ms2 = target_published

        if run_collision and effective_max_decel > 0.1 and aeb_outputs_ok:
            if brake_ttb_active:
                aeb_ff_decel = effective_max_decel
            elif threat_present:
                aeb_ff_decel = min(effective_required, effective_max_decel)
            else:
                aeb_ff_decel = 0.0
        else:
            aeb_ff_decel = 0.0

        warn_by_decel = (
            run_collision
            and aeb_outputs_ok
            and threat_present
            and effective_required >= warn_threshold
        )
        warn_by_ttb = (
            run_collision and aeb_outputs_ok and time_to_brake < cal.warn_ttb
        )
        warn_raw = bool(warn_by_decel or warn_by_ttb)
        # A beep about something this far out is not actionable, only noise.
        if warn_raw and nearest_colliding_range > cal.aeb_warn_max_range_m:
            warn_raw = False

        # Oblique warn occupancy gate; instant paths unchanged (README warn persistence).
        # Windows are refreshed per frame so a candidate calibration can move them.
        self._warn_confirm.window_s = cal.aeb_warn_confirm_oblique_s
        self._warn_confirm.observe(now_mono, warn_raw)

        # Every colliding target vetoed out of engagement and out of ego's lane
        # is the extrapolation-phantom class: warn waits on a longer window.
        warn_vetoed_only = bool(colliding_ids) and all(
            vid in engage_vetoed_ids and vid not in ego_lane_colliding_ids
            for vid in colliding_ids
        )
        self._warn_vetoed_confirm.window_s = cal.aeb_warn_confirm_vetoed_s
        self._warn_vetoed_confirm.observe(now_mono, warn_raw and warn_vetoed_only)

        # Two more phantom-beep classes: every colliding target oncoming, or
        # every one a full lane off the ego arc. Both gate latency, not silence.
        warn_oncoming_only = bool(colliding_ids) and colliding_ids <= oncoming_colliding_ids
        self._warn_oncoming_confirm.window_s = cal.aeb_warn_confirm_oncoming_s
        self._warn_oncoming_confirm.observe(now_mono, warn_raw and warn_oncoming_only)

        warn_wide_lat_only = bool(colliding_ids) and colliding_ids <= wide_lat_colliding_ids
        self._warn_wide_lat_confirm.window_s = cal.aeb_warn_confirm_wide_lat_s
        self._warn_wide_lat_confirm.observe(now_mono, warn_raw and warn_wide_lat_only)

        self._warn_instant_confirm.window_s = cal.aeb_warn_instant_min_s
        self._warn_instant_confirm.observe(now_mono, warn_raw)
        for _vid, _seen in list(self._warn_wide_seen.items()):
            if now_mono - _seen > cal.aeb_warn_wide_lat_sticky_s:
                del self._warn_wide_seen[_vid]

        if warn_raw:
            warn_latched = any(
                vid in self._latched_threat_ids for vid in colliding_ids
            )
            # The TTB slam assumes an in-path target; off-arc sets lose it.
            warn_ttb_hard = brake_ttb_active and not (
                cal.aeb_warn_ttb_needs_narrow and warn_wide_lat_only
            )
            warn_hard = bool(self._engaged or warn_ttb_hard or warn_latched)
            warn_instant = warn_hard or (
                any(vid in nearcertain_geom_ids for vid in colliding_ids)
                and self._warn_instant_confirm.confirmed(now_mono)
            )
            aeb_warn = warn_instant or self._warn_confirm.confirmed(now_mono)
            if aeb_warn and not warn_hard:
                if warn_vetoed_only and not self._warn_vetoed_confirm.confirmed(now_mono):
                    aeb_warn = False
                if (aeb_warn and warn_oncoming_only
                        and not self._warn_oncoming_confirm.confirmed(now_mono)):
                    aeb_warn = False
                if (aeb_warn and warn_wide_lat_only
                        and not self._warn_wide_lat_confirm.confirmed(now_mono)):
                    aeb_warn = False
        else:
            aeb_warn = False


        user_braking_now = self._read_user_braking()
        # Shares the engage base so any engagement still warns while the driver
        # brakes: aeb_warn_near_full_frac must stay equal to it (TUNING.md).
        near_full_target = (
            capability_decel > 0.1
            and effective_required >= cal.aeb_warn_near_full_frac * capability_decel
        )
        # Suppression must outlive the state hold below, which re-asserts warn
        # on a held WARN/BRAKE and would otherwise undo it for up to 0.3 s.
        warn_suppressed = bool(user_braking_now and not near_full_target)
        if warn_suppressed:
            aeb_warn = False

        aeb_brake = bool(self._engaged and target_published > 0.0)

        if aeb_brake:
            new_state = AEBState.BRAKE
        elif aeb_warn:
            new_state = AEBState.WARN
        else:
            new_state = AEBState.STANDBY

        if self._prev_state.value > new_state.value and now_mono < self._state_hold_until:
            new_state = self._prev_state
            if new_state == AEBState.BRAKE:
                aeb_brake = True
                aeb_warn = not warn_suppressed
            elif new_state == AEBState.WARN:
                aeb_warn = not warn_suppressed
        if new_state != self._prev_state:
            self._state_hold_until = now_mono + 0.3

        self._prev_state = new_state

        if aeb_brake:
            if self._brake_span_start_mono is None:
                self._brake_span_start_mono = now_mono
            elif (not self._brake_popup_fired
                    and now_mono - self._brake_span_start_mono >= _BRAKE_POPUP_MIN_DURATION_S
                    and abs(ego_speed) <= _BRAKE_POPUP_STOPPED_SPEED_MS):
                # Imported lazily: keeps this module (and the AEB tests) Qt-free.
                from ui.popup.popup_window import PopupWindow
                PopupWindow.emit(
                    "AEB intervention",
                    "Automatic Emergency Braking intervention",
                    "w",
                )
                self._brake_popup_fired = True
        else:
            self._brake_span_start_mono = None
            self._brake_popup_fired = False

        realized_decel = 0.0
        try:
            st = registry.get_thread("sending_thread")
            if st is not None and st.is_alive():
                with st.data._lock:
                    realized_decel = float(
                        getattr(st.data, "decel_measured_lead_ms2",
                                st.data.decel_measured_ms2)
                    )
        except (KeyError, AttributeError):
            pass

        user_brake = self._read_user_braking()
        if not tmp_traffic_session:
            self._latched_filter_ego_kmh = None
        elif not aeb_warn and not user_brake:
            self._latched_filter_ego_kmh = None
        elif self._latched_filter_ego_kmh is None and (aeb_warn or user_brake):
            self._latched_filter_ego_kmh = ego_kmh_now

        snap = AEBSnapshot(
            ego_x=ego_x, ego_z=ego_z, ego_yaw=ego_yaw_rad,
            ego_speed=ego_speed, ego_half_w=ego_hw, ego_half_l=ego_half_l,
            ego_arc=ego_arc, ego_braked_arc=ego_braked_arc,
            ego_has_trailer=ego_has_trailer,
            vehicles=vehicle_dicts, vehicle_arcs=vehicle_arcs,
            colliding_ids=colliding_ids, suppressed_ids=suppressed_ids,
            braking_worsens_ids=braking_worsens_ids,
            evasion_filtered_ids=evasion_filtered_ids,
            oncoming_evasion_filtered_ids=oncoming_evasion_filtered_ids,
            los_vetoed_ids=los_vetoed_ids,
            engage_vetoed_ids=engage_vetoed_ids,
            aeb_state=new_state, time_to_collision=display_ttc,
            time_to_brake=time_to_brake,
            hit_x=best_hit_x, hit_z=best_hit_z,
            evasion_left_arc=ego_evasion_left,
            evasion_right_arc=ego_evasion_right,
            suppression_reasons=suppression_reasons,
            tmp_traffic_session=tmp_traffic_session,
        )

        if aeb_warn:
            self._sound_handler.start_warning()
        else:
            self._sound_handler.stop_warning()

        with self.data._lock:
            self.data.AEB_warn = aeb_warn
            self.data.AEB_brake = aeb_brake
            self.data.time_to_brake = time_to_brake
            self.data.em_stop_requested = aeb_brake
            self.data.AEB_target_decel_ms2 = target_published
            self.data.AEB_ff_decel_ms2 = aeb_ff_decel
            self.data.AEB_required_decel_ms2 = effective_required
            self.data.AEB_effective_max_decel_ms2 = effective_max_decel
            self.data.AEB_realized_decel_ms2 = realized_decel
            self.data.snapshot = snap
        self._last_snapshot = snap

        self._capture_aeb_tick(
            now_mono, radar_t_mono, aeb_active, snap,
            _max_brake_live, effective_required, effective_max_decel,
            target_published, aeb_ff_decel, aeb_warn, aeb_brake, new_state,
            tmp_traffic_session,
        )

        if Settings.debug and run_collision and (
            self._engaged or bool(colliding_ids) or required_decel > 0.5
        ):
            self._log_gap_debug(
                now_mono=now_mono,
                cal=cal,
                ego_x=ego_x,
                ego_z=ego_z,
                ego_yaw_rad=ego_yaw_rad,
                ego_speed=ego_speed,
                ego_hw=ego_hw,
                ego_half_l=ego_half_l,
                ego_cap_fwd=ego_cap_fwd,
                ego_cap_back=ego_cap_back,
                ego_front_to_surface=ego_front_to_surface,
                ego_arc=ego_arc,
                vehicles_eff=vehicles_eff,
                vehicle_collision_data=vehicle_collision_data,
                best_threat_vid=best_threat_vid,
                best_ttb=best_ttb,
                best_unbraked_ttc=best_unbraked_ttc,
                best_closing_distance=best_closing_distance,
                best_v_closing=best_v_closing,
                best_ego_travel=best_ego_travel,
                best_hit_dist=best_hit_dist,
                best_codir_cap=best_codir_cap,
                required_decel=required_decel,
                required_decel_engage=required_decel_engage,
                effective_required=effective_required,
                effective_required_engage=effective_required_engage,
                engage_threshold=engage_threshold,
                effective_max_decel=effective_max_decel,
                brake_ttb_active=brake_ttb_active,
                brake_ttb_engage_active=brake_ttb_engage_active,
                certain_engage=certain_engage,
                geom_threat_latched=geom_threat_latched,
                latched_distance_threat=latched_distance_threat,
                target_raw=target_raw,
                target_published=target_published,
                colliding_ids=colliding_ids,
                response_s=load_response_s,
            )

    def _log_gap_debug(
        self,
        *,
        now_mono: float,
        cal: AEBCalibration,
        ego_x: float,
        ego_z: float,
        ego_yaw_rad: float,
        ego_speed: float,
        ego_hw: float,
        ego_half_l: float,
        ego_cap_fwd: float,
        ego_cap_back: float,
        ego_front_to_surface: float,
        ego_arc: ArcPath,
        vehicles_eff: list[Vehicle],
        vehicle_collision_data: dict,
        best_threat_vid: int | None,
        best_ttb: float,
        best_unbraked_ttc: float,
        best_closing_distance: float,
        best_v_closing: float,
        best_ego_travel: float,
        best_hit_dist: float,
        best_codir_cap: float,
        required_decel: float,
        required_decel_engage: float,
        effective_required: float,
        effective_required_engage: float,
        engage_threshold: float,
        effective_max_decel: float,
        brake_ttb_active: bool,
        brake_ttb_engage_active: bool,
        certain_engage: bool,
        geom_threat_latched: bool,
        latched_distance_threat: bool,
        target_raw: float,
        target_published: float,
        colliding_ids: set[int],
        response_s: float,
    ) -> None:
        """INFO dump of why AEB thinks a stop gap remains (debug mode only)."""
        engage_edge = self._engaged and not self._gap_debug_was_engaged
        self._gap_debug_was_engaged = self._engaged
        if not engage_edge and (now_mono - self._gap_debug_last_mono) < 0.5:
            return
        self._gap_debug_last_mono = now_mono

        abs_v = abs(ego_speed)
        resp = abs(best_v_closing) * response_s
        d_rel = best_closing_distance - cal.stop_buffer - resp
        d_ego = best_ego_travel - cal.stop_buffer - abs_v * response_s
        ttb_slam_s = cal.brake_ttb + cal.brake_response_window_s
        slam_ttb_reach_m = abs_v * ttb_slam_s

        body_lines: list[str] = []
        if best_threat_vid is not None:
            threat = next((v for v in vehicles_eff if v.id == best_threat_vid), None)
            arcs = None
            pc = vehicle_collision_data.get(best_threat_vid)
            if pc is not None:
                arcs = pc[0]
            if threat is not None and arcs:
                # OBB face gap in ego frame (+rz ahead, +rx left); same as debug radar.
                c, s = math.cos(-ego_yaw_rad), math.sin(-ego_yaw_rad)

                def _w2e(wx: float, wz: float) -> tuple[float, float]:
                    dx = wx - ego_x
                    dz = wz - ego_z
                    return (-dx) * c - dz * s, (-dx) * s + dz * c

                corners = [_w2e(cx, cz) for cx, cz in threat.get_corners()]
                min_rz = min(rz for _, rz in corners)
                max_rz = max(rz for _, rz in corners)
                min_rx = min(rx for rx, _ in corners)
                max_rx = max(rx for rx, _ in corners)
                face_ahead = min_rz - ego_half_l
                face_behind = -max_rz - ego_half_l
                # Right-of-ego face: negative rx in this frame.
                face_right = -max_rx - ego_hw if max_rx < 0.0 else None

                tgt = arcs[0]
                dsq = pair_body_dist_sq(ego_arc, tgt, 0.0)
                body_sep = math.sqrt(max(dsq, 0.0))
                cosd = abs(
                    ego_arc.fwd_x * tgt.fwd_x + ego_arc.fwd_z * tgt.fwd_z
                )
                sind = math.sqrt(max(0.0, 1.0 - cosd * cosd))
                pms = min(
                    getattr(ego_arc, "parallel_margin_scale", 1.0),
                    getattr(tgt, "parallel_margin_scale", 1.0),
                )
                hw_sum = ego_arc.half_width + tgt.half_width
                thr = hw_sum + cal.corridor_margin * (pms + (1.0 - pms) * sind)
                clearance = body_sep - thr
                body_lines = [
                    f"threat id={best_threat_vid} L={threat.size.length:.2f} "
                    f"W={threat.size.width:.2f} trailer={threat.is_trailer} "
                    f"parked={threat.is_parked}",
                    f"obb face_ahead(bumper gap est)={face_ahead:+.3f}m "
                    f"face_behind={face_behind:+.3f}m "
                    f"rx=[{min_rx:+.2f},{max_rx:+.2f}] "
                    f"rz=[{min_rz:+.2f},{max_rz:+.2f}]",
                    (
                        f"obb face_right={face_right:+.3f}m"
                        if face_right is not None else "obb face_right=n/a"
                    ),
                    f"capsule t=0 body_sep={body_sep:.3f}m thr={thr:.3f}m "
                    f"(hw_sum={hw_sum:.3f} margin={cal.corridor_margin:.3f} "
                    f"pms={pms:.2f} sind={sind:.2f}) clearance={clearance:+.3f}m "
                    f"(<0 = already overlapping corridor)",
                    f"ego capsule fwd/back={ego_cap_fwd:.3f}/{ego_cap_back:.3f} "
                    f"_cap={ego_arc._cap_fwd:.3f}/{ego_arc._cap_back:.3f} "
                    f"hw={ego_arc.half_width:.3f}",
                    f"tgt capsule fwd/back={tgt.fwd_len:.3f}/{tgt.back_len:.3f} "
                    f"_cap={tgt._cap_fwd:.3f}/{tgt._cap_back:.3f} "
                    f"hw={tgt.half_width:.3f}",
                ]

        engage_why = []
        if brake_ttb_engage_active:
            engage_why.append("brake_ttb_slam")
        if effective_required_engage >= engage_threshold:
            engage_why.append("required_decel")
        if geom_threat_latched:
            engage_why.append("geom_latch")
        if latched_distance_threat:
            engage_why.append("latched_headway")
        if not engage_why:
            engage_why.append("none")

        logger.info(
            "AEB gap debug: engaged=%s edge=%s speed=%.1fkm/h colliding=%s\n"
            "  cal stop_buffer=%.3f response_s=%.3f corridor_margin=%.3f "
            "parallel_scale=%.2f slam_ttb=%.2fs reach@v slam=%.2fm\n"
            "  closing_dist=%.3f hit_dist=%.3f ego_travel=%.3f "
            "ego_front_to_surface=%.3f (cap_fwd+hw) v_close=%.2f\n"
            "  d_rel=%.3f d_ego=%.3f resp_term=%.3f codir_cap=%.2f\n"
            "  ttc=%.3f ttb=%.3f req=%.2f req_engage=%.2f "
            "eff_req=%.2f thr_engage=%.2f max=%.2f\n"
            "  flags slam=%s slam_eng=%s certain=%s "
            "why=%s raw=%.2f pub=%.2f\n"
            "  %s",
            self._engaged,
            engage_edge,
            abs_v * 3.6,
            sorted(colliding_ids)[:8],
            cal.stop_buffer,
            response_s,
            cal.corridor_margin,
            cal.capsule_parallel_margin_scale,
            ttb_slam_s,
            slam_ttb_reach_m,
            best_closing_distance if best_closing_distance < _INF else -1.0,
            best_hit_dist if best_hit_dist < _INF else -1.0,
            best_ego_travel if best_ego_travel < _INF else -1.0,
            ego_front_to_surface,
            best_v_closing,
            d_rel,
            d_ego,
            resp,
            best_codir_cap if best_codir_cap < _INF else -1.0,
            best_unbraked_ttc if best_unbraked_ttc < _INF else -1.0,
            best_ttb if best_ttb < _INF else -1.0,
            required_decel,
            required_decel_engage,
            effective_required,
            engage_threshold,
            effective_max_decel,
            brake_ttb_active,
            brake_ttb_engage_active,
            certain_engage,
            "+".join(engage_why),
            target_raw,
            target_published,
            ("\n  ".join(body_lines) if body_lines else "threat body: n/a"),
        )

    def teardown(self) -> None:
        self._sound_handler.cleanup()
        if self._radar_visualizer is not None:
            try:
                self._radar_visualizer.stop()
            except Exception:
                pass
        self._latched_filter_ego_kmh = None
        self._engaged = False
        self._engage_confirm.reset()
        self._warn_confirm.reset()
        self._warn_vetoed_confirm.reset()
        self._warn_oncoming_confirm.reset()
        self._warn_wide_lat_confirm.reset()
        self._warn_instant_confirm.reset()
        self._warn_wide_seen.clear()
        self._risk_confirm.clear()
        self._published_target_ms2 = 0.0
        self._last_target_change_mono = 0.0
        self._prev_loop_mono = None
        self._latched_threat_ids.clear()
        self._latched_scope_ok_mono.clear()
        self._gap_debug_last_mono = 0.0
        self._gap_debug_was_engaged = False
        self._follow_tracks.clear()
        self._follow_hold_until.clear()
        self._follow_threat_ids = set()
        self._capture_prev_state = AEBState.STANDBY
        self._prev_ego_speed_capture_ms = None
        self._brake_span_start_mono = None
        self._brake_popup_fired = False
        self._curvature_blender.prune(set())
        self._los_tracks.clear()
        self._d_miss_hist.clear()
        with self.data._lock:
            self.data.AEB_warn = False
            self.data.AEB_brake = False
            self.data.time_to_brake = _INF
            self.data.em_stop_requested = False
            self.data.AEB_target_decel_ms2 = 0.0
            self.data.AEB_ff_decel_ms2 = 0.0
            self.data.AEB_required_decel_ms2 = 0.0
            self.data.AEB_effective_max_decel_ms2 = 0.0
            self.data.AEB_realized_decel_ms2 = 0.0
            self.data.snapshot = AEBSnapshot()
        logger.debug("AEB teardown complete")

    def _read_radar_snapshot(
        self,
    ) -> tuple[list[Vehicle], float, float, float, float, float, float, float,
               bool, float | None, bool, bool, float] | None:
        """Radar snapshot tuple under lock; None if radar thread missing."""
        try:
            rt = registry.get_thread("radar_thread")
        except KeyError:
            return None
        if rt is None or not rt.is_alive():
            return None
        try:
            with rt.data._lock:
                vehicles = list(rt.data.vehicles)
                return (
                    vehicles,
                    float(rt.data.ego_x),
                    float(rt.data.ego_y),
                    float(rt.data.ego_z),
                    float(rt.data.ego_yaw_rad),
                    float(rt.data.ego_speed),
                    float(rt.data.ego_pitch_deg),
                    float(rt.data.ego_steer),
                    bool(rt.data.ego_has_trailer),
                    rt.data.ego_curvature,
                    bool(rt.data.tmp_session),
                    bool(rt.data.paused),
                    float(rt.data.t_mono),
                )
        except AttributeError:
            return None

