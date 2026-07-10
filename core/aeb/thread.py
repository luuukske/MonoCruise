"""
AEB Thread: Automatic Emergency Braking with arc-based collision detection.

TTB-based detection: see ``core/aeb/AGENTS.md`` §9 for full logic description.

Registry name: ``aeb_thread``
"""

from __future__ import annotations

import copy
import enum
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from core.radar.traffic import (
    Vehicle,
    ArcPath, build_arc, arc_arc_collision, _accel_to_arc_params,
)
from core.aeb.calibration import AEBCalibration, DEFAULT as _CAL_DEFAULT
from core.aeb.lane_frame import project_to_ego_arc, classify, Lane
from core.aeb.capture import get_recorder
from core.aeb.clip_schema import AEBTickRecord, AEBWarmState, ConsumedContext, LiveAEB
from core.aeb.filters import (
    FilterContext, FilterResult,
    _build_vehicle_collision_data, _world_to_ego_forward, _cross_zone_padding,
    _apply_cross_zone, _earliest_hit, _is_approaching, _dampen_turning_curvature,
    _vehicle_curvature_blend, VehicleCurvatureBlender,
    build_pipeline,
)

logger = logging.getLogger(__name__)

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

_AEB_SOUND_PATH = "core/aeb/aeb_warning.wav"
# Extra seamless-loop plays after stop_warning() (avoids a single short blip).
_AEB_WARNING_STOP_EXTRA_REPLAYS = 1

# Constants

_INF: float = 1e9
_GRAVITY_MS2: float = 9.81

# Brake-capacity floor used when sending_thread has not yet published an
# estimate. Slope-corrected per tick before use; never read as a flat ceiling.
_FULL_BRAKE_DECEL_FALLBACK: float = 7.8
# Fraction of max brake capacity assumed for ego stopping / TTB calculations.
# The brake system physically commands only this fraction, reserving headroom
# so a sudden increase in closing speed can still stop the vehicle.
_AEB_EGO_DECEL_FRAC: float = 0.9
_MAX_RANGE: float = 200.0
_MAX_RANGE_SQ: float = _MAX_RANGE ** 2
# TMP-only: |v_ego − v_target| (km/h) vs latched ref ego speed: see _latched_filter_ego_kmh.
_TMP_FILTER_EGO_SPLIT_KMH: float = 40.0
_TMP_FILTER_REL_ABOVE_SPLIT_KMH: float = 15.0
_TMP_FILTER_REL_AT_OR_BELOW_SPLIT_KMH: float = 40.0
_USER_BRAKE_LATCH_THRESHOLD: float = 0.12

_MIN_ARC_HORIZON: float = 2.5
_MAX_ARC_HORIZON: float = 3.0
_CORRIDOR_MARGIN: float = 0.5
_COLLISION_SAMPLES: int = 36

_WARN_TTB_THRESHOLD: float = 1.3
_BRAKE_TTB_THRESHOLD: float = 0.2
_BRAKE_RELEASE_THRESHOLD: float = 0.5
_TIME_TO_BRAKE_BUFFER: float = 0.0

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
# fwd_dot threshold for lateral-gap activation: deliberately looser than the
# head_on threshold (-0.7) to catch oncoming vehicles that never reach -0.7
# during a shared turn.  Does NOT affect target decel model, evasion filter
# bypass, or risk confirm duration: those all still use head_on (-0.7).
_NEAR_HEAD_ON_DOT: float = -0.5
# Minimum target curvature (1/m) to apply the turning-diverge suppression.
# 0.03 ≈ 33 m radius: tight enough to be a real corner, loose enough to
# exclude gentle curves that could still converge on ego.
_TURNING_DIVERGE_CURVATURE: float = 0.007
_EVASION_G_THRESHOLD_ONCOMING: float = 0.13 * 9.81
_EVASION_FILTER_MAX_DELTA_KAPPA: float = 0.008
# Lateral offset from ego's forward axis (m) at which an oncoming vehicle is
# considered to be clearly in its own lane.  Above this, delta_kappa_t is
# scaled up so the evasion arcs fan wider and are more likely to clear ego.
# Uses the cross product: lat = dx*ego_fwd_z - dz*ego_fwd_x (signed, left < 0).
_OPPOSITE_LANE_OFFSET: float = 2.0
_OPPOSITE_LANE_KAPPA_SCALE: float = 2.0
# For head-on vehicles sharing the same curved road (same-sign curvature), ego's
# heading axis cuts across the road and compresses the cross-product lateral offset
# measurement: a vehicle genuinely a full lane away may read as <2.0 m. Use a
# lower threshold when same-curve geometry is confirmed by v_curvature sign + magnitude.
_SAME_CURVE_OWN_LANE_LAT: float = 1.0
_CO_DIR_DIVERGE_LOOKAHEAD_S: float = 0.25
# Fix C: extended lookahead for co-directional same-turn outer-lane suppression.
# Inner/outer lane arcs overlap before their centerlines cross; 0.25 s is too short
# to see the divergence. At horizon × this scale the paths have clearly separated.
_CO_SAME_TURN_LOOKAHEAD_SCALE: float = 0.5
# Sweep-pass suppression: stationary cross-traffic ego turns through.
_SWEEP_PASS_MAX_TARGET_SPEED: float = 1.0    # m/s

# Intersection / shared-turn false-positive suppression
# Fix A: Ghost-arc scaling for near-head-on vehicles clearly in their own lane.
# cross_zone_padding peaks at sin(angle)≈0.8, producing ±4 m ghost arcs at 10 m/s,
# which phantom-widen the target corridor and prevent the ego evasion filter from
# clearing. Only fires when target is laterally displaced into its own lane.
_NEAR_HEAD_ON_CROSS_SCALE: float = 0.3       # ghost-arc reduction factor
_NEAR_HEAD_ON_LATERAL_MIN: float = 3.0       # m: minimum lateral offset to activate Fix A

# Fix B: Road-following curvature expansion for oncoming vehicles in shared turns.
# Expands delta_kappa_t so the oncoming evasion filter tests whether "target follows
# the same corner road as ego": not just a tiny ±0.006 1/m perturbation.
# Still evaluated via arc_arc_collision; not a blind suppression.
_SHARED_TURN_MAX_KAPPA: float = 0.05         # cap on road-following curvature (R ≥ 20 m)

# Fix D: target arc over-rotation suppression.
# A vehicle turning from a side road into the opposite lane maintains high curvature;
# the constant-curvature arc keeps rotating past lane alignment into ego's lane.
# Dampen target curvature when heading rotation over the arc horizon would exceed
# the angle to anti-parallel road alignment.
_TURN_COMPLETE_CURVATURE_SCALE: float = 3.0   # divisor applied when overshoot detected

_TRAILER_TRACTOR_RADIUS_M: float = 30.0
_TRAILER_SWAP_SPEED_THRESHOLD_MS: float = 0.5
_TRAILER_TRACTOR_HEADING_DOT: float = 0.9

# Boundary-negative (TN) sampler thresholds (debug clip capture only, plan
# trigger ``shadow_near``). Occasionally save a clip where AEB correctly stayed
# silent while a filter rejected a real candidate, so a future model learns the
# negatives it must not fire on. These are capture-sampling policy and are
# deliberately NOT in AEBCalibration: they change no braking behaviour and must
# never perturb the calibration fingerprint that keys the clip corpus. The
# rate-limit lives in the recorder (its ``tn_cooldown_s``).
_SHADOW_MIN_SPEED_MS: float = 2.0     # only sample while actually moving
_SHADOW_MAX_RANGE_M: float = 80.0     # ignore filtered radar slots beyond this
# Spatial/geometry filters worth auto-tagging as TN; excludes rel-speed / elevation
# gates that can fire on distant or non-threatening slots while the road looks empty.
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

# Crash clip capture (debug only, trigger ``auto_crash``). Same speed-drop
# criterion as main_pedal_thread emergency detection; capture-only policy, not
# in AEBCalibration.
_CRASH_MIN_SPEED_KMH: float = 40.0
_CRASH_SPEED_DROP_KMH: float = 5.0


def _find_tractor_for_trailer(trailer: Vehicle, vehicles: list[Vehicle]) -> Vehicle | None:
    """Nearest same-heading non-trailer vehicle within _TRAILER_TRACTOR_RADIUS_M.

    Covers both TMP convoy partners (is_tmp=True) and convoy-mode players that
    appear via the AI traffic slot (is_tmp=False). Heading-similarity gate
    prevents grabbing a perpendicular AI car as a phantom tractor.
    """
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
    """Patch trailer-as-vehicle entries with their tractor's kinematics when their
    own speed slot is empty.

    Trailers reported as standalone radar vehicles often have unreliable speed
    (the slot has no engine telemetry: common for TMP partners and for convoy
    players who route through the AI traffic slot). AEB collision math then
    treats them as stationary obstacles directly ahead and false-triggers.
    Keep the trailer's pose (its own arc geometry is correct) but inherit
    kinematics from the nearest same-heading non-trailer within 30 m.
    """
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
    ego_half_w: float = 1.15
    ego_half_l: float = 3.0
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
    """Return [arc] plus two ghost arcs at ±padding along the target heading."""
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
    """Ego-space forward component: rz > 0 = in front of ego."""
    return dx * math.sin(ego_yaw_rad) + dz * math.cos(ego_yaw_rad)


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
    """CBDR predicted miss distance from a target's recent measured track.

    Least-squares slopes of world-frame line-of-sight bearing and range over
    the veto window give d_miss = |omega_los| * R^2 / |v_rel|: the closest
    approach the *measurements* predict under constant relative velocity,
    independent of any arc extrapolation. A genuine collision course holds
    constant bearing (omega ~ 0, d_miss ~ 0); passing traffic drifts.

    Returns None when the track is too short or barely closing: callers must
    treat None as "cannot judge" and fail open (no veto).
    """
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


def _dampen_turning_curvature(
    v_curvature: float,
    fwd_dot: float,
    ego_fwd_x: float, ego_fwd_z: float,
    veh_fwd_x: float, veh_fwd_z: float,
    abs_v_speed: float,
    arc_length: float,
    cal: AEBCalibration = _CAL_DEFAULT,
) -> float:
    """Dampen target curvature when arc would over-rotate past anti-parallel lane alignment.

    Mirrors the ego evasion centerline-snap but on the primary target arc. A vehicle
    turning from a side road into the opposite lane has high curvature; constant-curvature
    propagation keeps rotating past the point where the vehicle straightens into its lane,
    producing a phantom collision in ego's lane.

    Only fires for cross-traffic geometry (fwd_dot in (-0.5, 0.7)) with confirmed rotation
    toward anti-parallel and a heading change that would exceed the alignment angle.
    """
    if (abs(v_curvature) <= cal.turning_diverge_kappa
            or abs_v_speed <= 0.5
            or fwd_dot <= cal.near_head_on_dot
            or fwd_dot >= cal.co_directional_dot):
        return v_curvature
    theta_max = abs(v_curvature) * arc_length
    theta_to_anti = math.acos(max(-1.0, min(1.0, -fwd_dot)))
    if theta_max <= theta_to_anti:
        return v_curvature
    # Direction guard: only dampen when rotating TOWARD anti-parallel.
    # cross(veh_fwd, anti_ego_fwd) > 0 means CW rotation needed (= negative curvature in ETS2).
    # Rotating toward anti-parallel: cross and curvature have opposite signs.
    cross = veh_fwd_x * (-ego_fwd_z) - veh_fwd_z * (-ego_fwd_x)
    if cross * v_curvature >= 0.0:
        return v_curvature  # rotating away: genuine cross-arc threat
    return v_curvature / cal.turn_complete_curvature_scale


def _ls_slope(samples, idx: int) -> float:
    """Least-squares slope (units/s) of samples[i][idx] against samples[i][0].

    Robust to per-tick jitter where an endpoint difference is not; TMP range
    and speed snapshots jitter several units tick to tick.
    """
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
    """Build collision arcs and derived vehicle geometry for a vehicle.

    Returns (all_target_arcs, cross_padding, cross_arcs_list,
             v_yaw_rad, abs_v_speed, veh_fwd_x, veh_fwd_z, v_curvature).
    """
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
    # Fix D: dampen curvature when constant-curvature arc would over-rotate past
    # anti-parallel lane alignment. v_curvature is preserved unchanged for same_curve
    # checks; arc_curvature is used only for arc building.
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


class _SoundState(enum.IntEnum):
    STOPPED = 0
    RUNNING = 1
    SHUTTING_DOWN = 2


class _AEBSoundHandler:
    """
    State-managed sound handler for seamless looping, non-blocking stops,
    and the ability to resume during shutdown.
    """

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
            logger.warning("AEB sound init failed (%s): sound disabled", exc)
            self._sound = None

    def start_warning(self) -> None:
        """
        Start the warning sound loop. If shutting down, cancels the shutdown
        and resumes looping seamlessly.
        """
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
        """
        Signal the warning to stop non-blockingly.
        Schedules ``_stop_extra_replays`` more overlapping plays, then stops;
        the last clip runs to completion.
        """
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
        # Separate edge tracker for clip-capture triggers (debug only).
        self._capture_prev_state: AEBState = AEBState.STANDBY
        self._prev_ego_speed_capture_ms: float | None = None
        self._last_snapshot: AEBSnapshot | None = None
        self._risk_first_seen: dict[int, float] = {}
        self._radar_visualizer = None
        self._radar_vis_last_vehicle_time: float = -1.0
        self._latched_filter_ego_kmh: float | None = None
        self._sound_handler = _AEBSoundHandler(_AEB_SOUND_PATH)
        self._cal: AEBCalibration = _CAL_DEFAULT
        self._pipeline = build_pipeline(self._cal)
        # Per-target One-Euro state for blended curvature; stepped once per
        # vehicle per frame, pruned at the end of the loop.
        self._curvature_blender = VehicleCurvatureBlender(self._cal)
        # Per-target measured world track (t, vx, vz, ego_x, ego_z) for the
        # line-of-sight-rate engagement veto; time-trimmed on append and
        # pruned against active ids each loop.
        self._los_tracks: dict[int, deque] = {}
        # Follow-threat tracker: per-target (t, dist, abs_speed, d_abs)
        # history plus a hold-until timestamp for ids that pass kinematic
        # qualification (sustained closing + own decel, co-directional).
        # The active set is rebuilt each frame by _update_follow_threats
        # (hold + geometric gate) and read by the pipeline (TmpRelSpeedFilter
        # bypass, evasion/diverge exemption) and the precompute prefilter.
        self._follow_tracks: dict[int, deque] = {}
        self._follow_hold_until: dict[int, float] = {}
        self._follow_threat_ids: set[int] = set()
        # Continuous-decel state
        self._engaged: bool = False
        # Monotonic time when the current uninterrupted engagement
        # qualification streak began; None when not qualifying or engaged.
        # Backs the aeb_engage_confirm_s tier of the entry certainty gate.
        self._engage_qual_since: float | None = None
        # Same for the raw warn condition; backs the oblique warn
        # persistence gate (aeb_warn_confirm_oblique_s).
        self._warn_qual_since: float | None = None
        self._published_target_ms2: float = 0.0
        self._last_target_change_mono: float = 0.0
        self._prev_loop_mono: float | None = None
        # Vehicle ids latched by a confirmed engagement event. Held across
        # frames so a TMP target that drops below the rel-speed pre-filter
        # (because ego is now matching its speed) keeps flowing through the
        # pipeline, and so engagement holds until the gap has actually
        # re-opened: not just until v_closing^2 collapses to zero.
        self._latched_threat_ids: set[int] = set()
        # Replay seams (headless clip eval). Default to live behaviour: the
        # loop's single clock read and the enable flag. The clip-eval driver
        # overrides these plus the three read-seam methods and the sound handler
        # to re-run the pipeline deterministically over recorded frames.
        self._now = time.monotonic
        self._aeb_active_fn = None

    def _read_user_braking(self) -> bool:
        try:
            pt = registry.get_thread("main_pedal_thread")
            if pt is None or not pt.is_alive():
                return False
            with pt.data._lock:
                return float(getattr(pt.data, "brakeval", 0.0)) > _USER_BRAKE_LATCH_THRESHOLD
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
        """Rebuild the follow-threat flag set from per-target behavior tracks.

        Two-part test (see ``AEBCalibration`` follow_threat_* comments):
        1) Kinematic qualification — co-directional target sustains closing
           range and own deceleration over the trailing window; starts a hold.
        2) Geometric gate — while the hold is active, the target must be in
           Lane.EGO or laterally converging (arc-projected d_abs shrinking).
        """
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
        """Return the current ACC primary lead vehicle id, or None.

        Visualizer-only consumer: never raises so a missing/dead ACC
        thread cannot impact AEB safety logic.
        """
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
        """Read the live max brake capacity from sending_thread.

        Falls back to ``_FULL_BRAKE_DECEL`` when the sending thread is
        unavailable or has not yet published a valid estimate.
        """
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

    def _read_pedals_for_capture(self) -> tuple[float, float]:
        """(brakeval, gasval) floats for the clip's consumed context. Never raises."""
        try:
            pt = registry.get_thread("main_pedal_thread")
            if pt is None or not pt.is_alive():
                return 0.0, 0.0
            with pt.data._lock:
                return (
                    float(getattr(pt.data, "brakeval", 0.0)),
                    float(getattr(pt.data, "gasval", 0.0)),
                )
        except (KeyError, AttributeError):
            return 0.0, 0.0

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
        """Record one AEB tick + fire the engagement trigger (debug capture, guarded).

        Never raises into the AEB loop: capture must not touch the safety path.
        """
        recorder = get_recorder()
        if recorder is None:
            return
        try:
            brakeval, gasval = self._read_pedals_for_capture()
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

            # Boundary-negative sampler: AEB stayed silent this tick but a filter
            # rejected a real candidate. The recorder throttles these hard and
            # auto-tags them true negatives; fired every qualifying tick, it
            # simply lands one clip per cooldown while such traffic is around.
            # Skip while the user is braking: filtered convoy traffic during a
            # manual stop is junk TN data (mirrors warn suppression policy).
            if (aeb_active
                    and new_state == AEBState.STANDBY
                    and not self._engaged
                    and brakeval <= _USER_BRAKE_LATCH_THRESHOLD
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

        if paused and self._last_snapshot is not None:
            with self.data._lock:
                self.data.snapshot = self._last_snapshot
            return

        now_mono = self._now()

        # Yaw-rate proxy: see AGENTS.md §1. Do NOT use RadarData.ego_curvature.
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

        stopping_buffer = cal.stop_buffer + ego_half_l

        _ego_fwd_x = -math.sin(ego_yaw_rad)
        _ego_fwd_z = -math.cos(ego_yaw_rad)
        _ego_body_offset = (cal.arc_start_pctg - 0.5) * (2.0 * ego_half_l)
        ego_front_x = ego_x + _ego_body_offset * _ego_fwd_x
        ego_front_z = ego_z + _ego_body_offset * _ego_fwd_z

        ego_arc = build_arc(
            ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
            ego_curvature, ego_hw, dynamic_horizon,
        )

        run_collision = aeb_active

        ego_braked_arc: ArcPath | None = None
        if run_collision:
            ego_braked_arc = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                ego_curvature, ego_hw, dynamic_horizon,
                decel=effective_decel,
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
                left_kappa = left_kappa / 1.5
            right_kappa = ego_curvature - delta_kappa
            if ego_curvature > 0 and right_kappa > 0:
                right_kappa = right_kappa / 1.5
            ego_evasion_left = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                left_kappa, ego_hw, dynamic_horizon,
            )
            ego_evasion_right = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                right_kappa, ego_hw, dynamic_horizon,
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
        # Engagement-eligible aggregates: same chain minus LOS-vetoed targets.
        # Only the engagement-entry decision reads these; warn / display /
        # disarm / holds keep the full aggregates so a veto can never silence
        # an active event or the warning.
        best_ttb_engage: float = _INF
        best_closing_distance_engage: float = _INF
        best_v_closing_engage: float = 0.0
        # Colliding targets with certain geometry (Lane.EGO, aligned heading):
        # these skip the confirm wait at engagement entry. nearcertain covers
        # targets in-lane OR aligned (one classification step from certain):
        # they get the short confirm window instead of the oblique one.
        certain_geom_ids: set[int] = set()
        nearcertain_geom_ids: set[int] = set()
        los_vetoed_ids: set[int] = set()
        los_veto_memo: dict[int, bool] = {}
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
                stopping_buffer=stopping_buffer,
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
                    aligned = abs(ctx.fwd_dot) >= cal.aeb_certain_fwd_dot
                    if ctx.lane == Lane.EGO and aligned:
                        certain_geom_ids.add(v.id)
                    if ctx.lane == Lane.EGO or aligned:
                        nearcertain_geom_ids.add(v.id)
                    newly_risky.add(v.id)
                    if v.id not in self._risk_first_seen:
                        self._risk_first_seen[v.id] = now_mono
                    confirm_duration = (
                        cal.risk_confirm_oncoming_s if head_on else cal.risk_confirm_s
                    )
                    if now_mono - self._risk_first_seen[v.id] < confirm_duration:
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
                    # relative velocity in world frame, not the axial projection
                    # onto ego's heading. The vector form correctly captures
                    # rear-end scenarios where a faster trailing target's
                    # closing rate INCREASES as ego brakes: the axial form
                    # clamps to zero and misses this entirely. Subsumes the
                    # legacy RearOvertakerFilter under one principled mechanism.
                    v_ego_x = ego_speed * ego_fwd_x
                    v_ego_z = ego_speed * ego_fwd_z
                    v_t_x = v.speed * veh_fwd_x
                    v_t_z = v.speed * veh_fwd_z
                    closing_unbraked = math.hypot(v_ego_x - v_t_x, v_ego_z - v_t_z)
                    v_target_along_ego = v_t_x * ego_fwd_x + v_t_z * ego_fwd_z

                    braking_worsens = False
                    if braked_hit is not None:
                        # Target moving faster than ego along ego's heading
                        # axis means braking can only increase relative impact
                        # speed: handles imminent rear-ends where t_braked is
                        # too small for the hysteresis comparison to fire.
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

                    if braking_worsens:
                        braking_worsens_ids.add(v.id)
                        found_hit = True
                        continue

                    ttb = unbraked_ttc

                    if ttb < best_ttb:
                        best_ttb = ttb
                        best_hit_x = unbraked_hit[1]
                        best_hit_z = unbraked_hit[2]
                        # Distance for required_decel = v_rel^2 / 2d: take the
                        # tighter of two frames, since neither is right for
                        # every geometry.
                        # - Relative gap (v_rel * ttc): correct for
                        #   co-directional leads. The world-frame hit distance
                        #   is ego's TRAVEL (~v_ego * ttc); pairing that with
                        #   v_rel underestimates required by ~v_rel/v_ego for
                        #   a moving lead (crash clip 0fe85c88 read 32% of max
                        #   at a 100%-of-capacity rear-end).
                        # - Ego travel to the hit point: tighter for crossing
                        #   or oncoming arcs where v_rel > v_ego; switching
                        #   those to the relative gap drops genuine crossing
                        #   TPs 898e3a46 / f64d2a6b below the engage ratio.
                        # For stationary targets both forms agree.
                        best_closing_distance = min(
                            closing_unbraked * unbraked_ttc,
                            math.hypot(
                                unbraked_hit[1] - ego_front_x,
                                unbraked_hit[2] - ego_front_z,
                            ),
                        )
                        best_v_closing = closing_unbraked

                    # LOS-rate veto (engagement entry only): once per vehicle
                    # per frame, memoized across its arcs.
                    vetoed = los_veto_memo.get(v.id)
                    if vetoed is None:
                        vetoed = False
                        if cal.los_veto_enabled and dist >= cal.los_veto_min_range_m:
                            d_miss = _los_predicted_miss(
                                self._los_tracks.get(v.id, ()), now_mono, cal,
                            )
                            vetoed = (d_miss is not None
                                      and d_miss > cal.los_veto_miss_dist_m)
                        los_veto_memo[v.id] = vetoed
                        if vetoed:
                            los_vetoed_ids.add(v.id)
                    if not vetoed and ttb < best_ttb_engage:
                        best_ttb_engage = ttb
                        # Same tighter-of-two-frames distance as
                        # best_closing_distance above.
                        best_closing_distance_engage = min(
                            closing_unbraked * unbraked_ttc,
                            math.hypot(
                                unbraked_hit[1] - ego_front_x,
                                unbraked_hit[2] - ego_front_z,
                            ),
                        )
                        best_v_closing_engage = closing_unbraked
                    found_hit = True

                vehicle_dicts.append(veh_dict)

        self._risk_first_seen = {
            k: v for k, v in self._risk_first_seen.items() if k in newly_risky
        }
        _active_vids = {v.id for v in vehicles_eff}
        self._curvature_blender.prune(_active_vids)
        for vid in list(self._los_tracks.keys()):
            if vid not in _active_vids:
                del self._los_tracks[vid]
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

        required_decel = 0.0
        if run_collision and best_closing_distance < _INF and best_v_closing > 0.0:
            d_remaining = max(best_closing_distance - cal.stop_buffer, 1e-3)
            required_decel = (best_v_closing * best_v_closing) / (2.0 * d_remaining)

        effective_required = required_decel + downhill_offset

        engage_threshold = cal.aeb_engage_frac * effective_max_decel
        disarm_threshold = cal.aeb_disarm_frac * effective_max_decel
        warn_threshold = cal.aeb_warn_frac * effective_max_decel

        # brake_ttb_active: unbraked geometry says collision is within the
        # emergency window. The window is `brake_ttb + brake_response_window_s`
        # to compensate for actuator lag and the rate-limited brake ramp —
        # without this headroom the slam fires after the pedal has already
        # needed to be at full. Handles path-crossing / arc-cross scenarios
        # where v_closing≈0 collapses required_decel, but the geometry still
        # says ego is about to hit something.
        brake_ttb_active = (
            run_collision
            and time_to_brake < cal.brake_ttb + cal.brake_response_window_s
        )

        # Geometry-driven engagement latch: once engaged, hold engagement while
        # any confirmed collision target's unbraked_ttc is still within warn
        # range. As ego brakes and slows, best_v_closing collapses faster than
        # d_remaining, which would otherwise trip the required_decel disarm
        # threshold mid-event while the impact is still imminent.
        geom_threat_latched = (
            run_collision
            and best_unbraked_ttc < cal.warn_ttb
            and bool(colliding_ids)
        )

        # Distance-based engagement latch over previously-latched targets.
        # required_decel = v_closing^2/2d collapses to zero when ego matches
        # target speed, but the physical gap may still be unsafe. Hold while
        # any latched id has headway < latched_min_headway_s; release the id
        # once its headway grows past latched_release_headway_s.
        active_vid_set = {v.id for v in vehicles_eff}
        ego_v_safe = max(ego_speed, 0.5)
        latched_headway_min = _INF
        for vid in list(self._latched_threat_ids):
            if vid not in active_vid_set:
                self._latched_threat_ids.discard(vid)
                continue
            pc = vehicle_collision_data.get(vid)
            if pc is None:
                self._latched_threat_ids.discard(vid)
                continue
            dist_vid = math.sqrt(pc[5])
            gap = max(dist_vid - cal.stop_buffer, 0.0)
            hw = gap / ego_v_safe
            if hw > cal.latched_release_headway_s:
                self._latched_threat_ids.discard(vid)
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
            d_remaining_e = max(best_closing_distance_engage - cal.stop_buffer, 1e-3)
            required_decel_engage = (
                (best_v_closing_engage * best_v_closing_engage) / (2.0 * d_remaining_e)
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
            self._engage_qual_since = None
        else:
            # New engagements are gated by |ego_speed|: below the threshold
            # the truck is essentially crawling, the user has authority
            ego_kmh_abs = abs(ego_speed) * 3.6
            qualified = (run_collision
                         and ego_kmh_abs >= cal.aeb_min_engage_speed_kmh
                         and (effective_required_engage >= engage_threshold
                              or brake_ttb_engage_active))
            if qualified:
                if self._engage_qual_since is None:
                    self._engage_qual_since = now_mono
                # Certainty tiers that engage without the confirm wait:
                # imminent (full brake barely avoids), certain geometry
                # (aligned in-lane target, engage-eligible), or continuity
                # with a previously latched threat. Everything else: crossing
                # arcs, oblique sweeps: must sustain qualification, which
                # single-tick extrapolation phantoms never do.
                certain = (
                    brake_ttb_engage_active
                    or any(vid in certain_geom_ids and vid not in los_vetoed_ids
                           for vid in colliding_ids)
                    or any(vid in self._latched_threat_ids
                           for vid in colliding_ids)
                )
                # Confirm window graded by geometry: near-certain targets
                # (in-lane or aligned) get the short window; oblique
                # out-of-lane crossers, the extrapolation-fragile class,
                # must sustain qualification longer.
                confirm_window = (
                    cal.aeb_engage_confirm_s
                    if any(vid in nearcertain_geom_ids for vid in colliding_ids)
                    else cal.aeb_engage_confirm_oblique_s
                )
                confirmed = (now_mono - self._engage_qual_since
                             >= confirm_window)
                if certain or confirmed:
                    self._engaged = True
            else:
                self._engage_qual_since = None

        # Promote every currently-colliding target into the latched set so
        # subsequent frames keep them in the pipeline and in the hold check.
        if self._engaged and run_collision and colliding_ids:
            self._latched_threat_ids.update(colliding_ids)

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
        if (abs(delta) < cal.aeb_target_deadband_ms2
                and time_since_change < cal.aeb_target_refresh_min_s):
            target_published = self._published_target_ms2
        else:
            slew_limit = cal.aeb_target_rate_ms3 * dt_loop
            step = max(-slew_limit, min(slew_limit, delta))
            target_published = self._published_target_ms2 + step
            target_published = max(0.0, target_published)
            if abs(target_published - self._published_target_ms2) > 1e-6:
                self._last_target_change_mono = now_mono
        self._published_target_ms2 = target_published

        if run_collision and effective_max_decel > 0.1:
            if brake_ttb_active:
                aeb_ff_decel = effective_max_decel
            elif effective_required > 0.0:
                aeb_ff_decel = min(effective_required, effective_max_decel)
            else:
                aeb_ff_decel = 0.0
        else:
            aeb_ff_decel = 0.0

        warn_by_decel = (
            run_collision and effective_required >= warn_threshold
        )
        warn_by_ttb = (run_collision and time_to_brake < cal.warn_ttb)
        warn_raw = bool(warn_by_decel or warn_by_ttb)

        # Warn persistence gate: mirrors the engagement certainty tiers.
        # Certain/near-certain geometry, imminent TTB, latched threats, and
        # active engagement warn instantly; oblique out-of-lane threats must
        # sustain the warn condition for aeb_warn_confirm_oblique_s, which
        # filters the transient phantom beeps of that class while keeping
        # >= 0.1 s of warning ahead of an oblique engagement (whose confirm
        # is aeb_engage_confirm_oblique_s; warn qualification is strictly
        # looser than engage qualification, so the streak starts no later).
        # The streak tracks the raw threat condition so the user-braking
        # display suppression below never resets it.
        if warn_raw:
            if self._warn_qual_since is None:
                self._warn_qual_since = now_mono
            warn_instant = (
                self._engaged
                or brake_ttb_active
                or any(vid in nearcertain_geom_ids for vid in colliding_ids)
                or any(vid in self._latched_threat_ids for vid in colliding_ids)
            )
            aeb_warn = (
                warn_instant
                or (now_mono - self._warn_qual_since
                    >= cal.aeb_warn_confirm_oblique_s)
            )
        else:
            self._warn_qual_since = None
            aeb_warn = False

        user_braking_now = self._read_user_braking()
        near_full_target = (
            effective_max_decel > 0.1
            and effective_required >= cal.aeb_warn_near_full_frac * effective_max_decel
        )
        if user_braking_now and not near_full_target:
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
                aeb_warn = True
            elif new_state == AEBState.WARN:
                aeb_warn = True
        if new_state != self._prev_state:
            self._state_hold_until = now_mono + 0.3

        self._prev_state = new_state

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

    def teardown(self) -> None:
        self._sound_handler.cleanup()
        if self._radar_visualizer is not None:
            try:
                self._radar_visualizer.stop()
            except Exception:
                pass
        self._latched_filter_ego_kmh = None
        self._engaged = False
        self._engage_qual_since = None
        self._warn_qual_since = None
        self._published_target_ms2 = 0.0
        self._last_target_change_mono = 0.0
        self._prev_loop_mono = None
        self._latched_threat_ids.clear()
        self._follow_tracks.clear()
        self._follow_hold_until.clear()
        self._follow_threat_ids = set()
        self._capture_prev_state = AEBState.STANDBY
        self._prev_ego_speed_capture_ms = None
        self._curvature_blender.prune(set())
        self._los_tracks.clear()
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
        """Read the radar thread's published snapshot under its data lock.

        Returns ``None`` when the radar thread is missing / not alive: AEB
        then skips the loop rather than fabricating an ego pose.
        """
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

