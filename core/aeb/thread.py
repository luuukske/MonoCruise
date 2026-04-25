"""
AEB Thread — Automatic Emergency Braking with arc-based collision detection.

TTB-based detection — see ``core/aeb/AGENTS.md`` §9 for full logic description.

Registry name: ``aeb_thread``
"""

from __future__ import annotations

import enum
import logging
import math
import threading
import time
from dataclasses import dataclass, field

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from core.radar.traffic import (
    Vehicle,
    ArcPath, build_arc, arc_arc_collision, _accel_to_arc_params,
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

_FULL_BRAKE_DECEL: float = 7.8
# Fraction of max brake capacity assumed for ego stopping / TTB calculations.
# The brake system physically commands only this fraction, reserving headroom
# so a sudden increase in closing speed can still stop the vehicle.
_AEB_EGO_DECEL_FRAC: float = 0.9
_MAX_RANGE: float = 200.0
_MAX_RANGE_SQ: float = _MAX_RANGE ** 2
# TMP-only: |v_ego − v_target| (km/h) vs latched ref ego speed — see _latched_filter_ego_kmh.
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
# fwd_dot threshold for lateral-gap activation — deliberately looser than the
# head_on threshold (-0.7) to catch oncoming vehicles that never reach -0.7
# during a shared turn.  Does NOT affect target decel model, evasion filter
# bypass, or risk confirm duration — those all still use head_on (-0.7).
_NEAR_HEAD_ON_DOT: float = -0.5
# Minimum target curvature (1/m) to apply the turning-diverge suppression.
# 0.03 ≈ 33 m radius — tight enough to be a real corner, loose enough to
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
# measurement — a vehicle genuinely a full lane away may read as <2.0 m. Use a
# lower threshold when same-curve geometry is confirmed by v_curvature sign + magnitude.
_SAME_CURVE_OWN_LANE_LAT: float = 1.0
_CO_DIR_DIVERGE_LOOKAHEAD_S: float = 0.25
# Fix C — extended lookahead for co-directional same-turn outer-lane suppression.
# Inner/outer lane arcs overlap before their centerlines cross; 0.25 s is too short
# to see the divergence. At horizon × this scale the paths have clearly separated.
_CO_SAME_TURN_LOOKAHEAD_SCALE: float = 0.5
# Sweep-pass suppression — stationary cross-traffic ego turns through.
_SWEEP_PASS_MAX_TARGET_SPEED: float = 1.0    # m/s

# Intersection / shared-turn false-positive suppression
# Fix A — Ghost-arc scaling for near-head-on vehicles clearly in their own lane.
# cross_zone_padding peaks at sin(angle)≈0.8, producing ±4 m ghost arcs at 10 m/s,
# which phantom-widen the target corridor and prevent the ego evasion filter from
# clearing. Only fires when target is laterally displaced into its own lane.
_NEAR_HEAD_ON_CROSS_SCALE: float = 0.3       # ghost-arc reduction factor
_NEAR_HEAD_ON_LATERAL_MIN: float = 3.0       # m — minimum lateral offset to activate Fix A

# Fix B — Road-following curvature expansion for oncoming vehicles in shared turns.
# Expands delta_kappa_t so the oncoming evasion filter tests whether "target follows
# the same corner road as ego" — not just a tiny ±0.006 1/m perturbation.
# Still evaluated via arc_arc_collision; not a blind suppression.
_SHARED_TURN_MAX_KAPPA: float = 0.05         # cap on road-following curvature (R ≥ 20 m)

# Fix D — target arc over-rotation suppression.
# A vehicle turning from a side road into the opposite lane maintains high curvature;
# the constant-curvature arc keeps rotating past lane alignment into ego's lane.
# Dampen target curvature when heading rotation over the arc horizon would exceed
# the angle to anti-parallel road alignment.
_TURN_COMPLETE_CURVATURE_SCALE: float = 3.0   # divisor applied when overshoot detected

def _tmp_collision_threat(ref_ego_kmh: float, rel_speed_kmh: float) -> bool:
    """TMP session only — True if target should participate in arc collision / TTB."""
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
    braking_suppressed_ids: set = field(default_factory=set)
    evasion_filtered_ids: set = field(default_factory=set)
    oncoming_evasion_filtered_ids: set = field(default_factory=set)

    aeb_state: AEBState = AEBState.STANDBY
    time_to_collision: float = _INF
    time_to_brake: float = _INF
    hit_x: float = 0.0
    hit_z: float = 0.0

    evasion_left_arc: ArcPath | None = None
    evasion_right_arc: ArcPath | None = None


@dataclass
class AEBData(ThreadData):
    AEB_warn: bool = False
    AEB_brake: bool = False
    time_to_brake: float = _INF
    em_stop_requested: bool = False
    snapshot: AEBSnapshot = field(default_factory=AEBSnapshot)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


def _cross_zone_padding(ego_yaw_rad: float, v_yaw_rad: float, v_speed_ms: float) -> float:
    """Perpendicular-target ghost-arc padding (peaks at 90° yaw diff)."""
    cross_factor = abs(math.sin(ego_yaw_rad - v_yaw_rad))
    return cross_factor * (_CROSS_SAFE_ZONE_BASE + _CROSS_SAFE_ZONE_SPEED * v_speed_ms)


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


def _dampen_turning_curvature(
    v_curvature: float,
    fwd_dot: float,
    ego_fwd_x: float, ego_fwd_z: float,
    veh_fwd_x: float, veh_fwd_z: float,
    abs_v_speed: float,
    arc_length: float,
) -> float:
    """Dampen target curvature when arc would over-rotate past anti-parallel lane alignment.

    Mirrors the ego evasion centerline-snap but on the primary target arc. A vehicle
    turning from a side road into the opposite lane has high curvature; constant-curvature
    propagation keeps rotating past the point where the vehicle straightens into its lane,
    producing a phantom collision in ego's lane.

    Only fires for cross-traffic geometry (fwd_dot in (-0.5, 0.7)) with confirmed rotation
    toward anti-parallel and a heading change that would exceed the alignment angle.
    """
    if (abs(v_curvature) <= _TURNING_DIVERGE_CURVATURE
            or abs_v_speed <= 0.5
            or fwd_dot <= -0.5    # already mostly anti-parallel — not mid-turn entry
            or fwd_dot >= 0.7):   # co-directional — other suppressions handle this
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
        return v_curvature  # rotating away — genuine cross-arc threat
    return v_curvature / _TURN_COMPLETE_CURVATURE_SCALE


def _build_vehicle_collision_data(
    v: Vehicle,
    dynamic_horizon: float,
    ego_yaw_rad: float,
    ego_fwd_x: float,
    ego_fwd_z: float,
) -> tuple[list[ArcPath], float, list[list[ArcPath]],
           float, float, float, float, float]:
    """Build collision arcs and derived vehicle geometry for a vehicle.

    Returns (all_target_arcs, cross_padding, cross_arcs_list,
             v_yaw_rad, abs_v_speed, veh_fwd_x, veh_fwd_z, v_curvature).
    """
    v_hw = v.size.width / 2.0
    v_hw_coll = max(v_hw - 0.1, 0.3)
    abs_v_speed = abs(v.speed)
    _vk = v.curvature_from_history()
    v_curvature = _vk if _vk is not None else (
        math.radians(v.angular_velocity) / abs_v_speed if abs_v_speed > 0.5 else 0.0
    )
    _, v_yaw_deg, _ = v.rotation.euler()
    v_yaw_rad = math.radians(v_yaw_deg)
    veh_fwd_x = -math.sin(v_yaw_rad)
    veh_fwd_z = -math.cos(v_yaw_rad)
    fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
    head_on = fwd_dot < -0.5
    target_override_decel = _FULL_BRAKE_DECEL if head_on else 0.0
    # Fix D — dampen curvature when constant-curvature arc would over-rotate past
    # anti-parallel lane alignment. v_curvature is preserved unchanged for same_curve
    # checks; arc_curvature is used only for arc building.
    arc_curvature = _dampen_turning_curvature(
        v_curvature, fwd_dot,
        ego_fwd_x, ego_fwd_z, veh_fwd_x, veh_fwd_z,
        abs_v_speed, abs_v_speed * dynamic_horizon,
    )
    # For trailer arcs built with build_arc() directly. get_arc() calls
    # _accel_to_arc_params internally so veh_arc_coll only needs override_decel.
    target_decel, target_accel = _accel_to_arc_params(v.accel_for_arc(), target_override_decel)
    veh_arc_coll = v.get_arc(
        dynamic_horizon,
        half_width=v_hw_coll,
        decel=target_override_decel,
        arc_start_pctg=_ARC_START_PCTG,
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
            (1.0 - _ARC_START_PCTG) if tr_is_rev_c else _ARC_START_PCTG
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
    cross_padding = _cross_zone_padding(ego_yaw_rad, v_yaw_rad, abs_v_speed)
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
            logger.warning("pygame not available — AEB sound disabled")
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=256)
                pygame.mixer.init()
            self._sound = pygame.mixer.Sound(sound_file_path)
            self._sound.set_volume(0.8)
        except Exception as exc:
            logger.warning("AEB sound init failed (%s) — sound disabled", exc)
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
                    "AEB sound: stop requested — %d extra replay(s) then finishing",
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
                        logger.debug("AEB sound: extra replays done — letting current sound finish")
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
        self._last_snapshot: AEBSnapshot | None = None
        self._risk_first_seen: dict[int, float] = {}
        self._radar_visualizer = None
        # Frozen ref ego (km/h) for TMP rel-speed split while WARN/brake-pedal active.
        self._latched_filter_ego_kmh: float | None = None
        self._sound_handler = _AEBSoundHandler(_AEB_SOUND_PATH)

    def _read_user_braking(self) -> bool:
        try:
            pt = registry.get_thread("main_pedal_thread")
            if pt is None or not pt.is_alive():
                return False
            with pt.data._lock:
                return float(getattr(pt.data, "brakeval", 0.0)) > _USER_BRAKE_LATCH_THRESHOLD
        except (KeyError, AttributeError):
            return False

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
        return _FULL_BRAKE_DECEL

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

        aeb_active = Settings.AEB_enabled

        snapshot = self._read_radar_snapshot()
        if snapshot is None:
            return
        (vehicles, ego_x, ego_y, ego_z, ego_yaw_rad, ego_speed, ego_pitch_deg,
         steer, ego_has_trailer, _ego_curvature_from_history, tmp_traffic_session,
         paused) = snapshot

        if paused and self._last_snapshot is not None:
            with self.data._lock:
                self.data.snapshot = self._last_snapshot
            return

        now_mono = time.monotonic()

        # AEB ego path — yaw-rate proxy only. Do NOT use the position-history
        # fit published by RadarThread: AEB's ego arc must not be smoothed or
        # lagged, and the proxy reacts instantly to steering input. ACC owns
        # the history-based curvature (see core/radar/AGENTS.md §11).
        if ego_speed > 0.5:
            yaw_rate_rad_s = math.radians(steer * ego_speed * 12.0)
            ego_curvature = yaw_rate_rad_s / ego_speed
        else:
            ego_curvature = 0.0

        ego_hw: float = 1.15
        ego_half_l: float = 3.0

        # Effective ego decel for TTB calculations: 90 % of the live max brake
        # capacity so the trigger fires early enough for the phased brake
        # controller to stop the vehicle even if the threat brakes harder.
        # _FULL_BRAKE_DECEL is still used for modelling the *target's* braking.
        _max_brake_live = self._read_max_brake_ms2()
        effective_decel = _AEB_EGO_DECEL_FRAC * _max_brake_live

        t_stop = ego_speed / effective_decel
        dynamic_horizon = min(max(_MIN_ARC_HORIZON, t_stop * 2.0), _MAX_ARC_HORIZON)

        stopping_buffer = _STOP_BUFFER_FIXED + ego_half_l

        _ego_fwd_x = -math.sin(ego_yaw_rad)
        _ego_fwd_z = -math.cos(ego_yaw_rad)
        _ego_body_offset = (_ARC_START_PCTG - 0.5) * (2.0 * ego_half_l)
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
                _EVASION_G_THRESHOLD / (ego_speed * ego_speed),
                _EVASION_FILTER_MAX_DELTA_KAPPA,
            )
            # Snap to center: when path would cross center line, cap curvature at 0
            # Left path: when ego turns right, left path can cross center → snap to center (curvature 0)
            left_kappa = ego_curvature + delta_kappa
            if ego_curvature < 0 and left_kappa < 0:
                left_kappa = left_kappa/1.5
            # Right path: when ego turns left, right path can cross center → snap to center (curvature 0)
            right_kappa = ego_curvature - delta_kappa
            if ego_curvature > 0 and right_kappa > 0:
                right_kappa = right_kappa/1.5
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
            # Push per-vehicle raw/filtered speed and filtered acceleration to the UI.
            for v in vehicles:
                f_spd, f_acc, r_spd = v.radar_speed_accel()
                self._radar_visualizer.push_data(v.id, f_spd, f_acc, r_spd)

        colliding_ids: set[int] = set()
        suppressed_ids: set[int] = set()
        braking_suppressed_ids: set[int] = set()
        evasion_filtered_ids: set[int] = set()
        oncoming_evasion_filtered_ids: set[int] = set()
        best_ttb: float = _INF
        best_unbraked_ttc: float = _INF
        best_raw_dist: float = _INF
        best_hit_x: float = 0.0
        best_hit_z: float = 0.0
        vehicle_dicts: list[dict] = []
        vehicle_arcs: dict[int, list[ArcPath]] = {}
        newly_risky: set[int] = set()

        ego_pitch_rad = math.radians(ego_pitch_deg)

        # Precompute per-vehicle collision arcs + derived geometry for the main loop.
        # Stores (all_target_arcs, cross_padding, cross_arcs_list,
        #         dx, dz, dist_sq,
        #         v_yaw_rad, abs_v_speed, veh_fwd_x, veh_fwd_z, v_curvature)
        vehicle_collision_data: dict[int, tuple] = {}
        if run_collision:
            for v in vehicles:
                vx, vz = v.position.x, v.position.z
                dx = vx - ego_x
                dz = vz - ego_z
                dist_sq = dx * dx + dz * dz
                if dist_sq > _MAX_RANGE_SQ:
                    continue
                rz = _world_to_ego_forward(dx, dz, ego_yaw_rad)
                expected_y = ego_y + rz * math.tan(ego_pitch_rad)
                if abs(v.position.y - expected_y) > _ELEVATION_MARGIN_M:
                    continue
                if tmp_traffic_session:
                    _, v_yaw_deg_pc, _ = v.rotation.euler()
                    v_yaw_rad_pc = math.radians(v_yaw_deg_pc)
                    vf_x = -math.sin(v_yaw_rad_pc)
                    vf_z = -math.cos(v_yaw_rad_pc)
                    dvx_pc = ego_speed * ego_fwd_x - v.speed * vf_x
                    dvz_pc = ego_speed * ego_fwd_z - v.speed * vf_z
                    rel_kmh_pc = 3.6 * math.hypot(dvx_pc, dvz_pc)
                    if not _tmp_collision_threat(ref_kmh_for_filter, rel_kmh_pc):
                        continue
                (all_t, cross_pad, cross_list,
                 pc_yaw, pc_aspd, pc_fx, pc_fz, pc_curv,
                 ) = _build_vehicle_collision_data(
                    v, dynamic_horizon, ego_yaw_rad, ego_fwd_x, ego_fwd_z
                )
                vehicle_collision_data[v.id] = (
                    all_t, cross_pad, cross_list,
                    dx, dz, dist_sq,
                    pc_yaw, pc_aspd, pc_fx, pc_fz, pc_curv,
                )

        for v in vehicles:
            vx, vz = v.position.x, v.position.z

            # Reuse precomputed data when available (skips range/elevation/TMP checks
            # and derived value recomputation — already done in the precompute pass).
            pc = vehicle_collision_data.get(v.id)
            if pc is not None:
                (all_target_arcs, cross_padding, precomputed_cross_arcs,
                 dx, dz, dist_sq,
                 v_yaw_rad, abs_v_speed, veh_fwd_x, veh_fwd_z, v_curvature) = pc
                dist = math.sqrt(dist_sq)
                v_hw = v.size.width / 2.0
                v_hw_coll = max(v_hw - 0.1, 0.3)
            else:
                dx = vx - ego_x
                dz = vz - ego_z
                dist_sq = dx * dx + dz * dz
                if dist_sq > _MAX_RANGE_SQ:
                    continue

                # Elevation filter (slope-aware) — see AGENTS.md §13
                rz = _world_to_ego_forward(dx, dz, ego_yaw_rad)
                expected_y = ego_y + rz * math.tan(ego_pitch_rad)
                if abs(v.position.y - expected_y) > _ELEVATION_MARGIN_M:
                    continue

                dist = math.sqrt(dist_sq)
                _, v_yaw_deg, _ = v.rotation.euler()
                v_yaw_rad = math.radians(v_yaw_deg)
                v_hw = v.size.width / 2.0
                v_hw_coll = max(v_hw - 0.1, 0.3)

                abs_v_speed = abs(v.speed)
                _vk = v.curvature_from_history()
                v_curvature = _vk if _vk is not None else (
                    math.radians(v.angular_velocity) / abs_v_speed if abs_v_speed > 0.5 else 0.0
                )
                veh_fwd_x = -math.sin(v_yaw_rad)
                veh_fwd_z = -math.cos(v_yaw_rad)
                precomputed_cross_arcs = None

            veh_arc = v.get_arc(dynamic_horizon, arc_start_pctg=_ARC_START_PCTG)
            trailer_dicts = []
            trailer_arcs: list[ArcPath] = []
            tr_hw_colls: list[float] = []
            for tr in v.trailers:
                tr_arc_pos = tr.position
                tr_dict_pos = tr.correct_position() if tr.is_tmp else tr.position
                _, tr_yaw_deg, _ = tr.rotation.euler()
                tr_yaw_rad = math.radians(tr_yaw_deg)
                tr_hw = tr.size.width / 2.0
                tr_hw_coll = max(tr_hw - 0.1, 0.3)
                tr_hw_colls.append(tr_hw_coll)

                tr_is_rev = v.speed < -1e-3
                tr_effective_p = (1.0 - _ARC_START_PCTG) if tr_is_rev else _ARC_START_PCTG
                tr_fwd_x_l = -math.sin(tr_yaw_rad)
                tr_fwd_z_l = -math.cos(tr_yaw_rad)
                tr_body_offset = (tr_effective_p - 0.5) * tr.size.length
                tr_arc = build_arc(
                    tr_arc_pos.x + tr_body_offset * tr_fwd_x_l,
                    tr_arc_pos.z + tr_body_offset * tr_fwd_z_l,
                    tr_yaw_rad,
                    v.speed, v_curvature, tr_hw, dynamic_horizon,
                )
                trailer_arcs.append(tr_arc)

                trailer_dicts.append({
                    "x": tr_dict_pos.x, "z": tr_dict_pos.z,
                    "yaw": tr_yaw_rad,
                    "half_w": tr_hw,
                    "length": tr.size.length,
                    "is_tmp": tr.is_tmp,
                })

            veh_dict = {
                "vid": v.id,
                "x": vx, "z": vz,
                "yaw": v_yaw_rad,
                "half_w": v_hw,
                "length": v.size.length,
                "is_tmp": v.is_tmp,
                "speed_kmh": abs(v.speed) * 3.6,
                "rear_suppressed": False,
                "trailers": trailer_dicts,
            }

            vehicle_arcs[v.id] = [veh_arc] + trailer_arcs

            if not run_collision:
                vehicle_dicts.append(veh_dict)
                continue

            # Rear-approach / overtaker suppression (veh_fwd_x/z already computed)
            to_veh_len = max(dist, 1e-6)
            dot_fwd = (dx * ego_fwd_x + dz * ego_fwd_z) / to_veh_len
            if dot_fwd < _REAR_DOT_THRESHOLD:
                approach_dot = veh_fwd_x * ego_fwd_x + veh_fwd_z * ego_fwd_z
                if approach_dot > 0.5 and v.speed > ego_speed + _OVERTAKE_SPEED_MARGIN:
                    veh_dict["rear_suppressed"] = True
                    suppressed_ids.add(v.id)
                    vehicle_dicts.append(veh_dict)
                    continue

            # TMP threat check — skip when precomputed (already passed in precompute phase)
            if pc is None and tmp_traffic_session:
                dvx = ego_speed * ego_fwd_x - v.speed * veh_fwd_x
                dvz = ego_speed * ego_fwd_z - v.speed * veh_fwd_z
                rel_kmh = 3.6 * math.hypot(dvx, dvz)
                if not _tmp_collision_threat(ref_kmh_for_filter, rel_kmh):
                    vehicle_dicts.append(veh_dict)
                    continue

            fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
            co_directional = fwd_dot > 0.7
            head_on = fwd_dot < -0.7

            near_head_on = fwd_dot < _NEAR_HEAD_ON_DOT

            # Lateral separation from ego's forward axis in the ego plane.
            lateral_offset = abs(dx * ego_fwd_z - dz * ego_fwd_x)

            # Get collision arcs — already extracted from precomputed, or build new
            if pc is None:
                target_override_decel = _FULL_BRAKE_DECEL if head_on else 0.0
                target_decel, target_accel = _accel_to_arc_params(v.accel_for_arc(), target_override_decel)
                arc_curvature_fb = _dampen_turning_curvature(
                    v_curvature, fwd_dot,
                    ego_fwd_x, ego_fwd_z, veh_fwd_x, veh_fwd_z,
                    abs_v_speed, abs_v_speed * dynamic_horizon,
                )
                veh_arc_coll = v.get_arc(dynamic_horizon, half_width=v_hw_coll,
                                         decel=target_override_decel, arc_start_pctg=_ARC_START_PCTG,
                                         curvature_override=arc_curvature_fb)
                trailer_arcs_coll: list[ArcPath] = []
                for idx, tr in enumerate(v.trailers):
                    tr_pos = tr.position
                    _, tr_yaw_deg, _ = tr.rotation.euler()
                    tr_yaw_rad = math.radians(tr_yaw_deg)
                    tr_is_rev_c = v.speed < -1e-3
                    tr_effective_p_c = (1.0 - _ARC_START_PCTG) if tr_is_rev_c else _ARC_START_PCTG
                    tr_fwd_x_c = -math.sin(tr_yaw_rad)
                    tr_fwd_z_c = -math.cos(tr_yaw_rad)
                    tr_body_offset_c = (tr_effective_p_c - 0.5) * tr.size.length
                    trailer_arcs_coll.append(build_arc(
                        tr_pos.x + tr_body_offset_c * tr_fwd_x_c,
                        tr_pos.z + tr_body_offset_c * tr_fwd_z_c,
                        tr_yaw_rad,
                        v.speed, arc_curvature_fb, tr_hw_colls[idx], dynamic_horizon,
                        decel=target_decel, accel=target_accel,
                    ))
                all_target_arcs = [veh_arc_coll] + trailer_arcs_coll
                cross_padding = _cross_zone_padding(ego_yaw_rad, v_yaw_rad, abs_v_speed)

            # Fix A — reduce ghost-arc padding for near-head-on vehicles clearly in their
            # own lane. cross_zone_padding peaks at sin(angle)≈0.8 for near-head-on
            # geometry, producing ghost arcs ±4 m wide at 10 m/s. This phantom-widens the
            # target corridor so the ego evasion filter always sees a hit even when the
            # vehicle is safely displaced into its own lane at an intersection approach.
            # The scale-down only fires when lateral_offset confirms own-lane placement.
            effective_cross_padding = cross_padding
            fix_a_active = False
            if near_head_on and lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN:
                effective_cross_padding *= _NEAR_HEAD_ON_CROSS_SCALE
                fix_a_active = True

            for arc_idx, base_target_arc in enumerate(all_target_arcs):
                # Reuse precomputed cross_arcs when Fix A didn't change the padding
                if precomputed_cross_arcs is not None and not fix_a_active:
                    cross_arcs = precomputed_cross_arcs[arc_idx]
                else:
                    cross_arcs = _apply_cross_zone(base_target_arc, effective_cross_padding)

                lateral_gap = _LATERAL_LANE_SEPARATION if near_head_on else 0.0

                unbraked_hit = _earliest_hit(
                    ego_arc, cross_arcs, _CORRIDOR_MARGIN, _COLLISION_SAMPLES, lateral_gap,
                )

                # Suppress diverging co-directional moving targets only.
                # Fix C — for a co-directional vehicle in the outer lane of the same
                # corner as ego (same-sign curvature, lateral displacement confirmed,
                # both in a real corner), the inner/outer arc corridors overlap well
                # before the centerlines actually cross.  At the standard 0.25 s
                # lookahead the paths are still converging toward that crossing point,
                # so the suppression does not fire.  Extending the lookahead to
                # horizon × _CO_SAME_TURN_LOOKAHEAD_SCALE gives enough time for the
                # paths to have clearly separated post-crossing.
                # Guards are intentionally strict:
                #   - lateral_offset: outer lane confirmed, not lane-sharing
                #   - both curvatures above threshold: real corner, not straight drift
                #   - same curvature sign: both turning the same direction
                if (unbraked_hit is not None
                        and co_directional
                        and base_target_arc.speed > 0.5):
                    co_diverge_dt = _CO_DIR_DIVERGE_LOOKAHEAD_S
                    g_lat = lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN
                    g_ego_k = abs(ego_curvature) >= _TURNING_DIVERGE_CURVATURE
                    g_veh_k = abs(v_curvature) >= _TURNING_DIVERGE_CURVATURE
                    g_sign = ego_curvature * v_curvature > 0
                    fix_c_active = g_lat and g_ego_k and g_veh_k and g_sign
                    if fix_c_active:
                        co_diverge_dt = dynamic_horizon * _CO_SAME_TURN_LOOKAHEAD_SCALE
                    suppressed = not _is_approaching(
                        ego_arc, base_target_arc,
                        unbraked_hit[0], dt=co_diverge_dt)
                    if suppressed:
                        unbraked_hit = None

                # Suppress a tightly-turning cross-traffic vehicle whose arc is
                # already diverging at the hit point.  Guards are intentionally
                # strict to avoid masking real threats:
                #   - not head_on: never suppress an oncoming vehicle
                #   - not co_directional: original branch already handles that
                #   - curvature guard: target must be in a real corner, not a
                #     gentle curve that could still converge
                #   - speed guard: stationary / near-stationary targets are not
                #     "turning away" in a meaningful sense
                if (unbraked_hit is not None
                        and not head_on
                        and not co_directional
                        and base_target_arc.speed > 0.5):
                    g_veh_k_ct = abs(base_target_arc.curvature) > _TURNING_DIVERGE_CURVATURE
                    approaching_ct = _is_approaching(ego_arc, base_target_arc, unbraked_hit[0])
                    suppressed_ct = g_veh_k_ct and not approaching_ct
                    if suppressed_ct:
                        unbraked_hit = None

                # Sweep-pass: stationary cross-traffic ego turns through.
                # Guards: target near-stationary, ego in a real corner.
                # At t_hit, ego's heading has rotated past the vehicle — not a real collision.
                if (unbraked_hit is not None
                        and abs_v_speed < _SWEEP_PASS_MAX_TARGET_SPEED
                        and abs(ego_curvature) > _TURNING_DIVERGE_CURVATURE):
                    _sp_dist = ego_arc._dist_at_time(unbraked_hit[0])
                    _sp_ex, _sp_ez = ego_arc.position_at_dist(_sp_dist)
                    _sp_yaw = ego_arc.heading_at_dist(_sp_dist)
                    _sp_fwd_x = -math.sin(_sp_yaw)
                    _sp_fwd_z = -math.cos(_sp_yaw)
                    if (vx - _sp_ex) * _sp_fwd_x + (vz - _sp_ez) * _sp_fwd_z <= 0.0:
                        unbraked_hit = None

                # Corner-entry stationary oncoming suppression.
                # At corner entry ego_curvature ≈ 0, so all κ-gated suppressions
                # fail. A stationary oncoming vehicle's yaw encodes how much the
                # road curves between ego and the vehicle: the signed yaw difference
                # from anti-parallel equals the road bend angle, so
                #   implied_kappa = acos(-fwd_dot) / dist
                # gives the average curvature of the road ahead. If this exceeds
                # the corner threshold the vehicle is on an upcoming curve.
                # lateral_offset guards against a stationary vehicle in *ego's* lane
                # on the same curve — that vehicle has near-zero lateral displacement
                # from ego's current heading axis (it's straight ahead, not to the
                # side), so the gate correctly fails and the threat is preserved.
                if (unbraked_hit is not None
                        and abs_v_speed < _SWEEP_PASS_MAX_TARGET_SPEED
                        and fwd_dot < -0.3
                        and abs(ego_curvature) < _TURNING_DIVERGE_CURVATURE
                        and lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN
                        and dist > 1.0):
                    road_bend = math.acos(max(-1.0, min(1.0, -fwd_dot)))
                    implied_kappa = road_bend / dist
                    if implied_kappa > _TURNING_DIVERGE_CURVATURE:
                        oncoming_evasion_filtered_ids.add(v.id)
                        unbraked_hit = None

                if unbraked_hit is None:
                    continue

                unbraked_ttc = unbraked_hit[0]

                # Evasion filter — bypassed for co-directional moving and head-on
                if (ego_evasion_left is not None
                        and ego_evasion_right is not None
                        and not head_on
                        and not (
                        co_directional
                        and base_target_arc.speed > 0.5
                        and lateral_offset
                        <= (ego_hw + base_target_arc.half_width + 0.25)
                        )):
                    left_hit = _earliest_hit(
                        ego_evasion_left, cross_arcs,
                        _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                    )
                    right_hit = _earliest_hit(
                        ego_evasion_right, cross_arcs,
                        _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                    )
                    # Only filter if at least one evasion path misses this target
                    # (evasion paths are checked for collision with this target only)
                    left_clear = left_hit is None
                    right_clear = right_hit is None
                    if left_clear or right_clear:
                        evasion_filtered_ids.add(v.id)
                        continue

                # Oncoming evasion filter — mirrors the ego evasion filter but asks
                # whether the *oncoming vehicle* could steer around ego instead.
                elif (head_on and abs_v_speed > 1.0):
                    delta_kappa_t = min(
                        _EVASION_G_THRESHOLD_ONCOMING / (abs_v_speed * abs_v_speed),
                        _EVASION_FILTER_MAX_DELTA_KAPPA,
                    )
                    # Scale delta_kappa_t when vehicle is clearly in its own lane —
                    # a vehicle already displaced laterally needs less curvature to
                    # miss ego, so we give its evasion arcs more room to work with.
                    # lateral_offset is already computed above (same formula).
                    # On tight curves, ego's heading axis compresses the cross-product
                    # lateral offset. Use a lower threshold when both vehicles are
                    # clearly on the same curved road (same-sign curvature above
                    # threshold) — a genuinely in-lane head-on vehicle would be <1 m.
                    same_curve = (
                        abs(v_curvature) >= _TURNING_DIVERGE_CURVATURE
                        and ego_curvature * v_curvature > 0
                    )
                    lane_threshold = _SAME_CURVE_OWN_LANE_LAT if same_curve else _OPPOSITE_LANE_OFFSET
                    own_lane = lateral_offset >= lane_threshold
                    if own_lane:
                        delta_kappa_t = min(
                            delta_kappa_t * _OPPOSITE_LANE_KAPPA_SCALE,
                            _EVASION_FILTER_MAX_DELTA_KAPPA * _OPPOSITE_LANE_KAPPA_SCALE,
                        )
                    # Fix B — road-following curvature expansion for shared turns.
                    # Guard: own lane only — ego_k guard removed because the yaw-rate
                    # proxy underestimates curvature on gentle corners and silently
                    # blocks Fix B when it's most needed.
                    fixb_fired = False
                    if own_lane and abs(ego_curvature) >= _TURNING_DIVERGE_CURVATURE:
                        new_dk = max(
                            delta_kappa_t,
                            min(abs(ego_curvature), _SHARED_TURN_MAX_KAPPA),
                        )
                        fixb_fired = new_dk > delta_kappa_t
                        delta_kappa_t = new_dk
                    # For own-lane vehicles, build evasion arcs without forced braking.
                    # The head-on decel model (7.8 m/s²) is correct for genuine threats
                    # but wrong here: we're asking "will this vehicle naturally clear ego
                    # by following the road" — not "what if both vehicles brake hard."
                    # A braking evasion arc stops in ~1.3 s right inside ego's curved
                    # forward path, causing both left_clears and right_clears to be False
                    # even when the vehicle is 5+ m into its own lane.
                    evasion_decel = 0.0 if own_lane else base_target_arc.decel
                    tgt_evasion_left = build_arc(
                        base_target_arc.start_x, base_target_arc.start_z,
                        base_target_arc.yaw_rad, v.speed,
                        base_target_arc.curvature + delta_kappa_t,
                        base_target_arc.half_width, base_target_arc.horizon,
                        decel=evasion_decel,
                    )
                    tgt_evasion_right = build_arc(
                        base_target_arc.start_x, base_target_arc.start_z,
                        base_target_arc.yaw_rad, v.speed,
                        base_target_arc.curvature - delta_kappa_t,
                        base_target_arc.half_width, base_target_arc.horizon,
                        decel=evasion_decel,
                    )
                    left_clears_ego = arc_arc_collision(
                        ego_arc, tgt_evasion_left, _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                    ) is None
                    right_clears_ego = arc_arc_collision(
                        ego_arc, tgt_evasion_right, _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                    ) is None
                    if left_clears_ego or right_clears_ego:
                        oncoming_evasion_filtered_ids.add(v.id)
                        continue

                colliding_ids.add(v.id)

                # Risk confirmation — oncoming vehicles require 2× duration
                newly_risky.add(v.id)
                if v.id not in self._risk_first_seen:
                    self._risk_first_seen[v.id] = now_mono
                confirm_duration = (
                    _RISK_CONFIRM_DURATION_ONCOMING if head_on else _RISK_CONFIRM_DURATION
                )
                if now_mono - self._risk_first_seen[v.id] < confirm_duration:
                    continue

                if unbraked_ttc < best_unbraked_ttc:
                    best_unbraked_ttc = unbraked_ttc
                    best_raw_dist = dist
                    best_hit_x = unbraked_hit[1]
                    best_hit_z = unbraked_hit[2]

                braked_hit = _earliest_hit(
                    ego_braked_arc, cross_arcs,
                    _CORRIDOR_MARGIN + stopping_buffer,
                    _COLLISION_SAMPLES,
                    lateral_gap,
                )

                if braked_hit is None:
                    ttb = max(unbraked_ttc - t_stop * _TIME_TO_BRAKE_BUFFER, 0.0)
                    braking_suppressed_ids.add(v.id)
                else:
                    ttb = 0.0

                if ttb < best_ttb:
                    best_ttb = ttb
                    best_hit_x = unbraked_hit[1]
                    best_hit_z = unbraked_hit[2]

            vehicle_dicts.append(veh_dict)

        self._risk_first_seen = {
            k: v for k, v in self._risk_first_seen.items() if k in newly_risky
        }

        new_state = AEBState.STANDBY
        time_to_brake = _INF
        display_ttc = best_unbraked_ttc

        if run_collision and best_ttb < _INF:
            time_to_brake = best_ttb
            if time_to_brake < _WARN_TTB_THRESHOLD:
                new_state = AEBState.WARN
            if time_to_brake < _BRAKE_TTB_THRESHOLD:
                new_state = AEBState.BRAKE

        # BRAKE latch — prevent rapid cycling near threshold
        if self._prev_state == AEBState.BRAKE and time_to_brake < _BRAKE_RELEASE_THRESHOLD:
            new_state = AEBState.BRAKE

        # Hold escalated state 0.3 s
        if self._prev_state.value > new_state.value and now_mono < self._state_hold_until:
            new_state = self._prev_state
        if new_state != self._prev_state:
            self._state_hold_until = now_mono + 0.3

        self._prev_state = new_state

        user_brake = self._read_user_braking()
        if not tmp_traffic_session:
            self._latched_filter_ego_kmh = None
        elif new_state < AEBState.WARN and not user_brake:
            self._latched_filter_ego_kmh = None
        elif self._latched_filter_ego_kmh is None and (
                new_state >= AEBState.WARN or user_brake):
            self._latched_filter_ego_kmh = ego_kmh_now

        snap = AEBSnapshot(
            ego_x=ego_x, ego_z=ego_z, ego_yaw=ego_yaw_rad,
            ego_speed=ego_speed, ego_half_w=ego_hw, ego_half_l=ego_half_l,
            ego_arc=ego_arc, ego_braked_arc=ego_braked_arc,
            ego_has_trailer=ego_has_trailer,
            vehicles=vehicle_dicts, vehicle_arcs=vehicle_arcs,
            colliding_ids=colliding_ids, suppressed_ids=suppressed_ids,
            braking_suppressed_ids=braking_suppressed_ids,
            evasion_filtered_ids=evasion_filtered_ids,
            oncoming_evasion_filtered_ids=oncoming_evasion_filtered_ids,
            aeb_state=new_state, time_to_collision=display_ttc,
            time_to_brake=time_to_brake,
            hit_x=best_hit_x, hit_z=best_hit_z,
            evasion_left_arc=ego_evasion_left,
            evasion_right_arc=ego_evasion_right,
        )

        if new_state >= AEBState.WARN:
            self._sound_handler.start_warning()
        else:
            self._sound_handler.stop_warning()

        with self.data._lock:
            self.data.AEB_warn = (new_state >= AEBState.WARN)
            self.data.AEB_brake = (new_state == AEBState.BRAKE)
            self.data.time_to_brake = time_to_brake
            self.data.em_stop_requested = (new_state == AEBState.BRAKE)
            self.data.snapshot = snap
        self._last_snapshot = snap

    def teardown(self) -> None:
        self._sound_handler.cleanup()
        if self._radar_visualizer is not None:
            try:
                self._radar_visualizer.stop()
            except Exception:
                pass
        self._latched_filter_ego_kmh = None
        with self.data._lock:
            self.data.AEB_warn = False
            self.data.AEB_brake = False
            self.data.time_to_brake = _INF
            self.data.em_stop_requested = False
            self.data.snapshot = AEBSnapshot()
        logger.debug("AEB teardown complete")

    def _read_radar_snapshot(
        self,
    ) -> tuple[list[Vehicle], float, float, float, float, float, float, float,
               bool, float | None, bool, bool] | None:
        """Read the radar thread's published snapshot under its data lock.

        Returns ``None`` when the radar thread is missing / not alive — AEB
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
                )
        except AttributeError:
            return None