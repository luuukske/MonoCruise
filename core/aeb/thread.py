"""
AEB Thread — Automatic Emergency Braking with arc-based collision detection
and evasive path planning.

!! CRITICAL — COLLISION LOGIC NOTE !!
  Detection is TTB-based (Time To Brake), NOT TTC-based against the braked arc.

  TWO arcs are built each loop:
    ego_arc         — constant speed; detects whether a collision exists on
                      the current path at all.
    ego_braked_arc  — full-brake decel; classifies urgency only.

  Per-vehicle logic:
    1. Check ego_arc vs target.  No hit → skip (nothing in our path).
    2. Check ego_braked_arc vs target.
         braked_hit is None     → braking fully avoids.  Compute TTB:
                                       TTB = max(unbraked_ttc
                                                 - t_stop × _TIME_TO_BRAKE_BUFFER,
                                                 0.0)
                                   Vehicle is marked braking_suppressed
                                   (visual distinction only).
         braked_hit is not None → braking is insufficient.  TTB = 0 (brake NOW).

  Decision threshold is best_ttb across all vehicles:
    TTB < _WARN_TTB_THRESHOLD  → WARN
    TTB < _BRAKE_TTB_THRESHOLD → BRAKE

  BRAKE latch: once triggered, AEB_brake stays active until TTB >= _BRAKE_RELEASE_THRESHOLD.

  Risk confirmation: a vehicle only contributes to TTB after it has been
  continuously detected as a risk for >= _RISK_CONFIRM_DURATION seconds.
  It is still shown visually (colliding_ids) before confirmation.

  Stopping buffer: the braked-arc collision check uses an expanded corridor
  (_CORRIDOR_MARGIN + stopping_buffer) where stopping_buffer is speed-dependent.
  This ensures ego stops with physical clearance rather than just touching.

  Trailer arcs: trailers inherit the tractor's curvature so their predicted
  path follows the tractor's turn rather than going straight.

  Why not "primary braked arc" (old approach):
    Old code triggered only when braking could NOT avoid a collision.
    This silenced every vehicle that braking *would* save from — which
    is exactly the set that needs the warning.  Stationary vehicles were
    completely silent until the stopping distance was exhausted, by which
    point it was already too late.

  _is_approaching filter:
    Applied only to co-directional MOVING targets on the unbraked arc.
    Never applied to stationary targets — a parked car in your lane is
    always a real hit regardless of heading alignment.

Registry name: ``aeb_thread``

Other threads read:
  registry.get_thread("aeb_thread").data.AEB_warn
  registry.get_thread("aeb_thread").data.AEB_brake
  registry.get_thread("aeb_thread").data.time_to_brake
  registry.get_thread("aeb_thread").data.em_stop_requested
  registry.get_thread("aeb_thread").data.snapshot
"""

from __future__ import annotations

import enum
import logging
import math
import struct
import mmap
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from .traffic import (
    Position, Quaternion, Size, Trailer, Vehicle,
    ArcPath, build_arc, arc_arc_collision,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INF: float = 1e9

_FULL_BRAKE_DECEL: float = 7.8
_MIN_SPEED_MS: float = 5.0 / 3.6 #testing 35.0 / 3.6 is recommended
_MAX_RANGE: float = 100.0
_MAX_RANGE_SQ: float = _MAX_RANGE ** 2

_MIN_ARC_HORIZON: float = 3.0
_MAX_ARC_HORIZON: float = 4.0
_CORRIDOR_MARGIN: float = 0.5
_COLLISION_SAMPLES: int = 48

_WARN_TTB_THRESHOLD: float = 1.3
_BRAKE_TTB_THRESHOLD: float = 0.1
_BRAKE_RELEASE_THRESHOLD: float = 0.3   # TTB must exceed this for BRAKE → WARN transition
_TIME_TO_BRAKE_BUFFER: float = 0.0

# Stopping-distance buffer: expands the braked-arc collision corridor so ego
# stops with physical clearance instead of just touching the target.
_STOP_BUFFER_FIXED: float = 1.2    # metres (baseline gap at rest)

# Arc start position along the vehicle body.
# 0.0 = physical back, 1.0 = physical front.
# Applies to ego, all traffic vehicles, and trailers.
# For reversing vehicles p is automatically mirrored so the leading edge is always used.
_ARC_START_PCTG: float = 0.2

# A vehicle must be continuously detected as a risk for this many seconds
# before it contributes to TTB / AEB state.
_RISK_CONFIRM_DURATION: float = 0.1

_REAR_DOT_THRESHOLD: float = -0.5
_OVERTAKE_SPEED_MARGIN: float = 2.0

# Elevation filter: do not track vehicles below ego (e.g. on road underneath).
# Uses slope (rotationY, positive = uphill) so vehicles in front on a downhill
# are not wrongly filtered. Margin per AGENTS.md ±6 m road-level filtering.
_ELEVATION_MARGIN_M: float = 5.0

# Cross-traffic safe zone: for near-perpendicular targets (90° yaw diff) we expand
# the collision check by ghost arcs ±padding along the target's heading.  This
# catches cases where a fast crosser's arc shifts past ego's path between samples,
# or where a long trailer body occupies the crossing zone slightly before/after the
# predicted centre.
#
# padding = cross_factor * (BASE + SPEED * target_speed_ms)
#   cross_factor = |sin(yaw_diff)| → 1.0 at 90°, 0.0 at 0°/180°
_CROSS_SAFE_ZONE_BASE: float = 0.5    # m — minimum padding at any speed
_CROSS_SAFE_ZONE_SPEED: float = 0.3  # m per m/s of target speed

# Evasion filter: two extra ego arcs offset by ±Δκ check whether a vehicle
# detected on the main arc could be avoided with a gentle steer (≤ 0.1 g).
# Δκ = _EVASION_G_THRESHOLD / v².  Clamped to _EVASION_FILTER_MAX_DELTA_KAPPA
# so the filter paths stay meaningful at low speed.
_EVASION_G_THRESHOLD: float = 0.1 * 9.81
_EVASION_FILTER_MAX_DELTA_KAPPA: float = 0.03

_VEHICLE_FORMAT = "ffffffffffffhhbb"
_TRAILER_FORMAT = "ffffffffff"
_VEHICLE_OBJECT_FORMAT = _VEHICLE_FORMAT + _TRAILER_FORMAT * 3
_TOTAL_FORMAT = "=" + _VEHICLE_OBJECT_FORMAT * 40
_BUF_SIZE = 6960
_VEH_STRIDE = 16 + 3 * 10


# ---------------------------------------------------------------------------
# AEB state enum
# ---------------------------------------------------------------------------

class AEBState(enum.IntEnum):
    STANDBY = 0
    WARN = 1
    BRAKE = 2


# ---------------------------------------------------------------------------
# Snapshot for debug window
# ---------------------------------------------------------------------------

@dataclass
class AEBSnapshot:
    ego_x: float = 0.0
    ego_z: float = 0.0
    ego_yaw: float = 0.0
    ego_speed: float = 0.0
    ego_half_w: float = 1.25
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

    aeb_state: AEBState = AEBState.STANDBY
    time_to_collision: float = _INF
    time_to_brake: float = _INF
    hit_x: float = 0.0
    hit_z: float = 0.0

    evasion_left_arc: ArcPath | None = None
    evasion_right_arc: ArcPath | None = None


# ---------------------------------------------------------------------------
# Thread data
# ---------------------------------------------------------------------------

@dataclass
class AEBData(ThreadData):
    AEB_warn: bool = False
    AEB_brake: bool = False
    time_to_brake: float = _INF
    em_stop_requested: bool = False
    snapshot: AEBSnapshot = field(default_factory=AEBSnapshot)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Traffic buffer reader
# ---------------------------------------------------------------------------

class _TrafficReader:
    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._last_vehicles: dict[int, Vehicle] = {}

    def open(self) -> bool:
        if self._buf is not None:
            return True
        try:
            self._buf = mmap.mmap(0, _BUF_SIZE, r"Local\ETS2LATraffic")
            logger.info("ETS2LATraffic shared-memory buffer opened")
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                pass
            self._buf = None

    def read(self) -> list[Vehicle] | None:
        if self._buf is None:
            if not self.open():
                return None
        try:
            self._buf.seek(0)
        except Exception:
            return None
        try:
            raw = struct.unpack(_TOTAL_FORMAT, self._buf[:_BUF_SIZE])
        except Exception:
            self._buf = None
            return None

        vehicles: list[Vehicle] = []
        data = raw
        for _ in range(40):
            position = Position(data[0], data[1], data[2])
            rotation = Quaternion(data[3], data[4], data[5], data[6])
            size = Size(data[7], data[8], data[9])
            speed = data[10]
            acceleration = data[11]
            trailer_count = data[12]
            vid = data[13]
            is_tmp = bool(data[14])
            is_trailer = bool(data[15])

            trailers: list[Trailer] = []
            for j in range(3):
                off = 16 + j * 10
                tp = Position(data[off], data[off + 1], data[off + 2])
                tr = Quaternion(data[off + 3], data[off + 4], data[off + 5], data[off + 6])
                ts = Size(data[off + 7], data[off + 8], data[off + 9])
                if not tp.is_zero():
                    trailers.append(Trailer(tp, tr, ts, is_tmp))

            if not position.is_zero() and not rotation.is_zero():
                vehicles.append(Vehicle(
                    position, rotation, size, speed, acceleration,
                    trailer_count, trailers, vid, is_tmp, is_trailer,
                ))
            data = data[_VEH_STRIDE:]

        t_now = time.time()
        for v in vehicles:
            if v.id in self._last_vehicles:
                v.update_from_last(self._last_vehicles[v.id], t_now)
        self._last_vehicles = {v.id: v for v in vehicles}
        return vehicles


# ---------------------------------------------------------------------------
# Arc approach check
# ---------------------------------------------------------------------------

def _cross_zone_padding(ego_yaw_rad: float, v_yaw_rad: float, v_speed_ms: float) -> float:
    """Longitudinal safe-zone padding for near-perpendicular targets.

    Returns metres to offset ghost arcs ±along the target's heading.
    Scales with |sin(yaw_diff)| so it peaks at 90° and vanishes at 0°/180°.
    """
    cross_factor = abs(math.sin(ego_yaw_rad - v_yaw_rad))
    return cross_factor * (_CROSS_SAFE_ZONE_BASE + _CROSS_SAFE_ZONE_SPEED * v_speed_ms)


def _apply_cross_zone(arc: ArcPath, padding: float) -> list[ArcPath]:
    """Return [arc] plus two ghost arcs at ±padding along the target's heading.

    Ghost arcs model the space the vehicle will/has occupied just ahead and
    just behind its current predicted centre, guarding against timing mismatches
    that cause perpendicular crossers to slip through the arc-arc sample grid.
    """
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
) -> tuple[float, float, float] | None:
    """Return the earliest arc_arc_collision hit across all check_arcs."""
    best: tuple[float, float, float] | None = None
    for ca in check_arcs:
        h = arc_arc_collision(ego_arc, ca, margin, n_samples)
        if h is not None and (best is None or h[0] < best[0]):
            best = h
    return best


def _world_to_ego_forward(dx: float, dz: float, ego_yaw_rad: float) -> float:
    """Return ego-space forward component (rz). rz > 0 = in front of ego.

    World→ego: rx, rz = rotate_point(-dx, dz, -ego_yaw_rad).
    We only need rz = (-dx)*sin(-yaw) + dz*cos(-yaw) = dx*sin(yaw) + dz*cos(yaw).
    """
    return dx * math.sin(ego_yaw_rad) + dz * math.cos(ego_yaw_rad)


def _is_approaching(a: ArcPath, b: ArcPath, t: float, dt: float = 0.1) -> bool:
    """Return True if arcs a and b are still closing at time t."""
    ax0, az0 = a.position_at_time(t)
    bx0, bz0 = b.position_at_time(t)
    ax1, az1 = a.position_at_time(t + dt)
    bx1, bz1 = b.position_at_time(t + dt)
    d0_sq = (ax0 - bx0) ** 2 + (az0 - bz0) ** 2
    d1_sq = (ax1 - bx1) ** 2 + (az1 - bz1) ** 2
    return d1_sq < d0_sq


# ---------------------------------------------------------------------------
# AEB Thread
# ---------------------------------------------------------------------------

class AEBThread(BaseThread):
    loop_interval = 1 / 30 # 30fps
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="aeb_thread")
        self.data = AEBData()
        self._traffic = _TrafficReader()
        self._prev_state: AEBState = AEBState.STANDBY
        self._state_hold_until: float = 0.0
        self._last_snapshot: AEBSnapshot | None = None
        # Per-vehicle timestamp of when a risk was first continuously detected.
        # Cleared when a vehicle is no longer risky for a full loop iteration.
        self._risk_first_seen: dict[int, float] = {}

    def setup(self) -> None:
        self._traffic.open()
        logger.debug("AEB setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        aeb_active = Settings.AEB_enabled
        (ego_x, ego_z, ego_yaw_norm, ego_speed, steer, paused, ego_has_trailer,
         ego_y, ego_pitch_deg) = self._read_ego()

        if paused and self._last_snapshot is not None:
            with self.data._lock:
                self.data.snapshot = self._last_snapshot
            return

        # ETS2 rotationX is a 0–1 normalised yaw (full CCW rotation).
        # Multiply by 2π to get radians.
        #
        # !! DO NOT add +0.5 (i.e. +π) offset here !!
        # That offset rotates the ego forward vector by 180°, making the ego
        # arc point backward.  It was previously written as:
        #
        #     ego_yaw_rad = (ego_yaw_norm + 0.5) * 2.0 * math.pi   ← WRONG
        #
        # The ETS2 telemetry rotationX=0 corresponds to yaw=0 (North).
        # traffic.py forward vector: fwd = (-sin(yaw), -cos(yaw)).
        # At yaw=0 that gives (0, -1) = North, which is correct.
        # Adding +0.5 gave yaw=π → fwd=(0,+1)=South — backwards!
        #
        # Traffic vehicles are unaffected (they use Quaternion.euler() → radians
        # with no offset), which is why only the ego path was wrong.
        ego_yaw_rad = ego_yaw_norm * 2.0 * math.pi

        # Curvature from steering
        if ego_speed > 0.5:
            yaw_rate_rad_s = math.radians(steer * ego_speed * 12.0)
            ego_curvature = yaw_rate_rad_s / ego_speed
        else:
            ego_curvature = 0.0

        ego_hw: float = 1.25
        ego_half_l: float = 3.0  # cab half-length (m); increases stopping buffer for longer vehicles

        t_stop = ego_speed / _FULL_BRAKE_DECEL
        dynamic_horizon = min(max(_MIN_ARC_HORIZON, t_stop * 2.0), _MAX_ARC_HORIZON)

        # Speed-proportional stopping buffer: expanded corridor for the braked-arc
        # check so ego comes to a halt with physical clearance, not a kiss.
        # ego_half_l adds clearance proportional to vehicle length.
        stopping_buffer = _STOP_BUFFER_FIXED + ego_half_l

        # Shift ego arc start along the vehicle body using _ARC_START_PCTG.
        # Ego is symmetric (back_ratio = 0.5). AEB only runs above _MIN_SPEED_MS
        # so no reversing mirror needed.
        # Formula: start = position + (p - 0.5) * length * fwd
        _ego_fwd_x = -math.sin(ego_yaw_rad)
        _ego_fwd_z = -math.cos(ego_yaw_rad)
        _ego_body_offset = (_ARC_START_PCTG - 0.5) * (2.0 * ego_half_l)
        ego_front_x = ego_x + _ego_body_offset * _ego_fwd_x
        ego_front_z = ego_z + _ego_body_offset * _ego_fwd_z

        # Current-speed ego arc (for unbraked TTC / warning display)
        ego_arc = build_arc(
            ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
            ego_curvature, ego_hw, dynamic_horizon,
        )

        run_collision = aeb_active and ego_speed >= _MIN_SPEED_MS

        # Braking ego arc — THIS is the primary collision test arc.
        # By testing against the braking arc, we only trigger when a
        # collision would happen even if we start braking NOW.
        ego_braked_arc: ArcPath | None = None
        if run_collision:
            ego_braked_arc = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                ego_curvature, ego_hw, dynamic_horizon,
                decel=_FULL_BRAKE_DECEL,
            )

        # Evasion-filter arcs: two extra ego paths offset by ±Δκ from the
        # current curvature.  Δκ = _EVASION_G_THRESHOLD / v² so the lateral
        # acceleration at speed equals the threshold (0.1 g).  A vehicle must
        # collide with all three ego paths (centre + left + right) to be
        # considered a genuine in-lane hazard; otherwise it is likely parked on
        # the shoulder or sitting in a corner and is filtered out.
        ego_evasion_left: ArcPath | None = None
        ego_evasion_right: ArcPath | None = None
        if run_collision and ego_speed > 1.0:
            delta_kappa = min(
                _EVASION_G_THRESHOLD / (ego_speed * ego_speed),
                _EVASION_FILTER_MAX_DELTA_KAPPA,
            )
            ego_evasion_left = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                ego_curvature + delta_kappa, ego_hw, dynamic_horizon,
            )
            ego_evasion_right = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                ego_curvature - delta_kappa, ego_hw, dynamic_horizon,
            )

        ego_fwd_x = ego_arc.fwd_x
        ego_fwd_z = ego_arc.fwd_z

        vehicles = self._traffic.read() or []

        # Monotonic timestamp used for both risk confirmation and state hold.
        now_mono = time.monotonic()

        colliding_ids: set[int] = set()
        suppressed_ids: set[int] = set()
        braking_suppressed_ids: set[int] = set()
        evasion_filtered_ids: set[int] = set()
        best_ttb: float = _INF           # time-to-brake — primary decision metric
        best_unbraked_ttc: float = _INF  # TTC at current speed — display
        best_raw_dist: float = _INF      # raw ego→vehicle distance for current lane target
        best_hit_x: float = 0.0
        best_hit_z: float = 0.0
        vehicle_dicts: list[dict] = []
        vehicle_arcs: dict[int, list[ArcPath]] = {}

        # Vehicles that are actively risky this iteration — used to clean up
        # _risk_first_seen for vehicles that are no longer a threat.
        newly_risky: set[int] = set()

        # Elevation filter: expected Y at vehicle (x,z) using slope so vehicles
        # in front on a downhill are not filtered. Skip vehicles below that level.
        ego_pitch_rad = math.radians(ego_pitch_deg)

        for v in vehicles:
            vx, vz = v.position.x, v.position.z
            dx = vx - ego_x
            dz = vz - ego_z
            dist_sq = dx * dx + dz * dz
            if dist_sq > _MAX_RANGE_SQ:
                continue

            # Do not track vehicles below or above ego (e.g. road underneath or
            # overpass). Use slope so vehicles in front on a slope keep correct expected Y.
            rz = _world_to_ego_forward(dx, dz, ego_yaw_rad)
            expected_y = ego_y + rz * math.tan(ego_pitch_rad)
            if v.position.y < expected_y - _ELEVATION_MARGIN_M:
                continue
            if v.position.y > expected_y + _ELEVATION_MARGIN_M:
                continue

            dist = math.sqrt(dist_sq)
            _, v_yaw_deg, _ = v.rotation.euler()
            v_yaw_rad = math.radians(v_yaw_deg)
            v_hw = v.size.width / 2.0
            # Collision arcs use a 0.2 m narrower corridor (0.1 m per side) to
            # reduce false positives from width measurement noise.
            v_hw_coll = max(v_hw - 0.1, 0.3)

            # Tractor curvature — trailers inherit this so their arc follows
            # the turn instead of projecting straight ahead.
            abs_v_speed = abs(v.speed)
            if abs_v_speed > 0.5:
                v_curvature = math.radians(v.angular_velocity) / abs_v_speed
            else:
                v_curvature = 0.0

            veh_arc = v.get_arc(dynamic_horizon, arc_start_pctg=_ARC_START_PCTG)  # visual / unbraked
            trailer_dicts = []
            trailer_arcs: list[ArcPath] = []
            tr_hw_colls: list[float] = []
            for tr in v.trailers:
                tr_pos = tr.correct_position() if tr.is_tmp else tr.position
                _, tr_yaw_deg, _ = tr.rotation.euler()
                tr_yaw_rad = math.radians(tr_yaw_deg)
                tr_hw = tr.size.width / 2.0
                tr_hw_coll = max(tr_hw - 0.1, 0.3)
                tr_hw_colls.append(tr_hw_coll)

                # Use tractor curvature so trailer arc follows the vehicle's turn.
                # Trailer positions are always centered (back_ratio = 0.5):
                #   TMP:     correct_position() shifts to center.
                #   non-TMP: already centered per AGENTS.md.
                tr_is_rev = v.speed < -1e-3
                tr_effective_p = (1.0 - _ARC_START_PCTG) if tr_is_rev else _ARC_START_PCTG
                tr_fwd_x_l = -math.sin(tr_yaw_rad)
                tr_fwd_z_l = -math.cos(tr_yaw_rad)
                tr_body_offset = (tr_effective_p - 0.5) * tr.size.length
                tr_arc = build_arc(
                    tr_pos.x + tr_body_offset * tr_fwd_x_l,
                    tr_pos.z + tr_body_offset * tr_fwd_z_l,
                    tr_yaw_rad,
                    v.speed, v_curvature, tr_hw, dynamic_horizon,
                )
                trailer_arcs.append(tr_arc)

                trailer_dicts.append({
                    "x": tr_pos.x, "z": tr_pos.z,
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

            # Store all arcs (tractor + trailers) for debug visualisation and collision
            vehicle_arcs[v.id] = [veh_arc] + trailer_arcs

            if not run_collision:
                vehicle_dicts.append(veh_dict)
                continue

            # 1. Rear-approach / overtaker suppression
            to_veh_len = max(dist, 1e-6)
            dot_fwd = (dx * ego_fwd_x + dz * ego_fwd_z) / to_veh_len
            if dot_fwd < _REAR_DOT_THRESHOLD:
                veh_fwd_x = -math.sin(v_yaw_rad)
                veh_fwd_z = -math.cos(v_yaw_rad)
                approach_dot = veh_fwd_x * ego_fwd_x + veh_fwd_z * ego_fwd_z
                if approach_dot > 0.5 and v.speed > ego_speed + _OVERTAKE_SPEED_MARGIN:
                    veh_dict["rear_suppressed"] = True
                    suppressed_ids.add(v.id)
                    vehicle_dicts.append(veh_dict)
                    continue

            # Only suppress diverging hits for co-directional vehicles
            # (merging / same-direction traffic). Perpendicular and head-on
            # crossers must still trigger even if arcs are briefly diverging.
            veh_fwd_x = -math.sin(v_yaw_rad)
            veh_fwd_z = -math.cos(v_yaw_rad)
            fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
            co_directional = abs(fwd_dot) > 0.7
            # Head-on: both vehicles closing toward each other.  Model the target
            # as also braking at full decel — realistic since oncoming drivers
            # react to the same hazard.
            head_on = fwd_dot < -0.7

            # Build collision arcs: narrowed half-width; braked if head-on.
            target_decel = _FULL_BRAKE_DECEL if head_on else 0.0
            # Clamp reported acceleration; zeroed when the target is modelled as braking.
            target_accel = (
                0.0 if target_decel > 0.0
                else max(-6.0, min(4.0, v.acceleration))
            )
            veh_arc_coll = v.get_arc(dynamic_horizon, half_width=v_hw_coll,
                                     decel=target_decel, arc_start_pctg=_ARC_START_PCTG)
            trailer_arcs_coll: list[ArcPath] = []
            for idx, tr in enumerate(v.trailers):
                tr_pos = tr.correct_position() if tr.is_tmp else tr.position
                _, tr_yaw_deg, _ = tr.rotation.euler()
                tr_yaw_rad = math.radians(tr_yaw_deg)
                # Trailer collision arcs also use tractor curvature.
                tr_is_rev_c = v.speed < -1e-3
                tr_effective_p_c = (1.0 - _ARC_START_PCTG) if tr_is_rev_c else _ARC_START_PCTG
                tr_fwd_x_c = -math.sin(tr_yaw_rad)
                tr_fwd_z_c = -math.cos(tr_yaw_rad)
                tr_body_offset_c = (tr_effective_p_c - 0.5) * tr.size.length
                trailer_arcs_coll.append(build_arc(
                    tr_pos.x + tr_body_offset_c * tr_fwd_x_c,
                    tr_pos.z + tr_body_offset_c * tr_fwd_z_c,
                    tr_yaw_rad,
                    v.speed, v_curvature, tr_hw_colls[idx], dynamic_horizon,
                    decel=target_decel, accel=target_accel,
                ))

            # Collision is checked for both the vehicle and each of its trailers:
            # if the tractor OR any trailer is on ego's path, we trigger AEB.
            # Visual arcs (veh_arc / trailer_arcs) are unbraked and full-width.
            # Collision arcs (veh_arc_coll / trailer_arcs_coll) use narrowed width
            # and head-on braking decel.
            all_target_arcs = [veh_arc_coll] + trailer_arcs_coll

            # Cross-traffic safe zone: compute yaw-diff-weighted padding.
            # Peaks at 90° (perpendicular crossers / trailers), vanishes at 0°/180°.
            cross_padding = _cross_zone_padding(ego_yaw_rad, v_yaw_rad, abs_v_speed)

            for base_target_arc in all_target_arcs:
                # Expand each arc into up to 3 check arcs (centre + front/rear ghosts).
                cross_arcs = _apply_cross_zone(base_target_arc, cross_padding)

                # 1. PRIMARY: unbraked arc — is there anything in our path at all?
                unbraked_hit = _earliest_hit(
                    ego_arc, cross_arcs, _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                )
                # Suppress diverging co-directional hits for MOVING targets only
                # (e.g. ego just overtook a slower vehicle, arcs briefly overlap
                # but are separating).  Never suppress stationary targets —
                # a parked vehicle in the lane is always real regardless of heading.
                if (unbraked_hit is not None
                        and co_directional
                        and base_target_arc.speed > 0.5
                        and not _is_approaching(ego_arc, base_target_arc, unbraked_hit[0])):
                    unbraked_hit = None

                if unbraked_hit is None:
                    continue  # nothing in our path — skip braked check entirely

                unbraked_ttc = unbraked_hit[0]

                # Evasion-filter: check if the target also collides with both
                # offset ego paths.  If it misses either, ego could steer around
                # it within 0.1 g — likely a parked/corner vehicle, not an
                # in-lane hazard.  Skip the filter for moving co-directional
                # targets (they are genuinely in our lane) and for head-on
                # traffic (evasion is not meaningful when closing head-on).
                if (ego_evasion_left is not None
                        and ego_evasion_right is not None
                        and not head_on
                        and not (co_directional and base_target_arc.speed > 0.5)):
                    left_hit = _earliest_hit(
                        ego_evasion_left, cross_arcs,
                        _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                    )
                    right_hit = _earliest_hit(
                        ego_evasion_right, cross_arcs,
                        _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                    )
                    if left_hit is None or right_hit is None:
                        evasion_filtered_ids.add(v.id)
                        continue

                # Mark vehicle as colliding (for debug visualisation) regardless
                # of confirmation — so the debug window always reflects raw detections.
                colliding_ids.add(v.id)

                # Risk confirmation: only proceed with TTB computation once the
                # vehicle has been continuously detected as risky for
                # _RISK_CONFIRM_DURATION seconds.
                newly_risky.add(v.id)
                if v.id not in self._risk_first_seen:
                    self._risk_first_seen[v.id] = now_mono
                if now_mono - self._risk_first_seen[v.id] < _RISK_CONFIRM_DURATION:
                    continue  # detected but not yet confirmed — skip TTB

                if unbraked_ttc < best_unbraked_ttc:
                    best_unbraked_ttc = unbraked_ttc
                    best_raw_dist = dist
                    best_hit_x = unbraked_hit[1]
                    best_hit_z = unbraked_hit[2]

                # 2. SECONDARY: braked arc — does braking avoid the collision?
                # Use an expanded corridor (+ stopping_buffer) to guarantee ego
                # stops with clearance rather than just barely touching.
                # Cross zone applied here too — same geometric miss applies.
                braked_hit = _earliest_hit(
                    ego_braked_arc, cross_arcs,
                    _CORRIDOR_MARGIN + stopping_buffer,
                    _COLLISION_SAMPLES,
                )

                if braked_hit is None:
                    # Braking fully avoids — compute how long until we MUST brake.
                    # TTB = time remaining before the braking window closes:
                    #   unbraked_ttc     — when we hit at current speed
                    #   t_stop × buffer  — how long braking takes (with margin)
                    # When TTB reaches 0, delaying further makes a collision
                    # unavoidable regardless of braking.
                    ttb = max(unbraked_ttc - t_stop * _TIME_TO_BRAKE_BUFFER, 0.0)
                    braking_suppressed_ids.add(v.id)
                else:
                    # Braking is insufficient — collision happens even under full
                    # braking.  TTB = 0: must act now.
                    ttb = 0.0

                if ttb < best_ttb:
                    best_ttb = ttb
                    best_hit_x = unbraked_hit[1]
                    best_hit_z = unbraked_hit[2]

            vehicle_dicts.append(veh_dict)

        # Clean up confirmation timers for vehicles no longer risky this iteration.
        self._risk_first_seen = {
            k: v for k, v in self._risk_first_seen.items() if k in newly_risky
        }

        # AEB decision — based on TTB (time to brake), not raw TTC.
        # TTB = 0 means the braking window has closed or braking is insufficient.
        new_state = AEBState.STANDBY
        time_to_brake = _INF

        display_ttc = best_unbraked_ttc

        # Log the raw distance to the current in-lane vehicle (closest by TTC)
        # whenever a valid lane target exists. This is intended for debugging
        # and tuning; it does not affect AEB behaviour.
        if run_collision and best_unbraked_ttc < _INF and best_raw_dist < _INF:
            logger.info(
                "AEB lane vehicle: raw_distance=%.2f m, ttc=%.2f s",
                best_raw_dist,
                best_unbraked_ttc,
            )

        if run_collision and best_ttb < _INF:
            time_to_brake = best_ttb

            if time_to_brake < _WARN_TTB_THRESHOLD:
                new_state = AEBState.WARN
            if time_to_brake < _BRAKE_TTB_THRESHOLD:
                new_state = AEBState.BRAKE

        # BRAKE latch: once BRAKE is active, hold it until TTB clears
        # _BRAKE_RELEASE_THRESHOLD.  This prevents rapid on/off cycling when a
        # cut-in vehicle's TTB briefly fluctuates just above _BRAKE_TTB_THRESHOLD.
        if self._prev_state == AEBState.BRAKE and time_to_brake < _BRAKE_RELEASE_THRESHOLD:
            new_state = AEBState.BRAKE

        # Hold escalated state 0.3s
        if self._prev_state.value > new_state.value:
            if now_mono < self._state_hold_until:
                new_state = self._prev_state
        if new_state != self._prev_state:
            self._state_hold_until = now_mono + 0.3
        self._prev_state = new_state

        snap = AEBSnapshot(
            ego_x=ego_x, ego_z=ego_z, ego_yaw=ego_yaw_rad,
            ego_speed=ego_speed, ego_half_w=ego_hw, ego_half_l=ego_half_l,
            ego_arc=ego_arc, ego_braked_arc=ego_braked_arc,
            ego_has_trailer=ego_has_trailer,
            vehicles=vehicle_dicts, vehicle_arcs=vehicle_arcs,
            colliding_ids=colliding_ids, suppressed_ids=suppressed_ids,
            braking_suppressed_ids=braking_suppressed_ids,
            evasion_filtered_ids=evasion_filtered_ids,
            aeb_state=new_state, time_to_collision=display_ttc,
            time_to_brake=time_to_brake,
            hit_x=best_hit_x, hit_z=best_hit_z,
            evasion_left_arc=ego_evasion_left,
            evasion_right_arc=ego_evasion_right,
        )

        with self.data._lock:
            self.data.AEB_warn = (new_state >= AEBState.WARN)
            self.data.AEB_brake = (new_state == AEBState.BRAKE)
            self.data.time_to_brake = time_to_brake
            self.data.em_stop_requested = (new_state == AEBState.BRAKE)
            self.data.snapshot = snap
        self._last_snapshot = snap

    def teardown(self) -> None:
        self._traffic.close()
        with self.data._lock:
            self.data.AEB_warn = False
            self.data.AEB_brake = False
            self.data.time_to_brake = _INF
            self.data.em_stop_requested = False
            self.data.snapshot = AEBSnapshot()
        logger.debug("AEB teardown complete")

    def _read_ego(self) -> tuple[float, float, float, float, float, bool, bool, float, float]:
        try:
            tel = registry.get_thread("telemetry_thread")
            if tel is None or not tel.is_alive():
                return 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0.0
            with tel.data._lock:
                return (
                    tel.data.coordinateX,
                    tel.data.coordinateZ,
                    tel.data.rotationX,
                    tel.data.speed,
                    float(getattr(tel.data, "userSteer", 0.0)),
                    bool(getattr(tel.data, "paused", False)),
                    bool(getattr(tel.data, "ego_has_trailer", False)),
                    float(getattr(tel.data, "coordinateY", 0.0)),
                    float(getattr(tel.data, "rotationY", 0.0)),
                )
        except (KeyError, AttributeError):
            return 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0.0