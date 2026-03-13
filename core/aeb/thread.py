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
    TTB < _WARN_TTC_THRESHOLD  → WARN
    TTB < _BRAKE_TTS_THRESHOLD → BRAKE

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
  registry.get_thread("aeb_thread").data.evasion_curvature
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

_FULL_BRAKE_DECEL: float = 5.0
_MIN_SPEED_MS: float = 5.0 / 3.6 #testing 35.0 / 3.6 is recommended
_MAX_RANGE: float = 100.0
_MAX_RANGE_SQ: float = _MAX_RANGE ** 2

_MIN_ARC_HORIZON: float = 3.0
_MAX_ARC_HORIZON: float = 6.0
_CORRIDOR_MARGIN: float = 0.5
_COLLISION_SAMPLES: int = 36

_WARN_TTC_THRESHOLD: float = 1.3
_BRAKE_TTS_THRESHOLD: float = 0.1
_TIME_TO_BRAKE_BUFFER: float = 0.5

_REAR_DOT_THRESHOLD: float = -0.5
_OVERTAKE_SPEED_MARGIN: float = 2.0

# Collision avoidance (evasive path) — disabled; code kept for possible re-enable.
_RUN_COLLISION_AVOIDANCE_PATH: bool = False

_EVASION_CANDIDATES: int = 13
_EVASION_MAX_CURVATURE: float = 0.08
_EVASION_STEER_PENALTY: float = 5.0
_EVASION_G_THRESHOLD: float = 0.0 * 9.81

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

    aeb_state: AEBState = AEBState.STANDBY
    time_to_collision: float = _INF
    time_to_brake: float = _INF
    hit_x: float = 0.0
    hit_z: float = 0.0

    evasion_curvature: float = 0.0
    evasion_arc: ArcPath | None = None


# ---------------------------------------------------------------------------
# Thread data
# ---------------------------------------------------------------------------

@dataclass
class AEBData(ThreadData):
    AEB_warn: bool = False
    AEB_brake: bool = False
    time_to_brake: float = _INF
    em_stop_requested: bool = False
    evasion_curvature: float = 0.0
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

        for v in vehicles:
            if v.id in self._last_vehicles:
                v.update_from_last(self._last_vehicles[v.id])
        self._last_vehicles = {v.id: v for v in vehicles}
        return vehicles


# ---------------------------------------------------------------------------
# Arc approach check
# ---------------------------------------------------------------------------

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
# Evasive path planner
# ---------------------------------------------------------------------------

def _plan_evasion(
    ego_x: float,
    ego_z: float,
    ego_yaw_rad: float,
    ego_speed: float,
    current_curvature: float,
    ego_hw: float,
    threats: list[ArcPath],
    horizon: float = _MIN_ARC_HORIZON,
) -> tuple[float, ArcPath | None, bool]:
    """Find optimal evasive curvature.

    Uses the BRAKING ego arc for each candidate (since we're in WARN,
    the truck will be decelerating).

    Returns (best_curvature, best_arc_or_None, evasion_is_clear).
    ``evasion_is_clear`` is True when the best candidate has no collision.
    """
    if not threats or ego_speed < 1.0:
        return current_curvature, None, False

    n = _EVASION_CANDIDATES
    step = 2.0 * _EVASION_MAX_CURVATURE / max(n - 1, 1)
    candidates = [-_EVASION_MAX_CURVATURE + step * i for i in range(n)]
    candidates.append(current_curvature)

    best_score: float = -_INF
    best_kappa: float = current_curvature
    best_arc: ArcPath | None = None

    for kappa in candidates:
        ego_arc = build_arc(
            ego_x, ego_z, ego_yaw_rad, ego_speed,
            kappa, ego_hw, horizon,
            decel=_FULL_BRAKE_DECEL,
        )

        worst_ttc = _INF
        for ta in threats:
            result = arc_arc_collision(ego_arc, ta, _CORRIDOR_MARGIN, _COLLISION_SAMPLES)
            if result is not None and result[0] < worst_ttc:
                worst_ttc = result[0]

        score = worst_ttc - _EVASION_STEER_PENALTY * abs(kappa - current_curvature)
        if score > best_score:
            best_score = score
            best_kappa = kappa
            best_arc = ego_arc

    evasion_is_clear = best_score > (_INF * 0.5)
    return best_kappa, best_arc, evasion_is_clear


# ---------------------------------------------------------------------------
# AEB Thread
# ---------------------------------------------------------------------------

class AEBThread(BaseThread):
    loop_interval = 1 / 60 # 30fps
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="aeb_thread")
        self.data = AEBData()
        self._traffic = _TrafficReader()
        self._prev_state: AEBState = AEBState.STANDBY
        self._state_hold_until: float = 0.0
        self._last_snapshot: AEBSnapshot | None = None
        self._diag_counter: int = 0

    def setup(self) -> None:
        self._traffic.open()
        logger.debug("AEB setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        aeb_active = Settings.AEB_enabled
        ego_x, ego_z, ego_yaw_norm, ego_speed, steer, paused, ego_has_trailer = (
            self._read_ego()
        )

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

        t_stop = ego_speed / _FULL_BRAKE_DECEL
        dynamic_horizon = min(max(_MIN_ARC_HORIZON, t_stop * 2.0), _MAX_ARC_HORIZON)

        # Current-speed ego arc (for unbraked TTC / warning display)
        ego_arc = build_arc(
            ego_x, ego_z, ego_yaw_rad, ego_speed,
            ego_curvature, ego_hw, dynamic_horizon,
        )

        run_collision = aeb_active and ego_speed >= _MIN_SPEED_MS

        # Braking ego arc — THIS is the primary collision test arc.
        # By testing against the braking arc, we only trigger when a
        # collision would happen even if we start braking NOW.
        ego_braked_arc: ArcPath | None = None
        if run_collision:
            ego_braked_arc = build_arc(
                ego_x, ego_z, ego_yaw_rad, ego_speed,
                ego_curvature, ego_hw, dynamic_horizon,
                decel=_FULL_BRAKE_DECEL,
            )

        ego_fwd_x = ego_arc.fwd_x
        ego_fwd_z = ego_arc.fwd_z

        vehicles = self._traffic.read() or []

        colliding_ids: set[int] = set()
        suppressed_ids: set[int] = set()
        braking_suppressed_ids: set[int] = set()
        best_ttb: float = _INF           # time-to-brake — primary decision metric
        best_braked_ttc: float = _INF    # TTC under braking — evasion planner only
        best_unbraked_ttc: float = _INF  # TTC at current speed — display
        best_hit_x: float = 0.0
        best_hit_z: float = 0.0
        vehicle_dicts: list[dict] = []
        vehicle_arcs: dict[int, list[ArcPath]] = {}
        threat_arcs: list[ArcPath] = []

        for v in vehicles:
            vx, vz = v.position.x, v.position.z
            dx = vx - ego_x
            dz = vz - ego_z
            dist_sq = dx * dx + dz * dz
            if dist_sq > _MAX_RANGE_SQ:
                continue

            dist = math.sqrt(dist_sq)
            _, v_yaw_deg, _ = v.rotation.euler()
            v_yaw_rad = math.radians(v_yaw_deg)
            v_hw = v.size.width / 2.0

            veh_arc = v.get_arc(dynamic_horizon)
            trailer_dicts = []
            trailer_arcs: list[ArcPath] = []
            for tr in v.trailers:
                tr_pos = tr.correct_position() if tr.is_tmp else tr.position
                _, tr_yaw_deg, _ = tr.rotation.euler()
                tr_yaw_rad = math.radians(tr_yaw_deg)
                tr_hw = tr.size.width / 2.0

                tr_arc = build_arc(
                    tr_pos.x, tr_pos.z, tr_yaw_rad,
                    v.speed, 0.0, tr_hw, dynamic_horizon,
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
                "speed_kmh": v.speed * 3.6,
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
            co_directional = abs(ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z) > 0.7

            # Collision is checked for both the vehicle and each of its trailers:
            # if the tractor OR any trailer is on ego's path, we trigger AEB.
            all_target_arcs = [veh_arc] + trailer_arcs

            for target_arc in all_target_arcs:
                # 1. PRIMARY: unbraked arc — is there anything in our path at all?
                unbraked_hit = arc_arc_collision(
                    ego_arc, target_arc, _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                )
                # Suppress diverging co-directional hits for MOVING targets only
                # (e.g. ego just overtook a slower vehicle, arcs briefly overlap
                # but are separating).  Never suppress stationary targets —
                # a parked vehicle in the lane is always real regardless of heading.
                if (unbraked_hit is not None
                        and co_directional
                        and target_arc.speed > 0.5
                        and not _is_approaching(ego_arc, target_arc, unbraked_hit[0])):
                    unbraked_hit = None

                if unbraked_hit is None:
                    continue  # nothing in our path — skip braked check entirely

                unbraked_ttc = unbraked_hit[0]
                colliding_ids.add(v.id)

                if unbraked_ttc < best_unbraked_ttc:
                    best_unbraked_ttc = unbraked_ttc
                    best_hit_x = unbraked_hit[1]
                    best_hit_z = unbraked_hit[2]

                # 2. SECONDARY: braked arc — does braking avoid the collision?
                braked_hit = arc_arc_collision(
                    ego_braked_arc, target_arc, _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
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
                    threat_arcs.append(target_arc)
                    if braked_hit[0] < best_braked_ttc:
                        best_braked_ttc = braked_hit[0]

                if ttb < best_ttb:
                    best_ttb = ttb
                    best_hit_x = unbraked_hit[1]
                    best_hit_z = unbraked_hit[2]

            vehicle_dicts.append(veh_dict)

        # Diagnostics
        self._diag_counter += 1
        if self._diag_counter >= 60:
            self._diag_counter = 0
            if vehicle_dicts:
                logger.info(
                    "AEB diag: run=%s spd=%.1f vehs=%d colliding=%d "
                    "brake_insuff=%d braking_avoids=%d supp=%d "
                    "ttb=%.2f unbraked_ttc=%.2f braked_ttc=%.2f κ=%.4f",
                    run_collision, ego_speed * 3.6, len(vehicle_dicts),
                    len(colliding_ids),
                    len(colliding_ids) - len(braking_suppressed_ids),
                    len(braking_suppressed_ids), len(suppressed_ids),
                    best_ttb if best_ttb < _INF else -1.0,
                    best_unbraked_ttc if best_unbraked_ttc < _INF else -1.0,
                    best_braked_ttc if best_braked_ttc < _INF else -1.0,
                    ego_curvature,
                )

        # AEB decision — based on TTB (time to brake), not raw TTC.
        # TTB = 0 means the braking window has closed or braking is insufficient.
        new_state = AEBState.STANDBY
        time_to_brake = _INF
        evasion_kappa = ego_curvature
        evasion_arc: ArcPath | None = None
        evasion_viable = False

        display_ttc = best_unbraked_ttc

        if run_collision and best_ttb < _INF:
            time_to_brake = best_ttb

            if _RUN_COLLISION_AVOIDANCE_PATH:
                evasion_kappa, evasion_arc, evasion_is_clear = _plan_evasion(
                    ego_x, ego_z, ego_yaw_rad, ego_speed,
                    ego_curvature, ego_hw, threat_arcs, dynamic_horizon,
                )
                if evasion_is_clear:
                    delta_kappa = abs(evasion_kappa - ego_curvature)
                    a_lat = ego_speed ** 2 * delta_kappa
                    evasion_viable = a_lat <= _EVASION_G_THRESHOLD

            if not evasion_viable:
                if time_to_brake < _WARN_TTC_THRESHOLD:
                    new_state = AEBState.WARN
                if time_to_brake < _BRAKE_TTS_THRESHOLD:
                    new_state = AEBState.BRAKE
            else:
                new_state = AEBState.STANDBY

        # Hold escalated state 0.3s
        now_mono = time.monotonic()
        if self._prev_state.value > new_state.value:
            if now_mono < self._state_hold_until:
                new_state = self._prev_state
        if new_state != self._prev_state:
            self._state_hold_until = now_mono + 0.3
        self._prev_state = new_state

        snap = AEBSnapshot(
            ego_x=ego_x, ego_z=ego_z, ego_yaw=ego_yaw_rad,
            ego_speed=ego_speed, ego_half_w=ego_hw, ego_half_l=3.0,
            ego_arc=ego_arc, ego_braked_arc=ego_braked_arc,
            ego_has_trailer=ego_has_trailer,
            vehicles=vehicle_dicts, vehicle_arcs=vehicle_arcs,
            colliding_ids=colliding_ids, suppressed_ids=suppressed_ids,
            braking_suppressed_ids=braking_suppressed_ids,
            aeb_state=new_state, time_to_collision=display_ttc,
            time_to_brake=time_to_brake,
            hit_x=best_hit_x, hit_z=best_hit_z,
            evasion_curvature=evasion_kappa, evasion_arc=evasion_arc,
        )

        with self.data._lock:
            self.data.AEB_warn = (new_state >= AEBState.WARN)
            self.data.AEB_brake = (new_state == AEBState.BRAKE)
            self.data.time_to_brake = time_to_brake
            self.data.em_stop_requested = (new_state == AEBState.BRAKE)
            self.data.evasion_curvature = evasion_kappa
            self.data.snapshot = snap
        self._last_snapshot = snap

    def teardown(self) -> None:
        self._traffic.close()
        with self.data._lock:
            self.data.AEB_warn = False
            self.data.AEB_brake = False
            self.data.time_to_brake = _INF
            self.data.em_stop_requested = False
            self.data.evasion_curvature = 0.0
            self.data.snapshot = AEBSnapshot()
        logger.debug("AEB teardown complete")

    def _read_ego(self) -> tuple[float, float, float, float, float, bool, bool]:
        try:
            tel = registry.get_thread("telemetry_thread")
            if tel is None or not tel.is_alive():
                return 0.0, 0.0, 0.0, 0.0, 0.0, False, False
            with tel.data._lock:
                return (
                    tel.data.coordinateX,
                    tel.data.coordinateZ,
                    tel.data.rotationX,
                    abs(tel.data.speed),
                    float(getattr(tel.data, "userSteer", 0.0)),
                    bool(getattr(tel.data, "paused", False)),
                    bool(getattr(tel.data, "ego_has_trailer", False)),
                )
        except (KeyError, AttributeError):
            return 0.0, 0.0, 0.0, 0.0, 0.0, False, False