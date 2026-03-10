"""
AEB Thread — Automatic Emergency Braking collision detection.

Reads ego state from the telemetry thread and traffic vehicles from the
shared-memory traffic buffer.  Predicts paths, classifies threats, and
exposes AEB_warn / AEB_brake / time_to_stop for the sending thread and
the debug window.

Registry name: ``aeb_thread``

Other threads read:
  registry.get_thread("aeb_thread").data.AEB_warn
  registry.get_thread("aeb_thread").data.AEB_brake
  registry.get_thread("aeb_thread").data.time_to_stop
  registry.get_thread("aeb_thread").data.em_stop_requested
  registry.get_thread("aeb_thread").data.snapshot   (debug window)
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
    Position, Quaternion, Size, Trailer, Vehicle, VehiclePathPoint,
    _predict_position, _segment_intersection_params,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INF: float = 1e9

_FULL_BRAKE_DECEL: float = 12.0             # m/s^2 (magnitude)
_MIN_SPEED_MS: float = 35.0 / 3.6           # ~9.72 m/s  (35 km/h activation threshold)
_MAX_RANGE: float = 200.0                   # metres — ignore vehicles beyond this

_EGO_PATH_HORIZON: float = 5.0             # seconds of forward prediction for ego
_EGO_PATH_STEP: float = 0.1                # seconds between ego path samples

_WARN_TTC_THRESHOLD: float = 4.0           # seconds — enter WARN when TTC < this
_BRAKE_TTS_THRESHOLD: float = 0.8          # seconds — enter BRAKE when time_to_stop < this

_CORRIDOR_MARGIN: float = 0.5              # extra metres added to half-widths for collision
_ONCOMING_MIN_LATERAL: float = 3.0         # metres — oncoming must be closer than this laterally
_ONCOMING_YAW_LO: float = 140.0            # degrees — yaw diff range for "oncoming"
_ONCOMING_YAW_HI: float = 220.0

_REAR_DOT_THRESHOLD: float = -0.5          # dot product threshold for rear-approach

# Traffic buffer
_VEHICLE_FORMAT = "ffffffffffffhhbb"
_TRAILER_FORMAT = "ffffffffff"
_VEHICLE_OBJECT_FORMAT = _VEHICLE_FORMAT + _TRAILER_FORMAT * 3
_TOTAL_FORMAT = "=" + _VEHICLE_OBJECT_FORMAT * 40
_BUF_SIZE = 6960

_SEG_EPS: float = 1e-9
_TIME_EPS: float = 0.15  # relaxed vs traffic.py because ego path is synthetic


# ---------------------------------------------------------------------------
# AEB state enum (used by debug window HUD)
# ---------------------------------------------------------------------------

class AEBState(enum.IntEnum):
    STANDBY = 0
    WARN = 1
    BRAKE = 2


# ---------------------------------------------------------------------------
# Snapshot — passed to the debug window each frame
# ---------------------------------------------------------------------------

@dataclass
class AEBSnapshot:
    ego_x: float = 0.0
    ego_z: float = 0.0
    ego_yaw: float = 0.0
    ego_speed: float = 0.0
    ego_half_w: float = 1.25
    ego_half_l: float = 3.0
    ego_path: list = field(default_factory=list)

    vehicles: list = field(default_factory=list)
    colliding_ids: set = field(default_factory=set)
    suppressed_ids: set = field(default_factory=set)

    aeb_state: AEBState = AEBState.STANDBY
    time_to_collision: float = _INF
    time_to_brake: float = _INF


# ---------------------------------------------------------------------------
# Thread data
# ---------------------------------------------------------------------------

@dataclass
class AEBData(ThreadData):
    AEB_warn: bool = False
    AEB_brake: bool = False
    time_to_stop: float = _INF
    em_stop_requested: bool = False
    snapshot: AEBSnapshot = field(default_factory=AEBSnapshot)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_diff_deg(a: float, b: float) -> float:
    """Smallest unsigned angle between two yaw values (degrees)."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _ego_yaw_to_rad(rotation_x: float) -> float:
    """Convert ETS2 ego rotationX (0–1 normalised) to radians.  0.0 → South."""
    return (rotation_x + 0.5) * 2.0 * math.pi


def _build_ego_path(
    x: float,
    z: float,
    yaw_deg: float,
    speed: float,
    angular_velocity_deg_s: float,
    horizon: float,
    step: float,
) -> list[VehiclePathPoint]:
    """Generate a forward-predicted ego path as VehiclePathPoints.

    Uses the same prediction model as traffic vehicles (speed + yaw +
    angular_velocity with exponential decay) so ego and traffic paths are
    comparable.
    """
    now = time.time()
    pts: list[VehiclePathPoint] = []
    n = max(int(horizon / step), 2)
    for i in range(n + 1):
        t = i * step
        px, pz = _predict_position(
            x,
            z,
            speed,
            yaw_deg,
            angular_velocity_deg_s,
            t,
        )
        pts.append(
            VehiclePathPoint(
                x=px,
                z=pz,
                timestamp=now + t,
                yaw_deg=yaw_deg,
            )
        )
    return pts


def _build_vehicle_path(
    v: Vehicle,
    horizon: float,
    step: float,
) -> list[VehiclePathPoint]:
    """Predict a traffic vehicle's future path from its current state."""
    now = time.time()
    _, yaw_deg, _ = v.rotation.euler()
    pts: list[VehiclePathPoint] = []
    n = max(int(horizon / step), 2)
    for i in range(n + 1):
        t = i * step
        px, pz = _predict_position(
            v.position.x,
            v.position.z,
            v.speed,
            yaw_deg,
            v.angular_velocity,
            t,
        )
        pts.append(
            VehiclePathPoint(
                x=px,
                z=pz,
                timestamp=now + t,
                yaw_deg=yaw_deg,
            )
        )
    return pts


def _corridor_intersects(
    ego_path: list[VehiclePathPoint],
    veh_path: list[VehiclePathPoint],
    ego_hw: float,
    veh_hw: float,
) -> Optional[tuple[float, float, float]]:
    """Check whether two paths' corridors overlap.

    Because pure centre-line intersection is too strict, we inflate each path
    into a corridor of half-width ``ego_hw + veh_hw + margin`` and test the
    centre-lines offset by ± that corridor width.

    Returns (hit_time, hit_x, hit_z) of the earliest collision, or None.
    """
    corridor_w = ego_hw + veh_hw + _CORRIDOR_MARGIN

    if len(ego_path) < 2 or len(veh_path) < 2:
        return None

    best_time: Optional[float] = None
    best_x: float = 0.0
    best_z: float = 0.0

    for i in range(len(ego_path) - 1):
        ea = ego_path[i]
        eb = ego_path[i + 1]
        e_dx = eb.x - ea.x
        e_dz = eb.z - ea.z
        e_dt = eb.timestamp - ea.timestamp
        e_len_sq = e_dx * e_dx + e_dz * e_dz
        if e_dt <= 0.0 and e_len_sq < _SEG_EPS:
            continue

        for j in range(len(veh_path) - 1):
            va = veh_path[j]
            vb = veh_path[j + 1]
            v_dx = vb.x - va.x
            v_dz = vb.z - va.z
            v_dt = vb.timestamp - va.timestamp
            v_len_sq = v_dx * v_dx + v_dz * v_dz
            if v_dt <= 0.0 and v_len_sq < _SEG_EPS:
                continue

            # Check centre-line intersection
            params = _segment_intersection_params(
                ea.x, ea.z, eb.x, eb.z,
                va.x, va.z, vb.x, vb.z,
            )
            if params is not None:
                s, t = params
                time1 = ea.timestamp + s * e_dt
                time2 = va.timestamp + t * v_dt
                if abs(time1 - time2) <= _TIME_EPS:
                    hit_t = (time1 + time2) / 2.0
                    if best_time is None or hit_t < best_time:
                        best_time = hit_t
                        best_x = ea.x + s * e_dx
                        best_z = ea.z + s * e_dz
                    continue

            # Check closest approach — if the segments pass within corridor
            # width of each other at roughly the same time, treat as a hit.
            for s_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                ex = ea.x + s_frac * e_dx
                ez = ea.z + s_frac * e_dz
                et = ea.timestamp + s_frac * e_dt

                for t_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                    vx = va.x + t_frac * v_dx
                    vz = va.z + t_frac * v_dz
                    vt = va.timestamp + t_frac * v_dt

                    if abs(et - vt) > _TIME_EPS:
                        continue

                    dist_sq = (ex - vx) ** 2 + (ez - vz) ** 2
                    if dist_sq < corridor_w * corridor_w:
                        hit_t = (et + vt) / 2.0
                        if best_time is None or hit_t < best_time:
                            best_time = hit_t
                            best_x = (ex + vx) / 2.0
                            best_z = (ez + vz) / 2.0

    if best_time is None:
        return None
    return best_time, best_x, best_z


# ---------------------------------------------------------------------------
# Traffic buffer reader (adapted from Downloads/main.py)
# ---------------------------------------------------------------------------

class _TrafficReader:
    """Thin wrapper around the shared-memory traffic buffer."""

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

            data = data[16 + 3 * 10:]

        for v in vehicles:
            if v.id in self._last_vehicles:
                v.update_from_last(self._last_vehicles[v.id])

        self._last_vehicles = {v.id: v for v in vehicles}
        return vehicles


# ---------------------------------------------------------------------------
# AEB Thread
# ---------------------------------------------------------------------------

class AEBThread(BaseThread):
    loop_interval = 0.033  # ~30 Hz
    max_restarts = 5

    def __init__(self) -> None:
        super().__init__(name="aeb_thread")
        self.data = AEBData()
        self._traffic = _TrafficReader()
        self._prev_state: AEBState = AEBState.STANDBY
        self._state_hold_until: float = 0.0
        self._last_snapshot: AEBSnapshot | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self._traffic.open()
        logger.debug("setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        aeb_active = Settings.AEB_enabled

        # --- Ego state (always read for debug visualisation) -------------
        ego_x, ego_z, ego_yaw_norm, ego_speed, steer, paused = self._read_ego()

        # When telemetry is paused, keep the last snapshot static so paths
        # and vehicles don't drift while the game is paused.
        if paused and self._last_snapshot is not None:
            with self.data._lock:
                self.data.snapshot = self._last_snapshot
            return
        ego_yaw_rad = _ego_yaw_to_rad(ego_yaw_norm)
        ego_yaw_deg = ego_yaw_norm * 360.0

        # --- Traffic (always read for debug visualisation) ---------------
        vehicles = self._traffic.read()
        if vehicles is None:
            vehicles = []

        # --- Ego path ----------------------------------------------------
        # Approximate ego yaw rate from steering input (userSteer ~= -1..1).
        # Scale by speed so low-speed steering doesn't create extreme curves.
        yaw_rate_deg_s = steer * ego_speed * 7.0

        ego_path = _build_ego_path(
            ego_x,
            ego_z,
            ego_yaw_deg,
            ego_speed,
            yaw_rate_deg_s,
            _EGO_PATH_HORIZON,
            _EGO_PATH_STEP,
        )
        ego_hw = 1.25  # approximate ego truck half-width

        # Only run collision detection when AEB is enabled and above speed
        run_collision = aeb_active and ego_speed >= _MIN_SPEED_MS

        # --- Per-vehicle analysis ----------------------------------------
        colliding_ids: set[int] = set()
        suppressed_ids: set[int] = set()
        best_ttc: float = _INF
        best_hit_x: float = 0.0
        best_hit_z: float = 0.0
        vehicle_dicts: list[dict] = []

        for v in vehicles:
            vx = v.position.x
            vz = v.position.z

            dx = vx - ego_x
            dz = vz - ego_z
            dist = math.sqrt(dx * dx + dz * dz)
            if dist > _MAX_RANGE:
                continue

            _, v_yaw_deg, _ = v.rotation.euler()
            v_yaw_rad = math.radians(v_yaw_deg)
            v_hw = v.size.width / 2.0

            veh_path = _build_vehicle_path(
                v,
                _EGO_PATH_HORIZON,
                _EGO_PATH_STEP,
            )
            path_tuples = [(p.x, p.z) for p in veh_path]

            # --- Trailer info for debug ---
            trailer_dicts: list[dict] = []
            for tr in v.trailers:
                tr_pos = tr.correct_position() if tr.is_tmp else tr.position
                _, tr_yaw, _ = tr.rotation.euler()
                trailer_dicts.append({
                    "x": tr_pos.x,
                    "z": tr_pos.z,
                    "yaw": math.radians(tr_yaw),
                    "half_w": tr.size.width / 2.0,
                    "length": tr.size.length,
                    "is_tmp": tr.is_tmp,
                })

            veh_dict: dict = {
                "vid": v.id,
                "x": vx,
                "z": vz,
                "yaw": v_yaw_rad,
                "half_w": v_hw,
                "length": v.size.length,
                "is_tmp": v.is_tmp,
                "path": path_tuples,
                "rear_suppressed": False,
                "trailers": trailer_dicts,
            }

            if not run_collision:
                vehicle_dicts.append(veh_dict)
                continue

            # --- Classification ------------------------------------------

            # 1. Rear-approach suppression
            ego_fwd_x = -math.sin(ego_yaw_rad)
            ego_fwd_z = -math.cos(ego_yaw_rad)

            to_veh_len = max(dist, 1e-6)
            dot_fwd = (dx * ego_fwd_x + dz * ego_fwd_z) / to_veh_len

            if dot_fwd < _REAR_DOT_THRESHOLD:
                veh_fwd_x = -math.sin(v_yaw_rad)
                veh_fwd_z = -math.cos(v_yaw_rad)
                approach_dot = veh_fwd_x * ego_fwd_x + veh_fwd_z * ego_fwd_z
                if approach_dot > 0.5:
                    veh_dict["rear_suppressed"] = True
                    suppressed_ids.add(v.id)
                    vehicle_dicts.append(veh_dict)
                    continue

            # 2. Oncoming / opposite-lane down-weighting
            yaw_diff = _yaw_diff_deg(v_yaw_deg, ego_yaw_deg)
            is_oncoming = _ONCOMING_YAW_LO <= yaw_diff <= _ONCOMING_YAW_HI

            neg_dx = -dx
            c = math.cos(-ego_yaw_rad)
            s = math.sin(-ego_yaw_rad)
            rx = neg_dx * c - dz * s

            # 3. Collision check
            if len(veh_path) >= 2:
                hit = _corridor_intersects(ego_path, veh_path, ego_hw, v_hw)
            else:
                now = time.time()
                synth_path = [
                    VehiclePathPoint(vx, vz, now, v_yaw_deg),
                    VehiclePathPoint(
                        vx - v.speed * 3.0 * math.sin(v_yaw_rad),
                        vz - v.speed * 3.0 * math.cos(v_yaw_rad),
                        now + 3.0, v_yaw_deg,
                    ),
                ]
                hit = _corridor_intersects(ego_path, synth_path, ego_hw, v_hw)

            if hit is not None:
                hit_time, hit_x, hit_z = hit
                ttc = hit_time - time.time()
                if ttc < 0:
                    ttc = 0.0

                if is_oncoming and abs(rx) > _ONCOMING_MIN_LATERAL:
                    vehicle_dicts.append(veh_dict)
                    continue

                colliding_ids.add(v.id)
                if ttc < best_ttc:
                    best_ttc = ttc
                    best_hit_x = hit_x
                    best_hit_z = hit_z

            vehicle_dicts.append(veh_dict)

        # --- AEB decision ------------------------------------------------
        new_state = AEBState.STANDBY
        time_to_stop = _INF

        if run_collision and best_ttc < _INF:
            t_stop_duration = ego_speed / _FULL_BRAKE_DECEL
            time_to_stop = max(best_ttc - t_stop_duration, 0.0)

            if best_ttc < _WARN_TTC_THRESHOLD:
                new_state = AEBState.WARN
            if time_to_stop < _BRAKE_TTS_THRESHOLD:
                new_state = AEBState.BRAKE

        # Hold escalated state for at least 0.3s to avoid flicker
        now_mono = time.monotonic()
        if self._prev_state.value > new_state.value:
            if now_mono < self._state_hold_until:
                new_state = self._prev_state
        if new_state != self._prev_state:
            self._state_hold_until = now_mono + 0.3

        self._prev_state = new_state

        # --- Build snapshot & update data --------------------------------
        snap = AEBSnapshot(
            ego_x=ego_x,
            ego_z=ego_z,
            ego_yaw=ego_yaw_rad,
            ego_speed=ego_speed,
            ego_half_w=ego_hw,
            ego_half_l=3.0,
            ego_path=[(p.x, p.z) for p in ego_path],
            vehicles=vehicle_dicts,
            colliding_ids=colliding_ids,
            suppressed_ids=suppressed_ids,
            aeb_state=new_state,
            time_to_collision=best_ttc,
            time_to_brake=time_to_stop,
        )

        with self.data._lock:
            self.data.AEB_warn = (new_state == AEBState.WARN or new_state == AEBState.BRAKE)
            self.data.AEB_brake = (new_state == AEBState.BRAKE)
            self.data.time_to_stop = time_to_stop
            self.data.em_stop_requested = (new_state == AEBState.BRAKE)
            self.data.snapshot = snap

        self._last_snapshot = snap

    def teardown(self) -> None:
        self._traffic.close()
        with self.data._lock:
            self.data.AEB_warn = False
            self.data.AEB_brake = False
            self.data.time_to_stop = _INF
            self.data.em_stop_requested = False
            self.data.snapshot = AEBSnapshot()
        logger.debug("teardown complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_ego(self) -> tuple[float, float, float, float, float, bool]:
        """Return (ego_x, ego_z, ego_yaw_normalised, speed_m_s, userSteer, paused).

        Falls back to zeros if the telemetry thread is unavailable.
        """
        try:
            tel = registry.get_thread("telemetry_thread")
            if tel is None or not tel.is_alive():
                return 0.0, 0.0, 0.0, 0.0, 0.0, False
            with tel.data._lock:
                return (
                    tel.data.coordinateX,
                    tel.data.coordinateZ,
                    tel.data.rotationX,
                    abs(tel.data.speed),
                    float(getattr(tel.data, "userSteer", 0.0)),
                    bool(getattr(tel.data, "paused", False)),
                )
        except (KeyError, AttributeError):
            return 0.0, 0.0, 0.0, 0.0, 0.0, False

