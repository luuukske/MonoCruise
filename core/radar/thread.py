"""
Radar thread: central traffic + ego snapshot producer.

Reads the ETS2LA shared-memory traffic buffer and ego state from the
telemetry thread every loop (30 Hz) and publishes a coherent ``RadarData``
snapshot.  AEB and ACC both consume this data instead of opening the
traffic buffer themselves, so Vehicle smoothing / yaw EMA / position
history / curvature state is computed once per frame and shared.

Registry name: ``radar_thread``.

Readers:
    rt = registry.get_thread("radar_thread")
    with rt.data._lock:
        vehicles     = list(rt.data.vehicles)     # or copy just the ids you need
        trailer_vehicles = list(rt.data.trailer_vehicles)  # ACC-only: nested trailers
        ego_x        = rt.data.ego_x
        ego_yaw_rad  = rt.data.ego_yaw_rad
        ...

Vehicle instances are shared references: do not mutate them from consumer
threads.  The reader rebuilds the list every frame, and per-id smoothing
state is carried forward via ``Vehicle.update_from_last``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry

from .ego_path import EGO_POSITION_HISTORY_LEN, ego_curvature_from_history
from .reader import TrafficReader
from .traffic import Vehicle

from core.aeb.capture import get_recorder
from core.aeb.clip_schema import EgoTelemetry, RadarFrameRecord


logger = logging.getLogger(__name__)


@dataclass
class RadarData(ThreadData):
    # Vehicles: Vehicle instances with per-id smoothing carried forward.
    vehicles: list[Vehicle] = field(default_factory=list)
    # Nested trailers wrapped as standalone Vehicles (road-train trailers
    # behind the first). ACC-only: AEB already walks nested trailers via
    # Vehicle.trailers, so consuming this list there would double-count.
    trailer_vehicles: list[Vehicle] = field(default_factory=list)
    tmp_session: bool = False

    # Ego state (copied from telemetry under telemetry's lock each frame).
    ego_x: float = 0.0
    ego_y: float = 0.0
    ego_z: float = 0.0
    ego_yaw_rad: float = 0.0       # rotationX * 2π: see aeb AGENTS.md §2
    ego_yaw_norm: float = 0.0      # raw rotationX (0..1)
    ego_speed: float = 0.0         # m/s
    ego_pitch_deg: float = 0.0
    ego_pitch_rad: float = 0.0
    ego_steer: float = 0.0
    ego_has_trailer: bool = False
    paused: bool = False

    # Geometry-based ego curvature (1/m). None ⇒ caller falls back to yaw-rate proxy.
    ego_curvature: float | None = None

    # Monotonic time when snapshot was published.
    t_mono: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class RadarThread(BaseThread):
    loop_interval = 1 / 30
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="radar_thread")
        self.data = RadarData()
        self._traffic = TrafficReader()
        self._ego_position_history: list[tuple[float, float, float]] = []

    def setup(self) -> None:
        self._traffic.open()
        logger.debug("radar setup complete")

    def teardown(self) -> None:
        self._traffic.close()
        self._ego_position_history.clear()
        with self.data._lock:
            self.data.vehicles = []
            self.data.trailer_vehicles = []
            self.data.tmp_session = False
        logger.debug("radar teardown complete")

    def _read_ego(self) -> tuple[float, float, float, float, float, float, bool, bool, float]:
        """(x, y, z, yaw_norm, speed, steer, paused, ego_has_trailer, pitch_deg)."""
        try:
            tel = registry.get_thread("telemetry_thread")
            if tel is None or not tel.is_alive():
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0
            with tel.data._lock:
                return (
                    float(getattr(tel.data, "coordinateX", 0.0)),
                    float(getattr(tel.data, "coordinateY", 0.0)),
                    float(getattr(tel.data, "coordinateZ", 0.0)),
                    float(getattr(tel.data, "rotationX", 0.0)),
                    float(getattr(tel.data, "speed", 0.0)),
                    float(getattr(tel.data, "userSteer", 0.0)),
                    bool(getattr(tel.data, "paused", False)),
                    bool(getattr(tel.data, "ego_has_trailer", False)),
                    float(getattr(tel.data, "rotationY", 0.0)),
                )
        except (KeyError, AttributeError):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0

    def _capture_frame(
        self, recorder, now_mono: float, t_wall: float,
        ego_x: float, ego_y: float, ego_z: float, ego_yaw_norm: float,
        ego_speed: float, ego_steer: float, ego_pitch_raw: float,
        ego_has_trailer: bool, paused: bool,
        traffic_buf: bytes | None, parked_buf: bytes | None,
    ) -> None:
        """Push one radar frame to the clip recorder (debug capture, guarded).

        Never raises into the radar loop: a capture fault must not touch the
        safety path.
        """
        try:
            ego = EgoTelemetry(
                coordinateX=ego_x, coordinateY=ego_y, coordinateZ=ego_z,
                rotationX=ego_yaw_norm, rotationY=ego_pitch_raw, speed=ego_speed,
                userSteer=ego_steer, paused=paused, ego_has_trailer=ego_has_trailer,
            )
            recorder.push_radar_frame(RadarFrameRecord(
                t_wall=t_wall, t_mono=now_mono, ego=ego,
                traffic_buf=traffic_buf, parked_buf=parked_buf,
            ))
        except Exception:
            logger.debug("radar clip capture failed", exc_info=True)

    def loop(self) -> None:
        if not self.running:
            return

        (ego_x, ego_y, ego_z, ego_yaw_norm, ego_speed, ego_steer,
         paused, ego_has_trailer, ego_pitch_deg) = self._read_ego()

        # Paused → hold the previous snapshot; downstream threads treat an
        # unchanged t_mono as "no new frame".  Vehicle list is left as-is so
        # stale references don't flicker, matching AEB's paused-path behaviour.
        if paused:
            with self.data._lock:
                self.data.paused = True
                self.data.ego_x = ego_x
                self.data.ego_y = ego_y
                self.data.ego_z = ego_z
                self.data.ego_yaw_norm = ego_yaw_norm
                self.data.ego_yaw_rad = ego_yaw_norm * 2.0 * math.pi
                self.data.ego_speed = ego_speed
                self.data.ego_steer = ego_steer
                _pv = (ego_pitch_deg + 0.5) % 1.0 - 0.5
                _pr = -_pv * 2.0 * math.pi
                self.data.ego_pitch_deg = math.degrees(_pr)
                self.data.ego_pitch_rad = _pr
                self.data.ego_has_trailer = ego_has_trailer
            capture = get_recorder()
            if capture is not None:
                self._capture_frame(
                    capture, time.monotonic(), time.time(),
                    ego_x, ego_y, ego_z, ego_yaw_norm, ego_speed, ego_steer,
                    ego_pitch_deg, ego_has_trailer, True, None, None,
                )
            return

        now_mono = time.monotonic()

        self._ego_position_history.append((now_mono, ego_x, ego_z))
        if len(self._ego_position_history) > EGO_POSITION_HISTORY_LEN:
            self._ego_position_history = self._ego_position_history[-EGO_POSITION_HISTORY_LEN:]

        ego_curvature = ego_curvature_from_history(self._ego_position_history)

        capture = get_recorder()
        self._traffic.capture_raw = capture is not None
        read_result = self._traffic.read(ego_x, ego_y, ego_z, ego_speed)
        if read_result is None:
            vehicles, trailer_vehicles = [], []
        else:
            vehicles, trailer_vehicles = read_result
        tmp_session = any(v.is_tmp for v in vehicles)

        ego_yaw_rad = ego_yaw_norm * 2.0 * math.pi
        _pv = (ego_pitch_deg + 0.5) % 1.0 - 0.5
        ego_pitch_rad = -_pv * 2.0 * math.pi

        with self.data._lock:
            self.data.vehicles = vehicles
            self.data.trailer_vehicles = trailer_vehicles
            self.data.tmp_session = tmp_session
            self.data.ego_x = ego_x
            self.data.ego_y = ego_y
            self.data.ego_z = ego_z
            self.data.ego_yaw_rad = ego_yaw_rad
            self.data.ego_yaw_norm = ego_yaw_norm
            self.data.ego_speed = ego_speed
            self.data.ego_steer = ego_steer
            self.data.ego_pitch_deg = math.degrees(ego_pitch_rad)
            self.data.ego_pitch_rad = ego_pitch_rad
            self.data.ego_has_trailer = ego_has_trailer
            self.data.ego_curvature = ego_curvature
            self.data.paused = False
            self.data.t_mono = now_mono

        if capture is not None:
            self._capture_frame(
                capture, now_mono, self._traffic.last_t_wall,
                ego_x, ego_y, ego_z, ego_yaw_norm, ego_speed, ego_steer,
                ego_pitch_deg, ego_has_trailer, False,
                self._traffic.last_traffic_bytes, self._traffic.last_parked_bytes,
            )

