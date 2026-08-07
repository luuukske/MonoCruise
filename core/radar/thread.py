"""Radar thread: traffic + ego snapshot (``radar_thread``). See core/radar/README.md §12."""

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
    # Nested trailers as Vehicles for ACC scoring only. See core/radar/README.md §12.
    trailer_vehicles: list[Vehicle] = field(default_factory=list)
    tmp_session: bool = False

    # Ego state (copied from telemetry under telemetry's lock each frame).
    ego_x: float = 0.0
    ego_y: float = 0.0
    ego_z: float = 0.0
    ego_yaw_rad: float = 0.0       # rotationX * 2π: see aeb README.md §2
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
    # Held (not bumped) while paused so AEB/ACC treat the frame as stale.
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
        self._last_ego_hist_t: float = 0.0
        # None until the first frame; then True when using SCS simulatedTime.
        self._kin_use_sim: bool | None = None
        self._was_paused: bool = False

    def setup(self) -> None:
        self._traffic.open()
        logger.debug("radar setup complete")

    def teardown(self) -> None:
        self._traffic.close()
        self._ego_position_history.clear()
        self._last_ego_hist_t = 0.0
        self._kin_use_sim = None
        self._was_paused = False
        with self.data._lock:
            self.data.vehicles = []
            self.data.trailer_vehicles = []
            self.data.tmp_session = False
        logger.debug("radar teardown complete")

    def _reset_kinematics_clock(self) -> None:
        """Re-anchor after wall↔sim clock domain changes."""
        self._traffic.clear_kinematics_state()
        self._ego_position_history.clear()
        self._last_ego_hist_t = 0.0
        logger.debug("radar kinematics clock domain reset")

    def _read_ego(self) -> tuple[
        float, float, float, float, float, float, bool, bool, float, int, float,
    ]:
        """Ego tuple: pose, speed, steer, pause, trailer, pitch_deg, sim time, mass."""
        try:
            tel = registry.get_thread("telemetry_thread")
            if tel is None or not tel.is_alive():
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0, 0.0
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
                    int(getattr(tel.data, "simulated_time_us", 0) or 0),
                    float(getattr(tel.data, "estimated_total_mass_kg", 0.0)),
                )
        except (KeyError, AttributeError):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0, 0.0

    @staticmethod
    def _kinematics_t(simulated_time_us: int) -> float:
        """Seconds on the SCS simulated clock, or wall time if unavailable."""
        if simulated_time_us > 0:
            return simulated_time_us / 1_000_000.0
        return time.time()

    def _read_blinkers(self) -> tuple[bool, bool]:
        """Telemetry blinkers for clip capture; False,False if telemetry is down."""
        try:
            tel = registry.get_thread("telemetry_thread")
            if tel is None or not tel.is_alive():
                return False, False
            with tel.data._lock:
                return (
                    bool(getattr(tel.data, "blinkerLeft", False)),
                    bool(getattr(tel.data, "blinkerRight", False)),
                )
        except (KeyError, AttributeError):
            return False, False

    def _capture_frame(
        self, recorder, now_mono: float, t_wall: float,
        ego_x: float, ego_y: float, ego_z: float, ego_yaw_norm: float,
        ego_speed: float, ego_steer: float, ego_pitch_raw: float,
        ego_has_trailer: bool, paused: bool, ego_mass_kg: float,
        traffic_buf: bytes | None, parked_buf: bytes | None,
    ) -> None:
        """Debug clip capture; never raises into the radar loop."""
        try:
            bl_left, bl_right = self._read_blinkers()
            ego = EgoTelemetry(
                coordinateX=ego_x, coordinateY=ego_y, coordinateZ=ego_z,
                rotationX=ego_yaw_norm, rotationY=ego_pitch_raw, speed=ego_speed,
                userSteer=ego_steer, paused=paused, ego_has_trailer=ego_has_trailer,
                estimated_total_mass_kg=ego_mass_kg,
                blinkerLeft=bl_left, blinkerRight=bl_right,
            )
            recorder.push_radar_frame(RadarFrameRecord(
                t_wall=t_wall, t_mono=now_mono, ego=ego,
                traffic_buf=traffic_buf, parked_buf=parked_buf,
            ))
        except Exception:
            logger.debug("radar clip capture failed", exc_info=True)

    def _publish_ego_fields(
        self,
        ego_x: float,
        ego_y: float,
        ego_z: float,
        ego_yaw_norm: float,
        ego_speed: float,
        ego_steer: float,
        ego_has_trailer: bool,
        ego_pitch_deg: float,
        paused: bool,
        *,
        vehicles: list[Vehicle] | None = None,
        trailer_vehicles: list[Vehicle] | None = None,
        tmp_session: bool | None = None,
        ego_curvature: float | None = None,
        bump_t_mono: bool = False,
    ) -> None:
        ego_yaw_rad = ego_yaw_norm * 2.0 * math.pi
        _pv = (ego_pitch_deg + 0.5) % 1.0 - 0.5
        ego_pitch_rad = -_pv * 2.0 * math.pi
        with self.data._lock:
            if vehicles is not None:
                self.data.vehicles = vehicles
            if trailer_vehicles is not None:
                self.data.trailer_vehicles = trailer_vehicles
            if tmp_session is not None:
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
            # Only overwrite curvature on a published (unpaused) frame.
            if bump_t_mono:
                self.data.ego_curvature = ego_curvature
            self.data.paused = paused
            if bump_t_mono:
                self.data.t_mono = time.monotonic()

    def _read_traffic(
        self,
        ego_x: float,
        ego_y: float,
        ego_z: float,
        ego_speed: float,
        t_kin: float,
    ) -> tuple[list[Vehicle], list[Vehicle]] | None:
        capture = get_recorder()
        self._traffic.capture_raw = capture is not None
        return self._traffic.read(
            ego_x, ego_y, ego_z, ego_speed, t_now=t_kin,
        )

    def loop(self) -> None:
        if not self.running:
            return

        (ego_x, ego_y, ego_z, ego_yaw_norm, ego_speed, ego_steer,
         paused, ego_has_trailer, ego_pitch_deg, simulated_time_us,
         ego_mass_kg) = self._read_ego()
        t_kin = self._kinematics_t(simulated_time_us)
        use_sim = simulated_time_us > 0
        # Wall↔sim switch without a reset leaves Vehicle.time in the old domain
        # and produces huge/negative dt (stuck sub-frame path).
        if self._kin_use_sim is not None and self._kin_use_sim != use_sim:
            self._reset_kinematics_clock()
        self._kin_use_sim = use_sim

        # Pause / wall vs sim clock handling. See core/radar/README.md §7.
        if paused:
            self._was_paused = True
            self._publish_ego_fields(
                ego_x, ego_y, ego_z, ego_yaw_norm, ego_speed, ego_steer,
                ego_has_trailer, ego_pitch_deg, True,
                bump_t_mono=False,
            )
            capture = get_recorder()
            if capture is not None:
                self._capture_frame(
                    capture, time.monotonic(), time.time(),
                    ego_x, ego_y, ego_z, ego_yaw_norm, ego_speed, ego_steer,
                    ego_pitch_deg, ego_has_trailer, True, ego_mass_kg, None, None,
                )
            return

        if self._was_paused:
            self._was_paused = False
            self._traffic.request_reanchor()

        # Ego path history: advance only when the kinematics clock moves so a
        # hitch (frozen simulatedTime) does not stretch curvature samples.
        if t_kin > self._last_ego_hist_t:
            self._ego_position_history.append((t_kin, ego_x, ego_z))
            self._last_ego_hist_t = t_kin
            if len(self._ego_position_history) > EGO_POSITION_HISTORY_LEN:
                self._ego_position_history = self._ego_position_history[-EGO_POSITION_HISTORY_LEN:]

        ego_curvature = ego_curvature_from_history(self._ego_position_history)

        traffic = self._read_traffic(ego_x, ego_y, ego_z, ego_speed, t_kin)
        if traffic is None:
            vehicles, trailer_vehicles = [], []
        else:
            vehicles, trailer_vehicles = traffic
        tmp_session = any(v.is_tmp for v in vehicles)

        self._publish_ego_fields(
            ego_x, ego_y, ego_z, ego_yaw_norm, ego_speed, ego_steer,
            ego_has_trailer, ego_pitch_deg, False,
            vehicles=vehicles,
            trailer_vehicles=trailer_vehicles,
            tmp_session=tmp_session,
            ego_curvature=ego_curvature,
            bump_t_mono=True,
        )

        capture = get_recorder()
        if capture is not None:
            with self.data._lock:
                now_mono = self.data.t_mono
            self._capture_frame(
                capture, now_mono, self._traffic.last_t_wall,
                ego_x, ego_y, ego_z, ego_yaw_norm, ego_speed, ego_steer,
                ego_pitch_deg, ego_has_trailer, False, ego_mass_kg,
                self._traffic.last_traffic_bytes, self._traffic.last_parked_bytes,
            )

