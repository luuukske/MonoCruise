"""
Telemetry thread: checks ETS2 SDK connection and exposes sdkActive state + game data.

Other threads read:
  registry.get_thread("telemetry_thread").data.is_connected    : SDK active
  registry.get_thread("telemetry_thread").data.manual_start    : user started app before game
  registry.get_thread("telemetry_thread").data.request_quit    : thread requests app shutdown
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
import threading

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry

from core.settings import Settings
from core.sending_thread.accel_to_pedals import compute_estimated_mass_kg
from core.radar.ego_geometry import EgoGeometry, geometry_from_sdk


logger = logging.getLogger(__name__)

_MASS_LOG_INTERVAL_S = 1.0
# Same mapping the background checker opens (checker/ets2_checker.py).
_TELEMETRY_SHM_NAME = "Local\\SCSTelemetry"
# Wheel slip is meaningless at walking pace, where rim speed and truck speed both vanish.
_SLIP_MIN_SPEED_MS = 2.0


def sdk_shm_active() -> bool | None:
    """True/False/None: SDK-active flag in SCS telemetry shared memory (Windows)."""
    try:
        from multiprocessing.shared_memory import SharedMemory

        shm = SharedMemory(name=_TELEMETRY_SHM_NAME, create=False)
    except FileNotFoundError:
        return False
    except (ValueError, OSError):
        return False
    except Exception:
        logger.debug("sdk_shm_active probe failed", exc_info=True)
        return None
    try:
        return bool(shm.buf[0])
    finally:
        shm.close()


def _wheel_slip(velocities, radii, on_ground, count, speed_ms: float) -> float:
    """Largest braking slip of a grounded wheel: 0 rolling free, 1 locked."""
    speed = abs(float(speed_ms or 0.0))
    if speed < _SLIP_MIN_SPEED_MS or not velocities or not radii or not on_ground:
        return 0.0
    worst = 0.0
    for i in range(min(int(count or 0), len(velocities), len(radii), len(on_ground))):
        if on_ground[i] and radii[i] > 0.0:
            rim = abs(velocities[i]) * 2.0 * math.pi * radii[i]
            worst = max(worst, 1.0 - rim / speed)
    return worst


def _grip_fields(raw: dict, speed_ms: float) -> tuple[str, float, float]:
    """Surface under most truck wheels and the worst truck / trailer wheel slip."""
    names = raw.get("substances") or []
    grounded = raw.get("truckWheelOnGround") or []
    ids = [s for s, g in zip(raw.get("truck_wheelSubstance") or [], grounded) if g]
    surface = ""
    if ids:
        common = Counter(ids).most_common(1)[0][0]
        surface = str(names[common]) if 0 <= common < len(names) else str(common)
    truck = _wheel_slip(
        raw.get("truck_wheelVelocity"), raw.get("truckWheelRadius"), grounded,
        raw.get("truckWheelCount", 0), speed_ms,
    )
    trailer = 0.0
    for unit in raw.get("trailer") or []:
        if not unit.get("attached", False):
            break
        trailer = max(trailer, _wheel_slip(
            unit.get("wheelVelocity"), unit.get("wheelRadius"), unit.get("wheelOnGround"),
            unit.get("wheelCount", 0), speed_ms,
        ))
    return surface, truck, trailer


_GRIP_DEBUG_COLUMNS: tuple[tuple[str, str], ...] = (
    ("lateral_ms2", "lv_accelerationX"),
    ("steer", "userSteer"),
    ("brake_temp_c", "brake_temperature_c"),
    ("air_psi", "air_pressure_psi"),
    ("wipers", "wipers"),
    ("surface", "surface"),
    ("truck_slip", "truck_wheel_slip"),
    ("trailer_slip", "trailer_wheel_slip"),
)


def grip_debug_fields(data: object) -> dict[str, object]:
    """Grip-related columns for brake_debug.csv; read under the data lock."""
    return {column: getattr(data, attr, "") for column, attr in _GRIP_DEBUG_COLUMNS}


def _window_open_on_taskbar() -> bool:
    """Return True when the main window reports it is visible to the user."""
    try:
        main_window = registry.get("main_window")
    except KeyError:
        return False
    return bool(getattr(main_window, "is_open_on_taskbar", False))


def _log_geometry_change(truck_id: str, geometry: EgoGeometry | None) -> None:
    if geometry is None:
        logger.info("ego geometry: wheel layout unreadable for %s, using calibration body", truck_id)
        return
    logger.info(
        "ego geometry for %s: length %.2f m (front %.2f, rear %.2f), width %.2f m, "
        "path origin %.2f m behind placement",
        truck_id, geometry.front_m + geometry.rear_m, geometry.front_m, geometry.rear_m,
        2.0 * geometry.half_width_m, -geometry.path_origin_m,
    )


@dataclass
class TelemetryThreadData(ThreadData):
    # Connection state
    is_connected: bool = False
    manual_start: bool = False # opens the UI if true, otherwise hide it

    # Game / SDK metadata
    game: int = 0               # 1 = ETS2, 2 = ATS
    game_version_major: int = 0
    game_version_minor: int = 0
    sdk_version: int = 0        # telemetry_plugin_revision

    # Simulation state
    paused: bool = False
    # SCS simulatedTime (us); radar integration clock; 0 until SDK publishes.
    simulated_time_us: int = 0

    # Truck
    speed: float = 0.0          # m/s: convert to km/h: speed * 3.6
    cruise_control_speed: float = 0.0  # m/s, 0.0 when CC inactive
    speedLimit: float = 0.0     # m/s, 0.0 when no speed limit
    blinkerRight = False
    blinkerLight = False
    retarderBrake: int = 0

    # Engine
    engine_rpm: float = 0.0
    engine_rpm_max: float = 2000.0

    # Gear
    gear: int = 0               # current gear (negative = reverse, 0 = neutral)
    gear_dashboard: int = 0     # displayed gear on dashboard

    # Raw inputs
    userThrottle: float = 0.0
    userBrake: float = 0.0
    userSteer: float = 0.0
    userClutch: float = 0.0

    # Game-applied inputs
    gameThrottle: float = 0.0
    gameBrake: float = 0.0
    gameClutch: float = 0.0

    # Cargo / mass (SCS config_f + truck_f)
    cargoMass: float = 0.0     # kg
    unitMass: float = 0.0      # kg: tractor/chassis mass from SDK
    fuel: float = 0.0        # litres (current tank)
    trailer_count: int = 0   # attached trailers with wheels (from SDK list)
    estimated_total_mass_kg: float = 0.0  # unitMass + cargoMass + fuel mass

    # Truck-local LATERAL accel (right/left), not longitudinal, despite the SDK's field
    # name. Do not use as accel/decel feedback; see AGENTS.md domain invariants.
    lv_accelerationX: float = 0.0  # m/s²
    # Commanded accel for ACC / cruise (other threads may set; default 0).
    commanded_accel_ms2: float = 0.0
    # ACC is keeping speed behind a lead it measures moving: a mild decel is not a stop.
    commanded_crawl_follow: bool = False

    # Grip diagnostics for brake_debug.csv only: full-pedal stops on one rig spread
    # 0.4-1.2x of the model while partial braking held steady (core/sending_thread/README.md).
    brake_temperature_c: float = 0.0
    air_pressure_psi: float = 0.0
    wipers: bool = False
    surface: str = ""
    truck_wheel_slip: float = 0.0
    trailer_wheel_slip: float = 0.0

    # Vehicle state
    parkBrake: bool = False
    rotationY: float = 0.0          # pitch normalized [0, 1] full-circle: use _road_grade_from_norm to convert
    hazardsActive: bool = False

    # SDK truck model id ("vehicle.scania.r" and the like). Runtime only: the
    # ego path model drops its learned steer gain when the vehicle changes.
    truck_id: str = ""

    coordinateX: float = 0.0
    coordinateY: float = 0.0   # elevation (m): used for road-level filtering
    coordinateZ: float = 0.0
    rotationX: float = 0.0

    # Ego trailer: True when at least one attached trailer with wheels exists.
    ego_has_trailer: bool = False

    # Total wheels in contact with the ground (tractor + all attached trailers).
    wheels_on_ground: int = 0

    # Body size and path origin from the SDK wheel layout; None when unreadable.
    ego_geometry: EgoGeometry | None = None

    request_quit: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


def _apply_telemetry(data: TelemetryThreadData, raw: dict) -> None:
    """Write all telemetry fields under the data lock."""
    with data._lock:
        data.game                = raw.get("game", 0)
        data.game_version_major  = raw.get("telemetry_version_game_major", 0)
        data.game_version_minor  = raw.get("telemetry_version_game_minor", 0)
        data.sdk_version         = raw.get("telemetry_plugin_revision", 0)
        data.paused              = raw.get("paused", False)
        data.simulated_time_us   = int(raw.get("simulatedTime", 0) or 0)
        data.coordinateX         = raw.get("coordinateX", 0.0)
        data.coordinateY         = raw.get("coordinateY", 0.0)
        data.coordinateZ         = raw.get("coordinateZ", 0.0)
        data.rotationX           = raw.get("rotationX", 0.0)
        data.speed               = raw.get("speed", 0.0)
        data.blinkerLeft         = raw.get("blinkerLeftActive", False)
        data.blinkerRight        = raw.get("blinkerRightActive", False)
        data.cruise_control_speed= raw.get("cruiseControlSpeed", 0.0)
        data.speedLimit          = raw.get("speedLimit", 0.0)
        data.engine_rpm          = raw.get("engineRpm", 0.0)
        data.engine_rpm_max      = raw.get("engineRpmMax", 2000.0)
        data.gear                = raw.get("gear", 0)
        data.gear_dashboard      = raw.get("gearDashboard", 0)
        data.retarderBrake       = raw.get("retarderBrake", 0)
        data.userThrottle        = raw.get("userThrottle", 0.0)
        data.userBrake           = raw.get("userBrake", 0.0)
        data.userSteer           = raw.get("gameSteer", 0.0) # use game values for ETS2LA compatibility for example
        data.userClutch          = raw.get("userClutch", 0.0)
        data.gameThrottle        = raw.get("gameThrottle", 0.0)
        data.gameBrake           = raw.get("gameBrake", 0.0)
        data.gameClutch          = raw.get("gameClutch", 0.0)
        data.cargoMass           = raw.get("cargoMass", 0.0)
        data.unitMass            = 10000
        data.fuel                = raw.get("fuel", 0.0)
        trailer_count = 0
        for trailer in raw.get("trailer", []):
            if trailer.get("wheelCount", 0) > 0 and trailer.get("attached", False):
                trailer_count += 1
            else:
                break
        data.trailer_count = trailer_count
        truck_wog = sum(1 for w in raw.get("truckWheelOnGround", []) if w)
        trailer_wog = 0
        for trailer in raw.get("trailer", []):
            if not trailer.get("attached", False):
                break
            trailer_wog += sum(1 for w in trailer.get("wheelOnGround", []) if w)
        data.wheels_on_ground = truck_wog + trailer_wog
        data.lv_accelerationX    = raw.get("lv_accelerationX", 0.0)
        data.brake_temperature_c = float(raw.get("brakeTemperature", 0.0) or 0.0)
        data.air_pressure_psi    = float(raw.get("airPressure", 0.0) or 0.0)
        data.wipers              = bool(raw.get("wipers", False))
        try:
            data.surface, data.truck_wheel_slip, data.trailer_wheel_slip = _grip_fields(
                raw, data.speed,
            )
        except (TypeError, ValueError, AttributeError, IndexError):
            logger.debug("grip fields unreadable", exc_info=True)
        data.parkBrake           = raw.get("parkBrake", False)
        data.rotationY           = raw.get("rotationY", 0.0)
        data.hazardsActive       = raw.get("lightsHazards", False)
        data.truck_id            = str(raw.get("truckId", "") or "")
        geometry = geometry_from_sdk(raw)
        if geometry != data.ego_geometry:
            _log_geometry_change(data.truck_id, geometry)
        data.ego_geometry = geometry
        # trailer_count already counts leading slots with wheels and attached;
        # raw["trailer"] is an array of structs, so flat "trailer[0].x" keys never match.
        data.ego_has_trailer = data.trailer_count > 0
        data.estimated_total_mass_kg = compute_estimated_mass_kg(
            data.unitMass,
            data.cargoMass,
            data.fuel,
            trailer_count=data.trailer_count,
        )

class TelemetryThread(BaseThread):
    loop_interval = 0.02   # 50 Hz: game updates ~60 Hz; consumers read at polling_rate
    max_restarts = 2

    def __init__(self) -> None:
        super().__init__(name="telemetry_thread")
        self.data = TelemetryThreadData()
        self.sdk_initialized = False
        self._first = True
        self._manual_start = False
        self._telemetry = None
        self._last_mass_log_mono: float = 0.0

    def setup(self) -> None:
        time.sleep(0.2)
        logger.info("SDK check thread starting...")
        try:
            import truck_telemetry
            self._telemetry = truck_telemetry
            self._telemetry.init()
            self.sdk_initialized = True
            raw = self._telemetry.get_data()
            if not raw.get("sdkActive", False):
                raise Exception("SDK_NOT_ACTIVE")
            with self.data._lock:
                self.data.is_connected = True
            _apply_telemetry(self.data, raw)
            self._manual_start = False
            logger.info("starting in auto start mode")
        except Exception:
            self.sdk_initialized = False
            with self.data._lock:
                self.data.is_connected = False
            self._manual_start = True
            self.data.manual_start = True
            logger.info("starting in manual start mode")
        logger.debug("setup complete")

    def loop(self) -> None:
        if not self.running:
            return
        try:
            if self._telemetry is None:
                import truck_telemetry
                self._telemetry = truck_telemetry
            if not self.sdk_initialized:
                self._telemetry.init()
                self.sdk_initialized = True
            raw = self._telemetry.get_data()
            if not raw.get("sdkActive", False):
                raise Exception("SDK_NOT_ACTIVE")
            if not self.data.is_connected:
                # Imported lazily: keeps this module importable without Qt.
                from ui.popup.popup_window import PopupWindow
                PopupWindow.emit("SDK connected", "MonoCruise is now connected to the game", "c", 3000)
            with self.data._lock:
                self.data.is_connected = True
            _apply_telemetry(self.data, raw)
            Settings.save(values={"last_game": self.data.game})
            now_mono = time.monotonic()
            if now_mono - self._last_mass_log_mono >= _MASS_LOG_INTERVAL_S:
                self._last_mass_log_mono = now_mono
                with self.data._lock:
                    total = self.data.estimated_total_mass_kg
                    unit = self.data.unitMass
                    cargo = self.data.cargoMass
                    fuel_l = self.data.fuel
                    trailer_count = self.data.trailer_count
                # Disabled: trailer JSON debug logging.
        except Exception:
            self.sdk_initialized = False
            with self.data._lock:
                self.data.is_connected = False
            if self._first:
                self._manual_start = True
                with self.data._lock:
                    self.data.manual_start = True
            if (
                Settings.autostart_variable
                and not self._first
                and not self.data.is_connected
                and not self._manual_start
                and not _window_open_on_taskbar()
            ):
                # Set request_quit; main loop waits for viz settle before exit.
                if not self.data.request_quit:
                    logger.info("game disconnected: requesting auto-close")
                    with self.data._lock:
                        self.data.request_quit = True
            elif not self._manual_start:
                self._manual_start = True
                with self.data._lock:
                    self.data.manual_start = True
        self._first = False

    def stay_open_after_disconnect(self) -> None:
        """Cancel a pending auto-close and treat this as a manual session."""
        self._manual_start = True
        with self.data._lock:
            self.data.manual_start = True
            self.data.request_quit = False

    def teardown(self) -> None:
        logger.debug("teardown complete")

