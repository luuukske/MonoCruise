"""
Telemetry thread — checks ETS2 SDK connection and exposes sdkActive state + game data.

Other threads read:
  registry.get_thread("telemetry_thread").data.is_connected     — SDK active
  registry.get_thread("telemetry_thread").data.manual_start     — user started app before game
  registry.get_thread("telemetry_thread").data.request_quit     — thread requests app shutdown
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
import threading

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from ui.popup.popup_window import PopupWindow

from core.settings import Settings


logger = logging.getLogger(__name__)


def _window_open_on_taskbar() -> bool:
    """Return True when the main window reports it is visible to the user."""
    try:
        main_window = registry.get("main_window")
    except KeyError:
        return False
    return bool(getattr(main_window, "is_open_on_taskbar", False))


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

    # Truck motion
    speed: float = 0.0          # m/s — convert to km/h: speed * 3.6
    cruise_control_speed: float = 0.0  # m/s, 0.0 when CC inactive

    # Engine
    engine_rpm: float = 0.0
    engine_rpm_max: float = 2000.0

    # Gear
    gear: int = 0               # current gear (negative = reverse, 0 = neutral)
    gear_dashboard: int = 0     # displayed gear on dashboard

    # Raw inputs
    user_throttle: float = 0.0
    user_brake: float = 0.0
    user_steer: float = 0.0

    # Game-applied inputs
    game_throttle: float = 0.0
    game_brake: float = 0.0

    # Cargo
    cargo_mass: float = 0.0     # kg

    # Vehicle state
    park_brake: bool = False
    slope: float = 0.0          # rotationY — positive = uphill
    hazards_active: bool = False

    request_quit: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


def _apply_telemetry(data: TelemetryThreadData, raw: dict) -> None:
    """Write all telemetry fields under the data lock."""
    with data._lock:
        data.game                 = raw.get("game", 0)
        data.game_version_major   = raw.get("telemetry_version_game_major", 0)
        data.game_version_minor   = raw.get("telemetry_version_game_minor", 0)
        data.sdk_version          = raw.get("telemetry_plugin_revision", 0)
        data.paused               = raw.get("paused", False)
        data.speed                = raw.get("speed", 0.0)
        data.cruise_control_speed = raw.get("cruiseControlSpeed", 0.0)
        data.engine_rpm           = raw.get("engineRpm", 0.0)
        data.engine_rpm_max       = raw.get("engineRpmMax", 2000.0)
        data.gear                 = raw.get("gear", 0)
        data.gear_dashboard       = raw.get("gearDashboard", 0)
        data.user_throttle        = raw.get("userThrottle", 0.0)
        data.user_brake           = raw.get("userBrake", 0.0)
        data.user_steer           = raw.get("userSteer", 0.0)
        data.game_throttle        = raw.get("gameThrottle", 0.0)
        data.game_brake           = raw.get("gameBrake", 0.0)
        data.hazards_active       = raw.get("lightsHazards", False)
        data.cargo_mass           = raw.get("cargoMass", 0.0)
        data.park_brake           = raw.get("parkBrake", False)
        data.slope                = raw.get("rotationY", 0.0)


class TelemetryThread(BaseThread):
    loop_interval = 0.02
    max_restarts = 2

    def __init__(self) -> None:
        super().__init__(name="telemetry_thread")
        self.data = TelemetryThreadData()
        self.sdk_initialized = False
        self._first = True
        self._manual_start = False
        self._telemetry = None

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
                PopupWindow.emit("SDK connected", "MonoCruise is now connected to the game", "c", 3000)
            with self.data._lock:
                self.data.is_connected = True
            _apply_telemetry(self.data, raw)
            Settings.save(values={"last_game": self.data.game})
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
                logger.info("shutting down")
                with self.data._lock:
                    self.data.request_quit = True
                self._stop_event.set()
                return
            elif not self._manual_start:
                self._manual_start = True
                with self.data._lock:
                    self.data.manual_start = True
        self._first = False

    def teardown(self) -> None:
        logger.debug("teardown complete")