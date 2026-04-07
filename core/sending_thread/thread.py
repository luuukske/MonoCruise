from __future__ import annotations

"""
Sending Thread — owns SCSController and pushes inputs to the game.

Responsibilities:
- Open and manage the SCS shared-memory controller.
- Apply pedal shaping and adaptive accel-to-pedal mapping, then write aforward/abackward to the game.
- Expose toggle_bool() for timed boolean presses (False → True → False).
- Expose set_bool() for persistent boolean overrides.
- Expose change_hazards() for verified hazard toggling with retrigger (max 3).
"""

import logging
import time
from dataclasses import dataclass, field
import threading

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from .accel_to_pedals import AccelToPedals
from .brake_efficiency import BrakeEfficiencyTracker
from .scscontroller import SCSController
from .visualization_bar import VisualizationBar

logger = logging.getLogger(__name__)


def create_visualization_bar() -> VisualizationBar:
    """
    Create and return the pedal visualization bar (gas/brake + em_stop).
    Must be called from the Qt main thread.
    """
    return VisualizationBar()

BOOL_PRESS_DURATION: float = 0.1
HAZARD_PRESS_DURATION: float = 0.4
HAZARD_VERIFY_DELAY: float = 0.1
HAZARD_MAX_RETRIGGERS: int = 3


@dataclass
class SendingThreadData(ThreadData):
    aforward: float = 0.0
    abackward: float = 0.0
    hazardsActive: bool = False
    horn_active: bool = False
    airhorn_active: bool = False
    decel_measured_ms2: float = 0.0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


class SendingThread(BaseThread):
    loop_interval = 1.0 / Settings.polling_rate
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="sending_thread")
        self.data = SendingThreadData()
        self._controller: SCSController | None = None
        self._lock = threading.Lock()

        self._bool_presses: dict[str, float] = {}
        self._bool_overrides: dict[str, bool] = {}

        self._hazard_wanted: bool | None = None
        self._hazard_duration: float = HAZARD_PRESS_DURATION
        self._hazard_press_until: float = 0.0
        self._hazard_verify_until: float = 0.0
        self._hazard_retriggers: int = 0
        self._hazard_phase: str = "idle"

        self._last_should_force: bool = False
        self._hazard_user_override: bool = False
        self._prev_tel_hazards: bool = False
        self._accel_mapper = AccelToPedals()
        self._brake_tracker = BrakeEfficiencyTracker()
        self._key_listener = None

    def toggle_bool(self, name: str, duration: float = BOOL_PRESS_DURATION) -> None:
        """One-shot timed press: set *name* True for *duration* seconds, then False."""
        with self._lock:
            self._bool_presses[name] = time.monotonic() + max(duration, 0.0)
        logger.debug("toggle_bool: %s for %.3fs", name, duration)

    def set_bool(self, name: str, value: bool = True) -> None:
        """Persistently hold *name* at *value* on the controller every tick."""
        with self._lock:
            self._bool_overrides[name] = value
        logger.debug("set_bool: %s = %s", name, value)

    def change_hazards(self, wanted: bool, duration: float = HAZARD_PRESS_DURATION) -> None:
        """Request hazards ON or OFF with verification and up to 3 retriggers."""
        with self._lock:
            self._hazard_wanted = wanted
            self._hazard_duration = max(duration, 0.0)
            self._hazard_retriggers = 0
            self._hazard_phase = "idle"
        logger.debug("change_hazards: wanted=%s duration=%.3fs", wanted, duration)

    def reset_accel_mapper_smoothing(self) -> None:
        """Clear mapper smoothing/correction when cruise stops commanding."""
        self._accel_mapper.reset_smoothing()

    def _read_speed(self) -> float:
        try:
            tel = registry.get_thread("telemetry_thread")
            with tel.data._lock:
                return float(tel.data.speed)
        except (KeyError, AttributeError):
            return 0.0

    def _reset_controller(self, controller: SCSController | None) -> None:
        if controller is None:
            return
        try:
            controller.aforward = 0.0
            controller.abackward = 0.0
            controller.flasher4way = False
        except (AttributeError, OSError, TypeError):
            pass
        with self._lock:
            for name in list(self._bool_presses) + list(self._bool_overrides):
                try:
                    setattr(controller, name, False)
                except (AttributeError, OSError, TypeError):
                    pass
        logger.debug("controller outputs reset to defaults")

    def setup(self) -> None:
        logger.info("initialising SCSController shared-memory interface")
        try:
            self._controller = SCSController()
            self._reset_controller(self._controller)
        except Exception:
            logger.exception("SCSController init failed; thread will run with zero output")
            self._controller = None
        self._hazard_phase = "idle"
        self._hazard_wanted = None
        self._last_should_force = False
        self._hazard_user_override = False
        self._prev_tel_hazards = False

        if self._controller is not None:
            logger.debug("SCSController initialised")

    def loop(self) -> None:
        if not self.running:
            return

        self.loop_interval = 1.0 / max(Settings.polling_rate, 10)

        controller = self._controller
        if controller is None:
            return

        try:
            self._loop_body(controller)
        except Exception:
            logger.exception("unexpected error in loop; zeroing outputs")
            try:
                controller.aforward = 0.0
                controller.abackward = 0.0
            except (AttributeError, OSError, TypeError):
                pass
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0

    def _loop_body(self, controller: SCSController) -> None:
        try:
            pedal_thread = registry.get_thread("main_pedal_thread")
        except KeyError:
            pedal_thread = None
            logger.warning("main sending code limited;\nno pedal thread found")

        pedal_alive = pedal_thread is not None and pedal_thread.is_alive()

        em_stop = False
        AEB_brake = False
        AEB_warn = False
        if pedal_thread is not None and pedal_alive:
            try:
                with pedal_thread.data._lock:
                    em_stop = bool(pedal_thread.data.em_stop)
            except Exception as e:
                logger.debug("em_stop read failed: %s", e)
            try:
                aeb_thread = registry.get_thread("aeb_thread")
                if aeb_thread is not None and aeb_thread.is_alive():
                    with aeb_thread.data._lock:
                        AEB_brake = bool(aeb_thread.data.AEB_brake)
                        AEB_warn = bool(aeb_thread.data.AEB_warn)
            except (KeyError, Exception):
                pass

            em_stop = em_stop or AEB_brake

        connected = False
        gear = 0
        tel_hazards = False
        speed_ms = 0.0

        try:
            tel_thread = registry.get_thread("telemetry_thread")
        except KeyError:
            tel_thread = None
            logger.warning("sending thread limited;\nno telemetry thread found")

        if tel_thread is not None and tel_thread.is_alive():
            try:
                with tel_thread.data._lock:
                    connected = tel_thread.data.is_connected
                    gear = tel_thread.data.gear_dashboard
                    tel_hazards = bool(tel_thread.data.hazardsActive)
                    speed_ms = tel_thread.data.speed
            except Exception as e:
                logger.debug("telemetry read failed: %s", e)

        # Resolve cruise_active before mapper so tracker can gate on it.
        cruise_active = False
        try:
            cruise_t = registry.get_thread("cruise_control_thread")
            if cruise_t is not None and cruise_t.is_alive():
                with cruise_t.data._lock:
                    cruise_active = bool(cruise_t.data.active)
        except (KeyError, AttributeError):
            pass

        mapper_gas = 0.0
        mapper_brake = 0.0
        mapper_command_brake = 0.0
        measured_decel_ms2 = 0.0
        if connected and tel_thread is not None and tel_thread.is_alive():
            try:
                with tel_thread.data._lock:
                    wanted_a = float(tel_thread.data.commanded_accel_ms2)
                    raw_a = float(tel_thread.data.lv_accelerationX)
                    mass_kg = float(tel_thread.data.estimated_total_mass_kg)
                    spd_ms = float(tel_thread.data.speed)
                    has_t = bool(tel_thread.data.ego_has_trailer)
                    slope_rad = float(tel_thread.data.rotationY)
                    tel_gear_dashboard = int(tel_thread.data.gear_dashboard)
                    game_throttle = float(tel_thread.data.gameThrottle)
                    game_clutch = float(tel_thread.data.gameClutch)
                measured_decel_ms2 = max(0.0, -raw_a)
                targets = self._accel_mapper.step(
                    wanted_a,
                    raw_a,
                    spd_ms,
                    mass_kg,
                    has_t,
                    slope_rad=slope_rad,
                    cruise_commanding=cruise_active,
                    gear_dashboard=tel_gear_dashboard,
                    game_throttle=game_throttle,
                    game_clutch=game_clutch,
                )
                mapper_gas = float(targets.gas)
                mapper_brake = float(targets.brake)
                mapper_command_brake = float(targets.command_brake)
            except Exception as e:
                logger.debug("accel_mapper step failed: %s", e)
                slope_rad = 0.0

            # Update brake efficiency tracker only during cruise-commanded braking.
            if cruise_active and mapper_command_brake > 0.05:
                self._brake_tracker.update(
                    mapper_command_brake, measured_decel_ms2, speed_ms, slope_rad
                )
            if cruise_active:
                self._brake_tracker.check_warnings()

        speed_kmh = speed_ms * 3.6
        if connected:
            if tel_hazards and not self._prev_tel_hazards:
                with self._lock:
                    wanted_on = self._hazard_wanted is True
                if not wanted_on:
                    self._hazard_user_override = True
            if not tel_hazards:
                self._hazard_user_override = False
            elif speed_kmh <= 12.0:
                self._hazard_user_override = False
            self._prev_tel_hazards = tel_hazards

        if Settings.autodisable_hazards and pedal_thread is not None and pedal_alive:
            try:
                with pedal_thread.data._lock:
                    gas_pct = pedal_thread.data.gasval
                    brake_pct = pedal_thread.data.brakeval
                if (
                    speed_kmh > 12.0
                    and gas_pct >= 0.60
                    and brake_pct < 0.05
                    and not AEB_warn
                    and not self._hazard_user_override
                ):
                    with self._lock:
                        self._hazard_wanted = False
                        if self._hazard_phase == "idle":
                            self._hazard_retriggers = 0
            except Exception as e:
                logger.debug("autodisable_hazards read failed: %s", e)

        should_force = not pedal_alive or em_stop
        if should_force and not self._last_should_force:
            self.change_hazards(True)
        self._last_should_force = should_force

        self._tick_hazards(controller, tel_hazards)
        self._tick_bool_overrides(controller)


        if not connected:
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
            return

        if not pedal_alive:
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
            return

        try:
            with pedal_thread.data._lock:
                gas_output = pedal_thread.data.gas_output
                brake_output = pedal_thread.data.brake_output
                gasval = pedal_thread.data.gasval
                brakeval = pedal_thread.data.brakeval
        except Exception as e:
            logger.debug("pedal read failed: %s", e)
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
            return

        gas_exp = Settings.gas_exponent_variable or 1.0
        brake_exp = Settings.brake_exponent_variable or 1.0

        a = complex(gas_output).real
        b = complex(brake_output).real

        try:
            if gear != 0:
                a = float(gasval) ** float(gas_exp)
        except Exception:
            a = float(gasval)

        try:
            b = max(b, max(float(brakeval), 0.0) ** float(brake_exp))
        except Exception:
            b = max(b, max(float(brakeval), 0.0))

        b = b ** 0.91 # from going from 110% braking to 100% braking intensity
        a = float(complex(a).real)
        b = float(complex(b).real)

        # Cruise / ACC: mapper gas when active; idle-creep brake from mapper whenever connected.
        if cruise_active:
            a = max(a, mapper_gas)
        b = max(b, mapper_brake)

        controller.aforward = a
        controller.abackward = b

        self._tick_bool_presses(controller)

        with self.data._lock:
            self.data.aforward = a
            self.data.abackward = b
            self.data.hazardsActive = tel_hazards
            self.data.horn_active = bool(getattr(controller, "horn", False))
            self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
            self.data.decel_active = False
            self.data.decel_brake_output = 0.0
            self.data.decel_measured_ms2 = 0.0

    def teardown(self) -> None:
        if self._key_listener is not None:
            try:
                self._key_listener.stop()
            except Exception:
                pass
            self._key_listener = None

        if self._controller is not None:
            try:
                self._reset_controller(self._controller)
                self._controller.close()
            except Exception:
                logger.exception("error closing SCSController (suppressed)")
            self._controller = None
        with self.data._lock:
            self.data.aforward = 0.0
            self.data.abackward = 0.0
            self.data.hazardsActive = False
            self.data.horn_active = False
            self.data.airhorn_active = False
            self.data.decel_active = False
            self.data.decel_brake_output = 0.0
            self.data.decel_measured_ms2 = 0.0
        self._accel_mapper.close()
        logger.debug("teardown complete")

    def _tick_bool_overrides(self, controller: SCSController) -> None:
        with self._lock:
            overrides = dict(self._bool_overrides)
        for name, value in overrides.items():
            try:
                setattr(controller, name, value)
            except (AttributeError, OSError, TypeError):
                pass

    def _tick_bool_presses(self, controller: SCSController) -> None:
        now = time.monotonic()
        with self._lock:
            items = list(self._bool_presses.items())

        for name, release_at in items:
            if now < release_at:
                try:
                    setattr(controller, name, True)
                except (AttributeError, OSError, TypeError):
                    pass
            else:
                try:
                    setattr(controller, name, False)
                except (AttributeError, OSError, TypeError):
                    pass
                with self._lock:
                    self._bool_presses.pop(name, None)

    def _tick_hazards(self, controller: SCSController, tel_hazards: bool) -> None:
        """
        Hazard state machine with 3 phases: idle → pressing → verifying.

        - idle: if wanted != telemetry, start pressing.
        - pressing: hold flasher4way=True for the hold duration, then verify.
        - verifying: wait 0.1s, check telemetry. If matched → done. If not → retrigger (up to 3).
        """
        now = time.monotonic()

        with self._lock:
            wanted = self._hazard_wanted

        if wanted is None:
            try:
                controller.flasher4way = False
            except (AttributeError, OSError, TypeError):
                pass
            return

        if wanted and not Settings.hazards_variable:
            try:
                controller.flasher4way = False
            except (AttributeError, OSError, TypeError):
                pass
            return

        if self._hazard_phase == "idle":
            if tel_hazards != wanted:
                logger.debug("hazard: starting press (telemetry=%s, wanted=%s)", tel_hazards, wanted)
                self._hazard_phase = "pressing"
                self._hazard_press_until = now + self._hazard_duration
            else:
                with self._lock:
                    self._hazard_wanted = None
                try:
                    controller.flasher4way = False
                except (AttributeError, OSError, TypeError):
                    pass
                return

        if self._hazard_phase == "pressing":
            if now < self._hazard_press_until:
                try:
                    controller.flasher4way = True
                except (AttributeError, OSError, TypeError):
                    pass
            else:
                try:
                    controller.flasher4way = False
                except (AttributeError, OSError, TypeError):
                    pass
                self._hazard_phase = "verifying"
                self._hazard_verify_until = now + HAZARD_VERIFY_DELAY
                logger.debug("hazard: press released, verifying (wanted=%s)", wanted)

        elif self._hazard_phase == "verifying":
            try:
                controller.flasher4way = False
            except (AttributeError, OSError, TypeError):
                pass

            if tel_hazards == wanted:
                logger.debug("hazard: confirmed (telemetry=%s)", tel_hazards)
                self._hazard_phase = "idle"
                with self._lock:
                    self._hazard_wanted = None
                    self._hazard_retriggers = 0
            elif now >= self._hazard_verify_until:
                if self._hazard_retriggers >= HAZARD_MAX_RETRIGGERS:
                    logger.warning(
                        "hazard: retrigger limit (%d) reached, giving up (wanted=%s)",
                        HAZARD_MAX_RETRIGGERS, wanted,
                    )
                    self._hazard_phase = "idle"
                    with self._lock:
                        self._hazard_wanted = None
                        self._hazard_retriggers = 0
                else:
                    self._hazard_retriggers += 1
                    logger.debug(
                        "hazard: verify timeout, retrigger %d/%d (telemetry=%s, wanted=%s)",
                        self._hazard_retriggers, HAZARD_MAX_RETRIGGERS, tel_hazards, wanted,
                    )
                    self._hazard_phase = "pressing"
                    self._hazard_press_until = now + self._hazard_duration
