from __future__ import annotations

"""
Sending Thread — owns SCSController and pushes inputs to the game.

Responsibilities:
- Open and manage the SCS shared-memory controller.
- Apply gas/brake exponents and write aforward/abackward to the game.
- Expose toggle_bool() for timed boolean presses (False → True → False).
- Expose set_bool() for persistent boolean overrides.
- Expose change_hazards() for verified hazard toggling with retrigger (max 3).
- Expose start_decel() / stop_decel() for closed-loop deceleration control.
  Other threads (e.g. ACC) call these via registry.get_thread("sending_thread").

DecelController public API (via SendingThread):
  sending_thread.start_decel(target_ms2)  — begin closed-loop braking
  sending_thread.stop_decel()             — cancel immediately
  sending_thread.data.decel_active        — bool
  sending_thread.data.decel_brake_output  — float [0..1]
  sending_thread.data.decel_measured_ms2  — float latest measured decel
"""

import logging
import time
from dataclasses import dataclass, field
import threading

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from .scscontroller import SCSController
from .visualization_bar import VisualizationBar
from .decel_controller import DecelController, TARGET_DEFAULT as DECEL_TARGET_DEFAULT

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
    decel_active: bool = False
    decel_brake_output: float = 0.0
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

        self._decel = DecelController()
        self._decel_target: float = DECEL_TARGET_DEFAULT
        self._decel_key_toggle: bool = False
        self._decel_key_lock = threading.Lock()
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

    def start_decel(self, target_ms2: float = DECEL_TARGET_DEFAULT) -> None:
        """
        Begin closed-loop braking at *target_ms2* (m/s², must be negative).
        Safe to call from any thread via registry.get_thread("sending_thread").start_decel().
        """
        speed = self._read_speed()
        self._decel_target = target_ms2
        self._decel.start(speed)

    def stop_decel(self) -> None:
        """Cancel closed-loop braking immediately."""
        self._decel.cancel()

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

        try:
            from pynput import keyboard as kb

            def _on_press(key):
                try:
                    if hasattr(key, "char") and key.char in ("u", "U"):
                        with self._decel_key_lock:
                            self._decel_key_toggle = True
                except Exception:
                    pass

            self._key_listener = kb.Listener(on_press=_on_press)
            self._key_listener.start()
            logger.info("decel controller: U key registered for test mode")
        except Exception:
            logger.warning(
                "decel controller: pynput unavailable — U-key test disabled",
                extra={"popup": True},
            )
            self._key_listener = None

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
        rotationY = 0.0

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
                    rotationY = tel_thread.data.rotationY
            except Exception as e:
                logger.debug("telemetry read failed: %s", e)

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

        toggle_requested = False
        with self._decel_key_lock:
            if self._decel_key_toggle:
                toggle_requested = True
                self._decel_key_toggle = False

        if toggle_requested:
            if not self._decel.active:
                if connected:
                    self._decel.start(speed_ms)
                else:
                    logger.warning(
                        "decel test: cannot start — SDK not connected",
                        extra={"popup": True},
                    )
            else:
                self._decel.cancel()

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

        if self._decel.active:
            brake_out = self._decel.tick(self._decel_target, self.loop_interval)
            controller.aforward = 0.0
            controller.abackward = brake_out
            self._tick_bool_presses(controller)
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = brake_out
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = self._decel.active
                self.data.decel_brake_output = brake_out
                self.data.decel_measured_ms2 = (
                    self._decel._samples[-1].measured_decel if self._decel._samples else 0.0
                )
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
            if gear == 0:
                a = float(gasval) ** float(gas_exp)
        except Exception:
            a = float(gasval)

        try:
            b = max(b, max(float(brakeval), 0.0) ** float(brake_exp))
        except Exception:
            b = max(b, max(float(brakeval), 0.0))

        b = b ** 0.91
        a = float(complex(a).real)
        b = float(complex(b).real)

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

        self._decel.cancel()

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