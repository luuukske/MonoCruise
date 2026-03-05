from __future__ import annotations

"""
Sending Thread — owns SCSController and pushes inputs to the game.

Responsibilities:
- Open and manage the SCS shared-memory controller.
- Apply gas/brake exponents and write aforward/abackward to the game.
- Expose toggle_bool() for timed boolean presses (False → True → False).
- Expose set_bool() for persistent boolean overrides.
- Expose change_hazards() for verified hazard toggling with auto-retoggle (max 3).
- Force hazards ON (with a lock) when pedal thread is down or em_stop is True.
"""

import logging
import time
from dataclasses import dataclass, field
import threading

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from .scscontroller import SCSController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

BOOL_PRESS_DURATION: float = 0.1
HAZARD_PRESS_DURATION: float = 0.4
HAZARD_VERIFY_TIMEOUT: float = 0.1
EM_STOP_LOCK_DURATION: float = 0.5
HAZARD_MAX_RETRIGGERS: int = 3

# ---------------------------------------------------------------------------
# Thread data
# ---------------------------------------------------------------------------


@dataclass
class SendingThreadData(ThreadData):
    aforward: float = 0.0
    abackward: float = 0.0
    hazards_active: bool = False
    horn_active: bool = False
    airhorn_active: bool = False
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


# ---------------------------------------------------------------------------
# Internal state helpers
# ---------------------------------------------------------------------------


@dataclass
class _BoolPress:
    """One-shot timed press: True for [duration] seconds, then False."""
    pressing: bool = False
    release_at: float = 0.0


@dataclass
class _HazardState:
    """
    State for the 4-way blinker toggle state machine.

    Sequence per toggle attempt:
      idle      → telemetry != effective_wanted → start press
      pressing  → flasher4way=True for hold_duration, then release
      verifying → flasher4way=False, wait up to HAZARD_VERIFY_TIMEOUT
                  confirmed → done
                  timeout   → retrigger (if retrigger_count < HAZARD_MAX_RETRIGGERS)

    user_wanted: target set by change_hazards(). None = hands-off.
    lock_until:  monotonic time until which hazards are forced ON.
                 Refreshed every tick while the forcing condition is active;
                 expires EM_STOP_LOCK_DURATION after the condition clears.
    hold_duration: how long to hold the toggle press (set per change_hazards call).
    retrigger_count: number of retrigger attempts so far; resets on new request.
    """

    user_wanted: bool | None = None
    hold_duration: float = HAZARD_PRESS_DURATION
    lock_until: float = 0.0
    pressing: bool = False
    press_until: float = 0.0
    verifying: bool = False
    verify_until: float = 0.0
    retrigger_count: int = 0


# ---------------------------------------------------------------------------
# Thread
# ---------------------------------------------------------------------------


class SendingThread(BaseThread):
    loop_interval = 1.0 / Settings.polling_rate
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="sending_thread")
        self.data = SendingThreadData()
        self._controller: SCSController | None = None
        self._lock = threading.Lock()
        # Timed one-shot presses: name → _BoolPress
        self._bool_presses: dict[str, _BoolPress] = {}
        # Persistent overrides: name → bool (applied every tick until cleared)
        self._bool_overrides: dict[str, bool] = {}
        self._hazard = _HazardState()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def toggle_bool(self, name: str, duration: float = BOOL_PRESS_DURATION) -> None:
        """
        One-shot timed press: set *name* True for *duration* seconds, then False.

        Calling again while a press is active extends the release time.
        Default hold duration: 0.1 s.
        """
        with self._lock:
            press = self._bool_presses.setdefault(name, _BoolPress())
            press.pressing = True
            press.release_at = time.monotonic() + max(duration, 0.0)
        logger.debug("toggle_bool: %s for %.3fs", name, duration)

    def set_bool(self, name: str, value: bool = True) -> None:
        """
        Persistently hold *name* at *value* on the controller.

        The override is applied every tick until you call set_bool(name, False)
        or set_bool(name, True) again.  Use this for inputs that should stay
        in a fixed state (e.g. horn held, a mode latch).
        """
        with self._lock:
            self._bool_overrides[name] = value
        logger.debug("set_bool: %s = %s", name, value)

    def change_hazards(self, wanted: bool, duration: float = HAZARD_PRESS_DURATION) -> None:
        """
        Request a 4-way blinker state.

        Compares *wanted* against live telemetry.  If they differ, triggers a
        toggle press of *duration* seconds (default 0.4 s).  Waits
        HAZARD_VERIFY_TIMEOUT (0.1 s) for telemetry to confirm; retriggles if
        not, up to HAZARD_MAX_RETRIGGERS (3) times.  Each retrigger resets the
        full hold duration.

        Requests made while an em_stop lock is active are queued and honoured
        once the lock expires.
        """
        with self._lock:
            self._hazard.user_wanted = wanted
            self._hazard.hold_duration = max(duration, 0.0)
            self._hazard.retrigger_count = 0
        logger.debug("change_hazards: wanted=%s duration=%.3fs", wanted, duration)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

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
        self._controller = SCSController()
        self._reset_controller(self._controller)
        self._hazard = _HazardState()
        logger.debug("SCSController initialised")

    def loop(self) -> None:
        if not self.running:
            return

        self.loop_interval = 1.0 / max(Settings.polling_rate, 10)

        controller = self._controller
        if controller is None:
            return

        # ------------------------------------------------------------------
        # Pedal thread state.
        # ------------------------------------------------------------------
        pedal_thread = registry.get_thread("main_pedal_thread")
        pedal_alive = pedal_thread is not None and pedal_thread.is_alive()

        em_stop = False
        if pedal_thread is not None:
            try:
                with pedal_thread.data._lock:
                    em_stop = bool(pedal_thread.data.em_stop)
            except Exception as e:
                logger.debug("em_stop read failed: %s", e)

        # ------------------------------------------------------------------
        # Telemetry.
        # ------------------------------------------------------------------
        tel_thread = registry.get_thread("telemetry_thread")
        with tel_thread.data._lock:
            connected = tel_thread.data.is_connected
            gear = tel_thread.data.gear_dashboard
            tel_hazards = bool(tel_thread.data.hazards_active)
            speed_ms = tel_thread.data.speed

        # ------------------------------------------------------------------
        # Auto-disable hazards when driving (speed > 12 km/h, gas >= 40%,
        # brake not pressed) — only when autodisable_hazards is enabled.
        # ------------------------------------------------------------------
        if Settings.autodisable_hazards and pedal_thread is not None:
            speed_kmh = speed_ms * 3.6
            with pedal_thread.data._lock:
                gas_pct = pedal_thread.data.gasval
                brake_pct = pedal_thread.data.brakeval
            if speed_kmh > 12.0 and gas_pct >= 0.40 and brake_pct < 0.05:
                with self._lock:
                    self._hazard.user_wanted = False

        # ------------------------------------------------------------------
        # Hazard state machine — always runs regardless of SDK/pedal state.
        # ------------------------------------------------------------------
        should_force = not pedal_alive or em_stop
        self._tick_hazards(controller, tel_hazards, should_force)

        # ------------------------------------------------------------------
        # Persistent bool overrides.
        # ------------------------------------------------------------------
        self._tick_bool_overrides(controller)

        if not connected:
            controller.aforward = 0.0
            controller.abackward = 0.0
            return

        if not pedal_alive:
            return

        # ------------------------------------------------------------------
        # Pedal inputs.
        # ------------------------------------------------------------------
        with pedal_thread.data._lock:
            gas_output = pedal_thread.data.gas_output
            brake_output = pedal_thread.data.brake_output
            gasval = pedal_thread.data.gasval
            brakeval = pedal_thread.data.brakeval

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
            self.data.hazards_active = tel_hazards
            self.data.horn_active = bool(controller.horn)
            self.data.airhorn_active = bool(controller.airhorn)

    def teardown(self) -> None:
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
            self.data.hazards_active = False
            self.data.horn_active = False
            self.data.airhorn_active = False
        logger.debug("teardown complete")

    # ------------------------------------------------------------------ #
    # Tick helpers                                                         #
    # ------------------------------------------------------------------ #

    def _tick_bool_overrides(self, controller: SCSController) -> None:
        """Apply all persistent set_bool overrides to the controller."""
        with self._lock:
            overrides = dict(self._bool_overrides)
        for name, value in overrides.items():
            try:
                setattr(controller, name, value)
            except (AttributeError, OSError, TypeError):
                pass

    def _tick_bool_presses(self, controller: SCSController) -> None:
        """Drive one-shot timed toggle_bool presses."""
        now = time.monotonic()
        with self._lock:
            items = list(self._bool_presses.items())

        for name, press in items:
            if not press.pressing:
                continue
            if now < press.release_at:
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
                    press.pressing = False

    def _tick_hazards(
        self,
        controller: SCSController,
        tel_hazards: bool,
        should_force: bool,
    ) -> None:
        """
        4-way blinker state machine.

        should_force=True (em_stop or pedal thread down):
            Refreshes lock_until each tick; lock expires EM_STOP_LOCK_DURATION
            after the forcing condition clears.  During the lock effective_wanted
            is always True regardless of user_wanted.

        Retrigger cap: after HAZARD_MAX_RETRIGGERS failed attempts the state
        machine gives up and clears user_wanted.
        """
        now = time.monotonic()
        state = self._hazard

        if should_force:
            state.lock_until = now + EM_STOP_LOCK_DURATION

        locked = now < state.lock_until
        with self._lock:
            user_wanted = state.user_wanted

        effective_wanted: bool | None = True if locked else user_wanted

        if effective_wanted and not Settings.hazards_variable:
            effective_wanted = False

        if effective_wanted is None:
            try:
                controller.flasher4way = False
            except (AttributeError, OSError, TypeError):
                pass
            return

        press_flasher = False

        if state.pressing:
            if now < state.press_until:
                press_flasher = True
            else:
                # Trailing edge: release, open verification window.
                state.pressing = False
                state.verifying = True
                state.verify_until = now + HAZARD_VERIFY_TIMEOUT
                logger.debug(
                    "hazard press released, verifying for %.2fs (wanted=%s)",
                    HAZARD_VERIFY_TIMEOUT,
                    effective_wanted,
                )

        elif state.verifying:
            if tel_hazards == effective_wanted:
                logger.debug("hazard confirmed: telemetry=%s", tel_hazards)
                state.verifying = False
                state.retrigger_count = 0
                if not locked:
                    with self._lock:
                        state.user_wanted = None
            elif now >= state.verify_until:
                if state.retrigger_count >= HAZARD_MAX_RETRIGGERS:
                    logger.warning(
                        "hazard retrigger limit (%d) reached, giving up (wanted=%s)",
                        HAZARD_MAX_RETRIGGERS,
                        effective_wanted,
                    )
                    state.verifying = False
                    state.retrigger_count = 0
                    if not locked:
                        with self._lock:
                            state.user_wanted = None
                else:
                    state.retrigger_count += 1
                    logger.debug(
                        "hazard verify timeout: telemetry=%s, wanted=%s — retrigger %d/%d",
                        tel_hazards,
                        effective_wanted,
                        state.retrigger_count,
                        HAZARD_MAX_RETRIGGERS,
                    )
                    state.verifying = False
                    state.pressing = True
                    state.press_until = now + state.hold_duration
                    press_flasher = True

        else:
            # Idle: start a press if telemetry doesn't match.
            if tel_hazards != effective_wanted:
                logger.debug(
                    "hazard starting press: telemetry=%s, wanted=%s",
                    tel_hazards,
                    effective_wanted,
                )
                state.pressing = True
                state.press_until = now + state.hold_duration
                press_flasher = True

        try:
            controller.flasher4way = press_flasher
        except (AttributeError, OSError, TypeError):
            pass