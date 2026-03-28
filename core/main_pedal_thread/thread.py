"""
Main Pedal Thread — owns joystick input and computes pedal outputs.

Responsibilities:
  - Initialize and manage the pygame joystick (connect / disconnect / reconnect).
  - Read raw gas and brake axis values every loop tick.
  - Apply the One-Pedal-Drive transformation (stub — full OPD logic is a TODO).
  - Apply weight-based brake adjustment.
  - Manage the `stopped` hold-brake state and park-brake detection.
  - Detect emergency braking events (sudden pedal slam / crash) and hold
    full brake until the user releases, exposing `em_stop` for the sending thread.

Does NOT own:
  - Sending values to the game  → sending_thread (reads ThreadData).
  - Hazard / horn actuation     → sending_thread (reads ThreadData flags).
  - Cruise control / ACC        → cruise_control_thread reads CC button holds from this data.
  - AEB / radar                 → future feature thread.
  - Live visualization          → future feature thread.

Other threads read state via:
  registry.get_thread("main_pedal_thread").data.<field>
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import pygame

from ui.popup.popup_window import PopupWindow
from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_axis(device: pygame.joystick.JoystickType, axis: int, inverted: bool) -> float:
    """Return a normalised [0.0, 1.0] value from a joystick axis."""
    raw = device.get_axis(axis)
    if inverted:
        raw = -raw
    return round((raw + 1) / 2, 3)


def _find_joystick(guid_hex: str) -> pygame.joystick.JoystickType | None:
    """Return the first joystick whose GUID matches *guid_hex*, or None."""
    for i in range(pygame.joystick.get_count()):
        try:
            js = pygame.joystick.Joystick(i)
            if js.get_guid() == guid_hex:
                return js
        except Exception as exc:
            logger.debug("skipping joystick %d during enumeration: %s", i, exc)
    return None


def _onepedaldrive(gasval: float, brakeval: float) -> tuple[float, float]:
    """
    Transform raw gas/brake inputs into OPD gas/brake outputs.

    # TODO: implement full OPD logic using Settings.offset_variable,
    #       Settings.opd_mode_variable, and Settings.max_opd_brake_variable.
    """
    return gasval, brakeval


# ---------------------------------------------------------------------------
# ThreadData
# ---------------------------------------------------------------------------

@dataclass
class MainPedalThreadData(ThreadData):
    # Computed outputs — sending_thread reads these every tick.
    gas_output: float = 0.0
    brake_output: float = 0.0

    # Raw pedal values — other threads may read for display or CC logic.
    gasval: float = 0.0
    brakeval: float = 0.0

    # State flags — sending_thread uses these to decide what to send.
    device_lost: bool = True    # True until a joystick is successfully opened.
    em_stop: bool = False       # Full emergency brake engaged.
    stopped: bool = False       # Vehicle is in hold-brake "stopped" state.

    # Device info — UI / sending thread may display this.
    device_name: str = ""

    # Cruise-control buttons (read on the pygame thread only). Hat/keyboard bindings: future.
    cc_start_held: bool = False
    cc_inc_held: bool = False
    cc_dec_held: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Thread
# ---------------------------------------------------------------------------

class MainPedalThread(BaseThread):
    loop_interval = 1.0 / Settings.polling_rate
    max_restarts  = 3

    def __init__(self) -> None:
        super().__init__(name="main_pedal_thread")
        self.data = MainPedalThreadData()

        # Joystick state (private to this thread).
        self._device: pygame.joystick.JoystickType | None = None
        self._device_instance_id: int | None = None

        # Operational state (private).
        self._prev_brakeval: float = 0.0
        self._prev_speed: float = 0.0
        self._prev_opdbrakeval: float = 0.0
        self._prev_stop: float = 0.0    # monotonic timestamp of stop event start
        self._latency_ts: float = 0.0   # monotonic timestamp of previous loop tick

        # Reconnect state machine — advances across normal loop() ticks, no sleeping.
        # States: None → "initial_wait" → "attempt" → "attempt_wait" → "reinit_wait" → "reinit"
        self._reconnect_state: str | None = None
        self._reconnect_deadline: float = 0.0   # monotonic time to leave a wait state
        self._reconnect_attempt: int = 0
        self._reconnect_js: pygame.joystick.JoystickType | None = None  # held during reinit wait

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        pygame.init()
        pygame.joystick.init()

        guid = Settings.device
        if guid:
            js = _find_joystick(guid)
            if js is not None:
                js.init()
                self._device = js
                self._device_instance_id = js.get_instance_id()
                with self.data._lock:
                    self.data.device_lost = False
                    self.data.device_name = js.get_name()
                logger.info("connected to pedals")
            else:
                logger.warning("pedals not found. please reconnect or configure again.", extra={"popup": True})
        else:
            logger.info("no joystick configured — running without pedal input")

        self._latency_ts = time.monotonic()
        logger.debug("setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        # Keep loop_interval in sync with the live setting value.
        polling_rate = max(Settings.polling_rate, 10)
        self.loop_interval = 1.0 / polling_rate

        # --- Latency tracking (used to scale emergency-brake thresholds) ------
        now = time.monotonic()
        latency = now - self._latency_ts
        self._latency_ts = now
        # Multiplier mirrors the old code: scales relative to a 15 ms baseline.
        latency_multiplier = (latency / 0.015) * 2

        # --- Read telemetry ---------------------------------------------------
        tel = self._get_telemetry()
        if tel is None:
            # SDK not connected; zero outputs and wait.
            with self.data._lock:
                self.data.gas_output  = 0.0
                self.data.brake_output = 0.0
                self.data.cc_start_held = False
                self.data.cc_inc_held = False
                self.data.cc_dec_held = False
            # TODO: live visualization — clear frame here
            return

        speed      = tel["speed"]
        gear       = tel["gear"]
        paused     = tel["paused"]
        slope      = tel["rotationY"]
        park_brake = tel["parkBrake"]
        cargo_mass = tel["cargoMass"]  # kg

        # --- Process pygame events (input + hot-plug) -------------------------
        gasval, brakeval = self._process_pygame_events(speed)

        # --- Advance reconnect state machine (no sleeping) --------------------
        self._tick_reconnect()

        # Convenience snapshot for stopping logic below.
        prev_brakeval = self._prev_brakeval
        prev_speed    = self._prev_speed

        # --- Device lost guard ------------------------------------------------
        if self.data.device_lost:
            with self.data._lock:
                self.data.gasval      = 0.0
                self.data.brakeval    = 0.0
                self.data.gas_output  = 0.0
                self.data.brake_output = 0.15  # slight brake so vehicle slows gently
                self.data.em_stop     = speed > 0.1
                self.data.cc_start_held = False
                self.data.cc_inc_held = False
                self.data.cc_dec_held = False
            # TODO: hazards/horn actuation on device_lost (handled by sending_thread)
            # TODO: live visualization — clear frame here
            return

        # --- One-Pedal-Drive transform ----------------------------------------
        opdgasval, opdbrakeval = _onepedaldrive(gasval, brakeval)
        # Cruise blends mapper output in sending_thread (commanded_accel_ms2).

        # --- Weight adjustment ------------------------------------------------
        if Settings.weight_adjustment and cargo_mass > 0:
            try:
                total_weight_tons = (cargo_mass / 1000) + 8.93  # approx truck base weight
                weight_var = (0.27 * ((total_weight_tons - 8.93) / 12.7) + 1)
            except Exception:
                weight_var = 1.0
                logger.warning("error calculating weight adjustment")
        else:
            weight_var = 1.0
        opdbrakeval = (opdbrakeval ** (1 / weight_var)).real

        # --- Stopping logic ---------------------------------------------------
        effective_gas = max(gasval, opdgasval)
        offset = Settings.offset_variable
        a = 0.035 - slope / 2

        stopped = self.data.stopped

        if stopped:
            if gear > 0 and speed < 3 and effective_gas <= (0.7 + offset * 0.7) and effective_gas != 0:
                opdbrakeval += min(
                    0.03 * (((-round(speed + 0.8, 1) + 4) ** 5) / (4 ** 5)) + slope * 2,
                    0.3,
                )
            elif gear < 0 and speed > -3 and effective_gas <= (0.7 + offset * 0.7) and effective_gas != 0:
                opdbrakeval += min(
                    0.03 * (((round(speed + 0.8, 1) + 4) ** 5) / (4 ** 5)) - slope * 2,
                    0.3,
                )
            elif effective_gas == 0 and gear != 0:
                opdbrakeval += 0.06
            delta_time = time.monotonic() - self._prev_stop
            t = 0.5
            if self._prev_stop != 0 and delta_time < t:
                opdbrakeval = (
                    opdbrakeval * (delta_time / t)
                    + self._prev_opdbrakeval * (1 - delta_time / t)
                )
            else:
                self._prev_stop = 0
        elif opdgasval == 0 and Settings.opd_mode_variable and opdbrakeval < 0.3:
            if speed > 0:
                b = max(opdbrakeval ** 0.8 / 2, 0.3)
                opdbrakeval = max(
                    opdbrakeval * ((-1 / (b * speed + 1)) + 1)
                    + a * (1 - (-1 / (b * speed + 1) + 1)),
                    0,
                )
            elif speed < 0:
                b = max(opdbrakeval ** 0.8 / 2, 0.3)
                opdbrakeval = max(
                    opdbrakeval * ((-1 / (b * -speed + 1)) + 1)
                    + a * (1 - (-1 / (b * -speed + 1) + 1)),
                    0,
                )

        # --- Stopped state transitions ----------------------------------------
        if speed <= 0.1 and speed >= -0.1 and gasval == 0 and gear != 0 and not stopped:
            stopped = True
            self._prev_opdbrakeval = opdbrakeval
            self._prev_stop = time.monotonic()
        elif stopped and ((speed >= 4 and gear > 0) or (speed <= -4 and gear < 0)):
            stopped = False
            self._prev_stop = 0
        elif stopped and opdgasval > 0.75:
            stopped = False
            self._prev_stop = 0
        if park_brake and -2 <= speed <= 2 and not stopped:
            stopped = True
            self._prev_opdbrakeval = opdbrakeval
            self._prev_stop = time.monotonic()

        gas_output   = opdgasval
        brake_output = opdbrakeval

        # AEB override — when AEB thread requests emergency brake, apply full brake this tick.
        try:
            aeb = registry.get_thread("aeb_thread")
            if aeb is not None and aeb.is_alive() and gas_output < 0.8:
                with aeb.data._lock:
                    if aeb.data.AEB_brake:
                        gas_output = 0.0
                        brake_output = 1.0
        except (KeyError, AttributeError):
            pass

        # --- Emergency stop detection -----------------------------------------
        em_stop = self.data.em_stop

        sudden_brake_slam = (
            prev_brakeval - brakeval <= -0.07 * latency_multiplier
            or brakeval >= 0.8
            or park_brake
        )
        crash_detected = prev_speed - speed >= 5

        if sudden_brake_slam and not stopped and speed > 10 and not paused:
            stopped  = True
            em_stop  = True
            gas_output   = 0.0
            brake_output = 1.0
            if self.data.em_stop is not True:
                logger.warning("emergency stop triggered (speed=%.1f km/h, brakeval=%.3f)", speed, brakeval)
            # TODO: hazards/horn actuation — sending_thread reads em_stop flag
        elif crash_detected and not paused:
            stopped  = True
            gas_output   = 0.0
            brake_output = 1.0
            # TODO: hazards actuation — sending_thread reads stopped + em_stop flags

        if em_stop:
            # Maintain full brake each tick until the pedal is released.
            # No inner loop — loop() returns normally every tick so the
            # base class can update the heartbeat and the watchdog stays happy.
            brake_output = 1.0
            gas_output   = 0.0

            still_braking = brakeval > 0.8 or park_brake or (
                prev_brakeval - brakeval <= -0.03 * latency_multiplier
            )
            if not still_braking and not self.data.device_lost:
                em_stop      = False
                gas_output   = 0.0
                brake_output = 0.0

        cc_start_held, cc_inc_held, cc_dec_held = self._read_cc_button_states()

        # --- Write outputs ----------------------------------------------------
        with self.data._lock:
            self.data.gasval      = gasval
            self.data.brakeval    = brakeval
            self.data.gas_output  = gas_output
            self.data.brake_output = brake_output
            self.data.stopped     = stopped
            self.data.em_stop     = em_stop
            self.data.cc_start_held = cc_start_held
            self.data.cc_inc_held = cc_inc_held
            self.data.cc_dec_held = cc_dec_held

        self._prev_brakeval = brakeval
        self._prev_speed    = speed

        # TODO: live visualization — update frame here

    def teardown(self) -> None:
        try:
            if pygame.joystick.get_init():
                pygame.joystick.quit()
            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass
        logger.debug("teardown complete")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_telemetry(self) -> dict | None:
        """Return a lightweight snapshot of the fields we need from telemetry_thread.
        Returns None when the SDK is not connected."""
        try:
            tel = registry.get_thread("telemetry_thread")
        except KeyError:
            return None
        with tel.data._lock:
            if not tel.data.is_connected:
                return None
            return {
                "speed":      tel.data.speed * 3.6,  # m/s → km/h
                "gear":       tel.data.gear_dashboard,
                "paused":     tel.data.paused,
                "rotationY":  tel.data.rotationY,
                "parkBrake": tel.data.parkBrake,
                "cargoMass": tel.data.cargoMass,
            }

    def _read_cc_button_states(self) -> tuple[bool, bool, bool]:
        """Return (cc_start, cc_inc, cc_dec) held states; plain button indices only."""
        if self._device is None:
            return False, False, False
        try:
            n = self._device.get_numbuttons()
        except Exception:
            return False, False, False

        def pressed(spec: object) -> bool:
            if spec is None or not isinstance(spec, int):
                return False
            if spec < 0 or spec >= n:
                return False
            try:
                return bool(self._device.get_button(spec))
            except Exception:
                return False

        return (
            pressed(Settings.cc_start_button),
            pressed(Settings.cc_inc_button),
            pressed(Settings.cc_dec_button),
        )

    def _process_pygame_events(self, speed: float) -> tuple[float, float]:
        """Handle all pending pygame events and return the current (gasval, brakeval)."""
        gasval   = self.data.gasval
        brakeval = self.data.brakeval

        if not pygame.get_init():
            pygame.init()
            pygame.joystick.init()
            return gasval, brakeval

        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEREMOVED:
                if event.instance_id == self._device_instance_id:
                    logger.warning("Pedals disconnected. Please reconnect.", extra={"popup": True})
                    with self.data._lock:
                        self.data.device_lost = True
                        self.data.device_name = ""
                    self._device = None
                    # TODO: main window behavior — bring window to front on disconnect

            elif event.type == pygame.JOYDEVICEADDED:
                if self.data.device_lost and self._reconnect_state is None:
                    logger.info("device added — waiting before reconnect attempt")
                    self._reconnect_state   = "initial_wait"
                    self._reconnect_deadline = time.monotonic() + 3
                    self._reconnect_attempt  = 0
                    self._reconnect_js       = None

            elif event.type == pygame.JOYAXISMOTION and self._device is not None:
                try:
                    brakeval = _read_axis(self._device, Settings.brakeaxis, Settings.brake_inverted)
                    gasval   = _read_axis(self._device, Settings.gasaxis,   Settings.gas_inverted)
                except Exception as exc:
                    logger.warning("error reading joystick axis: %s", exc)

        return gasval, brakeval

    def _tick_reconnect(self) -> None:
        """
        Advance the reconnect state machine by one tick.  Called every loop()
        iteration — never sleeps, never blocks, so the heartbeat is always updated.

        State transitions:
          initial_wait  → (after 3 s)   → attempt
          attempt       → (js found)    → reinit_wait
                        → (not found)   → attempt_wait  or  None (give up)
                        → (exception)   → attempt_wait  or  None (give up)
          attempt_wait  → (after 0.2 s) → attempt
          reinit_wait   → (after 4 s)   → reinit
          reinit        →               → None (done)
        """
        if self._reconnect_state is None:
            return

        now = time.monotonic()

        if self._reconnect_state in ("initial_wait", "attempt_wait"):
            if now >= self._reconnect_deadline:
                self._reconnect_state = "attempt"
            return

        if self._reconnect_state == "attempt":
            guid = Settings.device
            try:
                js = _find_joystick(guid) if guid else None
                if js is not None:
                    js.init()
                    self._reconnect_js           = js
                    self._device                 = js
                    self._device_instance_id     = js.get_instance_id()
                    try:
                        brakeval = _read_axis(js, Settings.brakeaxis, Settings.brake_inverted)
                        gasval   = _read_axis(js, Settings.gasaxis,   Settings.gas_inverted)
                    except Exception:
                        brakeval = 0.0
                        gasval   = 0.0
                    with self.data._lock:
                        self.data.brakeval    = brakeval
                        self.data.gasval      = gasval
                        self.data.device_name = js.get_name()
                    # quit then wait before re-init to avoid axis drift
                    js.quit()
                    self._reconnect_state    = "reinit_wait"
                    self._reconnect_deadline = now + 4
                else:
                    self._reconnect_attempt += 1
                    # Keep retrying indefinitely; surface a popup periodically so the user
                    # knows they may need to reconfigure, but do not stop the state machine.
                    self._reconnect_state    = "attempt_wait"
                    self._reconnect_deadline = now + 0.2
            except Exception as exc:
                self._reconnect_attempt += 1
                logger.warning("reconnect attempt %d failed: %s", self._reconnect_attempt, exc)
                # Keep retrying indefinitely on errors as well. Notify the user
                # occasionally, but do not give up automatically.
                if self._reconnect_attempt % 30 == 0:
                    logger.error("failed to reconnect after multiple attempts.\nPlease reconfigure.", extra={"popup": True})
                self._reconnect_state    = "attempt_wait"
                self._reconnect_deadline = now + 0.2
            return

        if self._reconnect_state == "reinit_wait":
            if now >= self._reconnect_deadline:
                self._reconnect_state = "reinit"
            return

        if self._reconnect_state == "reinit":
            try:
                self._reconnect_js.init()
                with self.data._lock:
                    self.data.device_lost = False
                PopupWindow.emit("Pedals reconnected", "Pedals reconnected to pedals", "c", 2000)
                # TODO: main window behavior — update connected joystick label
            except Exception as exc:
                logger.error("reinit failed after reconnect: %s", exc, extra={"popup": True})
                with self.data._lock:
                    self.data.device_lost = True
                    self.data.device_name = ""
                self._device = None
            self._reconnect_js    = None
            self._reconnect_state = None
