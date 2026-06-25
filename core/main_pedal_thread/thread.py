"""
Main Pedal Thread: owns joystick input and computes pedal outputs.

Responsibilities:
  - Initialize and manage all required pygame joysticks:
      * The configured pedal device (critical path: drives brake/gas outputs).
      * Any additional joystick devices referenced by button bindings.
  - Read raw gas and brake axis values every loop tick.
  - Apply the One-Pedal-Drive transformation (disabled while cruise control is commanding).
  - Apply weight-based brake adjustment.
  - Manage the `stopped` hold-brake state and park-brake detection.
  - Detect emergency braking events (sudden pedal slam / crash) and hold
    full brake until the user releases, exposing `em_stop` for the sending thread.
  - Resolve button bindings (joystick buttons, hat directions, keyboard keys) and
    publish cc_*_held booleans for cruise_control_thread to read.
  - Expose joystick_button_states for input_bindings.resolve_held() callers.
  - Expose a capture API (start_capture / cancel_capture / consume_capture) for
    future UI assignment flows.

Does NOT own:
  - Sending values to the game  → sending_thread (reads ThreadData).
  - Hazard / horn actuation     → sending_thread (reads ThreadData flags).
  - Cruise control / ACC        → cruise_control_thread reads CC button holds from this data.
  - AEB / radar                 → aeb_thread / radar_thread.
  - Keyboard hook lifecycle     → keyboard_thread.

Other threads read state via:
  registry.get_thread("main_pedal_thread").data.<field>

Hat direction virtual-button encoding (matches legacy MonoCruise):
  virtual_code = button_count + hat_index * 4 + direction_index
  direction_index: 0=up  1=right  2=down  3=left
  pygame hat xy:   (0,1) (1,0)    (0,-1)  (-1,0)
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
from core.input_bindings import migrate_binding, keyboard_is_pressed, resolve_held

logger = logging.getLogger(__name__)

# Hat direction index → pygame hat (x, y) value
_DIR_IDX_TO_XY: dict[int, tuple[int, int]] = {
    0: (0,  1),   # up
    1: (1,  0),   # right
    2: (0, -1),   # down
    3: (-1, 0),   # left
}


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _opd_interpolate(y_lo: float, y_hi: float, x: float, x_lo: float, x_hi: float) -> float:
    """Linear map: x in [x_lo, x_hi] → y in [y_lo, y_hi]."""
    if abs(x_hi - x_lo) < 1e-12:
        return y_hi
    t = (x - x_lo) / (x_hi - x_lo)
    return y_lo + t * (y_hi - y_lo)


def _onepedaldrive(
    gasval: float,
    brakeval: float,
    *,
    apply_opd_mapping: bool = True,
) -> tuple[float, float]:
    """
    One-pedal mapping: combined axis → gas/brake, with gas and brake exponents applied
    (brake curve uses sqrt(Settings.brake_exponent_variable) as in legacy MonoCruise).
    When *apply_opd_mapping* is False, only the two-pedal path and exponents are
    used: same as legacy opd_mode off.
    """
    offset = float(Settings.offset_variable or 0.0)
    be_full = float(Settings.brake_exponent_variable or 1.0)
    brake_exponent = be_full**0.5
    gas_exponent = float(Settings.gas_exponent_variable or 1.0)
    opd_mode = bool(Settings.opd_mode_variable) and apply_opd_mapping
    max_opd_brake = max(0.0, float(Settings.max_opd_brake_variable or 0.0))

    val1 = min(max(float(gasval), 0.0), 1.0)
    val2 = (min(max(float(brakeval), 0.0), 1.0) ** brake_exponent) * -1.0
    sum_values = val1 + val2

    if opd_mode:
        if sum_values <= offset:
            value = (1.0 / (1.0 + offset)) * sum_values - (offset / (1.0 + offset))
        else:
            denom = 1.0 - offset
            if abs(denom) < 1e-9:
                value = sum_values
            else:
                value = (1.0 / denom) * sum_values - (offset / denom)
    else:
        value = sum_values

    if abs(brake_exponent) < 1e-12:
        a = 0.0
    else:
        a = -(max_opd_brake ** (1.0 / brake_exponent))

    if opd_mode and offset > 1e-12:
        if sum_values >= 0.0 and sum_values <= offset:
            off_exp = offset**brake_exponent
            if abs(off_exp) > 1e-12:
                value = (a / off_exp) * ((-sum_values + offset) ** brake_exponent)

    if sum_values < 0.0 and opd_mode:
        value = _opd_interpolate(-1.0, a, sum_values, -1.0, 0.0)

    gas_out = max(0.0, value)
    brake_out = -min(min(0.0, value), val2)

    gas_out = gas_out**gas_exponent
    brake_out = brake_out**brake_exponent
    return gas_out, brake_out


# ── ThreadData ────────────────────────────────────────────────────────────────

@dataclass
class MainPedalThreadData(ThreadData):
    # Computed outputs: sending_thread reads these every tick.
    gas_output: float = 0.0
    brake_output: float = 0.0

    # Raw pedal values: other threads may read for display or CC logic.
    gasval: float = 0.0
    brakeval: float = 0.0

    # OPD-transformed gas, published before any AEB override so sending_thread
    # can drive its stopped-state FSM off the user's true intent.
    opdgasval: float = 0.0

    # State flags: sending_thread uses these to decide what to send.
    device_lost: bool = True    # True until the *pedal* joystick is successfully opened.
    em_stop: bool = False       # Full emergency brake engaged.

    # Device info: UI / sending thread may display this.
    device_name: str = ""

    # Cruise-control button held states (all button sources unified here).
    cc_start_held: bool = False
    cc_inc_held: bool = False
    cc_dec_held: bool = False
    acc_dist_inc_held: bool = False
    acc_dist_dec_held: bool = False

    # Full joystick button state snapshot: {device_guid: {virtual_code: bool}}.
    # Includes hat directions encoded as virtual button indices.
    # Used by input_bindings.resolve_held() for external callers.
    joystick_button_states: dict = field(default_factory=dict, repr=False)

    # Capture API (for future UI button-assignment flows).
    capture_active: bool = False
    capture_event: object = None  # ("joystick", guid, code) when a joystick input captured

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


# ── Thread ────────────────────────────────────────────────────────────────────

class MainPedalThread(BaseThread):
    loop_interval = 1.0 / Settings.polling_rate
    max_restarts  = 3

    def __init__(self) -> None:
        super().__init__(name="main_pedal_thread")
        self.data = MainPedalThreadData()

        # ── Pedal device (critical path) ──────────────────────────────────────
        self._device: pygame.joystick.JoystickType | None = None
        self._device_instance_id: int | None = None

        # Pedal reconnect state machine: non-blocking, advances each loop() tick.
        # States: None → "initial_wait" → "attempt" → "attempt_wait" → "reinit_wait" → "reinit"
        self._reconnect_state: str | None = None
        self._reconnect_deadline: float = 0.0
        self._reconnect_attempt: int = 0
        self._reconnect_js: pygame.joystick.JoystickType | None = None

        # ── Button devices (non-critical, any GUID referenced by button bindings) ──
        # device_guid → Joystick | None (None = not yet found / lost)
        self._button_devices: dict[str, pygame.joystick.JoystickType | None] = {}
        # device_guid → instance_id (for JOYDEVICEREMOVED matching)
        self._button_instance_ids: dict[str, int] = {}
        # device_guid → is_lost flag
        self._button_device_lost: dict[str, bool] = {}
        # device_guid → human-readable name (for popups)
        self._button_device_names: dict[str, str] = {}

        # ── Joystick state snapshots ──────────────────────────────────────────
        # {guid: {virtual_code: bool}}: updated every tick by _update_joystick_states
        self._joystick_states: dict[str, dict[int, bool]] = {}
        # previous tick's states: used for 0→1 capture detection
        self._prev_capture_states: dict[str, dict[int, bool]] = {}

        # ── Operational state ─────────────────────────────────────────────────
        self._prev_brakeval: float = 0.0
        self._prev_speed: float = 0.0
        self._latency_ts: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        pygame.init()
        pygame.joystick.init()

        # Pedal device (critical).
        pedal_guid = Settings.device
        if pedal_guid:
            js = _find_joystick(pedal_guid)
            if js is not None:
                js.init()
                self._device = js
                self._device_instance_id = js.get_instance_id()
                with self.data._lock:
                    self.data.device_lost = False
                    self.data.device_name = js.get_name()
                logger.info("connected to pedals")
            else:
                logger.warning(
                    "pedals not found. please reconnect or configure again.",
                    extra={"popup": True},
                )
        else:
            logger.info("no joystick configured: running without pedal input")

        # Button devices (non-critical).
        self._init_button_devices()

        self._latency_ts = time.monotonic()
        logger.debug("setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        polling_rate = max(Settings.polling_rate, 10)
        self.loop_interval = 1.0 / polling_rate

        now = time.monotonic()
        latency = now - self._latency_ts
        self._latency_ts = now
        latency_multiplier = (latency / 0.015) * 2

        tel = self._get_telemetry()
        if tel is None:
            with self.data._lock:
                self.data.gas_output  = 0.0
                self.data.brake_output = 0.0
                self.data.opdgasval   = 0.0
                self.data.cc_start_held = False
                self.data.cc_inc_held = False
                self.data.cc_dec_held = False
                self.data.acc_dist_inc_held = False
                self.data.acc_dist_dec_held = False
            return

        speed      = tel["speed"]
        gear       = tel["gear"]
        paused     = tel["paused"]
        slope      = tel["rotationY"]
        park_brake = tel["parkBrake"]
        cargo_mass = tel["cargoMass"]

        # Only treat CC as "commanding the pedal" in Cruise control mode.
        # In Speed limiter mode CC just CAPS the user's gas pedal in
        # sending_thread (`min(user, mapper_gas)`)
        cruise_commanding = False
        try:
            cc = registry.get_thread("cruise_control_thread")
            if cc is not None and cc.is_alive():
                with cc.data._lock:
                    cc_active = bool(cc.data.active)
                    cc_winner = str(cc.data.active_controller)
                # Only treat the CC controller (or ACC) winning arbitration as
                # cruise commanding. A limiter-only win (CC button-disabled,
                # global limit active) must NOT suppress OPD coast-down brake.
                if cc_active and Settings.cc_mode == "Cruise control" and cc_winner == "cc":
                    cruise_commanding = True
        except (KeyError, AttributeError):
            pass

        # Process pygame events (input + hot-plug)
        gasval, brakeval = self._process_pygame_events(speed)

        # Advance pedal reconnect state machine.
        self._tick_reconnect()

        # Update all joystick button/hat state snapshots (button devices + pedals).
        # Also handles capture detection and publishes joystick_button_states.
        self._update_joystick_states()

        # Ensure newly-referenced button device GUIDs are tracked.
        self._ensure_button_devices()

        prev_brakeval = self._prev_brakeval
        prev_speed    = self._prev_speed

        # Device lost guard (pedals only).
        if self.data.device_lost:
            with self.data._lock:
                self.data.gasval      = 0.0
                self.data.brakeval    = 0.0
                self.data.gas_output  = 0.0
                self.data.brake_output = 0.15
                self.data.opdgasval   = 0.0
                self.data.em_stop     = speed > 0.1
                self.data.cc_start_held = False
                self.data.cc_inc_held = False
                self.data.cc_dec_held = False
                self.data.acc_dist_inc_held = False
                self.data.acc_dist_dec_held = False
            return

        # opdgasval is always OPD-mapped: it is the user-side input for the
        # CC override comparison in sending_thread, so the override criterion
        # matches the same OPD feel the driver gets when CC is off. The brake
        # side of this call is discarded when CC is commanding; only the
        # two-pedal output below is sent to the truck in that case.
        opdgasval, opdbrakeval_full = _onepedaldrive(
            gasval, brakeval, apply_opd_mapping=True
        )
        if cruise_commanding:
            # CC owns the longitudinal command: drop OPD coast-down and shaping
            # from the output path so the truck doesn't fight CC. User brake
            # pedal still passes through (two-pedal raw, exponented).
            _, opdbrakeval = _onepedaldrive(
                gasval, brakeval, apply_opd_mapping=False
            )
        else:
            opdbrakeval = opdbrakeval_full

        # Weight adjustment.
        if Settings.weight_adjustment and cargo_mass > 0:
            try:
                total_weight_tons = (cargo_mass / 1000) + 8.93
                weight_var = (0.27 * ((total_weight_tons - 8.93) / 12.7) + 1)
            except Exception:
                weight_var = 1.0
                logger.warning("error calculating weight adjustment")
        else:
            weight_var = 1.0
        opdbrakeval = (opdbrakeval ** (1 / weight_var)).real

        # OPD coast-down brake (user-pedal shaping only). The stopped-state
        # hold brake and its FSM live in sending_thread so they apply to every
        # output source (user, mapper, AEB): `stopped` is read back from there
        # to preserve the original if/elif gating.
        sending_stopped = False
        try:
            st = registry.get_thread("sending_thread")
            if st is not None and st.is_alive():
                with st.data._lock:
                    sending_stopped = bool(st.data.stopped)
        except (KeyError, AttributeError):
            pass

        a = 0.035 - slope / 2
        if (
            not sending_stopped
            and opdgasval == 0
            and Settings.opd_mode_variable
            and not cruise_commanding
            and opdbrakeval < 0.3
        ):
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

        gas_output   = opdgasval
        brake_output = opdbrakeval

        # AEB override: on engagement (AEB_brake), slam brake to 1.0 and zero
        # gas. The graduated FF additive path in sending_thread handles the
        # sub-engagement assist band on top of user braking. Engagement, by
        # definition, means the system has decided full braking is warranted.
        # Full-gas user authority preserved via the gas_output < 0.8 gate.
        try:
            aeb = registry.get_thread("aeb_thread")
            if aeb is not None and aeb.is_alive() and gas_output < 0.8:
                with aeb.data._lock:
                    if aeb.data.AEB_brake:
                        gas_output = 0.0
                        brake_output = 1.0
        except (KeyError, AttributeError):
            pass

        # Emergency stop detection.
        em_stop = self.data.em_stop

        sudden_brake_slam = (
            prev_brakeval - brakeval <= -0.07 * latency_multiplier
            or brakeval >= 0.8
            or park_brake
        )
        crash_detected = prev_speed - speed >= 5

        if sudden_brake_slam and speed > 10 and not paused:
            em_stop  = True
            gas_output   = 0.0
            brake_output = 1.0
            if self.data.em_stop is not True:
                logger.warning(
                    "emergency stop triggered (speed=%.1f km/h, brakeval=%.3f)",
                    speed, brakeval,
                )
        elif crash_detected and not paused:
            gas_output   = 0.0
            brake_output = 1.0

        if em_stop:
            brake_output = 1.0
            gas_output   = 0.0
            still_braking = brakeval > 0.8 or park_brake or (
                prev_brakeval - brakeval <= -0.03 * latency_multiplier
            )
            if not still_braking and not self.data.device_lost:
                em_stop      = False
                gas_output   = 0.0
                brake_output = 0.0

        (
            cc_start_held,
            cc_inc_held,
            cc_dec_held,
            acc_dist_inc_held,
            acc_dist_dec_held,
        ) = self._read_cc_button_states()

        with self.data._lock:
            self.data.gasval      = gasval
            self.data.brakeval    = brakeval
            self.data.gas_output  = gas_output
            self.data.brake_output = brake_output
            self.data.opdgasval   = opdgasval
            self.data.em_stop     = em_stop
            self.data.cc_start_held = cc_start_held
            self.data.cc_inc_held = cc_inc_held
            self.data.cc_dec_held = cc_dec_held
            self.data.acc_dist_inc_held = acc_dist_inc_held
            self.data.acc_dist_dec_held = acc_dist_dec_held

        self._prev_brakeval = brakeval
        self._prev_speed    = speed

    def teardown(self) -> None:
        try:
            if pygame.joystick.get_init():
                pygame.joystick.quit()
            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass
        logger.debug("teardown complete")

    # ── Capture API ───────────────────────────────────────────────────────────

    def start_capture(self) -> None:
        """Enable joystick capture mode: next button/hat press populates capture_event."""
        with self.data._lock:
            self.data.capture_active = True
            self.data.capture_event = None

    def cancel_capture(self) -> None:
        """Abort capture without saving anything."""
        with self.data._lock:
            self.data.capture_active = False
            self.data.capture_event = None

    def consume_capture(self) -> tuple | None:
        """Read + clear the captured joystick event. Returns None if nothing captured.

        Returns ("joystick", guid, virtual_code) on success.
        """
        with self.data._lock:
            ev = self.data.capture_event
            self.data.capture_event = None
            self.data.capture_active = False
            return ev

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_telemetry(self) -> dict | None:
        try:
            tel = registry.get_thread("telemetry_thread")
        except KeyError:
            return None
        with tel.data._lock:
            if not tel.data.is_connected:
                return None
            return {
                "speed":      tel.data.speed * 3.6,
                "gear":       tel.data.gear_dashboard,
                "paused":     tel.data.paused,
                "rotationY":  tel.data.rotationY,
                "parkBrake": tel.data.parkBrake,
                "cargoMass": tel.data.cargoMass,
            }

    # ── Button device management ──────────────────────────────────────────────

    def _collect_button_guids(self) -> set[str]:
        """Return all unique device GUIDs referenced by current joystick button bindings."""
        guids: set[str] = set()
        for name in (
            "cc_start_button", "cc_inc_button", "cc_dec_button",
            "acc_dist_inc_button", "acc_dist_dec_button",
        ):
            raw = getattr(Settings, name)
            b = migrate_binding(raw)
            if b and b.get("source") == "joystick":
                g = b.get("device_guid")
                if g:
                    guids.add(g)
        return guids

    def _init_button_devices(self) -> None:
        """Find and init button-binding devices not yet tracked. Called at setup."""
        pedal_guid = Settings.device or ""
        for guid in self._collect_button_guids():
            if not guid or guid == pedal_guid:
                continue  # pedals handled on the critical path
            if guid in self._button_devices:
                continue
            self._try_connect_button_device(guid, popup_on_missing=True)

    def _ensure_button_devices(self) -> None:
        """During loop: track any GUIDs that appeared since setup (binding reassignment)."""
        pedal_guid = Settings.device or ""
        for guid in self._collect_button_guids():
            if not guid or guid == pedal_guid:
                continue
            if guid not in self._button_devices:
                self._try_connect_button_device(guid, popup_on_missing=True)

    def _try_connect_button_device(self, guid: str, *, popup_on_missing: bool) -> bool:
        """Try to find and init the button device with this GUID. Returns True on success."""
        js = _find_joystick(guid)
        if js is not None:
            try:
                js.init()
                self._button_devices[guid] = js
                self._button_instance_ids[guid] = js.get_instance_id()
                self._button_device_lost[guid] = False
                name = js.get_name()
                self._button_device_names[guid] = name
                logger.info("connected to button device: %s", name)
                return True
            except Exception:
                logger.debug("failed to init button device %s", guid, exc_info=True)

        # Device not available.
        self._button_devices[guid] = None
        self._button_device_lost[guid] = True
        if popup_on_missing:
            name = self._button_device_names.get(guid, guid)
            logger.warning(
                "Button device %r not found: reconnect to use cruise control buttons",
                name,
                extra={"popup": True},
            )
        return False

    # ── Joystick state snapshot ───────────────────────────────────────────────

    def _update_joystick_states(self) -> None:
        """Read all tracked joystick button/hat states and publish them.

        Also performs capture-mode 0→1 transition detection.
        Hat directions are encoded as virtual button indices per the module-level scheme.
        """
        new_states: dict[str, dict[int, bool]] = {}

        # Gather all devices: pedals + button-binding devices.
        pedal_guid = Settings.device or ""
        all_devices: dict[str, pygame.joystick.JoystickType | None] = {}
        if pedal_guid:
            all_devices[pedal_guid] = self._device
        for guid, js in self._button_devices.items():
            all_devices.setdefault(guid, js)

        with self.data._lock:
            capture_active = self.data.capture_active

        for guid, js in all_devices.items():
            # Determine lost status.
            if guid == pedal_guid:
                lost = self.data.device_lost
            else:
                lost = self._button_device_lost.get(guid, True)

            if lost or js is None:
                new_states[guid] = {}
                continue

            try:
                button_count = js.get_numbuttons()
                hat_count    = js.get_numhats()
                states: dict[int, bool] = {}

                for i in range(button_count):
                    states[i] = bool(js.get_button(i))

                for hat_idx in range(hat_count):
                    hat_xy = js.get_hat(hat_idx)
                    base   = button_count + hat_idx * 4
                    for dir_idx, xy in _DIR_IDX_TO_XY.items():
                        states[base + dir_idx] = (hat_xy == xy)

                new_states[guid] = states

                # Capture: detect first 0→1 transition on any tracked device.
                if capture_active:
                    prev = self._prev_capture_states.get(guid, {})
                    for code, held in states.items():
                        if held and not prev.get(code, False):
                            with self.data._lock:
                                if self.data.capture_active and self.data.capture_event is None:
                                    self.data.capture_event = ("joystick", guid, code)
                                    self.data.capture_active = False
                            break

            except Exception:
                logger.debug("failed to read joystick states for %s", guid, exc_info=True)
                new_states[guid] = {}

        self._joystick_states = new_states
        self._prev_capture_states = {k: dict(v) for k, v in new_states.items()}

        with self.data._lock:
            self.data.joystick_button_states = dict(new_states)

    # ── Button binding resolution ─────────────────────────────────────────────

    def _read_cc_button_states(self) -> tuple[bool, bool, bool, bool, bool]:
        """Return (cc_start, cc_inc, cc_dec, acc_dist_inc, acc_dist_dec) held states.

        Bindings are resolved against self._joystick_states (joystick/hat) or via
        keyboard.is_pressed() (keyboard), so results are always current for this tick.
        """
        results: list[bool] = []
        for name in (
            "cc_start_button", "cc_inc_button", "cc_dec_button",
            "acc_dist_inc_button", "acc_dist_dec_button",
        ):
            results.append(resolve_held(getattr(Settings, name)))
        return tuple(results)  # type: ignore[return-value]

    def _resolve_joystick_binding(self, binding: dict) -> bool:
        """Look up the virtual button code in the current tick's joystick state snapshot."""
        guid = binding.get("device_guid")
        code = binding.get("code")
        if not guid or code is None:
            return False
        states = self._joystick_states.get(guid, {})
        return bool(states.get(code, False))

    def _resolve_keyboard_binding(self, binding: dict) -> bool:
        """Ask the keyboard library whether this key is currently held."""
        key = binding.get("code")
        if not key:
            return False
        return keyboard_is_pressed(str(key))

    # ── Pygame event processing ───────────────────────────────────────────────

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
                self._handle_device_removed(event.instance_id)

            elif event.type == pygame.JOYDEVICEADDED:
                self._handle_device_added()

            elif event.type == pygame.JOYAXISMOTION and self._device is not None:
                try:
                    brakeval = _read_axis(self._device, Settings.brakeaxis, Settings.brake_inverted)
                    gasval   = _read_axis(self._device, Settings.gasaxis,   Settings.gas_inverted)
                except Exception as exc:
                    logger.warning("error reading joystick axis: %s", exc)

        return gasval, brakeval

    def _handle_device_removed(self, instance_id: int) -> None:
        """Handle JOYDEVICEREMOVED for both pedal and button devices."""
        # Pedal device.
        if instance_id == self._device_instance_id:
            logger.warning(
                "Pedals disconnected. Please reconnect.",
                extra={"popup": True},
            )
            with self.data._lock:
                self.data.device_lost = True
                self.data.device_name = ""
            self._device = None
            return

        # Button devices.
        for guid, iid in self._button_instance_ids.items():
            if iid == instance_id:
                name = self._button_device_names.get(guid, guid)
                logger.warning(
                    "Button device %r disconnected: reconnect to use cruise control buttons",
                    name,
                    extra={"popup": True},
                )
                self._button_devices[guid] = None
                self._button_device_lost[guid] = True
                return

    def _handle_device_added(self) -> None:
        """Handle JOYDEVICEADDED: attempt reconnect for any lost device."""
        # Pedal reconnect uses the full non-blocking FSM (to avoid axis drift).
        if self.data.device_lost and self._reconnect_state is None:
            logger.info("device added: waiting before pedal reconnect attempt")
            self._reconnect_state    = "initial_wait"
            self._reconnect_deadline = time.monotonic() + 3
            self._reconnect_attempt  = 0
            self._reconnect_js       = None

        # Button devices reconnect immediately (no axis drift concern).
        for guid, lost in list(self._button_device_lost.items()):
            if not lost:
                continue
            js = _find_joystick(guid)
            if js is None:
                continue
            try:
                js.init()
                self._button_devices[guid] = js
                self._button_instance_ids[guid] = js.get_instance_id()
                self._button_device_lost[guid] = False
                name = js.get_name()
                self._button_device_names[guid] = name
                logger.info(
                    "Button device %r reconnected",
                    name,
                    extra={"popup": True},
                )
            except Exception:
                logger.debug("failed to init reconnected button device %s", guid, exc_info=True)

    # ── Pedal reconnect FSM ───────────────────────────────────────────────────

    def _tick_reconnect(self) -> None:
        """
        Advance the pedal reconnect state machine by one tick.  Called every loop()
        iteration: never sleeps, never blocks.

        State transitions:
          initial_wait  → (after 3 s)   → attempt
          attempt       → (js found)    → reinit_wait
                        → (not found)   → attempt_wait  (retry indefinitely)
                        → (exception)   → attempt_wait  (retry indefinitely)
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
                    js.quit()
                    self._reconnect_state    = "reinit_wait"
                    self._reconnect_deadline = now + 4
                else:
                    self._reconnect_attempt += 1
                    self._reconnect_state    = "attempt_wait"
                    self._reconnect_deadline = now + 0.2
            except Exception as exc:
                self._reconnect_attempt += 1
                logger.warning("reconnect attempt %d failed: %s", self._reconnect_attempt, exc)
                if self._reconnect_attempt % 30 == 0:
                    logger.error(
                        "failed to reconnect after multiple attempts.\nPlease reconfigure.",
                        extra={"popup": True},
                    )
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
            except Exception as exc:
                logger.error("reinit failed after reconnect: %s", exc, extra={"popup": True})
                with self.data._lock:
                    self.data.device_lost = True
                    self.data.device_name = ""
                self._device = None
            self._reconnect_js    = None
            self._reconnect_state = None

