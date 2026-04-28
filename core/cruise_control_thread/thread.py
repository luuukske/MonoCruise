"""
Cruise control worker — CC button edge/timing logic and minimal speed→accel loop.

Reads CC button holds from main_pedal_thread, writes commanded_accel_ms2 on
telemetry_thread for the accel-to-pedals mapper in sending_thread.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

from core.settings import Settings
from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from ui.popup.popup_window import PopupWindow

from .acc_controller import AdaptiveCruiseController

logger = logging.getLogger(__name__)

# Button timing (legacy MonoCruise main_cruise_control)
_LONG_PRESS_DEC_INC_FIRST_S = 0.3
_LONG_PRESS_DEC_INC_REPEAT_S = 0.6
_LONG_PRESS_START_S = 0.5

_SPEED_MIN_KMH = 30.0
_SPEED_MAX_KMH = 130.0

# Default PID constants — overridden at runtime by Settings.cc_k* / Settings.cc_accel_*
_KP_DEFAULT = 0.35
_KI_DEFAULT = 0.00
_KD_DEFAULT = 0.15
_INTEGRAL_CLAMP_DEFAULT = 3.0
_ACCEL_MIN_MS2_DEFAULT = -2.0
_ACCEL_MAX_MS2_DEFAULT = 2.0

# Low-pass time constant (s) for telemetry speed used only in the PID D-term.
# Reduces derivative noise from game telemetry; P/I still use instantaneous speed.
_CC_KD_SPEED_SMOOTH_TAU_S = 0.15

# EMA time constant (s) for the final commanded acceleration sent to telemetry.
# Slower than KD smoothing so pedal commands change gradually.
_CC_OUTPUT_EMA_TAU_S = 0.40
_CC_TARGET_SPEED_EMA_TAU_S = 0.5

# Gearshift D-term freeze. Mirrors the timings used by the sending_thread mapper
# (accel_to_pedals._GEARSHIFT_BLOCK_DURATION_S / _RAMP_DURATION_S) so CC and
# mapper release together. Speed stalls while the clutch is in and jumps on
# re-engage — a raw D-term would spike and send a transient gas/brake command.
_CC_CLUTCH_ACTIVE_THRESHOLD = 0.05
_CC_GEARSHIFT_BLOCK_DURATION_S = 0.5
_CC_GEARSHIFT_RAMP_DURATION_S = 1.0

# Disable-on-stop guard. CC only disables when an *event* (crash or AEB_brake)
# is followed by the vehicle coming to a stop. A normal stop (e.g., user just
# braking at a light) does not disable — CC stays armed and resumes when the
# user releases the brake. Re-enable happens when the user presses CC start/inc.
_CC_DISARM_SPEED_MS = 0.3   # ~1.1 km/h
# Per-tick speed drop (m/s) interpreted as a crash. Mirrors the threshold used
# by main_pedal_thread for its crash_detected branch.
_CC_CRASH_SPEED_DROP_MS = 5.0
# Window after a triggering event during which a stop will disable CC. If the
# vehicle stays moving past this, the event is forgotten.
_CC_DISARM_PENDING_TIMEOUT_S = 5.0

# Game throttle threshold above which CC bypasses output smoothing on the
# brake side so it reacts immediately to a user override.
_CC_GAME_THROTTLE_OVERRIDE = 0.1


@dataclass
class CruiseControlThreadData(ThreadData):
    """Published state for sending_thread / future UI."""

    active: bool = False
    cc_enabled: bool = False
    target_speed_kmh: float | None = None
    wanted_accel_ms2: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class CruiseControlThread(BaseThread):
    loop_interval = 1.0 / Settings.polling_rate
    max_restarts = 5

    def __init__(self) -> None:
        super().__init__(name="cruise_control_thread")
        self.data = CruiseControlThreadData()
        self._acc = AdaptiveCruiseController()

        self._cc_enabled = False
        self._target_speed_kmh: float | None = None

        self._time_pressed_dec: float | None = None
        self._time_pressed_inc: float | None = None
        self._time_pressed_start: float | None = None
        self._time_pressed_acc_dist_inc: float | None = None
        self._time_pressed_acc_dist_dec: float | None = None
        self._long_press_dec = False
        self._long_press_inc = False
        self._long_press_start = False
        self._long_press_acc_dist_inc = False
        self._long_press_acc_dist_dec = False

        self._integral_error = 0.0
        self._kd_smooth_speed_ms: float | None = None
        self._prev_loop_mono = time.monotonic()
        self._was_commanding = False
        self._last_target_for_integral: float | None = None
        self._last_assign_warn_mono: float = 0.0
        self._last_block_msg_mono: float = 0.0
        self._output_ema_accel_ms2: float | None = None
        self._target_speed_ema_ms: float | None = None

        # Clutch/gearshift tracking for D-term freeze
        self._cc_clutch_active: bool = False
        self._cc_clutch_release_mono: float = -math.inf
        self._cc_prev_d_factor: float = 1.0

        # Disarm-on-stop state. True transiently in button FSM resets; the
        # disable-on-stop guard sets _cc_enabled=False directly instead.
        self._cc_disarmed: bool = False
        # Disarm-trigger tracking. `_cc_disarm_pending_until` is a monotonic
        # deadline — set when a crash or AEB_brake event is observed; if the
        # vehicle stops before then, CC disarms.
        self._cc_disarm_pending_until: float = 0.0
        self._cc_prev_speed_ms: float | None = None

    def setup(self) -> None:
        self._prev_loop_mono = time.monotonic()
        logger.debug("cruise_control_thread setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        # Idle throttle: when telemetry is disconnected, button presses are
        # ignored (guarded by `connected` below) and no accel command is
        # published, so full polling_rate is wasted CPU. Drop to 2 Hz.
        try:
            tel = registry.get_thread("telemetry_thread")
            is_connected = tel.is_alive() and bool(tel.data.is_connected)
        except (KeyError, AttributeError):
            is_connected = False
        if is_connected:
            self.loop_interval = 1.0 / max(Settings.polling_rate, 10)
        else:
            self.loop_interval = 0.5

        now = time.monotonic()
        dt = max(now - self._prev_loop_mono, 1e-4)
        self._prev_loop_mono = now

        tel = self._snapshot_telemetry()
        pedal = self._snapshot_pedal()

        commanding = False
        wanted_accel = 0.0

        try:
            if tel is None or pedal is None:
                self._publish_telemetry_command(0.0)
                self._publish_data(False, 0.0)
                self._maybe_reset_mapper_on_commanding_end(False)
                return

            connected = tel["is_connected"]
            paused = tel["paused"]
            em_stop = pedal["em_stop"]
            device_lost = pedal["device_lost"]
            cc_dec = pedal["cc_dec_held"]
            cc_inc = pedal["cc_inc_held"]
            cc_start = pedal["cc_start_held"]
            acc_dist_inc = pedal["acc_dist_inc_held"]
            acc_dist_dec = pedal["acc_dist_dec_held"]

            all_assigned = self._all_cc_buttons_assigned()

            # Auto-deactivate when the truck is no longer in a drivable state.
            # Brake input (joystick or game keyboard) does NOT deactivate CC —
            # the user can override transiently and CC stays armed.
            if (
                self._cc_enabled
                and connected
                and Settings.cc_mode == "Cruise control"
                and (tel["park_brake"] or tel["gear_dashboard"] <= 0)
            ):
                self._cc_enabled = False
                self._cc_disarmed = False
                if tel["park_brake"]:
                    logger.info("CC disabled — parking brake engaged", extra={"popup": True})
                else:
                    logger.info("CC disabled — gear neutral or reverse", extra={"popup": True})

            if connected and Settings.cc_mode == "Cruise control" and (cc_inc or cc_start):
                if tel["park_brake"] or tel["gear_dashboard"] <= 0:
                    if now - self._last_block_msg_mono > 2.0:
                        self._last_block_msg_mono = now
                        if tel["park_brake"]:
                            logger.info("CC cannot be used with parking brake engaged", extra={"popup": True})
                        else:
                            logger.info("CC can only be used in drive", extra={"popup": True})

            if any((cc_dec, cc_inc, cc_start)):
                logger.debug(
                    "CC button held — start=%s inc=%s dec=%s | "
                    "connected=%s paused=%s device_lost=%s all_assigned=%s",
                    cc_start, cc_inc, cc_dec,
                    connected, paused, device_lost, all_assigned,
                )

            if connected and not paused and not device_lost:
                if not all_assigned and (cc_dec or cc_inc or cc_start):
                    if now - self._last_assign_warn_mono > 2.0:
                        self._last_assign_warn_mono = now
                        logger.info(
                            "Please assign all cruise control buttons in the settings",
                            extra={"popup": True},
                        )
                elif all_assigned:
                    self._tick_button_fsm(tel, now, cc_dec, cc_inc, cc_start)
                self._tick_acc_distance_fsm(now, acc_dist_inc, acc_dist_dec)
            elif any((cc_dec, cc_inc, cc_start)):
                logger.debug(
                    "CC button press ignored — guard blocked "
                    "(need: connected=%s, not paused=%s, not device_lost=%s)",
                    connected, not paused, not device_lost,
                )

            # Disable-on-stop guard. CC stays enabled through normal stops;
            # it only disables when a *triggering event* (crash or AEB_brake)
            # is followed by a full stop within the timeout window.
            speed_ms = tel["speed_ms"]
            if self._cc_enabled:
                aeb_brake = self._read_aeb_brake()
                crash_event = (
                    self._cc_prev_speed_ms is not None
                    and (self._cc_prev_speed_ms - speed_ms) >= _CC_CRASH_SPEED_DROP_MS
                )
                if crash_event or aeb_brake:
                    self._cc_disarm_pending_until = now + _CC_DISARM_PENDING_TIMEOUT_S
                if (
                    now < self._cc_disarm_pending_until
                    and speed_ms < _CC_DISARM_SPEED_MS
                ):
                    self._cc_enabled = False
                    self._cc_disarmed = False
                    self._cc_disarm_pending_until = 0.0
                
                    logger.info(
                        f'{"ACC" if Settings.acc_enabled else "CC"} disabled for safety\ntap set/+ to resume',
                    )
                    PopupWindow.emit(f'{"ACC" if Settings.acc_enabled else "CC"} disabled', "disabled for safety\ntap set/+ to resume", "w")
            else:
                self._cc_disarmed = False
                self._cc_disarm_pending_until = 0.0
            self._cc_prev_speed_ms = speed_ms

            if (
                self._cc_enabled
                and self._target_speed_kmh is not None
                and connected
                and not paused
                and not device_lost
                and not em_stop
            ):
                wanted_accel = self._speed_to_accel(tel, dt)
                if self._acc_should_cap():
                    wanted_accel = min(wanted_accel, self._acc.accel_cap_ms2(tel["speed_ms"]))
                else:
                    self._acc.reset()
                        # User game-throttle override: when the user is pressing gas
                # in-game and CC wants to brake, bypass output EMA so the
                # brake response is immediate instead of softened over ~0.4s.
                game_throttle = float(tel.get("game_throttle", 0.0))
                user_overriding = (
                    game_throttle > _CC_GAME_THROTTLE_OVERRIDE and wanted_accel < 0.0
                )
                if user_overriding:
                    self._output_ema_accel_ms2 = wanted_accel
                wanted_accel = self._smooth_output_accel_ema(wanted_accel, dt)
                accel_min = float(Settings.cc_accel_min_ms2)
                accel_max = float(Settings.cc_accel_max_ms2)
                wanted_accel = max(accel_min, min(accel_max, wanted_accel))
                commanding = True
            else:
                self._integral_error = 0.0
                self._kd_smooth_speed_ms = None
                self._output_ema_accel_ms2 = None
                self._target_speed_ema_ms = None
                self._acc.reset()

            self._publish_telemetry_command(wanted_accel if commanding else 0.0)
            self._publish_data(commanding, wanted_accel if commanding else 0.0)
            self._maybe_reset_mapper_on_commanding_end(commanding)

        except Exception:
            logger.exception("cruise_control_thread loop error; clearing command")
            self._publish_telemetry_command(0.0)
            self._publish_data(False, 0.0)
            self._maybe_reset_mapper_on_commanding_end(False)

    def teardown(self) -> None:
        try:
            tel = registry.get_thread("telemetry_thread")
            with tel.data._lock:
                tel.data.commanded_accel_ms2 = 0.0
        except (KeyError, AttributeError):
            pass
        self._request_mapper_reset()
        logger.debug("cruise_control_thread teardown complete")

    def _acc_should_cap(self) -> bool:
        return bool(Settings.acc_enabled) and Settings.cc_mode == "Cruise control"

    @staticmethod
    def _all_cc_buttons_assigned() -> bool:
        return (
            Settings.cc_start_button is not None
            and Settings.cc_inc_button is not None
            and Settings.cc_dec_button is not None
        )

    def _snapshot_telemetry(self) -> dict | None:
        try:
            tel = registry.get_thread("telemetry_thread")
        except KeyError:
            return None
        try:
            with tel.data._lock:
                return {
                    "is_connected": bool(tel.data.is_connected),
                    "paused": bool(tel.data.paused),
                    "speed_ms": float(tel.data.speed),
                    "gear_dashboard": int(tel.data.gear_dashboard),
                    "park_brake": bool(tel.data.parkBrake),
                    "game_clutch": float(tel.data.gameClutch),
                    "game_throttle": float(tel.data.gameThrottle),
                }
        except Exception:
            return None

    def _read_aeb_brake(self) -> bool:
        try:
            aeb = registry.get_thread("aeb_thread")
        except KeyError:
            return False
        try:
            if not aeb.is_alive():
                return False
            with aeb.data._lock:
                return bool(aeb.data.AEB_brake)
        except (AttributeError, KeyError):
            return False

    def _snapshot_pedal(self) -> dict | None:
        try:
            pt = registry.get_thread("main_pedal_thread")
        except KeyError:
            return None
        try:
            with pt.data._lock:
                return {
                    "device_lost": bool(pt.data.device_lost),
                    "em_stop": bool(pt.data.em_stop),
                    "cc_dec_held": bool(pt.data.cc_dec_held),
                    "cc_inc_held": bool(pt.data.cc_inc_held),
                    "cc_start_held": bool(pt.data.cc_start_held),
                    "acc_dist_inc_held": bool(getattr(pt.data, "acc_dist_inc_held", False)),
                    "acc_dist_dec_held": bool(getattr(pt.data, "acc_dist_dec_held", False)),
                }
        except Exception:
            return None

    def _clamp_target_kmh(self, v: float) -> float:
        return max(_SPEED_MIN_KMH, min(_SPEED_MAX_KMH, v))

    def _change_target_kmh(self, delta: float) -> None:
        """Adjust set speed like legacy change_target_speed: coarse steps use grid math, fine steps add."""
        if self._target_speed_kmh is None:
            return
        inc = int(round(float(delta)))
        abs_inc = abs(inc)
        if abs_inc >= 5:
            ts = int(round(float(self._target_speed_kmh)))
            if inc > 0:
                ts = ((ts // abs_inc) + 1) * abs_inc
            elif ts % abs_inc == 0:
                ts = ((ts // abs_inc) - 1) * abs_inc
            else:
                ts = (ts // abs_inc) * abs_inc
            self._target_speed_kmh = self._clamp_target_kmh(float(ts))
        else:
            self._target_speed_kmh = self._clamp_target_kmh(
                float(self._target_speed_kmh) + float(delta),
            )

    def _set_target_from_speed_kmh(self, speed_kmh: float) -> None:
        self._target_speed_kmh = self._clamp_target_kmh(round(speed_kmh))

    def _tick_button_fsm(
        self,
        tel: dict,
        now: float,
        cc_dec: bool,
        cc_inc: bool,
        cc_start: bool,
    ) -> None:
        speed_kmh = tel["speed_ms"] * 3.6
        short_i = int(Settings.short_increments) if Settings.short_increments is not None else 1
        long_i = int(Settings.long_increments) if Settings.long_increments is not None else 5

        cruise_mode = Settings.cc_mode == "Cruise control"
        block_inc_start = (
            cruise_mode
            and (tel["park_brake"] or tel["gear_dashboard"] <= 0)
        )

        all_assigned = self._all_cc_buttons_assigned()

        # Decrease
        if cc_dec and not cc_inc and not cc_start and all_assigned:
            if self._time_pressed_dec is None:
                self._time_pressed_dec = now
            dt_dec = now - self._time_pressed_dec
            if (not self._long_press_dec and dt_dec > _LONG_PRESS_DEC_INC_FIRST_S) or (
                self._long_press_dec and dt_dec > _LONG_PRESS_DEC_INC_REPEAT_S
            ):
                self._long_press_dec = True
                self._time_pressed_dec = now
                if self._target_speed_kmh is not None:
                    self._change_target_kmh(-float(long_i))
        elif self._time_pressed_dec is not None:
            if not self._long_press_dec and self._target_speed_kmh is not None:
                self._change_target_kmh(-float(short_i))
            else:
                self._long_press_dec = False
            self._time_pressed_dec = None

        # Increase
        if cc_inc and not cc_dec and not cc_start and all_assigned and not block_inc_start:
            if self._time_pressed_inc is None:
                self._time_pressed_inc = now
            dt_inc = now - self._time_pressed_inc
            if (not self._long_press_inc and dt_inc > _LONG_PRESS_DEC_INC_FIRST_S) or (
                self._long_press_inc and dt_inc > _LONG_PRESS_DEC_INC_REPEAT_S
            ):
                self._long_press_inc = True
                self._time_pressed_inc = now
                if self._cc_enabled:
                    self._change_target_kmh(float(long_i))
                elif self._target_speed_kmh is None or speed_kmh > (self._target_speed_kmh or 0):
                    self._set_target_from_speed_kmh(speed_kmh)
                if not self._cc_enabled:
                    self._cc_enabled = True
                    logger.info("Cruise control enabled")
                self._cc_disarmed = False
        elif self._time_pressed_inc is not None:
            if not self._long_press_inc:
                if self._cc_enabled:
                    self._change_target_kmh(float(short_i))
                elif self._target_speed_kmh is None or speed_kmh > (self._target_speed_kmh or 0):
                    self._set_target_from_speed_kmh(speed_kmh)
                if not self._cc_enabled:
                    self._cc_enabled = True
                    logger.info("Cruise control enabled")
                self._cc_disarmed = False
            else:
                self._long_press_inc = False
            self._time_pressed_inc = None

        # Start / toggle
        if cc_start and not cc_dec and not cc_inc and all_assigned:
            if self._time_pressed_start is None:
                self._time_pressed_start = now
            dt_start = now - self._time_pressed_start
            if not self._long_press_start and dt_start > _LONG_PRESS_START_S:
                self._long_press_start = True
                if Settings.long_press_reset and not block_inc_start:
                    self._set_target_from_speed_kmh(speed_kmh)
                    if not self._cc_enabled:
                        self._cc_enabled = True
                    self._cc_disarmed = False
                    logger.info("Cruise target reset to current speed")
                elif not Settings.long_press_reset:
                    logger.info("Long press to reset is disabled")
        elif self._time_pressed_start is not None:
            if not self._long_press_start:
                if self._cc_enabled:
                    self._cc_enabled = False
                    logger.info("Cruise control disabled")
                else:
                    self._cc_enabled = True
                    self._cc_disarmed = False
                    logger.info("Cruise control enabled")
                if self._target_speed_kmh is None:
                    self._set_target_from_speed_kmh(speed_kmh)
            else:
                self._long_press_start = False
            self._time_pressed_start = None

    def _tick_acc_distance_fsm(self, now: float, inc_held: bool, dec_held: bool) -> None:
        """Drive the ACC gap level from one or two dedicated buttons.

        Mirrors the cc inc/dec timing: short release = one step, sustained
        hold = auto-repeat at the same cadence.

        Both bound: inc/dec apply ±1 with hard clamp at [1, 4] (no wrap, so
        a held button can't run past the ends and "wrap around" unexpectedly).
        Only one bound: that single button cycles 1→2→3→4→1.
        Neither bound: warn (rate-limited) when the user presses something
        that mapped to nothing.
        """
        inc_assigned = Settings.acc_dist_inc_button is not None
        dec_assigned = Settings.acc_dist_dec_button is not None

        if not inc_assigned and not dec_assigned:
            if (inc_held or dec_held) and now - self._last_assign_warn_mono > 2.0:
                self._last_assign_warn_mono = now
                logger.info(
                    "Please assign the ACC distance button(s) in the settings",
                    extra={"popup": True},
                )
            self._time_pressed_acc_dist_inc = None
            self._time_pressed_acc_dist_dec = None
            self._long_press_acc_dist_inc = False
            self._long_press_acc_dist_dec = False
            return

        cycle_mode = inc_assigned ^ dec_assigned
        if cycle_mode:
            held = inc_held if inc_assigned else dec_held

            def _apply() -> None:
                self._step_acc_gap_level(+1, wrap=True)

            self._time_pressed_acc_dist_inc, self._long_press_acc_dist_inc = self._tick_dist_button(
                now, held, self._time_pressed_acc_dist_inc, self._long_press_acc_dist_inc, _apply,
            )
            self._time_pressed_acc_dist_dec = None
            self._long_press_acc_dist_dec = False
            return

        # Both assigned — clamped step. Suppress when both are held to avoid fights.
        if inc_held and dec_held:
            self._time_pressed_acc_dist_inc = None
            self._time_pressed_acc_dist_dec = None
            self._long_press_acc_dist_inc = False
            self._long_press_acc_dist_dec = False
            return

        self._time_pressed_acc_dist_inc, self._long_press_acc_dist_inc = self._tick_dist_button(
            now, inc_held, self._time_pressed_acc_dist_inc, self._long_press_acc_dist_inc,
            lambda: self._step_acc_gap_level(+1, wrap=False),
        )
        self._time_pressed_acc_dist_dec, self._long_press_acc_dist_dec = self._tick_dist_button(
            now, dec_held, self._time_pressed_acc_dist_dec, self._long_press_acc_dist_dec,
            lambda: self._step_acc_gap_level(-1, wrap=False),
        )

    @staticmethod
    def _tick_dist_button(
        now: float,
        held: bool,
        time_pressed: float | None,
        long_press: bool,
        apply,
    ) -> tuple[float | None, bool]:
        """Generic short/long-press FSM. Returns the new (time_pressed, long_press)."""
        if held:
            if time_pressed is None:
                time_pressed = now
            held_dt = now - time_pressed
            if (not long_press and held_dt > _LONG_PRESS_DEC_INC_FIRST_S) or (
                long_press and held_dt > _LONG_PRESS_DEC_INC_REPEAT_S
            ):
                long_press = True
                time_pressed = now
                apply()
            return time_pressed, long_press
        if time_pressed is not None:
            if not long_press:
                apply()
            else:
                long_press = False
            time_pressed = None
        return time_pressed, long_press

    @staticmethod
    def _step_acc_gap_level(delta: int, *, wrap: bool) -> None:
        try:
            current = int(Settings.acc_gap_level)
        except (TypeError, ValueError):
            current = 2
        current = max(1, min(4, current))
        new_level = current + int(delta)
        if wrap:
            if new_level > 4:
                new_level = 1
            elif new_level < 1:
                new_level = 4
        else:
            new_level = max(1, min(4, new_level))
        if new_level == current:
            return
        try:
            Settings.save(values={"acc_gap_level": new_level})
        except Exception:
            logger.exception("failed to persist acc_gap_level")
            return
        logger.info("ACC gap set to %d/4", new_level)

    def _gearshift_d_factor(self, now: float, clutch: float) -> float:
        """Returns 0.0 while clutched or in post-release block, 0→1 over ramp, 1 otherwise.

        Mirrors sending_thread mapper's gearshift state machine so CC's D-term
        releases in lockstep with the mapper's integrators.
        """
        now_safe = now if math.isfinite(now) else 0.0
        clutch_pressed = clutch > _CC_CLUTCH_ACTIVE_THRESHOLD

        if clutch_pressed and not self._cc_clutch_active:
            self._cc_clutch_active = True
        elif not clutch_pressed and self._cc_clutch_active:
            self._cc_clutch_active = False
            self._cc_clutch_release_mono = now_safe

        if clutch_pressed:
            return 0.0
        time_since_release = now_safe - self._cc_clutch_release_mono
        if time_since_release < _CC_GEARSHIFT_BLOCK_DURATION_S:
            return 0.0
        if time_since_release < _CC_GEARSHIFT_BLOCK_DURATION_S + _CC_GEARSHIFT_RAMP_DURATION_S:
            return (time_since_release - _CC_GEARSHIFT_BLOCK_DURATION_S) / _CC_GEARSHIFT_RAMP_DURATION_S
        return 1.0

    def _speed_to_accel(self, tel: dict, dt: float) -> float:
        speed_ms = tel["speed_ms"]
        target_kmh = self._target_speed_kmh
        if target_kmh is None:
            return 0.0

        if self._last_target_for_integral != target_kmh:
            self._integral_error = 0.0
            self._last_target_for_integral = target_kmh

        if Settings.cc_mode == "Speed limiter":
            target_ms = max((target_kmh - 0.1) / 3.6, 0.0)
        else:
            target_ms = target_kmh / 3.6
        target_ms = self._smooth_target_speed_ema(target_ms, dt)

        # Read PID gains from Settings each call for live hot-reload tuning.
        kp = float(Settings.cc_kp)
        ki = float(Settings.cc_ki)
        kd = float(Settings.cc_kd)
        clamp = float(Settings.cc_integral_clamp)

        error_ms = target_ms - speed_ms
        self._integral_error += error_ms * dt
        self._integral_error = max(-clamp, min(clamp, self._integral_error))

        d_factor = self._gearshift_d_factor(time.monotonic(), float(tel.get("game_clutch", 0.0)))

        raw_speed_ms = speed_ms
        if d_factor <= 0.0:
            # Freeze the smoothed-speed EMA during clutch + block window. Prevents
            # it from tracking the stalled-then-jumping speed and producing a
            # spurious derivative spike on re-engage.
            speed_deriv = 0.0
        else:
            # Leading edge of the ramp: reseed EMA to current speed so the first
            # post-block derivative starts at zero instead of comparing against
            # the stale pre-clutch value.
            if self._cc_prev_d_factor <= 0.0:
                self._kd_smooth_speed_ms = raw_speed_ms
            tau = _CC_KD_SPEED_SMOOTH_TAU_S
            alpha = dt / (tau + dt) if tau > 0.0 else 1.0
            if self._kd_smooth_speed_ms is None:
                self._kd_smooth_speed_ms = raw_speed_ms
                speed_deriv = 0.0
            else:
                prev_smooth = self._kd_smooth_speed_ms
                self._kd_smooth_speed_ms = alpha * raw_speed_ms + (1.0 - alpha) * prev_smooth
                speed_deriv = (self._kd_smooth_speed_ms - prev_smooth) / dt
        self._cc_prev_d_factor = d_factor

        p_term = kp * error_ms
        i_term = ki * self._integral_error
        d_term = -kd * speed_deriv * d_factor

        logger.debug(
            "cc pid: error=%.3f p=%.3f i=%.3f d=%.3f (df=%.2f) integral=%.3f output=%.3f",
            error_ms, p_term, i_term, d_term, d_factor, self._integral_error,
            p_term + i_term + d_term,
        )
        return p_term + i_term + d_term

    def _smooth_output_accel_ema(self, raw_ms2: float, dt: float) -> float:
        tau = _CC_OUTPUT_EMA_TAU_S
        alpha = dt / (tau + dt) if tau > 0.0 else 1.0
        if self._output_ema_accel_ms2 is None:
            self._output_ema_accel_ms2 = raw_ms2
        else:
            self._output_ema_accel_ms2 = (
                alpha * raw_ms2 + (1.0 - alpha) * self._output_ema_accel_ms2
            )
        return self._output_ema_accel_ms2

    def _smooth_target_speed_ema(self, target_ms: float, dt: float) -> float:
        tau = _CC_TARGET_SPEED_EMA_TAU_S
        alpha = dt / (tau + dt) if tau > 0.0 else 1.0
        if self._target_speed_ema_ms is None:
            self._target_speed_ema_ms = target_ms
        else:
            self._target_speed_ema_ms = (
                alpha * target_ms + (1.0 - alpha) * self._target_speed_ema_ms
            )
        return self._target_speed_ema_ms

    def _publish_telemetry_command(self, wanted_accel_ms2: float) -> None:
        if not math.isfinite(wanted_accel_ms2):
            wanted_accel_ms2 = 0.0
        try:
            tel = registry.get_thread("telemetry_thread")
            with tel.data._lock:
                tel.data.commanded_accel_ms2 = float(wanted_accel_ms2)
        except (KeyError, AttributeError):
            pass

    def _publish_data(self, commanding: bool, wanted_accel_ms2: float) -> None:
        with self.data._lock:
            self.data.active = commanding
            self.data.cc_enabled = self._cc_enabled
            self.data.target_speed_kmh = self._target_speed_kmh
            self.data.wanted_accel_ms2 = wanted_accel_ms2

    def _maybe_reset_mapper_on_commanding_end(self, commanding: bool) -> None:
        if self._was_commanding and not commanding:
            self._request_mapper_reset()
        self._was_commanding = commanding

    def _request_mapper_reset(self) -> None:
        try:
            st = registry.get_thread("sending_thread")
        except KeyError:
            return
        try:
            if st.is_alive():
                st.reset_accel_mapper_smoothing()
        except Exception:
            logger.debug("reset_accel_mapper_smoothing failed", exc_info=True)
