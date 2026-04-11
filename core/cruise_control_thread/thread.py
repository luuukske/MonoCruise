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

from .acc_stub import AdaptiveCruiseControllerStub

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
        self._acc = AdaptiveCruiseControllerStub()

        self._cc_enabled = False
        self._target_speed_kmh: float | None = None

        self._time_pressed_dec: float | None = None
        self._time_pressed_inc: float | None = None
        self._time_pressed_start: float | None = None
        self._long_press_dec = False
        self._long_press_inc = False
        self._long_press_start = False

        self._integral_error = 0.0
        self._kd_smooth_speed_ms: float | None = None
        self._prev_loop_mono = time.monotonic()
        self._was_commanding = False
        self._last_target_for_integral: float | None = None
        self._last_assign_warn_mono: float = 0.0
        self._last_block_msg_mono: float = 0.0
        self._output_ema_accel_ms2: float | None = None

    def setup(self) -> None:
        self._prev_loop_mono = time.monotonic()
        logger.debug("cruise_control_thread setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        self.loop_interval = 1.0 / max(Settings.polling_rate, 10)
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
            brakeval = pedal["brakeval"]
            cc_dec = pedal["cc_dec_held"]
            cc_inc = pedal["cc_inc_held"]
            cc_start = pedal["cc_start_held"]

            all_assigned = self._all_cc_buttons_assigned()

            if brakeval > 0.1 and self._cc_enabled:
                self._cc_enabled = False
                logger.info("Cruise control disabled due to brake input", extra={"popup": True})

            if connected and Settings.cc_mode == "Cruise control" and (cc_inc or cc_start):
                if tel["park_brake"] or tel["gear_dashboard"] <= 0:
                    if now - self._last_block_msg_mono > 2.0:
                        self._last_block_msg_mono = now
                        if tel["park_brake"]:
                            logger.info(
                                "Cruise control cannot be used with parking brake engaged",
                                extra={"popup": True},
                            )
                        else:
                            logger.info(
                                "Cruise control can only be used in drive",
                                extra={"popup": True},
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
                    wanted_accel = min(wanted_accel, self._acc.accel_cap_ms2())
                wanted_accel = self._smooth_output_accel_ema(wanted_accel, dt)
                accel_min = float(Settings.cc_accel_min_ms2)
                accel_max = float(Settings.cc_accel_max_ms2)
                wanted_accel = max(accel_min, min(accel_max, wanted_accel))
                commanding = True
            else:
                self._integral_error = 0.0
                self._kd_smooth_speed_ms = None
                self._output_ema_accel_ms2 = None

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
                }
        except Exception:
            return None

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
                    "brakeval": float(pt.data.brakeval),
                    "cc_dec_held": bool(pt.data.cc_dec_held),
                    "cc_inc_held": bool(pt.data.cc_inc_held),
                    "cc_start_held": bool(pt.data.cc_start_held),
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

        # --- Decrease ---
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

        # --- Increase ---
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
                    logger.info("Cruise control enabled", extra={"popup": True})
        elif self._time_pressed_inc is not None:
            if not self._long_press_inc:
                if self._cc_enabled:
                    self._change_target_kmh(float(short_i))
                elif self._target_speed_kmh is None or speed_kmh > (self._target_speed_kmh or 0):
                    self._set_target_from_speed_kmh(speed_kmh)
                if not self._cc_enabled:
                    self._cc_enabled = True
                    logger.info("Cruise control enabled", extra={"popup": True})
            else:
                self._long_press_inc = False
            self._time_pressed_inc = None

        # --- Start / toggle ---
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
                    logger.info("Cruise target reset to current speed", extra={"popup": True})
                elif not Settings.long_press_reset:
                    logger.info("Long press to reset is disabled", extra={"popup": True})
        elif self._time_pressed_start is not None:
            if not self._long_press_start:
                if self._cc_enabled:
                    self._cc_enabled = False
                    logger.info("Cruise control disabled", extra={"popup": True})
                else:
                    self._cc_enabled = True
                    logger.info("Cruise control enabled", extra={"popup": True})
                if self._target_speed_kmh is None:
                    self._set_target_from_speed_kmh(speed_kmh)
            else:
                self._long_press_start = False
            self._time_pressed_start = None

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

        # Read PID gains from Settings each call for live hot-reload tuning.
        kp = float(Settings.cc_kp)
        ki = float(Settings.cc_ki)
        kd = float(Settings.cc_kd)
        clamp = float(Settings.cc_integral_clamp)

        error_ms = target_ms - speed_ms
        self._integral_error += error_ms * dt
        self._integral_error = max(-clamp, min(clamp, self._integral_error))

        raw_speed_ms = speed_ms
        tau = _CC_KD_SPEED_SMOOTH_TAU_S
        alpha = dt / (tau + dt) if tau > 0.0 else 1.0
        if self._kd_smooth_speed_ms is None:
            self._kd_smooth_speed_ms = raw_speed_ms
            speed_deriv = 0.0
        else:
            prev_smooth = self._kd_smooth_speed_ms
            self._kd_smooth_speed_ms = alpha * raw_speed_ms + (1.0 - alpha) * prev_smooth
            speed_deriv = (self._kd_smooth_speed_ms - prev_smooth) / dt

        p_term = kp * error_ms
        i_term = ki * self._integral_error
        d_term = -kd * speed_deriv

        logger.debug(
            "cc pid: error=%.3f p=%.3f i=%.3f d=%.3f integral=%.3f output=%.3f",
            error_ms, p_term, i_term, d_term, self._integral_error, p_term + i_term + d_term,
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
