"""
Cruise control orchestrator thread.

Owns the longitudinal controller stack (ACC + CC children from
`core/longitudinal/`), the CC button FSM, and arbitration. Each tick:

1. Read telemetry + pedal + AEB state, build a `LongCtx`.
2. Run the CC button FSM — drives `self._cc_ctrl` enable/target.
3. Run the ACC distance FSM — drives `Settings.acc_gap_level`.
4. Step the controllers; arbitrate `min(...)` over active bids.
5. Publish wanted accel to telemetry for `accel_to_pedals.step()` and to
   `self.data` for UI consumers.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from core.longitudinal.acc import AdaptiveCruiseController
from core.longitudinal.base import LongCtx, LongOutput
from core.longitudinal.cc import CruiseController
from core.settings import Settings
from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry

logger = logging.getLogger(__name__)

# Button timing (legacy MonoCruise main_cruise_control)
_LONG_PRESS_DEC_INC_FIRST_S = 0.3
_LONG_PRESS_DEC_INC_REPEAT_S = 0.6
_LONG_PRESS_START_S = 0.5


@dataclass
class CruiseControlThreadData(ThreadData):
    """Published state for sending_thread / UI."""

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

        # Longitudinal controllers — children of LongitudinalController.
        self._cc_ctrl = CruiseController()
        self._acc_ctrl = AdaptiveCruiseController()

        # Button FSM state — owns press timing only; acts on CC via _cc_ctrl.
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

        # Loop cadence
        self._prev_loop_mono = time.monotonic()
        self._was_commanding = False

        # Rate-limited UI messages
        self._last_assign_warn_mono: float = 0.0
        self._last_block_msg_mono: float = 0.0

    def setup(self) -> None:
        self._prev_loop_mono = time.monotonic()
        logger.debug("cruise_control_thread setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        # Idle throttle when telemetry disconnected — buttons are gated below.
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
            aeb_brake = self._read_aeb_brake()

            all_assigned = self._all_cc_buttons_assigned()

            # Block-message: warn when user presses inc/start but truck is in
            # park/neutral/reverse (cruise mode only).
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

            # Drive CC button FSM and ACC distance FSM.
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

            # Build context for controllers.
            ctx = LongCtx(
                now=now,
                dt=dt,
                speed_ms=float(tel["speed_ms"]),
                gear_dashboard=int(tel["gear_dashboard"]),
                park_brake=bool(tel["park_brake"]),
                game_throttle=float(tel["game_throttle"]),
                game_clutch=float(tel["game_clutch"]),
                game_brake=0.0,
                aeb_brake=bool(aeb_brake),
                connected=bool(connected),
                paused=bool(paused),
                em_stop=bool(em_stop),
                device_lost=bool(device_lost),
            )

            # CC steps unconditionally — it owns its own enable/disarm guards.
            cc_out = self._cc_ctrl.step(ctx)

            # ACC bids only when CC is active. ACC alone (without a setpoint
            # source) would have no upper bound, so we keep the legacy gating.
            if cc_out.active:
                acc_out = self._acc_ctrl.step(ctx)
            else:
                self._acc_ctrl.reset()
                acc_out = LongOutput(None, False)

            # Arbitrate: lowest active m/s² bid wins.
            wanted_accel, commanding = self._arbitrate(cc_out, acc_out)

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

    @staticmethod
    def _arbitrate(*outs: LongOutput) -> tuple[float, bool]:
        bids = [o.wanted_ms2 for o in outs if o.active and o.wanted_ms2 is not None]
        if not bids:
            return 0.0, False
        return min(bids), True

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

    def _tick_button_fsm(
        self,
        tel: dict,
        now: float,
        cc_dec: bool,
        cc_inc: bool,
        cc_start: bool,
    ) -> None:
        """CC inc/dec/start press timing → drives `self._cc_ctrl`."""
        speed_kmh = tel["speed_ms"] * 3.6
        short_i = int(Settings.short_increments) if Settings.short_increments is not None else 1
        long_i = int(Settings.long_increments) if Settings.long_increments is not None else 5

        cruise_mode = Settings.cc_mode == "Cruise control"
        block_inc_start = (
            cruise_mode
            and (tel["park_brake"] or tel["gear_dashboard"] <= 0)
        )

        cc = self._cc_ctrl

        # Decrease
        if cc_dec and not cc_inc and not cc_start:
            if self._time_pressed_dec is None:
                self._time_pressed_dec = now
            dt_dec = now - self._time_pressed_dec
            if (not self._long_press_dec and dt_dec > _LONG_PRESS_DEC_INC_FIRST_S) or (
                self._long_press_dec and dt_dec > _LONG_PRESS_DEC_INC_REPEAT_S
            ):
                self._long_press_dec = True
                self._time_pressed_dec = now
                if cc.target_speed_kmh is not None:
                    cc.change_target_kmh(-float(long_i))
        elif self._time_pressed_dec is not None:
            if not self._long_press_dec and cc.target_speed_kmh is not None:
                cc.change_target_kmh(-float(short_i))
            else:
                self._long_press_dec = False
            self._time_pressed_dec = None

        # Increase (and enable on first press if disabled)
        if cc_inc and not cc_dec and not cc_start and not block_inc_start:
            if self._time_pressed_inc is None:
                self._time_pressed_inc = now
            dt_inc = now - self._time_pressed_inc
            if (not self._long_press_inc and dt_inc > _LONG_PRESS_DEC_INC_FIRST_S) or (
                self._long_press_inc and dt_inc > _LONG_PRESS_DEC_INC_REPEAT_S
            ):
                self._long_press_inc = True
                self._time_pressed_inc = now
                if cc.enabled:
                    cc.change_target_kmh(float(long_i))
                elif cc.target_speed_kmh is None or speed_kmh > (cc.target_speed_kmh or 0):
                    cc.set_target_from_speed_kmh(speed_kmh)
                if not cc.enabled:
                    cc.enable()
                    logger.info("Cruise control enabled")
        elif self._time_pressed_inc is not None:
            if not self._long_press_inc:
                if cc.enabled:
                    cc.change_target_kmh(float(short_i))
                elif cc.target_speed_kmh is None or speed_kmh > (cc.target_speed_kmh or 0):
                    cc.set_target_from_speed_kmh(speed_kmh)
                if not cc.enabled:
                    cc.enable()
                    logger.info("Cruise control enabled")
            else:
                self._long_press_inc = False
            self._time_pressed_inc = None

        # Start / toggle
        if cc_start and not cc_dec and not cc_inc:
            if self._time_pressed_start is None:
                self._time_pressed_start = now
            dt_start = now - self._time_pressed_start
            if not self._long_press_start and dt_start > _LONG_PRESS_START_S:
                self._long_press_start = True
                if Settings.long_press_reset and not block_inc_start:
                    cc.set_target_from_speed_kmh(speed_kmh)
                    if not cc.enabled:
                        cc.enable()
                    logger.info("Cruise target reset to current speed")
                elif not Settings.long_press_reset:
                    logger.info("Long press to reset is disabled")
        elif self._time_pressed_start is not None:
            if not self._long_press_start:
                if cc.enabled:
                    cc.disable()
                    logger.info("Cruise control disabled")
                else:
                    cc.enable()
                    logger.info("Cruise control enabled")
                if cc.target_speed_kmh is None:
                    cc.set_target_from_speed_kmh(speed_kmh)
            else:
                self._long_press_start = False
            self._time_pressed_start = None

    def _tick_acc_distance_fsm(self, now: float, inc_held: bool, dec_held: bool) -> None:
        """Drive the ACC gap level from one or two dedicated buttons.

        Both bound: inc/dec apply ±1 with hard clamp at [1, 4].
        Only one bound: that single button cycles 1→2→3→4→1.
        Neither bound: warn (rate-limited).
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

        # Both assigned — clamped step. Suppress when both are held.
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
        """Generic short/long-press FSM."""
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

    def _publish_telemetry_command(self, wanted_accel_ms2: float) -> None:
        try:
            tel = registry.get_thread("telemetry_thread")
            with tel.data._lock:
                tel.data.commanded_accel_ms2 = float(wanted_accel_ms2)
        except (KeyError, AttributeError):
            pass

    def _publish_data(self, commanding: bool, wanted_accel_ms2: float) -> None:
        with self.data._lock:
            self.data.active = commanding
            self.data.cc_enabled = self._cc_ctrl.enabled
            self.data.target_speed_kmh = self._cc_ctrl.target_speed_kmh
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
