"""Longitudinal orchestrator + CC button FSM. See core/cruise_control_thread/README.md."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

from core.cruise_control_thread.acc_distance import AccDistanceButtons
from core.cruise_control_thread.press_counter import PressCounter
from core.longitudinal.acc import AdaptiveCruiseController
from core.longitudinal.base import LongCtx, LongOutput
from core.longitudinal.cc import CruiseController
from core.longitudinal.limiter import SpeedLimiter
from core.settings import Settings
from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry

logger = logging.getLogger(__name__)

# CC disengage thresholds (CC-only: limiter is immune to these events)
_CC_RAW_BRAKE_DISENGAGE = 0.05
_CC_GAME_BRAKE_DISENGAGE = 0.2
_CC_DISARM_SPEED_MS = 0.3
_CC_CRASH_SPEED_DROP_MS = 5.0
_CC_DISARM_PENDING_TIMEOUT_S = 5.0

# User OPD-gas override of CC (cruise mode): while latched, CC/ACC bids are
_CC_OVERRIDE_GAS_RELEASE = 0.02
_CC_OVERRIDE_SPEED_MARGIN_KMH = 2.0

# Neutral: CC stays engaged (manual shift-through-N), but gas bids are cut.
# Popup waits this long so brief shift flashes never spam the driver.
_CC_NEUTRAL_GAS_POPUP_DWELL_S = 2.0
_CC_NEUTRAL_GAS_POPUP_COOLDOWN_S = 2.0

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
    active_controller: str = "none"  # "cc" | "limiter" | "none"

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class CruiseControlThread(BaseThread):
    loop_interval = 1.0 / Settings.polling_rate
    max_restarts = 5

    def __init__(self) -> None:
        super().__init__(name="cruise_control_thread")
        self.data = CruiseControlThreadData()

        # Longitudinal controllers: children of LongitudinalController.
        self._cc_ctrl = CruiseController()
        self._limiter_ctrl = SpeedLimiter()
        self._acc_ctrl = AdaptiveCruiseController()

        # Mode tracking for handover reset on mode flip.
        self._prev_cc_mode: str | None = None

        # Disarm-on-stop state (CC-only: moved from CruiseController).
        self._cc_disarm_pending_until: float = 0.0
        self._cc_prev_speed_ms: float | None = None

        # User OPD-gas override latch (cruise mode, limiter active only).
        self._cc_user_override: bool = False

        # Button FSM state: owns press timing only; acts on CC via _cc_ctrl.
        self._time_pressed_dec: float | None = None
        self._time_pressed_inc: float | None = None
        self._time_pressed_start: float | None = None
        self._long_press_dec = False
        self._long_press_inc = False
        self._long_press_start = False

        # Loop cadence
        self._prev_loop_mono = time.monotonic()
        self._was_commanding = False

        # Short presses fire per press counted by the pedal thread, so a tap
        # cannot be lost when this thread samples slower than the tap is long.
        self._presses = PressCounter()
        self._acc_distance = AccDistanceButtons(self._presses)

        # Rate-limited UI messages
        self._last_assign_warn_mono: float = 0.0
        self._last_block_msg_mono: float = 0.0
        self._neutral_since_mono: float | None = None
        self._last_neutral_gas_popup_mono: float = 0.0

    def setup(self) -> None:
        self._prev_loop_mono = time.monotonic()
        logger.debug("cruise_control_thread setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        # Idle throttle when telemetry disconnected: buttons are gated below.
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
                self._publish_data(False, 0.0, Settings.cc_mode)
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
            # park or reverse (neutral no longer blocks engage; gas is cut instead).
            if connected and Settings.cc_mode == "Cruise control" and (cc_inc or cc_start):
                if self._park_or_reverse_blocks_cc(
                    tel["park_brake"], tel["gear_dashboard"]
                ):
                    if now - self._last_block_msg_mono > 2.0:
                        self._last_block_msg_mono = now
                        if tel["park_brake"]:
                            logger.info("Cannot engage with parking brake on", extra={"popup": True})
                        else:
                            logger.info("Can only engage in drive", extra={"popup": True})

            if any((cc_dec, cc_inc, cc_start)):
                logger.debug(
                    "CC button held: start=%s inc=%s dec=%s | "
                    "connected=%s paused=%s device_lost=%s all_assigned=%s",
                    cc_start, cc_inc, cc_dec,
                    connected, paused, device_lost, all_assigned,
                )

            self._presses.sync(pedal.get("press_counts"))
            self._presses.audit(
                now,
                {
                    "cc_start_button": cc_start,
                    "cc_inc_button": cc_inc,
                    "cc_dec_button": cc_dec,
                    "acc_dist_inc_button": acc_dist_inc,
                    "acc_dist_dec_button": acc_dist_dec,
                },
                pedal.get("pedal_loop_hz", 0.0),
                self.avg_framerate,
            )

            # Drive CC button FSM and ACC distance FSM.
            if connected and not paused and not device_lost:
                if not all_assigned:
                    if cc_dec or cc_inc or cc_start:
                        if now - self._last_assign_warn_mono > 2.0:
                            self._last_assign_warn_mono = now
                            logger.info(
                                "Please assign all cruise control buttons in the settings",
                                extra={"popup": True},
                            )
                    self._presses.discard()
                else:
                    self._tick_button_fsm(tel, now, cc_dec, cc_inc, cc_start)
                self._acc_distance.tick(now, acc_dist_inc, acc_dist_dec)
            else:
                # Gated: drop what was counted so unpausing cannot replay it.
                self._presses.discard()
                if any((cc_dec, cc_inc, cc_start)):
                    blocked = []
                    if not connected:
                        blocked.append("telemetry disconnected")
                    if paused:
                        blocked.append("game paused")
                    if device_lost:
                        blocked.append("pedal device lost")
                    logger.debug(
                        "CC button press ignored: %s",
                        ", ".join(blocked) or "unknown reason",
                    )

            # Build context for controllers.
            user_raw_brake = float(pedal.get("brakeval", 0.0))
            game_brake_in = float(tel.get("game_brake", 0.0))
            commanded_recent_max = self._read_recent_commanded_brake_max()

            ctx = LongCtx(
                now=now,
                dt=dt,
                speed_ms=float(tel["speed_ms"]),
                gear_dashboard=int(tel["gear_dashboard"]),
                park_brake=bool(tel["park_brake"]),
                game_throttle=float(tel["game_throttle"]),
                game_clutch=float(tel["game_clutch"]),
                game_brake=game_brake_in,
                aeb_brake=bool(aeb_brake),
                connected=bool(connected),
                paused=bool(paused),
                em_stop=bool(em_stop),
                device_lost=bool(device_lost),
                user_raw_brake=user_raw_brake,
                commanded_brake_recent_max=commanded_recent_max,
            )

            mode = Settings.cc_mode

            # Reset the inactive controller's PID state on mode flip to avoid
            if mode != self._prev_cc_mode:
                if mode == "Speed limiter":
                    self._cc_ctrl.reset()
                    self._acc_ctrl.reset()
                self._prev_cc_mode = mode

            # Disengage conditions apply to CC only: the limiter is immune to
            # brake presses, gear changes, and crash events (matches original behaviour).
            if mode == "Cruise control":
                self._handle_cc_disengage_conditions(ctx)

            # Dispatch by mode.
            if mode == "Speed limiter":
                self._cc_user_override = False
                self._neutral_since_mono = None
                # CC's button-set target wins; global limit is the always-on fallback
                # when no target has been set via the buttons.
                if self._cc_ctrl.enabled and self._cc_ctrl.target_speed_kmh is not None:
                    # Re-apply the target clamp every tick: CC.step() owns the
                    self._cc_ctrl.set_target_kmh(self._cc_ctrl.target_speed_kmh)
                    self._limiter_ctrl.set_target_kmh(self._cc_ctrl.target_speed_kmh)
                    self._limiter_ctrl.enable()
                elif Settings.global_speed_limit_kmh is not None:
                    self._limiter_ctrl.set_target_kmh(float(Settings.global_speed_limit_kmh))
                    self._limiter_ctrl.enable()
                else:
                    self._limiter_ctrl.disable()

                limiter_out = self._limiter_ctrl.step(ctx)
                cc_out = LongOutput(None, False)
                self._acc_ctrl.reset()
                acc_out = LongOutput(None, False)
            else:
                # Global limiter runs in parallel with CC as an always-on cap
                if Settings.global_speed_limit_kmh is not None:
                    self._limiter_ctrl.set_target_kmh(float(Settings.global_speed_limit_kmh))
                    self._limiter_ctrl.enable()
                else:
                    self._limiter_ctrl.disable()

                # User OPD-gas override: while the user's gas exceeds CC's,
                self._update_cc_override_latch(ctx, pedal)

                if self._cc_user_override:
                    self._cc_ctrl.reset()
                    cc_out = LongOutput(None, False)
                else:
                    cc_out = self._cc_ctrl.step(ctx)
                limiter_out = self._limiter_ctrl.step(ctx)
                if cc_out.active:
                    acc_out = self._acc_ctrl.step(ctx)
                else:
                    self._acc_ctrl.reset()
                    acc_out = LongOutput(None, False)

                # Manual shift flashes N; keep CC on, cut gas, warn after dwell.
                cc_out, acc_out = self._apply_neutral_gas_hold(ctx, cc_out, acc_out)

            wanted_accel, commanding, winner = self._arbitrate_named(
                ("cc", cc_out), ("limiter", limiter_out), ("acc", acc_out),
            )

            self._publish_telemetry_command(wanted_accel if commanding else 0.0)
            self._publish_data(commanding, wanted_accel if commanding else 0.0, mode, winner)
            self._maybe_reset_mapper_on_commanding_end(commanding, paused=bool(paused))

        except Exception:
            logger.exception("cruise_control_thread loop error; clearing command")
            self._publish_telemetry_command(0.0)
            self._publish_data(False, 0.0, Settings.cc_mode)
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
    def _arbitrate_named(*items: tuple[str, LongOutput]) -> tuple[float, bool, str]:
        """Min-arbitrate and report which side dominates for the user-pedal merge. See `core/cruise_control_thread/README.md`."""
        active = [(name, o.wanted_ms2) for name, o in items if o.active and o.wanted_ms2 is not None]
        if not active:
            return 0.0, False, "none"
        winning_bid = min(bid for _, bid in active)
        cc_side_bidding = any(name != "limiter" for name, _ in active)
        winner = "cc" if cc_side_bidding else "limiter"
        return winning_bid, True, winner

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
                    "game_brake": float(getattr(tel.data, "gameBrake", 0.0)),
                }
        except Exception:
            return None

    def _read_recent_commanded_brake_max(self) -> float:
        """Max brake value sent to the game over the last few ticks. See `core/cruise_control_thread/README.md`."""
        try:
            st = registry.get_thread("sending_thread")
        except KeyError:
            return 0.0
        try:
            if not st.is_alive():
                return 0.0
            with st.data._lock:
                recent = tuple(st.data.recent_brake_outputs)
        except (AttributeError, KeyError):
            return 0.0
        if not recent:
            return 0.0
        return max(float(x) for x in recent)

    def _park_or_reverse_blocks_cc(self, park_brake: bool, gear_dashboard: int) -> bool:
        """Park brake or reverse. Neutral no longer blocks: gas is cut instead."""
        return bool(park_brake) or gear_dashboard < 0

    def _read_auto_neutral_holding(self) -> bool:
        """True while sending_thread's auto-neutral owns the gearbox. See README."""
        try:
            st = registry.get_thread("sending_thread")
        except KeyError:
            return False
        try:
            if not st.is_alive():
                return False
            with st.data._lock:
                return bool(getattr(st.data, "auto_neutral_holding", False))
        except (AttributeError, KeyError):
            return False

    @staticmethod
    def _cut_positive_gas_bid(out: LongOutput) -> tuple[LongOutput, bool]:
        """Clamp a positive m/s² bid to 0. Returns (out, True) if gas was cut."""
        if out.active and out.wanted_ms2 is not None and out.wanted_ms2 > 0.0:
            return LongOutput(0.0, True), True
        return out, False

    def _apply_neutral_gas_hold(
        self, ctx: LongCtx, cc_out: LongOutput, acc_out: LongOutput
    ) -> tuple[LongOutput, LongOutput]:
        """Keep CC on in N; cut gas; popup after dwell. See README."""
        in_neutral = ctx.gear_dashboard == 0
        cc_on = self._cc_ctrl.enabled
        if not (in_neutral and cc_on and ctx.connected):
            self._neutral_since_mono = None
            return cc_out, acc_out

        # Auto-neutral needs the published launch bid (>0.25 m/s²) to shift
        # back to drive; clamping here would leave the truck stuck in N.
        if self._read_auto_neutral_holding():
            self._neutral_since_mono = None
            return cc_out, acc_out

        if self._neutral_since_mono is None:
            self._neutral_since_mono = ctx.now

        cc_out, cc_cut = self._cut_positive_gas_bid(cc_out)
        acc_out, acc_cut = self._cut_positive_gas_bid(acc_out)
        cut_gas = cc_cut or acc_cut

        dwell_ok = (ctx.now - self._neutral_since_mono) >= _CC_NEUTRAL_GAS_POPUP_DWELL_S
        cooldown_ok = (
            ctx.now - self._last_neutral_gas_popup_mono
        ) >= _CC_NEUTRAL_GAS_POPUP_COOLDOWN_S
        if cut_gas and dwell_ok and cooldown_ok:
            self._last_neutral_gas_popup_mono = ctx.now
            logger.info("CC can't accelerate in neutral", extra={"popup": True})

        return cc_out, acc_out

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
                    "press_counts": dict(getattr(pt.data, "cc_button_press_counts", None) or {}),
                    "pedal_loop_hz": float(getattr(pt, "avg_framerate", 0.0) or 0.0),
                    "brakeval": float(getattr(pt.data, "brakeval", 0.0)),
                    "opdgasval": float(getattr(pt.data, "opdgasval", 0.0)),
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
            and self._park_or_reverse_blocks_cc(
                tel["park_brake"], tel["gear_dashboard"]
            )
        )

        cc = self._cc_ctrl

        # A press blocked by park or reverse is dropped, never queued.
        if block_inc_start:
            self._presses.discard(("cc_inc_button",))

        # Decrease: long press repeats while held, short presses fire per count.
        if cc_dec and not cc_inc and not cc_start:
            if self._time_pressed_dec is None:
                self._time_pressed_dec = now
            dt_dec = now - self._time_pressed_dec
            if (not self._long_press_dec and dt_dec > _LONG_PRESS_DEC_INC_FIRST_S) or (
                self._long_press_dec and dt_dec > _LONG_PRESS_DEC_INC_REPEAT_S
            ):
                if not self._long_press_dec:
                    self._presses.consume_one("cc_dec_button")
                self._long_press_dec = True
                self._time_pressed_dec = now
                if cc.target_speed_kmh is not None:
                    cc.change_target_kmh(-float(long_i))
        else:
            self._long_press_dec = False
            self._time_pressed_dec = None

        for _ in range(self._presses.take_short("cc_dec_button", cc_dec)):
            if cc.target_speed_kmh is not None:
                cc.change_target_kmh(-float(short_i))

        # Increase (and enable on first press if disabled)
        if cc_inc and not cc_dec and not cc_start and not block_inc_start:
            if self._time_pressed_inc is None:
                self._time_pressed_inc = now
            dt_inc = now - self._time_pressed_inc
            if (not self._long_press_inc and dt_inc > _LONG_PRESS_DEC_INC_FIRST_S) or (
                self._long_press_inc and dt_inc > _LONG_PRESS_DEC_INC_REPEAT_S
            ):
                if not self._long_press_inc:
                    self._presses.consume_one("cc_inc_button")
                self._long_press_inc = True
                self._time_pressed_inc = now
                if cc.enabled:
                    cc.change_target_kmh(float(long_i))
                elif cc.target_speed_kmh is None or speed_kmh > (cc.target_speed_kmh or 0):
                    cc.set_target_from_speed_kmh(speed_kmh)
                if not cc.enabled:
                    cc.enable()
                    _lbl = "Cruise control" if cruise_mode else "Speed limiter"
                    logger.info(f"{_lbl} enabled")
        else:
            self._long_press_inc = False
            self._time_pressed_inc = None

        if not block_inc_start:
            for _ in range(self._presses.take_short("cc_inc_button", cc_inc)):
                if cc.enabled:
                    cc.change_target_kmh(float(short_i))
                elif cc.target_speed_kmh is None or speed_kmh > (cc.target_speed_kmh or 0):
                    cc.set_target_from_speed_kmh(speed_kmh)
                if not cc.enabled:
                    cc.enable()
                    _lbl = "Cruise control" if cruise_mode else "Speed limiter"
                    logger.info(f"{_lbl} enabled")

        # Start / toggle
        if cc_start and not cc_dec and not cc_inc:
            if self._time_pressed_start is None:
                self._time_pressed_start = now
            dt_start = now - self._time_pressed_start
            if not self._long_press_start and dt_start > _LONG_PRESS_START_S:
                self._presses.consume_one("cc_start_button")
                self._long_press_start = True
                if Settings.long_press_reset and not block_inc_start:
                    cc.set_target_from_speed_kmh(speed_kmh)
                    if not cc.enabled:
                        cc.enable()
                    _lbl = "Cruise target" if cruise_mode else "Speed limit"
                    logger.info(f"{_lbl} reset to current speed")
                elif not Settings.long_press_reset:
                    logger.info("Long press to reset is disabled")
        else:
            self._long_press_start = False
            self._time_pressed_start = None

        for _ in range(self._presses.take_short("cc_start_button", cc_start)):
            _lbl = "Cruise control" if cruise_mode else "Speed limiter"
            if cc.enabled:
                cc.disable()
                logger.info(f"{_lbl} disabled")
            else:
                cc.enable()
                logger.info(f"{_lbl} enabled")
            if cc.target_speed_kmh is None:
                cc.set_target_from_speed_kmh(speed_kmh)

    def _publish_telemetry_command(self, wanted_accel_ms2: float) -> None:
        try:
            tel = registry.get_thread("telemetry_thread")
            with tel.data._lock:
                tel.data.commanded_accel_ms2 = float(wanted_accel_ms2)
        except (KeyError, AttributeError):
            pass

    def _publish_data(
        self,
        commanding: bool,
        wanted_accel_ms2: float,
        mode: str,
        winner: str = "none",
    ) -> None:
        if not commanding:
            active_ctrl = "none"
        else:
            active_ctrl = winner if winner in ("cc", "limiter") else (
                "limiter" if mode == "Speed limiter" else "cc"
            )
        with self.data._lock:
            self.data.active = commanding
            self.data.cc_enabled = self._cc_ctrl.enabled
            self.data.target_speed_kmh = self._cc_ctrl.target_speed_kmh
            self.data.wanted_accel_ms2 = wanted_accel_ms2
            self.data.active_controller = active_ctrl

    def _handle_cc_disengage_conditions(self, ctx: LongCtx) -> None:
        """Disengage CC on user brake, park/reverse, or crash-then-stop. See `core/cruise_control_thread/README.md`."""
        cc = self._cc_ctrl

        if cc.enabled:
            game_brake_excess = ctx.game_brake - ctx.commanded_brake_recent_max
            if (
                ctx.user_raw_brake > _CC_RAW_BRAKE_DISENGAGE
                or game_brake_excess > _CC_GAME_BRAKE_DISENGAGE
            ):
                cc.disable()
                logger.info("CC disabled: brake pressed", extra={"popup": True})

        if (
            cc.enabled
            and ctx.connected
            and self._park_or_reverse_blocks_cc(ctx.park_brake, ctx.gear_dashboard)
        ):
            cc.disable()
            if ctx.park_brake:
                logger.info("Cannot engage with parking brake on", extra={"popup": True})
            else:
                logger.info("Can only engage in drive", extra={"popup": True})

        if cc.enabled:
            crash_event = (
                self._cc_prev_speed_ms is not None
                and (self._cc_prev_speed_ms - ctx.speed_ms) >= _CC_CRASH_SPEED_DROP_MS
            )
            if crash_event or ctx.aeb_brake:
                self._cc_disarm_pending_until = ctx.now + _CC_DISARM_PENDING_TIMEOUT_S
            if ctx.now < self._cc_disarm_pending_until and ctx.speed_ms < _CC_DISARM_SPEED_MS:
                cc.disable()
                self._cc_disarm_pending_until = 0.0
                label = "ACC" if Settings.acc_enabled else "CC"
                logger.info(f"{label} disabled for safety\ntap set/+ to resume")
                # Imported lazily: keeps this module importable without Qt.
                from ui.popup.popup_window import PopupWindow
                PopupWindow.emit(
                    f"{label} disabled", "disabled for safety\ntap set/+ to resume", "w",
                )
        else:
            self._cc_disarm_pending_until = 0.0
        self._cc_prev_speed_ms = ctx.speed_ms

    def _update_cc_override_latch(self, ctx: LongCtx, pedal: dict) -> None:
        """Latch/unlatch the user OPD-gas override of CC (cruise mode only). See `core/cruise_control_thread/README.md`."""
        if not (self._cc_ctrl.active and self._limiter_ctrl.active):
            self._cc_user_override = False
            return

        opd_gas = float(pedal.get("opdgasval", 0.0))
        target_kmh = self._cc_ctrl.target_speed_kmh
        speed_kmh = ctx.speed_ms * 3.6
        below_target = (
            target_kmh is not None
            and speed_kmh < float(target_kmh) - _CC_OVERRIDE_SPEED_MARGIN_KMH
        )

        if self._cc_user_override:
            if opd_gas <= _CC_OVERRIDE_GAS_RELEASE or target_kmh is None or below_target:
                self._cc_user_override = False
        elif (
            opd_gas > _CC_OVERRIDE_GAS_RELEASE
            and not below_target
            and self._read_user_gas_above_mapper_flag()
        ):
            self._cc_user_override = True

    @staticmethod
    def _read_user_gas_above_mapper_flag() -> bool:
        """sending_thread's user-OPD-gas-above-mapper flag from the previous tick."""
        try:
            st = registry.get_thread("sending_thread")
        except KeyError:
            return False
        try:
            if not st.is_alive():
                return False
            with st.data._lock:
                return bool(getattr(st.data, "user_gas_above_mapper", False))
        except (AttributeError, KeyError):
            return False

    def _maybe_reset_mapper_on_commanding_end(
        self, commanding: bool, *, paused: bool = False
    ) -> None:
        # Pause gates the CC bid without a real disengage. Keep mapper state
        if paused:
            return
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

