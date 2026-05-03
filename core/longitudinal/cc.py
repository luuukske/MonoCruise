"""
Cruise control longitudinal child — owns set-speed PID, output smoothing,
target management, gearshift D-term freeze, and disarm-on-stop guard.

Buttons (enable/disable/inc/dec/start) are driven by the orchestrator
(`cruise_control_thread`) which calls `enable()`, `disable()`,
`change_target_kmh()`, etc. The class itself only owns control state.
"""

from __future__ import annotations

import logging
import math

from core.settings import Settings
from ui.popup.popup_window import PopupWindow

from .base import LongCtx, LongitudinalController, LongOutput

logger = logging.getLogger(__name__)

# Speed clamp for the set point (km/h)
_SPEED_MIN_KMH = 30.0
_SPEED_MAX_KMH = 130.0

# Low-pass time constant (s) for telemetry speed used only in the PID D-term.
_CC_KD_SPEED_SMOOTH_TAU_S = 0.15

# EMA time constants (s)
_CC_OUTPUT_EMA_TAU_S = 0.40
_CC_TARGET_SPEED_EMA_TAU_S = 0.5

# Gearshift D-term freeze — mirrors mapper's gearshift state machine
_CC_CLUTCH_ACTIVE_THRESHOLD = 0.05
_CC_GEARSHIFT_BLOCK_DURATION_S = 0.5
_CC_GEARSHIFT_RAMP_DURATION_S = 1.0

# Disable-on-stop guard (event + stop window)
_CC_DISARM_SPEED_MS = 0.3            # ~1.1 km/h
_CC_CRASH_SPEED_DROP_MS = 5.0        # per-tick speed drop
_CC_DISARM_PENDING_TIMEOUT_S = 5.0   # arming window after a triggering event

# Game throttle threshold above which CC bypasses brake-side output EMA
_CC_GAME_THROTTLE_OVERRIDE = 0.1

# Brake threshold for one-shot CC disengage (classic cruise behaviour:
# tap the brake → CC turns off, user must re-enable).
_CC_USER_BRAKE_DISENGAGE = 0.05


class CruiseController(LongitudinalController):
    """Set-speed PID with output smoothing, target management, and disarm guard."""

    name = "cc"

    def __init__(self) -> None:
        self._enabled: bool = False
        self._target_kmh: float | None = None

        # PID state
        self._integral_error: float = 0.0
        self._last_target_for_integral: float | None = None

        # D-term: speed EMA
        self._kd_smooth_speed_ms: float | None = None

        # Output EMA + target EMA
        self._output_ema_accel_ms2: float | None = None
        self._target_speed_ema_ms: float | None = None

        # Gearshift D-freeze state
        self._cc_clutch_active: bool = False
        self._cc_clutch_release_mono: float = -math.inf
        self._cc_prev_d_factor: float = 1.0

        # Disarm-on-stop tracking
        self._cc_disarm_pending_until: float = 0.0
        self._cc_prev_speed_ms: float | None = None

    # External state — read by the orchestrator for publishing/UI

    @property
    def active(self) -> bool:
        """True when this controller is bidding (enabled + has a target)."""
        return self._enabled and self._target_kmh is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def target_speed_kmh(self) -> float | None:
        return self._target_kmh

    # Lifecycle controls — driven by the orchestrator's button FSM

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def set_target_kmh(self, v: float) -> None:
        self._target_kmh = self._clamp_target_kmh(v)

    def set_target_from_speed_kmh(self, speed_kmh: float) -> None:
        self._target_kmh = self._clamp_target_kmh(round(speed_kmh))

    def change_target_kmh(self, delta: float) -> None:
        """Adjust set speed: coarse steps grid-snap, fine steps add."""
        if self._target_kmh is None:
            return
        inc = int(round(float(delta)))
        abs_inc = abs(inc)
        if abs_inc >= 5:
            ts = int(round(float(self._target_kmh)))
            if inc > 0:
                ts = ((ts // abs_inc) + 1) * abs_inc
            elif ts % abs_inc == 0:
                ts = ((ts // abs_inc) - 1) * abs_inc
            else:
                ts = (ts // abs_inc) * abs_inc
            self._target_kmh = self._clamp_target_kmh(float(ts))
        else:
            self._target_kmh = self._clamp_target_kmh(
                float(self._target_kmh) + float(delta),
            )

    # Main step

    def step(self, ctx: LongCtx) -> LongOutput:
        # Re-clamp target each tick so a tightened global limit drags the
        # active CC target down with it (user can never command above the
        # global speed limit, even via stale state).
        if self._target_kmh is not None:
            clamped = self._clamp_target_kmh(self._target_kmh)
            if clamped != self._target_kmh:
                self._target_kmh = clamped

        # User-brake disengage. Two independent triggers:
        #   1. Raw physical brake pedal (pre-OPD) exceeds threshold — direct
        #      user intent, ignores One Pedal Drive synthesised brake.
        #   2. In-game brake readback (gameBrake) exceeds the max brake we
        #      commanded over the last few ticks by more than the threshold —
        #      means user pressed the in-game brake (keyboard etc.) on top of
        #      whatever we sent. Comparing against recent commands tolerates
        #      the readback lag from the game.
        if self._enabled:
            game_brake_excess = ctx.game_brake - ctx.commanded_brake_recent_max
            if (
                ctx.user_raw_brake > _CC_USER_BRAKE_DISENGAGE
                or game_brake_excess > _CC_USER_BRAKE_DISENGAGE
            ):
                self._enabled = False
                logger.info("CC disabled — brake pressed", extra={"popup": True})

        # Auto-disable when truck enters a non-drivable state (cruise mode only)
        if (
            self._enabled
            and ctx.connected
            and Settings.cc_mode == "Cruise control"
            and (ctx.park_brake or ctx.gear_dashboard <= 0)
        ):
            self._enabled = False
            if ctx.park_brake:
                logger.info("CC disabled — parking brake engaged", extra={"popup": True})
            else:
                logger.info("CC disabled — gear neutral or reverse", extra={"popup": True})

        # Disarm-on-stop guard. CC stays armed through normal stops; only
        # disables when a crash or AEB_brake is followed by a full stop within
        # the timeout window.
        if self._enabled:
            crash_event = (
                self._cc_prev_speed_ms is not None
                and (self._cc_prev_speed_ms - ctx.speed_ms) >= _CC_CRASH_SPEED_DROP_MS
            )
            if crash_event or ctx.aeb_brake:
                self._cc_disarm_pending_until = ctx.now + _CC_DISARM_PENDING_TIMEOUT_S
            if (
                ctx.now < self._cc_disarm_pending_until
                and ctx.speed_ms < _CC_DISARM_SPEED_MS
            ):
                self._enabled = False
                self._cc_disarm_pending_until = 0.0
                label = "ACC" if Settings.acc_enabled else "CC"
                logger.info(f"{label} disabled for safety\ntap set/+ to resume")
                PopupWindow.emit(
                    f"{label} disabled", "disabled for safety\ntap set/+ to resume", "w",
                )
        else:
            self._cc_disarm_pending_until = 0.0
        self._cc_prev_speed_ms = ctx.speed_ms

        # Inactive — clear runtime smoothing/integrators so handover to active
        # later starts clean. Target and enabled flag are preserved.
        if not self.active:
            self._reset_runtime_state()
            return LongOutput(None, False)

        # External gates: telemetry, pause, device, em_stop
        if not (ctx.connected and not ctx.paused and not ctx.device_lost and not ctx.em_stop):
            return LongOutput(None, False)

        wanted_accel = self._speed_to_accel(ctx)

        # User game-throttle override: when user is pressing gas in-game and CC
        # wants to brake, bypass output EMA so brake response is immediate.
        if ctx.game_throttle > _CC_GAME_THROTTLE_OVERRIDE and wanted_accel < 0.0:
            self._output_ema_accel_ms2 = wanted_accel
        wanted_accel = self._smooth_output_accel_ema(wanted_accel, ctx.dt)

        accel_min = float(Settings.cc_accel_min_ms2)
        accel_max = float(Settings.cc_accel_max_ms2)
        wanted_accel = max(accel_min, min(accel_max, wanted_accel))

        return LongOutput(wanted_accel, True)

    def reset(self) -> None:
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        """Clear smoothers/integrators. Does not change enabled/target."""
        self._integral_error = 0.0
        self._kd_smooth_speed_ms = None
        self._output_ema_accel_ms2 = None
        self._target_speed_ema_ms = None

    # Internals

    @staticmethod
    def _clamp_target_kmh(v: float) -> float:
        upper = _SPEED_MAX_KMH
        glim = Settings.global_speed_limit_kmh
        if glim is not None and isinstance(glim, (int, float)) and math.isfinite(float(glim)):
            upper = min(upper, float(glim))
        return max(_SPEED_MIN_KMH, min(upper, v))

    def _gearshift_d_factor(self, now: float, clutch: float) -> float:
        """0.0 while clutched or in post-release block, 0→1 over ramp, 1 otherwise.

        Mirrors `accel_to_pedals` mapper's gearshift state machine so the
        D-term releases in lockstep with the mapper's integrators.
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

    def _speed_to_accel(self, ctx: LongCtx) -> float:
        speed_ms = ctx.speed_ms
        target_kmh = self._target_kmh
        if target_kmh is None:
            return 0.0

        if self._last_target_for_integral != target_kmh:
            self._integral_error = 0.0
            self._last_target_for_integral = target_kmh

        # In Speed-limiter mode, CC's m/s² output caps the user pedal in
        # sending_thread (`min(user, cc_mapper)`) instead of driving the
        # setpoint, so no target offset is needed here.
        target_ms = target_kmh / 3.6
        target_ms = self._smooth_target_speed_ema(target_ms, ctx.dt)

        kp = float(Settings.cc_kp)
        ki = float(Settings.cc_ki)
        kd = float(Settings.cc_kd)
        clamp = float(Settings.cc_integral_clamp)

        error_ms = target_ms - speed_ms
        self._integral_error += error_ms * ctx.dt
        self._integral_error = max(-clamp, min(clamp, self._integral_error))

        d_factor = self._gearshift_d_factor(ctx.now, ctx.game_clutch)

        if d_factor <= 0.0:
            # Freeze D-term during clutch + block window — prevents spurious
            # derivative spike from stalled-then-jumping speed on re-engage.
            speed_deriv = 0.0
        else:
            # Leading edge of the ramp: reseed EMA so first post-block
            # derivative starts at zero, not against stale pre-clutch value.
            if self._cc_prev_d_factor <= 0.0:
                self._kd_smooth_speed_ms = speed_ms
            tau = _CC_KD_SPEED_SMOOTH_TAU_S
            alpha = ctx.dt / (tau + ctx.dt) if tau > 0.0 else 1.0
            if self._kd_smooth_speed_ms is None:
                self._kd_smooth_speed_ms = speed_ms
                speed_deriv = 0.0
            else:
                prev_smooth = self._kd_smooth_speed_ms
                self._kd_smooth_speed_ms = alpha * speed_ms + (1.0 - alpha) * prev_smooth
                speed_deriv = (self._kd_smooth_speed_ms - prev_smooth) / ctx.dt
        self._cc_prev_d_factor = d_factor

        p_term = kp * error_ms
        i_term = ki * self._integral_error
        d_term = -kd * speed_deriv * d_factor

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
