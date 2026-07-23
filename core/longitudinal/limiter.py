"""Speed limiter PID child. See core/longitudinal/README.md."""

from __future__ import annotations

import logging
import math

from core.settings import Settings

from .base import LongCtx, LongitudinalController, LongOutput

logger = logging.getLogger(__name__)

_TARGET_SPEED_EMA_TAU_S = 0.5

# Overshoot kp boost: multiplier reached once overshoot exceeds the blend band.
_OVERSHOOT_BOOST = 2.0
_OVERSHOOT_BOOST_BAND_MS = 0.3

# Overshoot protection.
_OVERSHOOT_CUBIC_K = 0.25

# Deadband before overshoot protection bids at all, in m/s (0.15 m/s is ~0.5 km/h).
# Leaves ACC room to sit at the cap without the protection floor winning the
# arbitration min() on every small drift over the limit. Not a gate on the
# limiter bid itself: the continuous tracker still runs every tick (AGENTS.md).
_OVERSHOOT_DEADBAND_MS = 0.15

# Engagement gate: overshoot protection is for a standing overshoot the mapper
# is failing to clear, not one already coming down.
# See `core/longitudinal/README.md`.
_OVERSHOOT_CLEAR_S = 1.0
_CUBIC_ENGAGE_TAU_S = 0.35
_CUBIC_LEAK_TAU_S = 0.25

# Measured accel for the gate: tracking differentiator on speed. Shorter tau
# See `core/longitudinal/README.md`.
_ACCEL_TRACK_TAU_S = 0.15


class SpeedLimiter(LongitudinalController):
    """Set-speed PID for the speed-limiter mode."""

    name = "limiter"

    def __init__(self) -> None:
        self._enabled: bool = False
        self._target_kmh: float | None = None

        self._integral_error: float = 0.0
        self._prev_shaped_error: float | None = None
        self._last_target_for_integral: float | None = None
        self._target_speed_ema_ms: float | None = None

        # Overshoot-protection gate state.
        # See `core/longitudinal/README.md`.
        self._accel_track_smooth_ms: float | None = None
        # Protection engagement scalar in [0, 1], separately smoothed so the
        # cubic builds slowly and leaks fast.
        # See `core/longitudinal/README.md`.
        self._cubic_engage: float = 0.0

    @property
    def active(self) -> bool:
        return self._enabled and self._target_kmh is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def target_speed_kmh(self) -> float | None:
        return self._target_kmh

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def set_target_kmh(self, v: float) -> None:
        self._target_kmh = v

    def reset(self) -> None:
        self._integral_error = 0.0
        self._prev_shaped_error = None
        self._last_target_for_integral = None
        self._target_speed_ema_ms = None
        self._accel_track_smooth_ms = None
        self._cubic_engage = 0.0

    def step(self, ctx: LongCtx) -> LongOutput:
        if not self.active:
            self.reset()
            return LongOutput(None, False)

        target_kmh = float(self._target_kmh)

        if self._last_target_for_integral != target_kmh:
            self._integral_error = 0.0
            self._last_target_for_integral = target_kmh

        target_ms = target_kmh / 3.6
        target_ms = self._smooth_target_ema(target_ms, ctx.dt)

        kp = float(Settings.limiter_kp)
        ki = float(Settings.limiter_ki)
        kd = float(Settings.limiter_kd)
        clamp = float(Settings.limiter_integral_clamp)

        error_ms = target_ms - ctx.speed_ms
        # Power-shaped error: small errors amplified, large errors compressed.
        shaped_error = math.copysign(abs(error_ms) ** 0.9, error_ms)
        self._integral_error += shaped_error * ctx.dt
        self._integral_error = max(-clamp, min(clamp, self._integral_error))

        d_term = kd * (shaped_error - self._prev_shaped_error) / ctx.dt if self._prev_shaped_error is not None else 0.0
        self._prev_shaped_error = shaped_error

        # Boost kp when overshooting so ego decelerates faster back to the limit.
        # See `core/longitudinal/README.md`.
        overshoot_frac = min(1.0, max(0.0, -error_ms) / _OVERSHOOT_BOOST_BAND_MS)
        effective_kp = kp * (1.0 + (_OVERSHOOT_BOOST - 1.0) * overshoot_frac)
        wanted = effective_kp * shaped_error + ki * self._integral_error + d_term

        accel_min = float(Settings.limiter_accel_min_ms2)
        accel_min = self._overshoot_floor(accel_min, -error_ms, ctx)
        # Asymmetric clamp: only bound the lower side. Positive bids are left
        # See `core/longitudinal/README.md`.
        wanted = max(accel_min, wanted)

        # Return active=True every tick while enabled: continuous-tracker invariant
        # See `core/longitudinal/README.md`.
        return LongOutput(wanted, True)

    def _overshoot_floor(self, accel_min: float, overshoot_ms: float, ctx: LongCtx) -> float:
        """Deepen the decel floor for a standing overshoot the mapper isn't clearing. See `core/longitudinal/README.md`."""
        dt = max(float(ctx.dt), 1e-4)
        accel_meas = self._update_accel_track(float(ctx.speed_ms), dt)

        # Cubic runs off the excess past the deadband, so protection still enters
        # with zero slope at its own boundary rather than stepping in.
        excess_ms = overshoot_ms - _OVERSHOOT_DEADBAND_MS

        if excess_ms <= 0.0:
            # Inside the deadband or below the limit: no protection bid, and let
            # any built-up scale drain so a fresh overshoot starts from zero
            # rather than a stale value.
            self._cubic_engage += (dt / (_CUBIC_LEAK_TAU_S + dt)) * (0.0 - self._cubic_engage)
            return accel_min

        cubic_ms2 = min(abs(accel_min), _OVERSHOOT_CUBIC_K * excess_ms ** 3)

        # Decel that would clear the current overshoot within the target window.
        # Measured against the full overshoot, not the excess: the recovery
        # target is the limit itself, the deadband only holds the bid off.
        required_decel = overshoot_ms / _OVERSHOOT_CLEAR_S
        decel_meas = max(0.0, -accel_meas)
        # External decel = whatever ego is shedding BEYOND what the limiter floor
        # See `core/longitudinal/README.md`.
        commanded_floor_decel = abs(accel_min) + cubic_ms2 * self._cubic_engage
        decel_external = max(0.0, decel_meas - commanded_floor_decel)
        # 1 when nothing else is slowing ego (mapper failing), 0 once external
        # decel alone would clear the overshoot in time. Linear between.
        shortfall = 1.0 - decel_external / required_decel if required_decel > 0.0 else 0.0
        target = min(1.0, max(0.0, shortfall))

        # Asymmetric smoothing: build slowly, leak fast. Biases toward not
        # See `core/longitudinal/README.md`.
        tau = _CUBIC_ENGAGE_TAU_S if target > self._cubic_engage else _CUBIC_LEAK_TAU_S
        self._cubic_engage += (dt / (tau + dt)) * (target - self._cubic_engage)

        return accel_min - cubic_ms2 * self._cubic_engage

    def _update_accel_track(self, speed_ms: float, dt: float) -> float:
        """Tracking-differentiator longitudinal accel (m/s²), positive = speeding up."""
        tau = _ACCEL_TRACK_TAU_S
        if self._accel_track_smooth_ms is None:
            self._accel_track_smooth_ms = speed_ms
            return 0.0
        accel = (speed_ms - self._accel_track_smooth_ms) / tau
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
        self._accel_track_smooth_ms += alpha * (speed_ms - self._accel_track_smooth_ms)
        return accel

    def _smooth_target_ema(self, target_ms: float, dt: float) -> float:
        tau = _TARGET_SPEED_EMA_TAU_S
        alpha = dt / (tau + dt) if tau > 0.0 else 1.0
        if self._target_speed_ema_ms is None:
            self._target_speed_ema_ms = target_ms
        else:
            self._target_speed_ema_ms = (
                alpha * target_ms + (1.0 - alpha) * self._target_speed_ema_ms
            )
        return self._target_speed_ema_ms

