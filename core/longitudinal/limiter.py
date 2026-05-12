"""
Speed limiter — LongitudinalController subclass.

Lifecycle is driven by the orchestrator (CruiseControlThread):
  enable() / disable() / set_target_kmh(v) / reset()

No disengage logic lives here. The orchestrator decides when to enable or
disable the limiter; this class only runs the PID.

Gains: Settings.limiter_kp/ki/kd/integral_clamp/accel_min_ms2 — independent
of the CC gains so each controller can be tuned separately.
"""

from __future__ import annotations

import logging
import math

from core.settings import Settings

from .base import LongCtx, LongitudinalController, LongOutput

logger = logging.getLogger(__name__)

_TARGET_SPEED_EMA_TAU_S = 0.5


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
        effective_kp = kp * 1.3 if error_ms < 0 else kp
        wanted = effective_kp * shaped_error + ki * self._integral_error + d_term

        accel_min = float(Settings.limiter_accel_min_ms2)
        # Asymmetric clamp: only bound the lower side. Positive bids are left
        # uncapped so the mapper engages and the gas pedal cap tightens smoothly
        # as ego approaches the limit (continuous-tracker invariant, AGENTS.md).
        wanted = max(accel_min, wanted)

        # Return active=True every tick while enabled — continuous-tracker invariant
        # (AGENTS.md): the PID must run even when below the limit so the gas pedal
        # cap tightens progressively rather than snapping on at the boundary.
        return LongOutput(wanted, True)

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
