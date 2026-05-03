"""
Global speed limiter — always on, never user-overridable, no buttons.

PID only — no mapper. The limiter publishes a wanted m/s² request that
sending_thread folds into `min(cruise_wanted, limiter_wanted)` before
the single shared mapper runs. This avoids the per-instance state drift
that two parallel mappers used to suffer at the limit boundary (cruise
target dropping below the cap could fail to take over because each
mapper had its own `wanted_smooth` / fast PID state).

Activation: when `Settings.global_speed_limit_kmh` is set. None → inactive.
`step()` returns `None` when inactive; sending_thread skips the merge.
`is_constraining` is True when the last computed wanted was negative
(truck at/over the limit) — sending_thread uses this to force the
post-mapping `min(user, mapper_gas)` user-cap regardless of cc_mode.
"""

from __future__ import annotations

import logging
import math
import time

from core.settings import Settings

from .base import LongCtx, LongitudinalController, LongOutput

logger = logging.getLogger(__name__)

_TARGET_SPEED_EMA_TAU_S = 0.5


class SpeedLimiter(LongitudinalController):
    """Global speed limiter — PID-only m/s² source for the shared mapper."""

    name = "limiter"

    def __init__(self) -> None:
        # Internal speed-tracking PID state
        self._integral_error: float = 0.0
        self._target_speed_ema_ms: float | None = None
        self._last_target_for_integral: float | None = None

        # Own dt tracking — sending_thread doesn't build a LongCtx for us.
        self._prev_mono: float | None = None

        # Last computed wanted (for `is_constraining`). 0 = neutral.
        self._last_wanted_ms2: float = 0.0

    @property
    def active(self) -> bool:
        v = Settings.global_speed_limit_kmh
        return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))

    @property
    def is_constraining(self) -> bool:
        """True when the last `step()` wanted negative m/s² (truck at/over limit).

        Sending_thread uses this to force `min(user, mapper_gas)` post-mapping
        so the user can't push past the cap regardless of cc_mode.
        """
        return self.active and self._last_wanted_ms2 < 0.0

    def step(self, ctx: LongCtx) -> LongOutput:
        """Return the wanted m/s² request. Used by sending_thread (via
        `step_wanted`) and by any orchestrator wanting LongOutput semantics.
        """
        if not self.active:
            return LongOutput(None, False)
        wanted = self._compute_pid_with_dt(ctx.speed_ms, ctx.dt)
        return LongOutput(wanted, True)

    def step_wanted(self, speed_ms: float) -> float | None:
        """Compute wanted m/s². Tracks dt internally (sending_thread doesn't
        build a LongCtx). Returns None when inactive.
        """
        if not self.active:
            self._reset_runtime_state()
            self._prev_mono = None
            self._last_wanted_ms2 = 0.0
            return None

        now = time.monotonic()
        if self._prev_mono is None:
            dt = 0.02
        else:
            dt = max(min(now - self._prev_mono, 0.5), 1e-4)
        self._prev_mono = now

        return self._compute_pid_with_dt(speed_ms, dt)

    def reset(self) -> None:
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        self._integral_error = 0.0
        self._target_speed_ema_ms = None
        self._last_target_for_integral = None
        self._last_wanted_ms2 = 0.0

    def _compute_pid_with_dt(self, speed_ms: float, dt: float) -> float:
        """P+I controller against the global cap. Positive when below, negative when over."""
        target_kmh = float(Settings.global_speed_limit_kmh or 0.0)

        if self._last_target_for_integral != target_kmh:
            self._integral_error = 0.0
            self._last_target_for_integral = target_kmh

        target_ms = target_kmh / 3.6
        target_ms = self._smooth_target_ema(target_ms, dt)

        kp = float(Settings.cc_kp)
        ki = float(Settings.cc_ki)
        clamp = float(Settings.cc_integral_clamp)

        error_ms = target_ms - speed_ms
        self._integral_error += error_ms * dt
        self._integral_error = max(-clamp, min(clamp, self._integral_error))

        wanted = kp * error_ms + ki * self._integral_error
        # No upper clamp — limiter must allow user full gas while below cap.
        # Lower clamp bounds the limiter's brake authority.
        accel_min = float(Settings.cc_accel_min_ms2)
        wanted = max(accel_min, wanted)
        self._last_wanted_ms2 = wanted
        return wanted

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
