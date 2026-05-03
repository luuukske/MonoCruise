"""
Global speed limiter — always on, never user-overridable, no buttons.

Operates in pedal space (post-mapping). Owns its own `AccelToPedals`
instance that shares state with the cruise-side mapper so learned bias /
capacity / output continuity carry over at handover.

Activation: when `Settings.global_speed_limit_kmh` is set. None = inactive
(no constraint, returns no-cap/no-brake).

Output semantics from `step_pedal()`:
- Returns `(None, 0.0)` when inactive.
- Returns `(gas_cap, brake_demand)` when active.
  * `gas_cap` ∈ [0, 1] — final gas pedal must be ≤ this value.
  * `brake_demand` ∈ [0, 1] — final brake pedal must be ≥ this value.
"""

from __future__ import annotations

import logging
import math
import time

from core.sending_thread.accel_to_pedals import AccelToPedals, MapperSharedState
from core.settings import Settings

from .base import LongCtx, LongitudinalController, LongOutput

logger = logging.getLogger(__name__)

# Internal PID — reuses Settings.cc_* gains for now.
_TARGET_SPEED_EMA_TAU_S = 0.5


class SpeedLimiter(LongitudinalController):
    """Global speed limiter operating in pedal space (post-mapping)."""

    name = "limiter"

    def __init__(self, shared: MapperSharedState) -> None:
        # Own mapper; shares state with the cruise-side mapper for handover continuity.
        self._mapper = AccelToPedals(shared)

        # Internal speed-tracking PID state
        self._integral_error: float = 0.0
        self._target_speed_ema_ms: float | None = None
        self._last_target_for_integral: float | None = None

        # Own dt tracking — sending_thread doesn't build a LongCtx for us.
        self._prev_mono: float | None = None

    @property
    def active(self) -> bool:
        v = Settings.global_speed_limit_kmh
        return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))

    def step(self, ctx: LongCtx) -> LongOutput:
        """Return the m/s² request (LongitudinalController interface).

        Use `step_pedal()` for the post-mapping path that sending_thread takes.
        """
        if not self.active:
            return LongOutput(None, False)
        wanted_ms2 = self._compute_pid_with_dt(ctx.speed_ms, ctx.dt)
        return LongOutput(wanted_ms2, True)

    def step_pedal(
        self,
        *,
        speed_ms: float,
        gear_dashboard: int,
        game_throttle: float,
        game_clutch: float,
        raw_accel_ms2: float,
        total_mass_kg: float,
        has_trailer: bool,
        max_accel_ms2: float,
        max_brake_ms2: float,
        road_pitch: float,
        learn: bool = True,
    ) -> tuple[float | None, float]:
        """Compute pedal-space cap/brake. Runs the limiter's own mapper only
        when ego is at/over the limit and the PID wants negative accel.

        Asymmetric on purpose — a real speed limiter never *pushes* the truck
        toward the cap. When ego is below the cap, this returns `(None, 0.0)`
        so cruise/user pedals see no constraint. Otherwise the cruise
        controller couldn't ever command a target below the limit (limiter
        and cruise would fight at the boundary).

        Returns `(None, 0.0)` when inactive or passive (below limit).
        Returns `(gas_cap, brake_demand)` when actively constraining.
        """
        if not self.active:
            self._reset_runtime_state()
            self._prev_mono = None
            return None, 0.0

        # Track dt internally — sending_thread doesn't build a LongCtx.
        now = time.monotonic()
        if self._prev_mono is None:
            dt = 0.02
        else:
            dt = max(min(now - self._prev_mono, 0.5), 1e-4)
        self._prev_mono = now

        wanted_ms2 = self._compute_pid_with_dt(speed_ms, dt)

        # Asymmetric: only run the mapper / constrain when the PID wants
        # negative accel (truck is at/over the limit). Below the limit the
        # limiter is silent so cruise/user keep full authority.
        """
        if wanted_ms2 >= 0.0:
            return None, 0.0
        """

        targets = self._mapper.step(
            wanted_accel_ms2=wanted_ms2,
            raw_accel_ms2=raw_accel_ms2,
            speed_ms=speed_ms,
            total_mass_kg=total_mass_kg,
            has_trailer=has_trailer,
            max_accel_ms2=max_accel_ms2,
            max_brake_ms2=max_brake_ms2,
            cruise_commanding=True,
            road_pitch=road_pitch,
            gear_dashboard=gear_dashboard,
            game_throttle=game_throttle,
            game_clutch=game_clutch,
            learn=learn,
        )
        return float(targets.gas), float(targets.brake)

    def reset(self) -> None:
        self._reset_runtime_state()
        self._mapper.reset_smoothing()

    def _reset_runtime_state(self) -> None:
        self._integral_error = 0.0
        self._target_speed_ema_ms = None
        self._last_target_for_integral = None

    # Internals

    def _compute_pid_with_dt(self, speed_ms: float, dt: float) -> float:
        """P+I controller: positive m/s² when below cap, negative when over.

        When ego is well below the limit, the PID requests positive accel
        (mapper produces near-full gas). The cruise mapper's request will
        usually be lower → wins the post-mapping merge. When ego nears or
        exceeds the limit, the PID drives toward zero or negative → mapper
        emits gas cap or brake demand that overrides cruise.
        """
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
        # No upper clamp — when ego is below the cap, the limiter must request
        # enough positive m/s² that its mapper produces a high enough gas
        # pedal value not to throttle the user. The post-mapping `min(user,
        # limiter_gas)` merge in sending_thread then leaves user pedal intact
        # below the cap. Lower clamp keeps brake authority bounded.
        accel_min = float(Settings.cc_accel_min_ms2)
        return max(accel_min, wanted)

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
