"""
Longitudinal controller stack — parent class + shared types.

Three children inherit:
- AdaptiveCruiseController  (following distance)
- CruiseController          (set-speed PID)
- SpeedLimiter              (global cap, future)

Each child's `step(ctx)` returns a m/s² request. The orchestrator
(`cruise_control_thread`) arbitrates `min(...)` of active children and
publishes the winner to telemetry for `accel_to_pedals.step()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LongCtx:
    """Per-tick input snapshot for longitudinal controllers.

    Built once by the orchestrator from telemetry/pedal/AEB readings, then
    handed to every child's `step()` so children don't touch the registry.
    """
    now: float                  # monotonic clock
    dt: float                   # seconds since previous tick
    speed_ms: float
    gear_dashboard: int
    park_brake: bool
    game_throttle: float
    game_clutch: float
    game_brake: float
    aeb_brake: bool             # event flag (drives CC disarm-on-stop)
    connected: bool
    paused: bool
    em_stop: bool
    device_lost: bool


@dataclass(slots=True)
class LongOutput:
    """One controller's bid for a single tick.

    `wanted_ms2 is None` ⇔ inactive ⇔ this controller does not bid this tick
    (orchestrator excludes it from the `min()` arbitration).
    """
    wanted_ms2: float | None
    active: bool


class LongitudinalController:
    """Base class. Children override `step()` and `reset()`."""

    name: str = "base"

    def step(self, ctx: LongCtx) -> LongOutput:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal smoothing/integrator state."""
        pass

    @property
    def active(self) -> bool:
        """Is this controller currently bidding? Default False — children override."""
        return False

    # Shared math helpers — lifted from the cruise control PID and mapper.

    @staticmethod
    def _ema_step(current: float, sample: float, alpha: float) -> float:
        return current + alpha * (sample - current)

    @staticmethod
    def _ema_alpha(dt: float, tau_s: float) -> float:
        if tau_s <= 0.0:
            return 1.0
        return 1.0 - math.exp(-dt / max(tau_s, 1e-6))
