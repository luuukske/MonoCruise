"""
ACC longitudinal child — wraps the IIDM+CAH following-distance controller
from `core/cruise_control_thread/acc_controller.py`.

The inner controller publishes an upper bound on commanded longitudinal
accel (m/s²) from the lead chain. This child surfaces it as a `LongOutput`
so the orchestrator can arbitrate `min(ACC, CC, …)`.

Goal: smooth following distance. No set-speed PID lives here.
"""

from __future__ import annotations

from core.cruise_control_thread.acc_controller import AdaptiveCruiseController as _ACCFollowingDistance
from core.settings import Settings

from .base import LongCtx, LongitudinalController, LongOutput


class AdaptiveCruiseController(LongitudinalController):
    """Following-distance accel cap, exposed as a longitudinal child."""

    name = "acc"

    def __init__(self) -> None:
        self._inner = _ACCFollowingDistance()

    @property
    def active(self) -> bool:
        # ACC is "armed" whenever the user enabled it and we're in cruise mode.
        # Whether it actually constrains accel depends on lead presence — the
        # inner controller returns its no-lead ceiling when no lead is in path,
        # which the orchestrator will lose to CC's lower request via min().
        return bool(Settings.acc_enabled) and Settings.cc_mode == "Cruise control"

    def step(self, ctx: LongCtx) -> LongOutput:
        if not self.active:
            self._inner.reset()
            return LongOutput(None, False)
        cap_ms2 = self._inner.accel_cap_ms2(ctx.speed_ms)
        return LongOutput(cap_ms2, True)

    def reset(self) -> None:
        self._inner.reset()
