"""ACC following-distance cap wrapper. See core/cruise_control_thread/README.md."""

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
        # See `core/longitudinal/README.md`.
        return bool(Settings.acc_enabled) and Settings.cc_mode == "Cruise control"

    def step(self, ctx: LongCtx) -> LongOutput:
        if not self.active:
            self._inner.reset()
            return LongOutput(None, False)
        cap_ms2 = self._inner.accel_cap_ms2(ctx.speed_ms)
        return LongOutput(cap_ms2, True)

    def reset(self) -> None:
        self._inner.reset()

