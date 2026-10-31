"""ACC following-distance cap wrapper. See core/cruise_control_thread/README.md."""

from __future__ import annotations

import math

from core.cruise_control_thread.acc_controller import AdaptiveCruiseController as _ACCFollowingDistance
from core.settings import Settings

from .base import LongCtx, LongitudinalController, LongOutput


def free_road_target_ms2(cruise_bid_ms2: float | None, inner: _ACCFollowingDistance) -> float:
    """ACC's cap with no lead, and what a far lead lifts toward: the cruise bid.

    Bounded one onset dead band above `a_max`, so a lead that starts to bind is met at
    the plain jerk rate instead of a step. See core/longitudinal/README.md.
    """
    cfg = inner.config
    floor = cfg.no_lead_ceiling_ms2
    if cruise_bid_ms2 is None or not math.isfinite(cruise_bid_ms2):
        return floor
    return max(floor, min(cruise_bid_ms2, cfg.a_max_ms2 + cfg.j_onset_dead_ms2))


class AdaptiveCruiseController(LongitudinalController):
    """Following-distance cap, exposed as a longitudinal child."""

    name = "acc"

    def __init__(self) -> None:
        self._inner = _ACCFollowingDistance()

    @property
    def active(self) -> bool:
        # ACC is "armed" whenever the user enabled it and we're in cruise mode.
        # See `core/longitudinal/README.md`.
        return bool(Settings.acc_enabled) and Settings.cc_mode == "Cruise control"

    def step(self, ctx: LongCtx, cruise_bid_ms2: float | None = None) -> LongOutput:
        if not self.active:
            self._inner.reset()
            return LongOutput(None, False)
        cap_ms2 = self._inner.accel_cap_ms2(ctx.speed_ms, free_road_target_ms2(cruise_bid_ms2, self._inner))
        return LongOutput(cap_ms2, True)

    @property
    def crawl_follow(self) -> bool:
        """Following a lead measured moving, outside the standstill hold. ACC_ARCHITECTURE §10.3."""
        return self.active and self._inner.crawl_follow

    def reset(self) -> None:
        self._inner.reset()
