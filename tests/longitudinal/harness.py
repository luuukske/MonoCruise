"""Shared LongCtx builder for the longitudinal invariant tests. See core/longitudinal/README.md."""
from __future__ import annotations

from core.longitudinal.base import LongCtx


def make_ctx(speed_ms: float, *, dt: float = 0.02, now: float = 0.0, **overrides) -> LongCtx:
    """Nominal driving context: connected, unpaused, no pedals, drive gear."""
    fields = dict(
        now=now,
        dt=dt,
        speed_ms=speed_ms,
        gear_dashboard=1,
        park_brake=False,
        game_throttle=0.0,
        game_clutch=0.0,
        game_brake=0.0,
        aeb_brake=False,
        connected=True,
        paused=False,
        em_stop=False,
        device_lost=False,
    )
    fields.update(overrides)
    return LongCtx(**fields)
