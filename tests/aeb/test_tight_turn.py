"""TightTurnFilter: a walking-pace turn whose arc wraps onto a body ego pulls away from.

README "TightTurnFilter". The measured line of sight must not be closing, and the drop
ends at the braking deadline like every guess stage.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import build_pipeline
from core.aeb.turn_completion import TightTurnFilter

_ON = replace(CAL, tight_turn_enabled=True)


def _ctx(*, kappa=0.15, d_miss=3.0, rate=0.5, latched=(), deadline=False):
    v = SimpleNamespace(
        id=7, position=SimpleNamespace(x=4.0, z=-6.0),
        size=SimpleNamespace(length=4.5, width=1.9), trailers=(),
    )
    return SimpleNamespace(
        v=v, ego_curvature=kappa, d_miss=d_miss, d_miss_rate=rate,
        latched_threat_ids=set(latched), follow_threat_ids=set(),
        dx=4.0, dz=-6.0, ego_fwd_x=0.0, ego_fwd_z=-1.0, ego_hw=CAL.ego_half_width,
        v_yaw_rad=0.0, deadline_passed=deadline, deadline_released=False,
    )


def _suppressed(ctx, cal=_ON) -> bool:
    return TightTurnFilter(cal).apply(ctx).suppressed


def test_a_body_ego_pulls_away_from_in_a_tight_turn_is_dropped():
    assert _suppressed(_ctx())
    assert _suppressed(_ctx(kappa=-0.15))


def test_a_closing_line_a_gentle_turn_or_no_track_is_kept():
    assert not _suppressed(_ctx(rate=-0.5))
    assert not _suppressed(_ctx(kappa=0.05))
    assert not _suppressed(_ctx(d_miss=None, rate=None))
    assert not _suppressed(_ctx(latched={7}))


def test_the_drop_ends_at_the_braking_deadline_for_a_line_through_the_body():
    assert not _suppressed(_ctx(d_miss=0.5, deadline=True))
    assert _suppressed(_ctx(d_miss=0.5, deadline=False))
    assert _suppressed(_ctx(d_miss=5.0, deadline=True))


def test_switch_off_is_the_bare_pipeline():
    assert not _suppressed(_ctx(), replace(_ON, tight_turn_enabled=False))
    names = [s.name for s in build_pipeline(CAL)]
    assert names.index("TightTurnFilter") == names.index("EgoEvasionFilter") - 1
