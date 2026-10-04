"""Drop filters keep a braked-for vehicle whose body is still in ego's lane ahead.

README latched-threat hold, item 2. Card 131: these stages let go mid-brake.
"""

from __future__ import annotations

import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import (
    CoDirectionalDivergeFilter,
    OppositeLaneFilter,
    TmpCrossTrafficFilter,
    TurningCrossTrafficFilter,
    _latched_body_in_lane,
)
from core.radar.traffic import build_arc


class _ReachedGeometry(Exception):
    """The stage read past its early-outs, so the latch bypass was not taken."""


class _Vehicle:
    id = 1
    is_tmp = True

    def __getattr__(self, name):
        raise _ReachedGeometry(name)


class _LatchCtx:
    """Duck-typed FilterContext: a 13 m body, ego arc starting at ego's front."""

    def __init__(self, *, co_directional: bool, latched: bool = True,
                 centre_x: float = 0.5, centre_z: float = -15.0,
                 half_len: float = 6.5, ego_speed: float = 15.0) -> None:
        self.v = _Vehicle()
        self.latched_threat_ids = {1} if latched else set()
        self.ego_speed = ego_speed
        self.ego_arc = build_arc(0.0, 0.0, 0.0, ego_speed, 0.0, CAL.ego_half_width, 3.0)
        self.all_target_arcs = [build_arc(
            centre_x, centre_z, 0.0, 0.0, 0.0, 1.25, 3.0,
            fwd_len=half_len, back_len=half_len,
        )]
        self.co_directional = co_directional
        self.abs_v_speed = 10.0
        self.head_on = False
        self.reversing = False

    def __getattr__(self, name):
        raise _ReachedGeometry(name)


_STAGES = [
    (OppositeLaneFilter, False),
    (CoDirectionalDivergeFilter, True),
    (TurningCrossTrafficFilter, False),
    (TmpCrossTrafficFilter, False),
]


@pytest.mark.parametrize("stage_cls, co_dir", _STAGES)
def test_braked_for_body_in_lane_ahead_is_kept(stage_cls, co_dir):
    ctx = _LatchCtx(co_directional=co_dir)
    assert stage_cls(CAL).apply(ctx).suppressed is False


@pytest.mark.parametrize("variant", [
    {"latched": False},
    {"centre_x": 4.5},
    {"centre_z": 3.0, "half_len": 2.0},
    {"ego_speed": 1.0},
])
def test_stage_keeps_its_say_otherwise(variant):
    # Not braked for, out of lane, alongside ego (1a9f5ffa), or under the engage floor.
    assert _latched_body_in_lane(_LatchCtx(co_directional=True), CAL) is True
    ctx = _LatchCtx(co_directional=True, **variant)
    assert _latched_body_in_lane(ctx, CAL) is False


def test_engage_floor_is_the_speed_bar():
    floor = CAL.aeb_min_engage_speed_kmh
    above = _LatchCtx(co_directional=True, ego_speed=(floor + 0.5) / 3.6)
    below = _LatchCtx(co_directional=True, ego_speed=(floor - 0.5) / 3.6)
    assert _latched_body_in_lane(above, CAL) is True
    assert _latched_body_in_lane(below, CAL) is False
