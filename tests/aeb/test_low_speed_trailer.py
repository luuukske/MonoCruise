"""Trailers behind ego at manoeuvring speed are ignored: coupling drives under one."""

from __future__ import annotations

import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import LowSpeedTrailerFilter
from tests.aeb.harness import EgoState, Frame, evaluate_frame, make_vehicle

_STAGE = "LowSpeedTrailerFilter"
_FLOOR_MS = CAL.trailer_ignore_below_kmh / 3.6


def _frame(ego_speed_ms: float, *, z: float, is_trailer: bool = True) -> Frame:
    """Ego faces +Z: z < 0 puts the body behind the cab, as when coupling."""
    ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=ego_speed_ms)
    target = make_vehicle(
        vid=1, x=0.0, z=z, yaw_deg=180.0, speed=0.0, is_trailer=is_trailer,
    )
    return Frame(ego=ego, vehicles=[target], t=0.0)


def _reasons(result, vid: int = 1) -> list[str]:
    return [r.reason for r in result.suppression_reasons.get(vid, [])]


@pytest.mark.parametrize("ego_kmh", [12.0, 19.0, -12.0, -19.0])
def test_trailer_behind_the_cab_is_suppressed_below_the_floor(ego_kmh):
    result = evaluate_frame(_frame(ego_kmh / 3.6, z=-6.0), CAL)
    assert _STAGE in _reasons(result)
    assert 1 in result.suppressed_ids
    assert 1 not in result.colliding_ids


@pytest.mark.parametrize("ego_kmh", [20.0, 45.0, -45.0])
def test_trailer_behind_passes_the_stage_at_or_above_the_floor(ego_kmh):
    result = evaluate_frame(_frame(ego_kmh / 3.6, z=-6.0), CAL)
    assert _STAGE not in _reasons(result)


@pytest.mark.parametrize("ego_kmh", [12.0, -12.0])
def test_trailer_ahead_keeps_full_aeb(ego_kmh):
    """Queueing behind a parked trailer is a labelled TP class; do not touch it."""
    result = evaluate_frame(_frame(ego_kmh / 3.6, z=10.0), CAL)
    assert _STAGE not in _reasons(result)


def test_non_trailer_behind_is_untouched():
    result = evaluate_frame(_frame(12.0 / 3.6, z=-6.0, is_trailer=False), CAL)
    assert _STAGE not in _reasons(result)


class _Ctx:
    """Minimal duck-typed FilterContext: the stage reads five fields."""

    def __init__(self, ego_speed: float, dz: float, latched: set) -> None:
        self.v = type("V", (), {"is_trailer": True, "id": 1})()
        self.ego_speed = ego_speed
        self.latched_threat_ids = latched
        # yaw 0 rad points the cab along -Z, so dz > 0 sits behind it.
        self.dx, self.dz = 0.0, dz
        self.ego_yaw_rad = 0.0


def test_latched_threat_keeps_its_pipeline_seat_under_the_floor():
    stage = LowSpeedTrailerFilter(CAL)
    slow = _FLOOR_MS - 2.0
    assert stage.apply(_Ctx(slow, 6.0, set())).suppressed is True
    assert stage.apply(_Ctx(slow, 6.0, {1})).suppressed is False
