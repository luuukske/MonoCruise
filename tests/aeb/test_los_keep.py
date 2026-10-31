"""Measured line-of-sight keeps in TmpCrossTrafficFilter and OutOfLaneParallelFilter.

README "TmpCrossTrafficFilter" and "OutOfLaneParallelFilter", LOS keep. The harness
has no LOS track, so these tests feed the measured miss line into its context.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

import tests.aeb.harness as h
from core.aeb.calibration import DEFAULT as CAL
from tests.aeb.scenarios import fp_co_directional_outer_lane, fp_tmp_side_road_right_turn

_OFF = replace(CAL, tmp_cross_los_keep_m=0.0, tmp_cross_los_keep_straight_m=0.0,
               out_of_lane_los_keep_m=0.0)
_TMP = replace(
    _OFF, tmp_cross_los_keep_m=1.5, tmp_cross_los_keep_straight_m=2.5,
    tmp_cross_los_keep_range_m=28.0, tmp_cross_los_keep_min_fwd_dot=-0.93,
    tmp_cross_los_keep_min_speed_ms=6.7,
)
_OOL = replace(
    _OFF, out_of_lane_los_keep_m=1.95, out_of_lane_los_keep_rate_mps=0.0,
    out_of_lane_los_keep_range_m=20.0, out_of_lane_los_keep_min_lat_m=2.0,
)


def _drops(monkeypatch, scenario, stage: str, cal, d_miss, rate) -> int:
    """Frames where ``stage`` drops vehicle 1 with the given measured miss line."""
    orig = h.FilterContext

    def with_los(**kw):
        ctx = orig(**kw)
        ctx.d_miss, ctx.d_miss_rate = d_miss, rate
        return ctx

    monkeypatch.setattr(h, "FilterContext", with_los)
    n = 0
    for frame in scenario.build():
        reasons = h.evaluate_frame(frame, calibration=cal).suppression_reasons.get(1, [])
        n += any(r.reason == stage for r in reasons)
    return n


@pytest.fixture
def tmp_base(monkeypatch) -> int:
    n = _drops(monkeypatch, fp_tmp_side_road_right_turn, "TmpCrossTrafficFilter", _OFF, 0.5, -3.0)
    assert n > 0
    return n


def test_tmp_cross_keeps_a_crosser_on_a_near_closing_line(monkeypatch, tmp_base):
    kept = _drops(monkeypatch, fp_tmp_side_road_right_turn, "TmpCrossTrafficFilter",
                  _TMP, 0.5, -3.0)
    assert kept < tmp_base


@pytest.mark.parametrize("d_miss, rate, cal", [
    (3.0, -3.0, _TMP),                                        # clear of both bars
    (0.5, -0.5, _TMP),                                        # not closing
    (None, None, _TMP),                                       # no track
    (0.5, -3.0, replace(_TMP, tmp_cross_los_keep_range_m=5.0)),
    (0.5, -3.0, replace(_TMP, tmp_cross_los_keep_min_speed_ms=9.0)),
])
def test_tmp_cross_keep_needs_every_condition(monkeypatch, d_miss, rate, cal):
    def drops(c):
        return _drops(monkeypatch, fp_tmp_side_road_right_turn, "TmpCrossTrafficFilter",
                      c, d_miss, rate)
    assert drops(cal) == drops(_OFF) > 0


@pytest.fixture
def ool_base(monkeypatch) -> int:
    n = _drops(monkeypatch, fp_co_directional_outer_lane, "OutOfLaneParallelFilter",
               _OFF, 1.0, -1.0)
    assert n > 0
    return n


def test_out_of_lane_keeps_a_body_on_a_near_line(monkeypatch, ool_base):
    kept = _drops(monkeypatch, fp_co_directional_outer_lane, "OutOfLaneParallelFilter",
                  _OOL, 1.0, -1.0)
    assert kept < ool_base


@pytest.mark.parametrize("d_miss, rate, cal", [
    (2.5, -1.0, _OOL),                                        # miss line clear of ego
    (1.0, 0.5, _OOL),                                         # opening
    (None, None, _OOL),
    (1.0, -1.0, replace(_OOL, out_of_lane_los_keep_range_m=3.0)),
    (1.0, -1.0, replace(_OOL, out_of_lane_los_keep_min_lat_m=3.0)),
])
def test_out_of_lane_keep_needs_every_condition(monkeypatch, d_miss, rate, cal):
    def drops(c):
        return _drops(monkeypatch, fp_co_directional_outer_lane, "OutOfLaneParallelFilter",
                      c, d_miss, rate)
    assert drops(cal) == drops(_OFF) > 0
