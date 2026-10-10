"""Avoidability gate: a guessed drop of a parked body ends at the braking deadline.

README "Avoidability gate". The wrapped stages drop a stationary body on a guess
about where ego is going. The gate lets that guess stand only while AEB could
still stop short of the body if the guess turns out wrong.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clearance import clearance_required
from core.aeb.avoidability import AvoidabilityGate, measured_miss_inside_body
from core.aeb.filters import (
    CornerEntryStationaryFilter, CornerEntryStationaryFilterMirrored, EgoEvasionFilter,
    FilterResult, OppositeLaneFilter, OutOfLaneParallelFilter, SweepPassFilter,
    TmpCrossTrafficFilter,
    build_pipeline,
)
from core.aeb.lane_frame import Lane
from core.radar.traffic import build_arc, capsule_extents

_EGO_OFFSET = (CAL.arc_start_pctg - 0.5) * (2.0 * CAL.ego_half_length)
_CAP_FWD, _CAP_BACK = capsule_extents(
    CAL.ego_half_length, CAL.ego_half_length, _EGO_OFFSET,
)
_V = 15.0
_HALF_LEN = 2.25
_LAG_S = 0.3


class _AlwaysDrops:
    name = "AlwaysDrops"

    def apply(self, ctx) -> FilterResult:
        return FilterResult(suppressed=True, reason=self.name)


def _ctx(gap_m: float, *, deadline: float, ground: float | None = 0.0,
         speed_field: float = 0.0, d_miss: float | None = 0.0,
         trailers: tuple = ()):
    """Ego at the origin heading -z; a parked car whose rear sits ``gap_m`` ahead."""
    ego_arc = build_arc(
        0.0, 0.0, 0.0, _V, 0.0, CAL.ego_half_width, 3.0,
        fwd_len=_CAP_FWD, back_len=_CAP_BACK,
        parallel_margin_scale=CAL.capsule_parallel_margin_scale,
    )
    cz = -(gap_m + _HALF_LEN)
    body = build_arc(
        0.0, cz, 0.0, 0.0, 0.0, 0.95, 3.0, fwd_len=_HALF_LEN, back_len=_HALF_LEN,
        parallel_margin_scale=CAL.capsule_parallel_margin_scale,
    )

    def clearance_fn(arcs):
        return clearance_required(
            ego_arc, arcs, _V, CAL, lag_s=_LAG_S, pad_m=0.0,
            front_to_surface=_CAP_FWD, near_horizon_s=3.0,
        )

    v = SimpleNamespace(
        id=7, position=SimpleNamespace(x=0.0, z=cz),
        size=SimpleNamespace(length=2.0 * _HALF_LEN, width=1.9), trailers=trailers,
    )
    return SimpleNamespace(
        v=v, v_yaw_rad=0.0, abs_v_speed=speed_field, v_ground_meas=ground,
        d_miss=d_miss, dx=0.0, dz=cz, ego_fwd_x=0.0, ego_fwd_z=-1.0,
        ego_hw=CAL.ego_half_width, ego_arc=ego_arc, ego_speed=_V,
        all_target_arcs=[body], precomputed_cross_arcs=None, cross_padding=0.0,
        near_head_on=False, lane=Lane.OPPOSITE_OR_OUTER, lateral_gap=0.0,
        dynamic_horizon=3.0, brake_deadline_ms2=deadline, clearance_fn=clearance_fn,
        deadline_passed=None, deadline_released=False,
    )


def _demand(gap_m: float) -> float:
    ctx = _ctx(gap_m, deadline=1.0)
    return ctx.clearance_fn(ctx.all_target_arcs).required_ms2


def _suppressed(ctx, cal=CAL) -> bool:
    return AvoidabilityGate(_AlwaysDrops(), cal).apply(ctx).suppressed


def test_guess_stands_while_braking_can_still_stop():
    demand = _demand(25.0)
    assert _suppressed(_ctx(25.0, deadline=demand + 0.5))


def test_guess_ends_at_the_braking_deadline():
    demand = _demand(25.0)
    ctx = _ctx(25.0, deadline=demand - 0.5)
    assert not _suppressed(ctx)
    assert ctx.deadline_released


def test_a_moving_target_keeps_the_filter_verdict():
    assert _suppressed(_ctx(25.0, deadline=0.1, ground=6.0))


def test_a_tmp_speed_field_stuck_at_zero_is_not_parked():
    """TMP can report 0 for a car pulling away; its raw positions say otherwise."""
    assert _suppressed(_ctx(25.0, deadline=0.1, speed_field=0.0, ground=6.0))


def test_no_track_fails_toward_the_filter():
    assert _suppressed(_ctx(25.0, deadline=0.1, ground=None))
    assert _suppressed(_ctx(25.0, deadline=0.1, d_miss=None))


def test_a_measured_line_clear_of_the_body_keeps_the_drop():
    assert _suppressed(_ctx(25.0, deadline=0.1, d_miss=4.0))


def test_a_body_ego_already_overlaps_is_no_prediction():
    """Ghosted TMP traffic beside ego hits every path at t=0; braking changes nothing."""
    assert _suppressed(_ctx(-1.0, deadline=0.1))


def test_disabled_gate_is_the_bare_stage():
    off = replace(CAL, avoidability_gate_enabled=False)
    assert _suppressed(_ctx(25.0, deadline=0.1), cal=off)


def test_trailer_counts_only_on_the_side_it_reaches():
    """A tractor 10 m right of ego's line: its trailer decides, by the way it points."""
    def trailer(x_offset: float):
        return SimpleNamespace(
            position=SimpleNamespace(x=10.0 + x_offset, z=-20.0),
            rotation=SimpleNamespace(euler=lambda: (0.0, 90.0, 0.0)),
            size=SimpleNamespace(length=13.6, width=2.5),
        )

    def ctx_with(tr):
        ctx = _ctx(15.0, deadline=0.1, d_miss=10.0, trailers=(tr,))
        ctx.v.position = SimpleNamespace(x=10.0, z=-20.0)
        ctx.dx, ctx.dz = 10.0, -20.0
        return ctx

    assert measured_miss_inside_body(ctx_with(trailer(-8.0)))
    assert not measured_miss_inside_body(ctx_with(trailer(+8.0)))


def _oncoming(ctx, *, d_miss: float = 0.0, rate: float | None = -2.0, head_on: bool = True):
    ctx.head_on = head_on
    ctx.d_miss = d_miss
    ctx.d_miss_rate = rate
    ctx.abs_v_speed = ctx.v_ground_meas = 20.0
    return ctx


def _onc_suppressed(ctx, cal=CAL) -> bool:
    return AvoidabilityGate(_AlwaysDrops(), cal, oncoming=True).apply(ctx).suppressed


def test_oncoming_drifting_in_ends_the_drop_at_the_deadline():
    demand = _demand(25.0)
    assert not _onc_suppressed(_oncoming(_ctx(25.0, deadline=demand - 0.5)))
    assert _onc_suppressed(_oncoming(_ctx(25.0, deadline=demand + 0.5)))


def test_oncoming_needs_a_closing_miss_inside_ego_width():
    """Adjacent oncoming passes graze the bar; only a closing line through ego's own width counts."""
    assert _onc_suppressed(_oncoming(_ctx(25.0, deadline=0.1), rate=-0.5))
    assert _onc_suppressed(_oncoming(_ctx(25.0, deadline=0.1), rate=None))
    assert _onc_suppressed(_oncoming(_ctx(25.0, deadline=0.1), d_miss=CAL.ego_half_width + 0.1))
    assert _onc_suppressed(_oncoming(_ctx(25.0, deadline=0.1), head_on=False))
    assert _onc_suppressed(_oncoming(_ctx(25.0, deadline=0.1)),
                           replace(CAL, avoidability_gate_oncoming=False))


def test_only_guess_stages_are_gated():
    gated = {
        type(s._inner) for s in build_pipeline(CAL) if isinstance(s, AvoidabilityGate)
    }
    assert gated == {
        OutOfLaneParallelFilter, SweepPassFilter,
        CornerEntryStationaryFilter, CornerEntryStationaryFilterMirrored,
        OppositeLaneFilter,
    }
    onc = [s for s in build_pipeline(CAL) if isinstance(s, AvoidabilityGate) and s._oncoming]
    assert [type(s._inner) for s in onc] == [OppositeLaneFilter]
    bare = {type(s) for s in build_pipeline(CAL) if not isinstance(s, AvoidabilityGate)}
    assert {EgoEvasionFilter, TmpCrossTrafficFilter} <= bare
