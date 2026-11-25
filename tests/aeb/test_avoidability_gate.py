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
from core.aeb.bend_chain import BendChainFilter
from core.aeb.filters import (
    CornerEntryStationaryFilter, CornerEntryStationaryFilterMirrored, EgoEvasionFilter,
    FilterResult, OppositeLaneFilter, OutOfLaneParallelFilter, SweepPassFilter,
    TmpCrossTrafficFilter,
    build_pipeline, lane_clear_held,
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


def test_wide_oncoming_scope_needs_speed_and_a_car_off_the_opposite_lane():
    """The 1.3 to 2.4 m band counts only at road speed for a car the arc puts in ego's lane."""
    wide = replace(CAL, oncoming_gate_wide_m=2.4, oncoming_gate_wide_rate_mps=-0.25,
                   oncoming_gate_wide_min_speed_ms=14.0, oncoming_gate_wide_opposite_lane=False)

    def ctx(lane=Lane.EGO, speed=_V, dist=30.0, rate=-0.5):
        c = _oncoming(_ctx(25.0, deadline=0.1), d_miss=2.0, rate=rate)
        c.lane, c.ego_speed, c.dist = lane, speed, dist
        return c

    wide = replace(wide, oncoming_gate_wide_range_m=45.0)
    assert _onc_suppressed(ctx(), replace(CAL, oncoming_gate_wide_m=0.0))
    assert not _onc_suppressed(ctx(), wide)
    assert _onc_suppressed(ctx(lane=Lane.OPPOSITE_OR_OUTER), wide)
    assert _onc_suppressed(ctx(speed=10.0), wide)
    # Past the range a slow drift is a bend artefact; only the base closing rate counts.
    assert _onc_suppressed(ctx(dist=60.0), wide)
    assert not _onc_suppressed(ctx(dist=60.0, rate=-2.0), wide)


def _crossing(ctx, *, d_miss: float | None = 0.3, rate: float | None = 1.0, near: bool = True,
              fwd_dot: float = -0.8, lat: float = 0.0):
    """A slow TMP car turning across ego's lane; past the centre line its miss grows again."""
    ctx.head_on, ctx.near_head_on, ctx.fwd_dot = False, near, fwd_dot
    ctx.d_miss, ctx.d_miss_rate = d_miss, rate
    ctx.abs_v_speed = ctx.v_ground_meas = 4.5
    ctx.dx = lat
    return ctx


def _cross_suppressed(ctx, cal=CAL) -> bool:
    return AvoidabilityGate(_AlwaysDrops(), cal, crossing=True).apply(ctx).suppressed


def test_tmp_crosser_on_a_line_through_ego_ends_the_drop_at_the_deadline():
    """249e9d51: a left turner at 16 km/h, ego at 90; the miss line stays under 1 m to impact,
    the rate positive once it passes ego's centre, and TmpCross dropped it from 31.7 m to impact."""
    demand = _demand(25.0)
    ctx = _crossing(_ctx(25.0, deadline=demand - 0.5))
    assert not _cross_suppressed(ctx)
    assert ctx.deadline_released
    assert _cross_suppressed(_crossing(_ctx(25.0, deadline=demand + 0.5)))


def test_tmp_crossing_scope_needs_a_measured_line_inside_ego_width():
    assert _cross_suppressed(_crossing(_ctx(25.0, deadline=0.1), d_miss=CAL.ego_half_width + 0.1))
    assert _cross_suppressed(_crossing(_ctx(25.0, deadline=0.1), d_miss=None))
    assert _cross_suppressed(_crossing(_ctx(25.0, deadline=0.1)),
                             replace(CAL, avoidability_gate_tmp_cross=False))


def test_tmp_crossing_scope_is_cars_coming_at_ego():
    """A crosser coming the other way; a 90 degree crosser stays with the stage (5 more FPs)."""
    assert _cross_suppressed(_crossing(_ctx(25.0, deadline=0.1), near=False))


def test_a_head_on_car_counts_only_on_egos_own_path():
    """Head on in the next lane is the ordinary pass; within 0.5 m of ego's path it is not."""
    assert not _cross_suppressed(_crossing(_ctx(25.0, deadline=0.1), fwd_dot=-1.0, lat=0.4))
    assert _cross_suppressed(_crossing(_ctx(25.0, deadline=0.1), fwd_dot=-1.0, lat=0.6))
    assert not _cross_suppressed(_crossing(_ctx(25.0, deadline=0.1), fwd_dot=-0.85, lat=0.6))


def test_tmp_crossing_scope_skips_the_parked_body_map_rules():
    """Lane track and lane clear speak about parked bodies; a crosser keeps the bare gate."""
    ctx = _crossing(_ctx(25.0, deadline=0.1))
    ctx.lane_track_s, ctx.lane_clear_s, ctx.latched_threat_ids = 0.0, 99.0, set()
    assert lane_clear_held(ctx, CAL)
    assert not _cross_suppressed(ctx)


def test_only_guess_stages_are_gated():
    gated = {
        type(s._inner) for s in build_pipeline(CAL) if isinstance(s, AvoidabilityGate)
    }
    assert gated == {
        OutOfLaneParallelFilter, SweepPassFilter,
        CornerEntryStationaryFilter, CornerEntryStationaryFilterMirrored,
        OppositeLaneFilter, BendChainFilter, TmpCrossTrafficFilter,
    }
    onc = [s for s in build_pipeline(CAL) if isinstance(s, AvoidabilityGate) and s._oncoming]
    assert [type(s._inner) for s in onc] == [OppositeLaneFilter]
    cross = [s for s in build_pipeline(CAL) if isinstance(s, AvoidabilityGate) and s._crossing]
    assert [type(s._inner) for s in cross] == [TmpCrossTrafficFilter]
    bare = {type(s) for s in build_pipeline(CAL) if not isinstance(s, AvoidabilityGate)}
    assert {EgoEvasionFilter} <= bare
