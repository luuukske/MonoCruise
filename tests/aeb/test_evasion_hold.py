"""Latched-id and in-lane-closer bypasses for EgoEvasion / CornerEntry.

Clips 8c0df2b1, c7c9f6dd, ffe2a4e1 dropped the lead mid-brake or pre-brake
because these stages have no latch seat and treat Lane.EGO as non-evidence.
"""

from __future__ import annotations

import math

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import (
    CornerEntryStationaryFilter,
    CornerEntryStationaryFilterMirrored,
    EgoEvasionFilter,
)
from core.aeb.lane_frame import Lane, classify, in_lane_closing
from tests.aeb.harness import evaluate_frame
from tests.aeb.scenarios.fp_mp_stationary_corner_entry import build as build_corner
from tests.aeb.scenarios.fp_parked_shoulder import build as build_shoulder
from tests.aeb.scenarios.tp_stopped_in_lane import build as build_stopped


class _Boom:
    """If evasion geometry runs, the test failed to take an early bypass."""

    def __init__(self, speed: float = 0.0) -> None:
        self.speed = speed

    def __getattr__(self, name):
        raise AssertionError(f"evasion geometry must not run ({name})")


class _EvasionCtx:
    """Duck-typed FilterContext: EgoEvasionFilter early-outs only."""

    def __init__(
        self,
        *,
        vid: int = 1,
        latched: set | None = None,
        head_on: bool = False,
        co_directional: bool = True,
        v_speed: float = 0.0,
        ego_speed: float = 20.0,
        dx: float = 0.0,
        dz: float = 20.0,
        ego_fwd_x: float = 0.0,
        ego_fwd_z: float = 1.0,
        fwd_dot: float = 1.0,
        lane: Lane = Lane.EGO,
        follow: set | None = None,
        ref_kmh: float = 80.0,
    ) -> None:
        self.v = type("V", (), {"id": vid, "speed": v_speed})()
        self.latched_threat_ids = set() if latched is None else latched
        self.head_on = head_on
        self.co_directional = co_directional
        self.lane = lane
        self.all_target_arcs = [_Boom(v_speed)]
        self.follow_threat_ids = set() if follow is None else follow
        self.ref_kmh_for_filter = ref_kmh
        self.cal = CAL
        self.dx = dx
        self.dz = dz
        self.ego_fwd_x = ego_fwd_x
        self.ego_fwd_z = ego_fwd_z
        self.ego_speed = ego_speed
        self.fwd_dot = fwd_dot
        self.v_travel_speed = abs(v_speed)
        dummy = object()
        self.ego_evasion_left = dummy
        self.ego_evasion_right = dummy
        self.near_head_on = False
        self.cross_padding = 0.0
        self.precomputed_cross_arcs = None
        self.lateral_gap = 0.0
        self.ego_arc = dummy


_STAGE = EgoEvasionFilter(CAL)


def test_latched_id_skips_evasion_geometry():
    ctx = _EvasionCtx(latched={1}, dx=5.0, co_directional=False, head_on=True)
    assert _STAGE.apply(ctx).suppressed is False


def test_in_lane_stationary_skips_evasion_geometry():
    ctx = _EvasionCtx(dx=0.4, dz=18.0, v_speed=0.0)
    assert _STAGE.apply(ctx).suppressed is False


def test_in_lane_closing_codir_skips_evasion_geometry():
    ctx = _EvasionCtx(
        dx=0.5, dz=25.0, v_speed=15.0, ego_speed=25.0, co_directional=True,
    )
    assert _STAGE.apply(ctx).suppressed is False


def test_shoulder_offset_does_not_take_the_in_lane_bypass():
    ctx = _EvasionCtx(dx=2.2, dz=20.0, v_speed=0.0)
    assert in_lane_closing(
        ctx.dx, ctx.dz, ctx.ego_fwd_x, ctx.ego_fwd_z, ctx.ego_speed,
        ctx.v_travel_speed, ctx.fwd_dot, CAL.lane_half_width,
    ) is False


def test_faster_lead_does_not_take_the_in_lane_bypass():
    ctx = _EvasionCtx(
        dx=0.0, dz=20.0, v_speed=30.0, ego_speed=20.0, co_directional=True,
    )
    assert in_lane_closing(
        ctx.dx, ctx.dz, ctx.ego_fwd_x, ctx.ego_fwd_z, ctx.ego_speed,
        ctx.v_travel_speed, ctx.fwd_dot, CAL.lane_half_width,
    ) is False


def test_weak_closing_stays_on_the_evasion_check():
    ctx = _EvasionCtx(dx=0.0, dz=20.0, v_speed=20.0, ego_speed=20.5)
    assert in_lane_closing(
        ctx.dx, ctx.dz, ctx.ego_fwd_x, ctx.ego_fwd_z, ctx.ego_speed,
        ctx.v_travel_speed, ctx.fwd_dot, CAL.lane_half_width, CAL.aeb_min_closing_ms,
    ) is False


def test_classify_still_reaches_off_road():
    assert classify(0.5, CAL) is Lane.EGO
    assert classify(CAL.lane_half_width + 0.01, CAL) is not Lane.EGO
    far = 2.0 * CAL.lane_separation
    assert classify(far - 0.01, CAL) is Lane.OPPOSITE_OR_OUTER
    assert classify(far, CAL) is Lane.OFF_ROAD


def test_stopped_in_lane_is_not_evasion_filtered():
    result = evaluate_frame(build_stopped()[0], CAL)
    assert 1 not in result.evasion_filtered_ids
    assert 1 not in result.suppressed_ids


def test_parked_shoulder_still_suppressed():
    result = evaluate_frame(build_shoulder()[0], CAL)
    reasons = [r.reason for r in result.suppression_reasons.get(1, [])]
    assert reasons
    assert 1 not in result.colliding_ids


class _CornerCtx:
    """Mode B geometry from fp_mp_stationary_corner_entry v1 (x=-0.83, z=19.95)."""

    def __init__(self, *, latched: set | None = None, lane: Lane = Lane.EGO) -> None:
        yaw = math.radians(170.5)
        self.v = type("V", (), {"id": 1})()
        self.latched_threat_ids = set() if latched is None else latched
        self.follow_threat_ids: set = set()
        self.abs_v_speed = 0.0
        self.ego_curvature = 0.0
        self.dist = 19.97
        self.fwd_dot = 0.986
        self.lane = lane
        self.dx = -0.83
        self.dz = 19.95
        self.ego_fwd_x = 0.0
        self.ego_fwd_z = 1.0
        self.veh_fwd_x = -math.sin(yaw)
        self.veh_fwd_z = -math.cos(yaw)


def test_corner_entry_mode_b_still_fires_unlatched():
    stage = CornerEntryStationaryFilter(CAL)
    res = stage.apply(_CornerCtx())
    assert res.suppressed is True
    assert res.reason == "CornerEntryStationaryFilter"


def test_corner_entry_latched_id_passes():
    stage = CornerEntryStationaryFilter(CAL)
    assert stage.apply(_CornerCtx(latched={1})).suppressed is False


def test_corner_entry_mode_a_stays_suppressed_when_latched():
    stage = CornerEntryStationaryFilter(CAL)
    ctx = _CornerCtx(latched={1}, lane=Lane.OPPOSITE_OR_OUTER)
    assert stage.apply(ctx).suppressed is True


def test_corner_entry_mirrored_has_no_latch_seat():
    stage = CornerEntryStationaryFilterMirrored(CAL)
    ctx = _CornerCtx(latched={1}, lane=Lane.OPPOSITE_OR_OUTER)
    ctx.ego_curvature = 0.02
    assert stage.apply(ctx).suppressed is True


def test_corner_entry_mirrored_follow_threat_passes():
    stage = CornerEntryStationaryFilterMirrored(CAL)
    ctx = _CornerCtx(lane=Lane.OPPOSITE_OR_OUTER)
    ctx.ego_curvature = 0.02
    ctx.follow_threat_ids = {1}
    assert stage.apply(ctx).suppressed is False


def test_corner_queue_pipeline_releases_when_latched():
    frame = build_corner()[0]
    cold = evaluate_frame(frame, CAL)
    assert any(
        r.reason == "CornerEntryStationaryFilter"
        for r in cold.suppression_reasons.get(1, [])
    )
    hot = evaluate_frame(frame, CAL, latched_threat_ids={1})
    reasons = [r.reason for r in hot.suppression_reasons.get(1, [])]
    assert "CornerEntryStationaryFilter" not in reasons
