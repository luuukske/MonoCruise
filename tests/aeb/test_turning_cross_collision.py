"""Turning TMP centre contact before a clear endpoint needs measured corroboration."""
from dataclasses import replace
from types import SimpleNamespace
import math

from core.aeb.calibration import DEFAULT
from core.aeb.filters import TmpCrossTrafficFilter
from core.aeb.lane_frame import Lane, classify, project_to_ego_arc
from core.radar.traffic import build_arc


def _context(miss=1.0, rate=-1.0, lateral_shift=0.0):
    cal = replace(DEFAULT, max_evasion_lat_g=0.0)
    ego = build_arc(0.0, 0.0, 0.0, 20.0, 0.0, 1.2, 3.0)
    turn = build_arc(0.0, 0.0, math.pi / 2, 6.0, 0.05, 1.3, 3.0)
    tx, tz = turn.position_at_time(1.0)
    target = build_arc(-tx + lateral_shift, -20.0 - tz, math.pi / 2,
                       6.0, 0.05, 1.3, 3.0)
    v = SimpleNamespace(id=1, is_tmp=True, speed=6.0, size=SimpleNamespace(width=2.8),
                        position=SimpleNamespace(x=target.start_x, z=target.start_z))
    return cal, SimpleNamespace(
        v=v, latched_threat_ids=set(), co_directional=False, reversing=False, abs_v_speed=6.0,
        dx=target.start_x, dz=target.start_z, ego_fwd_x=ego.fwd_x,
        ego_fwd_z=ego.fwd_z, ego_hw=1.2, ego_arc=ego,
        v_curvature=0.05, all_target_arcs=[target], precomputed_cross_arcs=[[target]],
        d_miss=miss, d_miss_rate=rate, lateral_gap=0.0,
    )


def test_turning_collision_before_clear_endpoint_passes():
    cal, ctx = _context()
    end = ctx.all_target_arcs[0].position_at_time(3.0)
    assert classify(project_to_ego_arc(ctx.ego_arc, *end)[1], cal) != Lane.EGO
    assert not TmpCrossTrafficFilter(cal).apply(ctx).suppressed


def test_turning_phantom_without_measured_track_stays_suppressed():
    cal, ctx = _context(miss=None, rate=None)
    assert TmpCrossTrafficFilter(cal).apply(ctx).suppressed


def test_turning_measured_clear_pass_stays_suppressed():
    cal, ctx = _context(miss=5.0)
    assert TmpCrossTrafficFilter(cal).apply(ctx).suppressed


def test_turning_opening_miss_stays_suppressed():
    cal, ctx = _context(rate=1.0)
    assert TmpCrossTrafficFilter(cal).apply(ctx).suppressed
