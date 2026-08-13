"""Geometry-graded engage fraction and the oncoming body-separation guard.

Corpus context: the largest missed-positive class was in-lane co-directional
rear-ends the pipeline tracked and warned on, but whose demand peaked below
the flat 0.85 engage bar. The bar is a hedge against uncertain geometry, so it
is graded by the same certainty the confirm window already uses. See
core/aeb/README.md (geometry-graded engage, oncoming clearance).
"""
from __future__ import annotations

import math
import struct
from dataclasses import replace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clip_eval import run_headless
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, LiveAEB,
    RadarFrameRecord,
)
from core.aeb.filters import (
    FilterContext, OppositeLaneFilter, _build_vehicle_collision_data,
)
from core.aeb.lane_frame import Lane
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT
from core.radar.traffic import build_arc, capsule_extents
from tests.aeb.harness import make_vehicle

_HZ = 30.0
_FLAT = replace(CAL, aeb_engage_frac_certain=CAL.aeb_engage_frac)
# The shipped default currently equals aeb_engage_frac (0.85 in-game trial from
# 2026-08-11), so grading is flat. Drive the mechanism explicitly to keep it covered.
_GRADED = replace(CAL, aeb_engage_frac_certain=0.70)


def _stopped_ahead_clip(ego_ms: float, gap_m: float, capacity: float,
                        n: int = 60) -> Clip:
    """Ego closing on a stalled vehicle dead ahead, facing the same way it does.

    Co-directional and aligned in ego's lane, so it is the certain geometry the
    graded bar is scoped to.
    """
    dt = 1.0 / _HZ
    # yaw = pi faces +Z like ego: quaternion (cos(pi/2), 0, sin(pi/2), 0).
    flat: list = [0.0, 0.0, gap_m, 0.0, 0.0, 1.0, 0.0, 2.5, 3.0, 6.0, 0.0, 0.0]
    flat += [0, 3, 0, 0] + [0.0] * 30
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        buf = struct.pack(_TOTAL_FORMAT, *flat)
        assert len(buf) == _BUF_SIZE
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=ego_ms * t,
                             rotationX=0.5, rotationY=0.0, speed=ego_ms),
            traffic_buf=buf, parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=capacity, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    return Clip(metadata=ClipMetadata.create(session_kind="SP"),
                radar_frames=frames, aeb_ticks=ticks)


def _brake_range(clip: Clip, cal, ego_ms: float, gap_m: float) -> float | None:
    """Range to the obstacle at the first brake tick, or None if it never braked."""
    braked = [e for e in run_headless(clip, cal=cal) if e.aeb_brake]
    return gap_m - ego_ms * braked[0].t_rel if braked else None


def test_graded_bar_brakes_earlier_on_an_in_lane_obstacle():
    """Same clip, same capacity: a lower certain-geometry bar engages further out."""
    ego_ms, gap = 22.0, 60.0
    clip = _stopped_ahead_clip(ego_ms, gap, capacity=10.0)
    flat_r = _brake_range(clip, _FLAT, ego_ms, gap)
    graded_r = _brake_range(clip, _GRADED, ego_ms, gap)
    assert flat_r is not None and graded_r is not None
    assert graded_r > flat_r, (
        f"graded bar should brake further out: {graded_r:.1f} m vs {flat_r:.1f} m"
    )


def test_shipped_default_does_not_grade_certain_geometry_earlier():
    """Certain-geometry bar must not sit below the base bar.

    `aeb_engage_frac_certain` started as a lower bar (0.70 in preview.12). The
    2026-08-11 in-game trial flattened it onto `aeb_engage_frac`. Pin that
    certain traffic is never invited in earlier than oblique traffic.
    """
    assert CAL.aeb_engage_frac_certain >= CAL.aeb_engage_frac


def test_graded_bar_is_the_only_thing_that_changed():
    """A clip whose demand never reaches either bar still stays silent."""
    # Huge gap: required decel stays far below both fractions for the whole clip.
    clip = _stopped_ahead_clip(8.0, 180.0, capacity=10.0, n=30)
    assert not any(e.aeb_brake for e in run_headless(clip))


# TODO: re-enable once crawl engage has a length-aware or speed-aware gate.
# Keep the pad: ego length is unknown, a longer truck must still stop.
def _soft_crawl_rear_end_should_not_engage_without_slam():
    """Crawl in-lane stop: demand under 0.70 and TTB above slam stays warn-only."""
    # gap=14 m at 3.6 m/s: demand stays soft and TTB never reaches the 0.50 s
    # slam. The removed certain-TTB bridge used to engage here early.
    clip = _stopped_ahead_clip(3.6, 14.0, capacity=10.0, n=45)
    assert not any(e.aeb_brake for e in run_headless(clip))


def test_soft_crawl_far_threat_stays_silent():
    """Far soft demand with TTB well above the slam stays silent."""
    clip = _stopped_ahead_clip(3.6, 30.0, capacity=10.0, n=45)
    assert not any(e.aeb_brake for e in run_headless(clip))


def _oncoming_ctx(d_miss: float | None, lateral_m: float = 3.0):
    """Head-on target in its own lane, at ``lateral_m`` from ego's straight arc."""
    cal = CAL
    ego_speed = 20.0
    ego_hw, ego_half_l = cal.ego_half_width, cal.ego_half_length
    offset = (cal.arc_start_pctg - 0.5) * (2.0 * ego_half_l)
    fwd_len, back_len = capsule_extents(ego_half_l, ego_half_l, offset)
    # Ego at the origin heading +Z (yaw pi), straight.
    ego_arc = build_arc(0.0, offset, math.pi, ego_speed, 0.0, ego_hw,
                        cal.arc_horizon_max, fwd_len=fwd_len, back_len=back_len,
                        parallel_margin_scale=cal.capsule_parallel_margin_scale)
    v = make_vehicle(vid=7, x=lateral_m, z=40.0, yaw_deg=0.0, speed=20.0)
    (arcs, cross_pad, cross_arcs, v_yaw, abs_v, vfx, vfz, v_curv
     ) = _build_vehicle_collision_data(
        v, cal.arc_horizon_max, math.pi, ego_arc.fwd_x, ego_arc.fwd_z, cal,
    )
    return FilterContext(
        v=v, ego_arc=ego_arc, ego_braked_arc=ego_arc,
        ego_evasion_left=None, ego_evasion_right=None,
        ego_x=0.0, ego_y=0.0, ego_z=0.0, ego_yaw_rad=math.pi,
        ego_speed=ego_speed, ego_pitch_rad=0.0, ego_curvature=0.0,
        ego_fwd_x=ego_arc.fwd_x, ego_fwd_z=ego_arc.fwd_z, ego_hw=ego_hw,
        dynamic_horizon=cal.arc_horizon_max, tmp_traffic_session=False,
        ref_kmh_for_filter=ego_speed * 3.6, cal=cal,
        dx=lateral_m, dz=40.0, dist_sq=lateral_m ** 2 + 1600.0,
        dist=math.hypot(lateral_m, 40.0),
        v_yaw_rad=v_yaw, abs_v_speed=abs_v, veh_fwd_x=vfx, veh_fwd_z=vfz,
        v_curvature=v_curv, all_target_arcs=arcs,
        precomputed_cross_arcs=cross_arcs, cross_padding=cross_pad,
        head_on=True, near_head_on=True, lane=Lane.OPPOSITE_OR_OUTER,
        fwd_dot=-1.0, d_miss=d_miss,
    )


def test_body_separation_shortcut_survives_without_a_measurement():
    """No LOS track yet: the pose-only fast path is unchanged (fails open)."""
    res = OppositeLaneFilter(CAL).apply(_oncoming_ctx(d_miss=None))
    assert res.suppressed and res.reason == "OppositeLaneFilter"


def test_body_separation_shortcut_survives_a_measurement_that_agrees():
    res = OppositeLaneFilter(CAL).apply(_oncoming_ctx(d_miss=3.0))
    assert res.suppressed


def test_body_separation_shortcut_yields_to_a_measured_collision_course():
    """Pose says the bodies clear, the measured track says they meet."""
    res = OppositeLaneFilter(CAL).apply(_oncoming_ctx(d_miss=0.1))
    assert not res.suppressed, (
        "a measured head-on course must not be waved through on pose alone"
    )


def test_body_separation_yields_to_turn_into_path_closing(monkeypatch):
    """Inflated arc d_abs with collapsing |lat| and shrinking CBDR miss."""
    import core.aeb.filters as filters_mod

    ctx = _oncoming_ctx(d_miss=4.0, lateral_m=0.1)
    ctx.ego_curvature = 0.05
    ctx.d_miss_rate = -5.0
    # Honest adjacent has d_abs~|lat|; turn-into inflates arc offset (~30x).
    monkeypatch.setattr(
        filters_mod, "project_to_ego_arc",
        lambda arc, x, z: (40.0, 3.0),
    )
    # Inflated turn-into: d_abs/|lat| must clear the ratio gate (DEFAULT=10).
    assert 3.0 >= 0.1 * CAL.oncoming_closing_dabs_lat_ratio
    res = OppositeLaneFilter(CAL).apply(ctx)
    assert not res.suppressed


def test_body_separation_keeps_honest_adjacent_despite_closing_rate(monkeypatch):
    """Low d_abs/|lat| must not skip body-sep (6a35-style adjacent)."""
    import core.aeb.filters as filters_mod

    ctx = _oncoming_ctx(d_miss=5.0, lateral_m=0.8)
    ctx.ego_curvature = 0.05
    ctx.d_miss_rate = -10.0
    monkeypatch.setattr(
        filters_mod, "project_to_ego_arc",
        lambda arc, x, z: (40.0, 3.5),
    )
    # 3.5/0.8 = 4.375 < ratio 10 → closing false → body-sep.
    res = OppositeLaneFilter(CAL).apply(ctx)
    assert res.suppressed and res.reason == "OppositeLaneFilter"


def test_body_separation_keeps_shared_bend_when_lat_floors():
    """Miss-rate closing alone must not skip body-sep if |lat| stays wide."""
    ctx = _oncoming_ctx(d_miss=4.0, lateral_m=1.2)
    ctx.ego_curvature = 0.05
    ctx.d_miss_rate = -5.0
    res = OppositeLaneFilter(CAL).apply(ctx)
    assert res.suppressed and res.reason == "OppositeLaneFilter"
