"""Regression: clip 8aefa3a2, an articulated bus understeering into a head-on.

The driver winds to full lock while the tires are already saturated, so the old
steer-only ego path drew a 5 m circle where the bus was actually holding ~25 m.
These pin the path, not the warn: three other gates (elevation, cross-traffic
filters, the head-on braking arc) still drop the target in this clip.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

CLIP_NAME = "20260918T194047Z_auto_crash_8aefa3a2.json.gz"
_T_IMPACT = 4.45
_TARGET_VID = 190


def _clip_path() -> Path | None:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    p = Path(base) / "MonoCruise" / "aeb_clips" / CLIP_NAME
    return p if p.is_file() else None


pytestmark = [
    pytest.mark.needs_clips,
    pytest.mark.skipif(
        _clip_path() is None, reason="clip 8aefa3a2 not in local clip store",
    ),
]


@pytest.fixture(scope="module")
def clip():
    from core.aeb.clip_store import ClipStore
    c = ClipStore().load(_clip_path())
    assert c is not None
    return c


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@pytest.fixture(scope="module")
def path_track(clip):
    """Per-frame model state plus the curvature the bus actually held."""
    from core.aeb.calibration import DEFAULT as CAL, ego_path_params
    from core.aeb.clip_replay import ego_path_replay
    from core.aeb.clip_timebase import replay_frames
    from core.radar.ego_path_model import EgoPathModel

    frames = replay_frames(clip)
    _tkin, warm = ego_path_replay(clip)
    model = EgoPathModel(params=ego_path_params(CAL), gain=warm)
    t0 = frames[0].t_mono
    rows = []
    for i, f in enumerate(frames):
        e = f.ego
        state = model.step(f.t_wall, e.rotationX * 2.0 * math.pi, e.speed, e.userSteer)
        realized = None
        if 4 <= i < len(frames) - 4:
            span = frames[i + 4].t_wall - frames[i - 4].t_wall
            if span > 0 and abs(e.speed) > 1.0:
                realized = _wrap(
                    frames[i + 4].ego.rotationX * 2.0 * math.pi
                    - frames[i - 4].ego.rotationX * 2.0 * math.pi
                ) / span / e.speed
        rows.append((f.t_mono - t0, state, realized))
    return rows


def test_path_follows_the_line_the_bus_actually_held(path_track):
    """Through the understeer, the arc must sit on the driven radius, not the wheel's."""
    checked = 0
    for t, state, realized in path_track:
        if not (3.5 <= t <= 4.3) or realized is None:
            continue
        checked += 1
        assert abs(state.kappa_path - realized) <= 0.20 * abs(realized), (
            f"t={t:.2f}: path kappa {state.kappa_path:+.4f} vs realized {realized:+.4f}"
        )
        # The steer term alone is what used to be drawn; it is far off here.
        assert abs(state.kappa_steer) > 1.5 * abs(state.kappa_path)
    assert checked > 20


def test_grip_cap_engages_a_second_before_impact(path_track):
    first = next((t for t, s, _ in path_track if s.saturated), None)
    assert first is not None
    assert _T_IMPACT - first >= 1.0


def test_full_lock_no_longer_draws_a_five_metre_circle(path_track):
    at_lock = [
        s for t, s, _ in path_track
        if 4.05 <= t <= 4.4
    ]
    assert at_lock
    for s in at_lock:
        assert 1.0 / abs(s.kappa_steer) < 6.0      # what the wheel asked for
        assert 1.0 / abs(s.kappa_path) > 15.0      # what the bus could hold


def test_target_is_on_the_arc_for_two_seconds_before_impact(clip, path_track):
    """Geometry only: filters and the elevation gate still drop 190 in this clip."""
    from core.aeb.calibration import DEFAULT as CAL
    from core.aeb.clip_replay import decode_radar_stream, nearest_frame_t
    from core.radar.traffic import arc_arc_collision, build_arc, capsule_extents

    veh_by_t, ego_by_t, frame_t, _off = decode_radar_stream(clip)
    t0 = frame_t[0]
    state_by_t = {round(t, 6): s for t, s, _ in path_track}
    hits = []
    for ft in frame_t:
        t_rel = ft - t0
        if not (2.6 <= t_rel <= _T_IMPACT):
            continue
        ego = ego_by_t.get(ft)
        target = next((v for v in veh_by_t.get(ft, []) if v.id == _TARGET_VID), None)
        state = state_by_t.get(round(t_rel, 6))
        if ego is None or target is None or state is None:
            continue
        yaw = ego.rotationX * 2.0 * math.pi
        offset = (CAL.arc_start_pctg - 0.5) * 2.0 * CAL.ego_half_length
        fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
        cap_fwd, cap_back = capsule_extents(
            CAL.ego_half_length, CAL.ego_half_length, offset,
        )
        ego_arc = build_arc(
            ego.coordinateX + offset * fwd_x, ego.coordinateZ + offset * fwd_z,
            yaw, ego.speed, state.kappa_path, CAL.ego_half_width, 3.0,
            fwd_len=cap_fwd, back_len=cap_back,
            parallel_margin_scale=CAL.capsule_parallel_margin_scale,
        )
        target_arc = target.get_arc(
            3.0, half_width=max(target.size.width / 2.0 - 0.1, 0.3),
            arc_start_pctg=CAL.arc_start_pctg, body_capsule=True,
        )
        hit = arc_arc_collision(
            ego_arc, target_arc, CAL.corridor_margin, CAL.collision_samples,
        )
        hits.append((t_rel, hit is not None))
    assert hits
    misses = [t for t, ok in hits if not ok]
    assert not misses, f"ego arc lost the target at {misses}"

    nearest = nearest_frame_t(frame_t, t0 + 4.0)
    assert nearest is not None
