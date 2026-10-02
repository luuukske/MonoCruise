"""tools/clip_export: jitter removal, threat choice, trailer follower, framing, painting."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.aeb.clip_replay import ReviewFrame
from core.aeb.clip_schema import ConsumedContext, LiveAEB
from core.aeb.thread import AEBSnapshot
from core.radar.traffic import build_arc
from tools.clip_export import camera
from tools.clip_export.export import ExportOptions, open_loop_warning, output_times
from tools.clip_export.timeline import (
    Timeline, auto_window, local_linear, pick_primary_threat, threat_ticks, with_rerun_decisions,
)

TICK_S = 1.0 / 30.0


def _frame(t, *, ego=(0.0, 0.0, 0.0), speed=20.0, vehicles=(), colliding=(),
           warn=False, brake=False, arcs=None) -> ReviewFrame:
    snap = AEBSnapshot(
        ego_x=ego[0], ego_z=ego[1], ego_yaw=ego[2], ego_speed=speed,
        vehicles=[dict(v) for v in vehicles], colliding_ids=set(colliding),
        vehicle_arcs=dict(arcs or {}),
    )
    live = LiveAEB(aeb_warn=warn, aeb_brake=brake, target_decel_ms2=6.0 if brake else 0.0,
                   effective_max_decel_ms2=8.0)
    return ReviewFrame(t, t, snap, live, ConsumedContext())


def _car(vid, x, z, yaw, *, length=4.5):
    return {"vid": vid, "x": x, "z": z, "yaw": yaw, "half_w": 0.9, "length": length,
            "trailers": []}


def _crossing_clip(duration_s=6.0, *, cross_speed=15.0, meet_t=4.0):
    """Ego north at 20 m/s; a car crosses and meets ego's path at ``meet_t``."""
    frames = []
    n = int(duration_s / TICK_S)
    for i in range(n + 1):
        t = i * TICK_S
        car = _car(7, cross_speed * (t - meet_t), -20.0 * meet_t, -math.pi / 2.0)
        tracked = meet_t - 2.0 <= t <= meet_t + 0.5
        frames.append(_frame(
            t, ego=(0.0, -20.0 * t, 0.0), vehicles=[car, _car(9, -30.0, -20.0 * t, 0.0)],
            colliding=[7] if tracked else [],
            warn=meet_t - 1.0 <= t <= meet_t + 0.5, brake=meet_t - 0.7 <= t <= meet_t + 0.5,
        ))
    return frames


def test_local_linear_removes_the_60hz_tick_alternation():
    # 30 Hz timestamps over positions that advance 3 ticks then 1 tick, as recorded.
    v = 25.0
    ts, xs, ticks = [], [], 0
    for i in range(120):
        ts.append(i * TICK_S)
        xs.append(v * ticks / 60.0)
        ticks += 3 if i % 2 == 0 else 1
    grid = [1.0 + k / 60.0 for k in range(120)]
    smooth = [local_linear(ts, xs, t, 0.06) for t in grid]
    nearest = [xs[min(range(len(ts)), key=lambda j: abs(ts[j] - t))] for t in grid]

    def worst_accel(seq):
        return max(abs(seq[i + 1] - 2 * seq[i] + seq[i - 1]) for i in range(1, len(seq) - 1))

    assert worst_accel(nearest) > 0.5
    assert worst_accel(smooth) < 0.02


def test_local_linear_is_unbiased_at_track_ends():
    ts = [i * TICK_S for i in range(10)]
    vs = [3.0 + 2.0 * t for t in ts]
    for t in (ts[0], ts[-1], ts[4] + 0.01):
        assert local_linear(ts, vs, t, 0.06) == pytest.approx(3.0 + 2.0 * t, abs=1e-9)


def test_primary_threat_prefers_label_then_what_aeb_acted_on():
    frames = [
        _frame(0.0, vehicles=[_car(1, 0, -10, 0), _car(2, 5, -10, 0), _car(3, 9, -9, 0)],
               colliding=[1]),
        _frame(0.1, vehicles=[_car(1, 0, -10, 0), _car(2, 5, -10, 0)], colliding=[1]),
        _frame(0.2, vehicles=[_car(1, 0, -10, 0), _car(2, 5, -10, 0)], colliding=[2], brake=True),
    ]
    assert pick_primary_threat(frames, label_vid=3) == 3
    assert pick_primary_threat(frames, label_vid=99) == 2
    assert pick_primary_threat(frames, label_vid=None) == 2
    assert pick_primary_threat(frames[:2], label_vid=None) == 1


def test_auto_window_brackets_the_event():
    frames = _crossing_clip()
    lo, hi = auto_window(frames)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(6.0, abs=TICK_S)
    assert auto_window(frames, lead_s=1.0)[0] == pytest.approx(2.0, abs=TICK_S)


def test_trailer_trails_through_a_turn_and_straightens_after():
    speed, radius = 12.0, 25.0
    frames, x, z, yaw = [], 0.0, 0.0, 0.0
    for i in range(int(12.0 / TICK_S)):
        t = i * TICK_S
        frames.append(_frame(t, ego=(x, z, yaw), speed=speed))
        turning = 3.0 <= t < 3.0 + (math.pi / 2.0) * radius / speed
        yaw += (speed / radius) * TICK_S if turning else 0.0
        x += -math.sin(yaw) * speed * TICK_S
        z += -math.cos(yaw) * speed * TICK_S
    tl = Timeline(frames, ego_has_trailer=True)

    mid_turn = 3.0 + 0.5 * (math.pi / 2.0) * radius / speed
    ego, trailer = tl.ego_body(mid_turn), tl.ego_trailer_body(mid_turn)
    assert abs(math.degrees(ego.yaw - trailer.yaw)) > 10.0

    end = tl.t_last - 0.1
    ego, trailer = tl.ego_body(end), tl.ego_trailer_body(end)
    assert abs(math.degrees(ego.yaw - trailer.yaw)) < 2.0
    back = (trailer.x - ego.x) * math.sin(ego.yaw) + (trailer.z - ego.z) * math.cos(ego.yaw)
    assert back > ego.length / 2.0


def test_camera_keeps_ego_and_threat_in_frame_and_zooms_smoothly():
    tl = Timeline(_crossing_clip(), ego_has_trailer=True)
    times = [k / 60.0 for k in range(int(tl.t_last * 60.0))]
    states = tl.states(times)
    width, height = 1920, 1080
    cfg = camera.FramingConfig()
    cams = camera.solve(states, width, height, tl.primary_span, cfg)
    ppm_lo, ppm_hi = height / cfg.max_span_m, height / cfg.min_span_m
    entry = camera.entry_time(states, width, height, tl.primary_span, cfg)
    ramp = float("-inf") if entry <= times[0] else entry

    for st, cam in zip(states, cams):
        assert ppm_lo - 1e-9 <= cam.ppm <= ppm_hi + 1e-9
        if camera.threat_weight(st.t, tl.primary_span, cfg, ramp) < 1.0:
            continue
        pts = st.ego.corners() + st.ego_trailer.corners() + st.vehicle(7).body.corners()
        for x, z in pts:
            u, v = camera.to_view(x, z, cam.x, cam.z, cam.yaw)
            assert 0.0 <= width / 2.0 + u * cam.ppm <= width
            assert 0.0 <= height / 2.0 + v * cam.ppm <= height

    log_ppm = [math.log(c.ppm) for c in cams]
    kinks = [abs(log_ppm[i + 1] - 2 * log_ppm[i] + log_ppm[i - 1]) for i in range(1, len(cams) - 1)]
    assert max(kinks) < 0.005


def test_export_starts_when_the_threat_fits_and_never_zooms_out():
    tl = Timeline(_crossing_clip(7.0, cross_speed=60.0, meet_t=5.0), ego_has_trailer=True)
    opts = ExportOptions()
    times = output_times(tl, opts)
    ticks = tl.states(tl.tick_ts, corridors=False)
    entry = camera.entry_time(ticks, opts.width, opts.height, tl.primary_span)
    assert auto_window(tl.frames)[0] < entry == pytest.approx(times[0])

    cams = camera.solve(tl.states(times), opts.width, opts.height, tl.primary_span)
    log_ppm = [math.log(c.ppm) for c in cams]
    zoom_out = sum(max(a - b, 0.0) for a, b in zip(log_ppm, log_ppm[1:]))
    assert zoom_out < 0.01
    assert log_ppm[-1] - log_ppm[0] > 0.5


def test_ego_always_points_straight_up_in_both_aspects():
    tl = Timeline(_crossing_clip(), ego_has_trailer=True)
    states = tl.states([k / 30.0 for k in range(int(tl.t_last * 30.0))])
    for width, height in ((1920, 1080), (1080, 1920)):
        for st, cam in zip(states, camera.solve(states, width, height, tl.primary_span)):
            ahead = (st.ego.x - math.sin(st.ego.yaw) * 10.0, st.ego.z - math.cos(st.ego.yaw) * 10.0)
            u0, v0 = camera.to_view(st.ego.x, st.ego.z, cam.x, cam.z, cam.yaw)
            u1, v1 = camera.to_view(*ahead, cam.x, cam.z, cam.yaw)
            assert ((u1 - u0) / 10.0, (v1 - v0) / 10.0) == pytest.approx((0.0, -1.0), abs=1e-9)


def test_threat_highlight_bridges_dropouts_but_never_outlasts_the_recording():
    pattern = [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
    frames = [_frame(i * TICK_S, colliding=[5] if on else []) for i, on in enumerate(pattern)]
    lit = [5 in ids for ids in threat_ticks(frames)]
    assert lit == [False, True, True, True, True, True, True, True, False, False, False, True, False]


def test_every_moving_vehicle_gets_a_path_and_parked_ones_do_not():
    moving = build_arc(-30.0, -20.0, 0.0, 15.0, 0.0, 0.9, 3.0)
    parked = build_arc(10.0, -40.0, 0.0, 0.0, 0.0, 0.9, 3.0)
    frames = [_frame(i * TICK_S, vehicles=[_car(9, -30.0, -20.0, 0.0), _car(4, 10.0, -40.0, 0.0)],
                     arcs={9: [moving], 4: [parked]}) for i in range(5)]
    st = Timeline(frames, ego_has_trailer=False).states([2 * TICK_S])[0]
    assert not any(v.threat for v in st.vehicles)
    assert len(st.corridors[9][0][0]) > 2
    assert 4 not in st.corridors


def test_rerun_decisions_replace_the_recording_tick_for_tick():
    frames = _crossing_clip()[:3]
    tick = SimpleNamespace(aeb_warn=True, aeb_brake=True, engaged=True, target_decel_ms2=5.0,
                           required_decel_ms2=6.0, time_to_brake=0.5, time_to_collision=1.2,
                           colliding_ids={9})
    ticks = [SimpleNamespace(**vars(tick), t_mono=frames[0].t_mono),
             SimpleNamespace(**vars(tick), t_mono=frames[2].t_mono)]
    out = with_rerun_decisions(frames, ticks)
    assert [f.t_mono for f in out] == [frames[0].t_mono, frames[2].t_mono]
    assert all(f.live_aeb.aeb_brake and f.snapshot.colliding_ids == {9} for f in out)
    assert not frames[0].live_aeb.aeb_brake


def test_open_loop_warning_only_when_the_recording_braked_first():
    def tl(decisions, rec, shown):
        return SimpleNamespace(decisions=decisions, recorded_brake_t=rec, shown_brake_t=shown)

    assert "10.50" in open_loop_warning(tl("recomputed", 8.0, 10.5))
    assert "never" in open_loop_warning(tl("recomputed", 8.0, None))
    assert open_loop_warning(tl("recomputed", 8.0, 8.05)) is None
    assert open_loop_warning(tl("recomputed", None, 9.0)) is None
    assert open_loop_warning(tl("recorded", 8.0, 10.5)) is None


def test_state_label_cuts_on_the_nearest_tick():
    tl = Timeline(_crossing_clip(), ego_has_trailer=False)
    before, after = tl.states([2.98, 3.02])
    assert (before.state, after.state) == (0, 1)
    assert tl.states([3.32])[0].state == 2


def test_painter_draws_ego_where_the_camera_puts_it():
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication

    from tools.clip_export.painter import BG, ShowcaseRenderer, world_transform

    _app = QApplication.instance() or QApplication([])
    tl = Timeline(_crossing_clip(), ego_has_trailer=True)
    times = [3.5 + k / 60.0 for k in range(3)]
    states = tl.states(times)
    cams = camera.solve(states, 640, 360, tl.primary_span)
    renderer = ShowcaseRenderer(640, 360, origin=tl.origin, credit="credit line")
    image = renderer.render(states[1], cams[1])
    assert (image.width(), image.height()) == (640, 360)
    sx, sy = world_transform(cams[1], 640, 360).map(states[1].ego.x, states[1].ego.z)
    assert QColor(image.pixel(round(sx), round(sy))) != BG
    assert _app is not None


def test_cli_refuses_a_contributed_clip_without_the_flag(monkeypatch, capsys):
    import tools.aeb_agent.corpus as corpus
    from tools.clip_export import cli

    row = SimpleNamespace(origin="remote", short="abcd1234")
    monkeypatch.setattr(corpus, "build_index",
                        lambda *a, **k: SimpleNamespace(resolve=lambda ident: row))

    def _no_load(_row):
        raise AssertionError("a refused clip must not be decoded")

    monkeypatch.setattr(corpus, "load_clip", _no_load)
    assert cli.main(["abcd1234"]) == 2
    assert "--contributed-ok" in capsys.readouterr().err
