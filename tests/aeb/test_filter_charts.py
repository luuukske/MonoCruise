"""Tests for the filter-tuning charts: trace fidelity, lane wiring, review sync."""

from __future__ import annotations

import struct

import pytest

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from core.aeb.clip_replay import decode_radar_stream, replay_clip
from core.aeb.clip_schema import (
    AEBTickRecord, ConsumedContext, EgoTelemetry, LiveAEB, RadarFrameRecord,
)
from core.aeb.clip_store import ClipStore
from core.radar.elevation import BODY_DATUM_FRAC
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT
from core.radar.traffic import Position, Quaternion, Size, Vehicle
from tools import aeb_filter_charts as charts
from tools.aeb_filter_trace import _RESIDUAL, _lag_gates, build_trace
from tools.aeb_review import ReviewWindow
from tools.aeb_review_widgets import ClipLoader, Loaded, action_index, recorded_band

from tests.aeb.test_clip_capture import _make_clip

_BODY_H = 3.0
_BODY_Y = BODY_DATUM_FRAC * _BODY_H
# Radar frames closer together than _LOCATION_UPDATE_FREQUENCY are sub-frames and
# never run the chain, so the trace tests have to step past it.
_FULL_FRAME_DT = 0.06


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _traffic_buf(px: float, pz: float, vid: int, speed: float) -> bytes:
    flat: list = [px, _BODY_Y, pz, 1.0, 0.0, 0.0, 0.0, 2.5, _BODY_H, 6.0, speed, 0.0]
    flat += [0, vid, 0, 0] + [0.0] * 30
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def _braking_clip(frames: int = 44):
    """A lead that cruises then brakes hard, sampled fast enough to run the chain."""
    clip = _make_clip(clip_id="filter01")
    clip.radar_frames = []
    clip.aeb_ticks = []
    pos_z = 60.0
    speed = 22.0
    for i in range(frames):
        t = i * _FULL_FRAME_DT
        if i > frames // 2:
            speed = max(0.0, speed - 5.0 * _FULL_FRAME_DT)
        pos_z -= speed * _FULL_FRAME_DT
        clip.radar_frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=0.0, rotationX=0.5, speed=25.0),
            traffic_buf=_traffic_buf(1.5, pos_z, vid=7, speed=speed),
            parked_buf=None,
        ))
        clip.aeb_ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0),
            live_aeb=LiveAEB(
                aeb_warn=(i >= frames - 6), aeb_brake=(i >= frames - 3),
                engaged=(i >= frames - 3), colliding_ids=[7] if i >= frames - 6 else [],
                time_to_collision=1.0 if i >= frames - 6 else 1e9,
            ),
        ))
    return clip


def _trace_of(clip):
    return build_trace(clip, stream=decode_radar_stream(clip))


def test_step4_rebuild_reproduces_the_recorded_acc_speed():
    """The gate lane is only honest while this holds; a drift from traffic.py breaks it.

    ``_step4_internals`` restates the step 4 arithmetic to expose tau and the gates,
    which the filter itself does not return. The residual is the proof that the
    restatement still matches, so it is the whole reason the lane can be trusted.
    """
    trace = _trace_of(_braking_clip())
    residuals = [v for veh in trace.vehicles.values()
                 for v in veh.series[_RESIDUAL] if v == v]
    assert residuals, "no frame ran the chain, so the rebuild was never exercised"
    assert max(abs(v) for v in residuals) < 1e-9


def test_chain_ran_is_recorded_and_subframes_are_not_called_bypasses():
    trace = _trace_of(_braking_clip())
    veh = trace.vehicles[7]
    ran = [i for i, v in enumerate(veh.series["st_bypassed"]) if v < 0.5]
    assert len(ran) > 10
    # A sub-frame is a copy, not an anomaly: the two must never be set together.
    for sub, byp in zip(veh.series["st_subframe"], veh.series["st_bypassed"]):
        assert not (sub >= 0.5 and byp >= 0.5)


def test_step4_internals_are_held_across_subframes_not_left_blank():
    """Sub-frames carry the filter state forward, so the gates must too."""
    clip = _braking_clip()
    for i, frame in enumerate(clip.radar_frames):
        if i % 3 == 1:
            frame.t_wall = clip.radar_frames[i - 1].t_wall + 0.01
    trace = _trace_of(clip)
    veh = trace.vehicles[7]
    subframes = [i for i, v in enumerate(veh.series["st_subframe"]) if v >= 0.5]
    assert subframes, "the clip was meant to contain sub-frames"
    settled = [i for i in subframes if i > 8]
    assert any(veh.series["tau"][i] == veh.series["tau"][i] for i in settled)


def test_lag_elapsed_is_measured_on_the_vehicle_clock():
    """``_lag_since`` is stamped from the reader's wall clock, not the frame key.

    Subtracting the t_mono frame key from it produced an elapsed time of -1.8e9 s,
    which silently emptied the freeze trace instead of failing.
    """
    def _vehicle(t: float) -> Vehicle:
        v = Vehicle(Position(0.0, 0.0, 20.0), Quaternion(1.0, 0.0, 0.0, 0.0),
                    Size(2.5, 3.0, 6.0), 10.0, 0.0, 0, [], 7, True, False)
        v.time = t
        v._raw_x, v._raw_z = 0.0, 20.0
        return v

    prev, cur = _vehicle(1_700_000_000.0), _vehicle(1_700_000_000.4)
    cur._lag_since = 1_700_000_000.1
    ego_speed = 25.0
    out = _lag_gates(prev, cur, 0.4, gap_3d=20.0, ego_speed=ego_speed)
    assert out["lag_elapsed"] == pytest.approx(0.3, abs=1e-6)
    assert 0.0 < out["lag_freeze_dur"] <= 0.5


def test_every_lane_signal_resolves_against_a_real_trace(qapp):
    """A mistyped key draws nothing and says nothing; this is what notices."""
    trace = _trace_of(_braking_clip())
    chart = charts.FilterChart()
    chart.set_trace(trace, trace.vehicles[7])
    derived = {"ego_speed", "lag_frac"}
    for build in charts._LANES:
        for curve in build().signals:
            assert curve.key in derived or curve.key in trace.vehicles[7].series, curve.key
            assert chart._series(curve.key) is not None, curve.key
    for key, _name, _color in charts._STATE_ROWS:
        assert key in trace.vehicles[7].series, key


def test_thin_collapses_subframe_repeats_but_keeps_a_real_hold_flat():
    """52% of samples are carried copies; drawing them staircases the radar rate."""
    # Two frames apart is a sub-frame copy: it collapses to the update that made it.
    times = [0.00, 0.03, 0.06, 0.09]
    values = [10.0, 10.0, 11.0, 11.0]
    assert [(t, v) for t, v, _c in charts._thin(times, values)] == [(0.0, 10.0), (0.06, 11.0)]

    # A hold longer than _HOLD_MIN_S keeps both ends, so it draws flat then steep.
    times = [0.0, 0.1, 0.2, 0.3, 0.34]
    values = [5.0, 5.0, 5.0, 5.0, 9.0]
    thinned = [(t, v) for t, v, _c in charts._thin(times, values)]
    assert thinned == [(0.0, 5.0), (0.3, 5.0), (0.34, 9.0)]


def test_thin_drops_nans_and_marks_a_real_hole_as_a_break():
    times = [0.0, 0.05, 2.0]
    values = [1.0, float("nan"), 2.0]
    out = charts._thin(times, values)
    assert [(t, v) for t, v, _c in out] == [(0.0, 1.0), (2.0, 2.0)]
    assert [c for _t, _v, c in out] == [True, False]


def test_chart_window_follows_the_review_cursor_and_forwards_keys(qapp, tmp_path):
    store = ClipStore(root=tmp_path)
    clip = _braking_clip()
    path = store.write(clip)
    stream = decode_radar_stream(clip)
    frames = replay_clip(clip, stream=stream)

    win = ReviewWindow(store)
    loaded = Loaded(clip=clip, frames=frames, proposal=recorded_band(frames),
                    action_idx=action_index(frames), trace=build_trace(clip, stream=stream))
    win._path = str(path)
    win._cache[str(path)] = loaded
    win._show(str(path), loaded)
    assert win._charts is None

    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.NoModifier))
    assert win._charts is not None and win._charts.isVisible()
    assert win._charts._vid == 7

    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier))
    assert win._charts._chart._cursor == pytest.approx(win._cur_t())

    win._charts.seeked.emit(frames[-1].t_rel)
    assert win._cur_t() == pytest.approx(frames[-1].t_rel, abs=0.05)

    # A binding pressed over the chart window still reaches the label form.
    win._charts.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_4, Qt.NoModifier))
    assert win._class.currentText() == "fn"

    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.NoModifier))
    assert not win._charts.isVisible()
    win.close()


def test_the_replayed_band_is_a_separate_lazy_job(qapp, tmp_path):
    """0.8 s a clip, so it must never ride along with the load a tagging pass pays for."""
    store = ClipStore(root=tmp_path)
    clip = _braking_clip()
    path = store.write(clip)
    loader = ClipLoader(store)

    loads, evals = [], []
    loader.loaded.connect(lambda *a: loads.append(a))
    loader.evaluated.connect(lambda *a: evals.append(a))

    loader.load(str(path))
    assert loads and not evals, "loading a clip must not trigger the re-run"

    loader.evaluate(str(path), clip)
    assert len(evals) == 1
    _p, track = evals[0]
    assert track is not None and len(track) == len(clip.aeb_ticks)
    assert all(state in (0, 1, 2) for _t, state in track)


def test_the_chart_asks_for_the_replay_only_while_it_is_open(qapp, tmp_path):
    store = ClipStore(root=tmp_path)
    clip = _braking_clip()
    path = store.write(clip)
    stream = decode_radar_stream(clip)
    frames = replay_clip(clip, stream=stream)

    win = ReviewWindow(store)
    asked = []
    win.eval_requested.connect(lambda p, c: asked.append(p))
    loaded = Loaded(clip=clip, frames=frames, proposal=recorded_band(frames),
                    action_idx=action_index(frames), trace=build_trace(clip, stream=stream))
    win._path = str(path)
    win._cache[str(path)] = loaded
    win._show(str(path), loaded)
    assert not asked, "a closed chart must not cost the re-run"

    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.NoModifier))
    assert asked == [str(path)]
    assert win._charts._chart._replayed is None, "pending until the job answers"

    win._on_evaluated(str(path), [(0.0, 0), (0.5, 2)])
    assert win._charts._chart._replayed == [(0.0, 0), (0.5, 2)]
    assert loaded.evaluated == [(0.0, 0), (0.5, 2)]

    # A band that lands after the user moved on belongs to the clip that asked.
    win._on_evaluated("some/other/clip", [(0.0, 1)])
    assert win._charts._chart._replayed == [(0.0, 0), (0.5, 2)]
    win.close()


def test_the_loader_emits_a_trace_beside_the_frames(qapp, tmp_path):
    store = ClipStore(root=tmp_path)
    path = store.write(_braking_clip())
    loader = ClipLoader(store)
    received = []
    loader.loaded.connect(lambda *args: received.append(args))

    loader.load(str(path))
    assert len(received) == 1
    _path, clip, frames, trace = received[0]
    assert clip is not None and frames
    assert trace is not None and 7 in trace.vehicles


def test_a_trace_failure_never_costs_the_clip(qapp, tmp_path, monkeypatch):
    """The charts are a convenience; a clip must still open when the trace blows up."""
    import tools.aeb_filter_trace as trace_mod

    store = ClipStore(root=tmp_path)
    path = store.write(_braking_clip())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("trace exploded")

    monkeypatch.setattr(trace_mod, "build_trace", _boom)
    loader = ClipLoader(store)
    received = []
    loader.loaded.connect(lambda *args: received.append(args))

    loader.load(str(path))
    assert len(received) == 1
    _path, clip, frames, trace = received[0]
    assert clip is not None
    assert frames, "frames should survive a trace failure"
    assert trace is None
