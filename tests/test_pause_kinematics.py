"""Regression: pause gaps must not collapse or stall vehicle speeds."""

from __future__ import annotations

import math

import pytest

from core.aeb.clip_replay import decode_radar_stream
from core.aeb.clip_store import ClipStore, default_clip_root
from core.radar.reader import TrafficReader
from core.radar.traffic import (
    Position,
    Quaternion,
    Size,
    Vehicle,
    _READER_CLOCK_GAP_S,
)


def _vehicle(x: float, z: float, speed: float, vid: int = 1) -> Vehicle:
    return Vehicle(
        Position(x, 0.0, z),
        Quaternion(1.0, 0.0, 0.0, 0.0),
        Size(2.5, 3.0, 6.0),
        speed,
        0.0,
        0,
        [],
        vid,
        False,
        False,
    )


def test_reader_clock_gap_holds_speed_and_resets_history():
    """A pause-sized gap between reader clocks must not integrate Δpos/huge_τ."""
    prev = _vehicle(0.0, 0.0, 30.0)
    prev.time = 100.0
    prev._raw_x = 0.0
    prev._raw_z = 0.0
    prev._smooth_x = 0.0
    prev._smooth_z = 0.0
    prev._smooth_yaw = 0.0
    prev._speed_ema = 30.0
    prev._smooth_speed = 30.0
    prev._smooth_accel = 0.0
    prev.acc_speed = 30.0
    prev._raw_speed = 30.0
    prev._position_history = [(100.0 - i * 0.05, 0.0, i * 1.5) for i in range(10, -1, -1)]
    prev._speed_ema_history = [(100.0 - i * 0.05, 30.0) for i in range(10, -1, -1)]

    cur = _vehicle(0.0, -2.0, 30.0)
    assert _READER_CLOCK_GAP_S < 1.4
    cur._hold_across_clock_discontinuity(prev, 101.4)

    assert cur.speed == pytest.approx(30.0, abs=1e-6)
    assert cur.acc_speed == pytest.approx(30.0, abs=1e-6)
    assert cur._raw_speed == pytest.approx(30.0, abs=1e-6)
    assert cur.time == pytest.approx(101.4, abs=1e-9)
    assert len(cur._position_history) == 2
    # Seeded history should LS-fit back to the held speed.
    from core.radar.traffic import _raw_speed_from_position_history
    yaw = cur._smooth_yaw or 0.0
    fitted = _raw_speed_from_position_history(
        cur._position_history, -math.sin(yaw), -math.cos(yaw),
    )
    assert fitted == pytest.approx(30.0, abs=0.5)


def test_slow_full_update_dt_does_not_false_reanchor():
    """dt since last full update can exceed 0.2s without a reader gap; still track."""
    reader = TrafficReader()
    # Seed one vehicle via two close frames, then a 0.25s step (old false trigger).
    v0 = _vehicle(0.0, 0.0, 20.0, vid=7)
    v0.time = 10.0
    reader._last_vehicles = {7: v0}
    reader._last_kin_t = 10.0

    # Move ~5 m in 0.25 s → ~20 m/s if integrated normally.
    v1 = _vehicle(0.0, -5.0, 20.0, vid=7)
    out, _ = reader._smooth_and_build([v1], 10.25, 0.0, 0.0, 0.0, 15.0)
    assert len(out) == 1
    # Must NOT hold at seed and must not collapse: a real 0.25s step is valid motion.
    assert abs(out[0].speed) > 10.0


def _unpause_indices(frames):
    """Return (last_pre_i, first_post_i) for the first pause→unpause with history."""
    last_active_i = None
    in_pause = False
    for i, f in enumerate(frames):
        if f.ego.paused or f.traffic_buf is None:
            if last_active_i is not None:
                in_pause = True
            continue
        if in_pause and last_active_i is not None:
            return last_active_i, i
        last_active_i = i
        in_pause = False
    return None, None


def _assert_unpause_speeds(clip_glob: str, min_fast: int = 3) -> None:
    root = default_clip_root()
    matches = sorted(root.glob(clip_glob))
    if not matches:
        pytest.skip(f"clip {clip_glob} not found under {root}")

    clip = ClipStore(root=root).load(matches[0])
    assert clip is not None
    frames = sorted(clip.radar_frames, key=lambda f: f.t_mono)
    last_pre_i, first_post_i = _unpause_indices(frames)
    assert last_pre_i is not None and first_post_i is not None
    gap = frames[first_post_i].t_wall - frames[last_pre_i].t_wall
    assert gap > _READER_CLOCK_GAP_S

    veh_by_t, _, _ = decode_radar_stream(clip)
    pre = {v.id: v for v in veh_by_t[frames[last_pre_i].t_mono]}
    post = {v.id: v for v in veh_by_t[frames[first_post_i].t_mono]}
    # A few frames after unpause: speeds must still track (not stuck ~25%).
    post_later_i = min(first_post_i + 15, len(frames) - 1)
    while post_later_i > first_post_i and (
        frames[post_later_i].ego.paused or frames[post_later_i].traffic_buf is None
    ):
        post_later_i -= 1
    later = {v.id: v for v in veh_by_t[frames[post_later_i].t_mono]}

    shared = sorted(set(pre) & set(post) & set(later), key=lambda vid: -abs(pre[vid].speed))
    assert shared, "expected continuing vehicles across the pause"

    checked = 0
    for vid in shared:
        before = pre[vid]
        after = post[vid]
        after_later = later[vid]
        if abs(before.speed) < 10.0:
            continue
        raw_before = before._raw_speed if before._raw_speed is not None else before.speed
        raw_after = after._raw_speed if after._raw_speed is not None else after.speed
        assert abs(after.speed) > 0.5 * abs(before.speed), (
            f"id={vid} filtered speed collapsed across pause: "
            f"{before.speed:.2f} → {after.speed:.2f} (gap={gap:.2f}s)"
        )
        assert abs(raw_after) > 0.5 * abs(raw_before), (
            f"id={vid} raw speed collapsed across pause: "
            f"{raw_before:.2f} → {raw_after:.2f} (gap={gap:.2f}s)"
        )
        # Stall regression (b53e42df): must not sit near ~25% of pre-pause speed.
        assert abs(after_later.speed) > 0.5 * abs(before.speed), (
            f"id={vid} speed stalled after unpause: "
            f"{before.speed:.2f} → {after_later.speed:.2f}"
        )
        checked += 1
    assert checked >= min_fast, "expected several fast vehicles to validate"


@pytest.mark.needs_clips
def test_clip_f5b4e904_unpause_preserves_raw_speed():
    _assert_unpause_speeds("*f5b4e904*")


@pytest.mark.needs_clips
def test_clip_b53e42df_unpause_does_not_stall_speeds():
    _assert_unpause_speeds("*b53e42df*")


@pytest.mark.needs_clips
def test_clip_53a580e7_unpause_tracks_buffer_speed():
    """Post-unpause filtered speed must stay near the traffic buffer speed."""
    root = default_clip_root()
    matches = sorted(root.glob("*53a580e7*"))
    if not matches:
        pytest.skip(f"clip 53a580e7 not found under {root}")

    import struct
    from core.radar.reader import _TOTAL_FORMAT

    clip = ClipStore(root=root).load(matches[0])
    assert clip is not None
    frames = sorted(clip.radar_frames, key=lambda f: f.t_mono)
    last_pre_i, first_post_i = _unpause_indices(frames)
    assert last_pre_i is not None and first_post_i is not None

    veh_by_t, _, _ = decode_radar_stream(clip)
    # Check ~0.5 s after unpause (not the hold frame itself).
    check_i = min(first_post_i + 12, len(frames) - 1)
    while check_i > first_post_i and (
        frames[check_i].ego.paused or frames[check_i].traffic_buf is None
    ):
        check_i -= 1
    f = frames[check_i]
    raw = struct.unpack(_TOTAL_FORMAT, f.traffic_buf)
    buf_by_id: dict[int, float] = {}
    data = raw
    for _ in range(40):
        if not (data[0] == 0 and data[2] == 0):
            buf_by_id[int(data[13])] = float(data[10])
        data = data[46:]

    checked = 0
    for v in veh_by_t[f.t_mono]:
        buf = buf_by_id.get(v.id)
        if buf is None or abs(buf) < 10.0:
            continue
        ratio = abs(v.speed) / abs(buf)
        assert 0.7 <= ratio <= 1.35, (
            f"id={v.id} filtered {v.speed:.2f} vs buffer {buf:.2f} "
            f"(ratio={ratio:.2f}) at frame {check_i}"
        )
        checked += 1
    assert checked >= 3
