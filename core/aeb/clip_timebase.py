"""Physics-step timebase for clip replay: legacy ego re-pairing and the simulation clock.

See core/aeb/README.md section 16. Replay reads frames from here and never mutates the clip.
"""

from __future__ import annotations

import math
import statistics
import struct
from dataclasses import replace

from core.aeb.clip_schema import Clip, RadarFrameRecord
from core.radar.reader import TrafficReader, _TOTAL_FORMAT, _TOTAL_PARKED_FORMAT
from core.radar.traffic import Vehicle, _READER_CLOCK_GAP_S

PHYSICS_STEP_HZ: float = 60.0
# Clips up to this schema paired traffic with a telemetry-thread pose up to 3 steps old.
LEGACY_PAIRING_SCHEMA_MAX: int = 4

_STEP_TOLERANCE: float = 0.25
_AGREE_TOLERANCE: float = 0.3
_AGREE_SHARE: float = 0.6
_MIN_EGO_SPEED_MS: float = 1.0
_MIN_AI_SPEED_MS: float = 2.0
_MIN_TMP_SPEED_MS: float = 5.0
_MIN_TMP_VEHICLES: int = 2
_TMP_BASELINE_FRAMES: int = 6
_PHASE_WINDOW_FRAMES: int = 45
_PHASE_MAX_STEPS: int = 3
_CLOCK_MIN_COUNTED_SHARE: float = 0.5
_CLOCK_SPAN_TOLERANCE: float = 0.05
_MIN_SEGMENT_FRAMES: int = 3

_Steps = list[int | None]
_Snapshot = dict[int, Vehicle]


def decode_buffers(traffic_buf: bytes, parked_buf: bytes | None) -> list[Vehicle] | None:
    """Traffic plus parked decode, in the id-precedence order ``replay_frame`` uses."""
    try:
        vehicles = TrafficReader._build_vehicles_from_raw(
            struct.unpack(_TOTAL_FORMAT, traffic_buf)
        )
    except Exception:
        return None
    if parked_buf is not None:
        try:
            vehicles.extend(TrafficReader._build_parked_from_raw(
                struct.unpack(_TOTAL_PARKED_FORMAT, parked_buf),
                {int(v.id) for v in vehicles},
            ))
        except Exception:
            pass
    return vehicles


def replay_frames(clip: Clip, as_recorded: bool = False) -> list[RadarFrameRecord]:
    """Copies of the clip's radar frames in ``t_mono`` order, on the physics-step timebase.

    ``as_recorded`` skips both corrections, for probes that need the capture-time input.
    """
    frames = [
        replace(f, ego=replace(f.ego))
        for f in sorted(clip.radar_frames, key=lambda f: f.t_mono)
    ]
    if as_recorded:
        return frames
    repair = clip.metadata.schema_version <= LEGACY_PAIRING_SCHEMA_MAX
    for segment in _segments(frames):
        _apply_timebase(segment, repair)
    return frames


def _segments(frames: list[RadarFrameRecord]) -> list[list[RadarFrameRecord]]:
    """Runs of live frames the reader would integrate without a clock re-anchor."""
    segments: list[list[RadarFrameRecord]] = []
    current: list[RadarFrameRecord] = []
    for f in frames:
        live = f.traffic_buf is not None and not f.ego.paused
        if live and current:
            gap = f.t_wall - current[-1].t_wall
            if gap <= 0.0 or gap > _READER_CLOCK_GAP_S:
                segments.append(current)
                current = []
        if live:
            current.append(f)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _apply_timebase(frames: list[RadarFrameRecord], repair: bool) -> None:
    if len(frames) < _MIN_SEGMENT_FRAMES:
        return
    decoded = [decode_buffers(f.traffic_buf, f.parked_buf) for f in frames]
    if any(d is None for d in decoded):
        return
    snapshots = [{int(v.id): v for v in d} for d in decoded]
    ego_steps: _Steps = [None] + [_ego_steps(a, b) for a, b in zip(frames, frames[1:])]
    traffic_steps = _traffic_steps(frames, snapshots, ego_steps)
    if repair:
        _repair_ego_pairing(frames, traffic_steps, ego_steps)
    _rebuild_clock(frames, traffic_steps, ego_steps)


def _whole_steps(ratio: float) -> int | None:
    steps = round(ratio)
    return steps if abs(ratio - steps) <= _STEP_TOLERANCE else None


def _ego_steps(a: RadarFrameRecord, b: RadarFrameRecord) -> int | None:
    """Ego advances ``speed / 60`` per step, with speed read at the end of the step."""
    speed = abs(b.ego.speed)
    if speed < _MIN_EGO_SPEED_MS:
        return None
    moved = math.hypot(b.ego.coordinateX - a.ego.coordinateX, b.ego.coordinateZ - a.ego.coordinateZ)
    return _whole_steps(moved * PHYSICS_STEP_HZ / speed)


def _consensus(ratios: list[float], need: int) -> int | None:
    """One step count shared by every vehicle in the buffer, else None."""
    if len(ratios) < need:
        return None
    median = statistics.median(ratios)
    steps = _whole_steps(median)
    if steps is None:
        return None
    agree = sum(1 for r in ratios if abs(r - steps) < _AGREE_TOLERANCE)
    return steps if agree >= max(need, _AGREE_SHARE * len(ratios)) else None


def _displacement(a: Vehicle, b: Vehicle) -> float:
    return math.hypot(b.position.x - a.position.x, b.position.z - a.position.z)


def _pair_steps(
    snapshots: list[_Snapshot], i: int, lo: int, hi: int, window_steps: float,
) -> int | None:
    """Steps between snapshots ``i - 1`` and ``i``.

    AI traffic carries its own speed. TMP does not, so its per-step travel is measured
    over the ``lo``..``hi`` window, whose length in steps is ``window_steps``.
    """
    ai: list[float] = []
    tmp: list[float] = []
    for vid, b in snapshots[i].items():
        a = snapshots[i - 1].get(vid)
        if a is None or b.is_parked:
            continue
        if not b.is_tmp:
            if min(abs(a.speed), abs(b.speed)) >= _MIN_AI_SPEED_MS:
                ai.append(_displacement(a, b) * 2.0 * PHYSICS_STEP_HZ / (abs(a.speed) + abs(b.speed)))
            continue
        first, last = snapshots[lo].get(vid), snapshots[hi].get(vid)
        if first is None or last is None or window_steps <= 0.0:
            continue
        per_step = _displacement(first, last) / window_steps
        if per_step * PHYSICS_STEP_HZ >= _MIN_TMP_SPEED_MS:
            tmp.append(_displacement(a, b) / per_step)
    counted = _consensus(ai, 1)
    if counted is not None:
        return counted
    return _consensus(tmp, _MIN_TMP_VEHICLES)


def _step_index(frames: list[RadarFrameRecord], *steps: _Steps) -> tuple[list[int], int]:
    """Cumulative step index from the first available count, wall time as the last resort."""
    index = [0] * len(frames)
    counted = 0
    for i in range(1, len(frames)):
        step = next((s[i] for s in steps if s[i] is not None), None)
        if step is None:
            step = max(0, round((frames[i].t_wall - frames[i - 1].t_wall) * PHYSICS_STEP_HZ))
        else:
            counted += 1
        index[i] = index[i - 1] + step
    return index, counted


def _traffic_steps(
    frames: list[RadarFrameRecord], snapshots: list[_Snapshot], ego_steps: _Steps,
) -> _Steps:
    """Two passes: TMP travel per step on wall time first, then on the index that pass produced."""
    n = len(frames)
    windows = [(max(0, i - _TMP_BASELINE_FRAMES), min(n - 1, i + _TMP_BASELINE_FRAMES)) for i in range(n)]
    first: _Steps = [None]
    for i in range(1, n):
        lo, hi = windows[i]
        wall_steps = (frames[hi].t_wall - frames[lo].t_wall) * PHYSICS_STEP_HZ
        first.append(_pair_steps(snapshots, i, lo, hi, wall_steps))
    index, _ = _step_index(frames, first, ego_steps)
    refined: _Steps = [None]
    for i in range(1, n):
        lo, hi = windows[i]
        refined.append(_pair_steps(snapshots, i, lo, hi, float(index[hi] - index[lo])))
    return refined


def _per_step_rates(frames: list[RadarFrameRecord], ego_steps: _Steps) -> tuple[list[float], list[float]]:
    """Yaw (turns) and speed (m/s) change per physics step, held across uncounted pairs."""
    yaw = [0.0] * len(frames)
    speed = [0.0] * len(frames)
    for i in range(1, len(frames)):
        steps = ego_steps[i]
        if steps:
            turn = (frames[i].ego.rotationX - frames[i - 1].ego.rotationX + 0.5) % 1.0 - 0.5
            yaw[i] = turn / steps
            speed[i] = (frames[i].ego.speed - frames[i - 1].ego.speed) / steps
        else:
            yaw[i] = yaw[i - 1]
            speed[i] = speed[i - 1]
    return yaw, speed


def pairing_lag_steps(traffic_steps: _Steps, ego_steps: _Steps) -> list[int]:
    """How many physics steps each ego sample trails its traffic buffer, re-anchored locally."""
    n = len(traffic_steps)
    phase = [0] * n
    for i in range(1, n):
        nt, ne = traffic_steps[i], ego_steps[i]
        phase[i] = phase[i - 1] + (nt - ne if nt is not None and ne is not None else 0)
    lags = []
    for i in range(n):
        floor = min(phase[max(0, i - _PHASE_WINDOW_FRAMES): i + _PHASE_WINDOW_FRAMES + 1])
        lags.append(max(0, min(_PHASE_MAX_STEPS, phase[i] - floor)))
    return lags


def _repair_ego_pairing(
    frames: list[RadarFrameRecord], traffic_steps: _Steps, ego_steps: _Steps,
) -> None:
    lags = pairing_lag_steps(traffic_steps, ego_steps)
    yaw_rate, speed_rate = _per_step_rates(frames, ego_steps)
    last = len(frames) - 1
    for i, f in enumerate(frames):
        steps = lags[i]
        if steps == 0:
            continue
        ahead = min(i + 1, last)
        ego = f.ego
        heading = (ego.rotationX + 0.5 * yaw_rate[ahead] * steps) * 2.0 * math.pi
        mean_speed = ego.speed + speed_rate[ahead] * (steps + 1) * 0.5
        advance = mean_speed * steps / PHYSICS_STEP_HZ
        ego.coordinateX += -math.sin(heading) * advance
        ego.coordinateZ += -math.cos(heading) * advance
        ego.rotationX = (ego.rotationX + yaw_rate[ahead] * steps) % 1.0
        ego.speed += speed_rate[ahead] * steps


def _rebuild_clock(
    frames: list[RadarFrameRecord], traffic_steps: _Steps, ego_steps: _Steps,
) -> None:
    """Replace ``t_wall`` with simulated time when the counts explain the segment's duration."""
    index, counted = _step_index(frames, traffic_steps, ego_steps)
    if counted < _CLOCK_MIN_COUNTED_SHARE * (len(frames) - 1):
        return
    span = frames[-1].t_wall - frames[0].t_wall
    if abs(index[-1] / PHYSICS_STEP_HZ - span) > _CLOCK_SPAN_TOLERANCE * span + 2.0 / PHYSICS_STEP_HZ:
        return
    t0 = frames[0].t_wall
    for f, steps in zip(frames, index):
        f.t_wall = t0 + steps / PHYSICS_STEP_HZ
