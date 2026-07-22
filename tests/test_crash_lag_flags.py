"""Crash / lag flag discrimination (TMP vehicles).

A packet stall (network lag) freezes the whole pose and must never read as a
crash: not on stall entry, not on the resume snap. A physical crash keeps a
live data stream with violent content (rotation jerk plus a kinematic
anomaly) and must never be masked by the lag freeze. See radar AGENTS.md §7.
"""

from __future__ import annotations

import math

from core.radar.traffic import (
    _CRASH_HOLD_S, Position, Quaternion, Size, Vehicle,
)

DT = 0.0501


def _quat_yaw(yaw_deg: float) -> Quaternion:
    half = math.radians(yaw_deg) / 2.0
    return Quaternion(math.cos(half), 0.0, math.sin(half), 0.0)


def _vehicle(z: float, speed: float, yaw_deg: float = 0.0,
             y: float = 0.0) -> Vehicle:
    v = Vehicle(
        Position(0.0, y, z),
        _quat_yaw(yaw_deg),
        Size(2.5, 3.0, 6.0),
        speed,
        0.0,
        0,
        [],
        1,
        True,
        False,
    )
    return v


def _step(prev: Vehicle, t_now: float, z: float, speed: float,
          yaw_deg: float = 0.0, y: float = 0.0,
          ego_speed: float = 0.0) -> Vehicle:
    cur = _vehicle(z, speed, yaw_deg, y)
    cur.update_from_last(prev, t_now, 0.0, 0.0, 100.0, ego_speed)
    return cur


def _seed_cruise(speed: float, duration_s: float = 1.5,
                 yaw_rate_deg_s: float = 0.0):
    """Cruise with an optional steady yaw rate so rotation rates are nonzero."""
    t_now = 0.0
    z = 0.0
    yaw = 0.0
    prev = _vehicle(z, speed, yaw)
    prev.time = t_now
    for _ in range(int(duration_s / DT)):
        t_now += DT
        z -= speed * DT
        yaw += yaw_rate_deg_s * DT
        prev = _step(prev, t_now, z, speed, yaw)
    return prev, t_now, z, yaw


def test_packet_stall_never_reads_as_crash():
    """Freeze entry, frozen frames, and the resume snap must not fire crash."""
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed, yaw_rate_deg_s=5.0)
    assert prev.crash_confirmed is False

    # Stall: identical pose for 0.4 s (ego far away so the freeze window runs).
    stall_frames = int(0.4 / DT)
    for _ in range(stall_frames):
        t_now += DT
        prev = _step(prev, t_now, z, speed, yaw)
        assert prev.crash_confirmed is False

    # Resume where the vehicle actually is, rotation continued at its rate.
    stall_dur = stall_frames * DT
    z -= speed * (stall_dur + DT)
    yaw += 5.0 * (stall_dur + DT)
    for _ in range(4):
        t_now += DT
        prev = _step(prev, t_now, z, speed, yaw)
        assert prev.crash_confirmed is False
        z -= speed * DT
        yaw += 5.0 * DT


def test_crash_reversal_fires_and_latches():
    """A backward bounce with rotation jerk confirms and latches."""
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed)

    # Impact frame: position jumps backward, yaw snaps hard.
    t_now += DT
    z += 0.4
    yaw += 4.0
    prev = _step(prev, t_now, z, 0.0, yaw)
    assert prev.crash_confirmed is True

    # Latch survives benign frames afterward (wreck settles, no new jerk).
    fired_at = t_now
    while t_now - fired_at < _CRASH_HOLD_S - 2 * DT:
        t_now += DT
        prev = _step(prev, t_now, z, 0.0, yaw)
        assert prev.crash_confirmed is True

    # And expires once the hold runs out with no further qualifying frames.
    while t_now - fired_at < _CRASH_HOLD_S + 4 * DT:
        t_now += DT
        prev = _step(prev, t_now, z, 0.0, yaw)
    assert prev.crash_confirmed is False


def test_hard_stop_with_pitch_jerk_fires_crash():
    """A near-instant stop (displacement collapse) plus rotation jerk fires."""
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed)

    # The vehicle stops nearly dead: 10 % of expected displacement per frame,
    # cab pitching hard (rate flips 30 deg/s frame to frame).
    pitch_flip = 2.0
    fired = False
    for i in range(4):
        t_now += DT
        z -= 0.1 * speed * DT
        half = math.radians(pitch_flip if i % 2 == 0 else -pitch_flip) / 2.0
        cur = Vehicle(
            Position(0.0, 0.0, z),
            Quaternion(math.cos(half), math.sin(half), 0.0, 0.0),
            Size(2.5, 3.0, 6.0),
            0.0, 0.0, 0, [], 1, True, False,
        )
        cur.update_from_last(prev, t_now, 0.0, 0.0, 100.0, 0.0)
        prev = cur
        fired = fired or prev.crash_confirmed
    assert fired is True


def test_normal_cruise_and_brake_never_fire_crash():
    """Plain driving, curves, and a hard (but physical) brake stay clean."""
    speed = 20.0
    prev, t_now, z, yaw = _seed_cruise(speed, yaw_rate_deg_s=8.0)
    assert prev.crash_confirmed is False

    # Hard brake at 6 m/s²: displacement shrinks gradually, never collapses.
    true_speed = speed
    while true_speed > 0.0:
        next_speed = max(0.0, true_speed - 6.0 * DT)
        z -= 0.5 * (true_speed + next_speed) * DT
        yaw += 8.0 * DT
        t_now += DT
        prev = _step(prev, t_now, z, next_speed, yaw)
        assert prev.crash_confirmed is False
        true_speed = next_speed


def test_physical_stop_with_live_rotation_skips_lag_freeze():
    """Frozen position + clearly live rotation is a stop, not lag."""
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed)

    # Position freezes dead while the vehicle keeps rotating at 8 deg/s
    # (crash rock). Lag entry must not fire; the stop flows to the filters.
    for _ in range(6):
        t_now += DT
        yaw += 8.0 * DT
        prev = _step(prev, t_now, z, speed, yaw, ego_speed=8.0)
        assert prev._lag_since is None
        assert prev.lag_confirmed is False


def test_packet_stall_frozen_rotation_enters_lag_freeze():
    """Control: a full pose freeze still enters the lag freeze window."""
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed)

    t_now += DT
    prev = _step(prev, t_now, z, speed, yaw, ego_speed=8.0)
    assert prev._lag_since is not None


def test_subframe_carries_crash_latch():
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed)
    t_now += DT
    z += 0.4
    yaw += 4.0
    prev = _step(prev, t_now, z, 0.0, yaw)
    assert prev.crash_confirmed is True

    sub = _step(prev, t_now + 0.01, z, 0.0, yaw)
    assert sub.crash_confirmed is True
