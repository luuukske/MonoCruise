"""Crash vs lag flag discrimination on TMP vehicles. See core/radar/README.md §7."""
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


def test_braking_to_a_stop_never_enters_lag_freeze():
    """The regression from clips f7a2793c / b3419ab0.

    A target braking to a standstill in a straight line has ~0 rotation, so the
    rotation gate alone let it in. The filtered speed still reads high there, so the
    freeze pinned a stopped vehicle at ~9 m/s for over a second.
    """
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed)

    true_speed = speed
    while true_speed > 0.0:
        next_speed = max(0.0, true_speed - 6.0 * DT)
        z -= 0.5 * (true_speed + next_speed) * DT
        t_now += DT
        prev = _step(prev, t_now, z, next_speed, yaw, ego_speed=20.0)
        assert prev._lag_since is None
        true_speed = next_speed

    # And it stays out once fully stopped, so the stop reaches AEB raw.
    for _ in range(int(0.6 / DT)):
        t_now += DT
        prev = _step(prev, t_now, z, 0.0, yaw, ego_speed=20.0)
        assert prev._lag_since is None
        assert prev.lag_confirmed is False
    assert abs(prev.speed) < 0.5


def test_stopped_target_speed_converges_without_a_freeze_holding_it():
    """A stopped target must reach ~0 promptly, not sit at a stale filtered speed."""
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed)

    true_speed = speed
    while true_speed > 0.0:
        next_speed = max(0.0, true_speed - 6.0 * DT)
        z -= 0.5 * (true_speed + next_speed) * DT
        t_now += DT
        prev = _step(prev, t_now, z, next_speed, yaw, ego_speed=20.0)
        true_speed = next_speed

    stopped_at = t_now
    while abs(prev.speed) >= 1.0:
        t_now += DT
        prev = _step(prev, t_now, z, 0.0, yaw, ego_speed=20.0)
        assert t_now - stopped_at < 0.8, "stopped target still reads above 1 m/s"


def test_decaying_motion_before_a_dropout_blocks_lag_entry():
    """Gate A: raw motion already collapsing means a stop, even without a full stall."""
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed)

    # Decelerate hard enough that the recent raw window is well under the older one,
    # then drop a single near-zero-displacement frame.
    true_speed = speed
    for _ in range(6):
        true_speed = max(0.0, true_speed - 9.0 * DT)
        z -= true_speed * DT
        t_now += DT
        prev = _step(prev, t_now, z, true_speed, yaw, ego_speed=20.0)

    t_now += DT
    prev = _step(prev, t_now, z, true_speed, yaw, ego_speed=20.0)
    assert prev._lag_since is None


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


def _quat(pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0) -> Quaternion:
    """Quaternion whose ``euler()`` reads back (pitch, yaw, roll); undoes the x/y swap."""
    a, b, c = (math.radians(d) / 2.0 for d in (yaw_deg, pitch_deg, roll_deg))
    w = math.cos(a) * math.cos(b) * math.cos(c) + math.sin(a) * math.sin(b) * math.sin(c)
    x = math.sin(a) * math.cos(b) * math.cos(c) - math.cos(a) * math.sin(b) * math.sin(c)
    y = math.cos(a) * math.sin(b) * math.cos(c) + math.sin(a) * math.cos(b) * math.sin(c)
    z = math.cos(a) * math.cos(b) * math.sin(c) - math.sin(a) * math.sin(b) * math.cos(c)
    return Quaternion(w, y, x, z)


def _pose_step(prev: Vehicle, t_now: float, z: float, speed: float,
               pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0,
               y: float = 0.0, is_trailer: bool = False) -> Vehicle:
    cur = Vehicle(
        Position(0.0, y, z), _quat(pitch, yaw, roll), Size(2.5, 3.0, 13.6),
        speed, 0.0, 0, [], 1, True, is_trailer,
    )
    cur.update_from_last(prev, t_now, 0.0, 0.0, 100.0, 0.0)
    return cur


def test_quat_helper_round_trips():
    p, y, r = _quat(1.2, 30.0, -0.8).euler()
    assert abs(p - 1.2) < 1e-6 and abs(y - 30.0) < 1e-6 and abs(r + 0.8) < 1e-6


def test_rotation_built_over_two_frames_turns_a_shove_into_a_crash():
    """Clips dc3c6e29 / 168ba6a8: TMP spreads the impact over two packets.

    Neither frame steps the pitch rate past the bar on its own, so the one-frame
    test missed it and position mismatch held the trailer at cruise speed.
    """
    speed = 15.0
    prev, t_now, z, _ = _seed_cruise(speed)

    # Pitch rate 0 -> 8 -> 16 deg/s: each step is under the 12 deg/s bar.
    t_now += DT
    z -= speed * DT
    pitch = 8.0 * DT
    prev = _pose_step(prev, t_now, z, speed, pitch=pitch)
    assert prev.crash_confirmed is False

    # Second frame knocks it back against travel.
    t_now += DT
    z += 0.3
    pitch += 16.0 * DT
    prev = _pose_step(prev, t_now, z, speed, pitch=pitch)
    assert prev.crash_confirmed is True
    assert not prev.pos_mismatch_holding
    assert prev.position.z == z


def test_rewind_with_steady_rotation_is_still_held():
    """A step back with nothing new in the rotation is a netcode rewind, not a crash."""
    speed = 15.0
    prev, t_now, z, yaw = _seed_cruise(speed, yaw_rate_deg_s=8.0)
    held_z = prev.position.z

    t_now += DT
    yaw += 8.0 * DT
    prev = _pose_step(prev, t_now, z + 0.4, speed, yaw=yaw)
    assert prev.crash_confirmed is False
    assert prev.pos_mismatch_holding
    assert prev.position.z == held_z


def test_slow_step_back_with_rotation_is_not_a_shove():
    """Below walking pace a step back is jitter, whatever the rotation does."""
    speed = 2.0
    prev, t_now, z, _ = _seed_cruise(speed)

    for k in range(2):
        t_now += DT
        z += 0.02
        prev = _pose_step(prev, t_now, z, speed, roll=(k + 1) * 30.0 * DT)
    assert prev.crash_confirmed is False


def test_subframe_snap_does_not_read_as_displacement_collapse():
    """A sub-frame moves prev._raw_x forward, so the frame-to-frame step is short.

    Measured against it, every cruising frame looked collapsed and a lane-change
    roll alone confirmed a crash. The planar checks measure from the last live frame.
    """
    speed = 20.0
    prev, t_now, z, _ = _seed_cruise(speed)
    full_t = t_now
    roll = 0.0
    for k in range(10):
        # Roll rate steps 0 -> 25 deg/s at k = 5, past the 20 deg/s bar.
        roll_rate = 25.0 if k >= 5 else 0.0
        # Sub-frame 40 ms in, then the full frame 67 ms after the last one.
        prev = _pose_step(prev, full_t + 0.040, z - speed * 0.040, speed,
                          roll=roll + roll_rate * 0.040)
        full_t += 0.0667
        z -= speed * 0.0667
        roll += roll_rate * 0.0667
        prev = _pose_step(prev, full_t, z, speed, roll=roll)
        assert prev.crash_confirmed is False


def _knock_back(speed: float, is_trailer: bool = False):
    """Cruise, then two frames of pitch build-up that end in a step back: a crash shove."""
    prev, t_now, z, _ = _seed_cruise(speed)
    prev.is_trailer = is_trailer
    t_now += DT
    z -= speed * DT
    pitch = 8.0 * DT
    prev = _pose_step(prev, t_now, z, speed, pitch=pitch, is_trailer=is_trailer)
    t_now += DT
    z += 0.3
    pitch += 16.0 * DT
    prev = _pose_step(prev, t_now, z, speed, pitch=pitch, is_trailer=is_trailer)
    return prev, t_now, z, pitch


def _long_window_speed(v: Vehicle) -> float:
    from core.radar.traffic import _raw_speed_from_position_history
    yaw = v._smooth_yaw
    return _raw_speed_from_position_history(v._position_history, -math.sin(yaw), -math.cos(yaw))


def test_knocked_back_crash_reads_its_stop_through_the_short_window():
    """Clip 168ba6a8: the long window still read 21 m/s while the rig had stopped."""
    speed = 20.0
    prev, t_now, z, pitch = _knock_back(speed)
    assert prev.crash_confirmed is True
    for _ in range(4):
        t_now += DT
        prev = _pose_step(prev, t_now, z, 0.0, pitch=pitch)
    assert prev._raw_brake_active is True
    assert abs(prev._raw_speed) < 1.0
    assert _long_window_speed(prev) > 10.0


def test_crash_without_a_knock_keeps_the_long_window():
    """Rotation and a vertical jolt with the body still driving on: no stop to follow."""
    speed = 20.0
    prev, t_now, z, _ = _seed_cruise(speed)
    y = 0.0
    for k in range(2):
        t_now += DT
        z -= speed * DT
        y += 0.2 if k == 1 else 0.0
        prev = _pose_step(prev, t_now, z, speed, pitch=(k + 1) * 16.0 * DT, y=y)
    assert prev.crash_confirmed is True
    assert prev.time >= prev._crash_knock_until
    assert prev._raw_brake_active is False


def test_trailer_follows_its_crash_stop_then_hands_back():
    speed = 20.0
    prev, t_now, z, pitch = _knock_back(speed, is_trailer=True)
    assert prev.crash_confirmed is True
    for _ in range(4):
        t_now += DT
        prev = _pose_step(prev, t_now, z, 0.0, pitch=pitch, is_trailer=True)
    assert prev._raw_brake_active is True
    assert abs(prev._raw_speed) < 1.0
    # A trailer keeps it for the crash only: once the latch lapses it is off again.
    while prev.crash_confirmed:
        t_now += DT
        prev = _pose_step(prev, t_now, z, 0.0, pitch=pitch, is_trailer=True)
    t_now += DT
    prev = _pose_step(prev, t_now, z, 0.0, pitch=pitch, is_trailer=True)
    assert prev._raw_brake_active is False


def test_stall_that_resumes_behind_is_not_a_collapse():
    """Clip aeedafdf: a frozen frame, then TMP resumes from where it froze.

    Measured across the stall that reads as a collapse, and a cresting truck's pitch
    supplied the rotation jerk. Displacement across frozen frames is no evidence.
    """
    speed = 25.0
    prev, t_now, z, _ = _seed_cruise(speed)
    t_now += DT
    prev = _pose_step(prev, t_now, z, speed)                        # frozen
    pitch = 0.0
    for k in range(4):
        t_now += DT
        # Resumes 70 % short of where it should be, then drives on normally.
        z -= (0.3 if k == 0 else 1.0) * speed * DT
        pitch += 30.0 * (2 * DT if k == 0 else DT)
        prev = _pose_step(prev, t_now, z, speed, pitch=pitch)
        assert prev.crash_confirmed is False
