"""TMP pose jumps are relocations, not motion. See core/radar/README.md §7."""
from __future__ import annotations

import math

from core.radar.traffic import (
    _POSE_JUMP_SETTLE_S, Position, Quaternion, Size, Vehicle,
)

DT = 0.0667   # TMP full-frame cadence; 0.0501 sits on the sub-frame edge


def _quat_yaw(yaw_deg: float) -> Quaternion:
    half = math.radians(yaw_deg) / 2.0
    return Quaternion(math.cos(half), 0.0, math.sin(half), 0.0)


def _step(prev: Vehicle | None, t_now: float, x: float, z: float,
          yaw_deg: float = 0.0) -> Vehicle:
    cur = Vehicle(
        Position(x, 0.0, z), _quat_yaw(yaw_deg), Size(2.5, 3.0, 6.0),
        0.0, 0.0, 0, [], 1, True, False,
    )
    if prev is None:
        cur.time = t_now
    else:
        cur.update_from_last(prev, t_now, 0.0, 0.0, 400.0, 0.0)
    return cur


def _drive(speed: float, duration_s: float = 1.5):
    """Straight along -z (yaw 0) at a steady speed."""
    t_now = 0.0
    z = 0.0
    prev = _step(None, t_now, 0.0, z)
    for _ in range(int(duration_s / DT)):
        t_now += DT
        z -= speed * DT
        prev = _step(prev, t_now, 0.0, z)
    return prev, t_now, z


def test_standing_vehicle_flying_to_a_new_spot_keeps_standing():
    """Clips bc99e7ff / 2b81f188: a parked TMP vehicle swings round and flies metres a frame.

    Read as motion, the flight became 10 to 44 m/s of phantom speed next to ego and a
    false brake, and confirmed a crash on the way. Settling after it may read a crawl.
    """
    prev, t_now, z = _drive(0.0)
    x, yaw = 0.0, 0.0
    for dx, dz, dyaw in ((2.7, 0.0, -50.0), (5.0, 8.0, -70.0), (2.0, 3.5, -30.0),
                         (1.2, 0.8, -39.0), (0.1, 0.2, 3.0), (0.0, 0.1, 2.0)):
        t_now += DT
        x += dx
        z += dz
        yaw += dyaw
        prev = _step(prev, t_now, x, z, yaw)
        assert abs(prev.speed) < 3.0
        assert prev.crash_confirmed is False
        # The pose itself is taken: the vehicle is where TMP now puts it.
        assert (prev.position.x, prev.position.z) == (x, z)


def test_moving_vehicle_snapped_forward_keeps_its_speed():
    """A desync correction several metres ahead reads 100+ m/s for one frame."""
    prev, t_now, z = _drive(25.0)
    t_now += DT
    z -= 25.0 * DT + 8.0
    prev = _step(prev, t_now, 0.0, z)
    assert prev.position.z == z
    assert abs(prev.speed - 25.0) < 1.0
    for _ in range(10):
        t_now += DT
        z -= 25.0 * DT
        prev = _step(prev, t_now, 0.0, z)
    assert abs(prev.speed - 25.0) < 2.0


def test_fast_catch_up_is_real_motion():
    """TMP catching a vehicle up runs it well past its estimate; that is not a jump."""
    prev, t_now, z = _drive(15.0)
    for _ in range(12):
        t_now += DT
        z -= 45.0 * DT
        prev = _step(prev, t_now, 0.0, z)
        assert len(prev._position_history) > 2
    assert prev.speed > 30.0


def test_stall_catch_up_is_not_a_jump():
    """After a stall the position catches up by speed times the stall, all at once."""
    prev, t_now, z = _drive(25.0)
    for _ in range(10):
        t_now += DT
        prev = _step(prev, t_now, 0.0, z)
    t_now += DT
    z -= 25.0 * 11 * DT
    prev = _step(prev, t_now, 0.0, z)
    assert len(prev._position_history) > 2


def test_vehicle_that_drives_off_after_landing_is_let_go():
    """The tighter bars last one window from the first jump, never re-armed by the tail."""
    prev, t_now, z = _drive(0.0)
    t_now += DT
    z -= 6.0
    prev = _step(prev, t_now, 0.0, z)
    landed = t_now
    while t_now - landed < _POSE_JUMP_SETTLE_S + 1.5:
        t_now += DT
        z -= 20.0 * DT
        prev = _step(prev, t_now, 0.0, z)
    assert prev.speed > 15.0


def test_new_track_is_not_judged():
    """A vehicle entering range at speed has no speed estimate yet to judge a step by."""
    t_now = 0.0
    z = 0.0
    prev = _step(None, t_now, 0.0, z)
    for k in range(4):
        t_now += DT
        z -= 30.0 * DT
        prev = _step(prev, t_now, 0.0, z)
        assert len(prev._position_history) == k + 1
    assert prev.speed > 25.0
