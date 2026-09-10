"""Held TMP frames coast on their frozen accel. See core/radar/README.md §7."""
from __future__ import annotations

import math

from core.radar.traffic import (
    Position, Quaternion, Size, Vehicle, _hold_coast_speed,
)

DT = 0.0501


def _quat_yaw(yaw_deg: float) -> Quaternion:
    half = math.radians(yaw_deg) / 2.0
    return Quaternion(math.cos(half), 0.0, math.sin(half), 0.0)


def _vehicle(z: float, speed: float, yaw_deg: float = 0.0) -> Vehicle:
    return Vehicle(
        Position(0.0, 0.0, z),
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


def _step(prev: Vehicle, t_now: float, z: float, ego_speed: float = 0.0) -> Vehicle:
    cur = _vehicle(z, prev.speed)
    cur.update_from_last(prev, t_now, 0.0, 0.0, 400.0, ego_speed)
    return cur


def _seed_braking(speed: float, decel: float, duration_s: float = 1.2):
    """Drive a target down a steady decel ramp so ``acceleration`` settles negative."""
    t_now = 0.0
    z = 0.0
    prev = _vehicle(z, speed)
    prev.time = t_now
    for _ in range(int(duration_s / DT)):
        t_now += DT
        speed = max(0.0, speed - decel * DT)
        z -= speed * DT
        prev = _step(prev, t_now, z)
    return prev, t_now, z


def test_hold_coast_only_shrinks_magnitude():
    """Decel is integrated, accel is not, and the sign never flips."""
    assert _hold_coast_speed(14.0, -4.0, 0.25) == 14.0 - 1.0
    # Accelerating: extrapolating faster would bias every consumer toward less
    # braking, so a hold stays flat instead.
    assert _hold_coast_speed(14.0, 2.0, 0.25) == 14.0
    # Never crosses zero, however long the hold.
    assert _hold_coast_speed(1.0, -8.0, 1.0) == 0.0
    # Reverse: only the component that slows the target down is integrated.
    assert _hold_coast_speed(-5.0, 1.0, 0.5) == -4.5
    assert _hold_coast_speed(-5.0, -1.0, 0.5) == -5.0
    assert _hold_coast_speed(14.0, -4.0, 0.0) == 14.0


def test_position_mismatch_keeps_a_braking_target_decelerating():
    """Clip 2da7f2fb: a rewind held a hard-braking lead at cruise speed for 0.27 s."""
    prev, t_now, z = _seed_braking(18.0, 4.5)
    assert prev.acceleration < -2.0
    entry_speed = prev.speed
    frozen_accel = prev.acceleration
    frozen_acc_accel = prev.acc_accel
    entry_acc_speed = prev.acc_speed

    # Out-of-order packets: raw position steps backward along the heading.
    held = []
    for _ in range(4):
        t_now += DT
        z += 0.05
        prev = _step(prev, t_now, z)
        assert prev._pos_mismatch_frames > 0
        held.append(prev.speed)

    assert held == sorted(held, reverse=True), "held speed must keep falling"
    assert held[-1] < entry_speed - 0.5
    # The accel itself is frozen, not zeroed: consumers keep seeing the brake.
    assert prev.acceleration == frozen_accel
    assert prev.acc_accel == frozen_acc_accel
    assert prev.acc_speed < entry_acc_speed  # ACC lane coasts on its own accel
    # Coasted value drives the filter state, so the release does not snap back up.
    assert prev._speed_ema == prev.speed
    assert prev._smooth_speed == prev.speed


def test_position_mismatch_holds_flat_when_not_braking():
    """No decel evidence means no extrapolation: the old hold behaviour stands."""
    t_now = 0.0
    z = 0.0
    prev = _vehicle(z, 20.0)
    prev.time = t_now
    for _ in range(24):
        t_now += DT
        z -= 20.0 * DT
        prev = _step(prev, t_now, z)
    cruise_speed = prev.speed

    for _ in range(3):
        t_now += DT
        z += 0.05
        prev = _step(prev, t_now, z)
        assert prev._pos_mismatch_frames > 0
    assert prev.speed == cruise_speed


def test_lag_freeze_coasts_where_its_decay_ramp_is_still_flat():
    """``frac²`` is ~0 for the first frames, so the ramp alone holds a braking target."""
    prev, t_now, z = _seed_braking(20.0, 1.5, duration_s=1.6)
    assert prev.acceleration < -0.5
    entry_speed = prev.speed
    frozen_accel = prev.acceleration

    frozen = []
    for _ in range(4):
        t_now += DT
        prev = _step(prev, t_now, z, ego_speed=20.0)  # identical pose: packet stall
        frozen.append(prev.speed)
    assert prev._lag_since is not None

    # First frozen frame: the decay factor is 1.0, so only the coast can move it.
    assert frozen[0] < entry_speed
    assert frozen == sorted(frozen, reverse=True)
    # The accel is frozen, not zeroed as it was before the coast.
    assert prev.acceleration == frozen_accel


def test_lag_freeze_on_a_cruising_target_keeps_the_decay_ramp():
    """Entry gates mean most freezes open on a coasting target: ramp unchanged."""
    speed = 16.0
    t_now = 0.0
    z = 0.0
    prev = _vehicle(z, speed)
    prev.time = t_now
    for _ in range(30):
        t_now += DT
        z -= speed * DT
        prev = _step(prev, t_now, z, ego_speed=20.0)
    assert abs(prev.acceleration) < 1e-9
    entry_speed = prev.speed

    frozen = []
    for _ in range(6):
        t_now += DT
        prev = _step(prev, t_now, z, ego_speed=20.0)
        frozen.append(prev.speed)
    assert prev._lag_since is not None
    # frac = 0 on the first frame and the accel is zero, so nothing moves yet.
    assert frozen[0] == entry_speed
    assert frozen[-1] < entry_speed
