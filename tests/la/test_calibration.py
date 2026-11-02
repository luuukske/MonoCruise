"""Learning the game's own steering force per m/s^2 of cornering from steady bends."""

from __future__ import annotations

import math
import random

import pytest

from core.la.calibration import CalTuning, ForceCalibration, heading_angle

DT = 0.05
SPEED = 25.0


def yaw_for(angle: float) -> float:
    """The radar yaw whose travel direction is ``angle`` (``atan2(dz, dx)``)."""
    return math.atan2(-math.cos(angle), -math.sin(angle))


def drive(cal, accel_of_t, force_of, seconds, speed=SPEED, t0=0.0, start_angle=0.3, frames=True, gain=None):
    """Integrate heading from a lateral accel plan and feed the force the game would report for it.

    With ``gain`` the wheel sits where that steering gain puts it: ``kappa / gain`` of rightward lock.
    """
    angle = start_angle
    result = cal.result()
    for i in range(int(seconds / DT)):
        t = t0 + i * DT
        accel = accel_of_t(t)
        angle += accel / speed * DT
        steer = 0.0 if gain is None else accel / (speed * speed) / gain
        result = cal.update(t, speed, yaw_for(angle), steer, round(force_of(accel)), i if frames else 0)
    return result


def bends(t: float) -> float:
    """Steady bends both ways, held 6 s each, so most of the time is steady cornering."""
    return (0.0, 0.8, 0.0, -0.6, 0.4, -1.0)[int(t // 6.0) % 6]


def test_heading_angle_grows_while_turning_right():
    assert heading_angle(yaw_for(0.25)) == pytest.approx(0.25)
    assert heading_angle(yaw_for(-1.0)) == pytest.approx(-1.0)


@pytest.mark.parametrize("slope", [-2400.0, 1800.0])
def test_a_clean_game_force_is_learned_with_its_sign(slope):
    rng = random.Random(3)
    result = drive(ForceCalibration(), bends, lambda a: slope * a + 150.0 + rng.gauss(0.0, 250.0), 120.0)
    assert result.state == "ready"
    assert result.force_per_ms2 == pytest.approx(slope, rel=0.1)


def test_a_force_unrelated_to_cornering_never_calibrates():
    rng = random.Random(5)
    result = drive(ForceCalibration(), bends, lambda a: rng.gauss(0.0, 1500.0), 120.0)
    assert result.state == "learning" and result.force_per_ms2 is None


def test_straight_roads_alone_do_not_calibrate():
    result = drive(ForceCalibration(), lambda t: 0.0, lambda a: 2000.0 * a, 120.0)
    assert result.force_per_ms2 is None


def test_slow_driving_and_a_frozen_plugin_teach_nothing():
    slow = drive(ForceCalibration(), bends, lambda a: 2000.0 * a, 120.0, speed=8.0)
    frozen = drive(ForceCalibration(), bends, lambda a: 2000.0 * a, 120.0, frames=False)
    assert slow.weight_s == 0.0
    assert frozen.weight_s < 1.0


def test_a_saturated_game_force_is_not_a_sample():
    result = drive(ForceCalibration(), bends, lambda a: 9500.0, 120.0)
    assert result.weight_s == 0.0


def test_turn_in_transients_are_skipped():
    cal = ForceCalibration(CalTuning(min_weight_s=1e9))
    drive(cal, lambda t: 1.5 * math.sin(2.0 * math.pi * t / 2.0), lambda a: 2000.0 * a, 60.0)
    assert cal.result().weight_s < 5.0, "a wheel that never holds a bend gives almost no steady samples"


def test_a_sign_that_flips_later_locks_the_assist_out():
    cal = ForceCalibration()
    assert drive(cal, bends, lambda a: 2000.0 * a, 120.0).state == "ready"
    result = drive(cal, bends, lambda a: -2000.0 * a, 900.0, t0=120.0)
    assert result.state == "conflict" and result.force_per_ms2 is None
    assert drive(cal, bends, lambda a: 2000.0 * a, 600.0, t0=1020.0).state == "conflict"


def test_an_implausible_slope_is_refused():
    result = drive(ForceCalibration(), bends, lambda a: 20.0 * a, 120.0)
    assert result.state == "implausible"


def test_a_ready_calibration_holds_through_a_stretch_of_noise():
    cal = ForceCalibration()
    assert drive(cal, bends, lambda a: 2000.0 * a, 120.0).state == "ready"
    rng = random.Random(9)
    noisy = drive(cal, bends, lambda a: rng.gauss(0.0, 3000.0), 300.0, t0=120.0)
    assert noisy.state == "held"
    assert 0.2 * 2000.0 < noisy.force_per_ms2 <= 2000.0 * 1.05, "noise only ever weakens it, and it stays on"


def test_the_wheel_spring_and_the_steering_gain_are_learned():
    gain, spring = 0.19, 34570.0
    rng = random.Random(7)

    def force(accel):
        return spring * accel / (SPEED * SPEED) / gain + rng.gauss(0.0, 150.0)

    result = drive(ForceCalibration(), bends, force, 120.0, gain=gain)
    assert result.servo_ready
    assert result.force_per_ms2 * result.kappa_per_steer.value * SPEED * SPEED == pytest.approx(spring, rel=0.1)
    assert result.kappa_per_steer.value == pytest.approx(gain, rel=0.05)


def test_the_spring_is_right_at_every_speed_when_tight_bends_are_taken_slower():
    gain, per_v2 = 0.19, 130.0
    cal = ForceCalibration()

    def force(accel):
        return per_v2 * accel / gain

    drive(cal, lambda t: 2.0 * bends(t), force, 90.0, speed=15.0, gain=gain)
    result = drive(cal, lambda t: 0.6 * bends(t), force, 90.0, speed=25.0, t0=90.0, gain=gain)
    assert result.servo_ready
    for speed in (15.0, 25.0):
        spring = result.force_per_ms2 * result.kappa_per_steer.value * speed * speed
        assert spring == pytest.approx(per_v2 * speed * speed, rel=0.1)


def test_a_wheel_that_steers_backwards_is_refused():
    result = drive(ForceCalibration(), bends, lambda a: 2000.0 * a, 120.0, gain=-0.19)
    assert result.kappa_per_steer.value is None and result.kappa_per_steer.state == "implausible"
    assert not result.servo_ready


def test_a_still_wheel_teaches_no_steering_gain():
    result = drive(ForceCalibration(), bends, lambda a: 2000.0 * a, 120.0)
    assert result.force_per_ms2 is not None
    assert result.kappa_per_steer.value is None and not result.servo_ready

