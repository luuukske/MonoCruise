"""The wheel actuator: a damped servo designed on the wheel's inertia, and backing off when the plugin buzzes."""

from __future__ import annotations

import math

import pytest

from core.la.actuator import Gains, Wheel, WheelActuator, WheelTuning
from core.lane_departure.command import SERVO_OFF

GAINS = Gains(720.0, 0.19)


def test_the_servo_is_a_damped_second_order_response_on_the_wheel_inertia():
    tn = WheelTuning()
    kp, kd, ki = WheelActuator(tn).gains(1.0)
    w = math.sqrt(kp / tn.inertia)
    assert w == pytest.approx(2.0 * math.pi * tn.bandwidth_hz)
    assert kd / (2.0 * math.sqrt(kp * tn.inertia)) == pytest.approx(tn.damping)
    assert ki == pytest.approx(kp * w * tn.integral_share)


def test_the_game_spring_counts_toward_the_stiffness_so_the_total_stays_damped():
    tn = WheelTuning()
    alone = WheelActuator(tn).gains(1.0)
    kp, kd, ki = WheelActuator(tn).gains(1.0, game_spring=-40000.0)
    assert kp == pytest.approx(alone[0] - 40000.0), "only what the game's own spring does not give"
    assert (kd, ki) == pytest.approx(alone[1:]), "damping and integral are sized on the total"
    stiff_game = WheelActuator(tn).gains(1.0, game_spring=10 * alone[0])
    assert stiff_game[0] == pytest.approx(tn.own_floor * alone[0]), "the servo never drops below its floor"


def test_the_wheel_trails_its_target_longer_the_stiffer_the_games_spring():
    act = WheelActuator(WheelTuning())
    town, motorway = act.lag_s(GAINS, 22.0), act.lag_s(GAINS, 36.0)
    assert 0.1 < town < motorway < 0.4, "drives 5 and 6 measured 0.15 s at 80 to 90 km/h, 0.25 s at 100 to 130"
    assert act.lag_s(None, 30.0) is None


def test_strength_scales_the_stiffness_and_the_damping_keeps_its_share():
    weak = WheelActuator(WheelTuning(strength=0.5)).gains(1.0)
    full = WheelActuator(WheelTuning(strength=2.0)).gains(1.0)
    assert full[0] == pytest.approx(4.0 * weak[0])
    assert full[1] == pytest.approx(2.0 * weak[1]), "damping follows the square root, so the shape stays"
    assert WheelActuator(WheelTuning(strength=0.0)).commands(GAINS, 0.01, 0.0, 1.0, Wheel(ok=True)) is SERVO_OFF


def test_the_request_carries_the_target_rate_and_keeps_the_game_force_unless_asked():
    act = WheelActuator(WheelTuning())
    full = act.commands(GAINS, 0.01, 0.03, 1.0, Wheel(ok=True))
    half = act.commands(GAINS, 0.01, 0.03, 0.5, Wheel(ok=True))
    assert (full.target, full.rate, full.cancel, full.direction) == (0.01, 0.03, 0.0, 1), "the game's feel stays"
    assert half.kp < full.kp and half.cap < full.cap
    asked = WheelActuator(WheelTuning(cancel_share=0.6))
    assert asked.commands(GAINS, 0.01, 0.0, 1.0, Wheel(ok=True)).cancel == pytest.approx(0.6)
    assert 0.0 < asked.commands(GAINS, 0.01, 0.0, 0.5, Wheel(ok=True)).cancel < 0.6, "and follows the authority"
    assert act.commands(Gains(-720.0, 0.19), 0.01, 0.0, 1.0, Wheel(ok=True)).direction == -1
    assert act.commands(GAINS, 0.01, 0.0, 1.0, Wheel()) is SERVO_OFF, "no wheel reading, no servo"


def _guard_run(act, t, seconds, err_at, steer_at):
    for i in range(int(round(seconds / 0.05))):
        act.guard(t, 0.05, err_at(t), Wheel(ok=True, steer=steer_at(t)), 1.0)
        t += 0.05
    return t


def test_a_wheel_let_go_after_a_hold_snaps_over_without_cutting_the_servo():
    act = WheelActuator(WheelTuning())
    t = _guard_run(act, 0.0, 2.0, lambda t: 0.013, lambda t: 0.0144)
    t = _guard_run(act, t, 0.6, lambda t: 0.06 * math.sin(2.0 * math.pi * 2.0 * t), lambda t: 0.0144 + 0.06 * t)
    assert act.ring_scale == 1.0, "drive 8: hand or stalled reading held 2.5 s, the release swing took the servo to 0"
    t = _guard_run(act, t, 1.0, lambda t: 0.0, lambda t: 0.0)
    _guard_run(act, t, 0.6, lambda t: 0.06 * math.sin(2.0 * math.pi * 2.0 * t), lambda t: 0.06 * math.sin(t * 13.0))
    assert act.ring_scale < 0.05, "a wheel that swings without a hold still takes it out at once"


def test_the_slow_overshoot_of_a_big_return_is_no_ring():
    act = WheelActuator(WheelTuning())
    t = _guard_run(act, 0.0, 1.0, lambda t: 0.03, lambda t: 0.1 * t)
    t = _guard_run(act, t, 1.0, lambda t: -0.04, lambda t: 0.1 * t)
    _guard_run(act, t, 1.0, lambda t: 0.03, lambda t: 0.1 * t)
    assert act.ring_scale == 1.0, "drives 7 to 9: swings 0.7 to 5 s apart past 0.02 took the servo to 0, 11 times"


def test_a_wheel_chasing_a_jumping_target_is_no_ring_but_round_a_still_one_it_is():
    chased, still = WheelActuator(WheelTuning()), WheelActuator(WheelTuning())
    t = 0.0
    for _ in range(int(1.0 / 0.05)):
        target = 0.05 * math.sin(2.0 * math.pi * 2.0 * t)
        steer = 0.05 * math.sin(2.0 * math.pi * 2.0 * (t - 0.1))
        wheel = Wheel(ok=True, steer=steer, rate=1.0)
        chased.guard(t, 0.05, target - steer, wheel, 1.0, target)
        still.guard(t, 0.05, target - steer, wheel, 1.0, 0.0)
        t += 0.05
    assert chased.ring_scale == 1.0, "drive 10: a roundabout's lanes swung the target, the servo went to 0"
    assert still.ring_scale <= 0.5, "the same swings round a still target are a ring"


def test_a_buzz_from_the_plugin_backs_the_servo_off_once_per_drop():
    act = WheelActuator(WheelTuning())
    t = 0.0
    for buzz in (1.0, 1.0, 0.5, 0.5, 0.52, 0.25):
        act.guard(t, 0.05, 0.0, Wheel(ok=True, buzz=buzz), 1.0)
        t += 0.05
    assert act.ring_scale == pytest.approx(0.8 * 0.8, abs=0.01), "two drops, two cuts, none while it recovers"
    for _ in range(int(WheelTuning().ring_recover_max_s / 0.05) + 10):
        act.guard(t, 0.05, 0.0, Wheel(ok=True, buzz=1.0), 1.0)
        t += 0.05
    assert act.ring_scale == 1.0
