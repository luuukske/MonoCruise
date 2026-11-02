"""The driver's push estimate (core/la/hand_push.py) on a modelled wheel read the way the plugin reports it."""

from __future__ import annotations

import math

from core.la.hand_push import HandPush, HandPushEstimator, PushTuning

H = 0.001
TICK = 0.05
SPRING = 30000.0
INERTIA = 650.0


def ramp(t: float, start: float, size: float, over: float = 0.3) -> float:
    return size * min(1.0, max(0.0, (t - start) / over))


def drive(seconds, push, hand, only=0, direction=1, damping=0.3, steer=0.0):
    """1 kHz wheel under the game's centring, the plugin's push ``push(t)`` and a hand ``hand(t, steer, rate)``, all
    counterclockwise; read at 20 Hz as the plugin reports it (rate low-passed at 0.04 s, the game's force at 0.1 s).
    """
    est = HandPushEstimator(PushTuning(inertia=INERTIA))
    rate = rate_lp = 0.0
    game_lp = -SPRING * steer
    crit = 2.0 * math.sqrt(SPRING * INERTIA)
    rows = []
    for i in range(int(seconds / H)):
        t = i * H
        p, driver = push(t), hand(t, steer, rate)
        rate += (-SPRING * steer + p + driver - damping * crit * rate) / INERTIA * H
        steer += rate * H
        rate_lp += (rate - rate_lp) * (1.0 - math.exp(-H / 0.04))
        game_lp += (-SPRING * steer - game_lp) * (1.0 - math.exp(-H / 0.10))
        if i % round(TICK / H) == 0:
            out = est.update(TICK if i else 0.0, rate_lp, direction * game_lp, direction * p, 0.0, direction,
                             only if t >= 1.0 else 0)
            rows.append((t, out, driver))
    return rows


def free(t, steer, rate):
    return 0.0


def holding(at=0.0, bias=0.0):
    """A firm grip on the wheel at ``at`` (plus ``bias`` to hold it there against the game), passive."""
    return lambda t, steer, rate: bias + 2e6 * (at - steer) - 2.0 * math.sqrt(2e6 * INERTIA) * rate


def test_a_free_wheel_reads_little_hand_and_none_once_settled():
    rows = drive(3.0, lambda t: ramp(t, 0.5, 1500.0), free)
    assert all(abs(r.hand) < 0.4 * 1500.0 for t, r, _ in rows if t > 0.2), "the wheel's own damping on the way"
    assert all(abs(r.hand) < 100.0 for t, r, _ in rows if t > 2.0)


def test_a_wheel_held_still_reads_the_hand_cancelling_the_push():
    rows = drive(3.0, lambda t: ramp(t, 0.5, 1500.0), holding())
    assert all(abs(r.hand + 1500.0) < 75.0 for t, r, _ in rows if t > 1.5)


def test_it_reads_the_same_whichever_force_sign_turns_the_wheel_left():
    plus = drive(2.0, lambda t: ramp(t, 0.5, 1500.0), holding(), direction=1)
    minus = drive(2.0, lambda t: ramp(t, 0.5, 1500.0), holding(), direction=-1)
    assert all(abs(a.hand - b.hand) < 1e-6 for (_, a, _), (_, b, _) in zip(plus[1:], minus[1:]))


def test_passive_hands_never_go_past_the_pulse_push():
    """A pulse pushing counterclockwise (away from the right line) at 3000 into a bend the driver holds."""
    bend = 0.05
    for hand in (free, holding(bend, SPRING * bend), lambda t, s, r: SPRING * bend + 20000.0 * (bend - s)):
        rows = drive(3.0, lambda t: ramp(t, 1.0, 3000.0), hand, only=1, steer=bend)
        pulse = [r for t, r, _ in rows if t >= 1.0]
        assert pulse and all(r.beyond < 300.0 for r in pulse)
        assert max(r.push for r in pulse) > 2900.0


def test_a_driver_steering_toward_the_line_goes_past_it():
    def driver(t, steer, rate):
        return holding(at=-ramp(t, 1.5, 0.1))(t, steer, rate)
    rows = drive(3.0, lambda t: ramp(t, 1.0, 1500.0), driver, only=1)
    past = next(t for t, r, _ in rows if r.beyond >= 500.0)
    assert 1.5 < past < 1.9
    assert all(r.beyond < 300.0 for t, r, _ in rows if 1.0 <= t < 1.5), "holding still before it"


def test_no_direction_or_a_gap_gives_no_estimate_and_starts_over():
    est = HandPushEstimator()
    assert math.isnan(est.update(0.05, 0.0, 100.0, 0.0, 0.0, 0).hand)
    est.update(0.05, 0.0, 100.0, 0.0, 0.0, 1)
    assert math.isfinite(est.update(0.05, 0.0, 100.0, 0.0, 0.0, 1).hand)
    assert math.isnan(est.update(0.5, 0.0, 100.0, 0.0, 0.0, 1).hand), "a reading 0.5 s late"
    assert math.isfinite(est.update(0.05, 0.0, 100.0, 0.0, 0.0, 1).hand)


def test_outside_a_pulse_there_is_no_push_or_against():
    out = HandPush(hand=100.0, accel=0.0)
    assert math.isnan(out.push) and math.isnan(out.against) and math.isnan(out.beyond)
