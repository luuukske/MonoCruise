"""LA in closed loop on a modelled wheel and truck, plus every gate, mute and driver override."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.la.actuator import WheelActuator
from core.la.controller import STRENGTH_MAX, AssistTuning, Gains, LaneAssist, WantedPath, Wheel, bend_ceiling
from core.lane_departure.command import SERVO_OFF
from tests.la.servo_sim import FORCE, GAIN, SPRING, Rig, Track, bend_track, kink_track, loop
from tests.lane_departure.test_detector import DT, HALF_WIDTH, SPEED, START_S, frame_at, make_route

TN = AssistTuning()
QUIET = AssistTuning(hands_off_s=1e9, hands_warn_s=1e9)
GAINS = Gains(FORCE, GAIN)
# Rims on the model's spring: 1.16 Hz is the R3 estimate (inertia 650), 0.8 Hz twice as heavy, 2.3 Hz four times lighter.
WHEELS = [(0.8, 0.1), (1.16, 0.1), (1.16, 0.3), (2.3, 0.1)]
DAMPED = WHEELS[:3]
LIGHT = WHEELS[3]
# Dry friction of a wheel base, about 3.5 % of full force; resting hands as inertia and damping.
STICK = 350.0
RESTING = dict(stick=STICK, hand_mass=3.0, hand_damp=1.5)


QUICK = replace(QUIET, settle_s=0.0, rise_s=0.05)


def wheel_tuned(tuning, **kw):
    return replace(tuning, wheel=replace(tuning.wheel, **kw))


def released_in_bend(radius, ring_hz=1.5, friction=0.3, seconds=30.0, tuning=QUICK, gains=None, wobble=0.0,
                     **rig_kw):
    """On the lane's line in the bend, the assist engaged at once; ``rig_kw`` sets friction and hands (none: off).

    ``wobble`` starts the truck that many metres off its line.
    """
    rig = Rig(radius=radius, ring_hz=ring_hz, friction=friction, **{"stick": STICK, **rig_kw})
    rig.settle_on_lane()
    rig.e = wobble
    return loop(rig, seconds, tuning=tuning, gains=gains)


def gap(e):
    return TN.half_lane_m - (abs(e) + HALF_WIDTH)


def fixed(offset, seconds, assist, start=0.0, steer=None, route=None, gains=GAINS, **kw):
    """The truck held at an offset, the wheel still where ``steer`` says (default: where the assist wants it)."""
    route = route or make_route(None, length=2500.0)
    out = []
    for i in range(int(seconds / DT)):
        wheel_at = steer if steer is not None else (out[-1].target if out else 0.0)
        f = frame_at(start + i * DT, route, None, offset, 0.0, steer_right=-wheel_at, **kw)
        out.append(assist.update(f, gains, Wheel(ok=True, steer=wheel_at)))
    return out


@pytest.mark.parametrize("ring_hz, friction", DAMPED)
@pytest.mark.parametrize("radius", [800.0, 300.0, 200.0])
def test_hands_off_it_steers_the_bend_on_its_own(radius, ring_hz, friction):
    out = released_in_bend(radius, ring_hz, friction, wobble=0.5)
    after = [e for _, e, _, _ in out]
    assert min(gap(e) for e in after) > 0.0, "never leaves the lane"
    assert max(abs(e) for e in after[-int(5.0 / DT):]) < 0.7, "and settles in it, friction leaves a standing offset"
    assert not any(o.event == "override" for _, _, _, o in out), "a released wheel swinging is not a driver"


@pytest.mark.parametrize("lagged", [False, True])
@pytest.mark.parametrize("ring_hz, friction", WHEELS)
def test_with_resting_hands_and_friction_it_steers_the_bend(ring_hz, friction, lagged):
    out = released_in_bend(300.0, ring_hz, friction, wobble=0.5, lagged=lagged, **RESTING)
    assert min(gap(e) for _, e, _, _ in out) > 0.0
    assert max(abs(e) for t, e, _, _ in out if t > 20.0) < 0.4


def test_a_released_light_wheel_with_no_friction_rings_and_the_guard_cuts_it():
    out = released_in_bend(300.0, *LIGHT, wobble=0.5, stick=0.0)
    lane_steer = -1.0 / 300.0 / GAIN
    assert any(o.event == "ring" for _, _, _, o in out)
    assert max(abs(s - lane_steer) for _, _, s, _ in out) < 0.12, "the shake stays bounded"


@pytest.mark.parametrize("ring_hz, friction", DAMPED)
def test_a_drift_on_a_straight_is_caught_and_returned(ring_hz, friction):
    def kick(t, rig):
        if abs(t - 8.0) < 1e-9:
            rig.ed += 0.4

    out = loop(Rig(ring_hz=ring_hz, friction=friction, stick=STICK), 30.0, tuning=QUIET, kick=kick)
    es = [e for _, e, _, _ in out]
    assert max(es) < 0.6 and min(es) > -0.45, "friction leaves the truck hunting a little around the centre"
    assert max(abs(e) for e in es[-int(5.0 / DT):]) < 0.45


def test_an_offset_returns_to_the_centre_without_overshoot():
    def kick(t, rig):
        if abs(t - 8.0) < 1e-9:
            rig.e += 0.8

    out = loop(Rig(stick=STICK), 30.0, tuning=QUIET, kick=kick)
    es = [e for t, e, _, _ in out if t > 8.0]
    assert min(es) > -0.45 and max(abs(e) for e in es[-int(5.0 / DT):]) < 0.45


@pytest.mark.parametrize("scale", [0.85, 1.18])
def test_a_wrong_steering_gain_still_settles_on_the_lane(scale):
    out = released_in_bend(300.0, seconds=40.0, gains=Gains(FORCE, GAIN * scale), **RESTING)
    assert min(gap(e) for _, e, _, _ in out) > -0.1, "with friction a 15 % error can graze the line on the way in"
    late = [e for t, e, _, _ in out if t > 30.0]
    assert max(abs(e) for e in late) < 0.7, "the standing offset stays inside the lane"


def test_engaging_starts_at_the_wheel_and_never_yanks():
    assist = LaneAssist(QUIET)
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int(5.0 / DT)):
        f = frame_at(i * DT, route, None, 0.9, 0.0, steer_right=0.004)
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=-0.004)))
    first = next(o for o in out if o.servo is not SERVO_OFF)
    assert first.target == pytest.approx(-0.004, abs=TN.target_rate * DT + 1e-6)
    holds = [o.torque for o in out]
    assert max(abs(b - a) for a, b in zip(holds, holds[1:])) < 40.0


def test_the_servo_holds_the_target_moving_at_its_rate_on_top_of_the_game_force():
    out = fixed(0.9, 6.0, LaneAssist(QUIET))
    last = out[-1]
    kp, kd, ki = WheelActuator(TN.wheel).gains(1.0, SPRING)
    assert last.authority == 1.0 and last.torque == 0.0
    assert (last.servo.kp, last.servo.kd, last.servo.ki) == pytest.approx((kp, kd, ki))
    assert last.servo.cancel == 0.0 and last.servo.direction == 1
    moving = [o for o in out if o.servo.direction and abs(o.servo.rate) > 0.0]
    assert moving and all(abs(o.servo.rate) <= TN.target_rate + 1e-9 for o in out), "the walk's rate goes along"
    flipped = fixed(0.9, 6.0, LaneAssist(QUIET), gains=Gains(-FORCE, GAIN))[-1]
    assert flipped.servo.direction == -1


def test_the_wheel_feels_as_stiff_at_every_speed_the_game_spring_counted_in():
    slow = fixed(0.9, 6.0, LaneAssist(QUIET), speed_ms=0.8 * SPEED)[-1]
    fast = fixed(0.9, 6.0, LaneAssist(QUIET))[-1]
    assert slow.servo.kp + 0.64 * SPRING == pytest.approx(fast.servo.kp + SPRING), "servo plus game spring"
    assert slow.servo.kd == pytest.approx(fast.servo.kd)


def weaving(gains, seconds=20.0):
    """The truck swung across its lane every 4 s whatever the assist asks, the wheel on the target."""
    assist = LaneAssist(QUIET)
    route = make_route(None, length=2500.0)
    w = 2.0 * math.pi / 4.0
    target = 0.0
    for i in range(int(seconds / DT)):
        t = i * DT
        f = frame_at(t, route, None, 0.6 * math.sin(w * t), 0.6 * w / SPEED * math.cos(w * t), steer_right=-target)
        target = assist.update(f, gains, Wheel(ok=True, steer=target)).target
    return assist


def test_the_weave_guard_cuts_an_engaged_correction_but_not_the_drivers_own_steering():
    assert weaving(GAINS).gain < 1.0
    assert weaving(None).gain == 1.0, "unassisted, the flips are the driver's own"


def test_strength_scales_the_servo_and_keeps_its_cap():
    weak = fixed(0.9, 6.0, LaneAssist(wheel_tuned(QUIET, strength=0.5)))[-1]
    full = fixed(0.9, 6.0, LaneAssist(wheel_tuned(QUIET, strength=9.0)))[-1]
    assert weak.servo.cap == full.servo.cap == round(TN.wheel.servo_cap)
    assert full.servo.kp + SPRING == pytest.approx(STRENGTH_MAX / 0.5 * (weak.servo.kp + SPRING)), "the total scales"


def test_a_driver_holding_the_wheel_still_takes_over():
    assist = LaneAssist(QUIET)
    fixed(0.9, 5.0, assist)
    held = fixed(0.9, 1.5, assist, start=5.0, steer=-0.03)
    first = next(i for i, o in enumerate(held) if o.reason == "override")
    assert first * DT <= TN.override_s + 0.1
    assert held[first + int(TN.yield_s / DT) + 1].servo is SERVO_OFF


def test_a_wheel_on_the_move_is_not_yet_an_override():
    assist = LaneAssist(QUIET)
    fixed(0.9, 5.0, assist)
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int(0.5 / DT)):
        f = frame_at(5.0 + i * DT, route, None, 0.9, 0.0)
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=0.002 * i, rate=0.8)))
    assert all(o.reason == "assist" for o in out)


@pytest.mark.parametrize("rate, yields", [(0.0, True), (0.2, False)])
def test_a_still_wheel_with_the_servo_pinned_at_its_cap_counts_as_the_driver(rate, yields):
    assist = LaneAssist(QUIET)
    fixed(0.9, 5.0, assist)
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int(1.5 / DT)):
        f = frame_at(5.0 + i * DT, route, None, 0.9, 0.0)
        target = out[-1].target if out else 0.0
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=target + 0.005, rate=rate, push=TN.wheel.servo_cap)))
    assert any(o.event == "override" for o in out) == yields, "a wheel on its way is not held"


def test_a_servo_carrying_the_games_centring_through_a_bend_is_no_driver():
    assist = LaneAssist(QUIET)
    fixed(0.9, 5.0, assist)
    route = make_route(None, length=2500.0)
    cap = TN.wheel.servo_cap
    out = []
    for i in range(int(3.0 / DT)):
        f = frame_at(5.0 + i * DT, route, None, 0.9, 0.0)
        target = out[-1].target if out else 0.0
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=target + 0.005, push=-0.9 * cap, game=0.85 * cap)))
    assert not any(o.event == "override" for o in out), "drive 5: three bends handed back at the servo's cap"


@pytest.mark.parametrize("radius", [150.0, -150.0, 300.0])
def test_a_route_rebuilt_the_maps_way_steers_a_bend_as_smoothly_as_a_straight(radius):
    rig = Rig(radius=radius, ring_hz=1.16, friction=0.1)
    rig.settle_on_lane()
    out = loop(rig, 30.0, tuning=QUICK, rebuild=10.0)[int(10.0 / DT):]
    drift = [o.drift_ms for _, _, _, o in out]
    rates = sorted(abs(o.servo.rate) for _, _, _, o in out)
    jumps = sorted(abs(b - a) for a, b in zip(drift, drift[1:]))
    assert all(o.fitted for _, _, _, o in out)
    assert jumps[int(0.9 * len(jumps))] < 0.04, "the polyline read 0.29 m/s tick to tick at R150, as on drive 5"
    assert rates[int(0.9 * len(rates))] < 0.01, "the polyline kept the target's rate on its 0.04 limit"
    assert abs(sum(e for _, e, _, _ in out) / len(out)) < 0.1, "near its free zone; the polyline sat 0.11 m off"


def swinging(amplitude, seconds, assist, start=0.0, hz=2.0):
    """The wheel swinging through the target at ``hz``, the truck held on the lane centre."""
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int(seconds / DT)):
        t = start + i * DT
        target = out[-1].target if out else 0.0
        steer = target + amplitude * math.sin(2.0 * math.pi * hz * t)
        f = frame_at(t, route, None, 0.0, 0.0, steer_right=-steer)
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=steer)))
    return out


def test_a_wheel_ringing_through_the_target_has_its_stiffness_cut_then_given_back():
    assist = LaneAssist(QUIET)
    fixed(0.0, 5.0, assist)
    full = assist.ring_scale
    rung = swinging(0.008, 2.0, assist, start=5.0)
    assert full == 1.0 and any(o.event == "ring" for o in rung)
    assert min(o.servo.kp for o in rung) < 0.6 * WheelActuator(TN.wheel).gains(1.0, SPRING)[0]
    fixed(0.0, TN.wheel.ring_recover_s + 1.0, assist, start=7.0)
    assert assist.ring_scale == 1.0
    swinging(0.008, 2.0, assist, start=8.0 + TN.wheel.ring_recover_s)
    fixed(0.0, TN.wheel.ring_recover_s + 1.0, assist, start=10.0 + TN.wheel.ring_recover_s)
    assert assist.ring_scale < 1.0, "a wheel that rings again gets its stiffness back more slowly"


def test_a_big_swing_through_the_target_takes_the_stiffness_out_at_once():
    assist = LaneAssist(QUIET)
    fixed(0.0, 5.0, assist)
    swinging(0.04, 0.8, assist, start=5.0)
    assert assist.ring_scale < 0.05, "on its second swing, a quarter second after the first (Lukas, 2026-11-01)"


def test_hands_off_rumbles_then_lets_go_and_a_hand_brings_it_back():
    assist = LaneAssist()
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int((TN.hands_off_s + 8.0) / DT)):
        f = frame_at(i * DT, route, None, 0.2, 0.0)
        wheel_at = out[-1].target if out else 0.0
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=wheel_at)))
    warned = next(i for i, o in enumerate(out) if o.rumble)
    gone = next(i for i, o in enumerate(out) if o.event == "hands_off")
    assert (gone - warned) * DT == pytest.approx(TN.hands_off_s - TN.hands_warn_s, abs=0.2)
    assert out[-1].reason == "hands_off" and out[-1].servo is SERVO_OFF
    moved = assist.update(frame_at(len(out) * DT, route, None, 0.2, 0.0), GAINS, Wheel(ok=True, rate=0.1))
    assert moved.reason != "hands_off"


def test_a_hand_on_the_wheel_keeps_it_on():
    rng = random.Random(4)
    assist = LaneAssist()
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int(40.0 / DT)):
        f = frame_at(i * DT, route, None, 0.2, 0.0)
        wheel_at = (out[-1].target if out else 0.0) + rng.gauss(0.0, 0.002)
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=wheel_at)))
    assert not any(o.event == "hands_off" or o.rumble for o in out)


@pytest.mark.parametrize("change, reason", [
    (dict(speed_ms=12.0), "speed"),
    (dict(aeb_active=True), "aeb"),
    (dict(brake=0.8), "brake"),
    (dict(gear=-1), "reverse"),
    (dict(paused=True), "paused"),
    (dict(age_s=1.0), "stale"),
    (dict(route=None), "no_map"),
])
def test_every_gate_fades_it_out(change, reason):
    assist = LaneAssist(QUIET)
    fixed(0.9, 5.0, assist)
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int((TN.fall_s + 0.5) / DT)):
        f = frame_at(5.0 + i * DT, route, None, 0.9, 0.0)
        out.append(assist.update(replace(f, **change), GAINS, Wheel(ok=True, steer=0.0)))
    assert out[-1].reason == reason and out[-1].servo is SERVO_OFF and out[-1].torque == 0.0
    assert all(b.authority <= a.authority for a, b in zip(out, out[1:]))


def test_truckersmp_is_off_unless_allowed():
    route = make_route(None, length=2500.0)
    off, on = LaneAssist(QUIET), LaneAssist(replace(QUIET, allow_tmp=True))
    for i in range(int(5.0 / DT)):
        f = frame_at(i * DT, route, None, 0.9, 0.0)
        a, b = off.update(f, GAINS, Wheel(ok=True), tmp=True), on.update(f, GAINS, Wheel(ok=True), tmp=True)
    assert a.reason == "tmp" and a.servo is SERVO_OFF
    assert b.reason == "assist" and b.servo is not SERVO_OFF


def test_no_calibration_or_no_servo_plugin_means_nothing_sent():
    route = make_route(None, length=2500.0)
    learning, old = LaneAssist(QUIET), LaneAssist(QUIET)
    for i in range(int(5.0 / DT)):
        f = frame_at(i * DT, route, None, 0.9, 0.0)
        a, b = learning.update(f, None, Wheel(ok=True)), old.update(f, GAINS, Wheel())
    assert (a.reason, a.servo, a.torque) == ("calibrating", SERVO_OFF, 0.0)
    assert (b.reason, b.servo, b.torque) == ("plugin", SERVO_OFF, 0.0)


@pytest.mark.parametrize("extra, reason", [
    (dict(branch_s=60.0), "branch"),
    (dict(merge_s=60.0), "merge"),
    (dict(kind="prefab_curve"), "route_kind"),
    (dict(reach_m=60.0), "short_route"),
])
def test_forks_merges_and_junction_lanes_hand_back(extra, reason):
    out = fixed(0.9, 5.0, LaneAssist(QUIET), route=make_route(None, length=2500.0, **extra))
    assert out[-1].reason == reason and out[-1].servo is SERVO_OFF
    kept = fixed(0.9, 5.0, LaneAssist(replace(QUIET, follow_all=True)), route=make_route(None, length=2500.0, **extra))
    assert kept[-1].reason == "assist" and kept[-1].servo is not SERVO_OFF, "the follow-everywhere test keeps it"


@pytest.mark.parametrize("radius, reason", [(200.0, "assist"), (130.0, "bend"), (90.0, "bend_sharp")])
def test_following_everywhere_limits_bends_by_lateral_acceleration(radius, reason):
    route = make_route(radius, length=400.0)
    assist = LaneAssist(replace(QUIET, follow_all=True))
    out = []
    for i in range(80):
        wheel_at = out[-1].target if out else 0.0
        f = frame_at(i * DT, route, radius, 0.0, 0.0, steer_right=-wheel_at)
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=wheel_at)))
    assert out[-1].reason == reason


def test_following_everywhere_a_lane_switch_keeps_steering_on_the_bend_alone():
    plain, follow = LaneAssist(QUIET), LaneAssist(replace(QUIET, follow_all=True))
    for assist in (plain, follow):
        fixed(0.4, 5.0, assist)
    a = fixed(2.4, 1.0, plain, start=5.0)
    b = fixed(2.4, 1.0, follow, start=5.0)
    assert a[-1].reason == "lane_settle" and a[-1].authority < 1.0
    assert all(o.reason == "assist" and o.authority == 1.0 and o.accel_fb_ms2 == 0.0 for o in b)


def test_a_router_blip_shorter_than_the_confirm_does_not_touch_it():
    plain, blipped = LaneAssist(QUIET), LaneAssist(QUIET)
    for assist in (plain, blipped):
        fixed(0.9, 5.0, assist)
    blip = make_route(None, length=2500.0, branch_s=START_S + SPEED * 5.0 + 30.0)
    short = TN.soft_confirm_s - 0.1
    a = fixed(0.9, short, plain, start=5.0) + fixed(0.9, 1.0, plain, start=5.0 + short)
    b = fixed(0.9, short, blipped, start=5.0, route=blip) + fixed(0.9, 1.0, blipped, start=5.0 + short)
    assert all(o.reason == "assist" for o in b)
    assert max(abs(x.torque - y.torque) for x, y in zip(a, b)) < 1.0


def test_a_blinker_hands_back_and_it_waits_after_the_lamp_goes_off():
    assist = LaneAssist(QUIET)
    fixed(0.9, 5.0, assist)
    lamp = fixed(0.9, 1.0, assist, start=5.0, blinker_left=True)
    assert lamp[int(TN.yield_s / DT) + 1].servo is SERVO_OFF
    quiet = fixed(0.9, TN.blinker_hold_s + TN.rearm_s - 0.1, assist, start=6.0)
    assert all(o.servo is SERVO_OFF for o in quiet)
    back = fixed(0.9, 1.0, assist, start=6.0 + TN.blinker_hold_s + TN.rearm_s - 0.1)
    assert back[-1].servo is not SERVO_OFF


def test_the_bend_ceiling_is_smooth_and_hands_back_tight_bends():
    ks = [i / 20000.0 for i in range(0, 260)]
    ceilings = [bend_ceiling(k, TN) for k in ks]
    assert bend_ceiling(1.0 / 300.0, TN) == 1.0 and bend_ceiling(TN.kappa_max, TN) == 0.0
    assert all(a >= b for a, b in zip(ceilings, ceilings[1:]))
    assert max(abs(b - a) for a, b in zip(ceilings, ceilings[1:])) < 0.05
    route = make_route(90.0, length=400.0)
    assist = LaneAssist(QUIET)
    out = [assist.update(frame_at(i * DT, route, 90.0, 0.0, 0.0), GAINS, Wheel(ok=True)) for i in range(60)]
    assert out[-1].reason == "bend_sharp" and out[-1].servo is SERVO_OFF


def test_a_swerve_hands_back():
    assist = LaneAssist(QUIET)
    fixed(0.5, 5.0, assist)
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int(1.0 / DT)):
        f = frame_at(5.0 + i * DT, route, None, 0.5, 0.0, steer_right=0.2 * math.sin(i * DT * 6.0))
        out.append(assist.update(f, GAINS, Wheel(ok=True)))
    assert any(o.reason == "evade" for o in out)


ALWAYS = replace(TN, always_on=True)


def test_always_on_never_hands_back_to_the_driver():
    assist = LaneAssist(ALWAYS)
    fixed(0.5, 5.0, assist)
    lamp = fixed(0.5, 1.0, assist, start=5.0, blinker_left=True)
    held = fixed(0.5, 2.0, assist, start=6.0, steer=-0.03)
    route = make_route(None, length=2500.0)
    swerve = [assist.update(frame_at(8.0 + i * DT, route, None, 0.5, 0.0, steer_right=0.2 * math.sin(i * DT * 6.0)),
                            GAINS, Wheel(ok=True, push=TN.wheel.servo_cap)) for i in range(int(1.0 / DT))]
    still = fixed(0.2, TN.hands_off_s + 5.0, assist, start=9.0)
    for out in (lamp, held, swerve, still):
        assert all(o.reason == "assist" and o.authority == 1.0 and not o.event and not o.rumble for o in out)


@pytest.mark.parametrize("radius", [90.0, -90.0])
def test_always_on_keeps_full_authority_in_a_tight_bend(radius):
    route = make_route(radius, length=400.0)
    assist = LaneAssist(ALWAYS)
    out = []
    for i in range(80):
        wheel_at = out[-1].target if out else 0.0
        f = frame_at(i * DT, route, radius, 0.0, 0.0, steer_right=-wheel_at)
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=wheel_at)))
    assert out[-1].reason == "assist" and out[-1].authority == 1.0, "7 m/s^2 handed back in follow-everywhere"


@pytest.mark.parametrize("radius", [150.0, -150.0])
def test_a_bend_is_entered_and_left_on_the_lane_the_feedforward_leading_by_the_servos_lag(radius):
    spring = 68000.0
    rig = Rig(spring=spring, ring_hz=math.sqrt(spring / 650.0) / (2.0 * math.pi), friction=0.1, yaw_lag_s=0.05,
              track=bend_track(radius), hand_mass=3.0, hand_damp=1.5)
    rig.settle_on_lane()
    out = loop(rig, 34.0, tuning=replace(QUICK, always_on=True), rebuild=10.0)
    outside = [-math.copysign(1.0, radius) * e for t, e, _, _ in out if START_S + SPEED * t >= 300.0]
    assert max(outside) < 0.1, "the 0.4 s preview ran 0.34 m wide after the bend, as on drive 6"
    assert min(outside) > -0.1, "and cut 0.30 m inside on the way in"
    assert all(0.1 < o.preview_s < 0.25 for *_, o in out[int(1.0 / DT):])


def game_rig(**kw):
    """The game's spring at 90 km/h on the R3's estimated rim, the truck's path a tick behind its wheel."""
    spring = 68000.0
    return Rig(spring=spring, ring_hz=math.sqrt(spring / 650.0) / (2.0 * math.pi), friction=0.1, yaw_lag_s=0.05,
               hand_mass=3.0, hand_damp=1.5, **kw)


@pytest.mark.parametrize("k1, k2, ramp", [(1 / 150.0, 1 / 240.0, 12.0), (1 / 120.0, 1 / 180.0, 8.0)])
def test_a_prefab_kink_from_right_to_left_stays_in_the_lane(k1, k2, ramp):
    rig = game_rig(track=kink_track(k1, k2, ramp))
    rig.settle_on_lane()
    out = loop(rig, 30.0, tuning=replace(QUICK, always_on=True), rebuild=10.0)
    assert max(abs(e) for _, e, _, _ in out) < 0.8, "the 0.04 lock/s target walk left it 3.4 m out, as on drive 7"


def test_a_fast_drift_toward_the_line_is_stopped_harder():
    def peak(tuning):
        rig = game_rig()
        rig.settle_on_lane()

        def kick(t, r):
            if abs(t - 5.0) < 1e-6:
                r.ed = 2.0
        return max(e for _, e, _, _ in loop(rig, 15.0, tuning=tuning, kick=kick, rebuild=10.0))
    plain = peak(replace(QUICK, always_on=True, edge_max_ms2=0.0))
    assert peak(replace(QUICK, always_on=True)) < plain - 0.3


class _TwoLanes(Track):
    """A straight road with a second lane 4.5 m to the right; the route is the lane the truck's centre is in."""

    def __init__(self) -> None:
        super().__init__([(3000.0, 0.0)])
        self.rig: Rig | None = None

    def route(self, s_ego, step=10.0, reach=200.0):
        r = super().route(s_ego, step, reach)
        if self.rig is None or self.rig.e <= 2.25:
            return r
        return SimpleNamespace(**{**vars(r), "pts": tuple((x + 4.5, z) for x, z in r.pts)})


def test_always_on_pushes_the_truck_back_into_the_lane_it_left_when_the_route_moves_lanes():
    rig = game_rig(track=_TwoLanes())
    rig.track.rig = rig
    rig.settle_on_lane()

    def kick(t, r):
        r.hands, r.hands_at = (1e6, -0.012) if 5.0 <= t < 7.2 else (0.0, 0.0)
    out = loop(rig, 20.0, tuning=replace(QUICK, always_on=True), kick=kick, rebuild=10.0)
    assert max(e for _, e, _, _ in out) > 2.5, "the hand took it over the line, so the route moved lanes"
    assert any(o.event == "carry" for *_, o in out)
    after = [(t, e, o) for t, e, _, o in out if t >= 7.2]
    assert all(o.reason == "assist" for _, _, o in after), "no blind settle in the next lane"
    assert max(t for t, e, _ in after if e > 2.25) < 10.0, "back over the line within 3 s of letting go"
    assert max(abs(e) for t, e, _ in after if t > 13.0) < 0.3, "back on the lane it left, not in the next one"


@pytest.mark.parametrize("lag, overshoot", [(0.2, 0.1), (0.3, 0.15)])
def test_a_sideways_kick_settles_with_little_overshoot_on_a_slower_truck(lag, overshoot):
    rig = game_rig()
    rig.yaw_lag_s = lag
    rig.settle_on_lane()

    def kick(t, r):
        if abs(t - 5.0) < 1e-6:
            r.ed += 1.2
    out = loop(rig, 20.0, tuning=replace(QUICK, always_on=True), kick=kick, rebuild=10.0)
    after = [(t, e) for t, e, _, _ in out if t >= 5.0]
    assert -min(e for _, e in after) < overshoot, "a heading filter of 0.15 s swung it 0.39 and 0.58 m past"
    assert max(t for t, e in after if abs(e) > 0.15) < 13.0
    assert not any(o.event == "weave" for t, _, _, o in out if t >= 5.0), "no drift prediction: 0.36 m past, cut"


def test_always_on_carries_no_lane_across_a_gate_or_a_jump_past_a_lane():
    gated = LaneAssist(ALWAYS)
    fixed(0.4, 5.0, gated)
    fixed(0.4, 1.0, gated, start=5.0, speed_ms=12.0)
    back = fixed(-4.1, 1.0, gated, start=6.0)
    assert not any(o.event == "carry" for o in back) and back[-1].offset_m == pytest.approx(-4.1, abs=0.05)
    far = LaneAssist(ALWAYS)
    fixed(0.4, 5.0, far)
    out = fixed(-6.1, 1.0, far, start=5.0)
    assert not any(o.event == "carry" for o in out), "a parallel road or a re-plan is no lane crossing"


def test_a_lane_that_moves_faster_than_a_truck_holds_the_wheel_and_starts_the_lane_over():
    assist = LaneAssist(ALWAYS)
    before = fixed(0.0, 5.0, assist)[-1].target
    route = make_route(None, length=2500.0)
    out = []
    for i in range(int(3.0 / DT)):
        t = 5.0 + i * DT
        rate = -19.0 if t < 5.1 else (3.8 if t < 5.6 else 0.0)
        e = -19.0 * min(t - 5.0, 0.1) + 3.8 * max(0.0, min(t - 5.1, 0.5))
        target = out[-1].target if out else before
        f = frame_at(t, route, None, e, rate / SPEED, steer_right=-target)
        out.append(assist.update(f, GAINS, Wheel(ok=True, steer=target)))
    assert any(o.event == "geometry" for o in out)
    held = [o.target for o in out[: int(0.9 / DT)]]
    assert max(abs(x - before) for x in held) < 0.003, "drive 10: the target chased a roundabout's lanes"
    assert assist.ring_scale == 1.0 and out[-1].reason == "assist"


def test_a_cut_weave_gain_still_lets_a_big_correction_through():
    assist = LaneAssist(QUIET)
    assist._gain = 0.5
    assist._auth = 1.0
    assert abs(assist._lateral(0.05, 5.0, 2.0)) > 0.95 * TN.lat_max_ms2, "only the slope is softened"


def test_the_learned_centre_follows_the_driver_slowly_and_stays_bounded():
    assist = LaneAssist(QUIET)
    fixed(0.2, 60.0, assist)
    assert abs(assist.bias_m - 0.2) < 0.02
    fixed(0.9, 30.0, assist, start=60.0)
    assert assist.bias_m == pytest.approx(TN.bias_max_m)


def test_a_teleport_starts_over():
    assist = LaneAssist(QUIET)
    fixed(0.9, 6.0, assist)
    f = replace(frame_at(6.0, make_route(None, length=2500.0), None, 0.9, 0.0), x=5000.0)
    out = assist.update(f, GAINS, Wheel(ok=True))
    assert out.servo is SERVO_OFF and out.torque == 0.0


def test_a_wanted_path_moves_the_line_the_lane_follower_steers_to():
    lane, shifted = LaneAssist(QUIET), LaneAssist(QUIET)
    route = make_route(None, length=2500.0)
    wheel_at = {id(lane): 0.0, id(shifted): 0.0}
    for i in range(int(6.0 / DT)):
        f = frame_at(i * DT, route, None, 0.0, 0.0)
        a = lane.update(f, GAINS, Wheel(ok=True, steer=wheel_at[id(lane)]))
        b = shifted.update(f, GAINS, Wheel(ok=True, steer=wheel_at[id(shifted)]),
                           path=WantedPath(offset_m=-0.8, source="test"))
        wheel_at[id(lane)], wheel_at[id(shifted)] = a.target, b.target
    assert abs(a.accel_fb_ms2) < 1e-6 and a.source == "lane"
    assert b.accel_fb_ms2 < -0.2 and b.target > 0.0 and b.source == "test", "on the lane centre it steers left"
