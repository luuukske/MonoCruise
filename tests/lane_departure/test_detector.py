"""Lane departure detector against a synthetic lane: drifts, lane changes, bends and gates."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.lane_departure.detector import (
    LEVEL_CLOSE,
    LEVEL_WARN,
    SIDE_LEFT,
    SIDE_RIGHT,
    Frame,
    LaneDetector,
    Tuning,
    bend_allowance,
    bend_peaks,
    body_from_geometry,
    lane_kappa_right,
    project,
    smooth_heading,
)
from core.radar.ego_geometry import REF_PATH_ORIGIN_M, EgoGeometry

HZ = 20.0
DT = 1.0 / HZ
SPEED = 25.0
START_S = 40.0
DRIFT_START = 3.0
HALF_WIDTH = 1.265
FRONT = 3.333
REAR = 3.333
TN = Tuning()
LINE = -TN.warn_overlap_m


def make_route(radius=None, length=700.0, step=5.0, **extra):
    """Lane centre from the origin. ``radius`` > 0 bends right, < 0 left, None is straight north."""
    pts = []
    for i in range(int(length / step) + 1):
        s = i * step
        pts.append((_arc_x(s, radius), _arc_z(s, radius)))
    base = dict(pts=tuple(pts), kind="road", divergence=(), branch_s=None, merge_s=None, reach_m=length)
    base.update(extra)
    return SimpleNamespace(**base)


def _arc_x(s, radius):
    return 0.0 if radius is None else radius * (1.0 - math.cos(s / radius))


def _arc_z(s, radius):
    return -s if radius is None else -radius * math.sin(s / radius)


def frame_at(t, route, radius, lateral, psi, **kw):
    s = START_S + SPEED * t
    phi = 0.0 if radius is None else s / radius
    base = dict(
        t=t, age_s=0.0,
        x=_arc_x(s, radius) + lateral * math.cos(phi),
        z=_arc_z(s, radius) + lateral * math.sin(phi),
        yaw_rad=-(phi + psi), speed_ms=SPEED, steer_right=0.0, gear=1, brake=0.0,
        blinker_left=False, blinker_right=False, hazards=False, paused=False, aeb_active=False,
        half_width_m=HALF_WIDTH, front_m=FRONT, rear_m=REAR, path_m=0.0, route=route,
    )
    base.update(kw)
    return Frame(**base)


def drive(plan, seconds=14.0, radius=None, route=None, detector=None, **fixed):
    """Run ``plan(t) -> (lateral, psi, extras)`` at 20 Hz and return every decision."""
    route = route or make_route(radius)
    detector = detector or LaneDetector()
    out = []
    for i in range(int(seconds * HZ)):
        t = i * DT
        lateral, psi, extras = plan(t)
        out.append(detector.update(frame_at(t, route, radius, lateral, psi, **{**fixed, **extras})))
    return out


RAMP_S = 1.0


def drift(rate, side=1.0, start=DRIFT_START, **extras):
    """Lateral speed toward a side (+1 right, -1 left) that builds over ``RAMP_S`` from ``start``."""
    psi = side * math.atan2(rate, SPEED)

    def plan(t):
        tau = t - start
        if tau < 0.0:
            return 0.0, 0.0, {}
        lateral = rate * tau * tau / (2.0 * RAMP_S) if tau < RAMP_S else rate * (tau - 0.5 * RAMP_S)
        return side * lateral, psi * min(1.0, tau / RAMP_S), extras(t) if callable(extras) else extras

    return plan


def right_gap(r, psi):
    """Gap of the outer right corner, front or rear, on a straight lane."""
    return 2.25 - (r + HALF_WIDTH + max(FRONT * math.sin(psi), -REAR * math.sin(psi)))


def first_active(decisions):
    return next((i * DT for i, d in enumerate(decisions) if d.active), None)


def warns(decisions):
    return [d for d in decisions if d.started and d.level == LEVEL_WARN]


def cues(decisions):
    return [d for d in decisions if d.started and d.level == LEVEL_CLOSE]


def warn_seconds(decisions):
    return sum(d.active and d.level == LEVEL_WARN for d in decisions) * DT


def test_a_centred_truck_never_warns():
    decisions = drive(lambda t: (0.0, 0.0, {}))
    assert not any(d.active for d in decisions)
    assert decisions[-1].reason == "clear"


@pytest.mark.parametrize("side,expected", [(1.0, SIDE_RIGHT), (-1.0, SIDE_LEFT)])
def test_a_slow_drift_warns_before_the_tyre_reaches_the_line(side, expected):
    decisions = drive(drift(0.4, side))
    started = warns(decisions)
    assert len(started) == 1 and started[0].side == expected
    assert 5.4 <= decisions.index(started[0]) * DT <= 5.8
    assert LINE - 0.05 < started[0].gap_m <= LINE
    assert started[0].gap_m > 0.0


def test_the_warning_position_does_not_depend_on_lateral_speed():
    slow = warns(drive(drift(0.3)))
    fast = warns(drive(drift(1.2)))
    assert LINE - 0.03 < slow[0].gap_m <= LINE
    assert LINE - 0.07 < fast[0].gap_m <= LINE


def test_the_warning_stops_after_its_maximum_and_does_not_restart_while_still_out():
    decisions = drive(drift(0.4))
    assert 3.5 <= warn_seconds(decisions) <= 4.0 + 2 * DT
    assert len(warns(decisions)) == 1


def test_the_rear_corner_counts_when_angled_away_from_the_edge():
    """Pointing away from the right edge the rear corner is the outer one; moving inward, it only reports."""
    decisions = drive(lambda t: (1.0, -0.05, {}))
    assert decisions[-1].gap_m == pytest.approx(right_gap(1.0, -0.05), abs=0.01)
    assert decisions[-1].gap_m < right_gap(1.0, 0.0) - 0.1
    assert decisions[-1].near_miss == "returning" and not any(d.active for d in decisions)


def test_the_body_comes_from_the_sdk_geometry():
    geometry = EgoGeometry(half_width_m=1.3, front_m=4.1, rear_m=3.2, path_origin_m=-2.0)
    assert body_from_geometry(geometry) == (1.3, 4.1, 3.2, -2.0)
    assert body_from_geometry(None) == (HALF_WIDTH, FRONT, REAR, REF_PATH_ORIGIN_M)
    wide = drive(drift(0.4), half_width_m=1.5)
    narrow = drive(drift(0.4))
    assert warns(wide)[0].gap_m == pytest.approx(warns(narrow)[0].gap_m, abs=0.05)
    assert wide.index(warns(wide)[0]) < narrow.index(warns(narrow)[0]) - 8


def steer_back_plan(turn_at, rate=0.4):
    """Drift right at ``rate``, steer back at the same rate past ``turn_at``, straighten near the centre."""
    state = {"r": 0.0, "back": False, "gap": []}

    def plan(t):
        psi = 0.0
        if t >= DRIFT_START:
            state["back"] = state["back"] or state["r"] >= turn_at
            if state["back"] and state["r"] <= 0.3:
                psi = 0.0
            else:
                ramp = min(1.0, (t - DRIFT_START) / RAMP_S)
                psi = (-1.0 if state["back"] else ramp) * math.atan2(rate, SPEED)
            state["r"] += SPEED * math.sin(psi) * DT
        state["gap"].append(right_gap(state["r"], psi))
        return state["r"], psi, {}

    plan.state = state
    return plan


def test_the_warning_goes_on_across_the_line_and_ends_once_the_truck_steers_back():
    plan = steer_back_plan(turn_at=1.2)
    decisions = drive(plan, seconds=12.0)
    gaps = plan.state["gap"]
    active = [i for i, d in enumerate(decisions) if d.active and d.level == LEVEL_WARN]
    assert active and any(gaps[i] < -0.1 for i in active)
    turned = next(i for i, g in enumerate(gaps) if i > active[0] and g < -0.1)
    assert turned <= active[-1] <= turned + int(1.0 * HZ)
    assert not any(d.active for d in decisions[active[-1] + 1:])


def test_a_correction_before_the_warning_line_gives_no_warning_only_the_cue():
    plan = steer_back_plan(turn_at=0.68, rate=0.5)
    decisions = drive(plan, seconds=12.0)
    assert not warns(decisions) and cues(decisions)
    assert LINE < min(plan.state["gap"]) < LINE + 0.2


def test_a_second_drift_warns_again_without_recentring():
    """Warn, come back just inside the warning line, hold briefly, drift out again: a second warning."""
    a = math.atan2(0.4, SPEED)
    phases = [(DRIFT_START, 0.0), (5.25, a), (6.45, -a), (7.625, 0.0), (99.0, a)]
    state = {"r": 0.0, "min_after": math.inf}

    def plan(t):
        psi = next(p for end, p in phases if t < end)
        state["r"] += SPEED * math.sin(psi) * DT
        if t > 5.25:
            state["min_after"] = min(state["min_after"], state["r"])
        return state["r"], psi, {}

    decisions = drive(plan, seconds=10.0)
    started = [i * DT for i, d in enumerate(decisions) if d.started and d.level == LEVEL_WARN]
    assert len(started) == 2 and started[1] > 7.6
    assert all(d.side == SIDE_RIGHT for d in warns(decisions))
    assert state["min_after"] > 0.3


def test_the_rumble_ends_once_the_truck_is_far_enough_out():
    decisions = drive(drift(0.8))
    active = [i for i, d in enumerate(decisions) if d.active and d.level == LEVEL_WARN]
    assert active and decisions[active[-1]].gap_m > -1.7
    assert (active[-1] - active[0]) * DT < 3.5
    assert not any(d.active for d in decisions[active[-1] + 1:])


def test_straddling_the_line_is_cut_off_after_four_seconds():
    decisions = drive(lambda t: (1.3, 0.0, {}), seconds=12.0)
    assert 3.5 <= warn_seconds(decisions) <= 4.0 + 2 * DT
    assert len(warns(decisions)) == 1 and not cues(decisions)


def test_a_fork_ahead_does_not_restart_the_lane_settle():
    plain, fork = make_route(), make_route(branch_s=70.0)
    detector = LaneDetector()
    out = []
    for i in range(int(8 * HZ)):
        t = i * DT
        out.append(detector.update(frame_at(t, fork if 4.0 <= t < 5.0 else plain, None, 0.0, 0.0)))
    assert out[int(4.5 * HZ)].reason == "branch"
    assert out[int(5.2 * HZ)].reason == "clear"


def test_leaving_an_unusable_route_relearns_the_lane():
    plain, other = make_route(), make_route(kind="ramp")
    detector = LaneDetector()
    out = []
    for i in range(int(8 * HZ)):
        t = i * DT
        out.append(detector.update(frame_at(t, other if 4.0 <= t < 5.0 else plain, None, 0.0, 0.0)))
    assert out[int((5.0 + TN.steady_s - 0.2) * HZ)].reason == "lane_settle"
    assert out[int((5.0 + TN.steady_s + 0.2) * HZ)].reason == "clear"


def test_a_blinker_on_the_departure_side_mutes_the_warning():
    decisions = drive(drift(0.4, blinker_right=True))
    assert not any(d.active for d in decisions)
    assert any(d.near_miss == "blinker" for d in decisions)


def test_test_mode_warns_through_the_intent_rules():
    detector = LaneDetector(Tuning(test_mode=True))
    assert any(d.active for d in drive(drift(0.4, blinker_right=True), detector=detector))
    detector = LaneDetector(Tuning(test_mode=True))
    assert any(d.active for d in drive(drift(1.0), detector=detector))
    assert not any(d.active for d in drive(drift(0.4), detector=LaneDetector(Tuning(test_mode=True)), paused=True))


def test_a_blinker_on_the_other_side_does_not():
    decisions = drive(drift(0.4, blinker_left=True))
    assert any(d.active for d in decisions)


def test_a_fast_drift_with_a_still_wheel_is_an_emergency_not_a_lane_change():
    decisions = drive(drift(1.3))
    assert any(d.active for d in decisions)


def test_a_fast_move_with_a_clear_steering_action_toward_the_edge_is_deliberate():
    def plan(t):
        lateral, psi, _ = drift(1.3)(t)
        return lateral, psi, {"steer_right": 0.3 * max(0.0, t - DRIFT_START + 0.5)}

    assert not any(d.active for d in drive(plan))


@pytest.mark.parametrize("rate", [0.008, 0.02])
def test_smooth_slow_steering_toward_the_edge_still_warns(rate):
    def plan(t):
        lateral, psi, _ = drift(0.4)(t)
        return lateral, psi, {"steer_right": rate * max(0.0, t - DRIFT_START)}

    assert any(d.active for d in drive(plan))


def test_gentle_steering_away_from_the_edge_does_not_mute_it():
    def plan(t):
        lateral, psi, _ = drift(0.4)(t)
        return lateral, psi, {"steer_right": -0.03 * max(0.0, t - DRIFT_START)}

    assert any(d.active for d in drive(plan))


def no_reaction(radius):
    """A right bend with the wheel held straight: the truck runs out toward the left, the outside."""
    kappa = 1.0 / radius

    def plan(t):
        if t < DRIFT_START:
            return 0.0, 0.0, {}
        tau = t - DRIFT_START
        return -0.5 * SPEED * SPEED * kappa * tau * tau, -kappa * SPEED * tau, {}

    return plan


no_reaction_on_a_right_bend = no_reaction(300.0)


def test_no_reaction_on_a_slight_bend_cues_early_and_warns_at_the_outside_line():
    decisions = drive(no_reaction(700.0), radius=700.0)
    cue, warn = cues(decisions)[0], warns(decisions)[0]
    assert cue.side == warn.side == SIDE_LEFT
    assert decisions.index(warn) - decisions.index(cue) >= 4
    assert cue.gap_m <= LINE + TN.approach_range_m + 0.05
    assert LINE - 0.2 < warn.gap_m <= LINE and warn.allow_m == TN.warn_overlap_m


def test_no_reaction_in_a_tight_corner_cues_before_the_outside_line_and_warns_at_it():
    decisions = drive(no_reaction_on_a_right_bend, radius=300.0)
    cue, warn = cues(decisions)[0], warns(decisions)[0]
    assert cue.side == warn.side == SIDE_LEFT and decisions.index(cue) < decisions.index(warn)
    assert LINE - 0.2 < warn.gap_m <= LINE and warn.allow_m == TN.warn_overlap_m
    assert decisions.index(warn) * DT < DRIFT_START + 1.2


def swerving(amplitude=0.4, hz=0.8):
    """Hard wheel pulses left and right from the start of the drift."""

    def extras(t):
        return {"steer_right": amplitude * math.sin(2.0 * math.pi * hz * max(0.0, t - DRIFT_START))}

    return extras


def test_hard_steering_in_either_direction_is_ignored():
    def plan(t):
        lateral, psi, _ = drift(0.5)(t)
        return lateral, psi, swerving()(t)

    decisions = drive(plan)
    assert not any(d.active for d in decisions)
    assert any(d.near_miss in {"evade", "deliberate"} for d in decisions)


def test_a_hard_move_of_the_wheel_mutes_both_sides_not_only_the_one_it_turned_toward():
    """A hard pulse to the left, then a quiet wheel while the truck drifts right: the drift is muted."""

    def plan(t):
        lateral, psi, _ = drift(0.5)(t)
        pulse = -0.3 * min(1.0, max(0.0, (t - 3.2) / 0.3))
        return lateral, psi, {"steer_right": pulse}

    decisions = drive(plan, seconds=9.0)
    assert not any(d.active for d in decisions)
    assert any(d.near_miss == "evade" for d in decisions)
    soft = drive(lambda t: (*drift(0.5)(t)[:2], {"steer_right": -0.04 * min(1.0, max(0.0, (t - 3.2) / 0.3))}))
    assert warns(soft)


def test_the_same_drift_with_a_still_wheel_still_warns():
    assert warns(drive(drift(0.5)))


def hard_wheel(sign, rate=0.6):
    def extras(t):
        return {"steer_right": sign * rate * max(0.0, t - DRIFT_START)}

    return extras


def test_steering_with_a_tight_bend_is_not_intent_so_a_runout_still_warns():
    """Right bend R 300 m, the driver turns hard into it and still runs wide to the left."""

    def plan(t):
        lateral, psi, _ = no_reaction_on_a_right_bend(t)
        return lateral, psi, hard_wheel(1.0, 0.5)(t)

    decisions = drive(plan, radius=300.0)
    assert warns(decisions) and warns(decisions)[0].side == SIDE_LEFT
    assert not any(d.near_miss in {"evade", "deliberate"} for d in decisions)


def test_hard_steering_against_the_bend_toward_the_outside_edge_is_muted():
    def plan(t):
        lateral, psi, _ = no_reaction_on_a_right_bend(t)
        return lateral, psi, hard_wheel(-1.0, 0.5)(t)

    decisions = drive(plan, radius=300.0)
    assert not any(d.active for d in decisions)
    assert any(d.near_miss in {"evade", "deliberate"} for d in decisions)


@pytest.mark.parametrize("radius,warns_expected", [(800.0, False), (300.0, True)])
def test_only_a_tight_bend_exempts_steering_toward_the_inside(radius, warns_expected):
    def plan(t):
        lateral, psi, _ = drift(0.5, 1.0)(t)
        return lateral, psi, hard_wheel(1.0, 0.3)(t)

    assert bool(warns(drive(plan, radius=radius))) == warns_expected


def wheel_toward_right(rate):
    return lambda t: {"steer_right": rate * max(0.0, t - DRIFT_START)}


def test_a_gentle_lane_change_without_a_blinker_is_clear_intent():
    """The wheel moves toward the edge and the heading follows it: not a drift."""

    def plan(t):
        lateral, psi, _ = drift(0.8)(t)
        return lateral, psi, wheel_toward_right(0.02)(t)

    decisions = drive(plan)
    assert not any(d.active for d in decisions)
    assert any(d.near_miss == "intent" for d in decisions)


def test_the_same_motion_with_a_still_wheel_is_a_drift_and_warns():
    assert warns(drive(drift(0.8)))


def test_a_slow_creep_of_the_wheel_toward_the_edge_is_not_intent():
    def plan(t):
        lateral, psi, _ = drift(0.5)(t)
        return lateral, psi, wheel_toward_right(0.02)(t)

    assert warns(drive(plan))


def test_a_slight_drift_into_the_oncoming_side_of_a_gentle_left_bend_warns():
    """Following the curve with a little more wheel toward the inside than the road asks, sliding slowly across."""

    def plan(t):
        lateral, psi, _ = drift(0.45, -1.0)(t)
        return lateral, psi, {"steer_right": -0.01 * max(0.0, t - DRIFT_START)}

    decisions = drive(plan, radius=-600.0)
    assert warns(decisions) and warns(decisions)[0].side == SIDE_LEFT
    assert not any(d.near_miss == "intent" for d in decisions)


def test_steering_toward_the_edge_is_deliberate_even_when_slow():
    def plan(t):
        lateral, psi, _ = drift(0.3)(t)
        return lateral, psi, {"steer_right": 0.25 * max(0.0, t - DRIFT_START)}

    assert not any(d.active for d in drive(plan))


def test_a_driver_who_steers_back_in_time_gets_no_warning_line_rumble():
    state = {"r": 0.0}

    def plan(t):
        psi = 0.0
        if t >= DRIFT_START:
            psi = max(0.0, math.atan2(0.4, SPEED) * min(1.0, (t - DRIFT_START) / RAMP_S) - 0.05 * max(0.0, t - 4.3))
        state["r"] += SPEED * math.sin(psi) * DT
        return state["r"], psi, {}

    assert not warns(drive(plan, seconds=9.0))


def test_a_router_lane_switch_silences_the_lane_settle():
    def plan(t):
        return (0.0 if t < 5.0 else -4.5), 0.0, {}

    decisions = drive(plan, seconds=9.0)
    after = decisions[int(5.0 * HZ) + 2: int((5.0 + TN.steady_s - 0.1) * HZ)]
    assert after and all(d.reason == "lane_settle" for d in after)


def test_a_blinker_seen_while_settling_still_covers_a_truck_straddling_the_line():
    def plan(t):
        return 1.1, 0.0, {"blinker_right": t < 0.5}

    assert not any(d.active for d in drive(plan, seconds=10.0))
    assert any(d.active for d in drive(lambda t: (1.1, 0.0, {}), seconds=10.0))


def test_hugging_the_inside_of_a_bend_is_silent_but_not_on_a_straight():
    inside = drive(lambda t: (1.0, 0.0, {}), radius=300.0)
    assert not any(d.active for d in inside)
    assert inside[-1].near_miss == "bend_inner"
    straight = drive(lambda t: (1.0, 0.0, {}))
    assert any(d.active for d in straight)


def test_the_inside_allowance_grows_with_curvature_and_is_capped():
    assert bend_allowance(0.0, TN) == 0.0
    assert bend_allowance(-1 / 300.0, TN) == 0.0
    assert bend_allowance(1 / 1000.0, TN) < bend_allowance(1 / 400.0, TN) < TN.bend_cap_m
    assert bend_allowance(1 / 100.0, TN) == TN.bend_cap_m


def test_cutting_past_the_inside_allowance_warns_without_a_cue():
    """Keep drifting toward the inside of a bend and the warning comes, only later than on a straight."""
    decisions = drive(drift(0.4, 1.0), radius=300.0)
    assert any(d.near_miss == "bend_inner" for d in decisions)
    assert not cues(decisions)
    started = next(d for d in decisions if d.started)
    assert started.level == LEVEL_WARN and started.side == SIDE_RIGHT
    allow = TN.warn_overlap_m + bend_allowance(1 / 300.0, TN)
    assert started.allow_m == pytest.approx(allow, abs=0.02)
    assert -allow - 0.05 < started.gap_m <= -allow + 0.02


def bend_ahead_route(straight=300.0, radius=300.0, step=5.0):
    """Straight north for ``straight`` metres, then a right bend."""
    pts = [(0.0, -i * step) for i in range(int(straight / step) + 1)]
    for i in range(1, 80):
        u = i * step
        pts.append((radius * (1.0 - math.cos(u / radius)), -straight - radius * math.sin(u / radius)))
    return SimpleNamespace(pts=tuple(pts), kind="road", divergence=(), branch_s=None, merge_s=None, reach_m=700.0)


def test_turning_in_before_the_bend_counts_as_cutting_it():
    peaks = bend_peaks(bend_ahead_route().pts, 280.0, SPEED, TN)
    assert peaks[0] == 0.0 and peaks[1] == pytest.approx(1 / 300.0, rel=0.05)

    def hold(route, s):
        detector = LaneDetector()
        frames = (replace(frame_at(i * DT, route, None, 0.95, 0.0), z=-s) for i in range(int(4 * HZ)))
        return [detector.update(f) for f in frames]

    assert not any(d.active for d in hold(bend_ahead_route(), 280.0))
    assert any(d.active for d in hold(bend_ahead_route(), 150.0))


def test_a_concentric_line_close_to_the_inside_edge_is_not_a_departure():
    decisions = drive(lambda t: (0.85, 0.0, {}), radius=300.0)
    assert not any(d.active for d in decisions)


def test_understeering_toward_the_outside_of_a_bend_warns_at_the_line_with_no_allowance():
    decisions = drive(drift(0.4, -1.0), radius=300.0)
    warn = warns(decisions)[0]
    assert warn.side == SIDE_LEFT
    assert warn.allow_m == TN.warn_overlap_m and LINE - 0.05 < warn.gap_m <= LINE
    assert cues(decisions)[0].side == SIDE_LEFT


def test_a_sharp_bend_is_silent():
    decisions = drive(drift(0.4), radius=70.0, route=make_route(70.0, length=200.0))
    assert not any(d.active for d in decisions)
    assert {d.reason for d in decisions} <= {"bend_sharp", "short_route", "off_route"}
    assert decisions[int(3.0 * HZ)].reason == "bend_sharp"


def test_a_truck_far_from_its_lane_is_off_route():
    decisions = drive(lambda t: (8.0, 0.0, {}), seconds=4.0)
    assert {d.reason for d in decisions} == {"off_route"}


@pytest.mark.parametrize("name,kw,reason", [
    ("slow", dict(speed_ms=11.0), "speed"),
    ("paused", dict(paused=True), "paused"),
    ("aeb", dict(aeb_active=True), "aeb"),
    ("reverse", dict(gear=-1), "reverse"),
    ("brake", dict(brake=0.8), "brake"),
    ("stale", dict(age_s=0.4), "stale"),
])
def test_gates_keep_the_detector_silent(name, kw, reason):
    decisions = drive(drift(0.4), **kw)
    assert not any(d.active for d in decisions)
    assert decisions[-1].reason == reason


def test_moderate_braking_does_not_mute():
    assert any(d.active for d in drive(drift(0.4), brake=0.5))


@pytest.mark.parametrize("route,reason", [
    (make_route(kind="ramp"), "route_kind"),
    (make_route(kind="prefab", branch_s=70.0), "branch"),
    (make_route(kind="prefab", merge_s=80.0), "merge"),
    (make_route(branch_s=70.0), "branch"),
    (make_route(merge_s=80.0), "merge"),
    (make_route(length=150.0, reach_m=150.0), "short_route"),
])
def test_route_gates(route, reason):
    decisions = drive(drift(0.4), route=route)
    assert not any(d.active for d in decisions)
    assert decisions[-1].reason == reason


@pytest.mark.parametrize("rate", [0.5, 0.8, 1.2])
def test_the_cue_ignores_relative_speed_while_far_from_the_edge(rate):
    decisions = drive(drift(rate))
    cue = cues(decisions)[0]
    assert cue.gap_m <= LINE + TN.approach_range_m + 0.05
    far = [d for d in decisions if d.gap_m > LINE + TN.approach_range_m + 0.05]
    assert far and not any(d.active for d in far)


def test_a_slow_wander_gets_no_cue_only_the_line_warning():
    decisions = drive(drift(0.15), seconds=20.0)
    assert not cues(decisions) and len(warns(decisions)) == 1


def test_the_cue_comes_first_and_the_warning_follows_at_the_line():
    decisions = drive(drift(0.5))
    first, second = [d for d in decisions if d.started][:2]
    assert (first.level, second.level) == (LEVEL_CLOSE, LEVEL_WARN)
    assert first.tlc_s <= TN.approach_tlc_s and first.gap_m > second.gap_m


def test_the_cue_runs_at_least_one_second_and_ends_when_the_truck_stops_closing():
    state = {"r": 0.0}
    a = math.atan2(0.5, SPEED)

    def plan(t):
        psi = a if DRIFT_START <= t < 4.6 else 0.0
        state["r"] += SPEED * math.sin(psi) * DT
        return state["r"], psi, {}

    decisions = drive(plan, seconds=9.0)
    live = [i for i, d in enumerate(decisions) if d.active]
    assert cues(decisions) and not warns(decisions)
    assert 1.0 - DT <= (live[-1] - live[0] + 1) * DT <= 2.5


def test_a_brief_touch_of_the_line_still_rumbles_for_a_second():
    plan = steer_back_plan(turn_at=0.88)
    decisions = drive(plan, seconds=12.0)
    assert len(warns(decisions)) == 1
    assert warn_seconds(decisions) >= 1.0 - DT
    assert min(plan.state["gap"]) < LINE


def test_moving_toward_the_centre_mutes_the_warning():
    decisions = drive(lambda t: (1.0, -0.03, {}))
    assert not any(d.active for d in decisions)
    assert decisions[-1].near_miss == "returning" and decisions[-1].closing_ms < -0.5


def test_standing_over_the_line_without_moving_in_still_warns():
    decisions = drive(lambda t: (1.0, -0.002, {}))
    assert warns(decisions)


def test_a_cue_that_ends_does_not_disarm_the_line_warning():
    a = math.atan2(0.5, SPEED)
    state = {"r": 0.0, "hold": None}

    def plan(t):
        psi = 0.0
        if t >= DRIFT_START:
            if state["hold"] is None and state["r"] >= 0.76:
                state["hold"] = t
            if state["hold"] is None or t >= state["hold"] + 2.5:
                psi = a
        state["r"] += SPEED * math.sin(psi) * DT
        return state["r"], psi, {}

    decisions = drive(plan, seconds=12.0)
    cue_at = [i * DT for i, d in enumerate(decisions) if d.started and d.level == LEVEL_CLOSE]
    warn_at = [i * DT for i, d in enumerate(decisions) if d.started and d.level == LEVEL_WARN]
    assert len(cue_at) == 1 and len(warn_at) == 1
    assert warn_at[0] > state["hold"] + 2.4
    assert LINE - 0.1 < warns(decisions)[0].gap_m <= LINE


def test_the_inside_of_a_tight_corner_cues_only_for_a_fast_run_and_the_rest_like_a_straight():
    slow, fast = drive(drift(0.5, 1.0), radius=300.0), drive(drift(1.5, 1.0), radius=300.0)
    assert not cues(slow) and warns(slow)[0].side == SIDE_RIGHT
    assert any(d.near_miss == "inner_slow" for d in slow)
    assert cues(fast)[0].side == SIDE_RIGHT
    assert cues(drive(drift(0.5, -1.0), radius=300.0))[0].side == SIDE_LEFT
    for side, expected in ((1.0, SIDE_RIGHT), (-1.0, SIDE_LEFT)):
        slight = drive(drift(0.5, side), radius=700.0)
        assert cues(slight) and cues(slight)[0].side == expected


def test_a_straight_prefab_lane_without_branch_or_merge_is_a_standard_lane():
    route = make_route(kind="prefab")
    decisions = drive(drift(0.5), route=route)
    assert warns(decisions) and cues(decisions)
    assert not any(d.reason == "route_kind" for d in decisions)
    assert LINE - 0.05 < warns(decisions)[0].gap_m <= LINE


def test_a_curved_prefab_lane_is_still_not_a_lane():
    decisions = drive(drift(0.4), radius=300.0, route=make_route(300.0, kind="prefab"))
    assert not any(d.active for d in decisions)
    assert decisions[-1].reason == "route_kind"
    gentle = drive(drift(0.4), radius=1000.0, route=make_route(1000.0, kind="prefab"))
    assert warns(gentle)


def test_a_prefab_lane_that_bends_ahead_is_not_a_lane():
    route = SimpleNamespace(**{**bend_ahead_route().__dict__, "kind": "prefab"})
    detector = LaneDetector()
    frames = (replace(frame_at(i * DT, route, None, 0.0, 0.0), z=-280.0) for i in range(int(4 * HZ)))
    assert {detector.update(f).reason for f in frames} == {"route_kind"}


def test_no_map_is_silent():
    assert LaneDetector().update(frame_at(0.0, None, None, 0.0, 0.0)).reason == "no_map"
    assert LaneDetector().update(None).reason == "no_data"


def test_a_gate_ends_a_warning_at_once_even_inside_the_minimum_hold():
    detector = LaneDetector()
    plan = drift(0.5)
    decisions = drive(plan, detector=detector, seconds=5.45)
    assert decisions[-1].active and decisions[-1].level == LEVEL_WARN
    lateral, psi, _ = plan(5.45)
    gated = detector.update(frame_at(5.45, make_route(), None, lateral, psi, aeb_active=True))
    assert not gated.active and gated.reason == "aeb"


def test_project_reports_metres_right_of_the_lane():
    route = make_route()
    s, right, tx, tz = project(route.pts, 1.5, -50.0)
    assert (round(s, 3), round(right, 3), round(tx, 3), round(tz, 3)) == (50.0, 1.5, 0.0, -1.0)
    assert project(route.pts, -2.0, -50.0)[1] == pytest.approx(-2.0)


def test_project_extends_the_route_behind_its_start():
    s, right, _, _ = project(make_route().pts, 1.5, 3.0)
    assert s == pytest.approx(-3.0) and right == pytest.approx(1.5)


def test_lane_curvature_sign_follows_the_turn():
    assert lane_kappa_right(make_route(300.0).pts, 50.0) == pytest.approx(1 / 300.0, rel=0.02)
    assert lane_kappa_right(make_route(-300.0).pts, 50.0) == pytest.approx(-1 / 300.0, rel=0.02)
    assert lane_kappa_right(make_route().pts, 50.0) == pytest.approx(0.0, abs=1e-9)
    assert smooth_heading(make_route().pts, 1e6) == pytest.approx(math.atan2(-1.0, 0.0))
    assert smooth_heading((), 5.0) is None


@pytest.mark.parametrize("step", [5.0, 10.0, 25.0])
def test_a_coarse_polyline_does_not_make_the_heading_wobble(step):
    route = make_route(300.0, step=step)
    decisions = drive(lambda t: (0.0, 0.0, {}), radius=300.0, route=route)
    settled = decisions[int(3.0 * HZ):]
    assert max(abs(d.psi_deg) for d in settled) < 0.15
    assert not any(d.active or d.near_miss for d in settled)


def test_a_pose_jump_restarts_the_detector():
    detector = LaneDetector()
    assert drive(drift(0.4), detector=detector, seconds=6.5)[-1].active
    jumped = detector.update(replace(frame_at(6.55, make_route(), None, 1.2, 0.016), x=900.0, z=900.0))
    assert not jumped.active


def test_toward_an_unsafe_lane_a_blinker_no_longer_mutes_the_warning():
    assert any(d.active for d in drive(drift(0.4, blinker_right=True, unsafe_right=True)))
    assert not any(d.active for d in drive(drift(0.4, blinker_right=True, unsafe_left=True))), "the other side"


def test_toward_an_unsafe_lane_a_calm_lane_change_warns():
    def plan(t):
        lateral, psi, _ = drift(0.8)(t)
        return lateral, psi, {**wheel_toward_right(0.02)(t), "blinker_right": True, "unsafe_right": True}

    assert any(d.active for d in drive(plan))


def test_toward_an_unsafe_lane_a_swerve_still_mutes():
    def plan(t):
        lateral, psi, _ = drift(1.3)(t)
        return lateral, psi, {"steer_right": 0.3 * max(0.0, t - DRIFT_START + 0.5), "unsafe_right": True}

    decisions = drive(plan)
    assert not any(d.active for d in decisions)
    assert any(d.near_miss == "evade" for d in decisions)
