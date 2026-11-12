"""Lane departure detector: the start conditions the rumble and the lane keeping pulse share since 2026-11-11."""

from __future__ import annotations

import math

from core.la.lane_model import LocalLane
from core.lane_departure.detector import SIDE_RIGHT, LaneDetector
from tests.lane_departure.test_detector import (
    DRIFT_START, DT, LINE, RAMP_S, SPEED, TN, cues, drift, drive, right_gap, warn_seconds, warns, wheel_toward_right,
)


def curving_home(wheel_rate, rate=0.3, turn_at=0.83, inward=0.5):
    """Drift right at ``rate``, then curve home at ``inward`` m/s^2 from ``turn_at``; the wheel turns left 0.1 m before."""
    state = {"r": 0.0, "v": 0.0, "turned": None, "wheel": None, "gap": []}

    def plan(t):
        if t >= DRIFT_START:
            if state["wheel"] is None and state["r"] >= turn_at - 0.1:
                state["wheel"] = t
            if state["turned"] is None and state["r"] >= turn_at:
                state["turned"] = t
            if state["turned"] is None:
                state["v"] = rate * min(1.0, (t - DRIFT_START) / RAMP_S)
            elif state["r"] > 0.3:
                state["v"] = max(-rate, state["v"] - inward * DT)
            else:
                state["v"] = 0.0
            state["r"] += state["v"] * DT
        psi = math.asin(state["v"] / SPEED)
        state["gap"].append(right_gap(state["r"], psi))
        back = 0.0 if state["wheel"] is None else t - state["wheel"] + DT
        return state["r"], psi, {"steer_right": -wheel_rate * min(back, 1.0)}

    plan.state = state
    return plan


def test_a_driver_already_steering_back_is_not_rumbled_at_for_brushing_the_line():
    """The wheel is turned home as the corner reaches the warning line and the truck stops 0.05 m past it."""
    steering, hands_off = curving_home(0.04), curving_home(0.0)
    decisions = drive(steering, seconds=12.0)
    assert not any(d.active for d in decisions)
    assert any(d.near_miss == "correcting" for d in decisions)
    assert -TN.back_reach_m < min(steering.state["gap"]) - LINE < 0.0
    assert warns(drive(hands_off, seconds=12.0))


def test_a_wheel_turned_home_that_does_not_bring_the_truck_back_still_warns():
    def plan(t):
        lateral, psi, _ = drift(0.4)(t)
        return lateral, psi, {"steer_right": -0.03 * min(1.0, max(0.0, t - DRIFT_START - 1.0))}

    decisions = drive(plan)
    warn = warns(decisions)[0]
    assert any(d.near_miss == "correcting" for d in decisions)
    assert LINE - TN.back_reach_m - 0.1 < warn.gap_m < LINE


def test_a_rumble_does_not_start_on_a_truck_already_heading_home():
    """Over the line as the speed gate opens, creeping back at 0.1 m/s: silence; standing still there still warns."""
    creeping = drive(lambda t: (1.0, -0.004, {}))
    assert not any(d.active for d in creeping) and creeping[-1].near_miss == "returning"
    assert -0.2 < creeping[-1].closing_ms < -TN.home_ms
    assert warns(drive(lambda t: (1.0, 0.0, {})))


def test_a_running_rumble_is_not_ended_by_a_slow_creep_home():
    a = math.atan2(0.4, SPEED)
    state = {"r": 0.0}

    def plan(t):
        psi = 0.0 if t < DRIFT_START else a if state["r"] < 1.1 and t < 6.0 else -0.004
        state["r"] += SPEED * math.sin(psi) * DT
        return state["r"], psi, {}

    decisions = drive(plan, seconds=9.0)
    assert len(warns(decisions)) == 1 and warn_seconds(decisions) >= 2.0


def test_turning_into_a_tight_bend_and_cutting_it_is_intent_and_a_still_wheel_running_inside_is_not():
    """Right bend R 300, a fast run to the inside: with the wheel turned in it is a cut, with a still wheel a drift."""
    def turning_in(t):
        lateral, psi, _ = drift(1.0, 1.0)(t)
        return lateral, psi, wheel_toward_right(0.02)(t)

    cut = drive(turning_in, radius=300.0)
    assert not any(d.active for d in cut) and any(d.near_miss in {"intent", "latched"} for d in cut)
    still = drive(drift(1.0, 1.0), radius=300.0)
    assert cues(still)[0].side == SIDE_RIGHT and warns(still)[0].side == SIDE_RIGHT


def test_an_approach_cue_needs_the_truck_itself_drifting_out_by_the_lane_fit():
    """The polyline heading shows drift in fast bends that the fitted lane does not; the line warning is position only."""
    detector = LaneDetector()
    detector._lane.local = lambda *a: LocalLane(0.0, 0.0, 0.0, 0.0)
    decisions = drive(drift(0.6), detector=detector)
    assert not cues(decisions) and warns(decisions)
    assert any(d.near_miss == "no_drift" for d in decisions)
    assert cues(drive(drift(0.6)))
