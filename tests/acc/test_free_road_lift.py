"""A far lead lifts ACC toward the profile's bid; a lead that binds sees the law untouched.

Every tracked lead out to tracker range used to hold the profile near IIDM's 1.5 m/s2
free-road accel, so Sport was only uncapped on an empty road. See
core/cruise_control_thread/README.md, "Free-road lift".
"""
from __future__ import annotations

import itertools
import random

import pytest

from core.cruise_control_thread import acc_controller, free_road_lift
from core.cruise_control_thread.acc_controller import AdaptiveCruiseController, _LeadSnapshot
from core.cruise_control_thread.blinker_arbitration import BlinkerState

DT = 1.0 / 60.0
SPORT_TARGET = 2.5


@pytest.fixture
def clock(monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(acc_controller.time, "monotonic", lambda: 1000.0 + next(ticks) * DT)


def _caps(target, lead, v_ego, ticks=240):
    ctrl = AdaptiveCruiseController()
    ctrl._read_acc_snapshot = lambda: ([lead], None, BlinkerState())
    return ctrl, [ctrl.accel_cap_ms2(v_ego, target) for _ in range(ticks)]


def _states(n=400, seed=11):
    rng = random.Random(seed)
    for _ in range(n):
        v_ego = rng.uniform(0.0, 30.0)
        yield v_ego, rng.uniform(4.0, 160.0), max(0.0, v_ego + rng.uniform(-6.0, 4.0)), rng.uniform(-3.0, 1.0)


def test_the_weight_is_zero_once_the_law_binds_and_one_on_a_free_road():
    cfg = AdaptiveCruiseController().config
    for v in (0.0, 8.0, 15.0, 25.0):
        free = free_road_lift.free_road_ms2(cfg, v)
        assert free_road_lift.lift_weight(cfg, 0.87 * free, v) == 0.0
        assert free_road_lift.lift_weight(cfg, 0.98 * free, v) == pytest.approx(1.0)
        assert free_road_lift.lift_weight(cfg, 1.2 * free, v) == 1.0
        assert free_road_lift.lift_weight(cfg, -2.0, v) == 0.0


def test_a_lead_that_binds_gets_the_law_bit_for_bit(clock):
    """Close following, approach, landing and overlays: everything below the band."""
    checked = 0
    for v_ego, gap, v_lead, a_lead in _states():
        lead = _LeadSnapshot(vid=3, dist_m=gap, v_lead_ms=v_lead, a_lead_ms2=a_lead, score=10.0)
        plain_ctrl, plain = _caps(None, lead, v_ego)
        cfg = plain_ctrl.config
        if max(free_road_lift.lift_weight(cfg, c, v_ego) for c in plain) > 0.0:
            continue
        _, lifted = _caps(SPORT_TARGET, lead, v_ego)
        assert lifted == plain, (v_ego, gap, v_lead, a_lead)
        checked += 1
    assert checked > 150, "the sample must cover the binding regime"


def test_the_lift_never_lowers_the_law(clock):
    for v_ego, gap, v_lead, a_lead in _states(n=200, seed=5):
        lead = _LeadSnapshot(vid=3, dist_m=gap, v_lead_ms=v_lead, a_lead_ms2=a_lead, score=10.0)
        _, plain = _caps(None, lead, v_ego)
        _, lifted = _caps(SPORT_TARGET, lead, v_ego)
        assert all(b >= a - 1e-9 for a, b in zip(plain, lifted))


def test_a_far_lead_no_longer_caps_sport(clock):
    lead = _LeadSnapshot(vid=3, dist_m=150.0, v_lead_ms=8.33, a_lead_ms2=0.0, score=10.0)
    _, plain = _caps(None, lead, 8.33)
    _, lifted = _caps(SPORT_TARGET, lead, 8.33)
    assert plain[-1] < 1.5
    assert lifted[-1] > 2.3


def test_closing_in_on_a_lead_hands_back_to_the_law_smoothly(clock):
    """The weight falls with the law, so the command never steps on the way in."""
    ctrl = AdaptiveCruiseController()
    state = {"gap": 160.0}
    ctrl._read_acc_snapshot = lambda: (
        [_LeadSnapshot(vid=3, dist_m=state["gap"], v_lead_ms=8.0, a_lead_ms2=0.0, score=10.0)],
        None, BlinkerState())
    caps = []
    for _ in range(int(40.0 / DT)):
        caps.append(ctrl.accel_cap_ms2(12.0, SPORT_TARGET))
        state["gap"] = max(state["gap"] - 4.0 * DT, 15.0)
    steps = [abs(b - a) for a, b in zip(caps, caps[1:])]
    assert caps[0] > 2.0 and caps[-1] < 0.5
    assert max(steps) <= ctrl.config.j_max_ms3 * DT * 1.05


def test_no_target_keeps_every_tool_on_the_old_law(clock):
    """Probes call accel_cap_ms2 with one argument and must see the tuned controller."""
    lead = _LeadSnapshot(vid=3, dist_m=150.0, v_lead_ms=8.33, a_lead_ms2=0.0, score=10.0)
    ctrl = AdaptiveCruiseController()
    ctrl._read_acc_snapshot = lambda: ([lead], None, BlinkerState())
    one_arg = [ctrl.accel_cap_ms2(8.33) for _ in range(120)]
    _, none_arg = _caps(None, lead, 8.33, ticks=120)
    assert one_arg == none_arg
