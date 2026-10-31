"""With no lead, ACC's cap follows the cruise bid instead of a flat 1.5 m/s2.

The flat ceiling sat under every acceleration profile through the orchestrator
min(): Sport, meant to be full throttle, held exactly 1.50 on an empty road
(mapper log, 2026-10-19). With a lead the gap law is untouched.
"""
from __future__ import annotations

import itertools
import math

import pytest

from core.cruise_control_thread import acc_controller
from core.cruise_control_thread.acc_controller import AdaptiveCruiseController, _LeadSnapshot
from core.cruise_control_thread.blinker_arbitration import BlinkerState
from core.cruise_control_thread.thread import CruiseControlThread
from core.longitudinal import acc as long_acc
from core.longitudinal.acc import free_road_target_ms2
from core.longitudinal.base import LongOutput
from core.settings import Settings
from tests.longitudinal.harness import make_ctx

DT = 1.0 / 60.0


@pytest.fixture
def clock(monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(acc_controller.time, "monotonic", lambda: 1000.0 + next(ticks) * DT)


@pytest.fixture
def child(monkeypatch, clock):
    settings = Settings.instance()
    monkeypatch.setattr(settings, "acc_enabled", True)
    monkeypatch.setattr(settings, "cc_mode", "Cruise control")
    acc = long_acc.AdaptiveCruiseController()
    acc._inner._read_acc_snapshot = lambda: ([], None, BlinkerState())
    assert acc.active
    return acc


def _run(ctrl, no_lead, *, ticks=600, leads=(), v_ego=15.0):
    leads = list(leads)
    ctrl._read_acc_snapshot = lambda: (list(leads), None, BlinkerState())
    return [ctrl.accel_cap_ms2(v_ego, no_lead) for _ in range(ticks)]


def _settle(child, bid, ticks=600):
    out = None
    for _ in range(ticks):
        out = child.step(make_ctx(10.0), bid)
    return out.wanted_ms2


def test_without_a_target_the_old_ceiling_stands(clock):
    """Tools and probes that drive ACC alone keep the behaviour they were tuned on."""
    ctrl = AdaptiveCruiseController()
    assert _run(ctrl, None)[-1] == pytest.approx(ctrl.config.no_lead_ceiling_ms2)


def test_the_cap_follows_a_cruise_bid_above_the_old_ceiling(child):
    """Normal's 2.2 m/s2 launch must reach the truck on an empty road."""
    assert _settle(child, 2.2) == pytest.approx(2.2, abs=1e-6)


@pytest.mark.parametrize("bid", [-1.0, 0.0, 0.5, 1.2])
def test_the_cap_never_drops_below_the_old_ceiling(child, bid):
    """A cut-in still starts from where it always did once the cruise bid is small."""
    assert _settle(child, bid) == pytest.approx(child._inner.config.no_lead_ceiling_ms2)


def test_a_raw_sport_bid_is_bounded_one_dead_band_above_the_gap_law():
    """A lead appearing far away is met at the plain jerk rate, not stepped down to."""
    cfg = AdaptiveCruiseController().config
    bound = free_road_target_ms2(12.0, AdaptiveCruiseController())
    assert bound == pytest.approx(cfg.a_max_ms2 + cfg.j_onset_dead_ms2)
    assert bound >= 2.2, "must stay above Normal's launch or ACC caps the profile again"


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf])
def test_a_missing_or_broken_cruise_bid_falls_back_to_the_old_ceiling(bad):
    ctrl = AdaptiveCruiseController()
    assert free_road_target_ms2(bad, ctrl) == ctrl.config.no_lead_ceiling_ms2


def test_the_cap_rises_through_the_jerk_limit(child):
    """Following the bid must not step the command: the plain rate still applies."""
    _settle(child, None, ticks=120)
    caps = [child.step(make_ctx(10.0), 12.0).wanted_ms2 for _ in range(120)]
    steps = [b - a for a, b in zip(caps, caps[1:])]
    assert max(steps) <= child._inner.config.j_max_ms3 * DT + 1e-9


def test_a_far_lead_is_met_without_the_onset_chase(clock):
    """From the bound, a lead whose law grants 1.4 is reached at the plain jerk rate."""
    ctrl = AdaptiveCruiseController()
    cfg = ctrl.config
    top = free_road_target_ms2(12.0, ctrl)
    _run(ctrl, top, ticks=300)
    lead = _LeadSnapshot(vid=7, dist_m=140.0, v_lead_ms=15.0, a_lead_ms2=0.0, score=10.0)
    caps = _run(ctrl, top, ticks=120, leads=[lead])
    drops = [a - b for a, b in zip([top] + caps, caps)]
    assert max(drops) <= cfg.j_max_ms3 * DT * 1.25


def test_a_lead_ignores_the_cruise_bid_entirely(clock):
    """The gap law is the measured one; the cruise bid only replaces the empty-road constant."""
    lead = _LeadSnapshot(vid=7, dist_m=35.0, v_lead_ms=12.0, a_lead_ms2=0.0, score=10.0)
    plain = _run(AdaptiveCruiseController(), None, leads=[lead])
    followed = _run(AdaptiveCruiseController(), 2.5, leads=[lead])
    assert followed == plain


def test_an_empty_road_leaves_the_cruise_bid_in_charge(child):
    """Through the orchestrator min(): the merged command is CC's bid, not 1.5."""
    cc_out = LongOutput(2.2, True)
    acc_out = LongOutput(_settle(child, cc_out.wanted_ms2), True)
    wanted, commanding, winner = CruiseControlThread._arbitrate_named(("cc", cc_out), ("acc", acc_out))
    assert commanding and winner == "cc"
    assert wanted == pytest.approx(2.2, abs=1e-3)
