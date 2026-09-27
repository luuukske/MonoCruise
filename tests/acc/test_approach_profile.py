"""Approach profile: close on a slower or stopped lead at the constant rate that meets it,
instead of holding speed and then braking firmly, or braking hard early and crawling in.
See core/acc/ACC_ARCHITECTURE.md §13.5."""
from __future__ import annotations

import math
import random

import pytest

from core.cruise_control_thread.acc_controller import (
    T_HEADWAY_BY_LEVEL_S, AdaptiveCruiseController, _LeadSnapshot,
)
from core.cruise_control_thread.approach_profile import approach_band, slew_delta
from core.settings import Settings

DT = 1.0 / 60.0


def _cfg():
    return AdaptiveCruiseController().config


def _lead(dist_m, v_lead, a_lead=0.0):
    return _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead, a_lead_ms2=a_lead, score=6.0,
                         conf=1.0, a_lead_ff_ms2=a_lead)


def test_the_band_never_relaxes_past_the_glide_to_stop_rate():
    """Whatever it does to the law, a closing lead still gets at least -dv^2 / 2s."""
    cfg = _cfg()
    rng = random.Random(1515)
    for _ in range(20000):
        v_lead = rng.choice([0.0, rng.uniform(0.0, 25.0)])
        v_ego = v_lead + rng.uniform(0.0, 20.0)
        dist = rng.uniform(3.0, 160.0)
        law = rng.uniform(-6.55, 1.5)
        glide = -((v_ego - v_lead) ** 2) / (2.0 * dist)
        out = approach_band(cfg, law, _lead(dist, v_lead, rng.uniform(-0.4, 0.4)), v_ego,
                            rng.choice(T_HEADWAY_BY_LEVEL_S[1:]))
        assert cfg.max_decel_ms2 - 1e-9 <= out
        if law <= glide:
            assert out <= glide + 1e-9


@pytest.mark.parametrize("kw", [
    dict(a_lead=-1.2),  # braking lead: the law reads its accel
    dict(a_lead=1.2),  # pulling away: the constant-speed need is wrong
    dict(v_ego=21.0),  # barely closing: a gap correction, not an approach
], ids=["braking-lead", "accelerating-lead", "not-closing"])
def test_the_band_stays_out_of_the_way(kw):
    args = dict(dist=60.0, v_lead=20.0, a_lead=0.0, v_ego=28.0)
    args.update(kw)
    cfg = _cfg()
    for law in (-6.0, -3.0, -0.5, 0.8):
        lead = _lead(args["dist"], args["v_lead"], args["a_lead"])
        assert approach_band(cfg, law, lead, args["v_ego"], 1.1) == law


def test_a_critical_approach_to_a_moving_lead_keeps_the_laws_margin():
    """Past the critical need the law's front-loading is wanted: bit-identical."""
    cfg = _cfg()
    lead = _lead(20.0, 10.0)
    need = (25.0 - 10.0) ** 2 / (2.0 * (20.0 - cfg.s0_m - 10.0 * 1.1))
    assert need > cfg.approach_upper_zero_ms2
    assert approach_band(cfg, -6.4, lead, 25.0, 1.1) == -6.4


def test_a_stopped_lead_is_braked_for_at_least_its_need():
    """Its need is exact, so the lower edge stays on however tight the stop gets."""
    cfg = _cfg()
    for dist, v in ((150.0, 13.9), (60.0, 20.0), (40.0, 20.0)):
        need = v * v / (2.0 * (dist - cfg.s0_m - cfg.approach_stop_margin_m))
        assert approach_band(cfg, 1.0, _lead(dist, 0.0), v, 1.1) <= max(-need, cfg.max_decel_ms2) + 1e-9


def test_the_pull_builds_slowly_and_gives_braking_back_at_once():
    step, release = 0.05, 0.08
    assert slew_delta(0.0, -2.0, step, release) == pytest.approx(-step)
    assert slew_delta(-1.0, 0.0, step, release) == pytest.approx(-1.0 + release)
    assert slew_delta(0.0, 1.5, step, release) == pytest.approx(step)
    assert slew_delta(1.5, 0.0, step, release) == 0.0, "softening must drop out at once"
    assert slew_delta(1.5, -1.0, step, release) == pytest.approx(-step)


def _stop_for_a_stopped_vehicle(level, v0_kmh, seen_m, share=None):
    """Closed loop onto a stopped vehicle published at `seen_m`. Returns decel per sample."""
    previous = Settings.instance().acc_gap_level
    Settings.instance().acc_gap_level = level
    try:
        ctrl = AdaptiveCruiseController()
        cfg = ctrl.config
        if share is not None:
            cfg.approach_share = share
        v, gap, a_truck, t = v0_kmh / 3.6, seen_m, 0.0, 0.0
        decels = []
        while t < 60.0 and v > 0.3:
            ctrl._prev_mono = t
            raw = [_LeadSnapshot(vid=1, dist_m=gap, v_lead_ms=0.0, a_lead_ms2=0.0, score=6.0)]
            smooth = ctrl._smooth_chain(raw, DT, t)
            ctrl._landing.track_ego(v, DT)
            a_raw, emergency = ctrl._compute_command(raw, smooth, v, DT)
            cap = ctrl._output_filter(ctrl._jerk_limit(a_raw, DT, emergency, v), DT, emergency)
            a_truck += (min(cap, 0.0) - a_truck) * (1.0 - math.exp(-DT / 0.3))
            v = max(0.0, v + a_truck * DT)
            gap -= v * DT
            decels.append(-a_truck)
            t += DT
        return decels, gap
    finally:
        Settings.instance().acc_gap_level = previous


def _halves(decels):
    moving = [d for d in decels]
    h = len(moving) // 2
    return sum(moving[:h]) / h, sum(moving[h:]) / (len(moving) - h)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_a_stopped_vehicle_seen_early_gets_an_even_stop(level):
    """50 km/h, stopped vehicle at 150 m: the law held speed to ~80 m, then braked firmly."""
    decels, gap = _stop_for_a_stopped_vehicle(level, 50.0, 150.0)
    first, second = _halves(decels)
    old_first, old_second = _halves(_stop_for_a_stopped_vehicle(level, 50.0, 150.0, share=0.0)[0])
    assert old_first < 0.25 * old_second, "the fixture must reproduce the late, firm stop"
    assert 0.75 < first / second < 1.33
    assert max(decels) < 0.6 * max(_stop_for_a_stopped_vehicle(level, 50.0, 150.0, share=0.0)[0])
    assert gap > 3.0
