"""Approach profile: close on a slower or stopped lead at the constant rate that meets it,
instead of holding speed and then braking firmly, or braking hard early and crawling in.
See core/acc/ACC_ARCHITECTURE.md §13.5."""
from __future__ import annotations

import math
import random
from collections import deque

import pytest

from core.cruise_control_thread import acc_controller, approach_profile
from core.cruise_control_thread.acc_controller import (
    T_HEADWAY_BY_LEVEL_S, AdaptiveCruiseController, _LeadSnapshot,
)
from core.cruise_control_thread.approach_profile import (
    approach_band, brake_follow_limit, slew_delta,
)
from core.cruise_control_thread.blinker_arbitration import BlinkerState
from core.radar.traffic import _smooth_vehicle_kinematics
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


def test_closing_on_a_braking_lead_is_not_braked_past_that_decel():
    """The catch-up may meet the lead. It does not out-brake it while there is room."""
    cfg = _cfg()
    lead = _lead(25.0, 18.0, a_lead=-2.0)
    gentle = brake_follow_limit(cfg, -1.5, lead, 22.0, 1.1, gap_excess_ms=0.0)
    hard = brake_follow_limit(cfg, -5.0, lead, 22.0, 1.1, gap_excess_ms=0.0)
    assert gentle == -1.5
    assert hard == pytest.approx(-4.0)
    assert brake_follow_limit(cfg, -5.0, lead, 22.0, 1.1, gap_excess_ms=3.0) == -5.0
    cfg.follow_share = 0.0
    assert brake_follow_limit(cfg, -5.0, lead, 22.0, 1.1, gap_excess_ms=0.0) == -5.0


def test_the_follow_limit_is_for_a_braking_lead_only():
    """A smoothed accel trailing off just below 0 is a lead that stopped braking, not one braking."""
    cfg = _cfg()
    for a_lead in (-0.01, -0.05, -0.3):
        assert brake_follow_limit(cfg, -1.5, _lead(25.0, 18.0, a_lead), 22.0, 1.1,
                                  gap_excess_ms=0.0) == -1.5
    eased = brake_follow_limit(cfg, -1.5, _lead(25.0, 18.0, -0.45), 22.0, 1.1, gap_excess_ms=0.0)
    full = -(cfg.approach_upper_gain * 0.45 + cfg.approach_upper_margin_ms2)
    assert -1.5 < eased < full, "the gate fades in, it does not step"


def _far_lead_short_slowdown(level, v0_kmh, monkeypatch, old_gate=False):
    """Ego at 1.6x its wanted gap closing at 5 m/s, then the lead brakes 2.4 m/s^2 for 1 s and holds.

    The lead is seen through the ACC radar chain. Returns the smallest gap as a share of the wanted."""
    if old_gate:
        monkeypatch.setattr(approach_profile, "FOLLOW_BRAKING_ZERO_MS2", 0.0)
        monkeypatch.setattr(approach_profile, "FOLLOW_BRAKING_FULL_MS2", 0.0)
    previous = Settings.instance().acc_gap_level
    Settings.instance().acc_gap_level = level
    try:
        ctrl = AdaptiveCruiseController()
        s0, t_h = ctrl.config.s0_m, T_HEADWAY_BY_LEVEL_S[level]
        clock = [100.0]
        monkeypatch.setattr(acc_controller.time, "monotonic", lambda: clock[0])
        v_e = v0_kmh / 3.6
        v_l, v_end = v_e - 5.0, v_e - 5.0 - 2.4
        x_e, x_l, a_truck = 0.0, 1.6 * (s0 + v_e * t_h) + 5.0 * 2.0, 0.0
        positions: deque[tuple[float, float]] = deque(maxlen=20)
        chain = [None] * 7 + [False, 0.0]
        seen, t, next_radar, worst = (v_l, 0.0), 0.0, 0.0, math.inf
        while t < 20.0:
            if t >= 2.0 and v_l > v_end:
                v_l = max(v_end, v_l - 2.4 * DT)
            x_l += v_l * DT
            if t >= next_radar - 1e-9:
                next_radar += 0.05
                positions.append((t, x_l))
                raw = v_l
                if len(positions) >= 2:
                    tm = sum(p[0] for p in positions) / len(positions)
                    xm = sum(p[1] for p in positions) / len(positions)
                    raw = (sum((p[0] - tm) * (p[1] - xm) for p in positions)
                           / sum((p[0] - tm) ** 2 for p in positions))
                out = _smooth_vehicle_kinematics(raw, raw, 100.0 + t, 0.05, *chain)
                chain = [out[0], out[1], out[3], out[4], out[5], out[6], out[7], out[8], out[9]]
                seen = (out[7], out[5])
            lead = _LeadSnapshot(vid=1, dist_m=x_l - x_e, v_lead_ms=seen[0], a_lead_ms2=seen[1],
                                 score=6.0)
            monkeypatch.setattr(ctrl, "_read_acc_snapshot",
                                lambda lead=lead: ([lead], None, BlinkerState()))
            clock[0] += DT
            cap = ctrl.accel_cap_ms2(v_e)
            if t >= 2.0:
                # Ego held its speed up to the lead's brake, so the onset state is exact.
                a_truck += (min(cap, 1.0) - a_truck) * (1.0 - math.exp(-DT / 0.35))
                v_e = max(0.0, v_e + a_truck * DT)
                worst = min(worst, (x_l - x_e) / (s0 + v_e * t_h))
            x_e += v_e * DT
            t += DT
        return worst
    finally:
        Settings.instance().acc_gap_level = previous


@pytest.mark.parametrize("level, v0_kmh", [(3, 90.0), (2, 60.0)])
def test_ego_keeps_its_gap_after_a_far_lead_slows_briefly(level, v0_kmh, monkeypatch):
    """Card 152: the limit kept easing the brake on the decaying accel and ego settled at 0.7-0.8."""
    new = _far_lead_short_slowdown(level, v0_kmh, monkeypatch)
    old = _far_lead_short_slowdown(level, v0_kmh, monkeypatch, old_gate=True)
    assert old < 0.85, "the fixture must reproduce ego settling inside its gap"
    assert new > 0.9


def test_a_matched_or_critical_brake_keeps_the_law():
    cfg = _cfg()
    mild = _lead(30.0, 22.0, a_lead=-2.0)
    assert brake_follow_limit(cfg, -4.0, mild, 22.0, 1.1, gap_excess_ms=0.0) == -4.0
    critical = _lead(20.0, 10.0, a_lead=-6.0)
    assert brake_follow_limit(cfg, -6.4, critical, 25.0, 1.1, gap_excess_ms=0.0) == -6.4


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
