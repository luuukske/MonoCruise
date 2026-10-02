"""Brake landing: once the lead's smoothed speed stops falling, ACC brakes no harder
than it takes to reach that speed. The brake used to stay on while ego fell far
below a lead that had long stopped slowing. See core/acc/ACC_ARCHITECTURE.md §13.4."""
from __future__ import annotations

import math
import random
from collections import deque

import pytest

from core.cruise_control_thread import acc_controller
from core.cruise_control_thread.acc_controller import AdaptiveCruiseController, _LeadSnapshot
from core.cruise_control_thread.blinker_arbitration import BlinkerState
from core.cruise_control_thread.brake_landing import BrakeLanding
from core.radar.traffic import _smooth_vehicle_kinematics
from core.settings import Settings

DT = 1.0 / 60.0


def _cfg():
    return AdaptiveCruiseController().config


def _landed(law, v_ego, v_lead, dist, ego_decel, trend=0.0, gap_error=0.0, cfg=None,
            t_headway=1.5):
    """Prime 0.6 s of history, then land `law`. `gap_error` closes the gap faster than seen."""
    cfg = cfg or _cfg()
    landing = BrakeLanding()
    n = int(0.6 / DT)
    for i in range(n + 1):
        t = i * DT
        age = (n - i) * DT
        v = v_ego - ego_decel * age
        landing.track_ego(v, DT)
        vl = v_lead - trend * age
        d = dist - (v_lead - v_ego - gap_error) * age
        raw = _LeadSnapshot(vid=1, dist_m=d, v_lead_ms=vl, a_lead_ms2=0.0, score=6.0)
        smooth = _LeadSnapshot(vid=1, dist_m=d, v_lead_ms=vl, a_lead_ms2=0.0, score=6.0,
                               conf=1.0, a_lead_ff_ms2=0.0)
        out = landing.step(cfg, law, raw, smooth, v, t_headway, t)
    return out


def test_landing_never_adds_braking_and_never_pulls():
    rng = random.Random(4242)
    for _ in range(4000):
        law = rng.uniform(-6.55, -0.01)
        out = _landed(law, rng.uniform(5.0, 30.0), rng.uniform(4.0, 30.0),
                      rng.uniform(5.0, 80.0), rng.uniform(-6.0, 0.0), rng.uniform(-2.0, 0.5))
        assert law - 1e-12 <= out <= 0.0


def test_a_brake_that_already_lands_ego_lets_go():
    """Ego 1 m/s over a lead that stopped slowing, truck at -5: that alone lands it."""
    out = _landed(-5.0, 21.0, 20.0, 40.0, -5.0)
    assert out > -1.0


@pytest.mark.parametrize("kw", [
    dict(trend=-1.5),  # the lead is still slowing
    dict(ego_decel=-0.2),  # nothing to land
    dict(v_lead=2.0, v_ego=3.0),  # a stop, not a landing
    dict(gap_error=3.0),  # the gap closes faster than the lead speed allows
], ids=["lead-still-slowing", "truck-not-braking", "slow-lead", "gap-disagrees"])
def test_landing_stays_out_of_the_way(kw):
    args = dict(law=-5.0, v_ego=21.0, v_lead=20.0, dist=40.0, ego_decel=-5.0)
    args.update(kw)
    assert _landed(**args) == -5.0


def test_horizon_zero_disables_landing():
    cfg = _cfg()
    cfg.landing_horizon_s = 0.0
    assert _landed(-5.0, 21.0, 20.0, 40.0, -5.0, cfg=cfg) == -5.0


def test_a_short_gap_lands_below_the_lead_to_reopen_it():
    """397148fd: landing on the lead's speed alone kept ego parked inside its gap."""
    wanted = _cfg().s0_m + 20.0 * 1.5
    at_gap = _landed(-4.0, 22.0, 20.0, wanted, -2.0)
    short = _landed(-4.0, 22.0, 20.0, wanted - 15.0, -2.0)
    assert short < at_gap - 0.3


def _follow_through_the_radar(level: int, v0_kmh: float, decel: float, dv_kmh: float,
                              monkeypatch, old: bool = False):
    """Closed loop with the lead seen through the ACC radar chain. Returns (undershoot km/h, min gap)."""
    previous = Settings.instance().acc_gap_level
    Settings.instance().acc_gap_level = level
    try:
        ctrl = AdaptiveCruiseController()
        if old:
            cfg = ctrl.config
            cfg.j_release_tau_s = cfg.j_onset_tau_s = cfg.landing_horizon_s = cfg.gas_pace_s = 0.0
            cfg.at_clamp_slam = True
            cfg.follow_share = 0.0
        clock = [100.0]
        monkeypatch.setattr(acc_controller.time, "monotonic", lambda: clock[0])
        v0, v_end = v0_kmh / 3.6, (v0_kmh - dv_kmh) / 3.6
        x_l = ctrl.config.s0_m + v0 * acc_controller.T_HEADWAY_BY_LEVEL_S[level]
        x_e, v_e, v_l, a_truck = 0.0, v0, v0, 0.0
        positions: deque[tuple[float, float]] = deque(maxlen=20)
        chain = [None] * 7 + [False, 0.0]
        seen = (v0, 0.0)
        t, next_radar, v_min, gap_min = 0.0, 0.0, v0, math.inf
        while t < 16.0:
            if t >= 3.0 and v_l > v_end:
                v_l = max(v_end, v_l - decel * DT)
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
                (s_ema, acc, _corr, hist, a_ema, a_acc, a_hist, a_speed,
                 still, rel) = _smooth_vehicle_kinematics(raw, raw, 100.0 + t, 0.05, *chain)
                chain = [s_ema, acc, hist, a_ema, a_acc, a_hist, a_speed, still, rel]
                seen = (a_speed, a_acc)
            lead = _LeadSnapshot(vid=1, dist_m=x_l - x_e, v_lead_ms=seen[0],
                                 a_lead_ms2=seen[1], score=6.0)
            monkeypatch.setattr(ctrl, "_read_acc_snapshot",
                                lambda lead=lead: ([lead], None, BlinkerState()))
            clock[0] += DT
            cap = ctrl.accel_cap_ms2(v_e)
            a_truck += (min(cap, 1.0) - a_truck) * (1.0 - math.exp(-DT / 0.3))
            v_e = max(0.0, v_e + a_truck * DT)
            x_e += v_e * DT
            if t > 3.0:
                v_min = min(v_min, v_e)
            gap_min = min(gap_min, x_l - x_e)
            t += DT
        return (v_end - v_min) * 3.6, gap_min
    finally:
        Settings.instance().acc_gap_level = previous


@pytest.mark.parametrize("level, decel", [(2, 4.0), (3, 6.0)])
def test_ego_lands_on_a_lead_that_braked_then_held(level, decel, monkeypatch):
    """The TruckersMP complaint: the lead brakes 90 -> 65 km/h and holds."""
    under, gap = _follow_through_the_radar(level, 90.0, decel, 25.0, monkeypatch)
    old_under, old_gap = _follow_through_the_radar(level, 90.0, decel, 25.0, monkeypatch, old=True)
    assert old_under > 12.0, "the fixture must reproduce the hangover"
    assert under < 0.5 * old_under
    assert gap > old_gap - 1.0, "landing must not buy the speed back with the gap"
