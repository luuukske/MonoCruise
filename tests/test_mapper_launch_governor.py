"""Mapper launch governor: gas ceiling while a slipping clutch leaves the loop open."""
from __future__ import annotations

import math
from collections import deque

import pytest

from core.sending_thread import accel_to_pedals as atp
from core.sending_thread import launch_governor as lg
from core.sending_thread.accel_to_pedals import AccelToPedals
from core.settings import Settings

DT = 0.02
MASS_KG = 17_000.0
# Launch plant fitted on 62 logged clutch-slip launches (17 t rigs):
# accel = 0.25 + 2.75 * throttle(t - 0.3 s) - 0.68 * g sin(grade).
PLANT_CREEP_MS2 = 0.25
PLANT_GAIN = 2.75
PLANT_GRADE_COEF = 0.68
PLANT_DEAD_S = 0.3
PLANT_LAG_S = 0.15
CLUTCH_RELEASE_MS = 2.5
BID_START_S = 0.3
BID_JERK = 2.5


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    clk = _FakeClock()
    monkeypatch.setattr("core.sending_thread.accel_to_pedals.time.monotonic", clk)
    return clk


@pytest.fixture(autouse=True)
def crr():
    inst = Settings.instance()
    old = inst.mapper_rolling_resistance
    inst.mapper_rolling_resistance = 0.07
    try:
        yield
    finally:
        inst.mapper_rolling_resistance = old


@pytest.fixture()
def governor_off(monkeypatch):
    def _off():
        monkeypatch.setattr(lg, "ARM_SPEED_MS", -1.0)
    return _off


def _pitch_norm(rad: float) -> float:
    return rad / (2.0 * math.pi)


def _launch(clock, bid: float, capacity: float, grade_rad: float, dur_s: float = 4.0) -> dict:
    """Closed loop from rest: ACC-like bid ramp, fitted plant, clutch slipping below 2.5 m/s."""
    mapper = AccelToPedals()
    delay = deque([0.0] * (int(PLANT_DEAD_S / DT) + 1), maxlen=int(PLANT_DEAD_S / DT) + 1)
    v = a = t = 0.0
    spd_smooth = 0.0
    peak_slip_a = peak_gas = 0.0
    t90 = None
    grade_accel = PLANT_GRADE_COEF * atp.GRAVITY_MS2 * math.sin(grade_rad)
    max_accel = capacity * atp.weight_factor(MASS_KG, True)
    while t < dur_s:
        clock.t += DT
        t += DT
        wanted = 0.0 if t < BID_START_S else min(bid, BID_JERK * (t - BID_START_S))
        raw = (v - spd_smooth) / 0.30
        spd_smooth += (1.0 - math.exp(-DT / 0.30)) * (v - spd_smooth)
        clutch = 0.97 if v < CLUTCH_RELEASE_MS else 0.0
        gas = mapper.step(
            wanted, raw, v, MASS_KG, True,
            max_accel_ms2=max_accel,
            cruise_commanding=True,
            road_pitch=_pitch_norm(grade_rad),
            gear_dashboard=6,
            game_clutch=clutch,
        ).gas
        delay.appendleft(gas)
        drive = PLANT_CREEP_MS2 + PLANT_GAIN * delay[-1] - grade_accel
        a += (1.0 - math.exp(-DT / PLANT_LAG_S)) * (drive - a)
        v = max(0.0, v + a * DT)
        if clutch > 0.0:
            peak_slip_a = max(peak_slip_a, a)
        peak_gas = max(peak_gas, gas)
        if t90 is None and t > BID_START_S and a >= 0.9 * bid:
            t90 = t - BID_START_S
    return {"peak_slip_a": peak_slip_a, "peak_gas": peak_gas, "t90": t90}


def test_harness_reproduces_the_logged_rail(clock, governor_off):
    # 2026-09-28: bid 0.3-0.4 on a 5% grade, capacity 1.31, gas went to 1.0.
    governor_off()
    out = _launch(clock, bid=0.35, capacity=1.31, grade_rad=0.05)
    assert out["peak_gas"] >= 0.9
    assert out["peak_slip_a"] >= 2.0


def test_small_launch_bid_does_not_rail_the_gas(clock):
    out = _launch(clock, bid=0.35, capacity=1.31, grade_rad=0.05)
    assert out["peak_gas"] <= 0.6
    assert out["peak_slip_a"] <= 1.3


@pytest.mark.parametrize("bid", [0.35, 0.8, 1.5])
def test_flat_launch_tracks_the_bid(clock, bid):
    out = _launch(clock, bid=bid, capacity=3.0, grade_rad=0.0)
    assert out["peak_slip_a"] <= 1.15 * bid + 0.1
    # Reactivity guard: the bid itself needs 0.14 to 0.54 s to ramp in.
    assert out["t90"] is not None and out["t90"] <= 1.2


def test_hill_start_begins_at_the_holding_pedal():
    gov = lg.LaunchGovernor()
    kw = dict(dt=DT, enabled=True, clutch_pressed=True, speed_ms=0.0, factor=0.0,
              accel_ms2=0.0, hold_pedal=0.4, prev_gas=0.0, mapper_gas=1.0, gain_scale=1.0)
    gov, cap = lg.launch_gas_cap(gov, wanted_ms2=0.0, **kw)
    assert gov.phase == lg.SLIP and cap == pytest.approx(0.0)
    gov, cap = lg.launch_gas_cap(gov, wanted_ms2=0.05, **kw)
    assert cap >= 0.4


def _open_loop(clock, ticks, *, cap_mode=False, freeze_trim=False):
    mapper = AccelToPedals()
    gas = []
    for wanted, raw, speed, clutch, gear, pitch_rad in ticks:
        clock.t += DT
        gas.append(mapper.step(
            wanted, raw, speed, MASS_KG, True,
            max_accel_ms2=1.31 * atp.weight_factor(MASS_KG, True),
            cruise_commanding=True,
            road_pitch=_pitch_norm(pitch_rad),
            gear_dashboard=gear,
            game_clutch=clutch,
            cap_mode=cap_mode,
            freeze_trim=freeze_trim,
        ).gas)
    return gas, mapper


def _launch_ticks():
    ticks = [(0.0, 0.0, 0.0, 0.97, 6, 0.05)] * 15
    v = 0.0
    for k in range(150):
        wanted = min(0.4, 0.05 * k)
        raw = 1.2 if k > 20 else 0.0
        v += raw * DT
        clutch = 0.97 if v < 1.5 else 0.0
        ticks.append((wanted, raw, v, clutch, 6, 0.05))
    return ticks


def test_governor_only_lowers_gas(clock, governor_off):
    on, _ = _open_loop(clock, _launch_ticks())
    governor_off()
    off, _ = _open_loop(clock, _launch_ticks())
    assert all(g_on <= g_off + 1e-9 for g_on, g_off in zip(on, off))
    assert min(g_on - g_off for g_on, g_off in zip(on, off)) < -0.3


@pytest.mark.parametrize("mode", ["cap_mode", "freeze_trim"])
def test_limiter_and_aeb_are_untouched(clock, governor_off, mode):
    on, _ = _open_loop(clock, _launch_ticks(), **{mode: True})
    governor_off()
    off, _ = _open_loop(clock, _launch_ticks(), **{mode: True})
    assert on == off


def test_a_gearshift_while_moving_does_not_arm(clock, governor_off):
    ticks = [(0.8, 0.8, 6.0, 0.0, 6, 0.0)] * 60 + [(0.8, -0.6, 6.0, 0.97, 6, 0.0)] * 20
    ticks += [(0.8, 0.8, 6.0, 0.0, 7, 0.0)] * 60
    on, mapper = _open_loop(clock, ticks)
    assert mapper._launch.phase == lg.IDLE
    governor_off()
    off, _ = _open_loop(clock, ticks)
    assert on == off


def test_neutral_trajectory_does_not_step_in_on_gear_engage(clock, governor_off):
    # Auto neutral: the bid arrives in N, the gas trajectory runs up, then D engages.
    ticks = [(0.5, 0.0, 0.0, 0.0, 0, 0.0)] * 50 + [(0.5, 0.0, 0.0, 0.97, 6, 0.0)] * 5
    on, _ = _open_loop(clock, ticks)
    governor_off()
    off, _ = _open_loop(clock, ticks)
    assert off[50] >= 0.3
    assert on[50] <= 0.2


def test_hands_back_once_the_clutch_closes(clock, governor_off):
    ticks = _launch_ticks() + [(0.4, 0.4, 3.0, 0.0, 6, 0.05)] * 150
    on, mapper = _open_loop(clock, ticks)
    assert mapper._launch.phase == lg.IDLE
    governor_off()
    off, _ = _open_loop(clock, ticks)
    assert on[-20:] == pytest.approx(off[-20:])
