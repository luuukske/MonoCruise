"""The convoy plant's gearbox against the mapper debug log. See tools/acc_platoon/README.md, "Gear shifts".

An AMT upshift cuts the engine's drive for ~1.25 s and the length does not grow with
throttle, acceleration, gear or speed (|corr| <= 0.14 over 2128 logged upshifts). A
torque-converter automatic dips instead. The log constants below are what
`shifts.measure` read from the local `accel_to_pedals_debug.csv` on 2026-10-08.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from tools.acc_platoon import shifts
from tools.acc_platoon.plant import AMT, POWERSHIFT, TruckPlant, TruckSpec

DT = 1.0 / 60.0
LOG_AMT_BELOW_HALF_S = (0.90, 1.25, 1.55)
LOG_AMT_SHIFTS = 2128
LOG_POWERSHIFT_FLOOR = 0.68
QUANTILE_TOL_S = 0.15
PROFILE_TOL = 0.2
SCHEDULE_TOL = 0.04
LOG_PATH = Path(__file__).resolve().parents[2] / shifts.LOG_NAME


def _spec(box=AMT) -> TruckSpec:
    return TruckSpec("probe", mass_t=26.0, power_kw=400.0, brake_ms2=11.0, tau_brake_s=0.25, gearbox=box)


def test_an_amt_upshift_cuts_drive_as_long_as_the_log_shows():
    got = shifts.quantiles(shifts.model_upshifts(AMT, launches=40))
    for g, want in zip(got, LOG_AMT_BELOW_HALF_S):
        assert abs(g - want) <= QUANTILE_TOL_S, (got, LOG_AMT_BELOW_HALF_S)


def test_shift_length_does_not_grow_with_throttle():
    """The log shows no dependence; do not add one without new data that does."""
    gentle, hard = [], []
    for seed in range(30):
        for demand, out in ((0.6, gentle), (1.6, hard)):
            plant = TruckPlant(_spec(), 0.0, 0.0, DT, random.Random(seed))
            ts, vs, gs, t = [], [], [], 0.0
            while plant.v < 80.0 / 3.6 and t < 120.0:
                plant.step(demand, DT)
                t += DT
                if len(ts) == 0 or t - ts[-1] >= shifts.LOG_PERIOD_S:
                    ts.append(t)
                    vs.append(plant.v)
                    gs.append(plant.gear + 1)
            out.extend(shifts.measure(ts, vs, gs))
    assert abs(shifts.quantiles(gentle, (50,))[0] - shifts.quantiles(hard, (50,))[0]) <= QUANTILE_TOL_S


def test_a_torque_converter_box_dips_instead_of_cutting():
    measured = shifts.model_upshifts(POWERSHIFT, launches=30)
    assert shifts.quantiles(measured, (90,))[0] < 0.5
    assert abs(min(shifts.median_profile(measured)) - LOG_POWERSHIFT_FLOOR) <= PROFILE_TOL


def test_upshifts_follow_the_logged_schedule():
    plant = TruckPlant(_spec(), 0.0, 0.0, DT)
    seen: list[tuple[int, float]] = []
    t = 0.0
    while plant.v < 95.0 / 3.6 and t < 120.0:
        before = plant.shifts
        plant.step(1.0, DT)
        t += DT
        if plant.shifts > before:
            seen.append((plant.shifts - 1, plant.v * 3.6))
    assert [k for k, _ in seen] == list(range(len(AMT.up_kmh)))
    for k, kmh in seen:
        assert AMT.up_kmh[k] * (0.97 - SCHEDULE_TOL) <= kmh <= AMT.up_kmh[k] * (1.03 + SCHEDULE_TOL), seen


@pytest.mark.parametrize("kmh", [10.0, 25.0, 40.0, 55.0, 70.0, 80.0, 90.0])
def test_a_truck_holding_its_speed_does_not_hunt_gears(kmh):
    v = kmh / 3.6
    plant = TruckPlant(_spec(), 0.0, v, DT)
    for _ in range(int(30.0 / DT)):
        plant.step(1.5 * (v - plant.v), DT)
    assert plant.shifts <= 1


def test_asking_for_power_below_the_kickdown_speed_shifts_down():
    """Slowed from 80 to 62 km/h the box keeps its gear; full throttle then kicks it down."""
    plant = TruckPlant(_spec(), 0.0, 80.0 / 3.6, DT)
    top = plant.gear
    while plant.v > 62.0 / 3.6:
        plant.step(-1.5, DT)
    assert plant.gear == top
    for _ in range(int(0.5 / DT)):
        plant.step(2.0, DT)
    assert plant.shifts == 1


def test_braking_is_not_cut_by_a_shift():
    """A downshift while braking must not take any brake away."""
    plant = TruckPlant(_spec(), 0.0, 70.0 / 3.6, DT)
    decels = []
    for _ in range(int(6.0 / DT)):
        plant.step(-3.0, DT)
        if plant.v > 1.0:
            decels.append(plant.a)
    assert plant.shifts >= 1
    assert max(decels[60:]) < -2.5


@pytest.mark.needs_clips
@pytest.mark.skipif(not LOG_PATH.is_file(), reason="no local mapper debug log")
def test_the_model_reads_like_the_local_mapper_log():
    """Re-reads the log: the constants above must still describe it, and the plant must match it."""
    logged = shifts.amt(shifts.measure(*shifts.read_log(LOG_PATH)))
    assert len(logged) >= LOG_AMT_SHIFTS // 2
    model = shifts.model_upshifts(AMT, launches=40)
    for got, want in zip(shifts.quantiles(model), shifts.quantiles(logged)):
        assert abs(got - want) <= QUANTILE_TOL_S
    grid = shifts.grid()
    for g, m, w in zip(grid, shifts.median_profile(model), shifts.median_profile(logged)):
        if shifts.DIP_FROM_S < g < 1.0:
            assert abs(m - w) <= PROFILE_TOL, (g, m, w)
