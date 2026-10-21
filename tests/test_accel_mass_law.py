"""Gas capacity across a mass change. See core/sending_thread/README.md, "Gas capacity and mass".

2026-10-20: unhooking a 40.8 t load to a 10.5 t bobtail left the gas estimate 2-3x under the
truck, the gas map that much too steep, and CC and the limiter pumped the pedal at about 1 Hz.
"""
from __future__ import annotations

import math

import pytest

import core.sending_thread.accel_to_pedals as atp
import core.sending_thread.pedal_capacity as pc

GEAR = 10
OFFSET = 0.5
ROAD = 0.3
# Measured on full-throttle pulls matched on gear and speed, 17 t against 25-41 t.
MEASURED_EXP_LOW, MEASURED_EXP_HIGH = 0.75, 0.95


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    clk = _FakeClock()
    monkeypatch.setattr(pc.time, "monotonic", clk)
    return clk


class _FakeSettings:
    pedal_capacity_brake_scale = 1.0
    pedal_capacity_brake_model = pc._BRAKE_MODEL_VERSION
    pedal_capacity_max_accel_ms2 = 2.124
    pedal_capacity_accel_anchor_gain_ms2 = 2.0
    pedal_capacity_accel_ratio_step = 1.08
    pedal_capacity_accel_zero_offset_ms2 = OFFSET
    pedal_capacity_accel_mass_model = 0
    saves: list = []

    @classmethod
    def save(cls, values=None):
        cls.saves.append(dict(values or {}))


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    class S(_FakeSettings):
        saves: list = []

    monkeypatch.setattr(pc, "Settings", S)
    return S


def _true_slope(mass_kg: float, s20: float = 2.2, ratio: float = 1.08, gear: int = GEAR) -> float:
    """A truck whose pedal slope follows the measured law."""
    return s20 * ratio ** (pc._ANCHOR_GEAR - gear) * (mass_kg / 20_000.0) ** -atp._WEIGHT_MASS_EXP


def _believed_slope(t: pc.PedalCapacityTracker, mass_kg: float, gear: int = GEAR) -> float:
    return t.accel_slope_for_gear(gear) / pc.weight_factor(mass_kg, mass_kg > 12_000.0)


def _pull(t, clk, mass_kg, pedal=0.9, seconds=60.0, gear=GEAR):
    accel = OFFSET + _true_slope(mass_kg, gear=gear) * pedal - ROAD
    has_trailer = mass_kg > 12_000.0
    for _ in range(int(seconds / 0.01)):
        clk.t += 0.01
        t.update_accel(pedal, accel, 15.0, 0.0, 0.0, gear, mass_kg, has_trailer, road_load_ms2=ROAD)


def _learner(anchor: float = 1.4) -> pc.PedalCapacityTracker:
    t = pc.PedalCapacityTracker()
    t._accel_anchor_gain_ms2, t._accel_ratio_step, t._accel_zero_offset_ms2 = anchor, 1.08, OFFSET
    return t


def test_the_mass_exponent_is_the_measured_one():
    assert MEASURED_EXP_LOW <= atp._WEIGHT_MASS_EXP <= MEASURED_EXP_HIGH
    ratio = atp.weight_factor(41_000.0, True) / atp.weight_factor(17_300.0, True)
    assert ratio == pytest.approx((41.0 / 17.3) ** atp._WEIGHT_MASS_EXP)


def test_the_reference_mass_reads_the_same_under_both_laws():
    """The anchor is normalized at 20 t, so a 20 t rig keeps its map across the change."""
    for trailer in (False, True):
        assert atp.weight_factor(20_000.0, trailer) == pytest.approx(
            atp.legacy_weight_factor(20_000.0, trailer)
        )


def test_the_factor_is_not_clamped_inside_real_rigs():
    """The linear law's 1.85 ceiling sat at 37 t; a 60 t rig must still scale."""
    for mass in (8_000.0, 10_500.0, 40_800.0, 60_000.0):
        assert atp.weight_factor(mass, False) == pytest.approx(
            (mass / 20_000.0) ** atp._WEIGHT_MASS_EXP
        )


def test_unhooking_a_loaded_trailer_keeps_the_gas_estimate(clock):
    t = _learner()
    _pull(t, clock, 40_800.0)
    assert _believed_slope(t, 40_800.0) == pytest.approx(_true_slope(40_800.0), rel=0.03)
    assert _believed_slope(t, 10_500.0) == pytest.approx(_true_slope(10_500.0), rel=0.05)


def test_hooking_a_loaded_trailer_keeps_the_gas_estimate(clock):
    t = _learner(anchor=3.0)
    _pull(t, clock, 10_500.0, pedal=0.5)
    assert _believed_slope(t, 10_500.0) == pytest.approx(_true_slope(10_500.0), rel=0.03)
    assert _believed_slope(t, 40_800.0) == pytest.approx(_true_slope(40_800.0), rel=0.05)


def test_the_linear_law_read_the_bobtail_far_too_weak(clock, monkeypatch):
    """The 2026-10-20 defect, reproduced on the old law: the estimate the map divides by."""
    monkeypatch.setattr(pc, "weight_factor", atp.legacy_weight_factor)
    t = _learner()
    _pull(t, clock, 40_800.0)
    assert _believed_slope(t, 10_500.0) < 0.65 * _true_slope(10_500.0)


@pytest.mark.parametrize("mass_kg", [10_500.0, 17_300.0, 40_800.0])
def test_an_old_anchor_keeps_the_map_of_the_first_rig(clock, settings, mass_kg):
    t = pc.PedalCapacityTracker()
    t.load_persisted(baseline_brake=10.0, baseline_accel=1.0)
    trailer = mass_kg > 12_000.0
    before = t.accel_slope_for_gear(GEAR) / atp.legacy_weight_factor(mass_kg, trailer)
    clock.t += 0.01
    t.update_accel(0.9, 1.0, 15.0, 0.0, 0.0, GEAR, mass_kg, trailer, road_load_ms2=ROAD)
    after = t.accel_slope_for_gear(GEAR) / atp.weight_factor(mass_kg, trailer)
    assert after == pytest.approx(before)
    assert settings.saves[-1]["pedal_capacity_accel_mass_model"] == pc._ACCEL_MASS_MODEL_VERSION
    assert settings.saves[-1]["pedal_capacity_accel_anchor_gain_ms2"] == pytest.approx(
        t._accel_anchor_gain_ms2, abs=1e-3
    )


def test_a_migrated_anchor_is_not_migrated_again(clock, settings):
    settings.pedal_capacity_accel_mass_model = pc._ACCEL_MASS_MODEL_VERSION
    t = pc.PedalCapacityTracker()
    t.load_persisted(baseline_brake=10.0, baseline_accel=1.0)
    clock.t += 0.01
    t.update_accel(0.9, 1.0, 15.0, 0.0, 0.0, GEAR, 40_800.0, True, road_load_ms2=ROAD)
    assert t._accel_anchor_gain_ms2 == pytest.approx(2.0)
    assert settings.saves == []


def test_migration_waits_for_a_real_mass(clock, settings):
    """Telemetry can read 0 kg before the rig is known; that is no rig to keep the map of."""
    t = pc.PedalCapacityTracker()
    t.load_persisted(baseline_brake=10.0, baseline_accel=1.0)
    clock.t += 0.01
    t.update_accel(0.9, 1.0, 15.0, 0.0, 0.0, GEAR, 0.0, False, road_load_ms2=ROAD)
    assert t._accel_anchor_gain_ms2 == pytest.approx(2.0)
    assert t._accel_mass_model != pc._ACCEL_MASS_MODEL_VERSION


def test_every_shipped_save_carries_the_mass_model(clock, settings):
    """A save without it would let the next start migrate an already migrated anchor."""
    t = _learner()
    t._write(clock.t)
    assert settings.saves[-1]["pedal_capacity_accel_mass_model"] == pc._ACCEL_MASS_MODEL_VERSION
    assert math.isfinite(settings.saves[-1]["pedal_capacity_accel_anchor_gain_ms2"])
