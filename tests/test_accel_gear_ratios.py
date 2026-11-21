"""Gas capacity shaped by the truck's own gear ratios from the SDK. See core/sending_thread/README.md."""
from __future__ import annotations

import pytest

import core.sending_thread.gear_shape as gs
import core.sending_thread.pedal_capacity as pc
from core.sending_thread.accel_to_pedals import weight_factor
from core.telemetry_thread.thread import forward_gear_ratios

FH6 = (4.7, 2.21, 1.53, 1.0, 0.76, 0.67)   # Volvo FH6 6-speed, read from the SDK 2026-11-21
FH6_AXLE = 3.91
TWELVE = (14.94, 11.73, 9.04, 7.09, 5.54, 4.35, 3.44, 2.70, 2.08, 1.63, 1.27, 1.00)
TWELVE_AXLE = 2.64
MASS = 20_000.0
DT = 0.033
SPEED = 20.0


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


def _settings(**values):
    saved: dict = {}

    class _S:
        pedal_capacity_brake_scale = 0.0
        pedal_capacity_brake_model = pc._BRAKE_MODEL_VERSION
        pedal_capacity_max_brake_ms2 = 0.0
        pedal_capacity_max_accel_ms2 = 0.0
        pedal_capacity_accel_anchor_gain_ms2 = 0.0
        pedal_capacity_accel_ratio_step = 0.0
        pedal_capacity_accel_zero_offset_ms2 = 0.0
        pedal_capacity_accel_mass_model = pc._ACCEL_MASS_MODEL_VERSION
        mapper_brake_scale_ms2 = 6.5

        @staticmethod
        def save_async(values=None):
            saved.update(values or {})

    for key, value in values.items():
        setattr(_S, key, value)
    return _S, saved


@pytest.fixture(autouse=True)
def _no_settings_io(monkeypatch):
    stub, _ = _settings()
    monkeypatch.setattr(pc, "Settings", stub)


def _tracker(anchor: float = 1.5) -> pc.PedalCapacityTracker:
    t = pc.PedalCapacityTracker()
    t._accel_anchor_gain_ms2 = anchor
    t._accel_ratio_step = 1.27
    t._accel_zero_offset_ms2 = 0.5
    return t


def _feed(t, clk, gas, accel, gear, ticks, wet=False):
    for _ in range(ticks):
        clk.t += DT
        t.update_accel(gas, accel, SPEED, 0.0, 0.0, gear, MASS, False,
                       road_load_ms2=0.0, wet_road=wet)


def _per_ratio(t, gear):
    return t.accel_slope_for_gear(gear) / t.gear_total_ratio(gear)


def test_gear_strength_follows_the_trucks_own_ratios():
    """On the FH6 6-speed (2026-11-21, dry, 17 t) the pull per unit of gear ratio held
    1.2 to 1.9 from gear 1 to 6, while one learned step per gear was off up to 2x low."""
    t = _tracker()
    t.set_gear_ratios(FH6, FH6_AXLE)
    for gear, ratio in enumerate(FH6, 1):
        expected = 1.5 * ratio * FH6_AXLE / gs.REF_TOTAL
        assert t.accel_slope_for_gear(gear) == pytest.approx(expected)


def test_without_usable_ratios_the_gear_number_shape_is_unchanged():
    t = _tracker()
    for bad in (((), 0.0), ((4.7, 0.0), 3.91), (FH6, float("nan")), (FH6, -1.0)):
        t.set_gear_ratios(*bad)
        for gear in range(1, 15):
            assert t.accel_slope_for_gear(gear) == pytest.approx(1.5 * 1.27 ** (6 - gear))
        assert t.gear_total_ratio(3) is None


def test_ratios_that_go_missing_keep_the_last_good_set():
    t = _tracker()
    t.set_gear_ratios(FH6, FH6_AXLE)
    fourth = t.accel_slope_for_gear(4)
    t.set_gear_ratios((), 0.0)
    assert t.accel_slope_for_gear(4) == pytest.approx(fourth)


def test_a_gearbox_swap_moves_the_map_at_once():
    """The FH's 14-speed was swapped for a 6-speed on 2026-11-20 and one shared shape took
    the rest of the drive to follow. Strength per unit of ratio does not care which box."""
    t = _tracker()
    t.set_gear_ratios(TWELVE, TWELVE_AXLE)
    twelve_top = _per_ratio(t, 12)
    t.set_gear_ratios(FH6, FH6_AXLE)
    assert _per_ratio(t, 6) == pytest.approx(twelve_top)
    assert _per_ratio(t, 1) == pytest.approx(twelve_top)
    assert t.accel_anchor_ms2 == pytest.approx(1.5), "a swap must not move the learned strength"


def test_one_gear_teaches_every_gear(clock):
    """Strength learned in gear 2 carries to gear 6. The gear-number shape needed its step
    learned as well, at about a twentieth of the anchor's rate that far from gear 6."""
    truth = 2.0
    t = _tracker(anchor=1.0)
    t.set_gear_ratios(FH6, FH6_AXLE)
    wf = weight_factor(MASS, False)
    slope_2 = truth * FH6[1] * FH6_AXLE / gs.REF_TOTAL / wf
    _feed(t, clock, gas=0.8, accel=0.5 + slope_2 * 0.8, gear=2, ticks=4000)
    assert t.accel_samples_accepted > 0
    assert t.accel_slope_for_gear(2) / wf == pytest.approx(slope_2, rel=0.03)
    sixth = truth * FH6[5] * FH6_AXLE / gs.REF_TOTAL
    assert t.accel_slope_for_gear(6) == pytest.approx(sixth, rel=0.15)
    assert gs.EXP_MIN <= t.accel_ratio_exponent <= gs.EXP_MAX


def test_a_wet_road_teaches_the_gas_map_nothing(clock):
    """On the snowy drive of 2026-11-20 full-pedal pulls in gears 1 and 2 read a quarter to
    a half of the dry road's: the wheels spun, the engine did not get weaker."""
    t = _tracker()
    t.set_gear_ratios(FH6, FH6_AXLE)
    _feed(t, clock, gas=1.0, accel=0.8, gear=1, ticks=300, wet=True)
    assert t.accel_anchor_ms2 == pytest.approx(1.5)
    assert t.accel_samples_accepted == 0
    assert t.last_accel_gate == "wet"


def test_an_old_anchor_moves_to_the_ratios_once_keeping_gear_6(monkeypatch):
    """The persisted anchor was quoted at gear 6. The first truck with ratios keeps its
    gear 6 map exactly and every other gear moves to the ratios; it happens once."""
    stub, saved = _settings(
        pedal_capacity_accel_anchor_gain_ms2=0.88,
        pedal_capacity_accel_ratio_step=1.1795,
        pedal_capacity_accel_zero_offset_ms2=0.657,
    )
    monkeypatch.setattr(pc, "Settings", stub)
    t = pc.PedalCapacityTracker()
    t.load_persisted(10.0, 2.0)
    before = t.accel_slope_for_gear(6)
    t.set_gear_ratios(FH6, FH6_AXLE)
    assert t.accel_slope_for_gear(6) == pytest.approx(before)
    assert saved["pedal_capacity_accel_gear_model"] == gs.GEAR_MODEL_VERSION
    assert saved["pedal_capacity_accel_anchor_gain_ms2"] == pytest.approx(
        0.88 * gs.REF_TOTAL / (FH6[5] * FH6_AXLE), abs=1e-3,
    )

    saved.clear()
    t.set_gear_ratios(TWELVE, TWELVE_AXLE)
    assert saved == {}, "a later truck must not migrate again"

    stub.pedal_capacity_accel_anchor_gain_ms2 = 1.345
    stub.pedal_capacity_accel_gear_model = gs.GEAR_MODEL_VERSION
    restarted = pc.PedalCapacityTracker()
    restarted.load_persisted(10.0, 2.0)
    restarted.set_gear_ratios(FH6, FH6_AXLE)
    assert restarted.accel_anchor_ms2 == pytest.approx(1.345)
    assert saved == {}


def test_the_sdk_ratios_are_cut_to_the_gear_count():
    raw = {"gears": 6, "gearRatiosForward": list(FH6) + [0.0] * 18}
    assert forward_gear_ratios(raw) == FH6
    assert forward_gear_ratios({}) == ()
    assert forward_gear_ratios({"gears": 30, "gearRatiosForward": [2.0, 1.0]}) == (2.0, 1.0)
    assert forward_gear_ratios({"gears": None, "gearRatiosForward": None}) == ()
