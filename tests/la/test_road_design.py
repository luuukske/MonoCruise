"""The servo's second design for the road: blended in with the game's centring, picked by the calibration (model only)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.la.actuator import WheelActuator, WheelTuning, road_overrides
from core.la.runner import AssistRunner
from core.la.wheel_learner import WheelModel, learned_overrides
from core.lane_departure.lka_runner import LkaRunner
from core.settings import Settings
from tools.la_bench import road
from tools.la_bench.calibrate import push_of, rim_of, tuning_of
from tools.la_bench.design import Choice
from tests.la.test_wheel_calibration import R3_PROFILE

STANDING = WheelTuning(inertia=415.0, bandwidth_hz=1.79)
ROAD = replace(STANDING, road_inertia=145.0, road_bandwidth_hz=3.0, road_damping=1.0)


def test_standing_the_road_design_changes_nothing_and_from_60_kmh_the_gains_are_its_own():
    plain, both = WheelActuator(STANDING), WheelActuator(ROAD)
    alone = WheelActuator(WheelTuning(inertia=145.0, bandwidth_hz=3.0, damping=1.0))
    for spring in (0.0, 5000.0, ROAD.road_from):
        assert both.gains(1.0, spring) == plain.gains(1.0, spring), "the standing wheel and slow roads: as designed"
    for kmh in (61.0, 90.0, 130.0):
        spring = road.spring_at(kmh)
        assert both.gains(1.0, spring) == pytest.approx(alone.gains(1.0, spring))
    shares = [both.road_share(road.spring_at(kmh)) for kmh in (30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 90.0)]
    assert shares == sorted(shares) and shares[0] == 0.0 and shares[-1] == 1.0 and 0.0 < shares[3] < 1.0
    middle = road.spring_at(50.0)
    assert plain.gains(1.0, middle)[2] < both.gains(1.0, middle)[2] < alone.gains(1.0, middle)[2]
    assert plain.road_share(road.spring_at(130.0)) == 0.0, "no road design: none"


def test_the_ring_guard_and_the_strength_scale_the_road_design_like_the_standing_one():
    spring = road.spring_at(90.0)
    full = WheelActuator(ROAD).gains(1.0, spring)
    assert WheelActuator(replace(ROAD, strength=0.5)).gains(1.0, spring)[0] == pytest.approx(0.5 * full[0])
    assert WheelActuator(replace(ROAD, strength=0.0)).gains(1.0, spring) == (0.0, 0.0, 0.0)
    cut = WheelActuator(ROAD)
    cut._scale = 0.5
    assert cut.gains(1.0, spring)[0] == pytest.approx(0.5 * full[0]) and cut.gains(1.0, spring)[1] < full[1]


def test_with_it_the_wheel_trails_its_target_as_before_the_calibration():
    """2026-11-11: 2.0 Hz on 650 trailed 0.20 s at 90 km/h, the calibration's 1.4 Hz on 415 trailed 0.64 s."""
    before, standing, both = road.lags(WheelTuning()), road.lags(STANDING), road.lags(ROAD)
    assert standing[90.0] > 0.28 and both[90.0] == pytest.approx(before[90.0], abs=0.03)
    assert both[60.0] < standing[60.0] and both[130.0] < 0.65 * standing[130.0]
    assert "soft" not in road.describe(ROAD)


def test_the_keys_are_bounded_and_need_both_an_inertia_and_a_bandwidth():
    assert road_overrides(0.0, 3.0, 1.0) == {} and road_overrides(145.0, 0.0, 1.0) == {}
    assert road_overrides(145.0, 9.0, 0.0) == {"road_inertia": 145.0, "road_bandwidth_hz": 3.0, "road_damping": 0.0}
    assert road_overrides(10.0, 2.5, 5.0) == {"road_inertia": 50.0, "road_bandwidth_hz": 2.5, "road_damping": 2.0}
    assert WheelActuator(WheelTuning(road_inertia=145.0, road_bandwidth_hz=3.0)).gains(1.0, 1e5)[1] > 0.0, "its damping: the standing one's"


def test_a_wheel_that_hunts_loses_its_road_design_and_a_calibrated_one_keeps_it():
    tune = replace(ROAD, ff_friction=120.0, learn=True)
    hunted = WheelModel(rung=2, starts=20, start_force=900.0, start_top=1500.0)
    assert learned_overrides(tune, hunted, pushes_set=True)["road_bandwidth_hz"] == 0.0
    assert "road_bandwidth_hz" not in learned_overrides(tune, hunted, True, calibrated=True)
    act = WheelActuator(tune)
    act.learned(hunted)
    assert act.road_share(road.spring_at(90.0)) == 0.0
    kept = WheelActuator(replace(tune, calibrated=True))
    kept.learned(hunted)
    assert kept.road_share(road.spring_at(90.0)) == 1.0


@pytest.mark.parametrize("runner", [AssistRunner, LkaRunner])
def test_la_and_the_lane_keeping_assist_both_read_the_road_keys(monkeypatch, runner):
    settings = Settings.instance()
    assert runner()._tuning()[1].wheel.road_bandwidth_hz == 0.0, "no keys: no road design"
    for key, value in (("la_road_inertia", 145.0), ("la_road_bandwidth_hz", 3.0), ("la_road_damping", 1.0)):
        monkeypatch.setattr(settings, key, value, raising=False)
    wheel = runner()._tuning()[1].wheel
    assert (wheel.road_inertia, wheel.road_bandwidth_hz, wheel.road_damping) == (145.0, 3.0, 1.0)


def test_the_calibration_writes_the_road_keys_and_zeros_clear_an_older_design():
    assert Choice(415.0, 1.79, 0.7, None, None, 0.0).road_config() == {
        "la_road_inertia": 0, "la_road_bandwidth_hz": 0.0, "la_road_damping": 0.0}
    picked = Choice(415.0, 1.79, 0.7, None, None, 0.0, road_inertia=145.0, road_bandwidth_hz=3.0, road_damping=1.0)
    assert picked.road_config() == {"la_road_inertia": 145, "la_road_bandwidth_hz": 3.0, "la_road_damping": 1.0}
    assert "la_road_inertia" not in picked.config(), "the wheel check's keys are unchanged"
    assert tuning_of(picked, 1.79).road_bandwidth_hz == 3.0


def test_the_search_takes_a_road_design_that_holds_on_the_calibrated_rim_and_none_that_rings(monkeypatch):
    """The R3 as calibrated on 2026-11-10: 3.0 Hz on 145 holds every model case, 3.0 Hz on its own inertia does not."""
    rim = rim_of(R3_PROFILE)
    base = tuning_of(Choice(415.0, 1.79, 0.7, None, None, 0.0, **push_of(R3_PROFILE)), 1.79)
    monkeypatch.setattr(road, "ROAD_BANDWIDTHS", (3.0,))
    monkeypatch.setattr(road, "ROAD_DAMPINGS", (1.0,))
    monkeypatch.setattr(road, "ROAD_INERTIA_SHARES", (1.0,))
    assert road.road_design(rim, base, R3_PROFILE.ffb_hz) is None
    monkeypatch.setattr(road, "ROAD_INERTIA_SHARES", (0.35,))
    said = []
    got = road.road_design(rim, base, R3_PROFILE.ffb_hz, say=said.append)
    assert got == (145.0, 3.0, 1.0) and "holds" in said[-1]
    with_it = road.pulses(rim, replace(base, road_inertia=got[0], road_bandwidth_hz=got[1], road_damping=got[2]))
    assert with_it.ok and with_it.past_m < 0.6
