"""The short wheel calibration: open-loop pushes on the standing wheel and a fit of the rim (model wheels only so far)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.la.actuator import WheelActuator, WheelTuning
from core.la.wheel_calibration import DONE, FAILED, CalTuning, WheelProfile, _fresh, _relay_segments
from core.la.wheel_learner import WheelModel, learned_overrides
from tools.la_bench.calibrate import record, replay, rim_of, run_sim
from tools.la_bench.plant import SIM_WHEELS

OLD_WHEEL_RUNS = Path(__file__).parent / "data" / "old_wheel_calibrations_2026-11-09.json"

DRIVE_OLD = replace(SIM_WHEELS["old"], spring=0.0, inertia=104.0, damping=315.0, stick=77.0, breakaway=139.0,
                    delay_s=0.021)
WHEELS = {
    "old": SIM_WHEELS["old"],
    "old fitted on the 2026-11-09 drive": DRIVE_OLD,
    "sticky": SIM_WHEELS["sticky"],
    "r3 read at 60 Hz": replace(SIM_WHEELS["r3"], reading_hz=60.0),
    "bare with 4 mlock of play": replace(SIM_WHEELS["bare"], backlash=0.004),
}


@pytest.mark.parametrize("name", sorted(WHEELS))
def test_it_measures_the_rim_in_a_few_seconds_and_leaves_the_wheel_near_where_it_was(name):
    p = WHEELS[name]
    cal, wheel, seconds = run_sim(p)
    pr = cal.profile
    assert cal.state == DONE and pr.pinned, cal.failure
    assert seconds < 20.0
    assert abs(wheel.reading) < 0.15
    assert pr.inertia == pytest.approx(p.inertia, rel=0.1)
    assert pr.damping == pytest.approx(p.damping, rel=0.1)
    assert pr.slide == pytest.approx(p.stick, rel=0.1)
    assert pr.spring == pytest.approx(p.spring, rel=0.1, abs=50.0)
    assert pr.delay_s == pytest.approx(p.delay_s, abs=0.009), "a smooth hold hid the delay on a heavy rim"
    assert pr.reading_hz == pytest.approx(p.reading_hz or 120.6, rel=0.03)
    if p.quantum:
        assert pr.quantum == pytest.approx(p.quantum, rel=0.01)
    if p.breakaway:
        assert pr.breakaway == pytest.approx(p.breakaway, rel=0.25)
    if p.backlash:
        assert pr.play == pytest.approx(p.backlash, rel=0.25), "the motor crossing its play read as a start of 45"


def test_old_trace_samples_from_before_the_game_came_back_do_not_skew_the_reading_rate():
    cal, _, _ = run_sim(SIM_WHEELS["old"], backlog_s=16.0)
    assert cal.profile.reading_hz == pytest.approx(60.0, rel=0.03), "2026-11-09 on the old wheel: 26 and 37 for 62"


@pytest.mark.parametrize("name", sorted(WHEELS))
def test_the_relay_swings_the_wheel_a_little_and_its_readings_join_the_fit(name):
    cal, _, _ = run_sim(WHEELS[name])
    first, end, rest = cal._relay
    _, xs = _fresh(cal._t, cal._steer, cal._seq, first, end)
    pr = cal.profile
    assert 0.003 < max(xs) - min(xs) < 0.07, "a relay at twice the start swung the old wheel's model 76 mlock"
    assert _relay_segments(cal._t, cal._steer, cal._seq, cal._relay, cal.tune, pr.quantum, pr.reading_hz)
    assert cal.servo.direction == 0, "the servo is off again once it is done"


def test_its_travel_is_capped():
    cal, _, _ = run_sim(SIM_WHEELS["old"])
    xs = cal._steer
    assert max(xs) - min(xs) < 2.0 * CalTuning().travel


def test_a_held_rim_stops_it_and_nothing_is_saved():
    cal, wheel, _ = run_sim(replace(SIM_WHEELS["r3"], stick=4000.0))
    assert cal.state == FAILED and "hands" in cal.failure and cal.profile is None
    assert cal.step(1e6) == 0.0


def test_an_off_centre_wheel_is_never_pushed():
    cal, wheel, _ = run_sim(SIM_WHEELS["old"], steer0=0.3)
    assert cal.state == FAILED and "centre" in cal.failure
    assert max(abs(s) for s in cal._out) == 0.0


def test_a_saved_run_analyses_the_same_again():
    cal, _, _ = run_sim(DRIVE_OLD)
    again = replay(record(cal))
    assert again == cal.profile


@pytest.mark.parametrize("run", range(4))
def test_the_old_wheel_s_relay_runs_pin_the_rim_the_drive_fitted(run):
    pr = replay(json.loads(OLD_WHEEL_RUNS.read_text(encoding="utf-8"))[run])
    assert pr.pinned, "without the relay the error only slid toward an end of the delay scan"
    assert 80.0 < pr.inertia < 115.0 and 0.02 < pr.delay_s < 0.04, "the 2026-11-09 drive fit: 104 and 21 ms"
    assert 500.0 < pr.damping < 950.0 and 150.0 < pr.slide < 300.0
    assert pr.reading_hz == pytest.approx(62.0, rel=0.03)


def test_a_profile_survives_the_config_and_garbage_reads_as_none():
    pr = replay(json.loads(OLD_WHEEL_RUNS.read_text(encoding="utf-8"))[0])
    back = WheelProfile.from_dict(json.loads(json.dumps(pr.to_dict())))
    assert back.pinned and back.direction == pr.direction
    assert (back.inertia, back.delay_s, back.ffb_hz) == pytest.approx((pr.inertia, pr.delay_s, pr.ffb_hz), rel=1e-5)
    assert not WheelProfile.from_dict({"inertia": "heavy", "damping": -5.0, "delay_s": float("nan")}).pinned
    assert not WheelProfile.from_dict(None).pinned


def test_the_design_search_gets_the_calibrated_rim():
    pr = replay(json.loads(OLD_WHEEL_RUNS.read_text(encoding="utf-8"))[0])
    rim = rim_of(pr)
    assert (rim.inertia, rim.damping, rim.stick, rim.delay_s) == (pr.inertia, pr.damping, pr.slide, pr.delay_s)
    assert rim.breakaway == pr.breakaway and rim.breakaway * (1.0 + rim.breakaway_spread) == pytest.approx(pr.breakaway_top)
    assert rim.quantum == pr.quantum and rim.reading_hz == pr.reading_hz
    assert rim_of(WheelProfile(damping=700.0, slide=200.0)) is None, "no design on a rim whose inertia is unknown"


def test_a_calibrated_wheel_never_steps_its_design_down():
    tune = WheelTuning(bandwidth_hz=1.75, inertia=150.0, ff_friction=240.0, learn=True)
    hunted = WheelModel(rung=3, starts=20, start_force=900.0, start_top=1500.0)
    assert learned_overrides(tune, hunted, pushes_set=True)["bandwidth_hz"] < 1.75
    assert "bandwidth_hz" not in learned_overrides(tune, hunted, True, calibrated=True), "2026-11-09: rung 2 wove 3 m"
    act = WheelActuator(replace(tune, calibrated=True))
    act.learned(hunted)
    assert (act.tune.bandwidth_hz, act.tune.inertia) == (1.75, 150.0)


SLOW = {
    "drive-fitted old": DRIVE_OLD,
    "r3 read at 60 Hz": replace(SIM_WHEELS["r3"], reading_hz=60.0),
    "friction rising with speed": replace(DRIVE_OLD, stick_speed=150.0),
}


@pytest.mark.parametrize("name", sorted(SLOW))
def test_the_creep_reads_the_friction_a_slow_move_meets(name):
    p = SLOW[name]
    pr = run_sim(p)[0].profile
    assert pr.slow_friction == pytest.approx(p.stick + p.stick_speed * 0.462, rel=0.2), "the fast slides over-read it"


def test_the_reading_jitter_is_read_off_the_slides():
    r3 = replace(SIM_WHEELS["r3"], reading_hz=60.0)
    assert run_sim(r3)[0].profile.jitter_s < 0.0015
    assert run_sim(replace(r3, reading_jitter_s=0.004))[0].profile.jitter_s == pytest.approx(0.004, abs=0.0015)


def test_the_model_rim_rises_from_the_creep_friction_to_the_slides():
    rim = rim_of(WheelProfile(direction=1, inertia=100.0, damping=500.0, delay_s=0.025, slide=200.0, slow_friction=150.0,
                              breakaway=300.0, breakaway_top=330.0))
    assert rim.stick + rim.stick_speed * 0.462 == pytest.approx(150.0, rel=0.01)
    assert rim.stick + rim.stick_speed == pytest.approx(200.0)


def _check(params, bandwidth, inertia):
    from core.la.design_check import DesignCheck
    from tools.la_bench.calibrate import drive_sim

    cal, wheel, _ = run_sim(params)
    tune = WheelTuning(inertia=inertia, bandwidth_hz=bandwidth, damping=0.7, damping_first=True, ff_inertia=params.inertia,
                       ff_damping=params.damping, ff_friction=params.stick * params.stick / params.breakaway,
                       plan_s=params.delay_s)
    ck = DesignCheck(tune, cal._direction)
    drive_sim(ck, wheel)
    return ck


def test_the_design_check_passes_a_gentle_design_and_fails_a_wild_or_a_too_soft_one():
    gentle = _check(DRIVE_OLD, 1.5, 104.0)
    assert gentle.state == DONE and gentle.result.passed, gentle.result
    wild = _check(DRIVE_OLD, 3.0, 104.0)
    assert wild.state == DONE and not wild.result.passed and "overshoot" in wild.result.why and not wild.result.stiffer
    soft = _check(DRIVE_OLD, 0.5, 104.0)
    assert not soft.result.passed and soft.result.stiffer, "2026-11-09: soft designs never moved the old wheel's 0.02 step"
    assert gentle.servo.direction == 0 and wild.servo.direction == 0, "the servo is off once it is done"


def test_the_check_steps_a_rough_design_down_and_a_clean_one_up():
    from core.la.design_check import CheckResult
    from tools.la_bench.calibrate import verify
    from tools.la_bench.design import Choice

    def wheel(limit, kp_min=0.0):
        def run(tune):
            kp = tune.inertia * (2.0 * 3.141592653589793 * tune.bandwidth_hz) ** 2
            if kp < kp_min:
                return CheckResult(10.0, 0.0, 0, 0.01, False, False, "soft", 0.0, True)
            ok = tune.bandwidth_hz <= limit + 1e-9
            return CheckResult(50.0, 0.05 if ok else 0.6, 0, 0.001, ok, ok and tune.bandwidth_hz < limit * 0.9, "")
        return run

    def bandwidths(tried):
        return [round(c.bandwidth_hz, 2) for c, _ in tried]

    best, tried = verify(Choice(100.0, 2.5, 0.7, None, None, 0.0), wheel(1.7), say=lambda _m: None)
    assert best.bandwidth_hz == pytest.approx(1.6) and bandwidths(tried) == [2.5, 2.0, 1.6]
    best, tried = verify(Choice(100.0, 0.8, 0.7, None, None, 0.0), wheel(1.3), say=lambda _m: None)
    assert best.bandwidth_hz == pytest.approx(1.25) and bandwidths(tried) == [0.8, 1.0, 1.25]
    best, _ = verify(Choice(100.0, 0.5, 0.7, None, None, 0.0), wheel(0.3), say=lambda _m: None)
    assert best is None, "nothing held: nothing is saved"
    best, tried = verify(Choice(100.0, 1.5, 0.7, None, None, 0.0), wheel(1.6, kp_min=15000.0), say=lambda _m: None)
    assert best.inertia >= 150.0 and best.bandwidth_hz == pytest.approx(1.5), "a sticky wheel gets a stiffer design"
    best, tried = verify(Choice(150.0, 2.0, 0.7, None, None, 0.0), wheel(1.7, kp_min=15000.0), say=lambda _m: None,
                         breakaway=300.0)
    assert best.inertia * (2.0 * 3.141592653589793 * best.bandwidth_hz) ** 2 >= 15000.0, "gentler keeps its stiffness"


def test_half_the_friction_push_is_kept_when_the_wheel_overshoots_clearly_less_with_it():
    from core.la.design_check import CheckResult
    from tools.la_bench.calibrate import verify
    from tools.la_bench.design import Choice

    def run(tune):
        over = 0.3 if tune.ff_friction > 200.0 else 0.1
        return CheckResult(40.0, over, 0, 0.002, True, False, "clean")

    best, tried = verify(Choice(100.0, 2.0, 0.7, None, None, 0.0, ff_friction=261.0), run, say=lambda _m: None,
                         breakaway=466.0)
    assert best.ff_friction == pytest.approx(130.5) and len(tried) == 2, "2026-11-09: 261 overshot 30 % on the old wheel"
    kp = best.inertia * (2.0 * 3.141592653589793 * best.bandwidth_hz) ** 2
    assert kp >= (466.0 - 130.5) / 0.02 - 1.0, "half the push needs the stiffness to move a small step past the breakaway"
