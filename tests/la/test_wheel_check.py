"""The wheel check (tools/la_bench/check.py, design.py): design settings, ringing tests, design choice, smoothness."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.la.actuator import WheelActuator, WheelTuning, design_overrides
from core.la.runner import AssistRunner
from core.lane_departure.lka_runner import LkaRunner
from core.settings import Settings
from tools.la_bench.bench import BenchAbort, sim_bench
from tools.la_bench.check import accepted, rings, run_check, summary
from tools.la_bench.design import choose_design, smoothness
from tools.la_bench.plant import SIM_WHEELS, WheelParams
from tools.la_bench.protocol import Design, follow, ramp, step


def test_design_settings_are_neutral_by_default_and_bounded():
    assert design_overrides(0.0, 0.0, False) == {"damping_first": False}
    assert design_overrides(9.0, 9.0, True) == {"damping_first": True, "bandwidth_hz": 3.0, "damping": 2.0}
    assert design_overrides(0.01, 0.01, False)["bandwidth_hz"] == 0.3
    tuned = WheelTuning(inertia=380.0, **design_overrides(1.5, 1.0, True))
    assert WheelActuator(tuned).gains(1.0, 0.0) == Design(bandwidth_hz=1.5, damping=1.0, inertia=380.0).gains()
    assert (Settings.la_bandwidth_hz, Settings.la_damping, Settings.la_damping_first) == (0.0, 0.0, False)


@pytest.mark.parametrize("make", [lambda p: AssistRunner(debug_root=p), lambda p: LkaRunner(debug_root=p)])
def test_both_runners_carry_the_design_into_the_servo_tuning(make, monkeypatch, tmp_path):
    runner = make(tmp_path)
    _, plain, _ = runner._tuning()
    assert (plain.wheel.bandwidth_hz, plain.wheel.damping, plain.wheel.damping_first) == (2.0, 0.7, False)
    for name, value in (("la_bandwidth_hz", 1.5), ("la_damping", 1.0), ("la_damping_first", True),
                        ("la_wheel_inertia", 380.0), ("la_follow_all", True)):
        monkeypatch.setattr(Settings.instance(), name, value, raising=False)
    key, tuned, line = runner._tuning()
    assert (tuned.wheel.bandwidth_hz, tuned.wheel.damping, tuned.wheel.damping_first, tuned.wheel.inertia) == (
        1.5, 1.0, True, 380.0)
    assert line[0] % tuple(line[1:]), "the log line takes every setting it is given"


GRID = {"bandwidths": (1.0, 1.5, 2.0), "dampings": (0.7, 1.0), "scales": (1.0, 2.0)}


def summary_of(**over):
    base = {"aborted": None, "buzz_min": 1.0, "up": {"overshoot": 0.1, "ripple": 0.0002},
            "swing": {"overshoot": 0.2, "ripple": 0.0002}}
    base.update(over)
    return base


def test_ringing_is_an_abort_a_buzz_cut_a_big_swing_or_bad_tracking():
    assert not rings(summary_of()) and accepted(summary_of())
    assert rings(summary_of(aborted="wheel past 0.4 lock")) and rings(summary_of(buzz_min=0.5))
    assert rings(summary_of(swing={"overshoot": 1.5, "ripple": 0.0}))
    assert rings(summary_of(up={"overshoot": 0.1, "ripple": 0.05}))
    assert rings({"aborted": None, "buzz_min": 1.0, "track": {"rms": 0.05}})
    assert not accepted(summary_of(swing={"overshoot": 0.7, "ripple": 0.0})), "calm but loose is not accepted"


def test_the_design_is_fast_and_holds_up_on_the_fitted_rim():
    r3 = SIM_WHEELS["r3"]
    choice = choose_design(r3, 120.0, bandwidths=(1.0, 1.5, 2.0, 2.5), dampings=(0.7, 1.0))
    assert not choice.limited and choice.worst_overshoot <= 0.6
    assert choice.bandwidth_hz >= 1.5, "a verified rim should not be left at the bottom of the grid"
    config = choice.config()
    assert config["la_wheel_inertia"] >= round(r3.inertia), "searched from the fitted inertia up"
    assert {k: v for k, v in config.items() if k != "la_wheel_inertia"} == {
        "la_bandwidth_hz": choice.bandwidth_hz, "la_damping": choice.damping, "la_damping_first": True}


def test_nothing_holding_up_gives_the_gentlest_design_flagged():
    awful = replace(SIM_WHEELS["bare"], delay_s=0.09, damping=30.0)
    choice = choose_design(awful, 120.0, bandwidths=(1.5, 2.5), dampings=(0.7,))
    assert choice.limited and choice.bandwidth_hz == 1.5


def slow_runs(wheel: str, design: Design) -> list[dict]:
    bench = sim_bench(SIM_WHEELS[wheel])
    bench.say = lambda _msg: None
    bench.ready()
    return [bench.run_stage(s, design, 1) for s in (ramp(0.06, 0.05), follow(0.03, 0.4))]


def test_a_geared_wheel_reads_steppy_and_a_direct_drive_wheel_smooth():
    design = Design(bandwidth_hz=1.5, damping=1.0, inertia=380.0, damping_first=True)
    smooth = smoothness(slow_runs("r3", design), SIM_WHEELS["r3"].stick)
    assert smooth["verdict"] == "smooth" and not smooth["why"]
    geared = smoothness(slow_runs("geared", design), SIM_WHEELS["geared"].stick)
    assert geared["verdict"] == "steppy"
    assert any("steps of" in why for why in geared["why"]) and any("push" in why for why in geared["why"])


def test_the_check_fits_chooses_and_verifies_a_design_on_a_model_wheel():
    bench = sim_bench(SIM_WHEELS["r3"])
    bench.say = lambda _msg: None
    _, direction, home = bench.prepare(None, True)
    data = run_check(bench, direction, home, evals=150, say=lambda _msg: None, grid=GRID)
    assert data["verified"] and data["smoothness"]["verdict"] == "smooth"
    assert data["wheel"]["inertia"] == pytest.approx(SIM_WHEELS["r3"].inertia, rel=0.3)
    assert 0.5 <= data["config"]["la_bandwidth_hz"] <= 2.5 and data["config"]["la_damping_first"] is True
    assert data["step_rise_s"] is not None and data["step_overshoot"] <= 0.3
    text = summary(data)
    assert "verified on the wheel" in text and "'la_damping_first': true," in text
    json.dumps(data)


def test_a_wheel_nothing_holds_on_is_told_to_keep_the_defaults():
    hopeless = WheelParams(inertia=150.0, damping=100.0, stick=300.0, spring=1500.0, delay_s=0.03, cog=2500.0,
                           cog_period=0.02, quantum=0.004)
    bench = sim_bench(hopeless)
    bench.say = lambda _msg: None
    _, direction, home = bench.prepare(None, True)
    data = run_check(bench, direction, home, evals=60, say=lambda _msg: None, grid=GRID)
    assert not data["verified"] and data["config"] == {}
    assert "leave the la_* keys at their defaults" in summary(data)
    assert bench.channel.servo.direction == 0, "the servo is asked off"


def test_a_stop_midway_keeps_what_was_recorded():
    bench = sim_bench(SIM_WHEELS["r3"])
    bench.say = lambda _msg: None
    _, direction, home = bench.prepare(None, True)
    real, calls = bench.run_stage, []

    def flaky(*args, **kwargs):
        if len(calls) == 3:
            raise BenchAbort("the wheel rests +0.41 lock off centre")
        calls.append(1)
        return real(*args, **kwargs)

    bench.run_stage = flaky
    data = run_check(bench, direction, home, say=lambda _msg: None, grid=GRID)
    assert len(data["runs"]) == 3 and not data["verified"] and data["config"] == {}
    assert any(n.startswith("stopped") for n in data["notes"])
    assert "rim: not measured" in summary(data)
    json.dumps(data)
