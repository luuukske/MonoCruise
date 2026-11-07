"""The wheel check (tools/la_bench/check.py, design.py): design settings, ringing tests, design choice, smoothness."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from core.la.actuator import Gains, Wheel, WheelActuator, WheelTuning, design_overrides, push_ahead_overrides
from core.la.runner import AssistRunner
from core.lane_departure.command import ServoCommand
from core.lane_departure.lka_runner import LkaRunner
from core.settings import Settings
from tests.la.servo_sim import Plugin
from tools.la_bench.bench import BenchAbort, sim_bench
from tools.la_bench import check as check_module
from tools.la_bench.check import accepted, rings, run_check, summary
from tools.la_bench.design import Choice, choose_design, push_ahead_of, smoothness
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


def test_a_design_that_rings_is_fitted_again_with_the_ring_and_replaced(monkeypatch):
    real, calls = choose_design, []

    def eager_first(params, frame_hz, **kwargs):
        calls.append(1)
        return Choice(150.0, 3.0, 0.5, None, None, 0.0) if len(calls) == 1 else real(params, frame_hz, **kwargs)

    monkeypatch.setattr(check_module, "choose_design", eager_first)
    bench = sim_bench(SIM_WHEELS["r3"])
    bench.say = lambda _msg: None
    _, direction, home = bench.prepare(None, True)
    data = run_check(bench, direction, home, evals=150, say=lambda _msg: None, grid=GRID)
    assert len(calls) == 2 and data["verified"]
    assert any("rang on the wheel" in n for n in data["notes"])
    assert data["config"]["la_bandwidth_hz"] < 3.0


def test_a_held_design_is_tried_faster_and_the_fastest_accepted_is_kept(monkeypatch):
    seen = []
    real = check_module._verify

    def spy(bench, choice, *args, **kwargs):
        last, good, fast_only = real(bench, choice, *args, **kwargs)
        seen.append((choice.bandwidth_hz, good))
        return last, good, fast_only

    monkeypatch.setattr(check_module, "_verify", spy)
    bench = sim_bench(SIM_WHEELS["r3"])
    bench.say = lambda _msg: None
    _, direction, home = bench.prepare(None, True)
    data = run_check(bench, direction, home, evals=150, say=lambda _msg: None, grid=GRID)
    accepted_bw = [bw for bw, good in seen if good]
    assert data["verified"] and data["config"]["la_bandwidth_hz"] == max(accepted_bw)
    assert len(seen) >= 2, "at least one faster design was tried after the first held"


def test_the_servo_cap_is_opt_in_bounded_and_for_la_only(monkeypatch, tmp_path):
    assert "servo_cap" not in design_overrides(0.0, 0.0, False)
    assert design_overrides(0.0, 0.0, False, 5500.0)["servo_cap"] == 5500.0
    assert design_overrides(0.0, 0.0, False, 99999.0)["servo_cap"] == 6000.0
    assert design_overrides(0.0, 0.0, False, 50.0)["servo_cap"] == 1000.0
    assert Settings.la_servo_cap == 0.0
    monkeypatch.setattr(Settings.instance(), "la_servo_cap", 5500.0, raising=False)
    assert AssistRunner(debug_root=tmp_path)._tuning()[1].wheel.servo_cap == 5500.0
    assert LkaRunner(debug_root=tmp_path)._tuning()[1].wheel.servo_cap == 3000.0, "the lane keeping assist keeps its cap"
    monkeypatch.setattr(Settings.instance(), "la_follow_all", True, raising=False)
    line = AssistRunner(debug_root=tmp_path)._tuning()[2]
    assert "servo cap 5500" in line[0] % tuple(line[1:])


def pushed(cap: float) -> float:
    plugin = Plugin()
    servo = ServoCommand(target=1.0, kp=1e6, ki=0.0, cap=int(cap), direction=1)
    return max(abs(plugin.step(servo, 0.0, 0.0, 1 / 120, 0.0)) for _ in range(180))


def test_the_plugin_mirror_pushes_up_to_the_requested_cap_and_never_past_its_hard_maximum():
    assert pushed(3000) == pytest.approx(3000.0, abs=1.0)
    assert pushed(5500) == pytest.approx(5500.0, abs=1.0)
    assert pushed(60000) == pytest.approx(6000.0, abs=1.0), "the hard maximum"


def test_the_push_ahead_is_opt_in_bounded_and_shared_by_la_and_the_lane_keeping_assist(monkeypatch, tmp_path):
    assert push_ahead_overrides(0.0, 0.0, 0.0, 0.0) == {"ff_inertia": 0.0, "ff_damping": 0.0, "ff_friction": 0.0,
                                                       "lead_s": 0.0, "plan_s": 0.0, "ff_breakaway": 0.0}
    assert push_ahead_overrides(1e9, -5.0, 1e9, 1.0, 1.0, 1e9) == {
        "ff_inertia": 5000.0, "ff_damping": 0.0, "ff_friction": 1500.0, "lead_s": 0.08, "plan_s": 0.08,
        "ff_breakaway": 1500.0}
    for key in ("la_ff_inertia", "la_ff_damping", "la_ff_friction", "la_servo_lead_s", "la_ff_plan_s",
                "la_ff_breakaway"):
        assert getattr(Settings, key) == 0.0
    for key, value in (("la_ff_inertia", 352.0), ("la_ff_damping", 646.0), ("la_ff_friction", 164.0),
                       ("la_servo_lead_s", 0.0), ("la_ff_plan_s", 0.026), ("la_ff_breakaway", 380.0)):
        monkeypatch.setattr(Settings.instance(), key, value, raising=False)
    for runner in (AssistRunner(debug_root=tmp_path), LkaRunner(debug_root=tmp_path)):
        wheel = runner._tuning()[1].wheel
        assert (wheel.ff_inertia, wheel.ff_damping, wheel.ff_friction, wheel.lead_s, wheel.plan_s,
                wheel.ff_breakaway) == (352.0, 646.0, 164.0, 0.0, 0.026, 380.0)
    monkeypatch.setattr(Settings.instance(), "la_always_on", True, raising=False)
    for runner in (AssistRunner(debug_root=tmp_path), LkaRunner(debug_root=tmp_path)):
        line = runner._tuning()[2]
        assert ("push ahead inertia 352, damping 646, friction 164, lead 0.000 s, plan 0.026 s, breakaway 380"
                in line[0] % tuple(line[1:]))


def test_the_push_ahead_fades_in_with_the_authority_and_moves_on_the_plan():
    act = WheelActuator(WheelTuning(ff_inertia=352.0, ff_damping=646.0, ff_friction=164.0, lead_s=0.026))
    wheel = Wheel(ok=True)
    full = act.commands(Gains(580.0, 0.17), 0.1, 0.3, 1.0, wheel, 20.0, plan=(0.35, 2.0))
    half = act.commands(Gains(580.0, 0.17), 0.1, 0.3, 0.5, wheel, 20.0, plan=(0.35, 2.0))
    assert (full.ff_inertia, full.ff_damping, full.ff_friction, full.lead_s) == (352.0, 646.0, 164.0, 0.026)
    assert full.ff_plan and (full.plan_rate, full.plan_accel) == (0.35, 2.0)
    assert full.accel == 0.0, "the target itself moves on at its rate only, as before"
    assert half.ff_damping == pytest.approx(323.0) and half.lead_s == 0.026
    bare = act.commands(Gains(580.0, 0.17), 0.1, 0.3, 1.0, wheel, 20.0)
    assert bare.ff_plan and (bare.plan_rate, bare.plan_accel) == (0.3, 0.0), "no plan: its rate, now"
    plain = WheelActuator().commands(Gains(580.0, 0.17), 0.1, 0.3, 1.0, wheel, 20.0, plan=(0.35, 2.0))
    assert plain.ff_damping == 0.0 and not plain.ff_plan and plain.accel == 0.0, "off by default"


def test_the_plan_is_the_followers_own_motion_a_wheel_delay_on():
    from core.la.actuator import follower_ahead

    w, e, v, tau = 2.0 * math.pi * 1.5, 0.04, -0.1, 0.026
    h = 1e-5
    x, rate = e, v
    for _ in range(int(round(tau / h))):
        acc = -w * w * x - 2.0 * w * rate
        rate += acc * h
        x += rate * h
    plan_rate, plan_accel = follower_ahead(e, v, w, tau)
    assert plan_rate == pytest.approx(rate, abs=1e-4)
    assert plan_accel == pytest.approx(-w * w * x - 2.0 * w * rate, abs=1e-2)
    assert follower_ahead(e, v, w, 0.0) == pytest.approx((v, -w * w * e - 2.0 * w * v))


def test_the_bench_measures_how_often_the_game_reads_the_wheel():
    bench = sim_bench(replace(SIM_WHEELS["r3"], reading_hz=60.0))
    bench.say = lambda _m: None
    assert bench.reading_hz == 0.0
    bench.ready()
    bench.idle(1.5)
    assert bench.reading_hz == pytest.approx(60.0, rel=0.05)


def test_with_the_push_ahead_the_check_writes_the_rims_own_keys():
    rim = WheelParams(inertia=352.0, damping=646.0, stick=164.0, spring=2046.0, delay_s=0.0258, reading_hz=60.0)
    choice = choose_design(rim, 120.0, bandwidths=(1.0, 1.25), dampings=(0.7,), scales=(1.5,), ahead=True)
    assert not choice.limited and choice.plan_s == pytest.approx(0.0258)
    keys = choice.config()
    assert (keys["la_ff_inertia"], keys["la_ff_damping"], keys["la_ff_friction"], keys["la_ff_plan_s"]) == (
        352, 646, 164, 0.026)
    assert "la_ff_inertia" not in choose_design(rim, 120.0, bandwidths=(1.0,), dampings=(0.7,), scales=(1.5,)).config()


def test_a_wheel_the_fit_cannot_describe_gets_its_design_on_the_wheel():
    bench = sim_bench(replace(SIM_WHEELS["geared"], reading_hz=60.0))
    bench.say = lambda _m: None
    _, direction, home = bench.prepare(1, True)
    data = run_check(bench, direction, home, evals=200, say=lambda _m: None)
    assert data["verified"] and data["reading_hz"] == 60 and data["config"]["la_ff_friction"] > 0
    poor_fit = WheelParams(inertia=153.0, damping=562.0, stick=530.0, spring=3000.0, delay_s=0.068, reading_hz=60.0)
    held, runs = check_module._wheel_ladder(bench, poor_fit, direction, home, 20.0, lambda _m: None, [], [])
    assert held is not None and held.config()["la_ff_plan_s"] == 0.068, (
        "the geared model wheel: no design held in the model, 0.7 to 1.4 Hz with the push ahead held on the wheel")


def test_the_plan_holds_the_integral_on_fast_moves_and_lags_the_damping():
    servo = ServoCommand(target=0.01, kp=30000.0, ki=50000.0, cap=3000, direction=1, rate=0.5, kd=5000.0,
                         ff_damping=600.0, ff_plan=True, plan_rate=0.5)
    plugin = Plugin()
    plugin.step(servo, 0.0, 0.0, 1 / 120)
    assert plugin.integ == 0.0, "the target moves faster than 0.3 lock/s"
    slow = replace(servo, rate=0.1, plan_rate=0.1)
    plugin = Plugin()
    plugin.step(slow, 0.0, 0.0, 1 / 120)
    assert plugin.integ > 0.0
    plain = Plugin()
    plain.step(replace(servo, ff_plan=False), 0.0, 0.0, 1 / 120)
    assert plain.integ > 0.0 and plain.d_ref == 0.5, "without the plan, as before"
