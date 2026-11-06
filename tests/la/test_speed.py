"""The speed benchmark: the rim's own ceiling, and ramps and steps on a model wheel."""

from __future__ import annotations

import dataclasses
import json
import math
from argparse import Namespace

import pytest

from tools import la_servo_bench as bench_cli
from tools.la_bench.bench import sim_bench
from tools.la_bench.plant import SIM_WHEELS, WheelParams
from tools.la_bench.protocol import Design, Stage, ramp
from tools.la_bench.speed import (FOLLOW_OVER, RAMP_AMP, config_keys, half_stage, quickest, rim_accel, run_speed,
                                  speed_ramp, summary, top_rate)


def test_the_ceiling_is_the_bang_bang_move_on_a_free_rim():
    rim = WheelParams(inertia=400.0, damping=0.0, stick=0.0, spring=0.0)
    assert quickest(rim, 4000.0, 0.1, slew=0.0) == pytest.approx(2.0 * math.sqrt(0.1 * 400.0 / 4000.0), rel=0.01)


def test_the_force_slew_damping_and_friction_only_slow_the_ceiling():
    free = WheelParams(inertia=400.0, damping=0.0, stick=0.0, spring=0.0)
    real = WheelParams(inertia=400.0, damping=600.0, stick=150.0, spring=0.0)
    base = quickest(free, 4000.0, 0.1, slew=0.0)
    assert quickest(free, 4000.0, 0.1, slew=30000.0) > base
    assert quickest(real, 4000.0, 0.1, slew=0.0) > base
    assert quickest(WheelParams(inertia=400.0, damping=600.0, stick=5000.0), 4000.0, 0.1) is None, "cannot break away"
    assert top_rate(real, 4000.0) == pytest.approx((4000.0 - 150.0) / 600.0)


def test_a_model_wheel_follows_a_slow_ramp_and_settles_a_step():
    bench = sim_bench(SIM_WHEELS["r3"])
    bench.say = lambda _m: None
    _, direction, home = bench.prepare(1, True)
    design = Design(bandwidth_hz=1.0, inertia=650.0, damping_first=True, cap=3000)
    data = run_speed(bench, design, direction, home, say=lambda _m: None, rates=(0.1,), sizes=(0.05,))
    assert data["fastest_followed"] == 0.1 and data["ramps"][0]["lag_s"] <= 0.1
    assert data["steps"][0]["t90"] is not None and data["steps"][0]["clean"]
    text = summary(dict(data, label=design.label), SIM_WHEELS["r3"], 3000.0)
    assert "fastest ramp followed: 0.1 lock/s (45 deg/s)" in text and "rim allows" in text


def test_the_benchmark_takes_las_design_from_config_read_only(tmp_path, monkeypatch):
    raw = {"la_wheel_inertia": 528.0, "la_bandwidth_hz": 1.75, "la_damping": 0.0, "la_damping_first": True,
           "la_servo_cap": 5500.0}
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(bench_cli, "ROOT", tmp_path)
    assert bench_cli._config_design() == {"inertia": 528.0, "bandwidth": 1.75, "cap": 5500.0, "law": "damping-first"}
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == raw
    (tmp_path / "config.json").write_text("{broken", encoding="utf-8")
    assert bench_cli._config_design() == {}


def _args(**kw) -> Namespace:
    base = dict(simulate="rim", rim=None, bandwidth=1.0, damping=0.7, inertia=650.0, cap=3000.0, law="plain",
                direction=1, hz=20.0, rates=[0.1], steps=[0.02], rotation=900.0, note="", out=None, push_ahead=False,
                ff_inertia=0.0, ff_damping=0.0, ff_friction=0.0, lead=None, plan=None, corner=0.1,
                half_rates=[], no_half_at_once=True, half_accel=None)
    return Namespace(**dict(base, **kw))


def test_simulating_the_fitted_rim_or_pushing_its_rim_ahead_needs_a_fit(capsys):
    assert bench_cli.cmd_speed(_args()) == 1
    assert "needs --rim" in capsys.readouterr().out
    assert bench_cli.cmd_speed(_args(simulate=None, push_ahead=True)) == 1
    assert "--push-ahead needs --rim" in capsys.readouterr().out


def test_a_rounded_ramp_reaches_its_rate_and_stops_without_a_jump():
    stage = speed_ramp(0.8)
    segs = stage.segments()
    t0, t1 = segs[1][0], segs[1][1]
    h = 1e-5
    for t in [t0 + k * (t1 - t0) / 40 for k in range(41)]:
        x, v, a = stage.aim3(t)
        assert v == pytest.approx((stage.aim3(t + h)[0] - stage.aim3(t - h)[0]) / (2 * h), abs=1e-3)
        assert abs(a) <= 0.8 / stage.corner + 1e-9
    assert stage.aim3(t0 + 0.5 * (t1 - t0))[1] == pytest.approx(0.8), "cruising at its rate"
    assert stage.aim3(t1)[:2] == (pytest.approx(RAMP_AMP), 0.0)
    assert stage.corner == pytest.approx(0.5 * RAMP_AMP / 0.8), "half the move at the fastest"
    assert speed_ramp(0.2).corner == pytest.approx(0.1)
    braking = t1 - stage.corner
    assert any(abs(e - braking) < 1e-6 for e in stage.events(20.0)), "it sends when it starts to brake"
    assert Stage.from_dict(stage.to_dict()) == stage and "corner" not in ramp(0.06, 0.05).to_dict()


R3_FIT = WheelParams(inertia=352.0, damping=646.0, stick=164.0, spring=2046.0, delay_s=0.0258)  # check 2026-11-05


def test_the_push_ahead_follows_a_ramp_the_feedback_alone_cannot():
    rim = R3_FIT
    plain = Design(bandwidth_hz=1.75, inertia=528.0, damping_first=True, cap=5500)
    ahead = dataclasses.replace(plain, ff_inertia=rim.inertia, ff_damping=rim.damping, ff_friction=rim.stick)
    out = []
    for design in (plain, ahead):
        bench = sim_bench(rim)
        bench.say = lambda _m: None
        _, direction, home = bench.prepare(1, True, design.pushes_ahead)
        out.append(run_speed(bench, design, direction, home, say=lambda _m: None, rates=(0.8,), sizes=()))
    assert out[1]["ramps"][0]["overshoot"] < out[0]["ramps"][0]["overshoot"]
    assert out[1]["ramps"][0]["overshoot"] <= FOLLOW_OVER and out[1]["fastest_followed"] == 0.8
    assert "ahead" in ahead.label and ahead.pushes_ahead and not plain.pushes_ahead


def test_the_keys_are_offered_only_for_a_clean_run_that_used_them():
    clean = {"ramps": [{"aborted": None}], "steps": [{"clean": False, "rang": False}], "fastest_followed": 0.8,
             "design": {"ff_inertia": 352.4, "ff_damping": 646.2, "ff_friction": 164.0, "lead_s": 0.0258}}
    assert config_keys(clean) == {"la_ff_inertia": 352, "la_ff_damping": 646, "la_ff_friction": 164,
                                  "la_servo_lead_s": 0.026, "la_ff_plan_s": 0.0}, "a step's overshoot is the feedback's"
    assert config_keys(dict(clean, steps=[{"clean": False, "rang": True}])) == {}
    assert config_keys(dict(clean, fastest_followed=0.2)) == {}, "no faster than without it"
    assert config_keys(dict(clean, design={"ff_inertia": 0.0})) == {}


def test_the_bench_walks_a_wheel_resting_off_centre_to_the_centre_without_a_jolt():
    from tools.la_bench.bench import HOME_RATE, Bench
    from tools.la_bench.plant import SimChannel, SimClock

    clock = SimClock()
    channel = SimChannel(R3_FIT, clock, steer0=0.14)
    seen = []
    publish = channel.publish

    def spy(*a, **kw):
        st = publish(*a, **kw)
        seen.append((clock(), st.steer))
        return st

    channel.publish = spy
    bench = Bench(channel, clock, clock.sleep, say=lambda _m: None)
    _, _, home = bench.prepare(1, True, False)
    assert home == 0.0 and abs(channel.wheel.steer) < 0.01, "the bench centres the wheel (it rested 0.14 off on 2026-11-06)"
    assert min(s for _, s in seen) > -0.01, "no swing past the centre"
    rates = [abs(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(seen, seen[1:]) if b[0] - a[0] > 0.01]
    assert max(rates) < 1.5 * HOME_RATE, "a walk, not a snap"


def test_half_lock_goes_out_and_back_each_way_on_a_rim_paced_ramp():
    rim = dataclasses.replace(R3_FIT, reading_hz=60.0)
    design = Design(bandwidth_hz=1.25, damping=0.7, inertia=528.0, damping_first=True, cap=5500,
                    ff_inertia=rim.inertia, ff_damping=rim.damping, ff_friction=rim.stick)
    bench = sim_bench(rim)
    bench.say = lambda _m: None
    _, direction, home = bench.prepare(1, True, True, design)
    data = run_speed(bench, design, direction, home, say=lambda _m: None, rates=(), sizes=(), half_rates=(0.5,),
                     half_at_once=False, half_accel=rim_accel(rim, 5500.0))
    half = data["half"][0]
    assert half["followed"] and not half["rang"] and len(half["t90"]) == 4 and all(t < 1.2 for t in half["t90"])
    assert max(abs(s[1]) for s in data["runs"][0]["samples"]) > 0.48, "it really went half lock, past the 0.4 limit"
    stage = half_stage(2.0, accel=8.0)
    assert stage.corner == pytest.approx(0.25) and half_stage(0.0).name == "out 0.5 at once"
    assert rim_accel(None, 5500.0) == 8.0 and rim_accel(R3_FIT, 5500.0) == pytest.approx(0.7 * (5500.0 - 164.0) / 352.0)


def test_the_model_wheel_reads_the_angle_at_the_games_frame_rate():
    from tools.la_bench.plant import SimWheel

    wheel = SimWheel(dataclasses.replace(R3_FIT, reading_hz=60.0), frame_hz=120.0)
    for _ in range(4 * 120):
        wheel.substep()
    assert wheel.readings == pytest.approx(wheel.frames / 2, abs=2)
    every = SimWheel(R3_FIT, frame_hz=120.0)
    for _ in range(4 * 120):
        every.substep()
    assert every.readings == every.frames, "0 reads it at every FFB tick, as before"


def test_the_push_ahead_on_the_plan_a_wheel_delay_early_stops_a_half_lock_on_time():
    rim = dataclasses.replace(R3_FIT, reading_hz=60.0)
    out = []
    for plan in (0.0, rim.delay_s):
        design = Design(bandwidth_hz=1.25, damping=0.7, inertia=528.0, damping_first=True, cap=5500, plan_s=plan,
                        ff_inertia=rim.inertia, ff_damping=rim.damping, ff_friction=rim.stick)
        bench = sim_bench(rim)
        bench.say = lambda _m: None
        _, direction, home = bench.prepare(1, True, True, design)
        out.append(run_speed(bench, design, direction, home, say=lambda _m: None, rates=(0.8,), sizes=(),
                             half_rates=(2.0,), half_at_once=False, half_accel=rim_accel(rim, 5500.0)))
    now, early = out
    assert early["half"][0]["overshoot"] < 0.5 * now["half"][0]["overshoot"], "2026-11-06 model: 23 % to 5 %"
    assert early["fastest_followed"] == 0.8 and not now["ramps"][0]["followed"]
    assert "plan26ms" in Design(ff_damping=1.0, plan_s=0.026).label
