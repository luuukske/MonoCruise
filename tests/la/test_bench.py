"""The LA servo bench (tools/la_servo_bench.py): stages, metrics, capture on a model wheel, and the rim fit."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from argparse import Namespace

import pytest

from core.la.actuator import WheelActuator, WheelTuning
from core.lane_departure.command import SERVO_OFF
from tools import la_servo_bench as bench_cli
from tools.la_bench.metrics import describe, step_response, summarize
from tests.la import servo_sim
from tests.la.servo_sim import _sum
from tools.la_bench.plant import SimChannel, SimClock, SimWheel, WheelParams, fit, replay
from tools.la_bench.protocol import SUITES, Design, follow, ramp, step

RIM = WheelParams(inertia=700.0, damping=500.0, stick=150.0, spring=2000.0, delay_s=0.02)
R3 = WheelParams(inertia=270.0, damping=600.0, stick=185.0, spring=1550.0, delay_s=0.012)


def sim_bench(params: WheelParams = RIM, frame_hz: float = 120.0) -> bench_cli.Bench:
    clock = SimClock()
    return bench_cli.Bench(SimChannel(params, clock, frame_hz=frame_hz), clock, clock.sleep, say=lambda _msg: None)


def test_design_gains_are_the_actuators():
    kp, kd, ki = Design(bandwidth_hz=2.0).gains()
    assert (kp, kd, ki) == WheelActuator(WheelTuning()).gains(1.0, 0.0)
    assert Design(bandwidth_hz=1.0).gains()[0] == pytest.approx(kp / 4.0)


def test_game_spring_takes_over_part_of_the_stiffness():
    standing = Design().gains()
    at_speed = Design(spring=57000.0).gains()
    assert at_speed[0] < standing[0]
    assert at_speed[1] == pytest.approx(standing[1])


def test_step_stage_jumps_up_swings_through_zero_and_goes_home():
    stage = step(0.02, hold=1.5)
    assert stage.events(20.0) == [0.0, 0.5, 2.0, 3.5]
    assert [stage.aim(t)[0] for t in (0.0, 0.6, 2.1, 3.6)] == [0.0, 0.02, -0.02, 0.0]
    assert stage.duration == pytest.approx(5.0)


def test_moving_stages_send_at_the_target_rate_with_the_derivative():
    run = follow(0.03, 0.7)
    events = run.events(20.0)
    assert len(events) == int(run.duration * 20)
    t = events[7]
    offset, rate = run.aim(t)
    h = 1e-6
    assert rate == pytest.approx((run.aim(t + h)[0] - run.aim(t - h)[0]) / (2 * h), rel=1e-4)
    assert abs(ramp(0.06, 0.05).aim(0.5 + 0.3)[1]) == pytest.approx(0.05)
    assert ramp(0.06, 0.05).aim(0.5 + 0.06 / 0.05 + 0.5)[1] == 0.0


def test_step_response_reads_a_second_order_answer():
    zeta, wn = 0.5, 12.0
    wd = wn * math.sqrt(1 - zeta**2)
    ts = [k / 400 for k in range(1200)]
    ys = [1 - math.exp(-zeta * wn * t) * (math.cos(wd * t) + zeta * wn / wd * math.sin(wd * t)) for t in ts]
    got = step_response(ts, [0.1 * y for y in ys], 0.0, 0.0, 0.1, 3.0)
    assert got["overshoot"] == pytest.approx(math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2)), abs=0.01)
    assert 0.1 < got["rise_s"] < 0.2
    assert got["settle_s"] is not None and got["settle_s"] < 1.0


def test_a_wheel_that_never_arrives_does_not_settle():
    ts = [k / 100 for k in range(300)]
    got = step_response(ts, [0.0] * 300, 0.0, 0.0, 0.1, 3.0)
    assert got["rise_s"] is None and got["settle_s"] is None


def test_simulated_capture_records_every_run_and_finds_the_direction():
    data = bench_cli.capture(sim_bench(), [Design()], SUITES["step"], 20.0)
    assert data["meta"]["direction"] == 1 and "aborted" not in data["meta"]
    assert [r["name"] for r in data["runs"]] == ["step 0.01", "step 0.02"]
    run = data["runs"][0]
    assert len(run["samples"]) > 300 and run["sent"][0][0] == pytest.approx(0.0, abs=0.01)
    assert "rise" in describe(summarize(run))
    json.dumps(data)


def test_a_wheel_wired_the_other_way_is_detected_and_still_holds():
    data = bench_cli.capture(sim_bench(WheelParams(**{**RIM.__dict__, "sign": -1})), [Design(bandwidth_hz=1.0)],
                             [step(0.02)], 20.0)
    assert data["meta"]["direction"] == -1
    up = summarize(data["runs"][0])["up"]
    assert up["rise_s"] is not None and up["overshoot"] < 0.3


def test_a_wrong_direction_ends_the_stage_and_asks_the_servo_off():
    bench = sim_bench()
    data = bench_cli.capture(bench, [Design()], [step(0.02)], 20.0, direction=-1)
    assert "wheel past" in data["runs"][0]["aborted"]
    assert max(abs(s[1]) for s in data["runs"][0]["samples"]) <= bench_cli.ANGLE_LIMIT + 0.1
    assert bench.channel.servo == SERVO_OFF


def test_a_missing_plugin_ends_with_nothing_recorded():
    class Absent:
        def publish(self, *args, **kwargs):
            from core.lane_departure.command import PluginStatus

            return PluginStatus()

    clock = SimClock()
    data = bench_cli.capture(bench_cli.Bench(Absent(), clock, clock.sleep, say=lambda _msg: None), [Design()],
                             SUITES["step"], 20.0)
    assert data["runs"] == [] and "plugin not found" in data["meta"]["aborted"]


def test_replay_of_the_true_rim_matches_the_recording():
    data = bench_cli.capture(sim_bench(), [Design(bandwidth_hz=1.0)], [follow(0.03, 0.4)], 20.0)
    run = data["runs"][0]
    ts, ys = replay(run, RIM, 120.0)
    from tools.la_bench.metrics import interp

    err = [s[1] - interp(ts, ys, s[0]) for s in run["samples"]]
    assert math.sqrt(sum(e * e for e in err) / len(err)) < 0.001


def test_the_fit_recovers_a_known_rim_from_noisy_runs_at_another_frame_rate():
    stages = [step(0.01), ramp(0.06, 0.2), follow(0.03, 0.7)]
    designs = [Design(bandwidth_hz=1.0), Design(bandwidth_hz=1.5)]
    data = bench_cli.capture(sim_bench(frame_hz=100.0), designs, stages, 20.0)
    rng = random.Random(3)
    for run in data["runs"]:
        for row in run["samples"]:
            row[1] += rng.gauss(0.0, 0.0003)
    result = fit(data["runs"], 100.0, WheelParams(sign=1), max_evals=300)
    p = result.params
    assert result.used == len(data["runs"])
    assert result.rms < 0.001
    assert p.inertia == pytest.approx(RIM.inertia, rel=0.2)
    assert p.delay_s == pytest.approx(RIM.delay_s, abs=0.006)


def test_pinned_runs_are_fitted_on_their_swing_not_their_angle():
    data = bench_cli.capture(sim_bench(), [Design(bandwidth_hz=2.0)], [step(0.01), step(0.02), follow(0.03, 0.4)], 20.0)
    result = fit(data["runs"], 120.0, WheelParams(sign=1), max_evals=60)
    assert result.used < len(data["runs"])
    assert len(result.per_run) == len(data["runs"])


def test_sim_wheel_with_the_servo_off_and_no_torque_stays_put():
    wheel = SimWheel(RIM)
    for _ in range(2000):
        wheel.substep()
    assert wheel.steer == 0.0 and wheel.frames > 100


def test_report_and_predict_run_from_a_saved_recording(tmp_path, capsys):
    data = bench_cli.capture(sim_bench(), [Design(bandwidth_hz=1.0)], [step(0.01), follow(0.03, 0.4)], 20.0)
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert bench_cli.cmd_report(Namespace(file=str(path), json=False)) == 0
    assert "step 0.01" in capsys.readouterr().out
    assert bench_cli.cmd_predict(Namespace(
        file=str(path), bandwidth=[1.0], damping=0.7, inertia=650.0, strength=1.0, integral_share=0.3,
        game_spring=0.0, cap=3000, suite="step", hz=20.0, law="plain")) == 1
    assert "run fit first" in capsys.readouterr().out
    assert bench_cli.cmd_fit(Namespace(files=[str(path)], kind=["step", "follow"], evals=40, compare=False,
                                       no_ring=False, no_save=False)) == 0
    assert "fit" in json.loads(path.read_text(encoding="utf-8"))
    assert bench_cli.cmd_predict(Namespace(
        file=str(path), bandwidth=[1.0, 2.0], damping=0.7, inertia=650.0, strength=1.0, integral_share=0.3,
        game_spring=0.0, cap=3000, suite="step", hz=20.0, law="both")) == 0
    assert capsys.readouterr().out.count("step 0.01") == 4


def test_the_default_design_rings_on_the_measured_r3_and_damping_first_does_not(monkeypatch):
    monkeypatch.setattr(servo_sim, "SERVO_SLEW", 30000.0)  # the plugin it was measured on, before 2026-11-06
    plain = bench_cli.capture(sim_bench(R3), [Design(bandwidth_hz=2.0)], [step(0.02)], 20.0)["runs"][0]
    first = bench_cli.capture(sim_bench(R3), [Design(bandwidth_hz=2.0, damping_first=True)], [step(0.02)], 20.0)["runs"][0]
    ringing = summarize(plain)["swing"]
    assert plain.get("aborted") or ringing["overshoot"] > 1.0 or ringing["settle_s"] is None, "measured on the R3 2026-11-05"
    calm = summarize(first)["swing"]
    assert not first.get("aborted") and calm["overshoot"] < 0.4 and calm["settle_s"] is not None


def test_a_lighter_design_inertia_also_stops_the_standing_ring():
    run = bench_cli.capture(sim_bench(R3), [Design(bandwidth_hz=2.0, inertia=300.0)], [step(0.02)], 20.0)["runs"][0]
    swing = summarize(run)["swing"]
    assert not run.get("aborted") and swing["overshoot"] < 0.6 and swing["settle_s"] is not None


def test_damping_first_keeps_the_cap_and_brakes_a_pinned_servo():
    assert _sum(1e5, 0.0, 0.0, 2500.0, True) == 2500.0 and _sum(1e5, 0.0, 0.0, 2500.0, False) == 2500.0
    assert _sum(1e5, -800.0, 0.0, 2500.0, False) == 2500.0, "plain: the damping is hidden behind the pinned P"
    assert _sum(1e5, -800.0, 0.0, 2500.0, True) == pytest.approx(900.0)
    assert _sum(1e5, -1500.0, 0.0, 2500.0, True) == pytest.approx(-500.0)
    assert _sum(-1e5, 1500.0, 0.0, 2500.0, True) == pytest.approx(500.0)
    assert abs(_sum(1e5, 1500.0, 1e5, 2500.0, True)) <= 2500.0


def test_go_home_walks_a_resting_wheel_back_and_lets_go():
    bench = sim_bench(WheelParams(inertia=300.0, damping=600.0, stick=185.0, spring=0.0, delay_s=0.012))
    bench.ready()
    bench.channel.wheel.steer = 0.2
    bench.go_home(0.0, 1)
    assert abs(bench.channel.wheel.steer) < 0.02
    assert bench.channel.servo == SERVO_OFF


def test_damping_first_is_refused_on_a_plugin_that_cannot():
    class Old(SimChannel):
        def publish(self, *args, **kwargs):
            return dataclasses.replace(super().publish(*args, **kwargs), damping_first=False)

    clock = SimClock()
    old = bench_cli.Bench(Old(RIM, clock), clock, clock.sleep, say=lambda _msg: None)
    data = bench_cli.capture(old, [Design(damping_first=True)], SUITES["step"], 20.0)
    assert data["runs"] == [] and "cannot damp first" in data["meta"]["aborted"]


def test_a_note_and_the_law_land_in_the_recording():
    data = bench_cli.capture(sim_bench(), [Design(bandwidth_hz=1.0, damping_first=True)], [step(0.01)], 20.0, note="limiter on")
    assert data["meta"]["note"] == "limiter on"
    assert data["runs"][0]["gains"]["damping_first"] is True and "dfirst" in data["runs"][0]["label"]
    args = Namespace(bandwidth=[1.0, 2.0], damping=0.7, inertia=300.0, strength=1.0, integral_share=0.3, game_spring=0.0,
                     cap=3000, law="both")
    assert [(d.bandwidth_hz, d.damping_first) for d in bench_cli._designs(args)] == [
        (1.0, False), (1.0, True), (2.0, False), (2.0, True)]
