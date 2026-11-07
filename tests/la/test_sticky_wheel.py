"""A wheel that needs more to start than to slide, with a coarse angle (the old wheel, 2026-11-06): the model, the
plugin's breakaway push, and the wheel check reading and designing for it.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from core.lane_departure.command import ServoCommand
from tests.la import servo_sim
from tests.la.servo_sim import Plugin
from tools.la_bench.bench import Bench, BenchAbort, sim_bench
from tools.la_bench.check import accepted, design_rim, rings, run_check
from tools.la_bench.design import push_ahead_of, sticky
from tools.la_bench.metrics import interp, summarize
from tools.la_bench.probe import bracket_frictions, damping_probe, fit_slides, slide_estimate, slide_probe
from tools.la_bench.plant import (SIM_WHEELS, SimChannel, SimClock, SimWheel, WheelParams, breakaway_of,
                                  estimate_spring, fit, quantum_of, replay)
from tools.la_bench.protocol import Design, ramp, step

ROOT = Path(__file__).resolve().parents[2]
STICKY = replace(SIM_WHEELS["sticky"], reading_hz=60.0)
GENTLE = Design(bandwidth_hz=0.7, damping=0.7, inertia=650.0, damping_first=True, cap=2000)


def _runs(params: WheelParams, design: Design = GENTLE) -> list[dict]:
    bench = sim_bench(params)
    bench.say = lambda _m: None
    return [bench.run_stage(st, design, params.sign, 20.0, 0.0) for st in (step(0.02), ramp(0.06, 0.05))]


def _held_at(params: WheelParams, torque: float, seconds: float = 1.0) -> float:
    wheel = SimWheel(params)
    while wheel.time < seconds:
        wheel.substep(torque=torque)
    return wheel.steer


def test_the_model_wheel_stands_until_its_breakaway_and_slides_on_its_friction():
    assert _held_at(STICKY, 400.0) == 0.0, "between the sliding friction and the breakaway it stands"
    assert _held_at(STICKY, 520.0) > 0.01
    assert _held_at(replace(STICKY, breakaway=0.0), 400.0) > 0.01, "without a breakaway 400 moves it"


def test_the_model_wheel_reads_at_its_rate_whatever_the_ffb_rate():
    for frame_hz in (120.0, 120.6, 118.0):
        bench = sim_bench(STICKY, frame_hz)
        bench.say = lambda _m: None
        bench.ready()
        bench.idle(1.5)
        assert bench.reading_hz == pytest.approx(60.0, rel=0.03), f"at {frame_hz} Hz it read at 40 (2026-11-06)"


def test_the_breakaway_and_the_angle_steps_are_read_off_the_wheel():
    runs = _runs(STICKY)
    assert 400.0 < breakaway_of(runs) < 600.0, "the old wheel broke free at about 460, the model at 480"
    assert quantum_of(runs) == pytest.approx(0.00058, rel=0.1)
    fine = _runs(replace(SIM_WHEELS["r3"], reading_hz=60.0))
    assert quantum_of(fine) == 0.0
    assert not sticky(replace(SIM_WHEELS["r3"], breakaway=breakaway_of(fine))), "the R3 starts on its friction"
    assert breakaway_of(fine[:0]) == 0.0, "too few starts read nothing"


def _stand(servo: ServoCommand, seconds: float, plugin: Plugin | None = None, steer: float = 0.0) -> Plugin:
    plugin = plugin or Plugin()
    for i in range(round(seconds * 120)):
        plugin.step(servo, 0.0, steer, 1 / 120, fresh=i % 2 == 0)
    return plugin


def test_the_plugin_mirror_climbs_toward_the_breakaway_while_the_wheel_stands():
    servo = ServoCommand(target=0.0, kp=0.0, ki=0.0, cap=3000, direction=1, rate=0.5, ff_friction=200.0,
                         ff_breakaway=480.0, ff_plan=True, plan_rate=0.5)
    way = math.tanh(25.0)
    early = _stand(servo, 0.1)
    assert 200.0 * way < early.ff < 300.0 * way, "0.1 s standing: on its way up"
    plugin = _stand(servo, 0.45)
    assert plugin.ff == pytest.approx(480.0 * way), "after the ramp (0.3 s): the ceiling"
    plugin.step(servo, 0.0, 0.001, 1 / 120, fresh=True)
    assert plugin.ff == pytest.approx(200.0 * way), "the reading changed: the sliding friction at once"
    assert _stand(replace(servo, ff_breakaway=0.0), 0.45).ff == pytest.approx(200.0 * way), "off by default"
    still = replace(servo, rate=0.0, plan_rate=0.0, target=0.001)
    assert _stand(still, 0.45).ff == 0.0, "1 mlock short: no push at all (21:45: the rest push hunted it at 4 Hz)"
    far = _stand(replace(still, target=0.01), 0.45)
    assert far.ff == pytest.approx(480.0 * math.tanh(5.0)), "10 mlock short: the whole climb"
    plain = _stand(replace(still, ff_breakaway=0.0), 0.45)
    assert plain.ff == pytest.approx(200.0 * math.tanh(0.5)), "without the breakaway push the rest push is as it was"


def test_the_mirror_keeps_the_plugins_breakaway_constants():
    text = (ROOT / "plugins" / "ffb" / "ffb_signal.h").read_text(encoding="utf-8")
    for name, value in (("FFB_STUCK_FROM", servo_sim.STUCK_FROM), ("FFB_BREAK_RAMP_S", servo_sim.BREAK_RAMP_S),
                        ("FFB_BREAK_BAND", servo_sim.BREAK_BAND), ("FFB_BREAK_HOLD", servo_sim.BREAK_HOLD),
                        ("FFB_FF_REST_BAND", servo_sim.FF_REST_BAND)):
        assert float(re.search(rf"#define\s+{name}\s+(\S+)", text).group(1)) == value, name


def test_the_replay_moves_the_push_ahead_on_the_plan_the_run_had():
    design = Design(bandwidth_hz=1.0, damping=1.0, inertia=650.0, damping_first=True,
                    **push_ahead_of(SIM_WHEELS["r3"]))
    params = replace(SIM_WHEELS["r3"], reading_hz=60.0)
    bench = sim_bench(params)
    bench.say = lambda _m: None
    run = bench.run_stage(ramp(0.15, 0.4, 0.1), design, 1, 20.0, 0.0)

    def worst(r: dict) -> float:
        ts, ys = replay(r, params, bench.frame_hz)
        return max(abs(s[1] - interp(ts, ys, s[0])) for s in run["samples"])

    without = worst(dict(run, design=dict(run["design"], plan_s=0.0)))
    assert worst(run) < 0.6 * without and worst(run) < 0.02, (
        "the plan is rebuilt from the stage (what is left is the reading's phase: 4 mlock on a plain design)")


def test_rims_that_agree_as_well_are_designed_on_the_lowest_friction_and_held_on_the_rest():
    best = replace(STICKY, stick=358.0, inertia=30.0)
    low = replace(STICKY, stick=150.0, inertia=120.0)
    rim, also = design_rim(best, [low, replace(STICKY, stick=250.0)])
    assert rim == low and best in also and len(also) <= 2, "under-pushing the friction is the safe side"
    assert design_rim(best, []) == (best, [])


class _Blackout(SimChannel):
    """The game stops its force from ``off`` for ``seconds`` (another window in front), the steering goes on."""

    def __init__(self, params, clock, off: float, seconds: float) -> None:
        super().__init__(params, clock)
        self.off, self.seconds = off, seconds

    def publish(self, *args, **kwargs):
        st = super().publish(*args, **kwargs)
        t = self.clock() - self.t0
        if self.off <= t < self.off + self.seconds:
            self.wheel.plugin.value = 0.0
            return replace(st, frames=self.frozen, servo_applied=0, applied=0)
        self.frozen = st.frames
        return st


def test_a_stage_the_game_stopped_its_force_in_runs_again_once_it_is_back():
    clock = SimClock()
    said: list[str] = []
    bench = Bench(_Blackout(STICKY, clock, off=3.0, seconds=2.0), clock, clock.sleep, say=said.append)
    run = bench.run_stage(step(0.02), GENTLE, 1, 20.0)
    assert any("force feedback stopped" in m for m in said) and any("runs again" in m for m in said)
    assert summarize(run)["peak_servo"] > 0, "the stage that is kept ran with the force on"
    other = SimClock()
    stuck = Bench(_Blackout(STICKY, other, off=0.5, seconds=1e9), other, other.sleep, say=said.append)
    with pytest.raises(BenchAbort, match="stayed off"):
        stuck.run_stage(step(0.02), GENTLE, 1, 20.0)


def _probe(params: WheelParams) -> dict:
    bench = sim_bench(params)
    bench.say = lambda _m: None
    _, direction, home = bench.prepare(1, True)
    return slide_probe(bench, direction, home, say=lambda _m: None)


def test_the_slide_probe_tells_a_wheel_that_slides_on_little_from_one_that_needs_most_of_its_breakaway():
    base = replace(STICKY, breakaway=448.0, spring=570.0)
    low = _probe(replace(base, inertia=120.0, damping=205.0, stick=150.0, delay_s=0.042))
    high = _probe(replace(base, inertia=30.0, damping=410.0, stick=358.0, delay_s=0.027))
    assert 400.0 < low["breakaway"] < 560.0 and 400.0 < high["breakaway"] < 560.0, "both start near 448"
    assert low["slide_high"] is not None and low["slide_high"] < 200.0, "the old wheel's two rims: 150 keeps sliding"
    assert high["slide_high"] is None and high["slide_low"] > 300.0, "and 358 stops at every hold"
    assert all(f < 200.0 for f in bracket_frictions(low)) and all(f > 300.0 for f in bracket_frictions(high))
    sticky_probe = _probe(STICKY)
    assert sticky_probe["slide_low"] < STICKY.stick < sticky_probe["slide_high"]
    assert bracket_frictions({"breakaway": 0.0}) == []
    assert slide_estimate({"breakaway": 434.0, "slide_low": 298.0, "slide_high": None}) == 298.0, "17:16: all stopped"
    assert slide_estimate({"breakaway": 500.0, "slide_low": 150.0, "slide_high": 250.0}) == 200.0
    assert slide_estimate({"breakaway": 0.0}) is None


def test_a_sticky_wheel_gets_a_design_with_the_breakaway_push():
    bench = sim_bench(STICKY)
    bench.say = lambda _m: None
    _, direction, home = bench.prepare(1, True)
    assert bench.breakaway
    data = run_check(bench, direction, home, evals=200, say=lambda _m: None,
                     grid={"bandwidths": (0.5, 0.75), "dampings": (1.0,), "scales": (4.0,)})
    agree, total = data["model_agrees"]
    assert data["verified"] and agree >= 0.8 * total
    assert data["config"]["la_ff_breakaway"] > data["config"]["la_ff_friction"] > 0
    assert 400.0 < data["wheel"]["breakaway"] < 600.0 and data["wheel"]["quantum"] > 0.0
    assert any("to start the wheel" in n for n in data["notes"])
    assert data["probe"]["slide_low"] < STICKY.stick < data["probe"]["slide_high"], "the slide probe ran"


OLD = SIM_WHEELS["old"]


def test_the_model_wheel_starts_on_its_own_force_each_time():
    starts = []
    for seed in range(3):
        wheel = SimWheel(OLD)
        wheel._draws.seed(seed)
        for _ in range(12):
            wheel.standing = False
            wheel._stood()
            starts.append(wheel.start_force)
    assert min(starts) >= 0.8 * OLD.breakaway and max(starts) <= 1.2 * OLD.breakaway
    assert max(starts) - min(starts) > 0.25 * OLD.breakaway, "the old wheel: 311 to 526 from one try to the next"
    assert SimWheel(replace(OLD, breakaway_spread=0.0)).start_force == OLD.breakaway


def test_the_model_of_the_1916_run_stands_on_the_fixed_push_that_left_the_wheel_standing():
    design = Design(bandwidth_hz=0.5, damping=1.0, inertia=150.0, damping_first=True, cap=3000, ff_inertia=30.0,
                    ff_damping=594.8, ff_friction=305.2, ff_breakaway=370.6, plan_s=0.042)

    def up(d: Design) -> dict:
        bench = sim_bench(OLD, 119.8)
        bench.say = lambda _m: None
        _, direction, home = bench.prepare(1, True)
        return summarize(bench.run_stage(step(0.02), d, direction, 20.0, home))["up"]

    assert up(design)["rise_s"] is None, "19:16: the push reached 440, the wheel never rose"
    ramped = up(replace(design, ff_breakaway=560.0))
    assert ramped["rise_s"] is not None and ramped["overshoot"] < 0.2, "the push climbing to 560 starts it"


def test_the_damping_probe_reads_a_slow_sliding_wheel_and_leaves_a_fast_one_to_the_fit():
    for params in (OLD, replace(OLD, damping=1200.0, inertia=120.0)):
        bench = sim_bench(params, 119.8)
        bench.say = lambda _m: None
        _, direction, home = bench.prepare(1, True)
        out = damping_probe(bench, direction, home, say=lambda _m: None)
        assert out["damping"] == pytest.approx(params.damping, rel=0.15), "the push ahead holds within 0.7 to 1.2"
        assert out["spring"] == pytest.approx(params.spring, rel=0.25)
        assert out["friction"] == pytest.approx(params.stick, rel=0.1)
    bench = sim_bench(STICKY)
    bench.say = lambda _m: None
    _, direction, home = bench.prepare(1, True)
    assert damping_probe(bench, direction, home, say=lambda _m: None)["damping"] == 0.0, "too fast for steady slides"
    assert fit_slides([[0.1, 0.0, 400.0]] * 6)["damping"] == 0.0, "one way only reads nothing"


def test_a_wheel_with_a_breakaway_keeps_the_measured_spring_in_the_fit():
    runs = _runs(replace(OLD, breakaway_spread=0.0))
    spring = estimate_spring(runs, 1)
    base = WheelParams(sign=1, reading_hz=60.0, breakaway=436.0, quantum=0.00058, damping=595.0)
    held = fit(runs, 120.0, base, max_evals=60, damping_held=True).params
    assert held.spring == pytest.approx(spring) and held.damping == pytest.approx(595.0)
    assert fit(runs, 120.0, replace(base, breakaway=0.0), max_evals=60).params.spring != pytest.approx(spring)


def test_a_step_that_never_gets_there_is_not_held():
    s = {"buzz_min": 1.0, "up": {"rise_s": None, "overshoot": 0.0, "ripple": 0.0, "final_err": -0.88},
         "swing": {"rise_s": None, "overshoot": 0.0, "ripple": 0.0}}
    assert not rings(s) and not accepted(s), "19:16: a step that stopped 88 % short passed as held"
    s["up"]["final_err"] = -0.15
    assert accepted(s), "a smooth wheel that stops a little short of 90 % still holds"
    s["up"]["rise_s"] = 0.3
    assert accepted(s)


def test_the_old_wheel_model_gets_a_ramped_push_on_its_measured_damping():
    bench = sim_bench(OLD, 119.8)
    bench.say = lambda _m: None
    _, direction, home = bench.prepare(1, True)
    data = run_check(bench, direction, home, evals=200, say=lambda _m: None,
                     grid={"bandwidths": (0.5,), "dampings": (1.0,), "scales": (4.0,)})
    assert data["verified"], "19:16 on the wheel: nothing held"
    cfg = data["config"]
    assert cfg["la_ff_damping"] == pytest.approx(OLD.damping, rel=0.15), "designed on the damping probe"
    assert cfg["la_ff_breakaway"] >= data["probe"]["start_max"], "the push climbs past the highest start"
