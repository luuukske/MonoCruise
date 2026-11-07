"""The bench measures the wheel or says why it cannot: another writer on the servo, a game that stalls or reads the
wheel in bunches (the old wheel's 18:01 run, 2026-11-06), and slide probe trials that disagree.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.la_bench.bench import Bench, RunInvalid, reading_trouble, sim_bench
from tools.la_bench.check import run_check, summary
from tools.la_bench.plant import SIM_WHEELS, SimChannel, SimClock
from tools.la_bench.probe import bracket_frictions, slide_estimate, summarize_probe
from tools.la_bench.protocol import Design, step

WHEEL = replace(SIM_WHEELS["sticky"], reading_hz=60.0)
GENTLE = Design(bandwidth_hz=0.7, damping=0.7, inertia=650.0, damping_first=True, cap=2000)


def _times(gaps: list[float], n: int = 300) -> list[float]:
    t, out = 0.0, [0.0]
    for i in range(n):
        t += gaps[i % len(gaps)]
        out.append(t)
    return out


def test_wheel_readings_that_stall_or_come_in_bunches_are_trouble_steady_ones_are_not():
    assert reading_trouble(_times([0.0166, 0.0172, 0.0161])) is None, "60 Hz with jitter"
    assert reading_trouble(_times([0.0083, 0.0091])) is None, "120 Hz"
    assert reading_trouble(_times([0.0167, 0.0333])) is None, "45 frames a second on a 60 Hz screen"
    assert reading_trouble(_times([0.0156])) is None, "a bench polling at Windows' 15.6 ms"
    assert reading_trouble(_times([0.004, 0.006, 0.006, 0.036])) == "the game read the wheel in bunches", "18:01"
    stall = _times([0.0166]) + [5.0 + 0.36]
    assert reading_trouble(stall) == "the game stopped reading the wheel"
    assert reading_trouble([0.0, 0.02, 0.04]) is None, "too few to call bunches"


class _Writer(SimChannel):
    """MonoCruise's lane departure thread publishing beside the bench from ``at`` on."""

    def __init__(self, params, clock, at: float) -> None:
        super().__init__(params, clock)
        self.at = at

    @property
    def foreign(self) -> int:
        return int(self.clock() - self.t0 >= self.at)


def test_another_program_writing_the_servo_stops_the_check_and_keeps_nothing():
    clock = SimClock()
    bench = Bench(_Writer(WHEEL, clock, at=2.0), clock, clock.sleep, say=lambda _m: None)
    with pytest.raises(RunInvalid, match="another program writes to the wheel"):
        bench.run_stage(step(0.02), GENTLE, 1, 20.0)
    clock = SimClock()
    bench = Bench(_Writer(WHEEL, clock, at=4.0), clock, clock.sleep, say=lambda _m: None)
    _, direction, home = bench.prepare(1, True)
    with pytest.raises(RunInvalid):
        run_check(bench, direction, home, evals=20, say=lambda _m: None)


class _Stall(SimChannel):
    """The game stops reading the wheel from ``off`` for ``seconds``; its force goes on."""

    def __init__(self, params, clock, off: float, seconds: float) -> None:
        super().__init__(params, clock)
        self.off, self.seconds, self.held = off, seconds, None

    def publish(self, *args, **kwargs):
        st = super().publish(*args, **kwargs)
        t = self.clock() - self.t0
        if self.off <= t < self.off + self.seconds:
            self.held = self.held or st
            return replace(st, steer=self.held.steer, steer_frames=self.held.steer_frames)
        return st


def test_a_stage_the_game_stopped_reading_the_wheel_in_runs_again():
    clock = SimClock()
    said: list[str] = []
    bench = Bench(_Stall(WHEEL, clock, off=2.0, seconds=0.4), clock, clock.sleep, say=said.append)
    run = bench.run_stage(step(0.02), GENTLE, 1, 20.0)
    assert any("stopped reading the wheel during step 0.02: it runs again" in m for m in said)
    assert reading_trouble([s[0] for s in run["samples"]]) is None, "the run kept is the clean one"
    other = SimClock()
    stuck = Bench(_Stall(WHEEL, other, off=0.5, seconds=1e9), other, other.sleep, say=lambda _m: None)
    with pytest.raises(RunInvalid, match="again and again"):
        stuck.run_stage(step(0.02), GENTLE, 1, 20.0)


def test_finding_the_direction_does_not_run_a_light_rim_past_the_limit():
    light = replace(WHEEL, inertia=30.0, damping=595.0, stick=305.0, breakaway=436.0, spring=100.0)
    for steer0 in (0.0, 0.45, -0.45):
        clock = SimClock()
        bench = Bench(SimChannel(light, clock, steer0=steer0), clock, clock.sleep, say=lambda _m: None)
        assert bench.find_direction() == 1, "19:15 on the old wheel: aborted twice, wheel past 0.4 lock"
        assert 0.0 < bench.rest() - steer0 < 0.1, "it stops pushing once the wheel moved (0.25 in the whole 0.5 s)"


def test_slide_trials_that_disagree_bracket_both_and_say_so():
    held = [(174, False), (144, False), (244.5, True), (275.5, False), (360, True), (420, True)]
    trials = [{"breakaway": 500.0, "held": h, "sliding": s} for h, s in held]
    probe = summarize_probe(trials)
    assert (probe["slide_low"], probe["slide_high"], probe["mixed"]) == (244.5, 275.5, True), "18:01 read 276 to 244"
    assert slide_estimate(probe) == pytest.approx(260.0)
    assert min(bracket_frictions(probe)) == 244 and max(bracket_frictions(probe)) == 276
    clean = summarize_probe([{"breakaway": 450.0, "held": h, "sliding": h > 300} for h, _ in held])
    assert not clean["mixed"] and clean["slide_low"] < clean["slide_high"]
    data = {"wheel": {"inertia": 50.0, "damping": 500.0, "stick": 260.0, "delay_s": 0.05}, "choice": {},
            "smoothness": {"verdict": "steppy", "why": []}, "step_rise_s": None, "verified": False, "probe": probe, "config": {}, "notes": []}
    assert "the trials disagree" in summary(data)
