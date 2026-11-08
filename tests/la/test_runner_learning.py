"""The runners and the wheel learner: switched by ``la_wheel_learning``, fed from the plugin's trace, saved locally,
and never in the way of the assist when it fails."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.la import runner as runner_mod
from core.la.runner import AssistRunner
from core.la.wheel_learner import FLAG_SERVO, WheelLearner, WheelModel
from core.lane_departure.command import PluginStatus
from core.lane_departure.ffb_trace import TraceRead, TraceSample
from core.lane_departure.lka_runner import LkaRunner
from core.settings import Settings
from tests.lane_departure.test_detector import DT, frame_at, make_route


@pytest.fixture
def settings(monkeypatch):
    s = Settings.instance()
    monkeypatch.setattr(s, "la_force_per_ms2", 720.0, raising=False)
    monkeypatch.setattr(s, "la_kappa_per_steer", 0.19, raising=False)
    monkeypatch.setattr(s, "la_wheel_learned", {}, raising=False)
    return s


class _Reader:
    def __init__(self, fail: bool = False) -> None:
        self.t, self.fail = 0.0, fail

    def read(self, now=None) -> TraceRead:
        if self.fail:
            raise OSError("trace block gone")
        out = []
        for _ in range(6):
            self.t += 1 / 120
            seq = int(self.t * 60)
            out.append(TraceSample(self.t, 0.002 * (seq // 3 % 7), 0, 0, 0, seq, 0))
        return TraceRead(True, tuple(out))

    def close(self) -> None:
        return None


def _drive(runner, seconds=2.0) -> None:
    route = make_route(None, length=2500.0)
    for i in range(int(seconds / DT)):
        status = PluginStatus(present=True, hooked=True, version=5, steer_ok=True, steer_frames=i + 1)
        runner.step(frame_at(i * DT, route, None, 0.4, 0.0), False, status, False)


@pytest.mark.parametrize("cls", [AssistRunner, LkaRunner])
def test_the_switch_turns_the_hold_and_the_learned_design_on_and_off(monkeypatch, settings, tmp_path, cls):
    monkeypatch.setattr(settings, "la_wheel_learning", True, raising=False)
    on = cls(debug_root=tmp_path)._tuning()[1].wheel
    monkeypatch.setattr(settings, "la_wheel_learning", False, raising=False)
    off = cls(debug_root=tmp_path)._tuning()[1].wheel
    assert on.hold and on.settle and not off.hold and not off.settle and not off.learn
    assert on.learn == (cls is AssistRunner), "a lane keeping pulse keeps its configured design and pushes"
    assert replace(on, hold=False, learn=False, settle=False) == off, "nothing else in the tuning moves"


def test_the_runner_feeds_the_trace_to_the_learner_and_hands_the_assist_its_model(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(settings, "la_wheel_learning", True, raising=False)
    runner = AssistRunner(debug_root=tmp_path)
    runner._learner, runner._reader = WheelLearner(), _Reader()
    _drive(runner, 12.0)
    m = runner._learner.model
    assert m.reading_hz == pytest.approx(60.0, rel=0.05) and m.quantum == pytest.approx(0.002)
    assert runner._assist.wheel.model == m


def test_off_it_reads_nothing(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(settings, "la_wheel_learning", False, raising=False)
    runner = AssistRunner(debug_root=tmp_path)
    runner._learner, runner._reader = WheelLearner(), _Reader()
    _drive(runner)
    assert runner._learner.model == WheelModel()


def test_a_failing_learner_never_stops_the_assist(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(settings, "la_wheel_learning", True, raising=False)
    runner = AssistRunner(debug_root=tmp_path)
    runner._learner, runner._reader = WheelLearner(), _Reader(fail=True)
    _drive(runner)
    assert runner.output.reason == "assist"


def test_what_was_learned_is_saved_at_most_once_a_minute(monkeypatch, settings, tmp_path):
    clock = iter(i * 10.0 for i in range(10_000))
    monkeypatch.setattr(runner_mod.time, "monotonic", lambda: next(clock))
    runner = AssistRunner(debug_root=tmp_path)
    saved = []
    monkeypatch.setattr(runner_mod.Settings, "save", classmethod(lambda cls, values=None: saved.append(values)))
    for k in range(12):
        runner._keep_learned(WheelModel(reading_hz=60.0, calm_s=float(k)))
    assert 1 <= len(saved) <= 2 and saved[0]["la_wheel_learned"]["reading_hz"] == 60.0
    m = WheelModel.from_dict(saved[-1]["la_wheel_learned"])
    assert m.reading_hz == 60.0


def test_the_learner_ignores_the_servo_while_it_does_not_know_the_direction():
    learner = WheelLearner()
    for i in range(2000):
        learner.feed(i / 120.0, 0.0, 0.0, 600.0, 600.0, i // 2, FLAG_SERVO)
    assert learner.model.starts == 0 and learner.model.calm_s == 0.0
