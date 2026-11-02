"""The LA's 20 Hz debug recorder: written only with ``Settings.debug``, no coordinates in it."""

from __future__ import annotations

import csv

import pytest

from core.la import runner as runner_mod
from core.la.calibration import Calibration, Estimate
from core.la.runner import AssistRunner
from core.lane_departure.command import PluginStatus
from core.settings import Settings
from tests.lane_departure.test_detector import DT, frame_at, make_route

GAIN = 0.19


@pytest.fixture
def settings(monkeypatch):
    s = Settings.instance()
    monkeypatch.setattr(s, "la_force_per_ms2", 720.0, raising=False)
    monkeypatch.setattr(s, "la_kappa_per_steer", GAIN, raising=False)
    return s


def drive(runner, seconds=4.0):
    route = make_route(None, length=2500.0)
    for i in range(int(seconds / DT)):
        status = PluginStatus(present=True, hooked=True, version=4, steer_ok=True, steer_frames=i + 1)
        runner.step(frame_at(i * DT, route, None, 0.4, 0.0), False, status, False)
    runner.close()


def test_it_records_while_assisting_with_debug_on(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(settings, "debug", True, raising=False)
    clock = iter(i * DT for i in range(10_000))
    monkeypatch.setattr(runner_mod.time, "monotonic", lambda: next(clock))
    drive(AssistRunner(debug_root=tmp_path))
    with (tmp_path / "la_debug.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and {"steer", "target", "target_rate", "kp", "kd", "cancel", "game_force"} <= set(rows[0])
    assert {"hand", "steer_accel", "pulse_push", "hand_against", "servo_only", "rumbling"} <= set(rows[0])
    assert not any(k in rows[0] for k in ("x", "z", "pos_x", "pos_z")), "no coordinates"
    assert any(r["reason"] == "assist" for r in rows)


def test_nothing_is_written_without_debug(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(settings, "debug", False, raising=False)
    drive(AssistRunner(debug_root=tmp_path))
    assert not (tmp_path / "la_debug.csv").exists()


def test_a_ready_calibration_is_kept_for_the_next_start(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(settings, "la_saved_calibration", {}, raising=False)
    runner = AssistRunner(debug_root=tmp_path)
    runner._keep_calibration(Calibration(force_per_ms2=900.0, state="held",
                                         kappa_per_steer=Estimate(0.2, state="held")))
    assert runner_mod.saved_calibration() is None, "only live fits ready on both are kept"
    runner._keep_calibration(Calibration(force_per_ms2=706.0, state="ready",
                                         kappa_per_steer=Estimate(0.179, state="ready")))
    assert runner_mod.saved_calibration() == (706.0, 0.179)
