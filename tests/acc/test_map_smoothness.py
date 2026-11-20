"""The ACC law's steady map over cruising states has no jumps, no sharp corners and never brakes
less toward more danger. Ratchets: lower a baseline when the map improves, never raise one.
Before 2026-11-20 the soft floor stepped 0.09 m/s^2 wherever the law crossed zero (steps
0.091 / 0.094 / 0.091, corners 9.1 / 10.0 / 1.9 on decel / closing / gap), which pumped steady
following 7 to 12 times in 20 s. See `core/acc/ACC_ARCHITECTURE.md` §8.13 and `tests/acc/map_lint.py`."""
from __future__ import annotations

import pytest

from core.cruise_control_thread import acc_controller
from core.cruise_control_thread.acc_controller import (
    T_HEADWAY_BY_LEVEL_S, AdaptiveCruiseController, _LeadSnapshot,
)
from core.cruise_control_thread.blinker_arbitration import BlinkerState
from core.settings import Settings
from tests.acc import map_lint

BASELINE = {
    ("decel", "step"): 0.012, ("decel", "corner"): 2.22, ("decel", "loosen"): 0.19,
    ("closing", "corner"): 1.57, ("closing", "loosen"): 0.06, ("gap", "corner"): 0.41,
}
# No jump past a hundredth, no braking given back toward danger past two.
TARGET = {"step": 0.01, "loosen": 0.02}
# Reached: no jump along closing speed or gap, and braking never drops as the gap shrinks.
ABSOLUTE = {("closing", "step"): 0.01, ("gap", "step"): 0.01, ("gap", "loosen"): 0.0}


@pytest.fixture(scope="module")
def report():
    return map_lint.measure()


@pytest.mark.parametrize("key", sorted(BASELINE))
def test_the_map_is_no_rougher_than_its_baseline(report, key):
    worst = report.worst[key]
    assert worst.value <= BASELINE[key] + 1e-9, f"{key}: {worst.value:.3f} at {worst.where}"


@pytest.mark.parametrize("key", sorted(ABSOLUTE))
def test_what_is_reached_stays_reached(report, key):
    worst = report.worst[key]
    assert worst.value <= ABSOLUTE[key] + 1e-9, f"{key}: {worst.value:.3f} at {worst.where}"


def test_every_baseline_is_still_short_of_its_target():
    """Once a baseline reaches its target, move it to ABSOLUTE at the target."""
    for (axis, metric), value in BASELINE.items():
        if metric in TARGET:
            assert value > TARGET[metric], f"{axis} {metric} reached {TARGET[metric]}: make it absolute"


@pytest.mark.parametrize("level, kmh, mult, closing, decel", [
    (3, 80.0, 1.0, 1.0, 0.5), (3, 80.0, 0.7, 0.0, 0.0), (2, 50.0, 1.5, 2.0, 1.0),
    (4, 80.0, 1.0, 0.5, 0.3), (1, 50.0, 0.85, 1.0, 0.0),
])
def test_the_linted_map_is_the_controllers_settled_command(level, kmh, mult, closing, decel, monkeypatch):
    """The lint samples a stateless copy of the pipeline; it must stay the controller's own map."""
    previous = Settings.instance().acc_gap_level
    Settings.instance().acc_gap_level = level
    try:
        ctrl = AdaptiveCruiseController()
        cfg, t_h, v = ctrl.config, T_HEADWAY_BY_LEVEL_S[level], kmh / 3.6
        dist = (cfg.s0_m + v * t_h) * mult
        lead = _LeadSnapshot(vid=1, dist_m=dist, v_lead_ms=v - closing, a_lead_ms2=-decel, score=6.0)
        monkeypatch.setattr(ctrl, "_read_acc_snapshot", lambda: ([lead], None, BlinkerState()))
        clock = [100.0]
        monkeypatch.setattr(acc_controller.time, "monotonic", lambda: clock[0])
        cap = 0.0
        for _ in range(360):
            clock[0] += 1.0 / 60.0
            cap = ctrl.accel_cap_ms2(v)
        assert cap == pytest.approx(map_lint.steady_law(cfg, v, dist, closing, decel, t_h), abs=0.01)
    finally:
        Settings.instance().acc_gap_level = previous
