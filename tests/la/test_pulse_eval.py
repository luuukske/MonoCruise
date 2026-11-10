"""The LKA pulse eval (tools/la_bench/pulse_eval.py) against the LKA model it extends to the bench's wheels."""

from __future__ import annotations

import math

import pytest

from core.la.actuator import WheelTuning
from tests.la.servo_sim import SPRING
from tests.lane_departure.test_lka import events, keep, past_line
from tools.la_bench.plant import SIM_WHEELS, WheelParams
from tools.la_bench.pulse_eval import drivable, pulse_run


def _lka_model_wheel() -> WheelParams:
    """tests/la/servo_sim.Rig's wheel: rings at 1.5 Hz on the game's spring, 0.3 of critical, a frame plus 8 ms late."""
    inertia = SPRING / (2.0 * math.pi * 1.5) ** 2
    return WheelParams(inertia=inertia, damping=0.6 * math.sqrt(SPRING * inertia), stick=0.0, spring=0.0,
                       delay_s=1.0 / 60.0 + 0.008, reading_hz=60.0)


@pytest.mark.parametrize("drift", [0.8, 1.2])
def test_on_the_lka_models_own_wheel_it_gives_the_same_pulses(drift):
    run = pulse_run(_lka_model_wheel(), WheelTuning(), drift, instant=True)
    rows = keep(drift, seconds=14.0)
    assert run.pulses == sum(e == "start" for _, e in events(rows))
    assert run.past_m == pytest.approx(max(0.0, past_line(rows)), abs=0.03)


def test_the_presets_come_to_rest_under_the_games_force_with_resting_hands():
    assert all(drivable(p, "resting") for p in SIM_WHEELS.values())

