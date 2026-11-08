"""The worse-wheel model features (tools/la_bench/plant.py) and the simulated population and its hold metrics
(tools/la_bench/population.py). All simulated.
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from tools.la_bench.plant import SIM_WHEELS, SimWheel, WheelParams
from tools.la_bench.population import (HELD_OUT_EVERY, POPULATION_SIZE, _flips, force_scale, population, presets,
                                       split, unresolved)

NEW = ("dwell_s", "backlash", "stick_speed", "stall_every_s", "stall_s")


def _push(p: WheelParams, force: float, seconds: float) -> SimWheel:
    wheel = SimWheel(p)
    while wheel.time < seconds:
        wheel.substep(torque=force)
    return wheel


def test_the_new_features_are_off_by_default():
    defaults = {f.name: f.default for f in fields(WheelParams)}
    assert all(defaults[k] == 0.0 for k in NEW)
    for p in SIM_WHEELS.values():
        assert all(getattr(p, k) == 0.0 for k in NEW), "the five presets replay as before"


def test_backlash_lets_the_motor_move_in_its_play_while_the_rim_stands():
    p = replace(SIM_WHEELS["sticky"], backlash=0.002, quantum=0.0)
    w = _push(p, 150.0, 0.5)
    assert 0.0005 < w.steer <= 0.0015 and w.rim == pytest.approx(0.0), "150 is under the rim's friction"
    assert _push(replace(p, backlash=0.0), 150.0, 0.5).steer == 0.0


def test_dwell_stiction_restarts_a_wheel_stopped_a_moment_ago_on_less():
    p = replace(SIM_WHEELS["sticky"], dwell_s=0.5, quantum=0.0)
    w = SimWheel(p)
    assert w._start_force() == pytest.approx(p.breakaway), "at rest from the start: the whole breakaway"
    w.standing, w.stood_at = True, w.time
    assert w._start_force() == pytest.approx(p.stick), "just stopped: the sliding friction"
    w.n = int(2.0 / w.h)
    assert w._start_force() > 0.95 * p.breakaway


def test_sliding_friction_can_rise_with_speed():
    p = replace(SIM_WHEELS["sticky"], stick_speed=200.0)
    w = SimWheel(p)
    assert w._slide(0.0) == p.stick and w._slide(0.5) == pytest.approx(p.stick + 200.0, rel=0.01)


def test_reading_stalls_hold_the_reading():
    p = replace(SIM_WHEELS["r3"], reading_hz=60.0, stall_every_s=0.5, stall_s=0.1)
    w = SimWheel(p)
    gaps, last, at = [], None, 0.0
    while w.time < 10.0:
        w.substep(torque=400.0 if int(w.time) % 2 else -400.0)
        if w.readings != last:
            gaps.append(w.time - at)
            last, at = w.readings, w.time
    assert max(gaps) >= 0.05 and max(gaps) <= 0.12, "stalls of up to stall_s"


def test_the_population_repeats_and_holds_out_a_fifth():
    a, b = population(), population()
    assert a == b and len(a) == POPULATION_SIZE
    tune, held = split(a)
    assert len(held) == POPULATION_SIZE // HELD_OUT_EVERY and not set(n for n, _ in held) & set(n for n, _ in tune)
    for _, p in a:
        assert 0.02 <= p.delay_s <= 0.10 and 300.0 <= p.breakaway <= 700.0 and 30.0 <= p.reading_hz <= 60.0
        assert 0.0 <= p.quantum <= 0.005 and p.stick < p.breakaway
        assert p.backlash == 0.0 or 0.001 <= p.backlash <= 0.002
    assert len(presets()) == 5


def test_flips_need_the_hysteresis_and_the_reading_limits_the_final_error():
    assert _flips([10.0, -10.0, 10.0], 20.0) == 0
    assert _flips([30.0, -30.0, 5.0, 30.0], 20.0) == 2
    assert unresolved(0.02, 0.0) == 0.0
    assert unresolved(0.02, 0.004) == pytest.approx(0.002), "0.02 lies on a step: half a step"
    assert unresolved(0.021, 0.004) == pytest.approx(0.003)
    assert force_scale(SIM_WHEELS["r3"]) == SIM_WHEELS["r3"].stick
    assert force_scale(SIM_WHEELS["old"]) == SIM_WHEELS["old"].breakaway
