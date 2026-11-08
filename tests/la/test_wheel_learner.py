"""The wheel learner (core/la/wheel_learner.py): what it reads off normal driving, its bounds, and the design it hands
the servo. Every wheel here is a model (tools/la_bench/plant.py); nothing has been checked on a real wheel.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from core.la.actuator import WheelTuning
from core.la.wheel_learner import (FLAG_RUMBLE, FLAG_SERVO, FORCE_MAX, RUNG_BARE, RUNG_MAX, RUNG_PATIENCE, WheelLearner,
                                   WheelModel, learned_overrides, quantum_of)
from tools.la_bench.learn_sim import la_drive, manual_drive
from tools.la_bench.plant import SIM_WHEELS


def test_the_quantum_is_the_step_the_changes_are_whole_multiples_of():
    rng = random.Random(1)
    assert quantum_of([0.002 * rng.randint(1, 4) for _ in range(300)]) == pytest.approx(0.002)
    assert quantum_of([rng.uniform(0.0002, 0.006) for _ in range(300)]) == 0.0, "a fine encoder reads no grid"
    assert quantum_of([0.00005 * rng.randint(1, 4) for _ in range(300)]) == 0.0, "finer than a step that matters"
    assert quantum_of([0.002] * 5) == 0.0, "too few changes read nothing"


def test_a_manual_drive_teaches_the_reading_rate_the_quantum_and_the_centring():
    for name in ("old", "sticky", "geared"):
        p = replace(SIM_WHEELS[name], reading_hz=60.0)
        learner = WheelLearner()
        manual_drive(p, learner, seconds=30.0)
        m = learner.model
        assert m.reading_hz == pytest.approx(60.0, rel=0.05), name
        assert m.quantum == pytest.approx(p.quantum, rel=0.05), name
        assert m.spring == pytest.approx(p.spring, rel=0.05), name
        assert m.starts == 0 and m.rung == 0, "the servo was off: nothing about friction, no design step"


def test_a_fine_encoder_reads_no_quantum():
    learner = WheelLearner()
    manual_drive(replace(SIM_WHEELS["r3"], reading_hz=60.0), learner, seconds=30.0)
    assert learner.model.quantum == 0.0


def test_las_own_pushes_teach_the_start_force_without_pushing_anything_extra():
    p = SIM_WHEELS["sticky"]
    learner = WheelLearner()
    la_drive(p, learner, episodes=20)
    m = learner.model
    assert m.starts >= 8
    assert 0.6 * p.breakaway <= m.start_force <= 1.5 * p.breakaway, "what a standing wheel started on"
    assert m.start_top >= m.start_force


def test_nothing_is_learned_while_the_plugin_rumbles_or_from_samples_already_seen():
    learner = WheelLearner()
    for i in range(400):
        learner.feed(i / 120.0, 0.002 * (i // 4 % 5), 0.0, 0.0, 0.0, i // 2, FLAG_RUMBLE)
    assert learner.model == WheelModel()
    for i in range(400):
        learner.feed(i / 120.0, 0.0, 0.0, 0.0, 0.0, i // 2, 0)
    assert learner.model == WheelModel(), "the same times again (LA and LKA both feed it): skipped"


def test_unknown_leaves_the_design_as_configured_and_learning_never_stiffens_it():
    base = WheelTuning()
    assert learned_overrides(base, WheelModel()) == {}
    rng = random.Random(3)
    for _ in range(200):
        m = WheelModel(start_force=rng.uniform(0.0, 1500.0), start_top=rng.uniform(0.0, 1500.0),
                       starts=rng.randint(0, 40), slide_force=rng.uniform(0.0, 900.0), slides=rng.randint(0, 600),
                       damping=rng.uniform(0.0, 3000.0), rung=rng.randint(0, RUNG_MAX))
        tune = replace(base, **learned_overrides(base, m))
        assert tune.bandwidth_hz <= base.bandwidth_hz and tune.inertia <= base.inertia
        assert tune.damping >= base.damping
        assert tune.ff_friction <= FORCE_MAX and tune.ff_breakaway <= 1.2 * FORCE_MAX
        if m.starts < 8:
            assert tune.ff_friction == 0.0 and tune.ff_breakaway == 0.0, "no push without evidence"
            if m.rung <= RUNG_BARE + RUNG_PATIENCE:
                assert tune.inertia >= base.inertia * 0.7 ** RUNG_BARE - 1e-9, "no deep step without the push"
            else:
                assert tune.inertia >= base.inertia * 0.7 ** (m.rung - RUNG_PATIENCE) - 1e-9, "only after more hunts"


def test_a_configured_push_ahead_is_kept():
    base = replace(WheelTuning(), ff_friction=300.0, ff_breakaway=450.0)
    m = WheelModel(start_force=500.0, start_top=600.0, starts=20)
    assert "ff_friction" not in learned_overrides(base, m, pushes_set=True)


def test_a_hunt_steps_the_design_down_and_calm_driving_steps_it_back():
    learner = WheelLearner()
    for _ in range(RUNG_MAX + 2):
        learner.note_hunt()
    assert learner.model.rung == RUNG_MAX
    learner.direction = 1
    for i in range(int(601 * 120)):
        learner.feed(i / 120.0, 0.0, 0.0, 0.0, 0.0, i // 2, FLAG_SERVO)
    assert learner.model.rung == RUNG_MAX - 1, "ten minutes of servo time without a hunt"


def test_a_saved_model_is_read_back_bounded_and_garbage_is_unknown():
    m = WheelModel(reading_hz=60.0, quantum=0.00058, quantum_n=300, spring=650.0, start_force=436.0, start_top=520.0,
                   starts=20, rung=2)
    assert WheelModel.from_dict(m.to_dict()) == m
    bad = WheelModel.from_dict({"quantum": "x", "spring": math.nan, "start_force": 1e9, "rung": 99, "reading_hz": -5,
                                "starts": True})
    assert bad.quantum == 0.0 and bad.spring == 0.0 and bad.reading_hz == 0.0 and bad.starts == 0
    assert bad.start_force == FORCE_MAX and bad.rung == RUNG_MAX
    assert WheelModel.from_dict(None) == WheelModel() and WheelModel.from_dict([1, 2]) == WheelModel()
