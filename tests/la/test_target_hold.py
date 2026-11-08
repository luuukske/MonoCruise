"""The target hold (core/la/hold.py) and the actuator with it: the servo parks on a wheel standing on its target,
comes back when either moves, and never asks for more than the configured servo. Simulated wheels only.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.la.actuator import Gains, Wheel, WheelActuator, WheelTuning
from core.la.hold import HoldTuning, TargetHold
from core.la.wheel_learner import WheelModel
from tools.la_bench.plant import SIM_WHEELS
from tools.la_bench.design import push_ahead_of
from tools.la_bench.population import Design, hold_test, population

DT = 0.05


def _run(hold: TargetHold, readings, target=0.02, rate=0.0, quantum=0.0, pushes=None):
    outs = []
    for i, r in enumerate(readings):
        outs.append(hold.update(DT, target, rate, r, quantum, pushes[i] if pushes else 0.0))
    return outs


def test_a_wheel_standing_on_its_target_is_parked_and_the_gains_fade_out():
    outs = _run(TargetHold(), [0.0201] * 20)
    assert outs[1].gain == 1.0 and not outs[1].held, "not before it stood for still_s"
    assert outs[-1].held and outs[-1].gain == 0.0
    steps = [a.gain - b.gain for a, b in zip(outs, outs[1:])]
    assert max(steps) <= DT / HoldTuning().fade_in_s + 1e-9, "faded, never stepped"
    assert all(o.gain <= 1.0 and o.integral <= 1.0 for o in outs), "it only ever lowers the gains"


def test_it_comes_back_when_the_wheel_or_the_target_moves():
    hold = TargetHold()
    _run(hold, [0.02] * 10)
    assert hold.held
    out = _run(hold, [0.025] * 3)
    assert not out[-1].held and out[-1].gain == 1.0, "pushed off the target: the servo back within 0.05 s"
    _run(hold, [0.02] * 10)
    assert hold.held
    assert not _run(hold, [0.02], target=0.03)[0].held, "a new target is no longer parked on"


def test_a_moving_target_or_a_wheel_off_its_target_is_never_parked():
    assert not any(o.held for o in _run(TargetHold(), [0.02] * 40, rate=0.05))
    assert not any(o.held for o in _run(TargetHold(), [0.025] * 20)), "5 mlock off, before the wide band"


def test_a_coarse_reading_parks_in_the_step_nearest_the_target():
    hold = TargetHold()
    assert hold.band(0.004) == pytest.approx(0.00225)
    assert _run(hold, [0.022] * 10, quantum=0.004)[-1].held, "2 mlock off on a 4 mlock step is on target"


def test_a_motor_rattling_in_its_play_round_the_target_is_parked():
    rattle = [0.0185, 0.0215] * 12
    assert _run(TargetHold(), rattle)[-1].held
    assert not any(o.held for o in _run(TargetHold(), [0.010, 0.030] * 12)), "a 20 mlock swing is no rattle"


def test_a_hunt_round_a_still_target_is_reported_once():
    hold = TargetHold(HoldTuning(rattle=0.0, wide=0.0))
    readings = [0.015, 0.025] * 20
    outs = _run(hold, readings, pushes=[300.0 if i % 2 else -300.0 for i in range(len(readings))])
    assert sum(o.hunted for o in outs) == 1


def test_a_hunt_is_counted_on_pushes_that_matter_for_that_wheel():
    readings = [0.015, 0.025] * 20
    pushes = [60.0 if i % 2 else -60.0 for i in range(len(readings))]
    hold = TargetHold(HoldTuning(rattle=0.0, wide=0.0))
    assert sum(hold.update(DT, 0.02, 0.0, r, 0.0, f, 300.0).hunted for r, f in zip(readings, pushes)) == 1
    hold = TargetHold(HoldTuning(rattle=0.0, wide=0.0))
    assert not any(hold.update(DT, 0.02, 0.0, r, 0.0, f, 700.0).hunted for r, f in zip(readings, pushes)), (
        "60 units either way is no hunt on a wheel that starts on 700")


def test_off_it_does_nothing():
    assert all(o.gain == 1.0 and not o.held for o in _run(TargetHold(HoldTuning(enabled=False)), [0.02] * 20))


def _commands(tune: WheelTuning, steer: float, ticks: int = 20):
    act = WheelActuator(tune)
    gains = Gains(500.0, 0.19)
    wheel = Wheel(ok=True, steer=steer)
    return [act.commands(gains, 0.02, 0.0, 1.0, wheel, 25.0, dt=DT) for _ in range(ticks)], act


def test_the_actuator_without_the_hold_asks_exactly_what_it_did():
    plain, _ = _commands(WheelTuning(), 0.02)
    act = WheelActuator(WheelTuning())
    old = act.commands(Gains(500.0, 0.19), 0.02, 0.0, 1.0, Wheel(ok=True, steer=0.02), 25.0)
    assert all(c == old for c in plain)


def test_the_actuator_parks_within_its_cap_and_keeps_the_direction():
    held, act = _commands(replace(WheelTuning(), hold=True), 0.02)
    assert held[0].kp > 0.0 and held[-1].kp == 0.0 and held[-1].kd == 0.0 and held[-1].ki == 0.0
    assert all(c.cap == held[0].cap and c.direction == held[0].direction for c in held)
    assert act.hunts == 0


def test_the_actuator_takes_the_learned_design_only_when_asked():
    m = WheelModel(start_force=400.0, start_top=600.0, starts=20, rung=4)
    act = WheelActuator(WheelTuning())
    act.learned(m)
    assert act.tune == WheelTuning(), "learn off: the configured design"
    act = WheelActuator(replace(WheelTuning(), learn=True))
    act.learned(m)
    assert act.tune.bandwidth_hz < 2.0 and act.tune.ff_friction > 0.0 and act.tune.ff_breakaway > act.tune.ff_friction


def test_on_a_model_wheel_with_play_the_hold_stops_the_hunt_at_the_target():
    name, p = population()[3]
    assert name == "w003" and p.backlash > 0.0
    design = Design(bandwidth_hz=0.7, damping=1.0, inertia=225.0, damping_first=True,
                    **push_ahead_of(p, breakaway_share=1.2))
    plain = hold_test(p, design)
    parked = hold_test(p, design, hold=HoldTuning())
    assert plain.flips > 10, "without the hold it hunts round the target (23 flips in 2 s)"
    assert parked.flips <= 1 and parked.jitter <= 0.05 and parked.final_mlock <= 0.5
