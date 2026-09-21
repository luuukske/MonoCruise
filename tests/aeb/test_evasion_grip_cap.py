"""The grip cap reaches the escape arcs only while the tires are at the cap.

e36c64d bounded both evasion arcs by ``EgoPathState.kappa_cap``, but the model
published that ceiling on every frame where a yaw measurement agreed in sign
with the wheel, armed or not. Over a 200 clip sample it bit 78.1% of moving
frames, 98.5% of them with saturation unarmed, and 22.3% of frames lost both
escape arcs at once: the corridor collapsed onto the turn ego was already in,
and EgoEvasionFilter cannot suppress a target that both arcs still hit.
"""

from __future__ import annotations

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.thread import _evasion_kappas
from core.radar.ego_path_model import EgoPathModel, EgoPathParams

_DT = 1.0 / 30.0


def _drive(model: EgoPathModel, *, steer: float, kappa_true: float,
           speed: float, seconds: float):
    """Step the model along a vehicle that actually holds ``kappa_true``."""
    t, yaw, states = 100.0, 0.0, []
    for _ in range(int(seconds / _DT)):
        yaw += kappa_true * speed * _DT
        t += _DT
        states.append(model.step(t, yaw, speed, steer))
    return states


def _arcs(state, speed: float) -> tuple[float, float]:
    return _evasion_kappas(
        state.kappa_path, speed, CAL, state.kappa_cap, state.sat_weight,
    )


def test_ordinary_bend_keeps_the_whole_evasion_offset():
    """A motorway bend at 2.3 m/s2 lateral: the 0.08 g is there for the taking."""
    speed = 20.0
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    states = _drive(model, steer=0.03, kappa_true=model.gain * 0.03,
                    speed=speed, seconds=2.0)
    last = states[-1]
    assert last.kappa_cap is None and last.sat_weight == 0.0
    left, right = _arcs(last, speed)
    uncapped_left, uncapped_right = _evasion_kappas(last.kappa_path, speed, CAL)
    assert left == uncapped_left and right == uncapped_right
    # And the corridor is genuinely wider than the path on both sides.
    assert left > last.kappa_path > right


def test_saturated_corner_caps_the_arc_that_asks_for_more_grip():
    speed = 17.0
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    states = _drive(model, steer=0.60, kappa_true=0.030, speed=speed, seconds=1.0)
    last = states[-1]
    assert last.saturated and last.kappa_cap is not None
    left, right = _arcs(last, speed)
    uncapped_left, _ = _evasion_kappas(last.kappa_path, speed, CAL)
    # Turning tighter into a corner the truck is already sliding through is
    # not an escape route, so that arc is pulled back to the ceiling.
    assert left < uncapped_left
    assert abs(left) <= last.kappa_cap + 1e-9
    # Unwinding costs grip rather than spending it: that arc is untouched.
    assert right == _evasion_kappas(last.kappa_path, speed, CAL)[1]


def test_the_cap_ramps_onto_the_corridor_instead_of_stepping_it():
    speed = 17.0
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    states = _drive(model, steer=0.60, kappa_true=0.030, speed=speed, seconds=1.0)
    lefts = [_arcs(s, speed)[0] for s in states]
    total = abs(lefts[0] - lefts[-1])
    steps = [abs(b - a) for a, b in zip(lefts, lefts[1:])]
    assert total > 0.0
    assert max(steps) < 0.25 * total, f"biggest step {max(steps):.5f} of {total:.5f}"
