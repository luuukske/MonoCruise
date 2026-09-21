"""EgoPathModel: steer-led path, learned gain, grip cap. See core/radar/README.md §11."""

import math

from core.radar.ego_path_model import (
    EgoPathModel, EgoPathParams, GAIN_MAX, GAIN_MIN, warm_gain,
)

_DT = 1.0 / 30.0


def _drive(
    model: EgoPathModel,
    *,
    steer: float,
    kappa_true: float | None = None,
    speed: float = 16.0,
    seconds: float = 2.0,
    t0: float = 100.0,
    yaw0: float = 0.0,
):
    """Step the model along a vehicle that actually holds ``kappa_true``."""
    t = t0
    yaw = yaw0
    states = []
    kappa = model.gain * steer if kappa_true is None else kappa_true
    for _ in range(int(seconds / _DT)):
        yaw += kappa * speed * _DT
        t += _DT
        states.append(model.step(t, yaw, speed, steer))
    return states


def test_linear_regime_is_exactly_gain_times_steer():
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    states = _drive(model, steer=0.10, kappa_true=model.gain * 0.10)
    last = states[-1]
    assert last.kappa_path == last.kappa_steer == model.gain * 0.10
    assert not last.saturated


def test_gain_adapts_down_to_a_long_vehicle():
    """A bus turns less per unit steer; the learner has to follow it there."""
    model = EgoPathModel(params=EgoPathParams())
    bus_gain = 0.12
    for i in range(6):
        steer = 0.08 if i % 2 == 0 else -0.08
        _drive(
            model, steer=steer, kappa_true=bus_gain * steer,
            speed=12.0, seconds=6.0, t0=100.0 + i * 6.0,
        )
    assert abs(model.gain - bus_gain) < 0.01


def test_gain_does_not_learn_from_saturated_cornering():
    """High lateral load underperformance is grip, not geometry: the cap's job."""
    params = EgoPathParams()
    model = EgoPathModel(params=params)
    start = model.gain
    # steer asks for 0.03, vehicle holds 0.012 at 9 m/s^2 lateral: pure saturation.
    _drive(model, steer=0.158, kappa_true=0.012, speed=27.0, seconds=8.0)
    assert model.gain == start


def test_gain_stays_inside_its_band():
    model = EgoPathModel(params=EgoPathParams())
    _drive(model, steer=0.10, kappa_true=0.001, speed=12.0, seconds=20.0)
    assert GAIN_MIN <= model.gain <= GAIN_MAX


def test_cap_follows_the_measured_line_once_saturated():
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    held = 0.030
    states = _drive(model, steer=0.60, kappa_true=held, speed=17.0, seconds=1.0)
    assert states[-1].saturated
    # Capped onto the line the vehicle actually holds, not the wheel's demand.
    assert math.isclose(states[-1].kappa_path, held * 1.10, rel_tol=0.05)
    assert abs(states[-1].kappa_path) < abs(states[-1].kappa_steer)
    # And it engages promptly: measurement window plus the ramp.
    engaged = next(i for i, s in enumerate(states) if s.saturated)
    assert engaged * _DT < 0.35


def test_cap_arrives_as_a_ramp_not_a_step():
    """A latched cap halved the corridor radius in one frame; the corpus p90
    frame step at engagement was 0.032 1/m against 0.0006 in normal driving."""
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    states = _drive(model, steer=0.60, kappa_true=0.030, speed=17.0, seconds=1.0)
    total = abs(states[0].kappa_path - states[-1].kappa_path)
    steps = [
        abs(b.kappa_path - a.kappa_path) for a, b in zip(states, states[1:])
    ]
    assert max(steps) < 0.25 * total, f"biggest step {max(steps):.4f} of {total:.4f}"
    weights = [s.sat_weight for s in states]
    assert weights[0] == 0.0 and weights[-1] == 1.0
    # Several frames strictly between: that is the difference from a latch.
    assert sum(1 for w in weights if 0.0 < w < 1.0) >= 3


def test_unwinding_the_wheel_is_instant_while_capped():
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    states = _drive(model, steer=0.60, kappa_true=0.030, speed=17.0, seconds=1.0)
    assert states[-1].saturated
    # Same vehicle line, but the driver has come off the lock.
    out = model.step(states[-1].kappa_path and 101.0 + _DT, 0.0, 17.0, 0.02)
    assert out.kappa_path == out.kappa_steer


def test_sign_flip_releases_the_cap_on_the_same_frame():
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    _drive(model, steer=0.60, kappa_true=0.030, speed=17.0, seconds=1.0)
    assert model.state.saturated
    out = model.step(model._hist[-1][0] + _DT, model._hist[-1][1], 17.0, -0.60)
    assert not out.saturated
    assert out.kappa_path == out.kappa_steer


def test_cap_releases_when_the_vehicle_regains_the_line():
    params = EgoPathParams(learn_enabled=False)
    model = EgoPathModel(params=params)
    states = _drive(model, steer=0.60, kappa_true=0.030, speed=17.0, seconds=1.0)
    assert states[-1].saturated
    t = model._hist[-1][0]
    yaw = model._hist[-1][1]
    for _ in range(30):
        kappa = model.gain * 0.60
        yaw += kappa * 17.0 * _DT
        t += _DT
        out = model.step(t, yaw, 17.0, 0.60)
    assert not out.saturated


def test_frozen_simulated_clock_is_a_no_op():
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    _drive(model, steer=0.10, seconds=0.5)
    before = model.state
    same = model.step(model._hist[-1][0], model._hist[-1][1] + 0.5, 16.0, 0.9)
    assert same is before


def test_yaw_wrap_does_not_invent_curvature():
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    states = _drive(
        model, steer=0.05, kappa_true=model.gain * 0.05,
        speed=16.0, seconds=1.0, yaw0=math.pi - 0.05,
    )
    meas = states[-1].kappa_meas
    assert meas is not None
    assert abs(meas - model.gain * 0.05) < 0.002


def test_no_curvature_below_the_speed_gate():
    model = EgoPathModel(params=EgoPathParams(learn_enabled=False))
    out = model.step(100.0, 0.0, 0.2, 0.8)
    assert out.kappa_path == 0.0
    out = model.step(100.1, 0.0, -8.0, 0.8)
    assert out.kappa_path == 0.0


def test_vehicle_change_drops_a_gain_learned_on_the_old_one():
    model = EgoPathModel(params=EgoPathParams())
    model.note_vehicle("vehicle.bus.long")
    for i in range(6):
        steer = 0.08 if i % 2 == 0 else -0.08
        _drive(
            model, steer=steer, kappa_true=0.12 * steer,
            speed=12.0, seconds=6.0, t0=100.0 + i * 6.0,
        )
    assert model.gain < 0.16
    model.note_vehicle("vehicle.scania.r")
    assert model.gain == model.params.gain_prior


def test_warm_gain_recovers_the_vehicle_gain_from_a_clip():
    samples = []
    t, yaw = 100.0, 0.0
    for i in range(int(20.0 / _DT)):
        steer = 0.08 if (i // 180) % 2 == 0 else -0.08
        kappa = 0.12 * steer
        yaw += kappa * 12.0 * _DT
        t += _DT
        samples.append((t, yaw, 12.0, steer))
    assert abs(warm_gain(samples) - 0.12) < 0.02


def test_reset_keeps_a_gain_that_still_describes_the_vehicle():
    model = EgoPathModel(params=EgoPathParams())
    for i in range(4):
        steer = 0.08 if i % 2 == 0 else -0.08
        _drive(
            model, steer=steer, kappa_true=0.12 * steer,
            speed=12.0, seconds=6.0, t0=100.0 + i * 6.0,
        )
    learned = model.gain
    assert learned < 0.17
    model.reset(keep_gain=True)
    assert model.gain == learned
    assert not model.state.saturated
    model.reset()
    assert model.gain == model.params.gain_prior


def test_impact_suspends_the_cap_and_the_learner():
    """A collision spins the vehicle; that yaw is not a line anyone is driving."""
    model = EgoPathModel(params=EgoPathParams())
    _drive(model, steer=0.60, kappa_true=0.030, speed=17.0, seconds=1.0)
    assert model.state.saturated
    t = model._hist[-1][0]
    yaw = model._hist[-1][1]
    gain_before = model.gain
    # Speed collapses in one frame, as an impact does.
    out = model.step(t + _DT, yaw + 0.2, 2.0, 0.60)
    assert not out.saturated
    assert out.kappa_path == out.kappa_steer
    for i in range(20):
        out = model.step(t + (i + 2) * _DT, yaw + 0.2 + 0.05 * i, 2.0, 0.60)
    assert model.gain == gain_before
