"""AEBDecelController: tracks the published target and nulls environment bias.

Regression cover for the 2026-08-11 finding that the engagement slam pinned the
brake at 1.0, so the controller never influenced the pedal. See
docs/aeb_high_speed_stop_overshoot.md.
"""

from __future__ import annotations

import math

import pytest

from core.sending_thread.accel_to_pedals import brake_curve_fraction
from core.sending_thread.thread import (
    _AEB_MEAS_TAU_S,
    _AEB_PLANT_DEAD_SOLO_S,
    _AEB_PLANT_DEAD_TRAILER_S,
    _AEB_PLANT_TAU_SOLO_S,
    _AEB_PLANT_TAU_TRAILER_S,
    AEBDecelController,
)

MAX_BRAKE = 10.0
DT = 0.01


def pedal_from_decel(decel: float, max_brake: float) -> float:
    if decel <= 0.0 or max_brake <= 0.1:
        return 0.0
    ratio = min(decel / max_brake, 1.0 - 1e-9)
    arg = -math.log(1.0 - ratio) / 2.4277
    return 0.0 if arg <= 0.0 else min(1.0, arg ** (1.0 / 0.8518))


def decel_from_pedal(pedal: float, max_brake: float) -> float:
    return max_brake * brake_curve_fraction(pedal)


class Plant:
    """Brake plant: dead time then first order, with a settable capacity error."""

    def __init__(self, true_max: float = MAX_BRAKE, offset: float = 0.0,
                 dead: float = _AEB_PLANT_DEAD_SOLO_S,
                 tau: float = _AEB_PLANT_TAU_SOLO_S) -> None:
        self.true_max = true_max
        self.offset = offset
        self.dead = dead
        self.tau = tau
        self.decel = 0.0
        self._hist: list[tuple[float, float]] = []

    def step(self, pedal: float, now: float, dt: float) -> float:
        self._hist.append((now, pedal))
        applied = 0.0
        for t, p in self._hist:
            if t <= now - self.dead:
                applied = p
            else:
                break
        steady = max(0.0, self.true_max * brake_curve_fraction(applied) + self.offset)
        self.decel += (1.0 - math.exp(-dt / self.tau)) * (steady - self.decel)
        return self.decel


def run(target: float, plant: Plant, duration: float = 2.5,
        has_trailer: bool = False, demand: float | None = None):
    """Drive the controller against `plant`; returns (t, commanded_decel) samples."""
    ctrl = AEBDecelController()
    ctrl.update_active(True)
    smooth = 0.0
    now = 0.0
    out = []
    while now < duration:
        measured = max(0.0, smooth)
        pedal = ctrl.step(
            target_decel_ms2=target,
            floor_decel_ms2=target,
            demand_decel_ms2=target if demand is None else demand,
            measured_decel_ms2=measured,
            max_brake_ms2=MAX_BRAKE,
            ff_pedal_fn=pedal_from_decel,
            decel_from_pedal_fn=decel_from_pedal,
            has_trailer=has_trailer,
            now=now,
            dt=DT,
        )
        ctrl.note_applied_pedal(pedal, now)
        realized = plant.step(pedal, now, DT)
        # Mirror the sending thread's tracking differentiator on the AEB path.
        smooth += (1.0 - math.exp(-DT / _AEB_MEAS_TAU_S)) * (realized - smooth)
        now += DT
        out.append((now, realized, pedal, ctrl.bias_ms2))
    return out


def test_tracks_target_instead_of_saturating():
    """A modest target must not produce a full-brake pedal (the old slam bug)."""
    trace = run(3.0, Plant())
    settled = [row for row in trace if row[0] > 0.8]
    assert max(p for _, _, p, _ in settled) < 0.5
    assert all(abs(d - 3.0) < 0.35 for _, d, _, _ in settled)


def test_nulls_capacity_underestimate():
    """Truck brakes weaker than the curve says: loop must find the extra pedal."""
    trace = run(5.0, Plant(true_max=8.0))
    settled = [d for t, d, _, _ in trace if t > 1.8]
    assert all(abs(d - 5.0) < 0.3 for d in settled)


def test_nulls_constant_environment_bias():
    """Constant offset (grade, engine brake, curve error) is estimated out."""
    for offset in (-1.5, -0.6, 1.0):
        trace = run(5.0, Plant(offset=offset))
        settled = [d for t, d, _, _ in trace if t > 1.8]
        assert all(abs(d - 5.0) < 0.35 for d in settled), (
            f"offset {offset}: settled {settled[-1]:.2f}"
        )


def test_never_overshoots_the_target_decel():
    """Overshoot is the dangerous direction; the model is biased slow to avoid it.

    Covers plants faster and slower than the model, in both load classes.
    """
    cases = [
        (5.0, dict(offset=-1.5), False),
        (5.0, dict(offset=1.0), False),
        (5.0, dict(true_max=12.5), False),
        (2.0, dict(true_max=8.0, offset=-0.6), False),
        (5.0, dict(dead=0.05, tau=0.10), False),
        (5.0, dict(dead=0.08, tau=0.30), False),
        (5.0, dict(offset=-0.6, dead=_AEB_PLANT_DEAD_TRAILER_S, tau=0.35), True),
        (5.0, dict(offset=-0.6, dead=_AEB_PLANT_DEAD_TRAILER_S, tau=0.65), True),
        (5.0, dict(offset=-0.6, dead=_AEB_PLANT_DEAD_TRAILER_S, tau=0.80), True),
    ]
    for target, kwargs, trailer in cases:
        trace = run(target, Plant(**kwargs), has_trailer=trailer)
        peak = max(d for _, d, _, _ in trace)
        assert peak <= 1.15 * target, (
            f"{kwargs} trailer={trailer}: peaked at {peak / target:.2f}x target"
        )


def test_trailer_model_is_slower_than_solo():
    """Load class must actually change the model, or trailers overshoot."""
    solo = AEBDecelController._plant_model(False)
    trailer = AEBDecelController._plant_model(True)
    assert trailer[0] > solo[0] and trailer[1] > solo[1]
    assert solo == (_AEB_PLANT_DEAD_SOLO_S, _AEB_PLANT_TAU_SOLO_S)
    assert trailer == (_AEB_PLANT_DEAD_TRAILER_S, _AEB_PLANT_TAU_TRAILER_S)


def test_floor_keeps_required_decel_when_target_is_stale():
    """A zero/stale published target must not silence AEB."""
    ctrl = AEBDecelController()
    ctrl.update_active(True)
    pedal = ctrl.step(
        target_decel_ms2=0.0,
        floor_decel_ms2=6.0,
        demand_decel_ms2=6.0,
        measured_decel_ms2=0.0,
        max_brake_ms2=MAX_BRAKE,
        ff_pedal_fn=pedal_from_decel,
        decel_from_pedal_fn=decel_from_pedal,
        has_trailer=False,
        now=0.0,
        dt=DT,
    )
    assert pedal == pytest.approx(pedal_from_decel(6.0, MAX_BRAKE), abs=1e-6)


def test_inactive_returns_zero_and_clears_state():
    ctrl = AEBDecelController()
    ctrl.update_active(True)
    for i in range(40):
        ctrl.note_applied_pedal(0.5, i * DT)
        ctrl.step(
            target_decel_ms2=5.0, floor_decel_ms2=5.0, demand_decel_ms2=5.0,
            measured_decel_ms2=0.0,
            max_brake_ms2=MAX_BRAKE, ff_pedal_fn=pedal_from_decel,
            decel_from_pedal_fn=decel_from_pedal, has_trailer=False,
            now=i * DT, dt=DT,
        )
    assert ctrl.bias_ms2 != 0.0
    ctrl.update_active(False)
    assert ctrl.bias_ms2 == 0.0
    assert not ctrl.active
    assert ctrl.step(
        target_decel_ms2=5.0, floor_decel_ms2=5.0, demand_decel_ms2=5.0,
        measured_decel_ms2=0.0,
        max_brake_ms2=MAX_BRAKE, ff_pedal_fn=pedal_from_decel,
        decel_from_pedal_fn=decel_from_pedal, has_trailer=False,
        now=0.0, dt=DT,
    ) == 0.0


def test_unmeetable_demand_goes_straight_to_full_pedal():
    """Demand past the pedal's reach must slam, not sit at the ego_decel_frac cap.

    The 0.9 headroom is a tracking margin. When the threat needs more decel than
    the truck has, there is nothing to track and holding back only costs metres.
    """
    ceiling = decel_from_pedal(1.0, MAX_BRAKE)
    ctrl = AEBDecelController()
    ctrl.update_active(True)

    def one(demand):
        return ctrl.step(
            target_decel_ms2=0.9 * MAX_BRAKE,   # what AEB may publish, capped
            floor_decel_ms2=0.9 * MAX_BRAKE,
            demand_decel_ms2=demand,
            measured_decel_ms2=0.0,
            max_brake_ms2=MAX_BRAKE,
            ff_pedal_fn=pedal_from_decel,
            decel_from_pedal_fn=decel_from_pedal,
            has_trailer=False,
            now=0.0,
            dt=DT,
        )

    # Below the ceiling the capped target still governs, unsaturated.
    assert one(8.0) < 1.0
    # At and beyond it, full pedal on the very first tick.
    assert one(ceiling) == 1.0
    assert one(14.0) == 1.0


def test_saturation_override_does_not_touch_normal_stops():
    """A routine tracked stop must be unchanged by the override."""
    trace = run(5.0, Plant(offset=-0.6))
    assert max(p for _, _, p, _ in trace) < 1.0
    settled = [d for t, d, _, _ in trace if t > 1.8]
    assert all(abs(d - 5.0) < 0.35 for d in settled)
