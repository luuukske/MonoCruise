"""AEBDecelController: tracks the published target and nulls environment bias.

Regression cover for the 2026-08-11 finding that the engagement slam pinned the
brake at 1.0, so the controller never influenced the pedal. See
docs/aeb_high_speed_stop_overshoot.md.
"""

from __future__ import annotations

import math

import pytest

from core.aeb.calibration import DEFAULT as AEB_CAL
from core.sending_thread.accel_to_pedals import brake_curve_fraction
from core.sending_thread.thread import (
    _AEB_MEAS_TAU_S,
    _AEB_PLANT_DEAD_SOLO_S,
    _AEB_PLANT_DEAD_TRAILER_S,
    _AEB_PLANT_TAU_SOLO_S,
    _AEB_PLANT_TAU_TRAILER_S,
    _AEB_PREFILL_MAX_S,
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


class LevelPlant(Plant):
    """ETS2's build-up: a full pedal bites in well under 0.1 s, a quarter pedal in ~0.25 s.

    Fitted on brake_debug.csv onsets (2026-09-29 to 10-06): sent 1.0 tau 0.02-0.10 s,
    sent 0.05-0.3 tau 0.1-0.5 s. Tau is interpolated on the applied pedal.
    """

    def __init__(self, true_max: float = MAX_BRAKE, dead: float = 0.06) -> None:
        super().__init__(true_max=true_max, dead=dead)

    def step(self, pedal: float, now: float, dt: float) -> float:
        self._hist.append((now, pedal))
        applied = 0.0
        for t, p in self._hist:
            if t <= now - self.dead:
                applied = p
            else:
                break
        share = min(1.0, max(0.0, (applied - 0.2) / 0.8))
        tau = 0.25 + (0.05 - 0.25) * share
        steady = self.true_max * brake_curve_fraction(applied)
        self.decel += (1.0 - math.exp(-dt / tau)) * (steady - self.decel)
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
    # A truck braking harder than believed may run 1.25x through the onset guard,
    # then must settle; approved by Lukas 2026-10-04 (README, onset guard).
    cases = [
        (5.0, dict(offset=-1.5), False, 1.15),
        (5.0, dict(offset=1.0), False, 1.25),
        (5.0, dict(true_max=12.5), False, 1.25),
        (2.0, dict(true_max=8.0, offset=-0.6), False, 1.15),
        (5.0, dict(dead=0.05, tau=0.10), False, 1.15),
        (5.0, dict(dead=0.08, tau=0.30), False, 1.15),
        (5.0, dict(offset=-0.6, dead=_AEB_PLANT_DEAD_TRAILER_S, tau=0.35), True, 1.15),
        # Measured trailer plants top out at 0.38 s; the 0.80 s case went 2026-10-04 (README).
        (5.0, dict(offset=-0.6, dead=_AEB_PLANT_DEAD_TRAILER_S, tau=0.65), True, 1.15),
    ]
    for target, kwargs, trailer, limit in cases:
        trace = run(target, Plant(**kwargs), has_trailer=trailer)
        peak = max(d for _, d, _, _ in trace)
        assert peak <= limit * target, (
            f"{kwargs} trailer={trailer}: peaked at {peak / target:.2f}x target"
        )
        if limit > 1.15:
            late = [d for t, d, _, _ in trace if t > 1.8]
            assert all(abs(d - target) < 0.35 for d in late), (
                f"{kwargs}: still {late[-1] / target:.2f}x target after the guard"
            )


def test_does_not_let_go_after_a_fast_hit():
    """Slams build in ~0.15 s solo or with a trailer (brake_debug.csv, 2026-10-04).

    Against the slower model the observer read that as bias and cut the pedal:
    64-77% of target in game at every slider, 0.69-0.74 here without the guard.
    """
    cases = [
        (7.5, dict(true_max=11.0, dead=0.10, tau=0.03), False),
        (7.5, dict(true_max=11.0, dead=0.075, tau=0.08), True),
        (7.5, dict(dead=0.10, tau=0.03), False),
        (5.0, dict(dead=0.10, tau=0.03), False),
    ]
    for target, kwargs, trailer in cases:
        trace = run(target, Plant(**kwargs), has_trailer=trailer)
        hit = next(t for t, d, _, _ in trace if d >= 0.9 * target)
        low = min(d for t, d, _, _ in trace if hit < t <= hit + 1.5)
        assert low >= 0.88 * target, (
            f"{kwargs} trailer={trailer}: fell to {low / target:.2f}x target after the hit"
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

    # Measured at the command: the onset pre-fill is over, so only the override can slam.
    def one(demand):
        return ctrl.step(
            target_decel_ms2=0.9 * MAX_BRAKE,   # what AEB may publish, capped
            floor_decel_ms2=0.9 * MAX_BRAKE,
            demand_decel_ms2=demand,
            measured_decel_ms2=0.9 * MAX_BRAKE,
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


def _time_to(trace, level: float) -> float:
    return next(t for t, d, _, _ in trace if d >= level)


def test_an_emergency_command_bites_at_slam_speed():
    """AEB engages at its bar's share of the truck, so its pedal sits just under full.

    In ETS2 that pedal still builds like gentle braking, about 0.37 s to 90% of the
    command, while the build-up pad assumes a slam. The pre-fill gets there in ~0.15 s.
    """
    target = AEB_CAL.aeb_engage_frac * MAX_BRAKE
    for true_max in (MAX_BRAKE, 0.85 * MAX_BRAKE, 1.2 * MAX_BRAKE):
        reach = min(target, decel_from_pedal(1.0, true_max))
        trace = run(target, LevelPlant(true_max=true_max))
        assert _time_to(trace, 0.9 * reach) <= 0.25, f"true max {true_max}"
        settled = [d for t, d, _, _ in trace if t > 1.8]
        assert all(abs(d - reach) < 0.45 for d in settled), f"true max {true_max}"
    exact = run(target, LevelPlant())
    assert max(d for _, d, _, _ in exact) <= 1.15 * target, "pre-fill must hand over, not slam on"


def test_a_routine_command_never_opens_at_full_pedal():
    """The pre-fill is for a command near the top of the pedal, not for tracked stops."""
    for target in (2.0, 3.0, 5.0):
        trace = run(target, LevelPlant())
        assert max(p for _, _, p, _ in trace) < 1.0, f"target {target}"


def test_the_pre_fill_ends_on_its_own_clock():
    """A measurement that never rises must not hold full pedal past the pre-fill window."""
    ctrl = AEBDecelController()
    ctrl.update_active(True)
    target = AEB_CAL.aeb_engage_frac * MAX_BRAKE
    pedals = []
    now = 0.0
    while now < 1.0:
        pedals.append((now, ctrl.step(
            target_decel_ms2=target, floor_decel_ms2=target, demand_decel_ms2=target,
            measured_decel_ms2=0.0, max_brake_ms2=MAX_BRAKE,
            ff_pedal_fn=pedal_from_decel, decel_from_pedal_fn=decel_from_pedal,
            has_trailer=False, now=now, dt=DT,
        )))
        now += DT
    assert pedals[0][1] == 1.0
    assert all(p < 1.0 for t, p in pedals if t >= _AEB_PREFILL_MAX_S + DT)
