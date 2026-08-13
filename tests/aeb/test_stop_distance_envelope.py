"""Closed-loop stop simulation: does AEB actually stop in the gap it engaged at?

The clip corpus cannot answer this. Its labels were tagged against one engage
point, so moving the entry bar re-labels clips rather than scoring them, and its
replay never touches the brake pedal at all. This drives the shipped required-
decel formula, entry bar and `AEBDecelController` against a brake plant fitted
to 61 recorded braking episodes, and asserts on residual gap.

Plant: dead time then first order on `frac(pedal) * capacity`. Fitted lag was
tau 0.19 s median / 0.31 s p90, dead time 0.12 s.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

import core.sending_thread.pedal_capacity as pc
from core.aeb.calibration import DEFAULT as CAL
from core.aeb.thread import AEBThread, _required_decel_two_frame
from core.sending_thread.accel_to_pedals import brake_curve_fraction
from core.sending_thread.thread import _AEB_MEAS_TAU_S, AEBDecelController

DT = 0.01
PLANT_TAU_MEDIAN, PLANT_TAU_P90, PLANT_DEAD = 0.19, 0.31, 0.12

# (label, capacity m/s2 at pedal 1.0, has_trailer). Measured full-pedal stops.
RIGS = (
    ("bobtail 6w 10.4t", 10.22, False),
    ("single 12w 17t", 12.58, True),
    ("double 18w 24t", 13.89, True),
    ("double loaded 18w 54t", 10.85, True),
)
SPEEDS_KMH = (40, 60, 80, 100, 120)


def _pedal_from_decel(decel: float, max_brake: float) -> float:
    if decel <= 0.0 or max_brake <= 0.1:
        return 0.0
    ratio = min(decel / max_brake, 1.0 - 1e-9)
    arg = -math.log(1.0 - ratio) / 2.4277
    return 0.0 if arg <= 0.0 else min(1.0, arg ** (1.0 / 0.8518))


def _decel_from_pedal(pedal: float, max_brake: float) -> float:
    return max_brake * brake_curve_fraction(pedal)


class _Plant:
    def __init__(self, capacity: float, tau: float) -> None:
        self.capacity, self.tau = capacity, tau
        self.decel = 0.0
        self._hist: list[tuple[float, float]] = []

    def step(self, pedal: float, now: float) -> float:
        self._hist.append((now, pedal))
        applied = 0.0
        for t, p in self._hist:
            if t > now - PLANT_DEAD:
                break
            applied = p
        steady = self.capacity * brake_curve_fraction(applied)
        self.decel += (1.0 - math.exp(-DT / self.tau)) * (steady - self.decel)
        return self.decel


def stop_against_stationary(
    v0_kmh: float, capacity: float, estimate: float, has_trailer: bool,
    plant_tau: float = PLANT_TAU_MEDIAN, start_gap: float = 500.0,
    grade_pct: float = 0.0, cal=None,
) -> dict:
    """Run one AEB stop; returns residual gap, engage range, peak, saturation duty."""
    CAL = globals()["CAL"] if cal is None else cal
    pad = (CAL.stop_buffer_response_trailer_s if has_trailer
           else CAL.stop_buffer_response_s)
    # Downhill: gravity is stolen from brake force and added to the demand.
    downhill = max(0.0, 9.81 * math.sin(math.atan(grade_pct / 100.0)))
    plant = _Plant(capacity, plant_tau)
    ctrl = AEBDecelController()

    effective_max = max(0.1, CAL.ego_decel_frac * estimate - downhill)
    engage_bar = CAL.aeb_engage_frac * max(0.1, estimate - downhill)

    v, gap, measured, now = v0_kmh / 3.6, start_gap, 0.0, 0.0
    engaged, published, last_change = False, 0.0, -1e9
    engage_gap, peak, sat, held = None, 0.0, 0, 0
    reserve = _Reserve(None, 0.0, engaged=False)

    while now < 90.0 and gap > 0.0:
        required = (
            _required_decel_two_frame(
                gap, v, gap, v, CAL, response_s=pad,
                response_dist_m=reserve.released(now, CAL),
            )
            if v > 0.0 else 0.0
        ) + downhill
        if not engaged:
            if required >= engage_bar and v * 3.6 >= CAL.aeb_min_engage_speed_kmh:
                engaged, engage_gap = True, gap
                reserve = _Reserve(pad * v, now)

        target_raw = max(0.0, min(required, effective_max)) if engaged else 0.0
        delta = target_raw - published
        if published <= 1e-6 and target_raw > 0.0:
            published, last_change = target_raw, now
        elif not (abs(delta) < CAL.aeb_target_deadband_ms2
                  and now - last_change < CAL.aeb_target_refresh_min_s):
            slew = CAL.aeb_target_rate_engaged_ms3 * DT
            published = max(0.0, published + max(-slew, min(slew, delta)))
            last_change = now

        ctrl.update_active(engaged)
        pedal = ctrl.step(
            target_decel_ms2=published,
            floor_decel_ms2=min(required, effective_max) if engaged else 0.0,
            demand_decel_ms2=required if engaged else 0.0,
            measured_decel_ms2=measured,
            max_brake_ms2=estimate,
            ff_pedal_fn=_pedal_from_decel,
            decel_from_pedal_fn=_decel_from_pedal,
            has_trailer=has_trailer,
            now=now, dt=DT,
        )
        ctrl.note_applied_pedal(pedal, now)
        if engaged:
            held += 1
            sat += pedal >= 0.999

        brake_decel = plant.step(pedal, now)
        net = brake_decel - downhill
        peak = max(peak, brake_decel)
        measured += (1.0 - math.exp(-DT / _AEB_MEAS_TAU_S)) * (net - measured)
        v = max(0.0, v - net * DT)
        gap -= v * DT
        now += DT
        if v <= 1e-3:
            break

    return {"gap": gap, "engage_gap": engage_gap, "peak": peak,
            "saturated": sat / max(held, 1)}


HELD = replace(CAL, aeb_reserve_release_s=0.0)


@pytest.mark.parametrize("label,capacity,trailer", RIGS)
@pytest.mark.parametrize("v0", SPEEDS_KMH)
def test_a_correct_estimate_always_stops_in_time(label, capacity, trailer, v0):
    """With the capacity estimate right, every rig stops short of the obstacle.

    Evaluated with the reserve held, which is the configuration that carries a
    stopping margin. The shipped release deliberately spends it; that cost is
    pinned separately in `test_the_shipped_release_trades_the_stopping_margin`
    so it stays visible instead of quietly relaxing this bound.
    """
    r = stop_against_stationary(v0, capacity, capacity, trailer, cal=HELD)
    assert r["engage_gap"] is not None, f"{label} at {v0} km/h never engaged"
    assert r["gap"] > 0.0, f"{label} at {v0} km/h collided ({r['gap']:.2f} m)"


def test_the_shipped_release_trades_the_stopping_margin():
    """The measured price of `aeb_reserve_release_s`, kept in front of us.

    Releasing the reserve is what keeps AEB responsive: it comes off the command
    cap mid-stop, so a lead braking harder gets an actual increase (0.13 s to
    answer a step) instead of "already at maximum". The cost is the margin. The
    stop then lands on the obstacle at walking-pace-zero rather than short of it,
    on every trailer rig, and no `stop_buffer` or partial release recovers it:
    swept 0.2-1.8 m and 70-100% release, every combination is negative at p90
    brake lag, and `stop_buffer` at 0.7 and above also brakes while creeping up
    to a queue. The conflict is set by the engage bar, not by these knobs.
    """
    assert CAL.aeb_reserve_release_s > 0.0, "shipped configuration is released"
    for _, capacity, trailer in RIGS:
        if not trailer:
            continue
        released = stop_against_stationary(100, capacity, capacity, trailer)
        held = stop_against_stationary(100, capacity, capacity, trailer, cal=HELD)
        assert held["gap"] > released["gap"] + 1.0, (
            "holding the reserve must still be the higher-margin option"
        )
        assert released["gap"] > -0.5, (
            f"released stop must still finish at the bumper, not through it "
            f"({released['gap']:.2f} m)"
        )


def test_the_measured_loaded_double_survives_the_model_over_reading_it():
    """2026-08-12 probe, and the reason the ceiling came down to 1.00.

    Four full-pedal stops on one double, cargo the only variable, peak-A method:
    24.3 t measured 15.10 m/s2 against a model 13.91 (model 8% low, safe), and
    40.5 t measured 11.37 against a model 11.87 (model 4% high, the collision
    direction). At the old 1.05 ceiling the loaded rig would have been believed
    at 12.46 against a real 11.30, which is 1.10x and collides at 80-120 km/h.
    """
    true_cap, model = 11.30, 11.87
    for v0 in (80, 100, 120):
        ceiling = stop_against_stationary(
            v0, true_cap, model * pc._BRAKE_SCALE_MAX, True, plant_tau=PLANT_TAU_P90,
        )
        assert ceiling["gap"] > 0.0, (
            f"{v0} km/h collides on the measured loaded double "
            f"({ceiling['gap']:.2f} m)"
        )


@pytest.mark.parametrize("label,capacity,trailer", RIGS)
def test_under_reading_capacity_is_the_safe_direction(label, capacity, trailer):
    """An under-read engages early and brakes soft, but must never collide."""
    for v0 in SPEEDS_KMH:
        r = stop_against_stationary(v0, capacity, capacity * 0.7, trailer)
        assert r["gap"] > 0.0, f"{label} at {v0} km/h collided while under-reading"


@pytest.mark.parametrize("label,capacity,trailer", RIGS)
def test_the_response_pad_covers_the_slow_plant_tail(label, capacity, trailer):
    """The pad is the only entry margin left, so it must survive p90 brake lag."""
    for v0 in SPEEDS_KMH:
        r = stop_against_stationary(v0, capacity, capacity, trailer,
                                    plant_tau=PLANT_TAU_P90)
        assert r["gap"] > -0.1, (
            f"{label} at {v0} km/h ran {-r['gap']:.2f} m past the obstacle "
            "with p90 brake build-up"
        )


@pytest.mark.parametrize("label,capacity,trailer", RIGS)
@pytest.mark.parametrize("grade_pct", (4, 6, 8))
def test_a_descent_engages_earlier_not_later(label, capacity, trailer, grade_pct):
    """The two bases diverge on a grade, and the divergence is the safe way round.

    `capability_decel` loses the gravity term while `effective_required` gains it,
    so in terms of raw required decel the bar is
    `frac * (capacity - downhill) - downhill`, which falls about twice as fast as
    capability does. On an 8% descent a bobtail engages on 7.2 m/s2 of required
    decel instead of 8.7, and finishes with more gap in hand, not less.
    """
    for v0 in SPEEDS_KMH:
        flat = stop_against_stationary(v0, capacity, capacity, trailer, cal=HELD)
        down = stop_against_stationary(v0, capacity, capacity, trailer,
                                       grade_pct=grade_pct, cal=HELD)
        assert down["gap"] > 0.0, f"{label} at {v0} km/h collided on {grade_pct}%"
        assert down["gap"] >= flat["gap"] - 1e-6, (
            f"{label} at {v0} km/h has less margin downhill than on the flat"
        )


class _Reserve:
    """Minimal stand-in for the engagement state `_released_reserve_m` reads."""

    def __init__(self, latched, at_mono, engaged=True):
        self._engage_pad_dist_m = latched
        self._engage_pad_at_mono = at_mono
        self._engaged = engaged

    released = AEBThread._released_reserve_m


def test_the_build_up_reserve_is_latched_at_engagement():
    """The reserve pays for build-up, which happens once, at the start.

    Charging `response_s * v_closing` every tick hands the metres back as ego
    slows: demand decays with no external cause and the brake fades out. On clip
    e9fb04c9 the command fell 10.68 -> 0.41 m/s2 while the warn still sounded.
    Latching it at engagement removes that decay.
    """
    latched = CAL.stop_buffer_response_trailer_s * 27.8   # ~11 m at 100 km/h

    assert _Reserve(latched, 0.0, engaged=False).released(1.0, CAL) is None, (
        "disengaged must keep the live term the entry decision needs"
    )
    assert _Reserve(latched, 0.0).released(0.0, CAL) == pytest.approx(latched)

    # Shipped: bled off over the measured build-up, so demand goes fully live.
    assert CAL.aeb_reserve_release_s > 0.0
    mid = _Reserve(latched, 0.0).released(CAL.aeb_reserve_release_s / 2, CAL)
    assert 0.0 < mid < latched
    gone = _Reserve(latched, 0.0).released(CAL.aeb_reserve_release_s * 2, CAL)
    assert gone == pytest.approx(0.0)

    # Setting it to 0 holds the reserve for the whole event instead.
    held = replace(CAL, aeb_reserve_release_s=0.0)
    assert _Reserve(latched, 0.0).released(30.0, held) == pytest.approx(latched)


def test_releasing_the_reserve_buys_headroom_to_answer_a_lead():
    """Why the release is shipped: it gets the command off the command cap.

    Held, the command sits at the cap for ~80% of the event, so a lead that
    suddenly brakes harder gets no increase because AEB is already at maximum.
    Released, it tracks what the threat actually needs and has somewhere to go.
    """
    released = stop_against_stationary(100, 13.89, 13.89, True)
    held = stop_against_stationary(100, 13.89, 13.89, True, cal=HELD)
    assert released["saturated"] < held["saturated"] - 0.1, (
        f"released {released['saturated']:.0%} vs held {held['saturated']:.0%}: "
        "the release must leave room to brake harder"
    )


def test_the_entry_bar_is_not_discounted_twice():
    """Entry asks what the truck can do; `ego_decel_frac` only caps the command.

    Both halves of the "AEB steps in far too early and then crawls to a stop"
    report, priced separately on an 18-wheel double at 100 km/h. The double hedge
    is the small one: 3.7 m. Reading capacity as 8.90 instead of 13.89, which is
    what 44 recorded engagements actually did, is worth 20 m on top, and it also
    caps `effective_max_decel` at 8.0 on a truck that has 13.9.
    """
    capacity, v = 13.89, 100 / 3.6
    stale = 8.90
    pad = CAL.stop_buffer_response_trailer_s

    def engage_gap(bar: float) -> float:
        """Gap at which required decel first reaches `bar`, pad included."""
        return (v * v) / (2.0 * bar) + CAL.stop_buffer + pad * v

    shipped = stop_against_stationary(100, capacity, capacity, True)
    assert shipped["engage_gap"] == pytest.approx(
        engage_gap(CAL.aeb_engage_frac * capacity), abs=0.5
    )

    hedged = engage_gap(CAL.aeb_engage_frac * CAL.ego_decel_frac * capacity)
    stale_cap = engage_gap(CAL.aeb_engage_frac * CAL.ego_decel_frac * stale)
    assert 3.0 < hedged - shipped["engage_gap"] < 5.0
    assert stale_cap - hedged > 18.0, "capacity error must dominate the hedge"

    assert shipped["peak"] > 0.85 * capacity, "and it must use the truck it has"
