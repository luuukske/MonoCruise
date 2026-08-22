"""Guards on pedal_capacity brake estimate (AEB denominator). See core/sending_thread/README.md."""
from __future__ import annotations

import pytest

import core.sending_thread.pedal_capacity as pc
from core.sending_thread.accel_to_pedals import brake_curve_fraction

BASE = 8.74
DT = 0.033
SPEED = 20.0


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    clk = _FakeClock()
    monkeypatch.setattr(pc.time, "monotonic", clk)
    return clk


@pytest.fixture(autouse=True)
def _no_settings_io(monkeypatch):
    """Learning ticks trigger _maybe_save; keep it away from the real config."""

    class _S:
        pedal_capacity_brake_scale = 0.0
        pedal_capacity_max_brake_ms2 = 0.0
        pedal_capacity_max_accel_ms2 = 0.0
        pedal_capacity_accel_anchor_gain_ms2 = 0.0
        pedal_capacity_accel_ratio_step = 0.0
        mapper_brake_scale_ms2 = 6.5

        @staticmethod
        def save(values=None):
            pass

    monkeypatch.setattr(pc, "Settings", _S)


def _fresh(scale: float = 1.0) -> pc.PedalCapacityTracker:
    t = pc.PedalCapacityTracker()
    t._brake_scale = scale
    t._max_brake_ms2 = scale * BASE
    return t


def _feed(t, clk, pedal, decel, ticks=60, dt=DT, speed=SPEED, aeb=False, baseline=BASE):
    for _ in range(ticks):
        clk.t += dt
        t.update_brake(pedal, decel, speed, 0.0, baseline, road_load_ms2=0.0,
                       aeb_active=aeb)


def test_contaminated_sample_rejected(clock):
    """Reject inflated decel at partial pedal (above 1.35x baseline cap after curve correction)."""
    t = _fresh()
    _feed(t, clock, pedal=0.2, decel=6.0)
    assert t.max_brake_ms2 == pytest.approx(BASE)


def test_legit_strong_brake_recovers_the_estimate_up_to_the_model(clock):
    """Upward learning works, but stops at the model. See `_BRAKE_SCALE_MAX`."""
    t = _fresh(scale=0.85)
    _feed(t, clock, pedal=0.9, decel=8.5)       # candidate ~9.54, scale ~1.09
    assert 0.85 * BASE < t.max_brake_ms2
    assert t.max_brake_ms2 <= BASE + 1e-6


def test_gentle_press_drifts_estimate_slowly(clock):
    """Soft presses still teach, but slowly: a 2 s gentle stop must barely move the estimate, so a
    stretch of abnormal braking cannot poison it (estimate fell 9 -> 4 m/s2 in routine driving,
    2026-07-19)."""
    t = _fresh()
    _feed(t, clock, pedal=0.3, decel=1.2, ticks=60)
    assert t.max_brake_ms2 == pytest.approx(BASE, rel=0.05)
    assert t.max_brake_ms2 < BASE  # direction still correct


def test_aeb_reteaches_fast_after_underperformance(clock):
    """The same settled underperforming stream during an AEB event re-teaches
    the estimate much faster than in normal driving."""
    slow = _fresh()
    fast = _fresh()
    _feed(slow, clock, pedal=1.0, decel=4.0, ticks=60, aeb=False)
    _feed(fast, clock, pedal=1.0, decel=4.0, ticks=60, aeb=True)
    expected = 4.0 / brake_curve_fraction(1.0)   # ~4.39 m/s2
    assert fast.max_brake_ms2 < slow.max_brake_ms2
    assert fast.max_brake_ms2 == pytest.approx(expected, rel=0.05)


def test_heavy_settled_aeb_is_excellent_data(clock):
    """A sustained hard AEB stop with settled decel is full-scale capacity data:
    the estimate must converge on it within the stop, up to the model ceiling."""
    t = _fresh(scale=0.70)
    _feed(t, clock, pedal=1.0, decel=8.5, ticks=60, aeb=True)
    assert t.max_brake_ms2 == pytest.approx(BASE, rel=0.01)


def test_tap_transient_rejected_then_settled_phase_teaches(clock):
    """An AEB tap whose decel is still rising contributes nothing; once the decel flattens, the
    settled phase re-teaches fast. This is the tap poisoning fix: before the decel settle gate,
    the rising phase's mid-transient ratios became the estimate within ~100 ms."""
    t = _fresh()
    # 1 s tap: pedal settled at 0.8, decel ramping 0 -> 6 (never flat).
    for i in range(30):
        clock.t += DT
        t.update_brake(0.8, 6.0 * (i + 1) / 30.0, SPEED, 0.0, BASE,
                       road_load_ms2=0.0, aeb_active=True)
    assert t.max_brake_ms2 == pytest.approx(BASE)

    # Same press held: decel flat at 6.0. Learning resumes once the ramp
    # leaves the decel window and converges onto the honest candidate.
    _feed(t, clock, pedal=0.8, decel=6.0, ticks=40, aeb=True)
    expected = 6.0 / brake_curve_fraction(0.8)   # ~6.93 m/s2
    assert t.max_brake_ms2 == pytest.approx(expected, rel=0.05)


def test_v_dip_rejected(clock):
    """A pedal dip that returns to its old level within one window used to pass the endpoint-only
    settle check while the decel was still chasing the dip. The excursion check + decel gate must
    reject every sample of the dip and its recovery."""
    t = _fresh()
    _feed(t, clock, pedal=0.6, decel=4.0, ticks=60, aeb=True)
    est0 = t.max_brake_ms2

    dip_decels = [3.6, 3.2, 2.9, 2.8, 2.8]
    for d in dip_decels:                       # pedal dips, decel follows
        clock.t += DT
        t.update_brake(0.42, d, SPEED, 0.0, BASE, road_load_ms2=0.0,
                       aeb_active=True)
    recover_decels = [3.0, 3.3, 3.6, 3.8, 3.9, 4.0]
    for d in recover_decels:                   # pedal back, decel recovering
        clock.t += DT
        t.update_brake(0.6, d, SPEED, 0.0, BASE, road_load_ms2=0.0,
                       aeb_active=True)
    _feed(t, clock, pedal=0.6, decel=4.0, ticks=7, aeb=True)
    assert t.max_brake_ms2 == est0


def test_release_gap_blocks_relearn_until_resettled(clock):
    """Between two presses the pedal drops out and the decel decays. The second press must not
    inherit the first press's settled window: no sample may fire until pedal AND decel are flat
    again."""
    t = _fresh()
    _feed(t, clock, pedal=0.7, decel=5.0, ticks=60, aeb=True)
    est0 = t.max_brake_ms2

    for i in range(10):                        # release: decel decays
        clock.t += DT
        t.update_brake(0.0, 5.0 - 0.4 * (i + 1), SPEED, 0.0, BASE,
                       road_load_ms2=0.0, aeb_active=False)
    for i in range(15):                        # re-press: decel rebuilding
        clock.t += DT
        t.update_brake(0.7, min(2.0 + 0.2 * (i + 1), 5.0), SPEED, 0.0, BASE,
                       road_load_ms2=0.0, aeb_active=True)
    assert t.max_brake_ms2 == est0


def test_ripple_averages_instead_of_rectifying(clock):
    """Telemetry cadence puts zero-mean ripple on the decel signal. Window means must make the
    estimate track the true mean, not the lower envelope (the asymmetric underperform alpha would
    otherwise rectify the ripple into downward drift)."""
    t = _fresh()
    for i in range(90):
        clock.t += DT
        t.update_brake(1.0, 5.0 + (0.3 if i % 2 == 0 else -0.3), SPEED, 0.0,
                       BASE, road_load_ms2=0.0, aeb_active=True)
    expected = 5.0 / brake_curve_fraction(1.0)   # ~5.48 m/s2
    envelope = 4.7 / brake_curve_fraction(1.0)   # ~5.15 m/s2
    assert t.max_brake_ms2 == pytest.approx(expected, rel=0.04)
    assert t.max_brake_ms2 > envelope + 0.1


def test_settle_window_blocks_early_samples(clock):
    """After a pedal step, no sample may fire until the smoothed pedal has
    been flat for the full window (EMA convergence + window span, ~0.8 s)."""
    t = _fresh(scale=0.85)
    clock.t += 1.0
    t.update_brake(0.0, 0.0, SPEED, 0.0, BASE)
    _feed(t, clock, pedal=0.9, decel=8.5, ticks=18)   # 0.59 s after step
    assert t.max_brake_ms2 == pytest.approx(0.85 * BASE)
    _feed(t, clock, pedal=0.9, decel=8.5, ticks=15)   # 1.09 s after step
    assert t.max_brake_ms2 > 0.85 * BASE


def test_over_reading_the_rig_is_refused(clock):
    """The learner may only correct the model down, never up. Deliberate.

    Believing more brake than the rig has is the collision direction: the stop
    simulation collides once the estimate reaches ~1.10x truth, because AEB then
    engages at a gap sized for a stop it cannot make.

    The margin for an upward correction is already spent by the model itself.
    Probed 2026-08-12, `baseline_brake_ms2` runs 4-5% *high* on a loaded double
    (40.5 t measured 11.37 against a model 11.87), and no refit removes that:
    every power law through the six measured points over-predicts that rig by
    3.8-4.5%. A ceiling of 1.02 on top of it already collides at 80-120 km/h
    under p90 brake lag, so the ceiling is 1.00.

    The mechanism this blocks is a carry-over, the same class as the bug that
    motivated the rig-relative rewrite: `brake_scale` is global, but the model's
    error is not. Learning 1.05 on an unloaded double (where the model is ~8%
    low, so samples honestly say 1.08) and then hooking cargo applies it to a
    model already 4% high.
    """
    t = _fresh()
    _feed(t, clock, pedal=1.0, decel=10.5, ticks=600)
    assert t.max_brake_ms2 <= BASE + 1e-6
    assert t.brake_scale == pytest.approx(1.0, rel=1e-3)
    assert pc._BRAKE_SCALE_MAX == 1.0, "raising this needs the mass exponent resolved"


def test_under_delivery_is_believed_all_the_way_down(clock):
    """The floor must stay low enough to represent a genuinely weak rig.

    Wet grip, worn brakes or a fade episode all reduce real capability, and AEB
    planning against a capability the truck no longer has is what hits things.
    """
    t = _fresh()
    _feed(t, clock, pedal=1.0, decel=4.0, ticks=400, aeb=True)
    expected = 4.0 / brake_curve_fraction(1.0)   # ~4.39 m/s2
    assert t.max_brake_ms2 == pytest.approx(expected, rel=0.05)
    assert expected > BASE * pc._BRAKE_SCALE_MIN, "floor would have hidden this"


def test_hooking_a_trailer_moves_the_estimate_the_same_tick(clock):
    """The learned quantity is a correction, so a rig change lands immediately.

    An absolute m/s2 scalar cannot follow: hooking a trailer roughly doubles the
    braked axles, and at the shipped EMA rate the old estimate needed hours of
    accepted samples to catch up. Measured live it never did: 44 recorded
    engagements read 8.90 m/s2 on an 18-wheel double whose baseline is 13.89.
    """
    solo_base, trailer_base = 10.22, 13.89
    t = _fresh()
    _feed(t, clock, pedal=0.4, decel=3.0, ticks=80, baseline=solo_base)
    learned_scale = t.brake_scale
    assert t.max_brake_ms2 == pytest.approx(learned_scale * solo_base)

    t.update_brake(0.0, 0.0, SPEED, 0.0, trailer_base, road_load_ms2=0.0)
    assert t.brake_scale == pytest.approx(learned_scale), "correction must survive"
    assert t.max_brake_ms2 == pytest.approx(learned_scale * trailer_base)


def test_load_persisted_clamps_poisoned_value(monkeypatch):
    class _FakeSettings:
        pedal_capacity_brake_scale = 3.7
        pedal_capacity_max_accel_ms2 = 2.0
        pedal_capacity_accel_anchor_gain_ms2 = 0.0
        pedal_capacity_accel_ratio_step = 0.0
        mapper_brake_scale_ms2 = 6.5

    monkeypatch.setattr(pc, "Settings", _FakeSettings)
    t = pc.PedalCapacityTracker()
    t.load_persisted(BASE, 2.0)
    assert t.max_brake_ms2 <= BASE * pc._BRAKE_SCALE_MAX + 1e-6

    _FakeSettings.pedal_capacity_brake_scale = 0.0
    t2 = pc.PedalCapacityTracker()
    t2.load_persisted(BASE, 2.0)
    assert t2.max_brake_ms2 == pytest.approx(BASE), "unset means believe the model"


def test_brake_baseline_rises_with_a_trailer(monkeypatch):
    """Braking must not use the acceleration mass model, which has it backwards.

    Measured over 134 clips on flat road: 9.5 m/s2 solo at 10 t against 14.1 with a
    trailer at 17-24 t. The old baseline divided by `weight_factor` and so predicted
    8.26 solo vs 5.87 loaded, which made the partial-pedal candidate cap reject every
    truthful loaded sample and froze the estimate. See core/sending_thread/README.md.
    """
    from core.sending_thread import accel_to_pedals as ap

    solo = ap.baseline_brake_ms2(10_000.0, False)
    loaded = ap.baseline_brake_ms2(24_000.0, True)
    assert loaded > solo, f"trailer baseline {loaded:.2f} must exceed solo {solo:.2f}"

    # Mass alone, within a load class, must not move it: a trailer's own braked
    # axles are the mechanism, not the tonnage.
    assert ap.baseline_brake_ms2(17_000.0, True) == pytest.approx(
        ap.baseline_brake_ms2(24_000.0, True)
    )
    assert ap.baseline_brake_ms2(8_000.0, False) == pytest.approx(solo)


# Full-pedal stops: (wheels, mass_kg, max decel at brake=1.0). Clip-derived rows
# are converted out of raw peak decel into the same units the probe reports.
MEASURED_RIGS = (
    (6, 10_470.0, 10.14),   # bobtail, brake probe, n=3 plateaued stops
    (12, 17_000.0, 12.70),  # single trailer
    (18, 24_000.0, 13.90),  # double, empty
    (18, 54_310.0, 10.85),  # double, ~29 t cargo
    (18, 24_300.0, 15.10),  # double, empty, 2026-08-12 probe, n=2
    (18, 40_500.0, 11.37),  # double, 16 t cargo, 2026-08-12 probe, n=2
)


def test_baseline_tracks_measured_capability(monkeypatch):
    """The 1.35x partial-pedal cap must sit above real capability, not below it."""
    from core.sending_thread import accel_to_pedals as ap

    for wheels, mass, measured in MEASURED_RIGS:
        base = ap.baseline_brake_ms2(mass, wheels > 6, wheels)
        assert 0.85 <= measured / base <= 1.15, (
            f"baseline {base:.2f} vs measured {measured:.2f}"
        )
        assert measured <= pc._brake_candidate_cap_ms2(base), (
            "a truthful partial-pedal candidate would be rejected"
        )


def test_weight_adjusted_nominal_tracks_the_measured_rigs():
    """The degradation warning's reference must be the same fitted capability.

    It used to carry its own `11.5 * wheels/12 * 17000/mass` model, whose 1/mass
    term collapses on a loaded rig: against the probe it read 45% low on the
    54 t double and 30% low at 40 t, so a healthy truck looked like it was
    over-performing and real fade could never trip the ratio.
    """
    from core.sending_thread.accel_to_pedals import brake_curve_fraction as frac
    from core.sending_thread.brake_efficiency import nominal_max_brake_decel_ms2

    for wheels, mass, measured_asymptote in MEASURED_RIGS:
        # The model answers "decel at brake=1.0"; the probe measures the curve's
        # asymptote, which full pedal only reaches frac(1.0) of.
        at_full_pedal = measured_asymptote * frac(1.0)
        got = nominal_max_brake_decel_ms2(wheels, mass)
        assert 0.85 <= got / at_full_pedal <= 1.15, (
            f"{wheels}w {mass / 1000:.1f}t: nominal {got:.2f} vs measured "
            f"{at_full_pedal:.2f} at brake=1.0"
        )


def test_the_efficiency_reference_follows_the_brake_curve():
    """Expected decel per pedal is the fitted curve, never a straight line.

    A linear read makes a healthy truck look weak at part pedal: at the 0.70
    sampling threshold the curve is already at 84% of full, so `pedal * nominal`
    under-predicts by a fifth and the grip ratio comes out high.
    """
    from core.sending_thread.accel_to_pedals import brake_curve_fraction as frac

    assert frac(0.70) / frac(1.0) > 0.70 * 1.15, (
        "the curve is far from linear at the sampling threshold"
    )


def test_loading_a_rig_barely_lowers_its_braking(monkeypatch):
    """Air brakes are load-sensed, so decel must not fall like 1/mass.

    Same 18-wheel double, empty vs ~29 t of cargo: measured 13.17 -> 10.83, a
    factor of 0.82 for 2.24x the mass. A 1/mass model predicts 5.88.
    """
    from core.sending_thread import accel_to_pedals as ap

    empty = ap.baseline_brake_ms2(24_000.0, True, 18)
    loaded = ap.baseline_brake_ms2(54_310.0, True, 18)
    assert loaded < empty, "loading must not increase predicted capability"
    assert loaded / empty > 0.70, (
        f"mass term too strong: {loaded / empty:.2f}, measured ratio is 0.78"
    )


def test_cargo_needs_a_trailer_to_count(monkeypatch):
    """Dropping the trailer must drop its cargo from the mass estimate.

    The SDK keeps reporting the assigned job's cargoMass after you unhook, which
    read a bobtail as 39.8 t and corrupted every mass-scaled term.
    """
    from core.sending_thread.accel_to_pedals import compute_estimated_mass_kg

    bobtail = compute_estimated_mass_kg(10_000.0, 29_000.0, 800.0, trailer_count=0)
    hooked = compute_estimated_mass_kg(10_000.0, 29_000.0, 800.0, trailer_count=2)
    assert bobtail < 12_000.0, f"bobtail read {bobtail / 1000:.1f} t"
    assert hooked - bobtail == pytest.approx(29_000.0 + 14_000.0)


# Gas-side capacity learning: the gear-shift poisoning fix.

MASS = 20_000.0
ANCHOR_GEAR = 6


def _accel_tracker(anchor: float = 3.4) -> pc.PedalCapacityTracker:
    t = pc.PedalCapacityTracker()
    t._accel_anchor_gain_ms2 = anchor
    t._accel_ratio_step = 1.27
    return t


def _feed_accel(t, clk, gas, accel, ticks=60, dt=DT, gear=ANCHOR_GEAR, speed=SPEED):
    for _ in range(ticks):
        clk.t += dt
        t.update_accel(gas, accel, speed, 0.0, 0.0, gear, MASS, False,
                       road_load_ms2=0.0)


def test_steady_gas_cruise_still_teaches_the_anchor(clock):
    """The gates must not starve normal learning: a settled cruise still moves it."""
    t = _accel_tracker(anchor=3.4)
    _feed_accel(t, clock, gas=0.7, accel=3.0, ticks=90)
    assert t._accel_anchor_gain_ms2 > 3.4, "settled cruise taught nothing"


def test_post_shift_recovery_is_not_learned_as_weakness(clock):
    """The bug: after a shift the driveline is still restoring torque, so accel is
    depressed for reasons unrelated to gas. Learning that sag drops the anchor,
    which raises gas = combined / max_a_use, which is the ~5-10% pedal step
    users saw after every gear change.
    """
    t = _accel_tracker(anchor=3.4)
    before = t._accel_anchor_gain_ms2
    # Settle in the old gear, then shift and ramp accel back up over ~1.5 s
    # with the pedal held steady, exactly the measured recovery shape.
    _feed_accel(t, clock, gas=0.7, accel=3.0, ticks=60, gear=ANCHOR_GEAR)
    steady = t._accel_anchor_gain_ms2
    for i in range(45):
        clock.t += DT
        accel = -0.7 + 3.7 * (i + 1) / 45.0
        t.update_accel(0.7, max(0.0, accel), SPEED, 0.0, 0.0, ANCHOR_GEAR + 1,
                       MASS, False, road_load_ms2=0.0)
    assert t._accel_anchor_gain_ms2 == pytest.approx(steady), (
        "anchor moved during the post-shift recovery"
    )
    assert t._accel_anchor_gain_ms2 >= before


def test_gear_dwell_covers_the_measured_recovery(clock):
    """Recovery was measured at about 1.5 s; the dwell has to outlast it."""
    assert pc._GEAR_DWELL_S >= 1.4


def test_moving_accel_is_rejected_even_with_a_settled_pedal(clock):
    """A settled pedal is not enough on its own, which is why the old gate leaked:
    60% of recovery ticks passed it because the gas ramp was slow."""
    t = _accel_tracker(anchor=3.4)
    _feed_accel(t, clock, gas=0.7, accel=3.0, ticks=60)
    settled = t._accel_anchor_gain_ms2
    for i in range(40):
        clock.t += DT
        t.update_accel(0.7, 1.0 + 2.5 * i / 40.0, SPEED, 0.0, 0.0, ANCHOR_GEAR,
                       MASS, False, road_load_ms2=0.0)
    assert t._accel_anchor_gain_ms2 == pytest.approx(settled), (
        "learned from an accel signal that was still ramping"
    )


def test_a_gap_in_the_call_stream_restarts_the_window(clock):
    """Gas is cut to zero in neutral, so update_accel stops being called mid-shift.
    The window must restart rather than straddle the gap and look settled."""
    t = _accel_tracker(anchor=3.4)
    _feed_accel(t, clock, gas=0.7, accel=3.0, ticks=60)
    after_cruise = t._accel_anchor_gain_ms2
    clock.t += 2.0                      # neutral: no calls at all
    clock.t += DT
    t.update_accel(0.7, 0.2, SPEED, 0.0, 0.0, ANCHOR_GEAR, MASS, False,
                   road_load_ms2=0.0)
    assert t._accel_anchor_gain_ms2 == pytest.approx(after_cruise), (
        "first sample after the gap was learned immediately"
    )


def test_a_depressed_sample_would_have_dropped_the_anchor_without_the_gates(clock):
    """Pins the mechanism: the same sag, fed with the gates bypassed, drops the
    anchor hard. This is what the fix prevents."""
    t = _accel_tracker(anchor=3.4)
    # Settled but genuinely weak: passes every gate, and should be learned.
    _feed_accel(t, clock, gas=0.7, accel=0.7, ticks=90)
    assert t._accel_anchor_gain_ms2 < 3.4, "a settled weak sample must still teach"
