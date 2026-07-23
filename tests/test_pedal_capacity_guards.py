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
        pedal_capacity_max_brake_ms2 = 0.0
        pedal_capacity_max_accel_ms2 = 0.0
        pedal_capacity_accel_anchor_gain_ms2 = 0.0
        pedal_capacity_accel_ratio_step = 0.0
        mapper_brake_scale_ms2 = 6.5

        @staticmethod
        def save(values=None):
            pass

    monkeypatch.setattr(pc, "Settings", _S)


def _fresh() -> pc.PedalCapacityTracker:
    t = pc.PedalCapacityTracker()
    t._max_brake_ms2 = BASE
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


def test_legit_strong_brake_still_learns(clock):
    t = _fresh()
    _feed(t, clock, pedal=0.9, decel=8.5)
    expected = 8.5 / brake_curve_fraction(0.9)   # ~9.54 m/s2
    assert BASE < t.max_brake_ms2 <= expected + 0.1


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
    """A sustained hard AEB stop with settled decel is full-scale capacity
    data: the estimate must converge to it within the stop."""
    t = _fresh()
    _feed(t, clock, pedal=1.0, decel=8.5, ticks=60, aeb=True)
    expected = 8.5 / brake_curve_fraction(1.0)   # ~9.32 m/s2
    assert t.max_brake_ms2 == pytest.approx(expected, rel=0.03)


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
    t = _fresh()
    clock.t += 1.0
    t.update_brake(0.0, 0.0, SPEED, 0.0, BASE)
    _feed(t, clock, pedal=0.9, decel=8.5, ticks=18)   # 0.59 s after step
    assert t.max_brake_ms2 == pytest.approx(BASE)
    _feed(t, clock, pedal=0.9, decel=8.5, ticks=15)   # 1.09 s after step
    assert t.max_brake_ms2 > BASE


def test_estimate_ceiling(clock):
    t = _fresh()
    ceiling = pc._brake_estimate_ceiling_ms2(BASE)
    _feed(t, clock, pedal=1.0, decel=10.5, ticks=600)
    assert t.max_brake_ms2 <= ceiling + 1e-6
    assert t.max_brake_ms2 >= 11.0   # actually learned to the ceiling


def test_high_pedal_recovers_past_mass_ceiling(clock):
    """Loaded mass baseline used to cap/reject honest ~9.5 full-brake samples.
    Temporary nominal ceiling must let settled high pedal relearn upward."""
    t = _fresh()
    t._max_brake_ms2 = 4.353
    heavy_base = 5.0
    old_cap = heavy_base * pc._ESTIMATE_UPPER_BOUND
    _feed(t, clock, pedal=1.0, decel=9.5, ticks=120, aeb=True, baseline=heavy_base)
    expected = 9.5 / brake_curve_fraction(1.0)
    assert t.max_brake_ms2 > old_cap
    assert t.max_brake_ms2 == pytest.approx(expected, rel=0.05)


def test_load_persisted_clamps_poisoned_value(monkeypatch):
    class _FakeSettings:
        pedal_capacity_max_brake_ms2 = 17.9
        pedal_capacity_max_accel_ms2 = 2.0
        pedal_capacity_accel_anchor_gain_ms2 = 0.0
        pedal_capacity_accel_ratio_step = 0.0
        mapper_brake_scale_ms2 = 6.5

    monkeypatch.setattr(pc, "Settings", _FakeSettings)
    t = pc.PedalCapacityTracker()
    t.load_persisted(BASE, 2.0)
    assert t.max_brake_ms2 <= pc._brake_estimate_ceiling_ms2(BASE) + 1e-6

    _FakeSettings.pedal_capacity_max_brake_ms2 = 0.0
    t2 = pc.PedalCapacityTracker()
    t2.load_persisted(BASE, 2.0)
    assert t2.max_brake_ms2 == pytest.approx(BASE)
