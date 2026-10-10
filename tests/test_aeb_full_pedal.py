"""AEB full-pedal credit and its sent axis. See core/sending_thread/README.md (AEB capacity per truck)."""
from __future__ import annotations

import pytest

import core.sending_thread.aeb_capacity as ac
from core.scs_profile.intensity import TUNE_BRAKE_INTENSITY, apply_brake_intensity
from core.sending_thread.accel_to_pedals import brake_curve_fraction, brake_curve_pedal

B = 10.25
F1 = brake_curve_fraction(1.0)
FH = ac.full_pedal_key(ac.truck_key("ets2", "vehicle.volvo.fh_2024", 0), 2.158)
PEDALS = [i / 40.0 for i in range(40)]


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    class _S:
        aeb_full_pedal: object = {}
        saved: list[dict] = []

        @classmethod
        def save(cls, values=None):
            cls.saved.append(dict(values or {}))

    _S.saved = []
    monkeypatch.setattr(ac, "Settings", _S)
    return _S


def test_the_curve_inverse_round_trips():
    for p in PEDALS + [1.0]:
        assert brake_curve_pedal(brake_curve_fraction(p)) == pytest.approx(p, abs=1e-9)
    assert brake_curve_pedal(0.0) == 0.0
    assert brake_curve_pedal(2.0) == 1.0


def test_a_measured_full_pedal_keeps_the_usual_room():
    """Bobtail FH at 135%: full pedal stops at 1.27x the model, AEB planned with 0.95x."""
    today = ac.aeb_capacity_ms2(B, 0.947, 2.158)
    credited = ac.aeb_capacity_ms2(B, 0.947, 2.158, full_ratio=1.27)
    assert today == pytest.approx(0.947 * B)
    assert credited == pytest.approx(1.27 * B / 1.1)
    assert credited * 0.90 / (1.27 * B) == pytest.approx(0.90 / 1.1)


def test_a_measurement_never_lowers_capacity():
    for intensity in (1.0 / 3.0, 1.0, 1.39, 3.0):
        for scale in (0.5, 0.95, 1.0):
            today = ac.aeb_capacity_ms2(B, scale, intensity)
            for ratio in (0.3, 0.8, 1.0, 1.4):
                assert ac.aeb_capacity_ms2(B, scale, intensity, ratio) >= today - 1e-12


def test_the_axis_is_the_identity_at_or_below_the_tune_slider():
    for intensity in (1.0 / 3.0, 0.5, 1.0, TUNE_BRAKE_INTENSITY):
        for full in (None, 1.2):
            axis = ac.AebPedalAxis(intensity, 0.95, 0.9, full)
            for p in PEDALS:
                assert axis.sent(p) == pytest.approx(p)
                assert axis.applied(p) == pytest.approx(p)
            assert axis.sent(1.0) == 1.0


def test_unmeasured_a_high_slider_tracks_through_the_remap():
    """2026-10-04 at 300%: the raw pedal braked 2.7x what AEB planned, then sat at 0.06."""
    for intensity in (1.39, 2.158, 3.0):
        axis = ac.AebPedalAxis(intensity, 0.95, 0.95)
        for p in PEDALS:
            sent = axis.sent(p)
            assert sent == pytest.approx(apply_brake_intensity(p, intensity))
            assert axis.applied(sent) == pytest.approx(p, abs=1e-6)
        assert axis.sent(1.0) == 1.0
        assert axis.applied(1.0) == pytest.approx(1.0)


def _physical(sent, intensity, scale, full):
    """The piecewise plant the axis assumes, in units of the baseline."""
    end = TUNE_BRAKE_INTENSITY / intensity
    if sent <= end:
        return scale * brake_curve_fraction(sent / end)
    return scale * F1 + (full - scale) * F1 * (sent - end) / (1.0 - end)


def test_measured_a_high_slider_spreads_the_extra_over_the_travel():
    scale, full = 0.947, 1.27
    for intensity in (1.39, 2.158, 3.0):
        cap = full / 1.1
        axis = ac.AebPedalAxis(intensity, scale, cap, full)
        sents = [axis.sent(p) for p in PEDALS]
        assert sents == sorted(sents), "monotone"
        for p, sent in zip(PEDALS, sents):
            assert _physical(sent, intensity, scale, full) == pytest.approx(
                cap * brake_curve_fraction(p), rel=1e-6
            )
            assert axis.applied(sent) == pytest.approx(p, abs=1e-6)
        assert sents[-1] < 1.0 and axis.sent(1.0) == 1.0
        assert axis.applied(1.0) == 1.0


def test_one_stop_is_not_enough():
    store = ac.FullPedalStore()
    store.observe_stop(FH, 10300.0, 1.3)
    assert store.ratio(FH, 10300.0) is None
    store.observe_stop(FH, 10300.0, 1.2)
    assert store.ratio(FH, 10300.0) == pytest.approx(1.2), "two stops: the lower"


def test_credit_is_the_weakest_recent_stop():
    """FH bobtail at 135%: 1.26-1.38 in the morning, 1.07-1.16 that evening (2026-10-04)."""
    store = ac.FullPedalStore()
    for r in (1.05, 1.36, 1.26, 1.30, 1.29, 1.12):
        store.observe_stop(FH, 10300.0, r)
    assert store.ratio(FH, 10300.0) == pytest.approx(1.12), "only the last five count"


def test_a_different_load_or_slider_is_not_credited():
    """Same trailer, 25.7 t read 1.19 and 39.6 t read 1.05 at 1.39 (2026-10-04 probe)."""
    store = ac.FullPedalStore()
    for _ in range(3):
        store.observe_stop(FH, 25700.0, 1.19)
    assert store.ratio(FH, 27000.0) == pytest.approx(1.19)
    assert store.ratio(FH, 39600.0) is None
    assert store.ratio(FH, 17500.0) is None
    other = ac.full_pedal_key(ac.truck_key("ets2", "vehicle.volvo.fh_2024", 0), 3.0)
    assert store.ratio(other, 25700.0) is None


def test_bad_readings_are_dropped():
    store = ac.FullPedalStore()
    for r in (float("nan"), 0.1, 9.0):
        store.observe_stop(FH, 10300.0, r)
    store.observe_stop(FH, 0.0, 1.2)
    assert store.stops(FH, 10300.0) == 0


def test_the_store_keeps_four_loads_and_recent_trucks():
    store = ac.FullPedalStore()
    for m in (10000.0, 15000.0, 20000.0, 26000.0, 33000.0):
        store.observe_stop(FH, m, 1.1)
    assert store.stops(FH, 10000.0) == 0
    assert store.stops(FH, 33000.0) == 1
    for n in range(ac._MAX_TRUCKS + 3):
        store.observe_stop(f"ets2|t{n}|0|2.16", 10000.0, 1.1)
    assert store.stops(FH, 33000.0) == 0


def test_stops_persist_and_load_sanitised(fake_settings):
    store = ac.FullPedalStore()
    store.observe_stop(FH, 10300.0, 1.2)
    store.observe_stop(FH, 10300.0, 1.3)
    saved = fake_settings.saved[-1]["aeb_full_pedal"]
    assert saved[FH] == [[10300.0, [1.2, 1.3]]]
    fake_settings.aeb_full_pedal = {
        FH: [[10300.0, [1.2, "x", float("nan"), 9.0, 1.3]], [0.0, [1.1]], "junk"],
        "bad": "not a list", 7: [[1.0, [1.0]]],
    }
    fresh = ac.FullPedalStore()
    fresh.load_persisted()
    assert fresh.ratio(FH, 10300.0) == pytest.approx(1.2)
    assert fresh.stops(FH, 10300.0) == 2
    fake_settings.aeb_full_pedal = "not a dict"
    fresh.load_persisted()
    assert fresh.stops(FH, 10300.0) == 0


def _stop(run, decel, v0=25.0, dt=0.016, road=0.6, t0=100.0, staircase=True):
    """Drive one full-pedal stop through `run`; the game's 20 Hz physics steps the speed."""
    t, v, done = t0, v0, None
    while v > 2.0:
        run.tick(True, FH, 10300.0, B, v, road, t)
        t += dt
        phys = round(t * 20.0) / 20.0 if staircase else t
        v = max(v0 - (decel + road) * max(phys - t0 - 0.15, 0.0), 0.0)
    done = run.tick(False, FH, 10300.0, B, v, road, t)
    return done


def test_a_slam_is_measured_off_its_speed_trace():
    """The differentiated decel failed every settle gate on 2026-10-04's slams."""
    run = ac.FullPedalRun()
    done = _stop(run, 1.27 * B * F1)
    assert done[0] == FH and done[1] == 10300.0
    assert done[2] == pytest.approx(1.27, rel=0.03)
    assert run.tick(False, FH, 10300.0, B, 0.0, 0.0, 200.0) is None


def _tap(run, held_s, decel=1.2 * B * F1, v0=25.0, dt=0.016, road=0.6, t0=400.0):
    t = t0
    while t - t0 < held_s:
        run.tick(True, FH, 10300.0, B, max(v0 - (decel + road) * (t - t0), 0.0), road, t)
        t += dt
    return run.tick(False, FH, 10300.0, B, v0 - (decel + road) * held_s, road, t)


def test_a_short_hard_tap_is_not_a_stop():
    """2026-10-06: 1.1 s taps from 72-85 km/h read 0.70 and 0.86 against 0.97-1.30 for
    stops and set the bobtail's weakest-of-five credit. Taps under 1.3 s scattered
    0.52-1.81x the model; a full pedal now has to be held past tap length."""
    assert _tap(ac.FullPedalRun(), 1.15) is None
    held = _tap(ac.FullPedalRun(), 1.45)
    assert held is not None and held[2] == pytest.approx(1.2, rel=0.03)


def test_a_short_or_slow_run_is_ignored():
    run = ac.FullPedalRun()
    assert _stop(run, 1.2 * B * F1, v0=8.0) is None, "under a second above 18 km/h"
    for i in range(10):
        run.tick(True, FH, 10300.0, B, 3.0, 0.6, 300.0 + 0.05 * i)
    assert run.tick(False, FH, 10300.0, B, 3.0, 0.6, 300.6) is None
