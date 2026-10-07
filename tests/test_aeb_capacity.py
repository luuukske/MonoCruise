"""Per-truck AEB brake capacity. See core/sending_thread/README.md (AEB capacity per truck)."""
from __future__ import annotations

import pytest

import core.sending_thread.aeb_capacity as ac
import core.sending_thread.pedal_capacity as pc
from core.scs_profile.intensity import TUNE_BRAKE_INTENSITY
from core.sending_thread.accel_to_pedals import brake_curve_fraction

B = 10.34
SLIDERS = (1.0 / 3.0, 0.5, 0.75, 1.0, 1.05, 1.1, 1.39, 2.158, 3.0)
FH = ac.truck_key("ets2", "vehicle.volvo.fh_2024", 0)
ATS = ac.truck_key("ats", "vehicle.peterbilt.579", 1)


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    class _S:
        aeb_brake_scales: object = {}
        saved: list[dict] = []

        @classmethod
        def save(cls, values=None):
            cls.saved.append(dict(values or {}))

    _S.saved = []
    monkeypatch.setattr(ac, "Settings", _S)
    return _S


def _teach(store, key, candidate, tune, intensity=1.0, n=300):
    for i in range(n):
        store.observe(key, candidate, tune, intensity, now=1000.0 + i)


def test_a_default_slider_user_keeps_todays_capacity():
    """Clean 100% users already engage at ~90% of full pedal; the 1/1.1 is that room."""
    for scale in (ac._PRIOR_SCALE, 1.0):
        assert ac.aeb_capacity_ms2(B, scale, 1.0) == pytest.approx(B / TUNE_BRAKE_INTENSITY)


def test_a_high_slider_extra_is_never_credited():
    for intensity in (1.1, 1.39, 2.158, 3.0):
        for scale in (0.6, ac._PRIOR_SCALE, 1.0):
            assert ac.aeb_capacity_ms2(B, scale, intensity) == pytest.approx(scale * B)


def test_capacity_never_exceeds_the_learned_scale_or_the_slider_cut():
    for intensity in SLIDERS:
        cut = min(intensity / TUNE_BRAKE_INTENSITY, 1.0)
        for scale in (0.35, 0.5, 0.8, 0.95, 1.0):
            cap = ac.aeb_capacity_ms2(B, scale, intensity)
            assert cap <= scale * B + 1e-9
            assert cap <= cut * B + 1e-9


def test_a_low_slider_is_not_counted_twice():
    """Clip cluster at slider minimum: learned scale ~0.5 on top of the force cut read 3x low."""
    old = 0.5 * B / 3.3
    new = ac.aeb_capacity_ms2(B, 0.5, 1.0 / 3.0)
    assert new == pytest.approx(B / 3.3)
    assert new > 1.9 * old


def test_capacity_rises_with_the_slider_and_the_scale():
    for scale in (0.5, 0.95):
        caps = [ac.aeb_capacity_ms2(B, scale, i) for i in SLIDERS]
        assert caps == sorted(caps)
    for intensity in SLIDERS:
        caps = [ac.aeb_capacity_ms2(B, s, intensity) for s in (0.35, 0.6, 0.9, 1.0)]
        assert caps == sorted(caps)


def test_an_unmeasured_truck_starts_below_the_model():
    store = ac.AebCapacityStore()
    assert store.scale(FH) == ac._PRIOR_SCALE < 1.0
    assert not store.is_measured(FH)


def test_light_braking_does_not_teach():
    """Light samples read 0.47-0.69 in brake_debug.csv where firm ones read 0.96."""
    store = ac.AebCapacityStore()
    _teach(store, FH, 0.5, tune=0.25)
    assert store.scale(FH) == ac._PRIOR_SCALE
    assert not store.is_measured(FH)


def test_firm_braking_moves_the_estimate_to_the_truck():
    store = ac.AebCapacityStore()
    _teach(store, FH, 0.85, tune=0.95)
    assert store.scale(FH) == pytest.approx(0.85, abs=0.01)
    _teach(store, FH, 1.0, tune=0.95)
    assert store.scale(FH) == pytest.approx(1.0, abs=0.01)


def test_a_full_stop_on_a_low_slider_counts_as_firm():
    store = ac.AebCapacityStore()
    full_sent_in_tune_units = 0.5 / TUNE_BRAKE_INTENSITY
    _teach(store, FH, 0.7, tune=full_sent_in_tune_units, intensity=0.5)
    assert store.scale(FH) == pytest.approx(0.7, abs=0.01)


def test_a_stronger_truck_stops_at_the_ceiling():
    store = ac.AebCapacityStore()
    _teach(store, FH, 1.6, tune=1.0)
    assert store.scale(FH) == pytest.approx(ac._SCALE_MAX, abs=1e-3)
    assert store.scale(FH) <= ac._SCALE_MAX


def test_a_weak_reading_stops_at_the_floor():
    store = ac.AebCapacityStore()
    _teach(store, FH, 0.05, tune=1.0)
    assert store.scale(FH) == pytest.approx(ac._SCALE_MIN, abs=1e-3)


def test_another_truck_or_game_does_not_move_this_one():
    """2026-10-03: a 19 min ATS drive dragged the shared scale 0.940 -> 0.888 for the FH."""
    store = ac.AebCapacityStore()
    _teach(store, FH, 1.0, tune=1.0)
    _teach(store, ATS, 0.6, tune=1.0)
    assert store.scale(FH) == pytest.approx(1.0, abs=0.01)
    assert store.scale(ATS) == pytest.approx(0.6, abs=0.01)
    assert ac.truck_key("ets2", "vehicle.volvo.fh_2024", 1) != FH


def test_the_store_keeps_the_most_recent_trucks():
    store = ac.AebCapacityStore()
    for n in range(ac._MAX_TRUCKS + 6):
        store.observe(f"ets2|truck{n}|0", 0.9, 1.0, 1.0, now=1000.0 + n)
    assert len(store._scales) == ac._MAX_TRUCKS
    assert not store.is_measured("ets2|truck0|0")
    assert store.is_measured(f"ets2|truck{ac._MAX_TRUCKS + 5}|0")


def test_learned_scales_persist_with_a_cooldown(fake_settings):
    store = ac.AebCapacityStore()
    store.observe(FH, 0.8, 1.0, 1.0, now=1000.0)
    assert len(fake_settings.saved) == 1
    assert FH in fake_settings.saved[0]["aeb_brake_scales"]
    store.observe(FH, 0.5, 1.0, 1.0, now=1005.0)
    assert len(fake_settings.saved) == 1, "inside the cooldown"
    store.observe(FH, 0.5, 1.0, 1.0, now=1000.0 + ac._SAVE_COOLDOWN_S + 1.0)
    assert len(fake_settings.saved) == 2


def test_persisted_scales_are_sanitised_on_load(fake_settings):
    fake_settings.aeb_brake_scales = {
        FH: 0.9, ATS: 7.0, "ets2|weak|0": 0.01, "ets2|junk|0": "x",
        "ets2|nan|0": float("nan"), 3: 0.8,
    }
    store = ac.AebCapacityStore()
    store.load_persisted()
    assert store.scale(FH) == pytest.approx(0.9)
    assert store.scale(ATS) == ac._SCALE_MAX
    assert store.scale("ets2|weak|0") == ac._SCALE_MIN
    for key in ("ets2|junk|0", "ets2|nan|0"):
        assert not store.is_measured(key)
    fake_settings.aeb_brake_scales = "not a dict"
    store.load_persisted()
    assert store.scale(FH) == ac._PRIOR_SCALE


def test_the_tracker_hands_over_a_settled_sample_before_its_cap_gate(monkeypatch):
    """A high-slider stop reads above the mapper's 1.35x cap; AEB still needs it."""

    class _S:
        mapper_brake_scale_ms2 = 6.5

        @staticmethod
        def save(values=None):
            pass

    monkeypatch.setattr(pc, "Settings", _S)
    clock = [1000.0]
    monkeypatch.setattr(pc.time, "monotonic", lambda: clock[0])
    tracker = pc.PedalCapacityTracker()
    decel = 1.6 * B * brake_curve_fraction(1.0)
    last = None
    for _ in range(80):
        clock[0] += 0.033
        tracker.update_brake(1.0, decel, 20.0, 0.0, B, road_load_ms2=0.0)
        last = tracker.last_settled_brake_sample or last
    assert tracker.last_brake_gate == "cap"
    assert last is not None
    assert last[0] == pytest.approx(1.6, rel=0.02)
    assert last[1] == pytest.approx(1.0)

    tracker.update_brake(0.0, 0.0, 20.0, 0.0, B, road_load_ms2=0.0)
    assert tracker.last_settled_brake_sample is None


def test_a_short_firm_tap_hands_aeb_no_sample(monkeypatch):
    """The tracker's hard-hold gate covers every learner fed from its settled samples."""

    class _S:
        mapper_brake_scale_ms2 = 6.5

        @staticmethod
        def save(values=None):
            pass

    monkeypatch.setattr(pc, "Settings", _S)
    clock = [1000.0]
    monkeypatch.setattr(pc.time, "monotonic", lambda: clock[0])
    tracker = pc.PedalCapacityTracker()
    decel = 0.9 * B * brake_curve_fraction(1.0)
    handed = []
    for tick in range(80):
        clock[0] += 0.033
        tracker.update_brake(1.0, decel, 20.0, 0.0, B, road_load_ms2=0.0)
        if tracker.last_settled_brake_sample is not None:
            handed.append(tick * 0.033)
    assert handed, "a long hold must still hand samples over"
    assert min(handed) >= pc.BRAKE_HOLD_MIN_S
