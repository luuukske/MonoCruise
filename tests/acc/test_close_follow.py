"""Close-follow warning: time-gap timer, its latch, and what the ACC thread publishes."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.acc import close_follow as cf
from core.acc.blinker import EGO_FRONT_OFFSET_M
from core.acc.close_follow import CloseFollow, delay_for_gap
from core.acc.thread import ACCThread
from core.settings import Settings

V = 25.0
DT = 1.0 / 30.0


def _dist(gap_s: float, v: float = V) -> float:
    return gap_s * v + EGO_FRONT_OFFSET_M


def _hold(w: CloseFollow, t: float, seconds: float, gap_s: float | None, v: float = V, rel: float = 0.0) -> float:
    """Step at 30 Hz for ``seconds``; returns the new time."""
    for _ in range(round(seconds / DT)):
        t += DT
        w.step(t, None if gap_s is None else _dist(gap_s, v), v, rel)
    return t


def _armed() -> tuple[CloseFollow, float]:
    w = CloseFollow()
    w.step(0.0, None, V)
    return w, 0.0


@pytest.mark.parametrize("gap_s, delay", [(0.0, 1.0), (0.4, 1.0), (0.45, 3.0), (0.5, 5.0), (2.0, 5.0)])
def test_the_delay_grows_from_one_second_to_five(gap_s, delay):
    assert delay_for_gap(gap_s) == pytest.approx(delay)


@pytest.mark.parametrize("gap_s, delay", [(0.1, 1.0), (0.39, 1.0), (0.45, 3.0), (0.49, 4.6)])
def test_it_warns_once_the_gap_was_held_for_its_delay(gap_s, delay):
    w, t = _armed()
    t = _hold(w, t, delay - 0.2, gap_s)
    assert not w.warn
    _hold(w, t, 0.4, gap_s)
    assert w.warn


def test_half_a_second_and_more_never_warns():
    w, t = _armed()
    _hold(w, t, 30.0, 0.5)
    assert not w.warn


def test_the_gap_is_bumper_to_bumper_over_ego_speed():
    w, t = _armed()
    for _ in range(300):
        t += DT
        w.step(t, EGO_FRONT_OFFSET_M + 0.6 * V, V)
    assert not w.warn, "0.6 s behind the lead's rear, measured from ego's front"


@pytest.mark.parametrize("rel_kmh", [-12.0, 12.0])
def test_closing_or_opening_faster_than_ten_kmh_does_not_count(rel_kmh):
    w, t = _armed()
    _hold(w, t, 30.0, 0.1, rel=rel_kmh / 3.6)
    assert not w.warn


@pytest.mark.parametrize("rel_kmh", [-9.0, 0.0, 9.0])
def test_steady_following_counts(rel_kmh):
    w, t = _armed()
    _hold(w, t, 1.2, 0.1, rel=rel_kmh / 3.6)
    assert w.warn


def test_standing_or_reversing_never_warns():
    w, t = _armed()
    for v in (0.0, 0.5, -3.0):
        for _ in range(300):
            t += DT
            w.step(t, EGO_FRONT_OFFSET_M + 0.1, v)
        assert not w.warn


def test_it_stays_on_while_easing_off_inside_half_a_second():
    w, t = _armed()
    t = _hold(w, t, 1.2, 0.2)
    assert w.warn
    _hold(w, t, 0.5, 0.45)
    assert w.warn, "0.45 s alone would need 3 s; a warning already showing does not blink off"


def test_backing_off_clears_it_and_the_timer_starts_over():
    w, t = _armed()
    t = _hold(w, t, 1.2, 0.2)
    t = _hold(w, t, cf.RELEASE_S + 0.1, 0.8)
    assert not w.warn
    t = _hold(w, t, 0.8, 0.2)
    assert not w.warn
    _hold(w, t, 0.4, 0.2)
    assert w.warn


def test_a_short_dropout_neither_clears_it_nor_restarts_the_timer():
    w, t = _armed()
    t = _hold(w, t, 1.2, 0.2)
    t = _hold(w, t, cf.RELEASE_S - 0.2, None)
    assert w.warn
    w2, t2 = _armed()
    t2 = _hold(w2, t2, 0.6, 0.2)
    t2 = _hold(w2, t2, cf.RELEASE_S - 0.2, None)
    _hold(w2, t2, 0.6, 0.2)
    assert w2.warn, "0.6 s before and 0.6 s after the dropout add up"


def test_a_frozen_clock_holds_the_timer():
    w, t = _armed()
    t = _hold(w, t, 0.5, 0.2)
    for _ in range(300):
        w.step(t, _dist(0.2), V)
    assert not w.warn


def test_a_long_gap_between_steps_counts_as_one_short_step():
    w, t = _armed()
    w.step(t + 0.1, _dist(0.2), V)
    w.step(t + 60.0, _dist(0.2), V)
    assert not w.warn


def _lead(dist_m: float, rel: float = 0.0):
    return SimpleNamespace(
        vehicle=SimpleNamespace(id=7), score=6.0, dist_m=dist_m, rel_speed_ms=rel,
        effective_speed_ms=V + rel, effective_accel_ms2=0.0,
    )


class _Rig:
    """An ACC thread fed a canned radar frame and lead list."""

    def __init__(self, monkeypatch):
        self.t = 0.0
        self.leads: list = []
        self.paused = False
        self.thread = ACCThread()
        self.thread.running = True
        monkeypatch.setattr(self.thread, "_read_radar", self._radar)
        monkeypatch.setattr(self.thread, "_read_blinkers", lambda: (False, False))
        monkeypatch.setattr(self.thread, "_read_map", lambda: None)
        monkeypatch.setattr(self.thread._tracker, "update", lambda **_: list(self.leads))

    def _radar(self):
        return ([], [], 0.0, 0.0, 0.0, 0.0, V, 0.0, 0.0, self.paused, self.t, frozenset(), None)

    def run(self, seconds: float) -> bool:
        for _ in range(round(seconds / DT)):
            if not self.paused:
                self.t += DT
            self.thread.loop()
        return self.thread.data.close_follow_warn


@pytest.fixture
def rig(monkeypatch):
    s = Settings.instance()
    monkeypatch.setattr(s, "acc_enabled", True)
    monkeypatch.setattr(s, "cc_mode", "Cruise control")
    return _Rig(monkeypatch)


def test_the_thread_publishes_it_for_the_primary_lead(rig):
    rig.leads = [_lead(_dist(0.2))]
    assert not rig.run(0.8)
    assert rig.run(0.4)


def test_the_thread_needs_the_acc_option_and_cruise_control_mode(rig, monkeypatch):
    rig.leads = [_lead(_dist(0.2))]
    assert rig.run(1.3)
    monkeypatch.setattr(Settings.instance(), "cc_mode", "Speed limiter")
    assert not rig.run(DT)
    monkeypatch.setattr(Settings.instance(), "cc_mode", "Cruise control")
    assert not rig.run(0.8), "back on, the timer starts over"
    monkeypatch.setattr(Settings.instance(), "acc_enabled", False)
    assert not rig.run(3.0)


def test_the_thread_holds_it_through_a_pause(rig):
    rig.leads = [_lead(_dist(0.2))]
    rig.run(0.6)
    rig.paused = True
    assert not rig.run(5.0), "a paused game adds no following time"
    rig.paused = False
    assert not rig.run(0.2)
    assert rig.run(0.4)


def test_the_thread_drops_it_without_a_lead(rig):
    rig.leads = [_lead(_dist(0.2))]
    assert rig.run(1.3)
    rig.leads = []
    assert not rig.run(DT), "no vehicle to draw it on"
