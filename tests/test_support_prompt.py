"""Support prompt: the usage clock, the escalating schedule, and the card itself."""
from __future__ import annotations

import threading

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from core.usage_hours import (
    MAX_TICK_S,
    PROMPT_BASE_HOURS,
    SECONDS_PER_HOUR,
    UsageTracker,
    prompt_is_due,
    prompt_threshold_hours,
    record_prompt_dismissed,
)
from ui.main_window.constants import PATREON_URL, PROJECT_URL, WINDOW_HEIGHT, WINDOW_WIDTH
from ui.main_window.support_overlay import (
    SHARE_LINK,
    SHARE_TARGETS,
    SupportOverlay,
)
from ui.main_window.window import _SUPPORT_PROMPT_DELAY_S


class _FakeSettings:
    """Enough of Settings for the tracker: the two fields, the lock, and save."""

    def __init__(self) -> None:
        self.usage_seconds = 0.0
        self.support_prompts_dismissed = 0
        self._state_lock = threading.RLock()
        self.saves = 0

    def save(self) -> None:
        self.saves += 1


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def host(qapp):
    w = QWidget()
    w.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    w.show()
    qapp.processEvents()
    yield w
    w.hide()
    w.deleteLater()


# Schedule


def test_the_first_prompt_is_due_at_one_hundred_hours():
    assert prompt_threshold_hours(0) == PROMPT_BASE_HOURS


@pytest.mark.parametrize(
    "dismissed, hours",
    [(0, 100.0), (1, 565.685), (2, 1558.846), (3, 3200.0)],
)
def test_the_schedule_follows_n_to_the_two_point_five(dismissed, hours):
    """h(N) = N ** 2.5 * 100, so 100 -> 566 -> 1559 -> 3200 hours."""
    assert prompt_threshold_hours(dismissed) == pytest.approx(hours, rel=1e-4)


def test_each_dismissal_pushes_the_next_prompt_further_out():
    thresholds = [prompt_threshold_hours(n) for n in range(6)]
    assert thresholds == sorted(thresholds)
    assert len(set(thresholds)) == len(thresholds)


def test_a_corrupt_dismissal_count_falls_back_to_the_first_threshold():
    assert prompt_threshold_hours("nonsense") == PROMPT_BASE_HOURS
    assert prompt_threshold_hours(-5) == PROMPT_BASE_HOURS


def test_due_only_once_the_threshold_is_reached():
    just_under = (100.0 * SECONDS_PER_HOUR) - 1.0
    assert not prompt_is_due(just_under, 0)
    assert prompt_is_due(100.0 * SECONDS_PER_HOUR, 0)


def test_hitting_the_first_threshold_is_not_enough_for_the_second():
    """Otherwise a dismissal at 100 h would re-prompt on the very next tick."""
    assert prompt_is_due(100.0 * SECONDS_PER_HOUR, 0)
    assert not prompt_is_due(100.0 * SECONDS_PER_HOUR, 1)


def test_unusable_usage_values_are_never_due():
    assert not prompt_is_due(None, 0)
    assert not prompt_is_due("later", 0)


# Usage clock


def test_time_accrues_only_while_the_game_is_connected():
    s = _FakeSettings()
    tracker = UsageTracker(s)

    tracker.tick(True, 0.0)
    tracker.tick(True, 1.0)
    assert s.usage_seconds == pytest.approx(1.0)

    tracker.tick(False, 2.0)
    tracker.tick(False, 3.0)
    assert s.usage_seconds == pytest.approx(1.0)


def test_reconnecting_does_not_backfill_the_disconnected_gap():
    s = _FakeSettings()
    tracker = UsageTracker(s)
    tracker.tick(True, 0.0)
    tracker.tick(False, 1.0)
    tracker.tick(True, 900.0)
    tracker.tick(True, 901.0)
    assert s.usage_seconds == pytest.approx(1.0)


def test_a_sleep_or_clock_step_is_dropped_rather_than_counted():
    s = _FakeSettings()
    tracker = UsageTracker(s)
    tracker.tick(True, 0.0)
    tracker.tick(True, MAX_TICK_S + 60.0)
    assert s.usage_seconds == 0.0


def test_a_backwards_clock_does_not_subtract_hours():
    s = _FakeSettings()
    tracker = UsageTracker(s)
    s.usage_seconds = 500.0
    tracker.tick(True, 10.0)
    tracker.tick(True, 5.0)
    assert s.usage_seconds == pytest.approx(500.0)


def test_the_counter_is_not_flushed_to_disk_on_every_tick():
    """A 100 ms poll writing config.json each time would hammer the disk."""
    s = _FakeSettings()
    tracker = UsageTracker(s)
    now = 0.0
    for _ in range(200):
        tracker.tick(True, now)
        now += 0.1
    assert s.saves == 0
    assert s.usage_seconds == pytest.approx(19.9, abs=0.01)


def test_the_counter_is_eventually_flushed():
    s = _FakeSettings()
    tracker = UsageTracker(s)
    now = 0.0
    for _ in range(200):
        tracker.tick(True, now)
        now += 2.0
    assert s.saves >= 1


def test_usage_hours_reports_the_stored_seconds():
    s = _FakeSettings()
    s.usage_seconds = 3.0 * SECONDS_PER_HOUR
    assert UsageTracker(s).usage_hours == pytest.approx(3.0)


def test_a_failing_save_does_not_lose_the_in_memory_count():
    class _Boom(_FakeSettings):
        def save(self):
            raise OSError("disk gone")

    s = _Boom()
    tracker = UsageTracker(s)
    now = 0.0
    for _ in range(400):
        tracker.tick(True, now)
        now += 2.0
    assert s.usage_seconds == pytest.approx(798.0, abs=0.01)


def test_dismissal_counts_up_and_persists():
    s = _FakeSettings()
    assert record_prompt_dismissed(s) == 1
    assert record_prompt_dismissed(s) == 2
    assert s.support_prompts_dismissed == 2
    assert s.saves == 2


def test_a_corrupt_dismissal_count_recovers_on_the_next_dismissal():
    s = _FakeSettings()
    s.support_prompts_dismissed = None
    assert record_prompt_dismissed(s) == 1


# The card


def test_the_card_carries_every_share_target(host, qapp):
    ov = SupportOverlay(host)
    qapp.processEvents()
    labels = [b.text().strip() for b in ov._share_buttons]
    assert labels == [t.label for t in SHARE_TARGETS]


def test_every_share_button_carries_its_brand_mark(host, qapp):
    """A text-only button in a row of logos reads as a broken asset path."""
    ov = SupportOverlay(host)
    qapp.processEvents()
    for btn, target in zip(ov._share_buttons, SHARE_TARGETS):
        assert not btn.icon().isNull(), target.icon


def test_the_footer_buttons_carry_their_brand_marks(host, qapp):
    ov = SupportOverlay(host)
    qapp.processEvents()
    assert not ov._star_btn.icon().isNull()
    assert not ov._patreon_btn.icon().isNull()


def test_every_brand_mark_ships_in_the_assets_directory():
    """monocruise.spec bundles this directory, so a stray path breaks the build."""
    import os

    from ui.main_window.support_overlay import _ASSET_DIR

    for name in [t.icon for t in SHARE_TARGETS] + ["github.png", "patreon.png"]:
        assert os.path.isfile(os.path.join(_ASSET_DIR, name)), name


def test_the_clipboard_carries_the_bare_link_and_nothing_else():
    """Driver-requested: people word their own post, so no canned blurb."""
    assert SHARE_LINK == PROJECT_URL
    assert SHARE_LINK.startswith("https://")
    assert " " not in SHARE_LINK.strip()
    assert SHARE_LINK == SHARE_LINK.strip()


def test_every_share_url_is_https():
    for target in SHARE_TARGETS:
        assert target.url.startswith("https://"), target.label


def test_patreon_points_at_a_creator_page_not_the_patreon_homepage():
    """A support button landing on patreon.com reads as broken, not as an ask."""
    tail = PATREON_URL.rstrip("/").rsplit("patreon.com", 1)[-1]
    assert tail not in ("", "/"), PATREON_URL


def test_dismissing_reports_exactly_once(host, qapp):
    calls = []
    ov = SupportOverlay(host, on_dismiss=lambda: calls.append(1))
    qapp.processEvents()
    ov._dismiss()
    ov._dismiss()
    assert calls == [1]


def test_a_failing_dismiss_callback_still_closes_the_card(host, qapp):
    def _boom():
        raise RuntimeError("no settings")

    ov = SupportOverlay(host, on_dismiss=_boom)
    qapp.processEvents()
    ov._dismiss()
    assert not ov.isVisible()


def test_sharing_copies_the_link_and_updates_the_hint(host, qapp, monkeypatch):
    opened = []
    monkeypatch.setattr("ui.main_window.support_overlay._open", opened.append)
    ov = SupportOverlay(host)
    qapp.processEvents()
    before = ov._hint.text()

    ov._share(SHARE_TARGETS[0])

    assert opened == [SHARE_TARGETS[0].url]
    assert ov._hint.text() != before
    assert QApplication.clipboard().text() == SHARE_LINK


def test_sharing_does_not_dismiss_the_card(host, qapp, monkeypatch):
    """The user has to close it themselves, so one share is still one dismissal."""
    monkeypatch.setattr("ui.main_window.support_overlay._open", lambda _u: None)
    calls = []
    ov = SupportOverlay(host, on_dismiss=lambda: calls.append(1))
    qapp.processEvents()
    ov._share(SHARE_TARGETS[0])
    assert calls == []
    assert ov.isVisible()


@pytest.mark.parametrize("width", [WINDOW_WIDTH, 800, 1000])
def test_the_share_row_fits_without_clipping_a_label(qapp, width):
    """Five buttons on one row is the tightest thing in the card."""
    host = QWidget()
    host.resize(width, WINDOW_HEIGHT)
    host.show()
    qapp.processEvents()
    try:
        ov = SupportOverlay(host)
        qapp.processEvents()
        for btn in ov._share_buttons:
            assert btn.width() >= btn.sizeHint().width(), btn.text()
    finally:
        host.hide()
        host.deleteLater()


@pytest.mark.parametrize("height", [WINDOW_HEIGHT, 420, 360])
def test_the_card_stays_inside_the_window(qapp, height):
    host = QWidget()
    host.resize(WINDOW_WIDTH, height)
    host.show()
    qapp.processEvents()
    try:
        ov = SupportOverlay(host)
        qapp.processEvents()
        assert ov._card.geometry().bottom() <= ov.height()
        assert ov._dismiss_btn.geometry().bottom() <= ov._card.height()
    finally:
        host.hide()
        host.deleteLater()


# Window gating


@pytest.fixture()
def window(qapp):
    """A real main window with its poll timer stopped, so tests drive the sync."""
    from core.settings import Settings
    from ui.main_window.window import MonoCruiseWindow

    settings = Settings.instance()
    settings.usage_seconds = 0.0
    settings.support_prompts_dismissed = 0

    w = MonoCruiseWindow(settings, version="v0.0.0-test")
    w._poll_timer.stop()
    yield w
    if w._cc_panel is not None:
        w._cc_panel.stop()
        w._cc_panel = None
    w.deleteLater()


def _arm(window, hours: float) -> None:
    """Put the window in the state the prompt needs, minus the visibility gate."""
    window._settings.usage_seconds = hours * SECONDS_PER_HOUR
    window._support_visible_since = -_SUPPORT_PROMPT_DELAY_S * 2


def test_no_prompt_below_the_first_threshold(window, qapp):
    window._open_on_taskbar = True
    _arm(window, 99.0)
    window._sync_support_prompt()
    assert window._support_overlay is None


def test_prompt_appears_at_the_threshold_on_a_visible_window(window, qapp):
    window._open_on_taskbar = True
    _arm(window, 100.0)
    window._sync_support_prompt()
    assert window._support_overlay is not None


def test_no_prompt_while_the_window_is_minimised(window, qapp):
    """Auto-launched behind the game is exactly when this must stay quiet."""
    window._open_on_taskbar = False
    _arm(window, 500.0)
    window._sync_support_prompt()
    assert window._support_overlay is None


def test_the_visible_delay_has_to_elapse_first(window, qapp):
    window._open_on_taskbar = True
    window._settings.usage_seconds = 100.0 * SECONDS_PER_HOUR
    window._support_visible_since = None
    window._sync_support_prompt()
    assert window._support_overlay is None


def test_minimising_restarts_the_visible_delay(window, qapp):
    window._open_on_taskbar = True
    _arm(window, 100.0)
    window._open_on_taskbar = False
    window._sync_support_prompt()
    assert window._support_visible_since is None


def test_dismissing_moves_the_next_prompt_to_the_second_threshold(window, qapp):
    window._open_on_taskbar = True
    _arm(window, 100.0)
    window._sync_support_prompt()
    window._support_overlay._dismiss()

    assert window._settings.support_prompts_dismissed == 1
    assert not prompt_is_due(150.0 * SECONDS_PER_HOUR, 1)
    assert prompt_is_due(600.0 * SECONDS_PER_HOUR, 1)


def test_only_one_prompt_per_run(window, qapp):
    window._open_on_taskbar = True
    _arm(window, 100.0)
    window._sync_support_prompt()
    window._support_overlay._dismiss()
    _arm(window, 100.0)
    window._sync_support_prompt()
    assert window._support_overlay is None


def test_the_usage_clock_runs_while_minimised(window, qapp):
    """Hours accrue behind the game; only the prompt itself waits for a window."""
    window._open_on_taskbar = False
    window._usage.tick(True, 0.0)
    window._usage.tick(True, 2.0)
    assert window._settings.usage_seconds == pytest.approx(2.0)


def test_a_missing_telemetry_thread_does_not_break_the_poll(window, qapp):
    """No telemetry thread in this process, so the sync must read it defensively."""
    from core.thread_management.registry import registry

    with pytest.raises(KeyError):
        registry.get_thread("telemetry_thread")

    window._open_on_taskbar = True
    _arm(window, 100.0)
    window._sync_support_prompt()
    assert window._support_overlay is not None
