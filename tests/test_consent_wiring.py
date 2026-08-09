"""Settings-panel wiring: the consent prompt gates the opt-in, nothing else moves."""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from core.settings import Settings
from ui.main_window.consent_overlay import CONSENT_VERSION
from ui.main_window.settings_panel import SettingsPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp):
    """A real panel with the consent prompt stubbed, so no overlay is shown."""
    settings = Settings.instance()
    settings.aeb_contribute = False
    settings.aeb_contribute_consent_version = 0
    settings.aeb_capture_screenshots = True

    host = QWidget()
    requests: list[dict] = []
    p = SettingsPanel(
        host,
        settings,
        on_save=lambda: None,
        on_reset=lambda: None,
        show_confirm=lambda *a, **k: None,
        show_consent=lambda **kw: requests.append(kw),
    )
    p._consent_requests = requests
    yield p
    host.deleteLater()


def test_ticking_asks_for_consent_and_changes_nothing_yet(panel):
    panel.chk_contribute.setChecked(True)

    assert len(panel._consent_requests) == 1
    assert Settings.instance().aeb_contribute is False
    # The box must not sit checked while the setting is still off.
    assert panel.chk_contribute.isChecked() is False


def test_accepting_turns_it_on_and_records_the_version(panel):
    panel.chk_contribute.setChecked(True)
    panel._consent_requests[0]["on_accept"](True)

    s = Settings.instance()
    assert s.aeb_contribute is True
    assert s.aeb_contribute_consent_version == CONSENT_VERSION
    assert panel.chk_contribute.isChecked() is True


def test_declining_leaves_it_off(panel):
    panel.chk_contribute.setChecked(True)
    # Decline is simply never calling on_accept.
    assert Settings.instance().aeb_contribute is False
    assert panel.chk_contribute.isChecked() is False


def test_the_screenshot_choice_from_the_prompt_is_applied(panel):
    panel.chk_contribute.setChecked(True)
    panel._consent_requests[0]["on_accept"](False)

    assert Settings.instance().aeb_capture_screenshots is False
    assert Settings.instance().aeb_contribute is True


def test_unticking_turns_it_off_without_prompting(panel):
    panel.chk_contribute.setChecked(True)
    panel._consent_requests[0]["on_accept"](True)
    panel._consent_requests.clear()

    panel.chk_contribute.setChecked(False)

    assert Settings.instance().aeb_contribute is False
    assert panel._consent_requests == []


def test_a_stale_consent_version_reopens_the_question(qapp):
    """Accepting an older document must not carry over to a changed one."""
    s = Settings.instance()
    s.aeb_contribute = True
    s.aeb_contribute_consent_version = CONSENT_VERSION - 1

    host = QWidget()
    p = SettingsPanel(
        host, s,
        on_save=lambda: None, on_reset=lambda: None,
        show_confirm=lambda *a, **k: None, show_consent=lambda **kw: None,
    )
    assert p.chk_contribute.isChecked() is False
    host.deleteLater()
