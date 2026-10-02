"""Settings-panel wiring for the ACC gap level: dropdown, bind buttons, and the
sync that keeps the dropdown honest when the wheel buttons move the level."""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from core.cruise_control_thread.acc_distance import step_gap_level
from core.settings import Settings
from ui.main_window.settings_panel import _GAP_LEVEL_LABELS, SettingsPanel


def _label(level: int) -> str:
    """Display text for a level, so the tests survive a relabelling."""
    return _GAP_LEVEL_LABELS[level - 1]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp):
    settings = Settings.instance()
    settings.acc_gap_level = 2
    settings.acc_enabled = True
    settings.acc_dist_inc_button = None
    settings.acc_dist_dec_button = None
    settings.cc_start_button = None

    host = QWidget()
    p = SettingsPanel(
        host,
        settings,
        on_save=lambda: None,
        on_reset=lambda: None,
        show_confirm=lambda *a, **k: None,
        show_consent=lambda **kw: None,
    )
    yield p
    host.deleteLater()


def _build_panel(settings, confirms: list | None = None) -> tuple[QWidget, SettingsPanel]:
    """A panel whose confirm prompt is captured rather than shown."""
    host = QWidget()
    p = SettingsPanel(
        host, settings,
        on_save=lambda: None, on_reset=lambda: None,
        show_confirm=lambda *a, **k: (confirms.append(k) if confirms is not None else None),
        show_consent=lambda **kw: None,
    )
    return host, p


def _gap_row_shown(p: SettingsPanel) -> bool:
    item = p._grid.itemAtPosition(p._acc_gap_row, 1)
    return item is not None and item.widget().isVisibleTo(p)


def test_both_gap_buttons_are_bindable(panel):
    assert "acc_dist_inc_button" in panel._bind_buttons
    assert "acc_dist_dec_button" in panel._bind_buttons


def test_dropdown_opens_on_the_persisted_level(qapp):
    s = Settings.instance()
    s.acc_gap_level = 4

    host = QWidget()
    p = SettingsPanel(
        host, s,
        on_save=lambda: None, on_reset=lambda: None,
        show_confirm=lambda *a, **k: None, show_consent=lambda **kw: None,
    )
    assert p.opt_acc_gap.currentText() == _label(4)
    host.deleteLater()


def test_picking_a_distance_persists_the_level(panel):
    panel.opt_acc_gap.setCurrentText(_label(3))

    assert Settings.instance().acc_gap_level == 3


def test_every_label_maps_back_to_its_own_level(panel):
    """The level is parsed out of the display text, so each one must round-trip."""
    for level in (1, 2, 3, 4):
        panel.opt_acc_gap.setCurrentText(_label(level))
        assert Settings.instance().acc_gap_level == level


def test_a_button_press_moves_the_dropdown(panel):
    # step_gap_level is what the CC thread calls; the panel never sees the click.
    step_gap_level(+1, wrap=False)
    panel._sync_gap_level()

    assert Settings.instance().acc_gap_level == 3
    assert panel.opt_acc_gap.currentText() == _label(3)


def test_syncing_does_not_write_back(panel):
    writes: list[int] = []
    panel.opt_acc_gap.currentTextChanged.connect(lambda v: writes.append(v))

    step_gap_level(-1, wrap=False)
    panel._sync_gap_level()

    assert Settings.instance().acc_gap_level == 1
    assert writes == []


def test_apply_settings_reloads_the_level(panel):
    s = Settings.instance()
    s.acc_gap_level = 4
    panel.apply_settings(s)

    assert panel.opt_acc_gap.currentText() == _label(4)


def test_the_distance_row_is_hidden_while_acc_is_off(qapp):
    s = Settings.instance()
    s.acc_enabled = False

    host, p = _build_panel(s)
    assert _gap_row_shown(p) is False
    host.deleteLater()


def test_the_distance_row_shows_when_acc_is_on(qapp):
    s = Settings.instance()
    s.acc_enabled = True

    host, p = _build_panel(s)
    assert _gap_row_shown(p) is True
    host.deleteLater()


def test_turning_acc_off_hides_the_distance_row(panel):
    panel.chk_acc.setChecked(False)

    assert Settings.instance().acc_enabled is False
    assert _gap_row_shown(panel) is False


def test_the_row_stays_hidden_until_the_beta_prompt_is_accepted(qapp):
    s = Settings.instance()
    s.acc_enabled = False

    confirms: list[dict] = []
    host, p = _build_panel(s, confirms)
    p.chk_acc.setChecked(True)

    # Ticking only opens the prompt; declining is never calling on_confirm.
    assert _gap_row_shown(p) is False

    confirms[0]["on_confirm"]()
    assert _gap_row_shown(p) is True
    host.deleteLater()


def test_apply_settings_restores_the_row_visibility(panel):
    s = Settings.instance()
    panel.chk_acc.setChecked(False)
    assert _gap_row_shown(panel) is False

    s.acc_enabled = True
    panel.apply_settings(s)

    assert _gap_row_shown(panel) is True


def test_a_gap_button_gives_its_input_up_to_a_cruise_button(panel):
    binding = {"source": "keyboard", "code": "f7"}
    panel._finish_capture("acc_dist_inc_button", binding)
    assert Settings.instance().acc_dist_inc_button == binding

    panel._finish_capture("cc_start_button", dict(binding))

    assert Settings.instance().acc_dist_inc_button is None
    assert panel._bind_buttons["acc_dist_inc_button"].text() == "None"
