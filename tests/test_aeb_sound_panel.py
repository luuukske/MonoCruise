"""Settings-panel wiring for the AEB warning style and volume."""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from core.aeb import warning_sounds
from core.settings import Settings
from ui.main_window import settings_panel as panel_mod
from ui.main_window.settings_panel import SettingsPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def previews(monkeypatch):
    """Records every warning the panel starts; only the test button may start one."""
    played: list[str] = []
    monkeypatch.setattr(panel_mod.WarningTest, "trigger", lambda self: played.append("test"))
    return played


def _panel(aeb_on: bool) -> tuple[QWidget, SettingsPanel]:
    settings = Settings.instance()
    settings.AEB_enabled = aeb_on
    settings.aeb_sound = warning_sounds.DEFAULT_SOUND
    settings.aeb_sound_volume = warning_sounds.DEFAULT_VOLUME_PCT
    host = QWidget()
    p = SettingsPanel(host, settings, on_save=lambda: None, on_reset=lambda: None,
                      show_confirm=lambda *a, **k: None, show_consent=lambda **kw: None)
    return host, p


def _shown(p: SettingsPanel, row: int) -> bool:
    item = p._grid.itemAtPosition(row, 1)
    return item is not None and item.widget().isVisibleTo(p)


def test_sound_rows_follow_the_aeb_toggle(qapp):
    host, p = _panel(aeb_on=False)
    assert not any(_shown(p, r) for r in p._aeb_sound_rows)
    host.deleteLater()
    host, p = _panel(aeb_on=True)
    assert all(_shown(p, r) for r in p._aeb_sound_rows)
    p.chk_aeb.setChecked(False)
    assert not any(_shown(p, r) for r in p._aeb_sound_rows)
    host.deleteLater()


def test_picking_a_style_saves_it_without_playing_it(qapp, previews):
    host, p = _panel(aeb_on=True)
    p.opt_aeb_sound.setCurrentText(warning_sounds.TESLA.label)
    assert Settings.instance().aeb_sound == warning_sounds.TESLA.label
    assert previews == []
    p._btn_aeb_test.click()
    assert previews == ["test"]
    host.deleteLater()


def test_a_typed_volume_is_clamped_and_saved(qapp, previews):
    host, p = _panel(aeb_on=True)
    p.ent_aeb_volume.setText("0")
    p.ent_aeb_volume.returnPressed.emit()
    assert Settings.instance().aeb_sound_volume == warning_sounds.MIN_VOLUME_PCT
    assert p.ent_aeb_volume.text() == str(warning_sounds.MIN_VOLUME_PCT)
    assert p.ent_aeb_volume._mc_unit_label.text() == "%"
    assert previews == []
    host.deleteLater()


def test_the_test_button_plays_a_volume_still_being_typed(qapp, previews):
    host, p = _panel(aeb_on=True)
    heard: list[int] = []
    p._aeb_test.trigger = lambda: heard.append(Settings.instance().aeb_sound_volume)
    p.ent_aeb_volume.setText("35")
    p._btn_aeb_test.click()
    assert heard == [35]
    host.deleteLater()
