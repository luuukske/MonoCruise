"""Reset to defaults: it has to actually reset, and leave usage history alone.

Settings is a singleton, so the obvious ``fresh = Settings()`` hands back the
live instance and the reset writes every value onto itself. These pin the fix.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication

from core.settings import Settings
from core.usage_hours import RESET_EXEMPT_FIELDS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    from ui.main_window.window import MonoCruiseWindow

    settings = Settings.instance()
    w = MonoCruiseWindow(settings, version="v0.0.0-test")
    w._poll_timer.stop()
    yield w
    if w._cc_panel is not None:
        w._cc_panel.stop()
        w._cc_panel = None
    w.deleteLater()


def _default(name: str):
    field = Settings.instance().__dataclass_fields__[name]
    return Settings._dataclass_field_default(field)


def test_the_singleton_still_returns_the_same_object():
    """The reason the old reset silently did nothing. If this ever changes,
    the comment in _reset_settings is stale."""
    assert Settings() is Settings()
    assert Settings() is Settings.instance()


def test_a_changed_setting_goes_back_to_its_default(window):
    s = window._settings
    s.opd_mode_variable = 7
    s.polling_rate = 33
    s.cc_mode = "Speed limiter"

    window._reset_settings()

    assert s.opd_mode_variable == _default("opd_mode_variable")
    assert s.polling_rate == _default("polling_rate")
    assert s.cc_mode == _default("cc_mode")


def test_reset_clears_a_bound_pedal(window):
    """Bindings are the point of a reset, so they must not survive it."""
    s = window._settings
    s.gasaxis = 4
    s.brakeaxis = 5
    s.gas_inverted = True

    window._reset_settings()

    assert s.gasaxis is None
    assert s.brakeaxis is None
    assert s.gas_inverted is False


def test_usage_history_survives_a_reset(window):
    """The whole point of RESET_EXEMPT_FIELDS: no re-nagging a 600 hour user."""
    s = window._settings
    s.usage_minutes = 612 * 60
    s.support_prompts_dismissed = 2

    window._reset_settings()

    assert s.usage_minutes == 612 * 60
    assert s.support_prompts_dismissed == 2


def test_reset_touches_every_public_field_it_is_not_exempting(window):
    """Guards against a reset that quietly skips a whole block of settings."""
    s = window._settings
    window._reset_settings()

    for name, field in s.__dataclass_fields__.items():
        if name.startswith("_") or name in RESET_EXEMPT_FIELDS:
            continue
        assert getattr(s, name) == Settings._dataclass_field_default(field), name


def test_the_panel_survives_being_repainted_from_defaults(window):
    """Reset repaints the panel with device/axis None; it must not raise."""
    s = window._settings
    s.device = None
    s.gasaxis = None
    window._reset_settings()
    assert window._settings_panel is not None


def test_reset_is_idempotent(window):
    window._reset_settings()
    first = {k: getattr(window._settings, k)
             for k in window._settings.__dataclass_fields__ if not k.startswith("_")}
    window._reset_settings()
    second = {k: getattr(window._settings, k)
              for k in window._settings.__dataclass_fields__ if not k.startswith("_")}
    assert first == second
