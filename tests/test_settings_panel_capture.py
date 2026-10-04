"""Settings-panel capture flows: pedal setup, button assignment and unassign.

The buttons carry their own state (label, border, armed colour), so each flow has
to start, finish and cancel cleanly with nothing on the parent window listening.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from core.input_bindings import binding_display_name
from core.settings import Settings
from ui.main_window.settings_panel import SettingsPanel
from ui.main_window.widgets import BindButton

_KEY = "cc_start_button"
_TOUCHED = (_KEY, "device", "gasaxis", "brakeaxis")
_BRAKE_TEXT = "Tap the brake pedal  (click to cancel)"
_GAS_TEXT = "Tap the gas pedal  (click to cancel)"


class _FakePedalThread:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            _lock=threading.Lock(),
            pedal_config_active=False,
            pedal_config_stage="brake",
            pedal_config_result=None,
        )
        self.cancelled = 0

    def is_alive(self) -> bool:
        return True

    def start_pedal_config(self) -> None:
        self.data.pedal_config_active = True
        self.data.pedal_config_stage = "brake"

    def cancel_pedal_config(self) -> None:
        self.cancelled += 1
        self.data.pedal_config_active = False

    def consume_pedal_config(self):
        result = self.data.pedal_config_result
        self.data.pedal_config_result = None
        self.data.pedal_config_active = False
        return result

    def start_capture(self) -> None:
        pass

    def cancel_capture(self) -> None:
        pass


class _FakeKeyboardThread:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            _lock=threading.Lock(), capture_active=False, capture_event=None
        )

    def start_capture(self) -> None:
        self.data.capture_active = True

    def cancel_capture(self) -> None:
        self.data.capture_active = False

    def consume_capture(self):
        event, self.data.capture_event = self.data.capture_event, None
        self.data.capture_active = False
        return event


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def threads(monkeypatch):
    """The sibling threads the panel can see; empty means none are running."""
    found: dict[str, object] = {}
    monkeypatch.setattr(
        SettingsPanel, "_get_thread", staticmethod(lambda name: found.get(name))
    )
    return found


@pytest.fixture()
def panel(qapp, threads):
    s = Settings.instance()
    saved = {name: getattr(s, name) for name in _TOUCHED}
    for name in _TOUCHED:
        setattr(s, name, None)

    host = QWidget()
    p = SettingsPanel(
        host, s,
        on_save=lambda: None, on_reset=lambda: None,
        show_confirm=lambda *a, **k: None, show_consent=lambda **kw: None,
    )
    p._bind_timer.stop()
    yield p
    host.deleteLater()
    for name, value in saved.items():
        setattr(s, name, value)


def _bind_button(p: SettingsPanel) -> BindButton:
    return p._bind_buttons[_KEY]


def _assign(p: SettingsPanel, code: str = "f7") -> dict:
    binding = {"source": "keyboard", "code": code}
    p._finish_capture(_KEY, binding)
    return binding


def test_a_bind_button_listens_and_a_second_click_cancels(panel):
    panel._on_bind_clicked(_KEY)
    assert panel._configuring_key == _KEY
    assert _bind_button(panel).text() == BindButton._CONFIGURING_TEXT

    panel._on_bind_clicked(_KEY)
    assert panel._configuring_key is None
    assert _bind_button(panel).text() == "None"


def test_a_captured_key_is_assigned_and_shown_on_its_button(panel, threads):
    kb = threads["keyboard_thread"] = _FakeKeyboardThread()
    panel._on_bind_clicked(_KEY)
    assert kb.data.capture_active

    kb.data.capture_event = "f8"
    panel._poll_capture()

    binding = {"source": "keyboard", "code": "f8"}
    assert panel._configuring_key is None
    assert Settings.instance().cc_start_button == binding
    assert _bind_button(panel).text() == binding_display_name(binding)


def test_escape_ends_the_capture_and_leaves_the_binding_alone(panel, threads):
    kb = threads["keyboard_thread"] = _FakeKeyboardThread()
    binding = _assign(panel)
    panel._on_bind_clicked(_KEY)

    kb.data.capture_active = False
    panel._poll_capture()

    assert panel._configuring_key is None
    assert Settings.instance().cc_start_button == binding
    assert _bind_button(panel).text() == binding_display_name(binding)


def test_unassign_arms_then_clears_the_next_button_clicked(panel):
    _assign(panel)

    panel._on_unassign_clicked()
    assert panel._unassign_armed
    assert panel._unassign_btn.property("armed") is True

    panel._on_bind_clicked(_KEY)
    assert not panel._unassign_armed
    assert panel._unassign_btn.property("armed") is False
    assert Settings.instance().cc_start_button is None
    assert _bind_button(panel).text() == "None"


def test_clicking_unassign_twice_disarms_without_clearing(panel):
    binding = _assign(panel)

    panel._on_unassign_clicked()
    panel._on_unassign_clicked()

    assert not panel._unassign_armed
    assert panel._unassign_btn.property("armed") is False
    assert Settings.instance().cc_start_button == binding


def test_unassign_while_listening_clears_that_button(panel):
    _assign(panel)
    panel._on_bind_clicked(_KEY)

    panel._on_unassign_clicked()

    assert panel._configuring_key is None
    assert Settings.instance().cc_start_button is None
    assert _bind_button(panel).text() == "None"


def test_closing_the_drawer_cancels_a_capture_and_an_armed_unassign(panel):
    panel._on_bind_clicked(_KEY)
    panel.cancel_configuring()
    assert panel._configuring_key is None
    assert _bind_button(panel).text() == "None"

    panel._on_unassign_clicked()
    panel.cancel_configuring()
    assert not panel._unassign_armed
    assert panel._unassign_btn.property("armed") is False


def test_pedal_setup_walks_the_brake_then_the_gas_then_finishes(panel, threads):
    pedals = threads["main_pedal_thread"] = _FakePedalThread()

    panel._on_connect_pedals()
    assert panel._pedal_configuring
    assert panel.btn_connect.text() == _BRAKE_TEXT

    pedals.data.pedal_config_stage = "gas"
    panel._poll_pedal_config()
    assert panel.btn_connect.text() == _GAS_TEXT

    pedals.data.pedal_config_result = {"device_name": "Test pedals"}
    panel._poll_pedal_config()
    assert not panel._pedal_configuring
    assert panel.btn_connect.text() == "Connect to pedals"
    assert panel.lbl_conn_error.text() == ""


def test_clicking_the_pedal_button_again_cancels_the_setup(panel, threads):
    pedals = threads["main_pedal_thread"] = _FakePedalThread()

    panel._on_connect_pedals()
    panel._on_connect_pedals()

    assert not panel._pedal_configuring
    assert pedals.cancelled == 1
    assert panel.btn_connect.text() == "Connect to pedals"


def test_a_restarted_pedal_thread_ends_the_setup(panel, threads):
    pedals = threads["main_pedal_thread"] = _FakePedalThread()
    panel._on_connect_pedals()

    pedals.data.pedal_config_active = False
    panel._poll_pedal_config()

    assert not panel._pedal_configuring
    assert panel.btn_connect.text() == "Connect to pedals"


def test_closing_the_drawer_cancels_the_pedal_setup(panel, threads):
    pedals = threads["main_pedal_thread"] = _FakePedalThread()
    panel._on_connect_pedals()

    panel.cancel_configuring()

    assert not panel._pedal_configuring
    assert pedals.cancelled == 1
    assert panel.btn_connect.text() == "Connect to pedals"


def test_pedal_setup_reports_when_pedal_input_is_not_running(panel):
    panel._on_connect_pedals()

    assert not panel._pedal_configuring
    assert panel.lbl_conn_error.text() == "Pedal input is not running."


def test_reinstalling_the_sdk_hands_its_callback_to_the_installer(panel, monkeypatch):
    started: list = []
    monkeypatch.setattr("core.sdk_installer.start_reinstall", started.append)

    panel._do_reinstall_sdk()

    assert len(started) == 1 and callable(started[0])
