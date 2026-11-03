"""Number-row digits and numpad digits are different keys.

The keyboard library names both "1" and reports either as held. Bindings use
the physical key: scan code plus the keypad flag, so Num Lock does not rename
a numpad digit into End or Insert.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.input_bindings import binding_state, keyboard_is_pressed, resolve_press_count
from core.keyboard_thread.keys import binding_code, is_digit_code
from core.keyboard_thread.thread import KeyboardThread
from core.thread_management.registry import registry

# XT set-1 / evdev KEY_KPn, and the name Windows uses for that key with Num Lock off.
_NUMPAD = (
    (82, "0", "insert"),
    (79, "1", "end"),
    (80, "2", "down"),
    (81, "3", "page down"),
    (75, "4", "left"),
    (76, "5", "clear"),
    (77, "6", "right"),
    (71, "7", "home"),
    (72, "8", "up"),
    (73, "9", "page up"),
)


def _event(name, scan, keypad, kind="down"):
    return SimpleNamespace(
        name=name, scan_code=scan, is_keypad=keypad, event_type=kind
    )


@pytest.mark.parametrize("scan,digit,nav", _NUMPAD)
def test_numpad_digit_stays_a_numpad_key_with_num_lock_off(scan, digit, nav):
    assert binding_code(_event(digit, scan, True)) == f"Numpad {digit}"
    assert binding_code(_event(nav, scan, True)) == f"Numpad {digit}"
    assert binding_code(_event(nav, scan, False)) == nav.capitalize()


@pytest.mark.parametrize("digit", "0123456789")
def test_number_row_stays_the_plain_digit(digit):
    assert binding_code(_event(digit, 2, False)) == digit
    assert binding_code(_event(digit, None, True)) == f"Numpad {digit}"


def test_a_nameless_numpad_digit_still_has_a_code():
    assert binding_code(_event(None, 79, True)) == "Numpad 1"
    assert binding_code(_event(None, 2, False)) is None


def test_digit_codes_are_recognised_either_way():
    assert is_digit_code("1")
    assert is_digit_code("numpad 1")
    assert not is_digit_code("End")
    assert not is_digit_code("Numpad 10")
    assert not is_digit_code("A")


def test_held_state_and_presses_keep_the_two_keys_apart():
    thread = KeyboardThread()
    thread.data.is_available = True
    thread._watched_keys = {"1", "Numpad 1"}
    previous = registry._threads.get("keyboard_thread")
    registry.replace(thread)
    try:
        thread._on_key_event(_event("1", 2, False))
        thread._on_key_event(_event("end", 79, True))
        thread._on_key_event(_event("1", 2, False))  # auto-repeat
        assert keyboard_is_pressed("1") is True
        assert keyboard_is_pressed("Numpad 1") is True
        assert binding_state({"source": "keyboard", "code": "1"}) is True
        assert resolve_press_count({"source": "keyboard", "code": "1"}) == 1
        assert resolve_press_count({"source": "keyboard", "code": "Numpad 1"}) == 1

        thread._on_key_event(_event("1", 2, False, "up"))
        assert keyboard_is_pressed("1") is False
        assert keyboard_is_pressed("numpad 1") is True
        assert binding_state({"source": "keyboard", "code": "Numpad 1"}) is True
    finally:
        if previous is None:
            registry.unregister("keyboard_thread")
        else:
            registry.replace(previous)


def test_an_unbound_digit_is_tracked_but_not_counted():
    thread = KeyboardThread()
    thread.data.is_available = True
    thread._watched_keys = set()
    thread._on_key_event(_event("4", 75, True))
    assert "Numpad 4" in thread.data.keys_held
    assert thread.data.key_press_counts == {}


def test_capture_stores_the_numpad_key_and_still_ignores_modifiers():
    thread = KeyboardThread()
    thread.data.capture_active = True
    thread._on_capture_press(_event("shift", 42, False))
    assert thread.data.capture_event is None
    assert thread.data.capture_active is True
    thread._on_capture_press(_event("end", 79, True))
    assert thread.data.capture_event == "Numpad 1"
    assert thread.data.capture_active is False

    thread.data.capture_active = True
    thread.data.capture_event = None
    thread._on_capture_press(_event("esc", 1, False))
    assert thread.data.capture_event is None
    assert thread.data.capture_active is False


def test_digit_held_is_unknown_until_the_keyboard_thread_is_up():
    previous = registry._threads.get("keyboard_thread")
    registry.unregister("keyboard_thread")
    try:
        assert keyboard_is_pressed("1") is False
        assert binding_state({"source": "keyboard", "code": "1"}) is None
    finally:
        if previous is not None:
            registry.replace(previous)
