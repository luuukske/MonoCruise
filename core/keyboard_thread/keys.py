"""Physical names for keyboard bindings.

The keyboard library gives a numpad digit and the number-row digit the same
name, and its held check is true when either key is down. Numpad digits are
stored as ``Numpad 1`` and so on, so a binding is one physical key.

The scan codes are the XT set-1 codes. Linux evdev uses the same numbers for
``KEY_KP0`` through ``KEY_KP9``. Num Lock renames those keys to Insert, End
and the arrows, but the scan code stays put, so the binding does not change.
"""

from __future__ import annotations

# XT set-1 / evdev KEY_KPn. Num Lock changes the name, not this code.
_NUMPAD_DIGIT_SCAN = {
    82: "0",
    79: "1",
    80: "2",
    81: "3",
    75: "4",
    76: "5",
    77: "6",
    71: "7",
    72: "8",
    73: "9",
}


def binding_code(event: object) -> str | None:
    """Name to store for this key event, or None when it has no name."""
    raw_name = getattr(event, "name", None)
    text = str(raw_name).strip().lower() if raw_name else ""
    scan = getattr(event, "scan_code", None)
    keypad = bool(getattr(event, "is_keypad", False))
    if (
        keypad
        and isinstance(scan, int)
        and not isinstance(scan, bool)
        and scan in _NUMPAD_DIGIT_SCAN
    ):
        return "Numpad " + _NUMPAD_DIGIT_SCAN[scan]
    if not text:
        return None
    if keypad and len(text) == 1 and text in "0123456789":
        return "Numpad " + text
    return text.capitalize()


def is_digit_code(code: object) -> bool:
    """True for a number-row digit or a numpad digit binding."""
    text = str(code).capitalize()
    if len(text) == 1 and text in "0123456789":
        return True
    prefix = "Numpad "
    return (
        text.startswith(prefix)
        and len(text) == len(prefix) + 1
        and text[-1] in "0123456789"
    )
