"""Resolve Settings bindings to held state. Formats: see INPUT_BINDINGS.md."""

from __future__ import annotations

import logging

from core.thread_management.registry import registry

logger = logging.getLogger(__name__)

_keyboard_available = False
_kb = None

try:
    import keyboard as _kb_module
    _kb = _kb_module
    _keyboard_available = True
except Exception:
    pass


def migrate_binding(raw: object) -> dict | None:
    """Legacy int/str to structured dict; dict with source unchanged."""
    if raw is None:
        return None
    if isinstance(raw, dict) and "source" in raw:
        return raw
    if isinstance(raw, int):
        # Legacy: bare button index on the configured pedal device.
        from core.settings import Settings  # lazy to avoid circular import at module load
        pedal_guid = Settings.device
        return {
            "source": "joystick",
            "device_guid": pedal_guid,
            "device_name": "",
            "code": raw,
        }
    if isinstance(raw, str):
        # Legacy: keyboard key name.
        return {"source": "keyboard", "code": raw}
    return None


def resolve_held(binding: object) -> bool:
    """True if binding is held; False on error or unassigned."""
    b = migrate_binding(binding)
    if b is None:
        return False
    source = b.get("source")
    if source == "joystick":
        return _resolve_joystick(b)
    if source == "keyboard":
        return _resolve_keyboard(b)
    if source == "button_device":
        return _resolve_button_device(b)
    return False


def binding_state(binding: object) -> bool | None:
    """Like resolve_held but None until the source device has published state."""
    b = migrate_binding(binding)
    if b is None:
        return False
    source = b.get("source")
    try:
        if source == "joystick":
            guid = b.get("device_guid")
            code = b.get("code")
            if not guid or code is None:
                return False
            pt = registry.get_thread("main_pedal_thread")
            with pt.data._lock:
                states = pt.data.joystick_button_states.get(guid)
            if not states:
                return None
            return bool(states.get(code, False))
        if source == "keyboard":
            key = b.get("code")
            if not key:
                return False
            if not _keyboard_available or _kb is None:
                return None
            return bool(_kb.is_pressed(key))
        if source == "button_device":
            vid_pid = b.get("vid_pid")
            button_id = b.get("button_id")
            if not vid_pid or button_id is None:
                return False
            bt = registry.get_thread("button_device_thread")
            with bt.data._lock:
                states = bt.data.button_states.get(vid_pid)
            if not states:
                return None
            return bool(states.get(button_id, False))
    except Exception:
        return None
    return False


def resolve_press_count(binding: object) -> int | None:
    """Monotonic press count from the source, or None if it does not count edges.

    Sources that see every edge (HID reports, the keyboard OS hook) count them,
    so a consumer polling slower than a tap lasts still sees the press.
    """
    b = migrate_binding(binding)
    if b is None:
        return None
    source = b.get("source")
    try:
        if source == "button_device":
            vid_pid = b.get("vid_pid")
            button_id = b.get("button_id")
            if not vid_pid or button_id is None:
                return None
            bt = registry.get_thread("button_device_thread")
            with bt.data._lock:
                counts = bt.data.button_press_counts.get(vid_pid) or {}
            return int(counts.get(button_id, 0))
        if source == "keyboard":
            key = b.get("code")
            if not key:
                return None
            kt = registry.get_thread("keyboard_thread")
            with kt.data._lock:
                counts = dict(kt.data.key_press_counts)
            return int(counts.get(str(key).capitalize(), 0))
    except (KeyError, AttributeError):
        return None
    except Exception:
        logger.debug("failed to resolve press count for %s", source, exc_info=True)
        return None
    return None


def binding_display_name(raw: object) -> str:
    """Display label for UI; prefers stored label."""
    b = migrate_binding(raw)
    if b is None:
        return "None"
    label = b.get("label")
    if label:
        return str(label)
    source = b.get("source")
    if source == "joystick":
        return f"button {b.get('code')}"
    if source == "keyboard":
        return str(b.get("code") or "None")
    if source == "button_device":
        return f"button {b.get('button_id')}"
    return "None"


def keyboard_is_pressed(key: str) -> bool:
    """Thread-safe wrapper around keyboard.is_pressed(). Returns False if unavailable."""
    if not _keyboard_available or _kb is None:
        return False
    if not key:
        return False
    try:
        return bool(_kb.is_pressed(key))
    except Exception:
        return False


def _resolve_joystick(binding: dict) -> bool:
    guid = binding.get("device_guid")
    code = binding.get("code")
    if not guid or code is None:
        return False
    try:
        pt = registry.get_thread("main_pedal_thread")
        with pt.data._lock:
            states = pt.data.joystick_button_states.get(guid, {})
        return bool(states.get(code, False))
    except (KeyError, AttributeError):
        return False
    except Exception:
        logger.debug("failed to resolve joystick binding %s/%s", guid, code, exc_info=True)
        return False


def _resolve_keyboard(binding: dict) -> bool:
    key = binding.get("code")
    if not key:
        return False
    return keyboard_is_pressed(key)


def _resolve_button_device(binding: dict) -> bool:
    vid_pid = binding.get("vid_pid")
    button_id = binding.get("button_id")
    if not vid_pid or button_id is None:
        return False
    try:
        bt = registry.get_thread("button_device_thread")
        with bt.data._lock:
            states = bt.data.button_states.get(vid_pid, {})
        return bool(states.get(button_id, False))
    except (KeyError, AttributeError):
        return False
    except Exception:
        logger.debug("failed to resolve button_device binding %s/%s", vid_pid, button_id, exc_info=True)
        return False

