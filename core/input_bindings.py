"""
Input binding resolution.

Converts a raw binding value (from Settings) into a live held/not-held bool.

Binding formats (stored in config.json / Settings):
  null / None                       → unassigned → always False
  int       (legacy)                → joystick button on the configured pedal device
  str       (legacy)                → keyboard key (capitalized name, e.g. "A")
  {"source": "joystick",
   "device_guid": str,
   "device_name": str,              → optional, display only
   "label": str,                    → optional, display only (e.g. "button 7", "hat up")
   "code": int}                     → button index if code < button_count,
                                       or virtual hat index using the encoding:
                                       hat_virtual = button_count + hat_idx*4 + dir
                                       dir: 0=up 1=right 2=down 3=left
  {"source": "keyboard",
   "code": str}                     → capitalized key name, e.g. "A", "Space", "F1"
  {"source": "button_device",
   "vid_pid": str,                  → "{vendor_id:04x}:{product_id:04x}", e.g. "0483:0001"
   "device_name": str,              → optional, display only
   "label": str,                    → optional, display only
   "button_id": int}                → byte_index * 8 + bit_index from raw HID report

Public API
----------
migrate_binding(raw)  : upgrade legacy int/str to dict; pass-through for dict/None
resolve_held(binding) : True if the described input is currently held
binding_state(binding): tri-state resolve: True/False, or None when the
                        source device has not reported any state yet
keyboard_is_pressed(key): safe wrapper around keyboard.is_pressed()
binding_display_name(raw): short human-readable name for UI display
"""

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
    """Upgrade a legacy bare-int or bare-str binding to the structured dict format.

    Called at read time so legacy configs continue working without any migration step.
    dict bindings with a "source" key are returned unchanged.
    """
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
    """Return True if the described input is currently held.

    Joystick state is read from main_pedal_thread.data.joystick_button_states.
    Keyboard state is read via keyboard.is_pressed().
    Returns False on any error: safe default.
    """
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
    """Tri-state version of resolve_held.

    Returns True/False when the source device has published state for this
    input, and None when it hasn't (device not tracked yet, no report seen,
    keyboard lib unavailable). The capture guard uses None to keep a freshly
    assigned binding suppressed until its device actually reports a release,
    instead of clearing on a not-yet-connected device.
    """
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


def binding_display_name(raw: object) -> str:
    """Short human-readable name for a binding, for UI display.

    Prefers the stored "label" (written at capture time, when the device's
    button/hat layout was known); falls back to a generic name derived from
    the code. Returns "None" for unassigned.
    """
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

