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
   "code": int}                     → button index if code < button_count,
                                       or virtual hat index using the encoding:
                                       hat_virtual = button_count + hat_idx*4 + dir
                                       dir: 0=up 1=right 2=down 3=left
  {"source": "keyboard",
   "code": str}                     → capitalized key name, e.g. "A", "Space", "F1"

Public API
----------
migrate_binding(raw)   — upgrade legacy int/str to dict; pass-through for dict/None
resolve_held(binding)  — True if the described input is currently held
keyboard_is_pressed(key) — safe wrapper around keyboard.is_pressed()
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
    Returns False on any error — safe default.
    """
    b = migrate_binding(binding)
    if b is None:
        return False
    source = b.get("source")
    if source == "joystick":
        return _resolve_joystick(b)
    if source == "keyboard":
        return _resolve_keyboard(b)
    return False


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
