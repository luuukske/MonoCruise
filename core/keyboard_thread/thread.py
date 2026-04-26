"""
Keyboard Thread — lifecycle wrapper for the `keyboard` library.

Provides:
  - Global keyboard state accessible via keyboard.is_pressed() from any thread.
  - Capture mode for UI button assignment (future use): next key press populates
    capture_event so the UI can read and save it.

This thread's own loop() is lightweight (heartbeat only). The `keyboard` library
runs its hooks on its own OS-level background thread; state is read directly via
keyboard.is_pressed() in core.input_bindings without going through this thread's data.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry  # noqa: F401 — keep for watchdog compat

logger = logging.getLogger(__name__)

_keyboard_available = False
_kb = None

try:
    import keyboard as _kb_module
    _kb = _kb_module
    _keyboard_available = True
except Exception:
    logger.warning("keyboard library not importable — keyboard button bindings disabled")


@dataclass
class KeyboardThreadData(ThreadData):
    is_available: bool = False
    capture_active: bool = False
    capture_event: str | None = None  # capitalized key name when captured, e.g. "A", "Space"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class KeyboardThread(BaseThread):
    loop_interval = 0.05   # 50 ms — just keeps the heartbeat alive
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="keyboard_thread")
        self.data = KeyboardThreadData()
        self._capture_hook = None

    def setup(self) -> None:
        if not _keyboard_available or _kb is None:
            with self.data._lock:
                self.data.is_available = False
            logger.warning(
                "keyboard library unavailable — keyboard button bindings will not work"
            )
            return

        try:
            _kb.unhook_all()
        except Exception:
            pass

        with self.data._lock:
            self.data.is_available = True
        logger.debug("keyboard_thread setup complete")

    def loop(self) -> None:
        if not self.running:
            return
        # keyboard lib manages its own OS hook thread; nothing to poll here.

    def teardown(self) -> None:
        self._remove_capture_hook()
        if _keyboard_available and _kb is not None:
            try:
                _kb.unhook_all()
            except Exception:
                pass
        logger.debug("keyboard_thread teardown complete")

    # --- Capture API (called from UI thread when user clicks "assign button") ---

    def start_capture(self) -> None:
        """Enable capture mode — next key press populates data.capture_event."""
        if not _keyboard_available:
            return
        with self.data._lock:
            self.data.capture_active = True
            self.data.capture_event = None
        self._install_capture_hook()

    def cancel_capture(self) -> None:
        """Abort capture without saving anything."""
        self._remove_capture_hook()
        with self.data._lock:
            self.data.capture_active = False
            self.data.capture_event = None

    def consume_capture(self) -> str | None:
        """Read + clear the captured key name. Returns None if nothing was captured."""
        self._remove_capture_hook()
        with self.data._lock:
            ev = self.data.capture_event
            self.data.capture_event = None
            self.data.capture_active = False
            return ev

    def _install_capture_hook(self) -> None:
        self._remove_capture_hook()
        if not _keyboard_available or _kb is None:
            return

        data = self.data

        def _on_press(event) -> None:
            name = getattr(event, "name", None)
            if not name:
                return
            key = name.capitalize()
            if key in ("Esc", "Escape"):
                with data._lock:
                    data.capture_active = False
                return
            # Skip modifier-only keys that can't be unambiguously bound
            if key in ("Shift", "Ctrl", "Alt", "Win", "Tab", "Caps Lock"):
                return
            with data._lock:
                if data.capture_active and data.capture_event is None:
                    data.capture_event = key
                    data.capture_active = False

        try:
            self._capture_hook = _kb.on_press(_on_press, suppress=False)
        except Exception:
            logger.debug("failed to install keyboard capture hook", exc_info=True)

    def _remove_capture_hook(self) -> None:
        hook = self._capture_hook
        if hook is not None and _keyboard_available and _kb is not None:
            try:
                _kb.unhook(hook)
            except Exception:
                pass
        self._capture_hook = None
