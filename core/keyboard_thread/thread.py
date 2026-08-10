"""Keyboard library lifecycle wrapper; is_pressed used from input_bindings."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry  # noqa: F401  (keep for watchdog compat)

logger = logging.getLogger(__name__)

_keyboard_available = False
_kb = None

try:
    import keyboard as _kb_module
    _kb = _kb_module
    _keyboard_available = True
except Exception:
    logger.warning("keyboard library not importable: keyboard button bindings disabled")


@dataclass
class KeyboardThreadData(ThreadData):
    is_available: bool = False
    # {capitalized key name: monotonic press count} for bound keys only. Read
    # from the OS hook so a tap shorter than a consumer's tick still counts.
    key_press_counts: dict = field(default_factory=dict, repr=False)
    capture_active: bool = False
    capture_event: str | None = None  # capitalized key name when captured, e.g. "A", "Space"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class KeyboardThread(BaseThread):
    loop_interval = 0.2    # 200 ms: heartbeat only; keyboard lib runs its own OS hook
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="keyboard_thread")
        self.data = KeyboardThreadData()
        self._capture_hook = None
        self._press_hook = None
        # Only keys currently bound to a CC button are counted, so this never
        # becomes a tally of everything typed.
        self._watched_keys: set[str] = set()
        self._keys_down: set[str] = set()

    def setup(self) -> None:
        if not _keyboard_available or _kb is None:
            with self.data._lock:
                self.data.is_available = False
            logger.warning(
                "keyboard library unavailable: keyboard button bindings will not work"
            )
            return

        try:
            _kb.unhook_all()
        except Exception:
            pass

        with self.data._lock:
            self.data.is_available = True
        self._refresh_watched_keys()
        self._install_press_counter()
        logger.debug("keyboard_thread setup complete")

    def loop(self) -> None:
        if not self.running:
            return
        # keyboard lib manages its own OS hook thread; only the watched-key set
        # needs refreshing here, in case a binding changed.
        self._refresh_watched_keys()

    def _refresh_watched_keys(self) -> None:
        """Track only the keys bound to CC buttons."""
        from core.input_bindings import migrate_binding
        from core.settings import Settings

        keys: set[str] = set()
        for name in (
            "cc_start_button", "cc_inc_button", "cc_dec_button",
            "acc_dist_inc_button", "acc_dist_dec_button",
        ):
            try:
                b = migrate_binding(getattr(Settings, name, None))
            except Exception:
                continue
            if b and b.get("source") == "keyboard" and b.get("code"):
                keys.add(str(b["code"]).capitalize())
        self._watched_keys = keys

    def _install_press_counter(self) -> None:
        """Count press edges for bound keys from the OS hook, ignoring auto-repeat."""
        if not _keyboard_available or _kb is None:
            return
        data = self.data

        def _on_event(event) -> None:
            name = getattr(event, "name", None)
            if not name:
                return
            key = str(name).capitalize()
            if key not in self._watched_keys:
                return
            if getattr(event, "event_type", None) == "down":
                # Held keys repeat at the OS repeat rate; only the first counts.
                if key in self._keys_down:
                    return
                self._keys_down.add(key)
                with data._lock:
                    counts = dict(data.key_press_counts)
                    counts[key] = counts.get(key, 0) + 1
                    data.key_press_counts = counts
            else:
                self._keys_down.discard(key)

        try:
            self._press_hook = _kb.hook(_on_event, suppress=False)
        except Exception:
            logger.debug("failed to install keyboard press counter", exc_info=True)

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
        """Enable capture mode: next key press populates data.capture_event."""
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

