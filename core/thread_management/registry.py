"""
Registry — central directory of named BaseThread instances.

Rules:
  • Typed dataclass access: registry.get_thread("telemetry").data.speed
  • No string key-value store.
  • Thread-safe via a single RLock.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.thread_management.base_thread import BaseThread


class Registry:
    def __init__(self) -> None:
        self._lock    = threading.RLock()
        self._threads: dict[str, "BaseThread"] = {}

    # registration

    def register(self, thread: "BaseThread") -> None:
        with self._lock:
            if thread.name in self._threads:
                raise KeyError(f"Thread '{thread.name}' already registered")
            self._threads[thread.name] = thread

    def unregister(self, name: str) -> None:
        with self._lock:
            self._threads.pop(name, None)

    def replace(self, thread: "BaseThread") -> None:
        """Unregister old + register new atomically (used by watchdog)."""
        with self._lock:
            self._threads[thread.name] = thread

    # lookup

    def get_thread(self, name: str) -> "BaseThread":
        with self._lock:
            t = self._threads.get(name)
        if t is None:
            raise KeyError(f"No thread named '{name}'")
        return t

    def all_threads(self) -> list["BaseThread"]:
        with self._lock:
            return list(self._threads.values())

    def names(self) -> list[str]:
        with self._lock:
            return list(self._threads.keys())


# Module-level singleton
registry = Registry()
