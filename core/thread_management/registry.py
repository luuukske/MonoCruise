"""
Registry: central directory of named components.

Rules:
  • Thread access: registry.get_thread("telemetry").data.speed
  • Object access: registry.get("main_window")
  • Thread-safe via a single RLock.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.thread_management.base_thread import BaseThread


class Registry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._threads: dict[str, "BaseThread"] = {}
        self._objects: dict[str, object] = {}

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
            return [*self._threads.keys(), *self._objects.keys()]

    # generic object registration

    def register_object(self, name: str, value: object) -> None:
        with self._lock:
            if name in self._threads or name in self._objects:
                raise KeyError(f"Name '{name}' already registered")
            self._objects[name] = value

    def unregister_object(self, name: str) -> None:
        with self._lock:
            self._objects.pop(name, None)

    def get(self, name: str) -> object:
        with self._lock:
            t = self._threads.get(name)
            if t is not None:
                return t
            obj = self._objects.get(name)
        if obj is None:
            raise KeyError(f"No component named '{name}'")
        return obj

    def object_names(self) -> list[str]:
        with self._lock:
            return list(self._objects.keys())


# Module-level singleton
registry = Registry()

