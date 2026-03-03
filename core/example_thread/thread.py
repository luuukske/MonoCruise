"""
TEMPLATE THREAD — copy this directory and rename it for your worker.

Convention:
  core/<feature_name>/thread.py   ← worker lives here
  core/<feature_name>/__init__.py ← keep empty

Steps:
  1. Copy core/example_thread/ → core/<your_name>/
  2. Rename MyThread / MyThreadData to match your feature.
  3. Implement setup(), loop(), teardown().
  4. Register + start in main.py.
  5. Other threads access data via:
       registry.get_thread("my_thread").data.my_field
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import threading
import time

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry    import registry

logger = logging.getLogger(__name__)


# Typed data container — other threads read fields directly (GIL-safe)

@dataclass
class MyThreadData(ThreadData):
    # Add your typed fields here:
    value: float = 0.0
    label: str   = ""

    # For consistent multi-field reads, expose snapshot():
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class MyThread(BaseThread):
    loop_interval = 0.50   # seconds — rate-limit your loop here
    max_restarts  = 5

    def __init__(self) -> None:
        super().__init__(name="my_thread")
        self.data = MyThreadData()
        self._settings = None   # inject via constructor or module import

    # lifecycle

    def setup(self) -> None:
        """Runs once before the loop. Raise to abort startup."""
        logger.debug("setup complete")

    def loop(self) -> None:
        """
        Main work unit. Called every `loop_interval` seconds.
        Do NOT call time.sleep() here — the base class handles pacing.
        Raise any exception to trigger watchdog handling.
        """
        # read from another thread (single primitive field, GIL-safe) ---
        # other = registry.get_thread("other_thread")
        # speed = other.data.speed   # float → atomic read

        # read multiple fields consistently ---
        # snap = other.data.snapshot()

        # update own data (single field, GIL-safe) ---
        self.data.value += 1.0

        # update multiple fields together (use lock) ---
        with self.data._lock:
            self.data.value += 1.0
            self.data.label  = "running"

    def teardown(self) -> None:
        """Runs once after loop exits. Exceptions are suppressed by base."""
        logger.debug("teardown complete")
