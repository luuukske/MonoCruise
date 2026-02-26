"""
BaseThread — inherit this for every worker thread.

Lifecycle:
    setup()   → called once before the loop starts
    loop()    → called repeatedly at `loop_interval` seconds
    teardown()→ called once after the loop exits (even on error)

The watchdog reads `heartbeat_at` and `restart_count` / `stable_loops`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal typed data container every thread exposes
# ---------------------------------------------------------------------------

@dataclass
class ThreadData:
    """Override in subclass to add typed fields."""
    pass


# ---------------------------------------------------------------------------
# BaseThread
# ---------------------------------------------------------------------------

class BaseThread(threading.Thread):
    # ── tunables ────────────────────────────────────────────────────────────
    loop_interval: float = 0.05          # seconds between loop() calls
    max_restarts:  int   = 2             # restarts allowed before giving up
    stable_after:  int   = 100           # loops without error → stable again
    watched:       bool  = True          # False → watchdog skips this thread

    def __init__(self, name: str, *, daemon: bool = True) -> None:
        super().__init__(name=name, daemon=daemon)

        self.data: ThreadData = ThreadData()
        self._lock             = threading.Lock()

        # ── watchdog-visible state (primitives → GIL-safe bare reads) ───────
        self.heartbeat_at:  float = 0.0
        self.restart_count: int   = 0
        self.stable_loops:  int   = 0
        self.running:       bool  = False
        self.healthy:       bool  = True   # set False by watchdog

        self._stop_event = threading.Event()

    # ── public API ───────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> dict[str, Any]:
        """Consistent multi-field read (acquire lock once)."""
        with self._lock:
            return {
                "name":          self.name,
                "restart_count": self.restart_count,
                "stable_loops":  self.stable_loops,
                "heartbeat_age": round(time.monotonic() - self.heartbeat_at, 3),
                "running":       self.running,
                "healthy":       self.healthy,
            }

    # ── override in subclass ─────────────────────────────────────────────────

    def setup(self) -> None:
        """Called once before the loop. Raise to abort startup."""

    def loop(self) -> None:
        """Called repeatedly. Raise to trigger watchdog restart logic."""

    def teardown(self) -> None:
        """Called after loop exits; exceptions are suppressed."""

    # ── internal ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        log = logging.getLogger(self.name)
        self.running = True
        self._stop_event.clear()

        log.debug("setup")
        try:
            self.setup()
        except Exception:
            log.exception("setup() failed — thread will not start")
            self.running = False
            self.healthy = False
            return

        log.debug("loop starting")
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self.loop()
                self.heartbeat_at = time.monotonic()
                self.stable_loops += 1
                if self.stable_loops >= self.stable_after:
                    if self.restart_count > 0:
                        log.info(
                            "stable after %d loops — restart quota reset",
                            self.stable_after,
                        )
                    self.restart_count = 0
                    self.stable_loops  = 0
            except Exception:
                if self.restart_count >= self.max_restarts:
                    log.critical("Reached max restarts. Check logs for errors and contact devs.")
                    self.running = False
                    break
                log.exception("Unexpected error in the loop. Restarting...")
                self.running = False
                break

            elapsed = time.monotonic() - t0
            wait    = self.loop_interval - elapsed
            if wait > 0:
                self._stop_event.wait(wait)

        log.debug("teardown")
        try:
            self.teardown()
        except Exception:
            log.exception("teardown() raised (suppressed)")

        self.running = False
        log.debug("exited")
