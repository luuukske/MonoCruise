"""
Watchdog — monitors heartbeats and optionally restarts dead threads.

Detection:
  • thread.running is False  → crashed
  • heartbeat_age > HEARTBEAT_TIMEOUT → frozen (no loop() tick)

Restart:
  • Allowed while thread.restart_count < thread.max_restarts
  • A new instance is created via _factory (callable → BaseThread)
  • The factory must accept no arguments (use functools.partial if needed)
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TYPE_CHECKING

from core.thread_management.base_thread import BaseThread
from core.thread_management.registry    import registry

if TYPE_CHECKING:
    pass

logger = logging.getLogger("watchdog")

HEARTBEAT_TIMEOUT: float = 2.0   # seconds
POLL_INTERVAL:     float = 0.5   # watchdog check frequency


class Watchdog(BaseThread):
    loop_interval = POLL_INTERVAL

    def __init__(self, auto_restart: bool = True) -> None:
        super().__init__(name="watchdog")
        self._auto_restart = auto_restart
        # name → factory callable
        self._factories: dict[str, Callable[[], BaseThread]] = {}

    # ── factory registration ─────────────────────────────────────────────────

    def register_factory(self, name: str, factory: Callable[[], BaseThread]) -> None:
        self._factories[name] = factory

    # ── loop ─────────────────────────────────────────────────────────────────

    def loop(self) -> None:
        now = time.monotonic()
        for thread in registry.all_threads():
            if thread is self or not thread.watched:
                continue

            crashed  = not thread.running and thread.is_alive() is False
            age      = now - thread.heartbeat_at
            frozen   = thread.heartbeat_at > 0 and age > HEARTBEAT_TIMEOUT

            if not (crashed or frozen):
                continue

            reason = "crashed" if crashed else f"frozen ({age:.1f}s)"
            logger.warning("[%s] detected as %s", thread.name, reason)

            if not self._auto_restart:
                thread.healthy = False
                continue

            if thread.restart_count >= thread.max_restarts:
                logger.critical(
                    "[%s] exceeded max_restarts=%d — giving up",
                    thread.name, thread.max_restarts,
                )
                thread.healthy = False
                continue

            self._restart(thread)

    # ── restart ──────────────────────────────────────────────────────────────

    def _restart(self, dead: BaseThread) -> None:
        # Stop cleanly if still somehow alive
        if dead.is_alive():
            dead.stop()
            dead.join(timeout=2.0)

        factory = self._factories.get(dead.name)
        if factory is None:
            logger.critical(
                "[%s] no factory registered — cannot restart", dead.name
            )
            dead.healthy = False
            return

        new_thread               = factory()
        new_thread.restart_count = dead.restart_count + 1
        new_thread.name          = dead.name           # preserve name

        logger.info(
            "[%s] restarting (attempt %d/%d)",
            new_thread.name, new_thread.restart_count, new_thread.max_restarts,
        )

        registry.replace(new_thread)
        new_thread.start()
