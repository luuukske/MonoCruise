"""Heartbeat watchdog: restart crashed/frozen workers via registered factories. See AGENTS.md."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, TYPE_CHECKING

from core.thread_management.base_thread import BaseThread
from core.thread_management.registry    import registry

if TYPE_CHECKING:
    pass

logger = logging.getLogger("watchdog")


def _log_name(name: str) -> str:
    return (name.replace("_thread", "") + " code").capitalize()


HEARTBEAT_TIMEOUT:     float = 1.5   # seconds
POLL_INTERVAL:         float = 0.25  # watchdog check frequency
FREEZE_STREAK_RESTART: int   = 3     # consecutive stale-heartbeat polls before a freeze restart
STOP_GRACE:            float = 0.2   # seconds: brief join() so a cleanly-stoppable thread can exit
LOW_FPS_THRESHOLD:    float = 0.70  # warn when actual FPS < this fraction of wanted FPS
LOW_FPS_STREAK_WARN:  int   = 5     # consecutive low-fps polls before warning
LOW_FPS_WARN_WINDOW:  float = 3.0   # seconds: if >1 thread warned in this window → "some threads"


class Watchdog(BaseThread):
    loop_interval = POLL_INTERVAL

    def __init__(self) -> None:
        super().__init__(name="watchdog")
        # name → factory callable
        self._factories: dict[str, Callable[[], BaseThread]] = {}
        # consecutive below-threshold poll counts per thread
        self._low_fps_streaks:    dict[str, int]   = {}
        # thread_name → monotonic time when streak threshold was first hit (pending warn)
        self._low_fps_candidates: dict[str, float] = {}
        # thread_name → monotonic time when its warning was shown (suppress re-warn)
        self._low_fps_warned_at:  dict[str, float] = {}
        # consecutive stale-heartbeat poll counts per thread (freeze debounce)
        self._frozen_streaks:     dict[str, int]   = {}

    # factory registration

    def register_factory(self, name: str, factory: Callable[[], BaseThread]) -> None:
        self._factories[name] = factory

    # loop

    def loop(self) -> None:
        now = time.monotonic()
        for thread in registry.all_threads():
            if thread is self or not getattr(thread, "watched", False) or not getattr(thread, "healthy", False):
                continue

            # Batch low-FPS popup after LOW_FPS_WARN_WINDOW.
            tname    = getattr(thread, "name", "")
            interval = getattr(thread, "loop_interval", 0.0)
            if interval > 0:
                wanted_fps    = 1.0 / interval
                actual_fps    = getattr(thread, "inst_fps", 0.0)
                threshold_fps = LOW_FPS_THRESHOLD * wanted_fps
                if actual_fps > 0 and actual_fps < threshold_fps:
                    self._low_fps_streaks[tname] = self._low_fps_streaks.get(tname, 0) + 1
                    if (
                        self._low_fps_streaks[tname] >= LOW_FPS_STREAK_WARN
                        and tname not in self._low_fps_warned_at
                        and tname not in self._low_fps_candidates
                    ):
                        self._low_fps_candidates[tname] = now
                else:
                    self._low_fps_streaks.pop(tname, None)
                    self._low_fps_candidates.pop(tname, None)  # recovered before window elapsed

            # Defensive compatibility: skip entries that are not restartable workers.
            restart_count = getattr(thread, "restart_count", None)
            max_restarts = getattr(thread, "max_restarts", None)
            if restart_count is None or max_restarts is None:
                continue

            crashed = not thread.running and thread.is_alive() is False
            age     = now - thread.heartbeat_at
            stale   = thread.heartbeat_at > 0 and age > HEARTBEAT_TIMEOUT

            # Heartbeat stale for FREEZE_STREAK_RESTART polls before restart.
            if stale and not crashed:
                self._frozen_streaks[tname] = self._frozen_streaks.get(tname, 0) + 1
            else:
                self._frozen_streaks.pop(tname, None)
            frozen = self._frozen_streaks.get(tname, 0) >= FREEZE_STREAK_RESTART

            if not (crashed or frozen):
                continue

            # telemetry_thread with request_quit: do not restart (main handles shutdown).
            if getattr(thread, "name", "") == "telemetry_thread":
                data = getattr(thread, "data", None)
                if getattr(data, "request_quit", False):
                    thread.healthy = False
                    continue


            if restart_count >= max_restarts:
                # logging is handled in base_thread.py
                thread.healthy = False
                if not crashed:
                    logger.error(
                        "%s froze and couldn't restart.\nRestart MonoCruise or contact a developer.",
                        _log_name(thread.name),
                        extra={"popup": True},
                    )
                    thread.stop(force=True)
                continue
            else:
                if not crashed:
                    logger.warning(
                        "%s has frozen.\nRestarting now...",
                        _log_name(thread.name),
                        extra={"popup": True},
                )

            if not getattr(self, "_auto_restart", True):
                thread.healthy = False
                continue

            self._restart(thread)

        # Flush pending low-FPS candidates once the collection window has elapsed.
        if self._low_fps_candidates:
            oldest = min(self._low_fps_candidates.values())
            if now - oldest >= LOW_FPS_WARN_WINDOW:
                pending = list(self._low_fps_candidates.keys())
                if len(pending) > 1:
                    logger.warning(
                        "Some threads are running below 70%% of their target FPS.",
                        extra={"popup": True},
                    )
                else:
                    logger.warning(
                        "%s is running below 70%% of its target FPS.",
                        _log_name(pending[0]),
                        extra={"popup": True},
                    )
                for n in pending:
                    self._low_fps_warned_at[n] = now
                self._low_fps_candidates.clear()

    # restart

    def _restart(self, dead: BaseThread) -> None:
        factory = self._factories.get(dead.name)
        if factory is None:
            logger.critical(
                "thread '%s' cannot be restarted: no factory registered: this is a MAJOR bug.",
                dead.name,
            )
            logger.critical(
                "Internal error: component can't restart. Please contact a developer.",
                extra={"popup": True},
            )
            dead.healthy = False
            return

        # safe_state on a throwaway thread before swap (may block on I/O).
        try:
            threading.Thread(
                target=dead.safe_state,
                name=f"{dead.name}-safe-state",
                daemon=True,
            ).start()
        except Exception:
            logger.exception(
                "could not launch safe_state for '%s' (suppressed)", dead.name
            )

        # Brief stop grace only; do not join a syscall-frozen thread indefinitely.
        if dead.is_alive():
            dead.stop(force=True)
            dead.join(timeout=STOP_GRACE)
            if dead.is_alive():
                logger.warning(
                    "old '%s' thread did not stop in time; starting its "
                    "replacement anyway (stale thread is a daemon and will "
                    "self-exit once unblocked).",
                    dead.name,
                )

        new_thread               = factory()
        new_thread.restart_count = dead.restart_count + 1
        new_thread.name          = dead.name           # preserve name

        logger.info(
            "Restarting '%s' (attempt %d of %d).",
            new_thread.name, new_thread.restart_count, new_thread.max_restarts,
        )

        registry.replace(new_thread)
        new_thread.start()
        self._frozen_streaks.pop(dead.name, None)

