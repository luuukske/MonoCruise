from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar


logger = logging.getLogger("monocruise")


DataT = TypeVar("DataT")


@dataclass
class SnapshotMixin:
    """
    Base mixin for per-thread dataclasses.

    - Owners (the thread itself) may mutate fields freely.
    - Other threads should treat instances as read-only and use snapshot()
      when they need a consistent multi-field view.
    """

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def snapshot(self) -> "SnapshotMixin":
        """Return a shallow, locked copy suitable for cross-thread reading."""
        # Import lazily to avoid a global dependency
        import copy

        with self._lock:
            return copy.copy(self)

    def update(self, **changes) -> None:
        """Atomically apply multiple updates under the internal lock."""
        with self._lock:
            for key, value in changes.items():
                setattr(self, key, value)


class BaseThread(Generic[DataT], threading.Thread):
    """
    Cooperative worker thread with:

    - Structured lifecycle hooks (on_startup / loop / on_shutdown)
    - Built-in loop rate limiting via loop_interval (seconds)
    - Heartbeat tracking for watchdog supervision
    - Basic performance metrics (iteration count, last loop duration)
    - Exception isolation (no worker exception escapes the thread)

    Subclasses must:
    - Provide a create_initial_data() method that returns the per-thread
      dataclass instance.
    - Implement loop(), which will be called repeatedly while the thread
      is running.
    """

    #: Maximum tolerated heartbeat age before a watchdog considers
    #: the thread stalled, in seconds. Watchdog can override per-thread
    #: behaviour via the registry if needed.
    heartbeat_timeout: float = 1.0

    def __init__(
        self,
        *,
        name: str,
        loop_interval: float,
        auto_restart: bool = True,
        daemon: bool = True,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        if loop_interval <= 0:
            raise ValueError("loop_interval must be > 0")

        self.loop_interval = float(loop_interval)
        self.auto_restart = bool(auto_restart)

        self._stop_event = threading.Event()
        self._running = threading.Event()

        self._iteration_count = 0
        self._last_loop_started = 0.0
        self._last_loop_duration = 0.0

        self._heartbeat_lock = threading.Lock()
        self._last_heartbeat = 0.0

        self._last_exception: Optional[str] = None

        # Per-thread public data container. Subclasses should override
        # create_initial_data() to return a typed dataclass (ideally
        # derived from SnapshotMixin).
        self.data: DataT = self.create_initial_data()

    # ------------------------------------------------------------------
    # Lifecycle API for subclasses
    # ------------------------------------------------------------------

    def create_initial_data(self) -> DataT:  # pragma: no cover - interface
        """
        Return the initial dataclass instance for this thread.

        Subclasses must override this method.
        """
        raise NotImplementedError

    def on_startup(self) -> None:  # pragma: no cover - default hook
        """Optional: one-time initialization before the main loop starts."""
        return None

    def loop(self) -> None:  # pragma: no cover - interface
        """
        One logical unit of work.

        This method is called repeatedly while the thread is running.
        Do not call time.sleep() here for rate limiting; instead adjust
        self.loop_interval when constructing the thread.
        """
        raise NotImplementedError

    def on_shutdown(self) -> None:  # pragma: no cover - default hook
        """Optional: best-effort cleanup after the loop exits."""
        return None

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request cooperative shutdown."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """True while the thread's main loop is active."""
        return self._running.is_set()

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    @property
    def last_loop_started(self) -> float:
        return self._last_loop_started

    @property
    def last_loop_duration(self) -> float:
        return self._last_loop_duration

    @property
    def last_heartbeat(self) -> float:
        with self._heartbeat_lock:
            return self._last_heartbeat

    @property
    def last_exception(self) -> Optional[str]:
        return self._last_exception

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_heartbeat(self, ts: Optional[float] = None) -> None:
        if ts is None:
            ts = time.monotonic()
        with self._heartbeat_lock:
            self._last_heartbeat = ts

    # ------------------------------------------------------------------
    # Thread.run implementation
    # ------------------------------------------------------------------

    def run(self) -> None:  # pragma: no cover - behaviour, not logic
        thread_name = self.name or self.__class__.__name__
        logger.debug("Thread %s starting", thread_name)

        self._running.set()
        self._update_heartbeat()

        try:
            try:
                self.on_startup()
            except Exception:
                self._last_exception = traceback.format_exc()
                logger.error(
                    "Fatal error during startup in %s:\n%s",
                    thread_name,
                    self._last_exception,
                )
                return

            next_run = time.monotonic()

            while not self._stop_event.is_set():
                start = time.monotonic()
                self._last_loop_started = start

                try:
                    self.loop()
                    self._last_exception = None
                except Exception:
                    # Never let worker exceptions escape the thread boundary.
                    self._last_exception = traceback.format_exc()
                    logger.error(
                        "Uncaught exception in %s loop:\n%s",
                        thread_name,
                        self._last_exception,
                    )
                    # Break the loop; a watchdog may restart us.
                    break
                finally:
                    end = time.monotonic()
                    self._last_loop_duration = end - start
                    self._iteration_count += 1
                    self._update_heartbeat(end)

                # Cooperative rate limiting using loop_interval. We use
                # Event.wait() rather than time.sleep() so that stop()
                # can interrupt the wait.
                next_run += self.loop_interval
                delay = max(0.0, next_run - time.monotonic())
                if delay > 0:
                    self._stop_event.wait(delay)

        finally:
            try:
                self.on_shutdown()
            except Exception:
                logger.exception("Error in %s.on_shutdown", thread_name)
            finally:
                self._running.clear()
                logger.debug("Thread %s stopped", thread_name)

