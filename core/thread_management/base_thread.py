"""
BaseThread — inherit this for every worker thread.

Lifecycle:
    setup()   → called once before the loop starts
    loop()    → called repeatedly at `loop_interval` seconds
    teardown()→ called once after the loop exits (even on error)

The watchdog reads `heartbeat_at` and `restart_count` / `stable_loops`.

Threading backend:
    By default, uses ``threading.Thread``.  Pass ``use_qthread=True`` to
    use ``QThread`` instead — required for any thread that creates Qt
    objects with timers/signals (e.g. the UI thread).

    The lifecycle, watchdog integration, heartbeat, restart logic, and
    ``stop(force=True)`` behaviour are **identical** regardless of backend.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ThreadForcedStop(BaseException):
    """Raised in the target thread when stop(force=True) is used; triggers clean exit."""
    pass


# ---------------------------------------------------------------------------
# Minimal typed data container every thread exposes
# ---------------------------------------------------------------------------

@dataclass
class ThreadData:
    """Override in subclass to add typed fields."""
    pass


# ---------------------------------------------------------------------------
# Internal backend wrappers — keep the two thread primitives behind one API
# ---------------------------------------------------------------------------

class _StdThreadBackend:
    """Wraps ``threading.Thread``."""

    def __init__(self, name: str, daemon: bool, target) -> None:
        self._thread = threading.Thread(
            name=name, daemon=daemon, target=target
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def ident(self) -> int | None:
        return self._thread.ident


class _QThreadBackend:
    """
    Wraps ``PySide6.QtCore.QThread``.

    The QThread is used purely as a thread host — we override its ``run()``
    to call the provided ``target`` callable, exactly like threading.Thread.
    This means Qt's event-loop integration (timers, signals across threads)
    works correctly, while all lifecycle logic stays in BaseThread.
    """

    def __init__(self, name: str, daemon: bool, target) -> None:
        # Late import so non-UI threads never pull in PySide6
        from PySide6.QtCore import QThread

        class _Worker(QThread):
            def __init__(self, fn, thread_name: str) -> None:
                super().__init__()
                self._fn = fn
                self.setObjectName(thread_name)

            def run(self) -> None:       # noqa: D401 — Qt override
                self._fn()

        self._qthread = _Worker(target, name)
        # QThread doesn't have a "daemon" flag.  We handle cleanup in
        # BaseThread.stop() / teardown() instead.  Setting the parent to
        # None ensures it won't be prematurely garbage-collected.

    def start(self) -> None:
        self._qthread.start()

    def join(self, timeout: float | None = None) -> None:
        if timeout is not None:
            # QThread.wait() takes milliseconds (unsigned long)
            self._qthread.wait(int(timeout * 1000))
        else:
            self._qthread.wait()

    def is_alive(self) -> bool:
        return self._qthread.isRunning()

    @property
    def ident(self) -> int | None:
        """
        Return the native thread ID.
        
        ``QThread.currentThread()`` inside the running thread would give us
        this directly, but we need it from *outside*.  We grab it during
        ``run()`` by having BaseThread stash ``threading.current_thread().ident``
        — see ``BaseThread._run_lifecycle``.
        """
        # Populated by BaseThread._run_lifecycle at the start of run()
        return getattr(self, "_captured_ident", None)


# ---------------------------------------------------------------------------
# BaseThread
# ---------------------------------------------------------------------------

class BaseThread:
    """
    Base class for all MonoCruise worker threads.

    Not a subclass of ``threading.Thread`` — instead it *owns* an internal
    thread backend (_StdThreadBackend or _QThreadBackend) and delegates
    start/join/is_alive to it.  The full lifecycle (setup → loop → teardown)
    runs inside ``_run_lifecycle`` which the backend calls as its target.
    """

    # tunables — override in subclass
    loop_interval: float = 0.05          # seconds between loop() calls
    max_restarts:  int   = 1             # restarts allowed before giving up
    stable_after:  int   = 200           # loops without error → stable again
    watched:       bool  = True          # False → watchdog skips this thread

    def __init__(
        self,
        name: str,
        *,
        daemon: bool = True,
        critical_thread: bool = False,
        use_qthread: bool = False,
    ) -> None:
        self.name: str = name
        self.data: ThreadData = ThreadData()
        self._lock             = threading.Lock()
        self.critical_thread   = critical_thread
        self._use_qthread      = use_qthread

        # watchdog-visible state (primitives → GIL-safe bare reads)
        self.heartbeat_at:  float = 0.0
        self.avg_framerate: float = 0.0
        self.restart_count: int   = 0
        self.stable_loops:  int   = 0
        self.running:       bool  = False
        self.healthy:       bool  = True

        self._stop_event = threading.Event()
        self._fps_samples = 0
        self._ident: int | None = None   # captured inside the running thread

        # Create the backend
        if use_qthread:
            self._backend = _QThreadBackend(
                name=name, daemon=daemon, target=self._run_lifecycle,
            )
        else:
            self._backend = _StdThreadBackend(
                name=name, daemon=daemon, target=self._run_lifecycle,
            )

    # ------------------------------------------------------------------
    # threading.Thread-compatible API (delegated to backend)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the thread."""
        self._backend.start()

    def join(self, timeout: float | None = None) -> None:
        """Block until the thread finishes."""
        self._backend.join(timeout=timeout)

    def is_alive(self) -> bool:
        """Return True if the thread is still running."""
        return self._backend.is_alive()

    @property
    def ident(self) -> int | None:
        """Native thread ID (available after start, needed for force-stop)."""
        return self._ident

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self, force: bool = False) -> None:
        """Signal the thread to stop.

        If force=True, also inject ThreadForcedStop into the thread so it can
        exit even when stuck inside loop().
        """
        self._stop_event.set()
        if force and self._ident is not None and self.is_alive():
            try:
                tid = ctypes.c_ulong(self._ident)
                if tid.value != self._ident:
                    logger.warning(
                        "force stop: thread id %d overflows c_ulong for '%s'",
                        self._ident, self.name,
                    )
                    return
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    tid,
                    ctypes.py_object(ThreadForcedStop),
                )
                if res == 0:
                    logger.debug("force stop: invalid thread id for '%s'", self.name)
                elif res > 1:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(self._ident), None
                    )
                    logger.warning(
                        "force stop: PyThreadState_SetAsyncExc hit %d threads, reverted",
                        res,
                    )
            except (AttributeError, TypeError):
                logger.debug(
                    "force stop not available for thread '%s' (PyThreadState_SetAsyncExc failed)",
                    self.name,
                )

    def snapshot(self) -> dict[str, Any]:
        """Consistent multi-field read (acquire lock once)."""
        with self._lock:
            return {
                "name":          self.name,
                "restart_count": self.restart_count,
                "stable_loops":  self.stable_loops,
                "heartbeat_age": round(time.monotonic() - self.heartbeat_at, 3),
                "avg_framerate": round(self.avg_framerate, 2),
                "running":       self.running,
                "healthy":       self.healthy,
            }

    # ------------------------------------------------------------------
    # Override in subclass
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Called once before the loop. Raise to abort startup."""

    def loop(self) -> None:
        """Called repeatedly. Raise to trigger watchdog restart logic."""

    def teardown(self) -> None:
        """Called after loop exits; exceptions are suppressed."""

    # ------------------------------------------------------------------
    # Internal lifecycle (runs inside the backend thread)
    # ------------------------------------------------------------------

    def _run_lifecycle(self) -> None:
        """
        The full thread lifecycle.  Identical regardless of whether we're
        running on threading.Thread or QThread.
        """
        # Capture the native thread ident so force-stop and the watchdog work
        self._ident = threading.current_thread().ident
        # Also store on the QThread backend if applicable
        if hasattr(self._backend, "_captured_ident"):
            self._backend._captured_ident = self._ident

        log = logging.getLogger(self.name)
        self.running = True
        self._stop_event.clear()

        try:
            log.debug("setup")
            try:
                self.setup()
            except Exception:
                log.exception(
                    "setup() failed — thread '%s' will not start", self.name
                )
                log.critical(
                    "A component failed to start. Restart MonoCruise or contact a developer.",
                    extra={"popup": True},
                )
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
                except ThreadForcedStop:
                    raise
                except Exception as e:
                    if self.restart_count >= self.max_restarts:
                        log.critical(
                            "Thread '%s' has crashed %d time(s) and reached its restart limit.",
                            self.name, self.restart_count,
                            extra={"popup": False},
                        )
                        log.critical(
                            "A process keeps crashing. Restart MonoCruise or contact a developer.",
                            extra={"popup": True},
                        )
                        self.running = False
                        break
                    else:
                        log.exception(
                            "Unexpected error in loop — thread '%s' has stopped (restart %d/%d):\n%s",
                            self.name,
                            self.restart_count + 1,
                            self.max_restarts,
                            str(e),
                            extra={"popup": False},
                        )
                        log.error(
                            "Thread '%s' crashed.\nRestarting now...",
                            self.name,
                            extra={"popup": True},
                        )
                    self.running = False
                    break

                elapsed = time.monotonic() - t0
                cycle_duration = max(elapsed, self.loop_interval)
                if cycle_duration > 0:
                    inst_fps = 1.0 / cycle_duration
                    self._fps_samples += 1
                    self.avg_framerate += (
                        inst_fps - self.avg_framerate
                    ) / self._fps_samples
                wait = self.loop_interval - elapsed
                if wait > 0:
                    self._stop_event.wait(wait)

        except ThreadForcedStop:
            log.debug("force stop received — exiting")

        log.debug("teardown")
        try:
            self.teardown()
        except Exception:
            log.exception(
                "teardown() raised in thread '%s' (suppressed)", self.name
            )

        self.running = False
        log.debug("exited")