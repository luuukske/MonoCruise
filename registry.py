from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterable, List, Optional

from base_thread import BaseThread


logger = logging.getLogger("monocruise")


@dataclass(slots=True)
class Event:
    """Pub/sub event used for inter-thread signalling."""

    name: str
    payload: Any = None
    sender: Optional[str] = None
    timestamp: float = field(default_factory=time.monotonic)


ThreadFactory = Callable[["Registry"], BaseThread]


@dataclass
class ThreadSpec:
    """Registry metadata for a managed worker thread."""

    name: str
    factory: ThreadFactory
    auto_start: bool = True
    auto_restart: bool = True
    heartbeat_timeout: Optional[float] = None

    thread: Optional[BaseThread] = None
    last_start_time: float = 0.0
    last_stop_time: float = 0.0


class Registry:
    """
    Central registry for worker threads and the pub/sub event bus.

    Responsibilities:
    - Owns the authoritative mapping of thread names -> thread instances.
    - Starts/stops/restarts threads on demand.
    - Provides registry.get_thread(\"name\") for direct data access.
    - Implements a simple pub/sub bus for cross-thread events using
      per-subscriber queues.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._threads: Dict[str, ThreadSpec] = {}

        # event_name -> set(subscriber_name)
        self._subscriptions: Dict[str, set[str]] = {}
        # subscriber_name -> Queue[Event]
        self._event_queues: Dict[str, Queue[Event]] = {}

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def register_thread(
        self,
        name: str,
        factory: ThreadFactory,
        *,
        auto_start: bool = True,
        auto_restart: bool = True,
        heartbeat_timeout: Optional[float] = None,
    ) -> None:
        """Register a managed worker thread by name."""
        with self._lock:
            if name in self._threads:
                raise ValueError(f"Thread {name!r} already registered")
            self._threads[name] = ThreadSpec(
                name=name,
                factory=factory,
                auto_start=auto_start,
                auto_restart=auto_restart,
                heartbeat_timeout=heartbeat_timeout,
            )

    def start_thread(self, name: str) -> Optional[BaseThread]:
        """Start a single thread by name, creating a fresh instance."""
        with self._lock:
            spec = self._threads.get(name)
            if spec is None:
                raise KeyError(f"Unknown thread {name!r}")

            # Do not attempt to re-start a live instance.
            if spec.thread is not None and spec.thread.is_alive():
                return spec.thread

            thread = spec.factory(self)
            thread.name = name  # Ensure threadName matches registry key
            spec.thread = thread
            spec.last_start_time = time.monotonic()

            logger.info("Starting thread %s", name)
            thread.start()
            return thread

    def stop_thread(self, name: str, timeout: float = 5.0) -> None:
        """Signal a thread to stop and wait (with timeout) for it to exit."""
        with self._lock:
            spec = self._threads.get(name)
            if spec is None:
                raise KeyError(f"Unknown thread {name!r}")

            thread = spec.thread
            if thread is None:
                return

            logger.info("Stopping thread %s", name)
            thread.stop()

        # Join outside the lock to avoid deadlocks.
        thread.join(timeout)
        with self._lock:
            spec.last_stop_time = time.monotonic()

    def restart_thread(self, name: str, timeout: float = 5.0) -> Optional[BaseThread]:
        """Stop and then start a thread."""
        self.stop_thread(name, timeout=timeout)
        return self.start_thread(name)

    def start_all(self) -> None:
        """Start all registered threads that are marked auto_start=True."""
        with self._lock:
            names = [name for name, spec in self._threads.items() if spec.auto_start]
        for name in names:
            self.start_thread(name)

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all threads, best-effort."""
        with self._lock:
            names = list(self._threads.keys())
        for name in names:
            try:
                self.stop_thread(name, timeout=timeout)
            except KeyError:
                continue

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_thread(self, name: str) -> Optional[BaseThread]:
        """Return the live thread instance for a name, if any."""
        with self._lock:
            spec = self._threads.get(name)
            return spec.thread if spec else None

    def iter_specs(self) -> Iterable[ThreadSpec]:
        """Thread-safe snapshot iterator over ThreadSpec objects."""
        with self._lock:
            return list(self._threads.values())

    def status_snapshot(self) -> List[dict]:
        """
        Return a JSON-serialisable snapshot of all known threads and
        their basic runtime metrics.  Suitable for the CLI and monitor.
        """
        now = time.monotonic()
        result: List[dict] = []

        with self._lock:
            specs = list(self._threads.values())

        for spec in specs:
            t = spec.thread
            if t is None:
                state = "stopped"
                alive = False
                last_heartbeat_age = None
                last_exception = None
                iteration_count = 0
                last_loop_duration = None
            else:
                alive = t.is_alive()
                state = "running" if alive and t.is_running else "starting"
                hb = t.last_heartbeat
                last_heartbeat_age = None if hb == 0.0 else max(0.0, now - hb)
                last_exception = t.last_exception
                iteration_count = t.iteration_count
                last_loop_duration = t.last_loop_duration or None

            result.append(
                {
                    "name": spec.name,
                    "state": state,
                    "alive": alive,
                    "auto_restart": spec.auto_restart,
                    "heartbeat_timeout": spec.heartbeat_timeout,
                    "last_start_time": spec.last_start_time or None,
                    "last_stop_time": spec.last_stop_time or None,
                    "last_heartbeat_age": last_heartbeat_age,
                    "iteration_count": iteration_count,
                    "last_loop_duration": last_loop_duration,
                    "last_exception": last_exception,
                }
            )

        return result

    # ------------------------------------------------------------------
    # Pub/sub event bus
    # ------------------------------------------------------------------

    def subscribe(self, event_name: str, subscriber_name: str) -> None:
        """
        Subscribe a logical subscriber (usually a thread name) to an
        event channel.  Events are delivered via a per-subscriber queue.
        """
        with self._lock:
            subs = self._subscriptions.setdefault(event_name, set())
            subs.add(subscriber_name)
            self._event_queues.setdefault(subscriber_name, Queue())

    def unsubscribe(self, event_name: str, subscriber_name: str) -> None:
        with self._lock:
            subs = self._subscriptions.get(event_name)
            if not subs:
                return
            subs.discard(subscriber_name)
            if not subs:
                self._subscriptions.pop(event_name, None)

    def publish(
        self,
        event_name: str,
        payload: Any = None,
        *,
        sender: Optional[str] = None,
    ) -> None:
        """
        Publish an event to all subscribers.  Delivery is best-effort:
        a failure to enqueue to one subscriber does not affect others.
        """
        event = Event(name=event_name, payload=payload, sender=sender)

        with self._lock:
            subscribers = set(self._subscriptions.get(event_name, ()))
            # Wildcard subscribers can opt-in to all events by
            # subscribing to "*".
            subscribers |= set(self._subscriptions.get("*", ()))
            queues = {name: self._event_queues.get(name) for name in subscribers}

        for name, q in queues.items():
            if q is None:
                continue
            try:
                q.put_nowait(event)
            except Exception:
                logger.exception("Failed to enqueue event %r for subscriber %s", event, name)

    def poll_events(
        self,
        subscriber_name: str,
        max_events: int | None = None,
    ) -> List[Event]:
        """
        Non-blocking retrieval of queued events for a subscriber.

        Args:
            subscriber_name: Typically the thread name.
            max_events: Optional upper bound on how many events to
                return in one call.  None means "all available".
        """
        with self._lock:
            q = self._event_queues.get(subscriber_name)
            if q is None:
                # Lazily create a queue so that callers do not need an
                # explicit subscribe() just to start polling.
                q = self._event_queues.setdefault(subscriber_name, Queue())

        events: List[Event] = []
        while max_events is None or len(events) < max_events:
            try:
                events.append(q.get_nowait())
            except Empty:
                break
        return events


__all__ = ["Registry", "Event", "ThreadSpec"]

