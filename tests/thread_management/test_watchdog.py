"""Watchdog restart policy: crash, freeze debounce, restart ceiling, low-FPS warning.

The watchdog is the crash-resilience layer itself, so it is driven here through
`Watchdog.loop()` against fake registry entries. No real threads are started.
"""
from __future__ import annotations

import threading
import types

import pytest

import core.thread_management.watchdog as wd_mod
from core.thread_management.registry import registry
from core.thread_management.watchdog import (
    FREEZE_STREAK_RESTART,
    HEARTBEAT_TIMEOUT,
    LOW_FPS_STREAK_WARN,
    LOW_FPS_WARN_WINDOW,
    Watchdog,
)


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeWorker:
    """Stands in for a registered BaseThread as far as the watchdog can see."""

    def __init__(self, name: str, clock: _Clock, **overrides):
        self.name = name
        self._clock = clock
        self.watched = True
        self.healthy = True
        self.loop_interval = 0.0
        self.inst_fps = 0.0
        self.restart_count = 0
        self.max_restarts = 3
        self.running = True
        self.heartbeat_at = clock.monotonic()
        self._alive = True
        self.data = types.SimpleNamespace(request_quit=False)
        self.stopped_forcefully = False
        self.started = False
        self.__dict__.update(overrides)

    def is_alive(self) -> bool:
        return self._alive

    def stop(self, force: bool = False) -> None:
        self.stopped_forcefully = self.stopped_forcefully or force
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        pass

    def start(self) -> None:
        self.started = True

    @staticmethod
    def safe_state() -> None:
        pass

    def beat(self) -> None:
        self.heartbeat_at = self._clock.monotonic()

    def crash(self) -> None:
        self.running = False
        self._alive = False


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(wd_mod, "time", types.SimpleNamespace(monotonic=c.monotonic))
    return c


@pytest.fixture
def rig(clock):
    """Watchdog plus one registered worker, with a factory that records replacements."""
    dog = Watchdog()
    worker = _FakeWorker("worker_thread", clock)
    registry.replace(worker)

    made: list[_FakeWorker] = []

    def factory():
        new = _FakeWorker("worker_thread", clock)
        made.append(new)
        return new

    dog.register_factory("worker_thread", factory)
    yield dog, worker, made, clock
    registry.unregister("worker_thread")


def test_healthy_worker_is_left_alone(rig):
    dog, worker, made, _ = rig
    for _ in range(10):
        dog.loop()
    assert made == []
    assert worker.healthy is True


def test_crashed_worker_is_restarted_and_swapped_into_the_registry(rig):
    dog, worker, made, _ = rig
    worker.crash()
    dog.loop()

    assert len(made) == 1
    replacement = made[0]
    assert replacement.restart_count == worker.restart_count + 1
    assert replacement.started is True
    assert registry.get_thread("worker_thread") is replacement


def test_freeze_needs_a_streak_before_a_restart(rig):
    """One stale poll is a hiccup; FREEZE_STREAK_RESTART in a row is a freeze."""
    dog, worker, made, clock = rig
    clock.advance(HEARTBEAT_TIMEOUT + 0.1)

    for _ in range(FREEZE_STREAK_RESTART - 1):
        dog.loop()
        assert made == [], "restarted before the debounce streak was reached"

    dog.loop()
    assert len(made) == 1


def test_a_recovered_heartbeat_resets_the_freeze_streak(rig):
    dog, worker, made, clock = rig
    clock.advance(HEARTBEAT_TIMEOUT + 0.1)
    for _ in range(FREEZE_STREAK_RESTART - 1):
        dog.loop()

    worker.beat()
    dog.loop()
    clock.advance(HEARTBEAT_TIMEOUT + 0.1)
    for _ in range(FREEZE_STREAK_RESTART - 1):
        dog.loop()
    assert made == [], "streak survived a heartbeat"

    dog.loop()
    assert len(made) == 1


def test_restart_ceiling_stops_the_loop_instead_of_restarting_forever(rig):
    dog, worker, made, _ = rig
    worker.restart_count = worker.max_restarts
    worker.crash()
    dog.loop()

    assert made == []
    assert worker.healthy is False


def test_missing_factory_marks_unhealthy_without_raising(clock):
    dog = Watchdog()
    worker = _FakeWorker("orphan_thread", clock)
    registry.replace(worker)
    try:
        worker.crash()
        dog.loop()
        assert worker.healthy is False
        assert registry.get_thread("orphan_thread") is worker
    finally:
        registry.unregister("orphan_thread")


def test_unwatched_threads_are_skipped(rig):
    dog, worker, made, _ = rig
    worker.watched = False
    worker.crash()
    dog.loop()
    assert made == []
    assert worker.healthy is True


def test_telemetry_quit_request_is_not_a_restart(clock):
    """A requested shutdown is main's business; restarting it would fight the user."""
    dog = Watchdog()
    worker = _FakeWorker("telemetry_thread", clock)
    worker.data.request_quit = True
    registry.replace(worker)
    made = []
    dog.register_factory("telemetry_thread", lambda: made.append(1))
    try:
        worker.crash()
        dog.loop()
        assert made == []
        assert worker.healthy is False
    finally:
        registry.unregister("telemetry_thread")


def test_low_fps_warns_only_after_the_streak_and_the_window(rig, caplog):
    dog, worker, _, clock = rig
    worker.loop_interval = 1 / 100.0   # wants 100 fps
    worker.inst_fps = 50.0             # delivers half

    with caplog.at_level("WARNING"):
        for _ in range(LOW_FPS_STREAK_WARN):
            worker.beat()
            dog.loop()
        assert "target FPS" not in caplog.text, "warned before the collection window elapsed"

        clock.advance(LOW_FPS_WARN_WINDOW + 0.1)
        worker.beat()
        dog.loop()
        assert "target FPS" in caplog.text


def test_recovered_fps_cancels_a_pending_low_fps_warning(rig, caplog):
    dog, worker, _, clock = rig
    worker.loop_interval = 1 / 100.0
    worker.inst_fps = 50.0

    with caplog.at_level("WARNING"):
        for _ in range(LOW_FPS_STREAK_WARN):
            worker.beat()
            dog.loop()

        worker.inst_fps = 100.0
        worker.beat()
        dog.loop()

        clock.advance(LOW_FPS_WARN_WINDOW + 0.1)
        worker.beat()
        dog.loop()
        assert "target FPS" not in caplog.text


def _parked_in_a_known_place(release: threading.Event) -> None:
    release.wait(10.0)


@pytest.fixture
def parked_thread():
    release = threading.Event()
    t = threading.Thread(target=_parked_in_a_known_place, args=(release,), daemon=True)
    t.start()
    yield t
    release.set()
    t.join(2.0)


def _freeze(dog, clock):
    clock.advance(HEARTBEAT_TIMEOUT + 0.1)
    for _ in range(FREEZE_STREAK_RESTART):
        dog.loop()


def test_a_freeze_logs_where_the_thread_is_stuck(rig, caplog, parked_thread):
    """A freeze leaves no trace of its own; the stack names the blocking call."""
    dog, worker, made, clock = rig
    worker.ident = parked_thread.ident
    with caplog.at_level("WARNING", logger="watchdog"):
        _freeze(dog, clock)

    assert len(made) == 1
    stack = [r for r in caplog.records if "frozen at" in r.getMessage()]
    assert len(stack) == 1
    msg = stack[0].getMessage()
    assert "tests/thread_management/test_watchdog.py" in msg
    assert "_parked_in_a_known_place" in msg
    assert "release.wait(10.0)" in msg
    assert str(wd_mod._REPO_ROOT) not in msg and "\\" not in msg   # no absolute paths
    assert not getattr(stack[0], "popup", False)


def test_the_stack_is_logged_before_the_restart_ceiling_too(rig, caplog, parked_thread):
    dog, worker, made, clock = rig
    worker.ident = parked_thread.ident
    worker.restart_count = worker.max_restarts
    with caplog.at_level("WARNING", logger="watchdog"):
        _freeze(dog, clock)
    assert made == []
    assert "_parked_in_a_known_place" in caplog.text


def test_no_stack_for_a_crash_or_an_unknown_thread(rig, caplog):
    dog, worker, made, clock = rig
    with caplog.at_level("WARNING", logger="watchdog"):
        _freeze(dog, clock)               # no ident at all
        assert len(made) == 1
        made[0].crash()
        made[0].ident = threading.get_ident()
        dog.loop()                        # crashed, not frozen
    assert "frozen at" not in caplog.text


def test_frozen_stack_unknown_ident_is_none():
    assert wd_mod.frozen_stack(None) is None
    assert wd_mod.frozen_stack(-12345) is None


def test_paths_outside_the_repo_keep_only_their_tail():
    outside = wd_mod._REPO_ROOT.parent.parent / "someone" / "Lib" / "threading.py"
    assert wd_mod._short_path(str(outside)) == "Lib/threading.py"
    inside = wd_mod._REPO_ROOT / "core" / "sending_thread" / "thread.py"
    assert wd_mod._short_path(str(inside)) == "core/sending_thread/thread.py"
    # Frozen builds carry relative code paths; they pass through as-is.
    assert wd_mod._short_path("core/sending_thread/thread.py") == "core/sending_thread/thread.py"
