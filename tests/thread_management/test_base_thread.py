"""BaseThread lifecycle: heartbeat, crash handling, restart quota, teardown guarantees.

`_run_lifecycle` is driven directly on the test thread rather than started for
real, so these tests are deterministic and cannot hang the suite.
"""
from __future__ import annotations

import pytest

from core.thread_management.base_thread import BaseThread


class _Worker(BaseThread):
    """Runs `loops` iterations then stops itself."""

    loop_interval = 0.0
    stable_after = 1000

    def __init__(self, loops: int = 3, **behaviour):
        super().__init__(name="probe_thread")
        self.calls: list[str] = []
        self._loops_left = loops
        self._raise_in_setup = behaviour.get("raise_in_setup", False)
        self._raise_in_loop = behaviour.get("raise_in_loop", False)
        self._raise_in_teardown = behaviour.get("raise_in_teardown", False)

    def setup(self) -> None:
        self.calls.append("setup")
        if self._raise_in_setup:
            raise RuntimeError("setup boom")

    def loop(self) -> None:
        self.calls.append("loop")
        if self._raise_in_loop:
            raise RuntimeError("loop boom")
        self._loops_left -= 1
        if self._loops_left <= 0:
            self.stop()

    def teardown(self) -> None:
        self.calls.append("teardown")
        if self._raise_in_teardown:
            raise RuntimeError("teardown boom")


def test_lifecycle_order_and_clean_exit():
    worker = _Worker(loops=3)
    worker._run_lifecycle()
    assert worker.calls == ["setup", "loop", "loop", "loop", "teardown"]
    assert worker.running is False
    assert worker.healthy is True


def test_heartbeat_advances_with_every_loop():
    worker = _Worker(loops=2)
    assert worker.heartbeat_at == 0.0
    worker._run_lifecycle()
    assert worker.heartbeat_at > 0.0


def test_a_failed_setup_never_enters_the_loop():
    worker = _Worker(raise_in_setup=True)
    worker._run_lifecycle()
    assert "loop" not in worker.calls
    assert worker.running is False
    assert worker.healthy is False


def test_a_crashing_loop_exits_without_propagating(caplog):
    """The watchdog restarts it; the exception must not escape the thread."""
    worker = _Worker(raise_in_loop=True)
    with caplog.at_level("ERROR"):
        worker._run_lifecycle()
    assert worker.calls == ["setup", "loop", "teardown"]
    assert worker.running is False
    assert "crashed" in caplog.text


def test_a_crash_at_the_restart_ceiling_still_tears_down():
    worker = _Worker(raise_in_loop=True)
    worker.restart_count = worker.max_restarts
    worker._run_lifecycle()
    assert "teardown" in worker.calls
    assert worker.running is False


def test_a_raising_teardown_is_suppressed():
    worker = _Worker(loops=1, raise_in_teardown=True)
    worker._run_lifecycle()  # must not raise
    assert worker.running is False


def test_stable_running_resets_the_restart_quota():
    """A thread that has recovered must not carry its old crash count forever."""
    worker = _Worker(loops=6)
    worker.stable_after = 3
    worker.restart_count = 2
    worker._run_lifecycle()
    assert worker.restart_count == 0


def test_a_stale_stop_event_is_cleared_on_entry():
    """A restarted worker reuses the object, so a leftover stop must not kill it."""
    worker = _Worker(loops=2)
    worker.stop()
    worker._run_lifecycle()
    assert worker.calls == ["setup", "loop", "loop", "teardown"]


def test_fps_tracking_is_populated():
    """loop_interval must be nonzero: a 0 interval leaves inst_fps at 0 on Windows,
    where the monotonic clock is too coarse to measure a fast loop."""
    worker = _Worker(loops=3)
    worker.loop_interval = 0.01
    worker._run_lifecycle()
    assert 0.0 < worker.inst_fps <= 100.0
    assert worker.avg_framerate > 0.0


def test_snapshot_reports_watchdog_visible_state():
    worker = _Worker(loops=1)
    worker._run_lifecycle()
    snap = worker.snapshot()
    assert snap["name"] == "probe_thread"
    assert snap["running"] is False
    assert snap["healthy"] is True
    assert snap["heartbeat_age"] >= 0.0


def test_safe_state_is_a_no_op_by_default():
    """The watchdog calls this before swapping a dead thread out."""
    assert BaseThread.safe_state() is None
