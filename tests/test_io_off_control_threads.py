"""A stalled disk must never hold a control thread (2026-11-20 freeze, core/sending_thread/README.md).

Every check blocks the disk side on an Event and times the caller: the debug CSVs, the
settings file, the log file and the brake intensity profile read.
"""
from __future__ import annotations

import csv
import logging
import queue
import threading
import time

import pytest

import core.scs_profile.intensity as intensity
import core.sending_thread.debug_csv as debug_csv
from core.settings import Settings
from core.thread_management.log_queue import QueuedLogging

FAST_S = 0.05


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


@pytest.fixture
def stalled_disk():
    gate = threading.Event()
    yield gate
    gate.set()


def test_a_stalled_debug_csv_never_holds_the_caller_and_lands_once_the_disk_is_back(tmp_path, monkeypatch,
                                                                                   stalled_disk):
    real_open = debug_csv.open_debug_csv

    def slow_open(path, header, max_bytes=None):
        stalled_disk.wait(5.0)
        return real_open(path, header, max_bytes)

    monkeypatch.setattr(debug_csv, "open_debug_csv", slow_open)
    log = debug_csv.BrakeDebugLog(tmp_path)
    t0 = time.perf_counter()
    for i in range(40):
        log.tick(100.0 + 0.06 * i, True, {"sent_brake": 0.5})
    log.close()
    assert time.perf_counter() - t0 < FAST_S
    assert not (tmp_path / debug_csv.BRAKE_LOG_NAME).exists()

    stalled_disk.set()
    assert debug_csv.drain_debug_csv()
    rows = _rows(tmp_path / debug_csv.BRAKE_LOG_NAME)
    assert rows[0] == debug_csv.BRAKE_LOG_HEADER and len(rows) == 41


def test_a_full_writer_queue_drops_rows_instead_of_waiting(tmp_path, monkeypatch, stalled_disk):
    real_open = debug_csv.open_debug_csv
    monkeypatch.setattr(debug_csv, "open_debug_csv",
                        lambda path, header, max_bytes=None: (stalled_disk.wait(5.0), real_open(path, header))[1])
    writer = debug_csv._CsvWriter()
    writer._queue = queue.Queue(maxsize=3)
    path = tmp_path / "x.csv"
    t0 = time.perf_counter()
    for i in range(20):
        writer.submit(("rows", path, ["a"], None, [[i]]))
    assert time.perf_counter() - t0 < FAST_S
    assert writer.dropped >= 15
    stalled_disk.set()
    assert writer.drain()


def test_settings_reads_and_save_async_never_wait_for_a_stalled_write(monkeypatch, stalled_disk):
    real_write = Settings._atomic_write.__func__
    entered = threading.Event()

    def slow_write(cls, payload):
        entered.set()
        stalled_disk.wait(5.0)
        real_write(cls, payload)

    monkeypatch.setattr(Settings, "_atomic_write", classmethod(slow_write))
    before = Settings.acc_gap_level
    try:
        Settings.save_async(values={"acc_gap_level": 3 if before != 3 else 2})
        assert entered.wait(2.0), "the settings writer thread took the write"
        t0 = time.perf_counter()
        for _ in range(100):
            _ = Settings.polling_rate
        Settings.save_async(values={"usage_minutes": Settings.usage_minutes})
        assert time.perf_counter() - t0 < FAST_S
        assert Settings.disk_write_pending()
        stalled_disk.set()
        assert Settings.flush_pending()
        assert not Settings.disk_write_pending()
    finally:
        stalled_disk.set()
        Settings.save_async(values={"acc_gap_level": before})
        assert Settings.flush_pending()


def test_save_async_applies_now_and_writes_the_file(monkeypatch):
    import core.settings as settings_mod

    before = Settings.acc_gap_level
    wanted = 4 if before != 4 else 1
    try:
        Settings.save_async(values={"acc_gap_level": wanted})
        assert Settings.acc_gap_level == wanted
        assert Settings.flush_pending()
        data, err = Settings._try_read_config(settings_mod.CONFIG_PATH)
        assert err is None and data["acc_gap_level"] == wanted
        assert Settings.written_mtime_ns() == settings_mod.CONFIG_PATH.stat().st_mtime_ns
    finally:
        Settings.save_async(values={"acc_gap_level": before})
        assert Settings.flush_pending()


class _BlockingHandler(logging.Handler):
    def __init__(self, gate: threading.Event) -> None:
        super().__init__(logging.DEBUG)
        self.gate = gate
        self.lines: list[str] = []

    def emit(self, record):
        self.gate.wait(5.0)
        self.lines.append(self.format(record))


def test_logging_never_waits_for_a_stalled_log_file(stalled_disk):
    sink = _BlockingHandler(stalled_disk)
    queued = QueuedLogging([sink], max_records=10)
    queued.start()
    log = logging.getLogger("test_io_off_control_threads")
    log.propagate = False
    log.setLevel(logging.DEBUG)
    log.addHandler(queued.handler)
    try:
        t0 = time.perf_counter()
        for i in range(50):
            log.info("tick %d", i)
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("failed")
        assert time.perf_counter() - t0 < FAST_S
        assert queued.dropped > 0
        stalled_disk.set()
        assert queued.stop()
        assert sink.lines[0] == "tick 0"
        assert all(line.startswith("tick ") for line in sink.lines[:-1] if "failed" not in line)
    finally:
        log.removeHandler(queued.handler)
        queued.stop()


def test_queued_records_keep_their_message_and_traceback():
    seen: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            seen.append(record)

    queued = QueuedLogging([_Keep(logging.DEBUG)])
    queued.start()
    log = logging.getLogger("test_io_off_control_threads.trace")
    log.propagate = False
    log.addHandler(queued.handler)
    try:
        try:
            raise KeyError("lost")
        except KeyError:
            log.error("lookup %s failed", "x", exc_info=True)
        assert queued.stop()
    finally:
        log.removeHandler(queued.handler)
    text = logging.Formatter("%(message)s").format(seen[0])
    assert text.startswith("lookup x failed") and "KeyError: 'lost'" in text


def test_the_sending_threads_brake_intensity_read_never_waits_for_the_profile(monkeypatch, stalled_disk):
    class _Profile:
        g_brake_intensity = 2.0

    def slow_read(game, live_shifter_type=None):
        stalled_disk.wait(5.0)
        return _Profile()

    monkeypatch.setattr(intensity, "read_selected_profile", slow_read)
    monkeypatch.setattr(intensity, "_last_read", {})
    cache = intensity.BrakeIntensityCache(background=True)
    t0 = time.perf_counter()
    assert cache.get("ets2") == pytest.approx(intensity.DEFAULT_BRAKE_INTENSITY)
    assert time.perf_counter() - t0 < FAST_S
    stalled_disk.set()
    deadline = time.monotonic() + 2.0
    while cache._reading and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert cache._value == pytest.approx(2.0)
    assert intensity.BrakeIntensityCache(background=True).get("ets2") == pytest.approx(2.0), \
        "a restarted sending thread starts from the last value read"
