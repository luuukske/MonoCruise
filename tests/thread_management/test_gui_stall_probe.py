"""Debug-only GUI stall probe: a stall is logged once it ends, with the GUI thread's place and thread CPU."""
from __future__ import annotations

import logging
import re
import threading
import time
from types import SimpleNamespace

import pytest

import core.thread_management.gui_stall_probe as gsp
from core.thread_management.base_thread import BaseThread, LoopTimes
from core.thread_management.registry import registry
from tools import lag_report


@pytest.fixture(autouse=True)
def _loop_times_off():
    yield
    LoopTimes.on = False


@pytest.fixture
def fake_threads():
    """Registers stand-ins under worker names and puts the registry back afterwards."""
    saved: dict[str, object] = {}

    def put(thread) -> None:
        try:
            saved.setdefault(thread.name, registry.get_thread(thread.name))
        except KeyError:
            saved.setdefault(thread.name, None)
        registry.replace(thread)

    yield put
    for name, old in saved.items():
        registry.unregister(name)
        if old is not None:
            registry.replace(old)


def _radar(**kw) -> SimpleNamespace:
    data = dict(
        _lock=threading.Lock(), t_mono=1.0, ego_x=12344.0, ego_z=-5678.0, ego_speed=15.0,
        tmp_session=True, paused=False, vehicles=[1, 2, 3], trailer_vehicles=[1],
    )
    data.update(kw)
    return SimpleNamespace(name="radar_thread", data=SimpleNamespace(**data))


def _map() -> SimpleNamespace:
    snap = SimpleNamespace(status="ready", window=SimpleNamespace(lines=(1, 2)), target_routes={7: None})
    return SimpleNamespace(name="map_thread", data=SimpleNamespace(_lock=threading.Lock(), snapshot=snap))


def _worker(name: str, interval: float, samples: list[float]) -> SimpleNamespace:
    times = LoopTimes()
    for s in samples:
        times.add(s)
    return SimpleNamespace(name=name, loop_interval=interval, loop_times=times, ident=None)


def _probe() -> gsp.GuiStallProbe:
    probe = gsp.GuiStallProbe(threading.get_ident())
    probe.running = True
    probe.setup()
    return probe


def _busy_paint(until: float) -> None:
    while time.monotonic() < until:
        pass


def test_a_stall_is_logged_when_the_gui_beats_again(caplog):
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    probe = _probe()
    gui = threading.Thread(target=_busy_paint, args=(time.monotonic() + 0.5,))
    gui.start()
    probe._gui_ident = gui.ident
    probe._beat = time.monotonic() - 0.3
    probe.loop()
    probe.loop()
    gui.join()
    assert not [r for r in caplog.records if "GUI stalled" in r.getMessage()]
    probe.beat()
    probe.loop()
    lines = [r.getMessage() for r in caplog.records if "GUI stalled" in r.getMessage()]
    assert len(lines) == 1
    assert re.search(r"GUI stalled \d+ ms", lines[0])
    assert "_busy_paint" in lines[0], "where the GUI thread was is named"
    assert "CPU" in lines[0]
    assert probe.data.stalls == 1 and probe.data.worst_ms >= 300.0


def test_its_stack_lines_carry_no_absolute_paths(caplog):
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    probe._beat = time.monotonic() - 0.6
    probe.loop()
    probe.beat()
    probe.loop()
    for record in caplog.records:
        assert not re.search(r"[A-Za-z]:[\\/]", record.getMessage())


def test_short_gaps_are_not_stalls(caplog):
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    for _ in range(3):
        probe.beat()
        probe.loop()
    assert probe.data.stalls == 0
    assert not [r for r in caplog.records if "GUI stalled" in r.getMessage()]


def test_stalls_closer_than_the_log_gap_are_counted_not_logged(caplog):
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    for _ in range(2):
        probe._beat = time.monotonic() - 0.4
        probe.loop()
        probe.beat()
        probe.loop()
    assert probe.data.stalls == 2
    assert len([r for r in caplog.records if "GUI stalled" in r.getMessage()]) == 1


def test_a_broken_cpu_read_never_stops_the_probe(caplog):
    probe = _probe()

    def broken():
        raise OSError("access denied")

    probe._cpu_reader = broken
    probe._beat = time.monotonic() - 0.6
    probe.loop()
    probe.beat()
    probe.loop()
    probe._report_at = 0.0
    probe.loop()
    assert probe.data.stalls == 1


def test_the_periodic_report_names_the_busiest_threads(caplog):
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    end = time.process_time() + 0.05
    while time.process_time() < end:
        pass
    probe._report_at = 0.0
    probe.loop()
    lines = [r.getMessage() for r in caplog.records if "GUI longest gap" in r.getMessage()]
    assert len(lines) == 1 and "ms (" in lines[0]


def _messages(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def _stall_once(probe: gsp.GuiStallProbe) -> None:
    probe._beat = time.monotonic() - 0.6
    probe.loop()
    probe.beat()
    probe.loop()


def test_loop_times_are_recorded_only_while_the_probe_is_on():
    class Busy(BaseThread):
        loop_interval = 0.005

        def loop(self) -> None:
            end = time.perf_counter() + 0.002
            while time.perf_counter() < end:
                pass

    worker = Busy(name="loop_times_test")
    worker.start()
    try:
        time.sleep(0.1)
        assert worker.loop_times.take() == [], "nothing is recorded with the probe off"
        LoopTimes.on = True
        time.sleep(0.1)
        times = worker.loop_times.take()
    finally:
        worker.stop()
        worker.join(1.0)
    assert len(times) >= 5
    assert min(times) >= 0.0019, "loop wall time comes from a precise clock, not a 15.6 ms tick"


def test_the_probe_turns_loop_timing_on_and_off():
    probe = _probe()
    assert LoopTimes.on
    probe.teardown()
    assert not LoopTimes.on


def test_the_loop_line_names_each_worker_against_its_budget(caplog, fake_threads):
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    fake_threads(_worker("fast_worker", 0.01, [0.002] * 1000 + [0.05]))
    fake_threads(_worker("late_worker", 0.05, [0.08] * 100))
    probe._report_t0 = time.monotonic() - gsp.REPORT_S
    probe._report_at = 0.0
    probe.loop()
    lines = _messages(caplog, "Loops in the last")
    assert len(lines) == 1
    assert "fast_worker 100/100 Hz, p50 2.0 p99 2.0 max 50 ms of 10" in lines[0]
    assert "late_worker 10/20 Hz, p50 80.0 p99 80.0 max 80 ms of 50" in lines[0]
    assert lines[0].index("late_worker") < lines[0].index("fast_worker"), "worst share of the budget first"


def test_reports_and_stalls_say_where_the_truck_was(caplog, fake_threads):
    fake_threads(_radar())
    fake_threads(_map())
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    _stall_once(probe)
    probe._report_at = 0.0
    probe.loop()
    place = "At x 12340 z -5680, 54 km/h, TMP, 3 vehicles + 1 trailers, map ready 2 lanes 1 routed"
    assert place in _messages(caplog, "GUI stalled")[0]
    assert place in _messages(caplog, "GUI longest gap")[0]


def test_a_report_keeps_the_most_traffic_seen_in_its_window(fake_threads):
    radar = _radar()
    fake_threads(radar)
    probe = _probe()
    probe._sample_context(time.monotonic())
    radar.data.vehicles = [1]
    probe._sample_context(time.monotonic())
    assert probe._context.vehicles == 3


def test_missing_radar_and_map_leave_the_place_unknown(caplog, fake_threads):
    fake_threads(SimpleNamespace(name="radar_thread", data=None))
    fake_threads(SimpleNamespace(name="map_thread", data=None))
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    probe._report_at = 0.0
    probe.loop()
    line = _messages(caplog, "GUI longest gap")[0]
    assert "At position unknown, 0 vehicles + 0 trailers, map unknown 0 lanes 0 routed" in line


def test_a_radar_without_a_frame_yet_has_no_place(fake_threads):
    fake_threads(_radar(t_mono=0.0))
    probe = _probe()
    probe._sample_context(time.monotonic())
    assert probe._context.x is None and probe._context.vehicles == 3


def test_the_lag_report_reads_what_the_probe_writes(caplog, fake_threads):
    fake_threads(_radar())
    fake_threads(_map())
    fake_threads(_worker("aeb_thread", 1 / 30, [0.010] * 300))
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    _stall_once(probe)
    probe._report_t0 = time.monotonic() - gsp.REPORT_S
    probe._report_at = 0.0
    probe.loop()
    lines = [f"12:00:00 [gui_stall   ] {r.levelname:<8} {r.getMessage()}" for r in caplog.records]
    windows, stalls = lag_report.parse(lines)
    assert len(windows) == 1 and len(stalls) == 1
    w = windows[0]
    assert (w.ctx.x, w.ctx.z, w.ctx.tmp, w.ctx.vehicles, w.ctx.lanes, w.ctx.routed) == (12340, -5680, True, 3, 2, 1)
    loop = w.loops["aeb_thread"]
    assert (loop.hz, loop.target_hz, loop.p99_ms, loop.budget_ms) == (30, 30, 10.0, 33)
    assert stalls[0].ctx.x == 12340
    assert stalls[0].cpu_share is not None, "a line with the From prefix is classified"


def test_stall_cpu_is_shared_over_the_time_it_was_read_not_the_whole_stall(caplog):
    probe = _probe()
    caplog.set_level(logging.DEBUG, logger="gui_stall")
    reads = iter([{1: 0.0}, {1: 0.040}])
    probe._cpu_reader = lambda: next(reads)
    probe._beat = time.monotonic() - 0.6
    probe.loop()
    time.sleep(0.05)
    probe.beat()
    probe.loop()
    line = _messages(caplog, "GUI stalled")[0]
    seen, cpu, over = map(int, re.search(r"From (\d+) ms in: CPU (\d+) ms over (\d+) ms", line).groups())
    assert seen >= 590 and cpu == 40
    assert 45 <= over < 300, "the share's span is the time between the reads, not the 650 ms stall"
