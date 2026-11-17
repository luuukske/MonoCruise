"""Debug-only GUI stall probe: a stall is logged once it ends, with the GUI thread's place and thread CPU."""
from __future__ import annotations

import logging
import re
import threading
import time

import core.thread_management.gui_stall_probe as gsp


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
