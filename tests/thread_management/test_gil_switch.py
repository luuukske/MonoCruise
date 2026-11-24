"""gil_switch_ms sets Python's lock switch interval; an unusable value keeps the interval it had."""
from __future__ import annotations

import logging
import sys
import time

import pytest

import core.thread_management.gui_stall_probe as gsp
from core.settings import Settings
from core.thread_management.gil_switch import apply_gil_switch


@pytest.fixture(autouse=True)
def _restore_interval():
    before = sys.getswitchinterval()
    yield
    sys.setswitchinterval(before)


def test_the_default_is_one_millisecond():
    assert Settings.gil_switch_ms == 1.0


def test_a_usable_value_sets_the_interval():
    assert apply_gil_switch(2.0)
    assert sys.getswitchinterval() == pytest.approx(0.002)


@pytest.mark.parametrize("bad", [None, "fast", 0.0, -1.0, 100.0, float("nan")])
def test_an_unusable_value_keeps_the_interval_it_had(bad, caplog):
    sys.setswitchinterval(0.005)
    caplog.set_level(logging.WARNING, logger="main")
    assert not apply_gil_switch(bad)
    assert sys.getswitchinterval() == pytest.approx(0.005)
    assert "unusable" in caplog.text


def test_the_probe_logs_the_timer_once(caplog):
    probe = gsp.GuiStallProbe(None)
    probe.running = True
    probe.setup()
    caplog.set_level(logging.INFO, logger="gui_stall")
    for _ in range(2):
        probe._report_at = 0.0
        probe.loop()
    lines = [r.getMessage() for r in caplog.records if "timed wait lasts" in r.getMessage()]
    assert len(lines) == 1 and "GIL switch interval" in lines[0]


def test_a_timed_wait_lasts_at_least_its_timeout():
    t0 = time.perf_counter()
    assert gsp.timed_wait_ms(samples=3) >= 0.9
    assert time.perf_counter() - t0 < 0.5, "never blocks the probe for long"
