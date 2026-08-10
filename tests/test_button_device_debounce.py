"""Contact bounce on a HID button must not become extra presses.

Replays the report stream captured from a MOZA Multi-function Stalk, where a
single physical press bounces for a few milliseconds. See
core/button_device_thread/README.md.
"""

from __future__ import annotations

import pytest

from core.button_device_thread.thread import (
    _DEBOUNCE_S,
    ButtonDeviceThread,
)

VID_PID = "346e:0024"
CC_INC = 26  # byte 3, bit 2
TICK_S = 0.01


class FakeHidDevice:
    """Queues reports and hands them out one read() at a time, like hidapi."""

    def __init__(self) -> None:
        self.queue: list[list[int]] = []

    def push(self, button_bits: int) -> None:
        self.queue.append([0, 0, 0, button_bits, 0, 0, 0, 0])

    def read(self, size: int, timeout_ms: int = 0) -> list[int]:
        return self.queue.pop(0) if self.queue else []


def _make_thread(device: FakeHidDevice) -> ButtonDeviceThread:
    thread = ButtonDeviceThread()
    thread.running = True
    thread._devices = {VID_PID: device}
    return thread


def _run(thread: ButtonDeviceThread, device: FakeHidDevice, events, duration: float):
    """Feed timestamped reports and sample the settled state every tick."""
    samples: list[tuple[float, bool]] = []
    pending = list(events)
    now = 0.0
    while now <= duration:
        while pending and pending[0][0] <= now:
            device.push(pending.pop(0)[1])
        thread._drain_reports(VID_PID, device, now)
        states = thread._settle_buttons(VID_PID, now)
        samples.append((now, states.get(CC_INC, False)))
        now += TICK_S
    return samples


def _count_presses(samples) -> int:
    """Rising edges, matching what the cruise control button FSM reacts to."""
    presses = 0
    prev = False
    for _, held in samples:
        if held and not prev:
            presses += 1
        prev = held
    return presses


SET = 0x04  # bit 2 of byte 3
CLEAR = 0x00


def test_clean_press_registers_once():
    device = FakeHidDevice()
    thread = _make_thread(device)
    events = [(0.10, SET), (0.20, CLEAR)]
    samples = _run(thread, device, events, 0.40)
    assert _count_presses(samples) == 1


def test_bouncing_press_registers_once():
    """Captured stalk bounce: 7.2 ms release then 1.6 ms re-press mid-edge."""
    device = FakeHidDevice()
    thread = _make_thread(device)
    events = [
        (0.100, SET),
        (0.1072, CLEAR),
        (0.1088, SET),
        (0.270, CLEAR),
    ]
    samples = _run(thread, device, events, 0.50)
    assert _count_presses(samples) == 1


def test_bounce_on_release_registers_once():
    device = FakeHidDevice()
    thread = _make_thread(device)
    events = [
        (0.100, SET),
        (0.200, CLEAR),
        (0.2015, SET),
        (0.2045, CLEAR),
    ]
    samples = _run(thread, device, events, 0.50)
    assert _count_presses(samples) == 1


def test_two_deliberate_presses_still_register_twice():
    device = FakeHidDevice()
    thread = _make_thread(device)
    events = [
        (0.10, SET), (0.19, CLEAR),
        (0.30, SET), (0.39, CLEAR),
    ]
    samples = _run(thread, device, events, 0.60)
    assert _count_presses(samples) == 2


def test_burst_of_queued_reports_does_not_stretch_a_glitch():
    """The old reader consumed one report per tick, stretching a 1.6 ms glitch."""
    device = FakeHidDevice()
    thread = _make_thread(device)
    for bits in (SET, CLEAR, SET):
        device.push(bits)
    thread._drain_reports(VID_PID, device, 0.0)
    assert device.queue == [], "every queued report must be consumed in one tick"
    assert thread._raw_bits[VID_PID][CC_INC] is True


def test_press_shorter_than_debounce_is_rejected():
    device = FakeHidDevice()
    thread = _make_thread(device)
    events = [(0.10, SET), (0.10 + _DEBOUNCE_S / 2, CLEAR)]
    samples = _run(thread, device, events, 0.40)
    assert _count_presses(samples) == 0


def test_reconnect_does_not_inherit_a_held_button():
    device = FakeHidDevice()
    thread = _make_thread(device)
    _run(thread, device, [(0.05, SET)], 0.20)
    assert thread._settle_buttons(VID_PID, 0.20)[CC_INC] is True

    thread._reset_button_state(VID_PID)
    assert thread._settle_buttons(VID_PID, 0.30).get(CC_INC, False) is False


@pytest.mark.parametrize("offset_ms", range(0, 10))
def test_bounce_filtered_regardless_of_tick_alignment(offset_ms):
    """A 7 ms glitch must not survive whichever tick boundary it straddles."""
    device = FakeHidDevice()
    thread = _make_thread(device)
    base = 0.100 + offset_ms / 1000.0
    events = [
        (base, SET),
        (base + 0.0072, CLEAR),
        (base + 0.0088, SET),
        (base + 0.200, CLEAR),
    ]
    samples = _run(thread, device, events, 0.50)
    assert _count_presses(samples) == 1
