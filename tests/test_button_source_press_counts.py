"""Presses are counted at the source, not by a slow consumer poll.

main_pedal_thread can tick at 100 ms (polling_rate 10). A tap is far shorter
than that, so watching for an edge there loses it. The HID thread and the
keyboard OS hook see every edge and count it. See core/INPUT_BINDINGS.md.
"""

from __future__ import annotations

import pytest

from core.button_device_thread.thread import ButtonDeviceThread

VID_PID = "346e:0024"
CC_INC = 26
SET = 0x04
CLEAR = 0x00
TICK_S = 0.01


class FakeHidDevice:
    def __init__(self) -> None:
        self.queue: list[list[int]] = []

    def push(self, bits: int) -> None:
        self.queue.append([0, 0, 0, bits, 0, 0, 0, 0])

    def read(self, size: int, timeout_ms: int = 0) -> list[int]:
        return self.queue.pop(0) if self.queue else []


def _make() -> tuple[ButtonDeviceThread, FakeHidDevice]:
    device = FakeHidDevice()
    thread = ButtonDeviceThread()
    thread.running = True
    thread._devices = {VID_PID: device}
    return thread, device


def _run(thread, device, taps, duration):
    """Drive the HID thread at its real 100 Hz over a tap sequence."""
    t = 0.0
    pending = list(taps)
    while t <= duration:
        while pending and pending[0][0] <= t:
            device.push(pending.pop(0)[1])
        thread._drain_reports(VID_PID, device, t)
        thread._settle_buttons(VID_PID, t)
        t += TICK_S
    return thread._press_counts(VID_PID).get(CC_INC, 0)


def test_hid_counts_a_single_tap():
    thread, device = _make()
    count = _run(thread, device, [(0.10, SET), (0.16, CLEAR)], 0.60)
    assert count == 1


@pytest.mark.parametrize("taps", [3, 8, 15])
def test_hid_counts_every_tap_in_a_burst(taps):
    thread, device = _make()
    events = []
    for i in range(taps):
        start = 0.10 + i * 0.12
        events += [(start, SET), (start + 0.05, CLEAR)]
    count = _run(thread, device, events, 0.10 + taps * 0.12 + 0.4)
    assert count == taps


def test_hid_bounce_still_counts_once():
    """The captured stalk bounce must not inflate the count."""
    thread, device = _make()
    events = [
        (0.100, SET),
        (0.1072, CLEAR),
        (0.1088, SET),
        (0.270, CLEAR),
    ]
    assert _run(thread, device, events, 0.60) == 1


def test_count_survives_a_consumer_polling_far_slower_than_the_tap():
    """polling_rate 10 means a 100 ms consumer tick; a 60 ms tap must survive."""
    thread, device = _make()
    events = [(0.10, SET), (0.16, CLEAR)]
    # The consumer only ever looks at t=0.0 and t=0.5, long after the tap ended.
    t = 0.0
    pending = list(events)
    while t <= 0.5:
        while pending and pending[0][0] <= t:
            device.push(pending.pop(0)[1])
        thread._drain_reports(VID_PID, device, t)
        thread._settle_buttons(VID_PID, t)
        t += TICK_S
    # Level is long since released, but the press is still on record.
    assert thread._settle_buttons(VID_PID, 0.5).get(CC_INC, False) is False
    assert thread._press_counts(VID_PID).get(CC_INC, 0) == 1


def test_press_counts_reset_with_the_device():
    thread, device = _make()
    _run(thread, device, [(0.10, SET), (0.16, CLEAR)], 0.40)
    assert thread._press_counts(VID_PID).get(CC_INC, 0) == 1
    thread._reset_button_state(VID_PID)
    assert thread._press_counts(VID_PID).get(CC_INC, 0) == 0
