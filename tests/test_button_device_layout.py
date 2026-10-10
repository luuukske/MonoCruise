"""HID button thread: descriptor-filtered bits, read failures, no GIL-held open.

See core/button_device_thread/README.md.
"""

from __future__ import annotations

import logging

import pytest

import core.button_device_thread.thread as bt_mod
from core.button_device_thread.hid_descriptor import parse_button_layout
from core.button_device_thread.thread import (
    _CAPTURE_CONFIRM_S,
    _RECONNECT_INTERVAL,
    _UNREADABLE_RETRY_S,
    ButtonDeviceThread,
    _pick_collection,
)
from tests.test_hid_button_layout import PEDALS, STALK

STALK_VP = "346e:0024"
PEDAL_VP = "1234:5678"
TICK_S = 0.01


class FakeHid:
    def __init__(self, reports: list[list[int]] | None = None, fail: bool = False) -> None:
        self.queue = list(reports or [])
        self.fail = fail
        self.closed = False

    def read(self, size: int, timeout_ms: int = 0) -> list[int]:
        if self.fail:
            raise OSError("read error")
        return self.queue.pop(0) if self.queue else []

    def close(self) -> None:
        self.closed = True


def _thread() -> ButtonDeviceThread:
    t = ButtonDeviceThread()
    t.running = True
    return t


def _popups(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if getattr(r, "popup", False)]


def test_tracked_pedal_axes_are_never_published():
    t = _thread()
    dev = FakeHid([[0xFF, 0x0F, 0x10, 0x00, 0x00, 0x00]])
    t._devices = {PEDAL_VP: dev}
    t._layouts = {PEDAL_VP: parse_button_layout(PEDALS)}
    t._drain_reports(PEDAL_VP, dev, 0.0)
    assert t._settle_buttons(PEDAL_VP, 0.0) == {}


def test_tracked_stalk_publishes_only_its_button_bits():
    t = _thread()
    dev = FakeHid([[0, 0, 0, 0x04, 0xFF, 0xFF, 0xFF, 0xFF]])
    t._devices = {STALK_VP: dev}
    t._layouts = {STALK_VP: parse_button_layout(STALK)}
    t._drain_reports(STALK_VP, dev, 0.0)
    states = t._settle_buttons(STALK_VP, 1.0)
    assert set(states) == set(range(32))
    assert states[26] is True
    assert not any(v for k, v in states.items() if k != 26)


def _run_capture(layout_desc: list[int] | None) -> object:
    t = _thread()
    dev = FakeHid()
    t._capture_opened = True
    t._capture_scan = {PEDAL_VP: dev}
    t._capture_scan_names = {PEDAL_VP: "Sim Pedals"}
    t._capture_scan_layouts = {
        PEDAL_VP: parse_button_layout(layout_desc) if layout_desc else None
    }
    t.data.capture_active = True
    # Rest, then the pedal pressed and held: axis bytes change and stay set.
    reports = [[0x00, 0x00, 0, 0, 0, 0]] + [[0xFF, 0x0F, 0, 0, 0, 0]] * 10
    now = 0.0
    for report in reports:
        dev.queue.append(report)
        t._tick_capture({}, now)
        now += max(TICK_S, _CAPTURE_CONFIRM_S / 3)
    return t.data.capture_event


def test_capture_takes_axis_bits_when_the_layout_is_unknown():
    """The old behaviour, kept for devices without a readable descriptor."""
    assert _run_capture(None) is not None


def test_capture_never_binds_a_pedal_axis():
    assert _run_capture(PEDALS) is None


def test_unreadable_device_warns_once_and_backs_off(monkeypatch, caplog):
    monkeypatch.setattr(bt_mod, "_hid_available", True)
    monkeypatch.setattr(ButtonDeviceThread, "_still_enumerated", staticmethod(lambda vp: True))
    t = _thread()
    t._device_names = {PEDAL_VP: "Sim Pedals"}
    caplog.set_level(logging.WARNING)

    for _ in range(3):
        dev = FakeHid(fail=True)
        t._devices = {PEDAL_VP: dev}
        t._reconnect_deadlines = {PEDAL_VP: float("inf")}
        t.loop()
        assert dev.closed and t._devices[PEDAL_VP] is None

    assert len(_popups(caplog)) == 1
    assert "cannot be read" in _popups(caplog)[0]
    wait = t._reconnect_deadlines[PEDAL_VP] - bt_mod.time.monotonic()
    assert wait > _UNREADABLE_RETRY_S - 1.0


def test_working_device_that_fails_reports_a_disconnect(monkeypatch, caplog):
    monkeypatch.setattr(bt_mod, "_hid_available", True)
    t = _thread()
    t._device_names = {STALK_VP: "MOZA Multi-function Stalk"}
    t._devices = {STALK_VP: FakeHid([[0] * 8])}
    t._reconnect_deadlines = {STALK_VP: float("inf")}
    t.loop()
    assert STALK_VP in t._read_ok

    caplog.set_level(logging.WARNING)
    t._devices[STALK_VP] = FakeHid(fail=True)
    t.loop()
    assert [m for m in _popups(caplog) if "disconnected" in m]
    wait = t._reconnect_deadlines[STALK_VP] - bt_mod.time.monotonic()
    assert wait <= _RECONNECT_INTERVAL


def test_connect_never_falls_back_to_open_by_vid_pid(monkeypatch):
    """hid.device().open(vid, pid) enumerates every device with the GIL held."""

    class _Device:
        def open(self, *args, **kwargs):
            raise AssertionError("open(vid, pid) must not be used")

    class _Hid:
        @staticmethod
        def enumerate(*args):
            return []

        device = _Device

    monkeypatch.setattr(bt_mod, "_hid_available", True)
    monkeypatch.setattr(bt_mod, "_hid", _Hid)
    t = _thread()
    assert t._try_connect_device(STALK_VP) is False
    assert t._devices[STALK_VP] is None


def test_pick_collection_prefers_the_game_controller():
    vendor = {"path": b"v", "usage_page": 0xFF00, "usage": 0x01}
    mouse = {"path": b"m", "usage_page": 0x01, "usage": 0x02}
    joystick = {"path": b"j", "usage_page": 0x01, "usage": 0x04}
    assert _pick_collection([vendor, mouse, joystick], controllers_only=False) is joystick
    assert _pick_collection([mouse, vendor], controllers_only=False) is vendor
    assert _pick_collection([vendor, mouse], controllers_only=True) is None


@pytest.mark.parametrize("guid, expected", [
    ("030007036e3400000500000000000000", "346e:0005"),  # MOZA R3, from SDL 2.28
    ("03000000000000000000000000000000", None),
    ("short", None),
])
def test_guid_vid_pid(guid, expected):
    from core.input_bindings import joystick_guid_vid_pid
    assert joystick_guid_vid_pid(guid) == expected
