"""Joystick opening without a long stall, and finding a device whose GUID changed.

pygame's Joystick() blocks every thread for the whole first open (75 to 150 ms
for a DirectInput device), and SDL GUIDs embed a CRC of the device name plus
its version. See core/main_pedal_thread/README.md.
"""

from __future__ import annotations

from core.main_pedal_thread.joystick_pool import JoystickPool, pick_joystick

R3 = "030007036e3400000500000000000000"
R3_RENAMED = "0300aaaa6e3400000500000001000000"
STALK = "030011116e3400002400000000000000"


class FakeJs:
    def __init__(self, guid: str, iid: int, axes: int = 8, name: str = "dev") -> None:
        self.guid, self.iid, self.axes, self.name = guid, iid, axes, name

    def init(self) -> None:
        pass

    def get_guid(self) -> str:
        return self.guid

    def get_instance_id(self) -> int:
        return self.iid

    def get_name(self) -> str:
        return self.name

    def get_numaxes(self) -> int:
        return self.axes


class Rig:
    """Connected devices plus a clock that a first open advances by 130 ms."""

    def __init__(self, devices: list[FakeJs]) -> None:
        self.devices = devices
        self.now = 0.0
        self.opened: set[int] = set()
        self.slow_opens = 0

    def open(self, index: int) -> FakeJs:
        js = self.devices[index]
        if isinstance(js, Exception):
            raise js
        if js.iid not in self.opened:
            self.opened.add(js.iid)
            self.slow_opens += 1
            self.now += 0.13
        return js

    def pool(self) -> JoystickPool:
        return JoystickPool(lambda: len(self.devices), self.open, clock=lambda: self.now)


def test_one_slow_open_per_tick():
    rig = Rig([FakeJs(f"g{i}", i) for i in range(4)])
    pool = rig.pool()
    for tick in range(1, 4):
        assert pool.step() is False
        assert rig.slow_opens == tick
    assert pool.step() is True
    assert set(pool.devices) == {"g0", "g1", "g2", "g3"}


def test_already_open_devices_cost_nothing_and_finish_in_one_tick():
    rig = Rig([FakeJs(f"g{i}", i) for i in range(4)])
    rig.opened = {0, 1, 2, 3}
    assert rig.pool().step() is True


def test_restart_and_hot_plug_rewalk_without_reopening():
    rig = Rig([FakeJs("g0", 0), FakeJs("g1", 1)])
    pool = rig.pool()
    while not pool.step():
        pass
    pool.restart()
    assert pool.step() is True and rig.slow_opens == 2

    rig.devices.append(FakeJs("g2", 2))
    assert pool.step() is True  # g0, g1 free, then the one slow open
    assert "g2" in pool.devices and rig.slow_opens == 3


def test_open_failures_are_recorded_and_removal_forgets_the_device():
    rig = Rig([FakeJs("g0", 0), OSError("Acquire failed")])
    pool = rig.pool()
    while not pool.step():
        pass
    assert any("Acquire failed" in e for e in pool.errors)
    pool.remove_instance(0)
    assert pool.devices == {}


def test_buttons_only_devices_are_left_to_the_hid_thread():
    stalk = FakeJs(STALK, 1, axes=0, name="MOZA Multi-function Stalk")
    rig = Rig([FakeJs("g0", 0), stalk])
    rig.opened = {0, 1}
    pool = rig.pool()
    assert pool.step() is True
    assert set(pool.devices) == {"g0"}
    assert 1 not in pool.guid_by_instance
    assert pool.hid_only == ["MOZA Multi-function Stalk"]


def test_axes_only_devices_such_as_pedals_stay_listed():
    pedals = FakeJs("pedals", 0, axes=3, name="CRP2 Pedals")
    pool = Rig([pedals]).pool()
    assert pool.step() is True
    assert pool.devices == {"pedals": pedals} and pool.hid_only == []


def test_exact_guid_wins_and_stops_opening():
    seen: list[str] = []

    def candidates():
        for guid in (R3, STALK):
            seen.append(guid)
            yield guid, guid

    assert pick_joystick(candidates(), R3) == (R3, False)
    assert seen == [R3]


def test_renamed_device_is_found_by_vid_pid():
    renamed = FakeJs(R3_RENAMED, 0)
    assert pick_joystick([(STALK, None), (R3_RENAMED, renamed)], R3) == (renamed, True)


def test_vid_pid_fallback_refuses_ambiguity_and_too_few_axes():
    a, b = FakeJs(R3_RENAMED, 0), FakeJs(R3_RENAMED, 1)
    assert pick_joystick([(R3_RENAMED, a), (R3_RENAMED, b)], R3) == (None, False)

    small = FakeJs(R3_RENAMED, 0, axes=2)
    assert pick_joystick(
        [(R3_RENAMED, small)], R3, accept=lambda js: js.get_numaxes() >= 3,
    ) == (None, False)


def test_an_unrelated_device_is_never_picked():
    assert pick_joystick([(STALK, FakeJs(STALK, 0))], R3) == (None, False)
    assert pick_joystick([(R3, FakeJs(R3, 0))], "") == (None, False)
