"""Opening joysticks without stalling the program, and finding one by GUID. See README.md."""

from __future__ import annotations

import time
from typing import Callable, Iterable

from core.input_bindings import joystick_guid_vid_pid

# A first DirectInput open blocks every thread for 75 to 150 ms; reopening an
# open device is free. Past this much time in a tick, no further opens start.
_OPEN_BUDGET_S = 0.02


def pick_joystick(
    candidates: Iterable[tuple[str, object]],
    wanted_guid: str,
    accept: Callable[[object], bool] = lambda js: True,
) -> tuple[object | None, bool]:
    """(joystick, found_by_vid_pid): exact GUID first, else the only accepted vid:pid match."""
    wanted_vid_pid = joystick_guid_vid_pid(wanted_guid) if wanted_guid else None
    same_device: list[object] = []
    for guid, js in candidates:
        if guid == wanted_guid:
            return js, False
        if wanted_vid_pid is not None and joystick_guid_vid_pid(guid) == wanted_vid_pid:
            same_device.append(js)
    accepted = [js for js in same_device if accept(js)]
    if len(same_device) == 1 and len(accepted) == 1:
        return accepted[0], True
    return None, False


class JoystickPool:
    """Opens every connected joystick, at most one slow open per tick. Owned by one thread."""

    def __init__(
        self,
        count_fn: Callable[[], int],
        open_fn: Callable[[int], object],
        *,
        budget_s: float = _OPEN_BUDGET_S,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._count_fn = count_fn
        self._open_fn = open_fn
        self._budget_s = budget_s
        self._clock = clock
        self.walk = 0
        self._begin_walk(None)

    def _begin_walk(self, count: int | None) -> None:
        self.devices: dict[str, object] = {}
        self.names: dict[str, str] = {}
        self.guid_by_instance: dict[int, str] = {}
        self.errors: list[str] = []
        self.hid_only: list[str] = []
        self._count = count
        self._next = 0
        self.walk += 1

    def restart(self) -> None:
        """Walk every index again on the next step: retries failed opens."""
        self._count = None

    def step(self) -> bool:
        """Open more joysticks; True once every connected index has been tried."""
        try:
            count = self._count_fn()
        except Exception:
            return False
        if count != self._count:
            self._begin_walk(count)
        start = self._clock()
        opened = False
        while self._next < count:
            if opened and self._clock() - start >= self._budget_s:
                break
            index = self._next
            self._next += 1
            opened = True
            try:
                js = self._open_fn(index)
                js.init()
                if js.get_numaxes() == 0:
                    # SDL 2.28 never listed buttons-only devices (MOZA stalk); the HID thread reads them.
                    self.hid_only.append(js.get_name())
                    continue
                guid = js.get_guid()
                self.devices[guid] = js
                self.names[guid] = js.get_name()
                self.guid_by_instance[js.get_instance_id()] = guid
            except Exception as exc:
                # A device another program holds exclusively fails here.
                self.errors.append(f"joystick {index} could not be opened ({exc})")
        return self._next >= count

    def remove_instance(self, instance_id: int) -> None:
        guid = self.guid_by_instance.pop(instance_id, None)
        if guid is not None:
            self.devices.pop(guid, None)
            self.names.pop(guid, None)
