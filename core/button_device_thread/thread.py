"""
Button Device Thread — polls HID button devices for button states.

Devices are identified by a "vid_pid" string ("{vendor_id:04x}:{product_id:04x}").
Button IDs encode raw HID report position: button_id = byte_index * 8 + bit_index,
matching the layout used by devices like the MOZA Multi-function Stalk.

Binding format (stored in config.json / Settings):
  {"source": "button_device",
   "vid_pid": "0483:0001",      ← "{vendor_id:04x}:{product_id:04x}"
   "device_name": str,          ← display only
   "button_id": int}            ← byte_index * 8 + bit_index
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict

from core.thread_management.base_thread import BaseThread, ThreadData
from core.settings import Settings
from core.input_bindings import migrate_binding

logger = logging.getLogger(__name__)

_hid_available = False
_hid = None

try:
    import hid as _hid_module
    _hid = _hid_module
    _hid_available = True
except Exception:
    logger.warning("hid library not importable — button device bindings disabled")


_RECONNECT_INTERVAL = 2.0  # seconds between reconnect attempts for a lost device


def _parse_vid_pid(vid_pid: str) -> tuple[int, int] | None:
    """Parse 'XXXX:YYYY' hex string into (vendor_id, product_id). Returns None on error."""
    try:
        parts = vid_pid.split(":")
        if len(parts) != 2:
            return None
        return int(parts[0], 16), int(parts[1], 16)
    except Exception:
        return None


@dataclass
class ButtonDeviceThreadData(ThreadData):
    # {vid_pid: {button_id: bool}} — updated every loop tick
    button_states: Dict[str, Dict[int, bool]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class ButtonDeviceThread(BaseThread):
    loop_interval = 0.05  # 20 Hz — non-blocking HID reads
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="button_device_thread")
        self.data = ButtonDeviceThreadData()
        # vid_pid → hid.device | None  (None = not yet connected / lost)
        self._devices: dict[str, object] = {}
        # vid_pid → last raw HID report (list[int]) — retained between ticks
        self._last_reports: dict[str, list[int]] = {}
        # vid_pid → human-readable name (for logging/popup)
        self._device_names: dict[str, str] = {}
        # vid_pid → monotonic time after which the next reconnect attempt is allowed
        self._reconnect_deadlines: dict[str, float] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        if not _hid_available:
            logger.warning("hid library unavailable — button device bindings will not work")
            return
        self._connect_tracked_devices()
        logger.debug("button_device_thread setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        if not _hid_available:
            return

        self._ensure_tracked_devices()

        new_states: dict[str, dict[int, bool]] = {}

        for vid_pid, device in list(self._devices.items()):
            if device is None:
                self._maybe_reconnect(vid_pid)
                new_states[vid_pid] = {}
                continue

            try:
                raw = device.read(64, timeout_ms=0)  # non-blocking
                if raw:
                    self._last_reports[vid_pid] = raw

                report = self._last_reports.get(vid_pid, [])
                states: dict[int, bool] = {}
                for byte_idx, byte_val in enumerate(report):
                    for bit in range(8):
                        states[byte_idx * 8 + bit] = bool((byte_val >> bit) & 1)
                new_states[vid_pid] = states

            except OSError:
                name = self._device_names.get(vid_pid, vid_pid)
                logger.warning("Button device %r disconnected", name, extra={"popup": True})
                try:
                    device.close()
                except Exception:
                    pass
                self._devices[vid_pid] = None
                self._reconnect_deadlines[vid_pid] = time.monotonic() + _RECONNECT_INTERVAL
                new_states[vid_pid] = {}

            except Exception:
                logger.debug("failed to read button device %s", vid_pid, exc_info=True)
                new_states[vid_pid] = {}

        with self.data._lock:
            self.data.button_states = new_states

    def teardown(self) -> None:
        for device in self._devices.values():
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
        self._devices.clear()
        logger.debug("button_device_thread teardown complete")

    # ── Device management ─────────────────────────────────────────────────────

    def _collect_vid_pids(self) -> set[str]:
        """Return all unique vid_pid strings from current button_device bindings."""
        vid_pids: set[str] = set()
        for name in (
            "cc_start_button", "cc_inc_button", "cc_dec_button",
            "acc_dist_inc_button", "acc_dist_dec_button",
        ):
            try:
                raw = getattr(Settings, name)
                b = migrate_binding(raw)
                if b and b.get("source") == "button_device":
                    vp = b.get("vid_pid")
                    if vp:
                        vid_pids.add(vp)
            except Exception:
                pass
        return vid_pids

    def _connect_tracked_devices(self) -> None:
        for vid_pid in self._collect_vid_pids():
            if vid_pid not in self._devices:
                self._try_connect_device(vid_pid)

    def _ensure_tracked_devices(self) -> None:
        """Track any vid_pids that appeared since setup (e.g. binding reassignment)."""
        for vid_pid in self._collect_vid_pids():
            if vid_pid not in self._devices:
                self._try_connect_device(vid_pid)

    def _maybe_reconnect(self, vid_pid: str) -> None:
        """Attempt reconnect only if the cooldown has elapsed."""
        deadline = self._reconnect_deadlines.get(vid_pid, 0.0)
        if time.monotonic() >= deadline:
            self._try_connect_device(vid_pid)
            if self._devices.get(vid_pid) is None:
                self._reconnect_deadlines[vid_pid] = time.monotonic() + _RECONNECT_INTERVAL

    def _try_connect_device(self, vid_pid: str) -> bool:
        """Attempt to open the HID device. Returns True on success."""
        parsed = _parse_vid_pid(vid_pid)
        if parsed is None:
            logger.debug("invalid vid_pid: %s", vid_pid)
            self._devices[vid_pid] = None
            return False

        vendor_id, product_id = parsed
        try:
            device = _hid.device()
            device.open(vendor_id, product_id)
            device.set_nonblocking(True)
            name = device.get_product_string() or vid_pid
            self._devices[vid_pid] = device
            self._device_names[vid_pid] = name
            logger.info("connected to button device: %s (%s)", name, vid_pid)
            return True
        except Exception:
            self._devices[vid_pid] = None
            logger.debug("button device %s not available", vid_pid, exc_info=True)
            return False
