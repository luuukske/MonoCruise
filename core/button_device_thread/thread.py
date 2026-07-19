"""
Button Device Thread: polls HID button devices for button states.

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
from core.thread_management.registry import registry
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
    logger.warning("hid library not importable: button device bindings disabled")


_RECONNECT_INTERVAL = 2.0  # seconds between reconnect attempts for a lost device

# Capture: raw HID reports mix button bits with axis/counter bits that change
# on their own. During the warm-up window every changing bit is marked noisy
# and excluded, so only a deliberate press after warm-up can be captured.
_CAPTURE_WARMUP_S = 0
_CAPTURE_MAX_SCAN_DEVICES = 16
# A candidate bit must stay set this many consecutive ticks (~150 ms at 20 Hz)
# before it is accepted: transient axis/jitter bits never hold that long.
_CAPTURE_CONFIRM_TICKS = 1
# Generic-desktop usages that must never be scanned: mice/keyboards/keypads
# are system input. Joystick/gamepad/multi-axis devices are not skipped here;
# anything pygame actually exposes is excluded via _pygame_joystick_vid_pids()
# so pygame-invisible stalks (e.g. MOZA) still reach the HID capture path.
_SKIP_GENERIC_USAGES = {0x02, 0x06, 0x07}


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
    # {vid_pid: {button_id: bool}}: updated every loop tick
    button_states: Dict[str, Dict[int, bool]] = field(default_factory=dict, repr=False)

    # Capture API (used by the settings panel button-assignment flow).
    capture_active: bool = False
    # ("button_device", vid_pid, button_id, label, device_name) when captured
    capture_event: object = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class ButtonDeviceThread(BaseThread):
    loop_interval = 0.05  # 20 Hz: non-blocking HID reads
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="button_device_thread")
        self.data = ButtonDeviceThreadData()
        # vid_pid → hid.device | None  (None = not yet connected / lost)
        self._devices: dict[str, object] = {}
        # vid_pid → last raw HID report (list[int]): retained between ticks
        self._last_reports: dict[str, list[int]] = {}
        # vid_pid → human-readable name (for logging/popup)
        self._device_names: dict[str, str] = {}
        # vid_pid → monotonic time after which the next reconnect attempt is allowed
        self._reconnect_deadlines: dict[str, float] = {}

        # ── Capture scan (thread-owned; UI only toggles data.capture_active) ──
        # vid_pid → open hid.device for devices opened just for this capture
        self._capture_scan: dict[str, object] = {}
        self._capture_scan_names: dict[str, str] = {}
        self._capture_scan_reports: dict[str, list[int]] = {}
        # vid_pid → previous tick's bits (first sighting doubles as baseline)
        self._capture_prev_bits: dict[str, dict[int, bool]] = {}
        # vid_pid → bits that changed during warm-up (axis/counter noise)
        self._capture_noise: dict[str, set[int]] = {}
        self._capture_opened = False
        self._capture_started_ts = 0.0
        # (vid_pid, button_id) being hold-confirmed, and ticks held so far
        self._capture_candidate: tuple[str, int] | None = None
        self._capture_candidate_ticks = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        if not _hid_available:
            logger.warning("hid library unavailable: button device bindings will not work")
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

        self._tick_capture(new_states)

        with self.data._lock:
            self.data.button_states = new_states

    def teardown(self) -> None:
        self._teardown_capture_scan()
        for device in self._devices.values():
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
        self._devices.clear()
        logger.debug("button_device_thread teardown complete")

    # ── Capture API (called from UI thread; loop owns all device handles) ─────

    def start_capture(self) -> None:
        """Enable capture mode: the loop scans all HID devices for a press."""
        with self.data._lock:
            self.data.capture_active = True
            self.data.capture_event = None

    def cancel_capture(self) -> None:
        """Abort capture; the loop closes scan devices on its next tick."""
        with self.data._lock:
            self.data.capture_active = False
            self.data.capture_event = None

    def consume_capture(self) -> tuple | None:
        """Read + clear the captured event. Returns None if nothing captured.

        Returns ("button_device", vid_pid, button_id, label, device_name).
        """
        with self.data._lock:
            ev = self.data.capture_event
            self.data.capture_event = None
            self.data.capture_active = False
            return ev

    # ── Capture internals (loop thread only) ──────────────────────────────────

    def _tick_capture(self, tracked_states: dict[str, dict[int, bool]]) -> None:
        with self.data._lock:
            active = self.data.capture_active
        if not active:
            if self._capture_opened:
                self._teardown_capture_scan()
            return

        if not self._capture_opened:
            if not self._pygame_capture_ready():
                # Pedal thread has not published the all-joysticks set yet;
                # opening now would raw-scan pygame-owned devices (e.g. wheel).
                return
            self._capture_opened = True
            self._capture_started_ts = time.monotonic()
            self._capture_prev_bits = {}
            self._capture_noise = {}
            self._capture_candidate = None
            self._capture_candidate_ticks = 0
            self._open_capture_scan()

        # Merge bit states: tracked devices (already read this tick) + scan.
        merged: dict[str, dict[int, bool]] = dict(tracked_states)
        for vid_pid, device in list(self._capture_scan.items()):
            try:
                raw = device.read(64, timeout_ms=0)
                if raw:
                    self._capture_scan_reports[vid_pid] = raw
                report = self._capture_scan_reports.get(vid_pid)
                if not report:
                    continue
                bits: dict[int, bool] = {}
                for byte_idx, byte_val in enumerate(report):
                    for bit in range(8):
                        bits[byte_idx * 8 + bit] = bool((byte_val >> bit) & 1)
                merged[vid_pid] = bits
            except OSError:
                try:
                    device.close()
                except Exception:
                    pass
                self._capture_scan.pop(vid_pid, None)
            except Exception:
                logger.debug("capture read failed for %s", vid_pid, exc_info=True)

        in_warmup = (time.monotonic() - self._capture_started_ts) < _CAPTURE_WARMUP_S

        # Hold-confirm the current candidate before looking for new presses.
        if self._capture_candidate is not None:
            vid_pid, button_id = self._capture_candidate
            bits = merged.get(vid_pid)
            if bits is None or not bits.get(button_id, False):
                # Released before confirmation: not a deliberate hold.
                # Not marked noisy, so a proper (longer) re-press still works.
                self._capture_candidate = None
                self._capture_candidate_ticks = 0
            else:
                self._capture_candidate_ticks += 1
                if self._capture_candidate_ticks >= _CAPTURE_CONFIRM_TICKS:
                    name = (
                        self._capture_scan_names.get(vid_pid)
                        or self._device_names.get(vid_pid, vid_pid)
                    )
                    with self.data._lock:
                        if self.data.capture_active and self.data.capture_event is None:
                            self.data.capture_event = (
                                "button_device", vid_pid, button_id,
                                f"button {button_id}", name,
                            )
                            self.data.capture_active = False
                    for vp, bits2 in merged.items():
                        self._capture_prev_bits[vp] = dict(bits2)
                    return

        for vid_pid, bits in merged.items():
            prev = self._capture_prev_bits.get(vid_pid)
            if prev is None:
                # First sighting is the baseline; detection starts next tick.
                # Devices that only report on change therefore need a release
                # + second press if their very first report was the press.
                self._capture_prev_bits[vid_pid] = dict(bits)
                continue

            noise = self._capture_noise.setdefault(vid_pid, set())
            for button_id, held in bits.items():
                if held == prev.get(button_id, False):
                    continue
                if in_warmup:
                    noise.add(button_id)  # moving without user intent → noise
                elif (
                    held
                    and button_id not in noise
                    and self._capture_candidate is None
                ):
                    self._capture_candidate = (vid_pid, button_id)
                    self._capture_candidate_ticks = 1
            self._capture_prev_bits[vid_pid] = dict(bits)

    def _open_capture_scan(self) -> None:
        """Open every enumerable HID device not already tracked.

        Skipped: system mice/keyboards/keypads, plus anything pygame currently
        exposes as a joystick (excluded by vid:pid). Pygame-owned devices are
        captured on the joystick path; their raw HID reports are dominated by
        axis bits that would false-trigger this scan. Joystick-class devices
        that pygame does not enumerate (e.g. MOZA stalks) are included.
        Devices that refuse to open are silently ignored.
        """
        if not _hid_available or _hid is None:
            return
        try:
            infos = _hid.enumerate()
        except Exception:
            logger.debug("hid enumerate failed", exc_info=True)
            return

        joystick_vid_pids = self._pygame_joystick_vid_pids()
        seen: set[str] = set()
        for info in infos:
            if len(self._capture_scan) >= _CAPTURE_MAX_SCAN_DEVICES:
                break
            usage_page = info.get("usage_page") or 0
            usage = info.get("usage") or 0
            if usage_page == 0x01 and usage in _SKIP_GENERIC_USAGES:
                continue
            vendor_id = info.get("vendor_id") or 0
            product_id = info.get("product_id") or 0
            if not vendor_id and not product_id:
                continue
            vid_pid = f"{vendor_id:04x}:{product_id:04x}"
            if vid_pid in seen or vid_pid in self._devices:
                continue
            if vid_pid in joystick_vid_pids:
                continue
            path = info.get("path")
            if not path:
                continue
            seen.add(vid_pid)
            try:
                device = _hid.device()
                device.open_path(path)
                device.set_nonblocking(True)
                self._capture_scan[vid_pid] = device
                self._capture_scan_names[vid_pid] = (
                    info.get("product_string") or vid_pid
                )
            except Exception:
                continue
        logger.debug("capture scan opened %d HID devices", len(self._capture_scan))

    @staticmethod
    def _pygame_capture_ready() -> bool:
        """True once main_pedal_thread has published all-joystick capture states.

        If the pedal thread is missing, proceed without exclusion (empty set).
        """
        try:
            pt = registry.get_thread("main_pedal_thread")
            with pt.data._lock:
                return bool(pt.data.joystick_capture_ready)
        except Exception:
            return True

    @staticmethod
    def _pygame_joystick_vid_pids() -> set[str]:
        """vid:pid of every joystick pygame currently sees.

        Parsed from the SDL GUIDs published by main_pedal_thread (bytes 4-5 =
        vendor little-endian, bytes 8-9 = product little-endian). During
        capture the pedal thread publishes state for ALL connected joysticks,
        so this covers devices with no binding yet.
        """
        out: set[str] = set()
        try:
            pt = registry.get_thread("main_pedal_thread")
            with pt.data._lock:
                guids = list(pt.data.joystick_button_states.keys())
        except Exception:
            return out
        for guid in guids:
            try:
                if len(guid) >= 20:
                    vid = int(guid[10:12] + guid[8:10], 16)
                    pid = int(guid[18:20] + guid[16:18], 16)
                    if vid or pid:
                        out.add(f"{vid:04x}:{pid:04x}")
            except ValueError:
                continue
        return out

    def _teardown_capture_scan(self) -> None:
        for device in self._capture_scan.values():
            try:
                device.close()
            except Exception:
                pass
        self._capture_scan = {}
        self._capture_scan_names = {}
        self._capture_scan_reports = {}
        self._capture_prev_bits = {}
        self._capture_noise = {}
        self._capture_candidate = None
        self._capture_candidate_ticks = 0
        self._capture_opened = False

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
        if not _hid_available or _hid is None:
            self._devices[vid_pid] = None
            return False

        path = None
        product_name = None
        try:
            for info in _hid.enumerate(vendor_id, product_id):
                usage_page = info.get("usage_page") or 0
                usage = info.get("usage") or 0
                if usage_page == 0x01 and usage in _SKIP_GENERIC_USAGES:
                    continue
                cand = info.get("path")
                if not cand:
                    continue
                path = cand
                product_name = info.get("product_string")
                break
        except Exception:
            logger.debug("hid enumerate failed for %s", vid_pid, exc_info=True)

        try:
            device = _hid.device()
            if path is not None:
                device.open_path(path)
            else:
                # No filtered collection found; last resort (single-collection devices).
                device.open(vendor_id, product_id)
            device.set_nonblocking(True)
            name = product_name or device.get_product_string() or vid_pid
            self._devices[vid_pid] = device
            self._device_names[vid_pid] = name
            logger.info("connected to button device: %s (%s)", name, vid_pid)
            return True
        except Exception:
            self._devices[vid_pid] = None
            logger.debug("button device %s not available", vid_pid, exc_info=True)
            return False

