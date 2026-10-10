"""HID button devices (vid_pid, byte*8+bit). Binding format: INPUT_BINDINGS.md."""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings
from core.input_bindings import joystick_guid_vid_pid, migrate_binding
from core.button_device_thread.hid_descriptor import ButtonLayout, read_button_layout

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
# A device that opens but never delivers a report is retried this slowly.
_UNREADABLE_RETRY_S = 30.0

# A release must hold this long to count; a shorter dip is contact bounce,
# which peaks near 7 ms on a MOZA stalk. Presses are never delayed.
_RELEASE_HOLD_S = 0.015
# Each published level is held at least this long so a consumer sampling on
# Windows timer granularity cannot step over an edge. See the README.
_MIN_DWELL_S = 0.020
# Above roughly 25 taps/s the edge queue cannot keep up; resync rather than lag.
_MAX_PENDING_EDGES = 8
# Bound the per-tick drain so a runaway device cannot stall the loop.
_MAX_REPORTS_PER_TICK = 64

# Capture warm-up marks noisy bits; the confirm window filters jitter.
_CAPTURE_WARMUP_S = 0
_CAPTURE_MAX_SCAN_DEVICES = 16
# A candidate bit must stay set this long before it is accepted: transient
# axis/jitter bits and contact bounce never hold that long.
_CAPTURE_CONFIRM_S = 0.030
# Skip generic mouse/keyboard usages; pygame joysticks excluded by vid:pid.
_SKIP_GENERIC_USAGES = {0x02, 0x06, 0x07}
# Joystick, gamepad, multi-axis: the only collections the capture scan opens.
_GAME_CONTROLLER_USAGES = {0x04, 0x05, 0x08}


def _pick_collection(infos: list[dict], *, controllers_only: bool) -> dict | None:
    """Prefer the game-controller collection of a device: it carries the buttons."""
    fallback = None
    for info in infos:
        if not info.get("path"):
            continue
        usage_page = info.get("usage_page") or 0
        usage = info.get("usage") or 0
        if usage_page == 0x01 and usage in _GAME_CONTROLLER_USAGES:
            return info
        if controllers_only or fallback is not None:
            continue
        if usage_page == 0x01 and usage in _SKIP_GENERIC_USAGES:
            continue
        fallback = info
    return fallback


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
class _ButtonState:
    """Per-bit debounce and publication state. See the README."""

    raw: bool = False
    logical: bool = False       # raw with contact bounce filtered out
    published: bool = False     # logical rate-limited so no edge is skipped
    presses: int = 0            # monotonic count of published press edges
    release_ts: float | None = None
    last_publish_ts: float = 0.0
    pending: list[bool] = field(default_factory=list)


@dataclass
class ButtonDeviceThreadData(ThreadData):
    # {vid_pid: {button_id: bool}}: updated every loop tick
    button_states: Dict[str, Dict[int, bool]] = field(default_factory=dict, repr=False)

    # {vid_pid: {button_id: int}}: monotonic press count. Consumers poll slower
    # than a tap lasts, so they read this instead of watching for an edge.
    button_press_counts: Dict[str, Dict[int, int]] = field(default_factory=dict, repr=False)

    # Capture API (used by the settings panel button-assignment flow).
    capture_active: bool = False
    # ("button_device", vid_pid, button_id, label, device_name) when captured
    capture_event: object = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class ButtonDeviceThread(BaseThread):
    loop_interval = 0.01  # 100 Hz: non-blocking HID reads, drained every tick
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="button_device_thread")
        self.data = ButtonDeviceThreadData()
        # vid_pid → hid.device | None  (None = not yet connected / lost)
        self._devices: dict[str, object] = {}
        # vid_pid → last raw HID report (list[int]): retained between ticks
        self._last_reports: dict[str, list[int]] = {}
        # vid_pid → {button_id: _ButtonState}: debounce and publication state
        self._buttons: dict[str, dict[int, _ButtonState]] = {}
        # vid_pid → human-readable name (for logging/popup)
        self._device_names: dict[str, str] = {}
        # vid_pid → monotonic time after which the next reconnect attempt is allowed
        self._reconnect_deadlines: dict[str, float] = {}
        # vid_pid → button bits from the report descriptor (None: unknown, all bits)
        self._layouts: dict[str, ButtonLayout | None] = {}
        # Devices that delivered a report since they were opened.
        self._read_ok: set[str] = set()
        # One popup per session for an unreadable device or a non-button binding.
        self._unreadable_warned: set[str] = set()
        self._binding_warned: set[str] = set()

        # ── Capture scan (thread-owned; UI only toggles data.capture_active) ──
        # vid_pid → open hid.device for devices opened just for this capture
        self._capture_scan: dict[str, object] = {}
        self._capture_scan_names: dict[str, str] = {}
        self._capture_scan_layouts: dict[str, ButtonLayout | None] = {}
        # vid_pid → {report_id: last raw report}
        self._capture_scan_reports: dict[str, dict[int, list[int]]] = {}
        # vid_pid → previous tick's bits (first sighting doubles as baseline)
        self._capture_prev_bits: dict[str, dict[int, bool]] = {}
        # vid_pid → bits that changed during warm-up (axis/counter noise)
        self._capture_noise: dict[str, set[int]] = {}
        self._capture_opened = False
        self._capture_started_ts = 0.0
        # (vid_pid, button_id) being hold-confirmed, and when the hold started
        self._capture_candidate: tuple[str, int] | None = None
        self._capture_candidate_ts = 0.0
        # One debug line per capture session while the pygame gate blocks
        self._capture_wait_logged = False

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

        now = time.perf_counter()
        new_states: dict[str, dict[int, bool]] = {}
        new_counts: dict[str, dict[int, int]] = {}

        for vid_pid, device in list(self._devices.items()):
            if not self.running:
                return

            if device is None:
                self._maybe_reconnect(vid_pid)
                self._reset_button_state(vid_pid)
                new_states[vid_pid] = {}
                new_counts[vid_pid] = {}
                continue

            try:
                self._drain_reports(vid_pid, device, now)

            except OSError:
                self._on_read_failure(vid_pid, device)
                new_states[vid_pid] = {}
                new_counts[vid_pid] = {}
                continue

            except Exception:
                # Keep the settled state: dropping to empty here would look like
                # a release to the CC button FSM and fire a spurious press.
                logger.debug("failed to read button device %s", vid_pid, exc_info=True)

            new_states[vid_pid] = self._settle_buttons(vid_pid, now)
            new_counts[vid_pid] = self._press_counts(vid_pid)

        self._tick_capture(new_states, now)

        with self.data._lock:
            self.data.button_states = new_states
            self.data.button_press_counts = new_counts

    def _on_read_failure(self, vid_pid: str, device) -> None:
        """Close a device whose read failed. Only a device that was working gets the disconnect popup."""
        name = self._device_names.get(vid_pid, vid_pid)
        try:
            device.close()
        except Exception:
            pass
        self._devices[vid_pid] = None
        self._reset_button_state(vid_pid)
        was_working = vid_pid in self._read_ok
        self._read_ok.discard(vid_pid)

        if was_working or not self._still_enumerated(vid_pid):
            logger.warning("Button device %r disconnected", name, extra={"popup": True})
            self._reconnect_deadlines[vid_pid] = time.monotonic() + _RECONNECT_INTERVAL
            return
        # Present but unreadable (no input reports, or held by another
        # program): a 2 s retry only repeated the popup forever.
        if vid_pid not in self._unreadable_warned:
            self._unreadable_warned.add(vid_pid)
            logger.warning(
                "Button device %r cannot be read: reassign its cruise control buttons",
                name,
                extra={"popup": True},
            )
        self._reconnect_deadlines[vid_pid] = time.monotonic() + _UNREADABLE_RETRY_S

    @staticmethod
    def _still_enumerated(vid_pid: str) -> bool:
        parsed = _parse_vid_pid(vid_pid)
        if parsed is None or _hid is None:
            return False
        try:
            return bool(_hid.enumerate(*parsed))
        except Exception:
            return False

    @staticmethod
    def _queue_edge(state: _ButtonState, value: bool) -> None:
        """Queue a logical edge for publication, keeping the queue alternating."""
        if state.pending:
            if state.pending[-1] == value:
                return
        elif state.published == value:
            return
        state.pending.append(value)
        if len(state.pending) > _MAX_PENDING_EDGES:
            state.pending = [] if state.published == value else [value]

    def _apply_release_hold(self, states: dict[int, _ButtonState], now: float) -> None:
        """Promote a release to logical once it has outlasted contact bounce."""
        for state in states.values():
            if state.release_ts is None or state.raw or not state.logical:
                continue
            if now - state.release_ts >= _RELEASE_HOLD_S:
                state.logical = False
                state.release_ts = None
                self._queue_edge(state, False)

    def _drain_reports(self, vid_pid: str, device, now: float) -> None:
        """Consume every queued report and turn raw bit changes into logical edges."""
        states = self._buttons.setdefault(vid_pid, {})
        layout = self._layouts.get(vid_pid)
        # A release that matured before this tick's reports must land first,
        # or a press arriving now would erase it.
        self._apply_release_hold(states, now)

        for _ in range(_MAX_REPORTS_PER_TICK):
            if not self.running:
                return
            raw = device.read(64, timeout_ms=0)  # non-blocking
            if not raw:
                return
            self._read_ok.add(vid_pid)
            self._last_reports[vid_pid] = raw
            # Axis bits are never buttons, and another report ID's bytes must
            # not overwrite this report's buttons.
            allowed = layout.bits_for(raw) if layout is not None else None
            for byte_idx, byte_val in enumerate(raw):
                for bit in range(8):
                    button_id = byte_idx * 8 + bit
                    if allowed is not None and button_id not in allowed:
                        continue
                    held = bool((byte_val >> bit) & 1)
                    state = states.setdefault(button_id, _ButtonState())
                    if held and not state.raw:
                        # A press is never delayed, and a re-press inside the
                        # hold window cancels the release: that dip was bounce.
                        state.release_ts = None
                        if not state.logical:
                            state.logical = True
                            self._queue_edge(state, True)
                    elif not held and state.raw:
                        state.release_ts = now
                    state.raw = held

    def _settle_buttons(self, vid_pid: str, now: float) -> dict[int, bool]:
        """Publish at most one edge per bit per tick, holding each level _MIN_DWELL_S."""
        states = self._buttons.setdefault(vid_pid, {})
        self._apply_release_hold(states, now)

        for state in states.values():
            if state.pending and now - state.last_publish_ts >= _MIN_DWELL_S:
                state.published = state.pending.pop(0)
                state.last_publish_ts = now
                if state.published:
                    state.presses += 1
        # Cover every bit the device has reported: binding_state() reads an
        # empty dict as "device has not reported yet".
        return {button_id: state.published for button_id, state in states.items()}

    def _press_counts(self, vid_pid: str) -> dict[int, int]:
        states = self._buttons.get(vid_pid, {})
        return {button_id: state.presses for button_id, state in states.items()}

    def _reset_button_state(self, vid_pid: str) -> None:
        """Drop settled state so a reconnect never inherits a stale held button."""
        self._buttons.pop(vid_pid, None)
        self._last_reports.pop(vid_pid, None)

    def _button_device_bindings(self) -> list[tuple[str, dict]]:
        """(setting name, binding) for every CC button bound to a HID device."""
        out: list[tuple[str, dict]] = []
        for name in (
            "cc_start_button", "cc_inc_button", "cc_dec_button",
            "acc_dist_inc_button", "acc_dist_dec_button",
        ):
            try:
                b = migrate_binding(getattr(Settings, name))
            except Exception:
                continue
            if b and b.get("source") == "button_device" and b.get("vid_pid"):
                out.append((name, b))
        return out

    def _check_bindings(self, vid_pid: str) -> None:
        """Warn once when a binding points at a bit the device does not declare as a button."""
        layout = self._layouts.get(vid_pid)
        if layout is None or vid_pid in self._binding_warned:
            return
        for _name, binding in self._button_device_bindings():
            if binding.get("vid_pid") != vid_pid:
                continue
            try:
                button_id = int(binding.get("button_id"))
            except (TypeError, ValueError):
                continue
            if layout.is_button(button_id):
                continue
            self._binding_warned.add(vid_pid)
            logger.warning(
                "A cruise control button is bound to a pedal or axis on %r: reassign it",
                self._device_names.get(vid_pid, vid_pid),
                extra={"popup": True},
            )
            return

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
        """Pop capture_event tuple or None; clears capture state."""
        with self.data._lock:
            ev = self.data.capture_event
            self.data.capture_event = None
            self.data.capture_active = False
            return ev

    # Capture internals (loop thread only)

    def _tick_capture(self, tracked_states: dict[str, dict[int, bool]], now: float) -> None:
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
                if not self._capture_wait_logged:
                    self._capture_wait_logged = True
                    logger.debug("capture waiting on main_pedal_thread joystick snapshot")
                return
            self._capture_opened = True
            self._capture_started_ts = time.monotonic()
            self._capture_prev_bits = {}
            self._capture_noise = {}
            self._capture_candidate = None
            self._capture_candidate_ts = 0.0
            self._open_capture_scan()

        # Merge bit states: tracked devices (already read this tick) + scan.
        merged: dict[str, dict[int, bool]] = dict(tracked_states)
        for vid_pid, device in list(self._capture_scan.items()):
            try:
                layout = self._capture_scan_layouts.get(vid_pid)
                reports = self._capture_scan_reports.setdefault(vid_pid, {})
                for _ in range(_MAX_REPORTS_PER_TICK):
                    if not self.running:
                        return
                    raw = device.read(64, timeout_ms=0)
                    if not raw:
                        break
                    report_id = raw[0] if layout is not None and layout.uses_report_ids else 0
                    reports[report_id] = raw
                if not reports:
                    continue
                merged[vid_pid] = self._scan_bits(reports, layout)
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
                self._capture_candidate_ts = 0.0
            else:
                if now - self._capture_candidate_ts >= _CAPTURE_CONFIRM_S:
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
                # Baseline first report; change-only devices need a second press.
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
                    self._capture_candidate_ts = now
            self._capture_prev_bits[vid_pid] = dict(bits)

    @staticmethod
    def _scan_bits(
        reports: dict[int, list[int]], layout: ButtonLayout | None,
    ) -> dict[int, bool]:
        """Button bits of a scanned device; every bit when its layout is unknown."""
        bits: dict[int, bool] = {}
        for report in reports.values():
            allowed = layout.bits_for(report) if layout is not None else None
            for byte_idx, byte_val in enumerate(report):
                for bit in range(8):
                    button_id = byte_idx * 8 + bit
                    if allowed is None or button_id in allowed:
                        bits[button_id] = bool((byte_val >> bit) & 1)
        return bits

    def _open_capture_scan(self) -> None:
        """Open the game-controller collection of each non-tracked device pygame does not own."""
        if not _hid_available or _hid is None:
            return
        try:
            infos = _hid.enumerate()
        except Exception:
            logger.debug("hid enumerate failed", exc_info=True)
            return

        joystick_vid_pids = self._pygame_joystick_vid_pids()
        by_vid_pid: dict[str, list[dict]] = {}
        for info in infos:
            vendor_id = info.get("vendor_id") or 0
            product_id = info.get("product_id") or 0
            if not vendor_id and not product_id:
                continue
            vid_pid = f"{vendor_id:04x}:{product_id:04x}"
            if vid_pid in self._devices or vid_pid in joystick_vid_pids:
                continue
            by_vid_pid.setdefault(vid_pid, []).append(info)

        # open_path holds the GIL for the whole open, so opening every HID
        # device (headsets, RGB controllers) can stall the whole program.
        for vid_pid, candidates in by_vid_pid.items():
            if len(self._capture_scan) >= _CAPTURE_MAX_SCAN_DEVICES:
                break
            info = _pick_collection(candidates, controllers_only=True)
            if info is None:
                continue
            try:
                device = _hid.device()
                device.open_path(info["path"])
                device.set_nonblocking(True)
            except Exception:
                continue
            self._capture_scan[vid_pid] = device
            self._capture_scan_names[vid_pid] = info.get("product_string") or vid_pid
            self._capture_scan_layouts[vid_pid] = read_button_layout(device)
        logger.debug("capture scan opened %d HID devices", len(self._capture_scan))

    @staticmethod
    def _pygame_capture_ready() -> bool:
        """True when main_pedal_thread published joystick_capture_ready."""
        try:
            pt = registry.get_thread("main_pedal_thread")
            if not pt.is_alive():
                # Nothing owns the joysticks then, and a dead thread never
                # publishes the flag. See the README.
                return True
            with pt.data._lock:
                return bool(pt.data.joystick_capture_ready)
        except Exception:
            return True

    @staticmethod
    def _pygame_joystick_vid_pids() -> set[str]:
        """vid:pid set from SDL GUIDs on main_pedal_thread (all connected joysticks)."""
        out: set[str] = set()
        try:
            pt = registry.get_thread("main_pedal_thread")
            with pt.data._lock:
                guids = list(pt.data.joystick_button_states.keys())
        except Exception:
            return out
        for guid in guids:
            vid_pid = joystick_guid_vid_pid(guid)
            if vid_pid is not None:
                out.add(vid_pid)
        return out

    def _teardown_capture_scan(self) -> None:
        for device in self._capture_scan.values():
            try:
                device.close()
            except Exception:
                pass
        self._capture_scan = {}
        self._capture_scan_names = {}
        self._capture_scan_layouts = {}
        self._capture_scan_reports = {}
        self._capture_prev_bits = {}
        self._capture_noise = {}
        self._capture_candidate = None
        self._capture_candidate_ts = 0.0
        self._capture_opened = False
        self._capture_wait_logged = False

    # ── Device management ─────────────────────────────────────────────────────

    def _collect_vid_pids(self) -> set[str]:
        """Return all unique vid_pid strings from current button_device bindings."""
        return {b["vid_pid"] for _name, b in self._button_device_bindings()}

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

        # hid.device().open(vid, pid) is not used: it enumerates every HID
        # device while holding the GIL, which stalled all threads per retry.
        try:
            infos = _hid.enumerate(vendor_id, product_id)
        except Exception:
            logger.debug("hid enumerate failed for %s", vid_pid, exc_info=True)
            infos = []
        info = _pick_collection(infos, controllers_only=False)
        if info is None:
            self._devices[vid_pid] = None
            logger.debug("button device %s not present", vid_pid)
            return False

        try:
            device = _hid.device()
            device.open_path(info["path"])
            device.set_nonblocking(True)
        except Exception:
            self._devices[vid_pid] = None
            logger.debug("button device %s not available", vid_pid, exc_info=True)
            return False

        name = info.get("product_string") or vid_pid
        self._devices[vid_pid] = device
        self._device_names[vid_pid] = name
        self._layouts[vid_pid] = read_button_layout(device)
        if vid_pid in self._unreadable_warned:
            logger.debug("reopened unreadable button device %s", vid_pid)
        else:
            logger.info("connected to button device: %s (%s)", name, vid_pid)
        self._check_bindings(vid_pid)
        return True

