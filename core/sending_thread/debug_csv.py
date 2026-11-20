"""Local debug CSVs written beside the project root. See core/sending_thread/README.md."""

from __future__ import annotations

import atexit
import csv
import logging
import math
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BRAKE_LOG_NAME: str = "brake_debug.csv"
_BRAKE_LOG_INTERVAL_S: float = 0.05    # 20 Hz, the game's physics rate
_BRAKE_LOG_TAIL_S: float = 2.0         # keep logging this long after braking ends
_LOG_RETRY_S: float = 30.0             # back off after a failed open
_QUEUE_MAX_ITEMS: int = 5000           # row batches waiting for the disk; more are dropped
_DRAIN_TIMEOUT_S: float = 2.0
BRAKE_LOG_HEADER: list[str] = [
    "t_s",
    "utc",
    "speed_ms",
    "accel_ms2",           # tracking differentiator, tau 0.30 s (what learning sees)
    "decel_fast_ms2",      # AEB loop measurement, tau 0.12 s
    "road_load_ms2",
    "slope_rad",
    "gear",
    "game_clutch",
    "game_throttle",
    "game_brake",
    "gas",
    "user_brake",
    "mapper_brake",
    "hold_brake",
    "aeb_pedal",           # AEB controller output, before AebPedalAxis.sent
    "logical_brake",       # merged pedal before the intensity remap
    "sent_brake",          # written to the game
    "full_authority",
    "brake_intensity",
    "tune_pedal",          # sent pedal read back in tune units (what learning inverts)
    "aeb_active",
    "aeb_warn",
    "aeb_target_decel_ms2",
    "aeb_required_decel_ms2",
    "cruise_active",
    "controller",
    "wanted_ms2",
    "est_brake_ms2",       # learned capacity, tune units
    "aeb_max_brake_ms2",
    "aeb_brake_scale",     # per-truck AEB scale (aeb_capacity.py)
    "aeb_full_ratio",      # credited full-pedal stop, decel / (baseline x frac(1))
    "brake_scale",
    "baseline_brake_ms2",
    "learn_gate",
    "learn_count",
    "learn_candidate",
    "mass_kg",
    "wheels_on_ground",
    "trailer_count",
    # Grip: why some full-pedal stops brake at 0.6x of others (telemetry_thread.grip_debug_fields).
    "lateral_ms2",
    "steer",
    "brake_temp_c",
    "air_psi",
    "wipers",
    "surface",
    "truck_slip",          # worst grounded wheel: 0 rolling free, 1 locked
    "trailer_slip",
    "wetness",             # traction plugin road wetness, empty without it
]

# Gas learning inputs while the pedal is down, whoever owns it. The mapper log only
# runs while the mapper commands, so the driver's own pulls never reached a replay.
GAS_LEARN_LOG_NAME: str = "gas_learn_debug.csv"
_GAS_LEARN_LOG_INTERVAL_S: float = 0.10    # 10 Hz, the mapper log's rate
_GAS_LEARN_LOG_MAX_BYTES: int = 512 * 1024 * 1024
GAS_LEARN_LOG_HEADER: list[str] = [
    "t_s",
    "utc",
    "speed_ms",
    "gear",
    "gas_cmd",             # gas written to the game, what update_accel learns from
    "gas_owner",           # mapper (its gas won the merge) or driver
    "controller",
    "game_throttle",
    "raw_ms2",             # update_accel's measured accel, before its max(0, .)
    "road_load_ms2",       # as passed to update_accel: road load less idle creep
    "slope_rad",
    "game_clutch",
    "mass_kg",
    "has_trailer",
    "learn_gate",          # gate that ended the learner's last tick, "ok" when it learned
    "learn_count",
    "accel_anchor",
    "accel_ratio",
    "accel_offset_ms2",
    "est_accel_ms2",       # full-pedal capability for this gear and mass, real units
]


def open_debug_csv(path: Path, header: list[str], max_bytes: int | None = None):
    """Open *path* for appending under *header*. Returns (file, writer), or (None, None).

    A file with a different header, or one past *max_bytes*, is rotated aside first:
    appending new columns under an old header silently misaligns every later row.
    Only the header line is ever read, so opening costs the same at any file size.
    """
    try:
        size = path.stat().st_size if path.exists() else 0
        write_header = size == 0
        if not write_header and (
            (max_bytes is not None and size > max_bytes) or _header_differs(path, header)
        ):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            try:
                path.rename(path.with_name(f"{path.stem}_{stamp}{path.suffix}"))
                write_header = True
            except OSError:
                logger.debug("%s rotate failed", path.name, exc_info=True)
        fh = path.open("a", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)
            fh.flush()
        return fh, writer
    except OSError:
        logger.debug("%s unavailable", path.name, exc_info=True)
        return None, None


def _header_differs(path: Path, header: list[str]) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            return next(csv.reader(fh), []) != header
    except OSError:
        return False


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else ""
    return str(value)


class _CsvWriter:
    """One daemon thread owns every debug CSV file; callers only enqueue rows (README, debug logs)."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX_ITEMS)
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._retry_at: dict[Path, float] = {}
        self.failed_paths: set[Path] = set()
        self.dropped = 0

    def submit(self, item: tuple) -> None:
        """Queue *item* without waiting. A full queue drops it and counts the drop."""
        self._ensure_started()
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def drain(self, timeout: float = _DRAIN_TIMEOUT_S) -> bool:
        """Wait until everything queued before this call is written. False on timeout."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        done = threading.Event()
        try:
            self._queue.put(("barrier", done), timeout=timeout)
        except queue.Full:
            return False
        return done.wait(timeout)

    def _ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="debug_csv_writer", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        files: dict[Path, tuple] = {}
        while True:
            item = self._queue.get()
            try:
                self._handle(files, item)
                if self._queue.empty():
                    self._flush_all(files)
            except Exception:
                logger.debug("debug CSV writer: %s failed", item[0], exc_info=True)

    def _handle(self, files: dict[Path, tuple], item: tuple) -> None:
        kind = item[0]
        if kind == "rows":
            _, path, header, max_bytes, rows = item
            entry = files.get(path)
            if entry is None:
                entry = self._open(path, header, max_bytes)
                if entry is None:
                    return
                files[path] = entry
            try:
                entry[1].writerows(rows)
            except OSError:
                logger.debug("%s write failed", path.name, exc_info=True)
        elif kind == "close":
            entry = files.pop(item[1], None)
            if entry is not None:
                try:
                    entry[0].close()
                except OSError:
                    pass
        elif kind == "barrier":
            self._flush_all(files)
            item[1].set()

    def _open(self, path: Path, header: list[str], max_bytes: int | None) -> tuple | None:
        now = time.monotonic()
        if now < self._retry_at.get(path, -math.inf):
            return None
        fh, writer = open_debug_csv(path, header, max_bytes)
        if writer is None:
            self._retry_at[path] = now + _LOG_RETRY_S
            self.failed_paths.add(path)
            return None
        self.failed_paths.discard(path)
        return fh, writer

    @staticmethod
    def _flush_all(files: dict[Path, tuple]) -> None:
        for path, (fh, _) in list(files.items()):
            try:
                fh.flush()
            except OSError:
                logger.debug("%s flush failed", path.name, exc_info=True)


_WRITER = _CsvWriter()


def drain_debug_csv(timeout: float = _DRAIN_TIMEOUT_S) -> bool:
    """Block until every queued debug row is on disk. Tests and shutdown only, never a control loop."""
    return _WRITER.drain(timeout)


def debug_csv_dropped() -> int:
    """Rows dropped because the writer fell behind (a stalled disk)."""
    return _WRITER.dropped


atexit.register(drain_debug_csv)


class DebugCsvSink:
    """Rows for one debug CSV, written by the shared writer thread. Never blocks, never raises OSError."""

    def __init__(self, path: Path, header: list[str], max_bytes: int | None = None) -> None:
        self.path = path
        self._header = header
        self._max_bytes = max_bytes

    def writerow(self, row: list) -> None:
        _WRITER.submit(("rows", self.path, self._header, self._max_bytes, [row]))

    def writerows(self, rows: list[list]) -> None:
        if rows:
            _WRITER.submit(("rows", self.path, self._header, self._max_bytes, list(rows)))

    def close(self) -> None:
        _WRITER.submit(("close", self.path))

    @property
    def failed(self) -> bool:
        """True once the writer could not open the file (it retries every 30 s)."""
        return self.path in _WRITER.failed_paths


class TickDebugLog:
    """Rate-limited rows while *active* plus a tail, written off the calling thread."""

    def __init__(
        self, root: Path, name: str, header: list[str],
        interval_s: float, tail_s: float, max_bytes: int | None = None,
    ) -> None:
        self._path = root / name
        self._header = header
        self._interval_s = interval_s
        self._tail_s = tail_s
        self._sink = DebugCsvSink(self._path, header, max_bytes)
        self._start_mono: float | None = None
        self._last_write_mono: float = -math.inf
        self._active_until: float = -math.inf

    def tick(self, now: float, active: bool, row: dict[str, object]) -> None:
        """Queue *row* when active or inside the tail. Never blocks on the disk, never raises OSError."""
        if active:
            self._active_until = now + self._tail_s
        if now > self._active_until or now - self._last_write_mono < self._interval_s:
            return
        self._last_write_mono = now
        if self._start_mono is None:
            self._start_mono = now
        values = dict(row)
        values["t_s"] = now - self._start_mono
        values["utc"] = datetime.now(timezone.utc).isoformat()
        self._sink.writerow([_fmt(values.get(k, "")) for k in self._header])

    def close(self) -> None:
        self._sink.close()


class BrakeDebugLog(TickDebugLog):
    """Every braking tick at 20 Hz plus a short tail, for offline brake-response fits."""

    def __init__(self, root: Path) -> None:
        super().__init__(
            root, BRAKE_LOG_NAME, BRAKE_LOG_HEADER, _BRAKE_LOG_INTERVAL_S, _BRAKE_LOG_TAIL_S,
        )


class GasLearnDebugLog(TickDebugLog):
    """Every gas tick the learner sees, driver's pedal included, for offline replays."""

    def __init__(self, root: Path) -> None:
        super().__init__(
            root, GAS_LEARN_LOG_NAME, GAS_LEARN_LOG_HEADER, _GAS_LEARN_LOG_INTERVAL_S, 0.0,
            max_bytes=_GAS_LEARN_LOG_MAX_BYTES,
        )
