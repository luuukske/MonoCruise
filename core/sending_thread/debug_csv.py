"""Local debug CSVs written beside the project root. See core/sending_thread/README.md."""

from __future__ import annotations

import csv
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BRAKE_LOG_NAME: str = "brake_debug.csv"
_BRAKE_LOG_INTERVAL_S: float = 0.05    # 20 Hz, the game's physics rate
_BRAKE_LOG_TAIL_S: float = 2.0         # keep logging this long after braking ends
_BRAKE_LOG_RETRY_S: float = 30.0       # back off after a failed open
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
]


def open_debug_csv(path: Path, header: list[str]):
    """Open *path* for appending under *header*. Returns (file, writer), or (None, None).

    A file with a different header is rotated aside first: appending new columns
    under an old header silently misaligns every later row.
    """
    try:
        write_header = not path.exists() or path.stat().st_size == 0
        if not write_header and _header_differs(path, header):
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


class BrakeDebugLog:
    """Every braking tick at 20 Hz plus a short tail, for offline brake-response fits."""

    def __init__(self, root: Path) -> None:
        self._path = root / BRAKE_LOG_NAME
        self._file = None
        self._writer = None
        self._start_mono: float | None = None
        self._last_write_mono: float = -math.inf
        self._active_until: float = -math.inf
        self._retry_at: float = -math.inf

    def tick(self, now: float, active: bool, row: dict[str, object]) -> None:
        """Write *row* when braking is active or inside its tail. Never raises OSError."""
        if active:
            self._active_until = now + _BRAKE_LOG_TAIL_S
        if now > self._active_until or now - self._last_write_mono < _BRAKE_LOG_INTERVAL_S:
            return
        if self._writer is None:
            if now < self._retry_at:
                return
            self._file, self._writer = open_debug_csv(self._path, BRAKE_LOG_HEADER)
            if self._writer is None:
                self._retry_at = now + _BRAKE_LOG_RETRY_S
                return
        self._last_write_mono = now
        if self._start_mono is None:
            self._start_mono = now
        values = dict(row)
        values["t_s"] = now - self._start_mono
        values["utc"] = datetime.now(timezone.utc).isoformat()
        try:
            self._writer.writerow([_fmt(values.get(k, "")) for k in BRAKE_LOG_HEADER])
            self._file.flush()
        except OSError:
            logger.debug("brake_debug write failed", exc_info=True)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
        self._file = None
        self._writer = None
