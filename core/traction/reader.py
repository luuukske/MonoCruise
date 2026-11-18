"""Road wetness from the traction plugin and the grip it leaves. See plugins/traction/README.md."""

from __future__ import annotations

import logging
import math
import mmap
import struct

logger = logging.getLogger(__name__)

# Written once per game frame by plugins/traction (monocruise_traction.dll), layout traction_shm.h.
_TAG = r"Local\MonoCruiseTraction"
_VERSION = 1
# Mapped whole: a smaller block created here first would leave the plugin unable to map its own.
_SIZE = 2172
_HEADER = struct.Struct("=IIIIIIf")
_WEATHER_LIVE = 2

# The plugin stops on game exit or unload; wetness changes over minutes, so this can be generous.
STALE_AFTER_S: float = 2.0
# Grip left on a wet road: dry x (1 - WET_GRIP_LOSS x wetness), fitted on 12 grip-limited stops (2026-11-18).
WET_GRIP_LOSS: float = 0.42
# Above this the road counts as wet for the AEB capacity learners.
WET_LEARN_MAX: float = 0.05


def wet_grip_factor(wetness: float | None) -> float:
    """Share of dry grip left; exactly 1.0 when the wetness is unknown or zero."""
    if wetness is None or not math.isfinite(wetness) or wetness <= 0.0:
        return 1.0
    return 1.0 - WET_GRIP_LOSS * min(wetness, 1.0)


def is_wet(wetness: float | None) -> bool:
    return wetness is not None and wetness > WET_LEARN_MAX


def decode_wetness(raw: bytes) -> tuple[int, float | None]:
    """(seq, wetness) from the block header; wetness None for no writer or no live weather."""
    version, seq, status, _frames, _cur, _target, wet = _HEADER.unpack_from(raw, 0)
    if version != _VERSION or not status & _WEATHER_LIVE or not math.isfinite(wet):
        return seq, None
    return seq, min(max(float(wet), 0.0), 2.0)


class TractionReader:
    """Reads ``Local\\MonoCruiseTraction``; absent, stale, torn or unknown data reads as None."""

    def __init__(self, tag: str = _TAG, buffer: bytearray | None = None) -> None:
        self._tag = tag
        self._buf: mmap.mmap | bytearray | None = buffer
        self._open_failed = False
        self._last_seq: int | None = None
        self._seq_seen_at = 0.0
        self._last_wetness: float | None = None

    def _open(self) -> bool:
        if self._buf is not None:
            return True
        if self._open_failed:
            return False
        try:
            self._buf = mmap.mmap(0, _SIZE, self._tag)
            return True
        except Exception:
            self._open_failed = True
            logger.debug("traction plugin buffer unavailable", exc_info=True)
            return False

    def wetness(self, now: float) -> float | None:
        """Road wetness 0..~1 while the plugin is live, else None. Never raises."""
        if not self._open():
            return None
        try:
            raw = bytes(self._buf[:_HEADER.size])
            seq, wet = decode_wetness(raw)
            if seq % 2:
                return self._last_wetness
            if seq != self._last_seq:
                self._last_seq = seq
                self._seq_seen_at = now
            if seq == 0 or now - self._seq_seen_at > STALE_AFTER_S:
                self._last_wetness = None
                return None
            self._last_wetness = wet
            return wet
        except Exception:
            logger.debug("traction plugin read failed", exc_info=True)
            return None

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                pass
        self._buf = None
        self._last_seq = None
        self._last_wetness = None
