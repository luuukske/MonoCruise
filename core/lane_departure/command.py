"""Shared-memory command channel to the game-side FFB plugin (plugins/ffb).

The plugin creates ``Local\\MonoCruiseFfbCmd`` and owns the waveform. This side sends intent only
and bumps a heartbeat every call, so a stalled app lets the plugin fade out on its own.
"""

from __future__ import annotations

import logging
import mmap
import struct
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TAG = r"Local\MonoCruiseFfbCmd"
SIZE = 32
VERSION = 1
PEAK_UNITS = 1600
PEAK_CAP_UNITS = 3000
CMD_NONE = 0
CMD_RUMBLE = 1
_RETRY_S = 2.0

_U32 = struct.Struct("=I")
_CMD = struct.Struct("=BBH")
_OFF_VERSION = 0
_OFF_HEARTBEAT = 4
_OFF_CMD = 8
_OFF_SEQ = 12
_OFF_STATUS = 20
_OFF_FRAMES = 24

STATUS_HOOKED = 1
STATUS_FORCE_SEEN = 2
STATUS_RUMBLING = 4
STATUS_DEAD = 8


@dataclass(frozen=True)
class PluginStatus:
    """What the plugin reports back. ``present`` is False until it has stamped the layout version."""

    present: bool = False
    hooked: bool = False
    force_seen: bool = False
    rumbling: bool = False
    dead: bool = False
    frames: int = 0


def decode_status(raw: bytes) -> PluginStatus:
    version = _U32.unpack_from(raw, _OFF_VERSION)[0]
    if version != VERSION:
        return PluginStatus()
    bits = _U32.unpack_from(raw, _OFF_STATUS)[0]
    return PluginStatus(
        present=True,
        hooked=bool(bits & STATUS_HOOKED),
        force_seen=bool(bits & STATUS_FORCE_SEEN),
        rumbling=bool(bits & STATUS_RUMBLING),
        dead=bool(bits & STATUS_DEAD),
        frames=_U32.unpack_from(raw, _OFF_FRAMES)[0],
    )


def intensity_units(scale: float) -> int:
    """Peak magnitude out of 10000 for a user scale, never above the cap the plugin also enforces."""
    return max(0, min(PEAK_CAP_UNITS, round(PEAK_UNITS * scale)))


class RumbleChannel:
    """Writer for the command block. Absent plugin, non-Windows or any failure reads as ``PluginStatus()``."""

    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._next_open = 0.0
        self._heartbeat = 0
        self._seq = 0
        self._sent: tuple[int, int, int] | None = None

    @property
    def is_open(self) -> bool:
        return self._buf is not None

    def _open(self, now: float) -> bool:
        if self._buf is not None:
            return True
        if sys.platform != "win32" or now < self._next_open:
            return False
        self._next_open = now + _RETRY_S
        try:
            self._buf = mmap.mmap(0, SIZE, TAG)
        except Exception:
            logger.debug("FFB command block unavailable", exc_info=True)
            return False
        return True

    def publish(self, active: bool, side: int, units: int, now: float | None = None) -> PluginStatus:
        """Send this tick's request. Nothing is written until the plugin has stamped its layout version.

        The request is written only when it changes, so a second writer (the rumble test tool)
        is not cancelled by this one repeating "no rumble"; the heartbeat is bumped every call.
        """
        now = time.monotonic() if now is None else now
        if not self._open(now):
            return PluginStatus()
        try:
            status = decode_status(bytes(self._buf[:SIZE]))
            if not status.present:
                self._sent = None
                return status
            want = (CMD_RUMBLE, side, units) if active else (CMD_NONE, 0, 0)
            if want != self._sent:
                if active and (self._sent is None or self._sent[0] != CMD_RUMBLE):
                    self._seq = (self._seq + 1) & 0xFFFFFFFF
                    _U32.pack_into(self._buf, _OFF_SEQ, self._seq)
                _CMD.pack_into(self._buf, _OFF_CMD, *want)
                self._sent = want
            self._heartbeat = (self._heartbeat + 1) & 0xFFFFFFFF or 1
            _U32.pack_into(self._buf, _OFF_HEARTBEAT, self._heartbeat)
            return status
        except Exception:
            logger.debug("FFB command write failed", exc_info=True)
            self.close()
            return PluginStatus()

    def stop(self) -> None:
        """Ask for silence once; used when the feature is switched off or the thread ends."""
        if self._buf is not None:
            self.publish(False, 0, 0)

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                logger.debug("FFB command block close failed", exc_info=True)
        self._buf = None
        self._sent = None
