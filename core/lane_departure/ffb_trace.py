"""Reader for the FFB plugin's trace block (plugins/ffb/ffb_trace.h) and the debug CSV it feeds.

The plugin keeps its last 256 constant-force updates (about 2 s at the game's FFB rate): the game's own force, what
went to the wheel and the wheel angle the servo used, plus the effects the game made. MonoCruise only reads it, and
only with ``Settings.debug`` on, to fit the game's own centring offline. See plugins/ffb/README.md, Trace block.
"""

from __future__ import annotations

import logging
import math
import mmap
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.sending_thread.debug_csv import open_debug_csv

logger = logging.getLogger(__name__)

TAG = r"Local\MonoCruiseFfbTrace"
VERSION = 1
LEN = 256
SIZE = 6336
FLAG_SERVO = 1
FLAG_RUMBLE = 2
FLAG_LIVE = 4
FLAG_DRIVING = 8
KINDS = ("constant", "ramp", "square", "sine", "triangle", "sawtooth up", "sawtooth down", "spring", "damper",
         "inertia", "friction", "custom")
CONDITIONS = ("spring", "damper", "inertia", "friction")
KIND_OTHER = 31
_RETRY_S = 2.0
_MARGIN = 2

_HEAD = struct.Struct("=IIIIIII")
_COND = struct.Struct("=iiiIIiII")
_SAMPLE = struct.Struct("=dfhhhHHh")
_U32 = struct.Struct("=I")
_OFF_HEAD = 8
_OFF_COND = 64
_OFF_SAMPLES = 192


@dataclass(frozen=True)
class TraceSample:
    """One constant-force update. ``t`` is the plugin's performance-counter clock in seconds."""

    t: float
    steer: float
    game: int
    out: int
    servo: int
    steer_seq: int
    flags: int


@dataclass(frozen=True)
class Condition:
    """A condition effect's first axis as the game last wrote it, out of 10000; ``sets`` counts the writes."""

    offset: int = 0
    pos_coeff: int = 0
    neg_coeff: int = 0
    pos_sat: int = 0
    neg_sat: int = 0
    deadband: int = 0
    gain: int = 0
    sets: int = 0


@dataclass(frozen=True)
class TraceRead:
    """New samples since the last read; ``lost`` counts the ones overwritten before they were read."""

    present: bool = False
    samples: tuple[TraceSample, ...] = ()
    lost: int = 0
    effects: int = 0
    creates: int = 0
    other_sets: int = 0
    const_gain: int = 0
    conditions: dict[str, Condition] = field(default_factory=dict)

    def kinds(self) -> list[str]:
        names = [name for i, name in enumerate(KINDS) if self.effects & (1 << i)]
        return names + (["other"] if self.effects & (1 << KIND_OTHER) else [])


def decode(raw: bytes, head_after: int, since: int | None) -> tuple[TraceRead, int | None]:
    """``raw`` copied between two reads of the head (``head_after`` the second); returns the read and the next ``since``.

    Slots the plugin may have rewritten during the copy are dropped. A head below ``since`` is a restarted plugin.
    """
    if len(raw) < SIZE:
        return TraceRead(), None
    version, capacity, head, effects, creates, other_sets, const_gain = _HEAD.unpack_from(raw, 0)
    if version != VERSION or capacity != LEN:
        return TraceRead(), None
    if since is not None and (head < since or head_after < head):
        since = None
    oldest = max(0, head_after - LEN + _MARGIN)
    start = oldest if since is None else max(since, oldest)
    lost = start - since if since is not None and since < start else 0
    samples = tuple(TraceSample(*_SAMPLE.unpack_from(raw, _OFF_SAMPLES + (i % LEN) * _SAMPLE.size)[:7])
                    for i in range(start, head))
    conditions = {name: Condition(*_COND.unpack_from(raw, _OFF_COND + k * _COND.size))
                  for k, name in enumerate(CONDITIONS)}
    return TraceRead(True, samples, lost, effects, creates, other_sets, const_gain, conditions), max(head, start)


class TraceReader:
    """Opens ``Local\\MonoCruiseFfbTrace`` lazily; absent plugin, non-Windows or any failure reads as ``TraceRead()``."""

    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._next_open = 0.0
        self._since: int | None = None

    def read(self, now: float | None = None) -> TraceRead:
        now = time.monotonic() if now is None else now
        if self._buf is None:
            if sys.platform != "win32" or now < self._next_open:
                return TraceRead()
            self._next_open = now + _RETRY_S
            try:
                self._buf = mmap.mmap(0, SIZE, TAG)
            except Exception:
                logger.debug("FFB trace block unavailable", exc_info=True)
                return TraceRead()
        try:
            raw = bytes(self._buf[:SIZE])
            head_after = _U32.unpack_from(self._buf, _OFF_HEAD)[0]
            out, self._since = decode(raw, head_after, self._since)
            return out
        except Exception:
            logger.debug("FFB trace read failed", exc_info=True)
            self.close()
            return TraceRead()

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                logger.debug("FFB trace block close failed", exc_info=True)
        self._buf = None
        self._since = None


TRACE_HEADER = [
    "t_s", "steer", "game", "out", "servo", "steer_seq", "servo_on", "rumble", "live", "lost", "read_s",
    "speed_kmh", "reason", "authority", "target", "target_rate", "kp", "kd", "ki", "cap", "damping_first", "only",
    "force_per_ms2", "kappa_per_steer", "kappa_ff", "const_gain", "spring_coeff", "spring_sat", "spring_gain",
    "damper_coeff", "damper_gain",
]
_TRACE_MAX_BYTES = 300_000_000
_EFFECT_LOG_S = 10.0


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.5f}" if math.isfinite(value) else ""
    return "" if value is None else str(value)


class TraceLog:
    """Every driving sample of the trace to ``name`` beside the project root, with this tick's context."""

    def __init__(self, root: Path, name: str, label: str) -> None:
        self._path = root / name
        self._label = label
        self._reader = TraceReader()
        self._file = None
        self._writer = None
        self._retry_at = -math.inf
        self._logged: tuple | None = None
        self._logged_at = -math.inf

    def tick(self, now: float, context: dict[str, object]) -> None:
        """Copies what the plugin added since the last tick. Never raises OSError."""
        read = self._reader.read(now)
        if not read.present:
            return
        self._note_effects(now, read)
        rows = [s for s in read.samples if s.flags & FLAG_DRIVING]
        if not rows:
            return
        if self._writer is None:
            if now < self._retry_at:
                return
            self._file, self._writer = open_debug_csv(self._path, TRACE_HEADER, _TRACE_MAX_BYTES)
            if self._writer is None:
                self._retry_at = now + 30.0
                return
        spring = read.conditions.get("spring", Condition())
        damper = read.conditions.get("damper", Condition())
        tail = [_fmt(context.get(k)) for k in TRACE_HEADER[11:25]] + [
            str(read.const_gain), str(spring.pos_coeff), str(spring.pos_sat), str(spring.gain),
            str(damper.pos_coeff), str(damper.gain)]
        read_s = f"{time.perf_counter():.5f}"
        try:
            for i, s in enumerate(rows):
                self._writer.writerow([
                    f"{s.t:.5f}", f"{s.steer:.5f}", s.game, s.out, s.servo, s.steer_seq,
                    int(bool(s.flags & FLAG_SERVO)), int(bool(s.flags & FLAG_RUMBLE)), int(bool(s.flags & FLAG_LIVE)),
                    read.lost if i == 0 else 0, read_s, *tail])
            self._file.flush()
        except OSError:
            logger.debug("%s write failed", self._path.name, exc_info=True)

    def _note_effects(self, now: float, read: TraceRead) -> None:
        """One log line when the game's effects or its centring conditions change, at most every 10 s."""
        state = (read.effects, read.const_gain, tuple(
            (c.pos_coeff, c.neg_coeff, c.pos_sat, c.offset, c.gain) for c in read.conditions.values() if c.sets))
        if state == self._logged or now - self._logged_at < _EFFECT_LOG_S:
            return
        self._logged, self._logged_at = state, now
        conds = ", ".join(f"{name} coefficient {c.pos_coeff}/{c.neg_coeff} saturation {c.pos_sat}/{c.neg_sat} "
                          f"offset {c.offset} gain {c.gain}" for name, c in read.conditions.items() if c.sets)
        logger.info("%s FFB trace: the game made %s effects (constant force gain %d)%s", self._label,
                    ", ".join(read.kinds()) or "no", read.const_gain, "; " + conds if conds else "")

    def close(self) -> None:
        self._reader.close()
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
        self._file = None
        self._writer = None
