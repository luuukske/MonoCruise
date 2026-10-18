"""TruckersMP SDK vehicle motion from MonoCruise's plugin (experimental). See core/radar/README.md §19.

Positions stay ETS2LA's; this supplies each matched TMP body's own speed and the age
of its last network sample, which together tell a netcode stall from a stop or a crash.
"""

from __future__ import annotations

import ctypes
import logging
import math
import mmap
import struct
import sys
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .traffic import Vehicle


logger = logging.getLogger(__name__)

# Written once per rendered frame by tmp_plugin/ (monocruise_tmp.dll, "Experimental vehicle block").
VEHICLES_TAG = r"Local\MonoCruiseTmpVehicles"
VEHICLES_VERSION = 1
MAX_SLOTS = 96
HEADER = struct.Struct("<IIqHHB3x")
SLOT = struct.Struct("<3d4f3f3ffIfBBH")
VEHICLES_SIZE = HEADER.size + MAX_SLOTS * SLOT.size

KIND_TRACTOR, KIND_TRAILER, KIND_LOCAL_TRACTOR, KIND_LOCAL_TRAILER = 0, 1, 2, 3
FLAG_PLACEMENT, FLAG_LINEAR, FLAG_ANGULAR, FLAG_SAMPLED = 1, 2, 4, 8

# Experimental switch: False keeps every TMP speed on the position fit.
SDK_SPEED_ENABLED: bool = True
# AEB switches to the raw SDK speed while the filter lags a hard change; ACC never does.
AEB_RAW_SHORT_WINDOW: bool = True
# AEB takes the raw speed once it sits this far below the filtered one.
SDK_SHORT_ENTER_MS: float = 1.0
# Pre-filter on ACC's SDK speed, speed-scaled: the §7 chain barely smooths at low speed.
SDK_SPEED_TAU_REST_S: float = 0.5
SDK_SPEED_TAU_FAST_S: float = 0.15
SDK_SPEED_TAU_REF_MS: float = 25.0
# Same-place bodies sit within 1.3 m (p99, tractors and trailers); queued rigs are farther apart.
MATCH_M: float = 3.0
# The client extrapolates a late player for ~300 ms, then freezes it (measured 2026-10-11).
STALE_MS: float = 300.0
# A match dropout (p90 4 frames) on the raw speed holds it, then fades to the position fit.
DROPOUT_HOLD_S: float = 0.3
DROPOUT_FADE_S: float = 0.5
# A writer that stops ticking (game closed, plugin unloaded) is no data.
WRITER_STALE_S: float = 0.5

# Crash evidence between two network samples. 30 min of crash-free TMP driving peaked at
# 24 m/s2 (a 7 -> 2 m/s stop), 17.5 above 8 m/s, and 156 deg/s below 8 m/s, 123 above.
CRASH_DECEL_MS2: float = 30.0
CRASH_MIN_LOSS_MS: float = 6.0
CRASH_TURN_DEG_S: float = 200.0
CRASH_TURN_MIN_SPEED_MS: float = 8.0
CRASH_REVERSE_COS: float = -0.3
CRASH_MIN_SPEED_MS: float = 3.0
_MIN_SAMPLE_DT_S: float = 0.05

_SEQ_READ_ATTEMPTS = 4
_F32 = struct.Struct("<f")


@dataclass(frozen=True)
class SdkBody:
    """One body: world pose, the sender's own velocity, and its sample age at read time."""

    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float
    vx: float
    vy: float
    vz: float
    wx: float
    wy: float
    wz: float
    age_ms: float
    updates: int
    distance_m: float
    kind: int
    flags: int
    rig: int
    arrival_s: float = 0.0

    @property
    def is_local(self) -> bool:
        return self.kind in (KIND_LOCAL_TRACTOR, KIND_LOCAL_TRAILER)

    @property
    def is_trailer(self) -> bool:
        return self.kind in (KIND_TRAILER, KIND_LOCAL_TRAILER)

    @property
    def has_velocity(self) -> bool:
        return bool(self.flags & FLAG_LINEAR)

    @property
    def sampled(self) -> bool:
        return bool(self.flags & FLAG_SAMPLED)

    @property
    def stale(self) -> bool:
        """No network sample for longer than the client extrapolates: the pose is frozen lag."""
        return self.sampled and self.age_ms > STALE_MS

    @property
    def speed_h(self) -> float:
        return math.hypot(self.vx, self.vz)

    def forward(self) -> tuple[float, float]:
        """Unit forward (x, z) from the SDK quaternion: local -Z rotated into the world."""
        w, x, y, z = self.qw, self.qx, self.qy, self.qz
        fx = -2.0 * (x * z + w * y)
        fz = -(1.0 - 2.0 * (x * x + y * y))
        n = math.hypot(fx, fz)
        return (fx / n, fz / n) if n > 1e-9 else (0.0, 0.0)

    def speed_long(self) -> float:
        fx, fz = self.forward()
        return self.vx * fx + self.vz * fz


@dataclass(frozen=True)
class SdkFrame:
    """One plugin write as seen at ``read_s`` (plugin clock, seconds); ages are at read time."""

    read_s: float
    connected: bool
    bodies: tuple[SdkBody, ...]


def qpc_s() -> float:
    """The plugin's clock (QueryPerformanceCounter) in seconds; perf_counter off Windows."""
    if sys.platform != "win32":
        return time.perf_counter()
    freq, t = ctypes.c_int64(), ctypes.c_int64()
    ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
    ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(t))
    return t.value / freq.value


def decode_frame(raw: bytes, read_s: float | None = None) -> SdkFrame | None:
    """Header plus used slots; ages move to ``read_s`` (default: the write time). Bad data is None."""
    if raw is None or len(raw) < HEADER.size:
        return None
    version, heartbeat, write_us, _ping, count, connected = HEADER.unpack_from(raw, 0)
    if version != VEHICLES_VERSION or heartbeat == 0 or count > MAX_SLOTS:
        return None
    if len(raw) < HEADER.size + count * SLOT.size:
        return None
    write_s = write_us / 1e6
    # Quantised the way a clip stores them, so a replayed frame equals the live one exactly.
    read_s = write_s if read_s is None else round(read_s * 1e6) / 1e6
    extra_ms = max(0.0, (read_s - write_s) * 1000.0)
    bodies = []
    for i in range(count):
        body = SdkBody(*SLOT.unpack_from(raw, HEADER.size + i * SLOT.size))
        if body.sampled:
            age = _F32.unpack(_F32.pack(body.age_ms + extra_ms))[0]
            body = replace(body, age_ms=age, arrival_s=read_s - age / 1000.0)
        bodies.append(body)
    return SdkFrame(read_s=read_s, connected=bool(connected), bodies=tuple(bodies))


def encode_frame(frame: SdkFrame) -> bytes:
    """Clip form: ages already at read time, so ``decode_frame`` gives the same frame back."""
    head = HEADER.pack(VEHICLES_VERSION, 1, int(round(frame.read_s * 1e6)), 0xFFFF,
                       len(frame.bodies), int(frame.connected))
    return head + b"".join(
        SLOT.pack(b.x, b.y, b.z, b.qw, b.qx, b.qy, b.qz, b.vx, b.vy, b.vz,
                  b.wx, b.wy, b.wz, b.age_ms, b.updates, b.distance_m, b.kind, b.flags, b.rig)
        for b in frame.bodies
    )


def crash_evidence(vx0: float, vz0: float, vx1: float, vz1: float, dt: float) -> bool:
    """Two consecutive samples of one body that no brake or steer can produce."""
    if dt < _MIN_SAMPLE_DT_S:
        return False
    v0, v1 = math.hypot(vx0, vz0), math.hypot(vx1, vz1)
    if v0 - v1 >= CRASH_MIN_LOSS_MS and (v0 - v1) / dt >= CRASH_DECEL_MS2:
        return True
    if v0 < CRASH_MIN_SPEED_MS or v1 < CRASH_MIN_SPEED_MS:
        return False
    cos = max(-1.0, min(1.0, (vx0 * vx1 + vz0 * vz1) / (v0 * v1)))
    if cos <= CRASH_REVERSE_COS:
        return True
    # Slow manoeuvres swing the velocity fast; at 8 m/s, 200 deg/s is 28 m/s2 of lateral pull.
    return (min(v0, v1) >= CRASH_TURN_MIN_SPEED_MS
            and math.degrees(math.acos(cos)) / dt >= CRASH_TURN_DEG_S)


def attach(vehicles: list["Vehicle"], frame: SdkFrame | None) -> None:
    """Give each TMP record the SDK body at its place (tractor to tractor, trailer to trailer)."""
    for v in vehicles:
        v.sdk = None
    if frame is None or not SDK_SPEED_ENABLED:
        return
    for is_trailer, kind in ((False, KIND_TRACTOR), (True, KIND_TRAILER)):
        bodies = [b for b in frame.bodies if b.kind == kind and b.has_velocity]
        recs = [v for v in vehicles if v.is_tmp and v.is_trailer == is_trailer and not v.is_parked]
        pairs = []
        for i, b in enumerate(bodies):
            for j, v in enumerate(recs):
                d = math.hypot(b.x - v.position.x, b.z - v.position.z)
                if d <= MATCH_M:
                    pairs.append((d, i, j))
        pairs.sort()
        used_b: set[int] = set()
        used_v: set[int] = set()
        for _d, i, j in pairs:
            if i in used_b or j in used_v:
                continue
            used_b.add(i)
            used_v.add(j)
            recs[j].sdk = bodies[i]


def matched_only(frame: SdkFrame | None, vehicles: list["Vehicle"]) -> SdkFrame | None:
    """The bodies ``attach`` paired, in frame order: replaying the match on them pairs the same."""
    if frame is None:
        return None
    used = {id(v.sdk) for v in vehicles if v.sdk is not None}
    return SdkFrame(frame.read_s, frame.connected, tuple(b for b in frame.bodies if id(b) in used))


class TmpSdkReader:
    """Reads ``Local\\MonoCruiseTmpVehicles``; absent, stale or torn data reads as None."""

    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._open_failed: bool = False
        self._last_heartbeat: int | None = None
        self._heartbeat_seen_at: float = 0.0
        self.last_frame: SdkFrame | None = None

    def _open(self) -> bool:
        if self._buf is not None:
            return True
        if self._open_failed:
            return False
        try:
            self._buf = mmap.mmap(0, VEHICLES_SIZE, VEHICLES_TAG)
            return True
        except Exception:
            self._open_failed = True
            logger.debug("TruckersMP SDK vehicle buffer unavailable", exc_info=True)
            return False

    def read_raw(self) -> bytes | None:
        """One whole plugin write: the heartbeat must read the same, non-zero, around the copy."""
        if not self._open():
            return None
        for _ in range(_SEQ_READ_ATTEMPTS):
            before = struct.unpack_from("<I", self._buf, 4)[0]
            copy = bytes(self._buf[:VEHICLES_SIZE])
            if before != 0 and before == struct.unpack_from("<I", self._buf, 4)[0]:
                return copy
        return None

    def read(self) -> SdkFrame | None:
        self.last_frame = None
        if not SDK_SPEED_ENABLED:
            return None
        # Runs inside the radar loop: any failure reads as no data, never as an exception.
        try:
            raw = self.read_raw()
            if raw is None:
                return None
            now_mono = time.monotonic()
            heartbeat = struct.unpack_from("<I", raw, 4)[0]
            if heartbeat != self._last_heartbeat:
                self._last_heartbeat = heartbeat
                self._heartbeat_seen_at = now_mono
            if now_mono - self._heartbeat_seen_at > WRITER_STALE_S:
                return None
            frame = decode_frame(raw, qpc_s())
            if frame is None or not frame.connected:
                return None
            self.last_frame = frame
            return frame
        except Exception:
            logger.debug("TruckersMP SDK vehicle read failed", exc_info=True)
            return None

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                pass
        self._buf = None
        self._last_heartbeat = None
        self.last_frame = None
