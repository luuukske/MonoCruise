"""Ego pose read from SCS telemetry in the traffic buffer's game frame. See core/radar/README.md section 16."""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TELEMETRY_SHM_NAME = "Local\\SCSTelemetry"
_RETRY_S = 1.0

# scs-sdk-plugin shared-memory layout; offsets are identical in revisions 10 and 12.
_SUPPORTED_REVISIONS = frozenset({10, 12})
_REVISION = struct.Struct("<I")
_REVISION_OFFSET = 40
_FIELDS: dict[str, tuple[int, struct.Struct]] = {
    "sdkActive": (0, struct.Struct("<?")),
    "paused": (4, struct.Struct("<?")),
    "simulatedTime": (16, struct.Struct("<Q")),
    "speed": (948, struct.Struct("<f")),
    # Steer belongs to the pose: read from the telemetry thread's own copy it
    # aliases against the radar frame and the ego path steps. README §16.
    "gameSteer": (972, struct.Struct("<f")),
    "coordinateX": (2200, struct.Struct("<d")),
    "coordinateY": (2208, struct.Struct("<d")),
    "coordinateZ": (2216, struct.Struct("<d")),
    "rotationX": (2224, struct.Struct("<d")),
    "rotationY": (2232, struct.Struct("<d")),
}
_LAYOUT_END = max(off + s.size for off, s in _FIELDS.values())


@dataclass(frozen=True)
class ScsPose:
    """The timing-critical ego fields, all from one read of the telemetry block."""

    simulated_time_us: int
    paused: bool
    x: float
    y: float
    z: float
    yaw_norm: float
    pitch_raw: float
    speed: float
    steer: float


def _field(buf, name: str):
    offset, unpacker = _FIELDS[name]
    return unpacker.unpack_from(buf, offset)[0]


def layout_supported(buf) -> bool:
    """True when the mapped block is long enough and carries a known plugin revision."""
    if len(buf) < max(_LAYOUT_END, _REVISION_OFFSET + _REVISION.size):
        return False
    return _REVISION.unpack_from(buf, _REVISION_OFFSET)[0] in _SUPPORTED_REVISIONS


def pose_from_buffer(buf) -> ScsPose | None:
    """Decode the pose fields; None while the SDK reports itself inactive."""
    if not _field(buf, "sdkActive"):
        return None
    return ScsPose(
        simulated_time_us=int(_field(buf, "simulatedTime")),
        paused=bool(_field(buf, "paused")),
        x=float(_field(buf, "coordinateX")),
        y=float(_field(buf, "coordinateY")),
        z=float(_field(buf, "coordinateZ")),
        yaw_norm=float(_field(buf, "rotationX")),
        pitch_raw=float(_field(buf, "rotationY")),
        speed=float(_field(buf, "speed")),
        steer=float(_field(buf, "gameSteer")),
    )


def library_agrees(buf) -> bool | None:
    """Cross-check the fixed offsets against truck_telemetry; None when it is not installed."""
    try:
        from truck_telemetry.telemetry_version import v1_10, v1_12
    except Exception:
        return None
    snapshot = bytes(buf)
    for version in (v1_10, v1_12):
        if len(snapshot) < version.struct_telemetry.size or not version.is_same_version(snapshot):
            continue
        parsed = version.parse_data(snapshot)
        return all(parsed.get(name) == _field(snapshot, name) for name in _FIELDS)
    return False


class ScsPoseReader:
    """Owns a read-only view of the SCS telemetry block for the radar thread."""

    def __init__(self) -> None:
        self._shm = None
        self._retry_at: float = 0.0

    def open(self) -> bool:
        if self._shm is not None:
            return True
        now = time.monotonic()
        if now < self._retry_at:
            return False
        self._retry_at = now + _RETRY_S
        try:
            from multiprocessing.shared_memory import SharedMemory

            shm = SharedMemory(name=_TELEMETRY_SHM_NAME, create=False)
        except Exception:
            return False
        try:
            usable = layout_supported(shm.buf) and library_agrees(shm.buf) is not False
        except Exception:
            usable = False
        if not usable:
            self._close_quietly(shm)
            # A layout mismatch is permanent for this plugin, so stop retrying every second.
            self._retry_at = float("inf")
            logger.warning("SCS telemetry layout not recognised: radar keeps the telemetry thread pose")
            return False
        self._shm = shm
        logger.info("SCS telemetry pose reader opened")
        return True

    def close(self) -> None:
        if self._shm is not None:
            self._close_quietly(self._shm)
            self._shm = None

    @staticmethod
    def _close_quietly(shm) -> None:
        try:
            shm.close()
        except Exception:
            logger.debug("SCS telemetry pose reader close failed", exc_info=True)

    def simulated_time_us(self) -> int | None:
        if self._shm is None and not self.open():
            return None
        try:
            return int(_field(self._shm.buf, "simulatedTime"))
        except Exception:
            self.close()
            return None

    def read(self) -> ScsPose | None:
        if self._shm is None and not self.open():
            return None
        try:
            return pose_from_buffer(self._shm.buf)
        except Exception:
            self.close()
            return None
