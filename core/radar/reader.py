"""
Shared-memory traffic reader for the ETS2LA traffic plugin.

Opens ``Local\\ETS2LATraffic`` mmap and decodes up to 40 vehicle slots per
frame.  Each frame is converted into a list of ``Vehicle`` instances with
per-id continuity (speed smoothing, yaw EMA, position history) preserved
across reads by calling ``update_from_last``.

This module is consumed by ``RadarThread``; AEB and ACC both receive the
resulting ``Vehicle`` list from the radar data snapshot rather than opening
the shared-memory buffer themselves.
"""

from __future__ import annotations

import logging
import mmap
import struct
import time

from .traffic import Position, Quaternion, Size, Trailer, Vehicle


logger = logging.getLogger(__name__)


# Shared-memory layout — mirrors the ETS2LA traffic plugin struct.
_VEHICLE_FORMAT = "ffffffffffffhhbb"
_TRAILER_FORMAT = "ffffffffff"
_VEHICLE_OBJECT_FORMAT = _VEHICLE_FORMAT + _TRAILER_FORMAT * 3
_TOTAL_FORMAT = "=" + _VEHICLE_OBJECT_FORMAT * 40
_BUF_SIZE = 6960
_VEH_STRIDE = 16 + 3 * 10

_PARKED_VEHICLE_FORMAT = "ffffffffffhb"
_TOTAL_PARKED_FORMAT = "=" + _PARKED_VEHICLE_FORMAT * 40
_PARKED_BUF_SIZE = 1720
_PARKED_STRIDE = 12


class TrafficReader:
    """Opens ``Local\\ETS2LATraffic`` mmap and reads the vehicle array.

    Reader keeps a ``_last_vehicles`` map so ``Vehicle.update_from_last`` can
    carry smoothed speed / yaw / position history forward across frames.
    """

    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._parked_buf: mmap.mmap | None = None
        self._parked_retry_at: float = 0.0
        self._last_vehicles: dict[int, Vehicle] = {}

    def open(self) -> bool:
        if self._buf is not None:
            if self._parked_buf is None:
                self._open_parked_buffer()
            return True
        try:
            self._buf = mmap.mmap(0, _BUF_SIZE, r"Local\ETS2LATraffic")
            logger.info("ETS2LATraffic shared-memory buffer opened")
            self._open_parked_buffer()
            return True
        except Exception:
            return False

    def _open_parked_buffer(self) -> None:
        now = time.monotonic()
        if now < self._parked_retry_at:
            return
        try:
            self._parked_buf = mmap.mmap(0, _PARKED_BUF_SIZE, r"Local\ETS2LAParkedVehicles")
            logger.info("ETS2LAParkedVehicles shared-memory buffer opened")
        except Exception:
            self._parked_buf = None
            self._parked_retry_at = now + 1.0

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                pass
            self._buf = None
        if self._parked_buf is not None:
            try:
                self._parked_buf.close()
            except Exception:
                pass
            self._parked_buf = None

    def read(self) -> list[Vehicle] | None:
        if self._buf is None and not self.open():
            return None
        try:
            self._buf.seek(0)
        except Exception:
            return None
        try:
            raw = struct.unpack(_TOTAL_FORMAT, self._buf[:_BUF_SIZE])
        except Exception:
            self._buf = None
            return None

        vehicles: list[Vehicle] = []
        data = raw
        for _ in range(40):
            position = Position(data[0], data[1], data[2])
            rotation = Quaternion(data[3], data[4], data[5], data[6])
            size = Size(data[7], data[8], data[9])
            speed = data[10]
            acceleration = data[11]
            trailer_count = data[12]
            vid = data[13]
            is_tmp = bool(data[14])
            is_trailer = bool(data[15])

            trailers: list[Trailer] = []
            for j in range(3):
                off = 16 + j * 10
                tp = Position(data[off], data[off + 1], data[off + 2])
                tr = Quaternion(data[off + 3], data[off + 4], data[off + 5], data[off + 6])
                ts = Size(data[off + 7], data[off + 8], data[off + 9])
                if not tp.is_zero():
                    trailers.append(Trailer(tp, tr, ts, is_tmp))

            if not position.is_zero() and not rotation.is_zero():
                vehicles.append(Vehicle(
                    position, rotation, size, speed, acceleration,
                    trailer_count, trailers, vid, is_tmp, is_trailer,
                ))
            data = data[_VEH_STRIDE:]

        vehicles.extend(self._read_parked_vehicles({int(v.id) for v in vehicles}))

        t_now = time.time()
        for v in vehicles:
            if v.id in self._last_vehicles:
                v.update_from_last(self._last_vehicles[v.id], t_now)
        self._last_vehicles = {v.id: v for v in vehicles}
        return vehicles

    def _read_parked_vehicles(self, existing_ids: set[int]) -> list[Vehicle]:
        if self._parked_buf is None:
            self._open_parked_buffer()
            if self._parked_buf is None:
                return []
        try:
            self._parked_buf.seek(0)
            raw = struct.unpack(_TOTAL_PARKED_FORMAT, self._parked_buf[:_PARKED_BUF_SIZE])
        except Exception:
            self._parked_buf = None
            self._parked_retry_at = time.monotonic() + 1.0
            return []

        vehicles: list[Vehicle] = []
        seen_ids: set[int] = set()
        data = raw
        for _ in range(40):
            position = Position(data[0], data[1], data[2])
            rotation = Quaternion(data[3], data[4], data[5], data[6])
            size = Size(data[7], data[8], data[9])
            vid = int(data[10])
            is_trailer = bool(data[11])

            if (
                position.is_zero()
                or rotation.is_zero()
                or vid in seen_ids
                or vid in existing_ids
            ):
                data = data[_PARKED_STRIDE:]
                continue
            seen_ids.add(vid)

            vehicles.append(Vehicle(
                position, rotation, size, 0.0, 0.0,
                0, [], vid, False, is_trailer, True,
            ))
            data = data[_PARKED_STRIDE:]

        return vehicles
