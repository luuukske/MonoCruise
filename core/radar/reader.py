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

from .traffic import Position, Quaternion, Size, Trailer, Vehicle, vehicle_from_trailer


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

# Synthetic-id base for trailer-as-vehicle records (see _build_trailer_vehicles).
# Far above the int16 id space the traffic buffer uses, so a wrapped trailer can
# never collide with a real vehicle or parked-vehicle id.
_TRAILER_VEHICLE_ID_BASE: int = 1_000_000


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
        self._last_trailer_vehicles: dict[int, Vehicle] = {}

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

    def read(
        self,
        ego_x: float,
        ego_y: float,
        ego_z: float,
        ego_speed: float,
    ) -> tuple[list[Vehicle], list[Vehicle]] | None:
        """Decode one frame.

        ``ego_x/y/z`` and ``ego_speed`` are forwarded to
        :meth:`Vehicle.update_from_last` for the TTC-scaled lag freeze
        (see ``core/radar/AGENTS.md`` §7).

        Returns ``(vehicles, trailer_vehicles)`` — the top-level radar
        vehicles and the synthetic trailer-as-vehicle records flattened from
        nested trailers (see :meth:`_build_trailer_vehicles`). Returns ``None``
        if the buffer is unavailable or the frame failed to decode.
        """
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
                    trailers.append(Trailer(tp, tr, ts, is_tmp, slot=j))

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
                v.update_from_last(
                    self._last_vehicles[v.id], t_now, ego_x, ego_y, ego_z, ego_speed,
                )
        self._last_vehicles = {v.id: v for v in vehicles}

        trailer_vehicles = self._build_trailer_vehicles(
            vehicles, t_now, ego_x, ego_y, ego_z, ego_speed,
        )
        return vehicles, trailer_vehicles

    def _build_trailer_vehicles(
        self,
        vehicles: list[Vehicle],
        t_now: float,
        ego_x: float,
        ego_y: float,
        ego_z: float,
        ego_speed: float,
    ) -> list[Vehicle]:
        """Flatten nested trailers into standalone Vehicles for ACC scoring.

        Only trailers that aren't already top-level radar vehicles are
        wrapped: AI trucks nest every trailer, and in TMP the first trailer is
        its own vehicle while trailers behind it are nested on it. The
        ``not is_tmp or is_trailer`` gate matches the ACC tail-length rule in
        ``acc_controller._read_chain`` so neither path double-counts.

        Synthetic ids derive from the parent id + buffer slot, so per-id
        smoothing and position-history carry-forward stay continuous across
        frames — exactly as for real vehicles.
        """
        trailer_vehicles: list[Vehicle] = []
        for v in vehicles:
            if v.id < 0:
                continue
            if v.is_tmp and not v.is_trailer:
                continue
            for tr in v.trailers:
                if tr.is_zero():
                    continue
                sid = _TRAILER_VEHICLE_ID_BASE + int(v.id) * 4 + int(tr.slot)
                trailer_vehicles.append(vehicle_from_trailer(v, tr, sid))

        for tv in trailer_vehicles:
            prev = self._last_trailer_vehicles.get(tv.id)
            if prev is not None:
                tv.update_from_last(prev, t_now, ego_x, ego_y, ego_z, ego_speed)
        self._last_trailer_vehicles = {tv.id: tv for tv in trailer_vehicles}
        return trailer_vehicles

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
