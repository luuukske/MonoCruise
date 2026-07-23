"""ETS2LA traffic/parked shared-memory reader for RadarThread. See core/radar/README.md §5."""

from __future__ import annotations

import logging
import mmap
import struct
import time

from .traffic import (
    Position,
    Quaternion,
    Size,
    Trailer,
    Vehicle,
    vehicle_from_trailer,
    _READER_CLOCK_GAP_S,
)


logger = logging.getLogger(__name__)


# Shared-memory layout: mirrors the ETS2LA traffic plugin struct.
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

# Synthetic id base for trailer-as-vehicle records. See core/radar/README.md §12.
_TRAILER_VEHICLE_ID_BASE: int = 1_000_000

# Cap tracked vehicles per frame (nearest to ego). See core/radar/README.md §12.
_MAX_TRACKED_VEHICLES: int = 24


class TrafficReader:
    """Traffic mmap reader; per-id state via ``update_from_last``. See core/radar/README.md §5."""

    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._parked_buf: mmap.mmap | None = None
        self._parked_retry_at: float = 0.0
        self._last_vehicles: dict[int, Vehicle] = {}
        self._last_trailer_vehicles: dict[int, Vehicle] = {}
        # Debug: when capture_raw, read() stashes raw mmap bytes for AEB clips.
        self.capture_raw: bool = False
        self.last_traffic_bytes: bytes | None = None
        self.last_parked_bytes: bytes | None = None
        self.last_t_wall: float = 0.0
        # Last kinematics clock passed to _smooth_and_build (reader-level gap).
        self._last_kin_t: float | None = None
        # Set by radar on pause→unpause when the sim clock may not have jumped.
        self._pending_reanchor: bool = False

    def clear_kinematics_state(self) -> None:
        """Clear per-id smoothing after clock domain change. See core/radar/README.md §7."""
        self._last_vehicles.clear()
        self._last_trailer_vehicles.clear()
        self._last_kin_t = None
        self._pending_reanchor = False

    def request_reanchor(self) -> None:
        """Force a discontinuity hold on the next ``read`` / ``replay_frame``."""
        self._pending_reanchor = True

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
        t_now: float | None = None,
    ) -> tuple[list[Vehicle], list[Vehicle]] | None:
        """Decode one frame; ``t_now`` is kinematics seconds. See core/radar/README.md §7."""
        self.last_traffic_bytes = None
        self.last_parked_bytes = None
        if self._buf is None and not self.open():
            return None
        try:
            self._buf.seek(0)
        except Exception:
            return None
        try:
            slice_bytes = self._buf[:_BUF_SIZE]
            raw = struct.unpack(_TOTAL_FORMAT, slice_bytes)
        except Exception:
            self._buf = None
            return None
        if self.capture_raw:
            self.last_traffic_bytes = bytes(slice_bytes)

        vehicles = self._build_vehicles_from_raw(raw)
        vehicles.extend(self._read_parked_vehicles({int(v.id) for v in vehicles}))

        self.last_t_wall = time.time()
        kin_t = self.last_t_wall if t_now is None else float(t_now)
        return self._smooth_and_build(vehicles, kin_t, ego_x, ego_y, ego_z, ego_speed)

    @staticmethod
    def _build_vehicles_from_raw(raw: tuple) -> list[Vehicle]:
        """40-slot traffic decode (live ``read`` and ``replay_frame``)."""
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
        return vehicles

    def _smooth_and_build(
        self, vehicles: list[Vehicle], t_now: float,
        ego_x: float, ego_y: float, ego_z: float, ego_speed: float,
    ) -> tuple[list[Vehicle], list[Vehicle]]:
        """Cull, smooth, build trailer vehicles. See core/radar/README.md §12."""
        if len(vehicles) > _MAX_TRACKED_VEHICLES:
            vehicles.sort(
                key=lambda v: (v.position.x - ego_x) ** 2
                + (v.position.y - ego_y) ** 2
                + (v.position.z - ego_z) ** 2
            )
            vehicles = vehicles[:_MAX_TRACKED_VEHICLES]

        # TMP sub-frame pose snap rules. See core/radar/README.md §7.
        reanchor = self._pending_reanchor
        self._pending_reanchor = False
        if self._last_kin_t is not None:
            gap = t_now - self._last_kin_t
            reanchor = reanchor or gap > _READER_CLOCK_GAP_S or gap < 0.0
        self._last_kin_t = t_now

        for v in vehicles:
            if v.id in self._last_vehicles:
                prev = self._last_vehicles[v.id]
                if reanchor:
                    v._hold_across_clock_discontinuity(prev, t_now)
                else:
                    v.update_from_last(
                        prev, t_now, ego_x, ego_y, ego_z, ego_speed,
                    )
            else:
                # TMP sub-frame pose snap rules. See core/radar/README.md §7.
                v.time = t_now
        self._last_vehicles = {v.id: v for v in vehicles}
        trailer_vehicles = self._build_trailer_vehicles(
            vehicles, t_now, ego_x, ego_y, ego_z, ego_speed, reanchor=reanchor,
        )
        return vehicles, trailer_vehicles

    def replay_frame(
        self,
        traffic_bytes: bytes | None,
        parked_bytes: bytes | None,
        ego_x: float, ego_y: float, ego_z: float, ego_speed: float,
        t_wall: float,
    ) -> tuple[list[Vehicle], list[Vehicle]] | None:
        """Headless clip replay through the same path as ``read``."""
        if traffic_bytes is None:
            return None
        try:
            raw = struct.unpack(_TOTAL_FORMAT, traffic_bytes)
        except Exception:
            return None
        vehicles = self._build_vehicles_from_raw(raw)
        if parked_bytes is not None:
            try:
                praw = struct.unpack(_TOTAL_PARKED_FORMAT, parked_bytes)
                vehicles.extend(
                    self._build_parked_from_raw(praw, {int(v.id) for v in vehicles})
                )
            except Exception:
                pass
        # TMP sub-frame pose snap rules. See core/radar/README.md §7.
        for v in vehicles:
            if v.id not in self._last_vehicles:
                v.time = t_wall
        return self._smooth_and_build(vehicles, t_wall, ego_x, ego_y, ego_z, ego_speed)

    def _build_trailer_vehicles(
        self,
        vehicles: list[Vehicle],
        t_now: float,
        ego_x: float,
        ego_y: float,
        ego_z: float,
        ego_speed: float,
        *,
        reanchor: bool = False,
    ) -> list[Vehicle]:
        """Nested trailers as ACC Vehicles (not AEB). See core/radar/README.md §12."""
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
                if reanchor:
                    tv._hold_across_clock_discontinuity(prev, t_now)
                else:
                    tv.update_from_last(prev, t_now, ego_x, ego_y, ego_z, ego_speed)
            else:
                tv.time = t_now
        self._last_trailer_vehicles = {tv.id: tv for tv in trailer_vehicles}
        return trailer_vehicles

    def _read_parked_vehicles(self, existing_ids: set[int]) -> list[Vehicle]:
        if self._parked_buf is None:
            self._open_parked_buffer()
            if self._parked_buf is None:
                return []
        try:
            self._parked_buf.seek(0)
            parked_slice = self._parked_buf[:_PARKED_BUF_SIZE]
            raw = struct.unpack(_TOTAL_PARKED_FORMAT, parked_slice)
        except Exception:
            self._parked_buf = None
            self._parked_retry_at = time.monotonic() + 1.0
            return []
        if self.capture_raw:
            self.last_parked_bytes = bytes(parked_slice)

        return self._build_parked_from_raw(raw, existing_ids)

    @staticmethod
    def _build_parked_from_raw(raw: tuple, existing_ids: set[int]) -> list[Vehicle]:
        """Parked-buffer decode (live and replay)."""
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

