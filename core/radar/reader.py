"""
Shared-memory traffic reader for the ETS2LA traffic plugin.

Opens ``Local\\ETS2LATraffic`` mmap and decodes up to 40 vehicle slots per
frame.  Each frame is converted into a list of ``Vehicle`` instances with
per-id continuity (speed smoothing, yaw EMA, position history) preserved
across reads by calling ``update_from_last``.

Traffic and parked vehicles combined are culled to the
``_MAX_TRACKED_VEHICLES`` nearest to ego before smoothing, so per-vehicle
CPU cost in radar/AEB/ACC (and debug rendering) stays bounded in dense
traffic.

This module is consumed by ``RadarThread``; AEB and ACC both receive the
resulting ``Vehicle`` list from the radar data snapshot rather than opening
the shared-memory buffer themselves.
"""

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

# Synthetic-id base for trailer-as-vehicle records (see _build_trailer_vehicles).
# Far above the int16 id space the traffic buffer uses, so a wrapped trailer can
# never collide with a real vehicle or parked-vehicle id.
_TRAILER_VEHICLE_ID_BASE: int = 1_000_000

# Max vehicles kept per frame (traffic + parked combined). When over the cap,
# the nearest to ego are kept and the rest are culled before the smoothing
# chain runs. Culled vehicles lose their per-id smoothing state and re-enter
# via the normal fresh-spawn init if they come back into range.
_MAX_TRACKED_VEHICLES: int = 24


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
        # AEB clip capture (debug only): when capture_raw is set, read() stashes
        # the exact decoded byte slices so the recorder gets the bytes radar
        # consumed with no second mmap read (see core/aeb/AGENTS.md capture plan).
        self.capture_raw: bool = False
        self.last_traffic_bytes: bytes | None = None
        self.last_parked_bytes: bytes | None = None
        self.last_t_wall: float = 0.0
        # Last kinematics clock passed to _smooth_and_build (reader-level gap).
        self._last_kin_t: float | None = None
        # Set by radar on pause→unpause when the sim clock may not have jumped.
        self._pending_reanchor: bool = False

    def clear_kinematics_state(self) -> None:
        """Drop per-id smoothing so the next frame re-anchors on a new clock.

        Used when radar switches between wall time and SCS ``simulatedTime``:
        carrying ``Vehicle.time`` across domains makes ``dt`` huge/negative and
        pins vehicles on the sub-frame path.
        """
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
        """Decode one frame.

        ``ego_x/y/z`` and ``ego_speed`` are forwarded to
        :meth:`Vehicle.update_from_last` for the TTC-scaled lag freeze
        (see ``core/radar/AGENTS.md`` §7).

        ``t_now`` is the kinematics clock (seconds) passed to
        ``update_from_last``. Live radar supplies SCS ``simulatedTime`` so
        pause/hitch gaps do not inflate ``dt``. When omitted, falls back to
        ``time.time()`` (tests / callers without a sim clock).

        Returns ``(vehicles, trailer_vehicles)``: the top-level radar
        vehicles and the synthetic trailer-as-vehicle records flattened from
        nested trailers (see :meth:`_build_trailer_vehicles`). Returns ``None``
        if the buffer is unavailable or the frame failed to decode.
        """
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
        """Construct unsmoothed Vehicles from an unpacked traffic buffer.

        Single source of truth for the 40-slot decode: the live ``read`` and the
        headless ``replay_frame`` both go through here so a format change can
        never drift between them.
        """
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
        """Cull to the nearest ``_MAX_TRACKED_VEHICLES``, then smooth and flatten.

        Shared tail of ``read`` and ``replay_frame``: culls the frame to the
        vehicles nearest ego (so live and replay stay identical), applies
        ``update_from_last`` against the reader's ``_last_vehicles`` state using
        the supplied clock, then builds the trailer-as-vehicle list.
        """
        if len(vehicles) > _MAX_TRACKED_VEHICLES:
            vehicles.sort(
                key=lambda v: (v.position.x - ego_x) ** 2
                + (v.position.y - ego_y) ** 2
                + (v.position.z - ego_z) ** 2
            )
            vehicles = vehicles[:_MAX_TRACKED_VEHICLES]

        # Reader-level gap (pause / hitch with no intermediate frames). Do not
        # use Vehicle.time here: sub-frames freeze it, so dt since last full
        # update is not a pause detector. Radar also sets _pending_reanchor on
        # pause→unpause when simulatedTime may not have jumped.
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
                # Anchor to the kinematics clock (sim or wall). Construction
                # defaults to time.time(); a mismatched domain would make the
                # next dt huge/negative and pin the vehicle on the sub-frame path.
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
        """Decode + smooth one captured frame headlessly (no mmap, injected clock).

        Feeds the recorded byte slices and ``t_wall`` through the exact same
        construction + ``update_from_last`` path as the live ``read`` so a
        captured clip reproduces the real radar smoothing (see the AEB capture
        plan). ``None`` traffic bytes (a paused frame) returns ``None``.
        """
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
        # Anchor freshly-seen vehicles to the recorded clock. A Vehicle is
        # constructed with time.time() (the replay wall clock); feeding the
        # recorded t_wall next frame would make dt negative and pin the vehicle
        # to the sub-frame path forever, so it never smooths. Live radar passes
        # SCS simulatedTime via read(t_now=...); _smooth_and_build anchors new
        # ids the same way.
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
        """Flatten nested trailers into standalone Vehicles for ACC scoring.

        Only trailers that aren't already top-level radar vehicles are
        wrapped: AI trucks nest every trailer, and in TMP the first trailer is
        its own vehicle while trailers behind it are nested on it. The
        ``not is_tmp or is_trailer`` gate matches the ACC tail-length rule in
        ``acc_controller._read_chain`` so neither path double-counts.

        Synthetic ids derive from the parent id + buffer slot, so per-id
        smoothing and position-history carry-forward stay continuous across
        frames: exactly as for real vehicles.
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
        """Construct parked Vehicles from an unpacked parked buffer.

        Shared by the live ``_read_parked_vehicles`` and headless ``replay_frame``.
        """
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

