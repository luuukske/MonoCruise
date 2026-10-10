"""TruckersMP no-collision zone state from MonoCruise's TruckersMP plugin. See core/radar/README.md §18."""

from __future__ import annotations

import logging
import math
import mmap
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from core.acc.trailer_lock import TRAILER_VEHICLE_ID_BASE, resolve_tractor

from .ego_geometry import REF_HALF_LENGTH_M, REF_HALF_WIDTH_M, EgoGeometry
from .traffic import Vehicle


logger = logging.getLogger(__name__)

# Written once per rendered frame by tmp_plugin/ (monocruise_tmp.dll).
_STATE_TAG = r"Local\MonoCruiseTmpState"
_STATE_FORMAT = "=IIBBHHH"
_STATE_SIZE = 16
_STATE_VERSION = 1

# A writer that stops (game closed, plugin unloaded) must close the gate.
STALE_AFTER_S: float = 0.5
# Entering needs the zone signal to hold this long; leaving is immediate.
ENTER_CONFIRM_S: float = 0.3
# A jump this far between radar frames is a teleport (ferry, respawn); no truck covers it in one frame.
TELEPORT_JUMP_M: float = 50.0
# How far a body must sit inside ego at the zone exit to count as a lingering ghost.
# Box slack between two real trucks (mirrors, rounded corners) stays under it.
EXIT_ARM_INSET_M: float = 0.2


@dataclass(frozen=True)
class TmpState:
    """One read of the plugin state; ``fresh`` means the writer is still ticking.
    The player counts are diagnostics: TruckersMP's per-player collision flag is not usable."""

    fresh: bool = False
    connected: bool = False
    in_no_collision_zone: bool = False
    players_streamed: int = 0
    players_collidable: int = 0

    @property
    def in_zone(self) -> bool:
        """The live plugin reports ego inside a no-collision zone on a TruckersMP server."""
        return self.fresh and self.connected and self.in_no_collision_zone


def decode_state(raw: bytes, heartbeat_fresh: bool) -> TmpState:
    """Unpack the 16-byte state; an unknown layout version reads as no data."""
    version, _heartbeat, connected, in_zone, streamed, collidable, _ = struct.unpack(
        _STATE_FORMAT, raw[:_STATE_SIZE],
    )
    if version != _STATE_VERSION:
        return TmpState()
    return TmpState(
        fresh=heartbeat_fresh,
        connected=bool(connected),
        in_no_collision_zone=bool(in_zone),
        players_streamed=int(streamed),
        players_collidable=int(collidable),
    )


class TmpStateReader:
    """Reads ``Local\\MonoCruiseTmpState``; absent, stale or unknown data reads as ``TmpState()``."""

    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._open_failed: bool = False
        self._last_heartbeat: int | None = None
        self._heartbeat_seen_at: float = 0.0

    def _open(self) -> bool:
        if self._buf is not None:
            return True
        if self._open_failed:
            return False
        try:
            self._buf = mmap.mmap(0, _STATE_SIZE, _STATE_TAG)
            return True
        except Exception:
            # Non-Windows or no shared memory: the gate simply never opens.
            self._open_failed = True
            logger.debug("TruckersMP state buffer unavailable", exc_info=True)
            return False

    def read(self, now: float) -> TmpState:
        if not self._open():
            return TmpState()
        # Runs inside the radar loop: any failure reads as no data, never as an exception.
        try:
            raw = bytes(self._buf[:_STATE_SIZE])
            heartbeat = struct.unpack_from("=I", raw, 4)[0]
            if heartbeat != self._last_heartbeat:
                self._last_heartbeat = heartbeat
                self._heartbeat_seen_at = now
            fresh = heartbeat != 0 and (now - self._heartbeat_seen_at) <= STALE_AFTER_S
            return decode_state(raw, fresh)
        except Exception:
            logger.debug("TruckersMP state read failed", exc_info=True)
            return TmpState()

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                pass
        self._buf = None
        self._last_heartbeat = None
        self._heartbeat_seen_at = 0.0


class NoCollisionZoneGate:
    """All-or-nothing: while it is active every TruckersMP vehicle is a ghost to ego."""

    def __init__(self) -> None:
        self._agree_since: float | None = None
        self.active: bool = False
        # True only on the step the gate closed because the live plugin reported "not in a zone".
        self.exited_zone: bool = False
        self._seen_data: bool = False
        self._last_xz: tuple[float, float] | None = None
        # Set by a teleport; only the plugin reporting "not in a zone" clears it.
        self._teleported: bool = False

    def step(self, state: TmpState, now: float, ego_xz: tuple[float, float] | None = None) -> bool:
        was_active = self.active
        self.exited_zone = False
        if state.fresh and not self._seen_data:
            self._seen_data = True
            logger.info("TruckersMP no-collision zone data from the game plugin is available")

        jumped = False
        if ego_xz is not None:
            if self._last_xz is not None and math.dist(ego_xz, self._last_xz) > TELEPORT_JUMP_M:
                self._teleported = jumped = True
            self._last_xz = ego_xz
        if self._teleported and state.fresh and state.connected and not state.in_no_collision_zone:
            self._teleported = False

        if not state.in_zone or self._teleported:
            self._agree_since = None
            self._set_active(False)
            self.exited_zone = (
                was_active and not jumped and not self._teleported
                and state.fresh and state.connected and not state.in_no_collision_zone
            )
            return False

        if self._agree_since is None:
            self._agree_since = now
        self._set_active(now - self._agree_since >= ENTER_CONFIRM_S)
        return self.active

    def _set_active(self, active: bool) -> None:
        if active == self.active:
            return
        self.active = active
        if active:
            logger.info("TruckersMP no-collision zone: AEB ignores other players")
        else:
            logger.info("TruckersMP no-collision zone left: AEB watches other players again")

    def clear(self) -> None:
        self._agree_since = None
        self._last_xz = None
        self._teleported = False
        self.exited_zone = False
        self._set_active(False)


@dataclass(frozen=True)
class Box:
    """Oriented ground-plane rectangle: centre, unit forward axis, half extents."""

    cx: float
    cz: float
    fx: float
    fz: float
    half_length: float
    half_width: float

    def shrunk(self, inset: float) -> "Box":
        return Box(self.cx, self.cz, self.fx, self.fz,
                   max(0.0, self.half_length - inset), max(0.0, self.half_width - inset))


def ego_box(x: float, z: float, yaw_rad: float, geometry: EgoGeometry | None) -> Box:
    """Ego's body from the SDK placement origin; no wheel layout falls back to the reference rig."""
    g = geometry or EgoGeometry(REF_HALF_WIDTH_M, REF_HALF_LENGTH_M, REF_HALF_LENGTH_M, 0.0, False)
    fx, fz = -math.sin(yaw_rad), -math.cos(yaw_rad)
    shift = 0.5 * (g.front_m - g.rear_m)
    return Box(x + shift * fx, z + shift * fz, fx, fz, g.half_length_m, g.half_width_m)


def vehicle_box(v: Vehicle) -> Box | None:
    """The same body ``Vehicle.get_corners`` draws; None for a degenerate size."""
    fr, fl, bl, br = v.get_corners()
    cx = 0.25 * (fr[0] + fl[0] + bl[0] + br[0])
    cz = 0.25 * (fr[1] + fl[1] + bl[1] + br[1])
    mx, mz = 0.5 * (fr[0] + fl[0]) - cx, 0.5 * (fr[1] + fl[1]) - cz
    half_length = math.hypot(mx, mz)
    if half_length < 1e-6:
        return None
    return Box(cx, cz, mx / half_length, mz / half_length, half_length,
               0.5 * math.dist(fr, fl))


def boxes_overlap(a: Box, b: Box) -> bool:
    """Separating-axis test on the four box axes; touching counts as overlap."""
    dx, dz = b.cx - a.cx, b.cz - a.cz
    for ax, az in ((a.fx, a.fz), (-a.fz, a.fx), (b.fx, b.fz), (-b.fz, b.fx)):
        ra = a.half_length * abs(a.fx * ax + a.fz * az) + a.half_width * abs(a.fz * ax - a.fx * az)
        rb = b.half_length * abs(b.fx * ax + b.fz * az) + b.half_width * abs(b.fz * ax - b.fx * az)
        if abs(dx * ax + dz * az) > ra + rb:
            return False
    return True


class ExitGhostHold:
    """Players still inside ego when the zone ends stay ghosts until they separate (§18)."""

    def __init__(self) -> None:
        self._rigs: list[frozenset[int]] = []

    @property
    def ids(self) -> frozenset[int]:
        return frozenset().union(*self._rigs)

    def step(
        self,
        gate_active: bool,
        exited_zone: bool,
        ego: Box | None,
        vehicles: Sequence[Vehicle],
        trailer_vehicles: Sequence[Vehicle] = (),
    ) -> frozenset[int]:
        if gate_active or ego is None:
            self.clear()
            return frozenset()
        if not exited_zone and not self._rigs:
            return frozenset()
        boxes = {
            v.id: b for v in (*vehicles, *trailer_vehicles)
            if v.is_tmp and (b := vehicle_box(v)) is not None
        }
        if exited_zone:
            self._rigs = _rigs_inside(ego, vehicles, boxes)
            if self._rigs:
                logger.info(
                    "TruckersMP zone left with %d player(s) still overlapping ego: AEB keeps "
                    "ignoring them until they separate", len(self._rigs),
                )
        kept = [rig for rig in self._rigs
                if any(i in boxes and boxes_overlap(ego, boxes[i]) for i in rig)]
        if len(kept) != len(self._rigs):
            logger.debug("TruckersMP zone exit: %d lingering ghost(s) separated from ego",
                         len(self._rigs) - len(kept))
        self._rigs = kept
        return self.ids

    def clear(self) -> None:
        self._rigs = []


def _rigs_inside(ego: Box, vehicles: Sequence[Vehicle], boxes: dict[int, Box]) -> list[frozenset[int]]:
    """Whole rigs (tractor, trailer records, nested trailers) with a body clearly inside ego."""
    inner = ego.shrunk(EXIT_ARM_INSET_M)
    hit = {i for i, b in boxes.items() if boxes_overlap(inner, b)}
    if not hit:
        return []
    cache: dict[int, int] = {}
    head_of: dict[int, int] = {}
    for v in vehicles:
        if v.is_tmp and v.id in boxes:
            tractor = resolve_tractor(v, vehicles, cache) if v.is_trailer else None
            head_of[v.id] = tractor.id if tractor is not None else v.id
    for i in boxes:
        if i >= TRAILER_VEHICLE_ID_BASE:
            parent = (i - TRAILER_VEHICLE_ID_BASE) // 4
            head_of[i] = head_of.get(parent, parent)
    heads = {head_of.get(i, i) for i in hit}
    return [frozenset(i for i in boxes if head_of.get(i, i) == h) for h in sorted(heads)]


def ncz_vehicle_ids(active: bool, vehicles: list[Vehicle], trailer_vehicles: list[Vehicle]) -> frozenset[int]:
    """Every TruckersMP id the radar publishes while the gate is active; AI traffic never."""
    if not active:
        return frozenset()
    return frozenset(v.id for v in (*vehicles, *trailer_vehicles) if v.is_tmp)
