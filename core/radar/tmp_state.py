"""TruckersMP no-collision zone state from the ETS2LA plugin. See core/radar/README.md §18."""

from __future__ import annotations

import logging
import mmap
import struct
from dataclasses import dataclass

from .traffic import Vehicle


logger = logging.getLogger(__name__)

# Written by the plugin's TruckersMP SDK side (MonoCruise NCZ build of ets2la_plugin).
_STATE_TAG = r"Local\ETS2LAMpState"
_STATE_FORMAT = "=IIBBHHH"
_STATE_SIZE = 16
_STATE_VERSION = 1

# The plugin writes at 20 Hz; a writer that stops (game closed, plugin unloaded) must close the gate.
STALE_AFTER_S: float = 0.5
# Entering needs every signal to agree this long; leaving is immediate.
ENTER_CONFIRM_S: float = 0.3


@dataclass(frozen=True)
class TmpState:
    """One read of the plugin state. ``fresh`` means the writer is still ticking."""

    fresh: bool = False
    connected: bool = False
    in_no_collision_zone: bool = False
    players_streamed: int = 0
    players_collidable: int = 0

    @property
    def nobody_can_collide(self) -> bool:
        """Every signal agrees that no TruckersMP player can touch ego right now."""
        return (
            self.fresh
            and self.connected
            and self.in_no_collision_zone
            and self.players_streamed > 0
            and self.players_collidable == 0
        )


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
    """Reads ``Local\\ETS2LAMpState``; absent, stale or unknown data reads as ``TmpState()``."""

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
        self._seen_data: bool = False

    def step(self, state: TmpState, now: float) -> bool:
        if state.fresh and not self._seen_data:
            self._seen_data = True
            logger.info("TruckersMP no-collision zone data from the game plugin is available")

        if not state.nobody_can_collide:
            self._agree_since = None
            self._set_active(False)
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
        self._set_active(False)


def ncz_vehicle_ids(active: bool, vehicles: list[Vehicle], trailer_vehicles: list[Vehicle]) -> frozenset[int]:
    """Every TruckersMP id the radar publishes while the gate is active; AI traffic never."""
    if not active:
        return frozenset()
    return frozenset(v.id for v in (*vehicles, *trailer_vehicles) if v.is_tmp)
