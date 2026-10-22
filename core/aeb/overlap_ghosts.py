"""Bodies deep inside ego are ghosts, not obstacles. See core/aeb/README.md, overlap ghosts."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from core.acc.trailer_lock import TRAILER_VEHICLE_ID_BASE
from core.aeb.calibration import AEBCalibration
from core.radar.tmp_state import Box, boxes_overlap, ids_inside, rigs_of, vehicle_box
from core.radar.traffic import Trailer, Vehicle


logger = logging.getLogger(__name__)

# Past this from ego's centre no rig member can reach a rig that touches ego.
_RIG_REACH_M: float = 60.0


def _nested_trailer_box(tr: Trailer) -> Box:
    """A nested trailer as AEB evaluates it: centred on ``tr.position``, unlike ACC's copy."""
    yaw = math.radians(tr.rotation.euler()[1])
    return Box(tr.position.x, tr.position.z, -math.sin(yaw), -math.cos(yaw),
               0.5 * tr.size.length, 0.5 * tr.size.width)


def rig_boxes(vehicles: Sequence[Vehicle]) -> dict[int, Box]:
    """The bodies AEB evaluates by id, nested trailers under the radar's synthetic ids (radar §12)."""
    boxes: dict[int, Box] = {}
    for v in vehicles:
        b = vehicle_box(v)
        if b is not None:
            boxes[v.id] = b
        if v.id < 0:
            continue
        for tr in v.trailers:
            if not tr.is_zero():
                boxes[TRAILER_VEHICLE_ID_BASE + int(v.id) * 4 + int(tr.slot)] = _nested_trailer_box(tr)
    return boxes


class OverlapGhostHold:
    """Rigs found deep inside ego, or inside it for too long, stay ghosts until they separate."""

    def __init__(self) -> None:
        self._rigs: list[frozenset[int]] = []
        self._inside_since: dict[int, float] = {}
        self._last_now: float | None = None

    @property
    def ids(self) -> frozenset[int]:
        return frozenset().union(*self._rigs)

    def step(
        self, ego: Box, vehicles: Sequence[Vehicle], now: float, cal: AEBCalibration,
    ) -> frozenset[int]:
        vehicles = [v for v in vehicles
                    if math.hypot(v.position.x - ego.cx, v.position.z - ego.cz) <= _RIG_REACH_M]
        boxes = rig_boxes(vehicles)
        kept = [rig for rig in self._rigs
                if any(i in boxes and boxes_overlap(ego, boxes[i]) for i in rig)]
        if len(kept) != len(self._rigs):
            logger.debug("AEB overlap ghost: %d rig(s) separated from ego",
                         len(self._rigs) - len(kept))

        # Persistence needs unbroken evidence: a gap between steps (pause) starts it over.
        if self._last_now is not None and now - self._last_now > cal.overlap_ghost_persist_s:
            self._inside_since.clear()
        self._last_now = now
        inside = ids_inside(ego, boxes, cal.overlap_ghost_contact_m)
        self._inside_since = {i: self._inside_since.get(i, now) for i in inside}
        lingering = {i for i, t in self._inside_since.items()
                     if now - t >= cal.overlap_ghost_persist_s}
        hit = ids_inside(ego, boxes, cal.overlap_ghost_depth_m) | lingering

        held = frozenset().union(*kept)
        for rig in rigs_of(hit, vehicles, boxes):
            if not rig <= held:
                kept.append(rig)
                held |= rig
                logger.info("AEB ignores a vehicle sitting inside ego: a ghost, not an obstacle")
        self._rigs = kept
        return held

    def clear(self) -> None:
        self._rigs = []
        self._inside_since = {}
        self._last_now = None
