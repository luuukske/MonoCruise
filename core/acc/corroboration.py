"""Inter-vehicular corroboration for slow and stationary traffic.

A vehicle below ``_VALIDATE_MIN_SPEED_MS`` has no usable trail and can never
latch ``moving_validated``, so its lateral is known only as well as the road
model happens to be at its position. Where several such vehicles form a line
that agrees with the road, each is better placed than it is alone.

Corroboration raises how well a vehicle's position is *known*, not how much it
looks like a lead: a well-corroborated vehicle 20 m off the centreline is
rejected harder, because ``offset_component`` scales the whole term by evidence
including its negative range. See ``core/acc/README.md`` §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Only vehicles too slow to have earned a trail of their own; matches the
# tracker's own validation gate, so this covers exactly what that cannot.
MAX_SPEED_MS: float = 2.0

# Group gates. Measured together at 90.8 % of accepted members genuinely on
# ego's road, against 8.9 % for unfiltered stationary traffic (README §9).
_ALIGN_TO_ROAD_DEG: float = 25.0
_PARALLEL_SPREAD_DEG: float = 15.0
_LANE_SPREAD_M: float = 2.0
_MIN_SPAN_M: float = 12.0

# A line needs this many to say anything, and saturates here. Below the minimum
# a vehicle is left exactly as it was: one on its own earns nothing.
_MIN_MEMBERS: int = 3
_FULL_MEMBERS: int = 8

# Corroboration is inference from other vehicles, so it never outranks having
# watched this one drive its own line (_VALIDATED_STATIONARY_EVIDENCE = 0.75).
_MIN_EVIDENCE: float = 0.50
_MAX_EVIDENCE: float = 0.75
# Without a confident road model the only anchor is ego's own arc, which is a
# weaker claim that the line is the road ego is on rather than a road.
_UNANCHORED_MAX_EVIDENCE: float = 0.60


@dataclass(slots=True, frozen=True)
class SlowVehicle:
    """A slow vehicle in road coordinates, ready to be grouped."""

    vid: int
    s_m: float
    offset_m: float
    # Signed angle between the vehicle's heading and the road tangent under it.
    heading_err_rad: float


def _clusters(members: list[SlowVehicle]) -> list[list[SlowVehicle]]:
    """Split by lateral offset: a queue holds a lane, so its offset is constant.

    Single linkage on offset, which keeps a lane together without merging it
    into the next one. Cross-lane neighbours land in separate clusters and so
    never corroborate each other."""
    ordered = sorted(members, key=lambda m: m.offset_m)
    out: list[list[SlowVehicle]] = []
    for member in ordered:
        if out and member.offset_m - out[-1][-1].offset_m <= _LANE_SPREAD_M:
            out[-1].append(member)
        else:
            out.append([member])
    return out


def _is_line(cluster: list[SlowVehicle]) -> bool:
    """Mutually parallel, and long enough along the road to be a line at all."""
    if len(cluster) < _MIN_MEMBERS:
        return False
    headings = [m.heading_err_rad for m in cluster]
    if math.degrees(max(headings) - min(headings)) > _PARALLEL_SPREAD_DEG:
        return False
    span = max(m.s_m for m in cluster) - min(m.s_m for m in cluster)
    return span >= _MIN_SPAN_M


def _evidence_for(count: int, road_anchored: bool) -> float:
    cap = _MAX_EVIDENCE if road_anchored else _UNANCHORED_MAX_EVIDENCE
    span = _FULL_MEMBERS - _MIN_MEMBERS
    frac = 1.0 if span <= 0 else (count - _MIN_MEMBERS) / span
    return _MIN_EVIDENCE + (cap - _MIN_EVIDENCE) * max(0.0, min(1.0, frac))


def slow_vehicle_evidence(
    vehicles,
    ego_x: float, ego_z: float,
    ego_fwd_x: float, ego_fwd_z: float,
    ego_yaw_rad: float,
    road,
    max_s_m: float,
) -> dict[int, float]:
    """Project the slow traffic into road coordinates, then corroborate it."""
    members: list[SlowVehicle] = []
    for v in vehicles:
        if v.id < 0 or getattr(v, "is_parked", False):
            continue
        if abs(v.speed) >= MAX_SPEED_MS:
            continue
        dx = v.position.x - ego_x
        dz = v.position.z - ego_z
        longi = dx * ego_fwd_x + dz * ego_fwd_z
        lat = dx * (-ego_fwd_z) + dz * ego_fwd_x
        if not (0.0 < longi <= max_s_m):
            continue
        s_m, offset_m = road.road_coords(longi, lat)
        v_yaw = (
            v._smooth_yaw
            if v._smooth_yaw is not None
            else math.radians(v.rotation.euler()[1])
        )
        # Heading against the road tangent under it, both as ego-frame vectors,
        # so no angle convention has to be reconciled by hand.
        heading = v_yaw - ego_yaw_rad
        fwd_x, fwd_y = math.cos(heading), -math.sin(heading)
        tan_x, tan_y = road.tangent_at(s_m)
        members.append(SlowVehicle(
            vid=v.id, s_m=s_m, offset_m=offset_m,
            heading_err_rad=math.atan2(
                tan_x * fwd_y - tan_y * fwd_x,
                tan_x * fwd_x + tan_y * fwd_y,
            ),
        ))
    return corroborated_evidence(members, road.confidence > 0.0)


def corroborated_evidence(
    members: list[SlowVehicle],
    road_anchored: bool,
) -> dict[int, float]:
    """Evidence each slow vehicle earns from the others lined up with it.

    Returns only the vehicles that earned something, so a lone one is absent and
    the caller's existing evidence stands untouched."""
    aligned = [
        m for m in members
        if abs(math.degrees(m.heading_err_rad)) <= _ALIGN_TO_ROAD_DEG
    ]
    if len(aligned) < _MIN_MEMBERS:
        return {}
    out: dict[int, float] = {}
    for cluster in _clusters(aligned):
        if not _is_line(cluster):
            continue
        earned = _evidence_for(len(cluster), road_anchored)
        for member in cluster:
            out[member.vid] = earned
    return out


# Re-exports for tests and tuning tools.
ALIGN_TO_ROAD_DEG = _ALIGN_TO_ROAD_DEG
PARALLEL_SPREAD_DEG = _PARALLEL_SPREAD_DEG
LANE_SPREAD_M = _LANE_SPREAD_M
MIN_SPAN_M = _MIN_SPAN_M
MIN_MEMBERS = _MIN_MEMBERS
FULL_MEMBERS = _FULL_MEMBERS
MAX_EVIDENCE = _MAX_EVIDENCE
UNANCHORED_MAX_EVIDENCE = _UNANCHORED_MAX_EVIDENCE
