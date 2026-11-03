"""Ego body size and path origin from the SDK wheel layout. See core/radar/README.md §17.

Distances run along ego's forward axis from the SDK placement origin (truck
vehicle space: x right, z backward). Pure functions, no telemetry access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Reference rig: vehicle.volvo.fh_2024 6x4, wheel layout read from the SDK on
# 2026-10-04. AEB's body (AEBCalibration ego_half_length/width) was fitted on it.
REF_HALF_LENGTH_M: float = 3.333
REF_HALF_WIDTH_M: float = 1.265
_REF_FRONT_AXLE_Z: float = -1.7925
_REF_LAST_AXLE_Z: float = 2.7678
_REF_MAX_TRACK_HALF_M: float = 1.04
REF_PATH_ORIGIN_M: float = -2.093

# Overhangs and tyre outboard carried over from the reference rig to any truck.
FRONT_OVERHANG_M: float = REF_HALF_LENGTH_M + _REF_FRONT_AXLE_Z
REAR_OVERHANG_M: float = REF_HALF_LENGTH_M - _REF_LAST_AXLE_Z
TYRE_OUTBOARD_M: float = REF_HALF_WIDTH_M - _REF_MAX_TRACK_HALF_M

# A wheel this far behind the front-most one is not on the front axle.
_FRONT_AXLE_BAND_M: float = 1.0
_LIFTED: float = 0.5

# Plausibility bounds; outside them the layout is treated as unreadable.
_MIN_WHEELS: int = 4
_MIN_SPAN_M: float = 2.0
_MAX_SPAN_M: float = 12.0
_MIN_TRACK_HALF_M: float = 0.6
_MAX_TRACK_HALF_M: float = 1.6


@dataclass(frozen=True)
class EgoGeometry:
    half_width_m: float
    front_m: float          # front bumper ahead of the origin
    rear_m: float           # rear end behind the origin
    path_origin_m: float    # path reference ahead of the origin; negative = behind
    from_wheels: bool = True

    @property
    def half_length_m(self) -> float:
        return 0.5 * (self.front_m + self.rear_m)

    @property
    def front_delta_m(self) -> float:
        """How much farther forward the bumper sits than on the reference rig."""
        return self.front_m - REF_HALF_LENGTH_M


def calibration_geometry(
    half_length_m: float, half_width_m: float, arc_start_pctg: float,
) -> EgoGeometry:
    """The fixed body AEB used before wheel data: symmetric, path at arc_start_pctg."""
    return EgoGeometry(
        half_width_m=half_width_m,
        front_m=half_length_m,
        rear_m=half_length_m,
        path_origin_m=(arc_start_pctg - 0.5) * 2.0 * half_length_m,
        from_wheels=False,
    )


def _finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def estimate_ego_geometry(
    xs: list[float],
    zs: list[float],
    steerable: list[bool],
    lift: list[float] | None = None,
) -> EgoGeometry | None:
    """Body and path origin from per-wheel vehicle-space positions, or None if implausible.

    The path origin is the mean of the grounded non-steered wheels: the point
    whose velocity follows the heading, so an arc launched there has no sideslip.
    """
    n = min(len(xs), len(zs), len(steerable))
    if n < _MIN_WHEELS:
        return None
    xs, zs, steerable = list(xs[:n]), list(zs[:n]), list(steerable[:n])
    lifted = [False] * n
    if lift is not None and len(lift) >= n:
        lifted = [float(v) > _LIFTED for v in lift[:n]]
    if not _finite(*xs, *zs):
        return None

    z_front = min(zs)
    z_last = max(zs)
    span = z_last - z_front
    track_half = max(abs(x) for x in xs)
    if not (_MIN_SPAN_M <= span <= _MAX_SPAN_M):
        return None
    if not (_MIN_TRACK_HALF_M <= track_half <= _MAX_TRACK_HALF_M):
        return None

    if any(steerable):
        rear = [i for i in range(n) if not steerable[i]]
    else:
        # No steer flags published: everything off the front axle is the rear group.
        rear = [i for i in range(n) if zs[i] - z_front > _FRONT_AXLE_BAND_M]
    if not rear:
        return None
    grounded = [i for i in rear if not lifted[i]] or rear
    z_path = sum(zs[i] for i in grounded) / len(grounded)
    if z_path - z_front < _FRONT_AXLE_BAND_M:
        return None

    return EgoGeometry(
        half_width_m=track_half + TYRE_OUTBOARD_M,
        front_m=-z_front + FRONT_OVERHANG_M,
        rear_m=z_last + REAR_OVERHANG_M,
        path_origin_m=-z_path,
    )


def geometry_from_sdk(raw: dict) -> EgoGeometry | None:
    """``estimate_ego_geometry`` on a ``truck_telemetry`` dict; None when fields are absent."""
    try:
        n = int(raw.get("truckWheelCount", 0) or 0)
        xs = raw.get("truckWheelPositionX")
        zs = raw.get("truckWheelPositionZ")
        steer = raw.get("truckWheelSteerable")
        lift = raw.get("truck_wheelLift")
        if n <= 0 or xs is None or zs is None or steer is None:
            return None
        return estimate_ego_geometry(
            [float(v) for v in xs[:n]],
            [float(v) for v in zs[:n]],
            [bool(v) for v in steer[:n]],
            [float(v) for v in lift[:n]] if lift is not None else None,
        )
    except (TypeError, ValueError):
        return None


# Attached trailers: the SDK gives their pose, kingpin and wheels, not their body, so the ends are estimated.
TRAILER_KINGPIN_SETBACK_M: float = 1.3
TRAILER_HALF_WIDTH_M: float = 1.275
_TRAILER_REAR_SHARE: float = 0.35
_TRAILER_REAR_MIN_M: float = 1.0
_TRAILER_REAR_MAX_M: float = 3.5
_TRAILER_MIN_SPAN_M: float = 1.0
_TRAILER_MAX_SPAN_M: float = 16.0
_TRAILER_MAX_OFFSET_M: float = 60.0


@dataclass(frozen=True)
class TrailerBody:
    """One attached trailer's body in the tractor's frame, from the SDK; see core/radar/README.md §17.

    ``yaw_rel_rad`` is its heading minus the tractor's, in ``ego_yaw_rad``'s sense. ``hitch_gap_m`` is how far
    its kingpin sits from the tractor's fifth wheel (first trailer only, NaN otherwise): a check on the reading.
    """

    ahead_m: float
    right_m: float
    yaw_rel_rad: float
    half_length_m: float
    half_width_m: float = TRAILER_HALF_WIDTH_M
    hitch_gap_m: float = math.nan


def _forward(yaw_norm: float) -> tuple[float, float]:
    yaw = yaw_norm * 2.0 * math.pi
    return -math.sin(yaw), -math.cos(yaw)


def _trailer_body(unit: dict, origin: tuple[float, float, float], fifth_wheel) -> TrailerBody | None:
    n = int(unit.get("wheelCount", 0) or 0)
    zs = [float(v) for v in (unit.get("wheelPositionZ") or ())[:n]]
    wx, wz, yaw_norm = float(unit["worldX"]), float(unit["worldZ"]), float(unit["rotationX"])
    hook_x, hook_z = float(unit.get("hookPositionX", 0.0)), float(unit.get("hookPositionZ", 0.0))
    if not zs or not _finite(wx, wz, yaw_norm, hook_x, hook_z, *zs):
        return None
    # Trailer space runs z backward, like the truck's.
    kingpin, last_axle = -hook_z, -max(zs)
    span = kingpin - last_axle
    if not (_TRAILER_MIN_SPAN_M <= span <= _TRAILER_MAX_SPAN_M):
        return None
    rear = last_axle - min(_TRAILER_REAR_MAX_M, max(_TRAILER_REAR_MIN_M, _TRAILER_REAR_SHARE * span))
    front = kingpin + TRAILER_KINGPIN_SETBACK_M
    tfx, tfz = _forward(yaw_norm)
    centre = 0.5 * (front + rear)
    cx, cz, truck_yaw = origin
    fx, fz = _forward(truck_yaw)
    dx, dz = wx + centre * tfx - cx, wz + centre * tfz - cz
    ahead, right = dx * fx + dz * fz, dx * -fz + dz * fx
    if math.hypot(ahead, right) > _TRAILER_MAX_OFFSET_M:
        return None
    rel = (yaw_norm - truck_yaw) * 2.0 * math.pi
    rel = math.atan2(math.sin(rel), math.cos(rel))
    gap = math.nan
    if fifth_wheel is not None:
        px, pz = wx + kingpin * tfx + hook_x * -tfz, wz + kingpin * tfz + hook_x * tfx
        gap = math.hypot(px - fifth_wheel[0], pz - fifth_wheel[1])
    return TrailerBody(ahead, right, rel, 0.5 * (front - rear), hitch_gap_m=gap)


def trailer_bodies_from_sdk(raw: dict) -> tuple[TrailerBody, ...]:
    """The leading attached trailers with wheels (as ``trailer_count``), in slot order; unreadable ones left out."""
    try:
        cx, cz, yaw = float(raw["coordinateX"]), float(raw["coordinateZ"]), float(raw["rotationX"])
        units = list(raw.get("trailer") or ())
    except (KeyError, TypeError, ValueError):
        return ()
    if not _finite(cx, cz, yaw):
        return ()
    fx, fz = _forward(yaw)
    try:
        hz, hx = float(raw.get("truckHookPositionZ", 0.0)), float(raw.get("truckHookPositionX", 0.0))
        fifth_wheel = (cx - hz * fx + hx * -fz, cz - hz * fz + hx * fx)
    except (TypeError, ValueError):
        fifth_wheel = None
    out: list[TrailerBody] = []
    first = True
    for unit in units:
        try:
            if not unit.get("attached", False) or int(unit.get("wheelCount", 0) or 0) <= 0:
                break
            body = _trailer_body(unit, (cx, cz, yaw), fifth_wheel if first else None)
        except (AttributeError, KeyError, TypeError, ValueError):
            body = None
        first = False
        if body is not None:
            out.append(body)
    return tuple(out)
