"""TMP trailer→tractor sticky lock for ACC LeadInfo kinematics (README §4)."""

from __future__ import annotations

import math

from core.radar.traffic import Vehicle

_TRACTOR_LOCK_LONGI_MIN_M: float = 3.0
_TRACTOR_LOCK_LONGI_MAX_M: float = 16.0
_TRACTOR_LOCK_LAT_MAX_M: float = 1.5
_TRACTOR_LOCK_YAW_MAX_DEG: float = 15.0
_TRACTOR_LOCK_VALID_LONGI_MIN_M: float = 1.0
_TRACTOR_LOCK_VALID_LONGI_MAX_M: float = 25.0
_TRACTOR_LOCK_VALID_LAT_MAX_M: float = 4.0
_TRACTOR_LOCK_VALID_YAW_MAX_DEG: float = 60.0
TRAILER_VEHICLE_ID_BASE: int = 1_000_000


def _trailer_local_frame(
    trailer: Vehicle, other: Vehicle,
) -> tuple[float, float, float]:
    trailer_yaw = (
        trailer._smooth_yaw
        if trailer._smooth_yaw is not None
        else math.radians(trailer.rotation.euler()[1])
    )
    fwd_x = -math.sin(trailer_yaw)
    fwd_z = -math.cos(trailer_yaw)
    dx = other.position.x - trailer.position.x
    dz = other.position.z - trailer.position.z
    longi = dx * fwd_x + dz * fwd_z
    lat = dx * (-fwd_z) + dz * fwd_x
    other_yaw = (
        other._smooth_yaw
        if other._smooth_yaw is not None
        else math.radians(other.rotation.euler()[1])
    )
    yaw_delta = math.degrees(
        (other_yaw - trailer_yaw + math.pi) % (2.0 * math.pi) - math.pi
    )
    return longi, lat, yaw_delta


def _passes_strict(longi: float, lat: float, yaw_delta_deg: float) -> bool:
    return (
        _TRACTOR_LOCK_LONGI_MIN_M <= longi <= _TRACTOR_LOCK_LONGI_MAX_M
        and abs(lat) <= _TRACTOR_LOCK_LAT_MAX_M
        and abs(yaw_delta_deg) <= _TRACTOR_LOCK_YAW_MAX_DEG
    )


def _passes_loose(longi: float, lat: float, yaw_delta_deg: float) -> bool:
    return (
        _TRACTOR_LOCK_VALID_LONGI_MIN_M <= longi <= _TRACTOR_LOCK_VALID_LONGI_MAX_M
        and abs(lat) <= _TRACTOR_LOCK_VALID_LAT_MAX_M
        and abs(yaw_delta_deg) <= _TRACTOR_LOCK_VALID_YAW_MAX_DEG
    )


def resolve_tractor(
    trailer: Vehicle,
    vehicles: list[Vehicle],
    cache: dict[int, int],
) -> Vehicle | None:
    """Sticky TMP tractor for a trailer; strict acquire, loose revalidate."""
    cached_id = cache.get(trailer.id)
    if cached_id is not None:
        cached = next((o for o in vehicles if o.id == cached_id), None)
        if cached is not None and cached.is_tmp and not cached.is_trailer:
            longi, lat, yaw_delta = _trailer_local_frame(trailer, cached)
            if _passes_loose(longi, lat, yaw_delta):
                return cached
        cache.pop(trailer.id, None)

    best: Vehicle | None = None
    best_cost = math.inf
    for other in vehicles:
        if other.id == trailer.id:
            continue
        if not other.is_tmp or other.is_trailer:
            continue
        longi, lat, yaw_delta = _trailer_local_frame(trailer, other)
        if not _passes_strict(longi, lat, yaw_delta):
            continue
        cost = abs(lat) + 0.05 * abs(longi - 10.0) + 0.2 * abs(yaw_delta)
        if cost < best_cost:
            best_cost = cost
            best = other
    if best is not None:
        cache[trailer.id] = best.id
    return best
