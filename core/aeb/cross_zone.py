"""Perpendicular capsule halo. See core/aeb/README.md (perpendicular body buffer)."""

from __future__ import annotations

import math
from dataclasses import replace

from core.radar.traffic import ArcPath
from core.aeb.calibration import AEBCalibration, DEFAULT as _CAL_DEFAULT


def _cross_zone_padding(ego_yaw_rad: float, v_yaw_rad: float, v_speed_ms: float,
                        cal: AEBCalibration) -> float:
    """Along-track halo; peaks at 90 deg heading."""
    cross_factor = abs(math.sin(ego_yaw_rad - v_yaw_rad))
    return cross_factor * (cal.cross_zone_base + cal.cross_zone_speed * v_speed_ms)


def _apply_cross_zone(arc: ArcPath, padding: float,
                      cal: AEBCalibration | None = None) -> list[ArcPath]:
    """Expand the capsule (radial halo plus along-track extra). Occupancy sees it too."""
    if padding <= 1e-9:
        return [arc]
    cfg = _CAL_DEFAULT if cal is None else cal
    denom = cfg.cross_zone_base + cfg.cross_zone_speed * arc.speed
    radial = (padding * (cfg.cross_zone_radial / denom)
              if denom > 1e-9 else cfg.cross_zone_radial)
    # Do not .build(): that re-derives fwd from yaw and drops reverse travel.
    out = replace(arc, half_width=arc.half_width + radial,
                  fwd_len=arc.fwd_len + padding, back_len=arc.back_len + padding)
    out._has_body = True
    out._cap_fwd = max(out.fwd_len - out.half_width, 0.0)
    out._cap_back = max(out.back_len - out.half_width, 0.0)
    return [out]
