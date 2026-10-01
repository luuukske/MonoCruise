"""ACC ego path: blended steer/history curvature and dynamic half-width.

Blend weights, corridor flare, and AEB vs history κ: ``core/acc/README.md`` §2."""

from __future__ import annotations

import math

from core.radar.traffic import ArcPath, build_arc


# Steer→κ gain (~0.17 legacy); tune if corridor leads/lags at low speed.
_STEER_KAPPA_GAIN: float = 0.17

# Speed thresholds for the steer / history blend (km/h).
_BLEND_LOW_KMH: float = 15.0
_BLEND_HIGH_KMH: float = 30.0

# History weight above _BLEND_HIGH_KMH (legacy kept ~30 % steer at high speed).
_HISTORY_WEIGHT_HIGH: float = 0.70

# Corridor half-width: base lane + steer flare (README §2 path half-width).
LANE_BASE_HALF_M: float = 1.25       # 2.5 m corridor on straight road
LANE_FLARE_HALF_M: float = 2.0       # up to +2.0 m extra per side

# Scoring horizon for the ego arc.  Short on purpose: 2 s @ 90 km/h = 50 m,
# well within the path decay.  AEB uses its own longer horizon.
_HORIZON_S: float = 2.5


def blend_curvature(
    steer: float,
    history_kappa: float | None,
    ego_speed_ms: float,
) -> float:
    """Blend steer κ and ``RadarData.ego_curvature`` by ego speed (km/h ramp)."""
    steer_kappa = steer * _STEER_KAPPA_GAIN
    if history_kappa is None:
        return steer_kappa

    kmh = abs(ego_speed_ms) * 3.6
    if kmh <= _BLEND_LOW_KMH:
        w_hist = 0.0
    elif kmh >= _BLEND_HIGH_KMH:
        w_hist = _HISTORY_WEIGHT_HIGH
    else:
        frac = (kmh - _BLEND_LOW_KMH) / (_BLEND_HIGH_KMH - _BLEND_LOW_KMH)
        w_hist = _HISTORY_WEIGHT_HIGH * frac

    return (1.0 - w_hist) * steer_kappa + w_hist * history_kappa


def path_half_width(steer: float) -> float:
    """Dynamic half-width corridor (m) from current steering input."""
    flare = math.sin(min(abs(steer) * 1.5, 1.0) * (math.pi * 0.5))
    return LANE_BASE_HALF_M + flare * LANE_FLARE_HALF_M


def build_ego_arc(
    ego_x: float,
    ego_z: float,
    ego_yaw_rad: float,
    ego_speed_ms: float,
    steer: float,
    history_kappa: float | None,
) -> ArcPath:
    """Scoring ego arc; blinker bias is applied in scoring, not here (README §5)."""
    kappa = blend_curvature(steer, history_kappa, ego_speed_ms)
    half_w = path_half_width(steer)

    return build_arc(
        ego_x, ego_z, ego_yaw_rad,
        max(abs(ego_speed_ms), 0.1),   # never let speed==0 collapse the arc
        kappa, half_w, _HORIZON_S,
    )

