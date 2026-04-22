"""
ACC ego path prediction — blended curvature + dynamic half-width.

The ego corridor used for scoring is an arc built from:

    curvature   = blend( steer-derived κ, history-derived κ, by speed )
    half_width  = LANE_BASE/2 + sin(|steer·1.5|·π/2) · LANE_FLARE
    horizon     = small fixed window — scoring cares about the next
                  few metres, not AEB's full collision horizon.

Why blend?  At very low speeds the position-history fit collapses to
zero (near-zero chord) and the only reliable signal is the steering
wheel.  Once we are moving well, the steering input leads the actual
trajectory (driver sets the wheel before the truck rotates), so the
history fit gives a more accurate "where I'm actually going" than
steer × gain.  Blend by speed:

    speed ≤ 15 km/h  → 100 % steer
    speed ≥ 30 km/h  → 70 % history + 30 % steer
    linear between.

The steer → curvature gain (``_STEER_KAPPA_GAIN``) is an empirical
small-number; it's not a precise wheel-to-road ratio, it just gives
the corridor a reasonable bend while we're below the history-fit
window.

History-derived curvature comes from RadarThread (``RadarData.ego_curvature``)
— the position-history circle fit shared with Vehicle targets so
"our path" and "their path" are measured the same way.  AEB does
*not* consume that value; see core/aeb/AGENTS.md.
"""

from __future__ import annotations

import math

from core.radar.traffic import ArcPath, build_arc


# Steering-wheel → curvature gain.  ~0.17 matches SCORING_REFERENCE
# legacy value once rewritten in metres.  Tune if the corridor leads /
# lags the actual truck trajectory noticeably at low speed.
_STEER_KAPPA_GAIN: float = 0.17

# Speed thresholds for the steer / history blend (km/h).
_BLEND_LOW_KMH: float = 15.0
_BLEND_HIGH_KMH: float = 30.0

# Weight given to the history fit once we are above _BLEND_HIGH_KMH.
# The legacy logic favoured steering (~30 %) because it leads — this
# keeps that asymmetry.
_HISTORY_WEIGHT_HIGH: float = 0.70

# Dynamic corridor width (m).  Straight driving → LANE_BASE half-width.
# As |steer| grows we fan the corridor wider because (a) the circle
# approximation loses accuracy in tight manoeuvres and (b) scoring
# should tolerate more lateral slop when the truck is turning.
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
    """Combine steer-derived κ and history-derived κ by ego speed.

    steer            — telemetry userSteer, signed, roughly [-1, 1].
    history_kappa    — RadarData.ego_curvature (1/m) or None.
    ego_speed_ms     — current ego speed (m/s).
    """
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
    """Ego corridor arc used for in-path scoring.

    Blinker bias is **not** applied to the arc geometry — legacy
    scoring subtracts ``blinker·4.5`` from the scored lateral offset
    directly (see ``core/acc/scoring.py`` and SCORING_REFERENCE §8.1).
    Shifting the arc would also distort arc-arc hit tests, which is
    not what we want.
    """
    kappa = blend_curvature(steer, history_kappa, ego_speed_ms)
    half_w = path_half_width(steer)

    return build_arc(
        ego_x, ego_z, ego_yaw_rad,
        max(abs(ego_speed_ms), 0.1),   # never let speed==0 collapse the arc
        kappa, half_w, _HORIZON_S,
    )
