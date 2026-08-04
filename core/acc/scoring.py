"""ACC in-lane score components and per-id accumulation (meter-native).

Formulas, baselines, and dt scaling: ``core/acc/README.md`` §3."""

from __future__ import annotations

import math
from dataclasses import dataclass


# Score clamp: asymmetric so lock is fast and unlock is slow. The ceiling is
# capped just above the consumer's confidence saturation (README §3).
_SCORE_MIN: float = -5.0
_SCORE_MAX: float = 8.0

# Offset Gaussian width (metres).  σ = 2.25 m legacy; the
# zero-crossing of ``offset_raw`` sits at |x| ≈ 2.58 m at angle_amp = 1.
_OFFSET_SIGMA_M: float = 2.25

# Path decay base b^(-d m). Legacy 1.03 left a distant in-lane lead accumulating
# at about an eighth of its close-range rate. See README §3 accumulation.
_PATH_DECAY_BASE: float = 1.022

# Slow-speed path amplifier: ``slow_amp = 1.4 + (kmh / 100) × 4.1``.
# Reference table in SCORING_REFERENCE §8.3.2.
_SLOW_AMP_BASE: float = 1.4
_SLOW_AMP_SLOPE: float = 4.1
_SLOW_AMP_REF_KMH: float = 100.0

# In-/out-of-path caps.  Reference: ``min(base, 5.0)`` and
# ``-min(base × 0.6, 4.0)``.
_PATH_IN_CAP: float = 5.0
_PATH_OUT_CAP: float = 4.0
_PATH_OUT_GAIN: float = 0.6

# Yaw component shape.  Kept legacy.
_YAW_SCALE_DEG: float = 90.0
_YAW_POWER: float = 5.0
_YAW_GAIN: float = 1.5

# Offset baseline increments (legacy §8.1).
OFFSET_BASELINE_HIT: float = 0.0
OFFSET_BASELINE_NO_ARC_HIT: float = -0.40
OFFSET_BASELINE_NO_HISTORY: float = -0.16

# Target-speed multiplier ref 90 m/s (legacy), floor 0.5 (README §3).
_SPEED_MULT_REF_MS: float = 90.0
_SPEED_MULT_EXP: float = 0.8
_SPEED_MULT_FLOOR: float = 0.5

# Legacy tick rate.  :func:`accumulate` scales its dt input by this so
# at 10 Hz the per-frame delta matches the legacy integer-tick maths.
_LEGACY_RATE_HZ: float = 10.0

# Boost offset vs path when arc intersection misses a real in-lane lead.
_OFFSET_WEIGHT: float = 1.5

# Path weight: scales ego-path intersection relative to offset/yaw.
# Reduced below 1.0 so the path component reacts slower to arc changes.
_PATH_WEIGHT: float = 0.7


@dataclass(slots=True)
class ScoreComponents:
    """Per-frame contribution to a track's score (pre speed / dt scaling)."""
    offset: float = 0.0
    yaw: float = 0.0
    path: float = 0.0
    angle: float = 0.0   # reserved: currently always 0.0

    def total(self) -> float:
        return self.offset + self.yaw + self.path + self.angle


def _distance_amp(dist_m: float) -> float:
    """Legacy offset distance amp; see README §3 offset table."""
    d = max(dist_m, 0.0)
    return (math.pow(2.0, -d / 100.0) + 8.0 / (d + 3.0) - 1.0) / 3.0 + 1.0


def offset_component(
    offset_m: float,
    dist_m: float,
    angle_amp: float = 1.0,
    baseline: float = OFFSET_BASELINE_HIT,
    evidence: float = 1.0,
    angle_evidence: float | None = None,
) -> float:
    """Gaussian offset on blinker-adjusted lateral, scaled by evidence (README §3).

    ``evidence`` is how well the target's lateral position is known and scales the
    whole term. ``angle_evidence`` is how well its heading is known and gates the
    arrival-angle penalty only; it defaults to ``evidence``. The road model supplies
    position knowledge for targets that have no trail of their own."""
    ev = max(0.0, min(1.0, evidence))
    if ev <= 0.0:
        return 0.0
    ang_ev = ev if angle_evidence is None else max(0.0, min(1.0, angle_evidence))
    # A short trail measures its own arrival angle badly, so the angle penalty
    # regresses to neutral with the evidence that produced it.
    amp = ang_ev * angle_amp + (1.0 - ang_ev)
    x = offset_m / _OFFSET_SIGMA_M
    gauss = math.exp(-(x * x) * math.log(2.0))         # 2^(-(x/σ)²)
    raw = gauss * 2.5 * amp - 1.0
    clamped = max(-1.0, min(1.0, raw * _distance_amp(dist_m)))
    outer = 1.5 * (amp * 0.4 + 0.6)
    value = baseline + clamped * outer
    if value > 0.0:
        # Calling a target in-lane needs to know it is travelling the lane, not
        # crossing it. Rejecting one only needs to know where it is.
        value *= ang_ev
    return ev * value


def yaw_component(yaw_diff_deg: float) -> float:
    """Heading mismatch penalty (legacy yaw term; README §3)."""
    d = abs(yaw_diff_deg)
    if d > 4.0 * _YAW_SCALE_DEG:
        return -_YAW_GAIN
    ratio = d / _YAW_SCALE_DEG
    return (math.pow(2.0, -math.pow(ratio, _YAW_POWER)) - 1.0) * _YAW_GAIN


def path_component(
    dist_m: float,
    ego_speed_kmh: float,
    in_path: bool,
    blinker_offset: float = 0.0,
    evidence: float = 1.0,
) -> float:
    """Path decay × slow_amp × blinker reduction; in/out caps (README §3).

    ``evidence`` gates the in-corridor reward only. Awarding "it is in my lane"
    needs to know where it is; the out-of-corridor penalty is the conservative
    direction and stays ungated, same asymmetry as ``offset_component``."""
    if dist_m < 0.0:
        dist_m = 0.0
    decay = math.pow(_PATH_DECAY_BASE, -dist_m)        # legacy 1.03^(-d)
    slow_amp = _SLOW_AMP_BASE + (ego_speed_kmh / _SLOW_AMP_REF_KMH) * _SLOW_AMP_SLOPE
    blinker_sq = blinker_offset * blinker_offset
    amp = slow_amp * (1.0 - blinker_sq * 0.4)
    base = decay * amp
    if in_path:
        return min(_PATH_IN_CAP, base) * max(0.0, min(1.0, evidence))
    return -min(_PATH_OUT_CAP, base * _PATH_OUT_GAIN)


def speed_multiplier(target_speed_ms: float) -> float:
    """Target-speed score multiplier (not ego); README §3 accumulation."""
    v = abs(target_speed_ms)
    if v <= 0.0:
        return _SPEED_MULT_FLOOR
    ratio = v / _SPEED_MULT_REF_MS
    return max(math.pow(ratio, _SPEED_MULT_EXP), _SPEED_MULT_FLOOR)


def accumulate(
    prev_score: float,
    dt: float,
    components: ScoreComponents,
    target_speed_ms: float,
    path_weight: float = _PATH_WEIGHT,
) -> float:
    """Integrate weighted components; clamp and scale by target speed × dt (README §3)."""
    weighted_total = (
        components.offset * _OFFSET_WEIGHT
        + components.yaw
        + components.path * path_weight
        + components.angle
    )
    delta = weighted_total * speed_multiplier(target_speed_ms) * dt * _LEGACY_RATE_HZ
    return max(_SCORE_MIN, min(_SCORE_MAX, prev_score + delta))


# Re-exports for tests / debug window.
SCORE_MIN = _SCORE_MIN
SCORE_MAX = _SCORE_MAX
IN_PATH_THRESHOLD: float = 0.0
OFFSET_WEIGHT = _OFFSET_WEIGHT
PATH_WEIGHT = _PATH_WEIGHT
LEGACY_RATE_HZ = _LEGACY_RATE_HZ

