"""Geometric lead failsafe: republishes a dead-ahead lead the scorer lost.

Runs on the raw ego arc alone, so a wrong road model cannot suppress it.
Gates, reasons, and the authority ramp: ``core/acc/README.md`` §10."""

from __future__ import annotations

import math
from dataclasses import dataclass


# Range cap. Past ~70 m no estimator resolves a lane width (README §9), and the
# ego arc alone is only competitive with the blend well inside that.
_MAX_DIST_M: float = 50.0

# Corridor the body has to overlap on the raw ego arc. Tighter than the scored
# corridor and never flared: this rescues dead-ahead traffic only.
_CORRIDOR_HALF_M: float = 1.10
# Second, independent statement: the body's centre is in front of ego, not just
# overlapping the edge of the band. An adjacent lane centre sits 3.5 m or more out.
_CENTER_HALF_M: float = 1.60
_MAX_YAW_DEG: float = 12.0

# Reasons. One must hold, or nothing depends on this lead and it is not rescued.
_CLOSE_M: float = 30.0
_CLOSING_MS: float = 1.5
_RECENT_LEAD_S: float = 2.0
_SAFETY_TTC_S: float = 5.0

# Confirm before any authority is granted; short when the scorer had the lead
# recently or the closing time is already short.
_CONFIRM_S: float = 0.5
_CONFIRM_FAST_S: float = 0.2
_URGENT_TTC_S: float = 3.0

# Authority ramp, so entry and exit are never a step in the consumer's
# confidence. Release is slower than entry: dropping a real lead is the costlier
# direction.
_RAMP_UP_S: float = 0.3
_RAMP_DOWN_S: float = 0.7

# Gap measurement matches the controller: bumper, not vehicle origin.
_EGO_FRONT_OFFSET_M: float = 2.5
_MIN_VCLOSE_MS: float = 0.3

_SCORE_FLOOR_FALLBACK: float = 5.0
_score_floor_cache: float | None = None


def score_floor() -> float:
    """Score a fully ramped rescue publishes, at the consumer's saturation.

    Below it the rescue would publish a lead the controller gives no authority,
    which is the same as not publishing at all (README §10)."""
    global _score_floor_cache
    if _score_floor_cache is None:
        try:
            from core.cruise_control_thread.acc_controller import ANT_SCORE_FULL

            _score_floor_cache = float(ANT_SCORE_FULL)
        except (ImportError, TypeError, ValueError, AttributeError):
            _score_floor_cache = _SCORE_FLOOR_FALLBACK
    return _score_floor_cache


@dataclass(slots=True)
class FailsafeState:
    """Per-track confirm timer and authority ramp; lives on ``TrackState``."""
    confirm_s: float = 0.0
    authority: float = 0.0
    reason: str = ""


@dataclass(slots=True, frozen=True)
class FailsafeInputs:
    """One frame of raw geometry for one target. Laterals are ego-arc only."""
    dist_m: float
    body_lat_min: float
    body_lat_max: float
    lat_uncertainty_m: float
    yaw_diff_deg: float
    ego_speed_ms: float
    lead_speed_ms: float
    desired_th_s: float
    recently_led: bool
    suppressed: bool


def _gap_m(inp: FailsafeInputs) -> float:
    return max(0.0, inp.dist_m - _EGO_FRONT_OFFSET_M)


def _ttc_s(inp: FailsafeInputs) -> float:
    v_close = inp.ego_speed_ms - inp.lead_speed_ms
    if v_close <= _MIN_VCLOSE_MS:
        return math.inf
    return _gap_m(inp) / v_close


def _headway_s(inp: FailsafeInputs) -> float:
    return _gap_m(inp) / max(inp.ego_speed_ms, 0.5)


def geometry_ok(inp: FailsafeInputs) -> bool:
    """Dead ahead on the raw ego arc, aligned, and inside the range cap."""
    if not math.isfinite(inp.dist_m) or inp.dist_m > _MAX_DIST_M:
        return False
    if abs(inp.yaw_diff_deg) > _MAX_YAW_DEG:
        return False
    if abs(0.5 * (inp.body_lat_min + inp.body_lat_max)) > _CENTER_HALF_M:
        return False
    sigma = max(0.0, inp.lat_uncertainty_m)
    near = _CORRIDOR_HALF_M - (inp.body_lat_min + sigma)
    far = (inp.body_lat_max - sigma) + _CORRIDOR_HALF_M
    return near >= 0.0 and far >= 0.0


def reason_for(inp: FailsafeInputs, ttc_s: float) -> str:
    """Why this lead matters, or "" when nothing depends on it."""
    if inp.recently_led:
        return "tracked"
    if inp.dist_m <= _CLOSE_M:
        return "close"
    if inp.ego_speed_ms - inp.lead_speed_ms >= _CLOSING_MS:
        return "closing"
    if ttc_s <= _SAFETY_TTC_S or _headway_s(inp) <= inp.desired_th_s:
        return "safety"
    return ""


def failsafe_step(state: FailsafeState, inp: FailsafeInputs, dt: float) -> float:
    """Advance one track's rescue and return the score floor it publishes."""
    ttc_s = _ttc_s(inp)
    reason = ""
    if not inp.suppressed and geometry_ok(inp):
        reason = reason_for(inp, ttc_s)
    if not reason:
        return failsafe_decay(state, dt)

    state.reason = reason
    state.confirm_s += dt
    need = _CONFIRM_FAST_S if (inp.recently_led or ttc_s <= _URGENT_TTC_S) else _CONFIRM_S
    if state.confirm_s < need:
        # Confirming is not confirmed. Releasing here rather than holding is what
        # stops a target flickering across the corridor edge from freezing
        # authority part way up instead of letting go of it.
        return _release(state, dt)
    state.authority = min(1.0, state.authority + dt / _RAMP_UP_S)
    return state.authority * score_floor()


def failsafe_decay(state: FailsafeState, dt: float) -> float:
    """Release ramp for a track whose gates failed or that is no longer seen."""
    state.confirm_s = 0.0
    return _release(state, dt)


def _release(state: FailsafeState, dt: float) -> float:
    state.authority = max(0.0, state.authority - dt / _RAMP_DOWN_S)
    if state.authority <= 0.0:
        state.reason = ""
    return state.authority * score_floor()


# Re-exports for the tracker, tests, and tuning tools.
MAX_DIST_M = _MAX_DIST_M
CORRIDOR_HALF_M = _CORRIDOR_HALF_M
CENTER_HALF_M = _CENTER_HALF_M
MAX_YAW_DEG = _MAX_YAW_DEG
RECENT_LEAD_S = _RECENT_LEAD_S
CONFIRM_S = _CONFIRM_S
CONFIRM_FAST_S = _CONFIRM_FAST_S
RAMP_UP_S = _RAMP_UP_S
RAMP_DOWN_S = _RAMP_DOWN_S
