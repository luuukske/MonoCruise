"""
ETS2/ATS traffic vehicle classes with arc-based path prediction.

Coordinate system and yaw conventions: see ``core/aeb/AGENTS.md`` §1–§3.
Shared between AEB and ACC (both consume the same Vehicle instances produced
by RadarThread).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

_MAX_ANGULAR_VELOCITY: float = 45.0
_LOCATION_UPDATE_FREQUENCY: float = 0.05

# TMP speed / accel EMA: same hyperbolic law α(|v|) with different endpoints.
# Reference speed for "at 90 km/h" is 25 m/s. See AGENTS.md §7.
_ALPHA_SPEED_SCALE: float = 90.0 / 3.6   # 25.0 m/s

# Speed EMA on raw_speed: 1.0 at rest → 0.25 at 90 km/h.
_SPEED_EMA_AT_REST: float = 1.0
_SPEED_EMA_AT_90_KMH: float = 0.25
_SPEED_EMA_CURVE_D: float = (
    _ALPHA_SPEED_SCALE
    * _SPEED_EMA_AT_90_KMH
    / (_SPEED_EMA_AT_REST - _SPEED_EMA_AT_90_KMH)
)

# Accel filter: least-squares slope of recent speed_ema samples over
# _ACCEL_FIT_WINDOW_S (low-noise derivative), then a light EMA. Feeds the
# responsive accel (AEB, speed_corr). See AGENTS.md §7.
_ACCEL_FIT_WINDOW_S: float = 0.70
_ACCEL_EMA_ALPHA: float = 0.45
# Holds (t, speed_ema) samples for the LS slope fits: `accel` over
# _ACCEL_FIT_WINDOW_S and `accel_trend` over _ACC_SPEED_ACCEL_WINDOW_S.
# 120 ≈ 6 s of headroom at full-update rate.
_SPEED_EMA_HISTORY_LEN: int = 120

# Accel-correction term clamp (m/s): caps how far accel·τ can shift a speed.
_SPEED_CORR_CLAMP_MS: float = 3.0

# ACC speed (acc_speed): adaptive filter on speed_corr. A per-tick change
# below _ACC_SPEED_DEADBAND_MS is treated as noise and filtered with the long
# _ACC_SPEED_TAU_SLOW_S time constant; the time constant ramps continuously
# down toward _ACC_SPEED_TAU_FAST_S as the change grows, reaching it at twice
# the deadband, so real motion is tracked fast. See AGENTS.md §7.
_ACC_SPEED_DEADBAND_MS: float = 0.7
_ACC_SPEED_TAU_SLOW_S: float = 2.0
_ACC_SPEED_TAU_FAST_S: float = 0.08

# Smoothing is speed-scaled: tau is multiplied by speed_factor: 1.0 at
# _ACC_SPEED_SMOOTH_REF_MS (90 km/h) and above, falling linearly to
# _ACC_SPEED_SMOOTH_MIN at rest, so at low speed acc_speed leans on the
# incoming speed_corr instead of smoothing it. See AGENTS.md §7.
_ACC_SPEED_SMOOTH_REF_MS: float = 90.0 / 3.6   # 25 m/s
_ACC_SPEED_SMOOTH_MIN: float = 0.15

# ...and acceleration-scaled. accel_trend is the de-noised acceleration: the
# LS slope of speed_ema over _ACC_SPEED_ACCEL_WINDOW_S, where coasting and
# cruise wobble average to ≈ 0 but a steady ramp registers its true rate.
# accel_factor multiplies tau: 1.0 while coasting, falling to
# _ACC_SPEED_ACCEL_FLOOR once |accel_trend| reaches _ACC_SPEED_ACCEL_HI_MS2.
# Below _ACC_SPEED_ACCEL_LO_MS2 a ramp counts as cruise noise and is ignored,
# so cruise smoothing is untouched. See AGENTS.md §7.
_ACC_SPEED_ACCEL_WINDOW_S: float = 1.5
_ACC_SPEED_ACCEL_LO_MS2: float = 0.3
_ACC_SPEED_ACCEL_HI_MS2: float = 1.5
_ACC_SPEED_ACCEL_FLOOR: float = 0.35

# Feed-forward: the low-pass alone lags a steady ramp by accel·tau. acc_speed
# is predicted one step along the responsive accel before the low-pass corrects
# the residual, which cancels that lag analytically (no windup, no overshoot).
# The prediction is gated by ff_gate, a ramp on the de-noised |accel_trend|:
# zero below _ACC_SPEED_FF_GATE_LO_MS2, full at _ACC_SPEED_FF_GATE_HI_MS2. Both
# sit just above the cruise-noise floor (|accel_trend| ≈ 0.06 m/s² while
# coasting) so the prediction is exactly zero while coasting: no cruise noise
# is fed forward: yet saturates within a fraction of a real ramp. accel is
# clamped to _ACC_SPEED_FF_ACCEL_CLAMP_MS2 so a crash spike cannot jump
# acc_speed. See AGENTS.md §7.
_ACC_SPEED_FF_GATE_LO_MS2: float = 0.12        # m/s² : feed-forward gate opens
_ACC_SPEED_FF_GATE_HI_MS2: float = 0.30        # m/s² : feed-forward gate fully open
_ACC_SPEED_FF_ACCEL_CLAMP_MS2: float = 6.0     # m/s² : clamp on the feed-forward accel

# Standstill latch on acc_speed: once the filtered speed settles near zero with
# no real de-noised ramp, acc_speed is clamped to exactly 0 and held until
# speed_corr exceeds the release threshold for consecutive full frames. Kills
# residual crash-bounce wobble around zero at standstill. See AGENTS.md §7.
_ACC_SPEED_STANDSTILL_ENTER_MS: float = 0.3    # m/s : |acc_speed| below this can latch
_ACC_SPEED_STANDSTILL_RELEASE_MS: float = 0.6  # m/s : |speed_corr| above this releases
_ACC_SPEED_STANDSTILL_RELEASE_S: float = 0.5   # s : sustained time above release speed

# Yaw EMA (wrap-safe): AI and TMP (arc curvature).
_RAW_YAW_ALPHA: float = 0.50

# TMP lag detection: see AGENTS.md §7 "Lag / freeze detection".
_LAG_MIN_SPEED_MS: float = 5.0           # m/s : below this no lag detection runs
_LAG_DISP_RATIO: float = 0.10           # flag lag if raw disp < 10 % of expected

# Freeze duration scales logarithmically with time-to-vehicle (TTC) so a close
# real stop is not masked by the filter. TTC = 3D distance to ego / ego_speed,
# with ego_speed floored at _LAG_FREEZE_EGO_SPEED_FLOOR. Curve: 0 s at TTC ≤
# _LAG_FREEZE_TTC_LO, _LAG_FREEZE_DUR_MAX at TTC ≥ _LAG_FREEZE_TTC_HI,
# dur = K · ln(ttc / lo) between (K chosen so dur(hi) = max).
_LAG_FREEZE_TTC_LO: float = 0.3                  # s   : freeze = 0 at/below this TTC
_LAG_FREEZE_TTC_HI: float = 4.0                  # s   : freeze = max at/above this TTC
_LAG_FREEZE_DUR_MAX: float = 0.5                 # s   : freeze cap (release after this)
_LAG_FREEZE_EGO_SPEED_FLOOR: float = 1.0         # m/s : TTC denom floor
_LAG_FREEZE_LOG_K: float = _LAG_FREEZE_DUR_MAX / math.log(
    _LAG_FREEZE_TTC_HI / _LAG_FREEZE_TTC_LO
)

# Position mismatch (TMP only): out-of-order packet rejection.
# Fires when raw position jumps backward along heading.  Max 3 consecutive frames.
_POS_MISMATCH_BACKWARD_THRESHOLD: float = 0.00   # m: min backward dot to flag
_POS_MISMATCH_MAX_FRAMES: int = 5

# Crash detection (TMP only): angular jerk vs last sample (every read, not only full frames).
_CRASH_PITCH_JERK: float = 2.0                  # deg/s² pitch angular jerk threshold
_CRASH_YAW_JERK: float = 15.0                   # deg/s² yaw angular jerk threshold
_CRASH_ROLL_JERK: float = 2.0                   # deg/s² roll angular jerk threshold
_CRASH_CONFIRM_DURATION: float = 0.00           # s jerk must hold before confirming

_MIN_CURVATURE_RADIUS: float = 5.0
_STRAIGHT_CURVATURE_EPS: float = 1e-6

# Vehicle position history buffer: newest last, length capped at
# _POSITION_HISTORY_LEN. Shared by:
#   - raw speed LS fit (uses last _RAW_SPEED_HISTORY_LEN samples internally)
#   - curvature_from_history() circumscribed-circle fit (uses full buffer)
#   - ACC trail-arc scoring (needs long history for stable fit).
#
# _POSITION_HISTORY_LEN is the buffer size; _RAW_SPEED_HISTORY_LEN is the
# window the speed LS fit considers: ~1.3 s, long enough to average out
# TMP's ~1 Hz position-reconciliation ripple (see AGENTS.md §7) so the derived
# speed doesn't oscillate. AI uses the same fit; sign comes from the buffer.
_POSITION_HISTORY_LEN: int = 25
_RAW_SPEED_HISTORY_LEN: int = 20
_RAW_SPEED_NEAR_ZERO_CHORD: float = 0.025  # m: same gate as per-frame displacement
_BUFFER_SIGN_SPEED_MS: float = 0.05        # m/s: below this, trust LS sign on AI


def _lag_freeze_duration(gap_3d: float, ego_speed: float) -> float:
    """Logarithmic TTC-scaled freeze window. See AGENTS.md §7 "Lag / freeze detection".

    TTC = gap_3d / max(ego_speed, _LAG_FREEZE_EGO_SPEED_FLOOR). Returns 0 s below
    _LAG_FREEZE_TTC_LO, _LAG_FREEZE_DUR_MAX above _LAG_FREEZE_TTC_HI, and
    K · ln(ttc / lo) between, so a close vehicle's real stop is not masked.
    """
    ttc = gap_3d / max(ego_speed, _LAG_FREEZE_EGO_SPEED_FLOOR)
    if ttc <= _LAG_FREEZE_TTC_LO:
        return 0.0
    if ttc >= _LAG_FREEZE_TTC_HI:
        return _LAG_FREEZE_DUR_MAX
    return _LAG_FREEZE_LOG_K * math.log(ttc / _LAG_FREEZE_TTC_LO)


def _raw_speed_from_position_history(
    history: list[tuple[float, float, float]],
    fwd_x: float,
    fwd_z: float,
) -> float | None:
    """Estimate signed longitudinal speed (m/s) from (t, x, z) samples, oldest first.

    Uses only the last _RAW_SPEED_HISTORY_LEN samples so the LS fit window is
    independent of the total buffer length (which can be longer for curvature
    / ACC trail arc).

    Fits s ≈ v·τ where s = dot(p(τ) − p₀, fwd) and τ = t − t₀. Uniform spacing is
    not required. Returns None if fewer than two samples (caller uses one interval).
    If the first→last chord is below _RAW_SPEED_NEAR_ZERO_CHORD, returns 0.0.
    """
    if len(history) < 2:
        return None
    window = history[-_RAW_SPEED_HISTORY_LEN:] if len(history) > _RAW_SPEED_HISTORY_LEN else history
    t0, x0, z0 = window[0]
    tn, xn, zn = window[-1]
    chord_dx = xn - x0
    chord_dz = zn - z0
    chord = math.sqrt(chord_dx * chord_dx + chord_dz * chord_dz)
    if chord < _RAW_SPEED_NEAR_ZERO_CHORD:
        return 0.0
    num = 0.0
    den = 0.0
    for t, x, z in window:
        tau = t - t0
        if tau <= 1e-9:
            continue
        s = (x - x0) * fwd_x + (z - z0) * fwd_z
        num += tau * s
        den += tau * tau
    if den < 1e-12:
        dt = tn - t0
        if dt < 1e-9:
            return 0.0
        direction = 1.0 if (chord_dx * fwd_x + chord_dz * fwd_z) >= 0.0 else -1.0
        return direction * chord / dt
    return num / den


def _raw_speed_from_kinematics(
    buffer_speed: float,
    position_history: list[tuple[float, float, float]],
    fwd_x: float,
    fwd_z: float,
    prev_raw_x: float,
    prev_raw_z: float,
    prev_y: float,
    raw_x: float,
    raw_z: float,
    raw_y: float,
    dt: float,
    preserve_buffer_sign: bool,
) -> float:
    """Longitudinal raw speed (m/s) feeding the shared filter chain.

    TMP: LS fit on position history (single-interval fallback); sign from fit.
    SP/AI: same fit when history allows; sign from buffer when |buffer| is
    above _BUFFER_SIGN_SPEED_MS so turning vehicles are not misclassified as
    reversing (see AGENTS.md §7 "Speed sign detection").
    """
    _ls = _raw_speed_from_position_history(position_history, fwd_x, fwd_z)
    if _ls is not None:
        if preserve_buffer_sign and abs(buffer_speed) > _BUFFER_SIGN_SPEED_MS:
            return math.copysign(abs(_ls), buffer_speed)
        return _ls
    disp_x = raw_x - prev_raw_x
    disp_z = raw_z - prev_raw_z
    dist = math.sqrt(
        disp_x * disp_x + (raw_y - prev_y) ** 2 + disp_z * disp_z
    )
    if dist > _RAW_SPEED_NEAR_ZERO_CHORD and dt > 1e-9:
        direction = 1.0 if (disp_x * fwd_x + disp_z * fwd_z) >= 0.0 else -1.0
        derived = direction * dist / dt
        if preserve_buffer_sign and abs(buffer_speed) > _BUFFER_SIGN_SPEED_MS:
            return math.copysign(abs(derived), buffer_speed)
        return derived
    return 0.0


def _accel_to_arc_params(accel: float, override_decel: float = 0.0) -> tuple[float, float]:
    """Convert raw vehicle acceleration to (decel, accel) for build_arc().

    - override_decel > 0  (e.g. head-on full brake) → (override_decel, 0.0).
    - accel < 0 (braking) → decel = min(|accel|, 6.0), accel = 0.0.
      Capped at 6 m/s² so crash-induced backward position jumps (which produce
      large negative acceleration spikes) are not mistaken for hard braking.
    - accel >= 0 (accelerating or constant) → decel = 0.0, accel = min(accel, 4.0).
    """
    if override_decel > 0.0:
        return override_decel, 0.0
    if accel < 0.0:
        return min(-accel, 6.0), 0.0
    return 0.0, min(accel, 4.0)


def _tmp_speed_ema_alpha(speed_ms: float) -> float:
    """Weight on the new raw speed sample. 1.0 at rest → 0.25 at 90 km/h."""
    return (_SPEED_EMA_AT_REST * _SPEED_EMA_CURVE_D) / (
        abs(speed_ms) + _SPEED_EMA_CURVE_D
    )


def _accel_from_speed_history(
    history: list[tuple[float, float]],
    window_s: float,
) -> float:
    """Least-squares slope (m/s²) of (t, speed) samples within `window_s`.

    Fits `speed ≈ a·t + b` over samples no older than `window_s`; the slope `a`
    is a low-noise acceleration estimate. Returns 0.0 with < 2 samples.
    """
    if len(history) < 2:
        return 0.0
    t_new = history[-1][0]
    window = [(t, s) for (t, s) in history if t_new - t <= window_s]
    n = len(window)
    if n < 2:
        return 0.0
    t_mean = sum(t for t, _ in window) / n
    s_mean = sum(s for _, s in window) / n
    num = 0.0
    den = 0.0
    for t, s in window:
        dt_c = t - t_mean
        num += dt_c * (s - s_mean)
        den += dt_c * dt_c
    if den < 1e-12:
        return 0.0
    return num / den


def _smooth_vehicle_kinematics(
    raw_speed: float,
    t_now: float,
    dt: float,
    prev_speed_ema: float | None,
    prev_accel: float | None,
    prev_acc_speed: float | None,
    prev_speed_ema_history: list[tuple[float, float]] | None,
    prev_acc_standstill: bool = False,
    prev_acc_release_s: float = 0.0,
) -> tuple[float, float, float, float, list[tuple[float, float]], bool, float]:
    """Speed/accel filter chain shared by AI and TMP vehicles. See AGENTS.md §7.

    Returns (speed_ema, accel, speed_corr, acc_speed, speed_ema_history,
    acc_standstill, acc_release_s).
    """
    # Step 1: plain EMA of raw speed (no lag compensation).
    if prev_speed_ema is None:
        speed_ema = raw_speed
        alpha_s = 1.0
    else:
        alpha_s = _tmp_speed_ema_alpha(abs((prev_speed_ema + raw_speed) * 0.5))
        speed_ema = alpha_s * raw_speed + (1.0 - alpha_s) * prev_speed_ema

    # Step 2: accel = LS slope of the speed_ema history (low-noise derivative),
    # then a light EMA. A windowed least-squares fit averages out the per-sample
    # noise that a tick-to-tick difference would amplify.
    history = list(prev_speed_ema_history) if prev_speed_ema_history else []
    history.append((t_now, speed_ema))
    if len(history) > _SPEED_EMA_HISTORY_LEN:
        history = history[-_SPEED_EMA_HISTORY_LEN:]
    accel_raw = _accel_from_speed_history(history, _ACCEL_FIT_WINDOW_S)
    if prev_accel is None:
        accel = accel_raw
    else:
        accel = prev_accel + _ACCEL_EMA_ALPHA * (accel_raw - prev_accel)

    # Step 3: lag-compensated speed. τ is the step-1 EMA's settling time.
    tau_eff = dt * (1.0 - alpha_s) / alpha_s if alpha_s > 1e-6 else 0.0
    correction = max(-_SPEED_CORR_CLAMP_MS, min(_SPEED_CORR_CLAMP_MS, accel * tau_eff))
    speed_corr = speed_ema + correction

    # Step 4: ACC speed: adaptive low-pass on speed_corr with a constant-accel
    # feed-forward. tau ramps slow→fast with the per-tick change (abrupt jumps)
    # and is scaled down on a confirmed ramp; the feed-forward predicts along
    # the de-noised accel so a steady ramp tracks with ≈ 0 lag. See AGENTS.md §7.
    acc_standstill = prev_acc_standstill
    acc_release_s = 0.0
    if prev_speed_ema is None or prev_acc_speed is None:
        acc_speed = speed_corr
        acc_standstill = False
    else:
        delta = speed_corr - prev_acc_speed
        ramp = (abs(delta) - _ACC_SPEED_DEADBAND_MS) / _ACC_SPEED_DEADBAND_MS
        ramp = max(0.0, min(1.0, ramp))
        # Smoothing scales with speed: full at the reference speed, light at rest.
        speed_factor = _ACC_SPEED_SMOOTH_MIN + (1.0 - _ACC_SPEED_SMOOTH_MIN) * min(
            1.0, abs(speed_corr) / _ACC_SPEED_SMOOTH_REF_MS)
        # ...and with acceleration: a steady ramp is low-noise, track it closely.
        accel_trend = _accel_from_speed_history(history, _ACC_SPEED_ACCEL_WINDOW_S)
        # Fast tau opens only when the residual points the same way as the
        # de-noised trend: a big residual disagreeing with the trend is bounce
        # or a packet snap, not motion, and stays on the slow tau.
        if abs(accel_trend) <= _ACC_SPEED_FF_GATE_LO_MS2 or accel_trend * delta <= 0.0:
            ramp = 0.0
        accel_span = _ACC_SPEED_ACCEL_HI_MS2 - _ACC_SPEED_ACCEL_LO_MS2
        accel_ramp = max(0.0, min(1.0,
            (abs(accel_trend) - _ACC_SPEED_ACCEL_LO_MS2) / accel_span))
        accel_factor = 1.0 - (1.0 - _ACC_SPEED_ACCEL_FLOOR) * accel_ramp
        tau = _ACC_SPEED_TAU_SLOW_S + (_ACC_SPEED_TAU_FAST_S - _ACC_SPEED_TAU_SLOW_S) * ramp
        tau *= speed_factor * accel_factor
        alpha_a = dt / (tau + dt)
        # Feed-forward: predict one step along the responsive accel, then
        # low-pass the residual. Cancels the filter's accel·tau ramp lag with no
        # windup. ff_gate (a ramp on the de-noised |accel_trend|) holds the
        # prediction at zero while coasting, so cruise noise is never fed forward.
        ff_gate = max(0.0, min(1.0,
            (abs(accel_trend) - _ACC_SPEED_FF_GATE_LO_MS2)
            / (_ACC_SPEED_FF_GATE_HI_MS2 - _ACC_SPEED_FF_GATE_LO_MS2)))
        accel_ff = max(-_ACC_SPEED_FF_ACCEL_CLAMP_MS2,
                       min(_ACC_SPEED_FF_ACCEL_CLAMP_MS2, accel))
        predicted = prev_acc_speed + accel_ff * dt * ff_gate
        acc_speed = predicted + alpha_a * (speed_corr - predicted)

        # Standstill latch: clamp acc_speed to 0 once it settles near zero with
        # no real ramp; release on sustained speed_corr (hysteresis, see AGENTS.md §7).
        if acc_standstill:
            if abs(speed_corr) > _ACC_SPEED_STANDSTILL_RELEASE_MS:
                acc_release_s = prev_acc_release_s + dt
            if acc_release_s >= _ACC_SPEED_STANDSTILL_RELEASE_S:
                acc_standstill = False
            else:
                acc_speed = 0.0
        elif (abs(acc_speed) < _ACC_SPEED_STANDSTILL_ENTER_MS
                and abs(accel_trend) < _ACC_SPEED_ACCEL_LO_MS2):
            acc_standstill = True
            acc_speed = 0.0

    return (speed_ema, accel, speed_corr, acc_speed, history,
            acc_standstill, acc_release_s)


class Position:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y, self.z - other.z)

    def tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def is_zero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0 and self.z == 0.0

    def distance_to(self, other: "Position") -> float:
        dx = self.x - other.x
        dz = self.z - other.z
        return math.sqrt(dx * dx + dz * dz)

    def __repr__(self) -> str:
        return f"Position({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Quaternion:
    """ETS2 traffic quaternion: x/y swap is intentional (AGENTS.md §3)."""
    __slots__ = ("w", "x", "y", "z", "_euler_cache")

    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self.w = w
        self.x = y
        self.y = x
        self.z = z
        self._euler_cache: tuple[float, float, float] | None = None

    def euler(self) -> tuple[float, float, float]:
        """(pitch, yaw, roll) in degrees. Cached: quaternion is immutable after init."""
        if self._euler_cache is not None:
            return self._euler_cache
        yaw = math.atan2(
            2.0 * (self.y * self.z + self.w * self.x),
            self.w * self.w - self.x * self.x - self.y * self.y + self.z * self.z,
        )
        pitch = math.asin(
            max(-1.0, min(1.0, -2.0 * (self.x * self.z - self.w * self.y)))
        )
        roll = math.atan2(
            2.0 * (self.x * self.y + self.w * self.z),
            self.w * self.w + self.x * self.x - self.y * self.y - self.z * self.z,
        )
        self._euler_cache = math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
        return self._euler_cache

    def is_zero(self) -> bool:
        return self.w == 0.0 and self.x == 0.0 and self.y == 0.0 and self.z == 0.0

    def __repr__(self) -> str:
        p, y, r = self.euler()
        return f"Quaternion(pitch={p:.1f}, yaw={y:.1f}, roll={r:.1f})"


class Size:
    __slots__ = ("width", "height", "length")

    def __init__(self, width: float, height: float, length: float) -> None:
        self.width = width
        self.height = height
        self.length = length

    def __repr__(self) -> str:
        return f"Size({self.width:.2f}, {self.height:.2f}, {self.length:.2f})"


class Trailer:
    __slots__ = ("position", "rotation", "size", "is_tmp", "slot")

    def __init__(self, position: Position, rotation: Quaternion,
                 size: Size, is_tmp: bool = False, slot: int = -1) -> None:
        self.position = position
        self.rotation = rotation
        self.size = size
        self.is_tmp = is_tmp
        # Buffer trailer-slot index (0..2). Stable across frames so the ACC
        # trailer-as-vehicle wrapper can derive a continuous synthetic id.
        self.slot = slot

    def correct_position(self) -> Position:
        """Shift TMP trailer pivot from front coupler to body center."""
        _, yaw_deg, _ = self.rotation.euler()
        yaw_rad = math.radians(yaw_deg)
        return Position(
            self.position.x + (self.size.length / 2.0) * math.sin(yaw_rad),
            self.position.y,
            self.position.z + (self.size.length / 2.0) * math.cos(yaw_rad),
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()


@dataclass(slots=True)
class ArcPath:
    """Predicted path as a circular arc or straight ray.  See AGENTS.md §8."""
    start_x: float = 0.0
    start_z: float = 0.0
    yaw_rad: float = 0.0
    speed: float = 0.0
    curvature: float = 0.0
    half_width: float = 1.15
    horizon: float = 3.0
    decel: float = 0.0
    accel: float = 0.0

    is_straight: bool = True
    center_x: float = 0.0
    center_z: float = 0.0
    radius: float = 0.0
    angle0: float = 0.0
    max_sweep: float = 0.0
    arc_length: float = 0.0
    fwd_x: float = 0.0
    fwd_z: float = -1.0
    _sign: float = 1.0

    def build(self) -> "ArcPath":
        """Compute cached fields (fwd, radius, center, arc_length, is_straight) from
        start, curvature, speed, decel/accel. Call after setting fields."""
        self.fwd_x = -math.sin(self.yaw_rad)
        self.fwd_z = -math.cos(self.yaw_rad)

        # Reversing: flip fwd to actual travel direction, normalise speed to abs.
        if self.speed < -1e-3:
            self.fwd_x = -self.fwd_x
            self.fwd_z = -self.fwd_z
        self.speed = abs(self.speed)

        if self.speed < 1e-3:
            self.is_straight = True
            self.arc_length = 0.0
            self.max_sweep = 0.0
            return self

        if self.decel > 0.0:
            t_stop = self.speed / self.decel
            if t_stop < self.horizon:
                self.arc_length = self.speed * t_stop - 0.5 * self.decel * t_stop * t_stop
            else:
                t = self.horizon
                self.arc_length = self.speed * t - 0.5 * self.decel * t * t
        elif self.accel < 0.0:
            t_stop = -self.speed / self.accel
            if t_stop < self.horizon:
                self.arc_length = self.speed * t_stop + 0.5 * self.accel * t_stop * t_stop
            else:
                t = self.horizon
                self.arc_length = self.speed * t + 0.5 * self.accel * t * t
        elif self.accel > 0.0:
            t = self.horizon
            self.arc_length = self.speed * t + 0.5 * self.accel * t * t
        else:
            self.arc_length = self.speed * self.horizon

        if abs(self.curvature) < _STRAIGHT_CURVATURE_EPS:
            self.is_straight = True
            self.radius = 0.0
            self.max_sweep = 0.0
        else:
            self.is_straight = False
            self.radius = max(abs(1.0 / self.curvature), _MIN_CURVATURE_RADIUS)

            self._sign = 1.0 if self.curvature > 0 else -1.0
            self.center_x = self.start_x + self._sign * self.radius * self.fwd_z
            self.center_z = self.start_z + self._sign * self.radius * (-self.fwd_x)

            self.angle0 = math.atan2(
                self.start_z - self.center_z,
                self.start_x - self.center_x,
            )
            self.max_sweep = -self._sign * self.arc_length / self.radius

        return self

    def _dist_at_time(self, t: float) -> float:
        """Distance travelled along the path at time t (constant speed, decel, or accel)."""
        if self.decel > 0.0:
            t_stop = self.speed / self.decel
            if t >= t_stop:
                return self.speed * t_stop - 0.5 * self.decel * t_stop * t_stop
            return self.speed * t - 0.5 * self.decel * t * t
        elif self.accel < 0.0:
            t_stop = -self.speed / self.accel
            if t >= t_stop:
                return self.speed * t_stop + 0.5 * self.accel * t_stop * t_stop
            return self.speed * t + 0.5 * self.accel * t * t
        elif self.accel > 0.0:
            return self.speed * t + 0.5 * self.accel * t * t
        return self.speed * t

    def position_at_dist(self, dist: float) -> tuple[float, float]:
        """(x, z) at distance along the centerline (straight segment or arc)."""
        dist = max(0.0, min(dist, self.arc_length))
        if self.is_straight:
            return (
                self.start_x + dist * self.fwd_x,
                self.start_z + dist * self.fwd_z,
            )
        frac = dist / self.arc_length if self.arc_length > 0 else 0.0
        angle = self.angle0 + frac * self.max_sweep
        return (
            self.center_x + self.radius * math.cos(angle),
            self.center_z + self.radius * math.sin(angle),
        )

    def position_at_time(self, t: float) -> tuple[float, float]:
        """(x, z) at time t along the path (via _dist_at_time)."""
        return self.position_at_dist(self._dist_at_time(t))

    def heading_at_dist(self, dist: float) -> float:
        """Heading (yaw_rad) at distance along the path."""
        if self.is_straight:
            return self.yaw_rad
        dist = max(0.0, min(dist, self.arc_length))
        frac = dist / self.arc_length if self.arc_length > 0 else 0.0
        return self.yaw_rad + frac * self.max_sweep

    def sample_points(self, n: int = 16) -> list[tuple[float, float]]:
        """n evenly spaced (x, z) points along the centerline."""
        if n < 2 or self.arc_length < 1e-6:
            return [(self.start_x, self.start_z)]
        pts = []
        for i in range(n):
            d = self.arc_length * i / (n - 1)
            pts.append(self.position_at_dist(d))
        return pts

    def sample_corridor(self, n: int = 16) -> tuple[
        list[tuple[float, float]], list[tuple[float, float]]
    ]:
        """Left and right boundary point lists for the path corridor (half_width)."""
        if n < 2 or self.arc_length < 1e-6:
            return [(self.start_x, self.start_z)], [(self.start_x, self.start_z)]

        if self.is_straight:
            left = []
            right = []
            for i in range(n):
                d = self.arc_length * i / (n - 1)
                x, z = self.position_at_dist(d)
                h = self.heading_at_dist(d)
                rx = -math.cos(h)
                rz = math.sin(h)
                left.append((x - rx * self.half_width, z - rz * self.half_width))
                right.append((x + rx * self.half_width, z + rz * self.half_width))
            return left, right

        r_inner = max(self.radius - self.half_width, 0.5)
        r_outer = self.radius + self.half_width
        left = []
        right = []
        for i in range(n):
            frac = i / (n - 1) if n > 1 else 1.0
            angle = self.angle0 + frac * self.max_sweep
            cx, cz = self.center_x, self.center_z
            c, s = math.cos(angle), math.sin(angle)
            inner_pt = (cx + r_inner * c, cz + r_inner * s)
            outer_pt = (cx + r_outer * c, cz + r_outer * s)
            if self._sign > 0:  # left turn: left = inner, right = outer
                left.append(inner_pt)
                right.append(outer_pt)
            else:  # right turn: left = outer, right = inner
                left.append(outer_pt)
                right.append(inner_pt)
        return left, right


def build_arc(
    x: float, z: float, yaw_rad: float, speed: float,
    curvature: float, half_width: float, horizon: float,
    decel: float = 0.0,
    accel: float = 0.0,
) -> ArcPath:
    """Build and cache an ArcPath from start (x,z), yaw, speed, curvature, half_width,
    horizon; optional decel/accel. Call this instead of constructing ArcPath directly."""
    return ArcPath(
        start_x=x, start_z=z, yaw_rad=yaw_rad, speed=speed,
        curvature=curvature, half_width=half_width, horizon=horizon,
        decel=decel, accel=accel,
    ).build()


def arc_arc_collision(
    a: ArcPath,
    b: ArcPath,
    margin: float = 0.5,
    n_samples: int = 24,
    min_lateral_gap: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Earliest time the two arc corridors overlap; uses closed-form ray-ray when
    both straight and constant speed, else time-sampled + bisection. Returns
    (time_s, hit_x, hit_z) or None. min_lateral_gap: suppress hit when centerlines
    stay that far apart (e.g. head-on turns in separate lanes)."""
    if a.arc_length < 1e-3 and b.arc_length < 1e-3:
        return None

    corridor_sq = (a.half_width + b.half_width + margin) ** 2
    horizon = min(a.horizon, b.horizon)

    if (a.is_straight and b.is_straight
            and a.decel <= 0 and b.decel <= 0
            and a.accel == 0.0 and b.accel == 0.0):
        return _ray_ray_collision(a, b, corridor_sq, horizon, min_lateral_gap)

    return _sampled_collision(a, b, corridor_sq, horizon, n_samples, min_lateral_gap)


def _ray_ray_collision(
    a: ArcPath, b: ArcPath, corridor_sq: float, horizon: float,
    min_lateral_gap: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Earliest time two straight rays' corridors touch: solve quadratic for
    |a_pos(t) − b_pos(t)|² = corridor_sq; returns (t, hit_x, hit_z) or None.
    min_lateral_gap suppresses hits when centerlines stay that far apart laterally."""
    dpx = a.start_x - b.start_x
    dpz = a.start_z - b.start_z
    dvx = a.speed * a.fwd_x - b.speed * b.fwd_x
    dvz = a.speed * a.fwd_z - b.speed * b.fwd_z

    A = dvx * dvx + dvz * dvz
    B = 2.0 * (dpx * dvx + dpz * dvz)
    C = dpx * dpx + dpz * dpz - corridor_sq

    if C <= 0:
        if min_lateral_gap > 0.0:
            lat = abs(dpz * a.fwd_x - dpx * a.fwd_z)
            if lat >= min_lateral_gap:
                return None
        return 0.0, (a.start_x + b.start_x) * 0.5, (a.start_z + b.start_z) * 0.5

    if abs(A) < 1e-12:
        return None

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)

    t_hit = None
    if 0.0 <= t1 <= horizon:
        t_hit = t1
    elif 0.0 <= t2 <= horizon:
        t_hit = t2
    elif t1 < 0 <= t2 and t2 <= horizon:
        t_hit = 0.0

    if t_hit is None:
        return None

    ax = a.start_x + t_hit * a.speed * a.fwd_x
    az = a.start_z + t_hit * a.speed * a.fwd_z
    bx = b.start_x + t_hit * b.speed * b.fwd_x
    bz = b.start_z + t_hit * b.speed * b.fwd_z

    if min_lateral_gap > 0.0:
        lat = abs((bz - az) * a.fwd_x - (bx - ax) * a.fwd_z)
        if lat >= min_lateral_gap:
            return None

    return t_hit, (ax + bx) * 0.5, (az + bz) * 0.5


def _sampled_collision(
    a: ArcPath, b: ArcPath, corridor_sq: float, horizon: float, n: int,
    min_lateral_gap: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Earliest corridor overlap for curved or non-constant-speed arcs: sample at n
    times, then bisect to refine hit time; respects min_lateral_gap. Returns
    (t, hit_x, hit_z) or None."""
    best_t: Optional[float] = None
    best_mx = 0.0
    best_mz = 0.0

    inv_n = 1.0 / n
    for i in range(n + 1):
        t = horizon * i * inv_n
        ax, az = a.position_at_time(t)
        bx, bz = b.position_at_time(t)
        dsq = (ax - bx) ** 2 + (az - bz) ** 2
        if dsq < corridor_sq:
            if min_lateral_gap > 0.0:
                h_a = a.heading_at_dist(a._dist_at_time(t))
                fwd_x_a = -math.sin(h_a)
                fwd_z_a = -math.cos(h_a)
                lat = abs((bz - az) * fwd_x_a - (bx - ax) * fwd_z_a)
                if lat >= min_lateral_gap:
                    continue
            lo = max(t - horizon * inv_n, 0.0)
            hi = t
            best_t = t
            best_mx = (ax + bx) * 0.5
            best_mz = (az + bz) * 0.5
            for _ in range(6):
                mid = (lo + hi) * 0.5
                ax2, az2 = a.position_at_time(mid)
                bx2, bz2 = b.position_at_time(mid)
                if (ax2 - bx2) ** 2 + (az2 - bz2) ** 2 < corridor_sq:
                    if min_lateral_gap > 0.0:
                        h_a2 = a.heading_at_dist(a._dist_at_time(mid))
                        fwd_x_a2 = -math.sin(h_a2)
                        fwd_z_a2 = -math.cos(h_a2)
                        lat2 = abs((bz2 - az2) * fwd_x_a2 - (bx2 - ax2) * fwd_z_a2)
                        if lat2 >= min_lateral_gap:
                            lo = mid
                            continue
                    hi = mid
                    best_t = mid
                    best_mx = (ax2 + bx2) * 0.5
                    best_mz = (az2 + bz2) * 0.5
                else:
                    lo = mid
            break

    if best_t is None:
        return None
    return best_t, best_mx, best_mz


class Vehicle:
    """Traffic vehicle with arc-based path prediction."""

    def __init__(
        self,
        position: Position,
        rotation: Quaternion,
        size: Size,
        speed: float,
        acceleration: float,
        trailer_count: int,
        trailers: list[Trailer],
        id: int,
        is_tmp: bool,
        is_trailer: bool,
        is_parked: bool = False,
    ) -> None:
        self.position = position
        self.rotation = rotation
        self.size = size
        self.speed = speed
        self.acc_speed = speed
        # Shared-memory acceleration is not used for physics; kinematic value is
        # filled in update_from_last(). Zero until the first update avoids spikes.
        self.acceleration = 0.0
        self.trailer_count = trailer_count
        self.trailers = trailers
        self.id = id
        self.is_tmp = is_tmp
        self.is_trailer = is_trailer
        self.is_parked = is_parked

        self.time: float = time.time()
        self.last_location = Position(0.0, 0.0, 0.0)
        self.last_rotation = Quaternion(0.0, 0.0, 0.0, 0.0)
        self.angular_velocity: float = 0.0

        self._smooth_x: Optional[float] = None
        self._smooth_z: Optional[float] = None
        self._smooth_yaw: Optional[float] = None
        self._raw_x: Optional[float] = None
        self._raw_z: Optional[float] = None

        # Speed/accel filter state (AI + TMP): see AGENTS.md §7.
        self._smooth_speed: Optional[float] = None
        self._smooth_accel: Optional[float] = None
        self._speed_ema: Optional[float] = None
        self._speed_ema_history: list[tuple[float, float]] = []
        self._raw_speed: Optional[float] = None
        # acc_speed standstill latch (hysteresis state): see AGENTS.md §7.
        self._acc_standstill: bool = False
        self._acc_release_s: float = 0.0
        # (time, x, z) per full update: newest last, capped at _POSITION_HISTORY_LEN.
        # Populated for both TMP and AI; speed LS fit uses a shorter internal window.
        self._position_history: list[tuple[float, float, float]] = []

        # TMP lag detection state.
        # _lag_since: monotonic time when the frozen-position window began.
        # lag_confirmed: True once the vehicle has been stationary for
        #   >= the TTC-scaled freeze_dur. AEB handles confirmed-stopped vehicles
        #   naturally via arc collision; no special-case needed in thread.py.
        self._lag_since: Optional[float] = None
        self.lag_confirmed: bool = False

        # Position mismatch (TMP only): consecutive frame counter.
        # Counts how many frames in a row the raw position jumped backward.
        # Resets to 0 on any clean frame or when the cap is reached.
        self._pos_mismatch_frames: int = 0

        # Crash detection (TMP only): per-axis rotation rates and displacement from prev frame.
        self._prev_pitch_rate: Optional[float] = None
        self._prev_yaw_rate: Optional[float] = None
        self._prev_roll_rate: Optional[float] = None
        self._crash_since: Optional[float] = None
        self.crash_confirmed: bool = False

        self._curvature_cache: float | None = None
        self._curvature_cache_valid: bool = False

    def accel_for_arc(self) -> float:
        """Longitudinal acceleration for arc / collision (kinematic filter output)."""
        return self.acceleration

    def radar_speed_accel(self) -> tuple[float, float, float, float, float]:
        """(raw_speed, speed_corr, speed_ema, acc_speed, accel) for the visualizer.

        speed_corr is the accel-corrected speed (AEB-facing, == self.speed),
        speed_ema the uncorrected EMA, acc_speed the adaptive-filtered ACC speed
        (ACC-facing, == self.acc_speed). See AGENTS.md §7.
        """
        raw_speed = self._raw_speed if self._raw_speed is not None else self.speed
        speed_ema = self._speed_ema if self._speed_ema is not None else self.speed
        return raw_speed, self.speed, speed_ema, self.acc_speed, self.acceleration

    def _tmp_apply_crash_rotation_jerk(self, prev: "Vehicle", t_now: float) -> None:
        """TMP: detect crash-level rotation jerk on every buffer read (sub-frame and full)."""
        dt = t_now - prev.time
        if not self.is_tmp or prev._raw_x is None or dt < 1e-9:
            return

        def _adiff(a: float, b: float) -> float:
            return (a - b + 180.0) % 360.0 - 180.0

        pitch_c, yaw_c, roll_c = self.rotation.euler()
        pitch_p, yaw_p, roll_p = prev.rotation.euler()
        pitch_rate = _adiff(pitch_c, pitch_p) / dt
        yaw_rate = _adiff(yaw_c, yaw_p) / dt
        roll_rate = _adiff(roll_c, roll_p) / dt

        _rot_jerk = False
        if prev._prev_pitch_rate is not None:
            if (
                abs(pitch_rate - prev._prev_pitch_rate) > _CRASH_PITCH_JERK
                or abs(yaw_rate - prev._prev_yaw_rate) > _CRASH_YAW_JERK
                or abs(roll_rate - prev._prev_roll_rate) > _CRASH_ROLL_JERK
            ):
                _rot_jerk = True

        self._prev_pitch_rate = pitch_rate
        self._prev_yaw_rate = yaw_rate
        self._prev_roll_rate = roll_rate

        if _rot_jerk:
            if self._crash_since is None:
                self._crash_since = t_now
            if t_now - self._crash_since >= _CRASH_CONFIRM_DURATION:
                self.crash_confirmed = True
        else:
            self._crash_since = None

    def update_from_last(
        self,
        prev: "Vehicle",
        t_now: float,
        ego_x: float,
        ego_y: float,
        ego_z: float,
        ego_speed: float,
    ) -> None:
        """Carry forward smoothed state or run a full update.  See AGENTS.md §7.

        ``ego_x/y/z`` and ``ego_speed`` feed the TTC-scaled lag freeze
        (see AGENTS.md §7 "Lag / freeze detection").
        """
        dt = t_now - prev.time

        # Sub-frame pass: carry forward all smoothed state unchanged.
        if dt < _LOCATION_UPDATE_FREQUENCY:
            self.time = prev.time
            self.last_location = prev.last_location
            self.last_rotation = prev.last_rotation
            self.angular_velocity = prev.angular_velocity
            self._smooth_x = prev._smooth_x
            self._smooth_z = prev._smooth_z
            self._smooth_yaw = prev._smooth_yaw
            self._raw_x = prev._raw_x
            self._raw_z = prev._raw_z
            self._lag_since = prev._lag_since
            self.lag_confirmed = prev.lag_confirmed
            self._pos_mismatch_frames = prev._pos_mismatch_frames
            self._prev_pitch_rate = prev._prev_pitch_rate
            self._prev_yaw_rate = prev._prev_yaw_rate
            self._prev_roll_rate = prev._prev_roll_rate
            self._crash_since = prev._crash_since
            self.crash_confirmed = prev.crash_confirmed
            self._smooth_speed = prev._smooth_speed
            self._smooth_accel = prev._smooth_accel
            self._speed_ema = prev._speed_ema
            self._raw_speed = prev._raw_speed
            self._position_history = list(prev._position_history)
            self._speed_ema_history = list(prev._speed_ema_history)
            self._acc_standstill = prev._acc_standstill
            self._acc_release_s = prev._acc_release_s
            if abs(self.angular_velocity) > _MAX_ANGULAR_VELOCITY:
                self.angular_velocity = 0.0
            self.speed = prev.speed
            self.acceleration = prev.acceleration
            self.acc_speed = prev.acc_speed

            self._tmp_apply_crash_rotation_jerk(prev, t_now)

            # Between full updates (dt < threshold), snap pose to the latest buffer
            # coordinates so arcs track sub-frame motion. Re-derive _raw_speed for
            # debug when movement exceeds the usual gate; filtered speed/accel stay
            # at the last full-tick values until dt ≥ _LOCATION_UPDATE_FREQUENCY.
            # TMP only: skip during lag freeze and position-mismatch hold unless
            # crash_confirmed bypasses both.
            _sf_lag_active = False
            if self.is_tmp and prev._lag_since is not None and prev._raw_x is not None:
                _sf_gap_3d = math.sqrt(
                    (prev._raw_x - ego_x) ** 2
                    + (prev.position.y - ego_y) ** 2
                    + (prev._raw_z - ego_z) ** 2
                )
                _sf_freeze_dur = _lag_freeze_duration(_sf_gap_3d, ego_speed)
                _sf_lag_active = (t_now - prev._lag_since) < _sf_freeze_dur
            _sf_snap_ok = (
                prev._raw_x is not None
                and prev._smooth_yaw is not None
            )
            if self.is_tmp and not self.crash_confirmed:
                if _sf_lag_active or prev._pos_mismatch_frames > 0:
                    _sf_snap_ok = False
            if _sf_snap_ok:
                rx = self.position.x
                rz = self.position.z
                ry = self.position.y
                dt_sf = t_now - prev.time
                ddx = rx - prev._raw_x
                ddz = rz - prev._raw_z
                dist = math.sqrt(
                    ddx * ddx + (ry - prev.position.y) ** 2 + ddz * ddz
                )
                if dt_sf > 1e-9 and dist > _RAW_SPEED_NEAR_ZERO_CHORD:
                    fwd_x = -math.sin(prev._smooth_yaw)
                    fwd_z = -math.cos(prev._smooth_yaw)
                    direction = 1.0 if (ddx * fwd_x + ddz * fwd_z) >= 0.0 else -1.0
                    # Raw kinematics for debug only: keep self.speed at filtered value.
                    self._raw_speed = direction * dist / dt_sf
                self._raw_x = rx
                self._raw_z = rz
                self._smooth_x = rx
                self._smooth_z = rz
                self.position.x = rx
                self.position.z = rz
            elif self._smooth_x is not None:
                self.position.x = self._smooth_x
                self.position.z = self._smooth_z
            return

        self.time = t_now
        self.last_location = prev.position
        self.last_rotation = prev.rotation
        self._smooth_x = prev._smooth_x
        self._smooth_z = prev._smooth_z
        self._smooth_yaw = prev._smooth_yaw
        self._lag_since = prev._lag_since
        self.lag_confirmed = False
        self._pos_mismatch_frames = prev._pos_mismatch_frames
        self._prev_pitch_rate = prev._prev_pitch_rate
        self._prev_yaw_rate = prev._prev_yaw_rate
        self._prev_roll_rate = prev._prev_roll_rate
        self._crash_since = prev._crash_since
        self.crash_confirmed = False
        self._smooth_speed = prev._smooth_speed
        self._smooth_accel = prev._smooth_accel
        self._speed_ema = prev._speed_ema
        self._raw_speed = prev._raw_speed
        self._position_history = list(prev._position_history)
        self._speed_ema_history = list(prev._speed_ema_history)
        self._acc_standstill = prev._acc_standstill
        self._acc_release_s = prev._acc_release_s

        raw_x = self.position.x
        raw_z = self.position.z
        self._raw_x = raw_x
        self._raw_z = raw_z

        # Type 3: Crash detection (TMP only): angular jerk; sub-frames call the same helper earlier.
        self._tmp_apply_crash_rotation_jerk(prev, t_now)

        # Type 1: Position mismatch (TMP only, max _POS_MISMATCH_MAX_FRAMES)
        # Raw position jumped backward along the vehicle's heading: out-of-order packet.
        # Yaw EMA and angular_velocity still run; position and carried speed/accel are held.
        # Bypassed when crash_confirmed: a crashed vehicle's backward jumps are real.
        _skip_position_update = False
        if (self.is_tmp
                and prev._smooth_yaw is not None
                and prev._raw_x is not None):
            _pm_dx = raw_x - prev._raw_x
            _pm_dz = raw_z - prev._raw_z
            _pm_fwd_x = -math.sin(prev._smooth_yaw)
            _pm_fwd_z = -math.cos(prev._smooth_yaw)
            if (_pm_dx * _pm_fwd_x + _pm_dz * _pm_fwd_z < -_POS_MISMATCH_BACKWARD_THRESHOLD
                    and self._pos_mismatch_frames < _POS_MISMATCH_MAX_FRAMES
                    and not self.crash_confirmed):
                self._pos_mismatch_frames = prev._pos_mismatch_frames + 1
                _skip_position_update = True
            else:
                self._pos_mismatch_frames = 0

        # Type 2: TMP lag detection (near-stationary freeze with speed decay)
        # Bypassed when crash_confirmed: any movement on a crashed vehicle is real position data.
        if self.is_tmp and prev._raw_x is not None and not _skip_position_update and not self.crash_confirmed:
            _raw_disp_sq = (raw_x - prev._raw_x) ** 2 + (raw_z - prev._raw_z) ** 2
            _expected_disp = abs(prev.speed) * dt
            _lag_threshold_sq = (_expected_disp * _LAG_DISP_RATIO) ** 2
            if abs(prev.speed) > _LAG_MIN_SPEED_MS and _raw_disp_sq < _lag_threshold_sq:
                if self._lag_since is None:
                    self._lag_since = t_now
                _lag_duration = t_now - self._lag_since
                _gap_3d = math.sqrt(
                    (raw_x - ego_x) ** 2
                    + (self.position.y - ego_y) ** 2
                    + (raw_z - ego_z) ** 2
                )
                _freeze_dur = _lag_freeze_duration(_gap_3d, ego_speed)
                if _freeze_dur <= 0.0:
                    # Too close to ego: a real stop must not be masked. Drop the
                    # freeze entirely and let the normal update run.
                    self._lag_since = None
                elif _lag_duration < _freeze_dur:
                    _lag_frac = _lag_duration / _freeze_dur
                    self._smooth_x = prev._smooth_x
                    self._smooth_z = prev._smooth_z
                    self._smooth_yaw = prev._smooth_yaw
                    self.angular_velocity = prev.angular_velocity
                    self.speed = prev.speed * (1.0 - _lag_frac * _lag_frac)
                    self.acceleration = 0.0
                    self._smooth_accel = 0.0
                    self._smooth_speed = self.speed
                    self._speed_ema = self.speed
                    self.acc_speed = self.speed
                    self._raw_speed = 0.0
                    if self._smooth_x is not None:
                        self.position.x = self._smooth_x
                        self.position.z = self._smooth_z
                    return
                else:
                    self.lag_confirmed = True
            else:
                self._lag_since = None

        # Wrap-safe yaw EMA: runs first so angular_velocity uses smooth derivative
        raw_yaw = math.radians(self.rotation.euler()[1])
        if self._smooth_yaw is None:
            self._smooth_yaw = raw_yaw
        else:
            diff = (raw_yaw - self._smooth_yaw + math.pi) % (2.0 * math.pi) - math.pi
            self._smooth_yaw = self._smooth_yaw + _RAW_YAW_ALPHA * diff

        # Angular velocity in deg/s: callers apply math.radians(), so keep degrees here
        _prev_smooth_yaw_deg = math.degrees(prev._smooth_yaw) if prev._smooth_yaw is not None else prev.rotation.euler()[1]
        _cur_smooth_yaw_deg = math.degrees(self._smooth_yaw)
        _yaw_diff_deg = (_cur_smooth_yaw_deg - _prev_smooth_yaw_deg + 180.0) % 360.0 - 180.0
        raw_av = _yaw_diff_deg / dt
        self.angular_velocity = 0.0 if abs(raw_av) > _MAX_ANGULAR_VELOCITY else raw_av

        # Position mismatch: hold smooth position and carry speed; yaw already updated above.
        if _skip_position_update:
            if self._smooth_x is not None:
                self.position.x = self._smooth_x
                self.position.z = self._smooth_z
            self.speed = prev.speed
            self.acceleration = prev.acceleration
            self.acc_speed = prev.acc_speed
            self._smooth_accel = prev._smooth_accel
            return

        # World position is unfiltered: arcs and debug use true coordinates.
        self._smooth_x = raw_x
        self._smooth_z = raw_z
        self.position.x = raw_x
        self.position.z = raw_z

        fwd_x = -math.sin(self._smooth_yaw)
        fwd_z = -math.cos(self._smooth_yaw)

        # Append this frame to the shared position history (both TMP and AI).
        # _position_history was already copied from prev; append directly.
        self._position_history.append((t_now, raw_x, raw_z))
        if len(self._position_history) > _POSITION_HISTORY_LEN:
            self._position_history = self._position_history[-_POSITION_HISTORY_LEN:]

        # Raw speed: position-history LS fit (single-interval fallback). SP/AI keeps
        # buffer sign when moving; TMP takes sign from the fit. Filter chain below
        # is shared (see AGENTS.md §7).
        _prx = prev._raw_x if prev._raw_x is not None else prev.position.x
        _prz = prev._raw_z if prev._raw_z is not None else prev.position.z
        raw_speed = _raw_speed_from_kinematics(
            self.speed,
            self._position_history,
            fwd_x,
            fwd_z,
            _prx,
            _prz,
            prev.position.y,
            raw_x,
            raw_z,
            self.position.y,
            dt,
            preserve_buffer_sign=not self.is_tmp,
        )

        (speed_ema, accel, speed_corr, acc_speed,
         speed_ema_history, acc_standstill, acc_release_s) = _smooth_vehicle_kinematics(
            raw_speed, t_now, dt,
            prev._speed_ema, prev._smooth_accel, prev.acc_speed,
            prev._speed_ema_history,
            prev._acc_standstill, prev._acc_release_s,
        )
        self._raw_speed = raw_speed
        self._speed_ema = speed_ema
        self._speed_ema_history = speed_ema_history
        self._smooth_accel = accel
        self._smooth_speed = speed_corr
        self.speed = speed_corr
        self.acceleration = accel
        self.acc_speed = acc_speed
        self._acc_standstill = acc_standstill
        self._acc_release_s = acc_release_s

    def curvature_from_history(self) -> float | None:
        """Curvature (1/m) from circumscribed circle fit over _position_history.

        Averages over up to four (oldest, mid, newest) triples for stability.
        Returns None when < 3 samples; 0.0 when near-stationary or near-straight.
        Falls back to angular_velocity / speed in get_arc() when None.
        Cached per frame: _position_history doesn't change within a tick.
        """
        if self._curvature_cache_valid:
            return self._curvature_cache
        result = self._compute_curvature()
        self._curvature_cache = result
        self._curvature_cache_valid = True
        return result

    def _compute_curvature(self) -> float | None:
        hist = self._position_history
        if len(hist) < 3:
            return None
        _, x0, z0 = hist[0]
        _, xn, zn = hist[-1]
        if (xn - x0) ** 2 + (zn - z0) ** 2 < 0.05 ** 2:
            return 0.0

        n = len(hist)
        candidates = [(0, n // 2, n - 1)]
        if n >= 5:
            candidates.append((1, (n - 1) // 2, n - 2))
        if n >= 7:
            candidates += [(0, n // 3, n - 1), (0, 2 * n // 3, n - 1)]

        total_k = 0.0
        count = 0
        for i, j, k in candidates:
            _, ax, az = hist[i]
            _, bx, bz = hist[j]
            _, cx, cz = hist[k]
            if (bx - ax) ** 2 + (bz - az) ** 2 < 0.05 ** 2:
                continue
            if (cx - bx) ** 2 + (cz - bz) ** 2 < 0.05 ** 2:
                continue
            D = 2.0 * (ax * (bz - cz) + bx * (cz - az) + cx * (az - bz))
            if abs(D) < 1e-6:
                count += 1  # collinear → κ = 0 contribution
                continue
            a2 = ax * ax + az * az
            b2 = bx * bx + bz * bz
            c2 = cx * cx + cz * cz
            ux = (a2 * (bz - cz) + b2 * (cz - az) + c2 * (az - bz)) / D
            uz = -(a2 * (bx - cx) + b2 * (cx - ax) + c2 * (ax - bx)) / D
            R = max(math.sqrt((ax - ux) ** 2 + (az - uz) ** 2), _MIN_CURVATURE_RADIUS)
            cross = (bx - ax) * (cz - bz) - (bz - az) * (cx - bx)
            total_k += (-1.0 if cross > 0.0 else 1.0) / R
            count += 1

        return total_k / count if count > 0 else None

    def get_arc(
        self,
        horizon: float = 3.0,
        half_width: float | None = None,
        decel: float = 0.0,
        arc_start_pctg: float = 1.0,
        curvature_override: float | None = None,
    ) -> ArcPath:
        """ArcPath for this vehicle from smoothed pose and curvature.

        Curvature is derived from position history when available (circumscribed
        circle fit), falling back to angular_velocity / speed. Crash-induced
        backward position spikes are suppressed by the 6 m/s² cap in _accel_to_arc_params().
        """
        yaw_rad = (
            self._smooth_yaw
            if self._smooth_yaw is not None
            else math.radians(self.rotation.euler()[1])
        )
        abs_speed = abs(self.speed)
        if curvature_override is not None:
            curvature = curvature_override
        else:
            _hist_k = self.curvature_from_history()
            if _hist_k is not None:
                curvature = _hist_k
            else:
                curvature = math.radians(self.angular_velocity) / abs_speed if abs_speed > 0.5 else 0.0
        effective_hw = half_width if half_width is not None else self.size.width / 2.0
        effective_decel, effective_accel = _accel_to_arc_params(self.accel_for_arc(), decel)

        is_reversing = self.speed < -1e-3
        effective_p = (1.0 - arc_start_pctg) if is_reversing else arc_start_pctg
        fwd_x = -math.sin(yaw_rad)
        fwd_z = -math.cos(yaw_rad)
        body_offset = (effective_p - 0.5) * self.size.length
        start_x = self.position.x + body_offset * fwd_x
        start_z = self.position.z + body_offset * fwd_z

        return build_arc(
            start_x, start_z, yaw_rad, self.speed,
            curvature, effective_hw, horizon,
            decel=effective_decel, accel=effective_accel,
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()

    def get_corners(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """World-space footprint corners (front-right, front-left, back-left, back-right).

        TMP uses symmetric ± length/2 about pivot; AI uses asymmetric
        (-0.18·length front, +0.82·length back) per AGENTS.md §6. Yaw
        comes from ``_smooth_yaw`` when available.
        """
        yaw_rad = (
            self._smooth_yaw
            if self._smooth_yaw is not None
            else math.radians(self.rotation.euler()[1])
        )
        fwd_x = -math.sin(yaw_rad)
        fwd_z = -math.cos(yaw_rad)
        right_x = -fwd_z
        right_z = fwd_x
        if self.is_tmp:
            front_d = self.size.length * 0.5
            back_d = self.size.length * 0.5
        else:
            front_d = self.size.length * 0.18
            back_d = self.size.length * 0.82
        hw = self.size.width * 0.5
        px = self.position.x
        pz = self.position.z
        return (
            (px + front_d * fwd_x + hw * right_x, pz + front_d * fwd_z + hw * right_z),
            (px + front_d * fwd_x - hw * right_x, pz + front_d * fwd_z - hw * right_z),
            (px - back_d * fwd_x - hw * right_x, pz - back_d * fwd_z - hw * right_z),
            (px - back_d * fwd_x + hw * right_x, pz - back_d * fwd_z + hw * right_z),
        )

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.id}, pos={self.position}, "
            f"speed={self.speed:.2f}, is_tmp={self.is_tmp}, "
            f"is_parked={self.is_parked})"
        )


def vehicle_from_trailer(parent: Vehicle, trailer: Trailer, synthetic_id: int) -> Vehicle:
    """Wrap a nested Trailer record as a standalone Vehicle.

    Road trains expose only the tractor and the first trailer as top-level
    radar vehicles; every trailer behind the first is a nested Trailer on
    that first trailer (AI trucks nest all of their trailers the same way).
    ACC scoring iterates Vehicles, so those nested trailers are invisible to
    it. Wrapping one as a Vehicle lets the tracker score it and lets
    RadarThread carry position history forward for it like any other id.

    The wrapped Vehicle gets its own Position (``update_from_last`` mutates
    position in place); rotation and size are immutable and shared. TMP's raw
    trailer pivot is the front coupler, so ``correct_position()`` shifts it to
    the body center: matching the symmetric +/- length/2 pivot the rest of
    the pipeline assumes for TMP vehicles.
    """
    if trailer.is_tmp:
        position = trailer.correct_position()
    else:
        src = trailer.position
        position = Position(src.x, src.y, src.z)
    return Vehicle(
        position=position,
        rotation=trailer.rotation,
        size=trailer.size,
        speed=parent.speed,
        acceleration=parent.acceleration,
        trailer_count=0,
        trailers=[],
        id=synthetic_id,
        is_tmp=trailer.is_tmp,
        is_trailer=True,
        is_parked=parent.is_parked,
    )

