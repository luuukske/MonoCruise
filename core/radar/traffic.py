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
# Gap between successive TrafficReader clocks that means a real pause/hitch
# (no intermediate frames). Must NOT be applied to Vehicle.update_from_last dt:
# sub-frames freeze Vehicle.time, so dt since last *full* update can exceed
# this during normal slow radar without a pause — that path was freezing
# speeds at a stale fraction of truth. See AGENTS.md §7.
_READER_CLOCK_GAP_S: float = 0.50

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
#
# TAU_SLOW was 2.0 s. At that length alpha_a is small enough that the step-4
# feed-forward dominates and acc_speed runs near open loop, which lagged a real
# lead brake and made ACC follow too close on a hard stop. Shortening it lets
# the low-pass correction rein the prediction back in. Measured on the TMP
# corpus, together with the ACCEL_FLOOR change below: gap conceded per hard
# decel 5.37 -> 4.24 m, with stall poison and worst-frame jerk unchanged.
#
# The binding constraint is convoy sawtooth pass-through, guarded by
# test_convoy_sawtooth_attenuated_at_cruise (bound 0.30 m/s on a 0.8 m/s input
# wobble). That cost is carried almost entirely by TAU_SLOW: ACCEL_FLOOR barely
# registers on it (1.6/0.35 -> 0.2462, 1.6/0.25 -> 0.2463), so the floor is
# taken in full and the tau is bought only down to where margin remains.
# 2.0 -> 0.1997, 1.6 -> 0.2462, 1.4 -> 0.2775, 1.2 -> 0.3158 (fails).
# 1.4 buys only 0.05 m more gap for half the remaining margin, so 1.6 it is.
#
# Both constants are step-4 only, so AEB is untouched by construction.
_ACC_SPEED_DEADBAND_MS: float = 0.7
_ACC_SPEED_TAU_SLOW_S: float = 1.6
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
_ACC_SPEED_ACCEL_FLOOR: float = 0.15

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

# Trend consistency: TMP convoy reconciliation wobbles speed in slow
# drift-and-snap cycles (~2-4 s). Each drift phase reads as a genuine ramp on
# the 1.5 s accel_trend window and used to open the feed-forward, which then
# integrated the wobble straight into acc_speed. A real ramp accumulates net
# speed change on a longer horizon; a wobble does not, so the LS slope over
# _ACC_SPEED_CONSIST_WINDOW_S stays near zero. consistency = the larger of
# the slope ratio |accel_long| / |accel_trend| and a magnitude ramp on
# |accel_long| (_ACC_SPEED_CONSIST_MAG_LO/HI): the ratio converges slowly on
# a fresh brake because the long window still holds pre-brake cruise, but any
# real brake pushes |accel_long| past the magnitude ramp within ~1-1.5 s
# while a zero-mean wobble never sustains it. 0 on sign mismatch, clamped to
# 1; scales ff_gate and accel_ramp. The fast-tau residual gate stays on the
# short trend so hard-brake onset tracking is not delayed. See AGENTS.md §7.
_ACC_SPEED_CONSIST_WINDOW_S: float = 4.0
_ACC_SPEED_CONSIST_MAG_LO_MS2: float = 0.4     # m/s² : |accel_long| where magnitude term starts
_ACC_SPEED_CONSIST_MAG_HI_MS2: float = 1.0     # m/s² : |accel_long| granting full consistency

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

# A hard-braking target temporarily uses a short position-fit window so the
# long cruise window cannot keep reporting motion after the target has stopped.
_RAW_BRAKE_SHORT_HISTORY_LEN: int = 5
_RAW_BRAKE_CONFIRM_FRAMES: int = 2
_RAW_BRAKE_MIN_DECEL_MS2: float = 2.0
_RAW_BRAKE_MIN_SPEED_LOSS_MS: float = 0.4
_RAW_BRAKE_MIN_INTERVAL_SPEED_MS: float = 0.25
_RAW_BRAKE_MONOTONIC_TOL_MS: float = 0.15
_RAW_BRAKE_CONVERGENCE_MS: float = 0.3
_RAW_BRAKE_RELEASE_FRAMES: int = 3
_RAW_BRAKE_STANDSTILL_SPEED_MS: float = 0.1


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


def _hard_brake_decel_from_position_history(
    history: list[tuple[float, float, float]],
    fwd_x: float,
    fwd_z: float,
) -> float | None:
    """Return confirmed short-window deceleration magnitude, else None."""
    if len(history) < _RAW_BRAKE_SHORT_HISTORY_LEN:
        return None
    window = history[-_RAW_BRAKE_SHORT_HISTORY_LEN:]
    intervals: list[tuple[float, float]] = []
    for (t0, x0, z0), (t1, x1, z1) in zip(window, window[1:]):
        dt = t1 - t0
        if dt <= 1e-9:
            return None
        speed = ((x1 - x0) * fwd_x + (z1 - z0) * fwd_z) / dt
        intervals.append(((t0 + t1) * 0.5, speed))

    direction = 1.0 if sum(speed for _, speed in intervals) >= 0.0 else -1.0
    magnitudes = [speed * direction for _, speed in intervals]
    if any(speed < _RAW_BRAKE_MIN_INTERVAL_SPEED_MS for speed in magnitudes):
        return None
    if any(
        later > earlier + _RAW_BRAKE_MONOTONIC_TOL_MS
        for earlier, later in zip(magnitudes, magnitudes[1:])
    ):
        return None

    early_speed = 0.5 * (magnitudes[0] + magnitudes[1])
    late_speed = 0.5 * (magnitudes[-2] + magnitudes[-1])
    speed_loss = early_speed - late_speed
    early_t = 0.5 * (intervals[0][0] + intervals[1][0])
    late_t = 0.5 * (intervals[-2][0] + intervals[-1][0])
    span = late_t - early_t
    if span <= 1e-9:
        return None
    decel = speed_loss / span
    if (
        speed_loss < _RAW_BRAKE_MIN_SPEED_LOSS_MS
        or decel < _RAW_BRAKE_MIN_DECEL_MS2
    ):
        return None
    return decel


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


def _speed_corr_chain(
    raw_speed: float,
    t_now: float,
    dt: float,
    prev_speed_ema: float | None,
    prev_accel: float | None,
    prev_speed_ema_history: list[tuple[float, float]] | None,
    responsive_brake_decel: float = 0.0,
) -> tuple[float, float, float, list[tuple[float, float]]]:
    """Steps 1-3 of the filter chain. See AGENTS.md §7.

    Returns (speed_ema, accel, speed_corr, speed_ema_history). Run once per
    consumer: AEB reads the hard-brake-selected raw speed, ACC reads the long
    position window (see ``_smooth_vehicle_kinematics``).
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
    if responsive_brake_decel > 0.0:
        accel = min(accel, -responsive_brake_decel)

    # Step 3: lag-compensated speed. τ is the step-1 EMA's settling time.
    tau_eff = dt * (1.0 - alpha_s) / alpha_s if alpha_s > 1e-6 else 0.0
    correction = max(-_SPEED_CORR_CLAMP_MS, min(_SPEED_CORR_CLAMP_MS, accel * tau_eff))
    speed_corr = speed_ema + correction
    return speed_ema, accel, speed_corr, history


def _acc_speed_step(
    speed_corr: float,
    history: list[tuple[float, float]],
    accel: float,
    dt: float,
    prev_speed_ema: float | None,
    prev_acc_speed: float | None,
    prev_acc_standstill: bool,
    prev_acc_release_s: float,
) -> tuple[float, bool, float]:
    """Step 4 of the filter chain. Returns (acc_speed, standstill, release_s)."""
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
        # Trend consistency: a real ramp sustains its slope on the long window,
        # convoy drift-and-snap wobble does not. Scales ff and tau reduction only.
        accel_long = _accel_from_speed_history(history, _ACC_SPEED_CONSIST_WINDOW_S)
        if accel_trend * accel_long <= 0.0:
            consistency = 0.0
        else:
            ratio = min(1.0, abs(accel_long) / max(abs(accel_trend), 1e-6))
            mag_span = _ACC_SPEED_CONSIST_MAG_HI_MS2 - _ACC_SPEED_CONSIST_MAG_LO_MS2
            mag = max(0.0, min(1.0,
                (abs(accel_long) - _ACC_SPEED_CONSIST_MAG_LO_MS2) / mag_span))
            consistency = max(ratio, mag)
        accel_span = _ACC_SPEED_ACCEL_HI_MS2 - _ACC_SPEED_ACCEL_LO_MS2
        accel_ramp = max(0.0, min(1.0,
            (abs(accel_trend) - _ACC_SPEED_ACCEL_LO_MS2) / accel_span)) * consistency
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
            / (_ACC_SPEED_FF_GATE_HI_MS2 - _ACC_SPEED_FF_GATE_LO_MS2))) * consistency
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

    return acc_speed, acc_standstill, acc_release_s


def _smooth_vehicle_kinematics(
    raw_speed: float,
    acc_raw_speed: float,
    t_now: float,
    dt: float,
    prev_speed_ema: float | None,
    prev_accel: float | None,
    prev_speed_ema_history: list[tuple[float, float]] | None,
    prev_acc_speed_ema: float | None,
    prev_acc_accel: float | None,
    prev_acc_speed_ema_history: list[tuple[float, float]] | None,
    prev_acc_speed: float | None,
    prev_acc_standstill: bool = False,
    prev_acc_release_s: float = 0.0,
    responsive_brake_decel: float = 0.0,
) -> tuple[float, float, float, list[tuple[float, float]],
           float, float, list[tuple[float, float]], float, bool, float]:
    """Speed/accel filter chain shared by AI and TMP vehicles. See AGENTS.md §7.

    Two chains, one per consumer. AEB's ``speed`` / ``acceleration`` run on
    ``raw_speed``, which is the hard-brake-selected estimate: the short position
    window when a brake is confirmed. ACC's ``acc_speed`` / ``acc_accel`` run on
    ``acc_raw_speed``, the long window only.

    The split exists because the short window is an AEB responsiveness device
    that ACC never needed. On TMP it latches on packet stalls and then stays
    selected a median of 1.83 s past the end of the stall (corpus measurement),
    which ACC read as a sustained lead brake and answered with a hard brake of
    its own. Filtering the stall out is not possible: a stall and a real brake
    are the same observable until the speed either returns or does not. Keeping
    the short window off the ACC path removes the exposure instead, and costs
    AEB nothing.

    Callers pass ``acc_raw_speed == raw_speed`` when ``crash_confirmed``, so a
    crashed vehicle reaches ACC through the same unfiltered path as AEB.

    Returns (speed_ema, accel, speed_corr, speed_ema_history,
    acc_speed_ema, acc_accel, acc_speed_ema_history, acc_speed,
    acc_standstill, acc_release_s).
    """
    speed_ema, accel, speed_corr, history = _speed_corr_chain(
        raw_speed, t_now, dt,
        prev_speed_ema, prev_accel, prev_speed_ema_history,
        responsive_brake_decel,
    )

    # Always run the ACC chain on its own state. It cannot be aliased to the AEB
    # chain on frames where the two raw speeds happen to agree: the chains carry
    # separate EMA and history state, so they stay diverged for a window after
    # any brake transient even once the inputs match again.
    acc_ema, acc_accel, acc_corr, acc_history = _speed_corr_chain(
        acc_raw_speed, t_now, dt,
        prev_acc_speed_ema, prev_acc_accel, prev_acc_speed_ema_history,
    )

    acc_speed, acc_standstill, acc_release_s = _acc_speed_step(
        acc_corr, acc_history, acc_accel, dt,
        prev_acc_speed_ema, prev_acc_speed,
        prev_acc_standstill, prev_acc_release_s,
    )

    return (speed_ema, accel, speed_corr, history,
            acc_ema, acc_accel, acc_history, acc_speed,
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

    # Capsule body extents along the heading, measured from the arc reference
    # point (start). fwd_len reaches toward the body front, back_len toward the
    # body rear. Both default 0.0, which collapses the capsule to the reference
    # point and preserves the legacy point/disc collision behavior for every
    # consumer that does not set them (ACC, rendering). Set by AEB build sites
    # so the collision test covers the whole vehicle body, not just its width.
    # Collision uses the derived _cap_fwd/_cap_back (extents minus half_width)
    # so the capsule's rounded end cap lands ON the bumper, not half_width past
    # it; raw fwd_len/back_len remain the physical body ends for centreline
    # sampling (_any_body_in_ego_lane). See core/radar/AGENTS.md 8.
    fwd_len: float = 0.0
    back_len: float = 0.0

    # Corridor-margin scale applied when this body meets another capsule at a
    # near-parallel heading (see _sampled_collision). The margin exists to
    # absorb the time-sampling risk of crossing paths sweeping through contact
    # between samples; near-parallel bodies hold their separation across many
    # samples, so the full margin only manufactures side-graze hits on
    # adjacent-lane traffic. A collision pair uses the smaller scale of its two
    # arcs. 1.0 (default) keeps the full margin for every consumer that does
    # not set it; AEB ego arcs set cal.capsule_parallel_margin_scale.
    parallel_margin_scale: float = 1.0

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
    _has_body: bool = False
    _cap_fwd: float = 0.0
    _cap_back: float = 0.0

    def build(self) -> "ArcPath":
        """Compute cached fields (fwd, radius, center, arc_length, is_straight) from
        start, curvature, speed, decel/accel. Call after setting fields."""
        self._has_body = self.fwd_len > 1e-9 or self.back_len > 1e-9
        # Collision segment extents: the capsule test adds half_width radially
        # in EVERY direction around the segment, so a segment reaching the
        # bumper would extend the body half_width past each end lengthwise
        # (~1.15 m ego + ~1.25 m target of phantom length). Retract each end by
        # half_width so the rounded cap's tip coincides with the body end; the
        # side faces stay exact and only the rectangle corners round off, which
        # the corridor margin absorbs.
        self._cap_fwd = max(self.fwd_len - self.half_width, 0.0)
        self._cap_back = max(self.back_len - self.half_width, 0.0)
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
    fwd_len: float = 0.0,
    back_len: float = 0.0,
    parallel_margin_scale: float = 1.0,
) -> ArcPath:
    """Build and cache an ArcPath from start (x,z), yaw, speed, curvature, half_width,
    horizon; optional decel/accel. Call this instead of constructing ArcPath directly.
    fwd_len/back_len give the body extents ahead of/behind the reference for a
    capsule collision body; 0.0 (default) keeps point/disc behavior.
    parallel_margin_scale < 1.0 shrinks the corridor margin for near-parallel
    capsule contacts (see ArcPath field comment)."""
    return ArcPath(
        start_x=x, start_z=z, yaw_rad=yaw_rad, speed=speed,
        curvature=curvature, half_width=half_width, horizon=horizon,
        decel=decel, accel=accel, fwd_len=fwd_len, back_len=back_len,
        parallel_margin_scale=parallel_margin_scale,
    ).build()


def capsule_extents(
    front_d: float, back_d: float, body_offset: float,
) -> tuple[float, float]:
    """Body capsule extents (fwd_len, back_len) measured from the arc reference.

    front_d/back_d are the body front/rear distances from the pivot; body_offset
    is the reference's signed offset from the pivot along the heading. Result is
    clamped non-negative for the rare reverse-pivot case. Used by the ego and
    trailer build sites so the reference-offset asymmetry is handled uniformly."""
    return max(front_d - body_offset, 0.0), max(back_d + body_offset, 0.0)


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

    # Capsule bodies (nonzero fwd_len/back_len) test segment overlap, not point
    # overlap, so the closed-form ray-ray path (point-only) does not apply.
    if (a.is_straight and b.is_straight
            and not a._has_body and not b._has_body
            and a.decel <= 0 and b.decel <= 0
            and a.accel == 0.0 and b.accel == 0.0):
        return _ray_ray_collision(a, b, corridor_sq, horizon, min_lateral_gap)

    return _sampled_collision(
        a, b, corridor_sq, horizon, n_samples, min_lateral_gap,
        hw_sum=a.half_width + b.half_width, margin=margin,
    )


def _seg_seg_dist_sq_mid(
    ax0: float, az0: float, ax1: float, az1: float,
    bx0: float, bz0: float, bx1: float, bz1: float,
) -> tuple[float, float, float]:
    """Squared distance between segments A(a0->a1), B(b0->b1) and the midpoint
    of the closest-point pair. Degenerate segments (a point) are handled by the
    point-to-segment projections. No allocations beyond the returned tuple."""
    ux = ax1 - ax0
    uz = az1 - az0
    vx = bx1 - bx0
    vz = bz1 - bz0
    wx = ax0 - bx0
    wz = az0 - bz0
    a = ux * ux + uz * uz
    b = ux * vx + uz * vz
    c = vx * vx + vz * vz
    d = ux * wx + uz * wz
    e = vx * wx + vz * wz
    den = a * c - b * b
    if a <= 1e-12 and c <= 1e-12:
        sc = 0.0
        tc = 0.0
    elif a <= 1e-12:
        sc = 0.0
        tc = min(1.0, max(0.0, e / c))
    elif c <= 1e-12:
        tc = 0.0
        sc = min(1.0, max(0.0, -d / a))
    else:
        if den > 1e-12:
            sc = (b * e - c * d) / den
            sc = 0.0 if sc < 0.0 else (1.0 if sc > 1.0 else sc)
        else:
            sc = 0.0
        tc = (b * sc + e) / c
        if tc < 0.0:
            tc = 0.0
            sc = min(1.0, max(0.0, -d / a))
        elif tc > 1.0:
            tc = 1.0
            sc = min(1.0, max(0.0, (b - d) / a))
    cpax = ax0 + sc * ux
    cpaz = az0 + sc * uz
    cpbx = bx0 + tc * vx
    cpbz = bz0 + tc * vz
    dxx = cpax - cpbx
    dzz = cpaz - cpbz
    return dxx * dxx + dzz * dzz, (cpax + cpbx) * 0.5, (cpaz + cpbz) * 0.5


def pair_body_dist_sq(a: ArcPath, b: ArcPath, t: float) -> float:
    """Squared distance between the two arc bodies at time t: capsule
    segment-to-segment when either arc carries body extents, else point-to-point
    of the reference positions. Used by the diverge/approaching predicates so
    they measure the same body geometry the collision test uses."""
    if a._has_body or b._has_body:
        da = a._dist_at_time(t)
        ax, az = a.position_at_dist(da)
        ha = a.heading_at_dist(da)
        afx = -math.sin(ha)
        afz = -math.cos(ha)
        db = b._dist_at_time(t)
        bx, bz = b.position_at_dist(db)
        hb = b.heading_at_dist(db)
        bfx = -math.sin(hb)
        bfz = -math.cos(hb)
        dsq, _, _ = _seg_seg_dist_sq_mid(
            ax + a._cap_fwd * afx, az + a._cap_fwd * afz,
            ax - a._cap_back * afx, az - a._cap_back * afz,
            bx + b._cap_fwd * bfx, bz + b._cap_fwd * bfz,
            bx - b._cap_back * bfx, bz - b._cap_back * bfz,
        )
        return dsq
    ax, az = a.position_at_time(t)
    bx, bz = b.position_at_time(t)
    return (ax - bx) ** 2 + (az - bz) ** 2


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
    hw_sum: float = 0.0, margin: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Earliest corridor overlap for curved, non-constant-speed, or capsule-body
    arcs: sample at n times, then bisect to refine hit time; respects
    min_lateral_gap. When either arc carries body extents (fwd_len/back_len) the
    overlap test is segment-to-segment (the swept body), not point-to-point.
    For capsule pairs where either arc sets parallel_margin_scale < 1, the
    corridor margin shrinks toward margin * scale as the two headings approach
    parallel (per sample): the margin's job is absorbing crossing paths that
    sweep through contact between time samples, and near-parallel bodies hold
    their separation, so the full margin only manufactures side-graze hits on
    adjacent-lane traffic. hw_sum/margin carry the corridor_sq components for
    that scaling. Returns (t, hit_x, hit_z) or None. hit is the contact-point
    midpoint for capsule pairs, the reference midpoint for point pairs."""
    has_body = a._has_body or b._has_body
    need_lat = min_lateral_gap > 0.0
    pms = (a.parallel_margin_scale
           if a.parallel_margin_scale < b.parallel_margin_scale
           else b.parallel_margin_scale)
    scale_margin = has_body and pms < 1.0 and margin > 0.0

    def _probe(t: float) -> tuple[float, float, float, float, float]:
        if has_body:
            da = a._dist_at_time(t)
            ax, az = a.position_at_dist(da)
            ha = a.heading_at_dist(da)
            afx = -math.sin(ha)
            afz = -math.cos(ha)
            a0x = ax + a._cap_fwd * afx
            a0z = az + a._cap_fwd * afz
            a1x = ax - a._cap_back * afx
            a1z = az - a._cap_back * afz
            db = b._dist_at_time(t)
            bx, bz = b.position_at_dist(db)
            hb = b.heading_at_dist(db)
            bfx = -math.sin(hb)
            bfz = -math.cos(hb)
            b0x = bx + b._cap_fwd * bfx
            b0z = bz + b._cap_fwd * bfz
            b1x = bx - b._cap_back * bfx
            b1z = bz - b._cap_back * bfz
            dsq, mx, mz = _seg_seg_dist_sq_mid(a0x, a0z, a1x, a1z,
                                               b0x, b0z, b1x, b1z)
            lat = abs((bz - az) * afx - (bx - ax) * afz) if need_lat else 0.0
            if scale_margin:
                cosd = afx * bfx + afz * bfz
                if cosd < 0.0:
                    cosd = -cosd
                sind_sq = 1.0 - cosd * cosd
                sind = math.sqrt(sind_sq) if sind_sq > 0.0 else 0.0
                thr = hw_sum + margin * (pms + (1.0 - pms) * sind)
                return dsq, mx, mz, lat, thr * thr
            return dsq, mx, mz, lat, corridor_sq
        ax, az = a.position_at_time(t)
        bx, bz = b.position_at_time(t)
        dsq = (ax - bx) ** 2 + (az - bz) ** 2
        if need_lat:
            h_a = a.heading_at_dist(a._dist_at_time(t))
            fwd_x_a = -math.sin(h_a)
            fwd_z_a = -math.cos(h_a)
            lat = abs((bz - az) * fwd_x_a - (bx - ax) * fwd_z_a)
        else:
            lat = 0.0
        return dsq, (ax + bx) * 0.5, (az + bz) * 0.5, lat, corridor_sq

    best_t: Optional[float] = None
    best_mx = 0.0
    best_mz = 0.0

    inv_n = 1.0 / n
    for i in range(n + 1):
        t = horizon * i * inv_n
        dsq, mx, mz, lat, thr_sq = _probe(t)
        if dsq < thr_sq:
            if need_lat and lat >= min_lateral_gap:
                continue
            lo = max(t - horizon * inv_n, 0.0)
            hi = t
            best_t = t
            best_mx = mx
            best_mz = mz
            for _ in range(6):
                mid = (lo + hi) * 0.5
                dsq2, mx2, mz2, lat2, thr2_sq = _probe(mid)
                if dsq2 < thr2_sq and not (need_lat and lat2 >= min_lateral_gap):
                    hi = mid
                    best_t = mid
                    best_mx = mx2
                    best_mz = mz2
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
        # ACC-side acceleration: long-window chain, no hard-brake floor.
        self.acc_accel = 0.0
        self.trailer_count = trailer_count
        self.trailers = trailers
        self.id = id
        self.is_tmp = is_tmp
        self.is_trailer = is_trailer
        self.is_parked = is_parked

        # Kinematics clock for update_from_last (seconds). Live radar overwrites
        # with SCS simulatedTime/1e6 so pause/hitch gaps are not wall-clock dt.
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
        self._raw_brake_confirm_frames: int = 0
        self._raw_brake_active: bool = False
        self._raw_brake_converged_frames: int = 0
        # Parallel steps 1-3 state for the ACC chain (long window only).
        self._acc_speed_ema: Optional[float] = None
        self._acc_smooth_accel: Optional[float] = None
        self._acc_speed_ema_history: list[tuple[float, float]] = []
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

    def _reset_raw_brake_transient(self) -> None:
        self._raw_brake_confirm_frames = 0
        self._raw_brake_active = False
        self._raw_brake_converged_frames = 0

    def _select_raw_speed(
        self,
        long_speed: float,
        buffer_speed: float,
        fwd_x: float,
        fwd_z: float,
    ) -> float:
        """Select the short position fit only during a confirmed hard brake."""
        if self.is_trailer:
            self._reset_raw_brake_transient()
            return long_speed
        short_history = self._position_history[-_RAW_BRAKE_SHORT_HISTORY_LEN:]
        short_speed = _raw_speed_from_position_history(short_history, fwd_x, fwd_z)
        if short_speed is None:
            self._raw_brake_confirm_frames = 0
            return long_speed
        if not self.is_tmp and abs(buffer_speed) > _BUFFER_SIGN_SPEED_MS:
            short_speed = math.copysign(abs(short_speed), buffer_speed)

        recent_decel = _hard_brake_decel_from_position_history(
            self._position_history, fwd_x, fwd_z,
        )
        qualifies = recent_decel is not None
        if not self._raw_brake_active:
            if qualifies:
                self._raw_brake_confirm_frames += 1
            else:
                self._raw_brake_confirm_frames = 0
            if self._raw_brake_confirm_frames >= _RAW_BRAKE_CONFIRM_FRAMES:
                self._raw_brake_active = True
                self._raw_brake_converged_frames = 0

        if not self._raw_brake_active:
            return long_speed

        if (
            abs(short_speed) > _RAW_BRAKE_STANDSTILL_SPEED_MS
            and abs(short_speed - long_speed) <= _RAW_BRAKE_CONVERGENCE_MS
        ):
            self._raw_brake_converged_frames += 1
        else:
            self._raw_brake_converged_frames = 0
        if self._raw_brake_converged_frames >= _RAW_BRAKE_RELEASE_FRAMES:
            self._reset_raw_brake_transient()
            return long_speed
        return short_speed

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

    def _hold_across_clock_discontinuity(self, prev: "Vehicle", t_now: float) -> None:
        """Re-base after a pause/hitch gap without integrating across it.

        Holds filtered speed/accel, snaps pose to the latest buffer coordinates,
        and restarts position / speed-EMA histories at ``t_now`` so the LS raw
        speed fit cannot span the gap (which would read Δpos/huge_τ ≈ 0).
        """
        self.time = t_now
        self.last_location = prev.last_location
        self.last_rotation = prev.last_rotation
        self.angular_velocity = prev.angular_velocity
        if abs(self.angular_velocity) > _MAX_ANGULAR_VELOCITY:
            self.angular_velocity = 0.0

        raw_x = self.position.x
        raw_z = self.position.z
        self._raw_x = raw_x
        self._raw_z = raw_z
        self._smooth_x = raw_x
        self._smooth_z = raw_z
        self._smooth_yaw = prev._smooth_yaw
        if self._smooth_yaw is None:
            self._smooth_yaw = math.radians(self.rotation.euler()[1])

        self._lag_since = None
        self.lag_confirmed = False
        self._pos_mismatch_frames = 0
        self._prev_pitch_rate = prev._prev_pitch_rate
        self._prev_yaw_rate = prev._prev_yaw_rate
        self._prev_roll_rate = prev._prev_roll_rate
        self._crash_since = None
        self.crash_confirmed = False

        self._smooth_speed = prev._smooth_speed
        self._smooth_accel = prev._smooth_accel
        self._speed_ema = prev._speed_ema
        self._acc_speed_ema = prev._acc_speed_ema
        self._acc_smooth_accel = prev._acc_smooth_accel
        self._raw_speed = prev._raw_speed
        self._reset_raw_brake_transient()
        self.speed = prev.speed
        self.acceleration = prev.acceleration
        self.acc_speed = prev.acc_speed
        self.acc_accel = prev.acc_accel
        self._acc_standstill = prev._acc_standstill
        self._acc_release_s = prev._acc_release_s

        # Fresh histories on the new clock, seeded so the next LS raw-speed fit
        # returns ~held speed instead of a 1-sample cold start or a catch-up spike.
        dt_seed = _LOCATION_UPDATE_FREQUENCY
        yaw = self._smooth_yaw if self._smooth_yaw is not None else 0.0
        held_speed = float(prev.speed)
        fwd_x = -math.sin(yaw)
        fwd_z = -math.cos(yaw)
        back_x = raw_x - held_speed * fwd_x * dt_seed
        back_z = raw_z - held_speed * fwd_z * dt_seed
        self._position_history = [
            (t_now - dt_seed, back_x, back_z),
            (t_now, raw_x, raw_z),
        ]
        if prev._speed_ema is not None:
            self._speed_ema_history = [
                (t_now - dt_seed, prev._speed_ema),
                (t_now, prev._speed_ema),
            ]
        else:
            self._speed_ema_history = []
        if prev._acc_speed_ema is not None:
            self._acc_speed_ema_history = [
                (t_now - dt_seed, prev._acc_speed_ema),
                (t_now, prev._acc_speed_ema),
            ]
        else:
            self._acc_speed_ema_history = []

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

        # Clock went backwards (domain glitch): re-base without integrating.
        if dt < 0.0:
            self._hold_across_clock_discontinuity(prev, t_now)
            return

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
            self._acc_speed_ema = prev._acc_speed_ema
            self._acc_smooth_accel = prev._acc_smooth_accel
            self._raw_speed = prev._raw_speed
            self._raw_brake_confirm_frames = prev._raw_brake_confirm_frames
            self._raw_brake_active = prev._raw_brake_active
            self._raw_brake_converged_frames = prev._raw_brake_converged_frames
            self._position_history = list(prev._position_history)
            self._speed_ema_history = list(prev._speed_ema_history)
            self._acc_speed_ema_history = list(prev._acc_speed_ema_history)
            self._acc_standstill = prev._acc_standstill
            self._acc_release_s = prev._acc_release_s
            if abs(self.angular_velocity) > _MAX_ANGULAR_VELOCITY:
                self.angular_velocity = 0.0
            self.speed = prev.speed
            self.acceleration = prev.acceleration
            self.acc_speed = prev.acc_speed
            self.acc_accel = prev.acc_accel

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
                # Snap pose to the latest buffer coords; do not recompute
                # _raw_speed here (Δpos/dt_sf spikes after pause re-anchors and
                # is unused by the filter chain until the next full update).
                rx = self.position.x
                rz = self.position.z
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
        self._acc_speed_ema = prev._acc_speed_ema
        self._acc_smooth_accel = prev._acc_smooth_accel
        self._raw_speed = prev._raw_speed
        self._raw_brake_confirm_frames = prev._raw_brake_confirm_frames
        self._raw_brake_active = prev._raw_brake_active
        self._raw_brake_converged_frames = prev._raw_brake_converged_frames
        self._position_history = list(prev._position_history)
        self._speed_ema_history = list(prev._speed_ema_history)
        self._acc_speed_ema_history = list(prev._acc_speed_ema_history)
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
                    self._reset_raw_brake_transient()
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
                    self._acc_speed_ema = self.speed
                    self._acc_smooth_accel = 0.0
                    self.acc_speed = self.speed
                    self.acc_accel = 0.0
                    self._raw_speed = 0.0
                    if self._smooth_x is not None:
                        self.position.x = self._smooth_x
                        self.position.z = self._smooth_z
                    return
                else:
                    self._reset_raw_brake_transient()
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
            self._raw_brake_confirm_frames = 0
            if self._smooth_x is not None:
                self.position.x = self._smooth_x
                self.position.z = self._smooth_z
            self.speed = prev.speed
            self.acceleration = prev.acceleration
            self.acc_speed = prev.acc_speed
            self.acc_accel = prev.acc_accel
            self._smooth_accel = prev._smooth_accel
            self._acc_smooth_accel = prev._acc_smooth_accel
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
        # ACC reads the long window; AEB reads whatever the brake transient
        # selects. On a confirmed crash both read the same unfiltered estimate so
        # nothing extra sits between a crashed vehicle and either consumer.
        long_raw_speed = raw_speed
        raw_speed = self._select_raw_speed(
            raw_speed, self.speed, fwd_x, fwd_z,
        )
        acc_raw_speed = raw_speed if self.crash_confirmed else long_raw_speed
        responsive_brake_decel = 0.0
        if self._raw_brake_active:
            recent_decel = _hard_brake_decel_from_position_history(
                self._position_history, fwd_x, fwd_z,
            )
            if recent_decel is not None:
                responsive_brake_decel = min(
                    recent_decel, _ACC_SPEED_FF_ACCEL_CLAMP_MS2,
                )

        (speed_ema, accel, speed_corr, speed_ema_history,
         acc_speed_ema, acc_accel, acc_speed_ema_history, acc_speed,
         acc_standstill, acc_release_s) = _smooth_vehicle_kinematics(
            raw_speed, acc_raw_speed, t_now, dt,
            prev._speed_ema, prev._smooth_accel, prev._speed_ema_history,
            prev._acc_speed_ema, prev._acc_smooth_accel,
            prev._acc_speed_ema_history, prev.acc_speed,
            prev._acc_standstill, prev._acc_release_s,
            responsive_brake_decel,
        )
        self._raw_speed = raw_speed
        self._speed_ema = speed_ema
        self._speed_ema_history = speed_ema_history
        self._smooth_accel = accel
        self._smooth_speed = speed_corr
        self.speed = speed_corr
        self.acceleration = accel
        self._acc_speed_ema = acc_speed_ema
        self._acc_speed_ema_history = acc_speed_ema_history
        self._acc_smooth_accel = acc_accel
        self.acc_speed = acc_speed
        self.acc_accel = acc_accel
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
        body_capsule: bool = False,
    ) -> ArcPath:
        """ArcPath for this vehicle from smoothed pose and curvature.

        Curvature is derived from position history when available (circumscribed
        circle fit), falling back to angular_velocity / speed. Crash-induced
        backward position spikes are suppressed by the 6 m/s² cap in _accel_to_arc_params().

        body_capsule=True gives the arc body extents (fwd_len/back_len) so the
        collision test covers the whole vehicle length. Extents are measured
        from the arc reference to the body front/rear using the same AI/TMP
        pivot convention as get_corners, so they stay correct despite the
        arc_start_pctg reference offset. Default False keeps point/disc behavior
        for non-AEB consumers.
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

        cap_fwd_len = 0.0
        cap_back_len = 0.0
        if body_capsule:
            # Symmetric +/- length/2 for AI and TMP alike (AGENTS.md §6).
            front_d = self.size.length * 0.5
            back_d = self.size.length * 0.5
            cap_fwd_len = max(front_d - body_offset, 0.0)
            cap_back_len = max(back_d + body_offset, 0.0)

        return build_arc(
            start_x, start_z, yaw_rad, self.speed,
            curvature, effective_hw, horizon,
            decel=effective_decel, accel=effective_accel,
            fwd_len=cap_fwd_len, back_len=cap_back_len,
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()

    def get_corners(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """World-space footprint corners (front-right, front-left, back-left, back-right).

        Symmetric ± length/2 about the pivot for AI and TMP alike (AGENTS.md
        §6). Yaw comes from ``_smooth_yaw`` when available.
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
        front_d = self.size.length * 0.5
        back_d = self.size.length * 0.5
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

