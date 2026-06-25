"""
ACC trail-arc fit: target arc projected onto the ego row.

Closes the gap documented in ``core/acc/AGENTS.md §3``. For each target,
the scorer asks: *"where would this vehicle's path cross the line
through ego perpendicular to ego heading, and at what angle?"*  The
two outputs

    offset_m    signed lateral from ego centerline at the arc-crossing
                of the ego row.  Positive = right of ego.
    arc_angle   absolute angle (rad) between the target's tangent at
                the crossing and ego's forward axis.  Drives
                ``angle_amp = 2^(-(arc_angle/0.06)²)`` in
                ``scoring.offset_component``.

feed ``scoring.offset_component`` and replace the "current lateral as
fallback" stand-in.

This module owns its own sampling and curvature pipeline so the radar /
TMP-speed / AEB code paths can keep their dense ``_position_history``
buffers untouched.  We downsample the raw buffer ourselves
(``_HISTORY_MIN_DIST_M = 1.0``, ``_HISTORY_MIN_DT_S = 0.05``) and fit
an algebraic least-squares circle directly to the kept points: the
centre, radius, and direction-of-travel sign all come from the
positions, not from the smoothed yaw.  That removes the yaw-jitter
sensitivity the previous "kappa from radar + centre from yaw" version
suffered at low speeds.

Two failure modes: both surfaced to the caller as the legacy baseline
buckets (`HIT` / `NO_ARC_HIT` / `NO_HISTORY`) from ``scoring``:

    fit_trail returns None
        fewer than ``_MIN_FIT_SAMPLES`` downsampled points, total
        chord < ``_MIN_PATH_LEN_M``, or the LS fit is singular
        (perfectly collinear and we couldn't recover a line direction).
        → NO_HISTORY
    crossing_offset_and_angle returns None
        target's fitted line/circle does not intersect the ego row.
        → NO_ARC_HIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Per-fit downsampling.  At raw radar cadence (~30 Hz) low-speed buffers
# bunch within centimetres and the LS fit's normal matrix collapses;
# at high speed the buffer is dense enough that every-other sample is
# redundant.  Keep only samples that are ≥ ``_HISTORY_MIN_DIST_M``
# apart AND ≥ ``_HISTORY_MIN_DT_S`` apart from the last kept sample.
_HISTORY_MIN_DIST_M: float = 1.0
_HISTORY_MIN_DT_S: float = 0.05

# Minimum kept samples before a fit is attempted.  Five is the legacy
# ``fit_circle`` gate and gives the 3×3 normal-equation system room to
# average out a noisy sample without collapsing.
_MIN_FIT_SAMPLES: int = 5

# Reject near-stationary tracks: sub-half-metre total chord across
# the downsampled window is well under the per-sample distance gate
# and would only happen if the buffer hasn't filled up yet.
_MIN_PATH_LEN_M: float = 0.5

# Below this curvature magnitude, treat the trail as a straight line.
# 1 / 2000 m ⇒ a 2 km-radius arc is indistinguishable from straight at
# realistic scoring ranges (≤150 m horizon).
_STRAIGHT_KAPPA_MAX: float = 1.0 / 2000.0

# Goodness-of-fit gate.  After the LS circle fit returns ``(cx, cz,
# R)``, we measure the actual max perpendicular distance from the
# first→last chord to any interior kept sample.  An R-of-truth curve
# of chord L produces sagitta ≈ L² / (8R); if the observed deviation
# is significantly less than that, the LS fit is being driven by
# position noise rather than real curvature and we collapse it to a
# straight line.  0.5 = "the curve has to account for at least half
# of the sagitta its own radius implies."  This catches the noise-
# driven small-R failure mode without rejecting genuine gentle curves
# whose sagittas happen to fall under any absolute threshold.
_MIN_SAGITTA_RATIO: float = 0.5

# Legacy SCORING_REFERENCE §8.1 angle-amp denominator.  σ = 0.06 rad
# (~3.4°).  Tangent within ~3.4° of ego fwd ⇒ amp ≥ 0.5; past ~7° amp
# falls below 1 / 16.
_ANGLE_AMP_SIGMA: float = 0.06


@dataclass(slots=True, frozen=True)
class TrailFit:
    """Result of fitting a trail to a target's position history.

    ``is_straight`` discriminates the active fields:

        straight  → (point_x, point_z, dir_x, dir_z)
        curved    → (center_x, center_z, radius, sign)

    ``sign`` follows ``ArcPath._sign``: +1 left-turning (centre on left
    of forward direction, target sweeps CW around centre), −1 right-
    turning (centre on right, sweeps CCW).  ``(dir_x, dir_z)`` is the
    target's unit forward direction derived from the trailing chord.
    """
    is_straight: bool
    center_x: float = 0.0
    center_z: float = 0.0
    radius: float = 0.0
    sign: float = 1.0
    point_x: float = 0.0
    point_z: float = 0.0
    dir_x: float = 1.0
    dir_z: float = 0.0


def _downsample(
    history: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Keep samples ≥ ``_HISTORY_MIN_DIST_M`` AND ≥ ``_HISTORY_MIN_DT_S``
    apart from the last kept one.  Always keeps the first and last raw
    samples so the final tangent reflects the freshest motion."""
    if not history:
        return []
    kept: list[tuple[float, float, float]] = [history[0]]
    for t, x, z in history[1:-1]:
        last_t, last_x, last_z = kept[-1]
        if (t - last_t) < _HISTORY_MIN_DT_S:
            continue
        dx = x - last_x
        dz = z - last_z
        if (dx * dx + dz * dz) < (_HISTORY_MIN_DIST_M * _HISTORY_MIN_DIST_M):
            continue
        kept.append((t, x, z))
    # Always append the newest sample so the trailing chord reflects
    # current motion even if it would have been gated out for being too
    # close to the previous kept sample.
    if len(history) > 1:
        last = history[-1]
        if not kept or kept[-1] is not last:
            kept.append(last)
    return kept


def _ls_circle_fit(
    samples: list[tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Algebraic LS circle fit on ``(t, x, z)`` samples.

    Solves Σ (x² + z² + A·x + B·z + C)² for (A, B, C).  Centre =
    (−A/2, −B/2), R² = A²/4 + B²/4 − C.  Returns ``(cx, cz, R)`` or
    ``None`` when the system is singular (collinear points) or the
    recovered radius is non-positive.

    **The data is centred around its mean before fitting.** In TMP
    world coordinates can sit in the 10⁵ range, which makes the raw
    normal equations operate on entries of order 10¹⁰–10²⁰ and the
    3 × 3 determinant collapses numerically: the recovered radius
    blows up to 10⁵+ regardless of the actual curvature, which is
    exactly the "always straight" symptom on a real map.  Centring
    decouples the (A, B) sub-system from C and keeps every entry of
    the normal matrix on a sane scale.
    """
    n = len(samples)
    if n < 3:
        return None

    mx = 0.0
    mz = 0.0
    for _, x, z in samples:
        mx += x
        mz += z
    mx /= n
    mz /= n

    sxx = syy = sxy = 0.0
    sxr = syr = sr = 0.0
    for _, x_w, z_w in samples:
        x = x_w - mx
        z = z_w - mz
        r2 = x * x + z * z
        sxx += x * x
        syy += z * z
        sxy += x * z
        sxr += x * r2
        syr += z * r2
        sr += r2

    # With centred data ``Σ x = Σ z = 0`` so the normal-equation
    # matrix decouples into a 2×2 block for (A, B) and an independent
    # equation for C.
    det_2 = sxx * syy - sxy * sxy
    if abs(det_2) < 1e-9:
        return None
    a = (-sxr * syy + syr * sxy) / det_2
    b = (-syr * sxx + sxr * sxy) / det_2
    c = -sr / n
    cx_c = -a * 0.5
    cz_c = -b * 0.5
    r2 = cx_c * cx_c + cz_c * cz_c - c
    if r2 <= 1e-6:
        return None
    return cx_c + mx, cz_c + mz, math.sqrt(r2)


def fit_trail(
    history: list[tuple[float, float, float]],
    target_yaw_rad_fallback: float,
) -> TrailFit | None:
    """LS-fit a circle (or line) to the target's downsampled history.

    ``target_yaw_rad_fallback`` is only consulted when the downsampled
    history collapses to two samples too close together to define a
    direction: in that degenerate case the smoothed yaw is used so
    the caller still gets a (best-effort) straight TrailFit instead of
    a hard ``None``.
    """
    samples = _downsample(history)
    if len(samples) < _MIN_FIT_SAMPLES:
        return None

    # Total path length across the kept subset.
    px, pz = samples[0][1], samples[0][2]
    path_len = 0.0
    for _, hx, hz in samples[1:]:
        path_len += math.hypot(hx - px, hz - pz)
        px, pz = hx, hz
    if path_len < _MIN_PATH_LEN_M:
        return None

    target_x, target_z = samples[-1][1], samples[-1][2]
    prev_x, prev_z = samples[-2][1], samples[-2][2]
    first_x, first_z = samples[0][1], samples[0][2]

    # Long-baseline direction: first → last sample.  Less jittery than
    # the last-two-samples chord; used as the straight-fit direction.
    long_dx = target_x - first_x
    long_dz = target_z - first_z
    long_len = math.hypot(long_dx, long_dz)
    if long_len > 1e-3:
        long_ux = long_dx / long_len
        long_uz = long_dz / long_len
    else:
        long_ux = -math.sin(target_yaw_rad_fallback)
        long_uz = -math.cos(target_yaw_rad_fallback)

    # Trailing chord: direction the target is heading right now.  This
    # is what we want for the freshest tangent on a curve.  Falls back
    # to the long chord (and then to smoothed yaw) only if the last
    # two kept samples coincide.
    chord_dx = target_x - prev_x
    chord_dz = target_z - prev_z
    chord_len = math.hypot(chord_dx, chord_dz)
    if chord_len > 1e-3:
        dir_x = chord_dx / chord_len
        dir_z = chord_dz / chord_len
    else:
        dir_x, dir_z = long_ux, long_uz

    fit = _ls_circle_fit(samples)
    if fit is None:
        return TrailFit(
            is_straight=True,
            point_x=target_x, point_z=target_z,
            dir_x=long_ux, dir_z=long_uz,
        )
    cx, cz, R = fit
    if (1.0 / R) < _STRAIGHT_KAPPA_MAX:
        return TrailFit(
            is_straight=True,
            point_x=target_x, point_z=target_z,
            dir_x=long_ux, dir_z=long_uz,
        )

    # Goodness-of-fit: compare observed max perpendicular against the
    # sagitta the LS radius implies for the first→last chord.  If the
    # data doesn't actually deviate from the chord enough to back up
    # the radius, the LS is fitting noise: collapse to straight.
    perp_x = -long_uz
    perp_z = long_ux
    max_perp = 0.0
    for _, x, z in samples[1:-1]:
        dx = x - first_x
        dz = z - first_z
        perp_d = abs(dx * perp_x + dz * perp_z)
        if perp_d > max_perp:
            max_perp = perp_d
    expected_sagitta = (long_len * long_len) / (8.0 * R)
    if max_perp < _MIN_SAGITTA_RATIO * expected_sagitta:
        return TrailFit(
            is_straight=True,
            point_x=target_x, point_z=target_z,
            dir_x=long_ux, dir_z=long_uz,
        )

    # ArcPath sign from the actual sweep direction.  v1 (prev →
    # centre) and v2 (target → centre) span a small wedge at the
    # centre; the cross product (v1 × v2) is positive iff target is
    # CCW from prev around centre.  ArcPath: +1 = left turn = sweep CW.
    v1x = prev_x - cx
    v1z = prev_z - cz
    v2x = target_x - cx
    v2z = target_z - cz
    cross = v1x * v2z - v1z * v2x
    sign = -1.0 if cross > 0.0 else 1.0
    return TrailFit(
        is_straight=False,
        center_x=cx, center_z=cz, radius=R, sign=sign,
        point_x=target_x, point_z=target_z,
        dir_x=dir_x, dir_z=dir_z,
    )


def crossing_offset_and_angle(
    fit: TrailFit,
    ego_x: float, ego_z: float,
    ego_fwd_x: float, ego_fwd_z: float,
) -> tuple[float, float] | None:
    """Return (offset_m, arc_angle_rad) at the trail's crossing of the
    ego row, or ``None`` when the trail does not reach the row.

    ``offset_m`` is signed, positive = right of ego centerline.
    ``arc_angle_rad`` ∈ [0, π]: 0 = target parallel to ego, π/2 =
    perpendicular, π = oncoming.  Caller feeds it through
    :func:`angle_amp_from`.
    """
    right_x = -ego_fwd_z
    right_z = ego_fwd_x

    if fit.is_straight:
        ddotf = fit.dir_x * ego_fwd_x + fit.dir_z * ego_fwd_z
        if abs(ddotf) < 1e-6:
            return None
        s_line = (
            (ego_x - fit.point_x) * ego_fwd_x
            + (ego_z - fit.point_z) * ego_fwd_z
        ) / ddotf
        cross_x = fit.point_x + s_line * fit.dir_x
        cross_z = fit.point_z + s_line * fit.dir_z
        offset = (cross_x - ego_x) * right_x + (cross_z - ego_z) * right_z
        cos_ang = max(-1.0, min(1.0, ddotf))
        return offset, math.acos(cos_ang)

    dx = ego_x - fit.center_x
    dz = ego_z - fit.center_z
    b_coef = 2.0 * (dx * right_x + dz * right_z)
    c_coef = dx * dx + dz * dz - fit.radius * fit.radius
    disc = b_coef * b_coef - 4.0 * c_coef
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    s1 = (-b_coef + sq) * 0.5
    s2 = (-b_coef - sq) * 0.5

    cross1_x = ego_x + s1 * right_x
    cross1_z = ego_z + s1 * right_z
    cross2_x = ego_x + s2 * right_x
    cross2_z = ego_z + s2 * right_z
    d1 = math.hypot(cross1_x - fit.point_x, cross1_z - fit.point_z)
    d2 = math.hypot(cross2_x - fit.point_x, cross2_z - fit.point_z)
    if d1 <= d2:
        s = s1
        cross_x, cross_z = cross1_x, cross1_z
    else:
        s = s2
        cross_x, cross_z = cross2_x, cross2_z
    offset = s

    rx = cross_x - fit.center_x
    rz = cross_z - fit.center_z
    # ArcPath ``max_sweep = -sign · arc_len / radius`` so target sweeps
    # CW around centre when sign = +1 and CCW when sign = -1.  CW
    # tangent at (rx, rz) is (rz, -rx); CCW is (-rz, rx).  Combined:
    # sign·(rz, -rx).
    tan_x = rz * fit.sign
    tan_z = -rx * fit.sign
    tan_mag = math.hypot(tan_x, tan_z)
    if tan_mag < 1e-9:
        return None
    tan_x /= tan_mag
    tan_z /= tan_mag
    cos_ang = max(-1.0, min(1.0, tan_x * ego_fwd_x + tan_z * ego_fwd_z))
    return offset, math.acos(cos_ang)


def angle_amp_from(arc_angle_rad: float) -> float:
    """``2^(-(arc_angle/0.06)²)``: legacy SCORING_REFERENCE §8.1.

    Sharp cutoff: at ≈3.4° (= σ) amp = 0.5; at ≈7° amp = 1/16.
    Perpendicular / oncoming arcs collapse to near zero.
    """
    x = arc_angle_rad / _ANGLE_AMP_SIGMA
    return math.exp(-(x * x) * math.log(2.0))


# Re-exports for tests / debug window.
MIN_FIT_SAMPLES = _MIN_FIT_SAMPLES
MIN_PATH_LEN_M = _MIN_PATH_LEN_M
ANGLE_AMP_SIGMA = _ANGLE_AMP_SIGMA
HISTORY_MIN_DIST_M = _HISTORY_MIN_DIST_M
HISTORY_MIN_DT_S = _HISTORY_MIN_DT_S

