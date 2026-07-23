"""Trail-arc fit: target path vs ego row for offset_m and angle_amp.

Downsample, LS circle fit, baselines: ``core/acc/README.md`` §3."""

from __future__ import annotations

import math
from dataclasses import dataclass


# Downsample raw history: min 1 m and 0.05 s between kept samples (README §3).
_HISTORY_MIN_DIST_M: float = 1.0
_HISTORY_MIN_DT_S: float = 0.05

# Legacy fit_circle gate: five samples minimum for a stable LS fit.
_MIN_FIT_SAMPLES: int = 5

# Total downsampled chord must exceed this (near-stationary → NO_HISTORY).
_MIN_PATH_LEN_M: float = 0.5

# κ below 1/2000 m⁻¹ treated as straight at scoring horizons.
_STRAIGHT_KAPPA_MAX: float = 1.0 / 2000.0

# Sagitta ratio gate: collapse noisy LS circles to straight (README §3).
_MIN_SAGITTA_RATIO: float = 0.5

# angle_amp Gaussian σ (rad); legacy SCORING_REFERENCE §8.1.
_ANGLE_AMP_SIGMA: float = 0.06


@dataclass(slots=True, frozen=True)
class TrailFit:
    """Straight (point+dir) or curved (centre+radius+sign); see README §3 trail-arc."""
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
    """Gate history by ``_HISTORY_MIN_DIST_M`` and ``_HISTORY_MIN_DT_S``; keep first/last."""
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
    # Always keep newest sample for trailing-chord direction.
    if len(history) > 1:
        last = history[-1]
        if not kept or kept[-1] is not last:
            kept.append(last)
    return kept


def _ls_circle_fit(
    samples: list[tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Algebraic LS circle on (x,z); data mean-centred for TMP coordinate scale (README §3)."""
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

    # Centred normal equations: 2×2 block for (A,B) plus C row.
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
    """Downsample, LS circle or straight line; sagitta gate; README §3 constants."""
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

    # Long baseline first→last for straight direction; trailing chord for tangent.
    long_dx = target_x - first_x
    long_dz = target_z - first_z
    long_len = math.hypot(long_dx, long_dz)
    if long_len > 1e-3:
        long_ux = long_dx / long_len
        long_uz = long_dz / long_len
    else:
        long_ux = -math.sin(target_yaw_rad_fallback)
        long_uz = -math.cos(target_yaw_rad_fallback)

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

    # Reject LS circle when interior sagitta is below ``_MIN_SAGITTA_RATIO`` of expected.
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

    # Sweep sign from prev→centre × target→centre (matches ``ArcPath._sign``).
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
    """(offset_m, arc_angle_rad) at trail ∩ ego row, or None → NO_ARC_HIT baseline."""
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
    # Tangent at crossing: sign·(rz, -rx) per ArcPath sweep convention.
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
    """``2^(-(arc_angle/σ)²)`` with σ = ``_ANGLE_AMP_SIGMA`` (README §3)."""
    x = arc_angle_rad / _ANGLE_AMP_SIGMA
    return math.exp(-(x * x) * math.log(2.0))


# Re-exports for tests / debug window.
MIN_FIT_SAMPLES = _MIN_FIT_SAMPLES
MIN_PATH_LEN_M = _MIN_PATH_LEN_M
ANGLE_AMP_SIGMA = _ANGLE_AMP_SIGMA
HISTORY_MIN_DIST_M = _HISTORY_MIN_DIST_M
HISTORY_MIN_DT_S = _HISTORY_MIN_DT_S

