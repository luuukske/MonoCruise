"""
Shared ego path helpers — curvature from position history.

Used by RadarThread to publish a geometry-based ego curvature alongside ego
state. Mirrors ``Vehicle._compute_curvature`` in ``traffic.py`` so ego and
target curvatures are derived from the same circumscribed-circle math over
equivalent-length position histories (``_POSITION_HISTORY_LEN``).

Callers:
    - AEB — uses ``ego_curvature_from_history(...)`` in place of the yaw-rate
      proxy ``steer * speed * 12.0 / speed`` when a valid fit is available.
    - ACC — same curvature feeds the path-arc scoring so in-path target
      selection matches AEB's predicted corridor.
"""

from __future__ import annotations

import math

from .traffic import _MIN_CURVATURE_RADIUS, _POSITION_HISTORY_LEN


EGO_POSITION_HISTORY_LEN: int = _POSITION_HISTORY_LEN

# Minimum chord length before a circle fit is attempted; below this the
# samples are treated as "standing still".  Matches Vehicle fit gate.
_MIN_TOTAL_CHORD: float = 0.05


def ego_curvature_from_history(
    history: list[tuple[float, float, float]],
) -> float | None:
    """Curvature (1/m) via circumscribed-circle fit on (t, x, z) samples.

    Signed: positive = left turn (κ > 0), matching ArcPath convention.
    Returns None if < 3 samples; 0.0 when near-stationary or near-collinear.
    Callers fall back to the yaw-rate proxy when this returns None.
    """
    if len(history) < 3:
        return None

    _, x0, z0 = history[0]
    _, xn, zn = history[-1]
    if (xn - x0) ** 2 + (zn - z0) ** 2 < _MIN_TOTAL_CHORD * _MIN_TOTAL_CHORD:
        return 0.0

    n = len(history)
    candidates = [(0, n // 2, n - 1)]
    if n >= 5:
        candidates.append((1, (n - 1) // 2, n - 2))
    if n >= 7:
        candidates += [(0, n // 3, n - 1), (0, 2 * n // 3, n - 1)]

    total_k = 0.0
    count = 0
    for i, j, k in candidates:
        _, ax, az = history[i]
        _, bx, bz = history[j]
        _, cx, cz = history[k]
        if (bx - ax) ** 2 + (bz - az) ** 2 < _MIN_TOTAL_CHORD * _MIN_TOTAL_CHORD:
            continue
        if (cx - bx) ** 2 + (cz - bz) ** 2 < _MIN_TOTAL_CHORD * _MIN_TOTAL_CHORD:
            continue
        D = 2.0 * (ax * (bz - cz) + bx * (cz - az) + cx * (az - bz))
        if abs(D) < 1e-6:
            count += 1
            continue
        a2 = ax * ax + az * az
        b2 = bx * bx + bz * bz
        c2 = cx * cx + cz * cz
        ux = (a2 * (bz - cz) + b2 * (cz - az) + c2 * (az - bz)) / D
        uz = -(a2 * (bx - cx) + b2 * (cx - ax) + c2 * (ax - bx)) / D
        R = max(math.sqrt((ax - ux) ** 2 + (az - uz) ** 2), _MIN_CURVATURE_RADIUS)
        cross = (bx - ax) * (cz - bz) - (bz - az) * (cx - bx)
        sign = -1.0 if cross > 0 else (1.0 if cross < 0 else 0.0)
        total_k += sign / R
        count += 1

    if count == 0:
        return None
    return total_k / count
