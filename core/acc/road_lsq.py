"""Least-squares kernels of the road model fit, unrolled for its four coefficients.

Every sum adds in the order of the loops they replaced, so the fit is bit-identical
(``core/acc/README.md`` §9, "Cost").
"""

from __future__ import annotations

import math


def solve(a: list[list[float]], b: list[float]) -> tuple[float, ...] | None:
    """Gaussian elimination with partial pivoting on an n x n system."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = m[row][n]
        for col in range(row + 1, n):
            acc -= m[row][col] * out[col]
        out[row] = acc / m[row][row]
    return tuple(out)


def predict(beta, basis) -> float:
    # sum() kept: from Python 3.12 it adds floats compensated, so a plain + chain would differ.
    return sum((basis[0] * beta[0], basis[1] * beta[1], basis[2] * beta[2], basis[3] * beta[3]))


def residuals(rows, beta) -> list[float]:
    """``y - predict(beta, basis)`` for every row, in order."""
    c0, c1, c2, c3 = beta
    return [y - sum((b0 * c0, b1 * c1, b2 * c2, b3 * c3)) for _, (b0, b1, b2, b3), y, _ in rows]


def normal_equations(rows) -> tuple[list[list[float]], list[float]]:
    """Sums of ``w*b_i*b_j`` and ``w*b_i*y`` over the rows, priors not yet added."""
    a00 = a01 = a02 = a03 = a10 = a11 = a12 = a13 = 0.0
    a20 = a21 = a22 = a23 = a30 = a31 = a32 = a33 = 0.0
    t0 = t1 = t2 = t3 = 0.0
    for _, (b0, b1, b2, b3), y, w in rows:
        w0 = w * b0
        w1 = w * b1
        w2 = w * b2
        w3 = w * b3
        t0 += w0 * y
        a00 += w0 * b0
        a01 += w0 * b1
        a02 += w0 * b2
        a03 += w0 * b3
        t1 += w1 * y
        a10 += w1 * b0
        a11 += w1 * b1
        a12 += w1 * b2
        a13 += w1 * b3
        t2 += w2 * y
        a20 += w2 * b0
        a21 += w2 * b1
        a22 += w2 * b2
        a23 += w2 * b3
        t3 += w3 * y
        a30 += w3 * b0
        a31 += w3 * b1
        a32 += w3 * b2
        a33 += w3 * b3
    ata = [[a00, a01, a02, a03], [a10, a11, a12, a13], [a20, a21, a22, a23], [a30, a31, a32, a33]]
    return ata, [t0, t1, t2, t3]


def circle_kappa(points: list[tuple[float, float]], min_span_m: float) -> float | None:
    """Signed curvature of the least-squares circle through a source's own
    samples; + is left, and a straight trail reads as zero.

    Its own trail, so its lane offset cancels instead of reading as a bend the
    way a bearing from ego would. Least squares over the whole trail rather
    than three points off it: the three-point circle reads the road off two
    gaps and jitters frame to frame, which the base arc then passes on to
    every lateral at once. None when the trail spans less than ``min_span_m``."""
    ordered = sorted(points, key=lambda p: p[0] * p[0] + p[1] * p[1])
    (x1, y1), (xn, yn) = ordered[0], ordered[-1]
    if math.hypot(xn - x1, yn - y1) < min_span_m:
        return None
    # Mean-centred algebraic fit: |p|² + D·x + E·y + F = 0.
    mx = sum(p[0] for p in ordered) / len(ordered)
    my = sum(p[1] for p in ordered) / len(ordered)
    # Row (x, y, 1): products with 1.0 are exact and x*y == y*x, so the sums match a 3x3 loop.
    sxx = sxy = sx = syy = sy = t0 = t1 = t2 = 0.0
    for px, py in ordered:
        x, y = px - mx, py - my
        rhs = -(x * x + y * y)
        t0 += x * rhs
        sxx += x * x
        sxy += x * y
        sx += x
        t1 += y * rhs
        syy += y * y
        sy += y
        t2 += rhs
    ata = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(len(ordered))]]
    atb = [t0, t1, t2]
    sol = solve(ata, atb)
    if sol is None:
        return None
    cx, cy = -0.5 * sol[0], -0.5 * sol[1]
    radius_sq = cx * cx + cy * cy - sol[2]
    if radius_sq <= 1e-9:
        return None
    radius = math.sqrt(radius_sq)
    # Centre to the right of travel means the road turns right, so kappa < 0.
    cross = (xn - x1) * (cy - (y1 - my)) - (yn - y1) * (cx - (x1 - mx))
    return (-1.0 if cross > 0.0 else 1.0) / radius
