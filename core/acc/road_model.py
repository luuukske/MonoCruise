"""Shared road centreline fitted from ego's own path and the traffic ahead.

One weighted solve per frame, stateless; the frame-to-frame carry lives in
``road_smoother``. Parameterisation, the per-source offset elimination, and why
fusion happens in sample space: ``core/acc/README.md`` §9."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# Basis scale (m): v = s / _S_REF_M keeps the cubic design matrix conditioned
# over the 150 m scoring range instead of spanning 10^6 in the s³ column.
_S_REF_M: float = 100.0

# Robust reweighting: Huber threshold on the per-sample fit residual, in metres.
_HUBER_DELTA_M: float = 1.0
_IRLS_PASSES: int = 2

# Source-level rejection: a vehicle changing lanes is inconsistent as a whole,
# not sample by sample, so its entire contribution is scaled down together.
_SOURCE_RESIDUAL_DELTA_M: float = 0.35
_SOURCE_REJECT_POWER: float = 2.0

# Confidence ramp on total effective sample weight after offset elimination.
_CONF_WEIGHT_MIN: float = 4.0
_CONF_WEIGHT_FULL: float = 30.0

# Agreement ramp, on an upper quantile of the per-source residuals: count-stable
# and dissenter-tolerant, which neither a range nor an RMS is (README §9).
_CONF_AGREE_QUANTILE: float = 0.75
_CONF_RESIDUAL_GOOD_M: float = 0.10
_CONF_RESIDUAL_BAD_M: float = 0.60

# Agreement, not volume. Source-level rejection has no quorum, so a lone source
# is unchecked and a vehicle turning off is absorbed as road shape. README §9.
_CONF_SINGLE_SOURCE_CAP: float = 0.25

# A source needs this many samples to say anything about shape once its own
# lateral offset is eliminated.
_MIN_SOURCE_SAMPLES: int = 3

# Raising this was measured to change nothing once the base arc carries ego's
# curvature exactly, so it stays at parity. Dropped entirely when re-based.
_EGO_SAMPLE_WEIGHT: float = 1.0

# Confidence falls to zero this far past the furthest sample.
_EXTRAPOLATION_FADE_M: float = 30.0

# Ridge weight pulling the centreline's heading at ego (c1) toward ego's own
# heading. Comparable to total sample weight; 0 leaves it free.
_HEADING_PRIOR_WEIGHT: float = 400.0

# Number of terms in n(s). Four, because with c1 pinned by the heading prior a
# cubic leaves only a linear curvature ramp, and a corner is a step (README §9).
_N_COEF: int = 4

# Fixed lookahead grid the smoothed centreline is carried on, in arc length.
# Sample space, not coefficient space: filtering the cubic terms sloshes.
_NODE_STEP_M: float = 10.0
_NODE_COUNT: int = 16
_NODE_S: tuple[float, ...] = tuple(i * _NODE_STEP_M for i in range(_NODE_COUNT))

# Base-arc guards: below this curvature the road is straight, and the arc is
# only evaluated within this fraction of its radius (beyond it x is degenerate).
_STRAIGHT_KAPPA: float = 1.0 / 5000.0
_BASE_ARC_MAX_FRAC: float = 0.95

# Re-basing the arc onto the traffic. Gated on the source's own trail span, not
# its range: the vote is that trail's curvature (README §9).
_REBASE_MIN_SPAN_M: float = 15.0
_REBASE_MAX_KAPPA: float = 1.0 / 25.0

# Damped steps used to walk arc length onto a requested forward distance.
_INVERT_PASSES: int = 12

# Half-step of the central difference the centreline tangent is read from.
_TANGENT_STEP_M: float = 0.5

# Fraction of half the circumference the arc parameterisation will use: at
# exactly pi*R the far end is opposite ego and its sign is ambiguous.
_ARC_MAX_SPAN_FRAC: float = 0.97

# Measured lateral error of the blended estimate against ego's own future path
# (clip corpus): 0.65 m at 30-60 m, 2.0 m at 60-90 m, 3.7 m at 90-130 m.
_SIGMA_MIN_M: float = 0.25
_SIGMA_KNEE_M: float = 35.0
_SIGMA_SLOPE: float = 0.047


def base_arc_lateral(kappa: float, x_m: float) -> float:
    """Exact circular-arc offset through ego at curvature ``kappa``.

    Indexed by forward distance, so it saturates at ``_BASE_ARC_MAX_FRAC`` of
    the radius. Kept for drawing and for tests; the fit itself is indexed by arc
    length and has no such limit. See README §9."""
    if abs(kappa) < _STRAIGHT_KAPPA:
        return 0.0
    radius = 1.0 / abs(kappa)
    sign = 1.0 if kappa > 0.0 else -1.0
    # Past the circle's forward extent this parameterisation is degenerate.
    reach = min(abs(x_m), radius * _BASE_ARC_MAX_FRAC)
    return sign * (math.sqrt(max(radius * radius - reach * reach, 0.0)) - radius)


def arc_point(kappa: float, s_m: float) -> tuple[float, float]:
    """Ego-frame point at arc length ``s_m`` along the base circle."""
    if abs(kappa) < _STRAIGHT_KAPPA:
        return s_m, 0.0
    radius = 1.0 / abs(kappa)
    sign = 1.0 if kappa > 0.0 else -1.0
    theta = s_m / radius
    return radius * math.sin(theta), -sign * radius * (1.0 - math.cos(theta))


def arc_normal(kappa: float, s_m: float) -> tuple[float, float]:
    """Unit normal to ego's right at arc length ``s_m``; (0, 1) at ego."""
    if abs(kappa) < _STRAIGHT_KAPPA:
        return 0.0, 1.0
    sign = 1.0 if kappa > 0.0 else -1.0
    theta = s_m * abs(kappa)
    return sign * math.sin(theta), math.cos(theta)


def arc_span_limit(kappa: float) -> float:
    """Largest ``|s|`` this base arc resolves unambiguously.

    A circle closes, so arc length is periodic and a sample past ``pi·R`` reads
    as one on the near side. A straight road has no limit; ego steering hard at
    a standstill reports R = 7 m and has almost none, which is correct, because
    that is a manoeuvre rather than a road anyone is tracking traffic along."""
    if abs(kappa) < _STRAIGHT_KAPPA:
        return math.inf
    return _ARC_MAX_SPAN_FRAC * math.pi / abs(kappa)


def arc_coords(kappa: float, x_m: float, y_m: float) -> tuple[float, float]:
    """Ego-frame point to (arc length, signed normal offset) on the base circle.

    This is the inverse of ``arc_point`` plus ``arc_normal`` and it is exact for
    any heading change up to half a turn either way. Arc length stays monotone
    around a bend, which is the whole reason the fit is indexed by it: a forward
    distance stops being unique at 90 deg and folds back after it."""
    if abs(kappa) < _STRAIGHT_KAPPA:
        return x_m, y_m
    radius = 1.0 / abs(kappa)
    sign = 1.0 if kappa > 0.0 else -1.0
    # Centre of the base circle, and the point's position relative to it.
    vx = x_m
    vy = y_m + sign * radius
    return radius * math.atan2(vx, sign * vy), sign * (math.hypot(vx, vy) - radius)


@dataclass(slots=True, frozen=True)
class RoadModel:
    """Centreline through ego: exact base arc plus a quartic normal deviation.

    ``n(s) = c1·v + c2·v² + c3·v³ + c4·v⁴`` with ``v = s / 100 m``, where ``s``
    is arc length along the base circle and ``n`` is offset along its normal,
    positive to ego's right. The centreline is
    ``arc_point(s) + n(s)·arc_normal(s)``. Anchored at ego, so ``n(0) = 0``."""

    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    base_kappa: float = 0.0
    confidence: float = 0.0
    residual_rms_m: float = 0.0
    n_samples: int = 0
    n_sources: int = 0
    # Furthest arc length any sample reached; beyond it the cubic is
    # extrapolating and its confidence has to decay.
    support_s_m: float = 0.0
    # Per-source residual RMS about the shared fit; drives the caller's trust EMA.
    source_rms: dict = field(default_factory=dict)
    # Inputs the smoother needs to recompute confidence on a filtered residual.
    agreement_rms_m: float = 0.0
    target_weight: float = 0.0
    # Temporally smoothed deviation on ``_NODE_S``; empty means evaluate the fit.
    nodes: tuple = ()

    def confidence_at(self, s_m: float) -> float:
        """Confidence for a query at arc length ``s_m``.

        A cubic fitted to samples ending at 60 m says nothing trustworthy at
        140 m, however well it fits the samples it does have."""
        if self.confidence <= 0.0:
            return 0.0
        if abs(s_m) > arc_span_limit(self.base_kappa):
            return 0.0
        if s_m <= self.support_s_m:
            return self.confidence
        over = s_m - self.support_s_m
        decay = max(0.0, 1.0 - over / _EXTRAPOLATION_FADE_M)
        return self.confidence * decay

    def deviation_at(self, s_m: float) -> float:
        """Centreline offset from the base arc, along the arc's normal."""
        if self.nodes:
            return _interp_nodes(self.nodes, s_m)
        return self.raw_deviation_at(s_m)

    def raw_deviation_at(self, s_m: float) -> float:
        """This frame's fit alone, before any temporal smoothing."""
        v = s_m / _S_REF_M
        return v * (self.c1 + v * (self.c2 + v * (self.c3 + v * self.c4)))

    def point_at(self, s_m: float) -> tuple[float, float]:
        """Ego-frame (forward, right) point on the centreline at arc length s."""
        bx, by = arc_point(self.base_kappa, s_m)
        nx, ny = arc_normal(self.base_kappa, s_m)
        deviation = self.deviation_at(s_m)
        return bx + deviation * nx, by + deviation * ny

    def tangent_at(self, s_m: float) -> tuple[float, float]:
        """Unit direction the centreline runs at arc length ``s_m``, ego frame.

        Central difference rather than the analytic derivative, so it follows
        the smoothed nodes when they are set instead of the raw cubic."""
        ax, ay = self.point_at(s_m - _TANGENT_STEP_M)
        bx, by = self.point_at(s_m + _TANGENT_STEP_M)
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        return (1.0, 0.0) if norm < 1e-9 else (dx / norm, dy / norm)

    def road_coords(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Ego-frame point to (arc length, offset right of the centreline).

        Prefer this over ``lateral_at``: it is defined all the way around a
        bend, where a forward distance is not."""
        s_m, normal = arc_coords(self.base_kappa, x_m, y_m)
        return s_m, normal - self.deviation_at(s_m)

    def offset_of(self, x_m: float, y_m: float) -> float:
        """Road-relative lateral of a point: + is right of the centreline."""
        return self.road_coords(x_m, y_m)[1]

    def lateral_at(self, x_m: float) -> float:
        """Centreline lateral at forward distance ``x_m``, for drawing and tests.

        A forward distance stops being unique past 90 deg of heading change, so
        this saturates where the base arc does. Anything holding a 2D point
        should call ``road_coords``, which has no such limit."""
        kappa = self.base_kappa
        if abs(kappa) < _STRAIGHT_KAPPA:
            s_m = x_m
        else:
            radius = 1.0 / abs(kappa)
            reach = min(abs(x_m), radius * _BASE_ARC_MAX_FRAC)
            s_m = math.copysign(radius * math.asin(reach / radius), x_m)
        # The deviation moves the point along the arc's normal, which shifts its
        # forward distance too; walk s onto the requested x before reading y.
        px, py = self.point_at(s_m)
        for _ in range(_INVERT_PASSES):
            step = s_m + (x_m - px) * 0.5
            nx, ny = self.point_at(step)
            if abs(nx - x_m) >= abs(px - x_m):
                break
            s_m, px, py = step, nx, ny
        return py

    def curvature_at(self, s_m: float) -> float:
        """Signed curvature (1/m) at arc length s; + is left, matching ArcPath."""
        v = s_m / _S_REF_M
        d2 = (
            2.0 * self.c2 + 6.0 * self.c3 * v + 12.0 * self.c4 * v * v
        ) / (_S_REF_M * _S_REF_M)
        return self.base_kappa - d2


def _interp_nodes(nodes: tuple, s_m: float) -> float:
    """Linear interpolation over ``_NODE_S``, extrapolating from the end slopes."""
    last = len(nodes) - 1
    if s_m <= _NODE_S[0]:
        slope = (nodes[1] - nodes[0]) / _NODE_STEP_M
        return nodes[0] + slope * (s_m - _NODE_S[0])
    if s_m >= _NODE_S[last]:
        slope = (nodes[last] - nodes[last - 1]) / _NODE_STEP_M
        return nodes[last] + slope * (s_m - _NODE_S[last])
    idx = int(s_m / _NODE_STEP_M)
    idx = max(0, min(last - 1, idx))
    frac = (s_m - _NODE_S[idx]) / _NODE_STEP_M
    return nodes[idx] + frac * (nodes[idx + 1] - nodes[idx])





def lateral_sigma_m(x_m: float) -> float:
    """1-sigma lateral uncertainty of any lane estimate at forward distance x.

    Measured, not assumed: see the constants above and ``core/acc/README.md`` §9.
    Consumers inflate their in-lane test by this so a distant target cannot be
    called in-lane on an estimate that cannot resolve a lane width."""
    return _SIGMA_MIN_M + _SIGMA_SLOPE * max(0.0, x_m - _SIGMA_KNEE_M)


def from_curvature(kappa: float) -> RoadModel:
    """Constant-curvature fallback when there is no traffic evidence."""
    return RoadModel(base_kappa=kappa, confidence=0.0)


def _solve(a: list[list[float]], b: list[float]) -> tuple[float, ...] | None:
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


def _basis(s_m: float) -> tuple[float, ...]:
    v = s_m / _S_REF_M
    out = []
    power = 1.0
    for _ in range(_N_COEF):
        power *= v
        out.append(power)
    return tuple(out)


def _predict(beta, basis) -> float:
    return sum(b * c for b, c in zip(basis, beta))


_EGO_SOURCE_ID = -1


def _grouped_rows(
    ego_samples: list[tuple[float, float]],
    target_samples: list[tuple[int, float, float, float]],
) -> tuple[list[tuple[int, tuple[float, float, float], float, float]], int]:
    """Design rows with each source's own lateral offset eliminated.

    Ego is the reference so its rows pass through untouched; every other source
    is centred on its own mean, which removes the unknown lane offset while
    keeping the shape it contributes."""
    rows: list[tuple[int, tuple[float, ...], float, float]] = []
    for x, y in ego_samples:
        rows.append((_EGO_SOURCE_ID, _basis(x), y, _EGO_SAMPLE_WEIGHT))

    grouped: dict[int, list[tuple[float, float, float]]] = {}
    for source_id, x, y, weight in target_samples:
        if weight <= 0.0:
            continue
        grouped.setdefault(source_id, []).append((x, y, weight))

    n_sources = 0
    for source_id, samples in grouped.items():
        if len(samples) < _MIN_SOURCE_SAMPLES:
            continue
        total_w = sum(w for _, _, w in samples)
        if total_w <= 0.0:
            continue
        mean_b = [0.0] * _N_COEF
        mean_y = 0.0
        for x, y, w in samples:
            basis = _basis(x)
            for i in range(_N_COEF):
                mean_b[i] += w * basis[i]
            mean_y += w * y
        for i in range(_N_COEF):
            mean_b[i] /= total_w
        mean_y /= total_w
        for x, y, w in samples:
            basis = _basis(x)
            centred = tuple(basis[i] - mean_b[i] for i in range(_N_COEF))
            rows.append((source_id, centred, y - mean_y, w))
        n_sources += 1
    return rows, n_sources


def _weighted_fit(rows) -> tuple[tuple[float, ...], float] | None:
    """Weighted normal equations plus the weighted residual RMS."""
    ata = [[0.0] * _N_COEF for _ in range(_N_COEF)]
    atb = [0.0] * _N_COEF
    for _, basis, y, w in rows:
        for i in range(_N_COEF):
            atb[i] += w * basis[i] * y
            for j in range(_N_COEF):
                ata[i][j] += w * basis[i] * basis[j]
    # Heading prior: the road leaves ego along ego's heading, so c1 is pulled to
    # zero. Without it a far source's local shape tilts the whole centreline.
    ata[0][0] += _HEADING_PRIOR_WEIGHT
    beta = _solve(ata, atb)
    if beta is None:
        return None
    total_w = 0.0
    sq = 0.0
    for _, basis, y, w in rows:
        sq += w * (y - _predict(beta, basis)) ** 2
        total_w += w
    rms = math.sqrt(sq / total_w) if total_w > 0.0 else 0.0
    return beta, rms


def _source_residuals(rows, beta) -> dict[int, float]:
    """Per-source residual RMS about the shared fit, ego excluded."""
    acc: dict[int, list[float]] = {}
    for source_id, basis, y, w in rows:
        if source_id == _EGO_SOURCE_ID:
            continue
        bucket = acc.setdefault(source_id, [0.0, 0.0])
        bucket[0] += w * (y - _predict(beta, basis)) ** 2
        bucket[1] += w
    return {
        sid: math.sqrt(sq / tw) if tw > 0.0 else math.inf
        for sid, (sq, tw) in acc.items()
    }


def _source_scales(rows, beta) -> dict[int, float]:
    """Per-source weight scale from its own residual RMS about the shared fit."""
    acc: dict[int, list[float]] = {}
    for source_id, basis, y, w in rows:
        if source_id == _EGO_SOURCE_ID:
            continue
        bucket = acc.setdefault(source_id, [0.0, 0.0])
        bucket[0] += w * (y - _predict(beta, basis)) ** 2
        bucket[1] += w
    scales: dict[int, float] = {}
    for source_id, (sq, total_w) in acc.items():
        if total_w <= 0.0:
            scales[source_id] = 0.0
            continue
        rms = math.sqrt(sq / total_w)
        if rms <= _SOURCE_RESIDUAL_DELTA_M:
            scales[source_id] = 1.0
        else:
            scales[source_id] = (_SOURCE_RESIDUAL_DELTA_M / rms) ** _SOURCE_REJECT_POWER
    return scales


def agreement_residual_m(source_rms: dict) -> float:
    """Upper quantile of the per-source residuals: how badly the sources agree.

    Indexing ``n`` rather than ``n - 1`` makes this the plain maximum at four
    sources or fewer. That is deliberate and measured: correcting it costs more
    lock latency than it buys coverage. See the rejected table in README §9."""
    finite = sorted(r for r in source_rms.values() if math.isfinite(r))
    if not finite:
        return 0.0
    return finite[min(len(finite) - 1, int(_CONF_AGREE_QUANTILE * len(finite)))]


def _confidence(
    total_weight: float,
    agreement_rms: float,
    n_sources: int,
) -> float:
    """Weight the fit earned, scaled by how well its sources agree with it."""
    span_w = _CONF_WEIGHT_FULL - _CONF_WEIGHT_MIN
    w_term = max(0.0, min(1.0, (total_weight - _CONF_WEIGHT_MIN) / span_w))
    span_r = _CONF_RESIDUAL_BAD_M - _CONF_RESIDUAL_GOOD_M
    r_term = max(0.0, min(1.0, (_CONF_RESIDUAL_BAD_M - agreement_rms) / span_r))
    conf = w_term * r_term
    if n_sources <= 1:
        conf = min(conf, _CONF_SINGLE_SOURCE_CAP)
    return conf


def _rows_for_base(
    ego_samples: list[tuple[float, float]],
    target_samples: list[tuple[int, float, float, float]],
    kappa: float,
):
    """Project every sample onto the base arc at ``kappa`` and build the rows.

    Into arc length and normal offset, so the cubic never carries large
    curvature and never folds. Past the span limit a sample aliases (README §9)."""
    span = arc_span_limit(kappa)
    base_ego = [
        coords for coords in
        (arc_coords(kappa, x, y) for x, y in ego_samples)
        if abs(coords[0]) <= span
    ]
    base_targets = []
    for sid, x, y, w in target_samples:
        s_m, normal = arc_coords(kappa, x, y)
        if abs(s_m) <= span:
            base_targets.append((sid, s_m, normal, w))
    rows, n_sources = _grouped_rows(base_ego, base_targets)
    support = max((s for _, s, _, w in base_targets if w > 0.0), default=0.0)
    return rows, n_sources, support


def _trail_kappa(points: list[tuple[float, float]]) -> float | None:
    """Signed curvature of the least-squares circle through a source's own
    samples; + is left, and a straight trail reads as zero.

    Its own trail, so its lane offset cancels instead of reading as a bend the
    way a bearing from ego would. Least squares over the whole trail rather
    than three points off it: the three-point circle reads the road off two
    gaps and jitters frame to frame, which the base arc then passes on to
    every lateral at once."""
    ordered = sorted(points, key=lambda p: p[0] * p[0] + p[1] * p[1])
    (x1, y1), (xn, yn) = ordered[0], ordered[-1]
    if math.hypot(xn - x1, yn - y1) < _REBASE_MIN_SPAN_M:
        return None
    # Mean-centred algebraic fit: |p|² + D·x + E·y + F = 0.
    mx = sum(p[0] for p in ordered) / len(ordered)
    my = sum(p[1] for p in ordered) / len(ordered)
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for px, py in ordered:
        x, y = px - mx, py - my
        row = (x, y, 1.0)
        rhs = -(x * x + y * y)
        for i in range(3):
            atb[i] += row[i] * rhs
            for j in range(3):
                ata[i][j] += row[i] * row[j]
    sol = _solve(ata, atb)
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


def _rebased_kappa(
    ego_kappa: float,
    target_samples: list[tuple[int, float, float, float]],
) -> float:
    """Base curvature to index arc length by: ego's, bent toward the traffic's.

    Ego's curvature describes the road at ego, and on corner entry the road
    ahead is a whole bend away from it. This is raw ego-frame geometry, so
    unlike a curvature read off the fit it does not need the base arc to
    already be right, which is exactly the case that fails."""
    grouped: dict[int, list[tuple[float, float]]] = {}
    for sid, x, y, w in target_samples:
        if w <= 0.0:
            continue
        grouped.setdefault(sid, []).append((x, y))
    votes = []
    for points in grouped.values():
        # Same floor as the fit: a source too thin to carry shape cannot re-base.
        if len(points) < _MIN_SOURCE_SAMPLES:
            continue
        kappa = _trail_kappa(points)
        if kappa is not None:
            votes.append(kappa)
    if not votes:
        return ego_kappa
    # Median over sources, so one vehicle turning off cannot re-base the road.
    votes.sort()
    bent = votes[len(votes) // 2]
    if abs(bent - ego_kappa) < _STRAIGHT_KAPPA:
        return ego_kappa
    return max(-_REBASE_MAX_KAPPA, min(_REBASE_MAX_KAPPA, bent))


def fit_road_model(
    ego_samples: list[tuple[float, float]],
    target_samples: list[tuple[int, float, float, float]],
    fallback_kappa: float = 0.0,
) -> RoadModel:
    """Fit the shared centreline; falls back to constant curvature when weak.

    ``ego_samples`` are ego's own recent path in the current ego frame (x behind
    ego is negative). ``target_samples`` are ``(source_id, x, y, weight)``."""
    base_kappa = _rebased_kappa(fallback_kappa, target_samples)
    # Ego's path lies on the base arc only while the base is ego's own
    # curvature, and it never described the road ahead anyway (README §9).
    ego_rows = ego_samples if base_kappa == fallback_kappa else []
    rows, n_sources, support = _rows_for_base(
        ego_rows, target_samples, base_kappa,
    )
    if len(rows) < 4 and base_kappa != fallback_kappa:
        base_kappa = fallback_kappa
        rows, n_sources, support = _rows_for_base(
            ego_samples, target_samples, base_kappa,
        )
    if len(rows) < 4:
        return from_curvature(fallback_kappa)

    fit = _weighted_fit(rows)
    if fit is None:
        return from_curvature(fallback_kappa)
    beta, rms = fit

    # Huber handles sample noise, source-level scaling drops a manoeuvring vehicle.
    # Each pass rescales the ORIGINAL weights; compounding craters every source.
    base_rows = rows
    for _ in range(_IRLS_PASSES):
        scales = _source_scales(rows, beta)
        reweighted = []
        for source_id, basis, y, w0 in base_rows:
            residual = abs(y - _predict(beta, basis))
            scale = 1.0 if residual <= _HUBER_DELTA_M else _HUBER_DELTA_M / residual
            scale *= scales.get(source_id, 1.0)
            reweighted.append((source_id, basis, y, w0 * scale))
        fit = _weighted_fit(reweighted)
        if fit is None:
            return from_curvature(fallback_kappa)
        beta, rms = fit
        rows = reweighted

    # Ego anchors the fit but samples no road ahead, so it buys no confidence.
    target_weight = sum(w for sid, _, _, w in rows if sid != _EGO_SOURCE_ID)
    source_rms = _source_residuals(rows, beta)
    agreement = agreement_residual_m(source_rms)
    confidence = _confidence(target_weight, agreement, n_sources)
    if confidence <= 0.0:
        # Keep the per-source diagnostics even when the fit is not trusted, or a
        # caller's trust loop can never bootstrap out of a cold start.
        return RoadModel(
            base_kappa=base_kappa, confidence=0.0,
            residual_rms_m=rms, source_rms=source_rms,
            agreement_rms_m=agreement, target_weight=target_weight,
            n_sources=n_sources, support_s_m=support,
        )
    return RoadModel(
        c1=beta[0], c2=beta[1], c3=beta[2], c4=beta[3],
        base_kappa=base_kappa,
        confidence=confidence, residual_rms_m=rms,
        n_samples=len(rows), n_sources=n_sources,
        support_s_m=support, source_rms=source_rms,
        agreement_rms_m=agreement, target_weight=target_weight,
    )


# Re-exports for tests and tuning tools.
S_REF_M = _S_REF_M
MIN_SOURCE_SAMPLES = _MIN_SOURCE_SAMPLES
NODE_S = _NODE_S
EGO_SAMPLE_WEIGHT = _EGO_SAMPLE_WEIGHT
SOURCE_RESIDUAL_DELTA_M = _SOURCE_RESIDUAL_DELTA_M
HUBER_DELTA_M = _HUBER_DELTA_M
