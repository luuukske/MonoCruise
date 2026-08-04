"""Shared road centreline fitted from ego's own path and the traffic ahead.

One weighted cubic per frame, stateless. Parameterisation, the per-source offset
elimination, and why fusion happens in sample space: ``core/acc/README.md`` §9."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


# Basis scale (m): u = x / _X_REF_M keeps the cubic design matrix conditioned
# over the 150 m scoring range instead of spanning 10^6 in the x³ column.
_X_REF_M: float = 100.0

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

# Ego anchors the absolute position. Raising this was measured to change nothing
# once the base arc carries ego's curvature exactly, so it stays at parity.
_EGO_SAMPLE_WEIGHT: float = 1.0

# Confidence falls to zero this far past the furthest sample.
_EXTRAPOLATION_FADE_M: float = 30.0

# Ridge weight pulling the centreline's heading at ego (c1) toward ego's own
# heading. Comparable to total sample weight; 0 leaves it free.
_HEADING_PRIOR_WEIGHT: float = 400.0

# Fixed lookahead grid the smoothed centreline is carried on. Sample space, not
# coefficient space: filtering the cubic terms frame to frame sloshes.
_NODE_STEP_M: float = 10.0
_NODE_COUNT: int = 16
_NODE_X: tuple[float, ...] = tuple(i * _NODE_STEP_M for i in range(_NODE_COUNT))

# Centreline slew limit: a rate limit, not a low-pass (README §9). Budgeted in
# curvature, because lateral offset grows as x^2/2 and a flat m/s cap is all far.
_SMOOTH_MAX_KAPPA_RATE: float = 0.010     # (1/m) per second
_SMOOTH_MIN_RATE_MS: float = 20.0         # floor so the near nodes are not frozen
_SMOOTH_RESET_JUMP_M: float = 20.0

# The agreement residual comes from an unrobust fit, so one outlier sample moves
# it. Smooth the measurement here; the rate limit below governs the decision.
_CONF_RESIDUAL_TAU_S: float = 0.30

# Confidence rate limit (per second). Unsmoothed it went 1 -> 0 inside a frame,
# retargeting the whole centreline and reading as a bounce between arcs.
_CONF_RATE_UP_PER_S: float = 3.0
_CONF_RATE_DOWN_PER_S: float = 1.5

# Base-arc guards: below this curvature the road is straight, and the arc is
# only evaluated within this fraction of its radius (beyond it x is degenerate).
_STRAIGHT_KAPPA: float = 1.0 / 5000.0
_BASE_ARC_MAX_FRAC: float = 0.95

# Measured lateral error of the blended estimate against ego's own future path
# (clip corpus): 0.65 m at 30-60 m, 2.0 m at 60-90 m, 3.7 m at 90-130 m.
_SIGMA_MIN_M: float = 0.25
_SIGMA_KNEE_M: float = 35.0
_SIGMA_SLOPE: float = 0.047


def base_arc_lateral(kappa: float, x_m: float) -> float:
    """Exact circular-arc offset through ego at curvature ``kappa``.

    A parabola undershoots a circle, so a polynomial-only centreline sits toward
    the outside of a tight bend. The base arc carries the large curvature exactly
    and the cubic only has to describe the deviation. See README §9."""
    if abs(kappa) < _STRAIGHT_KAPPA:
        return 0.0
    radius = 1.0 / abs(kappa)
    sign = 1.0 if kappa > 0.0 else -1.0
    # Past the circle's forward extent this parameterisation is degenerate.
    reach = min(abs(x_m), radius * _BASE_ARC_MAX_FRAC)
    return sign * (math.sqrt(max(radius * radius - reach * reach, 0.0)) - radius)


@dataclass(slots=True, frozen=True)
class RoadModel:
    """Centreline through ego: exact base arc plus a cubic deviation.

    ``y(x) = base_arc(base_kappa, x) + c1·u + c2·u² + c3·u³`` with
    ``u = x / 100 m``; +y is ego's right, +x is ego's forward. Anchored at ego,
    so ``y(0) = 0`` by construction."""

    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    base_kappa: float = 0.0
    confidence: float = 0.0
    residual_rms_m: float = 0.0
    n_samples: int = 0
    n_sources: int = 0
    # Furthest forward distance any sample reached; beyond it the cubic is
    # extrapolating and its confidence has to decay.
    support_x_m: float = 0.0
    # Per-source residual RMS about the shared fit; drives the caller's trust EMA.
    source_rms: dict = field(default_factory=dict)
    # Inputs the smoother needs to recompute confidence on a filtered residual.
    agreement_rms_m: float = 0.0
    target_weight: float = 0.0
    # Temporally smoothed centreline on ``_NODE_X``; empty means evaluate the fit.
    nodes: tuple = ()

    def confidence_at(self, x_m: float) -> float:
        """Confidence for a query at forward distance ``x_m``.

        A cubic fitted to samples ending at 60 m says nothing trustworthy at
        140 m, however well it fits the samples it does have."""
        if self.confidence <= 0.0:
            return 0.0
        if x_m <= self.support_x_m:
            return self.confidence
        over = x_m - self.support_x_m
        decay = max(0.0, 1.0 - over / _EXTRAPOLATION_FADE_M)
        return self.confidence * decay

    def lateral_at(self, x_m: float) -> float:
        """Centreline lateral offset at forward distance ``x_m``."""
        if self.nodes:
            return _interp_nodes(self.nodes, x_m)
        return self.raw_lateral_at(x_m)

    def raw_lateral_at(self, x_m: float) -> float:
        """This frame's fit alone, before any temporal smoothing."""
        u = x_m / _X_REF_M
        deviation = self.c1 * u + self.c2 * u * u + self.c3 * u * u * u
        return base_arc_lateral(self.base_kappa, x_m) + deviation

    def offset_of(self, x_m: float, y_m: float) -> float:
        """Road-relative lateral of a point: + is right of the centreline."""
        return y_m - self.lateral_at(x_m)

    def curvature_at(self, x_m: float) -> float:
        """Signed curvature (1/m); + is a left turn, matching ArcPath."""
        u = x_m / _X_REF_M
        d2 = (2.0 * self.c2 + 6.0 * self.c3 * u) / (_X_REF_M * _X_REF_M)
        return self.base_kappa - d2


def node_slew_budget_ms(x_m: float) -> float:
    """Lateral rate a node at ``x_m`` may move (m/s), from the curvature budget."""
    return max(_SMOOTH_MIN_RATE_MS, _SMOOTH_MAX_KAPPA_RATE * 0.5 * x_m * x_m)


def _interp_nodes(nodes: tuple, x_m: float) -> float:
    """Linear interpolation over ``_NODE_X``, extrapolating from the end slopes."""
    last = len(nodes) - 1
    if x_m <= _NODE_X[0]:
        slope = (nodes[1] - nodes[0]) / _NODE_STEP_M
        return nodes[0] + slope * (x_m - _NODE_X[0])
    if x_m >= _NODE_X[last]:
        slope = (nodes[last] - nodes[last - 1]) / _NODE_STEP_M
        return nodes[last] + slope * (x_m - _NODE_X[last])
    idx = int(x_m / _NODE_STEP_M)
    idx = max(0, min(last - 1, idx))
    frac = (x_m - _NODE_X[idx]) / _NODE_STEP_M
    return nodes[idx] + frac * (nodes[idx + 1] - nodes[idx])


def _resample(points: list[tuple[float, float]]) -> list[float | None]:
    """Sample an ascending (x, y) polyline onto ``_NODE_X``; None where unseen."""
    out: list[float | None] = []
    n = len(points)
    for node_x in _NODE_X:
        if n < 2 or node_x < points[0][0] or node_x > points[-1][0]:
            out.append(None)
            continue
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if points[mid][0] <= node_x:
                lo = mid
            else:
                hi = mid
        x0, y0 = points[lo]
        x1, y1 = points[hi]
        span = x1 - x0
        out.append(y0 if span < 1e-9 else y0 + (node_x - x0) * (y1 - y0) / span)
    return out


class RoadSmoother:
    """Carries the centreline across frames in sample space. See README §9.

    The previous frame's nodes are transformed into the current ego frame and
    resampled before the EMA, so ego's own motion is removed rather than being
    smoothed as if it were a change in the road."""

    def __init__(self) -> None:
        self._nodes: tuple | None = None
        self._pose: tuple[float, float, float, float] | None = None
        self._confidence: float = 0.0
        self._support_x_m: float = 0.0
        self._residual: float | None = None

    def reset(self) -> None:
        self._nodes = None
        self._pose = None
        self._confidence = 0.0
        self._support_x_m = 0.0
        self._residual = None

    def step(
        self,
        model: RoadModel,
        ego_x: float, ego_z: float,
        ego_fwd_x: float, ego_fwd_z: float,
        dt: float,
    ) -> RoadModel:
        """Return ``model`` carrying the smoothed centreline and confidence."""
        conf = self._step_confidence(self._filtered_confidence(model, dt), dt)
        support = (
            model.support_x_m if model.confidence >= self._confidence
            else max(model.support_x_m, self._support_x_m if conf > 0.0 else 0.0)
        )
        held = replace(model, confidence=conf, support_x_m=support)

        prior = self._prior_on_grid(ego_x, ego_z, ego_fwd_x, ego_fwd_z)
        target = self._target_grid(model, held, prior, conf)
        if prior is None:
            blended = target
        else:
            span = max(dt, 1e-6)
            blended = []
            for x, f, p in zip(_NODE_X, target, prior):
                if p is None:
                    blended.append(f)
                    continue
                step = node_slew_budget_ms(x) * span
                blended.append(max(p - step, min(p + step, f)))
        # Re-anchor so the centreline still passes through ego exactly.
        anchor = blended[0]
        blended = [value - anchor for value in blended]
        self._nodes = tuple(blended)
        self._pose = (ego_x, ego_z, ego_fwd_x, ego_fwd_z)
        self._confidence = conf
        self._support_x_m = support
        return replace(held, nodes=self._nodes)

    def _filtered_confidence(self, model: RoadModel, dt: float) -> float:
        """Confidence from the low-passed agreement residual, not the raw one."""
        if model.n_sources <= 0:
            self._residual = None
            return model.confidence
        fresh = model.agreement_rms_m
        if self._residual is None:
            self._residual = fresh
        else:
            alpha = 1.0 - math.exp(-max(dt, 1e-6) / _CONF_RESIDUAL_TAU_S)
            self._residual += alpha * (fresh - self._residual)
        return _confidence(model.target_weight, self._residual, model.n_sources)

    def _step_confidence(self, fresh: float, dt: float) -> float:
        rate = _CONF_RATE_UP_PER_S if fresh >= self._confidence else _CONF_RATE_DOWN_PER_S
        step = rate * max(dt, 1e-6)
        return max(self._confidence - step, min(self._confidence + step, fresh))

    @staticmethod
    def _target_grid(
        model: RoadModel, held: RoadModel, prior, conf: float,
    ) -> list[float]:
        """Where each node wants to be this frame.

        Nodes the fit still reaches take it. Nodes it does not keep the shape
        already carried, fading to the bare ego arc as the held confidence
        decays, so losing the last source is a fade rather than a snap."""
        out: list[float] = []
        for i, x in enumerate(_NODE_X):
            if model.confidence_at(x) > 0.0:
                out.append(model.raw_lateral_at(x))
                continue
            arc = base_arc_lateral(model.base_kappa, x)
            carried = None if prior is None else prior[i]
            if carried is None or conf <= 0.0 or held.confidence_at(x) <= 0.0:
                out.append(arc)
            else:
                out.append(arc + (carried - arc) * conf)
        return out

    def _prior_on_grid(
        self, ego_x: float, ego_z: float, ego_fwd_x: float, ego_fwd_z: float,
    ) -> list[float | None] | None:
        if self._nodes is None or self._pose is None:
            return None
        px, pz, pfx, pfz = self._pose
        if math.hypot(ego_x - px, ego_z - pz) > _SMOOTH_RESET_JUMP_M:
            return None
        prx, prz = -pfz, pfx
        rx, rz = -ego_fwd_z, ego_fwd_x
        points: list[tuple[float, float]] = []
        for node_x, node_y in zip(_NODE_X, self._nodes):
            wx = px + node_x * pfx + node_y * prx
            wz = pz + node_x * pfz + node_y * prz
            dx, dz = wx - ego_x, wz - ego_z
            points.append((dx * ego_fwd_x + dz * ego_fwd_z, dx * rx + dz * rz))
        points.sort(key=lambda pt: pt[0])
        return _resample(points)


def lateral_sigma_m(x_m: float) -> float:
    """1-sigma lateral uncertainty of any lane estimate at forward distance x.

    Measured, not assumed: see the constants above and ``core/acc/README.md`` §9.
    Consumers inflate their in-lane test by this so a distant target cannot be
    called in-lane on an estimate that cannot resolve a lane width."""
    return _SIGMA_MIN_M + _SIGMA_SLOPE * max(0.0, x_m - _SIGMA_KNEE_M)


def from_curvature(kappa: float) -> RoadModel:
    """Constant-curvature fallback when there is no traffic evidence."""
    return RoadModel(base_kappa=kappa, confidence=0.0)


def _solve3(a: list[list[float]], b: list[float]) -> tuple[float, float, float] | None:
    """Gaussian elimination with partial pivoting on a 3x3 system."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, 3):
            factor = m[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, 4):
                m[r][c] -= factor * m[col][c]
    out = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        acc = m[row][3]
        for col in range(row + 1, 3):
            acc -= m[row][col] * out[col]
        out[row] = acc / m[row][row]
    return out[0], out[1], out[2]


def _basis(x_m: float) -> tuple[float, float, float]:
    u = x_m / _X_REF_M
    return u, u * u, u * u * u


_EGO_SOURCE_ID = -1


def _grouped_rows(
    ego_samples: list[tuple[float, float]],
    target_samples: list[tuple[int, float, float, float]],
) -> tuple[list[tuple[int, tuple[float, float, float], float, float]], int]:
    """Design rows with each source's own lateral offset eliminated.

    Ego is the reference so its rows pass through untouched; every other source
    is centred on its own mean, which removes the unknown lane offset while
    keeping the shape it contributes."""
    rows: list[tuple[int, tuple[float, float, float], float, float]] = []
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
        mean_b = [0.0, 0.0, 0.0]
        mean_y = 0.0
        for x, y, w in samples:
            basis = _basis(x)
            for i in range(3):
                mean_b[i] += w * basis[i]
            mean_y += w * y
        for i in range(3):
            mean_b[i] /= total_w
        mean_y /= total_w
        for x, y, w in samples:
            basis = _basis(x)
            centred = (basis[0] - mean_b[0], basis[1] - mean_b[1], basis[2] - mean_b[2])
            rows.append((source_id, centred, y - mean_y, w))
        n_sources += 1
    return rows, n_sources


def _weighted_fit(rows) -> tuple[tuple[float, float, float], float] | None:
    """Weighted normal equations plus the weighted residual RMS."""
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for _, basis, y, w in rows:
        for i in range(3):
            atb[i] += w * basis[i] * y
            for j in range(3):
                ata[i][j] += w * basis[i] * basis[j]
    # Heading prior: the road leaves ego along ego's heading, so c1 is pulled to
    # zero. Without it a far source's local shape tilts the whole centreline.
    ata[0][0] += _HEADING_PRIOR_WEIGHT
    beta = _solve3(ata, atb)
    if beta is None:
        return None
    total_w = 0.0
    sq = 0.0
    for _, basis, y, w in rows:
        pred = beta[0] * basis[0] + beta[1] * basis[1] + beta[2] * basis[2]
        sq += w * (y - pred) ** 2
        total_w += w
    rms = math.sqrt(sq / total_w) if total_w > 0.0 else 0.0
    return beta, rms


def _source_residuals(rows, beta) -> dict[int, float]:
    """Per-source residual RMS about the shared fit, ego excluded."""
    acc: dict[int, list[float]] = {}
    for source_id, basis, y, w in rows:
        if source_id == _EGO_SOURCE_ID:
            continue
        pred = beta[0] * basis[0] + beta[1] * basis[1] + beta[2] * basis[2]
        bucket = acc.setdefault(source_id, [0.0, 0.0])
        bucket[0] += w * (y - pred) ** 2
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
        pred = beta[0] * basis[0] + beta[1] * basis[1] + beta[2] * basis[2]
        bucket = acc.setdefault(source_id, [0.0, 0.0])
        bucket[0] += w * (y - pred) ** 2
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

    A quantile survives one dissenter among several, and does not drift upward
    as sources are added the way a max or a range does."""
    finite = sorted(r for r in source_rms.values() if math.isfinite(r))
    if not finite:
        return 0.0
    idx = min(len(finite) - 1, int(_CONF_AGREE_QUANTILE * len(finite)))
    return finite[idx]


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


def fit_road_model(
    ego_samples: list[tuple[float, float]],
    target_samples: list[tuple[int, float, float, float]],
    fallback_kappa: float = 0.0,
) -> RoadModel:
    """Fit the shared centreline; falls back to constant curvature when weak.

    ``ego_samples`` are ego's own recent path in the current ego frame (x behind
    ego is negative). ``target_samples`` are ``(source_id, x, y, weight)``."""
    # Deviation from the exact base arc, so the cubic never has to represent
    # large curvature itself (README §9 tight-corner bias).
    base_ego = [(x, y - base_arc_lateral(fallback_kappa, x)) for x, y in ego_samples]
    base_targets = [
        (sid, x, y - base_arc_lateral(fallback_kappa, x), w)
        for sid, x, y, w in target_samples
    ]
    rows, n_sources = _grouped_rows(base_ego, base_targets)
    if len(rows) < 4:
        return from_curvature(fallback_kappa)
    support = max(
        (x for _, x, _, w in base_targets if w > 0.0),
        default=0.0,
    )

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
            pred = beta[0] * basis[0] + beta[1] * basis[1] + beta[2] * basis[2]
            residual = abs(y - pred)
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
            base_kappa=fallback_kappa, confidence=0.0,
            residual_rms_m=rms, source_rms=source_rms,
            agreement_rms_m=agreement, target_weight=target_weight,
            n_sources=n_sources, support_x_m=support,
        )
    return RoadModel(
        c1=beta[0], c2=beta[1], c3=beta[2], base_kappa=fallback_kappa,
        confidence=confidence, residual_rms_m=rms,
        n_samples=len(rows), n_sources=n_sources,
        support_x_m=support, source_rms=source_rms,
        agreement_rms_m=agreement, target_weight=target_weight,
    )


# Re-exports for tests and tuning tools.
X_REF_M = _X_REF_M
MIN_SOURCE_SAMPLES = _MIN_SOURCE_SAMPLES
NODE_X = _NODE_X
SMOOTH_MAX_KAPPA_RATE = _SMOOTH_MAX_KAPPA_RATE
SMOOTH_MIN_RATE_MS = _SMOOTH_MIN_RATE_MS
CONF_RATE_UP_PER_S = _CONF_RATE_UP_PER_S
CONF_RATE_DOWN_PER_S = _CONF_RATE_DOWN_PER_S
EGO_SAMPLE_WEIGHT = _EGO_SAMPLE_WEIGHT
SOURCE_RESIDUAL_DELTA_M = _SOURCE_RESIDUAL_DELTA_M
HUBER_DELTA_M = _HUBER_DELTA_M
