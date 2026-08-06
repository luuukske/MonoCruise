"""Temporal carry for the fitted centreline: rate limit, and confidence ramp.

The fit itself is stateless and lives in ``road_model``. Everything here is the
frame-to-frame half of it. Why the carry happens in sample space rather than
coefficient space: ``core/acc/README.md`` §9."""

from __future__ import annotations

import math
from dataclasses import replace

from .road_model import (
    NODE_S as _NODE_S,
)
from .road_model import (
    RoadModel,
    _confidence,
    arc_coords,
    arc_normal,
    arc_point,
    arc_span_limit,
)

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




def node_slew_budget_ms(s_m: float) -> float:
    """Rate a node at arc length ``s_m`` may move (m/s), from the curvature budget."""
    return max(_SMOOTH_MIN_RATE_MS, _SMOOTH_MAX_KAPPA_RATE * 0.5 * s_m * s_m)


def _resample(points: list[tuple[float, float]]) -> list[float | None]:
    """Sample an ascending (s, n) polyline onto ``_NODE_S``; None where unseen."""
    out: list[float | None] = []
    n = len(points)
    for node_s in _NODE_S:
        if n < 2 or node_s < points[0][0] or node_s > points[-1][0]:
            out.append(None)
            continue
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if points[mid][0] <= node_s:
                lo = mid
            else:
                hi = mid
        s0, n0 = points[lo]
        s1, n1 = points[hi]
        span = s1 - s0
        out.append(n0 if span < 1e-9 else n0 + (node_s - s0) * (n1 - n0) / span)
    return out



class RoadSmoother:
    """Carries the centreline across frames in sample space. See README §9.

    The previous frame's nodes are transformed into the current ego frame and
    resampled before the EMA, so ego's own motion is removed rather than being
    smoothed as if it were a change in the road."""

    def __init__(self) -> None:
        self._nodes: tuple | None = None
        self._pose: tuple[float, float, float, float] | None = None
        self._kappa: float = 0.0
        self._confidence: float = 0.0
        self._support_s_m: float = 0.0
        self._residual: float | None = None

    def reset(self) -> None:
        self._nodes = None
        self._pose = None
        self._kappa = 0.0
        self._confidence = 0.0
        self._support_s_m = 0.0
        self._residual = None

    def step(
        self,
        model: RoadModel,
        ego_x: float, ego_z: float,
        ego_fwd_x: float, ego_fwd_z: float,
        dt: float,
    ) -> RoadModel:
        """Return ``model`` carrying the smoothed centreline and confidence."""
        # A confident fit on top of nothing carried is not a step to suppress:
        # limiting it publishes the base arc at the fit's confidence (README §9).
        reacquiring = self._confidence <= 0.0 < model.confidence
        conf = self._step_confidence(self._filtered_confidence(model, dt), dt)
        support = (
            model.support_s_m if model.confidence >= self._confidence
            else max(model.support_s_m, self._support_s_m if conf > 0.0 else 0.0)
        )
        held = replace(model, confidence=conf, support_s_m=support)

        prior = None if reacquiring else self._prior_on_grid(
            ego_x, ego_z, ego_fwd_x, ego_fwd_z, model.base_kappa,
        )
        target = self._target_grid(model, held, prior, conf)
        if prior is None:
            blended = target
        else:
            span = max(dt, 1e-6)
            blended = []
            for s_m, fresh, carried in zip(_NODE_S, target, prior):
                if carried is None:
                    blended.append(fresh)
                    continue
                step = node_slew_budget_ms(s_m) * span
                blended.append(max(carried - step, min(carried + step, fresh)))
        # Re-anchor so the centreline still passes through ego exactly.
        anchor = blended[0]
        blended = [value - anchor for value in blended]
        self._nodes = tuple(blended)
        self._pose = (ego_x, ego_z, ego_fwd_x, ego_fwd_z)
        self._kappa = model.base_kappa
        self._confidence = conf
        self._support_s_m = support
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
        """Where each node's deviation wants to be this frame.

        Nodes the fit still reaches take it. Nodes it does not keep the shape
        already carried, fading to zero (the bare base arc) as the held
        confidence decays, so losing the last source is a fade, not a snap."""
        out: list[float] = []
        for i, s_m in enumerate(_NODE_S):
            if model.confidence_at(s_m) > 0.0:
                out.append(model.raw_deviation_at(s_m))
                continue
            carried = None if prior is None else prior[i]
            if carried is None or conf <= 0.0 or held.confidence_at(s_m) <= 0.0:
                out.append(0.0)
            else:
                out.append(carried * conf)
        return out

    def _prior_on_grid(
        self, ego_x: float, ego_z: float, ego_fwd_x: float, ego_fwd_z: float,
        kappa: float,
    ) -> list[float | None] | None:
        """Last frame's deviations, re-expressed against this frame's base arc.

        Nodes hold a deviation from a base arc, so carrying them over means
        rebuilding the world points under the old arc and reading them back
        under the new one. Ego's own motion drops out; a curvature change does
        not, and should not."""
        if self._nodes is None or self._pose is None:
            return None
        px, pz, pfx, pfz = self._pose
        if math.hypot(ego_x - px, ego_z - pz) > _SMOOTH_RESET_JUMP_M:
            return None
        prx, prz = -pfz, pfx
        rx, rz = -ego_fwd_z, ego_fwd_x
        was_valid = arc_span_limit(self._kappa)
        now_valid = arc_span_limit(kappa)
        points: list[tuple[float, float]] = []
        for node_s, node_n in zip(_NODE_S, self._nodes):
            if node_s > was_valid:
                break
            bx, by = arc_point(self._kappa, node_s)
            nx, ny = arc_normal(self._kappa, node_s)
            lx, ly = bx + node_n * nx, by + node_n * ny
            wx = px + lx * pfx + ly * prx
            wz = pz + lx * pfz + ly * prz
            dx, dz = wx - ego_x, wz - ego_z
            carried = arc_coords(
                kappa, dx * ego_fwd_x + dz * ego_fwd_z, dx * rx + dz * rz,
            )
            if abs(carried[0]) <= now_valid:
                points.append(carried)
        points.sort(key=lambda pt: pt[0])
        return _resample(points)


# Re-exports for tests and tuning tools.
SMOOTH_MAX_KAPPA_RATE = _SMOOTH_MAX_KAPPA_RATE
SMOOTH_MIN_RATE_MS = _SMOOTH_MIN_RATE_MS
CONF_RATE_UP_PER_S = _CONF_RATE_UP_PER_S
CONF_RATE_DOWN_PER_S = _CONF_RATE_DOWN_PER_S
