"""Speed-adaptive low-pass for one signal, used on each target's blended curvature."""

from __future__ import annotations

import math


class OneEuroFilter:
    """Speed-adaptive 1€ low-pass; see core/aeb/README.md (One-Euro on v_curvature)."""

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "beta_scale",
                 "_x_prev", "_dx_prev", "_t_prev")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0,
                 beta_scale: float = 0.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.beta_scale = beta_scale
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def step(self, x: float, t: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0.0:
            return self._x_prev
        dx_raw = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx_raw + (1.0 - a_d) * self._dx_prev
        beta_eff = self.beta / (1.0 + self.beta_scale * abs(self._x_prev))
        cutoff = self.min_cutoff + beta_eff * abs(dx_hat)
        a_x = self._alpha(cutoff, dt)
        x_hat = a_x * x + (1.0 - a_x) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat
