"""A smooth model of the lane around ego, fitted on the lane's own vertices.

The map rebuilds the route every tick from ego's foot point, interpolated on a 10 m chord. Read off that polyline,
ego's heading against the lane jumped at every rebuild in bends and the chords' sag pulled ego inside. The vertices
past the foot are the lane's own samples and do not move, so they are kept as ego passes them and fitted with a
weighted quadratic in ego's frame. Design: core/la/README.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalLane:
    """Ego against the fitted lane: ``offset`` metres right of it, ``psi`` heading right of it (radians).

    ``kappa`` is the lane's rightward curvature at ego, ``kappa_ahead`` at the preview point.
    """

    offset: float
    psi: float
    kappa: float
    kappa_ahead: float


class LaneModel:
    def __init__(self, half_m: float = 25.0, keep_m: float = 40.0, min_points: int = 3,
                 match_m: float = 1e-3) -> None:
        self.half_m = half_m
        self.keep_m = keep_m
        self.min_points = min_points
        self.match_m = match_m
        self.reset()

    def reset(self) -> None:
        self._route: object | None = None
        self._pts: list[tuple[float, float]] = []

    def observe(self, route) -> None:
        """Takes a new route's vertices; those ego has passed stay while the new route continues the same lane."""
        if route is self._route:
            return
        self._route = route
        new = [(float(p[0]), float(p[1])) for p in tuple(getattr(route, "pts", ()))[1:]]
        kept: list[tuple[float, float]] = []
        if new and self._pts:
            fx, fz = new[0]
            for i, (px, pz) in enumerate(self._pts):
                if abs(px - fx) <= self.match_m and abs(pz - fz) <= self.match_m:
                    kept = self._pts[:i]
                    break
        self._pts = kept + new

    def local(self, x: float, z: float, fx: float, fz: float, ahead_m: float) -> LocalLane | None:
        """The fit at ``(x, z)`` heading ``(fx, fz)``, and its curvature ``ahead_m`` further on; None when too sparse."""
        pts = self._pts
        while len(pts) > self.min_points and (pts[0][0] - x) * fx + (pts[0][1] - z) * fz < -self.keep_m:
            pts.pop(0)
        reach = 2.0 * self.half_m + max(0.0, ahead_m)
        local = [((px - x) * fx + (pz - z) * fz, (px - x) * -fz + (pz - z) * fx) for px, pz in pts
                 if (px - x) ** 2 + (pz - z) ** 2 < reach * reach]
        here = self._fit(local, 0.0, self.half_m) or self._fit(local, 0.0, 2.0 * self.half_m)
        if here is None:
            return None
        ahead = self._fit(local, ahead_m, self.half_m) if ahead_m > 0.0 else here
        c0, slope, curve = here
        norm = math.sqrt(1.0 + slope * slope)
        return LocalLane(
            offset=-c0 / norm,
            psi=-math.atan(slope),
            kappa=curve / norm ** 3,
            kappa_ahead=(ahead[2] / math.sqrt(1.0 + ahead[1] ** 2) ** 3) if ahead is not None else curve / norm ** 3,
        )

    def _fit(self, local, centre: float, w: float) -> tuple[float, float, float] | None:
        """``(lateral, slope, second derivative)`` of a quadratic weighted within ``w`` of ``centre`` on ego's axis.

        Ego's own fit widens to twice the window while it has no vertices behind it yet (a new lane).
        """
        s = [0.0] * 5
        b = [0.0] * 3
        n = 0
        lo = hi = None
        for xi, eta in local:
            u = (xi - centre) / w
            if abs(u) >= 1.0:
                continue
            k = (1.0 - u * u) ** 2
            n += 1
            lo = u if lo is None else min(lo, u)
            hi = u if hi is None else max(hi, u)
            p = k
            for j in range(5):
                s[j] += p
                if j < 3:
                    b[j] += p * eta
                p *= u
        if n < self.min_points or hi - lo < 0.5:
            return None
        coef = _solve3(((s[0], s[1], s[2]), (s[1], s[2], s[3]), (s[2], s[3], s[4])), b)
        if coef is None:
            return None
        return coef[0], coef[1] / w, 2.0 * coef[2] / (w * w)


def _solve3(m, b) -> tuple[float, float, float] | None:
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if abs(det) < 1e-12:
        return None
    out = []
    for col in range(3):
        mm = [list(row) for row in m]
        for row in range(3):
            mm[row][col] = b[row]
        out.append((mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1])
                    - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
                    + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0])) / det)
    return out[0], out[1], out[2]
