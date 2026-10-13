"""A predicted path whose curvature changes with distance: an ArcPath bent to the road.

``kappa_s`` / ``kappa_k`` are breakpoints of curvature against arc length from
the start (positive turns left, relative to travel), linear between them and
held past the last one. The first breakpoint is the vehicle's own curvature, so
the path leaves it exactly as the plain arc does. Everything AEB reads off an
arc (positions and headings over time, capsule bodies, corridors) is answered
from a polyline integrated once in ``build``. See core/radar/README.md §8 and
core/aeb/README.md, "Road-bent trajectories".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.radar.traffic import ArcPath, build_arc

_STEP_M = 2.0
_MAX_STEPS = 200
_COARSE = 6


@dataclass(slots=True)
class BentPath(ArcPath):
    """ArcPath with a curvature profile. ``curvature`` stays the start curvature."""

    kappa_s: tuple[float, ...] = ()
    kappa_k: tuple[float, ...] = ()
    bent: bool = True
    _ds: float = _STEP_M
    _inv_ds: float = 1.0 / _STEP_M
    _xs: list = field(default_factory=list)
    _zs: list = field(default_factory=list)
    _hs: list = field(default_factory=list)

    def build(self) -> "BentPath":
        ArcPath.build(self)
        self.is_straight = False
        self._integrate()
        return self

    def kappa_at(self, s: float) -> float:
        ks, kk = self.kappa_s, self.kappa_k
        if not ks:
            return self.curvature
        if s <= ks[0]:
            return kk[0]
        for i in range(1, len(ks)):
            if s <= ks[i]:
                f = (s - ks[i - 1]) / max(ks[i] - ks[i - 1], 1e-9)
                return kk[i - 1] + f * (kk[i] - kk[i - 1])
        return kk[-1]

    def _integrate(self) -> None:
        length = max(self.arc_length, 0.0)
        steps = max(1, min(_MAX_STEPS, int(math.ceil(length / _STEP_M))))
        ds = length / steps if length > 0.0 else 0.0
        self._ds = ds
        self._inv_ds = 1.0 / ds if ds > 0.0 else 0.0
        # Travel heading in the yaw convention: fwd = (-sin h, -cos h).
        h_travel = math.atan2(-self.fwd_x, -self.fwd_z)
        body_minus_travel = self.yaw_rad - h_travel
        x, z, h = self.start_x, self.start_z, h_travel
        xs, zs, hs = [x], [z], [self.yaw_rad]
        ks_, kk_ = self.kappa_s, self.kappa_k
        j = 1
        for i in range(steps):
            sm = (i + 0.5) * ds
            # Breakpoints are sorted: walk a cursor instead of searching every step.
            while j < len(ks_) and ks_[j] < sm:
                j += 1
            if not ks_:
                k_mid = self.curvature
            elif j >= len(ks_):
                k_mid = kk_[-1]
            elif sm <= ks_[0]:
                k_mid = kk_[0]
            else:
                f = (sm - ks_[j - 1]) / max(ks_[j] - ks_[j - 1], 1e-9)
                k_mid = kk_[j - 1] + f * (kk_[j] - kk_[j - 1])
            h_mid = h + 0.5 * k_mid * ds
            x += -math.sin(h_mid) * ds
            z += -math.cos(h_mid) * ds
            h += k_mid * ds
            xs.append(x)
            zs.append(z)
            # Same body-heading convention as ArcPath.heading_at_dist (turns opposite to
            # the motion), so a bend changes only where the path goes (README §8).
            hs.append(h_travel - (h - h_travel) + body_minus_travel)
        self._xs, self._zs, self._hs = xs, zs, hs

    def position_at_dist(self, dist: float) -> tuple[float, float]:
        xs, zs = self._xs, self._zs
        last = len(xs) - 1
        u = dist * self._inv_ds
        if last < 1 or u <= 0.0:
            return (xs[0], zs[0]) if xs else (self.start_x, self.start_z)
        i = int(u)
        if i >= last:
            return xs[last], zs[last]
        f = u - i
        return xs[i] + f * (xs[i + 1] - xs[i]), zs[i] + f * (zs[i + 1] - zs[i])

    def heading_at_dist(self, dist: float) -> float:
        hs = self._hs
        last = len(hs) - 1
        u = dist * self._inv_ds
        if last < 1 or u <= 0.0:
            return hs[0] if hs else self.yaw_rad
        i = int(u)
        if i >= last:
            return hs[last]
        return hs[i] + (u - i) * (hs[i + 1] - hs[i])

    def project(self, x: float, z: float) -> tuple[float, float]:
        """``(s, signed offset right of travel)`` of a point; s runs past both ends."""
        xs, zs = self._xs, self._zs
        n = len(xs)
        if n < 2:
            dx, dz = x - self.start_x, z - self.start_z
            return dx * self.fwd_x + dz * self.fwd_z, dx * (-self.fwd_z) + dz * self.fwd_x
        # Coarse pass over every few vertices, then the segments around the nearest.
        k_best, d_best = 0, math.inf
        for k in range(0, n, _COARSE):
            d = (xs[k] - x) ** 2 + (zs[k] - z) ** 2
            if d < d_best:
                k_best, d_best = k, d
        lo = max(0, k_best - _COARSE - 1)
        hi = min(n - 1, k_best + _COARSE + 1)
        best = None
        for i in range(lo, hi):
            ax, az = xs[i], zs[i]
            sx, sz = xs[i + 1] - ax, zs[i + 1] - az
            length2 = sx * sx + sz * sz
            if length2 < 1e-12:
                continue
            raw = ((x - ax) * sx + (z - az) * sz) / length2
            t = max(0.0, min(1.0, raw))
            d2 = (x - ax - t * sx) ** 2 + (z - az - t * sz) ** 2
            if best is None or d2 < best[0]:
                best = (d2, i, raw, t, sx, sz)
        if best is None:
            return 0.0, 0.0
        _d2, i, raw, t, sx, sz = best
        length = math.sqrt(sx * sx + sz * sz)
        ax, az = xs[i], zs[i]
        side = ((x - ax) * (-sz) + (z - az) * sx) / length
        if (i == 0 and raw < 0.0) or (i == n - 2 and raw > 1.0):
            # Off an end: run along the end tangent so s keeps its sign.
            return (i + raw) * self._ds, side
        return (i + t) * self._ds, side

    def sample_corridor(self, n: int = 16):
        left, right = [], []
        if n < 2 or self.arc_length < 1e-6:
            return [(self.start_x, self.start_z)], [(self.start_x, self.start_z)]
        for k in range(n):
            d = self.arc_length * k / (n - 1)
            x, z = self.position_at_dist(d)
            h = self.heading_at_dist(d)
            rx, rz = -math.cos(h), math.sin(h)
            left.append((x - rx * self.half_width, z - rz * self.half_width))
            right.append((x + rx * self.half_width, z + rz * self.half_width))
        return left, right


def bend_arc(arc: ArcPath, kappa_s, kappa_k, keep_flat: bool = False) -> ArcPath:
    """``arc`` with the given curvature profile; the plain arc when the profile is flat.

    ``keep_flat`` returns a ``BentPath`` even then, so lane tests run along its centreline.
    """
    if not kappa_s or arc.speed < 1e-3:
        return arc
    k0 = kappa_k[0]
    if not keep_flat and all(abs(k - k0) < 1e-6 for k in kappa_k):
        return arc
    return BentPath(
        start_x=arc.start_x, start_z=arc.start_z, yaw_rad=arc.yaw_rad,
        speed=_signed_speed(arc), curvature=arc.curvature, half_width=arc.half_width,
        horizon=arc.horizon, decel=arc.decel, accel=arc.accel,
        fwd_len=arc.fwd_len, back_len=arc.back_len,
        parallel_margin_scale=arc.parallel_margin_scale,
        kappa_s=tuple(kappa_s), kappa_k=tuple(kappa_k),
    ).build()


def offset_curvature(arc: ArcPath, dk: float, **overrides) -> ArcPath:
    """The same kind of path with ``dk`` added to its curvature everywhere."""
    if isinstance(arc, BentPath):
        fields = dict(
            start_x=arc.start_x, start_z=arc.start_z, yaw_rad=arc.yaw_rad,
            speed=_signed_speed(arc), curvature=arc.curvature + dk,
            half_width=arc.half_width, horizon=arc.horizon, decel=arc.decel,
            accel=arc.accel, fwd_len=arc.fwd_len, back_len=arc.back_len,
            parallel_margin_scale=arc.parallel_margin_scale,
            kappa_s=arc.kappa_s, kappa_k=tuple(k + dk for k in arc.kappa_k),
        )
        fields.update(overrides)
        return BentPath(**fields).build()
    fields = dict(
        x=arc.start_x, z=arc.start_z, yaw_rad=arc.yaw_rad, speed=_signed_speed(arc),
        curvature=arc.curvature + dk, half_width=arc.half_width, horizon=arc.horizon,
        decel=arc.decel, accel=arc.accel, fwd_len=arc.fwd_len, back_len=arc.back_len,
        parallel_margin_scale=arc.parallel_margin_scale,
    )
    fields.update({("x" if k == "start_x" else "z" if k == "start_z" else k): v
                   for k, v in overrides.items()})
    return build_arc(**fields)


def _signed_speed(arc: ArcPath) -> float:
    """``build`` stores |speed|; a path travelling against its yaw needs the sign back."""
    reversing = arc.fwd_x * -math.sin(arc.yaw_rad) + arc.fwd_z * -math.cos(arc.yaw_rad) < 0.0
    return -arc.speed if reversing else arc.speed
