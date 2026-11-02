"""What the wheel and the truck do per unit of steering, learned from steady bends while driving.

Pure logic. Two fits share one set of steady-cornering samples: the game's own force per m/s^2 of cornering
(its sign is the wheel's torque direction) and the curvature per unit of steering (the truck's steering gain).
See core/la/README.md.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from core.lane_departure.detector import _slope


@dataclass(frozen=True)
class CalTuning:
    min_speed_ms: float = 40.0 / 3.6
    rate_window_s: float = 0.4
    steady_window_s: float = 0.8
    steady_span_ms2: float = 0.3
    max_accel_ms2: float = 3.0
    force_limit: int = 9000
    fresh_s: float = 0.25
    tau_s: float = 120.0
    min_weight_s: float = 20.0
    min_spread_ms2: float = 0.25
    min_corr: float = 0.7
    slope_min: float = 100.0
    slope_max: float = 30000.0
    min_steer_spread: float = 0.004
    gain_min: float = 0.05
    gain_max: float = 0.8
    gain_corr: float = 0.7


@dataclass(frozen=True)
class Estimate:
    """One fit: ``value`` is None until usable. ``state`` is learning, implausible, ready, held or conflict."""

    value: float | None = None
    corr: float = 0.0
    spread: float = 0.0
    state: str = "learning"


@dataclass(frozen=True)
class Calibration:
    """``force_per_ms2`` is game force units per m/s^2 of rightward cornering; steering is rightward, of full lock."""

    force_per_ms2: float | None = None
    corr: float = 0.0
    spread_ms2: float = 0.0
    weight_s: float = 0.0
    state: str = "learning"
    kappa_per_steer: Estimate = field(default_factory=Estimate)

    @property
    def servo_ready(self) -> bool:
        return self.force_per_ms2 is not None and self.kappa_per_steer.value is not None


class _Fit:
    """Exponentially weighted least squares of y on x, forgetting with ``tau_s`` of accepted time."""

    def __init__(self, tau_s: float) -> None:
        self.tau_s = tau_s
        self.w = self.x = self.y = self.xx = self.xy = self.yy = 0.0

    def add(self, x: float, y: float, dt: float) -> None:
        keep = math.exp(-dt / self.tau_s)
        self.w = self.w * keep + dt
        self.x = self.x * keep + dt * x
        self.y = self.y * keep + dt * y
        self.xx = self.xx * keep + dt * x * x
        self.xy = self.xy * keep + dt * x * y
        self.yy = self.yy * keep + dt * y * y

    def stats(self) -> tuple[float, float, float]:
        """``(slope, correlation, spread of x)``; zeros while there is nothing to fit."""
        if self.w <= 0.0:
            return 0.0, 0.0, 0.0
        mx, my = self.x / self.w, self.y / self.w
        vx = max(0.0, self.xx / self.w - mx * mx)
        vy = max(0.0, self.yy / self.w - my * my)
        cov = self.xy / self.w - mx * my
        if vx < 1e-12 or vy < 1e-12:
            return 0.0, 0.0, math.sqrt(vx)
        return cov / vx, cov / math.sqrt(vx * vy), math.sqrt(vx)


class _Gauge:
    """A fit with its acceptance rule, its first sign locked and its last good value held.

    ``positive`` requires a positive slope; otherwise either sign is fine but a later flip is a conflict.
    """

    def __init__(self, tau_s: float, spread: float, corr: float, lo: float, hi: float, positive: bool = False) -> None:
        self.fit = _Fit(tau_s)
        self.limits = (spread, corr, lo, hi)
        self.positive = positive
        self.sign = 0
        self.conflict = False
        self.good: float | None = None

    def estimate(self, min_weight: float) -> Estimate:
        spread_min, corr_min, lo, hi = self.limits
        slope, corr, spread = self.fit.stats()
        base = dict(corr=corr, spread=spread)
        if self.conflict:
            return Estimate(state="conflict", **base)
        weak = self.fit.w < min_weight or spread < spread_min or abs(corr) < corr_min
        bad = not lo <= abs(slope) <= hi or (self.positive and slope <= 0.0)
        if weak or bad:
            if self.good is not None:
                return Estimate(value=self.good, state="held", **base)
            return Estimate(state="learning" if weak else "implausible", **base)
        sign = 1 if slope > 0.0 else -1
        if self.sign == 0:
            self.sign = sign
        elif sign != self.sign:
            self.conflict = True
            self.good = None
            return Estimate(state="conflict", **base)
        self.good = slope
        return Estimate(value=slope, state="ready", **base)


def heading_angle(yaw_rad: float) -> float:
    """Ego's travel direction as ``atan2(dz, dx)``; it grows while the truck turns right."""
    return math.atan2(-math.cos(yaw_rad), -math.sin(yaw_rad))


class ForceCalibration:
    """Regresses the game's force on lateral acceleration and ego's curvature on steering.

    Samples come from steady cornering only. Steering is the plugin's own wheel angle when it has one. Force on
    steering is not fitted: pooled over speeds it reads low, since tight bends are taken slower.
    """

    def __init__(self, tuning: CalTuning | None = None) -> None:
        tn = self.tune = tuning or CalTuning()
        self._accel = _Gauge(tn.tau_s, tn.min_spread_ms2, tn.min_corr, tn.slope_min, tn.slope_max)
        self._gain = _Gauge(tn.tau_s, tn.min_steer_spread, tn.gain_corr, tn.gain_min, tn.gain_max, positive=True)
        self._heading: deque[tuple[float, float]] = deque()
        self._steady: deque[tuple[float, float, float, float]] = deque()
        self._last_t: float | None = None
        self._last_frames: int | None = None
        self._frames_at = -math.inf

    def reset_motion(self) -> None:
        """Drop the motion history (a jump, a pause); what was learned stays."""
        self._heading.clear()
        self._steady.clear()
        self._last_t = None

    def update(
        self, t: float, speed_ms: float, yaw_rad: float, steer_right: float, game_force: int, frames: int,
    ) -> Calibration:
        tn = self.tune
        if frames != self._last_frames:
            self._last_frames = frames
            self._frames_at = t
        dt = 0.0 if self._last_t is None else t - self._last_t
        if dt < 0.0 or dt > 1.0:
            self.reset_motion()
            dt = 0.0
        self._last_t = t
        angle = heading_angle(yaw_rad)
        if self._heading:
            prev = self._heading[-1][1]
            angle = prev + math.atan2(math.sin(angle - prev), math.cos(angle - prev))
        self._heading.append((t, angle))
        while t - self._heading[0][0] > tn.rate_window_s:
            self._heading.popleft()
        if len(self._heading) >= 4 and t - self._heading[0][0] >= 0.75 * tn.rate_window_s:
            accel = speed_ms * _slope(self._heading)
            self._steady.append((t, accel, float(game_force), steer_right))
        while self._steady and t - self._steady[0][0] > tn.steady_window_s:
            self._steady.popleft()
        if dt > 0.0 and self._accept(t, speed_ms, game_force):
            n = len(self._steady)
            accel = sum(s[1] for s in self._steady) / n
            force = sum(s[2] for s in self._steady) / n
            steer = sum(s[3] for s in self._steady) / n
            self._accel.fit.add(accel, force, dt)
            self._gain.fit.add(steer, accel / (speed_ms * speed_ms), dt)

        return self.result()

    def _accept(self, t: float, speed_ms: float, game_force: int) -> bool:
        tn = self.tune
        if speed_ms < tn.min_speed_ms or t - self._frames_at > tn.fresh_s or abs(game_force) >= tn.force_limit:
            return False
        if len(self._steady) < 8 or t - self._steady[0][0] < 0.75 * tn.steady_window_s:
            return False
        accels = [s[1] for s in self._steady]
        return max(accels) - min(accels) <= tn.steady_span_ms2 and max(abs(a) for a in accels) <= tn.max_accel_ms2

    def result(self) -> Calibration:
        w = self.tune.min_weight_s
        accel = self._accel.estimate(w)
        return Calibration(
            force_per_ms2=accel.value, corr=accel.corr, spread_ms2=accel.spread, weight_s=self._accel.fit.w,
            state=accel.state, kappa_per_steer=self._gain.estimate(w),
        )
