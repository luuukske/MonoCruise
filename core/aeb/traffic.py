"""
ETS2/ATS traffic vehicle classes with arc-based path prediction.

Coordinate system and yaw conventions — see ``core/aeb/AGENTS.md`` §1–§3.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

_MAX_ANGULAR_VELOCITY: float = 45.0
_LOCATION_UPDATE_FREQUENCY: float = 0.05

# TMP speed / accel EMA — same hyperbolic law α(|v|) with different endpoints.
# Reference speed for “at 90 km/h” is 25 m/s. See AGENTS.md §7.
_ALPHA_SPEED_SCALE: float = 90.0 / 3.6   # 25.0 m/s

# Speed EMA on raw_speed: 0.5 at rest → 0.15 at 90 km/h.
_SPEED_EMA_AT_REST: float = 1.0
_SPEED_EMA_AT_90_KMH: float = 0.2
_SPEED_EMA_CURVE_D: float = (
    _ALPHA_SPEED_SCALE
    * _SPEED_EMA_AT_90_KMH
    / (_SPEED_EMA_AT_REST - _SPEED_EMA_AT_90_KMH)
)

# Accel EMA on kinematic (d filtered_speed / dt): 0.5 at rest → 0.2 at 90 km/h.
_ACCEL_EMA_AT_REST: float = 0.5
_ACCEL_EMA_AT_90_KMH: float = 0.2
_ACCEL_EMA_CURVE_D: float = (
    _ALPHA_SPEED_SCALE
    * _ACCEL_EMA_AT_90_KMH
    / (_ACCEL_EMA_AT_REST - _ACCEL_EMA_AT_90_KMH)
)

# Yaw EMA (wrap-safe) — AI and TMP (arc curvature).
_RAW_YAW_ALPHA: float = 0.5

# TMP lag detection — see AGENTS.md §7 "Lag / freeze detection".
_LAG_MIN_SPEED_MS: float = 5.0           # m/s  — below this no lag detection runs
_LAG_DISP_RATIO: float = 0.10           # flag lag if raw disp < 10 % of expected
_LAG_FREEZE_DURATION: float = 0.2       # s    — freeze window; release after this

# Position mismatch (TMP only) — out-of-order packet rejection.
# Fires when raw position jumps backward along heading.  Max 3 consecutive frames.
_POS_MISMATCH_BACKWARD_THRESHOLD: float = 0.00   # m — min backward dot to flag
_POS_MISMATCH_MAX_FRAMES: int = 5

# Crash detection (TMP only) — angular jerk vs last sample (every read, not only full frames).
_CRASH_PITCH_JERK: float = 3.0                 # deg/s² pitch angular jerk threshold
_CRASH_YAW_JERK: float = 10.0                   # deg/s² yaw angular jerk threshold
_CRASH_ROLL_JERK: float = 3.0                  # deg/s² roll angular jerk threshold
_CRASH_CONFIRM_DURATION: float = 0.00           # s jerk must hold before confirming

_MIN_CURVATURE_RADIUS: float = 5.0
_STRAIGHT_CURVATURE_EPS: float = 1e-6

# TMP raw speed — fit longitudinal motion over the last N full-frame samples (LS on s ≈ v·τ).
# Two samples reduce to the legacy single-interval Δs/Δt; more samples damp jitter.
_TMP_SPEED_HISTORY_LEN: int = 10
_TMP_SPEED_NEAR_ZERO_CHORD: float = 0.025  # m — same gate as per-frame displacement


def _tmp_raw_speed_from_position_history(
    history: list[tuple[float, float, float]],
    fwd_x: float,
    fwd_z: float,
) -> float | None:
    """Estimate signed longitudinal speed (m/s) from (t, x, z) samples, oldest first.

    Fits s ≈ v·τ where s = dot(p(τ) − p₀, fwd) and τ = t − t₀.  Uniform spacing is
    not required.  Returns None if fewer than two samples (caller uses one interval).
    If the first→last chord is below _TMP_SPEED_NEAR_ZERO_CHORD, returns 0.0.
    """
    if len(history) < 2:
        return None
    t0, x0, z0 = history[0]
    tn, xn, zn = history[-1]
    chord_dx = xn - x0
    chord_dz = zn - z0
    chord = math.sqrt(chord_dx * chord_dx + chord_dz * chord_dz)
    if chord < _TMP_SPEED_NEAR_ZERO_CHORD:
        return 0.0
    num = 0.0
    den = 0.0
    for t, x, z in history:
        tau = t - t0
        if tau <= 1e-9:
            continue
        s = (x - x0) * fwd_x + (z - z0) * fwd_z
        num += tau * s
        den += tau * tau
    if den < 1e-12:
        dt = tn - t0
        if dt < 1e-9:
            return 0.0
        direction = 1.0 if (chord_dx * fwd_x + chord_dz * fwd_z) >= 0.0 else -1.0
        return direction * chord / dt
    return num / den


def _accel_to_arc_params(accel: float, override_decel: float = 0.0) -> tuple[float, float]:
    """Convert raw vehicle acceleration to (decel, accel) for build_arc().

    - override_decel > 0  (e.g. head-on full brake) → (override_decel, 0.0).
    - accel < 0 (braking) → decel = min(|accel|, 6.0), accel = 0.0.
      Capped at 6 m/s² so crash-induced backward position jumps (which produce
      large negative acceleration spikes) are not mistaken for hard braking.
    - accel >= 0 (accelerating or constant) → decel = 0.0, accel = min(accel, 4.0).
    """
    if override_decel > 0.0:
        return override_decel, 0.0
    if accel < 0.0:
        return min(-accel, 6.0), 0.0
    return 0.0, min(accel, 4.0)


def _tmp_speed_ema_alpha(speed_ms: float) -> float:
    """Weight on the new raw speed sample. 0.5 at rest → 0.15 at 90 km/h."""
    return (_SPEED_EMA_AT_REST * _SPEED_EMA_CURVE_D) / (
        abs(speed_ms) + _SPEED_EMA_CURVE_D
    )


def _tmp_accel_ema_alpha(speed_ms: float) -> float:
    """Weight on new kinematic accel in the second TMP EMA. 0.5 at rest → 0.2 at 90 km/h."""
    return (_ACCEL_EMA_AT_REST * _ACCEL_EMA_CURVE_D) / (
        abs(speed_ms) + _ACCEL_EMA_CURVE_D
    )


class Position:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y, self.z - other.z)

    def tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def is_zero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0 and self.z == 0.0

    def distance_to(self, other: "Position") -> float:
        dx = self.x - other.x
        dz = self.z - other.z
        return math.sqrt(dx * dx + dz * dz)

    def __repr__(self) -> str:
        return f"Position({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Quaternion:
    """ETS2 traffic quaternion — x/y swap is intentional (AGENTS.md §3)."""
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self.w = w
        self.x = y
        self.y = x
        self.z = z

    def euler(self) -> tuple[float, float, float]:
        """(pitch, yaw, roll) in degrees."""
        yaw = math.atan2(
            2.0 * (self.y * self.z + self.w * self.x),
            self.w * self.w - self.x * self.x - self.y * self.y + self.z * self.z,
        )
        pitch = math.asin(
            max(-1.0, min(1.0, -2.0 * (self.x * self.z - self.w * self.y)))
        )
        roll = math.atan2(
            2.0 * (self.x * self.y + self.w * self.z),
            self.w * self.w + self.x * self.x - self.y * self.y - self.z * self.z,
        )
        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def is_zero(self) -> bool:
        return self.w == 0.0 and self.x == 0.0 and self.y == 0.0 and self.z == 0.0

    def __repr__(self) -> str:
        p, y, r = self.euler()
        return f"Quaternion(pitch={p:.1f}, yaw={y:.1f}, roll={r:.1f})"


class Size:
    __slots__ = ("width", "height", "length")

    def __init__(self, width: float, height: float, length: float) -> None:
        self.width = width
        self.height = height
        self.length = length

    def __repr__(self) -> str:
        return f"Size({self.width:.2f}, {self.height:.2f}, {self.length:.2f})"


class Trailer:
    __slots__ = ("position", "rotation", "size", "is_tmp")

    def __init__(self, position: Position, rotation: Quaternion,
                 size: Size, is_tmp: bool = False) -> None:
        self.position = position
        self.rotation = rotation
        self.size = size
        self.is_tmp = is_tmp

    def correct_position(self) -> Position:
        """Shift TMP trailer pivot from front coupler to body center."""
        _, yaw_deg, _ = self.rotation.euler()
        yaw_rad = math.radians(yaw_deg)
        return Position(
            self.position.x + (self.size.length / 2.0) * math.sin(yaw_rad),
            self.position.y,
            self.position.z + (self.size.length / 2.0) * math.cos(yaw_rad),
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()


@dataclass(slots=True)
class ArcPath:
    """Predicted path as a circular arc or straight ray.  See AGENTS.md §8."""
    start_x: float = 0.0
    start_z: float = 0.0
    yaw_rad: float = 0.0
    speed: float = 0.0
    curvature: float = 0.0
    half_width: float = 1.15
    horizon: float = 3.0
    decel: float = 0.0
    accel: float = 0.0

    is_straight: bool = True
    center_x: float = 0.0
    center_z: float = 0.0
    radius: float = 0.0
    angle0: float = 0.0
    max_sweep: float = 0.0
    arc_length: float = 0.0
    fwd_x: float = 0.0
    fwd_z: float = -1.0
    _sign: float = 1.0

    def build(self) -> "ArcPath":
        """Compute cached fields (fwd, radius, center, arc_length, is_straight) from
        start, curvature, speed, decel/accel. Call after setting fields."""
        self.fwd_x = -math.sin(self.yaw_rad)
        self.fwd_z = -math.cos(self.yaw_rad)

        # Reversing: flip fwd to actual travel direction, normalise speed to abs.
        if self.speed < -1e-3:
            self.fwd_x = -self.fwd_x
            self.fwd_z = -self.fwd_z
        self.speed = abs(self.speed)

        if self.speed < 1e-3:
            self.is_straight = True
            self.arc_length = 0.0
            self.max_sweep = 0.0
            return self

        if self.decel > 0.0:
            t_stop = self.speed / self.decel
            if t_stop < self.horizon:
                self.arc_length = self.speed * t_stop - 0.5 * self.decel * t_stop * t_stop
            else:
                t = self.horizon
                self.arc_length = self.speed * t - 0.5 * self.decel * t * t
        elif self.accel < 0.0:
            t_stop = -self.speed / self.accel
            if t_stop < self.horizon:
                self.arc_length = self.speed * t_stop + 0.5 * self.accel * t_stop * t_stop
            else:
                t = self.horizon
                self.arc_length = self.speed * t + 0.5 * self.accel * t * t
        elif self.accel > 0.0:
            t = self.horizon
            self.arc_length = self.speed * t + 0.5 * self.accel * t * t
        else:
            self.arc_length = self.speed * self.horizon

        if abs(self.curvature) < _STRAIGHT_CURVATURE_EPS:
            self.is_straight = True
            self.radius = 0.0
            self.max_sweep = 0.0
        else:
            self.is_straight = False
            self.radius = max(abs(1.0 / self.curvature), _MIN_CURVATURE_RADIUS)

            self._sign = 1.0 if self.curvature > 0 else -1.0
            self.center_x = self.start_x + self._sign * self.radius * self.fwd_z
            self.center_z = self.start_z + self._sign * self.radius * (-self.fwd_x)

            self.angle0 = math.atan2(
                self.start_z - self.center_z,
                self.start_x - self.center_x,
            )
            self.max_sweep = -self._sign * self.arc_length / self.radius

        return self

    def _dist_at_time(self, t: float) -> float:
        """Distance travelled along the path at time t (constant speed, decel, or accel)."""
        if self.decel > 0.0:
            t_stop = self.speed / self.decel
            if t >= t_stop:
                return self.speed * t_stop - 0.5 * self.decel * t_stop * t_stop
            return self.speed * t - 0.5 * self.decel * t * t
        elif self.accel < 0.0:
            t_stop = -self.speed / self.accel
            if t >= t_stop:
                return self.speed * t_stop + 0.5 * self.accel * t_stop * t_stop
            return self.speed * t + 0.5 * self.accel * t * t
        elif self.accel > 0.0:
            return self.speed * t + 0.5 * self.accel * t * t
        return self.speed * t

    def position_at_dist(self, dist: float) -> tuple[float, float]:
        """(x, z) at distance along the centerline (straight segment or arc)."""
        dist = max(0.0, min(dist, self.arc_length))
        if self.is_straight:
            return (
                self.start_x + dist * self.fwd_x,
                self.start_z + dist * self.fwd_z,
            )
        frac = dist / self.arc_length if self.arc_length > 0 else 0.0
        angle = self.angle0 + frac * self.max_sweep
        return (
            self.center_x + self.radius * math.cos(angle),
            self.center_z + self.radius * math.sin(angle),
        )

    def position_at_time(self, t: float) -> tuple[float, float]:
        """(x, z) at time t along the path (via _dist_at_time)."""
        return self.position_at_dist(self._dist_at_time(t))

    def heading_at_dist(self, dist: float) -> float:
        """Heading (yaw_rad) at distance along the path."""
        if self.is_straight:
            return self.yaw_rad
        dist = max(0.0, min(dist, self.arc_length))
        frac = dist / self.arc_length if self.arc_length > 0 else 0.0
        return self.yaw_rad + frac * self.max_sweep

    def sample_points(self, n: int = 16) -> list[tuple[float, float]]:
        """n evenly spaced (x, z) points along the centerline."""
        if n < 2 or self.arc_length < 1e-6:
            return [(self.start_x, self.start_z)]
        pts = []
        for i in range(n):
            d = self.arc_length * i / (n - 1)
            pts.append(self.position_at_dist(d))
        return pts

    def sample_corridor(self, n: int = 16) -> tuple[
        list[tuple[float, float]], list[tuple[float, float]]
    ]:
        """Left and right boundary point lists for the path corridor (half_width)."""
        if n < 2 or self.arc_length < 1e-6:
            return [(self.start_x, self.start_z)], [(self.start_x, self.start_z)]

        if self.is_straight:
            left = []
            right = []
            for i in range(n):
                d = self.arc_length * i / (n - 1)
                x, z = self.position_at_dist(d)
                h = self.heading_at_dist(d)
                rx = -math.cos(h)
                rz = math.sin(h)
                left.append((x - rx * self.half_width, z - rz * self.half_width))
                right.append((x + rx * self.half_width, z + rz * self.half_width))
            return left, right

        r_inner = max(self.radius - self.half_width, 0.5)
        r_outer = self.radius + self.half_width
        left = []
        right = []
        for i in range(n):
            frac = i / (n - 1) if n > 1 else 1.0
            angle = self.angle0 + frac * self.max_sweep
            cx, cz = self.center_x, self.center_z
            c, s = math.cos(angle), math.sin(angle)
            inner_pt = (cx + r_inner * c, cz + r_inner * s)
            outer_pt = (cx + r_outer * c, cz + r_outer * s)
            if self._sign > 0:  # left turn: left = inner, right = outer
                left.append(inner_pt)
                right.append(outer_pt)
            else:  # right turn: left = outer, right = inner
                left.append(outer_pt)
                right.append(inner_pt)
        return left, right


def build_arc(
    x: float, z: float, yaw_rad: float, speed: float,
    curvature: float, half_width: float, horizon: float,
    decel: float = 0.0,
    accel: float = 0.0,
) -> ArcPath:
    """Build and cache an ArcPath from start (x,z), yaw, speed, curvature, half_width,
    horizon; optional decel/accel. Call this instead of constructing ArcPath directly."""
    return ArcPath(
        start_x=x, start_z=z, yaw_rad=yaw_rad, speed=speed,
        curvature=curvature, half_width=half_width, horizon=horizon,
        decel=decel, accel=accel,
    ).build()


def arc_arc_collision(
    a: ArcPath,
    b: ArcPath,
    margin: float = 0.5,
    n_samples: int = 24,
    min_lateral_gap: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Earliest time the two arc corridors overlap; uses closed-form ray-ray when
    both straight and constant speed, else time-sampled + bisection. Returns
    (time_s, hit_x, hit_z) or None. min_lateral_gap: suppress hit when centerlines
    stay that far apart (e.g. head-on turns in separate lanes)."""
    if a.arc_length < 1e-3 and b.arc_length < 1e-3:
        return None

    corridor_sq = (a.half_width + b.half_width + margin) ** 2
    horizon = min(a.horizon, b.horizon)

    if (a.is_straight and b.is_straight
            and a.decel <= 0 and b.decel <= 0
            and a.accel == 0.0 and b.accel == 0.0):
        return _ray_ray_collision(a, b, corridor_sq, horizon, min_lateral_gap)

    return _sampled_collision(a, b, corridor_sq, horizon, n_samples, min_lateral_gap)


def _ray_ray_collision(
    a: ArcPath, b: ArcPath, corridor_sq: float, horizon: float,
    min_lateral_gap: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Earliest time two straight rays’ corridors touch: solve quadratic for
    |a_pos(t) − b_pos(t)|² = corridor_sq; returns (t, hit_x, hit_z) or None.
    min_lateral_gap suppresses hits when centerlines stay that far apart laterally."""
    dpx = a.start_x - b.start_x
    dpz = a.start_z - b.start_z
    dvx = a.speed * a.fwd_x - b.speed * b.fwd_x
    dvz = a.speed * a.fwd_z - b.speed * b.fwd_z

    A = dvx * dvx + dvz * dvz
    B = 2.0 * (dpx * dvx + dpz * dvz)
    C = dpx * dpx + dpz * dpz - corridor_sq

    if C <= 0:
        if min_lateral_gap > 0.0:
            lat = abs(dpz * a.fwd_x - dpx * a.fwd_z)
            if lat >= min_lateral_gap:
                return None
        return 0.0, (a.start_x + b.start_x) * 0.5, (a.start_z + b.start_z) * 0.5

    if abs(A) < 1e-12:
        return None

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)

    t_hit = None
    if 0.0 <= t1 <= horizon:
        t_hit = t1
    elif 0.0 <= t2 <= horizon:
        t_hit = t2
    elif t1 < 0 <= t2 and t2 <= horizon:
        t_hit = 0.0

    if t_hit is None:
        return None

    ax = a.start_x + t_hit * a.speed * a.fwd_x
    az = a.start_z + t_hit * a.speed * a.fwd_z
    bx = b.start_x + t_hit * b.speed * b.fwd_x
    bz = b.start_z + t_hit * b.speed * b.fwd_z

    if min_lateral_gap > 0.0:
        lat = abs((bz - az) * a.fwd_x - (bx - ax) * a.fwd_z)
        if lat >= min_lateral_gap:
            return None

    return t_hit, (ax + bx) * 0.5, (az + bz) * 0.5


def _sampled_collision(
    a: ArcPath, b: ArcPath, corridor_sq: float, horizon: float, n: int,
    min_lateral_gap: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Earliest corridor overlap for curved or non-constant-speed arcs: sample at n
    times, then bisect to refine hit time; respects min_lateral_gap. Returns
    (t, hit_x, hit_z) or None."""
    best_t: Optional[float] = None
    best_mx = 0.0
    best_mz = 0.0

    inv_n = 1.0 / n
    for i in range(n + 1):
        t = horizon * i * inv_n
        ax, az = a.position_at_time(t)
        bx, bz = b.position_at_time(t)
        dsq = (ax - bx) ** 2 + (az - bz) ** 2
        if dsq < corridor_sq:
            if min_lateral_gap > 0.0:
                h_a = a.heading_at_dist(a._dist_at_time(t))
                fwd_x_a = -math.sin(h_a)
                fwd_z_a = -math.cos(h_a)
                lat = abs((bz - az) * fwd_x_a - (bx - ax) * fwd_z_a)
                if lat >= min_lateral_gap:
                    continue
            lo = max(t - horizon * inv_n, 0.0)
            hi = t
            best_t = t
            best_mx = (ax + bx) * 0.5
            best_mz = (az + bz) * 0.5
            for _ in range(6):
                mid = (lo + hi) * 0.5
                ax2, az2 = a.position_at_time(mid)
                bx2, bz2 = b.position_at_time(mid)
                if (ax2 - bx2) ** 2 + (az2 - bz2) ** 2 < corridor_sq:
                    if min_lateral_gap > 0.0:
                        h_a2 = a.heading_at_dist(a._dist_at_time(mid))
                        fwd_x_a2 = -math.sin(h_a2)
                        fwd_z_a2 = -math.cos(h_a2)
                        lat2 = abs((bz2 - az2) * fwd_x_a2 - (bx2 - ax2) * fwd_z_a2)
                        if lat2 >= min_lateral_gap:
                            lo = mid
                            continue
                    hi = mid
                    best_t = mid
                    best_mx = (ax2 + bx2) * 0.5
                    best_mz = (az2 + bz2) * 0.5
                else:
                    lo = mid
            break

    if best_t is None:
        return None
    return best_t, best_mx, best_mz


class Vehicle:
    """Traffic vehicle with arc-based path prediction."""

    def __init__(
        self,
        position: Position,
        rotation: Quaternion,
        size: Size,
        speed: float,
        acceleration: float,
        trailer_count: int,
        trailers: list[Trailer],
        id: int,
        is_tmp: bool,
        is_trailer: bool,
    ) -> None:
        self.position = position
        self.rotation = rotation
        self.size = size
        self.speed = speed
        # TMP: shared-memory acceleration is not used for physics; smoothed value is filled
        # in update_from_last(). Zero until the first kinematic update avoids buffer spikes.
        self.acceleration = 0.0 if is_tmp else acceleration
        self.trailer_count = trailer_count
        self.trailers = trailers
        self.id = id
        self.is_tmp = is_tmp
        self.is_trailer = is_trailer

        self.time: float = time.time()
        self.last_location = Position(0.0, 0.0, 0.0)
        self.last_rotation = Quaternion(0.0, 0.0, 0.0, 0.0)
        self.angular_velocity: float = 0.0

        self._smooth_x: Optional[float] = None
        self._smooth_z: Optional[float] = None
        self._smooth_yaw: Optional[float] = None
        self._raw_x: Optional[float] = None
        self._raw_z: Optional[float] = None

        # TMP only — EMA state for speed/acceleration (buffer accel unused on TMP).
        self._smooth_speed: Optional[float] = None
        self._smooth_accel: Optional[float] = None
        self._raw_speed: Optional[float] = None
        # TMP only — (time, x, z) from full updates for multi-sample raw speed (newest last).
        self._position_history: list[tuple[float, float, float]] = []

        # TMP lag detection state.
        # _lag_since: monotonic time when the frozen-position window began.
        # lag_confirmed: True once the vehicle has been stationary for
        #   >= _LAG_FREEZE_DURATION s. AEB handles confirmed-stopped vehicles
        #   naturally via arc collision; no special-case needed in thread.py.
        self._lag_since: Optional[float] = None
        self.lag_confirmed: bool = False

        # Position mismatch (TMP only) — consecutive frame counter.
        # Counts how many frames in a row the raw position jumped backward.
        # Resets to 0 on any clean frame or when the cap is reached.
        self._pos_mismatch_frames: int = 0

        # Crash detection (TMP only) — per-axis rotation rates and displacement from prev frame.
        self._prev_pitch_rate: Optional[float] = None
        self._prev_yaw_rate: Optional[float] = None
        self._prev_roll_rate: Optional[float] = None
        self._crash_since: Optional[float] = None
        self.crash_confirmed: bool = False

    def accel_for_arc(self) -> float:
        """Longitudinal acceleration for arc / collision (TMP = filtered kinematic value)."""
        return self.acceleration

    def radar_speed_accel(self) -> tuple[float, float, float]:
        """(filtered_speed, filtered_accel, raw_speed) for the radar visualizer.

        The visualizer is intended for debugging the TMP kinematic filtering: it shows
        raw vs filtered speed and the filtered (smoothed) acceleration only.
        """
        if self.is_tmp:
            filtered_speed = self._smooth_speed if self._smooth_speed is not None else self.speed
            filtered_accel = self._smooth_accel if self._smooth_accel is not None else self.acceleration
            raw_speed = self._raw_speed if self._raw_speed is not None else self.speed
            return filtered_speed, filtered_accel, raw_speed

        return self.speed, self.acceleration, self.speed

    def _tmp_apply_crash_rotation_jerk(self, prev: "Vehicle", t_now: float) -> None:
        """TMP: detect crash-level rotation jerk on every buffer read (sub-frame and full)."""
        dt = t_now - prev.time
        if not self.is_tmp or prev._raw_x is None or dt < 1e-9:
            return

        def _adiff(a: float, b: float) -> float:
            return (a - b + 180.0) % 360.0 - 180.0

        pitch_c, yaw_c, roll_c = self.rotation.euler()
        pitch_p, yaw_p, roll_p = prev.rotation.euler()
        pitch_rate = _adiff(pitch_c, pitch_p) / dt
        yaw_rate = _adiff(yaw_c, yaw_p) / dt
        roll_rate = _adiff(roll_c, roll_p) / dt

        _rot_jerk = False
        if prev._prev_pitch_rate is not None:
            if (
                abs(pitch_rate - prev._prev_pitch_rate) > _CRASH_PITCH_JERK
                or abs(yaw_rate - prev._prev_yaw_rate) > _CRASH_YAW_JERK
                or abs(roll_rate - prev._prev_roll_rate) > _CRASH_ROLL_JERK
            ):
                _rot_jerk = True

        self._prev_pitch_rate = pitch_rate
        self._prev_yaw_rate = yaw_rate
        self._prev_roll_rate = roll_rate

        if _rot_jerk:
            if self._crash_since is None:
                self._crash_since = t_now
            if t_now - self._crash_since >= _CRASH_CONFIRM_DURATION:
                self.crash_confirmed = True
        else:
            self._crash_since = None

    def update_from_last(self, prev: "Vehicle", t_now: float) -> None:
        """Carry forward smoothed state or run a full update.  See AGENTS.md §7."""
        dt = t_now - prev.time

        # Sub-frame pass: carry forward all smoothed state unchanged.
        if dt < _LOCATION_UPDATE_FREQUENCY:
            self.time = prev.time
            self.last_location = prev.last_location
            self.last_rotation = prev.last_rotation
            self.angular_velocity = prev.angular_velocity
            self._smooth_x = prev._smooth_x
            self._smooth_z = prev._smooth_z
            self._smooth_yaw = prev._smooth_yaw
            self._raw_x = prev._raw_x
            self._raw_z = prev._raw_z
            self._lag_since = prev._lag_since
            self.lag_confirmed = prev.lag_confirmed
            self._pos_mismatch_frames = prev._pos_mismatch_frames
            self._prev_pitch_rate = prev._prev_pitch_rate
            self._prev_yaw_rate = prev._prev_yaw_rate
            self._prev_roll_rate = prev._prev_roll_rate
            self._crash_since = prev._crash_since
            self.crash_confirmed = prev.crash_confirmed
            self._smooth_speed = prev._smooth_speed
            self._smooth_accel = prev._smooth_accel
            self._raw_speed = prev._raw_speed
            self._position_history = list(prev._position_history)
            if abs(self.angular_velocity) > _MAX_ANGULAR_VELOCITY:
                self.angular_velocity = 0.0
            self.speed = prev.speed
            if self.is_tmp:
                self.acceleration = prev.acceleration

            self._tmp_apply_crash_rotation_jerk(prev, t_now)

            # TMP: between full updates (dt < threshold), do not freeze speed and pose at
            # the last full tick — the buffer may have new coordinates while prev.speed
            # still holds a bad EMA sample. Re-derive speed from (latest raw − prev raw) /
            # (t_now − prev.time) when movement exceeds the usual gate; always snap pose
            # to latest raw. Skipped during lag freeze and position mismatch hold unless
            # crash_confirmed, which bypasses both filters to keep position accurate.
            _tmp_sf_ok = (
                self.is_tmp
                and prev._raw_x is not None
                and prev._smooth_yaw is not None
                and (
                    self.crash_confirmed
                    or (
                        not (
                            prev._lag_since is not None
                            and (t_now - prev._lag_since) < _LAG_FREEZE_DURATION
                        )
                        and prev._pos_mismatch_frames == 0
                    )
                )
            )
            if _tmp_sf_ok:
                rx = self.position.x
                rz = self.position.z
                ry = self.position.y
                dt_sf = t_now - prev.time
                ddx = rx - prev._raw_x
                ddz = rz - prev._raw_z
                dist = math.sqrt(
                    ddx * ddx + (ry - prev.position.y) ** 2 + ddz * ddz
                )
                if dt_sf > 1e-9 and dist > 0.025:
                    fwd_x = -math.sin(prev._smooth_yaw)
                    fwd_z = -math.cos(prev._smooth_yaw)
                    direction = 1.0 if (ddx * fwd_x + ddz * fwd_z) >= 0.0 else -1.0
                    # Raw kinematics for debug only — keep self.speed at filtered value.
                    self._raw_speed = direction * dist / dt_sf
                self._raw_x = rx
                self._raw_z = rz
                self._smooth_x = rx
                self._smooth_z = rz
                self.position.x = rx
                self.position.z = rz
            elif self._smooth_x is not None:
                self.position.x = self._smooth_x
                self.position.z = self._smooth_z
            return

        self.time = t_now
        self.last_location = prev.position
        self.last_rotation = prev.rotation
        self._smooth_x = prev._smooth_x
        self._smooth_z = prev._smooth_z
        self._smooth_yaw = prev._smooth_yaw
        self._lag_since = prev._lag_since
        self.lag_confirmed = False
        self._pos_mismatch_frames = prev._pos_mismatch_frames
        self._prev_pitch_rate = prev._prev_pitch_rate
        self._prev_yaw_rate = prev._prev_yaw_rate
        self._prev_roll_rate = prev._prev_roll_rate
        self._crash_since = prev._crash_since
        self.crash_confirmed = False
        self._smooth_speed = prev._smooth_speed
        self._smooth_accel = prev._smooth_accel
        self._raw_speed = prev._raw_speed
        self._position_history = list(prev._position_history)

        raw_x = self.position.x
        raw_z = self.position.z
        self._raw_x = raw_x
        self._raw_z = raw_z

        # --- Type 3: Crash detection (TMP only) — angular jerk; sub-frames call the same helper earlier.
        self._tmp_apply_crash_rotation_jerk(prev, t_now)

        # --- Type 1: Position mismatch (TMP only, max _POS_MISMATCH_MAX_FRAMES) ---
        # Raw position jumped backward along the vehicle's heading — out-of-order packet.
        # Yaw EMA and angular_velocity still run; position and carried speed/accel are held.
        # Bypassed when crash_confirmed — a crashed vehicle's backward jumps are real.
        _skip_position_update = False
        if (self.is_tmp
                and prev._smooth_yaw is not None
                and prev._raw_x is not None):
            _pm_dx = raw_x - prev._raw_x
            _pm_dz = raw_z - prev._raw_z
            _pm_fwd_x = -math.sin(prev._smooth_yaw)
            _pm_fwd_z = -math.cos(prev._smooth_yaw)
            if (_pm_dx * _pm_fwd_x + _pm_dz * _pm_fwd_z < -_POS_MISMATCH_BACKWARD_THRESHOLD
                    and self._pos_mismatch_frames < _POS_MISMATCH_MAX_FRAMES
                    and not self.crash_confirmed):
                self._pos_mismatch_frames = prev._pos_mismatch_frames + 1
                _skip_position_update = True
            else:
                self._pos_mismatch_frames = 0

        # --- Type 2: TMP lag detection (near-stationary freeze with speed decay) ---
        # Bypassed when crash_confirmed — any movement on a crashed vehicle is real position data.
        if self.is_tmp and prev._raw_x is not None and not _skip_position_update and not self.crash_confirmed:
            _raw_disp_sq = (raw_x - prev._raw_x) ** 2 + (raw_z - prev._raw_z) ** 2
            _expected_disp = abs(prev.speed) * dt
            _lag_threshold_sq = (_expected_disp * _LAG_DISP_RATIO) ** 2
            if abs(prev.speed) > _LAG_MIN_SPEED_MS and _raw_disp_sq < _lag_threshold_sq:
                if self._lag_since is None:
                    self._lag_since = t_now
                _lag_duration = t_now - self._lag_since
                if _lag_duration < _LAG_FREEZE_DURATION:
                    _lag_frac = _lag_duration / _LAG_FREEZE_DURATION
                    self._smooth_x = prev._smooth_x
                    self._smooth_z = prev._smooth_z
                    self._smooth_yaw = prev._smooth_yaw
                    self.angular_velocity = prev.angular_velocity
                    self.speed = prev.speed * (1.0 - _lag_frac * _lag_frac)
                    self.acceleration = 0.0
                    self._smooth_accel = 0.0
                    self._smooth_speed = self.speed
                    self._raw_speed = 0.0
                    if self._smooth_x is not None:
                        self.position.x = self._smooth_x
                        self.position.z = self._smooth_z
                    return
                self.lag_confirmed = True
            else:
                self._lag_since = None

        # Wrap-safe yaw EMA — runs first so angular_velocity uses smooth derivative
        raw_yaw = math.radians(self.rotation.euler()[1])
        if self._smooth_yaw is None:
            self._smooth_yaw = raw_yaw
        else:
            diff = (raw_yaw - self._smooth_yaw + math.pi) % (2.0 * math.pi) - math.pi
            self._smooth_yaw = self._smooth_yaw + _RAW_YAW_ALPHA * diff

        # Angular velocity in deg/s — callers apply math.radians(), so keep degrees here
        _prev_smooth_yaw_deg = math.degrees(prev._smooth_yaw) if prev._smooth_yaw is not None else prev.rotation.euler()[1]
        _cur_smooth_yaw_deg = math.degrees(self._smooth_yaw)
        _yaw_diff_deg = (_cur_smooth_yaw_deg - _prev_smooth_yaw_deg + 180.0) % 360.0 - 180.0
        raw_av = _yaw_diff_deg / dt
        self.angular_velocity = 0.0 if abs(raw_av) > _MAX_ANGULAR_VELOCITY else raw_av

        # Position mismatch: hold smooth position and carry speed; yaw already updated above.
        if _skip_position_update:
            if self._smooth_x is not None:
                self.position.x = self._smooth_x
                self.position.z = self._smooth_z
            self.speed = prev.speed
            if self.is_tmp:
                self.acceleration = prev.acceleration
                self._smooth_accel = prev._smooth_accel
            return

        # World position is unfiltered — arcs and debug use true coordinates.
        self._smooth_x = raw_x
        self._smooth_z = raw_z
        self.position.x = raw_x
        self.position.z = raw_z

        fwd_x = -math.sin(self._smooth_yaw)
        fwd_z = -math.cos(self._smooth_yaw)

        # TMP: raw speed from last N positions (LS on longitudinal s vs τ); smooth with EMA.
        if self.is_tmp:
            _hist = list(self._position_history)
            _hist.append((t_now, raw_x, raw_z))
            if len(_hist) > _TMP_SPEED_HISTORY_LEN:
                _hist = _hist[-_TMP_SPEED_HISTORY_LEN:]
            self._position_history = _hist

            _ls = _tmp_raw_speed_from_position_history(_hist, fwd_x, fwd_z)
            if _ls is not None:
                raw_speed = _ls
            else:
                _prx = prev._raw_x if prev._raw_x is not None else prev.position.x
                _prz = prev._raw_z if prev._raw_z is not None else prev.position.z
                disp_x = raw_x - _prx
                disp_z = raw_z - _prz
                dist = math.sqrt(
                    disp_x ** 2
                    + (self.position.y - prev.position.y) ** 2
                    + disp_z ** 2
                )
                if dist > 0.025:
                    direction = (
                        1.0 if (disp_x * fwd_x + disp_z * fwd_z) >= 0.0 else -1.0
                    )
                    raw_speed = direction * dist / dt
                else:
                    raw_speed = 0.0

            # Raw-sample jerk (noisy) — only used until we have a filtered speed baseline.
            if prev._raw_speed is not None and dt > 1e-9:
                raw_accel = (raw_speed - prev._raw_speed) / dt
            else:
                raw_accel = 0.0

            if prev._smooth_speed is None:
                smooth_speed = raw_speed
                smooth_accel = raw_accel
            else:
                alpha = _tmp_speed_ema_alpha(abs((prev.speed + raw_speed)/2))
                smooth_speed = alpha * raw_speed + (1.0 - alpha) * prev.speed
                kin_accel = (
                    (smooth_speed - prev.speed) / dt if dt > 1e-9 else 0.0
                )
                prev_sa = (
                    prev._smooth_accel
                    if prev._smooth_accel is not None
                    else kin_accel
                )
                beta = _tmp_accel_ema_alpha(abs((prev.speed + raw_speed)/2))
                smooth_accel = beta * kin_accel + (1.0 - beta) * prev_sa

            self._raw_speed = raw_speed
            self._smooth_speed = smooth_speed
            self._smooth_accel = smooth_accel
            self.speed = smooth_speed
            self.acceleration = smooth_accel
        else:
            # AI: populate position history for curvature_from_history()
            _ai_hist = list(self._position_history)
            _ai_hist.append((t_now, raw_x, raw_z))
            if len(_ai_hist) > _TMP_SPEED_HISTORY_LEN:
                _ai_hist = _ai_hist[-_TMP_SPEED_HISTORY_LEN:]
            self._position_history = _ai_hist

    def curvature_from_history(self) -> float | None:
        """Curvature (1/m) from circumscribed circle fit over _position_history.

        Averages over up to four (oldest, mid, newest) triples for stability.
        Returns None when < 3 samples; 0.0 when near-stationary or near-straight.
        Falls back to angular_velocity / speed in get_arc() when None.
        """
        hist = self._position_history
        if len(hist) < 3:
            return None
        _, x0, z0 = hist[0]
        _, xn, zn = hist[-1]
        if (xn - x0) ** 2 + (zn - z0) ** 2 < 0.05 ** 2:
            return 0.0

        n = len(hist)
        candidates = [(0, n // 2, n - 1)]
        if n >= 5:
            candidates.append((1, (n - 1) // 2, n - 2))
        if n >= 7:
            candidates += [(0, n // 3, n - 1), (0, 2 * n // 3, n - 1)]

        total_k = 0.0
        count = 0
        for i, j, k in candidates:
            _, ax, az = hist[i]
            _, bx, bz = hist[j]
            _, cx, cz = hist[k]
            if (bx - ax) ** 2 + (bz - az) ** 2 < 0.05 ** 2:
                continue
            if (cx - bx) ** 2 + (cz - bz) ** 2 < 0.05 ** 2:
                continue
            D = 2.0 * (ax * (bz - cz) + bx * (cz - az) + cx * (az - bz))
            if abs(D) < 1e-6:
                count += 1  # collinear → κ = 0 contribution
                continue
            a2 = ax * ax + az * az
            b2 = bx * bx + bz * bz
            c2 = cx * cx + cz * cz
            ux = (a2 * (bz - cz) + b2 * (cz - az) + c2 * (az - bz)) / D
            uz = -(a2 * (bx - cx) + b2 * (cx - ax) + c2 * (ax - bx)) / D
            R = max(math.sqrt((ax - ux) ** 2 + (az - uz) ** 2), _MIN_CURVATURE_RADIUS)
            cross = (bx - ax) * (cz - bz) - (bz - az) * (cx - bx)
            total_k += (-1.0 if cross > 0.0 else 1.0) / R
            count += 1

        return total_k / count if count > 0 else None

    def get_arc(
        self,
        horizon: float = 3.0,
        half_width: float | None = None,
        decel: float = 0.0,
        arc_start_pctg: float = 1.0,
    ) -> ArcPath:
        """ArcPath for this vehicle from smoothed pose and curvature.

        Curvature is derived from position history when available (circumscribed
        circle fit), falling back to angular_velocity / speed. Crash-induced
        backward position spikes are suppressed by the 6 m/s² cap in _accel_to_arc_params().
        """
        yaw_rad = (
            self._smooth_yaw
            if self._smooth_yaw is not None
            else math.radians(self.rotation.euler()[1])
        )
        abs_speed = abs(self.speed)
        _hist_k = self.curvature_from_history()
        if _hist_k is not None:
            curvature = _hist_k
        else:
            curvature = math.radians(self.angular_velocity) / abs_speed if abs_speed > 0.5 else 0.0
        effective_hw = half_width if half_width is not None else self.size.width / 2.0
        effective_decel, effective_accel = _accel_to_arc_params(self.accel_for_arc(), decel)

        is_reversing = self.speed < -1e-3
        effective_p = (1.0 - arc_start_pctg) if is_reversing else arc_start_pctg
        back_ratio = 0.5 if self.is_tmp else 0.82
        fwd_x = -math.sin(yaw_rad)
        fwd_z = -math.cos(yaw_rad)
        body_offset = (effective_p - back_ratio) * self.size.length
        start_x = self.position.x + body_offset * fwd_x
        start_z = self.position.z + body_offset * fwd_z

        return build_arc(
            start_x, start_z, yaw_rad, self.speed,
            curvature, effective_hw, horizon,
            decel=effective_decel, accel=effective_accel,
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.id}, pos={self.position}, "
            f"speed={self.speed:.2f}, is_tmp={self.is_tmp})"
        )