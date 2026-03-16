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
_RAW_POSITION_ALPHA: float = 0.27
_RAW_POSITION_ALPHA_TMP: float = 0.15
_MIN_CURVATURE_RADIUS: float = 5.0
_STRAIGHT_CURVATURE_EPS: float = 1e-6


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
        """Compute cached fields. Call after setting fields."""
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
        return self.position_at_dist(self._dist_at_time(t))

    def heading_at_dist(self, dist: float) -> float:
        if self.is_straight:
            return self.yaw_rad
        dist = max(0.0, min(dist, self.arc_length))
        frac = dist / self.arc_length if self.arc_length > 0 else 0.0
        return self.yaw_rad + frac * self.max_sweep

    def sample_points(self, n: int = 16) -> list[tuple[float, float]]:
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
    """Earliest corridor collision: ``(time_s, hit_x, hit_z)`` or ``None``.

    ``min_lateral_gap`` — if > 0, a candidate hit is suppressed when the
    perpendicular distance between the two centerlines (measured along a's
    instantaneous heading) is >= this value.  Use for head-on turns where
    arc paths overlap in the forward dimension but the vehicles remain in
    their own lanes laterally.
    """
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
        self.acceleration = acceleration
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
            if abs(self.angular_velocity) > _MAX_ANGULAR_VELOCITY:
                self.angular_velocity = 0.0
            self.speed = prev.speed
            if self.is_tmp:
                self.acceleration = prev.acceleration
            if self._smooth_x is not None:
                self.position.x = self._smooth_x
                self.position.z = self._smooth_z
            return

        self.time = t_now
        self.last_location = prev.position
        self.last_rotation = prev.rotation
        self._smooth_x = prev._smooth_x
        self._smooth_z = prev._smooth_z
        self._smooth_yaw = prev._smooth_yaw

        last_yaw = prev.last_rotation.euler()[1]
        current_yaw = self.rotation.euler()[1]
        raw_av = (current_yaw - last_yaw) / dt / 2.0
        self.angular_velocity = (
            0.0 if abs(raw_av) > _MAX_ANGULAR_VELOCITY else raw_av
        )

        raw_x = self.position.x
        raw_z = self.position.z

        if self.is_tmp:
            prev_raw_x = prev._raw_x if prev._raw_x is not None else prev.position.x
            prev_raw_z = prev._raw_z if prev._raw_z is not None else prev.position.z
            disp_x = raw_x - prev_raw_x
            disp_z = raw_z - prev_raw_z
            dist = math.sqrt(
                disp_x ** 2
                + (self.position.y - prev.position.y) ** 2
                + disp_z ** 2
            )
            if dist > 0.025:
                _, yaw_deg, _ = self.rotation.euler()
                yaw_rad = math.radians(yaw_deg)
                fwd_x = -math.sin(yaw_rad)
                fwd_z = -math.cos(yaw_rad)
                direction = 1.0 if (disp_x * fwd_x + disp_z * fwd_z) >= 0.0 else -1.0
                self.speed = direction * dist / dt
            else:
                self.speed = 0.0
        elif self.speed > 1e-3:
            lp = prev.last_location
            disp_x = self.position.x - lp.x
            disp_z = self.position.z - lp.z
            if disp_x * disp_x + disp_z * disp_z > 0.025 ** 2:
                _, yaw_deg, _ = self.rotation.euler()
                yaw_rad = math.radians(yaw_deg)
                fwd_x = -math.sin(yaw_rad)
                fwd_z = -math.cos(yaw_rad)
                if (disp_x * fwd_x + disp_z * fwd_z) < 0.0:
                    self.speed = -self.speed

        self._raw_x = raw_x
        self._raw_z = raw_z

        # Prediction-corrected position EMA — see AGENTS.md §7
        if self._smooth_x is None:
            self._smooth_x = raw_x
            self._smooth_z = raw_z
        else:
            _pred_yaw = self._smooth_yaw if self._smooth_yaw is not None else math.radians(self.rotation.euler()[1])
            _pred_fwd_x = -math.sin(_pred_yaw)
            _pred_fwd_z = -math.cos(_pred_yaw)
            _clamped_accel = max(-6.0, min(4.0, self.acceleration))
            _pred_dist = self.speed * dt + 0.5 * _clamped_accel * dt * dt
            _pred_x = self._smooth_x + _pred_dist * _pred_fwd_x
            _pred_z = self._smooth_z + _pred_dist * _pred_fwd_z
            _alpha = _RAW_POSITION_ALPHA_TMP if self.is_tmp else _RAW_POSITION_ALPHA
            self._smooth_x = _alpha * raw_x + (1.0 - _alpha) * _pred_x
            self._smooth_z = _alpha * raw_z + (1.0 - _alpha) * _pred_z
        self.position.x = self._smooth_x
        self.position.z = self._smooth_z

        # Wrap-safe yaw EMA
        raw_yaw = math.radians(self.rotation.euler()[1])
        if self._smooth_yaw is None:
            self._smooth_yaw = raw_yaw
        else:
            diff = (raw_yaw - self._smooth_yaw + math.pi) % (2.0 * math.pi) - math.pi
            _yaw_alpha = _RAW_POSITION_ALPHA_TMP if self.is_tmp else _RAW_POSITION_ALPHA
            self._smooth_yaw = self._smooth_yaw + _yaw_alpha * diff

    def get_arc(
        self,
        horizon: float = 3.0,
        half_width: float | None = None,
        decel: float = 0.0,
        arc_start_pctg: float = 1.0,
    ) -> ArcPath:
        yaw_rad = (
            self._smooth_yaw
            if self._smooth_yaw is not None
            else math.radians(self.rotation.euler()[1])
        )
        abs_speed = abs(self.speed)
        curvature = math.radians(self.angular_velocity) / abs_speed if abs_speed > 0.5 else 0.0
        effective_hw = half_width if half_width is not None else self.size.width / 2.0
        clamped_accel = (
            0.0 if decel > 0.0
            else max(-6.0, min(4.0, self.acceleration))
        )

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
            decel=decel, accel=clamped_accel,
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.id}, pos={self.position}, "
            f"speed={self.speed:.2f}, is_tmp={self.is_tmp})"
        )