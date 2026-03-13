"""
ETS2/ATS traffic vehicle classes with arc-based path prediction.

Coordinate system (ETS2):
  X = East/West (increases East)
  Y = Up/Down   (ignored for ground-plane work)
  Z = North/South (increases South)
  Ground plane = XZ.

Yaw conventions (DO NOT CHANGE — this has caused backward ego path bugs before):
  - yaw=0 rad → North (forward = -Z direction)
  - yaw=π/2 rad → West, yaw=π rad → South, yaw=3π/2 rad → East
  - Positive yaw = counter-clockwise (CCW) when viewed from above
  - Forward vector:  fwd_x = -sin(yaw_rad),  fwd_z = -cos(yaw_rad)
    This formula is FIXED. If the ego path points backward, the bug is in
    thread.py (rotationX → ego_yaw_rad conversion), NOT here. Never flip
    the signs or use (sin, cos) — that would break traffic vehicle paths.

Vehicles carry an ArcPath (center, radius, angular span, start angle)
instead of sampled polylines.  This enables O(1) position lookups and
fast closed-form arc–arc collision detection.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_ANGULAR_VELOCITY: float = 90.0   # deg/s clamp
_LOCATION_UPDATE_FREQUENCY: float = 0.05
_RAW_POSITION_ALPHA: float = 0.25
_MIN_CURVATURE_RADIUS: float = 5.0    # metres — floor for turning radius
_STRAIGHT_CURVATURE_EPS: float = 1e-6 # curvature below this → straight line


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

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
    """ETS2 traffic quaternion — x/y are deliberately swapped."""
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self.w = w
        self.x = y   # intentional swap
        self.y = x   # intentional swap
        self.z = z

    def euler(self) -> tuple[float, float, float]:
        """Return (pitch, yaw, roll) in degrees."""
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
        """Move pivot backward by half the trailer length (TMP trailers)."""
        _, yaw_deg, _ = self.rotation.euler()
        yaw_rad = math.radians(yaw_deg)
        return Position(
            self.position.x + (self.size.length / 2.0) * math.sin(yaw_rad),
            self.position.y,
            self.position.z + (self.size.length / 2.0) * math.cos(yaw_rad),
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()


# ---------------------------------------------------------------------------
# Arc path — the core geometry primitive
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ArcPath:
    """A predicted path as a circular arc (or straight ray).

    For a vehicle at (start_x, start_z) heading yaw_rad with signed curvature κ:
      - |κ| < ε  → straight line
      - Else     → circular arc, R = 1/|κ|, center perpendicular to forward

    ``position_at_time(t)`` and ``position_at_dist(d)`` give O(1) lookups.
    Supports deceleration for braking arcs.
    """
    start_x: float = 0.0
    start_z: float = 0.0
    yaw_rad: float = 0.0
    speed: float = 0.0
    curvature: float = 0.0
    half_width: float = 1.25
    horizon: float = 3.0
    decel: float = 0.0

    # Cached (computed by build())
    is_straight: bool = True
    center_x: float = 0.0
    center_z: float = 0.0
    radius: float = 0.0
    angle0: float = 0.0
    max_sweep: float = 0.0
    arc_length: float = 0.0
    fwd_x: float = 0.0
    fwd_z: float = -1.0
    _sign: float = 1.0       # curvature sign for sweep direction

    def build(self) -> "ArcPath":
        """Compute cached fields from canonical state. Call after setting fields."""
        # DO NOT CHANGE: Forward vector formula. ETS2 convention: fwd = (-sin, -cos).
        # If ego path points backward, fix thread.py rotationX→yaw conversion, not this.
        self.fwd_x = -math.sin(self.yaw_rad)
        self.fwd_z = -math.cos(self.yaw_rad)

        if self.speed < 1e-3:
            self.is_straight = True
            self.arc_length = 0.0
            self.max_sweep = 0.0
            return self

        # Effective travel distance accounting for deceleration
        if self.decel > 0.0:
            t_stop = self.speed / self.decel
            if t_stop < self.horizon:
                self.arc_length = self.speed * t_stop - 0.5 * self.decel * t_stop * t_stop
            else:
                t = self.horizon
                self.arc_length = self.speed * t - 0.5 * self.decel * t * t
        else:
            self.arc_length = self.speed * self.horizon

        if abs(self.curvature) < _STRAIGHT_CURVATURE_EPS:
            self.is_straight = True
            self.radius = 0.0
            self.max_sweep = 0.0
        else:
            self.is_straight = False
            self.radius = max(abs(1.0 / self.curvature), _MIN_CURVATURE_RADIUS)

            # Center perpendicular to forward.  Positive κ → turn left (CCW).
            # DO NOT CHANGE: center offset uses fwd_z and -fwd_x; tied to fwd formula.
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
        """Arc-length distance at time t (handles decel clamping)."""
        if self.decel > 0.0:
            t_stop = self.speed / self.decel
            if t >= t_stop:
                return self.speed * t_stop - 0.5 * self.decel * t_stop * t_stop
            return self.speed * t - 0.5 * self.decel * t * t
        return self.speed * t

    def position_at_dist(self, dist: float) -> tuple[float, float]:
        """Return (x, z) at a given arc-length distance from start."""
        dist = max(0.0, min(dist, self.arc_length))
        if self.is_straight:
            # DO NOT CHANGE: start + dist*fwd extends FORWARD. Never use minus.
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
        """Return (x, z) at time t seconds from now."""
        return self.position_at_dist(self._dist_at_time(t))

    def heading_at_dist(self, dist: float) -> float:
        """Return heading (radians) at given arc distance."""
        if self.is_straight:
            return self.yaw_rad
        dist = max(0.0, min(dist, self.arc_length))
        frac = dist / self.arc_length if self.arc_length > 0 else 0.0
        return self.yaw_rad + frac * self.max_sweep

    def sample_points(self, n: int = 16) -> list[tuple[float, float]]:
        """Return n evenly-spaced (x, z) samples along the arc for debug."""
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
        """Return (left_edge, right_edge) as n-point polylines for corridor drawing."""
        if n < 2 or self.arc_length < 1e-6:
            return [(self.start_x, self.start_z)], [(self.start_x, self.start_z)]

        if self.is_straight:
            # Straight: offset perpendicular at each point.
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

        # Curved: sample true concentric arcs so inner = tighter, outer = gentler.
        # Positive curvature → turn left → center to left → inner = left, outer = right.
        # THIS SHOULD NOT BE CHANGED!
        # this is a working part of the code and should only be changed if functionality needs to be altered.
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
) -> ArcPath:
    """Convenience constructor."""
    return ArcPath(
        start_x=x, start_z=z, yaw_rad=yaw_rad, speed=speed,
        curvature=curvature, half_width=half_width, horizon=horizon,
        decel=decel,
    ).build()


# ---------------------------------------------------------------------------
# Arc–Arc collision detection
# ---------------------------------------------------------------------------

def arc_arc_collision(
    a: ArcPath,
    b: ArcPath,
    margin: float = 0.5,
    n_samples: int = 24,
) -> Optional[tuple[float, float, float]]:
    """Find the earliest time two arc corridors collide.

    Returns (time_seconds, hit_x, hit_z) or None.

    Strategy:
    - Both straight, no decel → closed-form quadratic (O(1)).
    - Otherwise → O(n) time-synchronised sampling with bisection refinement.
    """
    if a.arc_length < 1e-3 and b.arc_length < 1e-3:
        return None  # both stationary

    corridor_sq = (a.half_width + b.half_width + margin) ** 2
    horizon = min(a.horizon, b.horizon)

    # Fast path: both straight, no deceleration
    if a.is_straight and b.is_straight and a.decel <= 0 and b.decel <= 0:
        return _ray_ray_collision(a, b, corridor_sq, horizon)

    # General: time-synchronised sampling
    return _sampled_collision(a, b, corridor_sq, horizon, n_samples)


def _ray_ray_collision(
    a: ArcPath, b: ArcPath, corridor_sq: float, horizon: float,
) -> Optional[tuple[float, float, float]]:
    """Closed-form earliest collision between two straight, constant-speed corridors.

    Solves  ||(ΔP) + t·(ΔV)||² = corridor_sq  →  quadratic in t.
    """
    dpx = a.start_x - b.start_x
    dpz = a.start_z - b.start_z
    dvx = a.speed * a.fwd_x - b.speed * b.fwd_x
    dvz = a.speed * a.fwd_z - b.speed * b.fwd_z

    A = dvx * dvx + dvz * dvz
    B = 2.0 * (dpx * dvx + dpz * dvz)
    C = dpx * dpx + dpz * dpz - corridor_sq

    if C <= 0:
        return 0.0, (a.start_x + b.start_x) * 0.5, (a.start_z + b.start_z) * 0.5

    if abs(A) < 1e-12:
        return None  # parallel / same speed

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
    return t_hit, (ax + bx) * 0.5, (az + bz) * 0.5


def _sampled_collision(
    a: ArcPath, b: ArcPath, corridor_sq: float, horizon: float, n: int,
) -> Optional[tuple[float, float, float]]:
    """Time-synchronised sampling with bisection refinement."""
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
            # Bisect between this step and the previous to refine
            lo = max(t - horizon * inv_n, 0.0)
            hi = t
            best_t = t
            best_mx = (ax + bx) * 0.5
            best_mz = (az + bz) * 0.5
            for _ in range(6):  # ~1/64 of step size precision
                mid = (lo + hi) * 0.5
                ax2, az2 = a.position_at_time(mid)
                bx2, bz2 = b.position_at_time(mid)
                if (ax2 - bx2) ** 2 + (az2 - bz2) ** 2 < corridor_sq:
                    hi = mid
                    best_t = mid
                    best_mx = (ax2 + bx2) * 0.5
                    best_mz = (az2 + bz2) * 0.5
                else:
                    lo = mid
            break  # earliest found

    if best_t is None:
        return None
    return best_t, best_mx, best_mz


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------

class Vehicle:
    """A traffic vehicle with arc-based path prediction.

    Maintains the same public interface as the original for compatibility.
    """

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

    def update_from_last(self, prev: "Vehicle") -> None:
        now = time.time()
        dt = now - prev.time

        if dt < _LOCATION_UPDATE_FREQUENCY:
            self.time = prev.time
            self.last_location = prev.last_location
            self.last_rotation = prev.last_rotation
            self.angular_velocity = prev.angular_velocity
            self._smooth_x = prev._smooth_x
            self._smooth_z = prev._smooth_z
            self._smooth_yaw = prev._smooth_yaw
            if abs(self.angular_velocity) > _MAX_ANGULAR_VELOCITY:
                self.angular_velocity = 0.0
            if self.is_tmp:
                self.speed = prev.speed
                self.acceleration = prev.acceleration
            if self._smooth_x is not None:
                self.position.x = self._smooth_x
                self.position.z = self._smooth_z
            return

        self.time = now
        self.last_location = prev.position
        self.last_rotation = prev.rotation
        self._smooth_x = prev._smooth_x
        self._smooth_z = prev._smooth_z
        self._smooth_yaw = prev._smooth_yaw

        # Angular velocity
        last_yaw = prev.last_rotation.euler()[1]
        current_yaw = self.rotation.euler()[1]
        raw_av = (current_yaw - last_yaw) / dt / 2.0
        self.angular_velocity = (
            0.0 if abs(raw_av) > _MAX_ANGULAR_VELOCITY else raw_av
        )

        # TMP speed from position delta
        if self.is_tmp:
            lp = prev.last_location
            dist = math.sqrt(
                (self.position.x - lp.x) ** 2
                + (self.position.y - lp.y) ** 2
                + (self.position.z - lp.z) ** 2
            )
            self.speed = dist / dt if dist > 0.025 else 0.0

        # Smooth position
        ax, az = self.position.x, self.position.z
        if self._smooth_x is None:
            self._smooth_x = ax
            self._smooth_z = az
        else:
            self._smooth_x = _RAW_POSITION_ALPHA * ax + (1.0 - _RAW_POSITION_ALPHA) * self._smooth_x
            self._smooth_z = _RAW_POSITION_ALPHA * az + (1.0 - _RAW_POSITION_ALPHA) * self._smooth_z
        self.position.x = self._smooth_x
        self.position.z = self._smooth_z

        # ← ADD THIS BLOCK:
        raw_yaw = math.radians(self.rotation.euler()[1])
        if self._smooth_yaw is None:
            self._smooth_yaw = raw_yaw
        else:
            # Shortest-path EMA — handles the 0/2π wrap boundary correctly.
            # Without the modulo wrap, blending across 0/2π would spin the arc 180°.
            diff = (raw_yaw - self._smooth_yaw + math.pi) % (2.0 * math.pi) - math.pi
            self._smooth_yaw = self._smooth_yaw + _RAW_POSITION_ALPHA * diff

    def get_arc(self, horizon: float = 3.0, half_width: float | None = None) -> ArcPath:
        # !! USE _smooth_yaw, NOT rotation.euler() !!
        # Raw yaw from rotation.euler() is noisy. Even small frame-to-frame jitter
        # gets amplified across arc length — the arc tip jumps wildly while the
        # vehicle box looks stable (boxes use position, arcs use position+yaw).
        # Trailers have no arc which is why they don't show the same jumping.
        # _smooth_yaw uses the same EMA alpha (_RAW_POSITION_ALPHA) as position.
        yaw_rad = (
            self._smooth_yaw
            if self._smooth_yaw is not None
            else math.radians(self.rotation.euler()[1])
        )
        if self.speed > 0.5:
            omega_rad = math.radians(self.angular_velocity)
            curvature = omega_rad / self.speed
        else:
            curvature = 0.0
        effective_hw = half_width if half_width is not None else self.size.width / 2.0
        return build_arc(
            self.position.x, self.position.z, yaw_rad, self.speed,
            curvature, effective_hw, horizon,
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.id}, pos={self.position}, "
            f"speed={self.speed:.2f}, is_tmp={self.is_tmp})"
        )