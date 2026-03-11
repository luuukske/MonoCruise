"""
ETS2/ATS traffic vehicle classes with path history and intersection detection.

Coordinate system (see AGENTS notes for full reference):
  X = East/West (increases East)
  Y = Up/Down   (ignored for 2-D ground-plane work)
  Z = North/South (increases South)
  Ground plane = XZ.

Yaw conventions used throughout:
  - Traffic vehicles use a Quaternion whose x/y are deliberately swapped to
    compensate for ETS2's internal quaternion axis ordering.  Do NOT remove
    that swap.
  - euler() returns (pitch, yaw, roll) in degrees; yaw=0 → South, positive CCW.
  - Forward vector from yaw degrees:
      fwd_x = -sin(radians(yaw))
      fwd_z = -cos(radians(yaw))

Path filtering (per-update) – three successive smoothing stages:
  1. Raw-position EMA:  smoothed_actual = α_raw * actual + (1-α_raw) * prev_smoothed
  2. Prediction blend:  filtered = (1-α_blend) * predicted + α_blend * smoothed_actual
  3. Path-point EMA:    recorded = α_path * filtered + (1-α_path) * last_path_point

The last 10 recorded positions are kept as a VehiclePathPoint list and used
for intersection detection, and the vehicle's own ``position.x`` /
``position.z`` are also updated to the recorded values so that debug drawing
and geometry queries see the smoothed location.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PATH_POINTS: int = 10
_LOCATION_UPDATE_FREQUENCY: float = 0.05  # minimum seconds between path updates (≈20 fps)
_MAX_ANGULAR_VELOCITY: float = 90.0       # deg/s – clamp to suppress telemetry noise
_SEGMENT_EPS: float = 1e-9               # denominator epsilon for segment intersection
_TIME_EPS: float = 0.05                  # seconds – tolerance when averaging hit times
_SMOOTHING_FACTOR: float = 0.01          # weight on actual vs predicted in the blend stage
_RAW_POSITION_ALPHA: float = 0.25        # EMA weight on incoming raw position (lower = smoother)
_PATH_POINT_ALPHA: float = 0.45          # EMA weight on new filtered point vs previous path point (lower = smoother)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _rotate_around_point(
    point: list[float],
    center: list[float],
    pitch_deg: float,
    yaw_deg: float,
) -> list[float]:
    """Rotate *point* around *center* by pitch then yaw (degrees).  Roll is
    always 0 in top-down work so it is omitted here for speed.

    Returns [x, y, z].
    """
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)

    x = point[0] - center[0]
    y = point[1] - center[1]
    z = point[2] - center[2]

    # Pitch (around X-axis)
    y, z = (
        y * math.cos(pitch_rad) - z * math.sin(pitch_rad),
        y * math.sin(pitch_rad) + z * math.cos(pitch_rad),
    )

    # Yaw (around Y-axis)
    x, z = (
        x * math.cos(yaw_rad) - z * math.sin(yaw_rad),
        x * math.sin(yaw_rad) + z * math.cos(yaw_rad),
    )

    return [x + center[0], y + center[1], z + center[2]]


def _predict_position(
    x: float,
    z: float,
    speed: float,
    yaw_deg: float,
    angular_velocity_deg_s: float,
    dt: float,
) -> tuple[float, float]:
    """Return the predicted (x, z) position after *dt* seconds given current
    speed, yaw (degrees), and yaw angular velocity (degrees/second).

    Uses exponential decay on angular velocity to mimic the original
    ``get_position_in`` approach.  Returns (pred_x, pred_z).
    """
    yaw_rad = math.radians(yaw_deg)
    ang_rad = math.radians(angular_velocity_deg_s)

    if ang_rad != 0.0:
        decay_rate = 0.3
        total_decay = (1.0 - math.exp(-decay_rate * dt)) / decay_rate
        yaw_rad += ang_rad * total_decay
    # else: no angular adjustment needed

    distance = speed * dt
    pred_x = x - distance * math.sin(yaw_rad)
    pred_z = z - distance * math.cos(yaw_rad)
    return pred_x, pred_z


def _segment_intersection_params(
    ax: float, az: float, bx: float, bz: float,
    cx: float, cz: float, dx: float, dz: float,
) -> Optional[tuple[float, float]]:
    """Compute the parametric intersection (s, t) of segment AB and segment CD
    on the XZ plane.

    Returns (s, t) where s ∈ [0,1] is the parameter along AB and t ∈ [0,1] is
    the parameter along CD.  Returns None if the segments are parallel,
    near-parallel, or do not cross within their extent.
    """
    r_x = bx - ax
    r_z = bz - az
    s_x = dx - cx
    s_z = dz - cz

    # Cross product of r and s (2-D, gives scalar)
    denom = r_x * s_z - r_z * s_x
    if abs(denom) < _SEGMENT_EPS:
        return None  # parallel or collinear

    qp_x = cx - ax
    qp_z = cz - az

    s = (qp_x * s_z - qp_z * s_x) / denom
    t = (qp_x * r_z - qp_z * r_x) / denom

    if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
        return s, t
    return None


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

class Position:
    """3-D world position on the ETS2 XZ ground plane.  Y is carried along for
    completeness but never used in 2-D calculations.
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y, self.z - other.z)

    def tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def is_zero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0 and self.z == 0.0

    def __str__(self) -> str:
        return f"Position({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Quaternion:
    """ETS2 traffic-vehicle quaternion.

    The constructor deliberately swaps the x and y arguments to compensate for
    ETS2's internal quaternion axis ordering.  Removing this swap breaks all
    traffic vehicle rotations.

    euler() returns (pitch, yaw, roll) in degrees.  yaw=0 → South, positive CCW.
    """

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self.w: float = w
        self.x: float = y  # intentional swap
        self.y: float = x  # intentional swap
        self.z: float = z

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

    def __str__(self) -> str:
        p, y, r = self.euler()
        return (
            f"Quaternion({self.w:.2f}, {self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
            f" -> (pitch {p:.2f}, yaw {y:.2f}, roll {r:.2f})"
        )

    def __dict__(self) -> dict:  # type: ignore[override]
        p, y, r = self.euler()
        return {
            "w": self.w,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "pitch": p,
            "yaw": y,
            "roll": r,
        }


class Size:
    """Bounding-box dimensions of a vehicle."""

    __slots__ = ("width", "height", "length")

    def __init__(self, width: float, height: float, length: float) -> None:
        self.width: float = width
        self.height: float = height
        self.length: float = length

    def __str__(self) -> str:
        return f"Size({self.width:.2f}, {self.height:.2f}, {self.length:.2f})"


class Trailer:
    """A trailer attached to a vehicle."""

    __slots__ = ("position", "rotation", "size", "is_tmp")

    def __init__(
        self,
        position: Position,
        rotation: Quaternion,
        size: Size,
        is_tmp: bool = False,
    ) -> None:
        self.position: Position = position
        self.rotation: Quaternion = rotation
        self.size: Size = size
        self.is_tmp: bool = is_tmp

    def correct_position(self) -> Position:
        """Move pivot position backward by half the trailer length so that the
        returned point is the geometric centre.  Used for TMP trailers whose
        pivot is at the hitch rather than the centre.
        """
        _, yaw_deg, _ = self.rotation.euler()
        yaw_rad = math.radians(yaw_deg)
        return Position(
            self.position.x + (self.size.length / 2.0) * math.sin(yaw_rad),
            self.position.y,
            self.position.z + (self.size.length / 2.0) * math.cos(yaw_rad),
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()

    def __str__(self) -> str:
        return f"Trailer({self.position}, {self.rotation}, {self.size})"

    def __dict__(self) -> dict:  # type: ignore[override]
        pos = self.correct_position() if self.is_tmp else self.position
        return {
            "position": pos.__dict__,
            "rotation": self.rotation.__dict__(),
            "size": self.size.__dict__,
        }


@dataclass
class VehiclePathPoint:
    """A single filtered path sample for a vehicle.

    *x* and *z* are the filtered ground-plane coordinates (Y is ignored here).
    *timestamp* is the absolute ``time.time()`` value when the sample was
    recorded.  *yaw_deg* is the vehicle's heading in degrees at that moment
    (yaw=0 → South, positive CCW).
    """

    x: float
    z: float
    timestamp: float
    yaw_deg: float = 0.0


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------

class Vehicle:
    """A traffic vehicle with filtered path history.

    Public attributes mirror the originals in ``classes.py`` so existing
    consumers remain compatible.  Additional attributes:

      path_points : deque[VehiclePathPoint]
          The last ≤10 filtered positions, oldest-first.

    Use ``update_from_last(prev_vehicle)`` to advance state and update the
    path after receiving fresh telemetry.
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
        self.position: Position = position
        self.rotation: Quaternion = rotation
        self.size: Size = size
        self.speed: float = speed
        self.acceleration: float = acceleration
        self.trailer_count: int = trailer_count
        self.trailers: list[Trailer] = trailers
        self.id: int = id
        self.is_tmp: bool = is_tmp
        self.is_trailer: bool = is_trailer

        self.time: float = time.time()
        self.last_location: Position = Position(0.0, 0.0, 0.0)
        self.last_rotation: Quaternion = Quaternion(0.0, 0.0, 0.0, 0.0)
        self.angular_velocity: float = 0.0  # degrees per second around yaw axis

        # Filtered path history – at most _MAX_PATH_POINTS entries, oldest first.
        self.path_points: deque[VehiclePathPoint] = deque(maxlen=_MAX_PATH_POINTS)

        # Running EMA state for raw-position smoothing (stage 1).
        # None until the first update so we can seed from the actual position.
        self._smooth_pos_x: Optional[float] = None
        self._smooth_pos_z: Optional[float] = None

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_from_last(self, prev: "Vehicle") -> None:
        """Advance vehicle state using the previous frame's data and update the
        filtered path history.

        Mirrors the logic of the original ``classes.py`` ``update_from_last``
        method, and additionally computes a 70/30 filtered position which is
        appended to ``self.path_points``.
        """
        now = time.time()
        dt = now - prev.time

        if dt < _LOCATION_UPDATE_FREQUENCY:
            # Not enough time has passed – carry forward history and derived
            # quantities but do not record a new path point.
            self.time = prev.time
            self.last_location = prev.last_location
            self.last_rotation = prev.last_rotation
            self.angular_velocity = prev.angular_velocity
            self.path_points = prev.path_points
            self._smooth_pos_x = prev._smooth_pos_x
            self._smooth_pos_z = prev._smooth_pos_z
            if abs(self.angular_velocity) > _MAX_ANGULAR_VELOCITY:
                self.angular_velocity = 0.0
            if self.is_tmp:
                self.speed = prev.speed
                self.acceleration = prev.acceleration
            return

        # Enough time has elapsed – record a new update.
        self.time = now
        self.last_location = prev.position
        self.last_rotation = prev.rotation
        self.path_points = prev.path_points  # carry forward existing history
        self._smooth_pos_x = prev._smooth_pos_x
        self._smooth_pos_z = prev._smooth_pos_z

        # Angular velocity from yaw change.
        last_yaw = prev.last_rotation.euler()[1]
        current_yaw = self.rotation.euler()[1]
        raw_angular_velocity = (current_yaw - last_yaw) / dt / 2.0
        # Divide by 2 empirically gives more accurate results (mirrors original).
        self.angular_velocity = (
            0.0 if abs(raw_angular_velocity) > _MAX_ANGULAR_VELOCITY
            else raw_angular_velocity
        )

        # TMP speed from positional distance.
        if self.is_tmp:
            last_pos = prev.last_location
            dist = math.sqrt(
                (self.position.x - last_pos.x) ** 2
                + (self.position.y - last_pos.y) ** 2
                + (self.position.z - last_pos.z) ** 2
            )
            self.speed = dist / dt if dist > 0.025 else 0.0

        # Compute filtered position and append to path history.
        self._record_filtered_point(now, dt)

    def _record_filtered_point(self, timestamp: float, dt: float) -> None:
        """Compute the smoothed position (three stages) and push it onto path_points.

        Stage 1 – raw-position EMA:
            Reduces high-frequency jitter in the telemetry feed before the
            position is used in any further calculation.

        Stage 2 – prediction blend:
            Weights the kinematic prediction against the (now pre-smoothed)
            actual position.  The prediction dominates so the path stays
            physically plausible even when telemetry hiccups.

        Stage 3 – path-point EMA:
            Blends the result against the previously recorded path point so
            that the visible polyline does not jump between frames.
        """
        _, yaw_deg, _ = self.rotation.euler()

        # -- Stage 1: EMA on raw telemetry position ---------------------------
        actual_x = self.position.x
        actual_z = self.position.z

        if self._smooth_pos_x is None:
            # Seed from first observation.
            self._smooth_pos_x = actual_x
            self._smooth_pos_z = actual_z
        else:
            self._smooth_pos_x = (
                _RAW_POSITION_ALPHA * actual_x
                + (1.0 - _RAW_POSITION_ALPHA) * self._smooth_pos_x
            )
            self._smooth_pos_z = (
                _RAW_POSITION_ALPHA * actual_z
                + (1.0 - _RAW_POSITION_ALPHA) * self._smooth_pos_z
            )

        # -- Stage 2: kinematic prediction blended with smoothed actual -------
        prev_x = self.last_location.x
        prev_z = self.last_location.z
        pred_x, pred_z = _predict_position(
            prev_x, prev_z, self.speed, yaw_deg, self.angular_velocity, dt
        )

        filtered_x = (
            (1.0 - _SMOOTHING_FACTOR) * pred_x
            + _SMOOTHING_FACTOR * self._smooth_pos_x
        )
        filtered_z = (
            (1.0 - _SMOOTHING_FACTOR) * pred_z
            + _SMOOTHING_FACTOR * self._smooth_pos_z
        )

        # -- Stage 3: EMA against the previous path point ---------------------
        if self.path_points:
            last = self.path_points[-1]
            filtered_x = (
                _PATH_POINT_ALPHA * filtered_x
                + (1.0 - _PATH_POINT_ALPHA) * last.x
            )
            filtered_z = (
                _PATH_POINT_ALPHA * filtered_z
                + (1.0 - _PATH_POINT_ALPHA) * last.z
            )

        # Update the vehicle's live position so that all consumers (debug
        # window, geometry helpers, etc.) see the smoothed coordinates.
        self.position.x = filtered_x
        self.position.z = filtered_z

        self.path_points.append(
            VehiclePathPoint(
                x=filtered_x,
                z=filtered_z,
                timestamp=timestamp,
                yaw_deg=yaw_deg,
            )
        )

    # ------------------------------------------------------------------
    # Path accessors
    # ------------------------------------------------------------------

    def get_path_points(self) -> list[VehiclePathPoint]:
        """Return a list copy of the filtered path history (oldest first)."""
        return list(self.path_points)

    def get_path_segments(
        self,
    ) -> list[tuple[VehiclePathPoint, VehiclePathPoint]]:
        """Return consecutive pairs of path points as segments (oldest first).

        Each tuple is (start_point, end_point).  An empty list is returned if
        fewer than two path points exist.
        """
        pts = list(self.path_points)
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    def get_filtered_position(self) -> Optional[VehiclePathPoint]:
        """Return the most recent filtered path point, or None if no path yet."""
        return self.path_points[-1] if self.path_points else None

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def get_corners(
        self,
        offset: Optional[Position] = None,
        correction_multiplier: float = 1.0,
    ) -> tuple[Position, Position, Position, Position]:
        """Return (front_left, front_right, back_right, back_left) world corners.

        TMP vehicles use symmetric ±length/2 offsets; AI vehicles use
        asymmetric offsets (0.18 front, 0.82 rear) to account for the pivot
        point being near the front axle rather than the geometric centre.
        """
        gm = [self.position.x, self.position.y, self.position.z]
        if offset is not None:
            gm[0] += offset.x
            gm[1] += offset.y
            gm[2] += offset.z

        hw = self.size.width / 2.0
        if self.is_tmp:
            rear_z = gm[2] + self.size.length / 2.0 * correction_multiplier
            front_z = gm[2] - self.size.length / 2.0 * correction_multiplier
        else:
            rear_z = gm[2] + self.size.length * 0.82 * correction_multiplier
            front_z = gm[2] - self.size.length * 0.18 * correction_multiplier

        raw_corners = [
            [gm[0] - hw, gm[1], front_z],  # front_left
            [gm[0] + hw, gm[1], front_z],  # front_right
            [gm[0] + hw, gm[1], rear_z],   # back_right
            [gm[0] - hw, gm[1], rear_z],   # back_left
        ]

        pitch_deg, yaw_deg, _ = self.rotation.euler()
        rotated = [
            _rotate_around_point(c, gm, pitch_deg, -yaw_deg) for c in raw_corners
        ]

        return tuple(Position(*c) for c in rotated)  # type: ignore[return-value]

    def get_corrected_position(self) -> Position:
        """Return the geometric centre of the vehicle's bounding box."""
        fl, fr, br, bl = self.get_corners()
        return Position(
            (fl.x + fr.x + br.x + bl.x) / 4.0,
            (fl.y + fr.y + br.y + bl.y) / 4.0,
            (fl.z + fr.z + br.z + bl.z) / 4.0,
        )

    def is_zero(self) -> bool:
        return self.position.is_zero() and self.rotation.is_zero()

    # ------------------------------------------------------------------
    # Serialisation (mirrors original classes.py interface)
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"Vehicle(id={self.id}, pos={self.position}, "
            f"speed={self.speed:.2f}, is_tmp={self.is_tmp})"
        )

    def __dict__(self) -> dict:  # type: ignore[override]
        return {
            "position": self.position.__dict__,
            "rotation": self.rotation.__dict__(),
            "size": self.size.__dict__,
            "speed": self.speed,
            "acceleration": self.acceleration,
            "trailer_count": self.trailer_count,
            "trailers": [t.__dict__() for t in self.trailers],
            "id": self.id,
            "is_tmp": self.is_tmp,
            "is_trailer": self.is_trailer,
        }


# ---------------------------------------------------------------------------
# Intersection detection
# ---------------------------------------------------------------------------

def find_paths_intersection_time(
    path1: list[VehiclePathPoint],
    path2: list[VehiclePathPoint],
) -> Optional[tuple[float, Position]]:
    """Return the earliest intersection time and XZ position between two
    vehicle paths represented as filtered point lists, or ``None``.

    Each path is treated as a polyline on the XZ plane.  Every pair of
    consecutive segments (one from each path) is tested for 2-D intersection.
    When an intersection is found, the crossing time is interpolated linearly
    within the time interval of each segment.  If the two interpolated times
    differ by less than ``_TIME_EPS``, their average is used; otherwise the
    segment pair is skipped as a likely numerical artifact.

    Parameters
    ----------
    path1, path2:
        Chronologically ordered lists of ``VehiclePathPoint``.  At least two
        points are needed to form a segment.  Fewer than two points causes an
        immediate ``None`` return.

    Returns
    -------
    ``(timestamp, Position)`` of the earliest intersection, or ``None``.
    The *timestamp* is an absolute ``time.time()``-epoch value matching the
    values stored in ``VehiclePathPoint.timestamp``.  The *Position* has Y=0
    (top-down; elevation is not meaningful here).
    """
    if len(path1) < 2 or len(path2) < 2:
        return None

    best_time: Optional[float] = None
    best_pos: Optional[Position] = None

    for i in range(len(path1) - 1):
        p1a = path1[i]
        p1b = path1[i + 1]

        # Skip degenerate segments (zero spatial length or zero time span).
        seg1_dx = p1b.x - p1a.x
        seg1_dz = p1b.z - p1a.z
        seg1_dt = p1b.timestamp - p1a.timestamp
        if seg1_dt <= 0.0 and (seg1_dx * seg1_dx + seg1_dz * seg1_dz) < _SEGMENT_EPS:
            continue

        for j in range(len(path2) - 1):
            p2a = path2[j]
            p2b = path2[j + 1]

            seg2_dt = p2b.timestamp - p2a.timestamp
            seg2_dx = p2b.x - p2a.x
            seg2_dz = p2b.z - p2a.z
            if seg2_dt <= 0.0 and (seg2_dx * seg2_dx + seg2_dz * seg2_dz) < _SEGMENT_EPS:
                continue

            params = _segment_intersection_params(
                p1a.x, p1a.z, p1b.x, p1b.z,
                p2a.x, p2a.z, p2b.x, p2b.z,
            )
            if params is None:
                continue

            s, t = params

            # Interpolate crossing time within each segment's time interval.
            time1 = p1a.timestamp + s * seg1_dt
            time2 = p2a.timestamp + t * seg2_dt

            if abs(time1 - time2) > _TIME_EPS:
                # The crossing times are too far apart – treat as a false
                # positive caused by the paths occupying different moments.
                continue

            hit_time = (time1 + time2) / 2.0

            # Only keep the earliest hit.
            if best_time is None or hit_time < best_time:
                best_time = hit_time
                hit_x = p1a.x + s * seg1_dx
                hit_z = p1a.z + s * seg1_dz
                best_pos = Position(hit_x, 0.0, hit_z)

    if best_time is None:
        return None
    return best_time, best_pos  # type: ignore[return-value]


def find_path_intersection_time(
    v1: Vehicle,
    v2: Vehicle,
) -> Optional[tuple[float, Position]]:
    """Return the earliest intersection time and position between two vehicles'
    filtered path histories, or ``None`` if their paths do not cross.

    Uses only the last ≤10 filtered positions stored on each vehicle.  At least
    two positions are required on each vehicle to form any segment.

    Returns
    -------
    ``(timestamp, Position)`` – absolute epoch timestamp and XZ position – or
    ``None``.
    """
    return find_paths_intersection_time(
        v1.get_path_points(),
        v2.get_path_points(),
    )
