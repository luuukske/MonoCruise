"""
this is code partially by tumppi for ETS2LA.
i (Lukas Deschryver) did some minor modifications to make it work with MonoCruise and smoother in MP.
I DO NOT TAKE CREDIT FOR THIS CODE (except for the filtering).
"""

import math
import queue
import time
from collections import deque
import threading
from turtle import pos
import numpy as np

location_update_frequency = 0.1  # 10fps

# testing
from ETS2radar.radar_visualizer import RadarVisualizer

Visualizer = RadarVisualizer()
threading.Thread(target=Visualizer.start, daemon=True).start()

SPEED_CALCULATION_WINDOW = 1 # seconds - time window for calculating maxlen of position history
POSITION_QUEUE_MAXLEN = 20 # maximum number of positions to store in queue for averaging

# TODO: Switch __dict__ to __iter__ and dict() for typing support.
# TODO: f = Class() -> dict(f) instead of f.__dict__()



def rotate_around_point(point, center, pitch, yaw, roll):
    """Rotate a point around a center point by the given pitch, yaw, and roll angles (in degrees).

    Parameters
    ----------
    - point: [x, y, z] coordinates of the point to rotate
    - center: [x, y, z] coordinates of the center of rotation
    - pitch: rotation around X-axis (in degrees)
    - yaw: rotation around Y-axis (in degrees)
    - roll: rotation around Z-axis (in degrees)

    Returns
    -------
    - Rotated point [x, y, z]

    """
    # Convert angles from degrees to radians
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(roll)

    # Translate point to origin (relative to center)
    x = point[0] - center[0]
    y = point[1] - center[1]
    z = point[2] - center[2]

    # Pitch rotation (around X-axis)
    rotated_y = y * math.cos(pitch_rad) - z * math.sin(pitch_rad)
    rotated_z = y * math.sin(pitch_rad) + z * math.cos(pitch_rad)
    y, z = rotated_y, rotated_z

    # Roll rotation (around Z-axis)
    rotated_x = x * math.cos(roll_rad) - y * math.sin(roll_rad)
    rotated_y = x * math.sin(roll_rad) + y * math.cos(roll_rad)
    x, y = rotated_x, rotated_y

    # Yaw rotation (around Y-axis)
    rotated_x = x * math.cos(yaw_rad) - z * math.sin(yaw_rad)
    rotated_z = x * math.sin(yaw_rad) + z * math.cos(yaw_rad)
    x, z = rotated_x, rotated_z

    # Translate back
    return [x + center[0], y + center[1], z + center[2]]
from dataclasses import dataclass
from collections import deque
from typing import Tuple
import time


@dataclass
class FilterConfig: 
    # NEEDS MORE TUNING BASED
    ema_alpha: float = 0.07  # EMA smoothing factor (0-1, lower = more smoothing)
    ema_accel_alpha: float = 0.17  # EMA smoothing when accelerating
    min_samples_for_accel: int = 3  # Minimum samples before calculating acceleration
    accel_smoothing_alpha: float = 0.16  # Smoothing for acceleration calculation (output)
    
    # Pre-filter settings
    pre_filter_size: int = 4  # Number of samples to average before main filter
    
    # Prediction settings
    prediction_weight: float = 0.95  # How much to trust prediction vs pure EMA (0-1)
    accel_decay: float = 0.95  # Decay factor for acceleration prediction (prevents runaway)
    
    # Offset canceling settings
    offset_alpha: float = 0.15  # Smoothing for tracking the offset (higher = faster tracking)
    offset_correction_threshold: float = 0.5  # Minimum offset before correction kicks in (m/s)
    offset_correction_gain:  float = 0.3  # How aggressively to correct the offset (0-1)
    max_correction_rate: float = 2.0  # Maximum correction per update (m/s)


class SpeedFilter: 
    """
    Predictive EMA-based speed filter with pre-averaging and offset canceling 
    for accurate tracking during hard braking while maintaining smoothness at steady state.
    
    Uses a short moving average buffer to remove noise, then velocity and 
    acceleration to predict the next value, reducing lag during deceleration events.
    """

    def __init__(self, config: FilterConfig = None):
        self.cfg = config or FilterConfig()
        
        # Pre-filter buffer for raw speed values
        self.raw_speed_buffer: deque = deque(maxlen=self.cfg.pre_filter_size)
        
        # Filtered data with timestamps (kept in sync)
        self.filtered_data: deque = deque(maxlen=self.cfg.min_samples_for_accel + 2)
        
        # Current state
        self.filtered_speed: float = 0.0
        self.filtered_acceleration: float = 0.0
        self.smoothed_offset: float = 0.0  # Tracked offset between raw and filtered
        self.initialized: bool = False
        
        # Prediction state
        self.last_update_time: float = None
        self.predicted_speed: float = 0.0

    def _get_pre_filtered_speed(self, speed: float) -> float:
        """
        Add new speed to buffer and return the averaged value.
        
        Uses a simple moving average of the last N samples to remove
        noise and outliers before main filtering.
        """
        self.raw_speed_buffer.append(speed)
        
        # Calculate average of buffer
        if len(self.raw_speed_buffer) == 0:
            return speed
        
        return sum(self.raw_speed_buffer) / len(self.raw_speed_buffer)

    def _predict_next_speed(self, dt: float) -> float:
        """
        Predict the next speed value based on current speed and acceleration.
        
        Uses kinematic equation: v_next = v_current + a * dt
        with acceleration decay to prevent runaway predictions.
        """
        if dt <= 0:
            return self.filtered_speed
        
        # Apply decay to acceleration to prevent over-prediction
        effective_accel = self.filtered_acceleration * self.cfg.accel_decay
        
        # Kinematic prediction
        predicted = self.filtered_speed + effective_accel * dt
        
        # Clamp to non-negative
        return max(0.0, predicted)

    def update(self, speed: float, current_time: float = None) -> Tuple[float, float]: 
        """
        Update the filter with a new speed measurement.
        
        Args:
            speed:  Current speed measurement (m/s)
            current_time: Optional timestamp (uses time.time() if not provided)
            
        Returns: 
            Tuple of (filtered_speed, filtered_acceleration)
        """
        if current_time is None:
            current_time = time.time()
        
        # --- PRE-FILTER STEP ---
        # Average the last N raw samples to remove noise
        pre_filtered_speed = self._get_pre_filtered_speed(speed)
        
        # Initialize on first update
        if not self.initialized:
            self.filtered_speed = pre_filtered_speed
            self.predicted_speed = pre_filtered_speed
            self.last_update_time = current_time
            self.filtered_data.append((current_time, pre_filtered_speed))
            self.initialized = True
            return (self.filtered_speed, 0.0)
        
        # Calculate time delta
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # --- PREDICTION STEP ---
        # Predict where we expect the speed to be based on previous trend
        self.predicted_speed = self._predict_next_speed(dt)
        
        # --- FUSION STEP ---
        # Blend prediction with measurement for the EMA base
        # During steady state (low accel), trust EMA more
        # During deceleration, trust prediction more
        accel_magnitude = abs(self.filtered_acceleration)
        
        # Adaptive prediction weight:  increase weight during high deceleration
        adaptive_weight = self.cfg.prediction_weight
        if accel_magnitude > 1.0:  # Significant decel/accel
            adaptive_weight = min(0.8, self.cfg.prediction_weight + 0.2)
        
        # Use prediction as the base for EMA instead of previous filtered value
        # This makes the filter "anticipate" changes
        ema_base = (
            adaptive_weight * self.predicted_speed +
            (1 - adaptive_weight) * self.filtered_speed
        )
        
        # Calculate raw offset (negative when braking hard = filtered is too high)
        raw_offset = pre_filtered_speed - self.filtered_speed
        
        # Smooth the offset to filter out noise
        self.smoothed_offset = (
            self.cfg.offset_alpha * raw_offset +
            (1 - self.cfg.offset_alpha) * self.smoothed_offset
        )
        
        # Apply EMA filter with prediction-adjusted base
        ema_alpha = (
            self.cfg.ema_alpha
            if self.filtered_acceleration <= 0.5
            else self.cfg.ema_accel_alpha
        )
        
        # Key change: blend toward measurement from predicted base, not old filtered value
        self.filtered_speed = (
            ema_alpha * pre_filtered_speed +
            (1 - ema_alpha) * ema_base
        )
        
        # Apply offset correction if offset exceeds threshold
        if abs(self.smoothed_offset) > self.cfg.offset_correction_threshold:
            # Calculate correction amount
            excess_offset = self.smoothed_offset - (
                self.cfg.offset_correction_threshold *
                (1 if self.smoothed_offset > 0 else -1)
            )
            correction = excess_offset * self.cfg.offset_correction_gain
            
            # Clamp correction rate
            correction = max(-self.cfg.max_correction_rate,
                           min(self.cfg.max_correction_rate, correction))
            
            # Apply correction
            self.filtered_speed += correction
            
            # Reduce tracked offset by the correction applied
            self.smoothed_offset -= correction
        
        # Clamp to non-negative
        self.filtered_speed = max(0.0, self.filtered_speed)
        
        # Store filtered result with timestamp
        self.filtered_data.append((current_time, self.filtered_speed))
        
        # Calculate acceleration (this updates self.filtered_acceleration)
        self._calculate_acceleration()
        
        return self.filtered_speed, self.filtered_acceleration

    def _calculate_acceleration(self):
        """Calculate acceleration from filtered speed history."""
        if len(self.filtered_data) < 2:
            return
        
        # Get the two most recent samples
        prev_time, prev_speed = self.filtered_data[-2]
        curr_time, curr_speed = self.filtered_data[-1]
        
        dt = curr_time - prev_time
        if dt <= 0:
            return
        
        # Calculate instantaneous acceleration
        raw_accel = (curr_speed - prev_speed) / dt
        
        # Smooth the output acceleration
        self.filtered_acceleration = (
            self.cfg.accel_smoothing_alpha * raw_accel +
            (1 - self.cfg.accel_smoothing_alpha) * self.filtered_acceleration
        )

    def reset(self):
        """Reset the filter to initial state."""
        self.raw_speed_buffer.clear()
        self.filtered_data.clear()
        self.filtered_speed = 0.0
        self.filtered_acceleration = 0.0
        self.smoothed_offset = 0.0
        self.initialized = False
        self.last_update_time = None
        self.predicted_speed = 0.0


class Position:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y, self.z - other.z)

    def tuple(self):
        return (self.x, self.y, self.z)

    def is_zero(self):
        return self.x == 0 and self.y == 0 and self.z == 0

    def __str__(self):
        return f"Position({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = y
        self.y = x
        self.z = z

    def euler(self):  # Convert to pitch, yaw, roll
        """Var yaw = atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z);
        var pitch = asin(-2.0*(q.x*q.z - q.w*q.y));
        var roll = atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z);
        """
        yaw = math.atan2(
            2.0 * (self.y * self.z + self.w * self.x),
            self.w * self.w - self.x * self.x - self.y * self.y + self.z * self.z,
        )
        pitch = math.asin(-2.0 * (self.x * self.z - self.w * self.y))
        roll = math.atan2(
            2.0 * (self.x * self.y + self.w * self.z),
            self.w * self.w + self.x * self.x - self.y * self.y - self.z * self.z,
        )

        yaw = math.degrees(yaw)
        pitch = math.degrees(pitch)
        roll = math.degrees(roll)

        return pitch, yaw, roll

    def is_zero(self):
        return self.w == 0 and self.x == 0 and self.y == 0 and self.z == 0

    def __str__(self):
        x, y, z = self.euler()
        return f"Quaternion({self.w:.2f}, {self.x:.2f}, {self.y:.2f}, {self.z:.2f}) -> (pitch {x:.2f}, yaw {y:.2f}, roll {z:.2f})"

    def __dict__(self):  # type: ignore
        euler = self.euler()
        return {
            "w": self.w,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "pitch": euler[0],
            "yaw": euler[1],
            "roll": euler[2],
        }


class Size:
    width: float
    height: float
    length: float

    def __init__(self, width: float, height: float, length: float):
        self.width = width
        self.height = height
        self.length = length

    def __str__(self):
        return f"Size({self.width:.2f}, {self.height:.2f}, {self.length:.2f})"


class Trailer:
    position: Position
    rotation: Quaternion
    size: Size
    is_tmp: bool

    def __init__(self, position: Position, rotation: Quaternion, size: Size, is_tmp: bool):
        self.position = position
        self.rotation = rotation
        self.size = size
        self.is_tmp = is_tmp
        
    def correct_position(self) -> Position:
        # Move the position back by half the length in the current direction
        yaw = self.rotation.euler()[1]
        yaw = math.radians(yaw)
        new_position = Position(self.position.x, self.position.y, self.position.z)
        new_position.x += (self.size.length / 2) * math.sin(yaw)
        new_position.z += (self.size.length / 2) * math.cos(yaw)
        return new_position

    def get_corners(
        self, offset: Position = None, correction_multiplier: float = 1
    ) -> tuple[Position, Position, Position, Position]:
        """This function will output the corners of the vehicle in the following order:
        1.Front left
        2.Front right
        3.Back right
        4.Back left
        """
        ground_middle = [self.position.x, self.position.y, self.position.z]
        if offset:
            ground_middle[0] += offset.x
            ground_middle[1] += offset.y
            ground_middle[2] += offset.z

        # Back left
        back_left = [
            ground_middle[0] - self.size.width / 2,
            ground_middle[1],
            ground_middle[2] + (self.size.length / 2 * correction_multiplier)
            if self.is_tmp
            else ground_middle[2] + (self.size.length * 0.82 * correction_multiplier),
        ]

        # Back right
        back_right = [
            ground_middle[0] + self.size.width / 2,
            ground_middle[1],
            ground_middle[2] + (self.size.length / 2 * correction_multiplier)
            if self.is_tmp
            else ground_middle[2] + (self.size.length * 0.82 * correction_multiplier),
        ]

        # Front right
        front_right = [
            ground_middle[0] + self.size.width / 2,
            ground_middle[1],
            ground_middle[2] - (self.size.length / 2 * correction_multiplier)
            if self.is_tmp
            else ground_middle[2] - (self.size.length * 0.18 * correction_multiplier),
        ]

        # Front left
        front_left = [
            ground_middle[0] - self.size.width / 2,
            ground_middle[1],
            ground_middle[2] - (self.size.length / 2 * correction_multiplier)
            if self.is_tmp
            else ground_middle[2] - (self.size.length * 0.18 * correction_multiplier),
        ]

        # Rotate the corners
        pitch, yaw, roll = self.rotation.euler()
        front_left = rotate_around_point(front_left, ground_middle, pitch, -yaw, 0)
        front_right = rotate_around_point(front_right, ground_middle, pitch, -yaw, 0)
        back_right = rotate_around_point(back_right, ground_middle, pitch, -yaw, 0)
        back_left = rotate_around_point(back_left, ground_middle, pitch, -yaw, 0)

        front_left = Position(*front_left)
        front_right = Position(*front_right)
        back_right = Position(*back_right)
        back_left = Position(*back_left)

        return front_left, front_right, back_right, back_left

    def is_zero(self):
        return self.position.is_zero() and self.rotation.is_zero()

    def __str__(self):
        return f"Trailer({self.position}, {self.rotation}, {self.size})"

    def __dict__(self):  # type: ignore
        if self.is_tmp:
            position = self.correct_position()
        else:
            position = self.position
            
        return {
            "position": position.__dict__,
            "rotation": self.rotation.__dict__(),
            "size": self.size.__dict__,
        }
    
class Vehicle:
    position: Position
    rotation: Quaternion
    size: Size
    speed: float
    acceleration: float
    trailer_count: int
    id: int
    trailers: list[Trailer]

    is_tmp: bool
    is_trailer: bool
    time: float = 0.0

    last_location: Position = Position(0, 0, 0)
    last_rotation: Quaternion = Quaternion(0, 0, 0, 0)
    angular_velocity: float = 0.0  # degrees per second around yaw

    # Kalman filter for speed/acceleration estimation
    speed_filter: SpeedFilter

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
    ):
        self.position = position
        self.rotation = rotation
        self.size = size
        self.speed_raw = speed
        self.is_tmp = is_tmp
        self.speed = speed
        self.acceleration = acceleration
        self.trailer_count = trailer_count
        self.trailers = trailers
        self.id = id
        self.is_trailer = is_trailer

        self.time = time.time()

        if self.is_tmp:
            # Initialize speed filter for new tmp vehicles
            # Don't call update_from_last(self) - it causes early return and leaves speed at 0
            # main.py will call update_from_last(previous_vehicle) if a previous vehicle exists
            self.speed_filter = SpeedFilter()
            # Initialize last_location and last_rotation to current values for first update
            self.last_location = Position(self.position.x, self.position.y, self.position.z)
            self.last_rotation = Quaternion(self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z)

    def update_from_last(self, vehicle):
        # Copy filter FIRST, before any other logic
        if hasattr(vehicle, 'speed_filter'):
            self.speed_filter = vehicle.speed_filter
        else:
            self.speed_filter = SpeedFilter()
        
        time_diff = time.time() - vehicle.time
        
        if time_diff < location_update_frequency:
            self.time = vehicle.time
            self.last_location = vehicle.last_location
            self.last_rotation = vehicle.last_rotation
            self.angular_velocity = vehicle.angular_velocity
            if abs(self.angular_velocity) > 90:
                self.angular_velocity = 0

            if self.is_tmp:
                # Use the previous vehicle's filtered speed as speed_raw to maintain continuity
                self.speed_raw = vehicle.speed
            else:
                # For non-tmp vehicles, use speed_raw if available, otherwise use speed
                if hasattr(vehicle, 'speed_raw'):
                    self.speed_raw = vehicle.speed_raw
                else:
                    self.speed_raw = vehicle.speed
            
            # IMPORTANT: Still update the filter even in early return to maintain speed/acceleration
            speed, accel = self.speed_filter.update(self.speed_raw, self.time)
            self.speed = speed
            self.acceleration = accel
            return

        self.time = time.time()
        self.last_location = self.position
        self.last_rotation = self.rotation

        last_yaw = vehicle.last_rotation.euler()[1]
        current_yaw = self.rotation.euler()[1]
        yaw_diff = current_yaw - last_yaw
        self.angular_velocity = yaw_diff / time_diff / 2

        if abs(self.angular_velocity) > 90:
            self.angular_velocity = 0

        if self.is_tmp:
            last_position = vehicle.last_location
            distance = math.sqrt(
                (self.position.x - last_position.x) ** 2
                + (self.position.y - last_position.y) ** 2
                + (self.position.z - last_position.z) ** 2
            )
            if distance > 30:
                self.speed_raw = 0
                self.acceleration = 0
                self.speed_filter.reset()
            elif distance > 0.025:
                self.speed_raw = distance / time_diff
            else:
                self.speed_raw = 0
        else:
            if hasattr(vehicle, 'speed_raw'):
                self.speed_raw = vehicle.speed_raw
            else:
                self.speed_raw = vehicle.speed

        speed, accel = self.speed_filter.update(self.speed_raw, self.time)
        self.speed = speed
        self.acceleration = accel

        Visualizer.push_data(self.id, self.speed, self.acceleration, self.speed_raw, 0.0)

    def is_zero(self):
        return self.position.is_zero() and self.rotation.is_zero()

    def __str__(self):
        return f"Vehicle({self.position}, {self.rotation}, {self.size}, {self.speed_raw:.2f}, {self.acceleration:.2f}, {self.trailer_count}, {self.trailers})"

    def get_corners(
        self, offset: Position = None, correction_multiplier: float = 1
    ) -> tuple[Position, Position, Position, Position]:
        """This function will output the corners of the vehicle in the following order:
        1.Front left
        2.Front right
        3.Back right
        4.Back left
        """
        ground_middle = [self.position.x, self.position.y, self.position.z]
        if offset:
            ground_middle[0] += offset.x
            ground_middle[1] += offset.y
            ground_middle[2] += offset.z

        # Back left
        back_left = [
            ground_middle[0] - self.size.width / 2,
            ground_middle[1],
            ground_middle[2] + (self.size.length / 2 * correction_multiplier)
            if self.is_tmp
            else ground_middle[2] + (self.size.length * 0.82 * correction_multiplier),
        ]

        # Back right
        back_right = [
            ground_middle[0] + self.size.width / 2,
            ground_middle[1],
            ground_middle[2] + (self.size.length / 2 * correction_multiplier)
            if self.is_tmp
            else ground_middle[2] + (self.size.length * 0.82 * correction_multiplier),
        ]

        # Front right
        front_right = [
            ground_middle[0] + self.size.width / 2,
            ground_middle[1],
            ground_middle[2] - (self.size.length / 2 * correction_multiplier)
            if self.is_tmp
            else ground_middle[2] - (self.size.length * 0.18 * correction_multiplier),
        ]

        # Front left
        front_left = [
            ground_middle[0] - self.size.width / 2,
            ground_middle[1],
            ground_middle[2] - (self.size.length / 2 * correction_multiplier)
            if self.is_tmp
            else ground_middle[2] - (self.size.length * 0.18 * correction_multiplier),
        ]

        # Rotate the corners
        pitch, yaw, roll = self.rotation.euler()
        front_left = rotate_around_point(front_left, ground_middle, pitch, -yaw, 0)
        front_right = rotate_around_point(front_right, ground_middle, pitch, -yaw, 0)
        back_right = rotate_around_point(back_right, ground_middle, pitch, -yaw, 0)
        back_left = rotate_around_point(back_left, ground_middle, pitch, -yaw, 0)

        front_left = Position(*front_left)
        front_right = Position(*front_right)
        back_right = Position(*back_right)
        back_left = Position(*back_left)

        return front_left, front_right, back_right, back_left

    def get_corrected_position(self) -> Position | None:
        front_left, front_right, back_right, back_left = self.get_corners()
        center_x = (front_left.x + front_right.x + back_right.x + back_left.x) / 4
        center_y = (front_left.y + front_right.y + back_right.y + back_left.y) / 4
        center_z = (front_left.z + front_right.z + back_right.z + back_left.z) / 4
        return Position(center_x, center_y, center_z)

    def get_position_in(self, seconds: float) -> Position | None:
        distance = self.speed * seconds
        if distance == 0:
            return Position(self.position.x, self.position.y, self.position.z)

        # x and z are the ground plane, don't care about y
        pitch, yaw, roll = self.rotation.euler()
        yaw = math.radians(yaw)

        # adjust based on angular velocity, we assume
        # that the vehicle is slowly tapering out so we apply
        # an exponential decay.
        angular_velocity = math.radians(self.angular_velocity)
        if angular_velocity != 0:
            decay_rate = 0.25
            total_decay = (1 - math.exp(-decay_rate * seconds)) / decay_rate
            yaw += angular_velocity * total_decay
        else:
            yaw += angular_velocity * seconds

        # eventual new position
        x = self.position.x - distance * math.sin(yaw)
        y = self.position.y
        z = self.position.z - distance * math.cos(yaw)

        return Position(x, y, z)

    def get_path_for(self, seconds: float) -> Position | None:
        points_per_second = 10
        points = []
        for i in range(0, int(seconds * points_per_second)):
            point = self.get_position_in(i / points_per_second)
            if point:
                points.append(point)

        return points

    def __dict__(self):  # type: ignore
        return {
            "position": self.position.__dict__,
            "rotation": self.rotation.__dict__(),
            "size": self.size.__dict__,
            "speed": self.speed,
            "acceleration": self.acceleration,
            "trailer_count": self.trailer_count,
            "trailers": [trailer.__dict__() for trailer in self.trailers],
            "id": self.id,
            "is_tmp": self.is_tmp,
            "is_trailer": self.is_trailer,
        }