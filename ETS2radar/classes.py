import math
import queue
import time
from collections import deque
import threading
from turtle import pos
import numpy as np

# testing
#from ETS2radar.radar_visualizer import RadarVisualizer

#Visualizer = RadarVisualizer()
#threading.Thread(target=Visualizer.start, daemon=True).start()

SPEED_CALCULATION_WINDOW = 1 # seconds - time window for calculating maxlen of position history
POSITION_QUEUE_MAXLEN = 20 # maximum number of positions to store in queue for averaging

# TODO: Switch __dict__ to __iter__ and dict() for typing support.
# TODO: f = Class() -> dict(f) instead of f.__dict__()

def rotate_around_point(point, center, pitch, yaw, roll):
    """
    Rotate a point around a center point by the given pitch, yaw, and roll angles (in degrees).
    
    Parameters:
    - point: [x, y, z] coordinates of the point to rotate
    - center: [x, y, z] coordinates of the center of rotation
    - pitch: rotation around X-axis (in degrees)
    - yaw: rotation around Y-axis (in degrees)
    - roll: rotation around Z-axis (in degrees)
    
    Returns:
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
    return [
        x + center[0],
        y + center[1],
        z + center[2]
    ]
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FilterConfig:
    base_speed_kmh: float = 40.0
    buffer_len: int = 20
    buffer_window_s: float = 0.25

    min_dt: float = 1e-3
    max_dt: float = 1.0
    snap_distance: float = 25.0

    ema_alpha_min: float = 0.05
    ema_alpha_max: float = 0.8

    q_scale_min: float = 0.05
    q_scale_lag_multiplier: float = 0.1

    stationary_speed_kmh: float = 0.5
    stationary_frames_threshold: int = 2

    mahalanobis_gate: float = 5.0
    r_inflate_when_gated: float = 4.0

class KalmanFilter:
    """
    Kalman filter for estimating position, velocity, and acceleration.
    State: [x, y, z, vx, vy, vz, ax, ay, az]

    - Adaptive Q based on speed
    - EMA smoothing of measurements (speed-based alpha)
    - Median buffer for jitter
    - Mahalanobis gating to down-weight outliers
    """

    def __init__(self, config: FilterConfig = FilterConfig()):
        self.cfg = config

        self.state = np.zeros(9)
        self.P = np.eye(9) * 1000.0

        self.Q_base = np.eye(9)
        self.Q_base[0:3, 0:3] *= 0.001
        self.Q_base[3:6, 3:6] *= 0.0005
        self.Q_base[6:9, 6:9] *= 0.7

        self.R_base = np.eye(3) * 0.5

        self.H = np.zeros((3, 9))
        self.H[0:3, 0:3] = np.eye(3)
        self.H_T = self.H.T

        self.position_buffer = deque(maxlen=self.cfg.buffer_len)

        self.ema_position: Optional[np.ndarray] = None
        self.initialized = False
        self.last_time: Optional[float] = None

        self.stationary_frames = 0
        self.last_filtered_speed = 0.0
        self.prev_last_filtered_speed = 0.0

        self.last_raw_velocity_vec = np.zeros(3)
        self.last_raw_velocity: float = 0.0
        self.last_smoothed_measurement: Optional[np.ndarray] = None

        # Store last raw measurement for lag detection
        self.last_raw_position: Optional[np.ndarray] = None
        self.last_raw_time: Optional[float] = None

    # ---------- Helper calculations ----------
    def _calculate_raw_speed(self, current_time: float) -> float:
        """Calculate raw speed from position buffer (for other uses)."""
        if len(self.position_buffer) < 2:
            return 0.0

        cutoff_time = current_time - self.cfg.buffer_window_s
        valid_samples = [(p, t) for p, t in self.position_buffer if t >= cutoff_time] or list(self.position_buffer)
        if len(valid_samples) < 2:
            return 0.0

        pos1, t1 = valid_samples[0]
        pos2, t2 = valid_samples[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0

        distance = np.linalg.norm(pos2 - pos1)
        return distance / dt

    def _calculate_raw_velocity_and_accel(self, current_time: float) -> Tuple[np.ndarray, float, float]:
        """Calculate velocity and accel from buffer (for state initialization)."""
        if len(self.position_buffer) < 2:
            return np.zeros(3), 0.0, 0.0

        cutoff_time = current_time - self.cfg.buffer_window_s
        valid_samples = [(p, t) for p, t in self.position_buffer if t >= cutoff_time] or list(self.position_buffer)
        if len(valid_samples) < 2:
            return np.zeros(3), 0.0, 0.0

        pos1, t1 = valid_samples[0]
        pos2, t2 = valid_samples[-1]
        dt = t2 - t1
        if dt <= 0:
            return np.zeros(3), 0.0, 0.0

        raw_velocity = (pos2 - pos1) / dt
        raw_speed = np.linalg.norm(raw_velocity)

        raw_accel_vec = (raw_velocity - self.last_raw_velocity_vec) / dt
        raw_signed_accel = np.dot(raw_accel_vec, raw_velocity / raw_speed) if raw_speed > 0.3 else 0.0

        return raw_velocity, raw_speed, raw_signed_accel

    def _apply_ema(self, measurement: np.ndarray, raw_speed_ms: float, 
                   current_time: float) -> np.ndarray:
        """Apply EMA smoothing with lag-aware multiplier."""
        speed_kmh = raw_speed_ms * 3.6
        alpha = speed_kmh / self.cfg.base_speed_kmh
        alpha = np.clip(alpha, self.cfg.ema_alpha_min, self.cfg.ema_alpha_max)

        # Ensure alpha doesn't go below minimum
        alpha = max(alpha, self.cfg.ema_alpha_min * 0.5)

        if self.ema_position is None:
            self.ema_position = measurement.copy()
            return measurement

        self.ema_position = alpha * measurement + (1 - alpha) * self.ema_position
        return self.ema_position

    def _get_buffered_position(self, current_time: float) -> Optional[np.ndarray]:
        if not self.position_buffer:
            return None
        if len(self.position_buffer) == 1:
            return self.position_buffer[-1][0]

        cutoff_time = current_time - self.cfg.buffer_window_s
        valid_positions = [p for p, t in self.position_buffer if t >= cutoff_time] or [p for p, _ in self.position_buffer]
        if len(valid_positions) == 1:
            return valid_positions[0]

        return np.median(np.asarray(valid_positions), axis=0)

    def _mahalanobis_distance(self, residual: np.ndarray, S: np.ndarray) -> float:
        try:
            x = np.linalg.solve(S, residual)
            return float(np.sqrt(residual.T @ x))
        except np.linalg.LinAlgError:
            return np.inf

    def _extract_outputs_from_state(self) -> Tuple[float, float]:
        v = self.state[3:6]
        speed = float(np.linalg.norm(v))
        if speed <= 0.3:
            return 0.0, 0.0
        direction = v / speed
        accel = float(np.dot(self.state[6:9], direction))
        return speed, accel

    # ---------- Kalman steps ----------
    def predict(self, dt: float, raw_speed_ms: float):
        if dt <= self.cfg.min_dt:
            return

        speed_kmh = self.last_filtered_speed * 3.6
        stationary = (raw_speed_ms * 3.6) < self.cfg.stationary_speed_kmh

        if stationary:
            self.stationary_frames += 1
            if self.stationary_frames > self.cfg.stationary_frames_threshold:
                self.state[3:9] = 0.0
                return
        else:
            self.stationary_frames = 0

        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
        F[3:6, 6:9] = np.eye(3) * dt

        self.state = F @ self.state

        speed_ratio = max(speed_kmh, 0.0) / max(self.cfg.base_speed_kmh, 1e-6)
        q_scale = np.exp(1.0 - speed_ratio)
        q_scale = max(self.cfg.q_scale_min, q_scale)
        Q = self.Q_base * q_scale

        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: np.ndarray, raw_speed_ms: float):
        residual = measurement - (self.H @ self.state)
        R = self.R_base.copy()

        S = self.H @ self.P @ self.H_T + R
        md = self._mahalanobis_distance(residual, S)
        if md > self.cfg.mahalanobis_gate:
            R *= self.cfg.r_inflate_when_gated
            S = self.H @ self.P @ self.H_T + R

        try:
            K = self.P @ self.H_T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S += np.eye(3) * 1e-6
            K = self.P @ self.H_T @ np.linalg.inv(S)

        self.state = self.state + K @ residual
        I = np.eye(9)
        self.P = (I - K @ self.H) @ self.P

    # ---------- Public API ----------
    def process(self, position, current_time: float) -> Tuple[float, float]:
        measurement = np.array([position.x, position.y, position.z], dtype=float)
        dt = (current_time - self.last_time) if self.initialized else 0.1
        
        # Update raw position tracking
        self.last_raw_position = measurement.copy()
        self.last_raw_time = current_time

        # Add to buffer for median filtering
        self.position_buffer.append((measurement, current_time))

        # Get buffered velocity for state updates
        buffered_velocity_vec, buffered_speed_ms, _ = self._calculate_raw_velocity_and_accel(current_time)
        self.last_raw_velocity_vec = buffered_velocity_vec
        self.last_raw_velocity = buffered_speed_ms

        buffered_measurement = self._get_buffered_position(current_time)
        if buffered_measurement is None:
            return 0.0, 0.0

        # Apply EMA with lag-aware alpha
        smoothed_measurement = self._apply_ema(buffered_measurement, buffered_speed_ms, current_time)
        self.last_smoothed_measurement = smoothed_measurement

        if not self.initialized:
            self.state[0:3] = smoothed_measurement
            self.state[3:6] = buffered_velocity_vec
            self.state[6:9] = 0.0
            self.initialized = True
            self.last_time = current_time
            speed, accel = self._extract_outputs_from_state()
            self.last_filtered_speed = speed
            return speed, accel

        if dt <= self.cfg.min_dt or dt > self.cfg.max_dt:
            self.last_time = current_time
            self.state[0:3] = smoothed_measurement
            self.state[3:6] = buffered_velocity_vec
            self.state[6:9] = 0.0
            self.P = np.eye(9) * 500.0
            speed, accel = self._extract_outputs_from_state()
            self.last_filtered_speed = speed
            return speed, accel

        # Normal predict/update
        self.predict(dt, buffered_speed_ms)

        predicted_pos = self.state[0:3]
        distance = np.linalg.norm(smoothed_measurement - predicted_pos)
        if distance > self.cfg.snap_distance:
            self.state[0:3] = smoothed_measurement
            self.state[3:6] = buffered_velocity_vec
            self.state[6:9] = 0.0
            self.P = np.eye(9) * 1000.0
            self.position_buffer.clear()
            self.position_buffer.append((measurement, current_time))
            self.ema_position = None
            self.last_time = current_time
            speed, accel = self._extract_outputs_from_state()
            self.last_filtered_speed = speed
            return speed, accel

        self.update(smoothed_measurement, buffered_speed_ms)
        self.last_time = current_time

        filtered_speed, filtered_signed_accel = self._extract_outputs_from_state()
        self.prev_last_filtered_speed = self.last_filtered_speed
        self.last_filtered_speed = filtered_speed

        return filtered_speed, filtered_signed_accel

    def reset(self):
        self.__init__(self.cfg)
    
    @property
    def is_lag_recovery_active(self) -> bool:
        """Check if currently recovering from lag (for debugging/UI)."""
        return False
    
    @property
    def ema_multiplier(self) -> float:
        """Current EMA multiplier (for debugging/UI)."""
        return 1.0

class Position():
    x: float
    y: float
    z: float
    
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        
    def is_zero(self):
        return self.x == 0 and self.y == 0 and self.z == 0
        
    def __str__(self):
        return f"Position({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
class Quaternion():
    w: float
    x: float
    y: float
    z: float
    
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = y
        self.y = x
        self.z = z
        
    def euler(self): # Convert to pitch, yaw, roll
        """
        var yaw = atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z);
        var pitch = asin(-2.0*(q.x*q.z - q.w*q.y));
        var roll = atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z);
        """
        yaw = math.atan2(2.0*(self.y*self.z + self.w*self.x), self.w*self.w - self.x*self.x - self.y*self.y + self.z*self.z)
        pitch = math.asin(-2.0*(self.x*self.z - self.w*self.y))
        roll = math.atan2(2.0*(self.x*self.y + self.w*self.z), self.w*self.w + self.x*self.x - self.y*self.y - self.z*self.z)
        
        yaw = math.degrees(yaw)
        pitch = math.degrees(pitch)
        roll = math.degrees(roll)
        
        return pitch, yaw, roll 
        
    def is_zero(self):
        return self.w == 0 and self.x == 0 and self.y == 0 and self.z == 0
        
    def __str__(self):
        x, y, z = self.euler()
        return f"Quaternion({self.w:.2f}, {self.x:.2f}, {self.y:.2f}, {self.z:.2f}) -> (pitch {x:.2f}, yaw {y:.2f}, roll {z:.2f})"
    
    def __dict__(self): # type: ignore
        euler = self.euler()
        return {
            "w": self.w,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "pitch": euler[0],
            "yaw": euler[1],
            "roll": euler[2]
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
    
    def __init__(self, position: Position, rotation: Quaternion, size: Size):
        self.position = position
        self.rotation = rotation
        self.size = size
        
    def is_zero(self):
        return self.position.is_zero() and self.rotation.is_zero()
        
    def __str__(self):
        return f"Trailer({self.position}, {self.rotation}, {self.size})"
    
    def __dict__(self): # type: ignore
        return {
            "position": self.position.__dict__,
            "rotation": self.rotation.__dict__(),
            "size": self.size.__dict__
        }

    def get_corners(self):
        """
        This function will output the corners of the vehicle in the following order:
        1. Front left
        2. Front right
        3. Back right
        4. Back left
        """
        ground_middle = [
            self.position.x,
            self.position.y,
            self.position.z
        ]
        
        # Back left
        back_left = [
            ground_middle[0] - self.size.width/2,
            ground_middle[1],
            ground_middle[2] + self.size.length/2
        ]
        
        # Back right
        back_right = [
            ground_middle[0] + self.size.width/2,
            ground_middle[1],
            ground_middle[2] + self.size.length/2
        ]
        
        # Front right
        front_right = [
            ground_middle[0] + self.size.width/2,
            ground_middle[1],
            ground_middle[2] - self.size.length/2
        ]
        
        # Front left
        front_left = [
            ground_middle[0] - self.size.width/2,
            ground_middle[1],
            ground_middle[2] - self.size.length/2
        ]
        
        # Rotate the corners
        pitch, yaw, roll = self.rotation.euler()
        front_left = rotate_around_point(front_left, ground_middle, pitch, -yaw, 0)
        front_right = rotate_around_point(front_right, ground_middle, pitch, -yaw, 0)
        back_right = rotate_around_point(back_right, ground_middle, pitch, -yaw, 0)
        back_left = rotate_around_point(back_left, ground_middle, pitch, -yaw, 0)
        
        return front_left, front_right, back_right, back_left
    
class Vehicle:
    position: Position
    rotation: Quaternion
    size: Size
    speed: float
    raw_speed: float
    acceleration: float
    raw_accel: float
    trailer_count: int
    id: int
    trailers: list[Trailer]
    
    is_tmp: bool
    is_trailer: bool
    time: float = 0.0
    position_queue: deque
    
    # Kalman filter for speed/acceleration estimation
    kalman_filter: KalmanFilter
    
    def __init__(self, position: Position, rotation: Quaternion, size: Size, 
                speed: float, acceleration: float, 
                trailer_count: int, trailers: list[Trailer],
                id: int, is_tmp: bool, is_trailer: bool):
        self.position = position
        self.rotation = rotation
        self.size = size
        self.speed = speed
        self.raw_speed = speed
        self.acceleration = acceleration
        self.raw_accel = acceleration
        self.trailer_count = trailer_count
        self.trailers = trailers
        self.id = id
        self.is_tmp = is_tmp
        self.is_trailer = is_trailer
        
        self.high_res = True

        self.time = time.time()
        self.kalman_filter = KalmanFilter()
        self.position_queue = deque(maxlen=POSITION_QUEUE_MAXLEN)
        
        # Add initial position to queue for is_tmp vehicles
        if self.is_tmp and self.high_res:
            self.add_position_to_queue()

    def add_position_to_queue(self):
        """Add the vehicle's current position to the queue for averaging. Only used for is_tmp vehicles."""
        if self.is_tmp and self.high_res:
            self.position_queue.append(self.position)
    
    def _get_averaged_position(self) -> Position | None:
        """Calculate the average of all positions in the queue.
        
        Returns:
            Position object with averaged x, y, z coordinates, or None if queue is empty.
        """
        if len(self.position_queue) == 0:
            return None
        elif len(self.position_queue) == 1:
            return self.position_queue[0]
        
        # Calculate mean of x, y, z coordinates from all Position objects in queue
        count = len(self.position_queue)
        sum_x = sum(pos.x for pos in self.position_queue)
        sum_y = sum(pos.y for pos in self.position_queue)
        sum_z = sum(pos.z for pos in self.position_queue)
        pos = Position(sum_x / count, sum_y / count, sum_z / count)
        self.position_queue.clear()
        
        return pos
    
    def update_from_last(self, vehicle):
        """Update this vehicle's calculated properties from the previous frame"""
        # Calculate raw speed and acceleration from position delta
        self.position_queue = vehicle.position_queue.copy()
        self._calculate_raw_speed_and_acceleration(vehicle)
        
        # Only calculate Kalman-filtered speed/acceleration for TruckersMP multiplayer vehicles
        if self.is_tmp and self.high_res:
            # Add current position to queue
            self.add_position_to_queue()
            
            if hasattr(vehicle, 'kalman_filter'):
                self.kalman_filter = vehicle.kalman_filter
            else:
                self.kalman_filter = KalmanFilter()
            
            # Get averaged position from queue
            averaged_position = self._get_averaged_position()
            # Fall back to current position if queue is empty
            position_to_use = averaged_position if averaged_position is not None else self.position
            
            current_time = time.time()
            self._calculate_speed_and_acceleration(current_time, position_to_use)
        elif self.is_tmp:
            # For non-high-res TMP vehicles, just copy last speed/accel
            self.speed = self.raw_speed
            self.acceleration = self.raw_accel

        #Visualizer.push_data(self.id, self.speed, self.acceleration, self.kalman_filter.last_raw_velocity, 0.0)
        
    def _calculate_raw_speed_and_acceleration(self, prev_vehicle):
        """Calculate raw speed and acceleration from position delta"""
        dt = self.time - prev_vehicle.time
        
        if dt <= 0 or dt > 1.0:
            self.raw_speed = 0.0
            self.raw_accel = 0.0
            return
        
        cur_pos = self.position
        prev_pos = prev_vehicle.position

        dx = cur_pos.x - prev_pos.x
        dy = cur_pos.y - prev_pos.y
        dz = cur_pos.z - prev_pos.z
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        self.raw_speed = distance / dt
        
        self.raw_accel = 0.0
        
    def _calculate_speed_and_acceleration(self, current_time, position: Position | None = None):
        """Calculate speed and acceleration using Kalman filter
        
        Args:
            current_time: Current timestamp
            position: Optional Position to use instead of self.position (for averaged positions)
        """
        position_to_use = position if position is not None else self.position
        speed, accel = self.kalman_filter.process(position_to_use, current_time)
        self.speed = speed
        self.acceleration = accel
        
    def is_zero(self):
        return self.position.is_zero() and self.rotation.is_zero()
        
    def __str__(self):
        return f"Vehicle({self.position}, {self.rotation}, {self.size}, {self.speed:.2f}, {self.acceleration:.2f}, {self.trailer_count}, {self.trailers})"

    def get_corners(self):
        """
        This function will output the corners of the vehicle in the following order:
        1. Front left
        2. Front right
        3. Back right
        4. Back left
        """
        ground_middle = [
            self.position.x,
            self.position.y,
            self.position.z
        ]
        
        # Back left
        back_left = [
            ground_middle[0] - self.size.width/2,
            ground_middle[1],
            ground_middle[2] + self.size.length/2
        ]
        
        # Back right
        back_right = [
            ground_middle[0] + self.size.width/2,
            ground_middle[1],
            ground_middle[2] + self.size.length/2
        ]
        
        # Front right
        front_right = [
            ground_middle[0] + self.size.width/2,
            ground_middle[1],
            ground_middle[2] - self.size.length/2
        ]
        
        # Front left
        front_left = [
            ground_middle[0] - self.size.width/2,
            ground_middle[1],
            ground_middle[2] - self.size.length/2
        ]
        
        # Rotate the corners
        pitch, yaw, roll = self.rotation.euler()
        front_left = rotate_around_point(front_left, ground_middle, pitch, -yaw, 0)
        front_right = rotate_around_point(front_right, ground_middle, pitch, -yaw, 0)
        back_right = rotate_around_point(back_right, ground_middle, pitch, -yaw, 0)
        back_left = rotate_around_point(back_left, ground_middle, pitch, -yaw, 0)
        
        return front_left, front_right, back_right, back_left

    def __dict__(self): # type: ignore
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