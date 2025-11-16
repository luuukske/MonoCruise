import math
import time
from collections import deque
import threading
import numpy as np

# testing
from ETS2radar.radar_visualizer import RadarVisualizer

Visualizer = RadarVisualizer()
threading.Thread(target=Visualizer.start, daemon=True).start()

SPEED_CALCULATION_WINDOW = 1 # seconds - time window for calculating maxlen of position history

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

from collections import deque
import numpy as np

import numpy as np
from collections import deque

class KalmanFilter:
    """
    Kalman filter for estimating position, velocity (speed), and acceleration.
    State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    Uses historical position buffer to handle multiplayer lag and jitter.
    Adapts filtering strength based on raw speed calculated from buffered measurements.
    """
    def __init__(self):
        self.state = np.zeros(9)
        self.P = np.eye(9) * 1000
        
        self.Q_base = np.eye(9)
        self.Q_base[0:3, 0:3] *= 0.0
        self.Q_base[3:6, 3:6] *= 0.15
        self.Q_base[6:9, 6:9] *= 2.5
        
        self.R_base = np.eye(3) * 0.5
        
        self.H = np.zeros((3, 9))
        self.H[0:3, 0:3] = np.eye(3)
        
        self.position_buffer = deque(maxlen=10)
        self.buffer_window = 0.4
        
        self.ema_position = None
        self.base_speed_kmh = 60.0
        
        self.initialized = False
        self.last_time = None
        self.stationary_frames = 0
        
    def _calculate_raw_speed(self, current_time):
        if len(self.position_buffer) < 2:
            return 0.0
        
        cutoff_time = current_time - self.buffer_window
        valid_samples = [(pos, t) for pos, t in self.position_buffer if t >= cutoff_time]
        
        if len(valid_samples) < 2:
            valid_samples = list(self.position_buffer)
        
        if len(valid_samples) < 2:
            return 0.0
        
        pos1, t1 = valid_samples[0]
        pos2, t2 = valid_samples[-1]
        
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        
        distance = np.linalg.norm(pos2 - pos1)
        speed_ms = distance / dt
        
        return speed_ms
        
    def predict(self, dt, raw_speed_ms):
        if dt <= 0:
            return
        
        speed_kmh = raw_speed_ms * 3.6
        
        if speed_kmh < 0.5:
            self.stationary_frames += 1
            if self.stationary_frames > 3:
                self.state[3:9] = 0
                return
        else:
            self.stationary_frames = 0
            
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
        F[3:6, 6:9] = np.eye(3) * dt
        
        self.state = F @ self.state
        
        speed_factor = max(0.0 + (speed_kmh / self.base_speed_kmh), 0.1)
        Q = self.Q_base * speed_factor
        
        self.P = F @ self.P @ F.T + Q
        
    def update(self, measurement, raw_speed_ms, is_lag_spike):
        y = measurement - (self.H @ self.state)
        
        speed_kmh = raw_speed_ms * 3.6
        speed_factor = max(0.0 + (speed_kmh / self.base_speed_kmh), 0.1)
        R = self.R_base * speed_factor

        if is_lag_spike:
            R *= 10.0  # Increase measurement noise to distrust the measurement
        
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        
        I = np.eye(9)
        self.P = (I - K @ self.H) @ self.P
    
    def _apply_ema(self, measurement, raw_speed_ms):
        speed_kmh = raw_speed_ms * 3.6
        ema_alpha = speed_kmh / self.base_speed_kmh
        ema_alpha = np.clip(ema_alpha, 0.1, 1.0)
        
        if self.ema_position is None:
            self.ema_position = measurement.copy()
            return measurement
        
        self.ema_position = ema_alpha * measurement + (1 - ema_alpha) * self.ema_position
        return self.ema_position
    
    def _get_buffered_position(self, current_time):
        if len(self.position_buffer) < 2:
            return self.position_buffer[-1][0] if self.position_buffer else None
        
        cutoff_time = current_time - self.buffer_window
        valid_positions = [pos for pos, t in self.position_buffer if t >= cutoff_time]
        
        if len(valid_positions) == 0:
            valid_positions = [pos for pos, t in self.position_buffer]
        
        if len(valid_positions) == 1:
            return valid_positions[0]
        
        positions_array = np.array(valid_positions)
        return np.median(positions_array, axis=0)
    
    def _detect_lag_spike(self, measurement, dt):
        if not self.initialized or dt <= 0:
            return False

        # Estimate velocity from measurement change
        measured_velocity = (measurement - self.state[0:3]) / dt
        
        # Estimate acceleration
        measured_acceleration = (measured_velocity - self.state[3:6]) / dt

        # Check for huge acceleration/deceleration (e.g., > 1g)
        if np.linalg.norm(measured_acceleration) > 10:
            return True

        return False
        
    def process(self, position, current_time):
        measurement = np.array([position.x, position.y, position.z])
        
        self.position_buffer.append((measurement, current_time))
        
        raw_speed_ms = self._calculate_raw_speed(current_time)
        
        buffered_measurement = self._get_buffered_position(current_time)
        if buffered_measurement is None:
            return 0.0, 0.0

        smoothed_measurement = self._apply_ema(buffered_measurement, raw_speed_ms)
        
        if not self.initialized:
            self.state[0:3] = smoothed_measurement
            self.state[3:9] = 0
            self.initialized = True
            self.last_time = current_time
            return 0.0, 0.0
        
        dt = current_time - self.last_time
        if dt <= 0 or dt > 1.0:
            self.last_time = current_time
            # Re-initialize state if time gap is too large
            self.state[0:3] = smoothed_measurement
            self.state[3:9] = 0
            return 0.0, 0.0
        
        self.predict(dt, raw_speed_ms)
        
        predicted_pos = self.state[0:3]
        distance = np.linalg.norm(smoothed_measurement - predicted_pos)
        if distance > 25.0: # Reset if measurement is too far from prediction
            self.state[0:3] = smoothed_measurement
            self.state[3:9] = 0
            self.P = np.eye(9) * 1000
            self.position_buffer.clear()
            self.position_buffer.append((measurement, current_time))
            self.ema_position = None
            self.last_time = current_time
            return 0.0, 0.0
            
        is_lag_spike = self._detect_lag_spike(smoothed_measurement, dt)

        self.update(smoothed_measurement, raw_speed_ms, is_lag_spike)
        self.last_time = current_time
        
        velocity = self.state[3:6]
        speed = np.linalg.norm(velocity)
        
        # If speed is very low, consider it zero to prevent micro-movements
        if speed <= 0.3:
            speed = 0.0
            self.state[3:6] = 0 # Also zero out velocity state
            signed_accel = 0.0
        else:
            velocity_direction = velocity / speed
            acceleration_3d = self.state[6:9]
            signed_accel = np.dot(acceleration_3d, velocity_direction)
        
        return speed, signed_accel
        
    def reset(self):
        self.state = np.zeros(9)
        self.P = np.eye(9) * 1000
        self.ema_position = None
        self.initialized = False
        self.last_time = None
        self.stationary_frames = 0
        self.position_buffer.clear()

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
        return f"Position({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
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

        self.time = time.time()
        self.kalman_filter = KalmanFilter()

    def update_from_last(self, vehicle):
        """Update this vehicle's calculated properties from the previous frame"""
        # Calculate raw speed and acceleration from position delta
        self._calculate_raw_speed_and_acceleration(vehicle)
        
        # Only calculate Kalman-filtered speed/acceleration for TruckersMP multiplayer vehicles
        if self.is_tmp:
            if hasattr(vehicle, 'kalman_filter'):
                self.kalman_filter = vehicle.kalman_filter
            else:
                self.kalman_filter = KalmanFilter()
            
            current_time = time.time()
            self._calculate_speed_and_acceleration(current_time)

        Visualizer.push_data(self.id, self.speed, self.acceleration, self.raw_speed, self.raw_accel)
        
    def _calculate_raw_speed_and_acceleration(self, prev_vehicle):
        """Calculate raw speed and acceleration from position delta"""
        dt = self.time - prev_vehicle.time
        
        if dt <= 0 or dt > 1.0:
            self.raw_speed = 0.0
            self.raw_accel = 0.0
            return
        
        dx = self.position.x - prev_vehicle.position.x
        dy = self.position.y - prev_vehicle.position.y
        dz = self.position.z - prev_vehicle.position.z
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        self.raw_speed = distance / dt
        
        self.raw_accel = 0.0
        
    def _calculate_speed_and_acceleration(self, current_time):
        """Calculate speed and acceleration using Kalman filter"""
        speed, accel = self.kalman_filter.process(self.position, current_time)
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