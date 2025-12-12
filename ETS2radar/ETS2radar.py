from typing import List, Tuple, Dict, Optional, Any, Deque
import cv2
import numpy as np
import numpy.typing as npt
import truck_telemetry
from shapely.geometry import Polygon, Point

from ETS2radar.classes import Vehicle
from ETS2radar.main import Module

import collections
import time
import threading

# Type aliases for clarity
Coordinate = Tuple[float, float]
ScreenCoordinate = Tuple[int, int]
VehicleData = Tuple[int, float, float]  # (vehicle_id, distance, speed)
RGBColor = Tuple[int, int, int]
Corner = Tuple[float, float, float]  # (x, y, z)
ego_steering = collections.deque(maxlen=10)

# Constants
MAX_DISTANCE: int = 150  # meters
REFRESH_INTERVAL: float = 0.1  # only used when running run() loop
FOV_ANGLE: int = 25  # degrees (half-angle of cone)
POSITION_SNAP: float = 1.5  # meters: only save every 10 meters

PATH_WISTH_ORIGINAL: float = 2.0

class VehicleTracker:
    """
    Tracks individual vehicle state for lane detection.
    Used for managing state transitions and delays, so in-lane detection is smooth and robust.
    """
    def __init__(self) -> None:
        self.current_state: bool = False
        self.target_state: bool = False
        self.transition_time: float = 0.0
        self.last_seen: float = time.time()

class EgoPathTracker:
    """
    Tracks ego path collision state for vehicles.
    Used for future expansion: e.g. tracking how long a vehicle is intersecting with the ego's predicted path.
    """
    def __init__(self) -> None:
        self.collision_start: Optional[float] = None
        self.last_collision: Optional[float] = None
        self.is_colliding: bool = False

class VehicleGroup:
    """
    Represents a group of vehicles (main vehicle + trailers).
    Used for clustering vehicles which are physically colliding or close together.
    """
    def __init__(self, vehicle: Any, polygon: Polygon, distance: float) -> None:
        self.vehicle: Any = vehicle
        self.polygon: Polygon = polygon
        self.distance: float = distance
        self.centroid: Point = polygon.centroid
        self.trailers: List[Dict[str, Any]] = []

class ETS2Radar:
    """
    ETS2 Radar system for detecting vehicles "in lane" and displaying radar view.
    Main entry point is update() or run().
    """

    def __init__(
        self,
        show_window: bool = True,
        max_distance: int = MAX_DISTANCE,
        fov_angle: int = FOV_ANGLE
    ) -> None:
        """
        Initialize ETS2 Radar class.

        Args:
            show_window: Whether to display the radar window.
            max_distance: Maximum radar detection distance in meters.
            fov_angle: Field of view half-angle in degrees.
        """

        self.show_window: bool = show_window
        self.max_distance: int = max_distance
        self.fov_angle: int = fov_angle
        self.refresh_interval: float = REFRESH_INTERVAL
        self.blinker_time_window: float = 2.5 # seconds to estimate changing lane
        self.last_left_blinker: int = 0 # timestamp of last left blinker on
        self.last_right_blinker: int = 0 # timestamp of last right blinker on

        self.capturing_positions = False
        self.vehicles: List[Vehicle] = []

        self.in_lane_scores_reset: bool = False
        self.reset_in_lane_scores: bool = False
        self.last_timestamp: int = time.time()
        self.prev_dt_frame: float = 0.1

        # Initialize telemetry and module
        self.module: Module = Module()
        self.module.imports()
        try:
            truck_telemetry.init()
        except Exception:
            pass

        # Vehicle tracking with proper type hints
        self.vehicle_histories: Dict[int, Deque[Coordinate]] = collections.defaultdict(
            lambda: collections.deque(maxlen=25)
        )
        self.vehicle_last_saved_position: Dict[int, Optional[Coordinate]] = collections.defaultdict(lambda: None)

        self.ego_trajectory: Deque[Coordinate] = collections.deque(maxlen=10)
        self.ego_last_saved_position: Optional[Coordinate] = None

        self.vehicle_speeds: Dict[int, float] = {}
        self.lane_state_tracker: Dict[int, VehicleTracker] = {}
        self.ego_path_tracker: Dict[int, EgoPathTracker] = {}
        self.ego_path_length: float = MAX_DISTANCE
        self.vehicle_world_pos_history = {}

        # Scoring system for vehicles (used for robust lane detection)
        self.vehicle_scores: Dict[int, float] = collections.defaultdict(float)

        # Setup window if requested
        if self.show_window:
            cv2.namedWindow("ETS2 Radar", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ETS2 Radar", 600, 600) 

        print("ETS2 Radar initialized.")

    def calculate_offset_score(
        self,
        vid: int,
        offset: Optional[float],
        distance: float = 30,
        angle: float = 0.0,
        data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate the new score for a vehicle, including previous score.
        If the vehicle has a trail (i.e. history deque exists and not empty) but not enough points (<10), score is -1.
        If no trail (no history or empty), score is 0.
        Otherwise, use the offset formula as before.
        """
        # Check if vehicle has a trail/history
        history = self.vehicle_histories.get(vid, None)
        if history is not None and offset is None:
            # Has a trail but not intersecting with base: penalize
            score_increment = -0.4
        elif history is None:
            score_increment = -0.16
        else:
            score_increment = 0.0

        # Offset method
        if offset is not None:
            try:
                x = float(offset)
                angle_amp = (2 ** (-(angle / 0.06) ** 2)) / 1
                offset_score = (2 ** (-(x / 9) ** 2) * 2.5 * angle_amp - 1) / 1
                distance_amp = (2 ** (-(distance / 100)) + 1 / ((distance + 3) / 8)-1)/3+1
                offset_score = max(-1, min(1, offset_score * distance_amp))
                score_increment += offset_score * 1.5 * (angle_amp*0.4 + 0.6)
            except Exception:
                pass
            
        return score_increment

    def calculate_angle_score(
        self,
        vid: int,
        ego_path: List[Tuple[float, float]],
        ego_speed: float,
        vehicle_centroid: Tuple[float, float],
        distance: float
    ) -> float:
        """
        Calculate score contribution from the intersection between ego_path and invisible circle.

        Args:
            vid: Vehicle ID.
            ego_path: List of ego path points (ego-space coordinates).
            vehicle_centroid: (x, z) position of the vehicle in ego-space.
            distance: Distance between ego and vehicle (radius of circle).
            previous_score: Optional previous score.

        Returns:
            Score contribution from the angular difference.
        """
        # Find intersection of ego_path with circle of radius=distance, centered at ego (0,0)
        intersection_points = []
        for i in range(1, len(ego_path)):
            p1 = ego_path[i-1]
            p2 = ego_path[i]
            # Line segment from p1 to p2
            # Parametric line: p = p1 + t*(p2-p1), t ∈ [0,1]
            dx = p2[0] - p1[0]
            dz = p2[1] - p1[1]
            a = dx**2 + dz**2
            b = 2 * (p1[0]*dx + p1[1]*dz)
            c = p1[0]**2 + p1[1]**2 - distance**2

            discriminant = b**2 - 4*a*c
            if discriminant < 0 or a == 0:
                continue
            sqrt_disc = np.sqrt(discriminant)
            for sign in [-1, 1]:
                t = (-b + sign * sqrt_disc) / (2 * a)
                if 0.0 <= t <= 1.0:
                    ix = p1[0] + t * dx
                    iz = p1[1] + t * dz
                    intersection_points.append((ix, iz))
        if not intersection_points:
            return 0.0

        # Pick the intersection point closest to the ego (should be only one in forward path)
        intersection = min(intersection_points, key=lambda p: p[1])  # furthest forward

        # Calculate angle from ego to intersection and from ego to vehicle
        intersection_angle = np.arctan2(intersection[0], intersection[1])
        vehicle_angle = np.arctan2(vehicle_centroid[0], vehicle_centroid[1])
        angle_diff = intersection_angle - vehicle_angle
        # Normalize angle_diff to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2*np.pi
        while angle_diff < -np.pi:
            angle_diff += 2*np.pi

        # Use formula for scoring
        try:
            x = float(np.degrees(angle_diff))
            angle_score = ((3 ** (-(x / 5) ** 2) * 2 - 1)/2 - (abs(x)/20)**6)
            distance_amp = (2**(-(distance/100))+1/((distance+3)/8)-1)/2+1
            # Clamp each score increment to [-2, 0.5] # temporarily set to max 0
            angle_score = max(-2, min(0, angle_score*distance_amp))
        except Exception:
                pass

        return angle_score

    def world_to_screen(
        self,
        dx: float,
        dz: float,
        center: ScreenCoordinate,
        scale: float,
        int_out = True
    ) -> ScreenCoordinate:
        """
        Convert a world-relative point (dx, dz) to screen coordinates.
        Used for visualization.
        """
        cx, cy = center
        if int_out:
            sx: int = int(cx + dx * scale)
            sy: int = int(cy - dz * scale)
            return sx, sy
        else:
            sx: float = float(cx + dx * scale)
            sy: float = float(cy - dz * scale)
            return sx, sy

    def rotate_point(
        self,
        dx: float,
        dz: float,
        yaw_rad: float
    ) -> Coordinate:
        """
        Rotate a point (dx, dz) around origin by yaw_rad radians.
        Used for transforming coordinates into ego-space.
        """
        rx: float = dx * np.cos(yaw_rad) - dz * np.sin(yaw_rad)
        rz: float = dx * np.sin(yaw_rad) + dz * np.cos(yaw_rad)
        return rx, rz

    def is_in_front_cone(
        self,
        dx: float,
        dz: float
    ) -> bool:
        """
        Check if (dx, dz) is within the forward field-of-view cone.
        """
        if dz <= 0:
            return False
        angle_deg: float = np.degrees(np.arctan2(abs(dx), dz))
        return angle_deg <= self.fov_angle

    def get_vehicle_polygon(
        self,
        vehicle: Any,
        px: float,
        pz: float,
        yaw_rad: float
    ) -> Polygon:
        """
        Get a Shapely polygon for the vehicle, transformed into ego-space.
        """
        corners: List[Corner] = vehicle.get_corners()
        pts: List[Coordinate] = []

        for x, y, z in corners:
            dx: float = x - px
            dz: float = z - pz
            dxr, dzr = self.rotate_point(-dx, dz, -yaw_rad)
            pts.append((dxr, dzr))

        return Polygon(pts)

    def draw_vehicle(
        self,
        img: npt.NDArray[np.uint8],
        poly: Polygon,
        center: ScreenCoordinate,
        scale: float,
        color: RGBColor
    ) -> None:
        """
        Draw a vehicle or trailer polygon on the radar.
        """
        pts: List[ScreenCoordinate] = [
            self.world_to_screen(x, z, center, scale)
            for x, z in poly.exterior.coords
        ]
        arr: npt.NDArray[np.int32] = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [arr], isClosed=True, color=color, thickness=1)

    def draw_fov_cone(
        self,
        img: npt.NDArray[np.uint8],
        center: ScreenCoordinate,
        scale: float
    ) -> None:
        """
        Overlay the FOV cone rays and arc (for visualization).
        """
        cx, cy = center
        fov_rad: float = np.radians(self.fov_angle)
        cone_len: int = int(self.max_distance * scale)

        left: ScreenCoordinate = (
            int(cx - cone_len * np.sin(fov_rad)),
            int(cy - cone_len * np.cos(fov_rad))
        )
        right: ScreenCoordinate = (
            int(cx + cone_len * np.sin(fov_rad)),
            int(cy - cone_len * np.cos(fov_rad))
        )

        cv2.line(img, (cx, cy), left, (0, 255, 0), 1)
        cv2.line(img, (cx, cy), right, (0, 255, 0), 1)

        arc: List[ScreenCoordinate] = [
            (int(cx + cone_len * np.sin(a)), int(cy - cone_len * np.cos(a)))
            for a in np.linspace(-fov_rad, fov_rad, 80)
        ]
        cv2.polylines(img, [np.array(arc, np.int32)], False, (0, 255, 0), 1)

    def fit_circle(
        self,
        xs: List[float],
        zs: List[float]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Fit a circle to a set of points (xs, zs) using least squares.
        Returns center (xc, zc) and radius r.
        """
        xs_array: npt.NDArray[np.float64] = np.array(xs)
        zs_array: npt.NDArray[np.float64] = np.array(zs)

        if len(xs_array) < 5:
            return None

        # Least squares fitting
        A: npt.NDArray[np.float64] = np.c_[2*xs_array, 2*zs_array, np.ones(xs_array.shape)]
        b: npt.NDArray[np.float64] = xs_array**2 + zs_array**2

        try:
            c, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
            xc, zc, d = c
            r: float = np.sqrt(xc**2 + zc**2 + d)
            return xc, zc, r
        except Exception:
            return None

    def draw_fitted_arc(
        self,
        img: npt.NDArray[np.uint8],
        history: List[Coordinate],
        px: float,
        pz: float,
        yaw_rad: float,
        center: ScreenCoordinate,
        scale: float,
        arc_length: float = 20.0,
        color: RGBColor = (0, 255, 255),
        reverse: bool = False,
        ego_steer: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Draw a fixed-length arc (from a circle fit) behind or in front of the vehicle.
        Returns intersection info for scoring.
        Now: If there are not enough history points, does nothing (no synthetic straight trail).
        """
        if len(history) < 5:
            # Not enough points: do not draw anything, no offset/angle
            return None, None

        xs, zs = zip(*history)
        fit_result = self.fit_circle(list(xs), list(zs))
        if fit_result is None:
            return None, None

        xc, zc, r = fit_result
        x_end, z_end = xs[-1], zs[-1]
        x_prev, z_prev = xs[-2], zs[-2]

        v_end = np.array([x_end - xc, z_end - zc])
        v_prev = np.array([x_prev - xc, z_prev - zc])
        theta_end = np.arctan2(v_end[1], v_end[0])
        cross = v_end[0] * v_prev[1] - v_end[1] * v_prev[0]
        direction = 1 if cross > 0 else -1

        if reverse:
            direction *= -1

        arc_angle = arc_length / r if r != 0 else 0.0
        arc_points = []
        num_points = 30

        for i in range(num_points + 1):
            t = i / num_points
            angle = theta_end - direction * t * arc_angle
            x = xc + r * np.cos(angle)
            z = zc + r * np.sin(angle)
            dx = x - px
            dz = z - pz
            dxr, dzr = self.rotate_point(-dx, dz, -yaw_rad)
            arc_points.append(self.world_to_screen(dxr, dzr, center, scale))

        # Draw arc for visualization only
        if self.show_window and arc_points:
            arr = np.array(arc_points, dtype=np.int32)
            cv2.polylines(img, [arr], isClosed=False, color=color, thickness=1)
            curvature = 1.0 / r if r != 0 else 0.0
            cv2.putText(img, f"curv={curvature:.3f}", arc_points[-1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Find intersection with screen bottom
        intersection_point = None
        intersection_angle = None
        screen_height = img.shape[0] if self.show_window else 600

        for i, (x, y) in enumerate(arc_points):
            if y >= screen_height:
                intersection_point = float(x - 300)
                if i > 0:
                    prev_point = arc_points[i - 1]
                    tangent_x = float(x - prev_point[0])
                    tangent_y = float(y - prev_point[1])
                    tangent_angle = np.arctan2(tangent_y, tangent_x)
                    baseline_angle = abs(tangent_angle)
                    if baseline_angle > np.pi / 2:
                        baseline_angle = np.pi - baseline_angle
                    normalized_angle = 2.0 - (2.0 * baseline_angle / (np.pi / 2))
                    if tangent_x < 0:
                        normalized_angle = -abs(normalized_angle)
                    else:
                        normalized_angle = abs(normalized_angle)
                    intersection_angle = normalized_angle
                break

        return intersection_point, intersection_angle

    def draw_ego_rectangle(self, img, center, scale, width=2.0, length=6.0, color=(0, 255, 255), thickness=2):
        """
        Draw a rectangle representing the ego vehicle at the center point.
        Width: 2m, Length: 6m (extending forward from ego point)
        Returns the rectangle corners in world coordinates for intersection detection.
        """
        # Ego vehicle rectangle corners in world coordinates (ego is at origin 0,0)
        half_width = width / 2.0
        # Rectangle extends from 0 to length forward (positive z direction)
        corners_world = [
            (-half_width, 0),      # rear left
            (half_width, 0),       # rear right
            (half_width, length),  # front right
            (-half_width, length)  # front left
        ]
        
        # Convert to screen coordinates
        corners_screen = []
        for x, z in corners_world:
            screen_pt = self.world_to_screen(x, z, center, scale, int_out=True)
            corners_screen.append(screen_pt)
        
        # Draw the rectangle
        pts = np.array(corners_screen, np.int32)
        cv2.polylines(img, [pts], True, color, thickness)
        
        return corners_world
 




















    def check_vehicle_in_ego_path(
        self,
        path_points: List[Tuple[float, float]],
        vehicle_polygon: Polygon,
        path_width: float
    ) -> bool:
        """
        Check if a vehicle polygon intersects with the ego path polygon.
        
        Args:
            path_points: List of (x, z) points defining the ego path in ego-space
            vehicle_polygon: Shapely polygon of the vehicle in ego-space
            path_width: Width of the ego path in meters
            
        Returns:
            True if vehicle intersects with the ego path
        """
        if not path_points or len(path_points) < 2:
            return False
        
        try:
            path_polygon = self.create_path_polygon(path_points, path_width)
            if path_polygon is None:
                return False
            
            # Check if vehicle polygon intersects with path polygon
            return vehicle_polygon.intersects(path_polygon)
        except Exception:
            return False

    def draw_ego_path_with_width(
        self,
        img: npt.NDArray[np.uint8],
        path_points: List[Tuple[float, float]],
        center: ScreenCoordinate,
        scale: float,
        path_width: float,
        color: RGBColor = (0, 255, 0),
        thickness: int = 1,
        show_info: bool = True,
        curvature: Optional[float] = None
    ):
        """
        Draw the ego vehicle's predicted path with visible width on the radar.
        
        Args:
            path_width: Width of the path in meters (default 1.5m)
        """
        if not path_points or len(path_points) < 2:
            return

        # Create path polygon for visualization
        path_polygon = self.create_path_polygon(path_points, path_width)
        if path_polygon is not None:
            try:
                # Convert polygon exterior coordinates to screen coordinates
                exterior_coords = list(path_polygon.exterior.coords)
                screen_coords = []
                for x, z in exterior_coords:
                    screen_x = int(center[0] + x * scale)
                    screen_y = int(center[1] - z * scale)
                    screen_coords.append((screen_x, screen_y))
                
                # Draw filled path polygon with transparency effect
                if len(screen_coords) >= 3:
                    # Create a copy of the image for overlay
                    overlay = img.copy()
                    pts = np.array(screen_coords, np.int32)
                    cv2.fillPoly(overlay, [pts], (0, 100, 0))  # Dark green fill
                    # Blend with original image (30% opacity)
                    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                    
                    # Draw path outline
                    cv2.polylines(img, [pts], True, color, thickness)
            except Exception:
                pass

        # Draw center line as fallback/reference
        screen_points = []
        for x, z in path_points:
            screen_x = int(center[0] + x * scale)
            screen_y = int(center[1] - z * scale)
            screen_points.append((screen_x, screen_y))

        # Draw the connected path center line
        for i in range(len(screen_points) - 1):
            cv2.line(img, screen_points[i], screen_points[i + 1], color, thickness)

        # Optionally draw curvature text
        if show_info and curvature is not None:
            try:
                k = float(curvature)
                text = f"curv={k:.4f} m^-1"
                base_pos = (center[0] + 8, center[1] - 12)
                cv2.putText(img, text, base_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            except Exception:
                pass

    def draw_radar(
        self,
        vehicles: List[Vehicle],
        px: float,
        pz: float,
        yaw: float,
        ego_steer: float = 0.0,
        ego_speed: float = 0.0,
        data: Optional[Dict[str, Any]] = None,
        cc_app=None,
        paused: bool = False
    ) -> List[Any]:
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect("ETS2 Radar")
        except:
            win_w, win_h = 600, 600

        vehicles = self.vehicles
        
        yaw_rad = (yaw + 0.5) * 2 * np.pi
        yaw_deg = yaw * 360

        path_width = PATH_WISTH_ORIGINAL + np.sin((abs(ego_steer*1.5)) * (np.pi/2)) * 4.0
        
        scale = win_h / self.max_distance
        center = (win_w // 2, win_h)
        img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        mask = np.zeros((win_h, win_w), dtype=np.uint8)
        cone_len = int(self.max_distance * scale)
        pts = [(center[0], center[1])]
        for a in np.linspace(-np.radians(self.fov_angle), np.radians(self.fov_angle), 200):
            x = int(center[0] + cone_len * np.sin(a))
            y = int(center[1] - cone_len * np.cos(a))
            pts.append((x, y))
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

        for r in range(25, self.max_distance + 1, 25):
            try:
                radius = int(r * scale)
                tmp = np.zeros_like(img)
                cv2.circle(tmp, center, radius, (100, 100, 100), 1)
                img = cv2.bitwise_or(img, cv2.bitwise_and(tmp, tmp, mask=mask))
            except:
                continue

        tmp = np.zeros_like(img)
        cv2.line(tmp, (center[0], 0), (center[0], win_h), (100, 100, 100), 1)
        cv2.line(tmp, (0, center[1]), (win_w, center[1]), (100, 100, 100), 1)
        img = cv2.bitwise_or(img, cv2.bitwise_and(tmp, tmp, mask=mask))

        self.draw_fov_cone(img, center, scale)
        cv2.circle(img, center, 5, (0, 255, 255), -1)
        
        """
        # Draw ego vehicle rectangle (2m wide, 6m long)
        ego_rect_corners = self.draw_ego_rectangle(img, center, scale, width=2.0, length=6.0, color=(0, 255, 255), thickness=2)
        """

        path_pts, curvature = [], 0.0
        try:
            path_pts, curvature = self.predict_ego_path_using_history(px, pz, yaw_rad, path_length=self.ego_path_length, ego_steer=ego_steer, ego_speed=ego_speed)
            # Draw ego path with 2m width
            self.draw_ego_path_with_width(img, path_pts, center, scale, path_width=path_width, color=(0, 200, 0), thickness=1, show_info=True, curvature=curvature)
        except Exception:
            pass

        origin = Point(0, 0)
        vehicle_data = []
        detected_ids = set()
        for v in vehicles:
            poly = self.get_vehicle_polygon(v, px, pz, yaw_rad)
            closest_distance = min(Point(coord).distance(origin) for coord in poly.exterior.coords[:-1])
            if closest_distance <= self.max_distance:
                vid = getattr(v, 'id', id(v))
                detected_ids.add(vid)
                vehicle_data.append({
                    'vehicle': v,
                    'polygon': poly,
                    'distance': closest_distance,
                    'centroid': poly.centroid,
                    'trailers': []
                })
                for t in getattr(v, 'trailers', []):
                    poly_t = self.get_vehicle_polygon(t, px, pz, yaw_rad)
                    trailer_closest_distance = min(Point(coord).distance(origin) for coord in poly_t.exterior.coords[:-1])
                    if trailer_closest_distance <= self.max_distance:
                        vehicle_data[-1]['trailers'].append({
                            'trailer': t,
                            'polygon': poly_t,
                            'distance': trailer_closest_distance
                        })

        # --- VEHICLE DATA RESET LOGIC ---
        # Find all known vehicle IDs (with score/history etc.)
        all_known_ids = set(self.vehicle_scores.keys()) | set(self.vehicle_histories.keys()) | set(self.vehicle_last_saved_position.keys()) | set(self.vehicle_speed_history.keys()) if hasattr(self, 'vehicle_speed_history') else set()
        # Check for ID disappearance or jump
        ids_to_reset = set()
        for vid in all_known_ids:
            if vid not in detected_ids:
                ids_to_reset.add(vid)
            else:
                # Check for large position jumps (>100 m/s)
                # Compare previous and current position in self.vehicle_histories
                hist = self.vehicle_histories.get(vid, None)
                if hist and len(hist) >= 2:
                    prev_pos = hist[-2]
                    curr_pos = hist[-1]
                    dx = curr_pos[0] - prev_pos[0]
                    dz = curr_pos[1] - prev_pos[1]
                    dist = np.sqrt(dx * dx + dz * dz)
                    # Estimate velocity over last frame (assuming 0.1s/frame)
                    velocity = dist / 0.1
                    if velocity > 100.0:
                        ids_to_reset.add(vid)
        # Reset all data for these IDs
        for vid in ids_to_reset:
            self.vehicle_scores.pop(vid, None)
            self.vehicle_histories.pop(vid, None)
            self.vehicle_last_saved_position.pop(vid, None)
            if hasattr(self, 'vehicle_speed_history'):
                self.vehicle_speed_history.pop(vid, None)
            if hasattr(self, 'vehicle_world_pos_history'):
                self.vehicle_world_pos_history.pop(vid, None)
            if hasattr(self, 'lane_state_tracker'):
                self.lane_state_tracker.pop(vid, None)
            if hasattr(self, 'ego_path_tracker'):
                self.ego_path_tracker.pop(vid, None)
            # You may need to reset more if you add more vehicle-specific state

        now = time.time()

        # Edge detection and speed check for blinker activation
        if data.get("blinkerRightActive", False) and not getattr(self, 'prev_right_active', False) and ego_speed*3.6 >= 65:
            self.last_right_blinker = now
            for vid in self.vehicle_scores:
                self.vehicle_scores[vid] = 0.0
        if data.get("blinkerLeftActive", False) and not getattr(self, 'prev_left_active', False) and ego_speed*3.6 >= 65:
            self.last_left_blinker = now
            for vid in self.vehicle_scores:
                self.vehicle_scores[vid] = 0.0

        self.prev_right_active = data.get("blinkerRightActive", False)
        self.prev_left_active = data.get("blinkerLeftActive", False)

        # Calculate blinker offset with gradual return to origin
        t_left = (now - self.last_left_blinker) / self.blinker_time_window
        t_right = (now - self.last_right_blinker) / self.blinker_time_window

        val_left = -np.cos(t_left * np.pi / 2) if t_left <= 1 else 0
        val_right = np.cos(t_right * np.pi / 2) if t_right <= 1 else 0

        blinker_offset = val_left + val_right

        in_lane_vehicles = []
        for vdata in vehicle_data:
            v = vdata['vehicle']
            poly = vdata['polygon']
            distance = vdata['distance']
            dx, dz = vdata['centroid'].x, vdata['centroid'].y
            overall_closest_distance = distance#-vdata['vehicle'].size.length/2
            for trailer_data in vdata['trailers']:
                trailer_distance = trailer_data['distance']#-trailer_data['trailer'].size.length/2
                if trailer_distance < overall_closest_distance:
                    overall_closest_distance = trailer_distance + 3
                
            distance_amp = max((1/(overall_closest_distance*0.02)) - overall_closest_distance*0.01 + 0.6, 0)
            
            vid = getattr(v, 'id', id(v))

            # Update vehicle world pos history instead of screen pos history
            if vid not in self.vehicle_world_pos_history:
                self.vehicle_world_pos_history[vid] = collections.deque(maxlen=10)
            self.vehicle_world_pos_history[vid].append((dx, dz))

            # Use speed and acceleration directly from vehicle
            raw_vehicle_speed = getattr(v, 'speed', 0)  # Speed in m/s
            is_tmp = getattr(v, 'is_tmp', False)

            # Use speed/acceleration from vehicle class directly
            vehicle_speed = getattr(v, 'speed', 0)
            acceleration = getattr(v, 'acceleration', 0)

            offset_for_score: Optional[float] = None
            angle_for_score: Optional[float] = None
            score_val = self.vehicle_scores.get(vid, 0.0)
            if self.is_in_front_cone(dx, dz):
                hist = self.vehicle_histories.get(vid, [])
                if vehicle_speed < 2 and len(hist) == 0:
                    cx = np.mean([c[0] for c in v.get_corners()])
                    cz = np.mean([c[2] for c in v.get_corners()])
                    corners = v.get_corners()
                    front_center_x = (corners[0][0] + corners[1][0]) / 2
                    front_center_z = (corners[0][2] + corners[1][2]) / 2
                    rear_center_x = (corners[2][0] + corners[3][0]) / 2
                    rear_center_z = (corners[2][2] + corners[3][2]) / 2
                    forward_x = front_center_x - rear_center_x
                    forward_z = front_center_z - rear_center_z
                    forward_length = np.sqrt(forward_x**2 + forward_z**2)
                if hist:
                    offset, angle = self.draw_fitted_arc(img, hist, px, pz, yaw_rad, center, scale,
                                                arc_length=150, color=(0,80,80), reverse=True, ego_steer=ego_steer)
                    if offset is not None:
                        offset_for_score = offset - blinker_offset * 18
                    else:
                        offset_for_score = None
                    angle_for_score = angle
                previous_score = self.vehicle_scores.get(vid, 0.0)
                offset_score = self.calculate_offset_score(vid, offset_for_score, overall_closest_distance, angle_for_score, data)

                #temp
                angle_score = 0.0
                """
                angle_score = self.calculate_angle_score(
                    vid,
                    path_pts,
                    ego_speed,
                    (dx, dz),
                    overall_closest_distance,
                )
                """

                _, y, _ = getattr(v, 'rotation', None).euler()

                yaw_diff = min([abs(y - yaw_deg), abs((y - yaw_deg) + 360), abs((y - yaw_deg) - 360)])
                yaw_score = ((2**(-(abs(yaw_diff)/90)**5))/1 - 1) * 1.5
                
                angle_score = 0.0  # Temporarily disable angle score contribution

                # Check if vehicle is in ego path
                if path_pts:
                    is_in_path = self.check_vehicle_in_ego_path(path_pts, poly, path_width=path_width)
                    
                    # Also check trailers
                    trailer_in_path = False
                    for trailer_data in vdata['trailers']:
                        if self.check_vehicle_in_ego_path(path_pts, trailer_data['polygon'], path_width=path_width):
                            trailer_in_path = True
                            break
                    
                    slow_speed_score_amp = 1.4+(ego_speed*3.6/100)*(5.5-1.4)

                    path_score_base = pow(1.03, -overall_closest_distance) * (slow_speed_score_amp - abs(blinker_offset**2 * 0.4) * slow_speed_score_amp)

                    # Vehicle or any of its trailers in path = positive score
                    if is_in_path or trailer_in_path:
                        path_score = min(path_score_base, 5)
                    else:
                        path_score = -min(path_score_base * 0.6, 4)
                else:
                    path_score = 0.0
                
                if self.reset_in_lane_scores:
                    score_val = 0.0  # Reset to 0 on blinker change
                elif not paused:
                    score_val = previous_score + (offset_score + angle_score + yaw_score + path_score) * max((abs(vehicle_speed)/90)**0.8, 0.5)
                else:
                    score_val = previous_score

                # Clamp score to valid range
                score_val = max(-5.0, min(15.0, score_val))
                self.vehicle_scores[vid] = score_val

            if self.is_in_front_cone(dx, dz) and score_val > 0:
                in_lane_vehicles.append((v, overall_closest_distance - 4.5, acceleration))

            # Draw vehicle with ID labels
            veh_scr_x, veh_scr_y = self.world_to_screen(dx, dz, center, scale)
            
            if self.is_in_front_cone(dx, dz):
                if not v.is_tmp or not v.is_trailer:
                    if score_val > 0:
                        # Check if vehicle is in ego path for different coloring
                        vehicle_color = (0, 0, 255)  # Default red for in-lane
                        if path_pts and self.check_vehicle_in_ego_path(path_pts, poly, path_width=path_width):
                            vehicle_color = (0, 100, 255)  # Orange-red for vehicles in ego path
                        
                        self.draw_vehicle(img, poly, center, scale, vehicle_color)
                    else:
                        self.draw_vehicle(img, poly, center, scale, (0,0,100))
                else:
                    if score_val > 0:
                        # Use same color logic for trailers
                        trailer_color = (255, 0, 0)  # Default red for in-lane trailers
                        if path_pts and self.check_vehicle_in_ego_path(path_pts, poly, path_width=path_width):
                            trailer_color = (255, 50, 0)  # Orange for trailers in ego path
                        self.draw_vehicle(img, poly, center, scale, trailer_color)
                    else:
                        self.draw_vehicle(img, poly, center, scale, (100,0,0))
            else:
                self.draw_vehicle(img, poly, center, scale, (200,200,0))
            
            # ALWAYS draw ID label for debugging
            id_label = f"ID:{vid}"
            if v.is_tmp:
                id_label += " TMP"
            if v.is_trailer:
                id_label += " TRAILER"
            cv2.putText(img, id_label, (veh_scr_x-30, veh_scr_y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            if self.is_in_front_cone(dx, dz) and score_val > 0:
                cv2.putText(img, f"s:{score_val:.1f}", (veh_scr_x-30, veh_scr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)

            for trailer_idx, trailer_data in enumerate(vdata['trailers']):
                poly_t = trailer_data['polygon']
                tx, tz = poly_t.centroid.x, poly_t.centroid.y
                t_scr_x, t_scr_y = self.world_to_screen(tx, tz, center, scale)
                
                if self.is_in_front_cone(tx, tz):
                    if score_val > 0:
                        # Use same color logic for trailers
                        trailer_color = (255, 0, 0)  # Default red for in-lane trailers
                        if path_pts and self.check_vehicle_in_ego_path(path_pts, poly_t, path_width=path_width):
                            trailer_color = (255, 50, 0)  # Orange for trailers in ego path
                        self.draw_vehicle(img, poly_t, center, scale, trailer_color)
                    else:
                        self.draw_vehicle(img, poly_t, center, scale, (100,0,0))
                else:
                    self.draw_vehicle(img, poly_t, center, scale, (200,200,0))
                
                # Draw trailer label
                trailer_label = f"T{trailer_idx} of ID:{vid}"
                cv2.putText(img, trailer_label, (t_scr_x-30, t_scr_y-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1, cv2.LINE_AA)

        in_lane_vehicles.sort(key=lambda x: x[1])
        
        # Process vehicles with special handling for is_tmp trailers
        filtered_vehicles = []
        
        for i, (v, dist, accel) in enumerate(in_lane_vehicles):
            # Handle is_tmp trailers - use trailer's distance but speed/accel from vehicle behind it
            if v.is_tmp and v.is_trailer:
                # Get the next vehicle in the list (behind the trailer) for speed and acceleration
                if i + 1 < len(in_lane_vehicles):
                    next_v, _, next_accel = in_lane_vehicles[i + 1]
                    # Use trailer's distance but next vehicle's speed and acceleration
                    v.speed = next_v.speed
                    filtered_vehicles.append((v, dist, next_accel))
                else:
                    # No vehicle behind, use trailer's own values
                    filtered_vehicles.append((v, dist, accel))
            else:
                # Non-trailer vehicle, add normally
                filtered_vehicles.append((v, dist, accel))
        
        result = []
        for v, dist, accel in filtered_vehicles[:3]:
            # Use filtered speed if available, otherwise use raw speed
            vid = getattr(v, 'id', id(v))
            speed_raw = getattr(v, 'speed', 0)
            speed_kmh = speed_raw * 3.6
            result.append((v.id, dist, speed_kmh, accel))

        try:
            if len(filtered_vehicles) > 0 and (len(getattr(filtered_vehicles[0][0], 'trailers', [])) > 0 or getattr(filtered_vehicles[0][0], 'is_trailer', False)):
                if cc_app is not None:
                    cc_app.update(acc_truck=True)
            else:
                if cc_app is not None:
                    cc_app.update(acc_truck=False)
        except:
            if cc_app is not None:
                cc_app.update(acc_truck=False)

        if self.show_window:
            cv2.imshow("ETS2 Radar", img)

        return result
        
    
    









    def predict_ego_path_using_history(
        self,
        px: float,
        pz: float,
        yaw_rad: float,
        ego_steer: float,
        ego_speed: float,
        path_length: float = 40.0,
        max_history: int = 25
    ):
        """
        Predict the most likely ego path using the past positions stored in self.ego_trajectory.

        Approach:
        - Use up to `max_history` past positions (already in world coords).
        - Transform them into ego-space (so ego is at origin facing +z).
        - Fit a circle to the transformed points. Extract signed curvature (1/radius).
        - Blend fitted curvature with steering input based on speed:
          * < 15 km/h: 100% steering
          * 15-30 km/h: linear interpolation from 100% to 30% steering
          * >= 30 km/h: 30% steering, 70% history
        - Generate arc forward from ego origin using curvature, with center always below the screen (negative z).
        - If fit fails or radius is extremely large, fallback to straight line.
        - Return the generated path points (as ego-space (x, y) where y is forward) and signed curvature.
        """
        hist = list(self.ego_trajectory)[-max_history:]
        if len(hist) < 3:
            pts = [(0.0, float(i)) for i in range(int(path_length) + 1)]
            return pts, 0.0

        ego_pts: List[Tuple[float, float]] = []
        for wx, wz in hist:
            dx = wx - px
            dz = wz - pz
            x_e, z_e = self.rotate_point(dx, dz, -yaw_rad)
            ego_pts.append((x_e, z_e))

        xs = [p[0] for p in ego_pts]
        zs = [p[1] for p in ego_pts]

        fit = self.fit_circle(xs, zs)
        if fit is None:
            pts = [(0.0, float(i)) for i in range(int(path_length) + 1)]
            return pts, 0.0

        xc, zc, r = fit
        if r == 0 or r > 1e4:
            pts = [(0.0, float(i)) for i in range(int(path_length) + 1)]
            return pts, 0.0

        x_last, z_last = ego_pts[-1]
        x_prev, z_prev = ego_pts[-2]
        v_end = np.array([x_last - xc, z_last - zc])
        v_prev = np.array([x_prev - xc, z_prev - zc])
        cross = v_end[0] * v_prev[1] - v_end[1] * v_prev[0]
        direction = 1 if cross > 0 else -1

        fitted_curvature = (direction * (1.0 / r)) if r != 0 else 0.0
        steer_curvature = ego_steer * 0.17

        speed_kmh = abs(ego_speed)
        if speed_kmh < 15.0:
            steer_weight = 1.0
        elif speed_kmh < 30.0:
            steer_weight = 1.0 - ((speed_kmh - 15.0) / 15.0) * 0.7
        else:
            steer_weight = 0.3

        history_weight = 1.0 - steer_weight
        curvature = steer_weight * steer_curvature + history_weight * fitted_curvature

        if curvature == 0.0:
            pts = [(0.0, float(i)) for i in range(int(path_length) + 1)]
            return pts, 0.0

        radius = abs(1.0 / curvature)
        lateral_offset = radius if curvature < 0 else -radius
        center_x = lateral_offset
        center_z = -abs(np.sqrt(max(0, radius**2 - center_x**2)))

        theta0 = np.arctan2(-center_z, -center_x)
        turn_direction = -1 if curvature > 0 else 1

        num_points = max(2, int(path_length)) * 2
        arc_points: List[Tuple[float, float]] = []
        for s in np.linspace(0.0, path_length, num_points):
            delta_theta = s / radius
            angle = theta0 + turn_direction * delta_theta
            x = center_x + radius * np.cos(angle)
            z = center_z + radius * -np.sin(angle)
            arc_points.append((float(x), float(z)))

        return arc_points, curvature

    def create_path_polygon(
        self,
        path_points,
        width=3.5
    ):
        """
        Create a polygon representing the path with given width.
        Used for collision prediction.
        """
        if len(path_points) < 2:
            return None

        left_points = []
        right_points = []

        for i, (x, y) in enumerate(path_points):
            if i == 0:
                # For first point, use direction to next point
                next_x, next_y = path_points[1]
                dx, dy = next_x - x, next_y - y
            elif i == len(path_points) - 1:
                # For last point, use direction from previous point
                prev_x, prev_y = path_points[i-1]
                dx, dy = x - prev_x, y - prev_y
            else:
                # For middle points, use average direction
                prev_x, prev_y = path_points[i-1]
                next_x, next_y = path_points[i+1]
                dx, dy = next_x - prev_x, next_y - prev_y

            # Normalize direction
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx, dy = dx/length, dy/length
                # Perpendicular vector
                perp_x, perp_y = -dy, dx
                half_width = width / 2
                left_points.append((x + perp_x * half_width, y + perp_y * half_width))
                right_points.append((x - perp_x * half_width, y - perp_y * half_width))

        # Create polygon: left side + reversed right side
        polygon_points = left_points + right_points[::-1]

        try:
            from shapely.geometry import Polygon
            return Polygon(polygon_points)
        except:
            return None

    '''
    def draw_ego_path(
        self,
        img: npt.NDArray[np.uint8],
        path_points: List[Tuple[float, float]],
        center: ScreenCoordinate,
        scale: float,
        color: RGBColor = (0, 255, 0),
        thickness: int = 1,
        show_info: bool = True,
        curvature: Optional[float] = None
    ):
        """
        Draw the ego vehicle's predicted path on the radar.

        - path_points: list of (x, y) in ego-space where y is forward.
        - thickness: line thickness (user requested 1).
        - curvature: optional signed curvature value to display.
        """
        if not path_points or len(path_points) < 2:
            return

        # Convert path points to screen coordinates
        screen_points = []
        for x, y in path_points:
            screen_x = int(center[0] + x * scale)
            screen_y = int(center[1] - y * scale)
            screen_points.append((screen_x, screen_y))

        # Draw the connected path with specified thickness
        for i in range(len(screen_points) - 1):
            cv2.line(img, screen_points[i], screen_points[i + 1], color, thickness)

        # Optionally draw curvature text
        if show_info and curvature is not None:
            try:
                k = float(curvature)
                text = f"curv={k:.4f} m^-1"
                # Place the text near the first point of the path (just above ego)
                base_pos = (center[0] + 8, center[1] - 12)
                cv2.putText(img, text, base_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            except Exception:
                pass
    '''

    def should_save_position(self, last: Optional[Coordinate], current: Coordinate) -> bool:
        """
        Checks if the distance between last and current is at least POSITION_SNAP meters.
        """
        if last is None:
            return True
        dx = current[0] - last[0]
        dz = current[1] - last[1]
        distance = np.sqrt(dx*dx + dz*dz)
        return distance >= POSITION_SNAP

    def handle_duplicate_vehicle_ids(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """
        Handle duplicate vehicle IDs by appending index numbers.
        If there are 3 vehicles with ID 123, they become 1231, 1232, 1233.
        """
        # Count occurrences of each ID
        id_counts = {}
        for vehicle in vehicles:
            vehicle_id = getattr(vehicle, 'id', id(vehicle))
            id_counts[vehicle_id] = id_counts.get(vehicle_id, 0) + 1
        
        # Create new vehicles with unique IDs
        filtered_vehicles = []
        id_indices = {}  # Track current index for each original ID
        
        for vehicle in vehicles:
            original_id = getattr(vehicle, 'id', id(vehicle))
            
            # If this ID appears multiple times, create a unique ID
            if id_counts[original_id] > 1:
                if original_id not in id_indices:
                    id_indices[original_id] = 1
                else:
                    id_indices[original_id] += 1
                
                # Create new ID by appending index
                new_id = int(str(original_id) + str(id_indices[original_id]))
                
                # Create a copy of the vehicle with the new ID
                new_vehicle = Vehicle(
                    position=vehicle.position,
                    rotation=vehicle.rotation,
                    size=vehicle.size,
                    speed=vehicle.speed,
                    acceleration=vehicle.acceleration,
                    trailer_count=vehicle.trailer_count,
                    trailers=vehicle.trailers,
                    id=new_id,
                    is_tmp=vehicle.is_tmp,
                    is_trailer=vehicle.is_trailer
                )
                filtered_vehicles.append(new_vehicle)
            else:
                # Keep original vehicle if ID is unique
                filtered_vehicles.append(vehicle)
        
        return filtered_vehicles

    def update_vehicle_history(self, vid: int, position: Coordinate) -> None:
        """
        Only save vehicle positions if moved >= POSITION_SNAP meters since last saved.
        """
        last = self.vehicle_last_saved_position[vid]
        if self.should_save_position(last, position):
            self.vehicle_histories[vid].append(position)
            self.vehicle_last_saved_position[vid] = position

    def update_ego_trajectory(self, position: Coordinate) -> None:
        """
        Only save ego positions if moved >= POSITION_SNAP meters since last saved.
        """
        last = self.ego_last_saved_position
        if self.should_save_position(last, position):
            self.ego_trajectory.append(position)
            self.ego_last_saved_position = position

    def update(self, data=None, cc_app=None):
        """
        Update radar data and return closest 3 in-lane vehicles.
        This is the main loop for updating vehicle histories and in-lane status.
        """
        if data is None:
            data = truck_telemetry.get_data()
        px = data.get("coordinateX", 0.0)
        pz = data.get("coordinateZ", 0.0)
        yaw = data.get("rotationX", 0.0)
        yaw_rad = (yaw + 0.5) * 2 * np.pi
        ego_steer = data.get("gameSteer", 0.0)
        speed = data.get("speed", 0.0)
        self.ego_path_length = MAX_DISTANCE
        ego_steering.append(ego_steer)
        ego_steer = np.mean(ego_steering)

        paused = data.get("paused", False)

        new_capture = truck_telemetry.get_data()
        px = new_capture.get("coordinateX", 0.0)
        pz = new_capture.get("coordinateZ", 0.0)
        vehicles = self.module.run(paused)
        if vehicles is None:
            return []
        self.update_ego_trajectory((px, pz))

        # Filter vehicles not on the truck's plane
        vehicles = [v for v in vehicles if abs(v.position.y - data.get("coordinateY", 0.0)) < 6.0]
        
        # Handle duplicate vehicle IDs
        vehicles = self.handle_duplicate_vehicle_ids(vehicles)
        self.vehicles = vehicles

        if not paused:
            for v in vehicles:
                # Update histories with world coordinates
                vid = getattr(v, 'id', id(v))
                poly = self.get_vehicle_polygon(v, px, pz, yaw_rad)
                cx = np.mean([c[0] for c in v.get_corners()])
                cz = np.mean([c[2] for c in v.get_corners()])
                self.update_vehicle_history(vid, (cx, cz))
                # Always update speed for all vehicles (store speed value)
                self.vehicle_speeds[vid] = getattr(v, 'speed', 0)

        # Main detection and visualization
        in_lane_vehicles = self.draw_radar(vehicles, px, pz, yaw, ego_steer, speed, data, cc_app, paused)
        self.start_position_capture()
        return in_lane_vehicles

    def start_position_capture(self):
        """
        Start capturing vehicle positions for history.
        """
        if self.capturing_positions:
            return
        self.capturing_positions = True

        # Start a thread to capture positions every 0.02 seconds
        def capture_thread():
            """Thread function that captures positions for high_res vehicles."""
            while self.capturing_positions:
                try:
                    # Call add_position_to_queue for vehicles
                    for vehicle in self.vehicles:
                        vehicle.add_position_to_queue()
                    
                    # Sleep for 0.005 seconds (200 Hz)
                    time.sleep(0.005)
                except Exception as e:
                    # Continue running even if there's an error
                    print(f"Error in position capture thread: {e}")
                    time.sleep(0.02)
        
        # Start the thread as a daemon so it stops when main program exits
        capture_thread_obj = threading.Thread(target=capture_thread, daemon=True)
        capture_thread_obj.start()


    def run(self):
        """
        Main loop for radar operation (either with or without visualization).
        Press ESC to exit.
        """
        while True:
            in_lane_vehicles = self.update() #output: (vehicle, distance, speed)

            print(f"Vehicles in lane: {len(in_lane_vehicles)}")
            if in_lane_vehicles:
                for i, (vehicle, distance, speed) in enumerate(in_lane_vehicles, 1):
                    print(f"  {i}. Vehicle {vehicle} at {distance:.1f}m, speed: {speed:.1f} km/h")

            if self.show_window:
                if cv2.waitKey(int(self.refresh_interval * 1000)) & 0xFF == 27:
                    break
            else:
                import time
                time.sleep(self.refresh_interval)

    def cleanup(self):
        """
        Clean up window resources (call at exit).
        """
        self.capturing_positions = False
        if self.show_window:
            cv2.destroyAllWindows()

# Test code - runs when script is executed directly
if __name__ == "__main__":
    # Create radar with window display
    radar = ETS2Radar(show_window=True, fov_angle=45)
    try:
        radar.run()
    except KeyboardInterrupt:
        print("\nStopping radar...")
    finally:
        radar.cleanup()
    # Example of how to use without window:
    # radar_headless = ETS2Radar(show_window=False)
    # in_lane_vehicles = radar_headless.update()
    # print(f"Found {len(in_lane_vehicles)} in-lane vehicles")