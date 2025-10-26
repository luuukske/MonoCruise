from turtle import speed
from typing import List, Tuple, Dict, Optional, Any, Deque, Union
import cv2
import numpy as np
import numpy.typing as npt
import truck_telemetry
from shapely.geometry import Polygon, Point

try:
    from ETS2radar.classes import Vehicle
    from ETS2radar.main import Module
except:
    print("\n")
    print("#"*20)
    print("\nwrong file stubid\n")
    print("#"*20)
    exit()
import collections
import time

# Type aliases for clarity
Coordinate = Tuple[float, float]
ScreenCoordinate = Tuple[int, int]
VehicleData = Tuple[int, float, float]  # (vehicle_id, distance, speed)
RGBColor = Tuple[int, int, int]
Corner = Tuple[float, float, float]  # (x, y, z)

# Constants
MAX_DISTANCE: int = 150  # meters
REFRESH_INTERVAL: float = 0.1  # only used when running run() loop
FOV_ANGLE: int = 25  # degrees (half-angle of cone)
POSITION_SNAP: float = 1.5  # meters: only save every 10 meters

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
        
        # Store filtered vehicles with unique IDs for next run
        self.filtered_vehicles: List[Vehicle] = []

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
        previous_score: Optional[float] = None,
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
        if previous_score is None:
            previous_score = self.vehicle_scores.get(vid, 0.0)

        # Check if vehicle has a trail/history
        history = self.vehicle_histories.get(vid, None)
        if history is not None and offset is None:
            # Has a trail but not intersecting with base: penalize
            score_increment = -0.7
        elif history is None:
            score_increment = -0.15
        else:
            score_increment = 0.0

        # Offset method
        if offset is not None:
            try:
                x = float(offset)
                offset_score = (2 ** (-(x / 9) ** 2) * 2.5 - 1.5) / 1
                distance_amp = (2 ** (-(distance / 100)) + 1 / ((distance + 3) / 8)-1)/3+1
                offset_score = max(-1.5, min(1, offset_score * distance_amp))
                score_increment += offset_score
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
            score_increment += angle_score
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
        path_width: float = 2
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
        path_width: float = 1.5,
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
        vehicles: List[Any],
        px: float,
        pz: float,
        yaw_rad: float,
        ego_steer: float = 0.0,
        ego_speed: float = 0.0,
        data: Optional[Dict[str, Any]] = None,
        cc_app=None
    ) -> List[Any]:
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect("ETS2 Radar")
        except:
            win_w, win_h = 600, 600

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
            path_pts, curvature = self.predict_ego_path_using_history(px, pz, yaw_rad, path_length=self.ego_path_length)
            # Draw ego path with 1.5m width
            self.draw_ego_path_with_width(img, path_pts, center, scale, path_width=1.5, color=(0, 200, 0), thickness=1, show_info=True, curvature=curvature)
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
            if hasattr(self, 'vehicle_acceleration_filtered'):
                self.vehicle_acceleration_filtered.pop(vid, None)
            if hasattr(self, 'vehicle_filtered_speed_history'):
                self.vehicle_filtered_speed_history.pop(vid, None)
            if hasattr(self, 'vehicle_speed_outlier_count'):
                self.vehicle_speed_outlier_count.pop(vid, None)
            if hasattr(self, 'vehicle_last_valid_speed'):
                self.vehicle_last_valid_speed.pop(vid, None)
            if hasattr(self, 'vehicle_speed_suspicious_timestamp'):
                self.vehicle_speed_suspicious_timestamp.pop(vid, None)
            if hasattr(self, 'vehicle_pending_speed'):
                self.vehicle_pending_speed.pop(vid, None)
            if hasattr(self, 'vehicle_speed_trend_history'):
                self.vehicle_speed_trend_history.pop(vid, None)
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

            # --- Update vehicle world pos history instead of screen pos history ---
            if vid not in self.vehicle_world_pos_history:
                self.vehicle_world_pos_history[vid] = collections.deque(maxlen=10)
            self.vehicle_world_pos_history[vid].append((dx, dz))

            # Filter and calculate acceleration using vehicle's speed data
            raw_vehicle_speed = getattr(v, 'speed', 0)  # Speed in m/s
            is_tmp = getattr(v, 'is_tmp', False)
            
            # Apply speed filtering
            filtered_vehicle_speed = self.filter_vehicle_speed(vid, raw_vehicle_speed, is_tmp)
            
            # Calculate acceleration using filtered speed data
            acceleration = self.calculate_vehicle_acceleration(vid, filtered_vehicle_speed, is_tmp)

            offset_for_score: Optional[float] = None
            angle_for_score: Optional[float] = None
            score_val = self.vehicle_scores.get(vid, 0.0)
            if self.is_in_front_cone(dx, dz):
                hist = self.vehicle_histories.get(vid, [])
                # Use filtered speed for scoring calculations
                if filtered_vehicle_speed < 2 and len(hist) == 0:
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
                        offset_for_score = offset - blinker_offset * 20
                    else:
                        offset_for_score = None
                    angle_for_score = angle
                previous_score = self.vehicle_scores.get(vid, 0.0)
                offset_score = self.calculate_offset_score(vid, offset_for_score, overall_closest_distance, angle_for_score, data)
                angle_score = self.calculate_angle_score(
                    vid,
                    path_pts,
                    ego_speed,
                    (dx, dz),
                    overall_closest_distance,
                )
                # NEW: Check if vehicle is in ego path and adjust score accordingly
                if path_pts:  # Only if we have a valid ego path
                    is_in_path = self.check_vehicle_in_ego_path(path_pts, poly, path_width=1)
                    if is_in_path:
                        path_score = min(0.3 * distance_amp, 10)  # Boost score for vehicles in ego path
                    else:
                        path_score = -min(0.3 * distance_amp, 3)  # Penalty for vehicles not in ego path
                else:
                    path_score = 0.0
                
                if self.reset_in_lane_scores:
                    score_val = -2.0  # Reset to -2 on blinker change
                else:
                    score_val = previous_score + (offset_score + angle_score + path_score) * max((abs(filtered_vehicle_speed)/90)**0.8, 0.5)

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
                        if path_pts and self.check_vehicle_in_ego_path(path_pts, poly, path_width=1):
                            vehicle_color = (0, 100, 255)  # Orange-red for vehicles in ego path
                        
                        self.draw_vehicle(img, poly, center, scale, vehicle_color)
                    else:
                        self.draw_vehicle(img, poly, center, scale, (0,0,100))
                else:
                    if score_val > 0:
                        # Use same color logic for trailers
                        trailer_color = (255, 0, 0)  # Default red for in-lane trailers
                        if path_pts and self.check_vehicle_in_ego_path(path_pts, poly, path_width=1):
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
                        if path_pts and self.check_vehicle_in_ego_path(path_pts, poly_t, path_width=1):
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
        
        # Filter out vehicles that come after a vehicle with both is_tmp and is_trailer
        filtered_vehicles = []
        skip_next = False
        
        for i, (v, dist, accel) in enumerate(in_lane_vehicles):
            if skip_next:
                skip_next = False
                continue
            
            # Check if current vehicle has both is_tmp and is_trailer
            if v.is_tmp and v.is_trailer:
                # Add this vehicle (the trailer) and skip the next one (the tractor)
                filtered_vehicles.append((v, dist, accel))
                
                if hasattr(v, 'trailer_count'):
                    v.trailer_count += 1
                else:
                    v.trailer_count = 1
                skip_next = True
            else:
                # Add vehicle normally
                filtered_vehicles.append((v, dist, accel))
        
        result = []
        for v, dist, accel in filtered_vehicles[:3]:
            # Use filtered speed if available, otherwise use raw speed
            vid = getattr(v, 'id', id(v))
            if hasattr(self, 'vehicle_filtered_speed_history') and vid in self.vehicle_filtered_speed_history:
                # Get the most recent filtered speed
                filtered_speeds = list(self.vehicle_filtered_speed_history[vid])
                if filtered_speeds:
                    speed_raw = filtered_speeds[-1]  # Most recent filtered speed in m/s
                else:
                    speed_raw = getattr(v, 'speed', 0)
            else:
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
        
    def calculate_vehicle_acceleration(self, vehicle_id: int, current_speed: float, is_tmp: bool = False) -> float:
        """
        Calculate vehicle acceleration using the vehicle's reported speed data.
        Returns acceleration in m/s².
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            current_speed: Current speed in km/h
            is_tmp: Whether this is a temporary/transient vehicle (applies advanced filtering)
        
        Advanced filtering (for is_tmp=True vehicles):
        - Multi-stage outlier detection using statistical methods
        - Adaptive thresholds based on recent acceleration patterns
        - Robust filtering that handles both sudden spikes and gradual drifts
        - Exponential moving average for smooth but responsive filtering
        """

        now = time.time()
        dt_frame = (min(now - self.last_timestamp, 0.5)+self.prev_dt_frame*10)/11
        self.prev_dt_frame = dt_frame
        self.last_timestamp = now

        # Initialize tracking dictionaries if not exists
        if not hasattr(self, 'vehicle_speed_history') or not hasattr(self, 'vehicle_acceleration_filtered'):
            self.vehicle_speed_history = {}
            self.vehicle_acceleration_filtered = {}
            self.vehicle_last_valid_acceleration = {}
            self.vehicle_suspicious_accel_timestamp = {}
            self.vehicle_pending_acceleration = {}
            self.vehicle_acceleration_history = {}
            self.vehicle_outlier_count = {}
            self.vehicle_acceleration_ema = {}
            self.vehicle_accel_confirmation_count = {}
        
        if not hasattr(self, 'vehicle_last_valid_acceleration'):
            self.vehicle_last_valid_acceleration = {}
        
        if not hasattr(self, 'vehicle_suspicious_accel_timestamp'):
            self.vehicle_suspicious_accel_timestamp = {}
        
        if not hasattr(self, 'vehicle_pending_acceleration'):
            self.vehicle_pending_acceleration = {}
            
        if not hasattr(self, 'vehicle_acceleration_history'):
            self.vehicle_acceleration_history = {}
            
        if not hasattr(self, 'vehicle_outlier_count'):
            self.vehicle_outlier_count = {}
            
        if not hasattr(self, 'vehicle_acceleration_ema'):
            self.vehicle_acceleration_ema = {}
            
        if not hasattr(self, 'vehicle_accel_confirmation_count'):
            self.vehicle_accel_confirmation_count = {}
        
        if vehicle_id not in self.vehicle_speed_history:
            self.vehicle_speed_history[vehicle_id] = collections.deque(maxlen=10)

        if vehicle_id not in self.vehicle_acceleration_filtered:
            self.vehicle_acceleration_filtered[vehicle_id] = 0.0
            
        if vehicle_id not in self.vehicle_acceleration_history:
            self.vehicle_acceleration_history[vehicle_id] = collections.deque(maxlen=20)
            
        if vehicle_id not in self.vehicle_outlier_count:
            self.vehicle_outlier_count[vehicle_id] = 0
            
        if vehicle_id not in self.vehicle_acceleration_ema:
            self.vehicle_acceleration_ema[vehicle_id] = 0.0
            
        if vehicle_id not in self.vehicle_accel_confirmation_count:
            self.vehicle_accel_confirmation_count[vehicle_id] = 0
        
        # Store current speed (convert from km/h to m/s)
        current_speed_ms = current_speed / 3.6
        
        # Add current speed to history
        self.vehicle_speed_history[vehicle_id].append(current_speed_ms)
        
        speeds = list(self.vehicle_speed_history[vehicle_id])
        
        # Need at least 5 speed measurements to calculate over 4 intervals
        if len(speeds) < 5:
            return 0.0
        
        # Get the oldest speed (first element) and newest speed (last element)
        old_speed = np.mean(speeds[:-1]) # average of all speeds except the last one
        new_speed = speeds[-1]  # Last/newest speed (current speed)
        
        time_interval = (len(speeds) - 1) * dt_frame  # Time between first and last measurement
        
        # Calculate raw acceleration
        raw_acceleration = (new_speed - old_speed) / max(time_interval, 0.01)
        
        # Apply different filtering strategies based on is_tmp flag (tmp stands for TrcukersMP)
        if is_tmp:
            # Basic filtering for permanent vehicles
            filtered_acceleration = self._apply_acceleration_filtering(
                vehicle_id, raw_acceleration, now
            )
        else:
            filtered_acceleration = raw_acceleration
        
        # Apply adaptive EMA smoothing
        ema_acceleration = self._apply_acceleration_ema(
            vehicle_id, filtered_acceleration
        )
        
        # Apply final low pass filter to smooth the acceleration output
        alpha = 0.5  # Filter coefficient (0 = no filtering, 1 = no smoothing)
        smoothed_acceleration = alpha * ema_acceleration + (1 - alpha) * self.vehicle_acceleration_filtered[vehicle_id]
        self.vehicle_acceleration_filtered[vehicle_id] = round(smoothed_acceleration, 4)
        
        return round(smoothed_acceleration, 4)
    
    def _apply_acceleration_ema(self, vehicle_id: int, acceleration: float) -> float:
        """
        Apply exponential moving average to acceleration with adaptive alpha.
        Uses faster response for sustained changes, slower for noise/outliers.
        """
        if self.vehicle_acceleration_ema[vehicle_id] == 0.0:
            # Initialize EMA with first value
            self.vehicle_acceleration_ema[vehicle_id] = acceleration
            return acceleration
        
        current_ema = self.vehicle_acceleration_ema[vehicle_id]
        accel_diff = abs(acceleration - current_ema)
        
        # Adaptive alpha based on consistency of change direction
        if accel_diff > 5.0:  # Large change detected
            # Check if change is consistent (real) or sporadic (noise/lag)
            if self.vehicle_accel_confirmation_count[vehicle_id] >= 2:
                # Sustained change confirmed - use faster response
                alpha = 0.4
            else:
                # Possible outlier - use slower response
                alpha = 0.1
                self.vehicle_accel_confirmation_count[vehicle_id] += 1
        else:
            # Normal change - balanced response
            alpha = 0.3
            self.vehicle_accel_confirmation_count[vehicle_id] = 0
        
        # Apply EMA: new_ema = alpha * new_value + (1 - alpha) * old_ema
        new_ema = alpha * acceleration + (1 - alpha) * current_ema
        self.vehicle_acceleration_ema[vehicle_id] = new_ema
        
        return new_ema
    
    def _apply_acceleration_filtering(self, vehicle_id: int, raw_acceleration: float, now: float) -> float:
        """
        Basic acceleration filtering for permanent vehicles.
        Simple outlier detection with fixed thresholds.
        """
        # Check if we have a previous valid acceleration to compare against
        if vehicle_id in self.vehicle_last_valid_acceleration:
            last_valid_accel = self.vehicle_last_valid_acceleration[vehicle_id]
            
            # Calculate the change in acceleration
            accel_change = abs(raw_acceleration - last_valid_accel)
            
            if accel_change > 10.0:  # m/s² - suspicious change detected
                # Track when suspicious acceleration was first detected
                if vehicle_id not in self.vehicle_suspicious_accel_timestamp:
                    self.vehicle_suspicious_accel_timestamp[vehicle_id] = now
                    self.vehicle_pending_acceleration[vehicle_id] = raw_acceleration
                
                # Check if 0.2s has passed since first suspicious detection
                time_since_suspicious = now - self.vehicle_suspicious_accel_timestamp[vehicle_id]
                
                if time_since_suspicious < 0.2:
                    # Use last valid acceleration instead of the suspicious one
                    return last_valid_accel
                else:
                    # 0.2s has passed, accept the new acceleration
                    self.vehicle_last_valid_acceleration[vehicle_id] = raw_acceleration
                    if vehicle_id in self.vehicle_suspicious_accel_timestamp:
                        del self.vehicle_suspicious_accel_timestamp[vehicle_id]
                    if vehicle_id in self.vehicle_pending_acceleration:
                        del self.vehicle_pending_acceleration[vehicle_id]
            else:
                # Normal acceleration change, update last valid and clear suspicious flag
                self.vehicle_last_valid_acceleration[vehicle_id] = raw_acceleration
                if vehicle_id in self.vehicle_suspicious_accel_timestamp:
                    del self.vehicle_suspicious_accel_timestamp[vehicle_id]
                if vehicle_id in self.vehicle_pending_acceleration:
                    del self.vehicle_pending_acceleration[vehicle_id]
        else:
            # First time seeing this vehicle, set initial acceleration
            self.vehicle_last_valid_acceleration[vehicle_id] = raw_acceleration
        
        return raw_acceleration
    
    def filter_vehicle_speed(self, vehicle_id: int, current_speed: float, is_tmp: bool = False) -> float:
        """
        Filter vehicle speed data to remove noise and outliers.
        Returns filtered speed in km/h.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            current_speed: Current speed in km/h
            is_tmp: Whether this is a temporary/transient vehicle (applies advanced filtering)
        
        Advanced filtering (for is_tmp=True vehicles):
        - Multi-stage outlier detection using statistical methods
        - Adaptive thresholds based on recent speed patterns
        - Robust filtering that handles both sudden spikes and gradual drifts
        - Physics-based validation (realistic speed changes)
        - Exponential moving average for smooth but responsive filtering
        """
        
        now = time.time()
        
        # Initialize tracking dictionaries if not exists
        if not hasattr(self, 'vehicle_filtered_speed_history'):
            self.vehicle_filtered_speed_history = {}
            self.vehicle_speed_outlier_count = {}
            self.vehicle_last_valid_speed = {}
            self.vehicle_speed_suspicious_timestamp = {}
            self.vehicle_pending_speed = {}
            self.vehicle_speed_trend_history = {}
            self.vehicle_speed_ema = {}
            self.vehicle_speed_confirmation_count = {}
        
        if not hasattr(self, 'vehicle_speed_outlier_count'):
            self.vehicle_speed_outlier_count = {}
            
        if not hasattr(self, 'vehicle_last_valid_speed'):
            self.vehicle_last_valid_speed = {}
            
        if not hasattr(self, 'vehicle_speed_suspicious_timestamp'):
            self.vehicle_speed_suspicious_timestamp = {}
            
        if not hasattr(self, 'vehicle_pending_speed'):
            self.vehicle_pending_speed = {}
            
        if not hasattr(self, 'vehicle_speed_trend_history'):
            self.vehicle_speed_trend_history = {}
            
        if not hasattr(self, 'vehicle_speed_ema'):
            self.vehicle_speed_ema = {}
            
        if not hasattr(self, 'vehicle_speed_confirmation_count'):
            self.vehicle_speed_confirmation_count = {}
        
        # Initialize vehicle-specific tracking
        if vehicle_id not in self.vehicle_filtered_speed_history:
            self.vehicle_filtered_speed_history[vehicle_id] = collections.deque(maxlen=15)
            
        if vehicle_id not in self.vehicle_speed_outlier_count:
            self.vehicle_speed_outlier_count[vehicle_id] = 0
            
        if vehicle_id not in self.vehicle_speed_trend_history:
            self.vehicle_speed_trend_history[vehicle_id] = collections.deque(maxlen=10)
            
        if vehicle_id not in self.vehicle_speed_ema:
            self.vehicle_speed_ema[vehicle_id] = current_speed
            
        if vehicle_id not in self.vehicle_speed_confirmation_count:
            self.vehicle_speed_confirmation_count[vehicle_id] = 0
        
        # Apply different filtering strategies based on is_tmp flag (tmp stands for TrcukersMP)
        if is_tmp:
            # Basic filtering for permanent vehicles
            filtered_speed = self._apply_speed_filtering(
                vehicle_id, current_speed, now
            )
        else:
            filtered_speed = current_speed
        
        # Apply adaptive EMA smoothing
        ema_speed = round(self._apply_speed_ema(vehicle_id, filtered_speed), 4)
        
        # Add filtered speed to history for future analysis
        self.vehicle_filtered_speed_history[vehicle_id].append(ema_speed)
        
        return ema_speed
    
    def _apply_speed_ema(self, vehicle_id: int, speed: float) -> float:
        """
        Apply exponential moving average to speed with adaptive alpha.
        Uses faster response for sustained changes, slower for noise/outliers.
        """
        current_ema = self.vehicle_speed_ema[vehicle_id]
        speed_diff = abs(speed - current_ema)
        
        # Adaptive alpha based on magnitude and consistency of change
        if speed_diff > 2.0:  # Significant change (>7 km/h)
            # Check if change is consistent (real) or sporadic (noise/lag)
            if self.vehicle_speed_confirmation_count[vehicle_id] >= 2:
                # Sustained change confirmed (likely real crash/acceleration) - fast response
                alpha = 0.5
            else:
                # Possible lag spike - slower response to avoid overreacting
                alpha = 0.2
                self.vehicle_speed_confirmation_count[vehicle_id] += 1
        else:
            # Normal driving - balanced response
            alpha = 0.3
            self.vehicle_speed_confirmation_count[vehicle_id] = 0
        
        # Apply EMA: new_ema = alpha * new_value + (1 - alpha) * old_ema
        new_ema = alpha * speed + (1 - alpha) * current_ema
        self.vehicle_speed_ema[vehicle_id] = new_ema
        
        return new_ema
    
    def _apply_speed_filtering(self, vehicle_id: int, current_speed: float, now: float) -> float:
        """
        Basic speed filtering for permanent vehicles.
        Simple outlier detection with fixed thresholds and physics-based validation.
        """
        # Physics-based validation: check for unrealistic speed values
        if current_speed < 0 or current_speed > 200:  # 0-200 m/s (0-720 km/h) reasonable range
            # Use last valid speed if available, otherwise return 0
            if vehicle_id in self.vehicle_last_valid_speed:
                return self.vehicle_last_valid_speed[vehicle_id]
            return 0.0
        
        # Check if we have a previous valid speed to compare against
        if vehicle_id in self.vehicle_last_valid_speed:
            last_valid_speed = self.vehicle_last_valid_speed[vehicle_id]
            
            # Calculate the change in speed
            speed_change = abs(current_speed - last_valid_speed)
            
            # Physics-based threshold: maximum realistic acceleration is ~10 m/s²
            # Assuming 0.2s time step, max speed change should be ~1 m/s
            max_realistic_change = 3.0  # m/s per frame
            
            if speed_change > max_realistic_change:
                # Track when suspicious speed was first detected
                if vehicle_id not in self.vehicle_speed_suspicious_timestamp:
                    self.vehicle_speed_suspicious_timestamp[vehicle_id] = now
                    self.vehicle_pending_speed[vehicle_id] = current_speed
                
                # Check if 0.2s has passed since first suspicious detection
                time_since_suspicious = now - self.vehicle_speed_suspicious_timestamp[vehicle_id]
                
                if time_since_suspicious < 0.2:
                    # Use last valid speed instead of the suspicious one
                    return last_valid_speed
                else:
                    # 0.2s has passed, accept the new speed
                    self.vehicle_last_valid_speed[vehicle_id] = current_speed
                    if vehicle_id in self.vehicle_speed_suspicious_timestamp:
                        del self.vehicle_speed_suspicious_timestamp[vehicle_id]
                    if vehicle_id in self.vehicle_pending_speed:
                        del self.vehicle_pending_speed[vehicle_id]
            else:
                # Normal speed change, update last valid and clear suspicious flag
                self.vehicle_last_valid_speed[vehicle_id] = current_speed
                if vehicle_id in self.vehicle_speed_suspicious_timestamp:
                    del self.vehicle_speed_suspicious_timestamp[vehicle_id]
                if vehicle_id in self.vehicle_pending_speed:
                    del self.vehicle_pending_speed[vehicle_id]
        else:
            # First time seeing this vehicle, set initial speed
            self.vehicle_last_valid_speed[vehicle_id] = current_speed
            
        return current_speed
    
    









    def predict_ego_path_using_history(
        self,
        px: float,
        pz: float,
        yaw_rad: float,
        path_length: float = 40.0,
        max_history: int = 25
    ):
        """
        Predict the most likely ego path using the past positions stored in self.ego_trajectory.

        Approach:
        - Use up to `max_history` past positions (already in world coords).
        - Transform them into ego-space (so ego is at origin facing +z).
        - Fit a circle to the transformed points using fit_circle(). If the fit succeeds,
          generate an arc forward from the ego origin along that circle.
        - If the fit fails or the radius is extremely large (near-straight), fallback to a straight line.
        - Return the generated path points (as ego-space (x, y) where y is forward) and an
          approximate signed curvature (1/radius, sign indicates turn direction).
        """
        # Gather history (most recent last)
        hist = list(self.ego_trajectory)[-max_history:]
        if len(hist) < 3:
            # Not enough data -> straight fallback
            pts = [(0.0, float(i)) for i in range(int(path_length) + 1)]
            return pts, 0.0

        # Transform history into ego-space (x lateral, z forward)
        ego_pts: List[Tuple[float, float]] = []
        for wx, wz in hist:
            dx = wx - px
            dz = wz - pz
            # Use same transform as used elsewhere: rotate_point(-dx, dz, -yaw)
            x_e, z_e = self.rotate_point(-dx, dz, -yaw_rad)
            ego_pts.append((x_e, z_e))

        xs = [p[0] for p in ego_pts]
        zs = [p[1] for p in ego_pts]

        fit = self.fit_circle(xs, zs)
        # If fit is not usable or radius is too large -> straight line
        if fit is None:
            # Straight forward path
            pts = [(0.0, float(i)) for i in range(int(path_length) + 1)]
            return pts, 0.0

        xc, zc, r = fit
        if r == 0 or r > 1e4:
            pts = [(0.0, float(i)) for i in range(int(path_length) + 1)]
            return pts, 0.0

        # Determine direction of travel along circle from history (sign of curvature)
        # Use the last two history vectors from center to decide direction
        x_last, z_last = ego_pts[-1]
        x_prev, z_prev = ego_pts[-2]
        v_end = np.array([x_last - xc, z_last - zc])
        v_prev = np.array([x_prev - xc, z_prev - zc])
        cross = v_end[0] * v_prev[1] - v_end[1] * v_prev[0]
        direction = 1 if cross > 0 else -1

        # Angle of the vector from center to the ego origin (0,0)
        theta0 = np.arctan2(-zc, -xc)  # angle from center to origin

        # Generate arc points forward from the ego origin along the fitted circle
        num_points = max(2, int(path_length)) * 2  # denser sampling
        arc_points: List[Tuple[float, float]] = []
        for s in np.linspace(0.0, path_length, num_points):
            # param angle step based on arc length s = r * delta_theta
            delta_theta = s / r
            angle = theta0 - direction * delta_theta
            x = xc + r * np.cos(angle)
            z = zc + r * np.sin(angle)
            # Convert to ego-space coords where forward is positive y
            arc_points.append((float(x), float(z)))

        # Signed curvature
        curvature = (direction * (1.0 / r)) if r != 0 else 0.0
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
        yawn = data.get("rotationX", 0.0) + 0.5
        yawr = yawn * 2 * np.pi
        ego_steer = data.get("gameSteer", 0.0)
        speed = data.get("speed", 0.0)
        self.ego_path_length = MAX_DISTANCE
        self.update_ego_trajectory((px, pz))

        vehicles = self.module.run()
        # Filter vehicles not on the truck's plane
        vehicles = [v for v in vehicles if abs(v.position.y - data.get("coordinateY", 0.0)) < 6.0]
        
        # Handle duplicate vehicle IDs
        vehicles = self.handle_duplicate_vehicle_ids(vehicles)
        
        # Save filtered vehicles for next run
        self.filtered_vehicles = vehicles

        # Update histories with world coordinates
        if not data["paused"]:
            for v in vehicles:
                vid = getattr(v, 'id', id(v))
                # Apply speed filtering and update speed storage
                raw_speed = getattr(v, 'speed', 0)
                is_tmp = getattr(v, 'is_tmp', False)
                filtered_speed = self.filter_vehicle_speed(vid, raw_speed, is_tmp)
                
                # Only update history for moving vehicles (filtered speed > 5)
                # Stationary vehicles will get synthetic trajectories in draw_radar
                if filtered_speed > 5:
                    poly = self.get_vehicle_polygon(v, px, pz, yawr)
                    cx = np.mean([c[0] for c in v.get_corners()])
                    cz = np.mean([c[2] for c in v.get_corners()])
                    self.update_vehicle_history(vid, (cx, cz))
                # Always update speed for all vehicles (store filtered speed)
                self.vehicle_speeds[vid] = filtered_speed

        # Main detection and visualization
        in_lane_vehicles = self.draw_radar(vehicles, px, pz, yawr, ego_steer, speed, data, cc_app)
        return in_lane_vehicles

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