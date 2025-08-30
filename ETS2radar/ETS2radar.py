from typing import List, Tuple, Dict, Optional, Any, Deque, Union
import cv2
import numpy as np
import numpy.typing as npt
import truck_telemetry
from shapely.geometry import Polygon, Point
from ETS2radar.main import Module
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
REFRESH_INTERVAL: float = 0.1  # seconds
FOV_ANGLE: int = 25  # degrees (half-angle of cone)

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
        self.vehicle_speeds: Dict[int, float] = {}
        self.lane_state_tracker: Dict[int, VehicleTracker] = {}
        self.ego_path_tracker: Dict[int, EgoPathTracker] = {}
        self.ego_path_length: float = MAX_DISTANCE

        # Scoring system for vehicles (used for robust lane detection)
        self.vehicle_scores: Dict[int, float] = collections.defaultdict(float)

        # Setup window if requested
        if self.show_window:
            cv2.namedWindow("ETS2 Radar", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ETS2 Radar", 600, 600)

        print("ETS2 Radar initialized.")

    def calculate_score(
        self,
        vid: int,
        offset: Optional[float],
        previous_score: Optional[float] = None,
        distance: float = 30
    ) -> float:
        """
        Calculate the new score for a vehicle, including previous score.

        The offset score formula is:
            f(x) = 2^(-((x/10)^2))*2 - 1

        The score to be added to previous is clamped between -1 and 1.
        The cumulative score is clamped between -5 and +5.

        Args:
            vid: Vehicle ID.
            offset: Offset value from fitted arc (None if not available).
            previous_score: Optional previous score for this vehicle.

        Returns:
            The new clamped score for this vehicle.
        """
        if previous_score is None:
            previous_score = self.vehicle_scores.get(vid, 0.0)

        score_increment = -0.1

        # Offset method
        if offset is not None:
            try:
                x = float(offset)
                offset_score = (2 ** (-(x / 15) ** 2) * 2 - 1)/2
                distance_amp = (2**(-(distance/100))+1/((distance+3)/10))
                # Clamp each score increment to [-1, 1]
                offset_score = max(-1.0, min(1.0, offset_score*distance_amp))
                score_increment += offset_score
            except Exception:
                pass
        else:
            score_increment += -0.5

        # Clamp total score to [-10, +10]
        # Adjust score change rate: decrease faster if previous_score > 0, increase faster if previous_score < 0
        if (previous_score > 0 and score_increment < 0) or (previous_score < 0 and score_increment > 0):
            new_score = previous_score + score_increment * 1.5  # slower increase, faster decrease
        else:
            new_score = previous_score + score_increment
        new_score = max(-10.0, min(10.0, new_score))
        return new_score

    def world_to_screen(
        self,
        dx: float,
        dz: float,
        center: ScreenCoordinate,
        scale: float
    ) -> ScreenCoordinate:
        """
        Convert a world-relative point (dx, dz) to screen coordinates.
        Used for visualization.
        """
        cx, cy = center
        sx: int = int(cx + dx * scale)
        sy: int = int(cy - dz * scale)
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

        if len(xs_array) < 10:
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
        Used both for lane detection and visualization.

        Returns:
            tuple: (intersection_point, normalized_angle)
        """
        base_radius: float = 10.0 * scale
        r_steer: float = base_radius / abs(ego_steer) if abs(ego_steer) != 0.0 else 0.0

        if len(history) < 10:
            return None, None

        xs, zs = zip(*history)
        fit_result = self.fit_circle(list(xs), list(zs))
        if fit_result is None:
            return None, None

        xc, zc, r = fit_result
        x_end, z_end = xs[-1], zs[-1]
        x_prev, z_prev = xs[-2], zs[-2]

        v_end: npt.NDArray[np.float64] = np.array([x_end - xc, z_end - zc])
        v_prev: npt.NDArray[np.float64] = np.array([x_prev - xc, z_prev - zc])
        theta_end: float = np.arctan2(v_end[1], v_end[0])
        cross: float = v_end[0]*v_prev[1] - v_end[1]*v_prev[0]
        direction: int = 1 if cross > 0 else -1

        if reverse:
            direction *= -1

        arc_angle: float = arc_length / r if r != 0 else 0.0
        arc_points: List[ScreenCoordinate] = []
        num_points: int = 30

        for i in range(num_points + 1):
            t: float = i / num_points
            angle: float = theta_end - direction * t * arc_angle
            x: float = xc + r * np.cos(angle)
            z: float = zc + r * np.sin(angle)
            dx: float = x - px
            dz: float = z - pz
            dxr, dzr = self.rotate_point(-dx, dz, -yaw_rad)
            arc_points.append(self.world_to_screen(dxr, dzr, center, scale))

        # Draw arc for visualization only
        if self.show_window and arc_points:
            arr: npt.NDArray[np.int32] = np.array(arc_points, dtype=np.int32)
            cv2.polylines(img, [arr], isClosed=False, color=color, thickness=1)
            curvature: float = 1.0 / r if r != 0 else 0.0
            cv2.putText(img, f"curv={curvature:.3f}", arc_points[-1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Find intersection with screen bottom
        intersection_point: Optional[float] = None
        intersection_angle: Optional[float] = None
        screen_height: int = img.shape[0] if self.show_window else 600

        for i, (x, y) in enumerate(arc_points):
            if y >= screen_height:
                intersection_point = float(x - 300)
                if i > 0:
                    prev_point = arc_points[i-1]
                    tangent_x: float = float(x - prev_point[0])
                    tangent_y: float = float(y - prev_point[1])
                    tangent_angle: float = np.arctan2(tangent_y, tangent_x)
                    baseline_angle: float = abs(tangent_angle)
                    if baseline_angle > np.pi/2:
                        baseline_angle = np.pi - baseline_angle
                    normalized_angle: float = 2.0 - (2.0 * baseline_angle / (np.pi/2))
                    if tangent_x < 0:
                        normalized_angle = -abs(normalized_angle)
                    else:
                        normalized_angle = abs(normalized_angle)
                    intersection_angle = normalized_angle
                break

        return intersection_point, intersection_angle

    def draw_radar(
        self,
        vehicles: List[Any],
        px: float,
        pz: float,
        yaw_rad: float,
        ego_steer: float = 0.0,
        ego_speed: float = 0.0
    ) -> List[VehicleData]:
        """
        Render the radar and return in-lane vehicle info. Also updates vehicle scores.
        This is the main in-lane vehicle detection logic.
        """

        # Get window size (or fallback to defaults)
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect("ETS2 Radar")
        except:
            win_w, win_h = 600, 600

        # Scale so that z = max_distance → y = 0 (top of window)
        scale = win_h / self.max_distance
        center = (win_w // 2, win_h)

        # Blank canvas
        img = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        # Create FOV mask
        mask = np.zeros((win_h, win_w), dtype=np.uint8)
        cone_len = int(self.max_distance * scale)
        pts = [(center[0], center[1])]
        for a in np.linspace(-np.radians(self.fov_angle), np.radians(self.fov_angle), 200):
            x = int(center[0] + cone_len * np.sin(a))
            y = int(center[1] - cone_len * np.cos(a))
            pts.append((x, y))
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

        # Draw range rings
        for r in range(25, self.max_distance + 1, 25):
            try:
                radius = int(r * scale)
                tmp = np.zeros_like(img)
                cv2.circle(tmp, center, radius, (100, 100, 100), 1)
                img = cv2.bitwise_or(img, cv2.bitwise_and(tmp, tmp, mask=mask))
            except:
                continue

        # Draw crosshairs
        tmp = np.zeros_like(img)
        cv2.line(tmp, (center[0], 0), (center[0], win_h), (100, 100, 100), 1)
        cv2.line(tmp, (0, center[1]), (win_w, center[1]), (100, 100, 100), 1)
        img = cv2.bitwise_or(img, cv2.bitwise_and(tmp, tmp, mask=mask))

        # Draw FOV cone outline and ego-truck dot
        self.draw_fov_cone(img, center, scale)
        cv2.circle(img, center, 5, (0, 255, 255), -1)

        # Collect all vehicle polygons and check for collisions
        origin = Point(0, 0)
        vehicle_data = []
        for v in vehicles:
            poly = self.get_vehicle_polygon(v, px, pz, yaw_rad)
            closest_distance = min(Point(coord).distance(origin) for coord in poly.exterior.coords[:-1])
            if closest_distance <= self.max_distance:
                vehicle_data.append({
                    'vehicle': v,
                    'polygon': poly,
                    'distance': closest_distance,
                    'centroid': poly.centroid,
                    'trailers': []
                })
                # Add trailers to the same vehicle group
                for t in getattr(v, 'trailers', []):
                    poly_t = self.get_vehicle_polygon(t, px, pz, yaw_rad)
                    trailer_closest_distance = min(Point(coord).distance(origin) for coord in poly_t.exterior.coords[:-1])
                    if trailer_closest_distance <= self.max_distance:
                        vehicle_data[-1]['trailers'].append({
                            'trailer': t,
                            'polygon': poly_t,
                            'distance': trailer_closest_distance
                        })

        # Detect colliding vehicles and group them (used for clutter reduction)
        collision_threshold = 0.5
        vehicle_groups = []
        processed = set()
        for i, vdata in enumerate(vehicle_data):
            if i in processed:
                continue
            group = [vdata]
            processed.add(i)
            for j, other_vdata in enumerate(vehicle_data):
                if j in processed or i == j:
                    continue
                if vdata['polygon'].distance(other_vdata['polygon']) < collision_threshold:
                    group.append(other_vdata)
                    processed.add(j)
            vehicle_groups.append(group)

        # Generate the ego steering path polygon (used for future expansion/path collision)
        if not hasattr(self, 'ego_path_length'):
            self.ego_path_length = MAX_DISTANCE # 40 + int(ego_speed) * 0.8
        ego_path_points = self.generate_ego_steering_path(ego_steer, self.ego_path_length)
        ego_path_polygon = self.create_path_polygon(ego_path_points, width=1)
        self.draw_ego_path(img, ego_path_points, center, scale)

        # Main vehicle-in-lane detection loop
        in_lane_vehicles = []
        for group in vehicle_groups:
            # For colliding vehicles, only keep the closest one
            if len(group) > 1:
                group = [min(group, key=lambda x: x['distance'])]
            main_vehicle = group[0]
            v = main_vehicle['vehicle']
            poly = main_vehicle['polygon']
            distance = main_vehicle['distance']
            dx, dz = main_vehicle['centroid'].x, main_vehicle['centroid'].y
            # Use closest trailer distance if any
            overall_closest_distance = distance
            for trailer_data in main_vehicle['trailers']:
                trailer_distance = trailer_data['distance']
                if trailer_distance < overall_closest_distance:
                    overall_closest_distance = trailer_distance
            vid = getattr(v, 'id', id(v))

            # Lane detection only by trajectory, no path collision
            offset_for_score: Optional[float] = None
            score_val = self.vehicle_scores.get(vid, 0.0)
            # Only consider in-lane if score > 0
            if self.is_in_front_cone(dx, dz):
                hist = self.vehicle_histories.get(vid, [])
                vehicle_speed = getattr(v, 'speed', 0)
                if vehicle_speed < 2 and len(hist) == 0:
                    # Synthetic trajectory for stationary vehicles
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
                    if forward_length > 0:
                        forward_x /= forward_length
                        forward_z /= forward_length
                        synthetic_hist = []
                        rear_x = rear_center_x
                        rear_z = rear_center_z
                        for i in range(10):
                            back_distance = i * 5
                            hist_x = rear_x - forward_x * back_distance
                            hist_z = rear_z - forward_z * back_distance
                            synthetic_hist.insert(0, (hist_x, hist_z))
                        synthetic_hist.append((rear_x, rear_z))
                        synthetic_hist.append((cx, cz))
                        hist = synthetic_hist
                if hist:
                    offset, angle = self.draw_fitted_arc(img, hist, px, pz, yaw_rad, center, scale,
                                                arc_length=130, color=(0,80,80), reverse=True, ego_steer=ego_steer)
                    offset_for_score = offset
                previous_score = self.vehicle_scores.get(vid, 0.0)
                score_val = self.calculate_score(vid, offset_for_score, previous_score, overall_closest_distance)
                self.vehicle_scores[vid] = score_val
            # Lane status for output (score-based)
            if self.is_in_front_cone(dx, dz) and score_val > 0:
                in_lane_vehicles.append((v, overall_closest_distance - 4.2))

            # Draw vehicle based on lane status
            if self.is_in_front_cone(dx, dz):
                if score_val > 0:
                    self.draw_vehicle(img, poly, center, scale, (0,0,255))
                    # Draw score text
                    veh_scr_x, veh_scr_y = self.world_to_screen(dx, dz, center, scale)
                    cv2.putText(img, f"score:{score_val:.2f}", (veh_scr_x-25, veh_scr_y-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
                else:
                    self.draw_vehicle(img, poly, center, scale, (0,0,100))
            else:
                self.draw_vehicle(img, poly, center, scale, (200,200,0))
            for trailer_data in main_vehicle['trailers']:
                poly_t = trailer_data['polygon']
                tx, tz = poly_t.centroid.x, poly_t.centroid.y
                if self.is_in_front_cone(tx, tz):
                    if score_val > 0:
                        self.draw_vehicle(img, poly_t, center, scale, (255,0,0))
                    else:
                        self.draw_vehicle(img, poly_t, center, scale, (100,0,0))

        # Sort and output only the closest 3 vehicles in-lane
        in_lane_vehicles.sort(key=lambda x: x[1])
        result = []
        for v, dist in in_lane_vehicles[:3]:
            speed_raw = getattr(v, 'speed', 0)
            speed_kmh = speed_raw * 3.6
            result.append((v.id, dist, speed_kmh))

        # Show radar window if enabled
        if self.show_window:
            cv2.imshow("ETS2 Radar", img)

        return result

    def generate_ego_steering_path(
        self,
        ego_steer,
        path_length,
        wheelbase=4.0,
        max_steer_angle_deg=40.0
    ):
        """
        Generate path points using bicycle model based on steering input.
        Used for drawing ego path and for future collision prediction.
        """
        points = [(0.0, 0.0)]
        max_steer_angle = np.radians(max_steer_angle_deg)
        delta = -ego_steer * max_steer_angle  # map [-1, 1] -> [-max_angle, max_angle]

        if abs(delta) < 1e-3:
            for i in range(1, int(path_length)):
                points.append((0.0, float(i)))
            return points

        curvature = np.tan(delta) / wheelbase
        radius = 1.0 / curvature

        step = 1.0  # step size in meters
        num_steps = int(path_length / step)

        heading = 0.0
        x, y = 0.0, 0.0

        for _ in range(num_steps):
            heading += curvature * step
            x += step * np.sin(heading)
            y += step * np.cos(heading)
            points.append((x, y))

        return points

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
        img,
        path_points,
        center,
        scale
    ):
        """
        Draw the ego vehicle's predicted path on the radar.
        """
        if len(path_points) < 2:
            return
        # Convert path points to screen coordinates
        screen_points = []
        for x, y in path_points:
            screen_x = int(center[0] + x * scale)
            screen_y = int(center[1] - y * scale)
            screen_points.append((screen_x, screen_y))
        # Draw path as connected lines
        for i in range(len(screen_points) - 1):
            cv2.line(img, screen_points[i], screen_points[i+1], (0, 255, 0), 1)
        # Draw path boundaries (optional - shows width)
        # This would require converting the path polygon to screen coordinates

    def update(self, data=None):
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
        self.ego_path_length = MAX_DISTANCE # 40 + int(speed) * 0.8  # Dynamic path length based on speed

        vehicles = self.module.run()
        # Filter vehicles not on the truck's plane
        vehicles = [v for v in vehicles if abs(v.position.y - data.get("coordinateY", 0.0)) < 6.0]

        # Update histories with world coordinates
        if not data["paused"]:
            for v in vehicles:
                vid = getattr(v, 'id', id(v))
                # Only update history for moving vehicles (speed > 5)
                # Stationary vehicles will get synthetic trajectories in draw_radar
                if v.speed > 5:
                    poly = self.get_vehicle_polygon(v, px, pz, yawr)
                    cx = np.mean([c[0] for c in v.get_corners()])
                    cz = np.mean([c[2] for c in v.get_corners()])
                    self.vehicle_histories[vid].append((cx, cz))
                # Always update speed for all vehicles
                self.vehicle_speeds[vid] = getattr(v, 'speed', 0)

        # Main detection and visualization
        in_lane_vehicles = self.draw_radar(vehicles, px, pz, yawr, ego_steer, speed)
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