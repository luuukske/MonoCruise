"""Background-loaded advisory road geometry, never a braking authority."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
import math
from pathlib import Path
import threading

from core import settings
from core.road_map.boundaries import assess, dimensions, offset_edges
from core.road_map.template_extents import resolve_profiles, road_extent
from core.road_map.validation import instance_check, profile_checks, road_check

_log = logging.getLogger(__name__)
CELL = 128
MAX_CANDIDATES = 512
PREVIEW_BUCKET_LIMIT = 1024
PREVIEW_DRAW_LIMIT = 2048


class RoadMap:
    def __init__(self, payload):
        if payload.get("schema") != 1 or payload.get("game") != "ETS2":
            raise ValueError("Unsupported map format")
        if (payload.get("source") == "trucklib-mapexporter" and
                not any(road.get("kind") == "road" for road in payload["roads"])):
            raise ValueError("Incomplete road map: run Repair-Road-Map.cmd")
        self.version = str(payload.get("version", "unknown"))
        self.grid = defaultdict(list)
        self.preview_grid = defaultdict(list)
        self.edge_grid = defaultdict(list)
        self.model_edge_grid = defaultdict(list)
        self.boundary_widths = {}
        self.boundary_ends = {}
        self.model_widths = {}
        self.model_ends = {}
        profiles = resolve_profiles(payload)
        checks = profile_checks(payload, profiles)
        self.model_validation = {}
        check_cache = {}
        count = 0
        for road in payload["roads"]:
            points = road["points"]
            widths = dimensions(road)
            if widths is not None and len(points) >= 2:
                edges = offset_edges(points, widths)
                if edges:
                    self.boundary_widths[road['id']] = widths
                    self.boundary_ends[road['id']] = (points[0], points[-1])
                    self._index_edges(edges, self.edge_grid)
            model_widths = road_extent(road, profiles) if widths is None else None
            if model_widths is not None:
                key = road.get('road_type'), road.get('variant_right'), instance_check(road)
                if key not in check_cache:
                    check_cache[key] = road_check(road, checks)
                check = check_cache[key]
                self.model_validation[road['id']] = check
                if check['state'] == 'inactive_collision_part':
                    model_widths = None
            if model_widths is not None and len(points) >= 2:
                edges = offset_edges(points, model_widths)
                if edges:
                    self.model_widths[road['id']] = model_widths
                    self.model_ends[road['id']] = (points[0], points[-1])
                    self._index_edges(edges, self.model_edge_grid)
            visual_points = points[::4]
            if visual_points and visual_points[-1] != points[-1]:
                visual_points.append(points[-1])
            for a, b in zip(visual_points, visual_points[1:]):
                if not all(math.isfinite(float(v)) for v in a + b):
                    continue
                if math.hypot(b[0]-a[0], b[1]-a[1]) > 200:
                    continue
                for gx in range(math.floor(min(a[0], b[0])/CELL), math.floor(max(a[0], b[0])/CELL)+1):
                    for gz in range(math.floor(min(a[1], b[1])/CELL), math.floor(max(a[1], b[1])/CELL)+1):
                        bucket = self.preview_grid[gx, gz]
                        if len(bucket) < PREVIEW_BUCKET_LIMIT:
                            bucket.append((tuple(a), tuple(b)))
            for a, b in zip(points, points[1:]):
                if not all(math.isfinite(float(v)) for v in a + b):
                    continue
                dx, dz = b[0]-a[0], b[1]-a[1]
                length = math.hypot(dx, dz)
                if not 0.01 < length <= 50:
                    continue
                segment = (road["id"], int(road["lanes"]), road.get("kind", "road"), tuple(a), tuple(b), length)
                minx, maxx = math.floor((min(a[0], b[0])-40)/CELL), math.floor((max(a[0], b[0])+40)/CELL)
                minz, maxz = math.floor((min(a[1], b[1])-40)/CELL), math.floor((max(a[1], b[1])+40)/CELL)
                for x in range(minx, maxx+1):
                    for z in range(minz, maxz+1):
                        bucket = self.grid[x, z]
                        if len(bucket) <= MAX_CANDIDATES:
                            bucket.append(segment)
                count += 1
        if count == 0:
            raise ValueError("No usable road segments")
        self.count = count
        self.surface_widths = self.boundary_widths | self.model_widths
        self.surface_ends = self.boundary_ends | self.model_ends
        self.validation_counts = Counter(check['state'] for check in self.model_validation.values())

    @staticmethod
    def _index_edges(edges, grid):
        for edge in edges:
            visual_edge = edge[::4]
            if visual_edge[-1] != edge[-1]:
                visual_edge.append(edge[-1])
            for a, b in zip(visual_edge, visual_edge[1:]):
                if not all(math.isfinite(v) for v in a+b) or math.hypot(b[0]-a[0], b[1]-a[1]) > 200:
                    continue
                for gx in range(math.floor(min(a[0], b[0])/CELL), math.floor(max(a[0], b[0])/CELL)+1):
                    for gz in range(math.floor(min(a[1], b[1])/CELL), math.floor(max(a[1], b[1])/CELL)+1):
                        bucket = grid[gx, gz]
                        if len(bucket) < PREVIEW_BUCKET_LIMIT:
                            bucket.append((a, b))

    def match(self, x, z, elevation, yaw):
        if not all(math.isfinite(v) for v in (x, z, elevation, yaw)):
            return None
        candidates = self.grid.get((math.floor(x/CELL), math.floor(z/CELL)), [])
        if len(candidates) > MAX_CANDIDATES:
            return None
        forward = (-math.sin(yaw), -math.cos(yaw))
        matches = {}
        for identity, lanes, kind, a, b, length in candidates:
            dx, dz = b[0]-a[0], b[1]-a[1]
            heading = abs((forward[0]*dx + forward[1]*dz)/length)
            if heading < 0.7:
                continue
            t = max(0, min(1, ((x-a[0])*dx + (z-a[1])*dz)/(length*length)))
            height = a[2] + t*(b[2]-a[2])
            if abs(elevation-height) > 5:
                continue
            distance = math.hypot(x-a[0]-t*dx, z-a[1]-t*dz)
            if distance > 40:
                continue
            result = {"road_id": identity, "distance_m": round(distance, 2), "lanes": lanes, "kind": kind}
            if identity not in matches or distance < matches[identity][0]:
                matches[identity] = (distance, result)
        ordered = sorted(matches.values(), key=lambda item: item[0])
        if not ordered or (len(ordered)>1 and ordered[1][0]-ordered[0][0] < 2):
            return None
        return ordered[0][1]


    def nearby_junctions(self, x, z, elevation):
        if not all(math.isfinite(v) for v in (x, z, elevation)):
            return None
        candidates = self.grid.get((math.floor(x/CELL), math.floor(z/CELL)), [])
        if len(candidates) > MAX_CANDIDATES:
            return None
        matches = {}
        for identity, _, kind, a, b, length in candidates:
            if kind != "junction":
                continue
            dx, dz = b[0]-a[0], b[1]-a[1]
            t = max(0, min(1, ((x-a[0])*dx + (z-a[1])*dz)/(length*length)))
            if abs(elevation-a[2]-t*(b[2]-a[2])) > 5:
                continue
            distance = math.hypot(x-a[0]-t*dx, z-a[1]-t*dz)
            if distance <= 40:
                matches[identity] = min(distance, matches.get(identity, math.inf))
        return [{"route_id": identity, "distance_m": round(distance, 2)}
                for identity, distance in sorted(matches.items(), key=lambda item: item[1])[:16]]


class MapService:
    def __init__(self):
        self._lock = threading.Lock()
        self._map = None
        self.status = "Map not loaded"
        self._loading = False

    def load(self, path: Path | None = None):
        path = path or settings.CONFIG_PATH.parent / "maps" / "roads.json"
        with self._lock:
            if self._loading:
                return
            self._loading = True
            self.status = "Loading road map..."
        threading.Thread(target=self._load, args=(path,), daemon=True).start()

    def _load(self, path):
        result = None
        try:
            with path.open() as stream:
                payload = json.load(stream)
            if payload.get("source") == "trucklib-mapexporter":
                raise ValueError("Old export is unreliable: run Prepare-TruckLib-Map.cmd")
            result = RoadMap(payload)
            status = (f"ETS2 {result.version}: {result.count} segments loaded (advisory); "
                      f"widths: {len(result.boundary_widths)} roads; "
                      f"model surface candidates: {len(result.model_widths)} roads; "
                      f"surface geometry checked: {result.validation_counts['surface_geometry_checked']}; "
                      f"pending evidence: {sum(value for key, value in result.validation_counts.items() if key.startswith('needs_'))}")
        except FileNotFoundError:
            status = "No map: run Prepare-Road-Map.cmd"
        except ValueError as exc:
            status = str(exc)
        except Exception:
            _log.warning("Road map could not be loaded")
            status = "Invalid map; AEB unchanged"
        with self._lock:
            self._map, self.status, self._loading = result, status, False

    def match(self, x, z, elevation, yaw):
        with self._lock:
            road_map = self._map
        if road_map is None:
            return None
        result = road_map.match(x, z, elevation, yaw)
        if result is not None:
            result['surface_validation'] = road_map.model_validation.get(result['road_id'])
        return result

    def validation_at(self, x, z, elevation, yaw):
        result = self.match(x, z, elevation, yaw)
        return result.get('surface_validation') if result is not None else None


    def nearby_junctions(self, x, z, elevation):
        with self._lock:
            road_map = self._map
        return road_map.nearby_junctions(x, z, elevation) if road_map else None


    def preview_segments(self, x, z, elevation):
        return self._preview(x, z, elevation, 'preview_grid')

    def preview_boundaries(self, x, z, elevation):
        return self._preview(x, z, elevation, 'edge_grid')

    def preview_model_extents(self, x, z, elevation, *, all_heights=False):
        return self._preview(x, z, elevation, 'model_edge_grid', same_height=not all_heights)

    def assess_roadside(self, x, z, elevation, radius, speed):
        with self._lock:
            road_map = self._map
        if road_map is None:
            return {'state': 'unknown'}
        result = assess(road_map, x, z, elevation, radius, speed, CELL, MAX_CANDIDATES)
        if road_map.model_widths:
            model = assess(road_map, x, z, elevation, radius, speed, CELL, MAX_CANDIDATES,
                           widths=road_map.surface_widths, ends=road_map.surface_ends)
            model.update(source='model_surface_candidate', validated=False, advisory_only=True)
            model['surface_validation'] = road_map.model_validation.get(model.get('road_id'))
            result['model_surface'] = model
        return result

    def _preview(self, x, z, elevation, grid_name, same_height=False):
        if not all(math.isfinite(v) for v in (x, z, elevation)):
            return []
        with self._lock:
            road_map = self._map
        if road_map is None:
            return []
        gx, gz = math.floor(x/CELL), math.floor(z/CELL)
        segments = {}
        for ix in range(gx-1, gx+2):
            for iz in range(gz-1, gz+2):
                for a, b in getattr(road_map, grid_name).get((ix, iz), []):
                    dx, dz = b[0]-a[0], b[1]-a[1]
                    length_sq = dx*dx+dz*dz
                    if length_sq <= 0:
                        continue
                    t = max(0, min(1, ((x-a[0])*dx+(z-a[1])*dz)/length_sq))
                    if same_height and abs(elevation-a[2]-t*(b[2]-a[2])) > 5:
                        continue
                    distance = math.hypot(x-a[0]-t*dx, z-a[1]-t*dz)
                    if distance <= 120:
                        segments[a, b] = distance
        return [segment for segment, _ in
                sorted(segments.items(), key=lambda item: item[1])[:PREVIEW_DRAW_LIMIT]]


service = MapService()
