"""Per-vehicle in-path tracker: ego arc, scoring, top-3 leads, trailer swap.

Frame pipeline and blinker behaviour: ``core/acc/README.md`` §3–5."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from core.radar.traffic import Vehicle

from .ego_path import build_ego_arc, path_half_width
from .scoring import (
    IN_PATH_THRESHOLD,
    LEGACY_RATE_HZ,
    OFFSET_BASELINE_HIT,
    OFFSET_BASELINE_NO_ARC_HIT,
    OFFSET_BASELINE_NO_HISTORY,
    OFFSET_WEIGHT,
    ScoreComponents,
    accumulate,
    offset_component,
    path_component,
    speed_multiplier,
    yaw_component,
)
from .trail_arc import angle_amp_from, crossing_offset_and_angle, fit_trail


logger = logging.getLogger(__name__)


# Filter bounds: vehicles outside these are never scored.
_MAX_SCORE_RANGE_M: float = 150.0      # longitudinal cut-off.
_REAR_DOT_THRESHOLD: float = -0.2      # rear half cone: fwd-dot below → skip.
# Pitch elevation gate: same margin as AEB; README §8 / core/aeb (pitch-projected y).
_ELEVATION_MARGIN_M: float = 5.0

# Missing-target decay: same rate as out-of-path per frame so we don't
# pile artificial penalties onto a briefly occluded car.
_MISSING_OUT_DECAY_S: float = 2.0      # expires track after this long missing.

# Blinker cos decay; lateral shift via offset - scalar·4.5 m (README §5).
_BLINKER_HOLD_S: float = 2.5
_BLINKER_OFFSET_M: float = 4.5

# Highway lane change reset: zero all scores on blinker rising edge
# above this ego speed so a new lead can lock cleanly on the new side.
_BLINKER_SCORE_RESET_KMH: float = 65.0

# TMP trailer→tractor inference; strict acquire / loose revalidate (README §4).
_TRACTOR_LOCK_LONGI_MIN_M: float = 3.0
_TRACTOR_LOCK_LONGI_MAX_M: float = 16.0
_TRACTOR_LOCK_LAT_MAX_M: float = 1.5
_TRACTOR_LOCK_YAW_MAX_DEG: float = 15.0

_TRACTOR_LOCK_VALID_LONGI_MIN_M: float = 1.0
_TRACTOR_LOCK_VALID_LONGI_MAX_M: float = 25.0
_TRACTOR_LOCK_VALID_LAT_MAX_M: float = 4.0
_TRACTOR_LOCK_VALID_YAW_MAX_DEG: float = 60.0

# Synthetic nested-trailer ids; skip tractor lock (own acc_speed chain).
_TRAILER_VEHICLE_ID_BASE: int = 1_000_000


@dataclass(slots=True)
class TrackState:
    """Running score + last seen timestamp for one vehicle id."""
    score: float = 0.0
    last_seen_mono: float = 0.0
    in_path: bool = False
    dist_m: float = 0.0           # last longitudinal distance along ego path.

    # Last-frame component breakdown for the debug window only.
    last_offset: float = 0.0
    last_yaw: float = 0.0
    last_path: float = 0.0
    last_lat: float = 0.0
    last_offset_for_score: float = 0.0
    last_yaw_diff_deg: float = 0.0
    last_baseline: float = 0.0
    last_arc_angle_amp: float = 1.0   # 2^(-(arc_angle/0.06)²) from the fit
    last_offset_delta: float = 0.0    # per-frame score contribution from offset alone
    last_score_delta: float = 0.0     # per-frame total Δscore (all components)
    last_lat_margin: float = 0.0      # corridor_half + width/2 - |lat|; positive = inside gate
    last_corridor_half: float = 0.0
    last_seen_this_frame: bool = False
    # Trail-arc debug fields; unset when fit/crossing failed (README §3).
    last_trail_is_straight: bool = False
    last_trail_cx: float = 0.0
    last_trail_cz: float = 0.0
    last_trail_R: float = 0.0
    last_trail_sign: float = 1.0
    last_trail_dir_x: float = 1.0
    last_trail_dir_z: float = 0.0
    last_trail_point_x: float = 0.0
    last_trail_point_z: float = 0.0
    last_trail_valid: bool = False
    last_trail_crossing_x: float = 0.0
    last_trail_crossing_z: float = 0.0
    last_trail_crossing_valid: bool = False


@dataclass(slots=True)
class LeadInfo:
    """Top-ranked in-path vehicle."""
    vehicle: Vehicle
    score: float
    dist_m: float
    rel_speed_ms: float           # lead - ego (signed).
    # After trailer→tractor swap these report tractor kinematics even
    # when ``vehicle`` is the trailer (for debug / visualisation).
    effective_speed_ms: float
    effective_accel_ms2: float


@dataclass
class ACCTracker:
    tracks: dict[int, TrackState] = field(default_factory=dict)
    # Sticky TMP trailer→tractor id; cleared on track expiry or failed revalidation.
    _trailer_to_tractor: dict[int, int] = field(default_factory=dict)

    # Blinker decay state; scalar = right_side - left_side after release cos decay.
    _last_left_active: float = 0.0
    _last_right_active: float = 0.0
    _prev_left: bool = False
    _prev_right: bool = False

    # Last-frame debug snapshot: populated by `update()` so the debug
    # window can render the inputs the scorer saw.
    last_blinker_scalar: float = 0.0
    last_ego_kappa_used: float = 0.0
    last_corridor_half: float = 0.0

    def update_blinkers(
        self,
        now_mono: float,
        ego_speed_ms: float,
        blinker_left: bool,
        blinker_right: bool,
    ) -> None:
        rising = (blinker_left and not self._prev_left) or (
            blinker_right and not self._prev_right
        )
        if blinker_left:
            self._last_left_active = now_mono
        if blinker_right:
            self._last_right_active = now_mono
        self._prev_left = blinker_left
        self._prev_right = blinker_right

        if rising and (ego_speed_ms * 3.6) >= _BLINKER_SCORE_RESET_KMH:
            # Highway lane change: clear current locks.
            for st in self.tracks.values():
                st.score = 0.0

    def _side_scalar(self, now_mono: float, last_active_t: float) -> float:
        """Per-side [0,1]: 1 while held, cos decay to 0 over ``_BLINKER_HOLD_S``."""
        if last_active_t <= 0.0:
            return 0.0
        t = now_mono - last_active_t
        if t <= 0.0:
            return 1.0
        if t >= _BLINKER_HOLD_S:
            return 0.0
        return math.cos((t / _BLINKER_HOLD_S) * (math.pi * 0.5))

    def _blinker_scalar(self, now_mono: float) -> float:
        """Signed blinker scalar: -1 = full left, +1 = full right."""
        left = self._side_scalar(now_mono, self._last_left_active)
        right = self._side_scalar(now_mono, self._last_right_active)
        return right - left

    @staticmethod
    def _ego_local(
        ego_x: float, ego_z: float,
        ego_fwd_x: float, ego_fwd_z: float,
        target_x: float, target_z: float,
    ) -> tuple[float, float]:
        """Ego-frame (longi, lat); rear cone only (scoring uses ``_project_onto_arc``)."""
        dx = target_x - ego_x
        dz = target_z - ego_z
        longi = dx * ego_fwd_x + dz * ego_fwd_z
        # lateral axis = (-fwd_z, fwd_x) → positive = right of ego.
        lat = dx * (-ego_fwd_z) + dz * ego_fwd_x
        return longi, lat

    @staticmethod
    def _project_onto_arc(
        arc,
        target_x: float, target_z: float,
        fwd_x: float, fwd_z: float,
    ) -> tuple[float, float]:
        """Arc-frame (arc_dist, lateral); curved arcs use infinite-circle unwrap (README §2)."""
        if arc.is_straight:
            dx = target_x - arc.start_x
            dz = target_z - arc.start_z
            longi = dx * arc.fwd_x + dz * arc.fwd_z
            lat = dx * (-arc.fwd_z) + dz * arc.fwd_x
            return longi, lat

        dcx = target_x - arc.center_x
        dcz = target_z - arc.center_z
        r_t = math.hypot(dcx, dcz)
        if r_t < 1e-6:
            # Degenerate: target sitting on the arc center.  Fall back.
            dx = target_x - arc.start_x
            dz = target_z - arc.start_z
            longi = dx * fwd_x + dz * fwd_z
            lat = dx * (-fwd_z) + dz * fwd_x
            return longi, lat

        target_angle = math.atan2(dcz, dcx)
        span = target_angle - arc.angle0
        # Unwrap span to match sweep direction so ahead-on-curve stays positive longi.
        span = (span + math.pi) % (2.0 * math.pi) - math.pi
        if arc.max_sweep > 0.0 and span < 0.0:
            span += 2.0 * math.pi
        elif arc.max_sweep < 0.0 and span > 0.0:
            span -= 2.0 * math.pi

        arc_dist = -span * arc.radius * arc._sign
        lat = (r_t - arc.radius) * arc._sign
        return arc_dist, lat

    def update(
        self,
        now_mono: float,
        dt: float,
        vehicles: list[Vehicle],
        ego_x: float, ego_y: float, ego_z: float,
        ego_yaw_rad: float, ego_pitch_rad: float,
        ego_speed_ms: float, ego_steer: float,
        ego_history_kappa: float | None,
        blinker_left: bool, blinker_right: bool,
    ) -> list[LeadInfo]:
        """Tick the tracker.  Returns top-3 leads (after trailer swap)."""
        self.update_blinkers(now_mono, ego_speed_ms, blinker_left, blinker_right)
        blinker = self._blinker_scalar(now_mono)

        ego_fwd_x = -math.sin(ego_yaw_rad)
        ego_fwd_z = -math.cos(ego_yaw_rad)
        ego_kmh = ego_speed_ms * 3.6
        # Pitch-projected elevation axis (matches AEB ElevationFilter).
        ego_yaw_sin = math.sin(ego_yaw_rad)
        ego_yaw_cos = math.cos(ego_yaw_rad)
        ego_pitch_tan = math.tan(ego_pitch_rad)

        ego_arc = build_ego_arc(
            ego_x, ego_z, ego_yaw_rad, ego_speed_ms,
            ego_steer, ego_history_kappa,
        )
        corridor_half = path_half_width(ego_steer)

        self.last_blinker_scalar = blinker
        self.last_ego_kappa_used = ego_arc.curvature
        self.last_corridor_half = corridor_half

        for st in self.tracks.values():
            st.last_seen_this_frame = False

        seen_ids: set[int] = set()
        id_to_vehicle: dict[int, Vehicle] = {}

        for v in vehicles:
            if getattr(v, "is_parked", False):
                self.tracks.pop(v.id, None)
                continue
            if v.id < 0:
                continue
            # Elevation gate: pitch-projected expected_y vs target y (AEB-aligned).
            _dx = v.position.x - ego_x
            _dz = v.position.z - ego_z
            rz_ele = _dx * ego_yaw_sin + _dz * ego_yaw_cos
            expected_y = ego_y + rz_ele * ego_pitch_tan
            if abs(v.position.y - expected_y) > _ELEVATION_MARGIN_M:
                continue

            # Rear cone uses instantaneous ego frame (not arc projection).
            straight_longi, _ = self._ego_local(
                ego_x, ego_z, ego_fwd_x, ego_fwd_z, v.position.x, v.position.z,
            )
            chord_len = math.hypot(v.position.x - ego_x, v.position.z - ego_z)
            if chord_len > 1e-3:
                fwd_dot = straight_longi / chord_len
                if fwd_dot < _REAR_DOT_THRESHOLD:
                    continue

            # Scoring geometry: vehicle center on ego arc (offset/yaw/path tuning).
            longi, lat = self._project_onto_arc(
                ego_arc, v.position.x, v.position.z, ego_fwd_x, ego_fwd_z,
            )

            # Footprint corners: nearest forward corner dist_m; any corner in corridor → in_path.
            corner_projs = [
                self._project_onto_arc(ego_arc, cx, cz, ego_fwd_x, ego_fwd_z)
                for cx, cz in v.get_corners()
            ]
            fwd_corners = [(ad, lt) for ad, lt in corner_projs if ad >= 0.0]
            if not fwd_corners:
                continue
            dist_m = min(ad for ad, _ in fwd_corners)
            if dist_m > _MAX_SCORE_RANGE_M:
                continue
            in_path = any(abs(lt) <= corridor_half for _, lt in corner_projs)

            # Trail-arc offset baselines HIT / NO_ARC_HIT / NO_HISTORY (README §3).
            v_yaw_rad = (
                v._smooth_yaw
                if v._smooth_yaw is not None
                else math.radians(v.rotation.euler()[1])
            )
            trail = fit_trail(
                getattr(v, "_position_history", []) or [],
                v_yaw_rad,
            )
            crossing: tuple[float, float] | None = None
            if trail is None:
                arc_offset = lat
                arc_angle_amp = 1.0
                baseline = OFFSET_BASELINE_NO_HISTORY
            else:
                cx_cz_ang = crossing_offset_and_angle(
                    trail, ego_x, ego_z, ego_fwd_x, ego_fwd_z,
                )
                if cx_cz_ang is None:
                    arc_offset = lat
                    arc_angle_amp = 1.0
                    baseline = OFFSET_BASELINE_NO_ARC_HIT
                else:
                    arc_offset, arc_angle_rad = cx_cz_ang
                    arc_angle_amp = angle_amp_from(arc_angle_rad)
                    baseline = OFFSET_BASELINE_HIT
                    # World-space crossing point for debug rendering.
                    right_x = -ego_fwd_z
                    right_z = ego_fwd_x
                    crossing = (
                        ego_x + arc_offset * right_x,
                        ego_z + arc_offset * right_z,
                    )

            # Scored lateral: arc offset minus blinker·4.5 m (README §5).
            offset_for_score = arc_offset - blinker * _BLINKER_OFFSET_M

            off = offset_component(
                offset_for_score, longi,
                angle_amp=arc_angle_amp, baseline=baseline,
            )
            yaw_diff_deg = math.degrees(
                (v_yaw_rad - ego_yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
            )
            yaw_c = yaw_component(yaw_diff_deg)
            path_c = path_component(longi, ego_kmh, in_path, blinker_offset=blinker)
            comps = ScoreComponents(offset=off, yaw=yaw_c, path=path_c, angle=0.0)

            st = self.tracks.get(v.id)
            if st is None:
                st = TrackState()
                self.tracks[v.id] = st
            # Legacy §9 uses the **target's** speed, not ego's.
            prev_score = st.score
            st.score = accumulate(st.score, dt, comps, v.speed)
            spd_mult = speed_multiplier(v.speed)
            st.last_score_delta = st.score - prev_score
            st.last_offset_delta = (
                off * OFFSET_WEIGHT * spd_mult * dt * LEGACY_RATE_HZ
            )
            st.last_seen_mono = now_mono
            st.in_path = in_path
            st.dist_m = dist_m
            st.last_offset = off
            st.last_yaw = yaw_c
            st.last_path = path_c
            st.last_lat = lat
            st.last_offset_for_score = offset_for_score
            st.last_yaw_diff_deg = yaw_diff_deg
            st.last_baseline = baseline
            st.last_arc_angle_amp = arc_angle_amp
            if trail is not None:
                st.last_trail_valid = True
                st.last_trail_is_straight = trail.is_straight
                st.last_trail_cx = trail.center_x
                st.last_trail_cz = trail.center_z
                st.last_trail_R = trail.radius
                st.last_trail_sign = trail.sign
                st.last_trail_dir_x = trail.dir_x
                st.last_trail_dir_z = trail.dir_z
                st.last_trail_point_x = trail.point_x
                st.last_trail_point_z = trail.point_z
            else:
                st.last_trail_valid = False
            if crossing is not None:
                st.last_trail_crossing_x = crossing[0]
                st.last_trail_crossing_z = crossing[1]
                st.last_trail_crossing_valid = True
            else:
                st.last_trail_crossing_valid = False
            # Margin: positive when the nearest corner is inside the corridor.
            st.last_lat_margin = corridor_half - min(abs(lt) for _, lt in corner_projs)
            st.last_corridor_half = corridor_half
            st.last_seen_this_frame = True
            seen_ids.add(v.id)
            id_to_vehicle[v.id] = v

        # Unseen tracks: path-only decay; expire after ``_MISSING_OUT_DECAY_S``.
        expired: list[int] = []
        for vid, st in self.tracks.items():
            if vid in seen_ids:
                continue
            if now_mono - st.last_seen_mono > _MISSING_OUT_DECAY_S:
                expired.append(vid)
                continue
            decay_comps = ScoreComponents(
                path=path_component(
                    st.dist_m, ego_kmh, in_path=False, blinker_offset=blinker,
                ),
            )
            st.score = accumulate(st.score, dt, decay_comps, ego_speed_ms)
            st.in_path = False
        for vid in expired:
            self.tracks.pop(vid, None)
            self._trailer_to_tractor.pop(vid, None)

        # Rank and swap.
        leads = self._top_leads(id_to_vehicle, vehicles, ego_fwd_x, ego_fwd_z, ego_speed_ms)
        return leads

    def _top_leads(
        self,
        id_to_vehicle: dict[int, Vehicle],
        vehicles: list[Vehicle],
        ego_fwd_x: float, ego_fwd_z: float,
        ego_speed_ms: float,
    ) -> list[LeadInfo]:
        in_path = [
            (vid, st) for vid, st in self.tracks.items()
            if st.score > IN_PATH_THRESHOLD and vid in id_to_vehicle
        ]
        # Primary sort: closest first. Secondary: score (descending) breaks ties.
        top = sorted(in_path, key=lambda item: (item[1].dist_m, -item[1].score))[:3]

        out: list[LeadInfo] = []
        for vid, st in top:
            v = id_to_vehicle[vid]
            eff_speed = v.acc_speed
            eff_accel = v.acc_accel

            # TMP top-level trailer: sticky tractor lock + kinematic swap (README §4).
            if (
                v.is_tmp and v.is_trailer
                and v.id < _TRAILER_VEHICLE_ID_BASE
            ):
                tractor = self._resolve_tractor(v, vehicles)
                if tractor is not None:
                    eff_speed = tractor.acc_speed
                    eff_accel = tractor.acc_accel

            rel = eff_speed - ego_speed_ms  # signed closing = negative
            out.append(
                LeadInfo(
                    vehicle=v,
                    score=st.score,
                    dist_m=st.dist_m,
                    rel_speed_ms=rel,
                    effective_speed_ms=eff_speed,
                    effective_accel_ms2=eff_accel,
                )
            )
        return out

    @staticmethod
    def _trailer_local_frame(
        trailer: Vehicle, other: Vehicle,
    ) -> tuple[float, float, float]:
        """(longi, lat, yaw_delta_deg) of other in trailer's smoothed heading frame."""
        trailer_yaw = (
            trailer._smooth_yaw
            if trailer._smooth_yaw is not None
            else math.radians(trailer.rotation.euler()[1])
        )
        fwd_x = -math.sin(trailer_yaw)
        fwd_z = -math.cos(trailer_yaw)
        dx = other.position.x - trailer.position.x
        dz = other.position.z - trailer.position.z
        longi = dx * fwd_x + dz * fwd_z
        lat = dx * (-fwd_z) + dz * fwd_x
        other_yaw = (
            other._smooth_yaw
            if other._smooth_yaw is not None
            else math.radians(other.rotation.euler()[1])
        )
        yaw_delta = math.degrees(
            (other_yaw - trailer_yaw + math.pi) % (2.0 * math.pi) - math.pi
        )
        return longi, lat, yaw_delta

    @staticmethod
    def _passes_strict_gate(longi: float, lat: float, yaw_delta_deg: float) -> bool:
        return (
            _TRACTOR_LOCK_LONGI_MIN_M <= longi <= _TRACTOR_LOCK_LONGI_MAX_M
            and abs(lat) <= _TRACTOR_LOCK_LAT_MAX_M
            and abs(yaw_delta_deg) <= _TRACTOR_LOCK_YAW_MAX_DEG
        )

    @staticmethod
    def _passes_loose_gate(longi: float, lat: float, yaw_delta_deg: float) -> bool:
        return (
            _TRACTOR_LOCK_VALID_LONGI_MIN_M <= longi <= _TRACTOR_LOCK_VALID_LONGI_MAX_M
            and abs(lat) <= _TRACTOR_LOCK_VALID_LAT_MAX_M
            and abs(yaw_delta_deg) <= _TRACTOR_LOCK_VALID_YAW_MAX_DEG
        )

    def _resolve_tractor(
        self, trailer: Vehicle, vehicles: list[Vehicle],
    ) -> Vehicle | None:
        """Sticky TMP tractor for a trailer; strict acquire, loose revalidate (README §4)."""
        cached_id = self._trailer_to_tractor.get(trailer.id)
        if cached_id is not None:
            cached = next((o for o in vehicles if o.id == cached_id), None)
            if cached is not None and cached.is_tmp and not cached.is_trailer:
                longi, lat, yaw_delta = self._trailer_local_frame(trailer, cached)
                if self._passes_loose_gate(longi, lat, yaw_delta):
                    return cached
            self._trailer_to_tractor.pop(trailer.id, None)

        best: Vehicle | None = None
        best_cost = math.inf
        for other in vehicles:
            if other.id == trailer.id:
                continue
            if not other.is_tmp or other.is_trailer:
                continue
            longi, lat, yaw_delta = self._trailer_local_frame(trailer, other)
            if not self._passes_strict_gate(longi, lat, yaw_delta):
                continue
            cost = abs(lat) + 0.05 * abs(longi - 10.0) + 0.2 * abs(yaw_delta)
            if cost < best_cost:
                best_cost = cost
                best = other
        if best is not None:
            self._trailer_to_tractor[trailer.id] = best.id
        return best

