"""Per-vehicle in-path tracker: ego arc, scoring, top-3 leads, trailer swap.

Frame pipeline and blinker behaviour: ``core/acc/README.md`` §3–5."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from core.radar.ego_path import EGO_POSITION_HISTORY_LEN
from core.radar.traffic import Vehicle

from .blinker import (
    BLINKER_OFFSET_M,
    EGO_FRONT_OFFSET_M,
    BlinkerBias,
    desired_time_headway_s,
    ego_lat_vel_ms,
    is_vacating,
    measure_lane_offset,
    pick_sticky_candidate,
    train_corners,
)
from .trailer_lock import TRAILER_VEHICLE_ID_BASE, resolve_tractor
from .ego_path import build_ego_arc, path_half_width
from .corroboration import slow_vehicle_evidence
from .failsafe import (
    RECENT_LEAD_S,
    FailsafeInputs,
    FailsafeState,
    failsafe_decay,
    failsafe_step,
)
from .road_model import (
    SOURCE_RESIDUAL_DELTA_M,
    RoadModel,
    arc_coords,
    fit_road_model,
    lateral_sigma_m,
)
from .road_smoother import RoadSmoother
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
from .trail_arc import (
    angle_amp_from,
    crossing_offset_and_angle,
    fit_trail,
    observed_motion_m,
    trail_evidence,
)


logger = logging.getLogger(__name__)


# Filter bounds: vehicles outside these are never scored.
_MAX_SCORE_RANGE_M: float = 150.0      # longitudinal cut-off.
_REAR_DOT_THRESHOLD: float = -0.2      # rear half cone: fwd-dot below → skip.
# Pitch elevation gate: same margin as AEB; README §8 / core/aeb (pitch-projected y).
_ELEVATION_MARGIN_M: float = 5.0

# Missing-target decay: same rate as out-of-path per frame so we don't
# pile artificial penalties onto a briefly occluded car.
_MISSING_OUT_DECAY_S: float = 2.0      # expires track after this long missing.

# Trail validation latch (README §3). Floor must stay below validated evidence.
_VALIDATE_MIN_EVIDENCE: float = 0.5
_VALIDATE_MIN_SPEED_MS: float = 2.0
_VALIDATED_STATIONARY_EVIDENCE: float = 0.75
_IN_PATH_HYSTERESIS_M: float = 0.8
_ARC_EVIDENCE_FLOOR: float = 0.45

# Road-sample direction bands and span (README §9).
_ROAD_SAMPLE_CODIR_DEG: float = 30.0
_ROAD_SAMPLE_ONCOMING_DEG: float = 150.0
_ROAD_SAMPLE_ONCOMING_WEIGHT: float = 1.0
_ROAD_SAMPLE_TMP_WEIGHT: float = 0.5
_ROAD_SAMPLE_MIN_S_M: float = -30.0
_ROAD_SAMPLE_MAX_S_M: float = 170.0
_ROAD_TRUST_TAU_UP_S: float = 0.5
_ROAD_TRUST_TAU_DOWN_S: float = 0.15
_ROAD_TRUST_INITIAL: float = 0.5
_ROAD_TRUST_MIN: float = 0.05


def effective_score(st: "TrackState") -> float:
    """Published score: the scorer's own, floored by an active failsafe rescue."""
    return max(st.score, st.failsafe_score)


def _direction_weight(yaw_diff_deg: float) -> float:
    """Road-sample weight by heading relative to ego; 0 means not a road source."""
    heading = abs(yaw_diff_deg)
    if heading <= _ROAD_SAMPLE_CODIR_DEG:
        return 1.0
    if heading >= _ROAD_SAMPLE_ONCOMING_DEG:
        return _ROAD_SAMPLE_ONCOMING_WEIGHT
    return 0.0


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
    last_evidence: float = 0.0        # trail confidence applied to the offset term
    last_road_weight: float = 0.0     # road-model share of the blended lateral
    last_trail_offset: float = 0.0    # trail-crossing lateral before the road blend
    last_lat_uncertainty: float = 0.0  # metres added to the in-path gate
    # Set once the target held a usable trail while moving (README §3 validation).
    moving_validated: bool = False
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
    # Blinker candidacy (R2-R4/R12/R13); leads[] still needs unshifted in_path (R15).
    last_behind_mono: float = float("-inf")
    indicated_candidate: bool = False
    in_path_unshifted: bool = False
    last_road_lat: float = 0.0
    last_time_headway_s: float = float("inf")
    # Geometric failsafe (README §10): ramp state, the floor it publishes, and
    # the last frame the scorer itself called this track a lead.
    failsafe: FailsafeState = field(default_factory=FailsafeState)
    failsafe_score: float = 0.0
    last_failsafe_lat: float = 0.0
    last_scored_lead_mono: float = float("-inf")


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

    _blinker: BlinkerBias = field(default_factory=BlinkerBias)

    # Ego world path in kinematics time; feeds the road model's near anchor.
    _ego_history: list[tuple[float, float, float]] = field(default_factory=list)
    # Per-source road-model trust, earned by agreeing with the fitted road.
    _source_trust: dict[int, float] = field(default_factory=dict)
    # Carries the centreline across frames in sample space (README §9).
    _road_smoother: RoadSmoother = field(default_factory=RoadSmoother)

    _last_primary_vid: int | None = None

    last_blinker_scalar: float = 0.0
    last_b_eff: float = 0.0
    last_ego_lat_vel_ms: float = 0.0
    last_lane_offset_m: float = 0.0
    last_blinker_committed: bool = False
    last_indicated_lead: LeadInfo | None = None
    last_ego_kappa_used: float = 0.0
    last_corridor_half: float = 0.0
    last_road_model: RoadModel = field(default_factory=RoadModel)

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

    def _push_ego_history(self, now_mono: float, ego_x: float, ego_z: float) -> None:
        """Append ego's world position, capped like the radar ego path."""
        if self._ego_history and now_mono <= self._ego_history[-1][0]:
            return
        self._ego_history.append((now_mono, ego_x, ego_z))
        if len(self._ego_history) > EGO_POSITION_HISTORY_LEN:
            del self._ego_history[:-EGO_POSITION_HISTORY_LEN]

    def _update_source_trust(self, road: RoadModel, dt: float) -> None:
        """Raise trust on sources that agree with the road, drop it fast when not."""
        seen = set(road.source_rms)
        for sid, rms in road.source_rms.items():
            target = (
                1.0 if rms <= SOURCE_RESIDUAL_DELTA_M
                else max(0.0, SOURCE_RESIDUAL_DELTA_M / max(rms, 1e-6))
            )
            prev = self._source_trust.get(sid, _ROAD_TRUST_INITIAL)
            tau = _ROAD_TRUST_TAU_UP_S if target >= prev else _ROAD_TRUST_TAU_DOWN_S
            alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
            self._source_trust[sid] = prev + alpha * (target - prev)
        for sid in [s for s in self._source_trust if s not in seen]:
            self._source_trust.pop(sid, None)

    def _build_road_model(
        self,
        vehicles: list[Vehicle],
        ego_x: float, ego_z: float,
        ego_fwd_x: float, ego_fwd_z: float,
        ego_yaw_rad: float,
        fallback_kappa: float,
    ) -> RoadModel:
        """Fit the shared centreline from ego's path and the traffic trails."""
        ego_samples = [
            self._ego_local(ego_x, ego_z, ego_fwd_x, ego_fwd_z, hx, hz)
            for _, hx, hz in self._ego_history
        ]
        target_samples: list[tuple[int, float, float, float]] = []
        for v in vehicles:
            if v.id < 0 or getattr(v, "is_parked", False):
                continue
            v_yaw = (
                v._smooth_yaw
                if v._smooth_yaw is not None
                else math.radians(v.rotation.euler()[1])
            )
            yaw_diff = math.degrees(
                (v_yaw - ego_yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
            )
            direction_weight = _direction_weight(yaw_diff)
            if direction_weight <= 0.0:
                continue
            history = getattr(v, "_trail_history", []) or []
            weight = trail_evidence(observed_motion_m(history))
            if weight <= 0.0:
                continue
            weight *= direction_weight
            if v.is_tmp:
                weight *= _ROAD_SAMPLE_TMP_WEIGHT
            weight *= max(
                _ROAD_TRUST_MIN,
                self._source_trust.get(v.id, _ROAD_TRUST_INITIAL),
            )
            projected = [
                self._ego_local(ego_x, ego_z, ego_fwd_x, ego_fwd_z, hx, hz)
                for _, hx, hz in history
            ]
            in_span = [
                (sx, sy) for sx, sy in projected
                if _ROAD_SAMPLE_MIN_S_M
                <= arc_coords(fallback_kappa, sx, sy)[0]
                <= _ROAD_SAMPLE_MAX_S_M
            ]
            if not in_span:
                continue
            for sx, sy in in_span:
                target_samples.append((v.id, sx, sy, weight))
        return fit_road_model(ego_samples, target_samples, fallback_kappa)

    @staticmethod
    def _apply_validation(
        st: TrackState,
        v: Vehicle,
        lat: float,
        evidence: float,
        arc_offset: float,
        arc_angle_amp: float,
        baseline: float,
    ) -> tuple[float, float, float, float]:
        """Latch trail validation while moving; reuse it once stopped (README §3)."""
        moving = abs(v.speed) >= _VALIDATE_MIN_SPEED_MS
        if (moving and evidence >= _VALIDATE_MIN_EVIDENCE
                and baseline == OFFSET_BASELINE_HIT):
            st.moving_validated = True
            return evidence, arc_offset, arc_angle_amp, baseline
        if evidence <= 0.0 and not moving and st.moving_validated:
            # Watched it drive this line before it stopped, so its lateral holds.
            return _VALIDATED_STATIONARY_EVIDENCE, lat, 1.0, OFFSET_BASELINE_HIT
        return evidence, arc_offset, arc_angle_amp, baseline

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
        """Tick the tracker. Returns top-3 in-lane leads (after trailer swap).

        Indicated-lane candidates are published separately on
        ``last_indicated_lead`` (R15); they never enter ``leads``."""
        ego_fwd_x = -math.sin(ego_yaw_rad)
        ego_fwd_z = -math.cos(ego_yaw_rad)
        ego_kmh = ego_speed_ms * 3.6
        self._blinker.update_stalk(
            now_mono, ego_speed_ms, blinker_left, blinker_right,
            ego_x=ego_x, ego_z=ego_z, ego_fwd_x=ego_fwd_x, ego_fwd_z=ego_fwd_z,
        )
        b_eff = self._blinker.compute_b_eff(now_mono, ego_kmh)
        if self._blinker.rising_this_frame:
            ref_st = self.tracks.get(self._last_primary_vid)
            self._blinker.set_commit_reference(
                self._last_primary_vid,
                ref_st.last_road_lat if ref_st is not None else 0.0,
            )
        ego_yaw_sin = math.sin(ego_yaw_rad)
        ego_yaw_cos = math.cos(ego_yaw_rad)
        ego_pitch_tan = math.tan(ego_pitch_rad)

        ego_arc = build_ego_arc(
            ego_x, ego_z, ego_yaw_rad, ego_speed_ms,
            ego_steer, ego_history_kappa,
        )
        corridor_half = path_half_width(ego_steer)

        self._push_ego_history(now_mono, ego_x, ego_z)
        road = self._build_road_model(
            vehicles, ego_x, ego_z, ego_fwd_x, ego_fwd_z,
            ego_yaw_rad, ego_arc.curvature,
        )
        road = self._road_smoother.step(
            road, ego_x, ego_z, ego_fwd_x, ego_fwd_z, dt,
        )
        self._update_source_trust(road, dt)
        corroboration = slow_vehicle_evidence(
            vehicles, ego_x, ego_z, ego_fwd_x, ego_fwd_z, ego_yaw_rad,
            road, _ROAD_SAMPLE_MAX_S_M,
        )
        desired_th_s = desired_time_headway_s()
        lat_shift = b_eff * BLINKER_OFFSET_M
        # This frame's commitment needs this frame's laterals, so the corridor
        # gate below reads last frame's latch. One frame at 30 Hz.
        committed_prev = self._blinker.committed

        self.last_blinker_scalar = self._blinker.last_scalar
        self.last_b_eff = b_eff
        self.last_ego_lat_vel_ms = ego_lat_vel_ms(
            self._ego_history, ego_fwd_x, ego_fwd_z,
        )
        self.last_ego_kappa_used = ego_arc.curvature
        self.last_corridor_half = corridor_half
        self.last_road_model = road

        for st in self.tracks.values():
            st.last_seen_this_frame = False
            st.indicated_candidate = False

        seen_ids: set[int] = set()
        id_to_vehicle: dict[int, Vehicle] = {}
        candidate_ids: list[int] = []
        # TMP trailers are top-level vehicles; nest them under their tractor so
        # parallel / R4 see the train rear, not the cab rear alone.
        tmp_trailers_by_tractor: dict[int, list[Vehicle]] = {}
        for tv in vehicles:
            if not (tv.is_tmp and tv.is_trailer):
                continue
            if tv.id >= TRAILER_VEHICLE_ID_BASE:
                continue
            tractor = resolve_tractor(tv, vehicles, self._trailer_to_tractor)
            if tractor is not None:
                tmp_trailers_by_tractor.setdefault(tractor.id, []).append(tv)

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
            straight_longi, straight_lat = self._ego_local(
                ego_x, ego_z, ego_fwd_x, ego_fwd_z, v.position.x, v.position.z,
            )
            chord_len = math.hypot(v.position.x - ego_x, v.position.z - ego_z)
            if chord_len > 1e-3:
                fwd_dot = straight_longi / chord_len
                # Soften the rear cone on the indicated side so R3 can see overtakers.
                rear_thresh = _REAR_DOT_THRESHOLD
                if abs(b_eff) > 1e-6:
                    side_lat = straight_lat * math.copysign(1.0, b_eff)
                    if side_lat > 1.0:
                        rear_thresh = -0.85
                if fwd_dot < rear_thresh:
                    continue

            # Scoring geometry: vehicle center on ego arc (offset/yaw/path tuning).
            longi, lat = self._project_onto_arc(
                ego_arc, v.position.x, v.position.z, ego_fwd_x, ego_fwd_z,
            )

            # Footprint corners: nearest forward corner dist_m; any corner in corridor → in_path.
            corner_projs = []
            corner_lats = []
            for cx, cz in v.get_corners():
                arc_dist, arc_lat = self._project_onto_arc(
                    ego_arc, cx, cz, ego_fwd_x, ego_fwd_z,
                )
                corner_projs.append((arc_dist, arc_lat))
                sx, sy = self._ego_local(ego_x, ego_z, ego_fwd_x, ego_fwd_z, cx, cz)
                road_s, road_off = road.road_coords(sx, sy)
                w_road = road.confidence_at(road_s)
                corner_lats.append(
                    w_road * road_off + (1.0 - w_road) * arc_lat
                )
            fwd_corners = [(ad, lt) for ad, lt in corner_projs if ad >= 0.0]
            if not fwd_corners:
                # Stamp behind-history on an existing track only (no new tracks).
                st_behind = self.tracks.get(v.id)
                if st_behind is not None and straight_longi < EGO_FRONT_OFFSET_M:
                    st_behind.last_behind_mono = now_mono
                continue
            dist_m = min(ad for ad, _ in fwd_corners)
            if dist_m > _MAX_SCORE_RANGE_M:
                continue
            body_lat_min = min(corner_lats)
            body_lat_max = max(corner_lats)
            # Kept unblended for the failsafe: a wrong road model must not move
            # the geometry the rescue is judged on (README §10).
            arc_lat_min = min(lt for _, lt in corner_projs)
            arc_lat_max = max(lt for _, lt in corner_projs)
            # R4 / parallel: ego-local rear of the whole train (cab + trailers).
            extras = tmp_trailers_by_tractor.get(v.id)
            train_pts = train_corners(v, extras)
            body_rear_m = min(
                self._ego_local(
                    ego_x, ego_z, ego_fwd_x, ego_fwd_z, cx, cz,
                )[0]
                for cx, cz in train_pts
            )

            st = self.tracks.get(v.id)
            if st is None:
                st = TrackState()
                self.tracks[v.id] = st
            if straight_longi < EGO_FRONT_OFFSET_M:
                st.last_behind_mono = now_mono

            # Trail-arc offset baselines HIT / NO_ARC_HIT / NO_HISTORY (README §3).
            v_yaw_rad = (
                v._smooth_yaw
                if v._smooth_yaw is not None
                else math.radians(v.rotation.euler()[1])
            )
            history = getattr(v, "_trail_history", []) or []
            trail = fit_trail(history, v_yaw_rad)
            # Evidence is observed motion, not fit success: a slow target with no
            # usable circle still shows which lane it is travelling.
            evidence = trail_evidence(observed_motion_m(history))
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

            trail_ev, arc_offset, arc_angle_amp, baseline = self._apply_validation(
                st, v, lat, evidence, arc_offset, arc_angle_amp, baseline,
            )

            # Road model knows where the target is even when it has no trail of
            # its own; the trail only ever spoke to where it was going.
            st.last_trail_offset = arc_offset
            road_s, d_road = road.road_coords(straight_longi, straight_lat)
            road_w = road.confidence_at(road_s)
            if road_w > 0.0:
                arc_offset = road_w * d_road + (1.0 - road_w) * arc_offset
            # Floored: the ego arc is itself a measurement, so the blend is never
            # evidence-free and cannot fall through to scoring on path (README §3).
            evidence = max(
                trail_ev, road_w, corroboration.get(v.id, 0.0),
                _ARC_EVIDENCE_FLOOR,
            )
            st.last_road_lat = d_road

            # Unshifted corridor membership: leads[] only (R15).
            lat_uncertainty = lateral_sigma_m(dist_m) * (1.0 - trail_ev)
            hold_bonus = _IN_PATH_HYSTERESIS_M if st.in_path_unshifted else 0.0
            # Hysteresis absorbs lateral noise; a committed lane change is not
            # noise, so it must not hold the vehicle being left (README §5).
            if committed_prev and is_vacating(body_lat_min, body_lat_max, b_eff):
                hold_bonus = 0.0
            hold_half = corridor_half + hold_bonus
            near_side = hold_half - (body_lat_min + lat_uncertainty)
            far_side = (body_lat_max - lat_uncertainty) + hold_half
            in_path_unshifted = near_side >= 0.0 and far_side >= 0.0

            yaw_diff_deg = math.degrees(
                (v_yaw_rad - ego_yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
            )
            is_cand, th_s = BlinkerBias.is_candidate(
                last_behind_mono=st.last_behind_mono,
                now_mono=now_mono, b_eff=b_eff, dist_m=dist_m,
                body_rear_m=body_rear_m,
                road_lat=d_road, yaw_diff_deg=yaw_diff_deg,
                ego_speed_ms=ego_speed_ms, v_speed_ms=float(v.acc_speed),
                desired_th_s=desired_th_s,
            )
            st.last_time_headway_s = th_s
            st.indicated_candidate = is_cand

            # R14: per-candidate shift only. Non-candidates keep true lateral.
            if is_cand:
                offset_for_score = arc_offset - lat_shift
                shift_min = body_lat_min - lat_shift
                shift_max = body_lat_max - lat_shift
                near_s = hold_half - (shift_min + lat_uncertainty)
                far_s = (shift_max - lat_uncertainty) + hold_half
                in_path_for_score = near_s >= 0.0 and far_s >= 0.0
            else:
                offset_for_score = arc_offset
                in_path_for_score = in_path_unshifted

            off = offset_component(
                offset_for_score, longi,
                angle_amp=arc_angle_amp, baseline=baseline,
                evidence=evidence, angle_evidence=trail_ev,
            )
            yaw_c = yaw_component(yaw_diff_deg)
            # b² path-amp cut stays global (slows in-lane windup, empty-lane pass).
            path_c = path_component(
                longi, ego_kmh, in_path_for_score,
                blinker_offset=b_eff, evidence=evidence,
            )
            comps = ScoreComponents(offset=off, yaw=yaw_c, path=path_c, angle=0.0)

            # Legacy §9 uses the **target's** speed, not ego's.
            prev_score = st.score
            st.score = accumulate(st.score, dt, comps, v.speed)
            spd_mult = speed_multiplier(v.speed)
            st.last_score_delta = st.score - prev_score
            st.last_offset_delta = (
                off * OFFSET_WEIGHT * spd_mult * dt * LEGACY_RATE_HZ
            )
            st.last_seen_mono = now_mono
            st.in_path = in_path_unshifted
            st.in_path_unshifted = in_path_unshifted
            st.dist_m = dist_m
            if st.score > IN_PATH_THRESHOLD:
                st.last_scored_lead_mono = now_mono
            # Failsafe: raw ego-arc geometry only, so it survives the road model
            # being wrong. It floors the published score, never ``st.score``.
            st.last_failsafe_lat = 0.5 * (arc_lat_min + arc_lat_max)
            st.failsafe_score = failsafe_step(
                st.failsafe,
                FailsafeInputs(
                    dist_m=dist_m,
                    body_lat_min=arc_lat_min,
                    body_lat_max=arc_lat_max,
                    lat_uncertainty_m=lat_uncertainty,
                    yaw_diff_deg=yaw_diff_deg,
                    ego_speed_ms=ego_speed_ms,
                    lead_speed_ms=float(v.acc_speed),
                    desired_th_s=desired_th_s,
                    recently_led=(
                        now_mono - st.last_scored_lead_mono <= RECENT_LEAD_S
                    ),
                    suppressed=(
                        is_cand
                        or (abs(b_eff) > 1e-6
                            and is_vacating(arc_lat_min, arc_lat_max, b_eff))
                    ),
                ),
                dt,
            )
            st.last_offset = off
            st.last_yaw = yaw_c
            st.last_path = path_c
            st.last_lat = lat
            st.last_offset_for_score = offset_for_score
            st.last_yaw_diff_deg = yaw_diff_deg
            st.last_baseline = baseline
            st.last_arc_angle_amp = arc_angle_amp
            st.last_evidence = evidence
            st.last_road_weight = road_w
            st.last_lat_uncertainty = lat_uncertainty
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
            # Margin: slack on whichever side of the corridor binds first.
            st.last_lat_margin = min(near_side, far_side)
            st.last_corridor_half = corridor_half
            st.last_seen_this_frame = True
            seen_ids.add(v.id)
            id_to_vehicle[v.id] = v
            if is_cand:
                candidate_ids.append(v.id)

        # R10: floor-lift on rising edge above the R0 gate. Never wipe positives.
        if self._blinker.rising_this_frame:
            for st in self.tracks.values():
                if st.score < 0.0:
                    st.score = 0.0

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
                    st.dist_m, ego_kmh, in_path=False, blinker_offset=b_eff,
                ),
            )
            st.score = accumulate(st.score, dt, decay_comps, ego_speed_ms)
            st.in_path = False
            st.in_path_unshifted = False
            st.indicated_candidate = False
            # A track radar no longer reports has no geometry left to rescue on.
            st.failsafe_score = failsafe_decay(st.failsafe, dt)
        for vid in expired:
            self.tracks.pop(vid, None)
            self._trailer_to_tractor.pop(vid, None)

        ref = self.tracks.get(self._blinker.commit_ref_vid)
        lane_offset = measure_lane_offset(
            b_eff,
            ref.last_road_lat if ref is not None and ref.last_seen_this_frame else None,
            self._blinker.commit_ref_lat,
            self._blinker.lateral_commit_m(ego_x, ego_z),
        )
        self.last_lane_offset_m = lane_offset
        self._blinker.update_commit(lane_offset)
        self.last_blinker_committed = self._blinker.committed

        sticky = self._blinker.sticky_id
        sticky_in = bool(
            sticky is not None
            and sticky in self.tracks
            and self.tracks[sticky].in_path_unshifted
        )
        b_eff = self._blinker.maybe_collapse(b_eff, sticky_in_corridor=sticky_in)
        self.last_b_eff = b_eff

        leads = self._top_leads(
            id_to_vehicle, vehicles, ego_fwd_x, ego_fwd_z, ego_speed_ms,
        )
        self._last_primary_vid = leads[0].vehicle.id if leads else None
        self.last_indicated_lead = self._pick_indicated_lead(
            candidate_ids, id_to_vehicle, vehicles, ego_speed_ms, now_mono,
        )
        return leads

    def _make_lead_info(
        self,
        vid: int,
        st: TrackState,
        id_to_vehicle: dict[int, Vehicle],
        vehicles: list[Vehicle],
        ego_speed_ms: float,
    ) -> LeadInfo | None:
        v = id_to_vehicle.get(vid)
        if v is None:
            return None
        eff_speed = v.acc_speed
        eff_accel = v.acc_accel
        if (
            v.is_tmp and v.is_trailer
            and v.id < TRAILER_VEHICLE_ID_BASE
        ):
            tractor = resolve_tractor(v, vehicles, self._trailer_to_tractor)
            if tractor is not None:
                eff_speed = tractor.acc_speed
                eff_accel = tractor.acc_accel
        rel = eff_speed - ego_speed_ms
        return LeadInfo(
            vehicle=v,
            score=effective_score(st),
            dist_m=st.dist_m,
            rel_speed_ms=rel,
            effective_speed_ms=eff_speed,
            effective_accel_ms2=eff_accel,
        )

    def _top_leads(
        self,
        id_to_vehicle: dict[int, Vehicle],
        vehicles: list[Vehicle],
        ego_fwd_x: float, ego_fwd_z: float,
        ego_speed_ms: float,
    ) -> list[LeadInfo]:
        # Score > 0 is the historic publish rule. R15 only blocks live candidates
        # that lack unshifted in-path evidence (shifted scores must not enter leads[]).
        in_path = [
            (vid, st) for vid, st in self.tracks.items()
            if (
                effective_score(st) > IN_PATH_THRESHOLD
                and vid in id_to_vehicle
                and not (st.indicated_candidate and not st.in_path_unshifted)
            )
        ]
        # Primary sort: closest first. Secondary: score (descending) breaks ties.
        top = sorted(in_path, key=lambda item: (item[1].dist_m, -item[1].score))[:3]

        out: list[LeadInfo] = []
        for vid, st in top:
            lead = self._make_lead_info(
                vid, st, id_to_vehicle, vehicles, ego_speed_ms,
            )
            if lead is not None:
                out.append(lead)
        return out

    def _pick_indicated_lead(
        self,
        candidate_ids: list[int],
        id_to_vehicle: dict[int, Vehicle],
        vehicles: list[Vehicle],
        ego_speed_ms: float,
        now_mono: float,
    ) -> LeadInfo | None:
        """R9: most binding score-positive candidate, identity-sticky (R7)."""
        scored = [
            (vid, self.tracks[vid].last_time_headway_s)
            for vid in candidate_ids
            if vid in self.tracks
            and self.tracks[vid].score > IN_PATH_THRESHOLD
            and not self.tracks[vid].in_path_unshifted
        ]
        best_vid, sticky_mono = pick_sticky_candidate(
            scored,
            sticky_id=self._blinker.sticky_id,
            sticky_mono=self._blinker.sticky_mono,
            now_mono=now_mono,
        )
        self._blinker.sticky_id = best_vid
        self._blinker.sticky_mono = sticky_mono
        if best_vid is None:
            return None
        return self._make_lead_info(
            best_vid, self.tracks[best_vid], id_to_vehicle, vehicles, ego_speed_ms,
        )


# Re-exports for tests and tuning tools.
ARC_EVIDENCE_FLOOR = _ARC_EVIDENCE_FLOOR
VALIDATED_STATIONARY_EVIDENCE = _VALIDATED_STATIONARY_EVIDENCE
IN_PATH_HYSTERESIS_M = _IN_PATH_HYSTERESIS_M
