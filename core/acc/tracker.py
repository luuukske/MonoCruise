"""
Per-vehicle in-path tracker — accumulates score, selects top-3 lead.

State lives in :class:`ACCTracker` (one instance per ACC thread).

Per frame the tracker:
    1. Builds the ego arc (see ``ego_path.build_ego_arc``).
    2. Computes the signed blinker scalar (``-1`` full left, ``+1``
       full right, ``0`` neutral) — cos-decays over ``_BLINKER_HOLD_S``
       once the blinker turns off.  Legacy SCORING_REFERENCE §7.
    3. For every Vehicle in RadarData:
         - Skips rear / too-far / wrong-elevation targets.
         - Computes lateral offset from the ego heading, longitudinal
           distance along the arc, and heading mismatch.
         - Subtracts ``blinker_scalar · 4.5 m`` from the scored lateral
           offset so adjacent-lane targets gain score during a lane
           change.
         - Feeds the four scoring components into ``scoring.accumulate``,
           scaling by the **target** vehicle's speed (legacy §9).
    4. Decays scores for vehicles we didn't see this frame (missing →
       path penalty only — they drift toward the negative floor).
    5. Applies the trailer→tractor swap on the top-3 list so a lead
       trailer reports its tractor's speed/accel.

At ego speeds ≥ ``_BLINKER_SCORE_RESET_KMH`` the blinker rising edge
zeroes all accumulated scores once — legacy "highway lane change"
reset so a new lead gets picked cleanly on the other side.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from core.radar.traffic import Vehicle, arc_arc_collision

from .ego_path import build_ego_arc, path_half_width
from .scoring import (
    IN_PATH_THRESHOLD,
    OFFSET_BASELINE_HIT,
    OFFSET_BASELINE_NO_ARC_HIT,
    OFFSET_BASELINE_NO_HISTORY,
    ScoreComponents,
    accumulate,
    offset_component,
    path_component,
    yaw_component,
)


logger = logging.getLogger(__name__)


# Filter bounds — vehicles outside these are never scored.
_MAX_SCORE_RANGE_M: float = 150.0      # longitudinal cut-off.
_REAR_DOT_THRESHOLD: float = -0.2      # rear half cone: fwd-dot below → skip.
_ELEVATION_DIFF_M: float = 5.0         # |Δy| above this → different road level.

# Missing-target decay — same rate as out-of-path per frame so we don't
# pile artificial penalties onto a briefly occluded car.
_MISSING_OUT_DECAY_S: float = 2.0      # expires track after this long missing.

# Blinker scalar decay.  Rising edge pins |scalar| = 1; after release it
# cos-decays to 0 over _BLINKER_HOLD_S.  Applied as ``offset - s·4.5 m``
# on the scored lateral offset (SCORING_REFERENCE §7, §8.1).
_BLINKER_HOLD_S: float = 2.5
_BLINKER_OFFSET_M: float = 4.5

# Highway lane change reset — zero all scores on blinker rising edge
# above this ego speed so a new lead can lock cleanly on the new side.
_BLINKER_SCORE_RESET_KMH: float = 65.0

# Minimum target position history length required before we'd trust a
# trail-arc fit.  Matches legacy ``fit_circle`` / ``draw_fitted_arc``
# gate of ``len(history) < 5``: below 5 samples the LS circle fit is
# skipped and the NO_HISTORY baseline (-0.16) is applied.
# NOTE: the trail-arc LS circle fit itself is not yet implemented —
# see core/acc/AGENTS.md §3.  Until it lands, offset uses the target's
# current lateral as the crossing fallback and baseline switches
# between HIT (got a fwd chord) and NO_HISTORY (not enough samples).
_MIN_TRAIL_SAMPLES: int = 5


@dataclass(slots=True)
class TrackState:
    """Running score + last seen timestamp for one vehicle id."""
    score: float = 0.0
    last_seen_mono: float = 0.0
    in_path: bool = False
    dist_m: float = 0.0           # last longitudinal distance along ego path.

    # Per-frame scoring breakdown — populated for every scored vehicle each
    # tick. Consumed by the debug window to surface why a vehicle is or is
    # not being tracked. Not used by control logic.
    last_offset: float = 0.0
    last_yaw: float = 0.0
    last_path: float = 0.0
    last_lat: float = 0.0
    last_offset_for_score: float = 0.0
    last_yaw_diff_deg: float = 0.0
    last_baseline: float = 0.0
    last_arc_hit: bool = False
    last_corridor_half: float = 0.0
    last_seen_this_frame: bool = False


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

    # Blinker scalar state — ``_last_*_active`` is bumped every frame
    # while the blinker is on, so once it releases the cos decay
    # starts cleanly at t=0.  Scalar resolves to ``right - left``
    # (only one side is usually active).
    _last_left_active: float = 0.0
    _last_right_active: float = 0.0
    _prev_left: bool = False
    _prev_right: bool = False

    # Last-frame debug snapshot — populated by `update()` so the debug
    # window can render the inputs the scorer saw.
    last_blinker_scalar: float = 0.0
    last_ego_kappa_used: float = 0.0
    last_corridor_half: float = 0.0

    # ------------------------------------------------------------------
    # Blinker handling
    # ------------------------------------------------------------------
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
            # Highway lane change — clear current locks.
            for st in self.tracks.values():
                st.score = 0.0

    def _side_scalar(self, now_mono: float, last_active_t: float) -> float:
        """Per-side scalar in [0, 1] — 1 while held, cos decay after.

        Because ``last_active_t`` is bumped every frame while the
        blinker is on, ``t = now - last_active_t`` is 0 at the moment
        of release and grows afterwards, giving a clean decay curve.
        """
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

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ego_local(
        ego_x: float, ego_z: float,
        ego_fwd_x: float, ego_fwd_z: float,
        target_x: float, target_z: float,
    ) -> tuple[float, float]:
        """(longitudinal, lateral) of target in ego heading frame.  Right=+lat.

        Uses ego's *instantaneous* forward vector — good for the rear
        cone gate, not for scoring on a curve.  See :func:`_project_onto_arc`
        for the arc-relative projection used by the scorer.
        """
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
        """(arc_dist, lateral_from_arc) — signed, positive lateral = right.

        Straight arc: same as :func:`_ego_local`.

        Curved arc: the arc is a circle, so the target's distance from
        the arc center minus the arc radius gives a signed lateral
        (using `_sign` to keep "positive = right of the heading at
        that point").  Longitudinal is the angular span from ``angle0``
        unwrapped in the sweep direction, converted back to metres via
        ``-span·radius·_sign``.  Valid beyond the arc's finite horizon
        — treats the arc as an infinite circle.
        """
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
        # Unwrap to the same rotational direction as the sweep so a
        # target "ahead" on a curve of arbitrary radius yields a
        # positive arc_dist.
        span = (span + math.pi) % (2.0 * math.pi) - math.pi
        if arc.max_sweep > 0.0 and span < 0.0:
            span += 2.0 * math.pi
        elif arc.max_sweep < 0.0 and span > 0.0:
            span -= 2.0 * math.pi

        arc_dist = -span * arc.radius * arc._sign
        lat = (r_t - arc.radius) * arc._sign
        return arc_dist, lat

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(
        self,
        now_mono: float,
        dt: float,
        vehicles: list[Vehicle],
        ego_x: float, ego_y: float, ego_z: float,
        ego_yaw_rad: float, ego_speed_ms: float, ego_steer: float,
        ego_history_kappa: float | None,
        blinker_left: bool, blinker_right: bool,
    ) -> list[LeadInfo]:
        """Tick the tracker.  Returns top-3 leads (after trailer swap)."""
        self.update_blinkers(now_mono, ego_speed_ms, blinker_left, blinker_right)
        blinker = self._blinker_scalar(now_mono)

        ego_fwd_x = -math.sin(ego_yaw_rad)
        ego_fwd_z = -math.cos(ego_yaw_rad)
        ego_kmh = ego_speed_ms * 3.6

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
            if v.id < 0:
                continue
            # Elevation gate — other road levels.
            if abs(v.position.y - ego_y) > _ELEVATION_DIFF_M:
                continue

            # Rear cone gate runs on the *instantaneous* ego frame so a
            # car directly behind on a bend still counts as "rear" even
            # if the arc would curl back over it.
            straight_longi, _ = self._ego_local(
                ego_x, ego_z, ego_fwd_x, ego_fwd_z, v.position.x, v.position.z,
            )
            chord_len = math.hypot(v.position.x - ego_x, v.position.z - ego_z)
            if chord_len > 1e-3:
                fwd_dot = straight_longi / chord_len
                if fwd_dot < _REAR_DOT_THRESHOLD:
                    continue

            # Scoring-space geometry — project into the ego arc frame so
            # lateral is measured from the curved centerline, not from
            # the straight forward vector.  This is the fix for "car
            # in front on a corner shows huge lateral → never locks".
            longi, lat = self._project_onto_arc(
                ego_arc, v.position.x, v.position.z, ego_fwd_x, ego_fwd_z,
            )

            if longi < 0.0 or longi > _MAX_SCORE_RANGE_M:
                continue

            # In-path test — sample closest approach between ego arc and
            # target arc.  arc_arc_collision handles curved ego paths
            # correctly (whereas a straight-line lateral test would fail
            # on a bend).
            try:
                v_arc = v.get_arc(horizon=2.5)
            except Exception:
                logger.debug("get_arc failed for id=%d", v.id, exc_info=True)
                continue
            hit = arc_arc_collision(
                ego_arc, v_arc,
                margin=0.0,
                n_samples=12,
                min_lateral_gap=0.0,
            )
            in_path = hit is not None and abs(lat) <= corridor_half

            # Blinker scalar shifts the *scored* lateral offset by up to
            # 4.5 m toward the indicated side (SCORING_REFERENCE §7).
            # Targets in the adjacent lane thus score near 0 offset
            # during the signalled manoeuvre.
            offset_for_score = lat - blinker * _BLINKER_OFFSET_M

            # Baseline selection — trail-arc fitting not yet implemented,
            # so: HIT if we have enough samples to trust the current
            # lateral as a crossing proxy, else NO_HISTORY.  Once the
            # LS circle fit lands this will flip to NO_ARC_HIT when the
            # fitted arc doesn't cross the ego row.
            history_len = len(getattr(v, "position_history", []) or ())
            if history_len < _MIN_TRAIL_SAMPLES:
                baseline = OFFSET_BASELINE_NO_HISTORY
            else:
                baseline = OFFSET_BASELINE_HIT

            # Components.
            off = offset_component(
                offset_for_score, longi, angle_amp=1.0, baseline=baseline,
            )
            # Heading mismatch via smoothed yaw when available.
            v_yaw_rad = v._smooth_yaw if v._smooth_yaw is not None else math.radians(
                v.rotation.euler()[1]
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
            st.score = accumulate(st.score, dt, comps, v.speed)
            st.last_seen_mono = now_mono
            st.in_path = in_path
            st.dist_m = longi
            st.last_offset = off
            st.last_yaw = yaw_c
            st.last_path = path_c
            st.last_lat = lat
            st.last_offset_for_score = offset_for_score
            st.last_yaw_diff_deg = yaw_diff_deg
            st.last_baseline = baseline
            st.last_arc_hit = hit is not None
            st.last_corridor_half = corridor_half
            st.last_seen_this_frame = True
            seen_ids.add(v.id)
            id_to_vehicle[v.id] = v

        # Decay / expire unseen tracks.  We don't know the target's
        # current speed, so fall back to ego speed for the accumulator
        # multiplier — it's close enough for a decay-only tick.
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

        # Rank and swap.
        leads = self._top_leads(id_to_vehicle, vehicles, ego_fwd_x, ego_fwd_z, ego_speed_ms)
        return leads

    # ------------------------------------------------------------------
    # Top-N selection + trailer swap
    # ------------------------------------------------------------------
    def _top_leads(
        self,
        id_to_vehicle: dict[int, Vehicle],
        vehicles: list[Vehicle],
        ego_fwd_x: float, ego_fwd_z: float,
        ego_speed_ms: float,
    ) -> list[LeadInfo]:
        ranked: list[tuple[int, TrackState]] = sorted(
            (
                (vid, st) for vid, st in self.tracks.items()
                if st.score > IN_PATH_THRESHOLD and vid in id_to_vehicle
            ),
            key=lambda item: item[1].score,
            reverse=True,
        )
        top = ranked[:3]

        out: list[LeadInfo] = []
        for vid, st in top:
            v = id_to_vehicle[vid]
            eff_speed = v.speed
            eff_accel = v.acceleration

            # Trailer → tractor swap (TMP: if lead is a trailer, promote
            # the pulling tractor's kinematics so gap control reacts to
            # the actual driven vehicle not the dragged trailer).
            if v.is_tmp and v.is_trailer:
                tractor = _find_tractor_for_trailer(v, vehicles)
                if tractor is not None:
                    eff_speed = tractor.speed
                    eff_accel = tractor.acceleration

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


def _find_tractor_for_trailer(trailer: Vehicle, vehicles: list[Vehicle]) -> Vehicle | None:
    """Cheap nearest-non-trailer-TMP match within 30 m.  Good enough for gap control."""
    best: Vehicle | None = None
    best_d = 30.0 * 30.0
    for other in vehicles:
        if other.id == trailer.id:
            continue
        if not other.is_tmp or other.is_trailer:
            continue
        dx = other.position.x - trailer.position.x
        dz = other.position.z - trailer.position.z
        d = dx * dx + dz * dz
        if d < best_d:
            best_d = d
            best = other
    return best
