"""ACC thread: scores traffic and publishes in-lane leads on ``ACCData``.

Tracker only (no control law). Integration and ``LeadInfo``: ``core/acc/README.md`` §6–7."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from core.settings import Settings
from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry

from .road_model import RoadModel
from .tracker import ACCTracker, LeadInfo


logger = logging.getLogger(__name__)


@dataclass
class ACCData(ThreadData):
    """Published in-lane lead snapshot."""
    enabled: bool = False

    # Primary lead (== leads[0]) broken out for cheap reads.
    has_lead: bool = False
    lead_id: int = -1
    lead_dist_m: float = 0.0
    lead_rel_speed_ms: float = 0.0
    lead_score: float = 0.0

    # Top-3 ordered by score (post trailer→tractor swap).  Empty when
    # no vehicle has score > 0.
    leads: list[LeadInfo] = field(default_factory=list)

    # Monotonic time this snapshot corresponds to (= radar t_mono).
    t_mono: float = 0.0

    # Debug-window scoring breakdown keyed by vehicle id (see README §6).
    debug_components: dict[int, dict] = field(default_factory=dict)
    debug_blinker: float = 0.0
    debug_ego_kappa: float = 0.0
    debug_corridor_half: float = 0.0
    debug_road_model: RoadModel = field(default_factory=RoadModel)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class ACCThread(BaseThread):
    loop_interval = 1.0 / 30.0
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="acc_thread")
        self.data = ACCData()
        self._tracker = ACCTracker()
        self._last_snapshot_t: float = 0.0

    def setup(self) -> None:
        logger.debug("acc setup complete")

    def teardown(self) -> None:
        with self.data._lock:
            self.data.enabled = False
            self.data.has_lead = False
            self.data.leads = []
        logger.debug("acc teardown complete")

    def _read_radar(self) -> tuple | None:
        try:
            rt = registry.get_thread("radar_thread")
        except KeyError:
            return None
        if not rt.is_alive():
            return None
        try:
            with rt.data._lock:
                return (
                    list(rt.data.vehicles),
                    list(rt.data.trailer_vehicles),
                    rt.data.ego_x,
                    rt.data.ego_y,
                    rt.data.ego_z,
                    rt.data.ego_yaw_rad,
                    rt.data.ego_pitch_rad,
                    rt.data.ego_speed,
                    rt.data.ego_steer,
                    rt.data.ego_curvature,
                    rt.data.paused,
                    rt.data.t_mono,
                )
        except AttributeError:
            return None

    def _read_blinkers(self) -> tuple[bool, bool]:
        try:
            tel = registry.get_thread("telemetry_thread")
        except KeyError:
            return False, False
        if not tel.is_alive():
            return False, False
        try:
            with tel.data._lock:
                return (
                    bool(getattr(tel.data, "blinkerLeft", False)),
                    bool(getattr(tel.data, "blinkerRight", False)),
                )
        except AttributeError:
            return False, False

    def loop(self) -> None:
        if not self.running:
            return

        enabled = bool(Settings.acc_enabled) and Settings.cc_mode == "Cruise control"

        snap = self._read_radar()
        if snap is None:
            self._publish(enabled, [], self._last_snapshot_t)
            return

        (vehicles, trailer_vehicles, ex, ey, ez, eyaw, epitch, espeed, esteer,
         ekappa, paused, t_radar) = snap

        # Nested trailers (road-train trailers behind the first) ride in a
        # separate radar list; score them alongside top-level vehicles.
        targets = vehicles + trailer_vehicles

        # Paused / stale frame: hold the previous lead list.  Don't advance
        # the tracker so scores don't decay against a frozen frame.
        if paused or t_radar <= self._last_snapshot_t:
            self._publish(enabled, self.data.leads, self._last_snapshot_t)
            return

        dt = t_radar - self._last_snapshot_t if self._last_snapshot_t > 0 else self.loop_interval
        dt = max(1e-3, min(dt, 0.2))   # safety clamp.
        self._last_snapshot_t = t_radar

        bl_left, bl_right = self._read_blinkers()

        try:
            leads = self._tracker.update(
                now_mono=t_radar,
                dt=dt,
                vehicles=targets,
                ego_x=ex, ego_y=ey, ego_z=ez,
                ego_yaw_rad=eyaw, ego_pitch_rad=epitch,
                ego_speed_ms=espeed, ego_steer=esteer,
                ego_history_kappa=ekappa,
                blinker_left=bl_left, blinker_right=bl_right,
            )
        except Exception:
            logger.exception("acc tracker update failed; holding previous leads")
            self._publish(enabled, [], t_radar)
            return

        self._publish(enabled, leads, t_radar)

    def _publish(self, enabled: bool, leads: list[LeadInfo], t_mono: float) -> None:
        primary = leads[0] if leads else None
        debug_components = {
            vid: {
                "score": st.score,
                "in_path": st.in_path,
                "lat": st.last_lat,
                "dist_m": st.dist_m,
                "offset": st.last_offset,
                "yaw": st.last_yaw,
                "path": st.last_path,
                "baseline": st.last_baseline,
                "offset_for_score": st.last_offset_for_score,
                "yaw_diff_deg": st.last_yaw_diff_deg,
                "arc_angle_amp": st.last_arc_angle_amp,
                "evidence": st.last_evidence,
                "moving_validated": st.moving_validated,
                "road_weight": st.last_road_weight,
                "lat_uncertainty": st.last_lat_uncertainty,
                "offset_delta": st.last_offset_delta,
                "score_delta": st.last_score_delta,
                "trail_valid": st.last_trail_valid,
                "trail_is_straight": st.last_trail_is_straight,
                "trail_cx": st.last_trail_cx,
                "trail_cz": st.last_trail_cz,
                "trail_R": st.last_trail_R,
                "trail_sign": st.last_trail_sign,
                "trail_dir_x": st.last_trail_dir_x,
                "trail_dir_z": st.last_trail_dir_z,
                "trail_point_x": st.last_trail_point_x,
                "trail_point_z": st.last_trail_point_z,
                "trail_crossing_valid": st.last_trail_crossing_valid,
                "trail_crossing_x": st.last_trail_crossing_x,
                "trail_crossing_z": st.last_trail_crossing_z,
                "lat_margin": st.last_lat_margin,
                "corridor_half": st.last_corridor_half,
                "seen": st.last_seen_this_frame,
            }
            for vid, st in self._tracker.tracks.items()
        }
        with self.data._lock:
            self.data.enabled = enabled
            self.data.leads = leads
            self.data.has_lead = primary is not None
            self.data.lead_id = primary.vehicle.id if primary is not None else -1
            self.data.lead_dist_m = primary.dist_m if primary is not None else 0.0
            self.data.lead_rel_speed_ms = primary.rel_speed_ms if primary is not None else 0.0
            self.data.lead_score = primary.score if primary is not None else 0.0
            self.data.t_mono = t_mono
            self.data.debug_components = debug_components
            self.data.debug_blinker = self._tracker.last_blinker_scalar
            self.data.debug_ego_kappa = self._tracker.last_ego_kappa_used
            self.data.debug_corridor_half = self._tracker.last_corridor_half
            self.data.debug_road_model = self._tracker.last_road_model

