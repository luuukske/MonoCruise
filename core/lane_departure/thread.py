"""Lane departure worker: map route and telemetry in, one rumble and one steering request to the FFB plugin out.

Off unless ``Settings.ldw_enabled`` (rumble), ``Settings.lka_enabled`` (lane keeping pulses) or ``Settings.la_enabled``
(LA, core/la; it outranks the pulses). One thread owns the command block. A missing sibling reads as no data and never
raises. See core/lane_departure/README.md.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace

from core.la.calibration import ForceCalibration
from core.la.runner import AssistRunner, Request, saved_calibration
from core.radar.ego_geometry import EgoGeometry
from core.settings import Settings
from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry

from .command import PluginStatus, RumbleChannel, intensity_units
from .detector import LEVEL_CLOSE, SIDE_LEFT, SIDE_RIGHT, Decision, Frame, LaneDetector, Tuning, body_from_geometry
from .lka import LaneKeeping, Pulse
from .lka_runner import LkaRunner
from .side_traffic import BESIDE, LEFT, RIGHT, Ego, SideState, SideTraffic

logger = logging.getLogger(__name__)

_LOG_EVERY_S = 1.0
_SUMMARY_S = 10.0
_ARM_MARGIN_KMH = 10.0
_SIDE_SUMMARY_S = 60.0


@dataclass
class LaneDepartureData(ThreadData):
    active: bool = False
    side: int = 0
    reason: str = "off"
    assist_reason: str = "off"
    assist_torque: float = 0.0
    # For the CC panel's lane lines and the bar's corners: the detector's reason while LDW or LKA watches, the side
    # acted on, and whether side traffic makes the lane past that line unsafe (an alert, not a warning).
    watch_reason: str = "off"
    firing_side: int = 0
    firing_alert: bool = False
    # Side traffic (side_traffic.py): what makes moving over unsafe per side, "" when nothing does.
    side_left: str = ""
    side_right: str = ""
    plugin: PluginStatus = field(default_factory=PluginStatus)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class LaneDepartureThread(BaseThread):
    loop_interval = 1.0 / 20.0
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="lane_departure_thread")
        self.data = LaneDepartureData()
        self._channel = RumbleChannel()
        self._detector = LaneDetector()
        self._tuning_key = (self._detector.tune.arm_kmh, False)
        self._ldw_on = False
        self._la_on = False
        self._lka_on = False
        self._detect_on = False
        calibration = ForceCalibration(saved=saved_calibration())
        self._assist = AssistRunner(calibration=calibration)
        self._lka = LkaRunner(calibration=calibration)
        self._keeping = LaneKeeping()
        self._pulse = Pulse()
        self._status = PluginStatus()
        self._tmp_map = False
        self._plugin_state: tuple | None = None
        self._body: tuple | None = None
        self._last_frame_t = -1.0
        self._last_decision = Decision()
        self._last_near_log = 0.0
        self._seconds: Counter = Counter()
        self._window_start = 0.0
        self._last_loop = 0.0
        self._min_gap = math.inf
        self._min_margin = math.inf
        self._side = SideTraffic()
        self._side_state = SideState()
        self._traffic: tuple = ((), frozenset(), ())
        self._rig_logged: tuple | None = None
        self._side_lit: Counter = Counter()
        self._side_onsets: Counter = Counter()
        self._side_window = 0.0
        self._side_last = 0.0
        self._side_failed = False
        self._alerting = False

    def setup(self) -> None:
        logger.debug("lane departure setup complete")

    def teardown(self) -> None:
        self._channel.stop()
        self._channel.close()
        self._assist.close()
        self._lka.close()
        self._publish(Decision(), PluginStatus())
        logger.debug("lane departure teardown complete")

    def loop(self) -> None:
        if not self.running:
            return
        ldw, la = bool(Settings.ldw_enabled), bool(getattr(Settings, "la_enabled", False))
        lka = bool(getattr(Settings, "lka_enabled", False))
        if not (ldw or la or lka):
            self._release()
            return
        self._follow_settings(ldw, la, lka)
        lka = lka and not la
        frame = self._read_frame()
        repeat = frame is not None and frame.t == self._last_frame_t and frame.age_s <= self._detector.tune.max_age_s
        side = self._step_side(frame, repeat) if ldw or lka else SideState()
        if frame is not None:
            frame = replace(frame, unsafe_left=side.left is not None, unsafe_right=side.right is not None)
        decision = Decision()
        if ldw or lka:
            decision = replace(self._last_decision, started=False) if repeat else self._detector.update(frame)
        if la:
            assist = self._assist.step(frame, repeat, self._status, self._tmp_map)
        elif lka:
            assist = self._keep_lane(frame, repeat, decision)
        else:
            assist = Request()
        if not repeat:
            self._last_frame_t = frame.t if frame is not None else -1.0
        self._last_decision = decision
        tune = self._detector.tune
        # A pulse the detector has no warning behind (the catch after a pulse) still comes with the soft cue.
        caught = ldw and lka and not decision.active and self._pulse.side and self._pulse.elapsed_s < tune.min_rumble_s
        soft = caught or decision.level == LEVEL_CLOSE
        rumble = (ldw and decision.active) or assist.rumble or bool(caught)
        status = self._channel.publish(rumble, self._pulse.side if caught else decision.side,
                                       intensity_units(self._intensity() * (tune.approach_scale if soft else 1.0)),
                                       torque=assist.torque, servo=assist.servo)
        self._status = status
        self._note(decision if ldw else Decision(), status)
        if ldw:
            self._account(decision)
        firing = self._pulse.side if lka and self._pulse.side else decision.side if ldw and decision.active else 0
        alert = self._alert(firing, side)
        self._publish(decision if ldw else Decision(), status, decision.reason if ldw or lka else "off", firing, side,
                      alert)

    def _alert(self, firing: int, side: SideState) -> bool:
        """LDW or LKA acting toward a lane that side traffic makes unsafe; logged once per episode."""
        hit = side.at(LEFT if firing == SIDE_LEFT else RIGHT) if firing else None
        alert = hit is not None
        if alert and not self._alerting:
            logger.info("lane departure toward an unsafe lane: %s, %s %s", "left" if firing == SIDE_LEFT else "right",
                        hit.kind, "TMP" if hit.tmp else "AI")
        self._alerting = alert
        return alert

    def _step_side(self, frame: Frame | None, repeat: bool) -> SideState:
        """Side traffic for this tick: it lifts the driver-intent mutes toward an unsafe lane and drives the alert."""
        if repeat:
            return self._side_state
        try:
            return self._judge_side(frame)
        except Exception:
            # Side traffic must never stop the rumble or a pulse running in the same loop; warn once.
            (logger.debug if self._side_failed else logger.warning)("side traffic failed", exc_info=True)
            self._side_failed = True
            self._side_state = SideState(reason="error")
            return self._side_state

    def _judge_side(self, frame: Frame | None) -> SideState:
        vehicles, skip, trailers = self._traffic
        if frame is None or frame.paused or frame.age_s > self._detector.tune.max_age_s:
            state = self._side.update(time.monotonic(), None, (), skip, None, None)
        else:
            ego = Ego(frame.x, frame.z, frame.yaw_rad, frame.speed_ms, frame.half_width_m, frame.front_m,
                      frame.rear_m, trailers)
            state = self._side.update(frame.t, ego, vehicles, skip, frame.route, frame.lanes)
            self._note_rig(trailers)
        self._note_side(state)
        self._side_state = state
        return state

    def _note_rig(self, trailers) -> None:
        length = self._side.rig_length_m
        if not math.isfinite(length):
            return
        key = (len(trailers), round(length))
        if key == self._rig_logged:
            return
        self._rig_logged = key
        gap = getattr(trailers[0], "hitch_gap_m", math.nan) if trailers else math.nan
        logger.info("side traffic rig: %d trailer(s), %.1f m long%s", len(trailers), length,
                    f", kingpin {gap:.2f} m from the fifth wheel" if math.isfinite(gap) else "")

    def _note_side(self, state: SideState) -> None:
        """One line per side lighting up or changing kind; every 60 s, how long each side was lit."""
        now = time.monotonic()
        dt = min(now - self._side_last, 0.2) if self._side_last else 0.0
        self._side_last = now
        for side, name in ((LEFT, "left"), (RIGHT, "right")):
            hit, before = state.at(side), self._side_state.at(side)
            if hit is None:
                if before is not None:
                    logger.debug("side traffic %s clear", name)
                continue
            self._side_lit[name] += dt
            if before is not None and before.kind == hit.kind and before.vid == hit.vid:
                continue
            onset = before is None or before.kind != hit.kind
            if onset:
                self._side_onsets[hit.kind] += 1
            closing = "" if hit.kind == BESIDE else f", gap {hit.gap_m:.1f} m, time {hit.time_s:.1f} s"
            (logger.info if onset else logger.debug)(
                "side traffic %s: %s %s, %.1f m long, %.1f m off our lane centre, closing %.1f m/s%s, "
                "lane past the line %s", name, hit.kind, "TMP" if hit.tmp else "AI", hit.length_m,
                abs(hit.lateral_m), hit.closing_ms, closing, hit.lane)
        if now - self._side_window < _SIDE_SUMMARY_S:
            return
        if self._side_window and (self._side_lit or self._side_onsets):
            logger.info("side traffic last %.0f s: lit left %.1f s, right %.1f s | %s", now - self._side_window,
                        self._side_lit["left"], self._side_lit["right"],
                        ", ".join(f"{k} {n}" for k, n in self._side_onsets.most_common()) or "no onsets")
        self._side_lit.clear()
        self._side_onsets.clear()
        self._side_window = now

    def _keep_lane(self, frame: Frame | None, repeat: bool, decision: Decision) -> Request:
        """A lane keeping pulse where the rumble starts, on the detector's departures; it reads the detector only."""
        if repeat:
            self._pulse = replace(self._pulse, event="")
        else:
            t = frame.t if frame is not None else self._last_frame_t
            self._pulse = self._keeping.update(t, decision, self._detector, frame, self._lka.output)
        edge = self._detector.edges.get(self._pulse.side)
        return self._lka.step_pulse(frame, repeat, self._status, self._tmp_map, self._pulse, edge)

    @staticmethod
    def _intensity() -> float:
        try:
            return float(Settings.ldw_intensity)
        except (TypeError, ValueError):
            return 1.0

    def _follow_settings(self, ldw: bool, la: bool, lka: bool) -> None:
        detect = ldw or (lka and not la)
        if detect != self._detect_on:
            self._detect_on = detect
            self._detector = LaneDetector(self._detector.tune)
        if ldw != self._ldw_on:
            self._ldw_on = ldw
            logger.info("lane departure rumble %s", "on" if ldw else "off")
        if la != self._la_on:
            self._la_on = la
            self._assist.reset()
            logger.info("LA %s", "on (config-only test feature)" if la else "off")
        if lka != self._lka_on:
            self._lka_on = lka
            self._lka.reset()
            self._keeping.reset()
            self._pulse = Pulse()
            logger.info("LKA %s", ("on (one-way pulses), waiting while LA steers" if la else "on (one-way pulses)")
                        if lka else "off")
        try:
            arm = float(Settings.ldw_min_speed_kmh)
        except (TypeError, ValueError):
            arm = self._tuning_key[0]
        key = (arm, bool(Settings.ldw_test_mode))
        if key != self._tuning_key:
            self._tuning_key = key
            self._detector = LaneDetector(
                Tuning(arm_kmh=arm, disarm_kmh=max(0.0, arm - _ARM_MARGIN_KMH), test_mode=key[1]),
            )
            if key[1]:
                logger.info("lane departure test mode: every close call warns, intent rules off")

    def _account(self, d: Decision) -> None:
        """Every 10 s, one line saying where the time went, so a silent drive can be explained."""
        now = time.monotonic()
        if self._last_loop:
            self._seconds[("near:" + d.near_miss) if d.near_miss else d.reason] += min(now - self._last_loop, 0.2)
        self._last_loop = now
        if math.isfinite(d.gap_m):
            self._min_gap = min(self._min_gap, d.gap_m)
            self._min_margin = min(self._min_margin, d.gap_m + d.allow_m)
        if now - self._window_start < _SUMMARY_S:
            return
        idle = {"speed", "no_map", "no_data", "paused", "off"}
        if self._window_start and any(k not in idle for k in self._seconds):
            spent = ", ".join(f"{k} {v:.1f}" for k, v in self._seconds.most_common())
            gap = "none" if math.isinf(self._min_gap) else f"{self._min_gap:.2f} m"
            margin = "none" if math.isinf(self._min_margin) else f"{self._min_margin:.2f} m"
            logger.info("lane departure last %.0f s: %s | closest edge %s, closest warning line %s",
                        now - self._window_start, spent, gap, margin)
        self._seconds.clear()
        self._min_gap = math.inf
        self._min_margin = math.inf
        self._window_start = now

    def _release(self) -> None:
        if not (self._ldw_on or self._la_on or self._lka_on):
            return
        if self._ldw_on:
            logger.info("lane departure rumble off")
        if self._la_on:
            logger.info("LA off")
        if self._lka_on:
            logger.info("LKA off")
        self._ldw_on = self._la_on = self._lka_on = self._detect_on = False
        self._detector = LaneDetector(self._detector.tune)
        self._assist.reset()
        self._lka.reset()
        self._keeping.reset()
        self._pulse = Pulse()
        self._last_frame_t = -1.0
        self._channel.stop()
        self._channel.close()
        self._status = PluginStatus()
        self._side.reset()
        self._side_state = SideState()
        self._publish(Decision(), PluginStatus())

    @staticmethod
    def _sibling(name: str):
        try:
            thread = registry.get_thread(name)
        except KeyError:
            return None
        return thread if thread.is_alive() else None

    def _read_radar(self):
        radar = self._sibling("radar_thread")
        if radar is None:
            return None
        try:
            with radar.data._lock:
                d = radar.data
                return (
                    d.ego_x, d.ego_z, d.ego_yaw_rad, d.ego_speed, bool(d.paused), d.t_mono,
                    getattr(d, "ego_geometry", None), list(getattr(d, "vehicles", ()) or ()),
                    frozenset(getattr(d, "off_surface_ids", frozenset()))
                    | frozenset(getattr(d, "ncz_ids", frozenset())),
                )
        except (AttributeError, TypeError):
            return None

    def _read_telemetry(self):
        tel = self._sibling("telemetry_thread")
        if tel is None:
            return None
        try:
            with tel.data._lock:
                d = tel.data
                return (
                    float(d.userSteer), int(d.gear), float(d.userBrake),
                    bool(getattr(d, "blinkerLeft", False)), bool(getattr(d, "blinkerRight", False)),
                    bool(d.hazardsActive), tuple(getattr(d, "ego_trailers", ()) or ()),
                )
        except (AttributeError, TypeError, ValueError):
            return None

    def _read_aeb(self) -> bool:
        aeb = self._sibling("aeb_thread")
        if aeb is None:
            return False
        try:
            with aeb.data._lock:
                return bool(aeb.data.AEB_cue)
        except AttributeError:
            return False

    @staticmethod
    def _read_map():
        from core.map_thread.thread import map_snapshot
        try:
            return map_snapshot()
        except Exception:
            logger.debug("map snapshot unreadable", exc_info=True)
            return None

    def _read_frame(self) -> Frame | None:
        radar = self._read_radar()
        tel = self._read_telemetry()
        if radar is None or tel is None:
            return None
        x, z, yaw, speed, paused, t_radar, geometry, vehicles, skip = radar
        steer, gear, brake, left, right, hazards, trailers = tel
        self._traffic = (vehicles, skip, trailers)
        geometry = geometry if isinstance(geometry, EgoGeometry) else None
        body = body_from_geometry(geometry)
        if body != self._body:
            self._body = body
            logger.info("lane departure body: width %.2f m, %.2f m ahead and %.2f m behind the origin, "
                        "rear axle %.2f m behind (%s)", 2.0 * body[0], body[1], body[2], -body[3],
                        "SDK wheels" if geometry is not None else "reference rig")
        snap = self._read_map()
        self._tmp_map = bool(getattr(snap, "tmp", False))
        return Frame(
            t=t_radar, age_s=time.monotonic() - t_radar, x=x, z=z, yaw_rad=yaw, speed_ms=speed,
            steer_right=-steer, gear=gear, brake=brake, blinker_left=left, blinker_right=right,
            hazards=hazards, paused=paused, aeb_active=self._read_aeb(),
            half_width_m=body[0], front_m=body[1], rear_m=body[2], path_m=body[3],
            route=getattr(snap, "route", None), lanes=getattr(snap, "window", None),
        )

    def _note(self, d: Decision, status: PluginStatus) -> None:
        state = (status.present, status.hooked, status.dead, status.version)
        if state != self._plugin_state:
            self._plugin_state = state
            logger.info("FFB plugin: present=%s hooked=%s dead=%s version=%d", *state)
        features = ("side=%d gap=%.2f m allow=%.2f m tlc=%.2f s drift=%.2f m/s psi=%.1f deg kappa=%.4f "
                    "dkappa=%.4f speed=%.0f km/h steer_net=%.3f steer_peak=%.3f steer_away=%.3f")
        values = (d.side, d.gap_m, d.allow_m, d.tlc_s, d.closing_ms, d.psi_deg, d.kappa, d.kappa_change,
                  d.speed_kmh, d.steer_net, d.steer_peak, d.steer_away)
        if d.started:
            logger.info("lane departure %s: " + features, "approach cue" if d.level == LEVEL_CLOSE else "rumble", *values)
            return
        now = time.monotonic()
        if d.near_miss and now - self._last_near_log >= _LOG_EVERY_S:
            self._last_near_log = now
            logger.debug("lane departure muted by %s: " + features, d.near_miss, *values)

    def _publish(self, d: Decision, status: PluginStatus, watch_reason: str = "off", firing_side: int = 0,
                 side: SideState | None = None, alert: bool = False) -> None:
        side = side or SideState()
        with self.data._lock:
            self.data.active = d.active
            self.data.side = d.side
            self.data.reason = d.reason
            self.data.watch_reason = watch_reason
            self.data.firing_side = firing_side
            self.data.firing_alert = alert
            self.data.side_left = side.left.kind if side.left is not None else ""
            self.data.side_right = side.right.kind if side.right is not None else ""
            steering = self._assist if self._la_on else self._lka if self._lka_on else None
            self.data.assist_reason = steering.output.reason if steering is not None else "off"
            self.data.assist_torque = float(status.applied) if steering is not None else 0.0
            self.data.plugin = status
