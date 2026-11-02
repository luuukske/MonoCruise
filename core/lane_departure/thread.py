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
from .detector import LEVEL_CLOSE, Decision, Frame, LaneDetector, Tuning, body_from_geometry
from .lka import LaneKeeping, Pulse
from .lka_runner import LkaRunner

logger = logging.getLogger(__name__)

_LOG_EVERY_S = 1.0
_SUMMARY_S = 10.0
_ARM_MARGIN_KMH = 10.0


@dataclass
class LaneDepartureData(ThreadData):
    active: bool = False
    side: int = 0
    reason: str = "off"
    assist_reason: str = "off"
    assist_torque: float = 0.0
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
        scale = self._intensity() * (self._detector.tune.approach_scale if decision.level == LEVEL_CLOSE else 1.0)
        rumble = (ldw and decision.active) or assist.rumble
        status = self._channel.publish(rumble, decision.side, intensity_units(scale), torque=assist.torque,
                                       servo=assist.servo)
        self._status = status
        self._note(decision if ldw else Decision(), status)
        if ldw:
            self._account(decision)
        self._publish(decision if ldw else Decision(), status)

    def _keep_lane(self, frame: Frame | None, repeat: bool, decision: Decision) -> Request:
        """A lane keeping pulse on the detector's own departures; the detector is read, never changed."""
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
                    getattr(d, "ego_geometry", None),
                )
        except AttributeError:
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
                    bool(d.hazardsActive),
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
        x, z, yaw, speed, paused, t_radar, geometry = radar
        steer, gear, brake, left, right, hazards = tel
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

    def _publish(self, d: Decision, status: PluginStatus) -> None:
        with self.data._lock:
            self.data.active = d.active
            self.data.side = d.side
            self.data.reason = d.reason
            steering = self._assist if self._la_on else self._lka if self._lka_on else None
            self.data.assist_reason = steering.output.reason if steering is not None else "off"
            self.data.assist_torque = float(status.applied) if steering is not None else 0.0
            self.data.plugin = status
