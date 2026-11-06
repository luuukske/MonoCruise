"""Settings, calibration, logs and the time summary around ``LaneAssist``; the lane departure thread hands it frames.

Config-only test feature: ``Settings.la_enabled`` and friends, no UI. See core/la/README.md. The lane keeping assist
runs the same core through a subclass (core/lane_departure/lka_runner.py).
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core.lane_departure.command import SERVO_OFF, VERSION, PluginStatus, ServoCommand
from core.lane_departure.detector import Frame
from core.lane_departure.ffb_trace import TraceLog
from core.sending_thread.debug_csv import TickDebugLog
from core.settings import Settings

from .calibration import Calibration, ForceCalibration
from .actuator import design_overrides, push_ahead_overrides
from .controller import STRENGTH_MAX, AssistOutput, AssistTuning, Gains, LaneAssist, Wheel, WheelTuning
from .hand_push import HandPush, HandPushEstimator, PushTuning

logger = logging.getLogger(__name__)

_SUMMARY_S = 10.0
_REF_MS = 80.0 / 3.6
_EVENT_LOG_S = 1.0
_CAL_LOG_S = 60.0
_CAL_FLAP_S = 10.0
_CAL_SAVE_S = 60.0
_CAL_SAVE_CHANGE = 0.03
_STEER_STALE_S = 0.3
_ARM_MARGIN_KMH = 10.0
_IDLE = frozenset({"speed", "no_map", "no_data", "paused", "off"})
_DEBUG_LOG_NAME = "la_debug.csv"
_DEBUG_MAX_BYTES = 50_000_000
_DEBUG_HEADER = [
    "t_s", "utc", "speed_kmh", "reason", "event", "authority", "offset_m", "centre_m", "drift_ms", "kappa_ff",
    "accel_fb_ms2", "target", "target_rate", "steer", "steer_rate", "game_force", "applied", "servo_applied",
    "cancel_applied", "integ", "kp", "kd", "ki", "cap", "cancel", "stiffness_scale", "weave_gain", "servo_scale",
    "ff_scale", "frame_hz", "hands_rms", "lane_fit", "ffb_frames", "steer_frames", "preview_s", "servo_only",
    "rumbling", "steer_accel", "hand", "pulse_push", "hand_against",
]
_EVENTS = {
    "override": "yields, the driver steers",
    "evade": "yields to a swerve",
    "weave": "lowers its gain, the correction kept flipping sides",
    "ring": "lowers its stiffness, the wheel kept swinging through the target",
    "hands_off": "lets go, no hands on the wheel",
    "carry": "keeps the lane the truck left, the route moved to the next one",
    "geometry": "holds the wheel and starts the lane over, the lane moved faster than the truck can",
}


def _setting(name: str, default: float) -> float:
    try:
        value = float(getattr(Settings, name))
    except (AttributeError, TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _show(value: float | None) -> float:
    return math.nan if value is None else value


def saved_calibration() -> tuple[float, float] | None:
    """``(force_per_ms2, kappa_per_steer)`` kept from an earlier drive (``Settings.la_saved_calibration``), or None."""
    raw = getattr(Settings, "la_saved_calibration", None)
    try:
        force, gain = float(raw["force_per_ms2"]), float(raw["kappa_per_steer"])
    except (TypeError, KeyError, ValueError):
        return None
    return (force, gain) if math.isfinite(force) and math.isfinite(gain) and force and gain > 0.0 else None


@dataclass(frozen=True)
class Request:
    """What this tick sends to the plugin for the LA."""

    torque: float = 0.0
    servo: ServoCommand = SERVO_OFF
    rumble: bool = False


class AssistRunner:
    LABEL = "LA"
    IDLE = _IDLE
    DEBUG_NAME = _DEBUG_LOG_NAME
    TRACE_NAME = "la_ffb_debug.csv"

    def __init__(self, debug_root: Path | None = None, calibration: ForceCalibration | None = None) -> None:
        self._assist = LaneAssist()
        self._cal = calibration or ForceCalibration()
        self._key: tuple | None = None
        self._out = AssistOutput()
        self._cal_state: tuple = ()
        self._cal_logged = 0.0
        self._cal_saved_at = -math.inf
        self._saved_seen = ""
        self._plugin_warned = False
        self._last_event_log = 0.0
        self._steer_frames: int | None = None
        self._steer_at = -math.inf
        self._seconds: Counter = Counter()
        self._window_start = 0.0
        self._last_loop = 0.0
        self._peak = 0.0
        self._torque_sum = 0.0
        self._rms: list[float] = []
        self._ticks = 0
        self._game_sum = self._felt_sum = 0.0
        self._assist_ticks = 0
        self._last_gains: Gains | None = None
        self._hand = HandPushEstimator()
        self._push = HandPush()
        self._push_at: float | None = None
        root = debug_root or Path(__file__).resolve().parents[2]
        self._debug = TickDebugLog(root, self.DEBUG_NAME, _DEBUG_HEADER, 0.05, 2.0, _DEBUG_MAX_BYTES)
        self._trace = TraceLog(root, self.TRACE_NAME, self.LABEL)

    @property
    def output(self) -> AssistOutput:
        return self._out

    @property
    def push(self) -> HandPush:
        """The driver's estimated push on the rim this tick; a lane keeping pulse yields to it."""
        return self._push

    def reset(self) -> None:
        """Back to silence; the calibration survives a toggle and is kept for the next start."""
        self._assist = LaneAssist(self._assist.tune)
        self._out = AssistOutput()
        self._hand.reset()
        self._push = HandPush()

    def step(self, frame: Frame | None, repeat: bool, status: PluginStatus, tmp: bool, engage: int = 0) -> Request:
        """This tick's request. ``repeat`` means the radar frame did not move since the last call.

        ``engage`` is the edge a lane keeping pulse steers away from (pulse mode only).
        """
        self._follow_settings()
        if not repeat:
            wheel = self._wheel(status)
            gains = self._gains(frame, status, wheel)
            self._out = self._assist.update(frame, gains, wheel, tmp, engage=engage, hand=self._push)
            self._note(status)
        self._watch_push(status)
        self._account(self._out, status)
        self._record(status)
        return Request(self._out.torque, self._out.servo, self._out.rumble)

    def close(self) -> None:
        self._debug.close()
        self._trace.close()

    def _watch_push(self, status: PluginStatus) -> None:
        """The driver's push from the plugin's readings, every tick; it needs the wheel's torque direction.

        ``perf_counter``: the step is a divisor, and ``monotonic`` can tick at 15.6 ms on Windows.
        """
        now = time.perf_counter()
        dt = now - self._push_at if self._push_at is not None else 0.0
        self._push_at = now
        gains = self._last_gains
        direction = (1 if gains.force_per_ms2 > 0.0 else -1) if gains is not None and gains.force_per_ms2 else 0
        if not status.servo_capable:
            direction = 0
        self._push = self._hand.update(dt, status.steer_rate, status.game_force, status.applied, status.cancel_applied,
                                       direction, self._out.servo.only)

    def _record(self, status: PluginStatus) -> None:
        """20 Hz rows to the debug file while ``Settings.debug`` is on and the assist is not idle.

        The plugin's trace goes to its own file at the FFB rate whenever the game drives, idle or not.
        """
        out = self._out
        if not getattr(Settings, "debug", False):
            return
        servo = out.servo
        gains = self._last_gains
        self._trace.tick(time.monotonic(), {
            "speed_kmh": out.speed_kmh, "reason": out.reason, "authority": out.authority, "target": out.target,
            "target_rate": servo.rate, "kp": servo.kp, "kd": servo.kd, "ki": servo.ki, "cap": servo.cap,
            "damping_first": servo.damping_first, "only": servo.only, "kappa_ff": out.kappa_ff,
            "force_per_ms2": gains.force_per_ms2 if gains is not None else None,
            "kappa_per_steer": gains.kappa_per_steer if gains is not None else None,
        })
        self._debug.tick(time.monotonic(), out.reason not in self.IDLE, {
            "speed_kmh": out.speed_kmh, "reason": out.reason, "event": out.event, "authority": out.authority,
            "offset_m": out.offset_m, "centre_m": out.bias_m, "drift_ms": out.drift_ms, "kappa_ff": out.kappa_ff,
            "accel_fb_ms2": out.accel_fb_ms2, "target": out.target, "target_rate": servo.rate, "steer": status.steer,
            "steer_rate": status.steer_rate, "game_force": status.game_force, "applied": status.applied,
            "servo_applied": status.servo_applied, "cancel_applied": status.cancel_applied, "integ": status.servo_integ,
            "kp": servo.kp, "kd": servo.kd, "ki": servo.ki, "cap": servo.cap, "cancel": servo.cancel,
            "stiffness_scale": self._assist.ring_scale, "weave_gain": self._assist.gain,
            "servo_scale": status.servo_scale, "ff_scale": self._assist.ff_scale, "frame_hz": status.frame_hz,
            "hands_rms": out.hands_rms, "lane_fit": int(out.fitted), "ffb_frames": status.frames,
            "steer_frames": status.steer_frames, "preview_s": out.preview_s, "servo_only": servo.only,
            "rumbling": status.rumbling, "steer_accel": self._push.accel, "hand": self._push.hand,
            "pulse_push": self._push.push, "hand_against": self._push.against,
        })

    def _follow_settings(self) -> None:
        key, tuning, line = self._tuning()
        if key == self._key:
            return
        if self._key is not None or line:
            logger.info(*line)
        self._key = key
        self._assist = LaneAssist(tuning)
        self._hand = HandPushEstimator(PushTuning(inertia=tuning.wheel.inertia))

    def _tuning(self) -> tuple[tuple, AssistTuning, tuple]:
        """The settings key, the tuning it gives and a log line for it (empty: only log a change)."""
        arm = _setting("la_min_speed_kmh", 60.0)
        strength = max(0.0, min(STRENGTH_MAX, _setting("la_strength", 1.0)))
        inertia = _setting("la_wheel_inertia", 0.0) or WheelTuning().inertia
        cancel = max(0.0, min(1.0, _setting("la_cancel_share", 0.0)))
        design = (_setting("la_bandwidth_hz", 0.0), _setting("la_damping", 0.0),
                  bool(getattr(Settings, "la_damping_first", False)), _setting("la_servo_cap", 0.0))
        ahead = (_setting("la_ff_inertia", 0.0), _setting("la_ff_damping", 0.0), _setting("la_ff_friction", 0.0),
                 _setting("la_servo_lead_s", 0.0), _setting("la_ff_plan_s", 0.0))
        key = (arm, strength, bool(getattr(Settings, "la_tmp_enabled", False)),
               bool(getattr(Settings, "la_follow_all", False)), max(50.0, min(20000.0, inertia)), cancel,
               bool(getattr(Settings, "la_always_on", False)), *design, *ahead)
        line = ("LA settings: from %.0f km/h, strength %.2f, TruckersMP %s, follow everywhere %s, wheel inertia %.0f, "
                "game force cancelled %.2f, always on (test, no hand-backs) %s, servo design %.2f Hz, damping %.2f, "
                "damping first %s, servo cap %.0f (0 is 3000), push ahead inertia %.0f, damping %.0f, friction %.0f, "
                "lead %.3f s, plan %.3f s", *key)
        return key, AssistTuning(
            arm_kmh=arm, disarm_kmh=max(0.0, arm - _ARM_MARGIN_KMH),
            wheel=WheelTuning(strength=strength, inertia=key[4], cancel_share=key[5], **design_overrides(*design),
                              **push_ahead_overrides(*ahead)),
            allow_tmp=key[2], follow_all=key[3], always_on=key[6],
        ), line if key[3] or key[6] else ()

    def _wheel(self, status: PluginStatus) -> Wheel:
        """The plugin's wheel angle, usable while its steering channel keeps counting."""
        now = time.monotonic()
        if status.steer_frames != self._steer_frames:
            self._steer_frames = status.steer_frames
            self._steer_at = now
        ok = status.servo_capable and now - self._steer_at <= _STEER_STALE_S
        return Wheel(ok=ok, steer=status.steer, rate=status.steer_rate, push=float(status.servo_applied),
                     buzz=status.servo_scale, game=float(status.game_force))

    def _gains(self, frame: Frame | None, status: PluginStatus, wheel: Wheel) -> Gains | None:
        """Learned, or fixed by config (``la_force_per_ms2``, ``la_kappa_per_steer``, 0 learns)."""
        if frame is not None and status.torque_capable:
            steer_right = -status.steer if wheel.ok else frame.steer_right
            cal = self._cal.update(frame.t, frame.speed_ms, frame.yaw_rad, steer_right, status.game_force,
                                   status.frames)
        else:
            cal = self._cal.result()
        self._log_calibration(cal)
        self._keep_calibration(cal)
        force = _setting("la_force_per_ms2", 0.0) or cal.force_per_ms2
        kappa = _setting("la_kappa_per_steer", 0.0) or cal.kappa_per_steer.value
        self._last_gains = None if force is None or kappa is None or kappa <= 0.0 else Gains(force, kappa)
        return self._last_gains

    def _keep_calibration(self, cal: Calibration) -> None:
        """Says once what became of the saved calibration, and saves a live one ready on both fits for the next start,
        at most once a minute and only when it moved by 3 %.
        """
        if cal.saved != self._saved_seen:
            self._saved_seen = cal.saved
            if cal.saved == "confirmed":
                logger.info("%s calibration: using the one saved from an earlier drive until this drive's is ready; "
                            "the wheel's force direction checked against this drive", self.LABEL)
            elif cal.saved == "rejected":
                logger.warning("%s calibration: the saved one pushes the wheel the other way from this drive; it is "
                               "dropped and the calibration learns afresh", self.LABEL)
        now = time.monotonic()
        if cal.state != "ready" or cal.kappa_per_steer.state != "ready" or now - self._cal_saved_at < _CAL_SAVE_S:
            return
        self._cal_saved_at = now
        force, gain = cal.force_per_ms2, cal.kappa_per_steer.value
        old = saved_calibration()
        if old and abs(force / old[0] - 1.0) <= _CAL_SAVE_CHANGE and abs(gain / old[1] - 1.0) <= _CAL_SAVE_CHANGE:
            return
        try:
            Settings.save(values={"la_saved_calibration": {"force_per_ms2": round(force, 1),
                                                           "kappa_per_steer": round(gain, 4)}})
        except Exception:
            logger.debug("la_saved_calibration save failed", exc_info=True)

    def _log_calibration(self, cal: Calibration) -> None:
        now = time.monotonic()
        state = (cal.state, cal.kappa_per_steer.state)
        changed = state != self._cal_state
        if not changed and not (cal.servo_ready and now - self._cal_logged >= _CAL_LOG_S):
            return
        conflict = "conflict" in state
        if changed and not conflict and now - self._cal_logged < _CAL_FLAP_S:
            self._cal_state = state
            return
        self._cal_state = state
        self._cal_logged = now
        gain = cal.kappa_per_steer
        spring = math.nan
        if cal.servo_ready:
            spring = Gains(cal.force_per_ms2, gain.value).spring(_REF_MS)
        line = ("%s calibration: cornering %s %+.0f units per m/s^2 (r %.2f), steering %s %.3f 1/m per lock "
                "(r %.2f), wheel spring %+.0f units per lock at 80 km/h, %.0f s of samples")
        args = (self.LABEL, cal.state, _show(cal.force_per_ms2), cal.corr, gain.state, _show(gain.value), gain.corr,
                spring, cal.weight_s)
        if conflict:
            logger.warning(line + "; a sign flipped, assist off until restart", *args)
        else:
            logger.info(line, *args)

    def _note(self, status: PluginStatus) -> None:
        out = self._out
        if status.present and status.version < VERSION and not self._plugin_warned:
            self._plugin_warned = True
            logger.info("%s needs the version %d FFB plugin (damped servo); the installed one is version %d",
                        self.LABEL, VERSION, status.version)
        now = time.monotonic()
        if out.event and now - self._last_event_log >= _EVENT_LOG_S:
            self._last_event_log = now
            logger.info("%s %s (%s): offset %+.2f m, centre %+.2f m, drift %+.2f m/s, target %+.4f, "
                        "wheel %+.4f, hands rms %.4f, %.0f km/h", self.LABEL, _EVENTS.get(out.event, out.event),
                        out.event, out.offset_m, out.bias_m, out.drift_ms, out.target, out.steer, out.hands_rms, out.speed_kmh)

    def _account(self, out: AssistOutput, status: PluginStatus) -> None:
        """Every 10 s, one line on where the time went, how hard it held and how the hands read.

        While assisting it also shows the game's own centring and what is left of it with the assist added.
        """
        now = time.monotonic()
        if self._last_loop:
            self._seconds[out.reason] += min(now - self._last_loop, 0.2)
        self._last_loop = now
        self._peak = max(self._peak, abs(status.applied))
        self._torque_sum += abs(status.applied)
        if out.reason in ("assist", "pulse"):
            self._game_sum += abs(status.game_force)
            self._felt_sum += abs(status.game_force + status.applied + status.cancel_applied)
            self._assist_ticks += 1
        if math.isfinite(out.hands_rms):
            self._rms.append(out.hands_rms)
        self._ticks += 1
        if now - self._window_start < _SUMMARY_S:
            return
        if self._window_start and any(k not in self.IDLE for k in self._seconds):
            spent = ", ".join(f"{k} {v:.1f}" for k, v in self._seconds.most_common())
            rms = sorted(self._rms)
            median = rms[len(rms) // 2] if rms else math.nan
            n = max(1, self._assist_ticks)
            spring = self._last_gains.spring(out.speed_kmh / 3.6) if self._last_gains else math.nan
            logger.info("%s last %.0f s: %s | applied torque peak %.0f, mean %.0f, game centring %.0f, "
                        "with assist %.0f, spring %+.0f per lock at %.0f km/h, centre %+.2f m, gain %.2f, "
                        "stiffness %.2f, bend scale %.2f, servo scale %.2f, FFB %.0f Hz, hands rms median %.4f",
                        self.LABEL, now - self._window_start, spent, self._peak, self._torque_sum / max(1, self._ticks),
                        self._game_sum / n, self._felt_sum / n, spring, out.speed_kmh, self._assist.bias_m,
                        self._assist.gain, self._assist.ring_scale, self._assist.ff_scale, status.servo_scale,
                        status.frame_hz, median)
        self._seconds.clear()
        self._peak = self._torque_sum = self._game_sum = self._felt_sum = 0.0
        self._assist_ticks = 0
        self._rms = []
        self._ticks = 0
        self._window_start = now
