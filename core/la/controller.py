"""Lane keeping assist: tracks the lane (plus any wanted path) with a target wheel angle for the actuator.

Pure logic, no threads. It keeps the current lane only: no lane changes, no junctions, forks or merges,
no tight bends (``follow_all`` lifts those for testing), and the driver's hands must stay on the wheel.
A hand on the rim always wins. Design: core/la/README.md.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

from core.lane_departure.command import SERVO_OFF, ServoCommand
from core.lane_departure.detector import (
    SIDE_NONE,
    SIDE_RIGHT,
    Frame,
    Tuning,
    _steer_features,
    bend_peaks,
    lane_kappa_right,
    project,
    smooth_heading,
)

from .actuator import STRENGTH_MAX, Gains, Wheel, WheelActuator, WheelTuning, ease, follower_ahead
from .hand_push import HandPush
from .lane_model import LaneModel
from .path import LANE, WantedPath

__all__ = ["STRENGTH_MAX", "AssistOutput", "AssistTuning", "Gains", "LaneAssist", "WantedPath", "Wheel",
           "WheelTuning", "bend_ceiling", "soft_excess"]

_INF = math.inf
_LDW = Tuning()
_KAPPA_SPAN_M = 25.0
# Silences after which the lane, its heading filter and the learned centre start over.
_FRESH_START = frozenset({"no_map", "off_route", "stale"})
_DRIVER_MUTES = frozenset({"blinker", "evade", "override", "hands_off"})
# Mutes that are a wait of their own, so ending them needs no extra re-arm delay.
_NO_REARM = frozenset({"lane_settle", "calibrating", "plugin"})


@dataclass(frozen=True)
class AssistTuning:
    arm_kmh: float = 60.0
    disarm_kmh: float = 50.0
    half_lane_m: float = 2.25
    free_m: float = 0.05
    lat_kp: float = 1.0
    lat_kd: float = 2.0
    lat_max_ms2: float = 2.0
    lat_ki: float = 0.05
    lat_i_max_ms2: float = 0.8
    lat_i_leak_s: float = 20.0
    lat_predict_s: float = 0.15
    edge_max_ms2: float = 5.0
    edge_min_ms: float = 0.3
    edge_floor_m: float = 0.2
    edge_lag_s: float = 0.2
    edge_back_m: float = 1.0
    preview_s: float = 0.2
    lead_min_s: float = 0.05
    lead_max_s: float = 0.4
    target_rate: float = 0.2
    target_hz: float = 3.0
    lane_fit_m: float = 25.0
    wheel: WheelTuning = field(default_factory=WheelTuning)
    insist_hold: float = 0.8
    insist_hold_s: float = 0.5
    freeze_at: float = 0.4
    give_from: float = 0.5
    give_full: float = 0.8
    resist_tau_s: float = 0.15
    resist_min_push: float = 500.0
    resist_home_rate: float = 0.02
    freeze_floor: float = 300.0
    give_done: float = 0.98
    psi_tau_s: float = 0.05
    psi_poly_tau_s: float = 0.15
    rise_s: float = 2.0
    fall_s: float = 1.0
    yield_s: float = 0.4
    rearm_s: float = 1.0
    soft_confirm_s: float = 0.5
    bend_soft_kappa: float = 1.0 / 200.0
    kappa_max: float = 1.0 / 100.0
    lookahead_min_m: float = 40.0
    lookahead_s: float = 2.0
    settle_s: float = 1.5
    lane_jump_m: float = 1.5
    geom_rate_ms: float = 6.0
    geom_accel_ms2: float = 20.0
    geom_hold_s: float = 0.5
    carry_max_s: float = 10.0
    carry_max_m: float = 6.0
    max_age_s: float = 0.25
    brake_gate: float = 0.7
    blinker_hold_s: float = 2.0
    steer_window_s: float = 1.0
    steer_peak_dt_s: float = 0.3
    steer_min_s: float = 0.5
    evade_net: float = 0.15
    evade_rate: float = 0.6
    evade_hold_s: float = 3.0
    override_err: float = 0.015
    override_still: float = 0.05
    override_s: float = 0.3
    override_hold_s: float = 2.0
    rejoin_m: float = 0.6
    hands_tau_s: float = 1.0
    hands_window_s: float = 3.0
    hands_rms: float = 0.0008
    hands_move_rate: float = 0.02
    hands_warn_s: float = 10.0
    hands_off_s: float = 15.0
    weave_flips: int = 3
    weave_window_s: float = 8.0
    weave_push_ms2: float = 0.3
    weave_cut: float = 0.5
    weave_floor: float = 0.15
    weave_recover_s: float = 60.0
    bias_tau_s: float = 20.0
    bias_max_m: float = 0.3
    bias_still_ms: float = 0.1
    bias_kappa_max: float = 1.0 / 1000.0
    ff_kappa_min: float = 1.0 / 700.0
    ff_tau_s: float = 15.0
    ff_min: float = 0.8
    ff_max: float = 1.2
    jump_m: float = 80.0
    allow_tmp: bool = False
    follow_all: bool = False
    follow_soft_ms2: float = 4.0
    follow_max_ms2: float = 6.0
    follow_release_ms3: float = 1.0
    always_on: bool = False
    pulse: bool = False
    pulse_rise_s: float = 0.15
    pulse_fall_s: float = 0.5
    pulse_inset_m: float = 0.5


@dataclass(frozen=True)
class AssistOutput:
    """``servo`` and ``torque`` go to the plugin as they are. ``rumble`` asks for the hands-on warning."""

    servo: ServoCommand = SERVO_OFF
    torque: float = 0.0
    authority: float = 0.0
    reason: str = "off"
    event: str = ""
    rumble: bool = False
    offset_m: float = math.nan
    drift_ms: float = 0.0
    bias_m: float = 0.0
    kappa_ff: float = 0.0
    accel_fb_ms2: float = 0.0
    target: float = 0.0
    steer: float = 0.0
    hands_rms: float = math.nan
    hands_off_s: float = 0.0
    speed_kmh: float = 0.0
    source: str = "lane"
    fitted: bool = False
    preview_s: float = 0.0


@dataclass(frozen=True)
class _Geom:
    """``offset`` is metres right of the lane centre at the rear axle, ``psi`` the heading against the lane there.

    ``soft`` names a route condition worth leaving alone if it lasts, ``ceiling`` the authority the bend allows,
    ``fitted`` whether the lane fit gave the numbers (else the route polyline did).
    """

    offset: float
    psi: float
    kappa: float
    kappa_ahead: float
    soft: str
    ceiling: float
    fitted: bool = False


def soft_excess(u: float, width_left: float, width_right: float) -> float:
    """Distance past a soft free zone: about ``u**3 / (3 w**2)`` inside it, ``u - w`` well outside, smooth throughout."""
    width = width_right if u >= 0.0 else width_left
    return u - width * math.tanh(u / width)


def bend_ceiling(kappa: float, tn: AssistTuning) -> float:
    """Authority a bend leaves: full below ``bend_soft_kappa``, none from ``kappa_max``, smooth between."""
    return 1.0 - ease((kappa - tn.bend_soft_kappa) / (tn.kappa_max - tn.bend_soft_kappa))


class LaneAssist:
    def __init__(self, tuning: AssistTuning | None = None) -> None:
        self.tune = tuning or AssistTuning()
        self._fast = False
        self._last_xz: tuple[float, float] | None = None
        self._last_t: float | None = None
        self._steer: deque[tuple[float, float]] = deque()
        self._auth = 0.0
        self._bias = 0.0
        self._ff = 1.0
        self._target: float | None = None
        self._blink_until = self._override_until = self._evade_until = self._geom_until = -_INF
        self._muted_at = -_INF
        self._soft_reason = ""
        self._soft_since = 0.0
        self._gain = 1.0
        self._flips: deque[float] = deque()
        self._push_sign = 0
        self._wheel = WheelActuator(self.tune.wheel)
        self._lane = LaneModel(max(1.0, self.tune.lane_fit_m))
        self._target_rate = self._plan_rate = self._plan_accel = 0.0
        self._lat_hold = 0.0
        self._preview = self.tune.preview_s
        self._speed = 0.0
        self._gap = False
        self._pulse_side = SIDE_NONE
        self._fresh_lane()
        self._clear_hands()
        self._clear_resist()
        self._rejoin = True

    def _fresh_lane(self) -> None:
        self._lane_since: float | None = None
        self._last_offset: float | None = None
        self._psi: float | None = None
        self._override_t = 0.0
        self._insist_t = 0.0
        self._lat_i = 0.0
        self._shift = 0.0
        self._carry_since = 0.0
        self._geo: tuple[float, float] | None = None

    def _clear_resist(self) -> None:
        self._resist = 0.0
        self._give = 0.0
        self._frozen: float | None = None

    def _clear_hands(self) -> None:
        self._err_mean: float | None = None
        self._hands: deque[tuple[float, float]] = deque()
        self._hands_off_t = 0.0
        self._hands_latched = False

    def _restart(self) -> None:
        self._steer.clear()
        self._clear_resist()
        self._fresh_lane()
        self._clear_hands()
        self._auth = self._bias = self._lat_hold = 0.0
        self._target = None
        self._target_rate = self._plan_rate = self._plan_accel = 0.0
        self._lane.reset()
        self._blink_until = self._override_until = self._evade_until = self._muted_at = self._geom_until = -_INF
        self._soft_reason = ""

    @property
    def ff_scale(self) -> float:
        """The bend feedforward's learned scale: what the truck needs in steady bends per unit of map curvature."""
        return self._ff

    @property
    def bias_m(self) -> float:
        return self._bias

    @property
    def gain(self) -> float:
        """Below 1 after the weave guard cut it."""
        return self._gain

    @property
    def ring_scale(self) -> float:
        """Below 1 after the ring guard cut the pull."""
        return self._wheel.ring_scale

    def update(self, f: Frame | None, gains: Gains | None, wheel: Wheel, tmp: bool = False,
               path: WantedPath = LANE, engage: int = SIDE_NONE, hand: HandPush | None = None) -> AssistOutput:
        """One tick. ``gains`` is None while the calibration is learning; ``wheel.ok`` needs a version 4 plugin.

        ``path`` is what a path source wants on top of the lane; the lane follower passes ``LANE``. ``hand`` is the
        driver's estimated push (``hand_push.py``), which a pulse yields to.
        """
        if f is None:
            self._restart()
            self._last_xz = self._last_t = None
            return AssistOutput(reason="no_data")
        dt = self._advance(f)
        self._speed = f.speed_ms
        kmh = f.speed_ms * 3.6
        reason = self._gate(f, kmh, tmp)
        geom = None
        lag = self._wheel.lag_s(gains, f.speed_ms)
        self._preview = self.tune.preview_s if lag is None else max(self.tune.lead_min_s, min(self.tune.lead_max_s, lag))
        if not reason:
            reason, geom = self._geometry(f, dt)
        if geom is None:
            self._gap = True
            self._shift = 0.0
            if reason in _FRESH_START:
                self._fresh_lane()
                self._lane.reset()
                self._bias = 0.0
            return self._fade(f.t, dt, reason, kmh, gains, wheel)
        return self._assist(f, dt, kmh, geom, gains, wheel, path, engage, hand)

    def _advance(self, f: Frame) -> float:
        tn = self.tune
        dt = 0.0
        if self._last_xz is not None and self._last_t is not None:
            jumped = math.hypot(f.x - self._last_xz[0], f.z - self._last_xz[1]) > tn.jump_m
            if jumped or f.t - self._last_t > 1.0:
                self._restart()
            else:
                dt = max(0.0, f.t - self._last_t)
        self._last_xz = (f.x, f.z)
        self._last_t = f.t
        self._steer.append((f.t, f.steer_right))
        while f.t - self._steer[0][0] > tn.steer_window_s:
            self._steer.popleft()
        return dt

    def _gate(self, f: Frame, kmh: float, tmp: bool) -> str:
        tn = self.tune
        if f.route is None:
            return "no_map"
        if f.paused:
            return "paused"
        if f.age_s > tn.max_age_s:
            return "stale"
        if f.gear < 0:
            return "reverse"
        if kmh >= tn.arm_kmh:
            self._fast = True
        elif kmh < tn.disarm_kmh:
            self._fast = False
        if not self._fast:
            return "speed"
        if tmp and not tn.allow_tmp:
            return "tmp"
        if f.aeb_active:
            return "aeb"
        if f.brake >= tn.brake_gate:
            return "brake"
        return ""

    def _geometry(self, f: Frame, dt: float) -> tuple[str, _Geom | None]:
        """Only a truck off its route has no geometry; forks, merges and junction lanes are soft and still measured.

        Offset, heading and curvature come from the lane fit while it has vertices, the feedforward's curvature as far
        ahead as the wheel trails its target. ``follow_all`` drops the soft gates and limits bends by the lateral
        acceleration the lane asks for, held a while after it eases; ``always_on`` drops the bend limit too.
        """
        tn = self.tune
        route = f.route
        pts = getattr(route, "pts", ())
        fx, fz = -math.sin(f.yaw_rad), -math.cos(f.yaw_rad)
        rx, rz = f.x + fx * f.path_m, f.z + fz * f.path_m
        hit = project(pts, rx, rz)
        if hit is None or abs(hit[1]) > 3.0 * tn.half_lane_m:
            return "off_route", None
        s0 = hit[0]
        left, right = bend_peaks(pts, s0, f.speed_ms, _LDW)
        self._lane.observe(route)
        fit = self._lane.local(rx, rz, fx, fz, self._preview * f.speed_ms) if tn.lane_fit_m > 0.0 else None
        soft = ""
        look = max(tn.lookahead_min_m, tn.lookahead_s * f.speed_ms)
        if getattr(route, "reach_m", 0.0) - s0 < look:
            soft = "short_route"
        for name in ("branch", "merge"):
            at = getattr(route, name + "_s", None)
            if not soft and at is not None and at - s0 < look:
                soft = name
        kind = getattr(route, "kind", "")
        if not soft and kind != "road" and not (kind == "prefab" and max(left, right) <= _LDW.prefab_max_kappa):
            soft = "route_kind"
        if fit is not None:
            offset, psi, kappa, ahead = fit.offset, fit.psi, fit.kappa, fit.kappa_ahead
        else:
            offset = hit[1]
            tangent = smooth_heading(pts, s0)
            ego = math.atan2(fz, fx)
            psi = math.atan2(math.sin(ego - tangent), math.cos(ego - tangent))
            kappa = lane_kappa_right(pts, s0)
            ahead = lane_kappa_right(pts, s0 + max(0.0, self._preview * f.speed_ms - 0.5 * _KAPPA_SPAN_M))
        peak = max(abs(kappa), left, right)
        if tn.always_on or tn.pulse:
            soft = ""
            ceiling = 1.0
        elif tn.follow_all:
            soft = ""
            lat = max(abs(kappa), abs(ahead)) * f.speed_ms * f.speed_ms
            self._lat_hold = max(lat, self._lat_hold - dt * tn.follow_release_ms3)
            ceiling = 1.0 - ease((self._lat_hold - tn.follow_soft_ms2) / (tn.follow_max_ms2 - tn.follow_soft_ms2))
        else:
            ceiling = bend_ceiling(peak, tn)
        return "", _Geom(offset, psi, kappa, ahead, soft, ceiling, fit is not None)

    def _assist(self, f: Frame, dt: float, kmh: float, g: _Geom, gains: Gains | None, wheel: Wheel,
                path: WantedPath, engage: int = SIDE_NONE, hand: HandPush | None = None) -> AssistOutput:
        tn = self.tune
        t = f.t
        if engage != SIDE_NONE:
            if engage != self._pulse_side:
                self._clear_resist()
            self._pulse_side = engage
        event = self._carry(t, g.offset, f.speed_ms)
        g = replace(g, offset=g.offset + self._shift)
        if self._lane_since is None:
            self._lane_since = t
        tau = tn.psi_tau_s if g.fitted else tn.psi_poly_tau_s
        blend = 1.0 if self._psi is None or dt <= 0.0 else 1.0 - math.exp(-dt / tau)
        self._psi = g.psi if self._psi is None else self._psi + blend * (g.psi - self._psi)
        drift = f.speed_ms * math.sin(self._psi)
        if self._implausible(dt, g.offset, drift):
            event = event or ("geometry" if self._geom_until <= t else "")
            self._fresh_lane()
            self._lane_since, self._geom_until = t, t + tn.geom_hold_s
        follow = tn.follow_all or tn.always_on
        if (f.blinker_left or f.blinker_right or f.hazards) and not tn.always_on:
            self._blink_until = t + tn.blinker_hold_s
        net, right_rate, left_rate = _steer_features(self._steer, tn)
        swerve = abs(net) >= tn.evade_net or max(right_rate, left_rate) >= tn.evade_rate
        if swerve and not tn.always_on:
            event = event or ("evade" if self._evade_until <= t and not tn.pulse else "")
            self._evade_until = t + tn.evade_hold_s
        settling = t - self._lane_since < tn.settle_s
        held = settling or bool(self._shift) or self._blink_until > t or self._evade_until > t
        self._learn_bias(dt, g, drift, held or swerve)
        # Offset and drift ``lat_predict_s`` on, from the wheel's angle, which the lane shows about that late; zero in
        # a steady bend, capped at the correction's own ceiling so a stalled steering reading cannot run away with it.
        rel, ahead = g.offset - self._bias - path.offset_m, drift
        if tn.pulse and self._pulse_side != SIDE_NONE:
            rel = g.offset - self._pulse_line(f.half_width_m)
        h = tn.lat_predict_s
        if h > 0.0 and gains is not None and wheel.ok:
            slip = f.speed_ms * f.speed_ms * (-wheel.steer * gains.kappa_per_steer - self._ff * g.kappa - path.kappa)
            slip = max(-tn.lat_max_ms2, min(tn.lat_max_ms2, slip))
            rel, ahead = rel + h * drift + 0.5 * h * h * slip, drift + h * slip
        accel = self._lateral(dt, rel, ahead)
        if settling and follow:
            accel = 0.0
        if tn.pulse and engage == SIDE_NONE:
            accel = 0.0
        weaving = self._weave(t, dt, accel) if not tn.pulse else False
        if not settling:
            accel = self._edge_stop(accel, g.offset, drift, f.half_width_m)
        if weaving:
            event = event or "weave"
        want = None
        if gains is not None:
            kappa_cmd = self._ff * g.kappa_ahead + path.kappa + accel / max(f.speed_ms * f.speed_ms, 25.0)
            want = -kappa_cmd / gains.kappa_per_steer if self._geom_until <= t else None
            self._learn_ff(dt, g, drift, kappa_cmd, held)
        self._track_target(dt, want, wheel)
        err = (self._target - wheel.steer) if self._target is not None and wheel.ok else 0.0
        if not tn.pulse and self._wheel.guard(t, dt, err, wheel, self._auth, self._target):
            event = event or "ring"
        self._watch_resist(dt, hand, wheel)
        if self._override(t, dt, err, g.offset - self._bias, wheel):
            event = event or "override"
        hands = self._watch_hands(t, dt, err, wheel)
        if hands == "hands_off" and not self._hands_latched:
            event = event or "hands_off"
        self._hands_latched = self._hands_latched or hands == "hands_off"
        soft = self._soft(t, g.soft)
        mute = self._mute(t, settling and not follow, gains, wheel) or soft
        if tn.pulse:
            self._step_pulse(dt, engage, mute)
        else:
            self._step_authority(t, dt, mute, g.ceiling)
        if self._auth <= 0.0:
            self._target = wheel.steer if wheel.ok else None
            self._target_rate = self._plan_rate = self._plan_accel = 0.0
            self._pulse_side = SIDE_NONE
            self._clear_resist()
        reason = mute or ("bend_sharp" if g.ceiling <= 0.0 else "bend" if g.ceiling < 1.0 else "assist")
        if tn.pulse and not mute:
            reason = "pulse" if self._auth > 0.0 else "standby"
        only = (1 if self._pulse_side == SIDE_RIGHT else -1) if tn.pulse and self._pulse_side != SIDE_NONE else 0
        servo = self._wheel.commands(gains, self._target, self._target_rate, self._auth, wheel, self._speed, only,
                                     (self._plan_rate, self._plan_accel))
        if self._frozen is not None and servo.cap:
            servo = replace(servo, cap=min(servo.cap, round(self._frozen * ease(self._auth))))
        return AssistOutput(
            servo=servo, authority=self._auth, reason=reason, event=event, source=path.source, fitted=g.fitted,
            rumble=self._hands_off_t >= tn.hands_warn_s and self._auth > 0.0, offset_m=g.offset, drift_ms=drift,
            bias_m=self._bias, kappa_ff=g.kappa_ahead, accel_fb_ms2=accel,
            target=self._target if self._target is not None else 0.0, steer=wheel.steer,
            hands_rms=self._hands_rms(), hands_off_s=self._hands_off_t, speed_kmh=kmh, preview_s=self._preview,
        )

    def _carry(self, t: float, offset: float, speed: float) -> str:
        """The route jumping a lane sideways starts the lane over, unless ``always_on``: then the lane the truck left
        stays the reference (offset carried across), up to ``carry_max_s``; ``carry`` on the tick it starts.

        Never across a gap in the geometry, nor for a jump or a shift past ``carry_max_m``; expiry waits for a return.
        """
        tn = self.tune
        event = ""
        jump = offset - self._last_offset if self._last_offset is not None else 0.0
        if abs(jump) > tn.lane_jump_m:
            shift = self._shift - jump
            if not tn.always_on or self._gap or abs(jump) > tn.carry_max_m or abs(shift) > tn.carry_max_m:
                self._fresh_lane()
            else:
                event = "" if self._shift else "carry"
                self._carry_since = t if not self._shift else self._carry_since
                self._shift = 0.0 if abs(shift) < tn.lane_jump_m else shift
        drift = speed * math.sin(self._psi) if self._psi is not None else 0.0
        homing = (offset + self._shift) * drift < 0.0 and abs(drift) >= tn.edge_min_ms
        if self._shift and t - self._carry_since > tn.carry_max_s and not homing:
            self._fresh_lane()
        self._last_offset = offset
        self._gap = False
        return event

    def _implausible(self, dt: float, offset: float, drift: float) -> bool:
        """The lane moving under the truck faster than any truck moves: a roundabout's or junction's lanes trading
        places. Chasing it swung the target and the ring guard took the servo out for two minutes (drive 10).
        """
        tn = self.tune
        last, self._geo = self._geo, (offset, drift)
        if last is None or not 0.0 < dt <= 0.2:
            return False
        sideways = (offset - last[0]) / dt - 0.5 * (drift + last[1])
        return abs(sideways) > tn.geom_rate_ms or abs(drift - last[1]) / dt > tn.geom_accel_ms2

    def _lateral(self, dt: float, rel: float, drift: float) -> float:
        """Rightward correction toward the learned centre: a light spring past a 0.05 m free zone, and damping.

        A slow bounded integral, built only while fully engaged and steady, takes out what a wrong steering gain
        or map curvature leaves standing in a bend.
        """
        tn = self.tune
        excess = soft_excess(rel, tn.free_m, tn.free_m)
        if self._auth >= 0.9 and dt > 0.0 and abs(drift) < tn.bias_still_ms and not self._shift:
            self._lat_i = max(-tn.lat_i_max_ms2, min(tn.lat_i_max_ms2, self._lat_i - tn.lat_ki * rel * dt))
        elif self._auth < 0.9 and dt > 0.0:
            self._lat_i *= math.exp(-dt / tn.lat_i_leak_s)
        a_max = tn.lat_max_ms2
        return a_max * math.tanh(-(tn.lat_kp * excess + tn.lat_kd * drift) * self._gain / a_max) + self._lat_i

    def _edge_stop(self, accel: float, offset: float, drift: float, half_width: float) -> float:
        """At least the lateral decel that stops the drift before the body reaches the lane line, up to 5 m/s^2.

        Room counts from where the command takes effect. Fades out a metre short of the centre on the far side, so
        it never holds back the pull home from beyond the other line (it capped that return near 0.3 m/s, drive 8).
        """
        tn = self.tune
        side = 1.0 if drift > 0.0 else -1.0
        weight = ease((side * offset + tn.edge_back_m) / tn.edge_back_m)
        if abs(drift) < tn.edge_min_ms or weight <= 0.0:
            return accel
        room = tn.half_lane_m - half_width - side * offset - abs(drift) * (self._preview + tn.edge_lag_s)
        need = min(tn.edge_max_ms2, drift * drift / (2.0 * max(tn.edge_floor_m, room)))
        bound = min(accel, -need) if side > 0.0 else max(accel, need)
        return accel + weight * (bound - accel)

    def _track_target(self, dt: float, want: float | None, wheel: Wheel) -> None:
        """Bumpless: the target starts at the wheel and follows the wanted angle, at most at ``target_rate``.

        A critically damped second-order follower at ``target_hz``, so its rate is smooth: that rate goes to the
        plugin, which damps on it and moves the target on between ticks.
        """
        tn = self.tune
        if self._target is None:
            self._target = wheel.steer if wheel.ok else want
            self._target_rate = self._plan_rate = self._plan_accel = 0.0
        if want is None or self._target is None or dt <= 0.0:
            self._target_rate = self._plan_rate = self._plan_accel = 0.0
            return
        w = 2.0 * math.pi * tn.target_hz
        e0, v0 = self._target - want, self._target_rate
        decay = math.exp(-w * dt)
        e1 = (e0 + (v0 + w * e0) * dt) * decay
        v1 = (v0 - w * (v0 + w * e0) * dt) * decay
        if abs(v1) > tn.target_rate:
            v1 = math.copysign(tn.target_rate, v1)
            step = max(-tn.target_rate * dt, min(tn.target_rate * dt, want + e1 - self._target))
            self._target += step
            self._plan_rate, self._plan_accel = v1, 0.0
        else:
            self._target = want + e1
            self._plan_rate, self._plan_accel = follower_ahead(e1, v1, w, self._wheel.tune.plan_s)
        self._target_rate = v1

    def _pulse_line(self, half_width: float) -> float:
        """Where a pulse steers the rear axle: ``pulse_inset_m`` inside the line it saves, never past the centre."""
        tn = self.tune
        line = max(0.0, tn.half_lane_m - half_width - tn.pulse_inset_m)
        return line if self._pulse_side == SIDE_RIGHT else -line

    def _step_pulse(self, dt: float, engage: int, mute: str) -> None:
        """In fast while a pulse is asked for, out when it ends; quicker still when the driver acts, and down to what
        the driver's push leaves it (``_watch_resist``) at that rate.
        """
        tn = self.tune
        ceiling = 1.0 - self._give
        if engage != SIDE_NONE and not mute and self._auth > ceiling:
            self._auth = max(ceiling, self._auth - dt / tn.yield_s)
        elif engage != SIDE_NONE and not mute:
            self._auth = min(ceiling, self._auth + dt / tn.pulse_rise_s)
        else:
            self._auth = max(0.0, self._auth - dt / (tn.yield_s if mute in _DRIVER_MUTES else tn.pulse_fall_s))

    def _watch_resist(self, dt: float, hand: HandPush | None, wheel: Wheel) -> None:
        """The driver's push against a pulse as a share of the pulse's own (Lukas, 2026-11-03): from ``freeze_at`` the
        pulse pushes no harder, and from ``give_from`` to ``give_full`` it hands back, latched for the pulse.

        Its servo's integral wound the force to 3000 in 0.7 s against a held wheel, faster than the push showed. Not
        counted while the pulse turns the wheel home: resting hands' damping and friction resist only motion.
        """
        tn = self.tune
        if not tn.pulse or self._pulse_side == SIDE_NONE or hand is None or self._auth <= 0.0 or dt <= 0.0:
            return
        only = 1 if self._pulse_side == SIDE_RIGHT else -1
        pushed = hand.push >= tn.resist_min_push and math.isfinite(hand.against)
        if not pushed or only * wheel.rate > tn.resist_home_rate:
            return
        self._resist += (1.0 - math.exp(-dt / tn.resist_tau_s)) * (hand.against / hand.push - self._resist)
        if self._frozen is None and self._resist >= tn.freeze_at:
            self._frozen = max(tn.freeze_floor, hand.push) / max(0.2, ease(self._auth))
        self._give = max(self._give, ease((self._resist - tn.give_from) / (tn.give_full - tn.give_from)))

    def _override(self, t: float, dt: float, err: float, rel: float, wheel: Wheel) -> bool:
        """A wheel held still away from the target is the driver steering; True on the tick it yields.

        Still, because a released wheel swinging past the target is not a driver. So is a still wheel pushed near the
        servo's cap net of the game's force: firm hands hold it. In a bend the servo carries the game's centring.
        After the hold it rejoins only near a lane centre, so a lane change without a blinker is not pulled back. A pulse
        counts only a wheel held short of its target: past it, toward home, its one-sided servo does nothing. It also
        yields once the driver's push has taken all of its authority (``_watch_resist``).
        """
        tn = self.tune
        if tn.always_on:
            return False
        if tn.pulse and self._give >= tn.give_done and self._override_until <= t:
            self._override_t = self._insist_t = 0.0
            self._override_until = t + tn.override_hold_s
            self._rejoin = False
            return True
        if self._auth < 0.3 or self._override_until > t:
            self._override_t = self._insist_t = 0.0
            if self._override_until > t and abs(rel) > tn.rejoin_m and self._rejoin:
                self._override_until = max(self._override_until, t + dt)
            return False
        short = (err if self._pulse_side == SIDE_RIGHT else -err) if tn.pulse else abs(err)
        held = short > tn.override_err and abs(wheel.rate) < tn.override_still
        self._override_t = self._override_t + dt if held else 0.0
        cap = self._wheel.cap(self._auth)
        cap = min(cap, self._frozen * ease(self._auth)) if self._frozen is not None else cap
        net = wheel.push + wheel.game
        pinned = cap > 0.0 and abs(net) >= tn.insist_hold * cap and abs(wheel.rate) < tn.override_still
        self._insist_t = self._insist_t + dt if pinned else 0.0
        if self._override_t < tn.override_s and self._insist_t < tn.insist_hold_s:
            return False
        self._override_t = self._insist_t = 0.0
        self._override_until = t + tn.override_hold_s
        self._rejoin = True
        return True

    def _watch_hands(self, t: float, dt: float, err: float, wheel: Wheel) -> str:
        """Hands leave a tremor in the servo error; a wheel that follows the servo exactly has none.

        Returns ``hands_off`` once the error has been that smooth for ``hands_off_s`` while engaged.
        """
        tn = self.tune
        if not wheel.ok or dt <= 0.0 or tn.always_on or tn.pulse:
            return ""
        if self._hands_latched:
            if abs(wheel.rate) >= tn.hands_move_rate:
                self._clear_hands()
            return "hands_off" if self._hands_latched else ""
        if self._auth < 0.5:
            self._hands_off_t = 0.0
            self._err_mean = None
            self._hands.clear()
            return ""
        self._err_mean = err if self._err_mean is None else self._err_mean + (err - self._err_mean) * (
            1.0 - math.exp(-dt / tn.hands_tau_s))
        self._hands.append((t, (err - self._err_mean) ** 2))
        while t - self._hands[0][0] > tn.hands_window_s:
            self._hands.popleft()
        if t - self._hands[0][0] < 0.9 * tn.hands_window_s:
            return ""
        self._hands_off_t = self._hands_off_t + dt if self._hands_rms() < tn.hands_rms else 0.0
        return "hands_off" if self._hands_off_t >= tn.hands_off_s else ""

    def _hands_rms(self) -> float:
        if not self._hands:
            return math.nan
        return math.sqrt(sum(h[1] for h in self._hands) / len(self._hands))

    def _soft(self, t: float, soft: str) -> str:
        """A fork, merge or junction lane counts once it has lasted ``soft_confirm_s``; a router blip does not."""
        if soft != self._soft_reason:
            self._soft_reason = soft
            self._soft_since = t
        return soft if soft and t - self._soft_since >= self.tune.soft_confirm_s else ""

    def _weave(self, t: float, dt: float, accel: float) -> bool:
        """Lowers the gain when the correction keeps flipping sides (overreacting), gives it back slowly; True on a cut.

        Counts only while engaged: unassisted, the flips are the driver's own steering.
        """
        tn = self.tune
        if dt > 0.0:
            self._gain = min(1.0, self._gain + dt * (1.0 - tn.weave_floor) / tn.weave_recover_s)
        if self._auth < 0.5:
            self._flips.clear()
            self._push_sign = 0
            return False
        sign = (accel > tn.weave_push_ms2) - (accel < -tn.weave_push_ms2)
        if not sign or sign == self._push_sign:
            return False
        flipped = self._push_sign != 0
        self._push_sign = sign
        if not flipped:
            return False
        self._flips.append(t)
        while t - self._flips[0] > tn.weave_window_s:
            self._flips.popleft()
        if len(self._flips) < tn.weave_flips:
            return False
        self._flips.clear()
        self._gain = max(tn.weave_floor, self._gain * tn.weave_cut)
        return True

    def _step_authority(self, t: float, dt: float, mute: str, ceiling: float) -> None:
        """Fades out on a mute, follows the bend's ceiling down, and rises only after ``rearm_s`` of quiet."""
        tn = self.tune
        if mute:
            if mute not in _NO_REARM:
                self._muted_at = t
            self._auth = max(0.0, self._auth - dt / (tn.yield_s if mute in _DRIVER_MUTES else tn.fall_s))
        elif self._auth > ceiling:
            self._auth = max(ceiling, self._auth - dt / tn.fall_s)
        elif t - self._muted_at >= tn.rearm_s:
            self._auth = min(ceiling, self._auth + dt / tn.rise_s)

    def _learn_ff(self, dt: float, g: _Geom, drift: float, kappa_cmd: float, hold: bool) -> None:
        """In a steady bend the command is what the truck needs; its ratio to the map curvature scales the feedforward.

        The map and the steering gain disagreed by 7 to 12 % on the drives, enough to cut every bend.
        """
        tn = self.tune
        steady = self._auth >= 0.9 and abs(drift) < tn.bias_still_ms and abs(g.kappa) >= tn.ff_kappa_min
        if hold or dt <= 0.0 or not steady or g.kappa * g.kappa_ahead <= 0.0:
            return
        ratio = max(tn.ff_min, min(tn.ff_max, kappa_cmd / g.kappa_ahead))
        self._ff += (ratio - self._ff) * min(1.0, dt / tn.ff_tau_s)

    def _learn_bias(self, dt: float, g: _Geom, drift: float, hold: bool) -> None:
        """Follow where the driver keeps the truck, slowly and bounded: the map has no paint and can sit off it."""
        tn = self.tune
        if hold or dt <= 0.0 or abs(drift) >= tn.bias_still_ms or abs(g.kappa) >= tn.bias_kappa_max:
            return
        self._bias += (g.offset - self._bias) * min(1.0, dt / tn.bias_tau_s)
        self._bias = max(-tn.bias_max_m, min(tn.bias_max_m, self._bias))

    def _mute(self, t: float, settling: bool, gains: Gains | None, wheel: Wheel) -> str:
        if gains is None:
            return "calibrating"
        if not wheel.ok:
            return "plugin"
        if settling:
            return "lane_settle"
        if self._blink_until > t and not self.tune.pulse:
            return "blinker"
        if self._evade_until > t and not self.tune.pulse:
            return "evade"
        if self._override_until > t:
            return "override"
        if self._hands_latched:
            return "hands_off"
        return ""

    def _fade(self, t: float, dt: float, reason: str, kmh: float, gains: Gains | None, wheel: Wheel) -> AssistOutput:
        """No usable lane: the target stays where it is while the authority fades, and the re-arm wait starts."""
        self._muted_at = t
        self._auth = max(0.0, self._auth - dt / self.tune.fall_s)
        self._target_rate = self._plan_rate = self._plan_accel = 0.0
        self._lat_hold = 0.0
        if self._auth <= 0.0:
            self._target = wheel.steer if wheel.ok else None
        servo = self._wheel.commands(gains, self._target, 0.0, self._auth, wheel, self._speed)
        return AssistOutput(
            servo=servo, authority=self._auth, reason=reason, bias_m=self._bias,
            target=self._target if self._target is not None else 0.0, steer=wheel.steer, speed_kmh=kmh,
        )
