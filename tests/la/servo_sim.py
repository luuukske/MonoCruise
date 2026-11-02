"""Closed-loop model for the LA: wheel and hands, the game's force, the plugin, the truck, the controller.

Numbers are assumptions, not measurements: a rim that rings near 1.5 Hz on the game's spring, the plugin at 60 Hz
with one frame of delay and the actuator 8 ms after that, the truck's yaw following its steering with 0.3 s lag.
``lagged`` makes the game's force follow the truck's cornering (self-aligning torque) instead of the wheel angle, as
the calibration measured it in the game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

from core.la.controller import AssistTuning, Gains, LaneAssist, Wheel
from core.lane_departure.command import servo_fields
from tests.lane_departure.test_detector import DT, SPEED, START_S, _arc_x, _arc_z, frame_at, make_route

SPRING = 34570.0
GAIN = 0.19
# Game force per m/s^2 that gives SPRING at the model's speed.
FORCE = SPRING / (GAIN * SPEED * SPEED)
SERVO_HZ = 60
SUBSTEPS = 20
ACTUATOR_S = 0.008
TORQUE_TAU_S = 0.06
RATE_TAU_S = 0.006
CANCEL_TAU_S = 1.5
EXTRAP_S = 0.15
D_CAP = 1500.0
I_SHARE = 1.0
SERVO_IN_S = 0.3
SERVO_OUT_S = 0.2
SERVO_SLEW = 30000.0
BUZZ_MIN = 300.0
BUZZ_SWING = 0.002
BUZZ_FLIPS = 4
BUZZ_WINDOW_S = 0.5
BUZZ_CUT = 0.5
BUZZ_FLOOR = 0.1
BUZZ_RECOVER = 0.25


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _clamp(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


@dataclass
class Plugin:
    """Mirror of plugins/ffb/ffb_signal.h (layout 4): hold low-pass, damped PID with the target moved on at its
    rate, a share of the game's steady force cancelled when asked, fade, slew and buzz guard.
    """

    base: float = 0.0
    rate: float = 0.0
    prev: float | None = None
    gain: float = 0.0
    value: float = 0.0
    integ: float = 0.0
    scale: float = 1.0
    clock: float = 0.0
    sign: int = 0
    pushed: bool = False
    swing: int = 0
    peak: float | None = None
    flips: list = field(default_factory=list)
    game_lp: float = 0.0
    sent: tuple | None = None
    age: float = 0.0
    cancel: float = 0.0

    def _buzz(self, want: float, angle: float) -> None:
        sign = 1 if want > BUZZ_MIN else (-1 if want < -BUZZ_MIN else 0)
        if sign and sign != self.sign:
            self.pushed = self.pushed or self.sign != 0
            self.sign = sign
        if angle != angle:
            self.peak, self.swing, self.pushed = None, 0, False
            return
        if self.peak is None:
            self.peak = angle
            return
        if not self.swing:
            if abs(angle - self.peak) >= BUZZ_SWING:
                self.swing, self.peak = (1 if angle > self.peak else -1), angle
            return
        if (angle - self.peak) * self.swing >= 0.0:
            self.peak = angle
            return
        if abs(angle - self.peak) < BUZZ_SWING:
            return
        self.swing, self.peak = -self.swing, angle
        if not self.pushed:
            return
        self.pushed = False
        self.flips = [f for f in self.flips if self.clock - f <= BUZZ_WINDOW_S][:BUZZ_FLIPS - 1]
        self.flips.append(self.clock)
        if len(self.flips) >= BUZZ_FLIPS:
            self.scale = max(BUZZ_FLOOR, self.scale * BUZZ_CUT)
            self.flips = []

    def step(self, servo, torque, steer, dt, game=0.0):
        fields = servo_fields(servo)
        target, kp, ki, cap, direction, rate, kd, cancel, only = fields
        if fields != self.sent:
            self.sent = fields
            self.age = 0.0
        else:
            self.age += dt
        self.base += (max(-3000.0, min(3000.0, torque)) - self.base) * (1.0 - math.exp(-dt / TORQUE_TAU_S))
        if self.prev is not None:
            self.rate += ((steer - self.prev) / dt - self.rate) * (1.0 - math.exp(-dt / RATE_TAU_S))
        self.prev = steer
        self.game_lp += (game - self.game_lp) * (1.0 - math.exp(-dt / CANCEL_TAU_S))
        self.clock += dt
        self.scale = min(1.0, self.scale + BUZZ_RECOVER * dt)
        on = bool(direction)
        want = 0.0
        if on:
            cap = min(cap, 3000.0)
            aim = target + rate * min(self.age, EXTRAP_S)
            err = aim - steer
            p = kp * err
            d = _clamp(kd * (rate - self.rate), D_CAP)
            if abs(p + d + self.integ) < cap or (p + d + self.integ) * err < 0.0:
                self.integ = _clamp(self.integ + ki * err * dt, I_SHARE * cap)
            if only and self.integ * only < 0.0:
                self.integ = 0.0
            u = p + d + self.integ
            want = 0.0 if only and u * only < 0.0 else direction * _clamp(u, cap)
            self._buzz(want, steer)
        else:
            self.integ = 0.0
        self.gain = min(1.0, max(0.0, self.gain + (dt / SERVO_IN_S if on else -dt / SERVO_OUT_S)))
        goal = want * self.scale * _ease(self.gain)
        self.value += max(-SERVO_SLEW * dt, min(SERVO_SLEW * dt, goal - self.value))
        if not on and self.gain <= 0.0 and abs(self.value) < 0.5:
            self.value, self.sign, self.pushed, self.swing, self.peak, self.flips = 0.0, 0, False, 0, None, []
        self.cancel = _clamp(-min(1.0, cancel) * _ease(self.gain) * self.game_lp, 6000.0) if cancel > 0.0 else 0.0
        return max(-4000.0, min(4000.0, self.base + self.value)) + self.cancel

    @property
    def servo_applied(self) -> float:
        return self.value


class Track:
    """A lane from a curvature profile ``[(length, kappa_end), ...]`` from the origin heading north.

    Curvature runs linearly to each ``kappa_end`` (a clothoid; equal ends hold an arc), positive turns right.
    """

    STEP = 0.5

    def __init__(self, pieces) -> None:
        self.s, self.k, self.x, self.z, self.phi = [0.0], [pieces[0][1] if pieces else 0.0], [0.0], [0.0], [0.0]
        k0 = self.k[0]
        for length, k1 in pieces:
            for i in range(1, int(round(length / self.STEP)) + 1):
                k = k0 + (k1 - k0) * i * self.STEP / length
                phi = self.phi[-1] + 0.5 * (k + self.k[-1]) * self.STEP
                mid = 0.5 * (phi + self.phi[-1])
                self.x.append(self.x[-1] + math.sin(mid) * self.STEP)
                self.z.append(self.z[-1] - math.cos(mid) * self.STEP)
                self.s.append(self.s[-1] + self.STEP)
                self.k.append(k)
                self.phi.append(phi)
            k0 = k1

    def _at(self, s: float, values) -> float:
        i = max(0, min(len(self.s) - 2, int(s / self.STEP)))
        u = s / self.STEP - i
        return values[i] + (values[i + 1] - values[i]) * u

    def kappa_at(self, s: float) -> float:
        return self._at(s, self.k)

    def pose(self, s: float, lateral: float, psi: float) -> tuple[float, float, float]:
        """``(x, z, yaw)`` ``lateral`` metres right of the lane at ``s``, heading ``psi`` right of it."""
        phi = self._at(s, self.phi)
        return (self._at(s, self.x) + lateral * math.cos(phi), self._at(s, self.z) + lateral * math.sin(phi),
                -(phi + psi))

    def route(self, s_ego: float, step: float = 10.0, reach: float = 200.0) -> SimpleNamespace:
        """As the map builds it: ego's foot on a ``step`` m chord, then the lane's vertices every ``step``."""
        k = int(s_ego // step)
        u = s_ego / step - k
        a, b = self.pose(k * step, 0.0, 0.0), self.pose((k + 1) * step, 0.0, 0.0)
        pts = [(a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))]
        for j in range(k + 1, k + 2 + int(reach / step)):
            if j * step > self.s[-1]:
                break
            p = self.pose(j * step, 0.0, 0.0)
            pts.append((p[0], p[1]))
        return SimpleNamespace(pts=tuple(pts), kind="road", divergence=(), branch_s=None, merge_s=None,
                               reach_m=reach)


def bend_track(radius: float, arc_m: float = 200.0, clothoid_m: float = 60.0, lead_m: float = 300.0) -> Track:
    """Straight, clothoid in, arc, clothoid out, straight."""
    k = 1.0 / radius
    return Track([(lead_m, 0.0), (clothoid_m, k), (arc_m, k), (clothoid_m, 0.0), (600.0, 0.0)])


def kink_track(k1: float = 1.0 / 150.0, k2: float = 1.0 / 240.0, ramp_m: float = 12.0, hold_m: float = 20.0) -> Track:
    """Drive 7's prefab: a gentle right bend, then a sharp right kink straight into a left one."""
    k0 = 1.0 / 500.0
    return Track([(250.0, 0.0), (40.0, k0), (100.0, k0), (ramp_m, k1), (hold_m, k1), (ramp_m, -k2), (hold_m, -k2),
                  (ramp_m, 0.0), (600.0, 0.0)])


@dataclass
class Rig:
    """``hands`` adds a driver: a spring of that many units per lock toward ``hands_at`` (counterclockwise).

    ``stick`` is dry friction in units (wheel base and resting hands); ``hand_mass`` and ``hand_damp`` add resting
    hands as inertia (times the rim's) and damping (fraction of critical on the game's spring).
    """

    spring: float = SPRING
    gain: float = GAIN
    ring_hz: float = 1.5
    friction: float = 0.3
    yaw_lag_s: float = 0.3
    lagged: bool = False
    radius: float | None = None
    hands: float = 0.0
    hands_at: float = 0.0
    stick: float = 0.0
    hand_mass: float = 0.0
    hand_damp: float = 0.0
    steer: float = 0.0
    rate: float = 0.0
    e: float = 0.0
    ed: float = 0.0
    kappa: float = 0.0
    track: Track | None = None

    def __post_init__(self) -> None:
        self.inertia = self.spring / (2.0 * math.pi * self.ring_hz) ** 2
        self.plugin = Plugin()
        self.held = 0.0
        self.clock = 0.0
        self.pending: list = []

    @property
    def lane_kappa(self) -> float:
        if self.track is not None:
            return self.track.kappa_at(START_S + SPEED * self.clock)
        return 0.0 if self.radius is None else 1.0 / self.radius

    def settle_on_lane(self) -> None:
        """Start on the lane centre with the wheel already at the lane's steering angle."""
        self.kappa = self.lane_kappa
        self.steer = -self.lane_kappa / self.gain

    def run(self, servo, torque, dt):
        """Advance by ``dt``: the plugin sampled at 60 Hz with a frame of delay, wheel and truck integrated finer."""
        frames = max(1, round(dt * SERVO_HZ))
        frame = dt / frames
        critical = 2.0 * math.sqrt(self.spring * self.inertia)
        damping = (self.friction + self.hand_damp) * critical
        inertia = self.inertia * (1.0 + self.hand_mass)
        for _ in range(frames):
            angle = self.kappa / self.gain * -1.0 if self.lagged else self.steer
            self.pending.append((self.clock + frame + ACTUATOR_S, self.plugin.step(servo, torque, self.steer, frame,
                                                                                 -self.spring * angle)))
            h = frame / SUBSTEPS
            for _ in range(SUBSTEPS):
                while self.pending and self.pending[0][0] <= self.clock:
                    self.held = self.pending.pop(0)[1]
                ccw = self.held
                self.clock += h
                angle = self.kappa / self.gain * -1.0 if self.lagged else self.steer
                game = -self.spring * angle
                driver = self.hands * (self.hands_at - self.steer) - 2.0 * math.sqrt(self.hands * self.inertia) * 0.7 * self.rate
                force = ccw + game + driver - damping * self.rate
                if self.stick and abs(self.rate) < 1e-4 and abs(force) <= self.stick:
                    self.rate = 0.0
                    force = 0.0
                elif self.stick:
                    force -= math.copysign(self.stick, self.rate if abs(self.rate) >= 1e-4 else force)
                acc = force / inertia
                self.rate += acc * h
                self.steer += self.rate * h
                want = -self.gain * self.steer
                self.kappa += (want - self.kappa) * h / self.yaw_lag_s
                self.ed += SPEED * SPEED * (self.kappa - self.lane_kappa) * h
                self.e += self.ed * h

    @property
    def psi(self) -> float:
        return math.atan2(self.ed, SPEED)

    @property
    def game_force(self) -> float:
        return -self.spring * (self.kappa / self.gain * -1.0 if self.lagged else self.steer)


def game_route(radius, s_ego: float, step: float, reach: float = 200.0) -> SimpleNamespace:
    """The route as the map builds it every tick: ego's foot on a ``step`` m chord, then the lane's own vertices."""
    k = int(s_ego // step)
    u = s_ego / step - k
    a = (_arc_x(k * step, radius), _arc_z(k * step, radius))
    b = (_arc_x((k + 1) * step, radius), _arc_z((k + 1) * step, radius))
    pts = [(a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))]
    pts += [(_arc_x(j * step, radius), _arc_z(j * step, radius)) for j in range(k + 1, k + 2 + int(reach / step))]
    return SimpleNamespace(pts=tuple(pts), kind="road", divergence=(), branch_s=None, merge_s=None, reach_m=reach)


def loop(rig: Rig, seconds: float, tuning: AssistTuning | None = None, kick=None, gains=None, path=None,
         rebuild: float | None = None, **frame_kw):
    """20 Hz controller on the rig; ``kick(t, rig)`` disturbs it. Returns ``(t, rig.e, rig.steer, output)``.

    ``rebuild`` gives the route the map's way, rebuilt every tick on vertices that many metres apart.
    """
    assist = LaneAssist(tuning or AssistTuning(hands_off_s=1e9, hands_warn_s=1e9))
    gains = gains or Gains(rig.spring / (rig.gain * SPEED * SPEED), rig.gain)
    route = make_route(rig.radius, length=2500.0 if rig.radius is None else min(2500.0, 1.5 * math.pi * abs(rig.radius)))
    out = []
    for i in range(int(seconds / DT)):
        t = i * DT
        if rebuild and rig.track is None:
            route = game_route(rig.radius, START_S + SPEED * t, rebuild)
        if kick:
            kick(t, rig)
        if rig.track is not None:
            s = START_S + SPEED * t
            route = rig.track.route(s, rebuild or 10.0)
            x, z, yaw = rig.track.pose(s, rig.e, rig.psi)
            f = frame_at(t, route, None, 0.0, 0.0, x=x, z=z, yaw_rad=yaw, steer_right=-rig.steer, **frame_kw)
        else:
            f = frame_at(t, route, rig.radius, rig.e, rig.psi, steer_right=-rig.steer, **frame_kw)
        wheel = Wheel(ok=True, steer=rig.steer, rate=rig.plugin.rate, push=rig.plugin.servo_applied,
                      buzz=rig.plugin.scale, game=rig.game_force)
        o = assist.update(f, gains, wheel, **({"path": path} if path is not None else {}))
        out.append((t, rig.e, rig.steer, o))
        rig.run(o.servo, o.torque, DT)
    return out
