"""The wheel actuator: turns a target wheel angle into the plugin's servo request.

Pure logic, shared by every path source. It knows the wheel and the target angle, nothing about lanes. The plugin
(version 4) runs a damped PID on the wheel angle at the game's FFB rate on top of the game's own force, so the
driver keeps the game's feel and where the wheel goes needs no model of that force. See core/la/README.md.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from core.lane_departure.command import SERVO_OFF, ServoCommand

STRENGTH_MAX = 2.0


@dataclass(frozen=True)
class WheelTuning:
    """``inertia`` is the rim with hands, in force units per lock per s^2; the servo is designed on it."""

    inertia: float = 650.0
    bandwidth_hz: float = 2.0
    damping: float = 0.7
    integral_share: float = 0.3
    own_floor: float = 0.25
    cancel_share: float = 0.0
    servo_cap: float = 3000.0
    strength: float = 1.0
    ring_err: float = 0.004
    ring_big: float = 0.02
    ring_big_gap_s: float = 0.5
    ring_flips: int = 4
    ring_window_s: float = 2.0
    ring_cut: float = 0.5
    ring_floor: float = 0.0
    buzz_cut: float = 0.8
    buzz_drop: float = 0.05
    ring_recover_s: float = 30.0
    ring_recover_max_s: float = 120.0
    ring_repeat_s: float = 60.0
    hold_err: float = 0.008
    hold_rate: float = 0.05
    hold_s: float = 0.3
    release_s: float = 1.0


@dataclass(frozen=True)
class Wheel:
    """The plugin's own reading of the wheel: ``steer`` is ``truck.input.steering``, counterclockwise positive.

    ``push`` is the servo torque it applies now, ``game`` the game's own force, ``buzz`` its buzz guard's scale.
    """

    ok: bool = False
    steer: float = 0.0
    rate: float = 0.0
    push: float = 0.0
    buzz: float = 1.0
    game: float = 0.0


@dataclass(frozen=True)
class Gains:
    """``force_per_ms2``: game force per m/s^2 of rightward cornering (its sign is the wheel's torque direction).

    ``kappa_per_steer``: curvature per unit of steering, the truck's steering gain.
    """

    force_per_ms2: float
    kappa_per_steer: float

    def spring(self, speed_ms: float) -> float:
        """The game's force per unit of rightward steering at this speed, near straight ahead."""
        return self.force_per_ms2 * self.kappa_per_steer * speed_ms * speed_ms


def ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


class WheelActuator:
    def __init__(self, tuning: WheelTuning | None = None) -> None:
        self.tune = tuning or WheelTuning()
        self._scale = 1.0
        self._flips: deque[float] = deque()
        self._sign = 0
        self._flip_at = -math.inf
        self._flip_target: float | None = None
        self._recover = self.tune.ring_recover_s
        self._cut_at = -math.inf
        self._buzz = 1.0
        self._still_t = 0.0
        self._held_at = -math.inf
        self._last_steer: float | None = None

    @property
    def ring_scale(self) -> float:
        """Below 1 after the ring guard cut the servo."""
        return self._scale

    def strength(self) -> float:
        return max(0.0, min(STRENGTH_MAX, self.tune.strength))

    def cap(self, authority: float) -> float:
        """The servo's cap at this authority; nothing to cap at strength 0."""
        return self.tune.servo_cap * ease(authority) if self.strength() > 0.0 else 0.0

    def gains(self, authority: float, game_spring: float = 0.0) -> tuple[float, float, float]:
        """``(kp, kd, ki)``: with the game's own spring, a second-order response at ``bandwidth_hz`` on ``inertia``.

        The game's centring answers the wheel angle almost at once, so the servo adds only what is missing (at least
        ``own_floor`` of it). Strength and the ring guard scale the total; damping and integral follow the total.
        """
        tn = self.tune
        w = 2.0 * math.pi * tn.bandwidth_hz
        total = tn.inertia * w * w * self.strength() * self._scale
        if total <= 0.0:
            return 0.0, 0.0, 0.0
        own = max(tn.own_floor * total, total - abs(game_spring))
        stiff = own + min(abs(game_spring), total)
        level = ease(authority)
        kd = 2.0 * tn.damping * math.sqrt(stiff * tn.inertia)
        return own * level, kd * level, stiff * math.sqrt(stiff / tn.inertia) * tn.integral_share * level

    def lag_s(self, gains: Gains | None, speed_ms: float) -> float | None:
        """How far the wheel trails a moving target: the game's spring the integral must carry, over its gain.

        Measured 0.15 s at 80 to 90 km/h and 0.25 s at 100 to 130 km/h, as this predicts. None without gains.
        """
        if gains is None or not gains.force_per_ms2:
            return None
        spring = abs(gains.spring(speed_ms))
        ki = self.gains(1.0, spring)[2]
        return spring / ki if ki > 0.0 else None

    def guard(self, t: float, dt: float, err: float, wheel: Wheel, authority: float,
              target: float | None = None) -> bool:
        """Cuts the servo when the wheel keeps swinging through the target or the plugin's buzz guard fires.

        Gives it back slowly; each cut within a minute of the last doubles the time. True on a cut. A wheel
        held still off the target (a hand, or a reading that stopped) snaps over once let go: that is no ring.
        """
        tn = self.tune
        if dt > 0.0:
            self._scale = min(1.0, self._scale + dt * (1.0 - tn.ring_floor) / self._recover)
            moved = abs(wheel.steer - self._last_steer) / dt if self._last_steer is not None else math.inf
            still = abs(err) > tn.hold_err and abs(wheel.rate) < tn.hold_rate and moved < tn.hold_rate
            self._still_t = self._still_t + dt if still else 0.0
            if self._still_t >= tn.hold_s:
                self._held_at = t
        self._last_steer = wheel.steer
        buzzed = wheel.buzz < tn.buzz_cut and wheel.buzz < self._buzz - tn.buzz_drop
        self._buzz = wheel.buzz if buzzed or wheel.buzz > self._buzz else self._buzz
        if authority < 0.5:
            self._flips.clear()
            self._sign = 0
            self._flip_at = -math.inf
            self._flip_target = None
            return False
        before = self._scale
        if buzzed:
            self._scale = max(tn.ring_floor, before * tn.buzz_cut)
        elif not self._swing(t, err, target):
            return False
        if before - self._scale <= 0.05:
            return False
        repeat = t - self._cut_at < tn.ring_repeat_s
        self._recover = min(tn.ring_recover_max_s, 2.0 * self._recover) if repeat else tn.ring_recover_s
        self._cut_at = t
        return True

    def _swing(self, t: float, err: float, target: float | None = None) -> bool:
        """Four swings through the target past ``ring_err`` within 2 s halve the servo; one past ``ring_big`` ends it
        if it came within ``ring_big_gap_s`` of the last (a ring; the slow overshoot of a big return is none, drive 9).

        Only a swing past a target that moved less than the swing itself counts: a ring swings round a still target,
        a wheel chasing a target that jumps (a roundabout's lanes, drive 10) does not ring.
        """
        tn = self.tune
        sign = (err > tn.ring_err) - (err < -tn.ring_err)
        if not sign or sign == self._sign:
            return False
        flipped = self._sign != 0
        self._sign = sign
        if not flipped:
            self._flip_target = target
            return False
        gap, self._flip_at = t - self._flip_at, t
        moved = abs(target - self._flip_target) if target is not None and self._flip_target is not None else 0.0
        self._flip_target = target
        if t - self._held_at <= tn.release_s or moved > abs(err):
            return False
        if abs(err) >= tn.ring_big and gap <= tn.ring_big_gap_s:
            self._flips.clear()
            self._scale = tn.ring_floor
            return True
        self._flips.append(t)
        while t - self._flips[0] > tn.ring_window_s:
            self._flips.popleft()
        if len(self._flips) < tn.ring_flips:
            return False
        self._flips.clear()
        self._scale = max(tn.ring_floor, self._scale * tn.ring_cut)
        return True

    def commands(self, gains: Gains | None, target: float | None, rate: float, authority: float,
                 wheel: Wheel, speed_ms: float = 0.0, only: int = 0) -> ServoCommand:
        """The servo request: hold ``target`` moving at ``rate``; ``cancel_share`` of the game's steady force goes.

        ``only`` +1 or -1 lets it push only that way (counterclockwise positive), never back.
        """
        if gains is None or not gains.force_per_ms2 or not wheel.ok or target is None or authority <= 0.0:
            return SERVO_OFF
        kp, kd, ki = self.gains(authority, gains.spring(speed_ms))
        if kp <= 0.0:
            return SERVO_OFF
        return ServoCommand(
            target=target, rate=rate, kp=kp, kd=kd, ki=ki, cap=round(self.cap(authority)),
            direction=1 if gains.force_per_ms2 > 0.0 else -1, cancel=self.tune.cancel_share * ease(authority), only=only,
        )
