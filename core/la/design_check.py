"""The design check: a servo design run on the standing wheel for about three seconds, measuring what the driver feels
(force chatter on a slow move, overshoot and ringing round a small step), so the calibration's design is stepped down
or up on the wheel itself rather than trusted from a model. Pure logic like ``wheel_calibration``: fed the plugin's
trace samples, asking for one servo command per step. See core/la/README.md, Wheel calibration.

Out 0.15 lock at 0.2 lock/s with 0.1 s rounded corners and back (the speed bench's ramp, so its chatter compares
with the bench's), then steps of 0.02 and 0.05 lock and back, each followed by a hold. With ``road_spring`` the ramps
run on the gains the design has at that much of the game's centring: the chatter the servo adds on the road.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from core.la.actuator import WheelActuator, WheelTuning
from core.lane_departure.command import SERVO_OFF, ServoCommand

WAIT, RUN, DONE, FAILED = "wait", "run", "done", "failed"


@dataclass(frozen=True)
class CheckTuning:
    travel: float = 0.15
    rate: float = 0.2
    corner_s: float = 0.1
    hold_s: float = 0.4
    steps: tuple[float, ...] = (0.02, 0.05)
    step_hold_s: float = 0.7
    rest_s: float = 0.3
    rest_band: float = 0.0015
    rest_max_s: float = 4.0
    travel_max: float = 0.25
    stall_s: float = 0.25
    chatter_max: float = 90.0
    overshoot_max: float = 0.35
    ring_band: float = 0.002
    ring_crossings: int = 3
    room: float = 0.7
    room_overshoot: float = 0.15
    reach_min: float = 0.7
    tail_s: float = 0.2


@dataclass(frozen=True)
class CheckResult:
    """``chatter``: the median change of the servo's force from one FFB update to the next on the slow moves (the R3
    felt smooth at 53 to 56, crunchy at 116 to 137, 2026-11-09). ``overshoot``: past the small step, as a share of it.
    ``room``: clean enough to try faster."""

    chatter: float
    overshoot: float
    crossings: int
    rest_error: float
    passed: bool
    room: bool
    why: str
    reached: float = 1.0
    stiffer: bool = False


def _schedule(ck: CheckTuning) -> list[tuple[float, float, float]]:
    """(duration, offset at its start, way: +1 or -1 for a ramp, 0 to hold) per segment."""
    ramp = ck.travel / ck.rate + ck.corner_s
    out = [(ramp, 0.0, 1.0), (ck.hold_s, ck.travel, 0.0), (ramp, ck.travel, -1.0), (ck.hold_s, 0.0, 0.0)]
    for size in ck.steps:
        out += [(ck.step_hold_s, size, 0.0), (ck.step_hold_s, 0.0, 0.0)]
    return out


def _along(ck: CheckTuning, way: float, u: float, dur: float) -> tuple[float, float, float]:
    """Offset, rate and acceleration ``u`` into a ramp of ``dur``: the rate builds and falls over ``corner_s``."""
    r, c = ck.rate * way, ck.corner_s
    if u <= 0.0 or u >= dur:
        return (0.0 if u <= 0.0 else r * (dur - c)), 0.0, 0.0
    if u < c:
        return 0.5 * r * u * u / c, r * u / c, r / c
    if u > dur - c:
        left = dur - u
        return r * (dur - c) - 0.5 * r * left * left / c, r * left / c, -r / c
    return r * (u - 0.5 * c), r, 0.0


@dataclass
class DesignCheck:
    """``tune`` the design, ``direction`` the torque sign that turns the reading up (the calibration's).

    ``road_spring`` is the game's centring the slow ramps' gains are set for; the steps always run the standing gains.
    """

    tune: WheelTuning
    direction: int
    check: CheckTuning = field(default_factory=CheckTuning)
    state: str = WAIT
    failure: str = ""
    result: CheckResult | None = None
    servo: ServoCommand = SERVO_OFF
    road_spring: float = 0.0

    def __post_init__(self) -> None:
        self._t: list[float] = []
        self._steer: list[float] = []
        self._servo: list[float] = []
        self._seq: list[int] = []
        self._marks: list[int] = []
        self._rest = 0.0
        self._start: float | None = None
        self._phase_at: float | None = None
        self._fed_at: float | None = None
        self._seen = 0
        self._gains = WheelActuator(self.tune).gains(1.0)
        self._ramp_gains = WheelActuator(self.tune).gains(1.0, self.road_spring) if self.road_spring > 0.0 else self._gains
        self._ramps_s = sum(d for d, _, _ in _schedule(self.check)[:4])

    def feed(self, samples) -> None:
        if self.state in (DONE, FAILED):
            return
        for s in samples:
            if self._t and s.t <= self._t[-1]:
                continue
            self._t.append(s.t)
            self._steer.append(s.steer)
            self._servo.append(float(s.servo))
            self._seq.append(s.steer_seq)

    def abort(self, why: str) -> ServoCommand:
        self.state, self.failure, self.servo = FAILED, why, SERVO_OFF
        return self.servo

    def step(self, now: float) -> ServoCommand:
        ck = self.check
        if self.state in (DONE, FAILED):
            return SERVO_OFF
        if self._phase_at is None:
            self._phase_at = now
        if not self._t:
            return self.abort("no wheel readings") if now - self._phase_at > ck.rest_max_s else SERVO_OFF
        if len(self._t) != self._seen:
            self._seen, self._fed_at = len(self._t), now
        elif self.state == RUN and now - self._fed_at > ck.stall_s:
            return self.abort("the game stopped updating the wheel")
        steer = self._steer[-1]
        if self.state == WAIT:
            if not self._still(ck.rest_s):
                return self.abort("the wheel would not stand still") if now - self._phase_at > ck.rest_max_s else SERVO_OFF
            self._rest, self._start, self.state = steer, now, RUN
            self._marks = [len(self._t)]
        if abs(steer - self._rest) > ck.travel_max:
            return self.abort(f"the wheel ran {abs(steer - self._rest):.2f} lock off: the design is too strong")
        u, offset, rate = now - self._start, 0.0, 0.0
        segs = _schedule(ck)
        done = sum(d for d, _, _ in segs)
        while len(self._marks) < len(segs) and u >= sum(d for d, _, _ in segs[:len(self._marks)]):
            self._marks.append(len(self._t))
        if u >= done:
            self._marks.append(len(self._t))
            self.result = self._measure()
            self.state, self.servo = DONE, SERVO_OFF
            return self.servo
        offset, rate, accel = self._aim(u)
        self.servo = self._command(self._rest + offset, rate, accel, u)
        return self.servo

    def _aim(self, u: float) -> tuple[float, float, float]:
        ck, start = self.check, 0.0
        for dur, offset, way in _schedule(ck):
            if u < start + dur:
                if not way:
                    return offset, 0.0, 0.0
                along, rate, accel = _along(ck, way, u - start, dur)
                return offset + along, rate, accel
            start += dur
        return 0.0, 0.0, 0.0

    def _command(self, target: float, rate: float, accel: float, u: float) -> ServoCommand:
        tn, (kp, kd, ki) = self.tune, self._ramp_gains if u < self._ramps_s else self._gains
        _, plan_rate, plan_accel = self._aim(u + tn.plan_s)
        return ServoCommand(target=target, rate=rate, kp=kp, kd=kd, ki=ki, cap=round(tn.servo_cap),
                            direction=self.direction, damping_first=tn.damping_first, accel=accel,
                            ff_inertia=tn.ff_inertia, ff_damping=tn.ff_damping, ff_friction=tn.ff_friction,
                            lead_s=tn.lead_s, ff_plan=tn.plan_s > 0.0, plan_rate=plan_rate, plan_accel=plan_accel,
                            ff_breakaway=tn.ff_breakaway)

    def _still(self, seconds: float) -> bool:
        t_end = self._t[-1]
        if t_end - self._t[0] < seconds:
            return False
        tail = self._steer[max(0, bisect_right(self._t, t_end - seconds) - 1):]
        return max(tail) - min(tail) <= self.check.rest_band

    def _fresh(self, lo: int, hi: int) -> list[int]:
        return [i for i in range(max(1, lo), hi) if self._seq[i] != self._seq[i - 1]]

    def _measure(self) -> CheckResult:
        ck, m = self.check, self._marks
        steps = []
        for a, b in ((m[0], m[1]), (m[2], m[3])):
            # Per FFB update, what the motor gets: per reading the same designs scattered 73 to 113 and 150 to 250.
            steps += [abs(self._servo[i + 1] - self._servo[i]) for i in range(a, b - 1)]
        steps.sort()
        chatter = steps[len(steps) // 2] if steps else 0.0
        over, crossings, tail, reached = 0.0, 0, [], 1.0
        moves = []
        for k, size in enumerate(ck.steps):
            moves += [((m[4 + 2 * k], m[5 + 2 * k]), size, 1.0, size), ((m[5 + 2 * k], m[6 + 2 * k]), 0.0, -1.0, size)]
        for (a, b), goal, way, size in moves:
            idx = self._fresh(a, b)
            err = [(self._steer[i] - self._rest - goal) * way for i in idx]
            over = max([over] + [e / size for e in err])
            if size == min(ck.steps):
                start = (self._steer[a] - self._rest - goal) * way if b > a else -size
                reached = min(reached, 1.0 - max(err, default=start) / start if start < 0.0 else 1.0)
            sign, hit = 0, False
            for e in err:
                hit = hit or e >= -ck.ring_band
                s = 1 if e > ck.ring_band else (-1 if e < -ck.ring_band else 0)
                if hit and s and s != sign:
                    crossings += sign != 0
                    sign = s
            t_end = self._t[b - 1] if b > a else 0.0
            tail += [abs(self._steer[i] - self._rest - goal) for i in idx if self._t[i] >= t_end - ck.tail_s]
        rest = sum(tail) / len(tail) if tail else 0.0
        why = []
        if chatter > ck.chatter_max:
            why.append(f"force chatter {chatter:.0f} over {ck.chatter_max:.0f}")
        if over > ck.overshoot_max:
            why.append(f"overshoot {over:.0%} over {ck.overshoot_max:.0%}")
        if crossings >= ck.ring_crossings:
            why.append(f"rang ({crossings} swings past the target)")
        # A sticky wheel waits out a small step until the integral winds up, then lurches (the overshoot it makes
        # then is softness too): stiffer, not gentler, unless it rings.
        stiffer = reached < ck.reach_min and crossings < ck.ring_crossings
        if reached < ck.reach_min:
            why.append(f"followed the small step only {reached:.0%} of the way: too soft for its stiction")
        passed = not why
        room = passed and chatter <= ck.room * ck.chatter_max and over <= ck.room_overshoot and crossings <= 1
        return CheckResult(chatter, over, crossings, rest, passed, room, ", ".join(why) or "clean", reached, stiffer)
