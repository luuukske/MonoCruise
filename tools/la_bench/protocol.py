"""Bench stages and designs: what the target does, and the servo gains each run holds.

A stage is a target path relative to the angle the wheel rests at: ``step`` jumps (up, then a swing through
zero, then home), ``ramp`` moves at a fixed rate (its corners rounded over ``corner`` seconds when set), ``follow`` is a
sine, ``out`` goes out to one side and back, then to the other and back (half lock, at a rate or, at 0, at once). Moving stages are sent at the rate MonoCruise sends targets (20 Hz) with the target's rate and acceleration, so
the plugin extrapolates them as it does in the game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.la.actuator import WheelActuator, WheelTuning

LEAD_S = 0.5
STEP_HOLD_S = 1.5
HOLD_S = 1.0


@dataclass(frozen=True)
class Design:
    """The servo design under test: what ``WheelActuator.gains`` turns into kp, kd and ki.

    ``spring`` is the game's centring the design is told about (units per lock); standing it is about 0.
    ``ff_*`` and ``lead_s`` are the plugin's push ahead and lead (layout 5), 0 is none; ``ff_breakaway`` the friction
    push while the wheel stands (status bit 2048).
    """

    bandwidth_hz: float = 2.0
    damping: float = 0.7
    inertia: float = 650.0
    strength: float = 1.0
    integral_share: float = 0.3
    spring: float = 0.0
    cap: int = 3000
    damping_first: bool = False
    ff_inertia: float = 0.0
    ff_damping: float = 0.0
    ff_friction: float = 0.0
    lead_s: float = 0.0
    plan_s: float = 0.0
    ff_breakaway: float = 0.0
    settle: bool = False

    @property
    def label(self) -> str:
        base = f"bw{self.bandwidth_hz:g} z{self.damping:g} J{self.inertia:g}"
        base += " dfirst" if self.damping_first else ""
        base += " ahead" if self.ff_inertia or self.ff_damping or self.ff_friction else ""
        base += f" break{self.ff_breakaway:.0f}" if self.ff_breakaway > self.ff_friction > 0.0 else ""
        base += f" plan{self.plan_s * 1000:.0f}ms" if self.plan_s else ""
        base += " settle" if self.settle else ""
        return base + (f" lead{self.lead_s * 1000:.0f}ms" if self.lead_s else "")

    @property
    def pushes_ahead(self) -> bool:
        return bool(self.ff_inertia or self.ff_damping or self.ff_friction or self.lead_s)

    def gains(self) -> tuple[float, float, float]:
        tune = WheelTuning(
            inertia=self.inertia, bandwidth_hz=self.bandwidth_hz, damping=self.damping, strength=self.strength,
            integral_share=self.integral_share, servo_cap=float(self.cap),
        )
        return WheelActuator(tune).gains(1.0, self.spring)


def move_s(distance: float, rate: float, corner: float) -> float:
    """How long a move of ``distance`` takes at ``rate``, plus one corner (a triangle when it never reaches the rate)."""
    if corner <= 0.0:
        return distance / rate
    if distance >= rate * corner:
        return distance / rate + corner
    return 2.0 * (distance * corner / rate) ** 0.5


def rounded_move(u: float, dur: float, a0: float, a1: float, rate: float, corner: float) -> tuple[float, float, float]:
    """``u`` seconds into a move from ``a0`` to ``a1`` lasting ``dur``: constant acceleration up to ``rate`` over
    ``corner``, cruise, the same down. Where it is, how fast it moves and how fast that changes.
    """
    sign = 1.0 if a1 >= a0 else -1.0
    dist = abs(a1 - a0)
    if dist >= rate * corner:
        top, ramp_s = rate, corner
    else:
        ramp_s = 0.5 * dur
        top = dist / ramp_s if ramp_s > 0.0 else 0.0
    acc = top / ramp_s if ramp_s > 0.0 else 0.0
    if u >= dur:
        return a1, 0.0, 0.0
    u = max(0.0, u)
    if u < ramp_s:
        x, v, a = 0.5 * acc * u * u, acc * u, acc
    elif u < dur - ramp_s:
        x, v, a = 0.5 * top * ramp_s + top * (u - ramp_s), top, 0.0
    else:
        left = dur - u
        x, v, a = dist - 0.5 * acc * left * left, acc * left, -acc
    return a0 + sign * x, sign * v, sign * a


def accel_changes(dur: float, dist: float, rate: float, corner: float) -> tuple[float, float]:
    """When a rounded move stops speeding up and starts slowing down, from its start."""
    ramp_s = corner if dist >= rate * corner else 0.5 * dur
    return ramp_s, dur - ramp_s


@dataclass(frozen=True)
class Stage:
    """``param`` is the hold per level (step), the rate in lock per second (ramp) or the frequency in Hz (follow).

    ``corner`` is how long a ramp takes to reach its rate and to stop (constant acceleration); 0 is a sharp corner.
    """

    kind: str
    amp: float
    param: float
    corner: float = 0.0

    @property
    def name(self) -> str:
        if self.kind == "step":
            return f"step {self.amp:g}"
        if self.kind == "ramp":
            return f"ramp {self.amp:g} at {self.param:g}/s" + (f", {self.corner:g} s corners" if self.corner else "")
        if self.kind == "out":
            if not self.param:
                return f"out {self.amp:g} at once"
            return f"out {self.amp:g} at {self.param:g}/s" + (f", {self.corner:g} s corners" if self.corner else "")
        return f"follow {self.amp:g} at {self.param:g} Hz"

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "amp": self.amp, "param": self.param}
        return dict(out, corner=self.corner) if self.corner else out

    @classmethod
    def from_dict(cls, raw: dict) -> Stage:
        return cls(str(raw["kind"]), float(raw["amp"]), float(raw["param"]), float(raw.get("corner", 0.0)))

    def _move_s(self, distance: float) -> float:
        return move_s(distance, self.param, self.corner) if self.param > 0.0 else 0.0

    @property
    def rounded(self) -> bool:
        return self.kind in ("ramp", "out") and self.corner > 0.0 and self.param > 0.0

    def _segments(self) -> list[tuple[float, float, float, float]]:
        a, p = self.amp, self.param
        if self.kind == "step":
            plan = [(LEAD_S, 0.0), (0.0, a), (p, a), (0.0, -a), (p, -a), (0.0, 0.0), (p, 0.0)]
        elif self.kind == "out":
            m = self._move_s(a)
            plan = [(LEAD_S, 0.0), (m, a), (HOLD_S, a), (m, 0.0), (HOLD_S, 0.0), (m, -a), (HOLD_S, -a), (m, 0.0),
                    (HOLD_S, 0.0)]
        else:
            plan = [(LEAD_S, 0.0), (self._move_s(a), a), (HOLD_S, a), (self._move_s(2.0 * a), -a), (HOLD_S, -a),
                    (self._move_s(a), 0.0), (LEAD_S, 0.0)]
        segs = []
        t, level = 0.0, 0.0
        for dur, to in plan:
            segs.append((t, t + dur, level, to))
            t += dur
            level = to
        return segs

    @property
    def duration(self) -> float:
        if self.kind == "follow":
            return max(5.0, 3.0 / self.param)
        return self._segments()[-1][1]

    def segments(self) -> list[tuple[float, float, float, float]]:
        """``(start, end, from, to)`` of each piece of the stage."""
        return self._segments()

    def aim(self, t: float) -> tuple[float, float]:
        """The target's offset from the resting angle and its rate at stage time ``t``."""
        return self.aim3(t)[:2]

    def aim3(self, t: float) -> tuple[float, float, float]:
        """The target's offset, rate and acceleration at stage time ``t``."""
        if self.kind == "follow":
            w = 2.0 * math.pi * self.param
            return self.amp * math.sin(w * t), self.amp * w * math.cos(w * t), -self.amp * w * w * math.sin(w * t)
        level = 0.0
        for t0, t1, a0, a1 in self._segments():
            if t < t0:
                break
            if t < t1:
                if self.rounded:
                    return rounded_move(t - t0, t1 - t0, a0, a1, self.param, self.corner)
                return a0 + (a1 - a0) * (t - t0) / (t1 - t0), (a1 - a0) / (t1 - t0), 0.0
            level = a1
        return level, 0.0, 0.0

    def events(self, hz: float, early: float = 0.0) -> list[float]:
        """When a new request goes out: every corner, and every 1/hz inside a moving stage. A rounded ramp also sends
        where its acceleration changes, as a feature that plans its target would: the plugin brakes on time.
        """
        times = set()
        if self.kind != "follow":
            for t0, t1, a0, a1 in self._segments():
                times.update((t0, t1))
                if self.rounded and a0 != a1:
                    up, down = accel_changes(t1 - t0, abs(a1 - a0), self.param, self.corner)
                    times.update((t0 + up, t0 + down))
                    if early > 0.0:
                        times.update(max(0.0, x - early) for x in (t0, t0 + up, t0 + down, t1))
        if self.kind != "step" and not (self.kind == "out" and not self.param):
            times.update(k / hz for k in range(int(self.duration * hz) + 1))
        return sorted(t for t in (round(x, 6) for x in times) if t < self.duration)


def step(amp: float, hold: float = STEP_HOLD_S) -> Stage:
    return Stage("step", amp, hold)


def ramp(amp: float, rate: float, corner: float = 0.0) -> Stage:
    return Stage("ramp", amp, rate, corner)


def out(amp: float, rate: float = 0.0, corner: float = 0.0) -> Stage:
    return Stage("out", amp, rate, corner)


def follow(amp: float, hz: float) -> Stage:
    return Stage("follow", amp, hz)


SUITES = {
    "step": (step(0.01), step(0.02)),
    "full": (step(0.01), step(0.02), ramp(0.06, 0.05), ramp(0.08, 0.2), follow(0.03, 0.4), follow(0.03, 0.7)),
}
