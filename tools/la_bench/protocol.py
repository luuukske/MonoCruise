"""Bench stages and designs: what the target does, and the servo gains each run holds.

A stage is a target path relative to the angle the wheel rests at: ``step`` jumps (up, then a swing through
zero, then home), ``ramp`` moves at a fixed rate, ``follow`` is a sine. Moving stages are sent at the rate
MonoCruise sends targets (20 Hz) with the target's rate, so the plugin extrapolates them as it does in the game.
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
    """

    bandwidth_hz: float = 2.0
    damping: float = 0.7
    inertia: float = 650.0
    strength: float = 1.0
    integral_share: float = 0.3
    spring: float = 0.0
    cap: int = 3000
    damping_first: bool = False

    @property
    def label(self) -> str:
        base = f"bw{self.bandwidth_hz:g} z{self.damping:g} J{self.inertia:g}"
        return base + " dfirst" if self.damping_first else base

    def gains(self) -> tuple[float, float, float]:
        tune = WheelTuning(
            inertia=self.inertia, bandwidth_hz=self.bandwidth_hz, damping=self.damping, strength=self.strength,
            integral_share=self.integral_share, servo_cap=float(self.cap),
        )
        return WheelActuator(tune).gains(1.0, self.spring)


@dataclass(frozen=True)
class Stage:
    """``param`` is the hold per level (step), the rate in lock per second (ramp) or the frequency in Hz (follow)."""

    kind: str
    amp: float
    param: float

    @property
    def name(self) -> str:
        if self.kind == "step":
            return f"step {self.amp:g}"
        if self.kind == "ramp":
            return f"ramp {self.amp:g} at {self.param:g}/s"
        return f"follow {self.amp:g} at {self.param:g} Hz"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "amp": self.amp, "param": self.param}

    @classmethod
    def from_dict(cls, raw: dict) -> Stage:
        return cls(str(raw["kind"]), float(raw["amp"]), float(raw["param"]))

    def _segments(self) -> list[tuple[float, float, float, float]]:
        a, p = self.amp, self.param
        if self.kind == "step":
            plan = [(LEAD_S, 0.0), (0.0, a), (p, a), (0.0, -a), (p, -a), (0.0, 0.0), (p, 0.0)]
        else:
            plan = [(LEAD_S, 0.0), (a / p, a), (HOLD_S, a), (2.0 * a / p, -a), (HOLD_S, -a), (a / p, 0.0), (LEAD_S, 0.0)]
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

    def aim(self, t: float) -> tuple[float, float]:
        """The target's offset from the resting angle and its rate at stage time ``t``."""
        if self.kind == "follow":
            w = 2.0 * math.pi * self.param
            return self.amp * math.sin(w * t), self.amp * w * math.cos(w * t)
        level = 0.0
        for t0, t1, a0, a1 in self._segments():
            if t < t0:
                break
            if t < t1:
                return a0 + (a1 - a0) * (t - t0) / (t1 - t0), (a1 - a0) / (t1 - t0)
            level = a1
        return level, 0.0

    def events(self, hz: float) -> list[float]:
        """When a new request goes out: every corner, and every 1/hz inside a moving stage."""
        times = set()
        if self.kind != "follow":
            for t0, t1, _, _ in self._segments():
                times.update((t0, t1))
        if self.kind != "step":
            times.update(k / hz for k in range(int(self.duration * hz) + 1))
        return sorted(t for t in (round(x, 6) for x in times) if t < self.duration)


def step(amp: float, hold: float = STEP_HOLD_S) -> Stage:
    return Stage("step", amp, hold)


def ramp(amp: float, rate: float) -> Stage:
    return Stage("ramp", amp, rate)


def follow(amp: float, hz: float) -> Stage:
    return Stage("follow", amp, hz)


SUITES = {
    "step": (step(0.01), step(0.02)),
    "full": (step(0.01), step(0.02), ramp(0.06, 0.05), ramp(0.08, 0.2), follow(0.03, 0.4), follow(0.03, 0.7)),
}
