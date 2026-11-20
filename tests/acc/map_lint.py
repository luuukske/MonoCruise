"""Smoothness of the ACC law's steady map over cruising states: steps, corners and loosening.

The map is the immediate-lead law up to the brake follow limit (`approach_profile.banded_law`
then `brake_follow_limit`), stateless, so it can be sampled finely. Each line runs along one
input with the other two held, toward more danger: lead braking harder, closing faster, or the
gap shrinking. Run this file to print the report. See `core/acc/ACC_ARCHITECTURE.md` §8.13."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.cruise_control_thread import approach_profile
from core.cruise_control_thread.acc_controller import (
    T_HEADWAY_BY_LEVEL_S, AdaptiveCruiseController, _LeadSnapshot,
)

LEVELS = (1, 2, 3, 4)
SPEEDS_KMH = (50.0, 80.0)
GAP_MULTS = (0.7, 1.0, 1.5, 2.2)
HELD_CLOSINGS = (-1.0, -0.5, 0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0)
HELD_DECELS = (-0.5, 0.0, 0.3, 0.5, 1.0, 1.5)
H_DECEL, H_CLOSING, H_GAP = 0.01, 0.01, 0.05
# Cruising only: braking harder than this belongs to the stop and overlay regime.
FLOOR_MS2 = -3.0
AXES = ("decel", "closing", "gap")
METRICS = ("step", "corner", "loosen")


@dataclass
class Worst:
    value: float = 0.0
    where: str = ""

    def take(self, value: float, where: str) -> None:
        if value > self.value:
            self.value, self.where = value, where


@dataclass
class Report:
    """Per axis: `step` a jump over its neighbours' change (m/s^2), `corner` the slope change
    across one sample (m/s^2 per unit of the axis), `loosen` braking given back toward danger."""
    worst: dict[tuple[str, str], Worst] = field(
        default_factory=lambda: {(a, m): Worst() for a in AXES for m in METRICS})

    def lines(self) -> list[str]:
        return [f"{a:7s} {m:6s} {w.value:7.3f}  at {w.where}" for (a, m), w in self.worst.items()]


def steady_law(cfg, v_ego: float, dist_m: float, closing: float, decel: float, t_h: float) -> float:
    """The settled command for one constant lead, from the law to the follow limit."""
    lead = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=max(0.0, v_ego - closing),
                         a_lead_ms2=-decel, score=6.0, conf=1.0, a_lead_ff_ms2=-decel)
    law = approach_profile.banded_law(cfg, lead, v_ego, t_h)
    return approach_profile.brake_follow_limit(cfg, law, lead, v_ego, t_h, gap_excess_ms=0.0,
                                               monotone=True)


def _scan(values: list[float], h: float, axis: str, where: str, rep: Report) -> None:
    """`values` run toward more danger, one sample every `h`."""
    w = rep.worst
    least = math.inf
    for i, v in enumerate(values):
        if v >= FLOOR_MS2 and math.isfinite(least):
            w[axis, "loosen"].take(v - least, f"{where}, sample {i}")
        least = min(least, v)
    d = [b - a for a, b in zip(values, values[1:])]
    for i in range(1, len(d) - 1):
        if min(values[i - 1:i + 3]) < FLOOR_MS2:
            continue
        w[axis, "step"].take(abs(d[i]) - 0.5 * (abs(d[i - 1]) + abs(d[i + 1])), f"{where}, sample {i}")
        w[axis, "corner"].take(abs(d[i] - d[i - 1]) / h, f"{where}, sample {i}")


def measure(cfg=None) -> Report:
    cfg = cfg or AdaptiveCruiseController().config
    rep = Report()
    decels = [-1.0 + H_DECEL * i for i in range(int(round(3.0 / H_DECEL)) + 1)]
    closings = [-2.0 + H_CLOSING * i for i in range(int(round(5.0 / H_CLOSING)) + 1)]
    for level in LEVELS:
        t_h = T_HEADWAY_BY_LEVEL_S[level]
        for kmh in SPEEDS_KMH:
            v = kmh / 3.6
            s_want = cfg.s0_m + v * t_h
            for mult in GAP_MULTS:
                s = s_want * mult
                tag = f"L{level} {kmh:.0f} km/h gap x{mult}"
                for c in HELD_CLOSINGS:
                    _scan([steady_law(cfg, v, s, c, d, t_h) for d in decels], H_DECEL, "decel",
                          f"{tag} closing {c:+.1f}, decel from -1.0", rep)
                for d in HELD_DECELS:
                    _scan([steady_law(cfg, v, s, c, d, t_h) for c in closings], H_CLOSING, "closing",
                          f"{tag} decel {d:+.1f}, closing from -2.0", rep)
            gaps = [s_want * 2.2 - H_GAP * i for i in range(int(s_want * 1.6 / H_GAP))]
            for c in (0.0, 1.0, 2.0):
                for d in (0.0, 0.5, 1.0):
                    _scan([steady_law(cfg, v, g, c, d, t_h) for g in gaps], H_GAP, "gap",
                          f"L{level} {kmh:.0f} km/h closing {c:+.1f} decel {d:+.1f}, "
                          f"gap from {gaps[0]:.1f} m down", rep)
    return rep


if __name__ == "__main__":
    for line in measure().lines():
        print(line)
