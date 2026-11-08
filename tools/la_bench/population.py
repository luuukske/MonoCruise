"""A simulated population of wheels and how smoothly a servo design holds each one at its target.

Every wheel here is a model (``plant.SimWheel``); nothing in this module has seen a real wheel. The population is drawn
from fixed seeds, so a result repeats, and split into a tuning set and a held-out set (every fifth wheel).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from core.la.hold import HoldOut, HoldTuning, TargetHold
from core.lane_departure.command import ServoCommand
from tools.la_bench.plant import SIM_WHEELS, SimWheel, WheelParams
from tools.la_bench.protocol import Design

POPULATION_SEED = 20261107
POPULATION_SIZE = 200
HELD_OUT_EVERY = 5
PRESETS = ("r3", "bare", "geared", "sticky", "old")
STEP = 0.02
LEAD_S = 0.5
HOLD_S = 5.0
WINDOW_S = 2.0
FINAL_S = 0.5
FLIP_SHARE = 0.1
FORCE_FLOOR = 50.0
JITTER_MAX = 0.05
JUMP_MAX = 0.5
FLIPS_MAX = 1
FINAL_MAX = 0.0015


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_wheel(rng: random.Random) -> WheelParams:
    """One wheel: ranges from the brief, each worse-wheel feature on for about half of them."""
    breakaway = rng.uniform(300.0, 700.0)
    return WheelParams(
        inertia=_log_uniform(rng, 30.0, 800.0),
        damping=_log_uniform(rng, 100.0, 1500.0),
        stick=breakaway * rng.uniform(0.4, 0.95),
        spring=_log_uniform(rng, 300.0, 3000.0),
        delay_s=rng.uniform(0.02, 0.10),
        sign=rng.choice((1, -1)),
        quantum=0.0 if rng.random() < 0.2 else _log_uniform(rng, 0.0002, 0.005),
        reading_hz=rng.uniform(30.0, 60.0),
        breakaway=breakaway,
        breakaway_spread=rng.uniform(0.0, 0.4),
        dwell_s=rng.uniform(0.1, 1.0) if rng.random() < 0.5 else 0.0,
        backlash=rng.uniform(0.001, 0.002) if rng.random() < 0.5 else 0.0,
        stick_speed=rng.uniform(50.0, 250.0) if rng.random() < 0.5 else 0.0,
        stall_every_s=rng.uniform(1.0, 4.0) if rng.random() < 0.4 else 0.0,
        stall_s=rng.uniform(0.04, 0.12),
    )


def population(n: int = POPULATION_SIZE, seed: int = POPULATION_SEED) -> list[tuple[str, WheelParams]]:
    rng = random.Random(seed)
    return [(f"w{i:03d}", sample_wheel(rng)) for i in range(n)]


def split(wheels: list[tuple[str, WheelParams]]) -> tuple[list, list]:
    """``(tuning, held_out)``: every HELD_OUT_EVERY-th wheel is held out, 20 % of them."""
    held = [w for i, w in enumerate(wheels) if i % HELD_OUT_EVERY == HELD_OUT_EVERY - 1]
    return [w for i, w in enumerate(wheels) if i % HELD_OUT_EVERY != HELD_OUT_EVERY - 1], held


def presets() -> list[tuple[str, WheelParams]]:
    return [(name, SIM_WHEELS[name]) for name in PRESETS]


def force_scale(p: WheelParams) -> float:
    """``F`` of the thresholds: the breakaway, or the sliding friction for a wheel without one."""
    return max(FORCE_FLOOR, p.breakaway if p.breakaway > p.stick else p.stick)


FREE = HoldOut()


def command(design: Design, target: float, direction: int, hold: HoldOut = FREE) -> ServoCommand:
    """``design``'s request on a still ``target``, its gains and friction pushes scaled by the hold."""
    kp, kd, ki = design.gains()
    factor = hold.gain
    return ServoCommand(target=target, rate=0.0, kp=kp * factor, kd=kd * factor, ki=ki * hold.integral, cap=design.cap,
                        direction=direction, damping_first=design.damping_first, ff_inertia=design.ff_inertia,
                        ff_damping=design.ff_damping, ff_friction=design.ff_friction * factor, lead_s=design.lead_s,
                        ff_plan=design.pushes_ahead and design.plan_s > 0.0, ff_breakaway=design.ff_breakaway * factor)


@dataclass(frozen=True)
class Hold:
    jitter: float
    jump: float
    flips: int
    final_mlock: float
    overshoot: float

    @property
    def ok(self) -> bool:
        return (self.jitter <= JITTER_MAX and self.jump <= JUMP_MAX and self.flips <= FLIPS_MAX
                and self.final_mlock <= FINAL_MAX * 1000.0)


def _flips(values: list[float], band: float) -> int:
    side, count = 0, 0
    for v in values:
        s = 1 if v >= band else (-1 if v <= -band else 0)
        if s and side and s != side:
            count += 1
        side = s or side
    return count


def _window(times, outs, angles, t_end, goal, start, p: WheelParams) -> Hold:
    f = force_scale(p)
    w = [(o, a) for t, o, a in zip(times, outs, angles) if t_end - WINDOW_S <= t < t_end]
    seg = [a for t, a in zip(times, angles) if t_end - HOLD_S <= t < t_end]
    tail = [a for t, a in zip(times, angles) if t_end - FINAL_S <= t < t_end]
    d = [b[0] - a[0] for a, b in zip(w, w[1:])]
    jitter = math.sqrt(sum(x * x for x in d) / len(d)) / f if d else 0.0
    jump = max((abs(x) for x in d), default=0.0) / f
    err = abs(sum(tail) / len(tail) - goal) if tail else 0.0
    size = goal - start
    peak = max((a - goal) / size for a in seg) if seg and size else 0.0
    return Hold(jitter, jump, _flips([o for o, _ in w], FLIP_SHARE * f), max(0.0, err - unresolved(goal, p.quantum))
                * 1000.0, max(0.0, peak))


def unresolved(goal: float, quantum: float) -> float:
    """How far off ``goal`` a wheel parked in the reading step nearest it may stand: half a step plus that step's
    distance from the goal. A servo that only sees the reading cannot do better."""
    if quantum <= 0.0:
        return 0.0
    return 0.5 * quantum + abs(round(goal / quantum) * quantum - goal)


def hold_test(p: WheelParams, design: Design, frame_hz: float = 120.0, amp: float = STEP,
              hold: HoldTuning | None = None, quantum: float | None = None, send_hz: float = 20.0) -> Hold:
    """A step up from rest and the swing back through zero, each held HOLD_S; the worst of their last WINDOW_S.

    ``hold`` runs MonoCruise's target hold at ``send_hz`` on the reading, told the reading step ``quantum`` (the
    wheel's own when None).
    """
    wheel = SimWheel(p, frame_hz)
    parker = TargetHold(hold) if hold is not None else None
    q = p.quantum if quantum is None else quantum
    plan = ((0.0, 0.0), (LEAD_S, amp), (LEAD_S + HOLD_S, -amp))
    end = LEAD_S + 2.0 * HOLD_S
    times, outs, angles = [], [], []
    k = 0
    target = 0.0
    servo = command(design, 0.0, p.sign)
    sent_at = -1.0
    while wheel.time < end:
        while k < len(plan) and plan[k][0] <= wheel.time:
            target = plan[k][1]
            k += 1
        if wheel.time - sent_at >= 1.0 / send_hz - 1e-9:
            out = parker.update(wheel.time - sent_at if sent_at >= 0.0 else 0.0, target, 0.0, wheel.reading, q) \
                if parker is not None else FREE
            servo = command(design, target, p.sign, out)
            sent_at = wheel.time
        ticks = wheel.n % wheel.substeps == 0
        wheel.substep(servo)
        if ticks:
            times.append(wheel.time)
            outs.append(wheel.plugin.value)
            angles.append(wheel.steer)
    up = _window(times, outs, angles, LEAD_S + HOLD_S, amp, 0.0, p)
    swing = _window(times, outs, angles, end, -amp, amp, p)
    return Hold(max(up.jitter, swing.jitter), max(up.jump, swing.jump), max(up.flips, swing.flips),
                max(up.final_mlock, swing.final_mlock), max(up.overshoot, swing.overshoot))


def table(results: list[tuple[str, Hold]]) -> dict:
    """Pass share and the medians and worst of each metric."""
    def col(name: str) -> list[float]:
        return sorted(getattr(h, name) for _, h in results)

    out = {"n": len(results), "pass": sum(h.ok for _, h in results) / max(1, len(results))}
    for name in ("jitter", "jump", "flips", "final_mlock", "overshoot"):
        c = col(name)
        out[name] = {"median": c[len(c) // 2], "p90": c[int(0.9 * (len(c) - 1))], "worst": c[-1]} if c else {}
    return out


def with_hands(p: WheelParams, share: float = 0.5) -> WheelParams:
    """Resting hands as more inertia and damping (a share of the rim's)."""
    return replace(p, inertia=p.inertia * (1.0 + share), damping=p.damping * (1.0 + share))
