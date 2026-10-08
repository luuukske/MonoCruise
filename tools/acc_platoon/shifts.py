"""Upshift drive cut: the plant's gearbox against the mapper debug log. See tools/acc_platoon/README.md.

Both sides go through the same measurement: speed logged at the mapper's ~10 Hz,
interpolated, differentiated over 0.2 s, and each upshift read as the time its
acceleration spends below half of what it was before and after.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .plant import AMT, Gearbox, TruckPlant, TruckSpec

LOG_NAME: str = "accel_to_pedals_debug.csv"
LOG_PERIOD_S: float = 0.105
GRID_S: float = 0.05
GRID_FROM_S: float = -1.5
GRID_TO_S: float = 2.5
DIFF_HALF_S: float = 0.1
PRE_UNTIL_S: float = -0.9
POST_FROM_S: float = 1.8
DIP_FROM_S: float = -1.0
DIP_TO_S: float = 2.0
MIN_PRE_MS2: float = 0.4
MIN_POST_MS2: float = 0.3
SAMPLES_BEFORE: int = 20
SAMPLES_AFTER: int = 35
MIN_LOG_DT_S: float = 0.05
MAX_LOG_DT_S: float = 0.2


@dataclass
class Upshift:
    """One measured upshift: seconds of drive below half, and accel over the pre-shift level on the grid."""

    below_half_s: float
    profile: list[float]
    from_gear: int
    kmh: float
    box_gears: int = 0


def grid() -> list[float]:
    n = int(round((GRID_TO_S - GRID_FROM_S) / GRID_S))
    return [GRID_FROM_S + k * GRID_S for k in range(n)]


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + (ys[hi] - ys[lo]) * f


def _box_gears(t: list[float], gear: list[int]) -> list[int]:
    """Highest gear of each drive (split at log gaps over a minute): 6 is the power-shift box."""
    out = [0] * len(t)
    start = 0
    for k in range(1, len(t) + 1):
        if k == len(t) or t[k] - t[k - 1] > 60.0:
            top = max(gear[start:k])
            out[start:k] = [top] * (k - start)
            start = k
    return out


def measure(t: list[float], v: list[float], gear: list[int],
            braking: list[bool] | None = None) -> list[Upshift]:
    """Every clean upshift in one log: one shift in the window, steady sampling, no brake."""
    g_s = grid()
    box = _box_gears(t, gear)
    out: list[Upshift] = []
    for k in range(SAMPLES_BEFORE, len(t) - SAMPLES_AFTER):
        if gear[k] <= gear[k - 1] or gear[k - 1] <= 0:
            continue
        lo, hi = k - SAMPLES_BEFORE, k + SAMPLES_AFTER
        tt = [t[j] - t[k] for j in range(lo, hi)]
        if any(not MIN_LOG_DT_S < b - a <= MAX_LOG_DT_S for a, b in zip(tt, tt[1:])):
            continue
        if sum(1 for j in range(lo + 1, hi) if gear[j] != gear[j - 1]) != 1:
            continue
        if braking is not None and any(braking[j] for j in range(k - 15, k + 15)):
            continue
        vv = v[lo:hi]
        acc = [(_interp(tt, vv, g + DIFF_HALF_S) - _interp(tt, vv, g - DIFF_HALF_S)) / (2 * DIFF_HALF_S)
               for g in g_s]
        pre = [a for g, a in zip(g_s, acc) if g <= PRE_UNTIL_S]
        post = [a for g, a in zip(g_s, acc) if g >= POST_FROM_S]
        a_pre, a_post = sum(pre) / len(pre), sum(post) / len(post)
        if a_pre < MIN_PRE_MS2 or a_post < MIN_POST_MS2:
            continue
        half = 0.5 * min(a_pre, a_post)
        idx = [i for i, g in enumerate(g_s) if DIP_FROM_S < g < DIP_TO_S and acc[i] < half]
        below = 0.0
        if idx:
            i_min = min(idx, key=lambda i: acc[i])
            i0 = i1 = i_min
            while i0 - 1 in idx:
                i0 -= 1
            while i1 + 1 in idx:
                i1 += 1
            below = (i1 - i0 + 1) * GRID_S
        out.append(Upshift(below, [a / a_pre for a in acc], gear[k - 1], 3.6 * v[k], box[k]))
    return out


def read_log(path: Path) -> tuple[list[float], list[float], list[int], list[bool]]:
    """Time, speed, gear and brake flag from the mapper's debug CSV."""
    t: list[float] = []
    v: list[float] = []
    g: list[int] = []
    b: list[bool] = []
    t0 = None
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            try:
                stamp = datetime.fromisoformat(row["utc"]).timestamp()
                speed, gear = float(row["speed_ms"]), int(float(row["gear"]))
                brake = float(row["brake_cmd"] or 0.0) > 0.01
            except (KeyError, TypeError, ValueError):
                continue
            t0 = stamp if t0 is None else t0
            t.append(stamp - t0)
            v.append(speed)
            g.append(gear)
            b.append(brake)
    return t, v, g, b


def model_upshifts(box: Gearbox = AMT, launches: int = 40, seed: int = 1) -> list[Upshift]:
    """The plant pulling away under a steady command, logged the way the mapper logs."""
    rng = random.Random(seed)
    spec = TruckSpec("probe", mass_t=26.0, power_kw=400.0, brake_ms2=11.0, tau_brake_s=0.25,
                     gearbox=box)
    dt = 1.0 / 60.0
    out: list[Upshift] = []
    for _ in range(launches):
        plant = TruckPlant(spec, 0.0, 0.0, dt, random.Random(rng.getrandbits(64)))
        demand = rng.uniform(0.6, 1.6)
        t, next_log = 0.0, rng.uniform(0.0, LOG_PERIOD_S)
        ts: list[float] = []
        vs: list[float] = []
        gs: list[int] = []
        while plant.v < 85.0 / 3.6 and t < 90.0:
            plant.step(demand, dt)
            t += dt
            if t >= next_log:
                ts.append(t)
                vs.append(plant.v)
                gs.append(plant.gear + 1)
                next_log += LOG_PERIOD_S + rng.uniform(-0.004, 0.008)
        out.extend(measure(ts, vs, gs))
    return out


def amt(shifts: list[Upshift]) -> list[Upshift]:
    """The 12- to 14-speed boxes; the log's 6-speed drives are a torque-converter automatic."""
    return [s for s in shifts if s.box_gears >= 12]


def quantiles(shifts: list[Upshift], ps: tuple[float, ...] = (10, 50, 90)) -> list[float]:
    xs = sorted(s.below_half_s for s in shifts)
    return [xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))] for p in ps]


def median_profile(shifts: list[Upshift]) -> list[float]:
    cols = list(zip(*(s.profile for s in shifts)))
    return [sorted(c)[len(c) // 2] for c in cols]
