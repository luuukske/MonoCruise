"""A model wheel under the plugin's servo, and a stdlib fit of it to bench recordings.

The servo is ``tests.la.servo_sim.Plugin``, the mirror of plugins/ffb/ffb_signal.h, so a fit and a design study
run the servo the game runs. The rim is inertia, viscous damping, dry friction, a spring and a delay, in the
plugin's units: force out of 10000, angle in lock, ``inertia`` as in ``la_wheel_inertia``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

from core.lane_departure.command import SERVO_OFF, PluginStatus, ServoCommand
from tests.la.servo_sim import Plugin
from tools.la_bench.metrics import interp

STILL = 1e-4
INERTIA_RANGE = (30.0, 20000.0)
DELAY_MIN_S = 0.016
DELAY_MAX_S = 0.1
QUASI_STATIC_RATE = 0.02
SATURATED_SHARE = 0.9
DELAY_SCAN_S = (0.016, 0.024, 0.032, 0.04, 0.05)


@dataclass(frozen=True)
class WheelParams:
    """``sign`` is which way the plugin's output turns the wheel counterclockwise (the command block's direction).

    ``cog`` is the peak of a force that repeats every ``cog_period`` of lock: the teeth of a geared wheel.
    ``quantum`` is the size of the steps the angle reads in (a coarse encoder); 0 reads it continuously.
    ``reading_hz`` is how often the game reads the wheel angle (its frame rate, 60 Hz on the R3 on 2026-11-06); 0 reads
    it at every FFB tick.
    """

    inertia: float = 650.0
    damping: float = 150.0
    stick: float = 100.0
    spring: float = 0.0
    delay_s: float = 0.016
    sign: int = 1
    cog: float = 0.0
    cog_period: float = 0.01
    quantum: float = 0.0
    reading_hz: float = 0.0


class SimWheel:
    """The rim and the plugin on one time grid: the plugin runs every ``substeps`` steps, its force reaches the rim
    ``delay_s`` after it read the angle.
    """

    def __init__(self, params: WheelParams, frame_hz: float = 120.0, substeps: int = 4, steer0: float = 0.0) -> None:
        self.params = params
        self.frame_hz = frame_hz
        self.substeps = substeps
        self.h = 1.0 / (frame_hz * substeps)
        self.plugin = Plugin()
        self.steer = steer0
        self.rate = 0.0
        self.n = 0
        self.held = 0.0
        self.applied = 0.0
        self.frames = 0
        self.readings = 0
        self.held_reading: float | None = None
        self.read_at = 0.0
        self.pending: deque = deque()

    @property
    def time(self) -> float:
        return self.n * self.h

    @property
    def reading(self) -> float:
        return self.held_reading if self.held_reading is not None else self._read()

    def _read(self) -> float:
        q = self.params.quantum
        return round(self.steer / q) * q if q else self.steer

    def _advance(self, ccw: float) -> None:
        p, h = self.params, self.h
        force = ccw - p.spring * self.steer - p.damping * self.rate
        if p.cog:
            force += p.cog * math.sin(2.0 * math.pi * self.steer / p.cog_period)
        if p.stick > 0.0:
            if abs(self.rate) < STILL and abs(force) <= p.stick:
                self.rate = 0.0
                return
            force -= math.copysign(p.stick, self.rate if abs(self.rate) >= STILL else force)
        new = self.rate + force / p.inertia * h
        self.rate = 0.0 if p.stick > 0.0 and self.rate * new < 0.0 else new
        self.steer += self.rate * h

    def substep(self, servo: ServoCommand = SERVO_OFF, torque: float = 0.0) -> None:
        if self.n % self.substeps == 0:
            hz = self.params.reading_hz
            fresh = not hz or self.held_reading is None or self.time - self.read_at >= 1.0 / hz - 1e-9
            if fresh:
                self.held_reading, self.read_at = self._read(), self.time
                self.readings += 1
            self.applied = self.plugin.step(servo, torque, self.reading, 1.0 / self.frame_hz, 0.0,
                                            reading_age=self.time - self.read_at, fresh=fresh)
            self.pending.append((self.time + self.params.delay_s, self.applied * self.params.sign))
            self.frames += 1
        while self.pending and self.pending[0][0] <= self.time:
            self.held = self.pending.popleft()[1]
        self._advance(self.held)
        self.n += 1


class SimClock:
    """A clock that moves only when the bench sleeps, so a simulated capture runs as fast as the model."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


class SimChannel:
    """Stands in for ``RumbleChannel`` on a model wheel; ``clock`` is the bench's clock, in seconds."""

    def __init__(self, params: WheelParams, clock, frame_hz: float = 120.0, steer0: float = 0.0) -> None:
        self.wheel = SimWheel(params, frame_hz, steer0=steer0)
        self.clock = clock
        self.t0 = clock()
        self.servo = SERVO_OFF
        self.torque = 0.0

    def publish(self, active=False, side=0, units=0, now=None, torque=0, servo: ServoCommand = SERVO_OFF) -> PluginStatus:
        w = self.wheel
        until = self.clock() - self.t0
        while w.time + w.h <= until:
            w.substep(self.servo, self.torque)
        self.servo, self.torque = servo, torque
        p = w.params
        return PluginStatus(
            present=True, hooked=True, version=5, steer_ok=True, damping_first=True, push_ahead=True, ff_plan=True,
            servoing=bool(servo.direction), ff_applied=round(w.plugin.ff * p.sign),
            steer=w.reading,
            steer_rate=w.plugin.rate, steer_frames=w.readings, frames=w.frames, frame_hz=w.frame_hz,
            game_force=round(-p.spring * w.steer * p.sign), applied=round(w.applied),
            servo_applied=round(w.plugin.value), servo_scale=w.plugin.scale, servo_integ=w.plugin.integ,
        )

    def stop(self) -> None:
        self.servo, self.torque = SERVO_OFF, 0.0

    def close(self) -> None:
        return None


def replay(run: dict, params: WheelParams, frame_hz: float, substeps: int = 4) -> tuple[list[float], list[float]]:
    """The wheel angle over time on a model wheel given the requests ``run`` sent."""
    sent, g = run["sent"], run["gains"]
    wheel = SimWheel(params, frame_hz, substeps, steer0=run["samples"][0][1])
    servo = SERVO_OFF
    i = 0
    end = run["samples"][-1][0]
    ts, ys = [], []
    while wheel.time < end:
        while i < len(sent) and sent[i][0] <= wheel.time:
            servo = ServoCommand(target=sent[i][1], rate=sent[i][2], kp=g["kp"], kd=g["kd"], ki=g["ki"], cap=g["cap"],
                                 direction=g["direction"], damping_first=g.get("damping_first", False),
                                 accel=sent[i][3] if len(sent[i]) > 3 else 0.0, ff_inertia=g.get("ff_inertia", 0.0),
                                 ff_damping=g.get("ff_damping", 0.0), ff_friction=g.get("ff_friction", 0.0),
                                 lead_s=g.get("lead_s", 0.0))
            i += 1
        ts.append(wheel.time)
        ys.append(wheel.steer)
        wheel.substep(servo)
    return ts, ys


def estimate_spring(runs: list[dict], sign: int) -> float:
    """The game's centring standing: the slope of its force against the angle while the wheel is nearly still."""
    pts = [(s[1], s[5] * sign) for run in runs for s in run["samples"] if abs(s[2]) < QUASI_STATIC_RATE]
    if len(pts) < 20:
        return 0.0
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    var = sum((x - mx) ** 2 for x, _ in pts)
    return max(0.0, -sum((x - mx) * (y - my) for x, y in pts) / var) if var > 1e-9 else 0.0


SIM_WHEELS = {
    "r3": WheelParams(inertia=380.0, damping=630.0, stick=190.0, spring=2500.0, delay_s=0.036),
    "bare": WheelParams(inertia=150.0, damping=250.0, stick=190.0, spring=1500.0, delay_s=0.030),
    "geared": WheelParams(inertia=600.0, damping=900.0, stick=450.0, spring=3000.0, delay_s=0.025, cog=250.0,
                          cog_period=0.012, quantum=0.002),
}


def _params(x: list[float], base: WheelParams) -> WheelParams:
    return replace(base, inertia=math.exp(x[0]), damping=math.exp(x[1]) - 1.0, stick=math.exp(x[2]) - 1.0,
                   spring=math.exp(x[3]) - 1.0, delay_s=x[4])


def _vector(p: WheelParams) -> list[float]:
    return [math.log(p.inertia), math.log(p.damping + 1.0), math.log(p.stick + 1.0), math.log(p.spring + 1.0), p.delay_s]


def _nelder_mead(f, x0: list[float], steps: list[float], max_evals: int) -> tuple[list[float], float, int]:
    n = len(x0)
    pts = [list(x0)] + [[v + (steps[j] if j == i else 0.0) for j, v in enumerate(x0)] for i in range(n)]
    vals = [f(p) for p in pts]
    evals = n + 1
    while evals < max_evals:
        order = sorted(range(n + 1), key=vals.__getitem__)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if vals[-1] - vals[0] <= 1e-3 * vals[0]:
            break
        mid = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        worst = pts[-1]
        refl = [m + (m - w) for m, w in zip(mid, worst)]
        fr = f(refl)
        evals += 1
        if fr < vals[0]:
            far = [m + 2.0 * (m - w) for m, w in zip(mid, worst)]
            ff = f(far)
            evals += 1
            pts[-1], vals[-1] = (far, ff) if ff < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            anchor = refl if fr < vals[-1] else worst
            con = [m + 0.5 * (a - m) for m, a in zip(mid, anchor)]
            fc = f(con)
            evals += 1
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = con, fc
            else:
                best = pts[0]
                pts = [best] + [[b + 0.5 * (v - b) for b, v in zip(best, p)] for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
                evals += n
    i = min(range(n + 1), key=vals.__getitem__)
    return pts[i], vals[i], evals


@dataclass(frozen=True)
class FitResult:
    """``per_run`` is the error of each run's term in the fit: angle rms for a linear run, swing size for a pinned one."""

    params: WheelParams
    rms: float
    evals: int
    per_run: list[float]
    used: int


def _peak(run: dict) -> float:
    return max((abs(s[4]) for s in run["samples"]), default=0.0)


def _swing(ys: list[float], start: float) -> tuple[float, float]:
    """Largest and rms excursion from the resting angle: what a limit cycle's size is, whatever its phase."""
    dev = [y - start for y in ys]
    return max(abs(v) for v in dev), math.sqrt(sum(v * v for v in dev) / len(dev))


def fit(runs: list[dict], frame_hz: float, base: WheelParams, max_evals: int = 600, every: int = 2,
        ring: bool = True) -> FitResult:
    """Fit the rim to ``runs`` by Nelder-Mead: the wheel angle where the servo stayed linear, the swing where it pinned.

    A servo pinned at its cap with a delay is a relay, its limit cycle chaotic in phase but not in size, so those runs
    are matched on the size of the swing, which is what pins the delay. The delay is scanned first, it is flat in
    the small-signal runs. ``ring=False`` leaves the pinned runs out.
    """
    runs = [r for r in runs if len(r["samples"]) > 10]
    linear = [r for r in runs if _peak(r) < SATURATED_SHARE * r["gains"]["cap"]]
    pinned = [r for r in runs if r not in linear] if ring else []
    linear = linear or ([] if ring else runs)
    prepared = [([s[0] for s in r["samples"]][::every], [s[1] for s in r["samples"]][::every]) for r in linear]
    swings = [_swing([s[1] for s in r["samples"]], r["start"]) for r in pinned]

    def terms(p: WheelParams) -> list[float]:
        out = []
        for run, (ts, ys) in zip(linear, prepared):
            sim_t, sim_y = replay(run, p, frame_hz)
            out.append(math.sqrt(sum((y - interp(sim_t, sim_y, t)) ** 2 for t, y in zip(ts, ys)) / max(len(ts), 1)))
        for run, (peak, rms) in zip(pinned, swings):
            sim_peak, sim_rms = _swing(replay(run, p, frame_hz)[1], run["start"])
            out.append(math.sqrt(((sim_peak - peak) ** 2 + (sim_rms - rms) ** 2) / 2.0))
        return out

    def cost(x: list[float]) -> float:
        p = _params(x, base)
        if not INERTIA_RANGE[0] <= p.inertia <= INERTIA_RANGE[1] or not DELAY_MIN_S <= p.delay_s <= DELAY_MAX_S:
            return 1e3
        vals = terms(p)
        return math.sqrt(sum(v * v for v in vals) / len(vals))

    start = replace(base, inertia=400.0, damping=400.0, stick=150.0, spring=estimate_spring(linear or runs, base.sign))
    x0 = _vector(start)
    each = max(20, max_evals // (2 * len(DELAY_SCAN_S)))
    best = None
    evals = 0
    for delay in DELAY_SCAN_S:
        x, value, n = _nelder_mead(lambda v, d=delay: cost(v + [d]), x0[:4], [0.5, 0.5, 0.5, 0.7], each)
        evals += n
        if best is None or value < best[1]:
            best = (x + [delay], value)
    x, value, n = _nelder_mead(cost, best[0], [0.15, 0.15, 0.15, 0.2, 0.004], max(40, max_evals - evals))
    final = _params(x, base)
    return FitResult(final, value, evals + n, terms(final), len(linear))
