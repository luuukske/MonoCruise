"""Learns the wheel from normal driving, so the servo needs no calibration. Pure logic, offline, kept locally.

Fed the plugin's trace samples (``core/lane_departure/ffb_trace.py``: time, the reading, the game's force, what went to
the wheel, the servo's part, the steering frame count, flags) as they come. What normal driving shows, and only that:

- with or without the servo: how often the game reads the wheel (the reading rate);
- with the servo off: in what steps it reads it (the quantum; a servo that hunts reads in big steps only) and the
  game's centring against the angle (its spring, at the speeds driven);
- with the servo on, from LA's own pushes (nothing extra is ever pushed): the force a standing wheel starts on and
  the force it slides on, hands included, which is what the servo has to move;
- from the target hold: whether the servo hunts at a still target, which steps the servo design down a rung.

The rim's own inertia and the delay are not learned: with hands on the rim they do not show in normal driving.
Every estimate is bounded and starts unknown (0), which leaves the servo exactly as configured. See core/la/README.md.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, fields, replace

FLAG_SERVO = 1
FLAG_RUMBLE = 2
QUANTUM_MIN = 1e-4
QUANTUM_MAX = 0.01
GRID_TOL = 0.15
STALL_S = 0.2
STILL_READINGS = 3
MIN_START = 20.0
FORCE_MAX = 1500.0
RATE_MIN = 0.005
RATE_MAX = 0.3
RUNG_MAX = 6
CALM_UP_S = 600.0
START_WINDOW_S = 0.15
START_FLAT = 0.2


@dataclass(frozen=True)
class WheelModel:
    """What is known about the wheel; 0 is unknown. ``rung`` is how many steps below the configured design it runs."""

    reading_hz: float = 0.0
    quantum: float = 0.0
    quantum_n: int = 0
    spring: float = 0.0
    start_force: float = 0.0
    start_top: float = 0.0
    starts: int = 0
    slide_force: float = 0.0
    damping: float = 0.0
    slides: int = 0
    rung: int = 0
    calm_s: float = 0.0

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, raw: object) -> WheelModel:
        """A saved model, every field bounded; anything unreadable is unknown."""
        if not isinstance(raw, dict):
            return cls()
        out = {}
        for f in fields(cls):
            v = raw.get(f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            out[f.name] = int(v) if f.type in ("int", int) else float(v)
        return bound(cls(**out))


def bound(m: WheelModel) -> WheelModel:
    def clip(v: float, lo: float, hi: float) -> float:
        return 0.0 if v <= 0.0 else max(lo, min(hi, v))

    q = m.quantum if QUANTUM_MIN <= m.quantum <= QUANTUM_MAX else 0.0
    return replace(m, reading_hz=clip(m.reading_hz, 10.0, 1000.0), quantum=q, quantum_n=max(0, m.quantum_n),
                   spring=clip(m.spring, 0.0, 200000.0), start_force=clip(m.start_force, MIN_START, FORCE_MAX),
                   start_top=clip(max(m.start_top, m.start_force), MIN_START, FORCE_MAX), starts=max(0, m.starts),
                   slide_force=clip(m.slide_force, 0.0, FORCE_MAX), damping=clip(m.damping, 0.0, 5000.0),
                   slides=max(0, m.slides), rung=max(0, min(RUNG_MAX, m.rung)), calm_s=max(0.0, m.calm_s))


@dataclass(frozen=True)
class LearnTuning:
    """How much evidence each estimate needs before it is used."""

    quantum_min_n: int = 200
    starts_min: int = 8
    resisted_min: int = 20
    slides_min: int = 200
    spring_min_n: int = 200
    keep: int = 512


def _quantile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]


def quantum_of(steps: list[float]) -> float:
    """The reading's step from the changes it made: the smallest common step they are whole multiples of, 0 when they
    are not on a grid (a fine encoder) or finer than QUANTUM_MIN."""
    steps = [d for d in steps if d > 0.0]
    if len(steps) < 20:
        return 0.0
    q0 = _quantile(steps, 0.02)
    if q0 < QUANTUM_MIN:
        return 0.0
    near = [d for d in steps if d <= 6.0 * q0]
    q = _quantile([d / max(1, round(d / q0)) for d in near], 0.5)
    off = _quantile([abs(d / q - round(d / q)) for d in near], 0.5)
    return q if off <= GRID_TOL and q >= QUANTUM_MIN else 0.0


_SHARED: list[WheelLearner] = []


def shared_learner(saved: object = None) -> WheelLearner:
    """The one learner per process, LA and the lane keeping assist share the wheel; ``saved`` seeds the first."""
    if not _SHARED:
        _SHARED.append(WheelLearner(WheelModel.from_dict(saved)))
    return _SHARED[0]


class WheelLearner:
    def __init__(self, model: WheelModel | None = None, tuning: LearnTuning | None = None) -> None:
        self.tune = tuning or LearnTuning()
        self._model = bound(model or WheelModel())
        n = self.tune.keep
        self._intervals: deque[float] = deque(maxlen=n)
        self._steps: deque[float] = deque(maxlen=n)
        self._starts: deque[float] = deque(maxlen=64)
        self._resisted: deque[float] = deque(maxlen=128)
        self._held_max = 0.0
        self._slides: deque[tuple[float, float]] = deque(maxlen=n)
        self._spring = [0.0] * 6
        self._seq: int | None = None
        self._read: float | None = None
        self._read_t = 0.0
        self._still = 0
        self._forces: deque[tuple[float, float]] = deque()
        self._servo_t: float | None = None
        self._last_t = -math.inf
        self.direction = 0

    @property
    def model(self) -> WheelModel:
        return self._model

    def feed(self, t: float, steer: float, game: float, out: float, servo: float, seq: int, flags: int) -> None:
        """One trace sample; ``out`` and ``game`` in the plugin's torque sign, ``direction`` set while LA steers.
        A sample no newer than the last is skipped, so LA and the lane keeping assist can both feed one learner."""
        if not all(math.isfinite(v) for v in (t, steer, game, out, servo)) or t <= self._last_t:
            return
        self._last_t = t
        if flags & FLAG_RUMBLE:
            self._seq = None
            return
        on = bool(flags & FLAG_SERVO) and self.direction in (-1, 1)
        if on and self._servo_t is not None and 0.0 < t - self._servo_t < STALL_S:
            self._model = replace(self._model, calm_s=self._model.calm_s + (t - self._servo_t))
        self._servo_t = t if on else None
        if self._seq is None or seq < self._seq:
            self._seq, self._read, self._read_t, self._still = seq, steer, t, 0
            self._forces.clear()
            return
        if on:
            self._forces.append((t, abs(out)))
            while self._forces and t - self._forces[0][0] > START_WINDOW_S:
                self._held_max = max(self._held_max, self._forces.popleft()[1])
        else:
            self._forces.clear()
            self._held_max = 0.0
        if seq == self._seq:
            return
        gap = (t - self._read_t) / (seq - self._seq)
        if 0.0 < gap < STALL_S:
            self._intervals.append(gap)
        step = abs(steer - self._read)
        if step > 0.0:
            if not flags & FLAG_SERVO:
                self._steps.append(step)
            start = self._start_force(t)
            if on and self._still >= STILL_READINGS and start is not None:
                self._starts.append(start)
            if on and self._still >= STILL_READINGS and self._held_max >= MIN_START:
                self._resisted.append(min(FORCE_MAX, self._held_max))
            self._held_max = 0.0
            if on and self._still == 0 and 0.0 < gap < STALL_S:
                rate = (steer - self._read) / (t - self._read_t)
                if RATE_MIN <= abs(rate) <= RATE_MAX:
                    self._slides.append((abs(rate), out * self.direction * math.copysign(1.0, rate)))
            self._still = 0
        else:
            self._still += seq - self._seq
        if not flags & FLAG_SERVO:
            self._note_spring(steer, game)
        self._seq, self._read, self._read_t = seq, steer, t
        self._refresh()

    def _start_force(self, t: float) -> float | None:
        """The force a standing wheel started on: only after a slow climb (within START_FLAT over the last
        START_WINDOW_S, which covers the delay and the reading's age), else the force ran past the start unseen."""
        win = [f for ft, f in self._forces if t - ft <= START_WINDOW_S]
        if len(win) < 4 or self._forces[0][0] > t - 0.8 * START_WINDOW_S:
            return None
        low, high = min(win), max(win)
        if high < MIN_START or high - low > START_FLAT * high:
            return None
        return min(FORCE_MAX, low)

    def _note_spring(self, steer: float, game: float) -> None:
        keep = 1.0 - 1.0 / (4.0 * self.tune.keep)
        s = self._spring
        s[:] = [s[0] * keep + 1.0, s[1] * keep + steer, s[2] * keep + game, s[3] * keep + steer * steer,
                s[4] * keep + steer * game, s[5] + 1.0]

    def note_hunt(self) -> None:
        """The servo hunted at a still target it could not park on: one rung gentler, the calm count starts over."""
        m = self._model
        self._model = replace(m, rung=min(RUNG_MAX, m.rung + 1), calm_s=0.0)

    def _refresh(self) -> None:
        tn, m = self.tune, self._model
        if len(self._intervals) >= 20:
            m = replace(m, reading_hz=1.0 / _quantile(list(self._intervals), 0.5))
        if len(self._steps) >= tn.quantum_min_n:
            m = replace(m, quantum=quantum_of(list(self._steps)), quantum_n=len(self._steps))
        starts, held = list(self._starts), list(self._resisted)
        if len(starts) >= tn.starts_min:
            m = replace(m, start_force=_quantile(starts, 0.5), starts=len(starts),
                        start_top=max(_quantile(starts, 0.9), _quantile(held, 0.9) if held else 0.0))
        elif len(held) >= tn.resisted_min:
            m = replace(m, start_force=_quantile(held, 0.75), start_top=_quantile(held, 0.9), starts=len(held))
        if len(self._slides) >= tn.slides_min:
            m = self._fit_slides(m)
        n, sx, sy, sxx, sxy, seen = self._spring
        var = n * sxx - sx * sx
        if seen >= tn.spring_min_n and var > 1e-9 * n * n:
            m = replace(m, spring=abs(n * sxy - sx * sy) / var)
        if m.rung > 0 and m.calm_s >= CALM_UP_S:
            m = replace(m, rung=m.rung - 1, calm_s=0.0)
        self._model = bound(m)

    def _fit_slides(self, m: WheelModel) -> WheelModel:
        """Force along the motion against its rate on steady slides: friction plus damping times rate."""
        pts = list(self._slides)
        n = len(pts)
        sx, sy = sum(p[0] for p in pts), sum(p[1] for p in pts)
        sxx, sxy = sum(p[0] * p[0] for p in pts), sum(p[0] * p[1] for p in pts)
        var = n * sxx - sx * sx
        if var <= 1e-12:
            return m
        c = (n * sxy - sx * sy) / var
        f = (sy - c * sx) / n
        if c < 0.0:
            c, f = 0.0, sy / n
        return replace(m, slide_force=max(0.0, f), damping=c, slides=n)


RUNG_BANDWIDTH = 0.7
RUNG_INERTIA = 0.7
RUNG_DAMPING = 0.2
BANDWIDTH_FLOOR = 0.5
INERTIA_FLOOR = 150.0
DAMPING_TOP = 1.4
STICKY = 1.2
BREAKAWAY_SHARE = 1.2
PLAN_S = 0.04
RUNG_BARE = 2
RUNG_PATIENCE = 2
DESIGN_RUNGS = 4
SLIDE_SHARE = 0.6


def learned_overrides(tune, model: WheelModel, pushes_set: bool = False) -> dict:
    """``WheelTuning`` fields from what was learned, on top of ``tune`` (the configured design): ``rung`` steps
    gentler, never stiffer, and a friction push from the starts when none is configured (the slides' damping is not
    pushed: their fit read it up to four times high). Unknown: nothing.
    Below RUNG_BARE only with a friction push: a gentle servo alone cannot move a sticky wheel at all."""
    out: dict = {}
    pushes = pushes_set or model.starts >= LearnTuning.starts_min
    k = model.rung if pushes or model.rung <= RUNG_BARE else max(RUNG_BARE, model.rung - RUNG_PATIENCE)
    k = min(k, DESIGN_RUNGS)
    if k > 0:
        out.update(bandwidth_hz=min(tune.bandwidth_hz, max(BANDWIDTH_FLOOR, tune.bandwidth_hz * RUNG_BANDWIDTH ** k)),
                   inertia=min(tune.inertia, max(INERTIA_FLOOR, tune.inertia * RUNG_INERTIA ** k)),
                   damping=max(tune.damping, min(DAMPING_TOP, tune.damping + RUNG_DAMPING * k)), damping_first=True)
    if not pushes or pushes_set:
        return out
    friction = SLIDE_SHARE * model.start_force
    if model.slides >= LearnTuning.slides_min and model.slide_force > 0.0:
        friction = min(model.slide_force, friction)
    out.update(ff_friction=friction, plan_s=PLAN_S)
    if model.start_top > STICKY * friction:
        out["ff_breakaway"] = friction + BREAKAWAY_SHARE * (model.start_top - friction)
    return out
