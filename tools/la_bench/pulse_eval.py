"""LKA pulses on the simulated wheel population: how far the truck and the wheel overshoot, how the push chatters.

Simulated only. The detector, the pulse trigger and LA's core in pulse mode run closed loop at 20 Hz as in
tests/lane_departure/test_lka.py (on that test's own wheel this harness gives the same pulses and swings), here on a
``plant.SimWheel`` at road speed: the game's centring is computed from the game's own reading of the wheel at every
FFB update and reaches the rim with the wheel's delay, since the plugin adds its servo to that force; the wheel's
output is clamped at full scale, and the truck steers from the same reading. A drift kicks the truck toward one line.

    python -m tools.la_bench.pulse_eval presets --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, fields, replace

from core.la.actuator import Gains, WheelTuning
from core.la.controller import AssistTuning, LaneAssist, Wheel
from core.la.hand_push import HandPushEstimator, PushTuning
from core.la.wheel_learner import WheelLearner, WheelModel, learned_overrides
from core.lane_departure.detector import LaneDetector, Tuning
from core.lane_departure.lka import LaneKeeping
from tests.la.servo_sim import GAIN, SPRING
from tests.lane_departure.test_detector import DT, HALF_WIDTH, SPEED, frame_at, make_route
from tools.la_bench.learn_sim import la_drive, manual_drive
from tools.la_bench.plant import SimWheel, WheelParams
from tools.la_bench.population import population, presets, split
from tools.la_bench.smooth_eval import reference

LINE = Tuning().half_lane_m - HALF_WIDTH
KICK_AT = 3.0
SECONDS = 14.0
YAW_LAG_S = 0.3
DRIFTS = (0.4, 0.8, 1.2)
HANDS = {"off": (0.0, 0.0, 0.0), "resting": (700.0, 3000.0, 150.0)}
FULL = 10000.0
TAIL_S = 0.5
ON_UNITS = 300.0
OFF_UNITS = 50.0
TURN_LOCK = 0.002
PAST_MAX = 0.3
OTHER_MAX = 0.0
PULSES_MAX = 3
TURNS_MAX = 2
CHATTER_MAX = 1
WHEEL_OVER_MAX = 0.01
SETTLED_LOCK = 0.002


class _GameForce:
    """The plugin with the game's centring added to its output, as the game's force reaches the wheel with it."""

    def __init__(self, plugin, spring: float, sign: int) -> None:
        self.plugin, self.spring, self.sign, self.game = plugin, spring, sign, 0.0

    def step(self, servo, torque, steer, dt, game=0.0, reading_age=0.0, fresh=True):
        self.game = -self.spring * steer * self.sign
        out = self.plugin.step(servo, torque, steer, dt, self.game, reading_age=reading_age, fresh=fresh) + self.game
        return max(-FULL, min(FULL, out))

    def __getattr__(self, name):
        return getattr(self.plugin, name)


class RoadWheel(SimWheel):
    """A ``SimWheel`` on the road: the game's spring at speed through the wheel's delay (``instant`` puts it on the rim
    with none), and the truck's sideways motion from the game's reading, its yaw ``YAW_LAG_S`` behind."""

    def __init__(self, params: WheelParams, game_spring: float = SPRING, instant: bool = False) -> None:
        super().__init__(replace(params, spring=params.spring + game_spring) if instant else params)
        if not instant:
            self.plugin = _GameForce(self.plugin, game_spring, params.sign)
        self.game_spring = game_spring
        self.e = self.ed = self.kappa = 0.0

    @property
    def game_units(self) -> float:
        return -self.game_spring * self.reading * self.params.sign

    def _advance(self, ccw: float) -> None:
        super()._advance(ccw)
        self.kappa += (-GAIN * self.reading - self.kappa) * self.h / YAW_LAG_S
        self.ed += SPEED * SPEED * self.kappa * self.h
        self.e += self.ed * self.h


def with_hands(p: WheelParams, hands: str) -> WheelParams:
    """Resting hands as inertia, damping and dry friction on top of the rim's, in the plugin's units."""
    inertia, damping, friction = HANDS[hands]
    return replace(p, inertia=p.inertia + inertia, damping=p.damping + damping, stick=p.stick + friction,
                   breakaway=p.breakaway + friction if p.breakaway > 0.0 else 0.0)


def drivable(p: WheelParams, hands: str = "resting", game_spring: float = SPRING, start: float = 0.02,
             seconds: float = 4.0) -> bool:
    """Whether the wheel comes to rest under the game's force alone, servo off, let go ``start`` off centre (friction
    may hold it off centre): one that rings or runs away by itself at road speed nobody could drive."""
    wheel = RoadWheel(with_hands(p, hands), game_spring)
    wheel.steer = wheel.rim = start
    lo, hi = math.inf, -math.inf
    while wheel.time < seconds:
        wheel.substep()
        if wheel.time > seconds - 1.5:
            lo, hi = min(lo, wheel.steer), max(hi, wheel.steer)
    return math.isfinite(hi - lo) and hi - lo < SETTLED_LOCK


@dataclass(frozen=True)
class PulseRun:
    """``past_m`` beyond either line, ``other_m`` beyond the far one, ``wheel_over`` the wheel past the pulse's target
    in its push (lock; mostly a target walking home past a wheel friction holds); per pulse, the worst: ``turns`` the wheel's reversals, ``chatter`` times the push let go and
    came back; ``jump`` the largest change of the push in one FFB update (units of 10000)."""

    pulses: int
    past_m: float
    other_m: float
    wheel_over: float
    turns: int
    chatter: int
    jump: float

    @property
    def ok(self) -> bool:
        return (self.past_m <= PAST_MAX and self.other_m <= OTHER_MAX and self.pulses <= PULSES_MAX
                and self.turns <= TURNS_MAX and self.chatter <= CHATTER_MAX and self.wheel_over <= WHEEL_OVER_MAX)


def _chatter(values: list[float]) -> int:
    on, armed, count = False, False, 0
    for v in values:
        if abs(v) >= ON_UNITS:
            count += armed and not on
            on = armed = True
        elif abs(v) <= OFF_UNITS:
            on = False
    return count


def _turns(angles: list[float]) -> int:
    """Reversals of the wheel's motion by more than TURN_LOCK (a hysteresis, so a coarse reading does not count)."""
    if not angles:
        return 0
    lo = hi = angles[0]
    way, count = 0, 0
    for a in angles:
        lo, hi = min(lo, a), max(hi, a)
        if way >= 0 and hi - a > TURN_LOCK:
            count += way > 0
            way, lo = -1, a
        elif way <= 0 and a - lo > TURN_LOCK:
            count += way < 0
            way, hi = 1, a
    return count


def pulse_run(p: WheelParams, tune: WheelTuning, drift: float, hands: str = "off", instant: bool = False,
              game_spring: float = SPRING, seconds: float = SECONDS, trace: list | None = None,
              model: WheelModel | None = None) -> PulseRun:
    """One drift at ``drift`` m/s, the LKA closed loop at 20 Hz on the wheel at the plugin's rate. ``trace`` collects
    ``(t, e, reading, target, authority, push)`` per FFB update; ``model`` is what the wheel learner knows."""
    wheel = RoadWheel(with_hands(p, hands), game_spring, instant)
    assist, det, keeping = LaneAssist(AssistTuning(pulse=True, settle_s=1.0, wheel=tune)), LaneDetector(), LaneKeeping()
    if model is not None:
        assist.wheel.learned(model)
    gains = Gains(game_spring / (GAIN * SPEED * SPEED) * p.sign, GAIN)
    route = make_route(None, length=2500.0)
    est = HandPushEstimator(PushTuning(inertia=tune.inertia))
    out, hand, rate_lp, game_lp = None, None, 0.0, 0.0
    pulses, swing, over, peak, live_until = 0, 0.0, 0.0, 0.0, -math.inf
    ticks: list[tuple[int, float, float]] = []
    for i in range(int(seconds / DT)):
        t = i * DT
        if abs(t - KICK_AT) < 1e-6:
            wheel.ed = drift
        frame = frame_at(t, route, None, wheel.e, math.atan2(wheel.ed, SPEED), steer_right=-wheel.reading)
        pulse = keeping.update(t, det.update(frame), det, frame, out)
        pulses += pulse.event == "start"
        now = Wheel(ok=True, steer=wheel.reading, rate=wheel.plugin.rate, push=wheel.plugin.value,
                    buzz=wheel.plugin.scale, game=wheel.game_units)
        out = assist.update(frame, gains, now, engage=pulse.side, hand=hand)
        if out.authority > 0.0:
            live_until = t + DT + TAIL_S
            if out.servo.only:
                over = max(over, (wheel.reading - out.target) * out.servo.only)
        servo = out.servo
        while wheel.time < (i + 1) * DT - 1e-9:
            tick = wheel.n % wheel.substeps == 0
            wheel.substep(servo, out.torque)
            if tick and t < live_until:
                ticks.append((pulses, wheel.plugin.value, wheel.reading))
            if tick and trace is not None:
                trace.append((wheel.time, wheel.e, wheel.reading, out.target, out.authority, wheel.plugin.value))
            peak = max(peak, abs(wheel.e))
        if pulses:
            swing = max(swing, -wheel.e if drift > 0.0 else wheel.e)
        rate_lp += (wheel.plugin.rate - rate_lp) * (1.0 - math.exp(-DT / 0.04))
        game_lp += (wheel.game_units - game_lp) * (1.0 - math.exp(-DT / 0.10))
        hand = est.update(DT if i else 0.0, rate_lp, game_lp, wheel.plugin.value, 0.0, p.sign, out.servo.only)
    each = [[(v, a) for k, v, a in ticks if k == n] for n in range(1, pulses + 1)]
    steps = [abs(b[1] - a[1]) for a, b in zip(ticks, ticks[1:]) if a[0] == b[0]]
    return PulseRun(pulses=pulses, past_m=max(0.0, peak - LINE), other_m=max(0.0, swing - LINE),
                    wheel_over=max(0.0, over), turns=max((_turns([a for _, a in e]) for e in each), default=0),
                    chatter=max((_chatter([v for v, _ in e]) for e in each), default=0), jump=max(steps, default=0.0))


def learned_model(p: WheelParams, seed: int) -> WheelModel:
    """What the wheel learner knows after a manual drive and LA's holds on the bench condition (learn_sim)."""
    learner = WheelLearner()
    manual_drive(p, learner, seed=seed)
    la_drive(p, learner, seed=seed)
    return learner.model


def designs(p: WheelParams, model: WheelModel) -> dict[str, WheelTuning]:
    """Today's LKA (the configured default), ``branch`` with the hold and the settle flag (la-smooth-ffb's LKA), the
    learner's design with the hold as LA gets it, and the tuning grid's reference design on the true rim."""
    base = WheelTuning()
    ref = reference(p)
    names = {f.name for f in fields(WheelTuning)}
    return {
        "default": base,
        "branch": replace(base, hold=True, settle=True),
        "learned": replace(base, hold=True, **learned_overrides(base, model)),
        "reference": replace(base, **{k: v for k, v in asdict(ref).items() if k in names}),
    }


def evaluate(wheels: list[tuple[str, WheelParams]], hands=tuple(HANDS), drifts=DRIFTS) -> dict:
    """Every design on every wheel, hands and drift."""
    rows = []
    for i, (name, p) in enumerate(wheels):
        model = learned_model(p, i)
        runs = {who: {h: {str(d): asdict(pulse_run(p, tune, d, h, model=model)) for d in drifts} for h in hands}
                for who, tune in designs(p, model).items()}
        rows.append({"name": name, "params": asdict(p), "model": model.to_dict(), "runs": runs})
        print(name, {w: sum(PulseRun(**r).ok for h in v.values() for r in h.values()) for w, v in runs.items()},
              flush=True, file=sys.stderr)
    return {"rows": rows, "tables": tables(rows)}


def tables(rows: list[dict]) -> dict:
    """Per design and hands: the share of runs and of wheels that pass, and each metric's median, p90 and worst."""
    out: dict = {}
    for who in rows[0]["runs"] if rows else ():
        for h in rows[0]["runs"][who]:
            runs = [PulseRun(**r) for row in rows for r in row["runs"][who][h].values()]
            wheels = [all(PulseRun(**r).ok for r in row["runs"][who][h].values()) for row in rows]
            col = {"runs_pass": sum(r.ok for r in runs) / len(runs), "wheels_pass": sum(wheels) / len(wheels)}
            for f in fields(PulseRun):
                v = sorted(getattr(r, f.name) for r in runs)
                col[f.name] = {"median": v[len(v) // 2], "p90": v[int(0.9 * (len(v) - 1))], "worst": v[-1]}
            out[f"{who}/{h}"] = col
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=("held", "tune", "presets"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0, help="this slice of the wheels (with --of)")
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args(argv)
    tune, held = split(population())
    wheels = {"held": held, "tune": tune, "presets": presets()}[args.which][args.part::args.of]
    data = evaluate(wheels)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
    print(json.dumps(data["tables"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
