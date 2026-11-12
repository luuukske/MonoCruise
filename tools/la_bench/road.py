"""The road side of a servo design: the game's centring at speed, which the standing wheel never shows.

Simulated only. Standing, the servo is the whole stiffness on the rim. At speed the game's own centring is most of it
(about 122 per lock per (m/s)^2 on the test truck: 76000 at 90 km/h, 159000 at 130), the servo keeps a quarter of its
design as its own push and its integral has to carry the rest, so a design picked on the standing rim alone can be
smooth and slow: 1.4 Hz on 415 trailed its target by 0.64 s at 90 km/h and lane keeping pulses crossed the line
(2026-11-11). See core/la/README.md, "Fast on the road".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from core.la.actuator import WheelActuator, WheelTuning
from tools.la_bench.bench import sim_bench
from tools.la_bench.plant import WheelParams
from tools.la_bench.protocol import Design

SPRING_PER_V2 = 122.0
ROAD_KMH = (60.0, 90.0, 130.0)
CHATTER_KMH = 60.0
DRIFTS = (0.4, 1.2)
HANDS = ("off", "resting")
PAST_MAX_M = 1.0
TURNS_MAX = 8
LAG_SOFT_S = 0.35
LAG_SOFT_KMH = 90.0
ROAD_BANDWIDTHS = (3.0, 2.5, 2.0)
ROAD_INERTIA_SHARES = (0.35, 0.5, 0.7, 1.0)
ROAD_DAMPINGS = (1.0, 0.7)
ROAD_GAIN = 0.8
STEP = 0.02
STEP_OVER_MAX = 0.25
STEP_KMH = (45.0, 50.0, 60.0, 90.0, 130.0)
BUZZ_MIN = 0.6
LATE = 1.3
LATE_KMH = (90.0, 130.0)
PULSE_TRIES = 8
BETTER = 0.5


def spring_per_v2(saved: object = None) -> float:
    """The game's centring per lock per (m/s)^2 from a saved live calibration (``la_saved_calibration``), else the test
    truck's: it is in the game's own force units, so the wheel does not change it; the truck and the game's settings do."""
    try:
        got = abs(float(saved["force_per_ms2"]) * float(saved["kappa_per_steer"]))
    except (TypeError, KeyError, ValueError):
        return SPRING_PER_V2
    return got if math.isfinite(got) and 20.0 <= got <= 1000.0 else SPRING_PER_V2


def spring_at(kmh: float, per_v2: float = SPRING_PER_V2) -> float:
    return per_v2 * (kmh / 3.6) ** 2


def lags(tune: WheelTuning, per_v2: float = SPRING_PER_V2, speeds=ROAD_KMH) -> dict[float, float]:
    """How far the wheel trails a moving target per speed: the game's spring over the integral gain, in seconds."""
    act = WheelActuator(tune)
    out = {}
    for kmh in speeds:
        spring = spring_at(kmh, per_v2)
        ki = act.gains(1.0, spring)[2]
        out[kmh] = spring / ki if ki > 0.0 else math.inf
    return out


def chatter(rim: WheelParams, design: Design, frame_hz: float, spring: float) -> float:
    """The design's force chatter on a 0.2 lock/s ramp with the gains it has at ``spring`` (``design.chatter``): what
    the servo adds to the wheel on the road, where its own push is a share of what it is standing."""
    from tools.la_bench.design import CHATTER_STAGE
    from tools.la_bench.design import chatter as stage_chatter

    plant = replace(rim, spring=rim.spring + spring)
    bench = sim_bench(plant, frame_hz)
    bench.say = lambda _msg: None
    bench.ready()
    return stage_chatter(bench.run_stage(CHATTER_STAGE, replace(design, spring=spring), plant.sign))


@dataclass(frozen=True)
class RoadResult:
    """``past_m``: the worst a lane keeping pulse let the truck run past a line; ``turns``: the wheel's most reversals
    in one pulse (a ring). ``ok``: no run ran away or rang."""

    past_m: float
    turns: int
    worst: str

    @property
    def ok(self) -> bool:
        return self.past_m <= PAST_MAX_M and self.turns <= TURNS_MAX


def pulses(rim: WheelParams, tune: WheelTuning, per_v2: float = SPRING_PER_V2, speeds=ROAD_KMH, drifts=DRIFTS,
           hands=HANDS) -> RoadResult:
    """Lane keeping pulses on the rim at the game's spring for each speed, hands off and resting, a slow and a fast
    drift (``pulse_eval.pulse_run``; the truck's own motion is the model's at 90 km/h throughout). With the target
    hold on, as the app runs a pulse (``la_wheel_learning``): without it every design hunts round a still target.
    """
    from tools.la_bench.pulse_eval import pulse_run

    tune = replace(tune, hold=True)
    past, turns, worst = 0.0, 0, ""
    for kmh in speeds:
        for hand in hands:
            for drift in drifts:
                run = pulse_run(rim, tune, drift, hand, game_spring=spring_at(kmh, per_v2))
                if run.past_m > past:
                    past, worst = run.past_m, f"{kmh:.0f} km/h, hands {hand}, drift {drift} m/s"
                turns = max(turns, run.turns)
    return RoadResult(past, turns, worst)


def describe(tune: WheelTuning, per_v2: float = SPRING_PER_V2) -> str:
    got = lags(tune, per_v2)
    line = "on the road the wheel trails its target by " + ", ".join(f"{s:.2f} s at {k:.0f} km/h" for k, s in got.items())
    soft = lags(tune, per_v2, (LAG_SOFT_KMH,))[LAG_SOFT_KMH]
    return line + (f"; over {LAG_SOFT_S:.2f} s at {LAG_SOFT_KMH:.0f} km/h is soft for lane keeping" if soft > LAG_SOFT_S else "")


def step_over(rim: WheelParams, design: Design, frame_hz: float, spring: float) -> float:
    """The overshoot of a 0.02 step on the rim at ``spring`` of the game's centring, up and through zero; infinite
    when the run aborted or the plugin's buzz guard cut in."""
    from tools.la_bench.metrics import summarize
    from tools.la_bench.protocol import step

    plant = replace(rim, spring=rim.spring + spring)
    bench = sim_bench(plant, frame_hz)
    bench.say = lambda _msg: None
    bench.ready()
    run = bench.run_stage(step(STEP), replace(design, spring=spring), plant.sign)
    s = summarize(run)
    if run.get("aborted") or s["buzz_min"] < BUZZ_MIN or "up" not in s:
        return math.inf
    return max(s["up"]["overshoot"], s["swing"]["overshoot"])


def road_design(rim: WheelParams, base: WheelTuning, frame_hz: float, per_v2: float = SPRING_PER_V2,
                say=None) -> tuple[float, float, float] | None:
    """The road design ``(inertia, bandwidth, damping)`` to blend in beside ``base`` (the standing design with its
    pushes), or None when none holds: the candidate that trails least at 90 km/h among those whose 0.02 step stays
    under STEP_OVER_MAX through the blend and at 60, 90 and 130 km/h (the rim's delay a third longer too) and on
    which lane keeping pulses neither run away nor ring. Its chatter at 60 km/h is said, not judged (Lukas,
    2026-11-12: the push comes first; the limit left the R3 with no road design at all on a run that read 11 ms).
    The pulses run on the rim as fitted only: with a third more delay the model's wheel rings hands off at 130 km/h
    under the game's centring alone, with any design or none. On a rim where none of the first PULSE_TRIES holds, the
    one whose pulses run least far past a line, when that is under BETTER of the standing design's alone and rings no
    more: slower wheels in the model run metres wide on any design, and far wider on the standing one.
    """
    base = replace(base, road_inertia=0.0, road_bandwidth_hz=0.0, road_damping=0.0)
    want = ROAD_GAIN * lags(base, per_v2, (LAG_SOFT_KMH,))[LAG_SOFT_KMH]
    tried = []
    for bw in ROAD_BANDWIDTHS:
        for share in ROAD_INERTIA_SHARES:
            for z in ROAD_DAMPINGS:
                tune = replace(base, road_inertia=max(50.0, round(share * rim.inertia)), road_bandwidth_hz=bw, road_damping=z)
                lag = lags(tune, per_v2, (LAG_SOFT_KMH,))[LAG_SOFT_KMH]
                if lag <= want:
                    tried.append((lag, tune))
    late = replace(rim, delay_s=rim.delay_s * LATE)
    alone, best, runs = None, None, 0
    for lag, tune in sorted(tried, key=lambda c: c[0]):
        if runs >= PULSE_TRIES:
            break
        design = Design(bandwidth_hz=tune.bandwidth_hz, damping=tune.damping, inertia=tune.inertia, damping_first=True,
                        ff_inertia=tune.ff_inertia, ff_damping=tune.ff_damping, ff_friction=tune.ff_friction,
                        plan_s=tune.plan_s, road_inertia=tune.road_inertia, road_bandwidth_hz=tune.road_bandwidth_hz,
                        road_damping=tune.road_damping)
        name = f"{tune.road_bandwidth_hz:.1f} Hz on {tune.road_inertia:.0f}, damping {tune.road_damping:.1f}"
        over = max(step_over(plant, design, frame_hz, spring_at(kmh, per_v2))
                   for plant, speeds in ((rim, STEP_KMH), (late, LATE_KMH)) for kmh in speeds)
        if over > STEP_OVER_MAX:
            _say(say, f"  {name}: a 0.02 step overshoots {over:.0%} on the road")
            continue
        got = pulses(rim, tune, per_v2)
        runs += 1
        felt = chatter(rim, design, frame_hz, spring_at(CHATTER_KMH, per_v2)) if say is not None else 0.0
        _say(say, f"  {name}: trails {lag:.2f} s at {LAG_SOFT_KMH:.0f} km/h, step {over:.0%} over, chatter {felt:.0f}, pulses at worst "
                  f"{got.past_m:.2f} m past a line ({got.worst}), {got.turns} wheel reversals: "
                  + ("holds" if got.ok else "no"))
        if got.ok:
            return tune.road_inertia, tune.road_bandwidth_hz, tune.road_damping
        alone = alone or pulses(rim, base, per_v2)
        if (got.past_m <= BETTER * alone.past_m and got.turns <= max(TURNS_MAX, alone.turns)
                and (best is None or got.past_m < best[0])):
            best = (got.past_m, name, tune)
    if best is None:
        return None
    _say(say, f"  none holds every case; {best[1]} runs {best[0]:.2f} m past a line at worst where the standing design "
              f"alone runs {alone.past_m:.2f} m")
    return best[2].road_inertia, best[2].road_bandwidth_hz, best[2].road_damping


def _say(say, text: str) -> None:
    if say is not None:
        say(text)
