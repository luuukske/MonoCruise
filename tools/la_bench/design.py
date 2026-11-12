"""Pick the servo design for a measured rim, and score how smooth the wheel is.

The design is tried on the fitted rim and on four neighbours of it (more and less delay, a lighter rim, a heavier
rim with hands) so it does not rest on one fit. A design holds up when no stage aborts, the buzz guard stays quiet,
nothing overshoots past ``MAX_OVERSHOOT`` and the fitted rim itself stays under ``NOMINAL_OVERSHOOT``; the fastest one
wins. The design inertia is searched from the fitted one up: on the R3 a design on 650 (stiff, well damped for a rim of
300) was the fastest that held, and one designed on the fitted inertia was half as stiff. Thresholds are first guesses.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, replace

from tools.la_bench.bench import sim_bench
from tools.la_bench.metrics import aim_at, summarize
from tools.la_bench.plant import WheelParams
from tools.la_bench.protocol import Design, follow, ramp, step

BANDWIDTHS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
DAMPINGS = (0.7, 1.0, 1.4)
MAX_OVERSHOOT = 0.6
NOMINAL_OVERSHOOT = 0.4
INERTIA_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0)
INERTIA_MAX = 1000.0
MIN_BUZZ = 0.6
STAGES = (step(0.02), follow(0.03, 0.7))
AHEAD_STAGES = (step(0.02), ramp(0.15, 0.4, 0.1), follow(0.03, 0.7))
STICKY_STAGES = AHEAD_STAGES + (ramp(0.06, 0.05),)
BREAKAWAY_SHARES = (1.1, 1.4)
BREAKAWAY_LOW = 0.75
STICKY_RATIO = 1.2
MEASURED_LOW = 0.85
STUCK_MAX = 0.25
RESOLUTION_MAX = 2e-4
TRACK_MAX = 0.01
FRICTION_MAX = 0.12
SERVO_CAP = 3000.0
RAMP_TRACK_MAX = 0.015
NEIGHBOUR_TRACK_MAX = 0.03
STILL_RATE = 0.005
CHATTER_STAGE = ramp(0.15, 0.2, 0.1)
MOVING_RATE = 0.03


@dataclass(frozen=True)
class Choice:
    """A servo design for one rim; ``limited`` means nothing in the grid held up and this is the gentlest."""

    inertia: float
    bandwidth_hz: float
    damping: float
    rise_s: float | None
    lag_s: float | None
    worst_overshoot: float
    limited: bool = False
    ff_inertia: float = 0.0
    ff_damping: float = 0.0
    ff_friction: float = 0.0
    plan_s: float = 0.0
    ff_breakaway: float = 0.0
    road_inertia: float = 0.0
    road_bandwidth_hz: float = 0.0
    road_damping: float = 0.0

    def road_config(self) -> dict:
        """The road design's keys, zeros when there is none so an older one does not stay in the config."""
        return {"la_road_inertia": round(self.road_inertia), "la_road_bandwidth_hz": round(self.road_bandwidth_hz, 2),
                "la_road_damping": round(self.road_damping, 2)}

    def design(self) -> Design:
        return Design(bandwidth_hz=self.bandwidth_hz, damping=self.damping, inertia=self.inertia, damping_first=True,
                      ff_inertia=self.ff_inertia, ff_damping=self.ff_damping, ff_friction=self.ff_friction,
                      plan_s=self.plan_s, ff_breakaway=self.ff_breakaway)

    def config(self) -> dict:
        out = {"la_wheel_inertia": round(self.inertia), "la_bandwidth_hz": round(self.bandwidth_hz, 2),
               "la_damping": round(self.damping, 2), "la_damping_first": True}
        if self.ff_inertia or self.ff_damping or self.ff_friction:
            out.update({"la_ff_inertia": round(self.ff_inertia), "la_ff_damping": round(self.ff_damping),
                        "la_ff_friction": round(self.ff_friction), "la_ff_plan_s": round(self.plan_s, 3)})
        if self.ff_breakaway > self.ff_friction > 0.0:
            out["la_ff_breakaway"] = round(self.ff_breakaway)
        return out


def sticky(params: WheelParams) -> bool:
    """A rim that standing needs more to start than to slide (the breakaway read off the wheel). Only a wheel that
    stick-slipped in the climb is given a breakaway at all; the old wheel's slide probe read 434 to start, 298 to slide.
    """
    return params.breakaway > STICKY_RATIO * params.stick > 0.0


def breakaway_top(params: WheelParams) -> float:
    """The most a start of this rim takes: its breakaway and the spread from one start to the next."""
    return params.breakaway * (1.0 + params.breakaway_spread)


def push_ahead_of(params: WheelParams, share: float = 1.0, breakaway_share: float = 0.0) -> dict:
    """The push ahead for a fitted rim: its inertia, damping and dry friction (``share`` of it) on the plan its delay
    early, and the ceiling the plugin's push climbs to while the wheel stands: that friction plus ``breakaway_share`` of
    the gap up to the highest start (a fixed push between them left the old wheel standing at 19:16, 2026-11-06).
    """
    out = {"ff_inertia": params.inertia, "ff_damping": params.damping, "ff_friction": share * params.stick,
           "plan_s": params.delay_s}
    if breakaway_share > 0.0 and sticky(params):
        out["ff_breakaway"] = out["ff_friction"] + breakaway_share * (breakaway_top(params) - out["ff_friction"])
    return out


def neighbours(p: WheelParams, ahead: bool = False, measured: bool = False) -> list[WheelParams]:
    """The fitted rim first, then what a design must also hold up on. With the push ahead also a rim with half the
    friction: the push then over-pushes at every stop, which is where a friction push hunts. A sticky rim also with a
    quarter less breakaway, where the breakaway push over-pushes at every start. With the friction ``measured`` by the
    slide probe (the old wheel stopped at every hold of 298), that rim has only 15 % less, and the rim with hands no more
    than the breakaway: both of the old ones rejected a design that held on the measured rim.
    """
    hands = p.stick * 1.5 if not (measured and sticky(p)) else min(p.stick * 1.5, p.breakaway)
    out = [p, replace(p, delay_s=p.delay_s * 1.3), replace(p, delay_s=p.delay_s * 0.8),
           replace(p, inertia=p.inertia * 0.75),
           replace(p, inertia=p.inertia * 1.6, damping=p.damping * 1.3, stick=hands)]
    if not ahead:
        return out
    out.append(replace(p, stick=p.stick * (MEASURED_LOW if measured else 0.5), damping=p.damping * 0.8))
    if sticky(p):
        out.append(replace(p, breakaway=max(p.stick, p.breakaway * BREAKAWAY_LOW)))
    return out


def chatter(run: dict) -> float:
    """The median change of the servo's force from one wheel reading to the next while the target moves: the crunch
    of a servo whose gains amplify the reading's jitter (the R3 on 2026-11-09 felt it at 148 and 241, not at 42 and 85)."""
    sent, samples = run["sent"], run["samples"]
    times = [s[0] for s in sent]
    moving = [s for s in samples if abs(sent[max(0, bisect_right(times, s[0]) - 1)][2]) > 0.0]
    steps = sorted(abs(b[4] - a[4]) for a, b in zip(moving, moving[1:]))
    return steps[len(steps) // 2] if steps else 0.0


def evaluate(params: WheelParams, design: Design, frame_hz: float, track_max: float = RAMP_TRACK_MAX,
             chatter_max: float | None = None) -> dict:
    bench = sim_bench(params, frame_hz)
    bench.say = lambda _msg: None
    bench.ready()
    out = {"ok": True, "over": 0.0, "rise_s": None, "lag_s": None, "chatter": None}
    stages = (STICKY_STAGES if sticky(params) else AHEAD_STAGES) if design.pushes_ahead else STAGES
    if chatter_max is not None:
        out["chatter"] = chatter(bench.run_stage(CHATTER_STAGE, design, params.sign))
        if out["chatter"] > chatter_max:
            out["ok"] = False
            return out
    for stage in stages:
        run = bench.run_stage(stage, design, params.sign)
        s = summarize(run)
        if run.get("aborted") or s["buzz_min"] < MIN_BUZZ:
            out["ok"] = False
        if "up" in s:
            out["over"] = max(out["over"], s["up"]["overshoot"], s["swing"]["overshoot"])
            out["rise_s"] = s["up"]["rise_s"]
        elif stage.kind == "follow":
            out["lag_s"] = s["track"]["lag_s"]
        elif (s["track"]["rms"] or 0.0) > track_max:
            out["ok"] = False
    return out


def holds(plants: list[WheelParams], design: Design, frame_hz: float,
          chatter_max: float | None = None) -> tuple[bool, list[dict]]:
    """Whether a design holds up on the fitted rim (first) and its neighbours; stops at the first that it fails.

    On a sticky rim the neighbours may track worse (NEIGHBOUR_TRACK_MAX): they are there to catch rings, and the old
    wheel's model tracked the fast ramp 18 to 24 mlock rms on the rim with hands and the one with half the friction.
    """
    res: list[dict] = []
    loose = sticky(plants[0])
    for i, plant in enumerate(plants):
        res.append(evaluate(plant, design, frame_hz, NEIGHBOUR_TRACK_MAX if i and loose else RAMP_TRACK_MAX,
                            chatter_max if i == 0 else None))
        limit = NOMINAL_OVERSHOOT if i == 0 else MAX_OVERSHOOT
        if not res[-1]["ok"] or res[-1]["over"] > limit:
            return False, res
    return True, res


def choose_design(params: WheelParams, frame_hz: float = 120.0, bandwidths=BANDWIDTHS, dampings=DAMPINGS,
                  scales=INERTIA_SCALES, ahead: bool = False, breakaway: bool = False,
                  extra: tuple[WheelParams, ...] = (), measured: bool = False, friction_share: float = 1.0,
                  chatter_max: float | None = None) -> Choice:
    """The fastest design that holds up on the fitted rim and its neighbours, searched over the design inertia too.

    ``ahead`` designs it with the push ahead on the plan (the plugin's layout 5 with status bit 1024), from the fit.
    ``breakaway`` adds the breakaway push (status bit 2048) on a sticky rim, searched over BREAKAWAY_SHARES. ``extra``
    are other rims that matched the wheel as well: the design must hold up on them too.
    """
    plants = neighbours(params, ahead, measured) + list(extra)
    shares = BREAKAWAY_SHARES if ahead and breakaway and sticky(params) else (0.0,)
    pushes = [push_ahead_of(params, friction_share, s) if ahead else {} for s in shares]
    best = None
    tried: set[float] = set()
    for scale in scales:
        inertia = max(50.0, min(INERTIA_MAX if scale > 1.0 else 20000.0, params.inertia * scale))
        if inertia in tried:
            continue
        tried.add(inertia)
        for bw in bandwidths:
            held = False
            for z in dampings:
                for push in pushes:
                    design = Design(bandwidth_hz=bw, damping=z, inertia=inertia, damping_first=True, **push)
                    ok, res = holds(plants, design, frame_hz, chatter_max)
                    if not ok:
                        continue
                    held = True
                    score = (res[0]["rise_s"] or 1.0) + 2.0 * (res[0]["lag_s"] or 0.5)
                    if best is None or score < best[0]:
                        best = (score, Choice(inertia, bw, z, res[0]["rise_s"], res[0]["lag_s"],
                                              max(r["over"] for r in res), **push))
            if best is not None and not held:
                break
    if best is None:
        return Choice(max(50.0, min(20000.0, params.inertia)), bandwidths[0], dampings[-1], None, None, 1.0, limited=True)
    return best[1]


def fast_design(params: WheelParams, push: dict, frame_hz: float, chatter_max: float | None,
                bandwidths=BANDWIDTHS, damping: float = 0.7, kp_floor: float = 0.0,
                road_spring: float = 0.0) -> tuple[Choice, float | None]:
    """The fastest bandwidth, designed on the rim's own inertia, whose force chatter stays under ``chatter_max`` and
    that holds on the rim and its neighbours; seconds where ``choose_design`` takes minutes. Also its chatter.
    The full search's stiffer design inertias crunched on the R3 (2026-11-09); ``kp_floor`` raises the design
    inertia only as far as a sticky wheel needs to move a small step past its breakaway.

    With ``road_spring`` the chatter is the design's at that much of the game's centring (``road.chatter``), where the
    servo works, not standing: standing it is the whole stiffness, on the road a share of it.
    """
    plants = neighbours(params, ahead=True, measured=True)

    def inertia_at(bw: float) -> float:
        return min(INERTIA_SCALES[-1] * params.inertia, max(params.inertia, kp_floor / (2.0 * math.pi * bw) ** 2))

    for bw in sorted(bandwidths, reverse=True):
        design = Design(bandwidth_hz=bw, damping=damping, inertia=inertia_at(bw), damping_first=True, **push)
        ok, res = holds(plants, design, frame_hz, None if road_spring > 0.0 else chatter_max)
        if not ok:
            continue
        felt = res[0]["chatter"]
        if road_spring > 0.0 and chatter_max is not None:
            from tools.la_bench.road import chatter as road_chatter

            felt = road_chatter(params, design, frame_hz, road_spring)
            if felt > chatter_max:
                continue
        return Choice(design.inertia, bw, damping, res[0]["rise_s"], res[0]["lag_s"], max(r["over"] for r in res),
                      **push), felt
    low = min(bandwidths)
    return Choice(inertia_at(low), low, damping, None, None, 1.0, limited=True, **push), None


def smoothness(runs: list[dict], stick: float | None = None) -> dict:
    """How steppy the wheel is: breakaway force (the fitted dry friction) against what the servo can push, the wheel
    standing still while the target moves slowly, the size of its smallest steps, slow tracking.

    The R3 on 2026-11-05 read 6 % breakaway, 2 to 3 % still, steps of 0.00002 and 0.002 to 0.003 of error: smooth.
    """
    slow = [r for r in runs if (r["stage"]["kind"] == "ramp" and r["stage"]["param"] <= 0.05)
            or (r["stage"]["kind"] == "follow" and r["stage"]["param"] <= 0.4)]
    moving = still = 0.0
    for r in slow:
        s, times = r["samples"], [x[0] for x in r["sent"]]
        for a, b in zip(s, s[1:]):
            dt = b[0] - a[0]
            if dt > 0 and abs(aim_at(r["sent"], times, b[0]) - aim_at(r["sent"], times, a[0])) / dt > MOVING_RATE:
                moving += dt
                still += dt if abs(b[1] - a[1]) / dt < STILL_RATE else 0.0
    deltas = sorted(abs(b[1] - a[1]) for r in runs for a, b in zip(r["samples"], r["samples"][1:]) if b[1] != a[1])
    track = [summarize(r)["track"]["rms"] for r in slow if r["stage"]["kind"] == "ramp"]
    out = {"stuck": still / moving if moving else None, "resolution": deltas[len(deltas) // 50] if deltas else None,
           "track_rms": sum(track) / len(track) if track else None,
           "friction_share": stick / SERVO_CAP if stick is not None else None}
    why = []
    if out["friction_share"] is not None and out["friction_share"] > FRICTION_MAX:
        why.append(f"it takes {out['friction_share']:.0%} of the servo's push to get the rim moving")
    if out["stuck"] is not None and out["stuck"] > STUCK_MAX:
        why.append(f"the wheel stands still {out['stuck']:.0%} of the time the target moves slowly")
    if out["resolution"] is not None and out["resolution"] > RESOLUTION_MAX:
        why.append(f"the angle reads in steps of {out['resolution']:.4f} lock")
    if out["track_rms"] is not None and out["track_rms"] > TRACK_MAX:
        why.append(f"slow tracking is off by {out['track_rms']:.4f} lock rms")
    known = out["stuck"] is not None or out["friction_share"] is not None
    out["verdict"] = "steppy" if why else ("smooth" if known else "unknown")
    out["why"] = why
    return out
