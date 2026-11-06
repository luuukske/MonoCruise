"""Pick the servo design for a measured rim, and score how smooth the wheel is.

The design is tried on the fitted rim and on four neighbours of it (more and less delay, a lighter rim, a heavier
rim with hands) so it does not rest on one fit. A design holds up when no stage aborts, the buzz guard stays quiet,
nothing overshoots past ``MAX_OVERSHOOT`` and the fitted rim itself stays under ``NOMINAL_OVERSHOOT``; the fastest one
wins. The design inertia is searched from the fitted one up: on the R3 a design on 650 (stiff, well damped for a rim of
300) was the fastest that held, and one designed on the fitted inertia was half as stiff. Thresholds are first guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tools.la_bench.bench import sim_bench
from tools.la_bench.metrics import aim_at, summarize
from tools.la_bench.plant import WheelParams
from tools.la_bench.protocol import Design, follow, ramp, step

BANDWIDTHS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
DAMPINGS = (0.7, 1.0, 1.4)
MAX_OVERSHOOT = 0.6
NOMINAL_OVERSHOOT = 0.4
INERTIA_SCALES = (1.0, 1.5, 2.0, 3.0)
INERTIA_MAX = 1000.0
MIN_BUZZ = 0.6
STAGES = (step(0.02), follow(0.03, 0.7))
AHEAD_STAGES = (step(0.02), ramp(0.15, 0.4, 0.1), follow(0.03, 0.7))
STUCK_MAX = 0.25
RESOLUTION_MAX = 2e-4
TRACK_MAX = 0.01
FRICTION_MAX = 0.12
SERVO_CAP = 3000.0
RAMP_TRACK_MAX = 0.015
STILL_RATE = 0.005
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

    def design(self) -> Design:
        return Design(bandwidth_hz=self.bandwidth_hz, damping=self.damping, inertia=self.inertia, damping_first=True,
                      ff_inertia=self.ff_inertia, ff_damping=self.ff_damping, ff_friction=self.ff_friction,
                      plan_s=self.plan_s)

    def config(self) -> dict:
        out = {"la_wheel_inertia": round(self.inertia), "la_bandwidth_hz": round(self.bandwidth_hz, 2),
               "la_damping": round(self.damping, 2), "la_damping_first": True}
        if self.ff_inertia or self.ff_damping or self.ff_friction:
            out.update({"la_ff_inertia": round(self.ff_inertia), "la_ff_damping": round(self.ff_damping),
                        "la_ff_friction": round(self.ff_friction), "la_ff_plan_s": round(self.plan_s, 3)})
        return out


def push_ahead_of(params: WheelParams, share: float = 1.0) -> dict:
    """The push ahead for a fitted rim: its inertia, damping and dry friction (``share`` of it) on the plan its delay
    early. Every wheel gets its own from the fit; nothing here is the R3's.
    """
    return {"ff_inertia": params.inertia, "ff_damping": params.damping, "ff_friction": share * params.stick,
            "plan_s": params.delay_s}


def neighbours(p: WheelParams, ahead: bool = False) -> list[WheelParams]:
    """The fitted rim first, then what a design must also hold up on. With the push ahead also a rim with half the
    friction: the push then over-pushes at every stop, which is where a friction push hunts.
    """
    out = [p, replace(p, delay_s=p.delay_s * 1.3), replace(p, delay_s=p.delay_s * 0.8),
           replace(p, inertia=p.inertia * 0.75),
           replace(p, inertia=p.inertia * 1.6, damping=p.damping * 1.3, stick=p.stick * 1.5)]
    return out + [replace(p, stick=p.stick * 0.5, damping=p.damping * 0.8)] if ahead else out


def evaluate(params: WheelParams, design: Design, frame_hz: float) -> dict:
    bench = sim_bench(params, frame_hz)
    bench.say = lambda _msg: None
    bench.ready()
    out = {"ok": True, "over": 0.0, "rise_s": None, "lag_s": None}
    for stage in AHEAD_STAGES if design.pushes_ahead else STAGES:
        run = bench.run_stage(stage, design, params.sign)
        s = summarize(run)
        if run.get("aborted") or s["buzz_min"] < MIN_BUZZ:
            out["ok"] = False
        if "up" in s:
            out["over"] = max(out["over"], s["up"]["overshoot"], s["swing"]["overshoot"])
            out["rise_s"] = s["up"]["rise_s"]
        elif stage.kind == "follow":
            out["lag_s"] = s["track"]["lag_s"]
        elif (s["track"]["rms"] or 0.0) > RAMP_TRACK_MAX:
            out["ok"] = False
    return out


def holds(plants: list[WheelParams], design: Design, frame_hz: float) -> tuple[bool, list[dict]]:
    """Whether a design holds up on the fitted rim (first) and its neighbours; stops at the first that it fails."""
    res: list[dict] = []
    for i, plant in enumerate(plants):
        res.append(evaluate(plant, design, frame_hz))
        limit = NOMINAL_OVERSHOOT if i == 0 else MAX_OVERSHOOT
        if not res[-1]["ok"] or res[-1]["over"] > limit:
            return False, res
    return True, res


def choose_design(params: WheelParams, frame_hz: float = 120.0, bandwidths=BANDWIDTHS, dampings=DAMPINGS,
                  scales=INERTIA_SCALES, ahead: bool = False) -> Choice:
    """The fastest design that holds up on the fitted rim and its neighbours, searched over the design inertia too.

    ``ahead`` designs it with the push ahead on the plan (the plugin's layout 5 with status bit 1024), from the fit.
    """
    plants = neighbours(params, ahead)
    push = push_ahead_of(params) if ahead else {}
    best = None
    for scale in scales:
        inertia = max(50.0, min(INERTIA_MAX if scale > 1.0 else 20000.0, params.inertia * scale))
        for bw in bandwidths:
            held = False
            for z in dampings:
                design = Design(bandwidth_hz=bw, damping=z, inertia=inertia, damping_first=True, **push)
                ok, res = holds(plants, design, frame_hz)
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
