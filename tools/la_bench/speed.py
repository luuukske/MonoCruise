"""The speed benchmark: how fast the servo turns the standing wheel to a new angle, and how fast a target it follows.

Ramps get faster and steps bigger until the wheel falls behind, overshoots or rings. A ramp reaches its rate over
``corner`` seconds at most (half its move at the fastest), as a commanding feature that smooths its target would send
it, with the acceleration, so the plugin can push the rim ahead. ``quickest`` is what the rim
itself allows: the shortest move the fitted inertia, damping and friction permit at the servo's cap and the plugin's
force slew, so the gap to the measured moves is what the servo leaves on the table. Last comes half lock: out from
the centre to one side and back, then to the other, at faster rates and then at once. Standing only: at speed the
game's centring and damping slow every number here.
"""

from __future__ import annotations

from tests.la.servo_sim import SERVO_SLEW
from tools.la_bench.bench import Bench, BenchAbort
from tools.la_bench.check import RING_BUZZ, RING_OVERSHOOT, rings
from tools.la_bench.metrics import describe, step_response, summarize, tracking
from tools.la_bench.plant import WheelParams
from tools.la_bench.protocol import Design, Stage, out, ramp, step

RAMP_AMP = 0.15
RAMP_RATES = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
STEP_SIZES = (0.02, 0.05, 0.1, 0.2)
FOLLOW_LAG_S = 0.1
FOLLOW_OVER = 0.1
STEP_OVER = 0.2
RAMP_CORNER_S = 0.1
KEYS_MIN_RATE = 0.4
HALF_LOCK = 0.5
HALF_RATES = (0.5, 1.0, 2.0)
HALF_ACCEL = 8.0
ACCEL_SHARE = 0.7
STILL = 1e-4


def speed_ramp(rate: float, corner_s: float = RAMP_CORNER_S, amp: float = RAMP_AMP):
    """A ramp of ``amp`` at ``rate`` whose corners take ``corner_s`` or half the move, whichever is shorter."""
    return ramp(amp, rate, min(corner_s, 0.5 * amp / rate) if corner_s > 0.0 else 0.0)


def half_stage(rate: float, corner_s: float = RAMP_CORNER_S, amp: float = HALF_LOCK,
               accel: float = HALF_ACCEL) -> Stage:
    """Half lock out and back each way at ``rate`` (0: at once). It speeds up and slows down at most at ``accel``
    (lock/s^2), what a commanding feature that knows the rim would ask, over at least ``corner_s``.
    """
    if rate <= 0.0:
        return out(amp)
    corner = max(corner_s, rate / accel) if accel > 0.0 else corner_s
    return out(amp, rate, corner)


def rim_accel(rim: WheelParams | None, cap: float) -> float:
    """The acceleration a half-lock ramp may ask: ``ACCEL_SHARE`` of what the rim does at the cap from rest."""
    if rim is None or rim.inertia <= 0.0 or cap <= rim.stick:
        return HALF_ACCEL
    return ACCEL_SHARE * (cap - rim.stick) / rim.inertia


def half_result(run: dict) -> dict:
    """Each of the four moves (out, back, out the other way, back): time to 90 % and overshoot; the wheel's peak rate.

    A half-lock move trails its target by more than a small ramp does, so here only an abort, a buzz cut or a swing
    past the whole move counts as a ring, not the tracking error.
    """
    ts = [s[0] for s in run["samples"]]
    ys = [s[1] for s in run["samples"]]
    stage = Stage.from_dict(run["stage"])
    segs = stage.segments()
    start = run["start"]
    t90s, overs = [], []
    for i in (1, 3, 5, 7):
        t0, a0, a1 = segs[i][0], start + segs[i][2], start + segs[i][3]
        end = segs[i + 1][1]
        overs.append(step_response(ts, ys, t0, a0, a1, end - t0)["overshoot"])
        t90s.append(time_to(ts, ys, t0, a0, a1))
    s = summarize(run)
    over = max(overs) if overs else 0.0
    rang = bool(s.get("aborted")) or s["buzz_min"] < RING_BUZZ or over > RING_OVERSHOOT
    lag = tracking(ts, ys, run["sent"])["lag_s"] if ts and stage.param > 0.0 else None
    followed = stage.param > 0.0 and not rang and lag is not None and lag <= FOLLOW_LAG_S and over <= FOLLOW_OVER
    return {"rate": stage.param, "amp": stage.amp, "t90": t90s, "overshoot": over, "lag_s": lag, "rang": rang,
            "followed": followed, "peak_rate": max((abs(x[2]) for x in run["samples"]), default=0.0),
            "peak_servo": s["peak_servo"], "aborted": s.get("aborted")}


def time_to(ts: list[float], ys: list[float], t0: float, start: float, goal: float, share: float = 0.9) -> float | None:
    """Seconds from ``t0`` until the wheel first covers ``share`` of the move from ``start`` to ``goal``."""
    size = goal - start
    for t, y in zip(ts, ys):
        if t >= t0 and size and (y - start) / size >= share:
            return t - t0
    return None


def ramp_result(run: dict) -> dict:
    """Lag, overshoot at the end of each move, and the rate the wheel reached (10 to 90 % of the first move)."""
    ts = [s[0] for s in run["samples"]]
    ys = [s[1] for s in run["samples"]]
    stage = Stage.from_dict(run["stage"])
    a, rate, start = stage.amp, stage.param, run["start"]
    segs = stage.segments()
    up = step_response(ts, ys, segs[1][0], start, start + a, segs[2][1] - segs[1][0])
    swing = step_response(ts, ys, segs[3][0], start + a, start - a, segs[4][1] - segs[3][0])
    lag = tracking(ts, ys, run["sent"])["lag_s"] if ts else None
    reached = 0.8 * a / up["rise_s"] if up["rise_s"] else None
    s = summarize(run)
    over = max(up["overshoot"], swing["overshoot"])
    followed = (not s.get("aborted") and not rings(s) and lag is not None and lag <= FOLLOW_LAG_S
                and over <= FOLLOW_OVER)
    return {"rate": rate, "lag_s": lag, "overshoot": over, "reached": reached, "followed": followed,
            "peak_servo": s["peak_servo"], "aborted": s.get("aborted")}


def step_result(run: dict) -> dict:
    """Time to 90 % of the step and of the swing through zero, their overshoot and settling."""
    ts = [x[0] for x in run["samples"]]
    ys = [x[1] for x in run["samples"]]
    s = summarize(run)
    sent = run["sent"]
    a, start = run["stage"]["amp"], run["start"]
    out = {"size": a, "aborted": s.get("aborted"), "peak_servo": s["peak_servo"]}
    if "up" not in s:
        return dict(out, t90=None, swing_t90=None, overshoot=None, settle_s=None, clean=False, rang=True)
    t90 = time_to(ts, ys, sent[1][0], start, start + a)
    swing_t90 = time_to(ts, ys, sent[2][0], start + a, start - a)
    over = max(s["up"]["overshoot"], s["swing"]["overshoot"])
    rang = bool(s.get("aborted")) or rings(s)
    clean = not rang and over <= STEP_OVER
    return dict(out, t90=t90, swing_t90=swing_t90, overshoot=over, settle_s=s["up"]["settle_s"], clean=clean,
                rang=rang)


def _move(p: WheelParams, cap: float, slew: float, switch: float, h: float, t_max: float) -> tuple[float, float]:
    """Full force one way, then from ``switch`` the other, until the rim stops: where and when it stopped."""
    x = v = u = t = 0.0
    while t < t_max:
        goal = cap if t < switch else -cap
        u += max(-slew * h, min(slew * h, goal - u)) if slew > 0.0 else goal - u
        force = u - p.damping * v - p.spring * x
        if abs(v) < STILL and abs(force) <= p.stick:
            v = 0.0
        else:
            force -= p.stick if (v if abs(v) >= STILL else force) > 0.0 else -p.stick
            new = v + force / p.inertia * h
            if t > switch and new <= 0.0:
                return x, t
            v = new
            x += v * h
        t += h
    return x, t


def quickest(p: WheelParams, cap: float, move: float, slew: float = SERVO_SLEW, h: float = 2e-4,
             t_max: float = 3.0) -> float | None:
    """The shortest time to move ``move`` lock and stop at ``cap`` with the force changing at most ``slew`` per second.

    No delay and no game centring: the bang-bang move a perfect servo would make on this rim. None if it cannot.
    """
    if _move(p, cap, slew, t_max, h, t_max)[0] < move:
        return None
    lo, hi = 0.0, t_max
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _move(p, cap, slew, mid, h, t_max)[0] < move:
            lo = mid
        else:
            hi = mid
    return _move(p, cap, slew, hi, h, t_max)[1]


def top_rate(p: WheelParams, cap: float) -> float | None:
    """The steady rate the rim reaches at ``cap`` against its damping and friction, lock per second."""
    return (cap - p.stick) / p.damping if p.damping > 0.0 and cap > p.stick else None


def run_speed(bench: Bench, design: Design, direction: int, home: float, hz: float = 20.0, say=print,
              rates=RAMP_RATES, sizes=STEP_SIZES, corner_s: float = RAMP_CORNER_S, half_rates=HALF_RATES,
              half_at_once: bool = True, half_accel: float = HALF_ACCEL) -> dict:
    """Ramps from slow to fast, then steps from small to big, then half lock; a run that rings ends its own ladder.

    Half lock at once runs only when every half-lock ramp held.
    """
    runs: list[dict] = []
    ramps: list[dict] = []
    steps: list[dict] = []
    half: list[dict] = []
    notes: list[str] = []
    try:
        say(f"ramps of {RAMP_AMP:g} lock, faster each time [{design.label}, cap {design.cap}]")
        for rate in rates:
            runs.append(bench.run_stage(speed_ramp(rate, corner_s), design, direction, hz, home))
            ramps.append(ramp_result(runs[-1]))
            say("  " + describe(summarize(runs[-1])))
            if ramps[-1]["aborted"] or rings(summarize(runs[-1])):
                notes.append(f"ramps stopped at {rate:g} lock/s: the wheel rang")
                break
        say("steps, bigger each time")
        for size in sizes:
            runs.append(bench.run_stage(step(size), design, direction, hz, home))
            steps.append(step_result(runs[-1]))
            say("  " + describe(summarize(runs[-1])))
            if steps[-1]["aborted"] or rings(summarize(runs[-1])):
                notes.append(f"steps stopped at {size:g} lock: the wheel rang")
                break
        if half_rates or half_at_once:
            say(f"half lock ({HALF_LOCK:g}), out and back each way, at most {half_accel:.1f} lock/s^2: hands and "
                "everything else clear of the rim")
        held = True
        for rate in list(half_rates) + ([0.0] if half_at_once else []):
            if rate == 0.0 and not held:
                notes.append("half lock at once skipped: a half-lock ramp rang")
                break
            runs.append(bench.run_stage(half_stage(rate, corner_s, accel=half_accel), design, direction, hz, home))
            half.append(half_result(runs[-1]))
            say("  " + describe(summarize(runs[-1])))
            if half[-1]["rang"]:
                held = False
                notes.append(f"half lock stopped at {rate:g} lock/s: the wheel rang" if rate else
                             "half lock at once rang")
                break
    except BenchAbort as exc:
        notes.append(f"stopped: {exc}")
    followed = [r["rate"] for r in ramps if r["followed"]]
    half_followed = [r["rate"] for r in half if r["followed"]]
    return {"runs": runs, "ramps": ramps, "steps": steps, "half": half, "notes": notes, "half_accel": half_accel,
            "fastest_followed": max(followed) if followed else None,
            "fastest_half": max(half_followed) if half_followed else None}


def _deg(lock: float, rotation: float) -> float:
    return lock * rotation / 2.0


def _s(v: float | None, fmt: str = ".2f") -> str:
    return "never" if v is None else format(v, fmt) + " s"


def summary(data: dict, rim: WheelParams | None, cap: float, rotation: float = 900.0) -> str:
    """The result as text, each step beside what the rim allows at the same cap with and without the force slew."""
    lines = [f"Speed benchmark [{data['label']}, cap {cap:.0f}]"]
    if rim is not None:
        lines.append(f"  rim: inertia {rim.inertia:.0f}, damping {rim.damping:.0f}, dry friction {rim.stick:.0f}")
    lines.append(f"  ramps of {RAMP_AMP:g} lock ({_deg(RAMP_AMP, rotation):.0f} deg on a {rotation:.0f} deg wheel):")
    for r in data["ramps"]:
        reached = f"{r['reached']:.2f}/s" if r["reached"] else "never"
        verdict = "followed" if r["followed"] else ("ABORTED" if r["aborted"] else "not followed")
        lines.append(f"    {r['rate']:g}/s: {verdict}, lag {_s(r['lag_s'], '.3f')}, overshoot {r['overshoot'] * 100:.0f} %, "
                     f"wheel reached {reached}, servo peak {r['peak_servo']:.0f}")
    fast = data["fastest_followed"]
    lines.append("  fastest ramp followed: " + (f"{fast:g} lock/s ({_deg(fast, rotation):.0f} deg/s)" if fast else "none"))
    lines.append("  steps (time to 90 %, then the swing through zero, twice the size):")
    for s in data["steps"]:
        best = quickest(rim, cap, 0.9 * s["size"]) if rim else None
        free = quickest(rim, cap, 0.9 * s["size"], slew=0.0) if rim else None
        rim_txt = f" | rim allows {_s(best, '.3f')}, {_s(free, '.3f')} without the force slew" if rim else ""
        over = f"{s['overshoot'] * 100:.0f} %" if s["overshoot"] is not None else "-"
        verdict = "" if s["clean"] else (" ABORTED" if s["aborted"] else " not clean")
        lines.append(f"    {s['size']:g} ({_deg(s['size'], rotation):.0f} deg): {_s(s['t90'], '.3f')}, swing "
                     f"{_s(s['swing_t90'], '.3f')}, overshoot {over}{verdict}{rim_txt}")
    if data.get("half"):
        amp = data["half"][0]["amp"]
        lines.append(f"  half lock ({amp:g} lock, {_deg(amp, rotation):.0f} deg), out, back, out the other way, back "
                     f"(time to 90 % each), ramps at most {data.get('half_accel', HALF_ACCEL):.1f} lock/s^2:")
        for h in data["half"]:
            times = ", ".join(_s(t, ".2f") for t in h["t90"])
            what = f"{h['rate']:g}/s" if h["rate"] else "at once"
            verdict = "rang" if h["rang"] else ("followed" if h["followed"] else ("held" if not h["rate"] else
                                                                                  "not followed"))
            extra = f", lag {_s(h['lag_s'], '.3f')}" if h["rate"] else ""
            ceiling = quickest(rim, cap, 0.9 * amp) if rim and not h["rate"] else None
            rim_txt = f" | rim allows {_s(ceiling, '.2f')}" if ceiling else ""
            lines.append(f"    {what}: {verdict}{extra}, {times}, overshoot {h['overshoot'] * 100:.0f} %, wheel peak "
                         f"{h['peak_rate']:.2f}/s ({_deg(h['peak_rate'], rotation):.0f} deg/s){rim_txt}")
        fast = data.get("fastest_half")
        lines.append("  fastest half lock followed: " + (f"{fast:g} lock/s ({_deg(fast, rotation):.0f} deg/s)" if fast
                                                          else "none"))
        if any(not h["rate"] for h in data["half"]):
            lines.append("    (at once, damping first holds the wheel to about half the cap over kd: a feature that wants "
                         "a fast big move sends it as a ramp)")
    if rim is not None:
        top = top_rate(rim, cap)
        lines.append("  the rim at this cap: top rate " + (f"{top:.1f} lock/s ({_deg(top, rotation):.0f} deg/s), "
                     f"force slew {SERVO_SLEW:.0f}/s takes {cap / SERVO_SLEW:.2f} s to reach the cap" if top else "none"))
    lines += [f"  note: {n}" for n in data["notes"]]
    keys = config_keys(data)
    if keys:
        lines.append("  config.json (this run was clean with them):")
        lines += [f"    {k!r}: {v}," for k, v in keys.items()]
    return "\n".join(lines) + "\n"


def config_keys(data: dict) -> dict:
    """The push ahead and the lead as ``la_*`` keys when the run used them, followed ramps of at least
    ``KEYS_MIN_RATE`` (twice LA's own target limit) and no step rang. A step's overshoot is the feedback's, not theirs.
    """
    d = data.get("design") or {}
    if not (d.get("ff_inertia") or d.get("ff_damping") or d.get("ff_friction") or d.get("lead_s")):
        return {}
    if not data["steps"] or any(s.get("rang", not s["clean"]) for s in data["steps"]):
        return {}
    if (data.get("fastest_followed") or 0.0) < KEYS_MIN_RATE:
        return {}
    out = {"la_ff_inertia": round(d["ff_inertia"]), "la_ff_damping": round(d["ff_damping"]),
           "la_ff_friction": round(d["ff_friction"]), "la_servo_lead_s": round(d["lead_s"], 3),
           "la_ff_plan_s": round(d.get("plan_s", 0.0), 3)}
    if d.get("ff_breakaway", 0.0) > d["ff_friction"] > 0.0:
        out["la_ff_breakaway"] = round(d["ff_breakaway"])
    return out
