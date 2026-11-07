"""The slide probe: open-loop torque on the standing wheel, to measure what starts it and what keeps it sliding.

Closed-loop runs cannot tell those apart on a wheel that stick-slips: on the old wheel (2026-11-06) a rim sliding on
150 and one sliding on 358 agreed with the same runs, because the wheel barely slid in them, and only the first has a
servo design that holds. The probe ramps the torque slowly until the wheel starts (the breakaway), then holds a share of
it for PROBE_HOLD_S: still sliding at the end, the sliding friction is below that share; stopped, above it. Both ways
round, the wheel walked home between trials. Open loop, so nothing can ring; the travel is capped.
"""

from __future__ import annotations

from tools.la_bench.bench import FFB_RETRIES, Bench, RunInvalid, reading_trouble

PROBE_RAMP = 400.0
PROBE_MAX = 1500.0
PROBE_START = 0.001
PROBE_HOLD_S = 0.4
PROBE_TAIL_S = 0.15
PROBE_SLIDING = 0.02
PROBE_TRAVEL = 0.12
PROBE_SHARES = (0.3, 0.5, 0.7)
DAMP_LEVELS = (1.05, 1.2, 1.35)
DAMP_HOLD_S = 0.3
DAMP_TAIL_S = 0.12
DAMP_TRAVEL = 0.3
DAMP_MIN_POINTS = 6


def slide_trial(bench: Bench, way: int, share: float, direction: int, home: float) -> dict:
    """One trial ``way`` (+1 counterclockwise): the force that started the wheel, whether ``share`` of it kept it
    sliding, and the readings (time, angle, applied force).
    """
    bench.go_home(home, direction)
    start = bench.rest()
    t0 = bench.clock()
    breakaway = started = None
    samples: list[list[float]] = []
    last = None
    try:
        while True:
            t = bench.clock() - t0
            if breakaway is None:
                want = PROBE_RAMP * t
                if want > PROBE_MAX:
                    break
            else:
                want = share * breakaway
                if t - started >= PROBE_HOLD_S:
                    break
            st = bench.status(torque=way * direction * want)
            if st.steer_frames != last:
                last = st.steer_frames
                samples.append([t, st.steer, st.applied])
                if breakaway is None and (st.steer - start) * way >= PROBE_START:
                    breakaway, started = abs(st.applied), t
                if abs(st.steer - start) > PROBE_TRAVEL:
                    break
            bench.sleep(bench.poll_s)
    finally:
        bench.idle(0.0)
    end = bench.clock() - t0
    tail =[s for s in samples if started is not None and s[0] >= started + PROBE_HOLD_S - PROBE_TAIL_S]
    rate = (tail[-1][1] - tail[0][1]) * way / (tail[-1][0] - tail[0][0]) if len(tail) > 1 and tail[-1][0] > tail[0][0] else 0.0
    far = bool(samples) and abs(samples[-1][1] - start) > PROBE_TRAVEL
    return {"way": way, "share": share, "breakaway": breakaway, "held": share * breakaway if breakaway else None,
            "sliding": breakaway is not None and (far or rate > PROBE_SLIDING), "rate": rate, "start": start,
            "samples": samples, "end": end}


def slide_probe(bench: Bench, direction: int, home: float, say=print) -> dict:
    """Every share both ways, each again when the game's wheel readings stalled in it; the breakaway (median) and
    the bracket the sliding friction lies in."""
    trials = []
    for share in PROBE_SHARES:
        for way in (1, -1):
            for _ in range(FFB_RETRIES):
                tr = slide_trial(bench, way, share, direction, home)
                why = reading_trouble([s[0] for s in tr["samples"]] + [tr["end"]])
                if why is None:
                    break
                say(f"  {why} during the slide probe: once more")
            else:
                raise RunInvalid(f"{why} again and again: keep the game in front and close other programs during "
                                 "the check")
            trials.append(tr)
            say(f"  slide probe {'left' if way > 0 else 'right'}: " + (
                f"started at {tr['breakaway']:.0f}, at {tr['held']:.0f} it "
                + ("kept sliding" if tr["sliding"] else "stopped") if tr["breakaway"] else "did not start"))
    return summarize_probe(trials)


def summarize_probe(trials: list[dict]) -> dict:
    """The breakaway (the median start) and the sliding friction's bracket: above every hold that stopped the wheel,
    below every hold it kept sliding at (``None`` where no trial says). Where a hold stopped it above one it kept
    sliding at (one way slides on less, 18:01 on the old wheel) the bracket spans both and ``mixed`` says so.
    """
    starts = sorted(t["breakaway"] for t in trials if t["breakaway"])
    kept = [t["held"] for t in trials if t["breakaway"] and t["sliding"]]
    stopped = [t["held"] for t in trials if t["breakaway"] and not t["sliding"]]
    low, high = (max(stopped) if stopped else 0.0), (min(kept) if kept else None)
    mixed = high is not None and low > high
    if mixed:
        low, high = high, low
    return {"breakaway": starts[len(starts) // 2] if starts else 0.0, "slide_low": low, "slide_high": high,
            "mixed": mixed, "start_min": starts[0] if starts else 0.0, "start_max": starts[-1] if starts else 0.0,
            "trials": trials}


def start_spread(probe: dict) -> float:
    """How far one start strays from the next, as a share either side of the middle (19:16: 348 to 516, 0.19)."""
    low, high = probe.get("start_min") or 0.0, probe.get("start_max") or 0.0
    return (high - low) / (high + low) if high > low > 0.0 else 0.0


def slide_estimate(probe: dict) -> float | None:
    """The sliding friction to design on: the middle of the bracket, its bottom when no hold kept the wheel sliding
    (the old wheel's 17:16 run: stopped at every hold up to 298), three quarters of its top when none stopped it.
    """
    low, high = probe.get("slide_low") or 0.0, probe.get("slide_high")
    if not probe.get("breakaway"):
        return None
    if high is None:
        return low or None
    return 0.5 * (low + high) if low else 0.75 * high


def bracket_frictions(probe: dict) -> list[float]:
    """Sliding frictions for the model to choose from, inside the probe's bracket (none without a start)."""
    low, high, start = probe.get("slide_low") or 0.0, probe.get("slide_high"), probe.get("breakaway") or 0.0
    if not start:
        return []
    high = start if high is None else high
    low = 0.5 * high if not low else min(low, high)
    return sorted({round(low), round(0.5 * (low + high)), round(high)})


def damping_trial(bench: Bench, way: int, direction: int, home: float) -> dict:
    """Start the wheel ``way`` on the probe's ramp, then hold DAMP_LEVELS of that start for DAMP_HOLD_S each: one
    steady slide per level, (rate, angle, push) over its last DAMP_TAIL_S, counterclockwise, the travel capped.
    """
    bench.go_home(home, direction)
    start = bench.rest()
    t0 = bench.clock()
    breakaway = started = None
    samples: list[list[float]] = []
    last = None
    limit = abs(start) + DAMP_TRAVEL + 0.05
    try:
        while True:
            t = bench.clock() - t0
            if breakaway is None:
                want = PROBE_RAMP * t
                if want > PROBE_MAX:
                    break
            else:
                level = int((t - started) / DAMP_HOLD_S)
                if level >= len(DAMP_LEVELS):
                    break
                want = DAMP_LEVELS[level] * breakaway
            st = bench.status(torque=way * direction * want, limit=max(limit, 0.4))
            if st.steer_frames != last:
                last = st.steer_frames
                samples.append([t, st.steer, st.applied * direction])
                if breakaway is None and (st.steer - start) * way >= PROBE_START:
                    breakaway, started = abs(st.applied), t
                if abs(st.steer - start) > DAMP_TRAVEL:
                    break
            bench.sleep(bench.poll_s)
    finally:
        bench.idle(0.0)
    end = bench.clock() - t0
    points = []
    for i in range(len(DAMP_LEVELS)):
        if started is None:
            break
        hi = started + (i + 1) * DAMP_HOLD_S
        win = [s for s in samples if hi - DAMP_TAIL_S <= s[0] <= hi]
        if len(win) < 3 or win[-1][0] - win[0][0] < 0.5 * DAMP_TAIL_S or win[-1][0] < hi - 0.02:
            continue
        rate = (win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0])
        if rate * way > 0.0:
            points.append([rate, sum(s[1] for s in win) / len(win), sum(s[2] for s in win) / len(win)])
    return {"way": way, "breakaway": breakaway, "points": points, "samples": samples, "end": end}


def damping_probe(bench: Bench, direction: int, home: float, say=print) -> dict:
    """Both ways, again when the game's readings stalled; the slides fitted as push = damping * rate + spring * angle +
    friction the way it moves. The old wheel's closed-loop fits read its damping anywhere from 250 to 2473, and a
    push ahead holds it only within 0.7 to 1.2 times the real one (on its model, 2026-11-06)."""
    trials = []
    for way in (1, -1):
        for _ in range(FFB_RETRIES):
            tr = damping_trial(bench, way, direction, home)
            why = reading_trouble([s[0] for s in tr["samples"]] + [tr["end"]])
            if why is None:
                break
            say(f"  {why} during the damping probe: once more")
        else:
            raise RunInvalid(f"{why} again and again: keep the game in front and close other programs during the check")
        trials.append(tr)
    out = {"trials": trials, **fit_slides([p for tr in trials for p in tr["points"]])}
    if out["damping"]:
        say(f"  damping probe: damping {out['damping']:.0f}, spring {out['spring']:.0f}, sliding on {out['friction']:.0f}")
    else:
        say("  damping probe: too few steady slides to read the damping")
    return out


def fit_slides(points: list[list[float]]) -> dict:
    """Least squares of push on (rate, angle, its sign) over steady slides: damping, spring and sliding friction, all
    0 with fewer than DAMP_MIN_POINTS or a fit that reads negative."""
    none = {"damping": 0.0, "spring": 0.0, "friction": 0.0}
    if len(points) < DAMP_MIN_POINTS or len({p[0] > 0.0 for p in points}) < 2:
        return none
    rows = [(v, x, 1.0 if v > 0.0 else -1.0) for v, x, _ in points]
    a = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    b = [sum(r[i] * p[2] for r, p in zip(rows, points)) for i in range(3)]
    sol = _solve3(a, b)
    if sol is None or sol[0] <= 0.0 or sol[2] <= 0.0:
        return none
    return {"damping": sol[0], "spring": max(0.0, sol[1]), "friction": sol[2]}


def _solve3(a: list[list[float]], b: list[float]) -> list[float] | None:
    m = [row[:] + [v] for row, v in zip(a, b)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(3):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][3] / m[i][i] for i in range(3)]
