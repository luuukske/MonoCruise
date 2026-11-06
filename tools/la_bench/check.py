"""The wheel check: gentle designs on the standing wheel in a ladder, a fit, a design for that rim, a verification.

The ladder climbs the bandwidth of a damping-first design until one rings, so the check never asks more of the wheel
than the rung before it held. The rim is fitted on those runs, a design is chosen for it in the model, and the design
is then run on the wheel: if it rings, that run goes back into the fit (a ring is what pins the delay) and a new design is
chosen, up to three rounds. The result is what `la_*` keys in config.json should hold, how responsive the wheel is and how
smooth.
"""

from __future__ import annotations

from dataclasses import asdict, replace

from tools.la_bench.bench import Bench, BenchAbort
from tools.la_bench.design import Choice, choose_design, push_ahead_of, smoothness
from tools.la_bench.metrics import describe, summarize
from tools.la_bench.plant import WheelParams, fit
from tools.la_bench.protocol import Design, follow, ramp, step

LADDER = (0.7, 1.0, 1.4, 2.0)
RUNG_CAP = 2000
RUNG_STEPS = (step(0.01), step(0.02))
SLOW = (ramp(0.06, 0.05), follow(0.03, 0.4))
VERIFY = (step(0.02),) + SLOW + (ramp(0.08, 0.2), follow(0.03, 0.7))
VERIFY_AHEAD = VERIFY + (ramp(0.15, 0.4, 0.1),)
RING_OVERSHOOT = 1.0
RING_RIPPLE = 0.01
RING_BUZZ = 0.6
RING_TRACK = 0.02
ACCEPT_UP = 0.2
ACCEPT_SWING = 0.35
RETRY_SCALE = 0.8
ROUNDS = 3
REFINE_SCALE = 1.15
REFINE_STEPS = 4
BANDWIDTH_MAX = 3.0
WHEEL_LADDER = (0.5, 0.7, 1.0, 1.4)
WHEEL_LADDER_INERTIA = 650.0
WHEEL_LADDER_DAMPING = 1.0


def rings(s: dict) -> bool:
    """A run that aborted, cut the servo with the buzz guard, or swung well past its target or its tracking."""
    if s.get("aborted") or s["buzz_min"] < RING_BUZZ:
        return True
    for key in ("up", "swing"):
        if key in s and (s[key]["overshoot"] > RING_OVERSHOOT or (s[key]["ripple"] or 0.0) > RING_RIPPLE):
            return True
    return "track" in s and (s["track"]["rms"] or 0.0) > RING_TRACK


def accepted(s: dict, margin: float = 1.0) -> bool:
    """Not ringing and within the overshoot limits, scaled by ``margin`` (below 1: stricter)."""
    if rings(s):
        return False
    return "up" not in s or (s["up"]["overshoot"] <= ACCEPT_UP * margin and s["swing"]["overshoot"] <= ACCEPT_SWING * margin)


def _climb(bench: Bench, direction: int, home: float, hz: float, say, runs: list[dict]) -> float | None:
    """The ladder; the bandwidth of the last rung that held, None if even the first rang."""
    clean = None
    for bw in LADDER:
        design = Design(bandwidth_hz=bw, damping=0.7, inertia=650.0, damping_first=True, cap=RUNG_CAP)
        rang = False
        for stage in RUNG_STEPS + (SLOW if clean is None else ()):
            runs.append(bench.run_stage(stage, design, direction, hz, home))
            s = summarize(runs[-1])
            say("  " + describe(s))
            if rings(s):
                rang = True
                break
        if rang:
            break
        clean = bw
    return clean


def _verify_once(bench: Bench, choice: Choice, direction: int, home: float, hz: float, say,
                 runs: list[dict], margin: float = 1.0) -> tuple[list[dict], bool]:
    """Run a design on the wheel until a stage rings or it has held through all of them; the runs of this attempt."""
    design = choice.design()
    say(f"verifying {design.label}")
    last: list[dict] = []
    for stage in VERIFY_AHEAD if design.pushes_ahead else VERIFY:
        last.append(bench.run_stage(stage, design, direction, hz, home))
        runs.append(last[-1])
        s = summarize(last[-1])
        say("  " + describe(s))
        if not accepted(s, margin):
            return last, False
    return last, True


def _not_again(choice: Choice, failed: list[Choice]) -> Choice:
    """A design at least as fast as one that rang, with the same shape, goes back 20 % from the slowest that rang."""
    same = [f for f in failed if abs(f.inertia - choice.inertia) < 1.0 and f.damping == choice.damping
            and choice.bandwidth_hz >= f.bandwidth_hz]
    if not same:
        return choice
    return replace(choice, bandwidth_hz=round(min(f.bandwidth_hz for f in same) * RETRY_SCALE, 2))


def _refine(bench: Bench, held: Choice, last_runs: list[dict], failed: list[Choice], direction: int, home: float,
            hz: float, say, runs: list[dict]) -> tuple[Choice, list[dict]]:
    """Try the held design faster, 15 % of bandwidth at a time, keeping the fastest the wheel accepts."""
    for _ in range(REFINE_STEPS):
        faster = replace(held, bandwidth_hz=round(held.bandwidth_hz * REFINE_SCALE, 2))
        known = any(abs(f.inertia - faster.inertia) < 1.0 and f.damping == faster.damping
                    and f.bandwidth_hz <= faster.bandwidth_hz for f in failed)
        if faster.bandwidth_hz > BANDWIDTH_MAX or known:
            break
        say("trying it faster")
        tried, good = _verify_once(bench, faster, direction, home, hz, say, runs)
        if not good:
            failed.append(faster)
            break
        held, last_runs = faster, tried
    return held, last_runs


def _wheel_ladder(bench: Bench, push: dict, direction: int, home: float, hz: float, say, runs: list[dict],
                  failed: list[Choice]) -> tuple[Choice | None, list[dict]]:
    """Push-ahead designs tried on the wheel itself, gentlest first, keeping the fastest that holds.

    For a wheel the fit cannot describe (gear teeth, backlash, a coarse encoder): on the geared model wheel the fit read
    inertia 159 and 67 ms where it is 600 and 25, no design held in the model, and 1.0 Hz with the push ahead held on
    the wheel. The wheel decides, so this works on any wheel the push ahead can move.
    """
    held, last = None, []
    for bw in WHEEL_LADDER:
        cand = Choice(WHEEL_LADDER_INERTIA, bw, WHEEL_LADDER_DAMPING, None, None, 1.0, False, **push)
        tried, good = _verify_once(bench, cand, direction, home, hz, say, runs)
        if not good:
            failed.append(cand)
            break
        held, last = cand, tried
    return held, last


def run_check(bench: Bench, direction: int, home: float, hz: float = 20.0, evals: int = 500, say=print,
              grid: dict | None = None) -> dict:
    """The whole check. A wheel that cannot be brought back to its start ends it early with what was recorded."""
    runs: list[dict] = []
    notes: list[str] = []
    clean = result = choice = held = None
    last_runs: list[dict] = []
    failed: list[Choice] = []
    ahead = bench.push_ahead and bench.ff_plan
    reading = round(bench.reading_hz) if bench.reading_hz > 0.0 else 0.0
    try:
        say(f"the game reads the wheel {reading:.0f} times a second" if reading else "wheel reading rate unknown")
        if not ahead:
            notes.append("the plugin cannot push ahead on a plan (layout 5, status 1024): designed without it")
        say("climbing: gentle damping-first designs, stopping at the first that rings")
        clean = _climb(bench, direction, home, hz, say, runs)
        if clean is None:
            notes.append(f"even {LADDER[0]} Hz rang or hunted: a very light or loose wheel, or stepping and backlash")
        for turn in range(ROUNDS):
            say("fitting the rim" + (" again, with the run that rang" if turn else ""))
            result = fit(runs, bench.frame_hz, WheelParams(sign=direction, reading_hz=reading), max_evals=evals)
            say("choosing a design for it" + (", with the push ahead" if ahead else ""))
            choice = _not_again(choose_design(result.params, bench.frame_hz, ahead=ahead, **(grid or {})), failed)
            if choice.limited:
                notes.append("no design in the grid held up on the fitted rim and its neighbours")
                if ahead:
                    break
            last_runs, good = _verify_once(bench, choice, direction, home, hz, say, runs)
            if good:
                held = choice
                break
            failed.append(choice)
        if held is None and ahead and result is not None:
            say("the model's designs did not hold, trying push-ahead designs on the wheel, gentlest first")
            held, last_runs = _wheel_ladder(bench, push_ahead_of(result.params), direction, home, hz, say, runs, failed)
            if held is not None:
                notes.append("chosen on the wheel, not in the model: the fit does not describe this wheel well")
        if held is None:
            say("the model's designs rang on the wheel, trying the gentlest rung")
            gentle = Choice(650.0, LADDER[0], 0.7, None, None, 1.0, True)
            last_runs, good = _verify_once(bench, gentle, direction, home, hz, say, runs)
            held = gentle if good else None
        elif held is not None and not held.limited and held.rise_s is not None:
            held, last_runs = _refine(bench, held, last_runs, failed, direction, home, hz, say, runs)
        if failed:
            notes.append(f"{len(failed)} design(s) rang on the wheel" + ("" if held else ", none held"))
    except BenchAbort as exc:
        notes.append(f"stopped: {exc}")
        held = None
    if held is None:
        notes.append("no design held on this wheel: leave the la_* keys at their defaults")
    params = result.params if result else None
    step_run = next((summarize(r) for r in last_runs if r["name"] == "step 0.02" and not r.get("aborted")), None)
    return {
        "runs": runs, "wheel": asdict(params) if params else {}, "fit_rms": result.rms if result else None,
        "choice": asdict(held or choice) if (held or choice) else {}, "verified": held is not None,
        "ladder_clean_hz": clean, "smoothness": smoothness(last_runs if held else runs, params.stick if params else None),
        "step_rise_s": step_run["up"]["rise_s"] if step_run and held else None,
        "step_overshoot": step_run["up"]["overshoot"] if step_run and held else None,
        "config": held.config() if held else {}, "notes": notes, "reading_hz": reading,
    }


def summary(data: dict) -> str:
    """The result as text: responsiveness, smoothness and the config.json keys."""
    w, c, m = data["wheel"], data["choice"], data["smoothness"]
    rise = data["step_rise_s"]
    lines = ["Wheel check"]
    if w:
        lines.append(f"  rim: inertia {w['inertia']:.0f}, damping {w['damping']:.0f}, dry friction {w['stick']:.0f}, "
                     f"delay {w['delay_s'] * 1000:.0f} ms" + (f", read {data['reading_hz']:.0f} times a second"
                                                           if data.get("reading_hz") else ""))
    else:
        lines.append("  rim: not measured")
    if data["verified"]:
        lines.append(f"  responsiveness: {c['bandwidth_hz']:g} Hz design" + (" with the push ahead" if c.get("ff_damping")
                                                                           or c.get("ff_inertia") else "")
                     + ", verified on the wheel"
                     + (f", a 0.02 step rises in {rise:.2f} s" if rise else ""))
    else:
        lines.append("  responsiveness: no design held on the wheel")
    lines.append(f"  smoothness: {m['verdict']}" + (" (" + "; ".join(m["why"]) + ")" if m["why"] else ""))
    lines += [f"  note: {n}" for n in data["notes"]]
    if data["config"]:
        lines.append("  config.json:")
        lines += [f"    {k!r}: {str(v).lower() if isinstance(v, bool) else v}," for k, v in data["config"].items()]
    else:
        lines.append("  config.json: leave the la_* keys at their defaults")
    return "\n".join(lines) + "\n"
