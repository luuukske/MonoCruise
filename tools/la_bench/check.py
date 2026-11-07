"""The wheel check: gentle designs on the standing wheel in a ladder, a fit, a design for that rim, a verification.

The ladder climbs the bandwidth of a damping-first design until one rings, so the check never asks more of the wheel
than the rung before it held; a wheel that rings on the first rung climbs again on a lighter design. The rim is fitted
on those runs with the breakaway and the angle's steps read off the wheel, the fit is held to the wheel's own verdicts
(which runs rang), a design is chosen for it in the model, and the design is then run on the wheel: if it rings, that
run goes back into the fit (a ring is what pins the delay) and a new design is chosen, up to three rounds. A model that
disagrees with the wheel is not trusted: designs are then tried on the wheel itself. The result is what `la_*` keys in
config.json should hold, how responsive the wheel is and how smooth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from itertools import product

from tools.la_bench.bench import Bench, BenchAbort, RunInvalid, sim_bench
from tools.la_bench.design import Choice, choose_design, push_ahead_of, smoothness, sticky
from tools.la_bench.metrics import describe, summarize
from tools.la_bench.plant import DELAY_MIN_S, WheelParams, breakaway_of, fit, quantum_of
from tools.la_bench.probe import bracket_frictions, damping_probe, slide_estimate, slide_probe, start_spread
from tools.la_bench.protocol import Design, Stage, follow, ramp, step

LADDER = (0.7, 1.0, 1.4, 2.0)
LADDER_INERTIA = 650.0
LADDER_LIGHT = 160.0
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
ACCEPT_SHORT = 0.6
RETRY_SCALE = 0.8
ROUNDS = 3
REFINE_SCALE = 1.15
REFINE_STEPS = 4
BANDWIDTH_MAX = 3.0
WHEEL_LADDER = (0.5, 0.7, 1.0, 1.4)
WHEEL_LADDER_INERTIA = (150.0, 650.0)
WHEEL_LADDER_DAMPING = 1.0
WHEEL_LADDER_BREAKAWAY = 1.2
MATCH_INERTIA = (0.25, 0.5, 1.0, 2.0, 4.0)
MATCH_FRICTION = (0.25, 0.33, 0.5, 0.8)
MATCH_DAMPING = (0.5, 1.0, 2.0)
MATCH_DELAY_S = (-0.015, 0.0, 0.01)
STUCK_PROBE = 0.45
ALIKE_ERR = 1.5
ALIKE_MAX = 2
WHEEL_LADDER_SCALES = (4.0, 2.0)
TRUST_SHARE = 0.8


def rings(s: dict) -> bool:
    """A run that aborted, cut the servo with the buzz guard, or swung well past its target or its tracking."""
    if s.get("aborted") or s["buzz_min"] < RING_BUZZ:
        return True
    for key in ("up", "swing"):
        if key in s and (s[key]["overshoot"] > RING_OVERSHOOT or (s[key]["ripple"] or 0.0) > RING_RIPPLE):
            return True
    return "track" in s and (s["track"]["rms"] or 0.0) > RING_TRACK


def accepted(s: dict, margin: float = 1.0) -> bool:
    """Not ringing, a step that gets 40 % of the way or more, and within the overshoot limits, scaled by ``margin`` (below
    1: stricter). The old wheel's step that stopped 88 % short (push 440 against starts of 350 to 520, 19:16) passed."""
    if rings(s):
        return False
    if "up" not in s:
        return True
    there = s["up"].get("rise_s", 0.0) is not None or s["up"].get("final_err", 0.0) > -ACCEPT_SHORT
    return there and s["up"]["overshoot"] <= ACCEPT_UP * margin and s["swing"]["overshoot"] <= ACCEPT_SWING * margin


def _rerun(params: WheelParams, run: dict, frame_hz: float) -> dict:
    """``run``'s stage and design again on a model wheel, from where it started."""
    bench = sim_bench(params, frame_hz)
    bench.say = lambda _msg: None
    return bench.run_stage(Stage.from_dict(run["stage"]), Design(**run["design"]), params.sign, 20.0, run["start"])


def match_verdicts(params: WheelParams, runs: list[dict], frame_hz: float, frictions: list[float] | None = None,
                   damping_held: bool = False) -> tuple[WheelParams, int, int, list]:
    """The rim, near the fit, that rings on the runs that rang on the wheel and holds on the others; how many agree;
    and the other rims the scan found that agree as well (within ALIKE_ERR of its peak force error).

    The fit matches angle traces, and a run that rang is a limit cycle, whose trace a heavily damped rim matches best:
    on the old wheel (2026-11-06) the fit read inertia 152 and damping 1457 and the model held 2 Hz where the wheel rang
    at 4.3 Hz, while a rim of 40 to 70 agreed on all ten runs. Ties go to the closer peak force, then to the fit.

    A sticky rim, or one the slide probe ran on, is always scanned, inertia and the sliding friction first (a share of
    the breakaway), then damping and delay around
    the best: when the wheel barely moves the fit puts the sliding friction near the breakaway. On 16 runs of the old
    wheel the check's fit (inertia 30, friction 365) agreed on 12, inertia 120 with friction 150 on all 16, yet on the
    second run's six a rim on 358 agreed as well: ``frictions`` (the slide probe's bracket) replace the shares of the
    breakaway then, and the fit's friction moves to the nearest of them. A damping the damping probe measured is kept.
    """
    usable =[r for r in runs if len(r["samples"]) > 10 and summarize(r)["peak_servo"] > 0.0]
    wheel = [(rings(summarize(r)), max(summarize(r)["peak_servo"], 1.0)) for r in usable]
    scan = sticky(params) or bool(frictions)
    if frictions:
        params = replace(params, stick=min(frictions, key=lambda f: abs(f - params.stick)))

    def score(p: WheelParams) -> tuple[int, float]:
        agree, err = 0, 0.0
        for run, (rang, peak) in zip(usable, wheel):
            m = summarize(_rerun(p, run, frame_hz))
            agree += rings(m) == rang
            err += abs(math.log(max(m["peak_servo"], 1.0) / peak))
        return agree, -err

    scored = [(score(params), params)]
    if scored[0][0][0] == len(usable) and not scan:
        return params, scored[0][0][0], len(usable), []
    shares = [params.breakaway * f for f in MATCH_FRICTION if params.breakaway > 0.0]
    for j, f in product(MATCH_INERTIA, list(frictions or [params.stick] + shares)):
        p = replace(params, inertia=max(30.0, params.inertia * j), stick=f)
        if p != params:
            scored.append((score(p), p))
    centre = max(scored, key=lambda x: x[0])[1]
    for ci, dl in product((1.0,) if damping_held else MATCH_DAMPING, MATCH_DELAY_S):
        p = replace(centre, damping=centre.damping * ci, delay_s=max(DELAY_MIN_S, centre.delay_s + dl))
        if p != centre:
            scored.append((score(p), p))
    (agree, err), best = max(scored, key=lambda x: x[0])
    alike = [p for (a, e), p in scored if a == agree and p != best and -e <= -err * ALIKE_ERR]
    return best, agree, len(usable), alike


def design_rim(best: WheelParams, alike: list[WheelParams]) -> tuple[WheelParams, list[WheelParams]]:
    """The rim to design on and the rims the design must also hold up on, from the rims that agree with the wheel.

    The friction pushed ahead is the lowest any of them has (an under-push leaves the rest to the feedback, an over-push
    hunts at every stop): 16 runs of the old wheel agreed as well with a sliding friction of 150 as with 358, because the
    wheel barely slid in them. The two with the lowest and the highest friction go in with the neighbours. With the
    slide probe ``run_check`` then designs on the probe's own estimate (the old wheel's 17:16 run picked 434, its
    breakaway, where the probe had stopped it at every hold up to 298, and the breakaway push fell away).
    """
    if not alike:
        return best, []
    low = min(alike + [best], key=lambda p: p.stick)
    high = max(alike + [best], key=lambda p: p.stick)
    others = [p for p in (high, best) if p != low]
    return low, [p for i, p in enumerate(others) if p not in others[:i]][:ALIKE_MAX]


def _climb(bench: Bench, direction: int, home: float, hz: float, say, runs: list[dict],
           inertia: float = LADDER_INERTIA) -> float | None:
    """The ladder; the bandwidth of the last rung that held, None if even the first rang."""
    clean = None
    for bw in LADDER:
        design = Design(bandwidth_hz=bw, damping=0.7, inertia=inertia, damping_first=True, cap=RUNG_CAP)
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
    last, good, _ = _verify(bench, choice, direction, home, hz, say, runs, margin)
    return last, good


def _verify(bench: Bench, choice: Choice, direction: int, home: float, hz: float, say,
            runs: list[dict], margin: float = 1.0) -> tuple[list[dict], bool, bool]:
    """``_verify_once`` and whether it held all but the fast ramp the push ahead adds (``slow``): LA's own target
    moves at most 0.2 lock/s, so such a design still serves it (the old wheel's model rim on 358, 2026-11-06).
    """
    design = choice.design()
    say(f"verifying {design.label}")
    last: list[dict] = []
    stages = VERIFY_AHEAD if design.pushes_ahead else VERIFY
    for i, stage in enumerate(stages):
        last.append(bench.run_stage(stage, design, direction, hz, home))
        runs.append(last[-1])
        s = summarize(last[-1])
        say("  " + describe(s))
        if not accepted(s, margin):
            return last, False, stages is VERIFY_AHEAD and i == len(stages) - 1
    return last, True, False


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


def _wheel_ladder(bench: Bench, rim: WheelParams, direction: int, home: float, hz: float, say, runs: list[dict],
                  failed: list[Choice], slow: list[Choice] | None = None) -> tuple[Choice | None, list[dict]]:
    """Push-ahead designs tried on the wheel itself, gentlest first, keeping the fastest that holds; one that held all but
    the fast ramp goes into ``slow`` and ends the climb.

    For a wheel the model cannot describe (gear teeth, backlash, a coarse encoder): on the geared model wheel the fit read
    inertia 159 and 67 ms where it is 600 and 25, no design held in the model, and 1.0 Hz with the push ahead held on
    the wheel. Designed on four times the rim (a rim of 40 to 70 held on 200 in the model, the R3's 160 on 650), with
    the breakaway push on a sticky rim where the plugin has it; on twice it when even the gentlest rung rings (a fit
    that read the rim too heavy). The wheel decides, so this works on any wheel.
    """
    push = push_ahead_of(rim, breakaway_share=WHEEL_LADDER_BREAKAWAY if bench.breakaway else 0.0)
    held, last = None, []
    for scale in WHEEL_LADDER_SCALES:
        inertia = max(WHEEL_LADDER_INERTIA[0], min(WHEEL_LADDER_INERTIA[1], scale * rim.inertia))
        if held is not None or (slow and held is None) or any(
                abs(f.inertia - inertia) < 1.0 for f in failed if f.bandwidth_hz == WHEEL_LADDER[0]):
            break
        for bw in WHEEL_LADDER:
            cand = Choice(inertia, bw, WHEEL_LADDER_DAMPING, None, None, 1.0, False, **push)
            tried, good, fast_only = _verify(bench, cand, direction, home, hz, say, runs)
            if not good:
                failed.append(cand)
                if fast_only and held is None and slow is not None:
                    slow.append(cand)
                    last = tried
                break
            held, last = cand, tried
    return held, last


def run_check(bench: Bench, direction: int, home: float, hz: float = 20.0, evals: int = 500, say=print,
              grid: dict | None = None) -> dict:
    """The whole check. A wheel that cannot be brought back to its start ends it early with what was recorded."""
    runs: list[dict] = []
    notes: list[str] = []
    clean = result = choice = held = None
    params: WheelParams | None = None
    agree = total = 0
    also: list[WheelParams] = []
    probe: dict | None = None
    damp: dict | None = None
    slow: list[Choice] = []
    slow_runs: list[dict] = []
    last_runs: list[dict] = []
    failed: list[Choice] = []
    ahead = bench.push_ahead and bench.ff_plan
    ladder_inertia = LADDER_INERTIA
    reading = round(bench.reading_hz) if bench.reading_hz > 0.0 else 0.0
    try:
        say(f"the game reads the wheel {reading:.0f} times a second" if reading else "wheel reading rate unknown")
        if not ahead:
            notes.append("the plugin cannot push ahead on a plan (layout 5, status 1024): designed without it")
        elif not bench.breakaway:
            notes.append("the plugin has no breakaway push (status 2048): a sticky wheel is designed without it")
        say("climbing: gentle damping-first designs, stopping at the first that rings")
        clean = _climb(bench, direction, home, hz, say, runs)
        if clean is None:
            say(f"the first rung rang: climbing again on a lighter design ({LADDER_LIGHT:.0f})")
            ladder_inertia = LADDER_LIGHT
            clean = _climb(bench, direction, home, hz, say, runs, LADDER_LIGHT)
        if clean is None:
            notes.append(f"even {LADDER[0]} Hz rang or hunted: a very light or loose wheel, or stepping and backlash")
        slips = (smoothness(runs)["stuck"] or 0.0) > STUCK_PROBE
        if slips:
            say("the wheel stick-slips: short open-loop pushes to measure what starts it and what keeps it sliding")
            probe = slide_probe(bench, direction, home, say)
            if probe.get("breakaway"):
                say("and longer pushes above its start to measure its damping")
                damp = damping_probe(bench, direction, home, say)
        frictions = bracket_frictions(probe) if probe else []
        held_c = (damp or {}).get("damping") or 0.0
        for turn in range(ROUNDS):
            start = ((probe or {}).get("breakaway") or breakaway_of(runs)) if slips else 0.0
            base = WheelParams(sign=direction, reading_hz=reading, breakaway=start, quantum=quantum_of(runs),
                               breakaway_spread=start_spread(probe) if probe and slips else 0.0,
                               **({"damping": held_c} if held_c else {}))
            say("fitting the rim" + (" again, with the run that rang" if turn else "")
                + (f" (it starts at {base.breakaway:.0f}" if base.breakaway else "")
                + (f", slides on {frictions[0]:.0f} to {frictions[-1]:.0f})" if frictions else ")" if base.breakaway else ""))
            result = fit(runs, bench.frame_hz, base, max_evals=evals, damping_held=bool(held_c))
            best, agree, total, alike = match_verdicts(result.params, runs, bench.frame_hz, frictions, bool(held_c))
            params, also = design_rim(best, alike)
            slide = slide_estimate(probe) if probe else None
            if slide and abs(params.stick - slide) > 1.0:
                also = [params] + [p for p in also if p != params][:ALIKE_MAX - 1]
                params = replace(params, stick=slide)
            say(f"the model agrees with the wheel on {agree} of {total} runs"
                + (f", as well with a sliding friction of {min(p.stick for p in also + [params]):.0f} to "
                   f"{max(p.stick for p in also + [params]):.0f}: designed on the lowest, held on the others"
                   if also else ""))
            if agree < TRUST_SHARE * total:
                notes.append(f"the model agreed with the wheel on {agree} of {total} runs only: designs tried on the wheel")
                break
            say("choosing a design for it" + (", with the push ahead" if ahead else ""))
            choice = _not_again(choose_design(params, bench.frame_hz, ahead=ahead, breakaway=bench.breakaway,
                                              extra=tuple(also), measured=bool(frictions), **(grid or {})), failed)
            if choice.limited:
                notes.append("no design in the grid held up on the fitted rim and its neighbours")
                if ahead:
                    break
            tried, good, fast_only = _verify(bench, choice, direction, home, hz, say, runs)
            if good:
                held, last_runs = choice, tried
                break
            failed.append(choice)
            if fast_only and not slow:
                slow.append(choice)
                slow_runs = tried
            last_runs = tried
        if held is None and ahead and params is not None:
            say("the model's designs did not hold, trying push-ahead designs on the wheel, gentlest first")
            before = len(slow)
            held, tried = _wheel_ladder(bench, params, direction, home, hz, say, runs, failed, slow)
            last_runs = tried
            if len(slow) > before:
                slow_runs = tried
            if held is not None:
                notes.append("chosen on the wheel, not in the model: the model does not describe this wheel well")
        slow_held = held is None and bool(slow)
        if slow_held:
            held, last_runs = slow[0], slow_runs
            notes.append("it holds the wheel up to 0.2 lock/s (as fast as LA's own target moves) but rang on the "
                         "0.4 lock/s ramp: kept, not tried faster")
        if held is None:
            say("the model's designs did not hold on the wheel, trying the gentlest rung")
            gentle = Choice(ladder_inertia, LADDER[0], 0.7, None, None, 1.0, True)
            last_runs, good = _verify_once(bench, gentle, direction, home, hz, say, runs)
            held = gentle if good else None
        elif held is not None and not held.limited and held.rise_s is not None and not slow_held:
            held, last_runs = _refine(bench, held, last_runs, failed, direction, home, hz, say, runs)
        if failed:
            notes.append(f"{len(failed)} design(s) rang on the wheel" + ("" if held else ", none held"))
    except RunInvalid:
        raise
    except BenchAbort as exc:
        notes.append(f"stopped: {exc}")
        held = None
    if held is None:
        notes.append("no design held on this wheel: leave the la_* keys at their defaults")
    if params is not None and sticky(params):
        notes.append(f"it takes {params.breakaway:.0f} to start the wheel and {params.stick:.0f} to keep it sliding:"
                     " expect small steps on slow moves")
    step_run = next((summarize(r) for r in last_runs if r["name"] == "step 0.02" and not r.get("aborted")), None)
    return {
        "runs": runs, "wheel": asdict(params) if params else {}, "fit_rms": result.rms if result else None,
        "model_agrees": [agree, total], "also_agree": [asdict(p) for p in also], "probe": probe,
        "damping_probe": damp or {},
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
                                                           if data.get("reading_hz") else "")
                     + (f", starts at {w['breakaway']:.0f}" if w.get("breakaway", 0.0) > w["stick"] else "")
                     + (f", angle steps {w['quantum']:.4f} lock" if w.get("quantum") else ""))
        pr = data.get("probe") or {}
        if pr.get("mixed"):
            lines.append(f"  slide probe: starts at {pr['breakaway']:.0f}, slides on {pr['slide_low']:.0f} to "
                         f"{pr['slide_high']:.0f} (the trials disagree: one way or place slides on less)")
        elif pr.get("breakaway"):
            lines.append(f"  slide probe: starts at {pr['breakaway']:.0f}, "
                         + (f"stops at {pr['slide_low']:.0f}, " if pr.get("slide_low") else "")
                         + (f"keeps sliding at {pr['slide_high']:.0f}" if pr.get("slide_high") else "never kept sliding"))
        dp = data.get("damping_probe") or {}
        if dp.get("damping"):
            lines.append(f"  damping probe: damping {dp['damping']:.0f}, spring {dp['spring']:.0f}, sliding on "
                         f"{dp['friction']:.0f} (designed on that damping)")
        if data.get("model_agrees"):
            lines.append("  model: agrees with the wheel on {} of {} runs".format(*data["model_agrees"]))
    else:
        lines.append("  rim: not measured")
    if data["verified"]:
        lines.append(f"  responsiveness: {c['bandwidth_hz']:g} Hz design" + (" with the push ahead" if c.get("ff_damping")
                                                                           or c.get("ff_inertia") else "")
                     + (" and the breakaway push" if c.get("ff_breakaway", 0.0) > c.get("ff_friction", 0.0) else "")
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
