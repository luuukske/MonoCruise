"""The wheel check: gentle designs on the standing wheel in a ladder, a fit, a design for that rim, a verification.

The ladder climbs the bandwidth of a damping-first design until one rings, so the check never asks more of the wheel
than the rung before it held. The rim is fitted on those runs, a design is chosen for it in the model, and the design
is then run on the wheel: if it rings, the bandwidth steps back and it is run again. The result is what `la_*` keys in
config.json should hold, how responsive the wheel is and how smooth.
"""

from __future__ import annotations

from dataclasses import asdict, replace

from tools.la_bench.bench import Bench, BenchAbort
from tools.la_bench.design import Choice, choose_design, smoothness
from tools.la_bench.metrics import describe, summarize
from tools.la_bench.plant import WheelParams, fit
from tools.la_bench.protocol import Design, follow, ramp, step

LADDER = (0.7, 1.0, 1.4, 2.0)
RUNG_CAP = 2000
RUNG_STEPS = (step(0.01), step(0.02))
SLOW = (ramp(0.06, 0.05), follow(0.03, 0.4))
VERIFY = (step(0.02),) + SLOW + (ramp(0.08, 0.2), follow(0.03, 0.7))
RING_OVERSHOOT = 1.0
RING_RIPPLE = 0.01
RING_BUZZ = 0.6
RING_TRACK = 0.02
ACCEPT_UP = 0.3
ACCEPT_SWING = 0.5
RETRY_SCALE = 0.8
RETRIES = 2


def rings(s: dict) -> bool:
    """A run that aborted, cut the servo with the buzz guard, or swung well past its target or its tracking."""
    if s.get("aborted") or s["buzz_min"] < RING_BUZZ:
        return True
    for key in ("up", "swing"):
        if key in s and (s[key]["overshoot"] > RING_OVERSHOOT or (s[key]["ripple"] or 0.0) > RING_RIPPLE):
            return True
    return "track" in s and (s["track"]["rms"] or 0.0) > RING_TRACK


def accepted(s: dict) -> bool:
    if rings(s):
        return False
    return "up" not in s or (s["up"]["overshoot"] <= ACCEPT_UP and s["swing"]["overshoot"] <= ACCEPT_SWING)


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


def _try(bench: Bench, choice: Choice, direction: int, home: float, hz: float, say,
         runs: list[dict]) -> tuple[list[dict], Choice | None]:
    """Run a design on the wheel, stepping its bandwidth back while it rings.

    Returns the runs of the last attempt and the design that held (None: none did); every run is added to ``runs``.
    """
    last: list[dict] = []
    bw = choice.bandwidth_hz
    for _ in range(RETRIES + 1):
        design = Design(bandwidth_hz=bw, damping=choice.damping, inertia=choice.inertia, damping_first=True)
        say(f"verifying {design.label}")
        good = True
        last = []
        for stage in VERIFY:
            last.append(bench.run_stage(stage, design, direction, hz, home))
            runs.append(last[-1])
            s = summarize(last[-1])
            say("  " + describe(s))
            if not accepted(s):
                good = False
                break
        if good:
            return last, replace(choice, bandwidth_hz=bw)
        bw = round(bw * RETRY_SCALE, 2)
    return last, None


def run_check(bench: Bench, direction: int, home: float, hz: float = 20.0, evals: int = 500, say=print,
              grid: dict | None = None) -> dict:
    """The whole check. A wheel that cannot be brought back to its start ends it early with what was recorded."""
    runs: list[dict] = []
    notes: list[str] = []
    clean = result = choice = held = None
    last_runs: list[dict] = []
    try:
        say("climbing: gentle damping-first designs, stopping at the first that rings")
        clean = _climb(bench, direction, home, hz, say, runs)
        if clean is None:
            notes.append(f"even {LADDER[0]} Hz rang or hunted: a very light or loose wheel, or stepping and backlash")
        say("fitting the rim")
        result = fit(runs, bench.frame_hz, WheelParams(sign=direction), max_evals=evals)
        say("choosing a design for it")
        choice = choose_design(result.params, bench.frame_hz, **(grid or {}))
        if choice.limited:
            notes.append("no design in the grid held up on the fitted rim and its neighbours")
        last_runs, held = _try(bench, choice, direction, home, hz, say, runs)
        if held is None:
            say("the model's design rang on the wheel, trying the gentlest rung")
            last_runs, held = _try(bench, Choice(650.0, LADDER[0], 0.7, None, None, 1.0, True), direction, home, hz, say, runs)
            notes.append("the model's design rang on the wheel" + ("; the gentlest rung held" if held else ""))
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
        "config": held.config() if held else {}, "notes": notes,
    }


def summary(data: dict) -> str:
    """The result as text: responsiveness, smoothness and the config.json keys."""
    w, c, m = data["wheel"], data["choice"], data["smoothness"]
    rise = data["step_rise_s"]
    lines = ["Wheel check"]
    if w:
        lines.append(f"  rim: inertia {w['inertia']:.0f}, damping {w['damping']:.0f}, dry friction {w['stick']:.0f}, "
                     f"delay {w['delay_s'] * 1000:.0f} ms")
    else:
        lines.append("  rim: not measured")
    if data["verified"]:
        lines.append(f"  responsiveness: {c['bandwidth_hz']:g} Hz design, verified on the wheel"
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
