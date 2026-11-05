"""Servo bench for the LA wheel servo: steps, ramps and follow runs on the standing wheel, a fit, a what-if.

Capture needs the game running and the truck standing (driving, not paused), hands off the rim, the version 4
plugin installed and MonoCruise's LA, LKA and rumble idle (they would overwrite the requests). Gains, targets
and rates are runtime fields of the command block, so a new design needs no game restart.

  python tools/la_servo_bench.py capture                      the real wheel, two designs, steps to follow runs
  python tools/la_servo_bench.py capture --simulate           the same flow on a model wheel, no game
  python tools/la_servo_bench.py check                        the wheel check: fit this rim, pick and verify a design
  python tools/la_servo_bench.py report FILE                  rise, overshoot, settling, tracking per run
  python tools/la_servo_bench.py fit FILE                     inertia, damping, friction, spring, delay of the rim
  python tools/la_servo_bench.py predict FILE --bandwidth 1.5 2 2.5     other designs on the fitted rim

See core/la/README.md, "Bench".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lane_departure.command import SERVO_OFF, VERSION, PluginStatus, RumbleChannel, ServoCommand
from tools.la_bench.bench import (ANGLE_LIMIT, CENTRE_LIMIT, Bench, BenchAbort, WheelRunaway, _round, sim_bench)
from tools.la_bench.metrics import describe, summarize
from tools.la_bench.check import run_check, summary
from tools.la_bench.plant import SIM_WHEELS, SimChannel, SimClock, WheelParams, fit, replay
from tools.la_bench.protocol import SUITES, Design, Stage

OUT_DIR = ROOT / "tools" / "la_bench_out"
SIM_WHEEL = WheelParams(inertia=700.0, damping=500.0, stick=150.0, spring=2000.0, delay_s=0.02)
FORMAT = 1


def capture(bench: Bench, designs: list[Design], stages, hz: float, direction: int | None = None,
            note: str = "") -> dict:
    """Every stage under every design; a plugin or centring problem ends it early with what was recorded so far."""
    runs, aborted, st = [], None, None
    try:
        st, direction, home = bench.prepare(direction, any(d.damping_first for d in designs))
        for design in designs:
            kp, kd, ki = design.gains()
            bench.say(f"design {design.label}: kp {kp:.0f}, kd {kd:.0f}, ki {ki:.0f}, cap {design.cap}")
            for stage in stages:
                runs.append(bench.run_stage(stage, design, direction, hz, home))
                bench.say("  " + describe(summarize(runs[-1])))
    except BenchAbort as exc:
        aborted = str(exc)
        bench.say(f"aborted: {aborted}")
    return {"meta": _meta(bench, st, direction, hz, aborted, note), "runs": runs}


@contextmanager
def _fine_timer():
    winmm = None
    if sys.platform == "win32":
        try:
            import ctypes

            winmm = ctypes.WinDLL("winmm")
            winmm.timeBeginPeriod(1)
        except Exception:
            winmm = None
    try:
        yield
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(1)


def _designs(args) -> list[Design]:
    laws = {"plain": (False,), "damping-first": (True,), "both": (False, True)}[args.law]
    return [
        Design(bandwidth_hz=bw, damping=args.damping, inertia=args.inertia, strength=args.strength,
               integral_share=args.integral_share, spring=args.game_spring, cap=args.cap, damping_first=first)
        for bw in args.bandwidth for first in laws
    ]


def _add_design_args(p: argparse.ArgumentParser, bandwidth: list[float]) -> None:
    p.add_argument("--bandwidth", type=float, nargs="+", default=bandwidth, help="servo bandwidths in Hz, one design each")
    p.add_argument("--damping", type=float, default=0.7, help="fraction of critical damping")
    p.add_argument("--inertia", type=float, default=650.0, help="la_wheel_inertia the design is made on")
    p.add_argument("--strength", type=float, default=1.0, help="la_strength")
    p.add_argument("--integral-share", type=float, default=0.3, help="integral gain at this share of the bandwidth")
    p.add_argument("--game-spring", type=float, default=0.0, help="the game's centring the design is told about")
    p.add_argument("--cap", type=int, default=3000, help="servo cap out of 10000")
    p.add_argument("--law", choices=("plain", "damping-first", "both"), default="plain",
                   help="damping-first gives D its room under the cap before P and I (needs the rebuilt plugin)")
    p.add_argument("--suite", choices=sorted(SUITES), default="full")
    p.add_argument("--hz", type=float, default=20.0, help="how often a moving target is sent")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))


def cmd_capture(args) -> int:
    designs, stages = _designs(args), SUITES[args.suite]
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out = Path(args.out) if args.out else (None if args.simulate else OUT_DIR / f"la_bench_{stamp}.json")
    if args.simulate:
        data = capture(sim_bench(SIM_WHEEL), designs, stages, args.hz, args.direction, args.note)
    else:
        channel = RumbleChannel()
        try:
            with _fine_timer():
                data = capture(Bench(channel), designs, stages, args.hz, args.direction, args.note)
        finally:
            channel.stop()
            channel.close()
    if out is not None and data["runs"]:
        _save(out, data)
        print(f"saved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name}")
    return 1 if "aborted" in data["meta"] else 0


def _meta(bench: Bench, st, direction: int | None, hz: float, aborted: str | None, note: str = "") -> dict:
    meta = {"format": FORMAT, "plugin_version": st.version if st else 0, "frame_hz": round(bench.frame_hz, 2),
            "direction": direction or 0, "target_hz": hz,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if aborted:
        meta["aborted"] = aborted
    if note:
        meta["note"] = note
    return meta


def cmd_check(args) -> int:
    """The wheel check on the real wheel, or on a model wheel with --simulate."""
    st = direction = home = None
    aborted = None
    channel = None
    if args.simulate:
        bench = sim_bench(SIM_WHEELS[args.simulate])
    else:
        channel = RumbleChannel()
        bench = Bench(channel)
    data: dict = {"runs": []}
    try:
        with _fine_timer():
            st, direction, home = bench.prepare(args.direction, True)
            data = run_check(bench, direction, home, args.hz, args.evals, bench.say)
    except BenchAbort as exc:
        aborted = str(exc)
        print(f"aborted: {aborted}")
    finally:
        if channel is not None:
            channel.stop()
            channel.close()
    if aborted is not None:
        return 1
    data["meta"] = _meta(bench, st, direction, args.hz, None, args.note)
    print(summary(data))
    out = Path(args.out) if args.out else (None if args.simulate else OUT_DIR / f"la_check_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    if out is not None:
        _save(out, data)
        print(f"saved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name}")
    return 0


def cmd_report(args) -> int:
    summaries = [summarize(run) for run in _load(args.file)["runs"]]
    print(json.dumps(summaries, indent=1) if args.json else "\n".join(describe(s) for s in summaries))
    return 0


def _model_run(run: dict, params: WheelParams, frame_hz: float) -> dict:
    ts, ys = replay(run, params, frame_hz)
    return dict(run, samples=[[t, y, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for t, y in zip(ts, ys)])


def cmd_fit(args) -> int:
    files = [_load(path) for path in args.files]
    directions = {d["meta"]["direction"] for d in files}
    if len(directions) != 1 or 0 in directions:
        print("the recordings disagree on the torque direction, or none found it")
        return 1
    direction = directions.pop()
    frame_hz = sum(d["meta"]["frame_hz"] for d in files) / len(files)
    runs = [r for d in files for r in d["runs"] if not args.kind or r["stage"]["kind"] in args.kind]
    result = fit(runs, frame_hz, WheelParams(sign=direction), max_evals=args.evals, ring=not args.no_ring)
    p = result.params
    saved = {"params": asdict(p), "rms": result.rms, "evals": result.evals, "per_run": result.per_run,
             "used": result.used}
    if not args.no_save:
        for path, data in zip(args.files, files):
            data["fit"] = saved
            _save(Path(path), data)
    print(f"fit on {len(runs)} runs ({result.used} by angle, the rest by swing size) from {len(files)} recording(s), {result.evals} model runs: "
          f"wheel angle error rms {result.rms:.4f} lock")
    print(f"  inertia {p.inertia:.0f}  (la_wheel_inertia 650 by default; hands on the rim add to it)")
    print(f"  damping {p.damping:.0f} per lock/s, dry friction {p.stick:.0f}, spring {p.spring:.0f} per lock "
          f"(the game's centring standing), delay {p.delay_s * 1000:.1f} ms from the plugin reading the angle to the "
          "force at the rim")
    if args.compare:
        for run, rms in zip(runs, result.per_run):
            print(f"  rms {rms:.4f}  measured: {describe(summarize(run))}")
            print(f"            model:    {describe(summarize(_model_run(run, p, frame_hz)))}")
    return 0


def cmd_predict(args) -> int:
    data = _load(args.file)
    meta = data["meta"]
    saved = data.get("fit")
    if saved is None:
        print("no fit in the file: run fit first")
        return 1
    params = WheelParams(**saved["params"])
    bench = sim_bench(params)
    bench.say = lambda _msg: None
    data_sim = capture(bench, _designs(args), SUITES[args.suite], args.hz, meta["direction"])
    for run in data_sim["runs"]:
        print(describe(summarize(run)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture", help="run the bench on the wheel")
    _add_design_args(p, [1.0, 2.0])
    p.add_argument("--simulate", action="store_true", help="use a model wheel, no game")
    p.add_argument("--direction", type=int, choices=(1, -1), help="skip the torque test: the torque sign that turns it left")
    p.add_argument("--out", help="where to write the recording")
    p.add_argument("--note", default="", help="free text kept in the recording, e.g. the game's steering settings")
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("check", help="the wheel check: ladder, fit, a design for this rim, verification")
    p.add_argument("--simulate", choices=sorted(SIM_WHEELS), help="check a model wheel, no game")
    p.add_argument("--direction", type=int, choices=(1, -1), help="skip the torque test: the torque sign that turns it left")
    p.add_argument("--hz", type=float, default=20.0, help="how often a moving target is sent")
    p.add_argument("--evals", type=int, default=500, help="fit budget in model runs")
    p.add_argument("--note", default="", help="free text kept in the recording, e.g. the wheel's own settings")
    p.add_argument("--out", help="where to write the recording")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("report", help="metrics per run")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("fit", help="fit the rim to a recording")
    p.add_argument("files", nargs="+")
    p.add_argument("--kind", nargs="+", choices=("step", "ramp", "follow"), help="fit on these stages only")
    p.add_argument("--evals", type=int, default=600, help="model-run budget")
    p.add_argument("--compare", action="store_true", help="print measured against model per run")
    p.add_argument("--no-ring", action="store_true", help="leave runs whose servo reached its cap out of the fit")
    p.add_argument("--no-save", action="store_true", help="do not write the fit back into the recording")
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("predict", help="designs on the fitted rim")
    p.add_argument("file")
    _add_design_args(p, [2.0])
    p.set_defaults(fn=cmd_predict)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
