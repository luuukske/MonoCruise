"""Servo bench for the LA wheel servo: steps, ramps and follow runs on the standing wheel, a fit, a what-if.

Capture needs the game running and the truck standing (driving, not paused), hands off the rim, the version 4
plugin installed and MonoCruise's LA, LKA and rumble idle (they would overwrite the requests). Gains, targets
and rates are runtime fields of the command block, so a new design needs no game restart.

  python tools/la_servo_bench.py capture                      the real wheel, two designs, steps to follow runs
  python tools/la_servo_bench.py capture --simulate           the same flow on a model wheel, no game
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
from tools.la_bench.metrics import describe, summarize
from tools.la_bench.plant import SimChannel, SimClock, WheelParams, fit, replay
from tools.la_bench.protocol import SUITES, Design, Stage

OUT_DIR = ROOT / "tools" / "la_bench_out"
ANGLE_LIMIT = 0.4
CENTRE_LIMIT = 0.25
HOME_BAND = 0.01
HOME_RATE = 0.15
HOME_MAX_S = 6.0
HOME_DESIGN = Design(bandwidth_hz=1.0, cap=2000)
POLL_S = 0.001
DIRECTION_UNITS = 600
DIRECTION_S = 0.5
SIM_WHEEL = WheelParams(inertia=700.0, damping=500.0, stick=150.0, spring=2000.0, delay_s=0.02)
DIRECTION_MIN_MOVE = 0.004
FORMAT = 1


class BenchAbort(Exception):
    """Something the run cannot go on through; the servo is asked off before the bench returns."""


class WheelRunaway(BenchAbort):
    """The wheel went past the angle limit while the bench was pushing it; the stage ends, the next may still run."""


def _why_not(st: PluginStatus) -> str:
    if not st.present:
        return "plugin not found: start the game with monocruise_ffb.dll installed"
    if st.version < VERSION:
        return f"plugin version {st.version} cannot steer: install the version {VERSION} DLL"
    if not st.hooked:
        return "constant force not hooked yet: drive for a few seconds and rerun"
    return "steering channel not seen: the game must be driving, not paused"


class Bench:
    """Runs stages through a channel with the clock and sleep it is given, so a model wheel can stand in."""

    def __init__(self, channel, clock=time.perf_counter, sleep=time.sleep, poll_s: float = POLL_S, say=print) -> None:
        self.channel = channel
        self.clock = clock
        self.sleep = sleep
        self.poll_s = poll_s
        self.say = say
        self._hz = [0.0, 0]

    @property
    def frame_hz(self) -> float:
        return self._hz[0] / self._hz[1] if self._hz[1] else 120.0

    def status(self, servo: ServoCommand = SERVO_OFF, torque: float = 0.0, limit: float = ANGLE_LIMIT) -> PluginStatus:
        st = self.channel.publish(False, 0, 0, torque=torque, servo=servo)
        if not st.present or st.dead or not st.hooked:
            raise BenchAbort("plugin lost: is the game running and driving?")
        if (servo.direction or torque) and abs(st.steer) > limit:
            raise WheelRunaway(f"wheel past {limit} lock: servo off, the design rings or the direction is wrong")
        if st.frame_hz > 0.0:
            self._hz[0] += st.frame_hz
            self._hz[1] += 1
        return st

    def ready(self, wait_s: float = 10.0) -> PluginStatus:
        end = self.clock() + wait_s
        while True:
            st = self.channel.publish(False, 0, 0)
            if st.servo_capable:
                return st
            if self.clock() >= end:
                raise BenchAbort(_why_not(st))
            self.sleep(0.05)

    def idle(self, seconds: float, torque: float = 0.0) -> PluginStatus:
        end = self.clock() + seconds
        st = self.status(torque=torque)
        while self.clock() < end:
            st = self.status(torque=torque)
            self.sleep(self.poll_s)
        return st

    def rest(self, min_s: float = 1.0, max_s: float = 4.0) -> float:
        """Servo off until the wheel has been still for 0.3 s (and for at least ``min_s``); the angle it rests at."""
        t0 = self.clock()
        still = None
        while True:
            st = self.status()
            now = self.clock()
            still = (now if still is None else still) if abs(st.steer_rate) < 0.01 else None
            if now - t0 >= max_s or (now - t0 >= min_s and still is not None and now - still >= 0.3):
                return st.steer
            self.sleep(self.poll_s)

    def find_direction(self) -> int:
        """Push a little torque with the servo off: +1 if positive torque turns the wheel counterclockwise."""
        start = self.rest()
        moved = self.idle(DIRECTION_S, DIRECTION_UNITS).steer - start
        self.idle(0.0)
        if abs(moved) < DIRECTION_MIN_MOVE:
            raise BenchAbort("the wheel did not move under torque: hands on the rim, or the game paused? (--direction skips the test)")
        return 1 if moved > 0 else -1

    def go_home(self, home: float, direction: int) -> None:
        """Walk a wheel that rests away from ``home`` back with a slow, well damped servo, then let go."""
        start = self.rest()
        if abs(start - home) <= HOME_BAND:
            return
        kp, kd, ki = HOME_DESIGN.gains()
        sign = 1.0 if home > start else -1.0
        t0 = self.clock()
        still = None
        try:
            while (t := self.clock() - t0) < HOME_MAX_S:
                walk = min(abs(home - start), HOME_RATE * t)
                moving = walk < abs(home - start)
                servo = ServoCommand(target=start + sign * walk, rate=sign * HOME_RATE if moving else 0.0, kp=kp, kd=kd,
                                     ki=ki, cap=HOME_DESIGN.cap, direction=direction)
                st = self.status(servo, limit=ANGLE_LIMIT + 0.2)
                near = abs(st.steer - home) <= HOME_BAND / 2
                still = (t if still is None else still) if near else None
                if still is not None and t - still >= 0.3:
                    break
                self.sleep(0.05)
        finally:
            self.channel.publish(False, 0, 0, servo=SERVO_OFF)

    def run_stage(self, stage: Stage, design: Design, direction: int, hz: float = 20.0, home: float | None = None) -> dict:
        if home is not None:
            self.go_home(home, direction)
        start = self.rest()
        if abs(start) > CENTRE_LIMIT:
            raise BenchAbort(f"the wheel rests {start:+.2f} lock off centre: centre it and rerun")
        kp, kd, ki = design.gains()
        events = stage.events(hz)
        servo = SERVO_OFF
        sent, samples = [], []
        aborted = None
        last = None
        t0 = self.clock()
        i = 0
        try:
            while (t := self.clock() - t0) < stage.duration:
                while i < len(events) and events[i] <= t:
                    offset, rate = stage.aim(events[i])
                    servo = ServoCommand(target=start + offset, rate=rate, kp=kp, kd=kd, ki=ki, cap=design.cap,
                                         direction=direction, damping_first=design.damping_first)
                    sent.append([t, servo.target, rate])
                    i += 1
                st = self.status(servo)
                if st.steer_frames != last:
                    last = st.steer_frames
                    samples.append([t, st.steer, st.steer_rate, st.applied, st.servo_applied, st.game_force,
                                    st.servo_integ, st.servo_scale])
                self.sleep(self.poll_s)
        except WheelRunaway as exc:
            aborted = str(exc)
        finally:
            self.channel.publish(False, 0, 0, servo=SERVO_OFF)
        run = {
            "name": stage.name, "label": design.label, "stage": stage.to_dict(), "design": asdict(design),
            "gains": {"kp": kp, "kd": kd, "ki": ki, "cap": design.cap, "direction": direction,
                      "damping_first": design.damping_first},
            "start": start, "sent": _round(sent), "samples": _round(samples),
        }
        if aborted:
            run["aborted"] = aborted
        return run


def _round(rows: list[list[float]]) -> list[list[float]]:
    return [[round(v, 5) for v in row] for row in rows]


def capture(bench: Bench, designs: list[Design], stages, hz: float, direction: int | None = None,
            note: str = "") -> dict:
    """Every stage under every design; a plugin or centring problem ends it early with what was recorded so far."""
    runs, aborted, st = [], None, None
    try:
        st = bench.ready()
        bench.say(f"plugin present, version {st.version}, {st.frame_hz:.0f} Hz FFB: hands off the wheel")
        if any(d.damping_first for d in designs) and not st.damping_first:
            raise BenchAbort("the plugin cannot damp first: rebuild plugins/ffb and restart the game")
        direction = direction or bench.find_direction()
        home = bench.rest()
        if abs(home) > CENTRE_LIMIT:
            raise BenchAbort(f"the wheel rests {home:+.2f} lock off centre: centre it and rerun")
        bench.say(f"positive torque turns the wheel {'counterclockwise' if direction > 0 else 'clockwise'}")
        for design in designs:
            kp, kd, ki = design.gains()
            bench.say(f"design {design.label}: kp {kp:.0f}, kd {kd:.0f}, ki {ki:.0f}, cap {design.cap}")
            for stage in stages:
                runs.append(bench.run_stage(stage, design, direction, hz, home))
                bench.say("  " + describe(summarize(runs[-1])))
    except BenchAbort as exc:
        aborted = str(exc)
        bench.say(f"aborted: {aborted}")
    meta = {"format": FORMAT, "plugin_version": st.version if st else 0, "frame_hz": round(bench.frame_hz, 2),
            "direction": direction or 0, "target_hz": hz,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if aborted:
        meta["aborted"] = aborted
    if note:
        meta["note"] = note
    return {"meta": meta, "runs": runs}


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


def _sim_bench(params: WheelParams) -> Bench:
    clock = SimClock()
    return Bench(SimChannel(params, clock), clock, clock.sleep)


def cmd_capture(args) -> int:
    designs, stages = _designs(args), SUITES[args.suite]
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out = Path(args.out) if args.out else (None if args.simulate else OUT_DIR / f"la_bench_{stamp}.json")
    if args.simulate:
        data = capture(_sim_bench(SIM_WHEEL), designs, stages, args.hz, args.direction, args.note)
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
    result = fit(runs, frame_hz, WheelParams(sign=direction), max_evals=args.evals, saturated=args.include_saturated)
    p = result.params
    saved = {"params": asdict(p), "rms": result.rms, "evals": result.evals, "per_run": result.per_run,
             "used": result.used}
    if not args.no_save:
        for path, data in zip(args.files, files):
            data["fit"] = saved
            _save(Path(path), data)
    print(f"fit on {result.used} of {len(runs)} runs from {len(files)} recording(s), {result.evals} model runs: "
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
    bench = _sim_bench(params)
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
    p = sub.add_parser("report", help="metrics per run")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("fit", help="fit the rim to a recording")
    p.add_argument("files", nargs="+")
    p.add_argument("--kind", nargs="+", choices=("step", "ramp", "follow"), help="fit on these stages only")
    p.add_argument("--evals", type=int, default=400, help="model-run budget")
    p.add_argument("--compare", action="store_true", help="print measured against model per run")
    p.add_argument("--include-saturated", action="store_true", help="also fit runs whose servo reached its cap")
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
