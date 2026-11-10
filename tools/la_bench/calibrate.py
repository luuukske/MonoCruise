"""Runs ``core.la.wheel_calibration`` on the standing wheel (the plugin's command block and trace block) or on a model
wheel, and keeps the raw trace so a real run can be analysed again offline (``replay``). See core/la/README.md,
Wheel calibration.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

from core.la.actuator import WheelTuning
from core.la.design_check import CheckResult
from core.la.wheel_calibration import DONE, FAILED, CalTuning, WheelCalibration, WheelProfile, _Trial, analyse
from core.lane_departure.command import SERVO_OFF, TORQUE_VERSION
from core.lane_departure.ffb_trace import TraceSample
from tools.la_bench.design import Choice, fast_design
from tools.la_bench.plant import SimWheel, WheelParams

TICK_S = 0.05
MAX_S = 60.0


def record(cal: WheelCalibration) -> dict:
    """The run as saved: every sample the calibration saw (time, reading, the game's force, the whole push, reading
    count), its trials and what it made of them."""
    rows = zip(cal._t, cal._steer, cal._game, cal._out, cal._seq)
    return {"samples": [[t, x, g, o, q] for t, x, g, o, q in rows],
            "trials": [asdict(tr) for tr in cal._trials], "relay": cal._relay, "creep": cal._creep,
            "direction": cal._direction,
            "state": cal.state,
            "failure": cal.failure, "profile": asdict(cal.profile) if cal.profile else None}


def replay(data: dict, tune: CalTuning | None = None) -> WheelProfile:
    """The profile of a saved run, analysed again (with other tuning, say)."""
    rows = data["samples"]
    trials = [_Trial(**tr) for tr in data["trials"]]
    relay = data.get("relay")
    return analyse([r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows], [int(r[4]) for r in rows],
                   trials, data["direction"], tune or CalTuning(), tuple(relay) if relay else None,
                   data.get("creep") or [])


def _torque(machine, now: float) -> float:
    """The calibration's step is its torque, the design check's its servo command (kept on ``servo`` by both)."""
    got = machine.step(now)
    return got if isinstance(got, float) else 0.0


def drive_sim(machine, w: SimWheel, max_s: float = MAX_S) -> float:
    """Step ``machine`` (a calibration or a design check) on a model wheel at 20 Hz from where the wheel is now;
    its seconds."""
    t0 = t = w.time
    while t < t0 + max_s and machine.state not in (DONE, FAILED):
        torque = _torque(machine, t)
        buf, frames = [], w.frames
        while w.time < t + TICK_S:
            w.substep(machine.servo, torque)
            if w.frames != frames:
                frames = w.frames
                buf.append(TraceSample(w.time, w.reading, 0, round(w.applied), round(w.plugin.value), w.readings, 0))
        machine.feed(buf)
        t += TICK_S
    return t - t0


def run_sim(params: WheelParams, frame_hz: float = 120.6, tune: CalTuning | None = None, steer0: float = 0.0,
            backlog_s: float = 0.0) -> tuple[WheelCalibration, SimWheel, float]:
    """The calibration on a model wheel, stepped at 20 Hz like the lane departure thread; also its seconds.
    ``backlog_s`` puts a sample that old in front, as the trace held while the game was behind the terminal."""
    w = SimWheel(params, frame_hz, steer0=steer0)
    cal = WheelCalibration(tune or CalTuning())
    if backlog_s:
        cal.feed([TraceSample(-backlog_s, w.reading, 0, 0, 0, -1, 0)])
    return cal, w, drive_sim(cal, w)


def drive_live(machine, channel, reader, say=print, clock=time.perf_counter, sleep=time.sleep,
               max_s: float = MAX_S) -> None:
    """Step ``machine`` on the wheel: ``channel`` a ``RumbleChannel``, ``reader`` a ``TraceReader``."""
    t0 = clock()
    next_at = t0
    state = None
    try:
        while machine.state not in (DONE, FAILED):
            now = clock() - t0
            if now > max_s:
                machine.abort(f"not done in {max_s:.0f} s")
                break
            machine.feed(reader.read().samples)
            torque = _torque(machine, now)
            st = channel.publish(False, 0, 0, torque=torque, servo=machine.servo)
            if not st.present or st.version < TORQUE_VERSION or not st.steer_ok:
                machine.abort("the FFB plugin is not there, too old, or does not read the wheel")
                break
            if getattr(channel, "foreign", 0):
                machine.abort("another program wrote to the wheel: switch LA, the lane keeping assist and the rumble off")
                break
            if machine.state != state:
                state = machine.state
                if state == "ramp":
                    say(f"  trial {len(machine._trials) + 1}: pushing until the wheel starts")
                elif state == "creep":
                    say("  creep: the wheel moves slowly out and back")
                elif state == "relay":
                    say("  relay: the wheel buzzes round where it stands for a second")
            next_at += TICK_S
            sleep(max(0.0, next_at - clock()))
    finally:
        channel.publish(False, 0, 0, torque=0, servo=SERVO_OFF)


def run_live(channel, reader, say=print, clock=time.perf_counter, sleep=time.sleep,
             tune: CalTuning | None = None) -> WheelCalibration:
    """The calibration on the wheel."""
    cal = WheelCalibration(tune or CalTuning())
    drive_live(cal, channel, reader, say, clock, sleep)
    return cal


STEP_DOWN = 0.8
STEP_UP = 1.25
CHECKS_MAX = 4
BANDWIDTH_MIN = 0.5
BANDWIDTH_MAX = 2.5


def tuning_of(choice: Choice, bandwidth: float, cap: float = 3000.0) -> WheelTuning:
    """The servo tuning a design check runs: the choice at ``bandwidth``."""
    return WheelTuning(inertia=choice.inertia, bandwidth_hz=bandwidth, damping=choice.damping, damping_first=True,
                       servo_cap=cap, ff_inertia=choice.ff_inertia, ff_damping=choice.ff_damping,
                       ff_friction=choice.ff_friction, plan_s=choice.plan_s, ff_breakaway=choice.ff_breakaway)


STIFFER = 1.5
HALF_PUSH = 0.5
HALF_PUSH_GAIN = 0.05


def _floored(c: Choice, breakaway: float) -> Choice:
    """``c`` with the design inertia raised, if needed, to move a STEP_MIN step past ``breakaway`` with its push."""
    floor = max(0.0, breakaway - c.ff_friction) / STEP_MIN
    return replace(c, inertia=float(math.ceil(max(c.inertia, floor / (2.0 * math.pi * c.bandwidth_hz) ** 2))))


def verify(choice: Choice, run_check, say=print,
           breakaway: float = 0.0) -> tuple[Choice | None, list[tuple[Choice, CheckResult | None]]]:
    """The model's design tried on the wheel (``run_check(tuning) -> CheckResult | None``): 20 % gentler while it is
    rough, half as stiff again on its design inertia while a sticky wheel does not follow the small step, 25 % faster
    while it is clean with room; at most CHECKS_MAX runs. The fastest design that passed, and every run."""
    best, tried, cur = None, [], choice
    for _ in range(CHECKS_MAX):
        res = run_check(tuning_of(cur, cur.bandwidth_hz))
        tried.append((cur, res))
        say(f"  check at {cur.bandwidth_hz:.2f} Hz on {cur.inertia:.0f}: " + (
            f"chatter {res.chatter:.0f}, overshoot {res.overshoot:.0%}, {res.crossings} swings, followed the step "
            f"{res.reached:.0%}, rests {res.rest_error * 1000:.1f} mlock off: {res.why}" if res else "stopped"))
        if res is not None and res.passed:
            if best is None or cur.bandwidth_hz > best.bandwidth_hz:
                best = cur
            up = replace(cur, bandwidth_hz=round(cur.bandwidth_hz * STEP_UP, 2))
            failed = [c.bandwidth_hz for c, r in tried if not (r and r.passed) and c.inertia == cur.inertia]
            if not res.room or up.bandwidth_hz > BANDWIDTH_MAX or any(b <= up.bandwidth_hz + 1e-6 for b in failed):
                break
            cur = up
            continue
        if res is not None and res.stiffer:
            cur = replace(cur, inertia=round(cur.inertia * STIFFER))
            continue
        if best is not None:
            break
        # Gentler keeps the stiffness a sticky wheel needs for a small step: more design inertia, so more damping.
        cur = _floored(replace(cur, bandwidth_hz=round(cur.bandwidth_hz * STEP_DOWN, 2)), breakaway)
        if cur.bandwidth_hz < BANDWIDTH_MIN:
            break
    if best is not None and best.ff_friction > 0.0:
        # The friction a sticky wheel reads varies a quarter from run to run (the old wheel 210 to 336): the wheel
        # decides whether half the push is cleaner.
        passed = [r for c, r in tried if c == best][0]
        half = _floored(replace(best, ff_friction=round(best.ff_friction * HALF_PUSH, 1)), breakaway)
        res = run_check(tuning_of(half, half.bandwidth_hz))
        tried.append((half, res))
        say(f"  check with half the friction push ({half.ff_friction:.0f}) on {half.inertia:.0f}: " + (
            f"overshoot {res.overshoot:.0%}, {res.crossings} swings: {res.why}" if res else "stopped"))
        if res is not None and res.passed and res.overshoot <= passed.overshoot - HALF_PUSH_GAIN:
            best = half
    return best, tried


STICK_SPEED_AT_CREEP = 0.462
FRICTION_SHARE = 0.5
READING_JITTER_S = 0.004
CHATTER_MAX = 150.0


def rim_of(p: WheelProfile) -> WheelParams | None:
    """The model rim a pinned profile describes, for the design search; None when not pinned. With the creep's slow
    friction the rim's friction rises from it to the fast slides' (the plant's ``stick_speed``); its jitter measured."""
    if not p.pinned:
        return None
    spread = max(0.0, min(0.5, p.breakaway_top / p.breakaway - 1.0)) if p.breakaway > 0.0 else 0.0
    stick, rise = p.slide, 0.0
    if 0.0 < p.slow_friction < p.slide:
        rise = (p.slide - p.slow_friction) / (1.0 - STICK_SPEED_AT_CREEP)
        stick = max(0.3 * p.slow_friction, p.slide - rise)
        rise = p.slide - stick
    return WheelParams(inertia=p.inertia, damping=p.damping, stick=stick, stick_speed=rise, spring=p.spring,
                       delay_s=p.delay_s, quantum=p.quantum, reading_hz=p.reading_hz,
                       breakaway=max(p.breakaway, p.slide), breakaway_spread=spread,
                       reading_jitter_s=p.jitter_s or READING_JITTER_S)


def push_of(p: WheelProfile) -> dict:
    """The push ahead: the rim's inertia and damping and the friction a slow move meets (the creep's, else half the
    fast slides', never over the slides'), cut on a sticky wheel by its sliding over its breaking force, a delay early.
    A push over the sliding friction is already there as a sticky wheel breaks free: the old wheel lurched and rang on
    180 and 240 and was clean on 120 (2026-11-09). No breakaway push: on that wheel it changed nothing but ringing."""
    friction = min(p.slow_friction or FRICTION_SHARE * p.slide, p.slide)
    if p.breakaway > p.slide > 0.0:
        friction *= p.slide / p.breakaway
    return {"ff_inertia": p.inertia, "ff_damping": p.damping, "ff_friction": friction, "plan_s": p.delay_s}


STEP_MIN = 0.02


def kp_floor_of(p: WheelProfile, push: dict) -> float:
    """The stiffness that moves a STEP_MIN step past the breakaway with the friction push: the old wheel waited out
    its 0.02 steps entirely under about 16000 per lock (2026-11-09)."""
    return max(0.0, p.breakaway - push["ff_friction"]) / STEP_MIN


def design(p: WheelProfile) -> tuple[Choice, float | None] | None:
    """The design for the calibrated rim in seconds (``fast_design``) and its chatter in the model; None unpinned."""
    rim = rim_of(p)
    if rim is None:
        return None
    push = push_of(p)
    return fast_design(rim, push, p.ffb_hz or 120.0, CHATTER_MAX, kp_floor=kp_floor_of(p, push))





def save(p: WheelProfile, choice: Choice) -> dict:
    """The design's ``la_*`` keys and the profile written to config.json; the learner's design steps start over."""
    from core.settings import Settings

    Settings.load()
    learned = dict(getattr(Settings, "la_wheel_learned", None) or {})
    learned.update(rung=0, calm_s=0.0)
    values = {**choice.config(), "la_wheel_profile": p.to_dict(), "la_wheel_learned": learned}
    Settings.save(values=values)
    return values


def describe(p: WheelProfile) -> str:
    steps = f"in steps of {p.quantum * 1000:.2f} mlock" if p.quantum else "finely"
    lines = [f"reads the wheel {p.reading_hz:.0f} times a second {steps}",
             f"starts at {p.breakaway:.0f} (up to {p.breakaway_top:.0f}; the test pushed {p.pushed_start:.0f})"]
    if p.pinned:
        lines.append(f"rim: inertia {p.inertia:.0f}, damping {p.damping:.0f}, slides on {p.slide:.0f}, "
                     f"spring {p.spring:.0f}, delay {p.delay_s * 1000:.0f} ms (fit off by {p.fit_rms:.1%})")
    elif p.fitted:
        lines.append(f"rim: damping {p.damping:.0f}, slides on {p.slide:.0f}, spring {p.spring:.0f}; inertia and delay "
                     f"not pinned by these moves (fit off by {p.fit_rms:.1%})")
    else:
        lines.append(f"rim: no fit (off by {p.fit_rms:.1%})")
    if p.play:
        lines.append(f"play in the drive: {p.play * 1000:.1f} mlock")
    return "\n".join(lines)
