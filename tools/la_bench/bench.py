"""The bench: runs stages through a plugin channel (or a model wheel) with the clock it is given.

``Bench`` keeps the wheel safe: the servo goes off when the plugin is lost or the wheel runs past the angle limit, a
wheel that rests off its starting angle is walked home between stages, and a stage ends on a runaway, not the run.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from core.lane_departure.command import SERVO_OFF, VERSION, PluginStatus, ServoCommand
from tools.la_bench.plant import SimChannel, SimClock, WheelParams
from tools.la_bench.protocol import Design, Stage

ANGLE_LIMIT = 0.4
CENTRE_LIMIT = 0.25
HOME_BAND = 0.01
HOME_RATE = 0.15
HOME_MAX_S = 6.0
HOME_DESIGN = Design(bandwidth_hz=1.0, cap=2000)
POLL_S = 0.001
DIRECTION_UNITS = 600
DIRECTION_S = 0.5
DIRECTION_MIN_MOVE = 0.004


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

    def prepare(self, direction: int | None = None, damping_first: bool = False) -> tuple[PluginStatus, int, float]:
        """The plugin ready, the torque sign that turns the wheel counterclockwise and the angle the wheel rests at."""
        st = self.ready()
        self.say(f"plugin present, version {st.version}, {st.frame_hz:.0f} Hz FFB: hands off the wheel")
        if damping_first and not st.damping_first:
            raise BenchAbort("the plugin cannot damp first: rebuild plugins/ffb and restart the game")
        direction = direction or self.find_direction()
        self.say(f"positive torque turns the wheel {'counterclockwise' if direction > 0 else 'clockwise'}")
        home = self.rest()
        if abs(home) > CENTRE_LIMIT:
            raise BenchAbort(f"the wheel rests {home:+.2f} lock off centre: centre it and rerun")
        return st, direction, home

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


def sim_bench(params: WheelParams, frame_hz: float = 120.0) -> Bench:
    clock = SimClock()
    return Bench(SimChannel(params, clock, frame_hz=frame_hz), clock, clock.sleep)
