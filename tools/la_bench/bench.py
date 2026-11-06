"""The bench: runs stages through a plugin channel (or a model wheel) with the clock it is given.

``Bench`` keeps the wheel safe: the servo goes off when the plugin is lost or the wheel runs past the angle limit, the
wheel is walked to the centre before the first stage and between stages on a rounded move, and a stage ends on a
runaway, not the run.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from core.lane_departure.command import SERVO_OFF, VERSION, PluginStatus, ServoCommand
from tools.la_bench.plant import SimChannel, SimClock, WheelParams
from tools.la_bench.protocol import Design, Stage, accel_changes, move_s, rounded_move

ANGLE_LIMIT = 0.4
ANGLE_MARGIN = 0.15
CENTRE = 0.0
CENTRE_LIMIT = 0.25
REST_LIMIT = 0.6
HOME_BAND = 0.01
HOME_RATE = 0.3
HOME_CORNER_S = 0.3
HOME_SEND_S = 0.02
HOME_SETTLE_S = 0.3
HOME_SPARE_S = 3.0
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
        self.damping_first = False
        self.push_ahead = False
        self.ff_plan = False
        self._readings: list[tuple[float, int]] = []

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
        now = self.clock()
        if not self._readings or st.steer_frames < self._readings[-1][1]:
            self._readings = [(now, st.steer_frames)]
        else:
            self._readings = [self._readings[0], (now, st.steer_frames)]
        return st

    @property
    def reading_hz(self) -> float:
        """How often the game reads the wheel angle, from the steering frame count (0 until a second has passed)."""
        if len(self._readings) < 2 or self._readings[-1][0] - self._readings[0][0] < 1.0:
            return 0.0
        (t0, f0), (t1, f1) = self._readings[0], self._readings[-1]
        return (f1 - f0) / (t1 - t0)

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

    def prepare(self, direction: int | None = None, damping_first: bool = False, push_ahead: bool = False,
                walk: Design | None = None) -> tuple[PluginStatus, int, float]:
        """The plugin ready, the torque sign that turns the wheel counterclockwise, and the wheel walked to the centre,
        which is home from then on. ``walk`` lends the walk its push ahead.
        """
        st = self.ready()
        self.damping_first, self.push_ahead, self.ff_plan = st.damping_first, st.push_ahead, st.ff_plan
        self.say(f"plugin present, version {st.version}, {st.frame_hz:.0f} Hz FFB: hands off the wheel")
        if damping_first and not st.damping_first:
            raise BenchAbort("the plugin cannot damp first: rebuild plugins/ffb and restart the game")
        if push_ahead and not st.push_ahead:
            raise BenchAbort("the plugin cannot push ahead: install the layout 5 DLL and restart the game")
        direction = direction or self.find_direction()
        self.say(f"positive torque turns the wheel {'counterclockwise' if direction > 0 else 'clockwise'}")
        rest = self.rest()
        if abs(rest) > REST_LIMIT:
            raise BenchAbort(f"the wheel rests {rest:+.2f} lock off centre: centre it and rerun")
        self.go_home(CENTRE, direction, walk)
        return st, direction, CENTRE

    def find_direction(self) -> int:
        """Push a little torque with the servo off: +1 if positive torque turns the wheel counterclockwise."""
        start = self.rest()
        moved = self.idle(DIRECTION_S, DIRECTION_UNITS).steer - start
        self.idle(0.0)
        if abs(moved) < DIRECTION_MIN_MOVE:
            raise BenchAbort("the wheel did not move under torque: hands on the rim, or the game paused? (--direction skips the test)")
        return 1 if moved > 0 else -1

    def go_home(self, home: float, direction: int, design: Design | None = None) -> None:
        """Walk a wheel that rests away from ``home`` there on a rounded move (it speeds up and slows down over 0.3 s,
        at most 0.3 lock/s) with a gentle servo, damping first and with ``design``'s push ahead where the plugin can,
        sent every 20 ms with the move's rate and acceleration; let go once it has sat there for 0.3 s.
        """
        start = self.rest()
        dist = abs(home - start)
        if dist <= HOME_BAND:
            return
        walk = HOME_DESIGN
        if self.damping_first:
            walk = Design(bandwidth_hz=walk.bandwidth_hz, cap=walk.cap, damping_first=True)
        if design is not None and design.pushes_ahead and self.push_ahead:
            walk = Design(bandwidth_hz=walk.bandwidth_hz, cap=walk.cap, damping_first=walk.damping_first,
                          ff_inertia=design.ff_inertia, ff_damping=design.ff_damping, ff_friction=design.ff_friction)
        kp, kd, ki = walk.gains()
        dur = move_s(dist, HOME_RATE, HOME_CORNER_S)
        marks = [m for m in accel_changes(dur, dist, HOME_RATE, HOME_CORNER_S) if m > 0.0] + [dur]
        limit = max(ANGLE_LIMIT, abs(start), abs(home)) + ANGLE_MARGIN
        t0 = self.clock()
        still = None
        try:
            while (t := self.clock() - t0) < dur + HOME_SPARE_S:
                aim, rate, accel = rounded_move(t, dur, start, home, HOME_RATE, HOME_CORNER_S)
                servo = ServoCommand(target=aim, rate=rate, kp=kp, kd=kd, ki=ki, cap=walk.cap, direction=direction,
                                     damping_first=walk.damping_first, accel=accel, ff_inertia=walk.ff_inertia,
                                     ff_damping=walk.ff_damping, ff_friction=walk.ff_friction)
                st = self.status(servo, limit=limit)
                near = t >= dur and abs(st.steer - home) <= HOME_BAND / 2
                still = (t if still is None else still) if near else None
                if still is not None and t - still >= HOME_SETTLE_S:
                    break
                nxt = next((m for m in marks if m > t + 1e-6), None)
                self.sleep(min(HOME_SEND_S, nxt - t) if nxt is not None else HOME_SEND_S)
        finally:
            self.channel.publish(False, 0, 0, servo=SERVO_OFF)

    def run_stage(self, stage: Stage, design: Design, direction: int, hz: float = 20.0, home: float | None = None) -> dict:
        if home is not None:
            self.go_home(home, direction, design)
        start = self.rest()
        if abs(start) > CENTRE_LIMIT:
            raise BenchAbort(f"the wheel rests {start:+.2f} lock off centre: centre it and rerun")
        limit = max(ANGLE_LIMIT, abs(start) + stage.amp + ANGLE_MARGIN)
        kp, kd, ki = design.gains()
        plan = design.plan_s if design.pushes_ahead else 0.0
        events = stage.events(hz, plan)
        servo = SERVO_OFF
        sent, samples = [], []
        aborted = None
        last = None
        t0 = self.clock()
        i = 0
        try:
            while (t := self.clock() - t0) < stage.duration:
                while i < len(events) and events[i] <= t:
                    offset, rate, accel = stage.aim3(events[i])
                    _, plan_rate, plan_accel = stage.aim3(events[i] + plan) if plan else (0.0, 0.0, 0.0)
                    servo = ServoCommand(target=start + offset, rate=rate, kp=kp, kd=kd, ki=ki, cap=design.cap,
                                         direction=direction, damping_first=design.damping_first, accel=accel,
                                         ff_inertia=design.ff_inertia, ff_damping=design.ff_damping,
                                         ff_friction=design.ff_friction, lead_s=design.lead_s, ff_plan=bool(plan),
                                         plan_rate=plan_rate, plan_accel=plan_accel)
                    sent.append([t, servo.target, rate, accel])
                    i += 1
                st = self.status(servo, limit=limit)
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
                      "damping_first": design.damping_first, "ff_inertia": design.ff_inertia,
                      "ff_damping": design.ff_damping, "ff_friction": design.ff_friction, "lead_s": design.lead_s},
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
