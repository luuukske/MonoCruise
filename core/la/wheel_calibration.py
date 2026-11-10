"""Wheel calibration: a short open-loop measurement of the standing wheel, so LA and the lane keeping assist know the
rim before the first drive instead of guessing it from hunts. Pure logic: the caller feeds it the plugin's trace
samples and sends the torque ``step`` asks for and the servo in ``servo``. See core/la/README.md, Wheel calibration.

Truck parked, hands off the rim. Six trials, their ways alternating so the wheel ends near where it started: the
torque climbs until the wheel starts (its breakaway), then holds a share of that push (half, to see whether the wheel
keeps sliding, then 1.25 and 1.6 times, to see it accelerate and slide). One least-squares fit over every trial's
readings after its start, push = inertia * accel + damping * rate + sliding friction + spring * angle, scanned over
the delay between the push and the rim, gives inertia, damping, sliding friction, spring and delay. Last, a relay:
the plugin's servo saturated round the resting angle for a second, so the wheel swings a few mlock at the rate where
its delay counts most; its readings join the fit. Each trial's travel is capped, and the relay's.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from dataclasses import asdict, dataclass, field, fields

from core.la.wheel_learner import quantum_of
from core.lane_departure.command import SERVO_OFF, ServoCommand

REST, RAMP, HOLD, RELEASE, CREEP, RELAY, RELAY_OUT, DONE, FAILED = (
    "rest", "ramp", "hold", "release", "creep", "relay", "relay_out", "done", "failed")


@dataclass(frozen=True)
class CalTuning:
    ramp: float = 400.0
    ramp_max: float = 1500.0
    start_move: float = 0.002
    shares: tuple[float, ...] = (0.5, 0.5, 1.25, 1.25, 1.6, 1.6)
    wobble: float = 0.25
    wobble_steps: int = 1
    hold_s: float = 0.4
    travel: float = 0.12
    rest_s: float = 0.3
    rest_band: float = 0.0015
    rest_max_s: float = 4.0
    settle_s: float = 0.15
    settle_max_s: float = 1.5
    stuck_s: float = 0.5
    start_min: float = 50.0
    centre_max: float = 0.25
    angle_max: float = 0.45
    stall_s: float = 0.25
    retries: int = 2
    total_max_s: float = 45.0
    delay_max_s: float = 0.08
    delay_step_s: float = 0.002
    grid_s: float = 0.001
    stop_readings: int = 3
    stop_rate: float = 0.02
    confirm: float = 0.01
    climbs: int = 3
    fit_rms_max: float = 0.25
    pin_ratio: float = 1.05
    delay_prior_s: float = 0.02
    inertia_range: tuple[float, float] = (5.0, 3000.0)
    damping_max: float = 8000.0
    slide_max: float = 3000.0
    relay_share: float = 1.3
    relay_s: float = 1.0
    relay_band: float = 0.0005
    relay_offset: float = 0.002
    relay_travel: float = 0.03
    relay_window_s: float = 0.25
    relay_min_readings: int = 6
    play_min: float = 0.001
    creep_rate: float = 0.05
    creep_travel: float = 0.06
    creep_hold_s: float = 0.3
    creep_settle: float = 0.4
    creep_bandwidth_hz: float = 1.5
    creep_push_share: float = 0.6
    creep_inertia_min: float = 100.0
    creep_cap: int = 1500
    jitter_half: int = 3
    jitter_rate_min: float = 0.05
    jitter_leverage: float = 1.4
    jitter_scale: float = 4.8
    relay_score_s: float = 0.65
    relay_pin_ratio: float = 1.5
    relay_h_s: float = 0.0005
    servo_in_s: float = 0.3


@dataclass(frozen=True)
class WheelProfile:
    """What the calibration measured; 0 is unmeasured. Forces in the game's units, angles in lock, rates in lock/s.

    ``direction`` is the torque sign that turns the reading up. ``breakaway`` is the push on the rim (the plugin's
    output, the game's own force included) that started it, ``pushed_start`` the calibration's own torque then: their
    gap is the game's standing friction, which is gone once the truck rolls. ``fit_rms`` is the fit's residual as a
    share of the pushes it explains.
    """

    direction: int = 0
    reading_hz: float = 0.0
    quantum: float = 0.0
    breakaway: float = 0.0
    breakaway_top: float = 0.0
    pushed_start: float = 0.0
    slide: float = 0.0
    damping: float = 0.0
    inertia: float = 0.0
    spring: float = 0.0
    delay_s: float = 0.0
    play: float = 0.0
    fit_rms: float = 0.0
    trials: int = 0
    ffb_hz: float = 0.0
    slow_friction: float = 0.0
    jitter_s: float = 0.0

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, raw: object) -> WheelProfile:
        """A saved profile; anything unreadable is unmeasured, and a profile that does not read as pinned is none."""
        if not isinstance(raw, dict):
            return cls()
        out = {}
        for f in fields(cls):
            v = raw.get(f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 and f.name != "direction":
                continue
            out[f.name] = int(v) if f.type in ("int", int) else float(v)
        return cls(**out)

    @property
    def fitted(self) -> bool:
        return self.damping > 0.0

    @property
    def pinned(self) -> bool:
        """Inertia and delay known too: on the old wheel (2026-11-09) the moves gave its damping and friction only."""
        return self.fitted and self.inertia > 0.0 and self.delay_s > 0.0


@dataclass
class _Trial:
    way: int
    share: float
    begin: float
    rest: float
    first: int
    started: float | None = None
    pushed: float = 0.0
    level: float = 0.0
    released: float | None = None
    end: int = 0
    play: float = 0.0
    climbs: int = 0


@dataclass
class WheelCalibration:
    """Steps at the caller's rate (20 Hz in the lane departure thread); ``feed`` takes trace samples as they come."""

    tune: CalTuning = field(default_factory=CalTuning)
    state: str = REST
    failure: str = ""
    profile: WheelProfile | None = None
    servo: ServoCommand = SERVO_OFF

    def __post_init__(self) -> None:
        self._t: list[float] = []
        self._steer: list[float] = []
        self._out: list[float] = []
        self._game: list[float] = []
        self._seq: list[int] = []
        self._trials: list[_Trial] = []
        self._trial: _Trial | None = None
        self._direction = 0
        self._torque = 0.0
        self._retried = 0
        self._phase_at: float | None = None
        self._began: float | None = None
        self._fresh_at: float | None = None
        self._fed_at: float | None = None
        self._seen = 0
        self._held = 0
        self._relay: tuple[int, int, float] | None = None
        self._relay_from = 0
        self._creep: list[int] = []
        self._creep_rest = 0.0
        self._wobble = 0.0

    def feed(self, samples) -> None:
        """Trace samples with ``t``, ``steer``, ``out`` and ``steer_seq`` (``core.lane_departure.ffb_trace``); ``game`` is
        kept for the record."""
        if self.state in (DONE, FAILED):
            return
        for s in samples:
            if self._t and s.t <= self._t[-1]:
                continue
            if not self._seq or s.steer_seq != self._seq[-1]:
                self._fresh_at = s.t
            self._t.append(s.t)
            self._steer.append(s.steer)
            self._out.append(float(s.out))
            self._game.append(float(getattr(s, "game", 0)))
            self._seq.append(s.steer_seq)

    def abort(self, why: str) -> float:
        self.state, self.failure, self._torque, self.servo = FAILED, why, 0.0, SERVO_OFF
        return 0.0

    def step(self, now: float) -> float:
        """The torque to send until the next step, signed as the game's force; 0 once done or failed."""
        tn = self.tune
        if self.state in (DONE, FAILED):
            return 0.0
        if self._phase_at is None:
            self._phase_at = self._began = now
        if now - self._began > tn.total_max_s:
            return self.abort(f"not done in {tn.total_max_s:.0f} s")
        if not self._t:
            return self.abort("no wheel readings from the FFB plugin") if now - self._phase_at > tn.rest_max_s else 0.0
        if len(self._t) != self._seen or self._fed_at is None:
            self._seen, self._fed_at = len(self._t), now
        elif now - self._fed_at > tn.stall_s and self.state != REST:
            return self._retry(now, "the game stopped updating the wheel (paused, or another window in front?)")
        steer = self._steer[-1]
        if abs(steer) > tn.angle_max:
            return self.abort(f"the wheel went past {tn.angle_max:.2f} lock")
        if self._fresh_at is not None and self._t[-1] - self._fresh_at > tn.stall_s and self.state != REST:
            return self._retry(now, "the wheel readings stalled")
        if self.state == REST:
            return self._rest(now, steer)
        if self.state == CREEP:
            return self._creep_step(now)
        if self.state == RELAY:
            target = self.servo.target
            if now - self._phase_at >= tn.relay_s or abs(steer - target) > tn.relay_travel:
                self._relay = (self._relay_from, len(self._t), self._steer[self._relay_from - 1])
                self.state, self._phase_at, self.servo = RELAY_OUT, now, SERVO_OFF
            return 0.0
        if self.state == RELAY_OUT:
            if self._still(tn.settle_s) or now - self._phase_at > tn.settle_max_s:
                self._finish()
            return 0.0
        tr = self._trial
        if self.state == RAMP:
            moved = steer - tr.rest
            way = tr.way if self._direction else (1 if moved > 0.0 else -1)
            if abs(moved) >= tn.start_move and abs(self._torque) < tn.start_min:
                self._trial, self.state, self._phase_at, self._torque = None, REST, now, 0.0
                return 0.0
            if abs(moved) >= tn.start_move and (moved * tr.way > 0.0 or not self._direction):
                if not self._direction:
                    self._direction, tr.way = way, way
                tr.started, tr.pushed = now, abs(self._torque)
                tr.level = tr.share * tr.pushed
                self.state, self._phase_at = HOLD, now
                self._torque = self._sign(tr.way) * tr.level
                return self._torque
            want = min(tn.ramp_max, tn.ramp * (now - tr.begin))
            if want >= tn.ramp_max and now - tr.begin > tn.ramp_max / tn.ramp + tn.stuck_s:
                return self.abort("the wheel did not move under the test push: hands on the rim, or the game paused?")
            self._torque = self._sign(tr.way) * want
            return self._torque
        if self.state == HOLD:
            if now - self._phase_at >= tn.hold_s or abs(steer - tr.rest) > tn.travel:
                tr.released = now
                self.state, self._phase_at, self._torque = RELEASE, now, 0.0
                return 0.0
            moved = abs(steer - tr.rest)
            if tr.share >= 1.0 and moved < tn.confirm and now - self._phase_at >= tn.settle_s and self._still(tn.settle_s):
                if tr.climbs >= tn.climbs:
                    return self.abort("the wheel keeps stopping under more push than started it")
                # Stopped under more than its start: play in the drive taken up; climb again from here.
                tr.play, tr.climbs = max(tr.play, moved), tr.climbs + 1
                tr.rest, tr.first, tr.started, tr.begin = steer, len(self._t), None, now - tr.level / tn.ramp
                self.state, self._phase_at, self._wobble = RAMP, now, 0.0
                return self._torque
            if tr.share >= 1.0:
                # Switching the push every step shows the delay as phase: a smooth push hid it on a heavy rim.
                self._held += 1
                if not self._wobble or self._held >= tn.wobble_steps:
                    self._wobble, self._held = (-self._wobble if self._wobble else tn.wobble), 0
                self._torque = self._sign(tr.way) * tr.level * (1.0 + self._wobble)
            return self._torque
        if self._still(tn.settle_s) or now - self._phase_at > tn.settle_max_s:
            tr.end = len(self._t)
            self._trials.append(tr)
            self._trial = None
            self.state, self._phase_at = REST, now
        return 0.0

    def _sign(self, way: int) -> int:
        return way * self._direction if self._direction else 1

    def _rest(self, now: float, steer: float) -> float:
        tn = self.tune
        if not self._still(tn.rest_s):
            if now - self._phase_at > tn.rest_max_s:
                return self.abort("the wheel would not stand still: hands off the rim")
            return 0.0
        if not self._trials and self._trial is None and abs(steer) > tn.centre_max:
            return self.abort(f"the wheel is {abs(steer):.2f} lock off centre: centre it and start again")
        i = len(self._trials)
        if i == len(tn.shares):
            return self._start_creep(now, steer) if len(self._creep) < 4 else self._start_relay(now, steer)
        way = (1 if i % 2 == 0 else -1) * (self._direction or 1)
        self._trial = _Trial(way=way, share=tn.shares[i], begin=now, rest=steer, first=len(self._t))
        self.state, self._phase_at = RAMP, now
        return 0.0

    def _start_creep(self, now: float, steer: float) -> float:
        """A slow move out and back on a gentle servo: the push it takes to creep is the friction a slow move meets,
        which the fast slides over-read (the old wheel rang on a push of that, 2026-11-09)."""
        from core.la.actuator import WheelActuator, WheelTuning

        tn = self.tune
        if not self._direction:
            return self._start_relay(now, steer)
        inertia, slide = _rough_rim(self._t, self._steer, self._out, self._seq, self._trials, self._direction, tn)
        kp, kd, ki = WheelActuator(WheelTuning(inertia=max(tn.creep_inertia_min, inertia),
                                               bandwidth_hz=tn.creep_bandwidth_hz, damping=1.0,
                                               damping_first=True)).gains(1.0)
        # A friction push under the fast slides' friction: alone, the gentle servo never moved a sticky wheel.
        self._creep_gains = (kp, kd, ki, tn.creep_push_share * slide)
        self._creep, self._creep_rest = [len(self._t)], steer
        self.state, self._phase_at = CREEP, now
        return self._creep_step(now)

    def _creep_step(self, now: float) -> float:
        tn = self.tune
        ramp_s = tn.creep_travel / tn.creep_rate
        u = now - self._phase_at
        marks = (ramp_s, ramp_s + tn.creep_hold_s, 2.0 * ramp_s + tn.creep_hold_s)
        while len(self._creep) <= len(marks) and u >= marks[len(self._creep) - 1]:
            self._creep.append(len(self._t))
        if u >= marks[-1] + tn.creep_hold_s:
            self.servo = SERVO_OFF
            self.state, self._phase_at = REST, now
            return 0.0
        if u < marks[0]:
            offset, rate = tn.creep_rate * u, tn.creep_rate
        elif u < marks[1]:
            offset, rate = tn.creep_travel, 0.0
        elif u < marks[2]:
            offset, rate = tn.creep_travel - tn.creep_rate * (u - marks[1]), -tn.creep_rate
        else:
            offset, rate = 0.0, 0.0
        kp, kd, ki, push = self._creep_gains
        self.servo = ServoCommand(target=self._creep_rest + offset, rate=rate, kp=kp, kd=kd, ki=ki, cap=tn.creep_cap,
                                  direction=self._direction, damping_first=True, ff_friction=push)
        return 0.0

    def _start_relay(self, now: float, steer: float) -> float:
        tn = self.tune
        pushed = sorted(tr.pushed for tr in self._trials if tr.started is not None and tr.share >= 1.0)
        if not self._direction or not pushed:
            self._finish()
            return 0.0
        cap = min(tn.ramp_max, tn.relay_share * pushed[len(pushed) // 2])
        target = steer + tn.relay_offset * self._direction
        self.servo = ServoCommand(target=target, kp=cap / tn.relay_band, cap=round(cap), direction=self._direction)
        self._relay_from = len(self._t)
        self.state, self._phase_at = RELAY, now
        return 0.0

    def _still(self, seconds: float) -> bool:
        t_end = self._t[-1]
        if t_end - self._t[0] < seconds:
            return False
        i = bisect_right(self._t, t_end - seconds)
        tail = self._steer[max(0, i - 1):]
        return max(tail) - min(tail) <= self.tune.rest_band

    def _retry(self, now: float, why: str) -> float:
        if self._retried >= self.tune.retries:
            return self.abort(f"{why} again and again: keep the game in front and close other programs")
        self._retried += 1
        self._trial = None
        self.state, self._phase_at, self._torque, self.servo = REST, now, 0.0, SERVO_OFF
        return 0.0

    def _finish(self) -> None:
        self.profile = analyse(self._t, self._steer, self._out, self._seq, self._trials, self._direction, self.tune,
                               self._relay, self._creep)
        if not self.profile.fitted:
            self.state, self.failure = FAILED, "the wheel's moves did not fit a rim: try again with hands off the rim"
        else:
            self.state = DONE


def _fresh(t: list[float], steer: list[float], seq: list[int], lo: int, hi: int) -> tuple[list[float], list[float]]:
    ts, xs = [], []
    for i in range(lo, hi):
        if i == 0 or seq[i] != seq[i - 1]:
            ts.append(t[i])
            xs.append(steer[i])
    return ts, xs


class _Push:
    """The push in the reading's direction, held between updates, and its first and second integrals on a grid."""

    def __init__(self, t: list[float], out: list[float], direction: int, grid_s: float) -> None:
        self.t0, self.h = t[0], grid_s
        n = int((t[-1] - t[0]) / grid_s) + 2
        g = [0.0] * n
        k = [0.0] * n
        j = 0
        for i in range(1, n):
            tau = self.t0 + (i - 1) * grid_s
            while j + 1 < len(t) and t[j + 1] <= tau:
                j += 1
            f = direction * out[j]
            g[i] = g[i - 1] + f * grid_s
            k[i] = k[i - 1] + 0.5 * (g[i] + g[i - 1]) * grid_s
        self.g, self.k = g, k
        self._t, self._f = t, [direction * o for o in out]

    def _at(self, arr: list[float], tau: float) -> float:
        x = (tau - self.t0) / self.h
        if x <= 0.0:
            return arr[0]
        i = int(x)
        if i + 1 >= len(arr):
            return arr[-1]
        return arr[i] + (x - i) * (arr[i + 1] - arr[i])

    def force(self, tau: float) -> float:
        j = max(0, bisect_right(self._t, tau) - 1)
        return self._f[j]

    def double(self, t1: float, t: float, d: float) -> float:
        """The push's double integral from ``t1`` to ``t``, ``d`` seconds late."""
        return self._at(self.k, t - d) - self._at(self.k, t1 - d) - (t - t1) * self._at(self.g, t1 - d)


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    m = [row[:] + [v] for row, v in zip(a, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12 * max(1.0, abs(m[c][c])):
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _still_s(tune: CalTuning, quantum: float, reading_hz: float) -> float:
    """Stopped: no new step for as long as a wheel at stop_rate takes to cross one."""
    return max(tune.stop_readings / max(reading_hz, 1.0), quantum / tune.stop_rate)


def _segments(t, steer, seq, trials: list[_Trial], tune: CalTuning, quantum: float,
              reading_hz: float) -> list[tuple[list[float], list[float], list[float]]]:
    """Per trial, the fresh readings from its first clear move until the wheel stopped or turned back, and the
    friction column (its way times t^2 / 2)."""
    out = []
    for tr in trials:
        if tr.started is None:
            continue
        ts, xs = _fresh(t, steer, seq, tr.first, tr.end)
        k0 = next((i for i, x in enumerate(xs) if abs(x - tr.rest) >= tune.start_move), None)
        if k0 is None:
            continue
        still_s = _still_s(tune, quantum, reading_hz)
        k1, moved = len(xs), k0
        for i in range(k0 + 1, len(xs)):
            if (xs[i] - xs[i - 1]) * tr.way < 0.0:
                k1 = i - 1
                break
            if xs[i] != xs[i - 1]:
                moved = i
            elif ts[i] - ts[moved] >= still_s:
                k1 = moved + 1
                break
        if k1 - k0 >= 4:
            ts, xs = ts[k0:k1], xs[k0:k1]
            out.append((ts, xs, [tr.way * 0.5 * (tt - ts[0]) ** 2 for tt in ts]))
    return out


def _relay_segments(t, steer, seq, relay: tuple[int, int, float] | None, tune: CalTuning, quantum: float,
                    reading_hz: float) -> list[tuple[list[float], list[float], list[float]]]:
    """The relay's readings in windows, each with its friction column: the double integral of the way the wheel
    moved, reading to reading (an unchanged reading keeps the last way). Stretches where it stuck are left out."""
    if relay is None:
        return []
    ts, xs = _fresh(t, steer, seq, relay[0], relay[1])
    k0 = next((i for i, x in enumerate(xs) if x != relay[2]), None)
    if k0 is None:
        return []
    still_s, runs, start, moved = _still_s(tune, quantum, reading_hz), [], k0, k0
    for i in range(k0 + 1, len(xs)):
        if xs[i] != xs[i - 1]:
            if ts[i - 1] - ts[moved] >= still_s:
                runs.append((start, moved + 1))
                start = i - 1
            moved = i
    runs.append((start, moved + 1))
    out = []
    for i, k1 in runs:
        out += _windows(ts[i:k1], xs[i:k1], tune)
    return out


def _windows(ts: list[float], xs: list[float], tune: CalTuning) -> list[tuple[list[float], list[float], list[float]]]:
    out, i = [], 0
    while i < len(ts):
        j = i
        while j + 1 < len(ts) and ts[j + 1] - ts[i] <= tune.relay_window_s:
            j += 1
        if j - i + 1 >= tune.relay_min_readings:
            wt, wx = ts[i:j + 1], xs[i:j + 1]
            way = 1.0 if wx[1] >= wx[0] else -1.0
            s1 = s2 = 0.0
            fric = [0.0]
            for k in range(1, len(wt)):
                dx, dt = wx[k] - wx[k - 1], wt[k] - wt[k - 1]
                way = way if dx == 0.0 else math.copysign(1.0, dx)
                s2 += s1 * dt + 0.5 * way * dt * dt
                s1 += way * dt
                fric.append(s2)
            out.append((wt, wx, fric))
        i = j + 1
    return out


def _fit(segs, push: _Push, d: float) -> tuple[list[float], float, float] | None:
    """Least squares of the push's double integral on (angle, its integral, the friction column, its double
    integral, one start-rate column per segment): inertia, damping, friction, spring; the residual and the size of
    the pushes."""
    n = 4 + len(segs)
    a = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    ss = yy = 0.0
    rows = []
    for j, (ts, xs, fric) in enumerate(segs):
        t1, x1 = ts[0], xs[0]
        ia = iia = 0.0
        for i, (tt, x) in enumerate(zip(ts, xs)):
            if i:
                dt = tt - ts[i - 1]
                prev = ia
                ia += 0.5 * ((x - x1) + (xs[i - 1] - x1)) * dt
                iia += 0.5 * (ia + prev) * dt
            u = tt - t1
            row = [0.0] * n
            row[0], row[1], row[2], row[3] = x - x1, ia, fric[i], iia + x1 * 0.5 * u * u
            row[4 + j] = u
            y = push.double(t1, tt, d)
            rows.append((row, y))
    for row, y in rows:
        for r in range(n):
            if row[r]:
                b[r] += row[r] * y
                for c in range(n):
                    a[r][c] += row[r] * row[c]
        yy += y * y
    sol = _solve(a, b)
    if sol is None:
        return None
    for row, y in rows:
        e = y - sum(c * v for c, v in zip(row, sol))
        ss += e * e
    return sol, ss, yy


def _starts(t, steer, seq, trials: list[_Trial], push: _Push, d: float) -> list[float]:
    """The push on the rim, ``d`` before each hold's wheel first read moved: its breakaway, sorted."""
    out = []
    for tr in trials:
        if tr.started is None or tr.share < 1.0:
            continue
        ts, xs = _fresh(t, steer, seq, tr.first, tr.end)
        k = next((i for i, x in enumerate(xs) if x != tr.rest), None)
        if k is not None:
            out.append(abs(push.force(ts[k] - d)))
    return sorted(out)


def _relay_swings(t, steer, out, seq, relay) -> tuple[list[tuple[float, float]], float] | None:
    """The relay's fresh readings from its rest, timed from its first push flip, and its strongest push."""
    first, end, rest = relay
    flip = next((t[i] for i in range(first + 1, end) if out[i] * out[i - 1] < 0.0 and abs(out[i]) > 200.0), None)
    if flip is None:
        return None
    ts, xs = _fresh(t, steer, seq, first, end)
    return [(tt - flip, x - rest) for tt, x in zip(ts, xs)], max(abs(o) for o in out[first:end])


def simulate_relay(inertia: float, damping: float, slide: float, breakaway: float, delay_s: float, cap: float,
                   quantum: float, reading_hz: float, ffb_hz: float, tune: CalTuning, until_s: float,
                   target: float) -> list[tuple[float, float]] | None:
    """The relay on a model rim, in the reading's direction: the plugin's saturated P at its FFB rate, faded in over
    its servo fade, on a reading held at the game's rate; readings timed from the first push flip, None without one."""
    h, x, v, push, t = tune.relay_h_s, 0.0, 0.0, 0.0, 0.0
    stuck, pending, reading, out, flip, last = True, deque(), 0.0, [], None, 0.0
    next_ffb = next_read = 0.0
    while t < until_s + (flip if flip is not None else until_s):
        if t >= next_read:
            reading = round(x / quantum) * quantum if quantum else x
            out.append((t, reading))
            next_read += 1.0 / reading_hz
        if t >= next_ffb:
            fade = min(1.0, t / tune.servo_in_s)
            u = max(-cap, min(cap, (target - reading) / tune.relay_band * cap)) * (0.5 - 0.5 * math.cos(math.pi * fade))
            if flip is None and u * last < 0.0 and abs(u) > 200.0:
                flip = t
            last = u
            pending.append((t + delay_s, u))
            next_ffb += 1.0 / ffb_hz
        while pending and pending[0][0] <= t:
            push = pending.popleft()[1]
        if stuck and abs(push) <= breakaway:
            t += h
            continue
        stuck = False
        nv = v + (push - damping * v - slide * math.copysign(1.0, v if v else push)) / inertia * h
        if v * nv < 0.0:
            nv, stuck = 0.0, abs(push) <= breakaway
        v, x, t = nv, x + nv * h, t + h
        if t > 3.0:
            break
    if flip is None:
        return None
    return [(tt - flip, xx) for tt, xx in out]


def _relay_delay(t, steer, out, seq, relay, direction, fits, quantum, reading_hz, breakaway, tune) -> int | None:
    """The delay step whose fitted rim swings in the relay as the wheel did, when that is clearly better than either end
    of the scan: the readings' fit alone slid along inertia against delay on the old wheel (2026-11-09)."""
    if relay is None or not fits or not reading_hz:
        return None
    seen = _relay_swings(t, steer, [direction * o for o in out], seq, relay)
    if seen is None:
        return None
    swings, cap = seen
    swings = [(tt, x) for tt, x in swings if -0.05 <= tt <= tune.relay_score_s]
    ffb_hz = _ffb_hz(t)
    scores = []
    for i, f in enumerate(fits):
        sim = None
        if f is not None and f[0][0] > 1.0 and f[0][1] >= 0.0:
            sim = simulate_relay(f[0][0], f[0][1], max(0.0, f[0][2]), breakaway, i * tune.delay_step_s, cap, quantum,
                                 reading_hz, ffb_hz, tune, tune.relay_score_s, tune.relay_offset * direction)
        scores.append(_swing_error(swings, sim) if sim else math.inf)
    k = min(range(len(scores)), key=scores.__getitem__)
    if not math.isfinite(scores[k]) or min(scores[0], scores[-1]) < tune.relay_pin_ratio * scores[k]:
        return None
    return k


def _rough_rim(t, steer, out, seq, trials, direction: int, tune: CalTuning) -> tuple[float, float]:
    """Inertia and sliding friction from the trials' fit at the prior delay, for the creep's servo; 0 without one."""
    ts_all, xs_all = _fresh(t, steer, seq, 0, len(t))
    gaps = sorted(b - a for a, b in zip(ts_all, ts_all[1:]) if 0.0 < b - a < 0.1)
    quantum = quantum_of([abs(b - a) for a, b in zip(xs_all, xs_all[1:])])
    segs = _segments(t, steer, seq, trials, tune, quantum, 1.0 / gaps[len(gaps) // 2] if gaps else 60.0)
    got = _fit(segs, _Push(t, out, direction, tune.grid_s), tune.delay_prior_s) if len(segs) >= 3 else None
    return (max(0.0, got[0][0]), max(0.0, got[0][2])) if got else (0.0, 0.0)


def _jitter_s(t, steer, seq, trials, tune: CalTuning, quantum: float, reading_hz: float) -> float:
    """How late the game's readings are, uniformly, from the median scatter of each reading round a quadratic through
    its neighbours on the trials' slides, over the rate, the reading steps' share taken out: 0 when too few.
    The median, not the mean: lumps on a slide read the real R3 at 4.6 to 10.3 ms by the mean, 2.1 to 2.7 by it."""
    h, ratios = tune.jitter_half, []
    for ts, xs, _ in _segments(t, steer, seq, trials, tune, quantum, reading_hz):
        for i in range(h, len(ts) - h):
            w = [k for k in range(i - h, i + h + 1) if k != i]
            dt = [ts[k] - ts[i] for k in w]
            a = [[sum(d ** (p + q) for d in dt) for q in range(3)] for p in range(3)]
            b = [sum(xs[k] * d ** p for k, d in zip(w, dt)) for p in range(3)]
            sol = _solve(a, b)
            if sol is None or abs(sol[1]) < tune.jitter_rate_min:
                continue
            r = xs[i] - sol[0]
            ratios.append(max(0.0, r * r - quantum * quantum / 12.0) / (sol[1] * sol[1]))
    if len(ratios) < 10:
        return 0.0
    ratios.sort()
    return tune.jitter_scale * math.sqrt(ratios[len(ratios) // 2] / tune.jitter_leverage)


def _slow_friction(t, steer, out, seq, creep: list[int], direction: int, damping: float, tune: CalTuning) -> float:
    """The push the creep took while it moved, both ways averaged so a steady offset cancels, less its damping: 0 when
    the creep did not run or barely moved."""
    if len(creep) < 4 or not direction:
        return 0.0
    got = []
    for lo, hi, way in ((creep[0], creep[1], 1), (creep[2], creep[3], -1)):
        lo = lo + int(tune.creep_settle * (hi - lo))
        ts, xs = _fresh(t, steer, seq, lo, hi)
        if len(ts) < 6 or hi <= lo:
            return 0.0
        mt, mx = sum(ts) / len(ts), sum(xs) / len(xs)
        var = sum((a - mt) ** 2 for a in ts)
        v = sum((a - mt) * (b - mx) for a, b in zip(ts, xs)) / var if var > 0.0 else 0.0
        if v * way < 0.3 * tune.creep_rate:
            return 0.0
        push = sum(direction * o for o in out[lo:hi]) / (hi - lo)
        got.append(way * push - damping * abs(v))
    return max(0.0, sum(got) / len(got))


def _ffb_hz(t: list[float]) -> float:
    gaps = sorted(b - a for a, b in zip(t, t[1:]) if 0.0 < b - a < 0.05)
    return 1.0 / gaps[len(gaps) // 2] if gaps else 120.0


def _swing_error(seen: list[tuple[float, float]], sim: list[tuple[float, float]]) -> float:
    st = [a for a, _ in sim]
    err = [(sim[i][1] - x) ** 2 for tt, x in seen if (i := bisect_right(st, tt) - 1) >= 0]
    return math.sqrt(sum(err) / len(err)) if err else math.inf


def analyse(t, steer, out, seq, trials: list[_Trial], direction: int, tune: CalTuning,
            relay: tuple[int, int, float] | None = None, creep: list[int] | None = None) -> WheelProfile:
    """The profile from a finished run's samples; fields the run cannot pin stay 0."""
    ts_all, xs_all = _fresh(t, steer, seq, 0, len(t))
    gaps = sorted(b - a for a, b in zip(ts_all, ts_all[1:]) if 0.0 < b - a < 0.1)
    reading_hz = 1.0 / gaps[len(gaps) // 2] if gaps else 0.0
    quantum = quantum_of([abs(b - a) for a, b in zip(xs_all, xs_all[1:])])
    base = WheelProfile(direction=direction, reading_hz=reading_hz, quantum=quantum, trials=len(trials),
                        ffb_hz=_ffb_hz(t))
    segs = _segments(t, steer, seq, trials, tune, quantum, reading_hz)
    if not direction or len(segs) < 3 or len({s[2][-1] > 0.0 for s in segs}) < 2:
        return base
    held = sorted(tr.play for tr in trials if tr.started is not None and tr.share >= 1.0)
    if held and held[len(held) // 2] >= tune.play_min:
        # Every turn of the relay crosses the play, which the fit has no term for; one hold that stuck is no play.
        relay = None
    segs += _relay_segments(t, steer, seq, relay, tune, quantum, reading_hz)
    push = _Push(t, out, direction, tune.grid_s)
    steps = int(round(tune.delay_max_s / tune.delay_step_s))
    fits = [_fit(segs, push, i * tune.delay_step_s) for i in range(steps + 1)]
    rms_at = [math.sqrt(f[1] / f[2]) if f and f[2] > 0.0 else math.inf for f in fits]
    k = min(range(len(rms_at)), key=rms_at.__getitem__)
    if not math.isfinite(rms_at[k]):
        return base
    # The delay is pinned only by a clear dip inside the scan; else the fit at the prior gives damping and friction.
    pinned = 0 < k < steps and min(rms_at[0], rms_at[-1]) >= tune.pin_ratio * rms_at[k]
    prior = _starts(t, steer, seq, trials, push, tune.delay_prior_s)
    by_relay = _relay_delay(t, steer, out, seq, relay, direction, fits, quantum, reading_hz,
                            prior[len(prior) // 2] if prior else 0.0, tune)
    if by_relay is not None:
        k, pinned = by_relay, True
    if not pinned:
        k = min(steps, int(round(tune.delay_prior_s / tune.delay_step_s)))
    d, (sol, ss, yy), rms = k * tune.delay_step_s, fits[k], rms_at[k]
    inertia, damping, slide, spring = sol[:4]
    starts = _starts(t, steer, seq, trials, push, d)
    pushed = sorted(tr.pushed for tr in trials if tr.started is not None and tr.share >= 1.0)
    plays = held
    lo, hi = tune.inertia_range
    ok = 0.0 <= damping <= tune.damping_max and 0.0 <= slide <= tune.slide_max and rms <= tune.fit_rms_max
    pin = ok and pinned and lo <= inertia <= hi
    return WheelProfile(
        direction=direction, reading_hz=reading_hz, quantum=quantum,
        breakaway=starts[len(starts) // 2] if starts else 0.0, breakaway_top=starts[-1] if starts else 0.0,
        pushed_start=pushed[len(pushed) // 2] if pushed else 0.0,
        slide=max(0.0, slide) if ok else 0.0, damping=max(0.0, damping) if ok else 0.0,
        inertia=inertia if pin else 0.0, spring=max(0.0, spring) if ok else 0.0, delay_s=d if pin else 0.0,
        play=plays[len(plays) // 2] if plays else 0.0, fit_rms=rms, trials=len(trials), ffb_hz=base.ffb_hz,
        slow_friction=_slow_friction(t, steer, out, seq, creep or [], direction, max(0.0, damping), tune) if ok else 0.0,
        jitter_s=_jitter_s(t, steer, seq, trials, tune, quantum, reading_hz))
