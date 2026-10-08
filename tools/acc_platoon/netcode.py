"""How one TruckersMP client draws another player's truck. See tools/acc_platoon/README.md.

Two layers, both measured on the clip corpus. A playback clock replays the sender's
own past path a syncdelay behind: constant-speed segments, network stalls that freeze
the drawn truck, a ~1 Hz wander from clock sync. On top of it the receiver draws the
truck with a lagging speed, so a truck that brakes is drawn too far forward until a
correction holds it still or runs it backwards.
"""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass

PHYSICS_HZ: float = 60.0
# TMP API `syncdelay` for Simulation 1 and 2; other servers run 100 to 350 ms.
SYNC_DELAY_S: float = 0.20

SEG_MIN_S: float = 0.10
SEG_MAX_S: float = 0.20
SYNC_PERIOD_S: float = 1.0
CATCHUP_S: float = 0.6
RATE_MAX: float = 1.8
STALL_MAX_S: float = 1.0
# After a stall the stream stays fragile: the stall hazard is this many times higher for a while,
# up to a ceiling that keeps one stall from setting off an endless chain.
STALL_CLUSTER_GAIN: float = 4.0
STALL_CLUSTER_S: float = 1.5
STALL_CLUSTER_MAX_PER_MIN: float = 20.0
REWIND_MIN_S: float = 0.15
REWIND_MAX_S: float = 0.25
REWIND_RATE_MIN: float = -0.20
REWIND_RATE_MAX: float = -0.01
# Stalls per minute against the sender's speed (m/s) in the mean receiver session.
STALL_RATE_BY_SPEED: tuple[tuple[float, float], ...] = (
    (1.5, 2.0), (5.5, 1.35), (11.5, 0.9), (18.5, 0.8), (26.0, 0.4))
# Hard acceleration stalls every session alike: per minute, per m/s^2 above the knee.
ACCEL_STALL_KNEE_MS2: float = 0.8
ACCEL_STALL_PER_MIN: float = 1.0
# A truck pulling away below `LAUNCH_SPEED_MS` stutters: up to this many stalls per minute.
LAUNCH_SPEED_MS: float = 3.0
LAUNCH_STALL_PER_MIN: float = 10.0
# Segment rate jitter against speed: a sawtooth below ~18 m/s, nearly none on the highway.
JITTER_BY_SPEED: tuple[tuple[float, float], ...] = (
    (1.5, 0.17), (5.5, 0.11), (11.5, 0.08), (16.0, 0.06), (19.0, 0.03), (22.0, 0.008), (30.0, 0.006))
# Drawn speed lags the sender's by `DR_SPEED_TAU_S`; the drawn lead relaxes in `DR_PULL_TAU_S`.
DR_SPEED_TAU_S: float = 0.8
DR_PULL_TAU_S: float = 2.5
# Once the drawn lead passes `DR_ARM_M` the receiver corrects it after `DR_TRIGGER_S` (lognormal),
# provided it has reached `DR_TRIGGER_MIN_M`, or `DR_TRIGGER_CAP_S` of travel for a truck that stops.
DR_ARM_M: float = 0.3
# The lead either way is held within this many s^2/m times speed squared: a stopping truck sheds
# its overshoot on the way down instead of sliding back at rest (corpus rewinds under 3 m/s: ~5 cm).
DR_LEAD_CAP_S2_M: float = 0.09
DR_TRIGGER_S: float = 1.8
DR_TRIGGER_SIGMA: float = 0.35
DR_TRIGGER_MIN_M: float = 2.0
DR_TRIGGER_CAP_S: float = 0.45
# Each correction holds the truck still or runs it backwards for this long (lognormal),
# then the drawn truck follows the sender's speed while what is left of the lead bleeds off.
DR_FIX_S: float = 0.2
DR_FIX_SIGMA: float = 0.25
DR_BLEED_S: float = 1.8
DR_BLEED_TAU_S: float = 0.8
# Share of corrections run backwards rather than held: lead speed / (lead speed + half).
DR_REWIND_HALF_MS: float = 1.9
DR_REWIND_SHARE_MIN: float = 0.10
DR_REWIND_SHARE_MAX: float = 0.30


_STALL_SPEEDS: tuple[float, ...] = tuple(p[0] for p in STALL_RATE_BY_SPEED)
_JITTER_SPEEDS: tuple[float, ...] = tuple(p[0] for p in JITTER_BY_SPEED)


def _interp(table: tuple[tuple[float, float], ...], keys: tuple[float, ...], x: float) -> float:
    """Log-linear interpolation, clamped at both ends."""
    if x <= keys[0]:
        return table[0][1]
    if x >= keys[-1]:
        return table[-1][1]
    k = bisect.bisect_right(keys, x)
    (x0, y0), (x1, y1) = table[k - 1], table[k]
    f = (x - x0) / (x1 - x0)
    return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))


@dataclass(frozen=True)
class NetProfile:
    """One client's connection.

    The receiver's session sets how often its streams stall: lognormal around `stall_scale`
    times the corpus mean, `session_sigma` wide, each stream a further `pair_sigma`. A
    sender above the mean adds its own excess on top.
    """

    ping_s: float = 0.060
    stall_scale: float = 1.0
    session_sigma: float = 1.4
    pair_sigma: float = 0.6
    stall_median_s: float = 0.09
    stall_sigma: float = 0.9
    jitter_scale: float = 1.0
    sync_sigma_s: float = 0.012
    rewinds_per_min: float = 0.05
    dead_reckoning: bool = True

    def session(self, rng: random.Random) -> float:
        """Stall multiplier of one receiver session; its mean over sessions is `stall_scale`."""
        if self.stall_scale <= 0.0:
            return 0.0
        sig = self.session_sigma
        return self.stall_scale * math.exp(rng.gauss(-0.5 * sig * sig, sig))


NORMAL = NetProfile()
# About the 99th percentile of measured sessions, on a 250 ms ping.
LAGGY = NetProfile(ping_s=0.250, stall_scale=8.0, session_sigma=0.0, stall_median_s=0.15,
                   stall_sigma=0.9, jitter_scale=2.0, sync_sigma_s=0.040, rewinds_per_min=1.0)
CLEAN = NetProfile(ping_s=0.030, stall_scale=0.0, jitter_scale=0.0, sync_sigma_s=0.0,
                   rewinds_per_min=0.0, dead_reckoning=False)


@dataclass(frozen=True)
class Glitch:
    """A forced artefact on one pair: `freeze` pauses the drawn truck, `shift` moves it."""

    kind: str
    at_s: float
    duration_s: float
    shift_m: float = 0.0


class TruePath:
    """Authoritative front-bumper road position and speed of one truck, one sample per physics step."""

    def __init__(self, t0: float, s0: float, v0: float, pre_s: float = 4.0) -> None:
        n_pre = int(round(pre_s * PHYSICS_HZ))
        self.t_start = t0 - n_pre / PHYSICS_HZ
        self.xs = [s0 - v0 * (n_pre - k) / PHYSICS_HZ for k in range(n_pre)] + [s0]
        self.vs = [v0] * (n_pre + 1)

    def append(self, s: float, v: float | None = None) -> None:
        self.vs.append((s - self.xs[-1]) * PHYSICS_HZ if v is None else v)
        self.xs.append(s)

    def _index(self, tau: float) -> tuple[int, float] | None:
        f = (tau - self.t_start) * PHYSICS_HZ
        if f <= 0.0:
            return None
        i = int(f)
        return (i, f - i) if i < len(self.xs) - 1 else None

    def at(self, tau: float) -> float:
        k = self._index(tau)
        if k is None:
            return self.xs[0] if tau <= self.t_start else self.xs[-1]
        i, f = k
        x0 = self.xs[i]
        return x0 + (self.xs[i + 1] - x0) * f

    def speed_at(self, tau: float) -> float:
        k = self._index(tau)
        if k is None:
            return self.vs[0] if tau <= self.t_start else self.vs[-1]
        i, f = k
        return self.vs[i] + (self.vs[i + 1] - self.vs[i]) * f

    def accel_at(self, tau: float, half_s: float = 0.1) -> float:
        return (self.speed_at(tau + half_s) - self.speed_at(tau - half_s)) / (2.0 * half_s)


class TmpStream:
    """One remote truck as one receiving client draws it.

    `session` is the receiver's stall multiplier, shared by all of its streams; without
    one the stream draws its own from the receiver's profile.
    """

    def __init__(self, path: TruePath, sender: NetProfile, receiver: NetProfile,
                 rng: random.Random, t0: float, sync_delay_s: float = SYNC_DELAY_S,
                 glitches: tuple[Glitch, ...] = (), session: float | None = None) -> None:
        self.path = path
        self._rng = rng
        self.delay_s = sync_delay_s + 0.5 * (sender.ping_s + receiver.ping_s)
        if session is None:
            session = receiver.session(rng)
        pair = math.exp(rng.gauss(-0.5 * receiver.pair_sigma ** 2, receiver.pair_sigma))
        self.roughness = session * pair + max(0.0, sender.stall_scale - 1.0)
        self._accel_stalls = receiver.stall_scale > 0.0
        self._stall_on = self.roughness > 0.0 or self._accel_stalls
        worst = sender if sender.stall_median_s >= receiver.stall_median_s else receiver
        self._stall_mu = math.log(max(worst.stall_median_s, 1e-3))
        self._stall_sigma = worst.stall_sigma
        self._jitter = max(sender.jitter_scale, receiver.jitter_scale)
        self._sync_sigma = receiver.sync_sigma_s
        self._rewind_rate = max(sender.rewinds_per_min, receiver.rewinds_per_min) / 60.0
        self._dr = receiver.dead_reckoning
        self._glitches = glitches
        self.tau = t0 - self.delay_s
        self._rate = 1.0
        self._seg_end = t0
        self._offset = 0.0
        self._next_sync = t0 + rng.uniform(0.0, SYNC_PERIOD_S)
        self._paused_until = -math.inf
        self.stalled = False
        # Dead-reckoning display: drawn lead over the playback point, drawn speed, correction.
        self.lead_m = 0.0
        self._v_drawn = path.speed_at(self.tau)
        self._trigger_s = DR_TRIGGER_S * math.exp(rng.gauss(0.0, DR_TRIGGER_SIGMA))
        self._armed_s = 0.0
        self._fix: str | None = None
        self._fix_left = 0.0
        self._fix_rate = 0.0
        self.corrections = 0

    def _stall_rate(self, v: float, a: float) -> float:
        """Stalls per second at the sender's speed and accel."""
        if not self._stall_on:
            return 0.0
        per_min = self.roughness * _interp(STALL_RATE_BY_SPEED, _STALL_SPEEDS, max(v, 0.1))
        if self._accel_stalls:
            per_min += ACCEL_STALL_PER_MIN * max(0.0, a - ACCEL_STALL_KNEE_MS2)
            if v < LAUNCH_SPEED_MS:
                per_min += LAUNCH_STALL_PER_MIN * min(1.0, max(0.0, a - 0.2))
        return per_min / 60.0

    def _new_segment(self, t: float) -> None:
        rng = self._rng
        seg = rng.uniform(SEG_MIN_S, SEG_MAX_S)
        if self._rewind_rate > 0.0 and rng.random() < self._rewind_rate * seg:
            self._rate = rng.uniform(REWIND_RATE_MIN, REWIND_RATE_MAX)
            seg = rng.uniform(REWIND_MIN_S, REWIND_MAX_S)
        else:
            err = (t - self.delay_s + self._offset) - self.tau
            sigma = self._jitter * _interp(JITTER_BY_SPEED, _JITTER_SPEEDS, self.path.speed_at(self.tau))
            noise = rng.gauss(0.0, sigma) if sigma > 0.0 else 0.0
            self._rate = min(RATE_MAX, max(0.0, 1.0 + err / CATCHUP_S + noise))
        self._seg_end = t + seg

    def _forced_pause(self, t: float) -> float:
        for g in self._glitches:
            if g.kind == "freeze" and g.at_s <= t < g.at_s + g.duration_s:
                return g.at_s + g.duration_s
        return -math.inf

    def _dead_reckon(self, dt: float, step_m: float, v: float) -> None:
        """Advance the drawn lead over the playback point, which moved `step_m` this step."""
        rng = self._rng
        if self._fix == "bleed":
            self.lead_m *= math.exp(-dt / DR_BLEED_TAU_S)
            self._v_drawn = v
            self._fix_left -= dt
            if self._fix_left <= 0.0:
                self._fix = None
                self._armed_s = 0.0
                self._trigger_s = DR_TRIGGER_S * math.exp(rng.gauss(0.0, DR_TRIGGER_SIGMA))
            return
        if self._fix is not None:
            self.lead_m -= step_m + (self._fix_rate * v * dt if self._fix == "rewind" else 0.0)
            self._fix_left -= dt
            if self._fix_left <= 0.0:
                self._fix = "bleed"
                self._fix_left = DR_BLEED_S
            return
        self._v_drawn += (v - self._v_drawn) * (1.0 - math.exp(-dt / DR_SPEED_TAU_S))
        self.lead_m += ((self._v_drawn - v) - self.lead_m / DR_PULL_TAU_S) * dt
        cap = DR_LEAD_CAP_S2_M * v * v
        self.lead_m = min(cap, max(-cap, self.lead_m))
        self._armed_s = self._armed_s + dt if self.lead_m > DR_ARM_M else 0.0
        floor = min(DR_TRIGGER_MIN_M, DR_TRIGGER_CAP_S * v)
        stopping = self.lead_m > DR_TRIGGER_CAP_S * v + DR_ARM_M
        if (self._armed_s >= self._trigger_s and self.lead_m > floor) or stopping:
            ahead = max(0.0, self._v_drawn - v)
            self._fix = "rewind" if rng.random() < ahead / (ahead + DR_REWIND_HALF_MS) else "hold"
            self._fix_left = DR_FIX_S * math.exp(rng.gauss(0.0, DR_FIX_SIGMA))
            self._fix_rate = rng.uniform(DR_REWIND_SHARE_MIN, DR_REWIND_SHARE_MAX)
            self.corrections += 1

    def advance(self, t: float, dt: float) -> None:
        """Move the playback clock and the drawn truck to time `t`."""
        rng = self._rng
        v = self.path.speed_at(self.tau)
        rate = self._stall_rate(v, self.path.accel_at(self.tau)) if self._stall_on else 0.0
        if t < self._paused_until + STALL_CLUSTER_S:
            rate = max(rate, min(rate * (1.0 + STALL_CLUSTER_GAIN), STALL_CLUSTER_MAX_PER_MIN / 60.0))
        if t >= self._paused_until and rate > 0.0 and rng.random() < rate * dt:
            pause = min(STALL_MAX_S, math.exp(rng.gauss(self._stall_mu, self._stall_sigma)))
            self._paused_until = t + pause
        self._paused_until = max(self._paused_until, self._forced_pause(t))
        if t >= self._next_sync:
            if self._sync_sigma > 0.0:
                self._offset = rng.gauss(0.0, self._sync_sigma)
            self._next_sync = t + SYNC_PERIOD_S * rng.uniform(0.9, 1.1)
        self.stalled = t < self._paused_until
        if self.stalled:
            # A stall freezes the drawn truck and ends in a segment that carries the catch-up.
            self._seg_end = t
            return
        s0 = self.path.at(self.tau)
        self.tau += self._rate * dt
        if self._dr:
            self._dead_reckon(dt, self.path.at(self.tau) - s0, v)
        if t >= self._seg_end:
            self._new_segment(t)

    def position(self, t: float) -> float:
        """Front bumper of the remote truck as drawn on the receiving client at time `t`."""
        s = self.path.at(self.tau) + self.lead_m
        for g in self._glitches:
            if g.kind == "shift" and g.at_s <= t < g.at_s + g.duration_s:
                s += g.shift_m
        return s
