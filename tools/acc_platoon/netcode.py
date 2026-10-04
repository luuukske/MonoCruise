"""How one TruckersMP client draws another player's truck. See tools/acc_platoon/README.md.

TMP does not add noise to a remote position. It replays the sender's own past path
on a playback clock that runs a syncdelay behind, and every artefact the clip
corpus shows is that clock misbehaving: constant-speed segments, pauses, short
backward runs, and a ~1 Hz wander from clock sync.
"""
from __future__ import annotations

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
REWIND_MIN_S: float = 0.15
REWIND_MAX_S: float = 0.25
REWIND_RATE_MIN: float = -0.20
REWIND_RATE_MAX: float = -0.01
# Pair roughness: a clean pair scales every artefact by U(0, CLEAN_MAX), a rough one by U(ROUGH_LO, ROUGH_HI).
CLEAN_MAX: float = 0.15
ROUGH_LO: float = 0.4
ROUGH_HI: float = 1.8


@dataclass(frozen=True)
class NetProfile:
    """One client's connection. A pair sees the sum of both sides' pauses."""

    ping_s: float = 0.060
    stalls_per_min: float = 9.0
    stall_median_s: float = 0.03
    stall_sigma: float = 0.6
    rate_sigma: float = 0.07
    sync_sigma_s: float = 0.012
    rewinds_per_min: float = 1.2
    # Chance this side is clean; a pair is clean when both sides are.
    clean_share: float = 0.55


NORMAL = NetProfile()
LAGGY = NetProfile(ping_s=0.250, stalls_per_min=8.0, stall_median_s=0.15, stall_sigma=0.9,
                   rate_sigma=0.15, sync_sigma_s=0.040, rewinds_per_min=4.0, clean_share=0.0)
CLEAN = NetProfile(ping_s=0.030, stalls_per_min=0.0, rate_sigma=0.0, sync_sigma_s=0.0,
                   rewinds_per_min=0.0, clean_share=1.0)


@dataclass(frozen=True)
class Glitch:
    """A forced artefact on one pair: `freeze` pauses the drawn truck, `shift` moves it."""

    kind: str
    at_s: float
    duration_s: float
    shift_m: float = 0.0


class TruePath:
    """Authoritative front-bumper road position of one truck, one sample per physics step."""

    def __init__(self, t0: float, s0: float, v0: float, pre_s: float = 4.0) -> None:
        n_pre = int(round(pre_s * PHYSICS_HZ))
        self.t_start = t0 - n_pre / PHYSICS_HZ
        self.xs = [s0 - v0 * (n_pre - k) / PHYSICS_HZ for k in range(n_pre)] + [s0]

    def append(self, s: float) -> None:
        self.xs.append(s)

    def at(self, tau: float) -> float:
        f = (tau - self.t_start) * PHYSICS_HZ
        if f <= 0.0:
            return self.xs[0]
        i = int(f)
        if i >= len(self.xs) - 1:
            return self.xs[-1]
        x0 = self.xs[i]
        return x0 + (self.xs[i + 1] - x0) * (f - i)


class TmpStream:
    """One remote truck as one receiving client draws it."""

    def __init__(self, path: TruePath, sender: NetProfile, receiver: NetProfile,
                 rng: random.Random, t0: float, sync_delay_s: float = SYNC_DELAY_S,
                 glitches: tuple[Glitch, ...] = ()) -> None:
        self.path = path
        self._rng = rng
        self.delay_s = sync_delay_s + 0.5 * (sender.ping_s + receiver.ping_s)
        if rng.random() < sender.clean_share * receiver.clean_share:
            self.roughness = rng.uniform(0.0, CLEAN_MAX)
        else:
            self.roughness = rng.uniform(ROUGH_LO, ROUGH_HI)
        rough = self.roughness
        self._stall_rate = rough * (sender.stalls_per_min + receiver.stalls_per_min) / 60.0
        worst = sender if sender.stall_median_s >= receiver.stall_median_s else receiver
        self._stall_mu = math.log(max(worst.stall_median_s, 1e-3))
        self._stall_sigma = worst.stall_sigma
        self._rate_sigma = rough * receiver.rate_sigma
        self._sync_sigma = rough * receiver.sync_sigma_s
        self._rewind_rate = rough * receiver.rewinds_per_min / 60.0
        self._glitches = glitches
        self.tau = t0 - self.delay_s
        self._rate = 1.0
        self._seg_end = t0
        self._offset = 0.0
        self._next_sync = t0 + rng.uniform(0.0, SYNC_PERIOD_S)
        self._paused_until = -math.inf
        self.stalled = False

    def _new_segment(self, t: float) -> None:
        rng = self._rng
        seg = rng.uniform(SEG_MIN_S, SEG_MAX_S)
        if self._rewind_rate > 0.0 and rng.random() < self._rewind_rate * seg:
            self._rate = rng.uniform(REWIND_RATE_MIN, REWIND_RATE_MAX)
            seg = rng.uniform(REWIND_MIN_S, REWIND_MAX_S)
        else:
            err = (t - self.delay_s + self._offset) - self.tau
            noise = rng.gauss(0.0, self._rate_sigma) if self._rate_sigma > 0.0 else 0.0
            self._rate = min(RATE_MAX, max(0.0, 1.0 + err / CATCHUP_S + noise))
        self._seg_end = t + seg

    def _forced_pause(self, t: float) -> float:
        for g in self._glitches:
            if g.kind == "freeze" and g.at_s <= t < g.at_s + g.duration_s:
                return g.at_s + g.duration_s
        return -math.inf

    def advance(self, t: float, dt: float) -> None:
        """Move the playback clock to time `t`."""
        rng = self._rng
        if t >= self._paused_until and self._stall_rate > 0.0 and rng.random() < self._stall_rate * dt:
            pause = min(STALL_MAX_S, math.exp(rng.gauss(self._stall_mu, self._stall_sigma)))
            self._paused_until = t + pause
        self._paused_until = max(self._paused_until, self._forced_pause(t))
        if t >= self._next_sync:
            if self._sync_sigma > 0.0:
                self._offset = rng.gauss(0.0, self._sync_sigma)
            self._next_sync = t + SYNC_PERIOD_S * rng.uniform(0.9, 1.1)
        self.stalled = t < self._paused_until
        if self.stalled:
            # A pause ends in a segment that already carries the catch-up.
            self._seg_end = t
            return
        self.tau += self._rate * dt
        if t >= self._seg_end:
            self._new_segment(t)

    def position(self, t: float) -> float:
        """Front bumper of the remote truck as drawn on the receiving client at time `t`."""
        s = self.path.at(self.tau)
        for g in self._glitches:
            if g.kind == "shift" and g.at_s <= t < g.at_s + g.duration_s:
                s += g.shift_m
        return s
