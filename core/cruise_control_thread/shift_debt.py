"""A lead's gear-shift stall: hold its pull through the dip and credit the speed it lost.

An AMT upshift cuts a truck's drive for about a second. Behind it, at the wanted gap, the law
lifted off for every stall, and the follower then stalled again at its own upshift at the
same road speed: the same loss counted twice. While the lead stalls the law sees its pull
held and the speed it lost credited back, and ego's own stall since then repays the credit.
See core/acc/ACC_ARCHITECTURE.md §9.11.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .acc_controller import ACConfig, _LeadSnapshot

# A vehicle pulling harder than this has a pull level; a lead's must reach LEAD_PULL_MS2
# before a stall counts: easing off a weak pull looks the same and repays nothing.
PULL_MS2: float = 0.3
LEAD_PULL_MS2: float = 0.5
# A dip below this share of the pull level is a stall.
DIP_SHARE: float = 0.7
# Longest stall held: TruckersMP plateaus last 0.5 / 1.0 / 1.75 s (p10 / p50 / p90).
HOLD_S: float = 1.6
# Decelerating past this is braking, never a shift: the drive cut alone is drag.
BRAKE_MS2: float = -0.3
# The pull level fades with this time constant while nothing refreshes it.
PEAK_TAU_S: float = 1.65
# A stall drops within DROP_S off a pull that held STEADY_SHARE of its level for STEADY_S;
# a slower fall, or one off a fading pull, is a lead easing off.
DROP_S: float = 0.9
STEADY_S: float = 0.8
STEADY_SHARE: float = 0.8
CREDIT_MAX_MS: float = 0.8
# The credit lasts this long from the stall's start, then fades out over CREDIT_FADE_S.
CREDIT_S: float = 3.5
CREDIT_FADE_S: float = 1.0
# A stall that outlives the hold was no shift: its credit fades out over this.
EXPIRE_FADE_S: float = 0.6
# Per second of the gap: the credit never closes more than this share of it.
CREDIT_GAP_SHARE: float = 0.04
# The credit fades out between these shares of the wanted gap.
ROOM_LO: float = 0.6
ROOM_HI: float = 0.8
# A lead gaining on a vehicle ahead of it this much slower is easing off, not shifting.
AHEAD_SLOWER_MS: float = 0.5
# Below this the standstill gap dominates the law and a lead settling behind its own lead
# looks like a stall: it cost a 0.8 m/s^2 brake at the end of every slow pull-away.
SPEED_MIN_MS: float = 3.0


class PullDip:
    """Pull level and stalls of one accelerating vehicle, from its acceleration."""

    __slots__ = ("peak", "dip_t", "braked", "lost_ms", "expired", "_t", "_hist", "_steady", "_arm")

    def __init__(self, steady: bool = False, arm_ms2: float = PULL_MS2) -> None:
        self.peak = 0.0
        self.dip_t: float | None = None
        self.braked = False
        # Speed lost to stalls so far; the difference of two readings is the loss between them.
        self.lost_ms = 0.0
        # Set when a held stall outlived the hold: the pull ended, it was no shift.
        self.expired = False
        self._t: float | None = None
        self._hist: deque[tuple[float, float]] = deque()
        self._steady = steady
        self._arm = arm_ms2

    def step(self, now: float, accel: float, hold_s: float) -> float:
        """The accel to believe: the pull level while it stalls, else `accel`."""
        dt = 0.0 if self._t is None else min(0.2, max(0.0, now - self._t))
        self._t = now
        if self._steady:
            self._hist.append((now, accel))
            while self._hist[0][0] < now - DROP_S - STEADY_S:
                self._hist.popleft()
        if accel < BRAKE_MS2:
            self.braked, self.dip_t, self.peak = True, None, 0.0
            return accel
        if accel > PULL_MS2:
            self.braked = False
        if self.braked:
            return accel
        if accel >= DIP_SHARE * self.peak:
            if accel > PULL_MS2:
                self.peak = max(accel, self.peak * math.exp(-dt / PEAK_TAU_S))
            self.dip_t = None
            return accel
        if self.peak <= self._arm:
            return accel
        if self.dip_t is None:
            if self._steady and not self._steady_before(now):
                self.peak = max(accel, 0.0)
                return accel
            self.dip_t, self.expired = now, False
        if now - self.dip_t > hold_s:
            self.peak, self.dip_t, self.expired = 0.0, None, True
            return accel
        self.lost_ms += (self.peak - accel) * dt
        return self.peak

    def _steady_before(self, now: float) -> bool:
        before = [a for t, a in self._hist if t <= now - DROP_S]
        if not before or self._hist[0][0] > now - DROP_S - 0.5 * STEADY_S:
            return False
        return min(before) >= STEADY_SHARE * max(before) and max(before) > PULL_MS2


class ShiftDebt:
    """The immediate lead as the law should see it through a gear-shift stall."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._vid: int | None = None
        self._lead = PullDip(steady=True, arm_ms2=LEAD_PULL_MS2)
        self._ego = PullDip()
        # (stall start, lead speed then, pull level then, credit owed, ego's loss then)
        self._debt: tuple[float, float, float, float, float] | None = None
        self._fade: tuple[float, float] | None = None
        self.credit_ms = 0.0

    def step(self, cfg: ACConfig, now: float, raw: _LeadSnapshot, smooth: _LeadSnapshot,
             a_ego: float, s_want: float, ahead: list[_LeadSnapshot]) -> _LeadSnapshot:
        """`smooth` with the held pull and the credit; unchanged outside a stall."""
        if cfg.shift_hold_s <= 0.0:
            return smooth
        if raw.vid != self._vid:
            self._vid, self._lead, self._debt, self._fade = raw.vid, PullDip(steady=True, arm_ms2=LEAD_PULL_MS2), None, None
        self._ego.step(now, a_ego, cfg.shift_hold_s)
        easing = raw.v_lead_ms < SPEED_MIN_MS or any(
            v.v_lead_ms < raw.v_lead_ms - AHEAD_SLOWER_MS or v.a_lead_ms2 < BRAKE_MS2 for v in ahead)
        held = self._lead.step(now, raw.a_lead_ms2, 0.0 if easing else cfg.shift_hold_s)
        if easing:
            self._debt, self._fade, self.credit_ms = None, None, 0.0
            return smooth
        self.credit_ms = self._credit(cfg, now, raw.v_lead_ms, a_ego)
        room = (raw.dist_m / max(s_want, 1e-3) - ROOM_LO) / (ROOM_HI - ROOM_LO)
        credit = min(self.credit_ms * min(1.0, max(0.0, room)), CREDIT_GAP_SHARE * raw.dist_m)
        if held <= raw.a_lead_ms2 and credit <= 0.0:
            return smooth
        ff = smooth.a_lead_ff_ms2
        return replace(smooth, v_lead_ms=smooth.v_lead_ms + credit,
                       a_lead_ms2=max(smooth.a_lead_ms2, held),
                       a_lead_ff_ms2=None if ff is None else max(ff, held))

    def _credit(self, cfg: ACConfig, now: float, v_raw: float, a_ego: float) -> float:
        lead = self._lead
        if lead.braked or cfg.shift_credit_max_ms <= 0.0:
            self._debt, self._fade = None, None
            return 0.0
        if lead.expired and self._debt is not None:
            self._fade, self._debt = (now, self.credit_ms), None
        if self._fade is not None:
            t_f, c_f = self._fade
            c = c_f * math.exp(-(now - t_f) / EXPIRE_FADE_S)
            if c < 0.02 or lead.dip_t is not None:
                self._fade = None
            return c if c >= 0.02 else 0.0
        if lead.dip_t is not None and self._debt is None:
            self._debt = (lead.dip_t, v_raw, lead.peak, 0.0, self._ego.lost_ms)
        if self._debt is None:
            return 0.0
        t0, v0, a0, owed, ego0 = self._debt
        if lead.dip_t is not None:
            owed = max(owed, min(cfg.shift_credit_max_ms, v0 + a0 * (now - t0) - v_raw))
            self._debt = (t0, v0, a0, owed, ego0)
        credit = max(0.0, owed - (self._ego.lost_ms - ego0))
        # Only a follower that is pulling itself will stall at the same speed and repay it.
        if self._ego.peak <= PULL_MS2 and self._ego.dip_t is None:
            credit = 0.0
        age = now - t0
        if age > CREDIT_S:
            credit *= math.exp(-(age - CREDIT_S) / CREDIT_FADE_S)
            if credit < 0.02:
                self._debt = None
                return 0.0
        return credit
