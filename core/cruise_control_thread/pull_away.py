"""Pulling away behind a lead that is just starting to move. See core/acc/ACC_ARCHITECTURE.md §10.2."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

from . import idm_cah

if TYPE_CHECKING:
    from .acc_controller import ACConfig, _LeadSnapshot

# Only near standstill: above this acc_speed is off its latch and the law sees the lead.
MOTION_MAX_EGO_MS: float = 4.0
# Least-squares speed over this window; net displacement over the longer one.
MOTION_WINDOW_S: float = 0.5
MOTION_NET_WINDOW_S: float = 1.0
# Confirmed once the fitted speed holds this long and the lead has moved this far.
MOTION_CONFIRM_MS: float = 0.15
MOTION_CONFIRM_S: float = 0.2
MOTION_MIN_DISP_M: float = 0.12
MOTION_KEEP_MS: float = 0.08
# A start comes from rest; a crash-rocked vehicle swings back within its period.
MOTION_LOOKBACK_S: float = 2.0
MOTION_REVERSAL_M: float = 0.05
# A frame-to-frame jump this large is a teleport or a different body, not motion;
# a hole this long in the ticks means ego may have moved unseen. Both restart it.
MOTION_JUMP_M: float = 0.8
MOTION_HOLE_S: float = 0.3
# Above this acc_speed has released its latch and tracks the lead on its own.
LATCH_FLOOR_MAX_MS: float = 0.8

# Pace floor: a_lead + kv * dv + ks * gap error, full at rest, gone by PACE_ZERO_MS.
PACE_KV: float = 0.6
PACE_KS: float = 0.15
PACE_MAX_MS2: float = 1.0
PACE_MIN_LEAD_MS: float = 0.1
# At rest IIDM cannot see the lead's speed at all, so a lead seen pulling away gets
# a launch bid at once, above the standstill hold's release. Gone by PACE_LAUNCH_ZERO_MS.
PACE_LAUNCH_MS2: float = 0.35
PACE_LAUNCH_FULL_MS: float = 0.1
PACE_LAUNCH_ZERO_MS: float = 0.5
PACE_FULL_MS: float = 2.0
PACE_ZERO_MS: float = 5.0
# Never pulls inside this share of the wanted gap; the lift fades in over the band.
PACE_ROOM_LO: float = 0.75
PACE_ROOM_HI: float = 0.90
# The lift fades in over this much floor, so it never steps in or out.
PACE_FADE_MS2: float = 0.2


class LeadMotion:
    """The immediate lead's speed from its own displacement, while ego is near rest.

    `acc_speed` reads a lead under 0.6 m/s as exactly 0; this sees that crawl."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._vid: int | None = None
        self._odo_m = 0.0
        self._hist: deque[tuple[float, float]] = deque()
        self._above_s = 0.0
        self.confirmed = False

    def step(self, now: float, dt: float, vid: int, dist_m: float, v_ego: float,
             crashed: bool = False) -> float | None:
        """Fitted lead speed while motion is confirmed, else None."""
        if v_ego > MOTION_MAX_EGO_MS or crashed or not math.isfinite(dist_m):
            self.reset()
            return None
        if vid != self._vid or (self._hist and now - self._hist[-1][0] > MOTION_HOLE_S):
            self.reset()
            self._vid = vid
        self._odo_m += max(v_ego, 0.0) * dt
        pos = dist_m + self._odo_m
        if self._hist and abs(pos - self._hist[-1][1]) > MOTION_JUMP_M:
            self.reset()
            self._vid = vid
        self._hist.append((now, pos))
        while self._hist and now - self._hist[0][0] > MOTION_LOOKBACK_S:
            self._hist.popleft()
        if now - self._hist[0][0] < MOTION_WINDOW_S - 1e-6:
            self._above_s = 0.0
            self.confirmed = False
            return None
        speed = _fit_speed(self._hist, now - MOTION_WINDOW_S)
        if self.confirmed:
            self.confirmed = speed >= MOTION_KEEP_MS
        else:
            self._above_s = self._above_s + dt if speed >= MOTION_CONFIRM_MS else 0.0
            start = next(p for t, p in self._hist if now - t <= MOTION_NET_WINDOW_S)
            self.confirmed = (self._above_s >= MOTION_CONFIRM_S
                              and pos - start >= MOTION_MIN_DISP_M
                              and _largest_drop(self._hist) <= MOTION_REVERSAL_M)
        return speed if self.confirmed else None


def _largest_drop(hist: deque[tuple[float, float]]) -> float:
    peak, drop = -math.inf, 0.0
    for _, p in hist:
        peak = max(peak, p)
        drop = max(drop, peak - p)
    return drop


def _fit_speed(hist: deque[tuple[float, float]], t_from: float) -> float:
    pts = [(t, p) for t, p in hist if t >= t_from]
    n = len(pts)
    if n < 2:
        return 0.0
    t_mean = sum(t for t, _ in pts) / n
    p_mean = sum(p for _, p in pts) / n
    var = sum((t - t_mean) ** 2 for t, _ in pts)
    if var <= 1e-9:
        return 0.0
    return sum((t - t_mean) * (p - p_mean) for t, p in pts) / var


def follow_lead_motion(motion: LeadMotion, cfg: ACConfig, now: float, dt: float,
                       raw: _LeadSnapshot, chain_smooth: list[_LeadSnapshot],
                       v_ego: float) -> list[_LeadSnapshot]:
    """The smoothed chain with its immediate lead's speed floored by the measured one.

    The raw chain, which the safety overlays read, is never touched."""
    if cfg.pull_away_share <= 0.0:
        return chain_smooth
    moving = motion.step(now, dt, raw.vid, raw.dist_m, v_ego, raw.crashed)
    if moving is None:
        return chain_smooth
    lead0 = chain_smooth[0]
    return [replace(lead0, v_lead_ms=latch_floor(lead0.v_lead_ms, moving)), *chain_smooth[1:]]


def latch_floor(v_lead: float, moving: float | None) -> float:
    """The lead speed the law uses: the displacement speed while acc_speed still reads a crawl as 0."""
    if moving is None or v_lead >= LATCH_FLOOR_MAX_MS:
        return v_lead
    return max(v_lead, moving)


def pace_lift(cfg: ACConfig, law: float, s: float, v_ego: float, v_lead: float,
              a_lead: float, t_headway: float) -> float:
    """Raise the law toward the pace of a lead that is pulling away. Never lowers it.

    At rest IIDM sizes its ask from the gap alone (its lead-speed term is v * dv), and
    while crawling inside s0 + v*T it brakes even as the lead opens the gap. §10.2."""
    share = cfg.pull_away_share
    if share <= 0.0 or v_lead < PACE_MIN_LEAD_MS:
        return law
    gate = share * idm_cah.fade(v_ego, PACE_FULL_MS, PACE_ZERO_MS)
    if gate <= 0.0:
        return law
    v = max(v_ego, 0.0)
    s_want = max(cfg.s0_m + v * t_headway, 1e-3)
    room = idm_cah._cos_ramp((s / s_want - PACE_ROOM_LO) / (PACE_ROOM_HI - PACE_ROOM_LO))
    floor = max(a_lead, 0.0) + PACE_KV * (v_lead - v) + PACE_KS * (s - s_want)
    floor = max(floor, PACE_LAUNCH_MS2 * idm_cah.fade(v, PACE_LAUNCH_FULL_MS, PACE_LAUNCH_ZERO_MS))
    lift = max(0.0, min(floor, PACE_MAX_MS2) - law)
    return law + gate * room * idm_cah._cos_ramp(floor / PACE_FADE_MS2) * lift


def lift_for(cfg: ACConfig, law: float, s: float, lead: _LeadSnapshot, v_ego: float,
             t_headway: float) -> float:
    """`pace_lift` for a smoothed lead, on the slower feedforward accel estimate. §8.10."""
    a_ff = lead.a_lead_ms2 if lead.a_lead_ff_ms2 is None else lead.a_lead_ff_ms2
    return pace_lift(cfg, law, s, v_ego, lead.v_lead_ms, a_ff, t_headway)
