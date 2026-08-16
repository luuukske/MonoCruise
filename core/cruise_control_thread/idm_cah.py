"""IIDM + CAH + ACC blend primitives and the shaped gap law built from them.

Pure functions of a gap and the two vehicles' kinematics. Every stateful part of
the controller (smoothing, chain, overlays, anticipation) stays in
`acc_controller.py`. See core/acc/ACC_ARCHITECTURE.md."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .acc_controller import ACConfig


def iidm(
    s: float,
    v: float,
    v_lead: float,
    a_max: float,
    b_comfort: float,
    s0: float,
    t_headway: float,
    v0: float,
    delta: float,
) -> float:
    """Improved IDM piecewise control law (Treiber & Kesting 2013, §11.3.4)."""
    dv = v - v_lead
    sqrt_ab = math.sqrt(max(a_max * b_comfort, 1e-6))
    s_star_dyn = v * t_headway + (v * dv) / (2.0 * sqrt_ab)
    s_star = s0 + max(0.0, s_star_dyn)
    z = s_star / max(s, 1e-3)
    v_ratio = v / max(v0, 1e-3)
    a_free = a_max * (1.0 - v_ratio ** delta)
    if z >= 1.0:
        return a_max * (1.0 - z * z)
    if a_free <= 1e-6:
        return -a_max * (z * z)
    exponent = 2.0 * a_max / a_free
    return a_free * (1.0 - z ** exponent)


def cah(
    s: float,
    v: float,
    v_lead: float,
    a_lead: float,
    a_max: float,
) -> float:
    """Constant-Acceleration Heuristic (Kesting/Treiber/Helbing 2010)."""
    a_lead_eff = min(a_lead, a_max)
    s_safe = max(s, 1e-3)
    denom = v_lead * v_lead - 2.0 * s_safe * a_lead_eff
    selector = v_lead * (v - v_lead)
    # Strict: a stopped, non-accelerating lead makes both sides zero, and the
    # first branch is 0/0 there. Route it to the glide-to-stop branch instead.
    if selector < -2.0 * s_safe * a_lead_eff:
        if abs(denom) < 1e-6:
            return a_lead_eff
        return (v * v * a_lead_eff) / denom
    dv = v - v_lead
    heaviside = 1.0 if dv > 0.0 else 0.0
    return a_lead_eff - (dv * dv) * heaviside / (2.0 * s_safe)


def acc_blend(a_iidm: float, a_cah: float, b_comfort: float, c: float) -> float:
    """ACC model blend with cool factor c (Kesting et al. 2010)."""
    if a_iidm >= a_cah:
        return a_iidm
    b = max(b_comfort, 1e-6)
    return (1.0 - c) * a_iidm + c * (a_cah + b * math.tanh((a_iidm - a_cah) / b))


def _cos_ramp(t: float) -> float:
    """C1 ramp: 0 at t <= 0, 1 at t >= 1."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * t))


def level_gain(cfg: ACConfig, v_ego: float, t_headway: float) -> float:
    """How eager the loop is at this following-distance setting, level 2 = 1.0.

    A function of the wanted gap only, so it does not sag when the lead happens
    to be far away. See core/acc/ACC_ARCHITECTURE.md §8.4."""
    v = max(v_ego, 0.0)
    s_want = max(cfg.s0_m + v * t_headway, 1e-3)
    s_ref = cfg.s0_m + v * cfg.gap_gain_ref_headway_s
    return (s_ref / s_want) ** cfg.gap_gain_exponent


def comfort_gain(
    cfg: ACConfig,
    dist_m: float,
    v_ego: float,
    v_lead: float,
    t_headway: float,
) -> float:
    """Scales the comfort brake by the wanted gap, the current one, and opening.

    How smooth the loop is comes from the gap the driver asked for, not from how
    far away the lead happens to be. See core/acc/ACC_ARCHITECTURE.md §8.4."""
    v = max(v_ego, 0.0)
    s_want = max(cfg.s0_m + v * t_headway, 1e-3)
    s_ref = cfg.s0_m + v * cfg.gap_gain_ref_headway_s
    s_now = max(dist_m, 1e-3)

    # Smoothness from the wanted gap; firmness from being close in absolute
    # terms, which is the same distance whatever the driver asked for.
    level = level_gain(cfg, v_ego, t_headway)
    close = max(1.0, (s_ref / s_now) ** cfg.gap_gain_exponent)
    gain = min(cfg.gap_gain_max, max(cfg.gap_gain_min, level * close))

    opening_ms = v_lead - v_ego
    if opening_ms > 0.0:
        relief = 1.0 - (1.0 - cfg.opening_gain_min) * _cos_ramp(
            opening_ms / max(cfg.opening_gain_full_ms, 1e-6),
        )
        # Inside the wanted gap the lead cannot open it fast enough to be worth
        # waiting on, so the relief fades back out.
        span = s_want * (1.0 - cfg.opening_relief_fade_frac)
        deficit = _cos_ramp((s_want - s_now) / max(span, 1e-6))
        gain *= relief + (1.0 - relief) * deficit
    return gain


def lead_law(
    cfg: ACConfig,
    dist_m: float,
    v_ego: float,
    v_lead: float,
    a_lead: float,
    t_headway: float,
) -> float:
    """IIDM + CAH + ACC blend for one lead at its direct gap, then gap shaping."""
    eff_dist = max(dist_m, 1e-3)
    a_iidm = iidm(
        s=eff_dist,
        v=v_ego,
        v_lead=v_lead,
        a_max=cfg.a_max_ms2,
        b_comfort=cfg.b_comfort_ms2,
        s0=cfg.s0_m,
        t_headway=t_headway,
        v0=cfg.v0_ms,
        delta=cfg.delta,
    )
    a_cah_val = cah(
        s=eff_dist,
        v=v_ego,
        v_lead=v_lead,
        a_lead=a_lead,
        a_max=cfg.a_max_ms2,
    )
    a_acc = acc_blend(a_iidm, a_cah_val, cfg.b_comfort_ms2, cfg.cool_factor_c)
    if a_acc >= 0.0:
        # Upward only: a close setting closes on its target harder, a far one
        # keeps full pull rather than being throttled by its own smoothness.
        return a_acc * max(1.0, level_gain(cfg, v_ego, t_headway))
    a_soft = a_acc * comfort_gain(cfg, eff_dist, v_ego, v_lead, t_headway)
    if a_cah_val < 0.0:
        # Shaping may relax the comfort term, never past the deceleration the
        # kinematics already require, and never below the unshaped law.
        a_soft = min(a_soft, max(a_acc, a_cah_val))
    return a_soft
