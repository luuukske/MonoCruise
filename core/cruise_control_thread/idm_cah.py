"""IIDM + CAH + ACC blend primitives. See core/acc/ACC_ARCHITECTURE.md."""

from __future__ import annotations

import math


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
    if selector <= -2.0 * s_safe * a_lead_eff:
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
