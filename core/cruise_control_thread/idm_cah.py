"""IIDM + CAH + ACC blend primitives and the shaped gap law built from them.

Pure functions of a gap and the two vehicles' kinematics. Every stateful part of
the controller (smoothing, chain, overlays, anticipation) stays in
`acc_controller.py`. See core/acc/ACC_ARCHITECTURE.md."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .acc_controller import ACConfig

# Lead-braking feedforward. The blend discards a_cah whenever IIDM is already
# the harder demand, which leaves a_lead no path to the command. See §8.6.
LEAD_BRAKE_FF_SHARE: float = 0.50
# Matches b_comfort: a spurious a_lead spike costs one comfort brake, never more.
LEAD_BRAKE_FF_MAX_MS2: float = 2.0
# Cubic soft corner on the demand, so gain rises from zero instead of stepping
# there. 0 restores the hard corner. See §8.7.
LEAD_BRAKE_FF_SOFT_MS2: float = 0.50


# Softening width of the kinematic floor, in m/s^2 of command. 0 restores the
# hard clamp. Biased to the braking side so it can never relax the floor. §8.7.
LEAD_LAW_FLOOR_SOFT_MS2: float = 0.09

# Accel-side phase lead. Deliberately a fraction of the brake side: it buys feel,
# not safety, so it is bounded far tighter and gated off while closing. See §8.9.
LEAD_ACCEL_NUDGE_SHARE: float = 0.50
LEAD_ACCEL_NUDGE_MAX_MS2: float = 0.75
LEAD_ACCEL_NUDGE_SOFT_MS2: float = 0.30
LEAD_ACCEL_NUDGE_ZERO_MS: float = 2.0
# Room gate, as a fraction of the wanted gap. Without it the nudge pulls just as
# hard at 6 m as at 1000 m and can cancel a real brake. See §8.12.
LEAD_ACCEL_NUDGE_ROOM_LO: float = 0.70
LEAD_ACCEL_NUDGE_ROOM_HI: float = 1.05

# Gap-error braking is halved while ego is not actually catching the lead, and
# back to normal by 2 km/h of closing. The kinematic floor still holds. §8.11.
CLOSING_RELIEF_MIN: float = 0.50
CLOSING_RELIEF_FULL_MS: float = 2.0 / 3.6

# The feedforwards read a_lead through their own slower symmetric filter. CAH
# keeps the fast asymmetric one: it sizes a stopping requirement. See §8.10.
TAU_ALEAD_FF_S: float = 0.50


def ema_step(prev: float | None, new: float, dt: float, tau: float) -> float:
    if prev is None or not math.isfinite(prev):
        return new
    alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
    return prev + alpha * (new - prev)


def _soft_min(a: float, b: float, eps: float) -> float:
    """Log-sum-exp minimum, always <= min(a, b), by at most `eps` at a == b.

    Deviating downward keeps every "never relaxes past" assertion valid."""
    if eps <= 0.0:
        return min(a, b)
    lo, hi = (a, b) if a <= b else (b, a)
    return lo - eps * math.log1p(math.exp(-(hi - lo) * math.log(2.0) / eps)) / math.log(2.0)


def _soft_negative(d: float, eps: float) -> float:
    """C1 replacement for `min(0, d)`: zero above 0, exact below -eps.

    A hard corner rectifies jitter around it into a one-sided mean."""
    if d >= 0.0:
        return 0.0
    if eps <= 0.0 or d <= -eps:
        return d
    return -(d * d * d) / (eps * eps) - 2.0 * (d * d) / eps


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


def lead_brake_ff(
    cfg: ACConfig,
    s: float,
    v: float,
    v_lead: float,
    a_lead: float,
) -> float:
    """Bounded share of the braking CAH demands purely because the lead is braking.

    Exactly zero for a lead that is not decelerating, so it moves only the
    `a_lead` axis. See core/acc/ACC_ARCHITECTURE.md §8.6."""
    if a_lead >= 0.0:
        return 0.0
    coasting = cah(s=s, v=v, v_lead=v_lead, a_lead=0.0, a_max=cfg.a_max_ms2)
    braking = cah(s=s, v=v, v_lead=v_lead, a_lead=a_lead, a_max=cfg.a_max_ms2)
    demand = _soft_negative(braking - coasting, cfg.lead_brake_ff_soft_ms2)
    return max(cfg.lead_brake_ff_share * demand, -cfg.lead_brake_ff_max_ms2)


def alead_tau_s(cfg: ACConfig, innovation: float) -> float:
    """Filter time constant for a_lead, ramped over the innovation not switched.

    See core/acc/ACC_ARCHITECTURE.md §12.2."""
    span = max(cfg.a_lead_tau_ramp_ms2, 1e-9)
    urgency = _cos_ramp((-innovation - cfg.a_lead_deadband_ms2) / span)
    return (cfg.tau_alead_relax_s
            + (cfg.tau_alead_brake_s - cfg.tau_alead_relax_s) * urgency)


def lead_accel_nudge(
    cfg: ACConfig,
    s: float,
    v: float,
    v_lead: float,
    a_lead: float,
    t_headway: float,
) -> float:
    """Bounded accel-side phase lead for a lead that is pulling harder.

    Zero unless the lead is accelerating, and gated on both closing speed and
    room, so it cannot add pull toward a lead ego is catching or crowding.
    See §8.9 and §8.12."""
    if a_lead <= 0.0:
        return 0.0
    # Soft-cornered at both ends: at a_max because ego cannot follow what it
    # cannot produce, and at zero so jitter is not rectified into pull.
    eps = cfg.lead_accel_nudge_soft_ms2
    capped = cfg.a_max_ms2 + _soft_negative(a_lead - cfg.a_max_ms2, eps)
    demand = -_soft_negative(-capped, eps)
    gate = 1.0 - _cos_ramp((v - v_lead) / max(cfg.lead_accel_nudge_zero_ms, 1e-6))
    s_want = max(cfg.s0_m + max(v, 0.0) * t_headway, 1e-3)
    span = max(cfg.lead_accel_nudge_room_hi - cfg.lead_accel_nudge_room_lo, 1e-6)
    room = _cos_ramp((s / s_want - cfg.lead_accel_nudge_room_lo) / span)
    return min(cfg.lead_accel_nudge_share * demand * gate * room,
               cfg.lead_accel_nudge_max_ms2)


def fade(x: float, full: float, zero: float) -> float:
    """C1 cosine ramp: 1.0 at x <= full, 0.0 at x >= zero."""
    if x <= full:
        return 1.0
    if x >= zero:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * (x - full) / max(zero - full, 1e-6)))


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


def closing_relief(cfg: ACConfig, v_ego: float, v_lead: float) -> float:
    """Softens gap-error braking while the lead is level with ego or opening.

    Only the comfort term: the §8.4 floor still enforces the kinematics. §8.11."""
    closing = v_ego - v_lead
    ramp = _cos_ramp(closing / max(cfg.closing_relief_full_ms, 1e-6))
    return cfg.closing_relief_min + (1.0 - cfg.closing_relief_min) * ramp


def lead_law(
    cfg: ACConfig,
    dist_m: float,
    v_ego: float,
    v_lead: float,
    a_lead: float,
    t_headway: float,
    a_lead_ff: float | None = None,
) -> float:
    """IIDM + CAH + ACC blend for one lead at its direct gap, then gap shaping."""
    eff_dist = max(dist_m, 1e-3)
    a_ff = a_lead if a_lead_ff is None else a_lead_ff
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
    # Outside the blend, which steps across its own branch test. See §8.6.
    a_acc += lead_brake_ff(cfg, eff_dist, v_ego, v_lead, a_ff)
    a_acc += lead_accel_nudge(cfg, eff_dist, v_ego, v_lead, a_ff, t_headway)
    if a_acc >= 0.0:
        # Upward only: a close setting closes on its target harder, a far one
        # keeps full pull rather than being throttled by its own smoothness.
        return a_acc * max(1.0, level_gain(cfg, v_ego, t_headway))
    a_soft = (a_acc * comfort_gain(cfg, eff_dist, v_ego, v_lead, t_headway)
              * closing_relief(cfg, v_ego, v_lead))
    # Kinematic floor, softened downward only. Ungated and the inner max hard,
    # both for continuity reasons set out in §8.7.
    return _soft_min(a_soft, max(a_acc, a_cah_val), cfg.lead_law_floor_soft_ms2)
