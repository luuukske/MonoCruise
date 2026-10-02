"""Multi-vehicle anticipation: the delta the leads beyond the first add to the lead law.

Pure function of the smoothed chain. See core/acc/ACC_ARCHITECTURE.md §9."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import idm_cah

if TYPE_CHECKING:
    from .acc_controller import ACConfig, _LeadSnapshot


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def anticipation_delta(
    cfg: ACConfig,
    chain_raw: list[_LeadSnapshot],
    chain_smooth: list[_LeadSnapshot],
    v_ego: float,
    a_base: float,
    t_headway: float,
    ttc_min_vclose_ms: float,
) -> float:
    """Unfiltered anticipation adjustment (m/s^2) from leads beyond the first."""
    if len(chain_smooth) < 2:
        return 0.0

    # Stationary-lead failsafe on the RAW immediate lead speed: traffic
    # beyond a stopped vehicle predicts nothing about when it will move.
    moving_gate = _clamp(
        (chain_raw[0].v_lead_ms - cfg.ant_lead_moving_min_ms)
        / max(cfg.ant_lead_moving_full_ms - cfg.ant_lead_moving_min_ms, 1e-6),
        0.0,
        1.0,
    )
    if moving_gate <= 0.0:
        return 0.0

    # Coupling weights: pairwise time-gap ramp x tracker-score
    members: list[tuple[_LeadSnapshot, _LeadSnapshot, float]] = []
    prev = chain_smooth[0]
    w_run = moving_gate * prev.conf
    for cur in chain_smooth[1:]:
        v_ref = max(prev.v_lead_ms, cfg.ant_time_ref_floor_ms)
        gap_s = max(cur.dist_m - prev.dist_m, 0.0) / v_ref
        w_pair = idm_cah.fade(gap_s, cfg.ant_gap_full_s, cfg.ant_gap_zero_s)
        w_run *= w_pair * cur.conf
        if w_run < 1e-4:
            w_run = 0.0
        members.append((prev, cur, w_run))
        prev = cur

    if not any(w for _, _, w in members):
        return 0.0

    # Decel side: each anticipated lead is evaluated at its direct gap
    a_dec_delta = 0.0
    for _, lead, w in members:
        if w <= 0.0:
            continue
        a_n = _clamp(
            idm_cah.lead_law(cfg, lead.dist_m, v_ego, lead.v_lead_ms, lead.a_lead_ms2,
                             t_headway, lead.a_lead_ff_ms2),
            cfg.max_decel_ms2, cfg.max_accel_ms2)
        contrib = w * min(0.0, a_n - a_base)
        if contrib < a_dec_delta:
            a_dec_delta = contrib

    # Virtual lead: predict the immediate lead's near-future state from the
    # weighted upstream differentials.
    dv_up = 0.0
    da_up = 0.0
    for prev, cur, w in members:
        if w <= 0.0:
            continue
        dv_up += w * (cur.v_lead_ms - prev.v_lead_ms)
        da_up += w * (cur.a_lead_ms2 - prev.a_lead_ms2)

    primary = chain_smooth[0]
    v_virt = max(0.0, primary.v_lead_ms + cfg.ant_kv * dv_up)
    lo, hi = cfg.emergency_decel_ms2, cfg.max_accel_ms2
    a_virt = _clamp(primary.a_lead_ms2 + cfg.ant_ka * da_up, lo, hi)
    a_virt_ff = _clamp((primary.a_lead_ff_ms2 if primary.a_lead_ff_ms2 is not None
                        else primary.a_lead_ms2) + cfg.ant_ka * da_up, lo, hi)
    a_virt_cmd = _clamp(
        idm_cah.lead_law(cfg, primary.dist_m, v_ego, v_virt, a_virt, t_headway, a_virt_ff),
        cfg.max_decel_ms2, cfg.max_accel_ms2)
    # Differenced against the same law on the unmodified lead, never a_base:
    # a_base carries the confidence blend, ghost and blinker, which this must not undo.
    a_real_cmd = _clamp(
        idm_cah.lead_law(cfg, primary.dist_m, v_ego, primary.v_lead_ms, primary.a_lead_ms2,
                         t_headway, primary.a_lead_ff_ms2),
        cfg.max_decel_ms2, cfg.max_accel_ms2)
    virt_delta = a_virt_cmd - a_real_cmd
    if virt_delta < a_dec_delta:
        a_dec_delta = virt_delta
    lift = _clamp(virt_delta, 0.0, cfg.ant_lift_max_ms2)

    if lift > 0.0:
        # Kinematic gate on RAW immediate-lead data: no lift while
        # actually closing on the lead with an uncomfortable TTC.
        pr = chain_raw[0]
        v_close = v_ego - pr.v_lead_ms
        if v_close > ttc_min_vclose_ms:
            ttc = max(pr.dist_m, 0.01) / v_close
            lift *= 1.0 - idm_cah.fade(ttc, cfg.ant_lift_ttc_min_s, cfg.ant_lift_ttc_full_s)
        # Fade lift out when decel anticipation is binding so the two
        # sides never fight.
        lift *= _clamp(
            1.0 + a_dec_delta / max(cfg.ant_lift_fade_ms2, 1e-6), 0.0, 1.0,
        )

    return a_dec_delta + lift
