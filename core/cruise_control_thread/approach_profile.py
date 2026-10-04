"""ACC approach profile: close on a slower lead at the constant rate that meets its speed at the
target gap, instead of IIDM's late-then-hard profile. See core/acc/ACC_ARCHITECTURE.md §13.5."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import idm_cah

if TYPE_CHECKING:
    from .acc_controller import ACConfig, _LeadSnapshot

# Weight on the whole band. 0 disables. §13.5.
APPROACH_SHARE: float = 1.0
# An approach, not a gap correction: fades in over this closing speed (m/s).
APPROACH_DV_LO_MS: float = 1.5
APPROACH_DV_HI_MS: float = 3.0
# The band assumes the lead holds its speed. One speeding up or slowing down harder
# than this (m/s^2) is left to the law, which reads its accel.
APPROACH_A_LEAD_FULL_MS2: float = 0.5
APPROACH_A_LEAD_ZERO_MS2: float = 1.0
# A stopped lead is met this far beyond s0, inside the standstill hold's reach.
APPROACH_STOP_MARGIN_M: float = 1.0
# The band hands back to the law over this much room (m) short of its target.
APPROACH_ROOM_MIN_M: float = 2.0
APPROACH_ROOM_FULL_M: float = 5.0
# A moving lead's wanted gap is a comfort target the law may dip into: the lower edge
# only holds this share of it.
APPROACH_LOWER_GAP_FRAC: float = 0.6
# Lower edge: brake at least the need once it is worth braking for (fades in over the
# first pair), up to where the law's own margin takes over (fades out over the second).
APPROACH_NEED_ON_MS2: float = 0.15
APPROACH_NEED_FULL_MS2: float = 0.45
APPROACH_LOWER_FULL_MS2: float = 2.0
APPROACH_LOWER_ZERO_MS2: float = 3.0
# A stopped lead's need is exact: its lower edge stays on up to the clamp, and its upper
# edge reaches this much further. Full at or below the first lead speed (m/s), none above.
APPROACH_STOPPED_UPPER_EXTRA_MS2: float = 1.0
APPROACH_STOPPED_FULL_MS: float = 1.0
APPROACH_STOPPED_ZERO_MS: float = 3.0
# Upper edge: no more than this times the need plus the margin, until the approach
# turns critical (fades out between the pair) and the law's front-loading is wanted.
APPROACH_UPPER_GAIN: float = 1.15
APPROACH_UPPER_MARGIN_MS2: float = 0.2
APPROACH_UPPER_FULL_MS2: float = 3.0
APPROACH_UPPER_ZERO_MS2: float = 4.5
# The band's pull on the law builds at no more than this (m/s^3). Extra braking lets go
# at the plain jerk rate, a softened law gets its braking back at once. §13.5.
APPROACH_SLEW_MS3: float = 1.5
# While closing, do not brake harder than the lead and the decel that meets its speed.
# 0 disables. Matched speed is left to the lead law. See core/cruise_control_thread/README.md.
FOLLOW_SHARE: float = 1.0
FOLLOW_DV_LO_MS: float = 0.5
FOLLOW_DV_HI_MS: float = 1.5
# Short TTC belongs to the lead law and the overlays. The limit is fully off by the first.
FOLLOW_TTC_OFF_S: float = 2.5
FOLLOW_TTC_FULL_S: float = 4.0
# Gap closing this much faster than the seen speeds means the lead is still slower
# than it looks. Do not ease the brake on a late picture.
FOLLOW_GAP_EXCESS_MS: float = 2.5
# The most this may take off a too-hard brake. Releasing it fully made a hard stop late.
FOLLOW_LIFT_MAX_MS2: float = 1.0


def approach_band(
    cfg: ACConfig,
    a_law: float,
    lead: _LeadSnapshot,
    v_ego: float,
    t_headway: float,
) -> float:
    """The immediate lead's law, kept inside the constant-decel band around the need."""
    dv = v_ego - lead.v_lead_ms
    weight = (cfg.approach_share
              * (1.0 - idm_cah.fade(dv, cfg.approach_dv_lo_ms, cfg.approach_dv_hi_ms))
              * idm_cah.fade(abs(lead.a_lead_ms2), cfg.approach_a_lead_full_ms2,
                             cfg.approach_a_lead_zero_ms2))
    if weight <= 0.0:
        return a_law
    headway_m = max(lead.v_lead_ms, 0.0) * t_headway
    out = a_law
    stopped = idm_cah.fade(lead.v_lead_ms, cfg.approach_stopped_full_ms,
                           cfg.approach_stopped_zero_ms)
    room = lead.dist_m - cfg.s0_m - max(cfg.approach_stop_margin_m,
                                         cfg.approach_lower_gap_frac * headway_m)
    if room > cfg.approach_room_min_m:
        need = dv * dv / (2.0 * room)
        lower = max(-need, cfg.max_decel_ms2)
        # Fading the lower edge out as a stop turns critical lets the need run away from it.
        fade_out = idm_cah.fade(need, cfg.approach_lower_full_ms2, cfg.approach_lower_zero_ms2)
        w_lower = (weight * _room_weight(cfg, room)
                   * (1.0 - idm_cah.fade(need, cfg.approach_need_on_ms2,
                                         cfg.approach_need_full_ms2))
                   * (stopped + (1.0 - stopped) * fade_out))
        if lower < out:
            out += w_lower * (lower - out)
    room = lead.dist_m - cfg.s0_m - max(cfg.approach_stop_margin_m, headway_m)
    if room <= cfg.approach_room_min_m:
        return out
    need = dv * dv / (2.0 * room)
    upper = -(cfg.approach_upper_gain * need + cfg.approach_upper_margin_ms2)
    extra = cfg.approach_stopped_upper_extra_ms2 * stopped
    w_upper = weight * _room_weight(cfg, room) * idm_cah.fade(
        need, cfg.approach_upper_full_ms2 + extra, cfg.approach_upper_zero_ms2 + extra)
    if out < upper:
        out += w_upper * (upper - out)
    return out


def brake_follow_limit(
    cfg: ACConfig,
    a_law: float,
    lead: _LeadSnapshot,
    v_ego: float,
    t_headway: float,
    gap_excess_ms: float | None = None,
) -> float:
    """Raise a braking law that has run past the lead's decel while closing on it."""
    if (a_law >= 0.0 or cfg.follow_share <= 0.0 or lead.a_lead_ms2 >= 0.0
            or gap_excess_ms is None or gap_excess_ms > FOLLOW_GAP_EXCESS_MS):
        return a_law
    dv = v_ego - lead.v_lead_ms
    closing = 1.0 - idm_cah.fade(dv, cfg.follow_dv_lo_ms, cfg.follow_dv_hi_ms)
    room = lead.dist_m - cfg.s0_m
    ttc = lead.dist_m / dv if dv > 0.3 else math.inf
    far = 1.0 - idm_cah.fade(ttc, FOLLOW_TTC_OFF_S, FOLLOW_TTC_FULL_S)
    if closing <= 0.0 or far <= 0.0 or room <= cfg.approach_room_min_m:
        return a_law
    kinematic = (dv * dv) / (2.0 * room) if dv > 0.0 else 0.0
    need = max(kinematic, -lead.a_lead_ms2)
    upper = -(cfg.approach_upper_gain * need + cfg.approach_upper_margin_ms2)
    extra = cfg.approach_stopped_upper_extra_ms2 * idm_cah.fade(
        lead.v_lead_ms, cfg.approach_stopped_full_ms, cfg.approach_stopped_zero_ms)
    weight = (closing * far * cfg.follow_share * _room_weight(cfg, room)
              * idm_cah.fade(need, cfg.approach_upper_full_ms2 + extra,
                             cfg.approach_upper_zero_ms2 + extra))
    if weight <= 0.0 or a_law >= upper:
        return a_law
    return a_law + min(FOLLOW_LIFT_MAX_MS2, weight * (upper - a_law))


def _room_weight(cfg: ACConfig, room: float) -> float:
    return 1.0 - idm_cah.fade(room, cfg.approach_room_min_m, cfg.approach_room_full_m)


def slew_delta(current: float, target: float, step: float, release_step: float) -> float:
    """Move the band's pull toward `target`, slewed except when giving braking back."""
    if current > 0.0 and target < current:
        # A softened law gets its braking back at once: that is the safe direction.
        current = max(target, 0.0)
    if target >= current:
        up = release_step if current < 0.0 else step
        return min(target, current + up)
    return max(target, current - step)
