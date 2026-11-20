"""Local ACC debug log: what the gap law saw and what each of its terms asked, per tick.

`acc_debug.csv` beside the project root, written only with `debug` on in `config.json`, at
20 Hz while ACC has a lead plus a 2 s tail. No vehicle or player ids: a changed lead is a
flag. See core/cruise_control_thread/README.md, "ACC debug log"."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from core.sending_thread.debug_csv import TickDebugLog
from core.settings import Settings

from . import anticipation, approach_profile, free_road_lift, idm_cah

if TYPE_CHECKING:
    from .acc_controller import AdaptiveCruiseController

ACC_LOG_NAME: str = "acc_debug.csv"
_INTERVAL_S: float = 0.05
_TAIL_S: float = 2.0
_MAX_BYTES: int = 512 * 1024 * 1024

ACC_LOG_HEADER: list[str] = [
    "t_s", "utc", "gap_level", "headway_s",
    "v_ego_ms", "a_ego_ms2",           # landing's tracking differentiator on ego speed
    "cap_ms2",                         # what ACC handed the orchestrator this tick
    "jerk_cmd_ms2",                    # after the jerk limiter, before the output filter
    "law_out_ms2",                     # the whole command before free-road lift and jerk limit
    "lifted_ms2", "free_road_ms2", "emergency",
    "lead_grace", "lead_changed", "chain_len",
    "raw_dist_m", "raw_v_lead_ms", "raw_a_lead_ms2", "score",
    "dist_m", "v_lead_ms", "a_lead_ms2", "a_lead_ff_ms2", "conf",
    "closing_ms", "s_want_m", "gap_ratio", "ttc_s", "need_ms2",
    "iidm_ms2", "cah_ms2", "blend_ms2", "lead_ff_ms2", "nudge_ms2",
    "law_ms2",                         # idm_cah.lead_law on the smoothed lead
    "banded_ms2",                      # plus pull-away lift and approach band, unslewed
    "followed_ms2",                    # plus the monotone brake follow limit
    "approach_pull_ms2", "landing_engaged", "landing_gap_excess_ms", "ant_delta_ms2",
    "standstill_held", "leaving_lane",
    "lead2_dist_m", "lead2_v_ms", "lead2_a_ms2", "lead3_dist_m", "lead3_v_ms", "lead3_a_ms2",
    "ant_confirm", "ant_floor_ms2",    # §9.13 evidence and the lowest command anticipation may set
    "ant_virt_dv_ms", "ant_virt_da_ms2",  # the virtual lead's speed and accel offsets
    "ant_member", "ant_weight",        # chain index and coupling of the member asking most braking
    "ant_dist_m", "ant_v_ms", "ant_a_ms2", "ant_law_ms2",
]


class AccDebugLog(TickDebugLog):
    def __init__(self, root: Path) -> None:
        super().__init__(root, ACC_LOG_NAME, ACC_LOG_HEADER, _INTERVAL_S, _TAIL_S, _MAX_BYTES)
        self._prev_vid: int | None = None

    def tick(self, now: float, inner: AdaptiveCruiseController, cap_ms2: float,
             free_road_ms2: float | None, v_ego_ms: float) -> None:
        """Queue one row. Never blocks on the disk; the caller catches anything else."""
        if now - self._last_write_mono < self._interval_s:
            return
        row = acc_debug_row(inner, cap_ms2, free_road_ms2, v_ego_ms, self._prev_vid)
        self._prev_vid = row.pop("_vid", None)
        super().tick(now, row["chain_len"] > 0 or cap_ms2 < 0.0, row)


def acc_debug_row(inner: AdaptiveCruiseController, cap_ms2: float, free_road_ms2: float | None,
                  v_ego_ms: float, prev_vid: int | None) -> dict[str, object]:
    """One row from the controller's state after `accel_cap_ms2`. `_vid` is for the caller only."""
    from .acc_controller import _headway_for_level, _LeadSnapshot

    cfg = inner.config
    try:
        level = int(Settings.acc_gap_level)
    except (TypeError, ValueError):
        level = 0
    t_h = _headway_for_level(level) if level else cfg.t_headway_s
    row: dict[str, object] = {
        "gap_level": level, "headway_s": t_h, "v_ego_ms": v_ego_ms,
        "a_ego_ms2": inner._landing.ego_accel_ms2, "cap_ms2": cap_ms2,
        "jerk_cmd_ms2": math.nan if inner._prev_cmd_ms2 is None else inner._prev_cmd_ms2,
        "free_road_ms2": math.nan if free_road_ms2 is None else free_road_ms2,
        "approach_pull_ms2": inner._approach_delta, "ant_delta_ms2": inner._ant_delta_ms2,
        "landing_engaged": inner._landing._engaged,
        "landing_gap_excess_ms": (math.nan if inner._landing.gap_excess_ms is None
                                  else inner._landing.gap_excess_ms),
        "standstill_held": inner._standstill.held, "leaving_lane": inner._blinker.leaving_lane,
        "chain_len": 0, "_vid": None,
    }
    chain = inner._last_chain_raw
    if not chain or inner._prev_mono is None:
        return row
    raw = chain[0]
    ema = inner._lead_emas.get(raw.vid)
    row["chain_len"] = len(chain)
    row["_vid"] = raw.vid
    row["lead_grace"] = inner._last_chain_mono != inner._prev_mono
    row["lead_changed"] = prev_vid is not None and prev_vid != raw.vid
    row.update(raw_dist_m=raw.dist_m, raw_v_lead_ms=raw.v_lead_ms, raw_a_lead_ms2=raw.a_lead_ms2,
               score=raw.score)
    law_out, emergency = getattr(inner, "debug_raw", (math.nan, False))
    row.update(law_out_ms2=law_out, emergency=emergency,
               lifted_ms2=free_road_lift.lift(cfg, law_out, free_road_ms2, v_ego_ms, emergency))
    for k, name in ((1, "lead2"), (2, "lead3")):
        if len(chain) > k:
            row.update({f"{name}_dist_m": chain[k].dist_m, f"{name}_v_ms": chain[k].v_lead_ms,
                        f"{name}_a_ms2": chain[k].a_lead_ms2})
    if ema is None or ema.dist_m is None or ema.v_lead_ms is None or ema.a_lead_ms2 is None:
        return row
    lead = _LeadSnapshot(vid=raw.vid, dist_m=ema.dist_m, v_lead_ms=ema.v_lead_ms,
                         a_lead_ms2=ema.a_lead_ms2, score=raw.score,
                         conf=1.0 if ema.conf is None else ema.conf, a_lead_ff_ms2=ema.a_lead_ff_ms2)
    row.update(_law_terms(cfg, lead, v_ego_ms, t_h, inner._landing.gap_excess_ms))
    a_base = law_out - inner._ant_delta_ms2
    row.update(_anticipation_terms(cfg, inner, chain, v_ego_ms, t_h, a_base))
    return row


def _anticipation_terms(cfg, inner, chain, v: float, t_h: float, a_base: float) -> dict[str, object]:
    """Which member beyond the lead anticipation acts on, and whether its braking is confirmed."""
    from .acc_controller import _LeadSnapshot

    smooth = []
    for raw in chain:
        ema = inner._lead_emas.get(raw.vid)
        if ema is None or ema.dist_m is None or ema.v_lead_ms is None or ema.a_lead_ms2 is None:
            return {}
        smooth.append(_LeadSnapshot(vid=raw.vid, dist_m=ema.dist_m, v_lead_ms=ema.v_lead_ms,
                                    a_lead_ms2=ema.a_lead_ms2, score=raw.score,
                                    conf=1.0 if ema.conf is None else ema.conf,
                                    a_lead_ff_ms2=ema.a_lead_ff_ms2))
    members = anticipation.weighted_members(cfg, chain, smooth)
    if not members or not any(w for _, _, w in members):
        return {"ant_confirm": 0.0}
    evidence, worst, row = [], None, {}
    for k, (prev, cur, w) in enumerate(members, start=1):
        law_n = idm_cah.lead_law(cfg, cur.dist_m, v, cur.v_lead_ms, cur.a_lead_ms2, t_h, cur.a_lead_ff_ms2)
        evidence.append((w, cur.a_lead_ms2, law_n))
        pull = w * min(0.0, law_n - a_base)
        if worst is None or pull < worst[0]:
            worst = (pull, k, w, cur, law_n)
    confirm = anticipation.confirmation(smooth[0], evidence)
    _, k, w, cur, law_n = worst
    row.update(ant_confirm=confirm, ant_floor_ms2=anticipation.confirmed_floor(cfg, a_base, confirm),
               ant_virt_dv_ms=cfg.ant_kv * sum(w_ * (c.v_lead_ms - p.v_lead_ms) for p, c, w_ in members),
               ant_virt_da_ms2=cfg.ant_ka * sum(w_ * (c.a_lead_ms2 - p.a_lead_ms2) for p, c, w_ in members),
               ant_member=k, ant_weight=w, ant_dist_m=cur.dist_m, ant_v_ms=cur.v_lead_ms,
               ant_a_ms2=cur.a_lead_ms2, ant_law_ms2=law_n)
    return row


def _law_terms(cfg, lead, v: float, t_h: float, gap_excess_ms: float | None) -> dict[str, object]:
    s, v_lead, a_lead = max(lead.dist_m, 1e-3), lead.v_lead_ms, lead.a_lead_ms2
    a_ff = a_lead if lead.a_lead_ff_ms2 is None else lead.a_lead_ff_ms2
    closing = v - v_lead
    s_want = cfg.s0_m + max(v_lead, 0.0) * t_h
    room = s - s_want
    a_iidm = idm_cah.iidm(s=s, v=v, v_lead=v_lead, a_max=cfg.a_max_ms2, b_comfort=cfg.b_comfort_ms2,
                          s0=cfg.s0_m, t_headway=t_h, v0=cfg.v0_ms, delta=cfg.delta)
    a_cah = idm_cah.cah(s=s, v=v, v_lead=v_lead, a_lead=a_lead, a_max=cfg.a_max_ms2)
    law = idm_cah.lead_law(cfg, s, v, v_lead, a_lead, t_h, lead.a_lead_ff_ms2)
    banded = approach_profile.banded_law(cfg, lead, v, t_h)
    return {
        "dist_m": lead.dist_m, "v_lead_ms": v_lead, "a_lead_ms2": a_lead, "a_lead_ff_ms2": a_ff,
        "conf": lead.conf, "closing_ms": closing, "s_want_m": s_want, "gap_ratio": s / s_want,
        "ttc_s": s / closing if closing > 0.1 else math.nan,
        "need_ms2": closing * closing / (2.0 * room) if closing > 0.0 and room > 0.5 else math.nan,
        "iidm_ms2": a_iidm, "cah_ms2": a_cah,
        "blend_ms2": idm_cah.acc_blend(a_iidm, a_cah, cfg.b_comfort_ms2, cfg.cool_factor_c),
        "lead_ff_ms2": idm_cah.lead_brake_ff(cfg, s, v, v_lead, a_ff),
        "nudge_ms2": idm_cah.lead_accel_nudge(cfg, s, v, v_lead, a_ff, t_h),
        "law_ms2": law, "banded_ms2": banded,
        "followed_ms2": approach_profile.brake_follow_limit(cfg, banded, lead, v, t_h, gap_excess_ms,
                                                            monotone=True),
    }
