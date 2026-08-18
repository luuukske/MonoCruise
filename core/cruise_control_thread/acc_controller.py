"""IIDM+CAH accel cap. See core/acc/ACC_ARCHITECTURE.md and core/cruise_control_thread/README.md."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from core.settings import Settings
from core.thread_management.registry import registry

from .blinker_arbitration import (
    BLINKER_STAGE1_HEADWAY_SCALE,
    BLINKER_TTC_FLOOR_S,
    BlinkerArbiter,
    BlinkerState,
)
from . import idm_cah

logger = logging.getLogger(__name__)

# Re-export for fixtures.
__all__ = ("AdaptiveCruiseController", "BLINKER_TTC_FLOOR_S", "_LeadSnapshot")


A_MAX_MS2: float = 1.5
B_COMFORT_MS2: float = 2.0
DELTA: float = 4.0
V0_MS: float = 40.0

S0_M: float = 5.0
EGO_FRONT_OFFSET_M: float = 2.5
T_HEADWAY_S: float = 1.5

T_HEADWAY_BY_LEVEL_S: tuple[float, float, float, float, float] = (
    1.1,
    0.7,
    1.1,
    1.5,
    2.2,
)

COOL_FACTOR_C: float = 0.99

# Gap-error gain shaping. Smoothness follows the gap the driver asked for,
# measured against level 2; being inside that gap adds firmness back.
GAP_GAIN_REF_HEADWAY_S: float = T_HEADWAY_BY_LEVEL_S[2]
GAP_GAIN_EXPONENT: float = 0.6
GAP_GAIN_MIN: float = 0.35
GAP_GAIN_MAX: float = 1.5

# A lead pulling away restores the gap without ego lifting off, so its share of
# the comfort brake is cut. Full relief once it opens this fast.
OPENING_GAIN_MIN: float = 0.25
OPENING_GAIN_FULL_MS: float = 1.5
# The relief fades back out inside the wanted gap, reaching zero at this
# fraction of it: a deficit that deep is not worth leaving to the lead.
OPENING_RELIEF_FADE_FRAC: float = 0.6

MA_MAX_LEADS: int = 3
MA_MIN_CHAIN_GAP_M: float = 4.0

# Multi-vehicle anticipation coupling. Each anticipated lead's influence
ANT_GAP_FULL_S: float = 1.0
ANT_GAP_ZERO_S: float = 3.0
# Pair time gaps are normalised by the follower's speed, floored so
# slow-moving packed traffic keeps a finite gap time.
ANT_TIME_REF_FLOOR_MS: float = 5.0
# In-lane tracker score confidence ramp: zero at or below SCORE_MIN. This floor is
# what gates latch time; road-model smoothing measured as no influence on it.
ANT_SCORE_MIN: float = 0.5
ANT_SCORE_FULL: float = 5.0
# Per-vehicle confidence is EMA-filtered asymmetrically: fast upward so
ANT_CONF_TAU_UP_S: float = 0.10
ANT_CONF_TAU_DOWN_S: float = 0.80
# When the immediate lead vanishes from the chain (lane-edge drift, id
PRIMARY_GHOST_HOLD_S: float = 0.8
# Virtual-lead prediction: fraction of the weighted upstream speed /
ANT_KV: float = 0.4
ANT_KA: float = 0.5
# Accel-side anticipation may raise the command at most this far above
ANT_LIFT_MAX_MS2: float = 1.0
ANT_LIFT_TTC_MIN_S: float = 4.0
ANT_LIFT_TTC_FULL_S: float = 6.0
# When decel anticipation is binding, accel-side lift fades out over
# this many m/s^2 of decel delta so the two sides never fight.
ANT_LIFT_FADE_MS2: float = 0.3
# The total anticipation delta is EMA-filtered so chain membership
# changes (vehicle entering / leaving ego's lane) never step the output.
ANT_TAU_S: float = 0.4
# Stationary-lead failsafe: anticipation is fully disabled when the
ANT_LEAD_MOVING_MIN_MS: float = 0.75
ANT_LEAD_MOVING_FULL_MS: float = 1.5

TTC_HARD_S: float = 1.5
TTC_MIN_VCLOSE_MS: float = 0.3
D_EMERGENCY_M: float = 1.5
EMERGENCY_DECEL_MS2: float = -8.0
MAX_ACCEL_MS2: float = 1.5
MAX_DECEL_MS2: float = -6.55

STANDSTILL_SPEED_MS: float = 0.4
STANDSTILL_GAP_SLACK_M: float = 2.0
STANDSTILL_HOLD_DECEL_MS2: float = -0.6

J_MAX_MS3: float = 2.5

TAU_INPUT_NEAR_S: float = 0.12
TAU_INPUT_FAR_S: float = 0.20
D_INPUT_NEAR_M: float = 20.0
D_INPUT_FAR_M: float = 80.0

TAU_ALEAD_BRAKE_S: float = 0.080
TAU_ALEAD_RELAX_S: float = 0.350
A_LEAD_DEADBAND_MS2: float = 0.30
# Innovation span above the deadband over which the fast tau fades in. §12.2.
A_LEAD_TAU_RAMP_MS2: float = 0.45

TAU_OUTPUT_S: float = 0.05

DT_FALLBACK_S: float = 1.0 / 30.0
DT_MAX_S: float = 0.2
# Pinned to A_MAX_MS2 so _prev_cmd_ms2 stays in the same range as IIDM-
NO_LEAD_CEILING_MS2: float = A_MAX_MS2

# Lead-loss grace: brief empty-chain windows (vehicle-id flip after a
LEAD_LOSS_GRACE_S: float = 0.30

EMA_GC_TTL_S: float = 2.0


def _headway_for_level(level: int) -> float:
    if 1 <= level <= 4:
        return T_HEADWAY_BY_LEVEL_S[level]
    return T_HEADWAY_BY_LEVEL_S[0]


@dataclass(slots=True)
class ACConfig:
    a_max_ms2: float = A_MAX_MS2
    b_comfort_ms2: float = B_COMFORT_MS2
    delta: float = DELTA
    v0_ms: float = V0_MS
    s0_m: float = S0_M
    ego_front_offset_m: float = EGO_FRONT_OFFSET_M
    t_headway_s: float = T_HEADWAY_S
    cool_factor_c: float = COOL_FACTOR_C
    gap_gain_ref_headway_s: float = GAP_GAIN_REF_HEADWAY_S
    gap_gain_exponent: float = GAP_GAIN_EXPONENT
    gap_gain_min: float = GAP_GAIN_MIN
    gap_gain_max: float = GAP_GAIN_MAX
    lead_brake_ff_share: float = idm_cah.LEAD_BRAKE_FF_SHARE
    lead_brake_ff_max_ms2: float = idm_cah.LEAD_BRAKE_FF_MAX_MS2
    opening_gain_min: float = OPENING_GAIN_MIN
    opening_gain_full_ms: float = OPENING_GAIN_FULL_MS
    opening_relief_fade_frac: float = OPENING_RELIEF_FADE_FRAC
    ma_max_leads: int = MA_MAX_LEADS
    ma_min_chain_gap_m: float = MA_MIN_CHAIN_GAP_M
    ant_gap_full_s: float = ANT_GAP_FULL_S
    ant_gap_zero_s: float = ANT_GAP_ZERO_S
    ant_time_ref_floor_ms: float = ANT_TIME_REF_FLOOR_MS
    ant_score_min: float = ANT_SCORE_MIN
    ant_score_full: float = ANT_SCORE_FULL
    ant_conf_tau_up_s: float = ANT_CONF_TAU_UP_S
    ant_conf_tau_down_s: float = ANT_CONF_TAU_DOWN_S
    primary_ghost_hold_s: float = PRIMARY_GHOST_HOLD_S
    ant_kv: float = ANT_KV
    ant_ka: float = ANT_KA
    ant_lift_max_ms2: float = ANT_LIFT_MAX_MS2
    ant_lift_ttc_min_s: float = ANT_LIFT_TTC_MIN_S
    ant_lift_ttc_full_s: float = ANT_LIFT_TTC_FULL_S
    ant_lift_fade_ms2: float = ANT_LIFT_FADE_MS2
    ant_tau_s: float = ANT_TAU_S
    ant_lead_moving_min_ms: float = ANT_LEAD_MOVING_MIN_MS
    ant_lead_moving_full_ms: float = ANT_LEAD_MOVING_FULL_MS
    ttc_hard_s: float = TTC_HARD_S
    d_emergency_m: float = D_EMERGENCY_M
    emergency_decel_ms2: float = EMERGENCY_DECEL_MS2
    max_accel_ms2: float = MAX_ACCEL_MS2
    max_decel_ms2: float = MAX_DECEL_MS2
    standstill_speed_ms: float = STANDSTILL_SPEED_MS
    standstill_gap_slack_m: float = STANDSTILL_GAP_SLACK_M
    standstill_hold_decel_ms2: float = STANDSTILL_HOLD_DECEL_MS2
    j_max_ms3: float = J_MAX_MS3
    tau_input_near_s: float = TAU_INPUT_NEAR_S
    tau_input_far_s: float = TAU_INPUT_FAR_S
    d_input_near_m: float = D_INPUT_NEAR_M
    d_input_far_m: float = D_INPUT_FAR_M
    tau_alead_brake_s: float = TAU_ALEAD_BRAKE_S
    tau_alead_relax_s: float = TAU_ALEAD_RELAX_S
    tau_alead_ff_s: float = idm_cah.TAU_ALEAD_FF_S
    a_lead_deadband_ms2: float = A_LEAD_DEADBAND_MS2
    a_lead_tau_ramp_ms2: float = A_LEAD_TAU_RAMP_MS2
    lead_brake_ff_soft_ms2: float = idm_cah.LEAD_BRAKE_FF_SOFT_MS2
    lead_law_floor_soft_ms2: float = idm_cah.LEAD_LAW_FLOOR_SOFT_MS2
    lead_accel_nudge_share: float = idm_cah.LEAD_ACCEL_NUDGE_SHARE
    lead_accel_nudge_max_ms2: float = idm_cah.LEAD_ACCEL_NUDGE_MAX_MS2
    lead_accel_nudge_soft_ms2: float = idm_cah.LEAD_ACCEL_NUDGE_SOFT_MS2
    lead_accel_nudge_zero_ms: float = idm_cah.LEAD_ACCEL_NUDGE_ZERO_MS
    tau_output_s: float = TAU_OUTPUT_S
    no_lead_ceiling_ms2: float = NO_LEAD_CEILING_MS2


@dataclass(slots=True)
class _LeadSnapshot:
    vid: int
    dist_m: float
    v_lead_ms: float
    a_lead_ms2: float
    score: float = 0.0
    conf: float = 1.0
    a_lead_ff_ms2: float | None = None


@dataclass(slots=True)
class _LeadEMA:
    dist_m: float | None = None
    v_lead_ms: float | None = None
    a_lead_ms2: float | None = None
    a_lead_ff_ms2: float | None = None
    conf: float | None = None
    last_seen_mono: float = 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class AdaptiveCruiseController:
    """Commands an upper bound on longitudinal accel (m/s²) from the lead chain."""

    def __init__(self, config: ACConfig | None = None) -> None:
        self.config = config or ACConfig()
        self._prev_mono: float | None = None
        self._lead_emas: dict[int, _LeadEMA] = {}
        self._output_ema: float | None = None
        self._prev_cmd_ms2: float | None = None
        self._ant_delta_ms2: float = 0.0
        self._prev_primary_vid: int | None = None
        self._ghost_vid: int | None = None
        self._last_chain_raw: list[_LeadSnapshot] = []
        self._last_chain_mono: float = -math.inf
        self._blinker = BlinkerArbiter()

    def accel_cap_ms2(self, ego_speed_ms: float) -> float:
        now = time.monotonic()
        if self._prev_mono is None:
            dt = DT_FALLBACK_S
        else:
            dt = _clamp(now - self._prev_mono, 1e-3, DT_MAX_S)
        self._prev_mono = now

        chain_raw, indicated_raw, blinker = self._read_acc_snapshot()

        # Lead-loss grace. ACC's tracker can drop a lead for 1-2 ETS2 physics
        if chain_raw:
            self._last_chain_raw = chain_raw
            self._last_chain_mono = now
        elif self._last_chain_raw and (now - self._last_chain_mono) < LEAD_LOSS_GRACE_S:
            chain_raw = self._last_chain_raw

        v_ego = max(0.0, float(ego_speed_ms))
        if not chain_raw and indicated_raw is None:
            # Truly no lead. Route the ceiling through the SAME jerk + output
            self._gc_emas(now)
            self._ant_delta_ms2 = 0.0
            self._blinker.committed = False
            self._blinker.released_vid = None
            target = self.config.no_lead_ceiling_ms2
            a_jerk = self._jerk_limit(target, dt, is_emergency=False)
            return self._output_filter(a_jerk, dt, is_emergency=False)

        if chain_raw:
            chain_smooth = self._smooth_chain(chain_raw, dt, now)
        else:
            chain_smooth = []
        indicated_smooth = None
        if indicated_raw is not None:
            indicated_smooth = self._smooth_chain([indicated_raw], dt, now)[0]
        self._gc_emas(now)

        a_raw, is_emergency = self._compute_command(
            chain_raw, chain_smooth, v_ego, dt,
            indicated_raw=indicated_raw,
            indicated_smooth=indicated_smooth,
            b_eff=blinker.b_eff,
            committed=blinker.committed,
            lane_offset_m=blinker.lane_offset_m,
        )
        a_jerk = self._jerk_limit(a_raw, dt, is_emergency)
        return self._output_filter(a_jerk, dt, is_emergency)

    def reset(self) -> None:
        self._prev_mono = None
        self._lead_emas.clear()
        self._output_ema = None
        self._prev_cmd_ms2 = None
        self._ant_delta_ms2 = 0.0
        self._prev_primary_vid = None
        self._ghost_vid = None
        self._last_chain_raw = []
        self._last_chain_mono = -math.inf
        self._blinker.reset()

    def _lead_to_snapshot(self, lead: object) -> _LeadSnapshot | None:
        cfg = self.config
        try:
            vehicle = lead.vehicle
            if getattr(vehicle, "is_parked", False):
                return None
            vid = int(vehicle.id)
            dist_m = float(lead.dist_m)
            v_lead = float(lead.effective_speed_ms)
            a_lead = float(lead.effective_accel_ms2)
            score = float(getattr(lead, "score", 0.0))
        except (AttributeError, TypeError, ValueError):
            return None
        if not (math.isfinite(dist_m) and math.isfinite(v_lead) and math.isfinite(a_lead)):
            return None
        if not math.isfinite(score):
            score = 0.0
        dist_m = max(0.0, dist_m - cfg.ego_front_offset_m)
        if dist_m <= 0.0:
            return None
        a_lead = _clamp(a_lead, cfg.emergency_decel_ms2, cfg.max_accel_ms2)
        return _LeadSnapshot(vid, dist_m, v_lead, a_lead, score)

    def _read_acc_snapshot(
        self,
    ) -> tuple[list[_LeadSnapshot], _LeadSnapshot | None, BlinkerState]:
        """In-lane chain, indicated lead, blinker state. Anticipation uses chain only."""
        idle = BlinkerState()
        try:
            acc = registry.get_thread("acc_thread")
            if not acc.is_alive():
                return [], None, idle
        except (KeyError, AttributeError):
            return [], None, idle

        try:
            with acc.data._lock:
                raw_leads = list(acc.data.leads) if acc.data.leads else []
                indicated = getattr(acc.data, "indicated_lead", None)
                blinker = BlinkerState(
                    float(getattr(acc.data, "blinker_b_eff", 0.0) or 0.0),
                    bool(getattr(acc.data, "blinker_committed", False)),
                    float(getattr(acc.data, "blinker_lane_offset_m", 0.0) or 0.0),
                )
        except (AttributeError, TypeError, ValueError):
            return [], None, idle

        snapshots: list[_LeadSnapshot] = []
        for lead in raw_leads:
            snap = self._lead_to_snapshot(lead)
            if snap is not None:
                snapshots.append(snap)
        indicated_snap = self._lead_to_snapshot(indicated) if indicated is not None else None

        if not snapshots:
            return [], indicated_snap, blinker

        snapshots.sort(key=lambda s: s.dist_m)
        cfg = self.config
        chain: list[_LeadSnapshot] = [snapshots[0]]
        for s in snapshots[1:]:
            if len(chain) >= cfg.ma_max_leads:
                break
            if s.dist_m - chain[-1].dist_m < cfg.ma_min_chain_gap_m:
                continue
            chain.append(s)
        return chain, indicated_snap, blinker

    def _smooth_chain(
        self,
        chain: list[_LeadSnapshot],
        dt: float,
        now_mono: float,
    ) -> list[_LeadSnapshot]:
        """Per-vehicle EMA: distance-adaptive on (dist, v_lead), asymmetric on a_lead."""
        cfg = self.config
        smoothed: list[_LeadSnapshot] = []
        for raw in chain:
            ema = self._lead_emas.get(raw.vid)
            if ema is None:
                ema = _LeadEMA()
                self._lead_emas[raw.vid] = ema

            anchor_d = ema.dist_m if ema.dist_m is not None else raw.dist_m
            span = max(cfg.d_input_far_m - cfg.d_input_near_m, 1e-3)
            t = _clamp((anchor_d - cfg.d_input_near_m) / span, 0.0, 1.0)
            tau_input = cfg.tau_input_near_s + (cfg.tau_input_far_s - cfg.tau_input_near_s) * t

            ema.dist_m = idm_cah.ema_step(ema.dist_m, raw.dist_m, dt, tau_input)
            ema.v_lead_ms = idm_cah.ema_step(ema.v_lead_ms, raw.v_lead_ms, dt, tau_input)

            prev_a = ema.a_lead_ms2
            tau_a = (cfg.tau_alead_relax_s if prev_a is None
                     else idm_cah.alead_tau_s(cfg, raw.a_lead_ms2 - prev_a))
            ema.a_lead_ms2 = idm_cah.ema_step(prev_a, raw.a_lead_ms2, dt, tau_a)
            # Symmetric and slower: a static gain on a_lead would otherwise
            # transmit its noise straight to the pedal. See §8.10.
            ema.a_lead_ff_ms2 = idm_cah.ema_step(
                ema.a_lead_ff_ms2, raw.a_lead_ms2, dt, cfg.tau_alead_ff_s)

            conf_target = self._score_conf(raw.score)
            if ema.conf is None or conf_target >= ema.conf:
                tau_c = cfg.ant_conf_tau_up_s
            else:
                tau_c = cfg.ant_conf_tau_down_s
            ema.conf = idm_cah.ema_step(ema.conf, conf_target, dt, tau_c)
            ema.last_seen_mono = now_mono

            smoothed.append(_LeadSnapshot(
                vid=raw.vid,
                dist_m=ema.dist_m,
                v_lead_ms=ema.v_lead_ms,
                a_lead_ms2=ema.a_lead_ms2,
                score=raw.score,
                conf=ema.conf,
                a_lead_ff_ms2=ema.a_lead_ff_ms2,
            ))
        return smoothed

    def _gc_emas(self, now_mono: float) -> None:
        stale = [vid for vid, ema in self._lead_emas.items()
                 if now_mono - ema.last_seen_mono > EMA_GC_TTL_S]
        for vid in stale:
            self._lead_emas.pop(vid, None)

    def _safety_overlays(
        self,
        primary_raw: _LeadSnapshot,
        v_ego: float,
    ) -> tuple[float, bool] | None:
        """Emergency / hard-TTC / standstill overlays. None means continue normal law."""
        cfg = self.config
        eff_dist = max(primary_raw.dist_m, 0.01)
        if eff_dist <= cfg.d_emergency_m:
            self._ant_delta_ms2 = 0.0
            return cfg.emergency_decel_ms2, True

        v_close = v_ego - primary_raw.v_lead_ms
        if v_close > cfg.standstill_speed_ms and self._score_conf(primary_raw.score) > 0.0:
            ttc = eff_dist / max(v_close, TTC_MIN_VCLOSE_MS)
            if ttc < cfg.ttc_hard_s:
                self._ant_delta_ms2 = 0.0
                return cfg.max_decel_ms2, True

        if (
            v_ego < cfg.standstill_speed_ms
            and primary_raw.v_lead_ms < cfg.standstill_speed_ms
            and eff_dist <= cfg.s0_m + cfg.standstill_gap_slack_m
        ):
            self._ant_delta_ms2 = 0.0
            return 0.0, False
        return None

    def _compute_command(
        self,
        chain_raw: list[_LeadSnapshot],
        chain_smooth: list[_LeadSnapshot],
        v_ego: float,
        dt: float,
        indicated_raw: _LeadSnapshot | None = None,
        indicated_smooth: _LeadSnapshot | None = None,
        b_eff: float = 0.0,
        committed: bool = False,
        lane_offset_m: float = 0.0,
    ) -> tuple[float, bool]:
        cfg = self.config
        now = self._prev_mono if self._prev_mono is not None else 0.0

        try:
            level = int(Settings.acc_gap_level)
        except (TypeError, ValueError):
            level = 0
        t_headway = _headway_for_level(level) if level else cfg.t_headway_s

        if not chain_raw:
            # Empty in-lane chain with a published indicated lead: follow it.
            # Keep arbiter in pass so hysteresis is not stale when leads return.
            if indicated_smooth is None or indicated_raw is None:
                self._ant_delta_ms2 = 0.0
                self._blinker.mode = "lane"
                self._blinker.committed = False
                return cfg.no_lead_ceiling_ms2, False
            a_ind = self._indicated_accel(indicated_smooth, v_ego, t_headway)
            self._ant_delta_ms2 = 0.0
            self._blinker._set_mode("pass", now)
            self._blinker.committed = committed
            return a_ind, False

        primary_raw = chain_raw[0]
        overlay = self._safety_overlays(primary_raw, v_ego)
        if overlay is not None:
            return overlay

        # R8: stage-1 softening only while in-lane TTC and comfort allow it.
        t_lane = t_headway
        v_close_soft = v_ego - primary_raw.v_lead_ms
        ttc_soft = (
            max(primary_raw.dist_m, 0.01) / v_close_soft
            if v_close_soft > TTC_MIN_VCLOSE_MS else math.inf
        )
        a_req_soft = self._lead_law(
            primary_raw.dist_m, v_ego, primary_raw.v_lead_ms,
            primary_raw.a_lead_ms2, t_headway, primary_raw.a_lead_ff_ms2,
        )
        soft_ok = self._blinker.soft_ok(
            ttc=ttc_soft, a_req=a_req_soft,
            b_comfort=cfg.b_comfort_ms2, now=now,
        )
        if abs(b_eff) > 1e-6 and soft_ok:
            t_lane = max(
                T_HEADWAY_BY_LEVEL_S[1],
                t_headway * BLINKER_STAGE1_HEADWAY_SCALE,
            )

        primary = chain_smooth[0]
        a_base = self._lead_law(
            primary.dist_m, v_ego, primary.v_lead_ms, primary.a_lead_ms2, t_lane,
            primary.a_lead_ff_ms2,
        )
        a_base = _clamp(a_base, cfg.max_decel_ms2, cfg.max_accel_ms2)

        # What the chain would command without chain[0]: the fallback for a
        # marginal immediate lead, and the target when stage 2 releases it.
        a_free = self._chain_tail_accel(chain_smooth, v_ego, t_headway)

        # Immediate-lead confidence blend: a marginal chain[0] (tracker
        conf0 = primary.conf
        a_lead_only = a_base
        if conf0 < 1.0:
            a_base = conf0 * a_base + (1.0 - conf0) * a_free
            if a_lead_only < 0.0:
                # Low confidence may soften braking, never invert it into
                # acceleration toward the lead the law is braking for.
                a_base = min(a_base, 0.0)

        # A lead we drove away from is not a lead we lost; the ghost is for
        # id churn and lane-edge drift only. See core/acc/README.md §5.
        if self._blinker.released_vid is not None or (committed and abs(b_eff) > 1e-6):
            self._ghost_vid = None
            self._prev_primary_vid = chain_smooth[0].vid
        else:
            a_base = self._apply_primary_ghost(chain_smooth, v_ego, a_base, t_lane)

        if indicated_smooth is None:
            a_ind = cfg.no_lead_ceiling_ms2
            ind_vid = None
        else:
            a_ind = self._indicated_accel(indicated_smooth, v_ego, t_headway)
            ind_vid = indicated_smooth.vid

        a_base = self._blinker.arbitrate(
            a_base, a_ind,
            b_eff=b_eff, committed=committed, soft_ok=soft_ok,
            ind_vid=ind_vid, now=now,
            lead_ttc_s=ttc_soft, lead_gap_m=primary_raw.dist_m, v_ego=v_ego,
            lane_vid=primary.vid, a_free=a_free, lane_offset_m=lane_offset_m,
        )

        if a_base <= cfg.max_decel_ms2 + 1e-6:
            # At-clamp: immediate lead demands full brake; ant must not soften.
            self._ant_delta_ms2 = 0.0
            return a_base, True

        # Anticipation reads leads[] only. Skip during stage-2 pass release.
        if self._blinker.mode == "pass" or len(chain_smooth) < 2:
            delta_target = 0.0
        else:
            delta_target = self._anticipation_delta(
                chain_raw, chain_smooth, v_ego, a_base, t_headway,
            )
        self._ant_delta_ms2 = idm_cah.ema_step(
            self._ant_delta_ms2, delta_target, dt, cfg.ant_tau_s,
        )

        a_cmd = _clamp(a_base + self._ant_delta_ms2, cfg.max_decel_ms2, cfg.max_accel_ms2)
        is_emergency = a_cmd <= cfg.max_decel_ms2 + 1e-6
        return a_cmd, is_emergency

    def _chain_tail_accel(
        self, chain_smooth: list[_LeadSnapshot], v_ego: float, t_headway: float,
    ) -> float:
        """The command the chain would give with its immediate lead removed."""
        cfg = self.config
        if len(chain_smooth) <= 1:
            return cfg.no_lead_ceiling_ms2
        nxt = chain_smooth[1]
        return _clamp(
            self._lead_law(nxt.dist_m, v_ego, nxt.v_lead_ms, nxt.a_lead_ms2,
                           t_headway, nxt.a_lead_ff_ms2),
            cfg.max_decel_ms2, cfg.max_accel_ms2,
        )

    def _score_conf(self, score: float) -> float:
        cfg = self.config
        span = max(cfg.ant_score_full - cfg.ant_score_min, 1e-6)
        return _clamp((score - cfg.ant_score_min) / span, 0.0, 1.0)

    def _apply_primary_ghost(
        self,
        chain_smooth: list[_LeadSnapshot],
        v_ego: float,
        a_base: float,
        t_headway: float,
    ) -> float:
        """Fade out a vanished immediate lead instead of dropping it in one tick. See `core/cruise_control_thread/README.md`."""
        cfg = self.config
        now = self._prev_mono if self._prev_mono is not None else 0.0
        primary_vid = chain_smooth[0].vid
        chain_vids = {lead.vid for lead in chain_smooth}

        prev_vid = self._prev_primary_vid
        if prev_vid is not None and prev_vid != primary_vid:
            ema = self._lead_emas.get(prev_vid)
            if (
                prev_vid not in chain_vids
                and ema is not None
                and ema.dist_m is not None
                and ema.dist_m < chain_smooth[0].dist_m
            ):
                self._ghost_vid = prev_vid
        self._prev_primary_vid = primary_vid

        if self._ghost_vid is None:
            return a_base
        if self._ghost_vid in chain_vids:
            self._ghost_vid = None
            return a_base
        ema = self._lead_emas.get(self._ghost_vid)
        if ema is None or ema.dist_m is None or ema.conf is None:
            self._ghost_vid = None
            return a_base
        age = now - ema.last_seen_mono
        fade = idm_cah.fade(age, 0.0, cfg.primary_ghost_hold_s)
        if fade <= 0.0 or age > cfg.primary_ghost_hold_s:
            self._ghost_vid = None
            return a_base

        a_ghost = self._lead_law(
            ema.dist_m, v_ego, ema.v_lead_ms, ema.a_lead_ms2, t_headway,
            ema.a_lead_ff_ms2,
        )
        a_ghost = _clamp(a_ghost, cfg.max_decel_ms2, cfg.max_accel_ms2)
        w = ema.conf * fade
        return a_base + w * min(0.0, a_ghost - a_base)

    def _lead_law(
        self,
        dist_m: float,
        v_ego: float,
        v_lead: float,
        a_lead: float,
        t_headway: float,
        a_lead_ff: float | None = None,
    ) -> float:
        return idm_cah.lead_law(self.config, dist_m, v_ego, v_lead, a_lead,
                                t_headway, a_lead_ff)

    def _indicated_accel(
        self, ind: _LeadSnapshot, v_ego: float, t_headway: float,
    ) -> float:
        """Indicated-lane gap demand, bounded to comfort braking.

        A candidate is published only from outside ego's corridor, so there is
        no collision path to it. See `core/cruise_control_thread/README.md`."""
        cfg = self.config
        a = self._lead_law(ind.dist_m, v_ego, ind.v_lead_ms, ind.a_lead_ms2,
                           t_headway, ind.a_lead_ff_ms2)
        return _clamp(a, -cfg.b_comfort_ms2, cfg.max_accel_ms2)

    def _anticipation_delta(
        self,
        chain_raw: list[_LeadSnapshot],
        chain_smooth: list[_LeadSnapshot],
        v_ego: float,
        a_base: float,
        t_headway: float,
    ) -> float:
        """Unfiltered anticipation adjustment (m/s^2) from leads beyond the first. See `core/cruise_control_thread/README.md`."""
        cfg = self.config
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
        weights: list[float] = []
        w_run = moving_gate * chain_smooth[0].conf
        for n in range(1, len(chain_smooth)):
            prev = chain_smooth[n - 1]
            cur = chain_smooth[n]
            v_ref = max(prev.v_lead_ms, cfg.ant_time_ref_floor_ms)
            gap_s = max(cur.dist_m - prev.dist_m, 0.0) / v_ref
            w_pair = idm_cah.fade(gap_s, cfg.ant_gap_full_s, cfg.ant_gap_zero_s)
            w_run *= w_pair * cur.conf
            if w_run < 1e-4:
                w_run = 0.0
            weights.append(w_run)

        if not any(weights):
            return 0.0

        # Decel side: each anticipated lead is evaluated at its direct gap
        a_dec_delta = 0.0
        for n, w in enumerate(weights, start=1):
            if w <= 0.0:
                continue
            lead = chain_smooth[n]
            a_n = self._lead_law(
                lead.dist_m, v_ego, lead.v_lead_ms, lead.a_lead_ms2, t_headway,
                lead.a_lead_ff_ms2,
            )
            a_n = _clamp(a_n, cfg.max_decel_ms2, cfg.max_accel_ms2)
            contrib = w * min(0.0, a_n - a_base)
            if contrib < a_dec_delta:
                a_dec_delta = contrib

        # Virtual lead: predict the immediate lead's near-future state from
        dv_up = 0.0
        da_up = 0.0
        for n, w in enumerate(weights, start=1):
            if w <= 0.0:
                continue
            prev = chain_smooth[n - 1]
            cur = chain_smooth[n]
            dv_up += w * (cur.v_lead_ms - prev.v_lead_ms)
            da_up += w * (cur.a_lead_ms2 - prev.a_lead_ms2)

        primary = chain_smooth[0]
        v_virt = max(0.0, primary.v_lead_ms + cfg.ant_kv * dv_up)
        lo, hi = cfg.emergency_decel_ms2, cfg.max_accel_ms2
        a_virt = _clamp(primary.a_lead_ms2 + cfg.ant_ka * da_up, lo, hi)
        a_virt_ff = _clamp((primary.a_lead_ff_ms2 if primary.a_lead_ff_ms2 is not None
                            else primary.a_lead_ms2) + cfg.ant_ka * da_up, lo, hi)
        a_virt_cmd = self._lead_law(primary.dist_m, v_ego, v_virt, a_virt, t_headway,
                                    a_virt_ff)
        a_virt_cmd = _clamp(a_virt_cmd, cfg.max_decel_ms2, cfg.max_accel_ms2)
        virt_delta = a_virt_cmd - a_base
        if virt_delta < a_dec_delta:
            a_dec_delta = virt_delta
        lift = _clamp(virt_delta, 0.0, cfg.ant_lift_max_ms2)

        if lift > 0.0:
            # Kinematic gate on RAW immediate-lead data: no lift while
            # actually closing on the lead with an uncomfortable TTC.
            pr = chain_raw[0]
            v_close = v_ego - pr.v_lead_ms
            if v_close > TTC_MIN_VCLOSE_MS:
                ttc = max(pr.dist_m, 0.01) / v_close
                lift *= 1.0 - idm_cah.fade(ttc, cfg.ant_lift_ttc_min_s, cfg.ant_lift_ttc_full_s)
            # Fade lift out when decel anticipation is binding so the two
            # sides never fight.
            lift *= _clamp(
                1.0 + a_dec_delta / max(cfg.ant_lift_fade_ms2, 1e-6), 0.0, 1.0,
            )

        return a_dec_delta + lift

    def _jerk_limit(self, a_new: float, dt: float, is_emergency: bool) -> float:
        if is_emergency or self._prev_cmd_ms2 is None:
            self._prev_cmd_ms2 = a_new
            return a_new
        max_step = self.config.j_max_ms3 * dt
        delta = a_new - self._prev_cmd_ms2
        if delta > max_step:
            a_new = self._prev_cmd_ms2 + max_step
        elif delta < -max_step:
            a_new = self._prev_cmd_ms2 - max_step
        self._prev_cmd_ms2 = a_new
        return a_new

    def _output_filter(self, value: float, dt: float, is_emergency: bool) -> float:
        if is_emergency:
            self._output_ema = value
            return value
        if self._output_ema is None or not math.isfinite(self._output_ema):
            self._output_ema = value
            return value
        alpha = 1.0 - math.exp(-dt / max(self.config.tau_output_s, 1e-6))
        self._output_ema = self._output_ema + alpha * (value - self._output_ema)
        return self._output_ema

