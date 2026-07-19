"""
Adaptive Cruise Controller: IIDM core + CAH overlay, blended via the
Kesting/Treiber/Helbing 2010 ACC model, with multi-vehicle anticipation.

See ``core/acc/ACC_ARCHITECTURE.md`` for the design rationale and formulas.
This module implements that document.

The controller publishes an upper bound on commanded longitudinal accel
(m/s²); ``cruise_control_thread`` takes ``min(speed_pid_output, accel_cap)``
so the outer speed regulator owns set-speed tracking.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from core.settings import Settings
from core.thread_management.registry import registry

logger = logging.getLogger(__name__)


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

MA_MAX_LEADS: int = 3
MA_MIN_CHAIN_GAP_M: float = 4.0

# Multi-vehicle anticipation coupling. Each anticipated lead's influence
# is weighted by the pairwise time gap to the vehicle behind it in the
# chain: full coupling at <= ANT_GAP_FULL_S, zero at >= ANT_GAP_ZERO_S
# (cosine ramp between). Weights propagate multiplicatively down the
# chain, so one large gap anywhere removes everything beyond it, while a
# tightly packed platoon anticipates strongly.
ANT_GAP_FULL_S: float = 1.0
ANT_GAP_ZERO_S: float = 3.0
# Pair time gaps are normalised by the follower's speed, floored so
# slow-moving packed traffic keeps a finite gap time.
ANT_TIME_REF_FLOOR_MS: float = 5.0
# In-lane tracker score confidence ramp: zero at or below SCORE_MIN
# (scores hovering near the in-path threshold are tracker noise), full
# at SCORE_FULL. Scores ramp from 0 on lane entry, so a cutting-in
# vehicle fades in instead of snapping. Applied to anticipated leads
# and, via the immediate-lead confidence blend, to chain[0] itself.
ANT_SCORE_MIN: float = 1.0
ANT_SCORE_FULL: float = 5.0
# Per-vehicle confidence is EMA-filtered asymmetrically: fast upward so
# a genuine cut-in gains authority almost immediately, slow downward so
# a score flickering around the in-path threshold ratchets toward its
# recent high instead of squarewaving the command.
ANT_CONF_TAU_UP_S: float = 0.10
ANT_CONF_TAU_DOWN_S: float = 0.80
# When the immediate lead vanishes from the chain (lane-edge drift, id
# flip) while a farther vehicle takes its slot, its braking demand fades
# out over this hold instead of dropping in one tick. Min-only: a ghost
# can only hold brake briefly, never lift the cap.
PRIMARY_GHOST_HOLD_S: float = 0.8
# Virtual-lead prediction: fraction of the weighted upstream speed /
# accel differentials applied to the immediate lead's state for the
# accel-side (lift) evaluation.
ANT_KV: float = 0.4
ANT_KA: float = 0.5
# Accel-side anticipation may raise the command at most this far above
# the immediate-lead law, and only while the raw TTC to the immediate
# lead is comfortable (cosine ramp from zero lift at TTC_MIN to full
# lift at TTC_FULL).
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
# immediate lead's raw speed is below MOVING_MIN, fully enabled above
# MOVING_FULL, linear ramp between. A stopped lead must be treated on
# its own merits regardless of what traffic beyond it is doing.
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

TAU_OUTPUT_S: float = 0.05

DT_FALLBACK_S: float = 1.0 / 30.0
DT_MAX_S: float = 0.2
# Pinned to A_MAX_MS2 so _prev_cmd_ms2 stays in the same range as IIDM-
# commanded values during no-lead intervals. The cap is still permissive
# downstream: CC's speed PID is the lower bid via min(ACC, CC): but a
# lower ceiling keeps the jerk limiter's prev state from drifting far
# above IIDM's range, which gates how fast brake re-engages on lead
# reacquisition (a ceiling of 10 took ~5 s to ramp back down to -2 m/s²).
NO_LEAD_CEILING_MS2: float = A_MAX_MS2

# Lead-loss grace: brief empty-chain windows (vehicle-id flip after a
# classifier transient at close range, single-tick radar miss, etc.)
# reuse the last seen chain so a 1-2 ETS2 physics tick (50-100 ms) gap
# does not slam the output between the IIDM brake command and the no-lead
# ceiling and back.
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
    a_lead_deadband_ms2: float = A_LEAD_DEADBAND_MS2
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


@dataclass(slots=True)
class _LeadEMA:
    dist_m: float | None = None
    v_lead_ms: float | None = None
    a_lead_ms2: float | None = None
    conf: float | None = None
    last_seen_mono: float = 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ema_step(prev: float | None, new: float, dt: float, tau: float) -> float:
    if prev is None or not math.isfinite(prev):
        return new
    alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
    return prev + alpha * (new - prev)


def _fade(x: float, full: float, zero: float) -> float:
    """C1 cosine ramp: 1.0 at x <= full, 0.0 at x >= zero."""
    if x <= full:
        return 1.0
    if x >= zero:
        return 0.0
    t = (x - full) / max(zero - full, 1e-6)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def _iidm(
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


def _cah(
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


def _acc_blend(a_iidm: float, a_cah: float, b_comfort: float, c: float) -> float:
    """ACC model blend with cool factor c (Kesting et al. 2010)."""
    if a_iidm >= a_cah:
        return a_iidm
    b = max(b_comfort, 1e-6)
    return (1.0 - c) * a_iidm + c * (a_cah + b * math.tanh((a_iidm - a_cah) / b))


class AdaptiveCruiseController:
    """Commands an upper bound on longitudinal accel (m/s²) from the lead chain."""

    def __init__(self, config: ACConfig | None = None) -> None:
        self.config = config or ACConfig()
        self._prev_mono: float | None = None
        self._lead_emas: dict[int, _LeadEMA] = {}
        self._output_ema: float | None = None
        self._prev_cmd_ms2: float | None = None
        # Filtered multi-vehicle anticipation delta (m/s^2, added to the
        # immediate-lead command). EMA state so chain membership changes
        # fade instead of stepping.
        self._ant_delta_ms2: float = 0.0
        # Immediate-lead identity tracking for the ghost hold.
        self._prev_primary_vid: int | None = None
        self._ghost_vid: int | None = None
        # Lead-loss grace cache: see accel_cap_ms2.
        self._last_chain_raw: list[_LeadSnapshot] = []
        self._last_chain_mono: float = -math.inf

    def accel_cap_ms2(self, ego_speed_ms: float) -> float:
        now = time.monotonic()
        if self._prev_mono is None:
            dt = DT_FALLBACK_S
        else:
            dt = _clamp(now - self._prev_mono, 1e-3, DT_MAX_S)
        self._prev_mono = now

        chain_raw = self._read_chain()

        # Lead-loss grace. ACC's tracker can drop a lead for 1-2 ETS2 physics
        # ticks at low speed / close range (vehicle-id flip after a classifier
        # transient, brief radar miss). Treating each such gap as "no lead"
        # collapsed the controller's continuous state and slammed wanted_ms2
        # between the IIDM brake command and the no-lead ceiling: visible
        # ~3 m/s² step oscillation upstream of the mapper. Reuse the last
        # good chain for a short grace period so transient gaps are invisible.
        if chain_raw:
            self._last_chain_raw = chain_raw
            self._last_chain_mono = now
        elif self._last_chain_raw and (now - self._last_chain_mono) < LEAD_LOSS_GRACE_S:
            chain_raw = self._last_chain_raw

        if not chain_raw:
            # Truly no lead. Route the ceiling through the SAME jerk + output
            # pipeline as a real command so _prev_cmd_ms2 and _output_ema
            # stay continuous across the handover (architecture goal §15:
            # "continuous IIDM domain across all gap regimes"). Nulling them
            # here, as the previous code did, bypassed the jerk limit on the
            # very next tick and produced step changes proportional to the
            # difference between IIDM's last brake command and the ceiling.
            # Per-lead EMAs and the cached chain are left in place: _gc_emas
            # ages out stale ones via EMA_GC_TTL_S, and the cache is reseeded
            # the moment a real chain returns.
            self._gc_emas(now)
            self._ant_delta_ms2 = 0.0
            target = self.config.no_lead_ceiling_ms2
            a_jerk = self._jerk_limit(target, dt, is_emergency=False)
            return self._output_filter(a_jerk, dt, is_emergency=False)

        v_ego = max(0.0, float(ego_speed_ms))
        chain_smooth = self._smooth_chain(chain_raw, dt, now)
        self._gc_emas(now)

        a_raw, is_emergency = self._compute_command(chain_raw, chain_smooth, v_ego, dt)
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

    def _read_chain(self) -> list[_LeadSnapshot]:
        """Snapshot the in-lane lead chain from acc_thread under its lock.

        Returns leads sorted by ascending distance, capped at ma_max_leads,
        with chain spacing sanity filter applied.
        """
        cfg = self.config
        try:
            acc = registry.get_thread("acc_thread")
        except KeyError:
            return []
        try:
            if not acc.is_alive():
                return []
        except AttributeError:
            return []

        try:
            with acc.data._lock:
                if not acc.data.has_lead or not acc.data.leads:
                    return []
                raw_leads = list(acc.data.leads)
        except AttributeError:
            return []

        snapshots: list[_LeadSnapshot] = []
        for lead in raw_leads:
            try:
                vehicle = lead.vehicle
                if getattr(vehicle, "is_parked", False):
                    continue
                vid = int(vehicle.id)
                dist_m = float(lead.dist_m)
                v_lead = float(lead.effective_speed_ms)
                a_lead = float(lead.effective_accel_ms2)
                score = float(getattr(lead, "score", 0.0))
            except (AttributeError, TypeError, ValueError):
                continue

            if not (math.isfinite(dist_m) and math.isfinite(v_lead) and math.isfinite(a_lead)):
                continue
            if not math.isfinite(score):
                score = 0.0
            dist_m = max(0.0, dist_m - cfg.ego_front_offset_m)
            if dist_m <= 0.0:
                continue

            a_lead = _clamp(a_lead, cfg.emergency_decel_ms2, cfg.max_accel_ms2)
            snapshots.append(_LeadSnapshot(vid, dist_m, v_lead, a_lead, score))

        if not snapshots:
            return []

        snapshots.sort(key=lambda s: s.dist_m)

        chain: list[_LeadSnapshot] = [snapshots[0]]
        for s in snapshots[1:]:
            if len(chain) >= cfg.ma_max_leads:
                break
            if s.dist_m - chain[-1].dist_m < cfg.ma_min_chain_gap_m:
                continue
            chain.append(s)
        return chain

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

            ema.dist_m = _ema_step(ema.dist_m, raw.dist_m, dt, tau_input)
            ema.v_lead_ms = _ema_step(ema.v_lead_ms, raw.v_lead_ms, dt, tau_input)

            prev_a = ema.a_lead_ms2
            if prev_a is None:
                tau_a = cfg.tau_alead_relax_s
            else:
                delta_a = raw.a_lead_ms2 - prev_a
                if abs(delta_a) < cfg.a_lead_deadband_ms2:
                    tau_a = cfg.tau_alead_relax_s
                elif delta_a < 0.0:
                    tau_a = cfg.tau_alead_brake_s
                else:
                    tau_a = cfg.tau_alead_relax_s
            ema.a_lead_ms2 = _ema_step(prev_a, raw.a_lead_ms2, dt, tau_a)

            conf_target = self._score_conf(raw.score)
            if ema.conf is None or conf_target >= ema.conf:
                tau_c = cfg.ant_conf_tau_up_s
            else:
                tau_c = cfg.ant_conf_tau_down_s
            ema.conf = _ema_step(ema.conf, conf_target, dt, tau_c)
            ema.last_seen_mono = now_mono

            smoothed.append(_LeadSnapshot(
                vid=raw.vid,
                dist_m=ema.dist_m,
                v_lead_ms=ema.v_lead_ms,
                a_lead_ms2=ema.a_lead_ms2,
                score=raw.score,
                conf=ema.conf,
            ))
        return smoothed

    def _gc_emas(self, now_mono: float) -> None:
        stale = [vid for vid, ema in self._lead_emas.items()
                 if now_mono - ema.last_seen_mono > EMA_GC_TTL_S]
        for vid in stale:
            self._lead_emas.pop(vid, None)

    def _compute_command(
        self,
        chain_raw: list[_LeadSnapshot],
        chain_smooth: list[_LeadSnapshot],
        v_ego: float,
        dt: float,
    ) -> tuple[float, bool]:
        cfg = self.config

        primary_raw = chain_raw[0]
        eff_dist_raw = max(primary_raw.dist_m, 0.01)

        if eff_dist_raw <= cfg.d_emergency_m:
            # Safety overlays reset anticipation: on exit the controller
            # restarts from the pure immediate-lead law.
            self._ant_delta_ms2 = 0.0
            return cfg.emergency_decel_ms2, True

        v_close_raw = v_ego - primary_raw.v_lead_ms
        if v_close_raw > cfg.standstill_speed_ms:
            ttc = eff_dist_raw / max(v_close_raw, TTC_MIN_VCLOSE_MS)
            if ttc < cfg.ttc_hard_s:
                self._ant_delta_ms2 = 0.0
                return cfg.max_decel_ms2, True

        if (
            v_ego < cfg.standstill_speed_ms
            and primary_raw.v_lead_ms < cfg.standstill_speed_ms
            and eff_dist_raw <= cfg.s0_m + cfg.standstill_gap_slack_m
        ):
            # Standstill behind a stationary lead: publish 0 m/s² (no command).
            # The sending_thread HoldController is the single authority for
            # keeping the truck stationary on any slope; an ACC-side creep-hold
            # used to fight the FSM's slope-balance brake and prevented smooth
            # launches. The gating block above still detects the condition so
            # downstream consumers can treat this as "ACC is in standstill
            # context" if needed.
            self._ant_delta_ms2 = 0.0
            return 0.0, False

        try:
            level = int(Settings.acc_gap_level)
        except (TypeError, ValueError):
            level = 0
        t_headway = _headway_for_level(level) if level else cfg.t_headway_s

        primary = chain_smooth[0]
        a_base = self._lead_law(
            primary.dist_m, v_ego, primary.v_lead_ms, primary.a_lead_ms2, t_headway,
        )
        a_base = _clamp(a_base, cfg.max_decel_ms2, cfg.max_accel_ms2)

        # Immediate-lead confidence blend: a marginal chain[0] (tracker
        # score hovering near the in-path threshold, vehicle drifting on
        # the lane edge) fades in against the command we would hold
        # without it, instead of grabbing full authority the tick it
        # appears and dropping it the tick it leaves. Safety overlays
        # above already ran on the raw chain[0] regardless of score.
        conf0 = primary.conf
        if conf0 < 1.0:
            if len(chain_smooth) > 1:
                nxt = chain_smooth[1]
                a_alt = self._lead_law(
                    nxt.dist_m, v_ego, nxt.v_lead_ms, nxt.a_lead_ms2, t_headway,
                )
                a_alt = _clamp(a_alt, cfg.max_decel_ms2, cfg.max_accel_ms2)
            else:
                a_alt = cfg.no_lead_ceiling_ms2
            a_base = conf0 * a_base + (1.0 - conf0) * a_alt

        a_base = self._apply_primary_ghost(chain_smooth, v_ego, a_base, t_headway)

        if a_base <= cfg.max_decel_ms2 + 1e-6:
            # At-clamp hard overlay: the immediate lead alone demands full
            # braking authority. Anticipation must never soften this.
            self._ant_delta_ms2 = 0.0
            return a_base, True

        delta_target = self._anticipation_delta(
            chain_raw, chain_smooth, v_ego, a_base, t_headway,
        )
        self._ant_delta_ms2 = _ema_step(
            self._ant_delta_ms2, delta_target, dt, cfg.ant_tau_s,
        )

        a_cmd = _clamp(a_base + self._ant_delta_ms2, cfg.max_decel_ms2, cfg.max_accel_ms2)
        is_emergency = a_cmd <= cfg.max_decel_ms2 + 1e-6
        return a_cmd, is_emergency

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
        """Fade out a vanished immediate lead instead of dropping it in one tick.

        When chain[0] flips to a farther vehicle because the previous
        primary left the published list (lane-edge drift, id flip), its
        cached kinematics keep a decaying, min-only grip on the command
        for primary_ghost_hold_s. A ghost can only hold brake, never
        raise the cap; reappearance in the chain clears it.
        """
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
        fade = _fade(age, 0.0, cfg.primary_ghost_hold_s)
        if fade <= 0.0 or age > cfg.primary_ghost_hold_s:
            self._ghost_vid = None
            return a_base

        a_ghost = self._lead_law(
            ema.dist_m, v_ego, ema.v_lead_ms, ema.a_lead_ms2, t_headway,
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
    ) -> float:
        """IIDM + CAH + ACC blend for one lead at its direct gap."""
        cfg = self.config
        eff_dist = max(dist_m, 1e-3)
        a_iidm = _iidm(
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
        a_cah_val = _cah(
            s=eff_dist,
            v=v_ego,
            v_lead=v_lead,
            a_lead=a_lead,
            a_max=cfg.a_max_ms2,
        )
        return _acc_blend(a_iidm, a_cah_val, cfg.b_comfort_ms2, cfg.cool_factor_c)

    def _anticipation_delta(
        self,
        chain_raw: list[_LeadSnapshot],
        chain_smooth: list[_LeadSnapshot],
        v_ego: float,
        a_base: float,
        t_headway: float,
    ) -> float:
        """Unfiltered anticipation adjustment (m/s^2) from leads beyond the first.

        Negative values brake earlier / gentler than the immediate-lead law
        (unbounded, still clamped by max_decel downstream); positive values
        may lift the command by at most ant_lift_max_ms2 and only while the
        raw TTC to the immediate lead is comfortable.
        """
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
        # confidence, propagated multiplicatively down the chain. The
        # immediate lead's own confidence gates the whole chain: a shaky
        # chain[0] makes every prediction built on it shaky too.
        weights: list[float] = []
        w_run = moving_gate * chain_smooth[0].conf
        for n in range(1, len(chain_smooth)):
            prev = chain_smooth[n - 1]
            cur = chain_smooth[n]
            v_ref = max(prev.v_lead_ms, cfg.ant_time_ref_floor_ms)
            gap_s = max(cur.dist_m - prev.dist_m, 0.0) / v_ref
            w_pair = _fade(gap_s, cfg.ant_gap_full_s, cfg.ant_gap_zero_s)
            w_run *= w_pair * cur.conf
            if w_run < 1e-4:
                w_run = 0.0
            weights.append(w_run)

        if not any(weights):
            return 0.0

        # Decel side: each anticipated lead is evaluated at its direct gap
        # with the full law; its extra braking demand relative to the
        # immediate-lead command is scaled by its coupling weight. At w=1 it
        # binds fully, at w=0 it contributes nothing, smooth in between.
        a_dec_delta = 0.0
        for n, w in enumerate(weights, start=1):
            if w <= 0.0:
                continue
            lead = chain_smooth[n]
            a_n = self._lead_law(
                lead.dist_m, v_ego, lead.v_lead_ms, lead.a_lead_ms2, t_headway,
            )
            a_n = _clamp(a_n, cfg.max_decel_ms2, cfg.max_accel_ms2)
            contrib = w * min(0.0, a_n - a_base)
            if contrib < a_dec_delta:
                a_dec_delta = contrib

        # Virtual lead: predict the immediate lead's near-future state from
        # the weighted upstream speed / accel differentials and re-run the
        # law on the immediate gap. The negative part joins the decel side
        # (it reacts to upstream slowdowns long before the per-lead direct
        # gap does); the positive part becomes the bounded, gated lift.
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
        a_virt = _clamp(
            primary.a_lead_ms2 + cfg.ant_ka * da_up,
            cfg.emergency_decel_ms2,
            cfg.max_accel_ms2,
        )
        a_virt_cmd = self._lead_law(primary.dist_m, v_ego, v_virt, a_virt, t_headway)
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
                lift *= 1.0 - _fade(ttc, cfg.ant_lift_ttc_min_s, cfg.ant_lift_ttc_full_s)
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

