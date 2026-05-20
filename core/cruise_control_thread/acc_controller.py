"""
Adaptive Cruise Controller — IIDM core + CAH overlay, blended via the
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

S0_M: float = 3.0
EGO_FRONT_OFFSET_M: float = 2.5
T_HEADWAY_S: float = 1.5

T_HEADWAY_BY_LEVEL_S: tuple[float, float, float, float, float] = (
    1.5,
    1.0,
    1.5,
    2.0,
    2.5,
)

COOL_FACTOR_C: float = 0.99

MA_MAX_LEADS: int = 3
MA_WEIGHT_DECAY: float = 0.5
MA_MIN_CHAIN_GAP_M: float = 4.0

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
NO_LEAD_CEILING_MS2: float = 10.0

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
    ma_weight_decay: float = MA_WEIGHT_DECAY
    ma_min_chain_gap_m: float = MA_MIN_CHAIN_GAP_M
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
    tail_m: float


@dataclass(slots=True)
class _LeadEMA:
    dist_m: float | None = None
    v_lead_ms: float | None = None
    a_lead_ms2: float | None = None
    last_seen_mono: float = 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ema_step(prev: float | None, new: float, dt: float, tau: float) -> float:
    if prev is None or not math.isfinite(prev):
        return new
    alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
    return prev + alpha * (new - prev)


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

    def accel_cap_ms2(self, ego_speed_ms: float) -> float:
        now = time.monotonic()
        if self._prev_mono is None:
            dt = DT_FALLBACK_S
        else:
            dt = _clamp(now - self._prev_mono, 1e-3, DT_MAX_S)
        self._prev_mono = now

        chain_raw = self._read_chain()
        if not chain_raw:
            self._lead_emas.clear()
            self._output_ema = None
            self._prev_cmd_ms2 = None
            return self.config.no_lead_ceiling_ms2

        v_ego = max(0.0, float(ego_speed_ms))
        chain_smooth = self._smooth_chain(chain_raw, dt, now)
        self._gc_emas(now)

        a_raw, is_emergency = self._compute_command(chain_raw, chain_smooth, v_ego)
        a_jerk = self._jerk_limit(a_raw, dt, is_emergency)
        return self._output_filter(a_jerk, dt, is_emergency)

    def reset(self) -> None:
        self._prev_mono = None
        self._lead_emas.clear()
        self._output_ema = None
        self._prev_cmd_ms2 = None

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
                tail_m = 0.5 * float(vehicle.size.length)
                if not vehicle.is_tmp:
                    for trailer in vehicle.trailers:
                        if not trailer.is_zero():
                            tail_m += float(trailer.size.length)
            except (AttributeError, TypeError, ValueError):
                continue

            if not (math.isfinite(dist_m) and math.isfinite(v_lead) and math.isfinite(a_lead)):
                continue
            dist_m = max(0.0, dist_m - cfg.ego_front_offset_m)
            if dist_m <= 0.0:
                continue

            a_lead = _clamp(a_lead, cfg.emergency_decel_ms2, cfg.max_accel_ms2)
            snapshots.append(_LeadSnapshot(vid, dist_m, v_lead, a_lead, tail_m))

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
            ema.last_seen_mono = now_mono

            smoothed.append(_LeadSnapshot(
                vid=raw.vid,
                dist_m=ema.dist_m,
                v_lead_ms=ema.v_lead_ms,
                a_lead_ms2=ema.a_lead_ms2,
                tail_m=raw.tail_m,
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
    ) -> tuple[float, bool]:
        cfg = self.config

        primary_raw = chain_raw[0]
        eff_dist_raw = max(primary_raw.dist_m - primary_raw.tail_m, 0.01)

        if eff_dist_raw <= cfg.d_emergency_m:
            return cfg.emergency_decel_ms2, True

        v_close_raw = v_ego - primary_raw.v_lead_ms
        if v_close_raw > cfg.standstill_speed_ms:
            ttc = eff_dist_raw / max(v_close_raw, TTC_MIN_VCLOSE_MS)
            if ttc < cfg.ttc_hard_s:
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
            return 0.0, False

        try:
            level = int(Settings.acc_gap_level)
        except (TypeError, ValueError):
            level = 0
        t_headway = _headway_for_level(level) if level else cfg.t_headway_s

        a_chain = cfg.max_accel_ms2
        for n, lead in enumerate(chain_smooth):
            eff_dist = max(lead.dist_m - lead.tail_m, 1e-3)
            a_iidm = _iidm(
                s=eff_dist,
                v=v_ego,
                v_lead=lead.v_lead_ms,
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
                v_lead=lead.v_lead_ms,
                a_lead=lead.a_lead_ms2,
                a_max=cfg.a_max_ms2,
            )
            a_acc = _acc_blend(a_iidm, a_cah_val, cfg.b_comfort_ms2, cfg.cool_factor_c)
            a_acc = _clamp(a_acc, cfg.max_decel_ms2, cfg.max_accel_ms2)

            w_n = cfg.ma_weight_decay ** n
            a_eff = a_acc + (1.0 - w_n) * cfg.a_max_ms2
            if a_eff < a_chain:
                a_chain = a_eff

        a_chain = _clamp(a_chain, cfg.max_decel_ms2, cfg.max_accel_ms2)
        is_emergency = a_chain <= cfg.max_decel_ms2 + 1e-6
        return a_chain, is_emergency

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
