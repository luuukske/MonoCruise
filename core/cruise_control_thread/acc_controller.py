"""
Adaptive Cruise Controller — gap-control command in m/s² space.

Legacy PD + feed-forward law, unchanged in shape and gains — those gains were
already dimensioned in m/s² (see module notes below). Two targeted changes
relative to the legacy implementation:

1. **Continuous kinematic stopping floor** replaces the discrete
   ``lead.speed < 0.5`` branch that produced step-outputs when the lead came
   to rest. A single kinematic required-accel is evaluated every tick and
   used as a conservative floor on the PD+FF output. It smoothly degenerates
   to the classic stop-at-gap curve as ``v_lead → 0``.
2. **Distance-adaptive input EMA** on ``(dist, v_lead, a_lead)`` damps
   multiplayer jitter when the lead is far away without lagging the
   close-range response.

Creep suppression / pedal feedforward / road-load / slope compensation all
live in ``core/sending_thread/accel_to_pedals.py`` — this module trusts that
layer completely and never reasons about pedals.

Units note
----------
Legacy ``K_F * lead.accel * acc_amp * acceleration_amp`` is m/s² by
construction (lead.accel is m/s², rest are scalars). Since the controller
output is ``closeness_amp * (p + d) + ff`` and the whole thing must be
m/s², ``K_P`` carries units of 1/s² and ``K_D`` carries units of 1/s. With
the legacy numbers ``K_P = 0.06`` → ω_n ≈ 0.245 rad/s, ``K_D = 0.46`` →
damping ratio ≈ 0.94 — near-critically-damped 2nd-order.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from core.thread_management.registry import registry

logger = logging.getLogger(__name__)


# --- Legacy PD + FF gains (already m/s²-correct — do not rescale) -----------
K_P: float = 0.06     # [1/s²] — gap-error proportional
K_D: float = 0.46     # [1/s]  — relative-speed derivative
K_F: float = 1.0      # [1]    — feed-forward on lead accel
ACC_AMP: float = 1.0  # [1]    — overall FF scalar (legacy tune hook)

# --- Geometry --------------------------------------------------------------
MIN_GAP_M: float = 3.0
T_HEADWAY_S: float = 1.5

# --- Envelope --------------------------------------------------------------
MAX_ACCEL_MS2: float = 1.5
MAX_DECEL_MS2: float = -6.55
EMERGENCY_DECEL_MS2: float = -8.0
D_EMERGENCY_M: float = 1.5

# --- Input EMA (distance-adaptive) ------------------------------------------
# τ ramps linearly from TAU_NEAR at D_NEAR to TAU_FAR at D_FAR, clamped
# outside. Close → snappy; far → heavily filtered, kills MP jitter before it
# reaches the gains.
TAU_INPUT_NEAR_S: float = 0.08
TAU_INPUT_FAR_S: float = 0.12
D_INPUT_NEAR_M: float = 20.0
D_INPUT_FAR_M: float = 80.0

# --- Output low-pass (legacy α=0.6 ported to framerate-independent τ) ------
# α=0.6 per 1/30 s sample ↔ τ = −dt/ln(1−α) ≈ 0.036 s
TAU_OUTPUT_S: float = 0.036

# --- Misc ------------------------------------------------------------------
DT_FALLBACK_S: float = 1.0 / 30.0
DT_MAX_S: float = 0.2
NO_LEAD_CEILING_MS2: float = 10.0


@dataclass(slots=True)
class ACConfig:
    """Tunable parameters. Defaults match the legacy tune."""
    k_p: float = K_P
    k_d: float = K_D
    k_f: float = K_F
    acc_amp: float = ACC_AMP
    min_gap_m: float = MIN_GAP_M
    t_headway_s: float = T_HEADWAY_S
    max_accel_ms2: float = MAX_ACCEL_MS2
    max_decel_ms2: float = MAX_DECEL_MS2
    emergency_decel_ms2: float = EMERGENCY_DECEL_MS2
    d_emergency_m: float = D_EMERGENCY_M
    tau_input_near_s: float = TAU_INPUT_NEAR_S
    tau_input_far_s: float = TAU_INPUT_FAR_S
    d_input_near_m: float = D_INPUT_NEAR_M
    d_input_far_m: float = D_INPUT_FAR_M
    tau_output_s: float = TAU_OUTPUT_S
    no_lead_ceiling_ms2: float = NO_LEAD_CEILING_MS2


@dataclass(slots=True)
class _LeadSnapshot:
    dist_m: float
    v_lead_ms: float
    a_lead_ms2: float
    # Distance from the vehicle's tracked pivot to the physical rear of its
    # train (cab tail + all attached trailers).  Subtracted from dist_m before
    # any control calculation so the controller works in actual gap space, not
    # pivot-to-pivot space.
    tail_m: float = 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class AdaptiveCruiseController:
    """Commands an upper bound on longitudinal accel (m/s²) from the lead."""

    def __init__(self, config: ACConfig | None = None) -> None:
        self.config = config or ACConfig()
        self._prev_mono: float | None = None

        # Smoothed lead inputs
        self._lead_dist_ema: float | None = None
        self._lead_speed_ema: float | None = None
        self._lead_accel_ema: float | None = None

        # Output filter state
        self._output_ema: float | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def accel_cap_ms2(self, ego_speed_ms: float) -> float:
        now = time.monotonic()
        if self._prev_mono is None:
            dt = DT_FALLBACK_S
        else:
            dt = _clamp(now - self._prev_mono, 1e-3, DT_MAX_S)
        self._prev_mono = now

        lead = self._read_lead()
        if lead is None:
            # No lead → clear filters so next engagement starts clean.
            self._lead_dist_ema = None
            self._lead_speed_ema = None
            self._lead_accel_ema = None
            self._output_ema = None
            return self.config.no_lead_ceiling_ms2

        v_ego = max(0.0, float(ego_speed_ms))
        smooth = self._smooth_lead_inputs(lead, dt)
        a_raw = self._compute_command(smooth, v_ego)
        a_filt = self._output_filter(a_raw, dt)
        return a_filt

    def reset(self) -> None:
        self._prev_mono = None
        self._lead_dist_ema = None
        self._lead_speed_ema = None
        self._lead_accel_ema = None
        self._output_ema = None

    # -----------------------------------------------------------------------
    # Lead acquisition
    # -----------------------------------------------------------------------
    def _read_lead(self) -> _LeadSnapshot | None:
        try:
            acc = registry.get_thread("acc_thread")
        except KeyError:
            return None
        try:
            if not acc.is_alive():
                return None
        except AttributeError:
            return None
        try:
            with acc.data._lock:
                if not acc.data.has_lead or not acc.data.leads:
                    return None
                primary = acc.data.leads[0]
                dist_m = float(primary.dist_m)
                v_lead = float(primary.effective_speed_ms)
                a_lead = float(primary.effective_accel_ms2)

                # Tail length: pivot → rear of the lead train (held under lock
                # since vehicle is a shared reference from RadarThread).
                # AI pivot sits 18 % from front (82 % of cab behind pivot).
                # TMP pivot is at body centre (50 %).  TMP trailers are separate
                # vehicle entries and carry no nested trailers list.
                vehicle = primary.vehicle
                if vehicle.is_tmp:
                    tail_m = 0.5 * float(vehicle.size.length)
                else:
                    tail_m = 0.82 * float(vehicle.size.length)
                    for trailer in vehicle.trailers:
                        if not trailer.is_zero():
                            tail_m += float(trailer.size.length)
        except (AttributeError, IndexError):
            return None

        if not (math.isfinite(dist_m) and math.isfinite(v_lead) and math.isfinite(a_lead)):
            return None
        # Game accel can spike absurdly on spawn/teleport — clamp at source.
        a_lead = _clamp(a_lead, self.config.emergency_decel_ms2, self.config.max_accel_ms2)
        return _LeadSnapshot(dist_m=dist_m, v_lead_ms=v_lead, a_lead_ms2=a_lead, tail_m=tail_m)

    # -----------------------------------------------------------------------
    # Input smoothing — τ scales with distance
    # -----------------------------------------------------------------------
    def _smooth_lead_inputs(self, lead: _LeadSnapshot, dt: float) -> _LeadSnapshot:
        cfg = self.config
        # Anchor τ on the most recent smoothed distance so τ itself doesn't
        # oscillate when dist is jittering. Fall back to raw on first sample.
        anchor_d = self._lead_dist_ema if self._lead_dist_ema is not None else lead.dist_m
        span = max(cfg.d_input_far_m - cfg.d_input_near_m, 1e-3)
        t = _clamp((anchor_d - cfg.d_input_near_m) / span, 0.0, 1.0)
        tau = cfg.tau_input_near_s + (cfg.tau_input_far_s - cfg.tau_input_near_s) * t
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))

        def _step(prev: float | None, new: float) -> float:
            return new if prev is None else prev + alpha * (new - prev)

        self._lead_dist_ema = _step(self._lead_dist_ema, lead.dist_m)
        self._lead_speed_ema = _step(self._lead_speed_ema, lead.v_lead_ms)
        self._lead_accel_ema = _step(self._lead_accel_ema, lead.a_lead_ms2)
        return _LeadSnapshot(
            dist_m=self._lead_dist_ema,
            v_lead_ms=self._lead_speed_ema,
            a_lead_ms2=self._lead_accel_ema,
            tail_m=lead.tail_m,  # geometric constant — not smoothed
        )

    # -----------------------------------------------------------------------
    # Control law — legacy PD+FF with continuous kinematic stopping floor
    # -----------------------------------------------------------------------
    def _compute_command(self, lead: _LeadSnapshot, v_ego: float) -> float:
        cfg = self.config

        # Actual gap from ego reference to the physical rear of the lead train.
        # dist_m is pivot-to-pivot; tail_m corrects for cab tail + trailers.
        eff_dist = max(lead.dist_m - lead.tail_m, 0.01)

        # Emergency: inside safety floor, dump straight to hard decel.
        if eff_dist <= cfg.d_emergency_m:
            return cfg.emergency_decel_ms2

        # --- Legacy PD + FF ------------------------------------------------
        desired_gap = cfg.min_gap_m + cfg.t_headway_s * v_ego
        gap_error = eff_dist - desired_gap             # + too far, − too close
        speed_error = lead.v_lead_ms - v_ego           # + lead pulling away
        actual_time_gap = max(eff_dist / max(v_ego, 0.1), 0.0)

        closeness_amp = pow(0.8, actual_time_gap * 10.0 - 3.0) + 0.6
        if speed_error < 0.1:
            closeness_amp = closeness_amp ** 0.8
        else:
            closeness_amp = _clamp(closeness_amp, 0.7, 1.2)

        p_term = cfg.k_p * gap_error
        d_term = cfg.k_d * speed_error

        acceleration_amp = max(-((actual_time_gap / 2.0) ** 3) + 1.0, 0.2)
        ff_term = cfg.k_f * lead.a_lead_ms2 * cfg.acc_amp * acceleration_amp

        pid_accel = closeness_amp * (p_term + d_term) + ff_term

        # --- Continuous kinematic stopping floor --------------------------
        # a_req matches ego speed to lead speed at d_lead = MIN_GAP,
        # accounting for the lead's own accel. Same formula regardless of
        # whether the lead is cruising, braking, or stopped — when v_lead=0
        # it reduces to a_lead − v_ego²/(2·d_eff), the classic stop curve.
        v_close = v_ego - lead.v_lead_ms
        d_eff = max(eff_dist - cfg.min_gap_m, 0.5)
        if v_close > 0.0:
            a_req_close = lead.a_lead_ms2 - (v_close * v_close) / (2.0 * d_eff)
        else:
            # Not closing — kinematic floor disengages (coast with lead).
            a_req_close = cfg.max_accel_ms2

        # --- Lead-braking projection floor --------------------------------
        # When the lead is actively decelerating, project where it will stop
        # and demand ego stop by that point (minus min_gap). Anticipatory —
        # fires even at v_close ≈ 0 so brake onset doesn't wait for the gap
        # to collapse. Mirrors the legacy "decel" stopping branch.
        if lead.a_lead_ms2 < -0.5 and lead.v_lead_ms > 0.1:
            lead_stop_dist = -(lead.v_lead_ms * lead.v_lead_ms) / (2.0 * lead.a_lead_ms2)
            target_stop_pos = max(d_eff + lead_stop_dist, 0.5)
            a_req_brake = -(v_ego * v_ego) / (2.0 * target_stop_pos)
        else:
            a_req_brake = cfg.max_accel_ms2

        # Conservative combine: each kinematic law only binds when it demands
        # more braking than the PID. `min` is continuous, no branch-step.
        target = min(pid_accel, a_req_close, a_req_brake)

        return _clamp(target, cfg.max_decel_ms2, cfg.max_accel_ms2)

    # -----------------------------------------------------------------------
    # Output low-pass — emergency bypasses
    # -----------------------------------------------------------------------
    def _output_filter(self, value: float, dt: float) -> float:
        # Hard-brake bypass so emergency reaches the actuator without lag.
        if value < self.config.max_decel_ms2 + 1e-6 or value <= self.config.emergency_decel_ms2 + 1e-6:
            self._output_ema = value
            return value
        if self._output_ema is None or not math.isfinite(self._output_ema):
            self._output_ema = value
            return value
        alpha = 1.0 - math.exp(-dt / max(self.config.tau_output_s, 1e-6))
        self._output_ema = self._output_ema + alpha * (value - self._output_ema)
        return self._output_ema
