"""
Adaptive Cruise Controller — IDM-based gap-control command in m/s² space.

The previous implementation stacked legacy PD+FF + two kinematic floors via a
``min`` combine. It oscillated when ego sat near the desired gap (gap-PD pushed
positive → overshoot lead → swung negative) and the discrete stop branches
produced step-outputs at handover.

This rewrite uses the **Intelligent Driver Model** (Treiber/Helbing) as the
core control law. IDM is a single continuous formula that:

* Smoothly approaches a stationary leader without a separate stop branch —
  the desired dynamic distance ``s*`` shrinks naturally as ``v → 0``.
* Cancels exactly at ``v = v_lead`` and ``s = s*`` (no oscillation: free term
  and braking term balance).
* Reacts proportionally to closing speed via the ``v·Δv / (2·√(a·b))`` term in
  ``s*`` — anticipates collisions before they require hard braking.
* Drifts ego in slowly when the gap is much larger than desired (free-flow
  term ≈ ``a_max`` minus a tiny braking contribution).

The controller still publishes an **accel cap** (m/s²); ``cruise_control_thread``
takes ``min(speed_pid_output, accel_cap)`` so the outer speed regulator owns
set-speed tracking. We therefore use a high constant ``V0`` so IDM's free-flow
term never artificially throttles ego below the speed PID's request — this
module's job is purely "what is the maximum safe accel given the lead?".

Safety overlays sit on top of IDM:

* **TTC hard floor** — input noise can push IDM into a feasible-but-unsafe
  command for a few ticks; a time-to-collision check forces ``MAX_DECEL`` when
  ``TTC < TTC_HARD_S`` and ego is closing.
* **Emergency band** — inside ``D_EMERGENCY_M`` of the lead, dump straight to
  ``EMERGENCY_DECEL_MS2`` and bypass smoothing.
* **Standstill hold** — at near-zero speed inside the standstill gap, command
  a small holding decel so ego doesn't creep against torque converter.

Comfort overlay:

* **Jerk limiter** caps ``|da/dt|`` at ``J_MAX_MS3`` (≈ 2.5 m/s³, well below
  the 2.94 m/s³ comfort threshold from automated-vehicle UX literature). The
  limiter is bypassed for emergency commands so hard braking reaches the
  actuator without delay.
* **Light output EMA** (existing) on top, also emergency-bypassed.

Creep suppression / pedal feedforward / road-load / slope compensation all
live in ``core/sending_thread/accel_to_pedals.py`` — this module trusts that
layer completely and never reasons about pedals.

References
----------
- Treiber, Hennecke, Helbing (2000): "Congested traffic states in empirical
  observations and microscopic simulations" — original IDM paper.
- Treiber & Kesting (2025) "Twenty-Five Years of the Intelligent Driver Model"
  — IDM defaults for trucks: T = 1.7 s, s0 = 2 m, a = 0.3 m/s², b = 2 m/s²,
  δ = 4. We use slightly punchier ``a_max`` (1.5 m/s²) since the truck is
  controlled by an outer speed PID and we don't want IDM to be the binding
  constraint on free-road acceleration.
- ISO/comfort literature: longitudinal jerk < 2.94 m/s³ acceptable, < 0.5 m/s³
  imperceptible.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from core.settings import Settings
from core.thread_management.registry import registry

logger = logging.getLogger(__name__)


# Free-road acceleration. IDM uses a · [1 − (v/v0)^δ] for the free term; with
# our high V0 (controller is an accel-cap, not a speed regulator) the bracket
# stays ≈ 1 at all realistic speeds, so a_max effectively becomes the IDM
# upper bound on commanded accel before the lead-aware braking term subtracts.
A_MAX_MS2: float = 1.5

# Comfortable deceleration. Used in the s* denominator: 2·√(a·b). Larger b →
# IDM tolerates closer/faster approaches before commanding big decel; smaller
# b → smoother but earlier braking. 2.0 m/s² matches Treiber's truck default.
B_COMFORT_MS2: float = 2.0

# Free-road accel exponent. δ=4 gives the "smooth approach to v0" curve;
# unused in practice here because V0 ≫ v_ego, but kept for completeness.
DELTA: float = 4.0

# High constant. ACC is an accel-cap, not a speed regulator — V0 sits well
# above any plausible ego speed so the free-flow term ≈ a_max.
V0_MS: float = 40.0   # 144 km/h

# Standstill gap and default headway.
S0_M: float = 3.0
T_HEADWAY_S: float = 1.5

# Per-level headway times (seconds). Index 0 = fallback if Settings ever
# returns something invalid. 1..4 = closest..farthest. The Sensors-2025 PD
# stability paper requires h ≥ 2τ where τ is system delay (~0.2 s). All four
# levels satisfy this comfortably.
T_HEADWAY_BY_LEVEL_S: tuple[float, float, float, float, float] = (
    1.5,  # 0 — fallback
    1.0,  # 1 — closest
    1.5,  # 2 — default
    2.0,  # 3
    2.5,  # 4 — farthest
)


def _headway_for_level(level: int) -> float:
    if 1 <= level <= 4:
        return T_HEADWAY_BY_LEVEL_S[level]
    return T_HEADWAY_BY_LEVEL_S[0]


# Mild anticipatory FF on lead.accel. IDM responds to lead-decel through the
# Δv channel, but FF gives an earlier edge before relative speed has had time
# to develop. Kept small so it can't overpower the IDM core.
K_FF: float = 0.3

# Above this gap multiple (× s*), FF is faded out — we don't need anticipation
# when there's plenty of room.
FF_FADE_GAP_MULT: float = 2.0

# Safety overlay thresholds.
TTC_HARD_S: float = 1.5            # below this and closing → hard decel
TTC_MIN_VCLOSE_MS: float = 0.3     # avoid div-by-zero / slow-creep nuisance
D_EMERGENCY_M: float = 1.5
EMERGENCY_DECEL_MS2: float = -8.0
MAX_ACCEL_MS2: float = 1.5
MAX_DECEL_MS2: float = -6.55

# When ego is essentially stopped and within the standstill gap of a stopped
# lead, command a small holding decel so the truck doesn't creep against the
# torque converter / idle.
STANDSTILL_SPEED_MS: float = 0.4
STANDSTILL_GAP_SLACK_M: float = 1.0
STANDSTILL_HOLD_DECEL_MS2: float = -0.6

# Jerk cap (m/s³). 2.5 sits below the 2.94 m/s³ "unlikely to be acceptable"
# threshold from automated-vehicle comfort literature and well above the
# 0.5 m/s³ "imperceptible" floor. Bypassed for emergency commands.
J_MAX_MS3: float = 2.5

# Distance-adaptive input EMA (retained from legacy). τ ramps linearly from
# TAU_NEAR at D_NEAR to TAU_FAR at D_FAR, clamped outside. Close → snappy;
# far → heavily filtered, kills MP jitter before it reaches the IDM core.
TAU_INPUT_NEAR_S: float = 0.08
TAU_INPUT_FAR_S: float = 0.12
D_INPUT_NEAR_M: float = 20.0
D_INPUT_FAR_M: float = 80.0

# Output low-pass (legacy α=0.6 ported to framerate-independent τ).
TAU_OUTPUT_S: float = 0.036

DT_FALLBACK_S: float = 1.0 / 30.0
DT_MAX_S: float = 0.2
NO_LEAD_CEILING_MS2: float = 10.0


@dataclass(slots=True)
class ACConfig:
    """Tunable parameters. Defaults match the IDM truck tune."""
    a_max_ms2: float = A_MAX_MS2
    b_comfort_ms2: float = B_COMFORT_MS2
    delta: float = DELTA
    v0_ms: float = V0_MS
    s0_m: float = S0_M
    t_headway_s: float = T_HEADWAY_S
    k_ff: float = K_FF
    ff_fade_gap_mult: float = FF_FADE_GAP_MULT
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
    tau_output_s: float = TAU_OUTPUT_S
    no_lead_ceiling_ms2: float = NO_LEAD_CEILING_MS2


@dataclass(slots=True)
class _LeadSnapshot:
    dist_m: float
    v_lead_ms: float
    a_lead_ms2: float
    # Distance from the vehicle's tracked pivot to the physical rear of its
    # train (cab tail + all attached trailers). Subtracted from dist_m before
    # any control calculation so IDM works in actual gap space.
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

        # Output filter & jerk limiter state
        self._output_ema: float | None = None
        self._prev_cmd_ms2: float | None = None

    def accel_cap_ms2(self, ego_speed_ms: float) -> float:
        """Public entry point — unchanged signature so cruise_control_thread is untouched."""
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
            self._prev_cmd_ms2 = None
            return self.config.no_lead_ceiling_ms2

        v_ego = max(0.0, float(ego_speed_ms))
        smooth = self._smooth_lead_inputs(lead, dt)

        a_raw, is_emergency = self._compute_command(smooth, v_ego)
        a_jerk = self._jerk_limit(a_raw, dt, is_emergency)
        a_filt = self._output_filter(a_jerk, dt, is_emergency)
        return a_filt

    def reset(self) -> None:
        self._prev_mono = None
        self._lead_dist_ema = None
        self._lead_speed_ema = None
        self._lead_accel_ema = None
        self._output_ema = None
        self._prev_cmd_ms2 = None

    def _read_lead(self) -> _LeadSnapshot | None:
        """Snapshot the primary lead from acc_thread under its lock."""
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

                # Tail length: pivot → rear of the lead train. AI pivot sits
                # 18 % from front (82 % of cab behind pivot). TMP pivot is at
                # body centre (50 %). TMP trailers are separate vehicle
                # entries and carry no nested trailers list.
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

    def _smooth_lead_inputs(self, lead: _LeadSnapshot, dt: float) -> _LeadSnapshot:
        """Distance-adaptive EMA on (dist, v_lead, a_lead). Retained from legacy."""
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
            tail_m=lead.tail_m,
        )

    def _compute_command(self, lead: _LeadSnapshot, v_ego: float) -> tuple[float, bool]:
        """Return (target_accel_ms2, is_emergency).

        The is_emergency flag bypasses jerk and output smoothing downstream so
        hard-brake commands reach the actuator without lag.
        """
        cfg = self.config

        # Actual gap from ego reference to the physical rear of the lead train.
        # dist_m is pivot-to-pivot; tail_m corrects for cab tail + trailers.
        eff_dist = max(lead.dist_m - lead.tail_m, 0.01)

        # Emergency band: dump to hard decel and flag bypass.
        if eff_dist <= cfg.d_emergency_m:
            return cfg.emergency_decel_ms2, True

        # Closing speed (positive = ego catching up to lead).
        v_close = v_ego - lead.v_lead_ms

        # TTC hard floor. Input EMA can mask a fast-developing closure for
        # ~100 ms; a TTC check on the smoothed values still flags the hazard
        # early enough to save the IDM core from inadequate braking authority.
        if v_close > cfg.standstill_speed_ms:
            ttc = eff_dist / max(v_close, TTC_MIN_VCLOSE_MS)
            if ttc < cfg.ttc_hard_s:
                return cfg.max_decel_ms2, True

        # Standstill hold: ego stopped, lead stopped, gap inside standstill +
        # slack. Small decel prevents idle creep without surprising the user
        # with a hard brake.
        if (
            v_ego < cfg.standstill_speed_ms
            and lead.v_lead_ms < cfg.standstill_speed_ms
            and eff_dist <= cfg.s0_m + cfg.standstill_gap_slack_m
        ):
            return cfg.standstill_hold_decel_ms2, False

        # IDM core. Headway time follows the user's gap-level setting.
        try:
            level = int(Settings.acc_gap_level)
        except (TypeError, ValueError):
            level = 0
        t_headway = _headway_for_level(level) if level else cfg.t_headway_s

        # Desired dynamic distance s*. The Δv term is one-sided (positive
        # only): when the lead is pulling away (v_close < 0) it doesn't shrink
        # s*, it just disappears. Treiber's IDM uses max(0, …) here.
        sqrt_ab = math.sqrt(max(cfg.a_max_ms2 * cfg.b_comfort_ms2, 1e-6))
        s_star_dyn = v_ego * t_headway + (v_ego * v_close) / (2.0 * sqrt_ab)
        s_star = cfg.s0_m + max(0.0, s_star_dyn)

        # Free-flow term (≈ 1 in our regime since v0 ≫ v_ego).
        v_ratio = v_ego / max(cfg.v0_ms, 1e-3)
        free_term = 1.0 - v_ratio ** cfg.delta

        # Braking interaction term. (s_star / eff_dist)² is the crash-prevention
        # core: as eff_dist → s_star, the term saturates at 1 and exactly
        # cancels free_term (a → 0). When eff_dist < s_star, the term grows
        # quadratically, producing strong but continuous braking.
        gap_ratio = s_star / max(eff_dist, 1e-3)
        brake_term = gap_ratio * gap_ratio

        a_idm = cfg.a_max_ms2 * (free_term - brake_term)

        # Mild lead-FF gated by gap. Anticipates lead deceleration before Δv
        # has time to develop; faded out at large gap multiples.
        gap_mult = eff_dist / max(s_star, 1e-3)
        ff_gate = _clamp((cfg.ff_fade_gap_mult - gap_mult) / max(cfg.ff_fade_gap_mult - 1.0, 1e-3), 0.0, 1.0)
        a_ff = cfg.k_ff * lead.a_lead_ms2 * ff_gate

        a_target = a_idm + a_ff
        a_target = _clamp(a_target, cfg.max_decel_ms2, cfg.max_accel_ms2)

        # An IDM command at MAX_DECEL is treated as emergency for downstream
        # smoothing — same convention as the legacy controller.
        is_emergency = a_target <= cfg.max_decel_ms2 + 1e-6
        return a_target, is_emergency

    def _jerk_limit(self, a_new: float, dt: float, is_emergency: bool) -> float:
        """Cap |da/dt| at j_max_ms3. Bypassed on emergency."""
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
        """Light EMA on the commanded accel. Bypassed on emergency."""
        if is_emergency:
            self._output_ema = value
            return value
        if self._output_ema is None or not math.isfinite(self._output_ema):
            self._output_ema = value
            return value
        alpha = 1.0 - math.exp(-dt / max(self.config.tau_output_s, 1e-6))
        self._output_ema = self._output_ema + alpha * (value - self._output_ema)
        return self._output_ema
