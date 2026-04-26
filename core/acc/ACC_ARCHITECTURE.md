# ACC Gap-Control Architecture

This document describes the IDM-based adaptive cruise controller in
`core/cruise_control_thread/acc_controller.py`. ACC consumes the in-lane lead
published by `ACCThread` (see `core/acc/AGENTS.md`) and returns an upper
bound on commanded acceleration in m/s². The outer cruise control loop in
`cruise_control_thread.py` takes `min(speed_pid_output, acc_cap)` so this
module is a *cap*, not a speed regulator.

---

## 1. Why IDM

Earlier versions stacked a legacy PD + feed-forward law against two
kinematic safety floors via a `min` combine. Two failure modes recurred:

- **Oscillation near the desired gap.** Gap-PD pushed positive accel even
  when ego matched lead speed → ego overshot lead → PD swung negative →
  loop.
- **Discrete stop branches.** A dedicated `lead.speed < 0.5 m/s` branch
  produced step changes when the lead came to rest.

The Intelligent Driver Model (Treiber/Helbing 2000; updated review Treiber &
Kesting 2025) is a single continuous formula that resolves both problems:

| Concern | IDM mechanism |
|---|---|
| Smooth stop | `s*` shrinks with `v` → braking term degrades to a clean stop curve |
| No oscillation at match | At `v=v_lead`, `Δv=0` → `s* = s0 + v·T`; if `s = s*`, free term and braking term cancel, `a=0` |
| Slow approach far | `(s*/s)²` is small at large gaps → free term dominates, gentle drift-in |
| Crash-free | `(s*/s)²` is unbounded as `s → 0` |
| Lead-decel reaction | `Δv` term in `s*` lifts the desired distance proportional to closing speed |

Results from the bibliography we reviewed:

- **String stability paper (Sensors 2025, MDPI 25/11/3518):** PD with
  constant time headway is string-stable iff `h ≥ 2τ`, `kp > 0`,
  `kd > (τ−h)·kp`, and `h·kp + kd ≤ 1/(2mτ)`. Useful as a check on our
  effective gains under linearisation.
- **Comfort literature (Bellem et al., ScienceDirect 2022):** longitudinal
  jerk < 2.94 m/s³ acceptable, < 0.5 m/s³ imperceptible. We cap at
  **2.5 m/s³** in the comfort layer, bypassed for emergencies.
- **Treiber 2025:** truck IDM defaults `T = 1.7 s, s0 = 2 m, a = 0.3 m/s²,
  b = 2 m/s², δ = 4`. We use slightly more authority on `a` (1.5 m/s²)
  because ACC is a cap and the outer speed PID owns set-speed tracking.

---

## 2. Pipeline

```
  ACCThread.data.leads[0]
        │
        ▼
  _read_lead       — extracts (dist, v_lead, a_lead, tail) under lock
        │
        ▼
  _smooth_lead_inputs  — distance-adaptive EMA, kills MP jitter
        │
        ▼
  _compute_command — emergency band, TTC floor, standstill hold,
                     IDM core, mild lead-FF
        │
        ▼
  _jerk_limit      — |da/dt| ≤ 2.5 m/s³, bypassed on emergency
        │
        ▼
  _output_filter   — light EMA (τ ≈ 36 ms), bypassed on emergency
        │
        ▼
  cruise_control_thread.loop:
      wanted = min(speed_pid_accel, acc.accel_cap_ms2(v_ego))
        │
        ▼
  telemetry.commanded_accel_ms2  →  accel_to_pedals mapper
```

---

## 3. IDM core

```
s_star_dyn = v_ego · T  +  v_ego · Δv  /  (2 · √(a_max · b_comfort))
s_star     = s0 + max(0, s_star_dyn)

free_term   = 1 − (v_ego / v0)^δ                       # ≈ 1 in our regime
brake_term  = (s_star / eff_dist)²

a_idm       = a_max · (free_term − brake_term)
a_target    = a_idm + K_FF · a_lead · ff_gate
a_target    = clamp(a_target, MAX_DECEL, MAX_ACCEL)

  Δv     = v_ego − v_lead          (positive = closing)
  T      = headway, set by Settings.acc_gap_level (1.0 / 1.5 / 2.0 / 2.5 s)
  s0     = 3.0 m
  a_max  = 1.5 m/s²
  b      = 2.0 m/s²
  δ      = 4
  v0     = 40 m/s          (high constant — ACC is a cap, not a regulator)
  K_FF   = 0.3
  ff_gate = clamp((FF_FADE − gap_mult)/(FF_FADE − 1), 0, 1)   # off when far
  FF_FADE = 2.0   (× s_star)
```

`v0` is intentionally above any plausible ego speed so the free-flow term
stays ≈ 1. With this choice IDM acts as a pure lead-aware accel cap; the
outer speed PID owns set-speed tracking, and `min()` binds whichever is
tighter.

`eff_dist` is `lead.dist_m − tail_m` where `tail_m` is the pivot-to-rear
length of the lead train (cab tail + trailers; TMP pivot is mid-body, AI
pivot 18 % from front).

---

## 4. Safety overlays

Sit on top of the IDM core. Each can short-circuit the rest of the pipeline.

| Overlay | Trigger | Output | Bypass |
|---|---|---|---|
| Emergency band | `eff_dist ≤ 1.5 m` | `−8.0 m/s²` | jerk + EMA |
| TTC hard floor | `v_close > 0.3` AND `eff_dist / v_close < 1.5 s` | `MAX_DECEL = −6.55 m/s²` | jerk + EMA |
| Standstill hold | `v_ego < 0.4` AND `v_lead < 0.4` AND `eff_dist ≤ 4` | `−0.6 m/s²` | none |
| At-clamp hard | `a_target ≤ MAX_DECEL + 1e-6` | as-is | jerk + EMA |

The TTC floor exists because the input EMA (τ ≈ 80–120 ms) can briefly
mask a fast-developing closure. TTC computed on the smoothed values still
catches the hazard early enough to give the IDM core room to brake.

The standstill hold is a real-vehicle extension to the textbook IDM —
without it, IDM commands exactly zero at a dead stop and the truck creeps
against the torque converter / engine idle.

---

## 5. Comfort overlays

| Layer | Time constant / cap | Notes |
|---|---|---|
| Jerk limiter | `J_MAX = 2.5 m/s³` | Comfort literature: < 2.94 m/s³ acceptable. Bypassed on emergency. |
| Output EMA  | `τ = 36 ms` | Legacy α=0.6 per 30 Hz tick, ported to framerate-independent τ. Bypassed on emergency. |

---

## 6. Inputs and smoothing

Lead inputs (`dist_m`, `v_lead_ms`, `a_lead_ms2`) pass through a
distance-adaptive EMA — τ ramps from 80 ms at 20 m to 120 ms at 80 m. Close
range stays snappy for stop-and-go responsiveness; long range is heavily
filtered to kill TruckersMP packet jitter before it touches the IDM core.

`a_lead` is clamped to `[EMERGENCY_DECEL, MAX_ACCEL]` at source — game
physics can spike absurdly on spawn / teleport.

The `tail_m` geometric correction is **not** smoothed (it's a constant per
lead vehicle).

---

## 7. Comparison with old controller

| Concern | Old: PD + FF + kinematic floors | New: IDM + safety + comfort |
|---|---|---|
| Stop branch | discrete `lead.speed < 0.5` | continuous via `s*(v, Δv)` |
| Anti-oscillation | none — gap-PD always biased to close | exact cancel at `s = s*`, `Δv = 0` |
| Far approach | hard PD pull → lurch | gentle free-term drift-in |
| Lead reaction | `min(pid, a_req_close, a_req_brake)` | single continuous formula + TTC overlay |
| Authority shaping | `closeness_amp`, `acceleration_amp` non-linear scalars | intrinsic to IDM `(s*/s)²` |
| Jerk | unbounded | capped at 2.5 m/s³ |

---

## 8. Tuning hooks

All defaults live in module-level constants and are replicated on
`ACConfig` so tests can override without monkey-patching:

```
a_max_ms2, b_comfort_ms2, delta, v0_ms,
s0_m, t_headway_s,
k_ff, ff_fade_gap_mult,
ttc_hard_s, d_emergency_m, emergency_decel_ms2,
max_accel_ms2, max_decel_ms2,
standstill_speed_ms, standstill_gap_slack_m, standstill_hold_decel_ms2,
j_max_ms3,
tau_input_near_s, tau_input_far_s, d_input_near_m, d_input_far_m,
tau_output_s,
no_lead_ceiling_ms2,
```

Headway by gap level lives in `T_HEADWAY_BY_LEVEL_S` (module constant) and
respects `Settings.acc_gap_level` at every tick.

---

## 9. Public API (unchanged)

```
class AdaptiveCruiseController:
    def __init__(self, config: ACConfig | None = None) -> None: ...
    def accel_cap_ms2(self, ego_speed_ms: float) -> float: ...
    def reset(self) -> None: ...
```

`cruise_control_thread.py` is untouched.

---

## 10. References

- Treiber, M., Hennecke, A., Helbing, D. (2000). *Congested traffic states
  in empirical observations and microscopic simulations.* Physical
  Review E 62, 1805.
- Treiber, M., Kesting, A. (2025). *Twenty-Five Years of the Intelligent
  Driver Model.* arXiv:2506.05909.
- Bellem, H. et al. (2022). *Standards for passenger comfort in automated
  vehicles: Acceleration and jerk.* ScienceDirect S0003687022002046.
- Yamamura, K. et al. (2025). *String Stability Analysis and Design
  Guidelines for PD Controllers in ACC Systems.* Sensors 25(11), 3518.
- Vahidi, A., Eskandarian, A. (2003). *Research advances in intelligent
  collision avoidance and adaptive cruise control.* IEEE TITS 4(3).
