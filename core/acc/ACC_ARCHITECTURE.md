# ACC Gap-Control Architecture

This document describes the gap-control law in
`core/cruise_control_thread/acc_controller.py`. ACC consumes the in-lane lead
published by `ACCThread` (see `core/acc/AGENTS.md`) and returns an upper
bound on commanded acceleration in m/s². The outer cruise control loop in
`cruise_control_thread.py` takes `min(speed_pid_output, acc_cap)` so this
module is a *cap*, not a speed regulator.

---

## 1. Design goal — smooth, stable, and safe

The controller is **not** trying to imitate a human driver. It is trying to
be:

* **Smooth in equilibrium** — when ego is locked at the desired gap and
  speed, the command sits at zero with no micro-corrections.
* **String-stable in traffic** — disturbances must attenuate as they travel
  upstream. Any amplification creates the familiar stop-and-go waves known as
  phantom traffic.
* **Reactive when necessary** — genuine lead braking events must trigger an
  immediate response, but only to the degree required for safety.
* **Predictive beyond the immediate lead** — ACC can also observe vehicles
  ahead of the current lead, allowing earlier and gentler responses to forming
  slowdowns.

The primary objective is to maintain an optimal balance between **safety** and
**traffic-flow smoothness**. Excessively harsh braking, while safe in the short
term, can inject unnecessary disturbances into the traffic stream. Those
perturbations propagate upstream, often growing into phantom traffic jams.
A production-quality ACC should therefore brake no harder than necessary while
still preserving a robust safety margin.

This philosophy requires more than classical IDM. Classical IDM tends to
recover too aggressively after perturbations and can amplify disturbances near
equilibrium. Our controller instead prioritizes monotonic convergence,
string stability, and anticipatory behavior.

---

## 2. Control philosophy — equilibrium first, intervention second

The core is the **Improved IDM (IIDM)**, with a **Constant-Acceleration
Heuristic (CAH)** overlay blended in via the **ACC model** of Kesting,
Treiber & Helbing (2010).

This combination is chosen specifically because it supports three often
competing goals simultaneously:

1. **Stable equilibrium** — once the desired gap is reached, acceleration
   naturally converges to zero without oscillation.
2. **Minimal disturbance propagation** — small speed changes in upstream
   traffic are absorbed rather than amplified.
3. **Safety under rapid transients** — sudden lead braking still produces a
   decisive response.

IIDM serves as the default operating mode because it produces smooth,
well-damped behavior. CAH acts only when IIDM would otherwise respond too
softly to a developing hazard. This ensures that aggressive braking is treated
as an exception, not the norm.

---

## 3. Multi-vehicle anticipation

Unlike a purely reactive ACC that only considers the immediate lead, this
controller can also access vehicles further upstream.

This additional look-ahead enables:

* Earlier recognition of developing slowdowns.
* Reduced reliance on abrupt corrective braking.
* Improved damping of stop-and-go waves.
* More natural and confidence-inspiring longitudinal behavior.

In practice, this means the controller can begin easing off the throttle—or
apply mild braking—before the immediate lead has fully reacted. The result is
lower jerk, improved comfort, and significantly better traffic-flow stability.

This upstream awareness is especially valuable in dense traffic, where waiting
for the direct lead alone often forces unnecessarily sharp interventions.

---

## 4. Smoothness versus safety

Safety always has priority, but smoothness is a core design requirement rather
than a secondary comfort feature.

A controller that brakes too aggressively for routine disturbances may remain
collision-free, yet still perform poorly at the traffic-system level. Each
unnecessary deceleration creates a disturbance that following vehicles must
amplify to compensate. Repeated over many vehicles, these disturbances can
trigger phantom traffic jams.

Therefore, the braking strategy follows this hierarchy:

1. Use the **gentlest acceleration reduction** that preserves the target gap.
2. Escalate braking only when the predicted trajectory requires it.
3. Reserve strong deceleration for true safety-critical events.
4. Avoid rapid recovery immediately after a transient.

This approach preserves both occupant comfort and overall traffic throughput.

---

## 5. String stability as a first-class requirement

String stability is not merely a desirable property—it is a central design
constraint.

A string-stable ACC attenuates disturbances as they move upstream. This is
critical for preventing phantom traffic formation. Every component of the
controller is tuned with this in mind:

* IIDM provides monotonic convergence to the desired gap.
* Headway values remain at or above the stability boundary.
* Jerk limiting prevents high-frequency acceleration disturbances.
* Input filtering suppresses sensor and network jitter.
* Multi-vehicle anticipation reduces late, high-magnitude corrections.

Together, these measures ensure that the vehicle acts as a **traffic damper**
rather than a disturbance amplifier.

---

## 6. Practical behavior targets

The desired on-road behavior is:

* No oscillatory "rubber-banding" around the target gap.
* No unnecessary brake taps in steady traffic.
* Minimal jerk during routine speed adjustments.
* Early, gentle responses to upstream slowdowns.
* Strong braking only when required for safety.
* Smooth recovery after disturbances without overshoot.

An ideal ACC should feel calm and predictable while simultaneously improving
traffic flow for vehicles behind it.

---

## 7. Guiding principle

The controller should behave like a **traffic stabilizer**.

It must:

* protect safety margins,
* maintain realistic and comfortable vehicle dynamics,
* minimize unnecessary braking,
* and actively suppress the formation of phantom traffic.

The best ACC intervention is often the one that drivers barely notice, yet
which prevents a disturbance from ever developing into a larger traffic wave.

---

## 8. Control law — IIDM, CAH, and the ACC blend

The control law is built from three components, applied in order:

1. **Improved IDM (IIDM)** — primary continuous control law. Smooth in
   equilibrium, monotone in `s`, no free-term recovery overshoot.
2. **Constant-Acceleration Heuristic (CAH)** — kinematic safety floor.
   Provides correct authority when the lead is decelerating and ego's
   relative speed is small (the case classical IDM under-reacts to).
3. **ACC blend** — combines the two via a smooth `tanh` transition with
   cool factor `c`, so CAH only takes over when IIDM is unsafe.

### 8.1 IIDM core

Let `s` be the effective bumper-to-bumper gap, `v` ego speed, `v_lead`
the lead's effective speed, and `Δv = v − v_lead` the closing speed
(positive when ego is catching up).

Desired dynamic gap (Treiber/Helbing 2000):

```
s*(v, Δv) = s0 + max(0, v · T  +  v · Δv / (2 · √(a_max · b)))
```

Free-flow accel (limit when no lead is in range):

```
a_free = a_max · (1 − (v / v0)^δ)
```

In our regime `v0 ≫ v` so `a_free ≈ a_max`. The free term remains in the
formula because it shapes the transition near `s = s*` correctly even
when `a_free` happens to equal `a_max`.

IIDM piecewise (Treiber & Kesting 2013, ch. 11.3.4):

```
z = s*(v, Δv) / s

if z ≥ 1:                  # gap closer than desired — pure braking branch
    a_iidm = a_max · (1 − z²)

else:                      # gap at or beyond desired — bounded approach to a_free
    a_iidm = a_free · (1 − z^(2 · a_max / max(a_free, ε)))
```

The lower branch is the key correction over classical IDM: instead of
allowing `(s*/s)²` to vanish and the free term to dominate (overshoot),
IIDM caps growth at `a_free` and enforces C¹ continuity at `z = 1`.

| Property | Mechanism |
|---|---|
| `a = 0` exactly at `s = s*, Δv = 0` | both branches give 0 at `z = 1` |
| No free-term recovery overshoot | upper bound on lower branch is `a_free` |
| Smooth across `z = 1` | C¹ at the boundary by construction |
| Crash-free | `z²` unbounded as `s → 0` |
| String stable | provable for `T ≥ 2τ` and standard truck params |

### 8.2 CAH overlay

CAH is the closed-form maximum ego accel that avoids collision under the
assumption that both vehicles hold their current acceleration until stop.
It does not depend on `Δv` and therefore does not require the closing
speed to develop before responding — the failure mode classical IDM
exhibits when "lead brakes hard and ego is matching the deceleration".

Cap lead's accel by ego's authority so CAH cannot demand more than the
truck can produce:

```
a_lead_eff = min(a_lead, a_max)
```

Two-branch formula (Kesting, Treiber, Helbing 2010):

```
if v_lead · (v − v_lead)  ≤  −2 · s · a_lead_eff:
    a_cah = (v² · a_lead_eff) / (v_lead² − 2 · s · a_lead_eff)

else:
    a_cah = a_lead_eff − (v − v_lead)² · H(v − v_lead) / (2 · s)
```

where `H(x) = 1 if x > 0 else 0`. The first branch covers the case
where lead's residual stopping distance is shorter than the gap allows
(ego must mirror lead's decel, scaled by speed-squared ratios). The
second branch covers the case where there is enough room to glide to a
stop using only the closing-speed energy.

CAH supersedes the legacy `K_FF · a_lead` feedforward term entirely.
Anticipation now comes from a kinematically correct limit, not a hand-
tuned scalar.

### 8.3 ACC blend

CAH alone is too aggressive in steady state — it commits to the worst
case every tick. The ACC model (Kesting et al. 2010) blends IIDM with
CAH only when CAH demands more braking than IIDM:

```
if a_iidm ≥ a_cah:
    a_acc = a_iidm                               # IIDM passthrough — comfort regime

else:
    a_acc = (1 − c) · a_iidm
          + c · (a_cah  +  b · tanh((a_iidm − a_cah) / b))
```

with cool factor `c = 0.99` (Kesting et al recommend `c ∈ [0.95, 0.99]`)
and `b = b_comfort`. The `tanh` makes the blend C¹. In equilibrium the
first branch holds and CAH contributes nothing — no comfort cost.

---

## 9. Multi-vehicle anticipation

`ACCThread` already publishes the top-3 in-lane leads by score (see
`core/acc/AGENTS.md §6`). The controller treats them as a longitudinal
chain and applies a multi-anticipative extension:

### 9.1 Chain construction

Per tick, under `acc.data._lock`:

1. Copy `leads[:]` (shallow).
2. Filter to `dist_m > 0` and `effective_speed_ms` finite.
3. Sort ascending by `dist_m`. Index 0 is the immediate lead, indices
   1+ are anticipated leads.
4. Cap the chain at `MA_MAX_LEADS = 3`.

Vehicles that are not strictly ahead of the previous chain member by at
least `s0_m + 1.0 m` are dropped — they are either lateral noise or
ghost duplicates from the radar pipeline.

### 9.2 Per-lead control evaluation

For each chain member `n ∈ {0, 1, 2}`, the IIDM/CAH/ACC blend from §8 is
evaluated treating that vehicle as if it were the immediate lead. The
gap input is the **direct** gap to that vehicle (sum of intermediate
vehicle lengths and gaps is implicit in `dist_m`).

This produces three candidate caps `a_acc^{(0)}, a_acc^{(1)}, a_acc^{(2)}`.

### 9.3 Combining the chain

The final cap is a **weighted minimum**:

```
w_0 = 1.0
w_n = w_anticipation^n          # geometric decay; default w_anticipation = 0.5

a_eff^{(n)} = a_acc^{(n)}  +  (1 − w_n) · a_max     # soften further-out leads

a_chain = min over n of a_eff^{(n)}
```

The softening term `(1 − w_n) · a_max` is added to anticipated leads so
they only bind the cap when their command is *meaningfully tighter* than
the immediate lead's. With `w = 0.5`:

* `n = 0`: no softening, immediate lead always binds at full authority.
* `n = 1`: lead-of-lead must demand ≥ 0.5·a_max more braking than the
  immediate lead before it changes the output.
* `n = 2`: third-ahead must demand ≥ 0.75·a_max more braking before it
  binds.

This produces the **early, gentle response to forming slowdowns**
described in §3 without making anticipated leads dominate the command in
normal driving. A lead-of-lead that is only mildly slowing has no effect;
a lead-of-lead that is hammering the brakes pulls the cap down before
the immediate lead has even started reacting.

### 9.4 Safety overlays still use the immediate lead

The TTC, emergency, and standstill overlays in §10 are evaluated against
chain index 0 only. Anticipation is for *smoothing*, not for tripping
emergency action — a hazard two cars away that the immediate lead has
not yet reacted to is not yet an emergency for ego.

---

## 10. Safety overlays

Sit on top of the IIDM/CAH/ACC blend. Each can short-circuit the rest of
the pipeline. Overlays consume **raw** lead distance and lead speed (not
the EMA-smoothed values) so the smoothing layer cannot mask a true
emergency.

| Overlay | Trigger | Output | Bypass |
|---|---|---|---|
| Emergency band | `eff_dist ≤ 1.5 m` | `−8.0 m/s²` | jerk + EMA |
| TTC hard floor | `v_close > 0.3` AND `raw_eff_dist / v_close < 1.5 s` | `MAX_DECEL = −6.55 m/s²` | jerk + EMA |
| Standstill hold | `v_ego < 0.4` AND `v_lead < 0.4` AND `eff_dist ≤ s0 + 1.0` | `−0.6 m/s²` | none |
| At-clamp hard | `a_chain ≤ MAX_DECEL + 1e-6` | as-is | jerk + EMA |

The standstill hold is a real-vehicle extension to the textbook IIDM —
without it, IIDM commands exactly zero at a dead stop and the truck
creeps against the torque converter / engine idle.

---

## 11. Pipeline

```
  ACCThread.data.leads[0..2]
        │
        ▼
  _read_chain        — sort by dist, sanity filter, lock-scoped copy
        │
        ▼
  _smooth_inputs     — distance-adaptive EMA on (s, v_lead);
                       asymmetric EMA on a_lead (fast on brake, slow on relax)
        │
        ▼
  _compute_command   — emergency band, TTC floor, standstill hold,
                       per-lead IIDM/CAH/ACC blend, weighted-min over chain
        │
        ▼
  _jerk_limit        — |da/dt| ≤ 2.5 m/s³, bypassed on emergency
        │
        ▼
  _output_filter     — light EMA (τ ≈ 36 ms), bypassed on emergency
        │
        ▼
  cruise_control_thread.loop:
      wanted = min(speed_pid_accel, acc.accel_cap_ms2(v_ego))
        │
        ▼
  telemetry.commanded_accel_ms2  →  accel_to_pedals mapper
```

---

## 12. Inputs and smoothing

### 12.1 Distance and lead speed — symmetric distance-adaptive EMA

`dist_m` and `v_lead_ms` go through a distance-adaptive EMA — τ ramps
linearly from 80 ms at 20 m to 120 ms at 80 m. Close range stays snappy;
long range is filtered hard to kill TruckersMP packet jitter before it
reaches the IIDM core. Each chain member maintains its own EMA state
keyed on `vehicle.id` so a swap of the primary lead does not cause a
discontinuity on the new lead-of-lead.

### 12.2 Lead acceleration — asymmetric EMA

`a_lead_ms2` uses an **asymmetric** EMA:

```
τ_brake = 30 ms          # fast: when a_lead is becoming more negative
τ_relax = 150 ms         # slow: when a_lead is becoming more positive
τ      = τ_brake if (new_a_lead < prev_a_lead_ema) else τ_relax
```

Rationale: during a real braking event the truth is changing rapidly in
the negative direction; we want to follow it within one tick. On the
positive side (coasting, throttle reapplication, packet jitter), the
signal is mostly noise and should be filtered hard. Asymmetry costs
nothing in equilibrium (both directions converge to the same fixed
point) and preserves CAH reaction time without amplifying jitter into
the IIDM braking channel.

This is the **only** place in the pipeline where lead acceleration is
filtered. The TTC and emergency overlays do not consume `a_lead`, IIDM's
reaction is via `Δv`, so the residual smoothing lag affects only CAH —
which by construction (§8.2) is the part of the controller that *should*
track `a_lead` directly.

### 12.3 Hard-cap at source

`a_lead` is clamped to `[EMERGENCY_DECEL, MAX_ACCEL]` before the EMA —
game physics can spike absurdly on spawn / teleport.

### 12.4 Tail correction

`tail_m` (pivot-to-rear of the lead train: cab tail + trailers; TMP
pivot mid-body, AI pivot 18 % from front) is **not** smoothed — it is
constant per lead vehicle. `eff_dist = lead.dist_m − tail_m`.

---

## 13. Comfort overlays

| Layer | Time constant / cap | Notes |
|---|---|---|
| Jerk limiter | `J_MAX = 2.5 m/s³` | Below 2.94 m/s³ comfort threshold (Bellem 2022). Bypassed on emergency. |
| Output EMA  | `τ = 36 ms` | Legacy α=0.6 per 30 Hz tick, ported to framerate-independent τ. Bypassed on emergency. |

The jerk cap is the dominant smoothness shaper between the control law
(already smooth in `a` by §8) and the actuator. It is **not** bypassed
by sub-emergency CAH commands — only by the explicit safety overlays in
§10. Moderate CAH-driven braking events therefore stay jerk-limited and
feel firm rather than sharp.

---

## 14. String stability — quantitative

For a constant time-headway controller, the textbook PD-equivalent
condition (Yamamura et al. 2025) is

```
h ≥ 2τ
kp > 0
kd > (τ − h) · kp
h · kp + kd ≤ 1 / (2 · m · τ)
```

with τ ≈ system delay. In linearisation, IIDM with truck parameters and
`T ≥ 1.0 s` satisfies this comfortably. The ACC blend with `c → 1`
inherits string stability from IIDM in equilibrium (where `a_iidm ≥
a_cah` holds and CAH is dormant) and only departs from it during
transient under-braking — precisely the regime where giving up some
smoothness for safety is correct.

User-facing gap level (`Settings.acc_gap_level`) maps to four headway
values, all ≥ 1.0 s:

| Level | Headway T | Effective behaviour |
|---|---|---|
| 1 | 1.0 s | Closest — at the string-stability boundary; reactive |
| 2 | 1.5 s | Default — comfortable, stable |
| 3 | 2.0 s | Relaxed |
| 4 | 2.5 s | Farthest — very stable, large equilibrium gap |

Headways below 1.0 s are deliberately not exposed.

---

## 15. Anti-oscillation summary

Cumulative effect of the layers, mapped to specific failure modes the
classical implementation suffered from:

| Failure mode | Mitigation |
|---|---|
| Free-term recovery overshoot (classical IDM) | IIDM piecewise form caps the upper branch at `a_free` |
| Soft response when lead matches ego decel | CAH overlay, blended via cool-factor model |
| Late, sharp brake on slowdowns ahead of lead | Multi-vehicle anticipation chain (§9) |
| Jitter-driven micro-oscillation in equilibrium | symmetric EMA on `s, v_lead`; jerk limiter |
| Brake-event lag from filtering | asymmetric EMA on `a_lead`; TTC + emergency on raw |
| Double-counting of lead accel | legacy `K_FF · a_lead` term removed (CAH supersedes) |
| Phantom traffic jams from accel amplification | `T ≥ 1.0 s` + IIDM string stability + anticipation |
| Step changes at handover (e.g. lead → no-lead) | continuous IIDM domain across all gap regimes |

---

## 16. Tuning hooks

All defaults live in module-level constants and are replicated on
`ACConfig` so tests can override without monkey-patching:

```
a_max_ms2, b_comfort_ms2, delta, v0_ms,
s0_m, t_headway_s,
cool_factor_c,
ma_max_leads, ma_weight_decay, ma_min_chain_gap_m,
ttc_hard_s, d_emergency_m, emergency_decel_ms2,
max_accel_ms2, max_decel_ms2,
standstill_speed_ms, standstill_gap_slack_m, standstill_hold_decel_ms2,
j_max_ms3,
tau_input_near_s, tau_input_far_s, d_input_near_m, d_input_far_m,
tau_alead_brake_s, tau_alead_relax_s,
tau_output_s,
no_lead_ceiling_ms2,
```

Headway by gap level lives in `T_HEADWAY_BY_LEVEL_S` (module constant)
and respects `Settings.acc_gap_level` at every tick.

---

## 17. Public API (unchanged)

```
class AdaptiveCruiseController:
    def __init__(self, config: ACConfig | None = None) -> None: ...
    def accel_cap_ms2(self, ego_speed_ms: float) -> float: ...
    def reset(self) -> None: ...
```

`cruise_control_thread.py` is untouched. The chain is read internally
from `acc_thread.data.leads` under its lock — no signature change.

---

## 18. References

- Treiber, M., Hennecke, A., Helbing, D. (2000). *Congested traffic
  states in empirical observations and microscopic simulations.*
  Physical Review E 62, 1805. — Original IDM.
- Treiber, M., Hennecke, A., Helbing, D. (2006). *Delays, inaccuracies
  and anticipation in microscopic traffic models.* Physica A 360,
  71–88. — Multi-anticipative IDM.
- Kesting, A., Treiber, M., Helbing, D. (2010). *Enhanced Intelligent
  Driver Model to access the impact of driving strategies on traffic
  capacity.* Phil. Trans. R. Soc. A 368, 4585–4605. — IIDM, CAH, ACC
  blend, cool factor.
- Treiber, M., Kesting, A. (2013). *Traffic Flow Dynamics.* Springer,
  ch. 11 (IIDM) and ch. 15 (string stability).
- Treiber, M., Kesting, A. (2025). *Twenty-Five Years of the Intelligent
  Driver Model.* arXiv:2506.05909. — Truck defaults and review of
  variants.
- Schakel, W., van Arem, B., Netten, B. (2010). *Effects of cooperative
  adaptive cruise control on traffic flow stability.* IEEE ITSC. —
  IDM+ precursor; motivation for monotonicity in `s`.
- Bellem, H. et al. (2022). *Standards for passenger comfort in
  automated vehicles: Acceleration and jerk.* ScienceDirect
  S0003687022002046.
- Yamamura, K. et al. (2025). *String Stability Analysis and Design
  Guidelines for PD Controllers in ACC Systems.* Sensors 25(11), 3518.
- Sugiyama, Y. et al. (2008). *Traffic jams without bottlenecks —
  experimental evidence for the physical mechanism of the formation of
  a jam.* New J. Phys. 10, 033001. — Empirical phantom-jam baseline.
- Vahidi, A., Eskandarian, A. (2003). *Research advances in intelligent
  collision avoidance and adaptive cruise control.* IEEE TITS 4(3).
