# ACC Gap-Control Architecture

This document describes the gap-control law in
`core/cruise_control_thread/acc_controller.py`. ACC consumes the in-lane lead
published by `ACCThread` (see `core/acc/README.md`) and returns an upper
bound on commanded acceleration in m/s². The outer cruise control loop in
`cruise_control_thread.py` takes `min(speed_pid_output, acc_cap)` so this
module is a *cap*, not a speed regulator.

---

## 1. Design goal: smooth, stable, and safe

The controller is **not** trying to imitate a human driver. It is trying to
be:

* **Smooth in equilibrium**: when ego is locked at the desired gap and
  speed, the command sits at zero with no micro-corrections.
* **String-stable in traffic**: disturbances must attenuate as they travel
  upstream. Any amplification creates the familiar stop-and-go waves known as
  phantom traffic.
* **Reactive when necessary**: genuine lead braking events must trigger an
  immediate response, but only to the degree required for safety.
* **Predictive beyond the immediate lead**: ACC can also observe vehicles
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

## 2. Control philosophy: equilibrium first, intervention second

The core is the **Improved IDM (IIDM)**, with a **Constant-Acceleration
Heuristic (CAH)** overlay blended in via the **ACC model** of Kesting,
Treiber & Helbing (2010).

This combination is chosen specifically because it supports three often
competing goals simultaneously:

1. **Stable equilibrium**: once the desired gap is reached, acceleration
   naturally converges to zero without oscillation.
2. **Minimal disturbance propagation**: small speed changes in upstream
   traffic are absorbed rather than amplified.
3. **Safety under rapid transients**: sudden lead braking still produces a
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

## 8. Control law: IIDM, CAH, and the ACC blend

The control law is built from three components, applied in order:

1. **Improved IDM (IIDM)**: primary continuous control law. Smooth in
   equilibrium, monotone in `s`, no free-term recovery overshoot.
2. **Constant-Acceleration Heuristic (CAH)**: kinematic safety floor.
   Provides correct authority when the lead is decelerating and ego's
   relative speed is small (the case classical IDM under-reacts to).
3. **ACC blend**: combines the two via a smooth `tanh` transition with
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

if z ≥ 1:                  # gap closer than desired: pure braking branch
    a_iidm = a_max · (1 − z²)

else:                      # gap at or beyond desired: bounded approach to a_free
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
speed to develop before responding: the failure mode classical IDM
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

CAH alone is too aggressive in steady state: it commits to the worst
case every tick. The ACC model (Kesting et al. 2010) blends IIDM with
CAH only when CAH demands more braking than IIDM:

```
if a_iidm ≥ a_cah:
    a_acc = a_iidm                               # IIDM passthrough: comfort regime

else:
    a_acc = (1 − c) · a_iidm
          + c · (a_cah  +  b · tanh((a_iidm − a_cah) / b))
```

with cool factor `c = 0.99` (Kesting et al recommend `c ∈ [0.95, 0.99]`)
and `b = b_comfort`. The `tanh` makes the blend C¹. In equilibrium the
first branch holds and CAH contributes nothing: no comfort cost.

---

## 9. Multi-vehicle anticipation

`ACCThread` already publishes the top-3 in-lane leads by score (see
`core/acc/README.md §6`). The controller treats them as a longitudinal
chain. The command is composed as

```
a_cmd = a_base  +  EMA(delta_anticipation, tau = ant_tau_s)
```

where `a_base` is the IIDM/CAH/ACC blend on the immediate lead alone
(§8, after the confidence blend of §9.5) and `delta_anticipation` is a
smoothly weighted adjustment from the leads beyond it. Everything that
can change chain membership (lane entry/exit, score flicker, id flips)
passes through continuous weights plus the delta EMA, so membership
changes never step the output.

### 9.1 Chain construction

Per tick, under `acc.data._lock`:

1. Copy `leads[:]` (shallow).
2. Filter to `dist_m > 0` and `effective_speed_ms` finite; carry each
   lead's tracker `score`.
3. Sort ascending by `dist_m`. Index 0 is the immediate lead, indices
   1+ are anticipated leads.
4. Cap the chain at `MA_MAX_LEADS = 3`.

Vehicles that are not strictly ahead of the previous chain member by at
least `ma_min_chain_gap_m` are dropped: they are either lateral noise
or ghost duplicates from the radar pipeline.

### 9.2 Coupling weights

Each anticipated lead `n ≥ 1` gets a weight built from three factors:

```
gap_time_n = (dist_n − dist_{n−1}) / max(v_{n−1}, ant_time_ref_floor_ms)
pair_n     = cos-ramp(gap_time_n): 1 at ≤ ant_gap_full_s (1.0 s),
                                    0 at ≥ ant_gap_zero_s (3.0 s)
conf_n     = smoothed score confidence of vehicle n (§9.5)

W_n = moving_gate · conf_0 · Π_{k=1..n} (pair_k · conf_k)
```

Pairwise time gaps propagate **multiplicatively**: one large gap
anywhere in the chain removes everything beyond it, while a tightly
packed platoon anticipates strongly even when it starts far from ego.
A vehicle more than 3 s ahead of its follower is invisible by
construction. `moving_gate` is the stationary-lead failsafe (§9.6).

### 9.3 Decel side: weighted per-lead demand + virtual lead

Two mechanisms, combined by minimum:

1. **Per-lead direct gap.** Each anticipated lead is run through the
   full §8 law at its direct gap; its *extra* braking demand over the
   immediate-lead command is scaled by its weight:

   ```
   dec_n = W_n · min(0, a_acc^{(n)} − a_base)
   ```

   At `W = 1` the vehicle binds fully, at `W = 0` it contributes
   nothing, smooth in between. This handles geometry: a slow vehicle
   at a short direct gap.

2. **Virtual lead.** The immediate lead's near-future state is
   predicted from the weighted upstream differentials:

   ```
   v_virt = v_0 + ant_kv · Σ W_n (v_n − v_{n−1})
   a_virt = a_0 + ant_ka · Σ W_n (a_n − a_{n−1})
   ```

   and the §8 law re-run on the immediate gap with `(v_virt, a_virt)`.
   Its negative delta joins the decel side. This reacts to upstream
   *braking events* long before the per-lead direct gap does (the ACC
   blend ignores CAH in the comfort regime, so a distant decelerating
   vehicle barely registers through mechanism 1).

```
delta_dec = min(min over n of dec_n, min(0, a_virt_cmd − a_base))
```

### 9.4 Accel side: bounded, gated lift

The positive part of the virtual-lead delta becomes the lift:

```
lift = clamp(a_virt_cmd − a_base, 0, ant_lift_max_ms2)      # 0.5 m/s²
```

gated by two safety conditions:

* **TTC gate** on the raw immediate lead: zero lift below
  `ant_lift_ttc_min_s` (4 s) TTC, full above `ant_lift_ttc_full_s`
  (6 s), cosine ramp between. No lift while genuinely closing.
* **Decel-priority fade**: lift fades to zero as `delta_dec` grows past
  `ant_lift_fade_ms2`, so the two sides never fight.

`delta_anticipation = delta_dec + lift`, then EMA (`ant_tau_s = 0.4 s`)
before being added to `a_base`. Ego eases off the brake, or picks up
throttle slightly earlier, when the pack ahead of the lead accelerates;
it can never gain more than 0.5 m/s² over the immediate-lead law.

### 9.5 Confidence and the immediate-lead blend

Tracker score maps to confidence via a ramp: 0 at `ant_score_min` (1),
1 at `ant_score_full` (5). Confidence is EMA-filtered per vehicle,
asymmetrically: fast up (`ant_conf_tau_up_s = 0.1 s`) so a genuine
cut-in gains authority almost immediately, slow down
(`ant_conf_tau_down_s = 0.8 s`) so a score flickering around the
in-path threshold ratchets toward its recent high instead of
squarewaving the command.

The immediate lead itself is confidence-blended: with `conf_0 < 1`,

```
a_base = conf_0 · a_lead0  +  (1 − conf_0) · a_next
```

where `a_next` is the law on chain[1] (or the no-lead ceiling). A
marginal vehicle drifting on the lane edge fades into authority instead
of snapping in and out. Safety overlays (§10) always run on the raw
chain[0] regardless of score.

**Primary ghost hold**: when chain[0] flips to a farther vehicle
because the previous primary left the published list, the old primary's
cached kinematics keep a decaying min-only grip on `a_base` for
`primary_ghost_hold_s` (0.8 s). A ghost can only hold brake briefly,
never raise the cap.

### 9.6 Stationary-lead failsafe

Anticipation (both directions) is disabled when the immediate lead is
not moving: `moving_gate` ramps from 0 at raw
`v_lead ≤ ant_lead_moving_min_ms` (0.75 m/s) to 1 at
`ant_lead_moving_full_ms` (1.5 m/s). Traffic beyond a stopped vehicle
predicts nothing about when it will move; ego must handle the stopped
lead on its own merits.

### 9.7 Safety overlays still use the immediate lead

The TTC, emergency, and standstill overlays in §10 are evaluated against
chain index 0 only, on raw data. Anticipation is for *smoothing*, not
for tripping emergency action. Every overlay, and the at-clamp state
(immediate-lead law at `max_decel`), zeroes the anticipation EMA so the
controller restarts from the pure immediate-lead law on exit. The
anticipation delta is additive on top of `a_base`, so a hazard the
immediate lead poses is never masked: `delta_dec` only tightens, and
lift is bounded, TTC-gated, and disabled at the decel clamp.

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
  _read_chain       : sort by dist, sanity filter, lock-scoped copy
        │
        ▼
  _smooth_inputs    : distance-adaptive EMA on (s, v_lead);
                       asymmetric EMA on a_lead (fast on brake, slow on relax)
        │
        ▼
  _compute_command  : emergency band, TTC floor, standstill hold,
                       immediate-lead IIDM/CAH/ACC blend + confidence
                       blend + ghost hold, anticipation delta (EMA)
        │
        ▼
  _jerk_limit       : |da/dt| ≤ 2.5 m/s³, bypassed on emergency
        │
        ▼
  _output_filter    : light EMA (τ ≈ 36 ms), bypassed on emergency
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

### 12.1 Distance and lead speed: symmetric distance-adaptive EMA

`dist_m` and `v_lead_ms` go through a distance-adaptive EMA: τ ramps
linearly from 120 ms at 20 m to 200 ms at 80 m. Close range stays snappy;
long range is filtered hard to kill TruckersMP packet jitter before it
reaches the IIDM core. Each chain member maintains its own EMA state
keyed on `vehicle.id` so a swap of the primary lead does not cause a
discontinuity on the new lead-of-lead.

### 12.2 Lead acceleration: asymmetric EMA

`a_lead_ms2` uses a **deadbanded asymmetric** EMA:

```
τ_brake = 80 ms          # fast: real negative-going step exceeds deadband
τ_relax = 350 ms         # slow: relax / coast / sub-deadband noise
deadband = 0.30 m/s²     # AI tick-to-tick wobble + MP packet jitter floor

Δ = new_a_lead − prev_a_lead_ema
τ = τ_relax           if |Δ| < deadband   # noise: heaviest filter
τ = τ_brake           if Δ ≤ −deadband    # real brake event
τ = τ_relax           otherwise           # real positive change
```

Rationale: AI traffic in ETS2/ATS wobbles `a_lead` by ±0.5–1 m/s²
tick-to-tick as a game-AI artefact, and TruckersMP injects intermittent
single-frame discontinuities. A pure asymmetric EMA chases every
negative-going step within one tick, so CAH momentarily demands brake
on phantom events. The deadband suppresses both noise sources entirely
by routing sub-floor deltas through the relax constant; only deltas
that exceed the noise floor in the negative direction trigger the fast
brake-side τ. Asymmetry still costs nothing in equilibrium and
preserves CAH reaction time on genuine lead braking.

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
pivot mid-body, AI pivot 18 % from front) is **not** smoothed: it is
constant per lead vehicle. `eff_dist = lead.dist_m − tail_m`.

---

## 13. Comfort overlays

| Layer | Time constant / cap | Notes |
|---|---|---|
| Jerk limiter | `J_MAX = 2.5 m/s³` | Below 2.94 m/s³ comfort threshold (Bellem 2022). Bypassed on emergency. |
| Output EMA  | `τ = 36 ms` | Legacy α=0.6 per 30 Hz tick, ported to framerate-independent τ. Bypassed on emergency. |

The jerk cap is the dominant smoothness shaper between the control law
(already smooth in `a` by §8) and the actuator. It is **not** bypassed
by sub-emergency CAH commands: only by the explicit safety overlays in
§10. Moderate CAH-driven braking events therefore stay jerk-limited and
feel firm rather than sharp.

---

## 14. String stability: quantitative

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
transient under-braking: precisely the regime where giving up some
smoothness for safety is correct.

User-facing gap level (`Settings.acc_gap_level`) maps to four headway
values, all ≥ 1.0 s:

| Level | Headway T | Effective behaviour |
|---|---|---|
| 1 | 1.0 s | Closest: at the string-stability boundary; reactive |
| 2 | 1.5 s | Default: comfortable, stable |
| 3 | 2.0 s | Relaxed |
| 4 | 2.5 s | Farthest: very stable, large equilibrium gap |

Headways below 1.0 s are deliberately not exposed.

---

## 15. Anti-oscillation summary

Cumulative effect of the layers, mapped to specific failure modes the
classical implementation suffered from:

| Failure mode | Mitigation |
|---|---|
| Free-term recovery overshoot (classical IDM) | IIDM piecewise form caps the upper branch at `a_free` |
| Soft response when lead matches ego decel | CAH overlay, blended via cool-factor model |
| Late, sharp brake on slowdowns ahead of lead | Weighted per-lead demand + virtual-lead prediction (§9.3) |
| Chain membership snapping (lane entry/exit) | Continuous coupling weights + anticipation delta EMA (§9.2) |
| Marginal in-lane vehicle flapping the command | Score-confidence blend + asymmetric conf EMA + ghost hold (§9.5) |
| Over-braking into a dissolving slowdown | Bounded, TTC-gated accel-side lift (§9.4) |
| Anticipating past a stopped lead | Stationary-lead failsafe (§9.6) |
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
ma_max_leads, ma_min_chain_gap_m,
ant_gap_full_s, ant_gap_zero_s, ant_time_ref_floor_ms,
ant_score_min, ant_score_full,
ant_conf_tau_up_s, ant_conf_tau_down_s, primary_ghost_hold_s,
ant_kv, ant_ka,
ant_lift_max_ms2, ant_lift_ttc_min_s, ant_lift_ttc_full_s,
ant_lift_fade_ms2, ant_tau_s,
ant_lead_moving_min_ms, ant_lead_moving_full_ms,
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
from `acc_thread.data.leads` under its lock: no signature change.

---

## 18. References

- Treiber, M., Hennecke, A., Helbing, D. (2000). *Congested traffic
  states in empirical observations and microscopic simulations.*
  Physical Review E 62, 1805.: Original IDM.
- Treiber, M., Hennecke, A., Helbing, D. (2006). *Delays, inaccuracies
  and anticipation in microscopic traffic models.* Physica A 360,
  71–88.: Multi-anticipative IDM.
- Kesting, A., Treiber, M., Helbing, D. (2010). *Enhanced Intelligent
  Driver Model to access the impact of driving strategies on traffic
  capacity.* Phil. Trans. R. Soc. A 368, 4585–4605.: IIDM, CAH, ACC
  blend, cool factor.
- Treiber, M., Kesting, A. (2013). *Traffic Flow Dynamics.* Springer,
  ch. 11 (IIDM) and ch. 15 (string stability).
- Treiber, M., Kesting, A. (2025). *Twenty-Five Years of the Intelligent
  Driver Model.* arXiv:2506.05909.: Truck defaults and review of
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
  a jam.* New J. Phys. 10, 033001.: Empirical phantom-jam baseline.
- Vahidi, A., Eskandarian, A. (2003). *Research advances in intelligent
  collision avoidance and adaptive cruise control.* IEEE TITS 4(3).

