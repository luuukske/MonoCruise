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
It is the only term in the law that reads `a_lead` at all.

**`a_cah` not depending on `Δv` does not mean the command responds
without `Δv`.** An earlier revision of this section claimed CAH removes
the failure mode classical IDM exhibits when "lead brakes hard and ego is
matching the deceleration". It does not, and cannot: the blend in §8.3 is
one-directional, so it discards `a_cah` in exactly that state. Measured
in §8.5.

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

**The branch test is strict for a reason.** A stopped, non-accelerating
lead puts `v_lead · (v − v_lead)` and `−2 · s · a_lead_eff` both at zero.
With a non-strict `≤` that selects the first branch, whose numerator and
denominator are then both zero, and the divide-by-zero guard returned
`a_lead_eff` = 0: "a stopped vehicle needs no braking". The blend below
then relaxed IIDM's demand onto that non-answer, so the command flatlined
near `a_cah − b` = −2 m/s² the whole way in and the TTC overlay (§10) was
left to catch the stop at −6.55 m/s², unfiltered. The two branches agree
analytically at the boundary wherever both are defined, so `<` changes
nothing except the degenerate case, where it yields the correct
glide-to-stop rate `−v² / 2s`. Pinned by
`tests/acc/test_gap_law_shaping.py`.

The legacy `K_FF · a_lead` feedforward term was removed when CAH landed,
on the stated premise that CAH supersedes it. **That premise holds only
where the blend actually selects CAH.** Where it does not (§8.5), the
removal left `a_lead` with no path to the command at all. Treat the
absence of a feedforward term as an open question, not a settled one.

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

Note the direction: the blend output is always **≥ `a_iidm`**, bounded
below by `a_cah − b`. CAH relaxes IIDM's braking toward the kinematic
requirement; it never adds authority of its own. Braking authority in
this controller comes from IIDM's `z²` term and the overlays in §10.
That is why a wrong `a_cah` (above) shows up as under-braking rather
than over-braking.

### 8.4 Gap-error shaping

The blend answers "how hard, given this gap". It does not say how much
that gap error should matter, and IIDM's answer to that was too stiff far
out and too slack up close: rubber-banding behind a distant lead, and
braking to restore a gap that the lead was already opening on its own.

**How smooth the loop is comes from the gap the driver asked for, not
from how far away the lead happens to be.** A wanted gap is a statement
about how relaxed the following should feel; the instantaneous gap is
just where you are on the way there. Keying the gain on the current gap
was tried first and made a close setting go slack whenever the lead was a
long way off, which is the opposite of what a close setting means.

```
s_want = s0 + v · T                                # this level's wanted gap
s_ref  = s0 + v · GAP_GAIN_REF_HEADWAY_S           # level-2 wanted gap

w_level = (s_ref / s_want) ^ 0.6                   # the setting, not the gap
w_close = max(1, (s_ref / s) ^ 0.6)                # firmness once genuinely close
w_open  = 1 − (1 − 0.25) · cos-ramp(v_lead − v, 0 → 1.5 m/s)
          faded back to 1 as s drops from s_want to 0.6 · s_want

a_out   = a_acc · clamp(w_level · w_close, 0.35, 1.5) · w_open   if a_acc < 0
        = a_acc · max(1, w_level)                                if a_acc ≥ 0
```

* `w_level` is **1.0 at level 2 by construction**, so the calibration
  everything else was tuned against comes out unchanged. At 80 km/h:
  1.24 / 1.00 / 0.85 / 0.70 for levels 1 to 4, and **flat with distance**.
  A farther setting gets a lazier loop at every gap; a closer one stays
  eager at every gap.
* `w_close` keys on `s_ref`, not `s_want`, so "close" is the same physical
  distance whatever the driver asked for.
* `w_open` is the "it will get there by itself" relief, and it only holds
  where that reasoning does. At or beyond the wanted gap a lead opening at
  1.5 m/s or more keeps a quarter of the comfort brake. Inside the wanted
  gap the relief fades back out: a deficit that deep does not recover fast
  enough to be left to the lead, so a close vehicle pulling away gets the
  close-range gain in full.
* The accel side is scaled **upward only**. A close setting closes on its
  target harder; a far setting keeps full pull rather than being throttled
  by its own smoothness. Headroom there is small in practice, since
  `a_free` already sits near `max_accel_ms2`.
* `gap_gain_min` (0.35) cannot bind while the level table stays inside
  `[0.7, 2.2]` s. It is a bound on a computed quantity, not a tunable.

**Equilibrium is untouched at every level.** The weights scale a command
that is already zero at `s = s*`, so only the gain around the operating
point changes, never where it sits.

**The relief stops at the kinematics.** Where CAH demands deceleration,
the shaped output is floored at `max(a_acc, a_cah)`: shaping may give
back comfort margin, never the glide-to-stop rate, and never more than
the unshaped law was asking for in the first place. This is what keeps
the §8.2 fix from being cancelled by a relaxed gain on the approach to a
stopped vehicle.

Measured, 90 km/h onto a stopped vehicle first seen at 250 m: peak
command −6.55 → −2.56 m/s² at level 2 (−2.26 at level 4), peak jerk 133 →
0.5 m/s³, and the same run with the vehicle first published at 110 m goes
from a collision to a clean stop 6.8 m short. Catching a slower lead from
150 m settles 11 % sooner at level 4. Equilibrium gaps and the response to
a lead braking at −2/−4/−6 m/s² are unchanged.

### 8.5 Measured: `a_lead` did not reach the command (fixed in §8.6)

**The law was blind to how hard the lead was braking.** This is a
property of the model, not a bug in the implementation, but it
contradicts what §8.2 used to claim, so it is recorded here with numbers.
§8.6 is the term that closed it; everything below is the pre-fix
measurement and the reasoning that picked the exit.

`a_lead` enters the law through exactly one door, `cah()`. §8.3 shuts
that door whenever `a_iidm ≥ a_cah`:

```
if a_iidm >= a_cah:
    return a_iidm          # a_lead is now absent from the command
```

`iidm()` takes no `a_lead` argument, and neither does `level_gain` nor
`comfort_gain`. So in that branch the only residue of the lead's
deceleration is the `min(a_soft, max(a_acc, a_cah))` floor in §8.4, worth
about 0.01 m/s².

At 80 km/h, level 3, 37.5 m effective gap, speeds matched, sweeping the
lead from coasting to an emergency stop:

| lead decel | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| cap (m/s²) | −0.06 | −0.07 | −0.07 | −0.07 | −0.07 |

`a_iidm` is −0.06 and `a_cah` is −3.61 at the right-hand end, so the
blend returns −0.06 and discards CAH. At a 77.5 m gap with the lead
braking at 8 m/s² the cap is still **+1.07**: still commanding
acceleration.

Closed loop, lead brakes from matched speed and holds, perfect actuator
(so these are floors, real actuation lag adds on top):

| case | to −0.5 | to −1.0 | to −2.0 | peak | min gap | TTC overlay |
|---|---|---|---|---|---|---|
| 80 km/h, 40 m, lead −2 | 0.63 s | 1.23 s | 5.93 s | −2.07 | 7.6 m | no |
| 80 km/h, 40 m, lead −4 | 0.37 s | 0.63 s | 1.13 s | −3.71 | 6.5 m | no |
| 80 km/h, 40 m, lead −6 | 0.30 s | 0.50 s | 0.90 s | −6.55 | 8.6 m | fires at 3.70 s |

The response does develop, because closing speed builds and IIDM reacts
to that. Two costs. The gap collapses from 40 m to 6.5 m against an `s0`
of 5 m, spending nearly the whole buffer. And at lead −6 the gap law
never gets there: the TTC floor of §10 fires and slams −6.55 unfiltered,
which is the same coast-then-slam shape §8.2's strict branch test fixed
for stopped leads, reached by a different route.

**Why this is defensible.** IDM-family models exclude `a_lead` on
purpose. Feeding lead acceleration straight through re-transmits a
disturbance upstream with gain instead of absorbing it, which is a
string-stability hazard, and string stability is a first-class
requirement here (§5). The code matches Kesting et al. 2010 exactly.

**"Near matched speed" understated it.** The freeze is a threshold in
`a_lead`, not a band in closing speed, and it bit everywhere. At 80 km/h,
40 m, closing 2 m/s the cap was `-1.292` for every lead decel from 2 to
10 m/s². Worst in the pulling-away region, where anticipation is
cheapest: a lead 4 m/s faster and braking at 8 m/s² drew `+1.234`, bit
identical to one coasting away, so ACC accelerated through the half
second before the closing speed arrived.

**Three exits, none free:**

1. Accept it and let the overlays plus AEB be the backstop. Cost: the
   hard-brake case is handled by an unfiltered slam, which §4 exists to
   avoid.
2. Let CAH bind downward in a gated regime (`min` rather than
   passthrough). Cost: CAH commits to the worst case every tick, so this
   is where over-braking and string-stability loss come from.
3. Reinstate a bounded `a_lead` feedforward, sized so it cannot dominate.
   Cost: the double-counting §15 warns about, and a hand-tuned scalar
   again.

The virtual lead in §9.3 already does something close to option 3, but
only with two or more chain members, so it is unavailable in exactly the
single-lead case above.

**Exit 3 was taken.** See §8.6.

Reproduce with `tools/acc_response_map.py` (see `tools/README.md`):

```
python tools/acc_response_map.py --scenario matched-speed --report text
python tools/acc_response_map.py --scenario brake-onset
```

### 8.6 Lead-braking feedforward

The signal is not raw `a_lead`. It is the share of CAH's demand that
exists **only because** the lead is braking:

```
a_ff = max(FF_SHARE · min(0, cah(a_lead) − cah(0)), −FF_MAX)
a_acc = acc_blend(...) + a_ff
```

with `FF_SHARE = 0.50` and `FF_MAX = b_comfort = 2.0 m/s²`.

Why this signal rather than `K_FF · a_lead`, which §8.2 removed:

* **It is exactly zero whenever `a_lead ≥ 0`.** The equilibrium of every
  gap level, the accel side, and the whole stopped-lead approach are
  untouched by construction, not by tuning. Pinned by
  `test_feedforward_is_silent_for_a_lead_that_is_not_braking`.
* **It self-gates on range.** `cah` already folds in gap, ego speed and
  lead speed, so a lead braking hard far enough away contributes almost
  nothing without a distance ramp to tune. A scalar on `a_lead` needs one
  and gets it wrong at the edges.
* **It is bounded.** `FF_MAX` equals the comfort brake, so a spurious
  `a_lead` from laggy TMP traffic costs at most one comfort brake and can
  never reach the decel clamp. This bound is the string-stability
  argument and the jitter argument at once. Pinned by
  `test_feedforward_is_bounded`.

**Applied outside the blend, not inside it.** The blend steps across its
own branch test (`a_iidm ≥ a_cah`), so a term added on only one side
would be discontinuous there. Adding it to `a_acc` keeps `lead_law`
C⁰ across the boundary. Pinned by
`test_lead_law_is_continuous_across_the_cah_branch_test`.

**No double count with §9.3.** Anticipation composes differences
(`a_virt_cmd − a_base`, `a_n − a_base`), and both terms come from
`lead_law`, so a term inside `lead_law` cancels and only its *excess* on
the virtual lead survives.

Measured, level 3, single lead, closing 0 m/s, the `cap span over the
whole decel axis` that §8.5 reported as ~0:

| panel | before | after |
|---|---|---|
| 50 km/h, 40 m | 0.00 | 0.97 |
| 50 km/h, 60 m | 0.00 | 0.69 |
| 80 km/h, 30 m | 0.33 | 2.33 |
| 80 km/h, 40 m | 0.01 | 1.82 |
| 80 km/h, 60 m | 0.00 | 1.40 |

Closed loop from matched speed at the level's own equilibrium, perfect
actuator:

| case | peak before | peak after | TTC overlay |
|---|---|---|---|
| 80 km/h, 40 m, lead −2 | −2.07 | −1.92 | no → no |
| 80 km/h, 40 m, lead −4 | −3.71 | −3.36 | no → no |
| 80 km/h, 40 m, lead −6 | −6.55 | −4.44 | 3.70 s → **no** |
| 80 km/h, 40 m, lead −8 | −6.55 | −5.27 | 3.47 s → **no** |

The overlay no longer fires anywhere a truck lead can physically brake,
so the coast-then-slam shape of §4 is gone from this path. Peak ego decel
stays under the lead's at every rate tested, which is the string
stability §5 asks for. Pinned by
`test_following_a_braking_lead_is_string_stable` and
`test_a_hard_braking_lead_no_longer_needs_the_ttc_overlay`.

**What this did not fix.** Minimum gap still collapses to 6.0-6.6 m from
40 m whatever the lead does, and sweeping `FF_SHARE` from 0.35 to 0.80
does not move it: the buffer is set by the IIDM equilibrium and the
authority limit, not by the `a_lead` path. Lead decel beyond ~10 m/s²
still reaches the overlay, correctly.

### 8.7 Transition smoothness at small lead braking

Three hard clamps sat within 1 m/s² of `a_lead = 0`, which is where a
lead in traffic spends most of its time. Each one is a step in
`d(cap)/d(a_lead)`, and a step in gain rectifies jitter around it into a
one-sided mean: the command responds on one half-cycle of the noise and
not the other.

Measured at 80 km/h, 40 m, closing 0.5 m/s, sweeping `a_lead` at 0.0025
m/s² resolution, worst adjacent change in gain:

| clamp | site | before | after |
|---|---|---|---|
| `min(0, demand)` | feedforward corner at `a_lead = 0` | step | cubic soft corner |
| `min(a_soft, floor)` | kinematic floor at decel 0.525 | step | log-sum-exp soft min |
| `max(a_acc, a_cah)` | floor argument swap at decel 0.723 | step | **still hard** |
| worst gain jump | | **0.798** | **0.171** |

Closed loop against a lead holding −0.3 m/s² with 0.35 m/s²
sample-and-hold telemetry noise at 8 Hz, averaged over 12 seeds:

| | RMS jerk | peak jerk |
|---|---|---|
| HEAD | 0.139 | 0.705 |
| current | 0.278 | 1.010 |

The first version of this table read 0.500 against 0.431 and claimed an
improvement. Its baseline was wrong: `BASELINE` in the probe omitted the
two feedforward shares, so the feedforward was live in both columns and
the comparison was the change against itself. §8.10 is the consequence.

**Why the third clamp stays hard.** A C1 maximum necessarily approaches
from above: `f = max(a,b) − g(|a−b|)` has a gradient mismatch at `a = b`
for any `g`, and only the `+g` family (log-sum-exp) is smooth. Above is
the direction that *relaxes* the kinematic floor, and softening it by
0.09 m/s² lifted the command 0.0033 above the glide-to-stop rate at 150
m, breaking `test_shaping_never_relaxes_past_the_kinematic_requirement`.
Smoothness is not worth trading that assertion for. The soft min is
biased downward for the same reason, so it can only ever add braking.

**The floor band is not monotone in smoothness.** 0.09 m/s² is a
measured optimum, re-derived against a true HEAD baseline: worst gain
jump 0.394 / 0.205 / **0.171** / 0.155 / 0.146 and RMS jerk 0.332 /
0.296 / **0.278** / 0.285 / 0.462 at bands 0 / 0.045 / 0.09 / 0.14 /
0.20. Past 0.14 a wide band adds a standing downward bias the loop then
fights. Re-measure with the probe before moving it, and average over
seeds: one noise realisation ranks the variants wrongly.

Reproduce with `tools/acc_transition_probe.py`.

### 8.8 Why the accel side is flat near equilibrium

At the wanted gap with speeds matched, a lead accelerating gently commands
almost nothing: `a_lead` of 0.25 m/s² gives `+0.0013`, against `0.0000`
for a coasting lead. Measured gains at that state, level 3:

| | lead braking (−2..0) | lead accelerating (0..+2) |
|---|---|---|
| before §8.6 | +0.000 | +0.153 |
| after §8.6 | +0.371 | +0.153 |

**The feedforward did not cause this.** The accel side is unchanged to
four decimals; only the ratio moved. The flatness is the `tanh` blend
doing its job: with `a_iidm = 0` at
equilibrium the blend reduces to `a_cah + b·tanh(−a_cah/b)`, which is
`a_cah³/12` for small `a_cah`. Gain is zero to second order, which is
exactly the "CAH contributes nothing in equilibrium, no comfort cost"
property §8.3 is built on. Linearising it puts micro-corrections back
into steady following, against §1.

**Why the two sides are not worth the same.** `a_lead` integrates to
`Δv` and twice to gap error, both already high-gain inputs, so a
feedforward buys only phase lead. On the brake side that is worth a
bounded string-stability cost because lateness costs a collision. On the
accel side lateness costs a slightly wider gap that closes itself, while
earliness means surging at a vehicle that may brake again. Left alone,
`Δv` does get there: closed loop from equilibrium with the lead
accelerating at 1.0 m/s², the cap passed `+0.1` at 0.20 s and `+0.3` at
0.87 s with no accel-side term at all.

**This section originally concluded "do not add one".** That was
overruled on responsiveness grounds, and §8.9 is the result. The argument
above is not withdrawn: it is why §8.9 is a bounded gated nudge at 0.30
m/s² rather than a mirror of the 2.0 m/s² brake side, and it is the
reason to be suspicious of any future request to raise that bound.

`a_lead_eff = min(a_lead, a_max)` also clamps the accel side at 1.5 m/s².
That is correct, ego cannot follow an acceleration it cannot produce, and
it sits far from the `a_lead ≈ 0` band where telemetry jitter lives, so
it carries none of the §8.7 rectification risk.

### 8.9 Accel-side nudge

A bounded, gated phase lead for a lead that is pulling harder. It buys
feel, not safety, and is sized accordingly.

```
demand = soft-corner(min(a_lead, a_max))          # 0 for a_lead <= 0
gate   = 1 − cos-ramp((v − v_lead) / 2.0 m/s)     # 1 matched, 0 closing at 2
a_acc += min(0.25 · demand · gate, 0.30)
```

| | brake side (§8.6) | accel side |
|---|---|---|
| share | 0.50 | 0.50 |
| bound | 2.0 m/s² (`b_comfort`) | **0.75 m/s²** |
| gate | none, self-scales via `cah` | closing speed |

The accel share was raised from 0.25 to 0.50 once the gap cost was priced
honestly (below). Its bound had to go 0.30 to 0.75 with it: at share 0.50
a 0.30 bound clips from `a_lead` 0.6 upward, putting a 0.50 gain step in
the middle of the gentle-acceleration band. At 0.75 the bound cannot bind,
because `demand` is already capped at `a_max`, so ego's own authority is
the limiter. The `min(a_lead, a_max)` inside the nudge is soft-cornered
for the same reason; without that, doubling the share doubled the step at
`a_lead = 1.5` (0.399 to 0.899). With it the step is 0.407, unchanged.
What remains there is `cah`'s own hard `a_lead_eff = min(a_lead, a_max)`,
which predates all of this and only binds when a lead out-accelerates the
truck.

Three guards, each measured:

* **Zero unless the lead is accelerating.** `brake-onset` traces come back
  bit identical and the stopped-lead grid is untouched, so nothing on the
  safety-carrying side moved. Pinned by
  `test_accel_nudge_is_silent_unless_the_lead_is_accelerating`.
* **Gated on closing speed.** Contribution against a lead reporting +1.0
  m/s²: `0.250` at matched speed, `0.107` closing at 1 m/s, `0.000` at 2
  m/s. A phantom `a_lead` cannot add pull toward a lead ego is catching.
  Pinned by `test_accel_nudge_gates_off_while_closing`.
* **Soft-cornered and capped at `a_max`.** The worst gain jump of §8.7
  stays at 0.1435, so this does not reintroduce a step at `a_lead = 0`,
  and a lead accelerating at 20 m/s² reads the same as one at 1.5.

Measured at the wanted gap, matched speed, level 3:

| lead accel | 0.00 | 0.25 | 0.50 | 1.00 | 1.50 |
|---|---|---|---|---|---|
| before | 0.0000 | 0.0013 | 0.0101 | 0.0750 | 0.2274 |
| after | 0.0000 | 0.0620 | 0.1351 | 0.3250 | 0.5274 |

Equilibrium is untouched, exactly, because the nudge is zero at
`a_lead = 0`.

**The cost of more push was mispriced first time.** The original figure,
a 15.65 m standing gap loss at share 1.0, came from a test that held the
lead's *speed* constant while reporting a sustained `+1.0 m/s²`. Those
contradict each other: a real lead accelerating at 1.0 for a minute
reaches 216 km/h, and if the kinematics estimator ever reports sustained
acceleration against a flat speed that is a bug upstream of ACC, not a
case to tune against. Re-measured with `v_lead` integrated from the same
`a_lead` the lead reports, 8 seeds, σ 0.35 of noise:

| nudge share | resume to +0.3 | settled gap (wanted 38.33) | RMS jerk | peak jerk | cap at 6 m, 2 m/s |
|---|---|---|---|---|---|
| 0 | 0.97 s | 41.16 | 0.216 | 1.428 | −0.096 |
| 0.25 | 0.60 s | 41.12 | 0.244 | 1.425 | +0.182 |
| **0.50** | **0.40 s** | **41.07** | **0.281** | **1.512** | **+0.451** |
| 1.00 | 0.27 s | 40.94 | 0.382 | 1.732 | +0.987 |

Settled gap moves 0.2 m across a 4x share change, and jerk moves little.
Neither is a reason to hold the share down.

**The cost that is real is the close-range column.** The nudge takes no
distance argument, so its contribution is the same at 3 m as at 1000 m,
and it can invert a small braking command into a small pull: at 6 m and
2 m/s behind a lead reporting `+1.0`, `−0.096` becomes `+0.451`. The
bound is currently doing the job a range gate should do. Add the range
term before raising the share again.

Closed loop, lead accelerating at 1.0 m/s² from equilibrium: the cap
passes `+0.1` at 0.00 s (was 0.20), `+0.3` at 0.00 s (was 0.87) and `+0.5`
at 1.07 s (was 1.70). Peak gap 49.1 m rather than 52.4 m, so ego holds 3.3
m closer through the manoeuvre.

### 8.10 The feedforwards read their own `a_lead` estimate

`a_lead` is the noisiest signal in the pipeline, a second derivative of
telemetry. §8.5's defect and the old noise immunity were **the same
property**: the blend discarding `a_cah` was also what kept that noise
out of the command. §8.6 removed the defect and removed the filter with
it, and a static gain on `a_lead` then transmitted its noise to the
pedal.

Measured, lead holding a constant speed with 8 Hz sample-and-hold noise
on `a_lead`, 12 seeds, RMS commanded jerk:

| σ(`a_lead`) | HEAD | §8.6 as first written | with this section |
|---|---|---|---|
| 0.35 | 0.016 | 0.775 | 0.243 |
| 0.60 | 0.027 | 1.158 | 0.388 |

Peak jerk at σ 0.60 went 0.285 (HEAD) to **2.451**, against a
`J_MAX_MS3` of 2.5: the command was being rate-clipped while nothing was
happening.

**The fix is a filter, not a smaller gain.** Cutting `share` or widening
the soft corner would pay for the jerk with steady-state response, which
is the entire point of §8.6. A low-pass pays with phase lag instead, and
lag is the cheap currency here: the deadline before the TTC overlay fires
is seconds. It is also **invisible to the response map**, so every §8.5
and §8.6 steady-state number is preserved by construction rather than by
tuning. Verified: the decel-axis spans are identical at every tau tried.

`TAU_ALEAD_FF_S = 0.50 s`, symmetric, maintained in `_smooth_chain`
alongside the existing estimate and carried on `_LeadSnapshot`. CAH keeps
the fast asymmetric one of §12.2, because it sizes a stopping requirement
where lateness is dangerous. Only the additive bonus term is slowed, so
the worst case of over-filtering is that §8.6 degrades toward HEAD, which
is what shipped.

Filtered **before** the one-sided nonlinearity, not after. The
feedforward is zero for `a_lead ≥ 0`, so it rectifies; filtering after
the corner cuts variance only, filtering before cuts the standing bias
too.

Trade curve, lag measured from the moment a coasting lead starts braking
(the earlier trace published a lead already braking at t=0, and the EMA
seeds to its first input, so it showed no lag at all):

| tau | RMS jerk σ0.35 | peak σ0.60 | lag to −1.0, lead −2 / −4 / −6 |
|---|---|---|---|
| HEAD | 0.016 | 0.285 | 1.27 / 0.63 / 0.50 s, slams at −6 |
| 0.08 | 0.775 | 2.451 | 0.50 / 0.43 / 0.43 s |
| 0.35 | 0.299 | 2.027 | 0.63 / 0.43 / 0.43 s |
| **0.50** | **0.243** | **1.803** | **0.70 / 0.43 / 0.43 s** |
| 0.70 | 0.189 | 1.591 | 0.77 / 0.43 / 0.43 s |

The lag cost lands almost entirely on the gentlest brake. At −4 and −6 it
is 0.43 s at every tau, and no tau reintroduces the slam.

The virtual lead of §9.3 builds its own feedforward input the same way. If
it read the unfiltered estimate while `a_base` read the filtered one, the
`a_virt_cmd − a_base` difference would carry back exactly the noise this
section removes.

### 8.11 Closing-speed relief

Gap-error braking is halved while ego is not actually catching the lead,
and returns to normal by 2 km/h of closing:

```
relief = 0.50 + 0.50 · cos-ramp((v − v_lead) / (2 km/h))
a_soft = a_acc · comfort_gain · relief          # decel side only
```

Driver-reported: the truck braked harder than expected behind a lead that
was level or pulling away. §8.4's `w_open` was supposed to cover this and
does not, because it keys on *opening* speed and fades to nothing inside
0.6 · `s_want`, so it gives no relief at all at matched speed.

**Above 2 km/h of closing the command is bit identical.** That is the
requested boundary and it is worth keeping: everything the earlier
sections measured about approach and lead braking lives above it.

**This is a comfort term and the §8.4 floor still binds under it.** At a
20 m gap, matched speed, lead braking at 6 m/s², the command relaxes from
−6.47 to −4.038, and the floor there is −4.037. The relief spent exactly
the margin the previous build was adding *past* the floor and stopped on
it.

Be precise about what that floor is: it is `max(a_acc, a_cah)`, not
`a_cah`. Beyond roughly 30 m the blend has already relaxed CAH so `a_acc`
is the higher of the two and the floor follows it, which is why the
command legitimately sits above the raw CAH rate at 40 m (−1.399 against
−3.042). That is the ACC blend of §8.3 working as designed and predates
this section. The relief cannot pass that floor; it was never bounded by
`a_cah` alone.

Measured cost, closed loop with a perfect actuator:

| case | min gap, relief off | relief on |
|---|---|---|
| 80 km/h, 20 m, lead −6 | 6.2 | 5.9 |
| 80 km/h, 25 m, lead −6 | 6.3 | 6.1 |
| 50 km/h, 15 m, lead −6 | 5.4 | 5.2 |
| 80 km/h, 30 m, lead −8 | 8.6 | 8.6 |

0.2 to 0.3 m of minimum gap in hard-braking cases at short range, all in
the same direction. The `brake-onset` suite is unchanged (peaks −1.92 /
−3.35 / −4.44, no overlay trip), and the overlay trips that do appear at
20 m and below are **not new**: with the relief off those cases trip at
0.00 s, so the relief delays the trip rather than causing it.

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
| Standstill hold | `v_ego < 0.4` AND `v_lead < 0.4` AND `eff_dist ≤ s0 + 2.0` | `0.0 m/s²` | none |
| At-clamp hard | `a_chain ≤ MAX_DECEL + 1e-6` | as-is | jerk + EMA |

The standstill hold is a real-vehicle extension to the textbook IIDM.
Inside the window IIDM still reads the gap as larger than `s0` and asks
for a small positive command; pinning the cap at zero stops the truck
creeping up on a stopped lead. `standstill_hold_decel_ms2` (−0.6) is
**not** wired to anything: the overlay returns 0.0. Actually holding the
truck still is `sending_thread`'s hold and creep-cancel job.

---

## 11. Pipeline

```
  ACCThread.data.leads[0..2]
        │
        ▼
  _read_acc_snapshot: sort by dist, sanity filter, lock-scoped copy
        │
        ▼
  _smooth_chain     : distance-adaptive EMA on (s, v_lead);
                       asymmetric EMA on a_lead (fast on brake, slow on relax)
        │
        ▼
  _compute_command  : emergency band, TTC floor, standstill hold,
                       immediate-lead IIDM/CAH/ACC blend + gap shaping
                       + confidence blend + ghost hold, anticipation delta (EMA)
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
ramp = 0.45 m/s²         # innovation span the fast τ fades in over

urgency = cos-ramp(−Δ, deadband → deadband + ramp)      # 0 to 1, C1
τ = τ_relax + (τ_brake − τ_relax) · urgency
```

**The threshold used to be a switch, and that was a chatter source.** A
lead whose reported `a_lead` sits on the deadband edge re-picked between
a 80 ms and a 350 ms filter every tick, a 4.4x bandwidth change with no
transition. Worse, the switch is one-sided by design, so symmetric
telemetry noise was rectified: the filter jumped down fast on every
negative innovation past the floor and crawled back at 350 ms, dragging
the filtered `a_lead` systematically negative into a phantom brake. The
ramp keeps the intent (fast on a genuine brake, slow on noise) and
removes the edge. `alead_tau_s` in `idm_cah.py`, `A_LEAD_TAU_RAMP_MS2 =
0` restores the old switch for A/B runs.

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
values in `T_HEADWAY_BY_LEVEL_S`, which is indexed by level with index 0
holding the fallback used when the level is out of range:

| Level | Headway T | Effective behaviour |
|---|---|---|
| 1 | 0.7 s | Closest: below the string-stability bound; reactive |
| 2 | 1.1 s | Default: comfortable |
| 3 | 1.5 s | Relaxed |
| 4 | 2.2 s | Farthest: very stable, large equilibrium gap |

Only levels 3 and 4 clear the `T ≥ 1.0 s` bound the linearisation above
assumes, so 1 and 2 are not string-stable in the textbook sense: a
disturbance ahead can amplify down a queue. The safety overlays in §10
and AEB are what bound that, not the headway.

**Open:** why the set runs below 1.0 s is not recorded anywhere, and no
measurement in this repo justifies the current values. An earlier
revision of this table claimed 1.0 / 1.5 / 2.0 / 2.5 s and stated that
headways below 1.0 s were deliberately not exposed; the constants have
not matched that for some time. The numbers above are the shipped ones,
and the settings panel now prints each level's headway next to it, so
they are user-visible. Treat the values as unvalidated rather than
tuned, and re-derive them before citing this table as a rationale.

---

## 15. Anti-oscillation summary

Cumulative effect of the layers, mapped to specific failure modes the
classical implementation suffered from:

| Failure mode | Mitigation |
|---|---|
| Free-term recovery overshoot (classical IDM) | IIDM piecewise form caps the upper branch at `a_free` |
| Soft response when lead matches ego decel | **Still open.** CAH is discarded by the blend in exactly this state (§8.5) |
| Late, sharp brake on slowdowns ahead of lead | Weighted per-lead demand + virtual-lead prediction (§9.3) |
| Chain membership snapping (lane entry/exit) | Continuous coupling weights + anticipation delta EMA (§9.2) |
| Marginal in-lane vehicle flapping the command | Score-confidence blend + asymmetric conf EMA + ghost hold (§9.5) |
| Over-braking into a dissolving slowdown | Bounded, TTC-gated accel-side lift (§9.4) |
| Anticipating past a stopped lead | Stationary-lead failsafe (§9.6) |
| Jitter-driven micro-oscillation in equilibrium | symmetric EMA on `s, v_lead`; jerk limiter |
| Brake-event lag from filtering | asymmetric EMA on `a_lead`; TTC + emergency on raw |
| Double-counting of lead accel | legacy `K_FF · a_lead` term removed. Premise re-opened by §8.5 |
| Phantom traffic jams from accel amplification | `T ≥ 1.0 s` + IIDM string stability + anticipation |
| Step changes at handover (e.g. lead → no-lead) | continuous IIDM domain across all gap regimes |
| Coast to a stopped vehicle, then a full-authority slam | strict CAH branch test (§8.2) |
| Rubber-banding at a farther gap setting | `w_level` (§8.4) |
| Braking to close a gap the lead is already opening | `w_open` (§8.4) |
| A close setting going slack behind a far-off lead | gain keys on the wanted gap, not the current one (§8.4) |

---

## 16. Tuning hooks

All defaults live in module-level constants and are replicated on
`ACConfig` so tests can override without monkey-patching:

```
a_max_ms2, b_comfort_ms2, delta, v0_ms,
s0_m, t_headway_s,
cool_factor_c,
gap_gain_ref_headway_s, gap_gain_exponent, gap_gain_min, gap_gain_max,
opening_gain_min, opening_gain_full_ms, opening_relief_fade_frac,
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

