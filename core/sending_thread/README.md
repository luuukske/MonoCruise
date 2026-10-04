# Sending thread and pedal I/O

> Maps commanded accel to game pedals via shared memory (`SCSController`).
> Longitudinal arbitration in `cruise_control_thread`; agent rules in root `AGENTS.md`.

## Sending thread (`thread.py`)

Opens SCS controls, runs `AccelToPedals`, hold FSM, pedal capacity learning, optional
visualization bar, hazard toggling, AEB decel assist, auto-neutral, creep compensation,
and commander merge (CC/ACC/limiter/AEB/user). Publishes `aforward` / `abackward` on
`SendingThreadData` as the **logical** pedals (what the mapper and viz use). The value
written to `SCSController.abackward` is remapped by live `g_brake_intensity` as the
last step before send, except AEB and `em_stop` which keep the full brake axis.
See **Brake intensity** below.

## Hazards

`hazards_variable` gates every automatic press. Pedal loss turns hazards on
on the rising edge and leaves them on. So does a hard brake on the pedal
actually sent to the game, at the driver's slam floor (0.8), above 10 km/h
like the driver slam itself: a stomp at rest is a hold, not an emergency,
and holding it does not arm the lamps once the truck rolls. That check sits
after the user-override merge, so an ACC brake the driver has overridden
never lights the lamps. Braking-intensity compensation is applied after this
check and does not change the floor.

`autodisable_hazards` turns them off from those same sent pedals: gas at the
autodisable floor (0.60), brake clear, above 12 km/h. The lamps do not go
off just because the hard brake ended; they wait for accelerator.

## AccelToPedals (`accel_to_pedals.py`)

Single mapper instance for the process. Converts wanted m/s² to gas/brake with smoothing,
leaky integral, road-load feed-forward, adaptive full-pedal accel/brake estimates,
gearshift integrator freeze, and tuning CSV rows when high-demand estimates underperform.

- Road load is gravity along grade plus rolling resistance and aero. Pitch comes from
  telemetry `rotationY` as a normalized full-circle float, converted by
  `_road_grade_from_norm`. Rolling coefficient is `mapper_rolling_resistance`.
  Grade uses a slow EMA for pitch noise; a large grade error blends toward a
  short tau with `1-exp(-(err/ref)^2)` so a real hill still tracks. Rolling and
  aero stay slower. Slope compensation stays in this mapper, not the cruise PID.
  A positive accel bid is never mapped to brake by gravity (downhill launch).
  Positive `slow_integral` leftover from a climb bleeds off when already
  accelerating with no accel bid, so it cannot keep FF on gas over a crest.
- Adaptive accel/brake estimates learn from load-compensated accel (`raw + road_load`),
  so a slope cannot bias the learned full-pedal capability.
- Also holds the shared telemetry mass-estimate helper that `telemetry_thread` uses.

### Launch governor (`launch_governor.py`)

At a launch the game's automatic clutch slips from rest to somewhere between 5 and 20 km/h,
and `game_clutch` reads pressed the whole time. `_gearshift_factor` treats that as a
gearshift: it freezes the measured accel at its value when the clutch opened and stops the
fast trim integrating. The pedal during a launch was therefore pure feedforward,
`(wanted + road_load) / capacity`, and that feedforward is wrong at a launch:

- Fitted over 62 logged slip launches (September 2026, mostly 17 t): accel = 0.25 +
  2.75 x throttle (0.3 s dead time) - 0.68 x g sin(grade). The learned per-gear capacity
  the feedforward divides by read anywhere from 1.2 to 4.0 on those same launches, so it
  does not predict the launch.
- Road load includes `mapper_rolling_resistance` x g (0.69 m/s2), a coast fit a slipping
  clutch never sees, and the idle creep the game adds is only counted in gear 1.
- 2026-09-28, 5% grade, capacity 1.31: ACC asked for +0.3 to +0.4 m/s2, gas went to 1.0,
  the truck did about 2 m/s2, ACC braked, and it repeated every 4.5 s. That is the
  "lurching" launch behind slow traffic. Across recent launches the truck overshot the bid
  1.2 to 1.8x at a 1.5 bid and 3 to 6x at small bids.

The governor caps the mapper's gas while that lasts. It arms when the clutch is pressed
below 1.5 m/s in a forward gear with CC/ACC commanding, never in limiter mode or under
AEB. It is a PI servo in pedal units on the live accel (`raw_smooth_live`), with gains
divided by `gain_scale` because launch gain falls with mass: kp 0.30, ki 0.40 while the
truck is short of the bid and 0.80 while it is ahead. On the rising edge of a launch bid
the cap starts no lower than the pedal that holds the grade, so a hill start never waits
for the integrator. Once the clutch has closed and the gearshift ramp has run out (or
above 7 m/s) the cap opens at 1.0/s, and the governor goes idle on the first tick it no
longer binds, so handing back is bumpless. In `accel_to_pedals_debug.csv` it shows as
`gas_cmd` below `effort`.

Closed loop, the real mapper against the fitted plant (a reduced copy is in
`tests/test_mapper_launch_governor.py`). Peak accel while the clutch slips, and t90, the
time to 90% of the bid after it starts ramping:

| case | peak before | peak after | t90 before / after |
|---|---|---|---|
| bid 0.35, 5% grade, capacity 1.31 (the 09-28 rig) | 2.49 (7.1x) | 1.21 (3.5x) | n/a |
| bid 0.35, flat, capacity 3.0 | 0.93 (2.7x) | 0.40 (1.1x) | n/a |
| bid 0.8, flat | 1.46 (1.8x) | 0.85 (1.1x) | 0.56 / 0.84 s |
| bid 1.5, flat | 2.29 (1.5x) | 1.54 (1.0x) | 0.82 / 1.08 s |
| bid 0.8, 8% grade | 2.46 (3.1x) | 1.48 (1.8x) | 0.16 / 0.56 s |
| 10 t, bid 1.5, flat | 3.93 (2.6x) | 1.50 (1.0x) | 0.64 / 1.12 s |
| 40 t, bid 1.0, 3% grade | 1.05 (1.0x) | 1.03 (1.0x) | 0.80 / 0.90 s |

The truck reaches the bid 0.1 to 0.5 s later, because before it got there by overshooting.
Hills still overshoot: the holding pedal comes from the same learned capacity, which
under-reads at a launch, and it stays because a hill start must not stall.

Rules:

- Ceiling only. It never raises gas above what the mapper computed;
  `test_governor_only_lowers_gas` pins it.
- Never in limiter mode, where the mapper's gas caps the driver's own pedal, or under AEB.
- It starts from the gas actually sent, and while it binds the rate limiter's memory is held
  at the cap. In neutral the mapper keeps its gas trajectory running so gas returns at once
  when a gear engages. With auto neutral that trajectory had already reached 1.0 by the
  time drive engaged, and the first tick in gear stepped straight there.
- Do not fix this by un-freezing the measured accel during the slip instead. The freeze is
  there for gearshifts, and the fast trim's P and D cannot undo a feedforward at twice the
  truth before the loop delay lets them see it.
- Retune against the plant fit, not by feel. The loop delay is roughly half a second (plant
  dead time plus the measurement's tracking differentiator), and `KI_RISE` sets the phase
  margin against it: about 1 rad/s crossover and 55 degrees at the fitted gain.

## Pedal capacity (`pedal_capacity.py`)

Always-on brake decel and gas gain learning (replaces legacy brake efficiency tracker).

**Brake**: `update_brake` every tick; accept samples only when pedal and decel settle.
Candidate inverts the fitted brake curve; pedal³ weighting; a sample moves the estimate
at the same rate whether the stop was harder or softer than the current figure. Road load
canceled before sampling. Fast EMA during
deep settled AEB braking. Candidates reject above `_BRAKE_CANDIDATE_MAX_FRACTION` (1.35) of
the load baseline. The pedal is the value written to the game, read back into tune units
with `effective_brake_pedal` (the inverse of the cruise remap) so a slider change is not
learned as truck weakness. See **Brake intensity**.

**Gas**: `update_accel` every tick that the pedal is above zero, learning the zero-pedal
offset, the shape-function anchor and the per-gear ratio. Same acceptance discipline as the
brake side, and for the same reason.

### The pedal model is affine, not a line through the origin

`candidate = accel / pedal` is only a capability estimate if zero pedal means zero accel. It
does not. At gas 0 in gear the sim still drives the truck, measured at **+0.50 m/s2, flat**
across 10-55 t and 5-35 m/s over 4021 `sample_kind="coast"` rows, and flat over the length of
a coast episode (+0.540 at 0 s, +0.485 at 5 s), so it is physics and not differentiator lag.
An independent affine fit of the accepted samples per gear and mass bucket puts the intercept
at **0.51-0.76 in every bucket**, agreeing with the coast number.

Dividing an affine signal by the pedal makes the answer blow up as the pedal shrinks. Replayed
over 1.3 M rows of `accel_to_pedals_debug.csv`, the estimate walked from **9.36 m/s2 at pedal
0.05-0.10 down to 1.17 at full pedal**: an 8x spread that is a function of pedal position and
nothing else. It is one-directional, so it never averages out.

The gates did not catch it because they were never looking for it. Of the samples that passed
every gate, **75.9% were taken at constant speed and 13.3% while the truck was slowing down**,
carrying 39.8% and 10.0% of the pedal-cubed weight. `_WEIGHT_POWER` suppresses them but there
are too many for that to be enough.

Three changes close it:

- **Band split.** Below `_ACCEL_OFFSET_MAX_PEDAL` (0.25) the offset dominates, so that band
  fits the intercept and never touches the slope. Above `_ACCEL_SLOPE_MIN_PEDAL` (0.35) the
  slope dominates. Between them neither term is separable, so neither is learned. The offset
  band is additionally skipped whenever the modelled slope contribution already exceeds the
  offset, which is what keeps low gears (slope 5+ m/s2) out of the intercept fit.
- **Authority gate.** A slope sample must still carry `_ACCEL_MIN_AUTHORITY_MS2` (0.10) and
  `_ACCEL_MIN_AUTHORITY_FRAC` (0.25) of the measurement once the offset is removed. This is
  what keeps a constant-speed reading out of an acceleration-capability estimate.
- **Candidate rejection restored.** The per-candidate bound existed as
  `_clamp(candidate, _ACCEL_GAIN_MIN_MS2, _ACCEL_GAIN_MAX_MS2)` until commit `e151848`
  replaced the per-gear dict with the shape function; the bounds were renamed to
  `_ACCEL_ANCHOR_*` and re-applied to the anchor only. The sample-level guard is back, tested
  in the shape function's own units against the implied anchor.

`_ACCEL_ANCHOR_MIN/MAX_MS2` are 0.2 and 7.5 rather than 0.5 and 8.0 because the anchor is the
pedal **slope** now; full-pedal capability is `offset + slope`, so the old bounds move down by
the offset. `accel_gain_for_gear` returns that sum, scaling the offset by `weight_factor` so
the mapper's own division returns it whole: the offset is mass-independent, the slope is not.

### The anchor and ratio walked their degenerate direction

`log_ratio += lr_ratio * x * residual` carries the lever arm `x = _ANCHOR_GEAR - gear`, which
reaches -8 in top gear, while the anchor update does not. One top-gear sample therefore moved
the ratio 8x further than the anchor. With the gear distribution as lopsided as it is (gear 14
is 100 k of ~400 k accepted samples at 17 t against 842 at gear 1) the pair is only identified
along one direction, and the unnormalized step slid it along the other. Replayed over the full
log the ratio hit both rails repeatedly. Normalizing that step by `1 + x^2` fixes the lever arm
without touching the anchor rate the gear-shift replay tuned; the ratio then stays inside
1.05-1.33 over the same replay.

Replaying the fixed learner over the same 1.3 M rows, the offset converges to **0.509**,
independently reproducing the 0.504 measured from coast. Top gear at 17 t reads **1.029 m/s2**
against a measured **1.02**; before the fix the persisted state gave **0.203** against the same
truth, which is why a bid over 0.2 m/s2 floored the throttle.

**Still open, deliberately not fixed here.** `weight_factor` is a straight line in tonnes with
an effective mass exponent of ~0.46 where the data reads 0.65-1.0, so the residual error grows
with mass: top gear reads 1.6x high at 40 t while it is exact at 17 t. It is load-bearing for
the mapper's round trip on every rig, so re-fitting it is its own change with its own probe,
the way `baseline_brake_ms2` was done.

### Gear-shift poisoning (the post-shift pedal step)

For a long time CC/ACC stepped the gas pedal up about 5-10% for a few seconds after every
gear change, at unchanged demand. Repeated fixes to the mapper's clutch freeze, blend ramp
and capacity glide never removed it, because the fault was never in `accel_to_pedals.py`.

Measured over 36 h of debug logs, 1602 clean steady-demand upshifts:

- The driveline takes about **1.5 s** to restore torque. True accel goes to about
  **-0.7 m/s²** during the shift and stays below its pre-shift level for seconds after.
- The old gear-dwell gate was **0.30 s**, so it opened while accel was still deeply
  depressed. **60%** of ticks in the 0.3-2.0 s recovery window passed every remaining gate,
  because the gas pedal ramps slowly enough during recovery to read as settled.
- Those samples carry pedal³ weight of **0.514 against 0.329** in steady cruise (the pedal is
  high exactly then), and `_UNDERPERFORM_MULT` doubles the rate again because a recovery
  sample is by definition below expectation.
- So the tracker attributed the torque interruption to the truck being weak, dropped the
  anchor, and `gas = combined / max_a_use` rose. That is the pedal step.

Two gates close it, and both are needed. Replaying the real learner over the same 36 h:
today 3.433, dwell alone 3.910, settle gate alone 4.111, **both 4.778**, against a reference
of 4.708 built from pristine samples only. Both together land within 1.5% of that reference.

- `_GEAR_DWELL_S` **1.50 s**, sized on the measured recovery rather than the differentiator
  glitch it originally guarded. `_LOW_GEAR_DWELL_S` stays shorter (0.30 s) so brief low-gear
  holds during a launch can still learn; the settle gate covers them.
- An **accel-settled gate** mirroring the decel-settled gate: the accel signal itself has to
  have stopped moving, not just the pedal. A call-stream gap restarts both windows, which is
  what handles a shift through neutral (gas is cut to zero there, so `update_accel` stops
  being called and the window must not straddle the hole).

The candidate is a window mean on both sides for the same reason the brake side is.

Consequence to expect on first drive after this: the persisted anchor is currently poisoned
low, so it re-learns upward over tens of minutes rather than jumping. Steady-state gas will
fall as it does (the replay predicts roughly 28%), and `fast_i`, which sat permanently at
-0.2 to -0.5 absorbing the over-generous feedforward, should relax toward zero.

**What is learned is `brake_scale`, a dimensionless correction on the baseline, not an
absolute m/s².** Capacity is a property of the rig, and the rig changes the moment you back
under a trailer: braked axles roughly double while the EMA carries the old number. At the
shipped rate an absolute scalar needs hours of *accepted* samples to catch up, and accepted
samples are ~0.07% of ticks, so in practice it never does. Measured live: 44 recorded
engagements read `max_brake_ms2` 8.90 on an 18-wheel 24 t double whose baseline is 13.89,
10.53 on a 12-wheel single against 12.58, and 9.54 on a bobtail against 10.22 — the error
tracks rig size, because it is the rig change the estimator cannot follow. `update_brake`
now re-resolves `scale × baseline_brake_ms2(wheels, mass)` every tick, so a trailer hookup
lands in the same tick and only the correction is carried across.

The bounds are asymmetric on purpose:

- `_BRAKE_SCALE_MIN` 0.35 — a genuinely weak rig (wet grip, worn brakes, fade) must be
  believable all the way down. AEB planning against capability the truck no longer has is
  what hits things.
- `_BRAKE_SCALE_MAX` **1.00** — the learner may correct the model down but never up.
  Over-reading raises the entry bar, so AEB engages at a gap sized for a stop it cannot
  make; the stop simulation collides from about 1.10× truth. This replaces the old
  `_BRAKE_CEILING_NOMINAL_MULT` workaround, which allowed ~19.5 m/s².

**Why the ceiling is one-sided.** The room for an upward correction is already spent by the
model. Probed 2026-08-12 on one double with cargo as the only variable:

| rig | probe (peak-A) | model | model/probe |
|-----|---------------:|------:|------------:|
| double, empty, 24.3 t | 15.10 | 13.91 | 0.92 |
| double, 16 t cargo, 40.5 t | 11.37 | 11.87 | **1.04** |

The over-prediction on the loaded rig is not a fitting accident: refitting the power law
over all six measured points, or any subset, still runs 3.8-4.5% high there, and the two
controlled cargo-only pairs disagree on the mass exponent (0.31 from 24→54 t, 0.555 from
24.3→40.5 t). On top of a model already 4% high, a ceiling of 1.02 collides at 80-120 km/h
under p90 brake lag; 1.05 collides at every speed.

The mechanism a one-sided ceiling blocks is a **carry-over, the same class as the bug that
motivated this rewrite**: `brake_scale` is global but the model's error is not. Learning
1.05 on the empty double, where the model is 8% low and samples honestly say 1.08, then
hooking cargo applies it to a model already 4% high. Raising the ceiling needs the mass
exponent resolved first, which needs a probe at a third cargo mass.

The residual risk is baseline error rather than estimator drift, and the baseline is fitted
to one user's rigs.

A brake sample moves the estimate at `_UNDERPERFORM_MULT` (2.0) whether it is above or
below the current figure. It used to rise at half that rate, which settled the estimate
4-10% under the samples and made the cruise pedal harder than the commanded stop. A
weaker stop still drops it at that same rate, down to `_BRAKE_SCALE_MIN`. The ceiling
stays 1.00: AEB plans with this number, and believing more brake than the model engages
late. Gas learning still applies the 2.0 only when the sample is below the estimate.

**The brake baseline is braked axles vs mass** (`baseline_brake_ms2`), fitted as
`70.8 * wheels^0.52 * mass^-0.31`. It must never divide by `weight_factor`: that is the
*acceleration* model, where more mass means less accel. Measured on full-pedal stops:

| rig | wheels | mass | measured | fitted | 1/mass model |
|-----|-------:|-----:|---------:|-------:|-------------:|
| bobtail | 6 | 10.5 t | 10.14 | 10.20 | 9.48 |
| single trailer | 12 | 17.0 t | 12.70 | 12.58 | 11.93 |
| double, empty | 18 | 24.0 t | 13.90 | 13.96 | 12.68 |
| double, ~29 t cargo | 18 | 54.3 t | **10.85** | 10.84 | **5.65** |

All four within 1%. Everything is in *A_true* units, `(decel - road_load) / frac(pedal)`.
Clip-derived rows are raw peak decel at pedal ~1.0 and must be converted before use, since
`frac(1.0)` is 0.912, not 1: skipping that runs the fit about 7% low.

**The mass exponent is 0.31, not 1.** The loaded double is the controlled test: same rig,
same 18 wheels, cargo only. Mass rises 2.26x and decel falls just 22%, because air brakes
are load-sensed, so braking force scales with the weight on each axle. A pure `1/mass` form
is 48% low there, which is exactly the regime where AEB must not under-brake.

Getting here took three wrong turns worth recording. First reading said `1/mass` was broken,
from assuming 12 wheels on every trailer rig, when the 24 t combination is a double with 18.
Second reading said `wheels/mass` fitted to ±5%, but every rig then available had wheels and
mass moving together (+6 wheels and +7 t per trailer), so the two exponents were not
separately identified: only a cargo change at fixed wheels separates them. Third mixed units,
fitting clip raw peak decel against probe `A_true` and landing 7% low. **Do not fit this
model on rigs that vary wheels and mass together, and normalise units first.**

Fit caveat: one user's trucks. The `sample_kind="brake"` rows in `coast_debug.csv` are the
cheap way to extend it, since a full-pedal stop needs no traffic.

**Reading those rows: take the peak A per stop, not a windowed mean.** `decel / frac(pedal)`
is only a capacity estimate once the plant has plateaued, and a stop from low speed brakes
for well under a second, so it never gets there. A fixed settling window then under-reports
by up to 25% and fakes a decay across a run. With peak-A, four back-to-back bobtail stops
20 s apart measured 9.19 / 9.01 / 9.51 / 9.41 (mean 9.28, ±2.7%) against a 9.74 prediction.
No fade: ETS2 exposes neither brake temperature nor wear, so neither can be compensated.

The old baseline was inverted *and* low, so for a loaded rig the partial-pedal candidate cap
sat at 7.9 m/s² against a real 13-14: every truthful sample was rejected as contaminated and
the estimate froze at ~10. That is why AEB believed 10.0 while the truck delivered 14
(high-speed stop overshoot notes, clip ab291591).

**Cargo only counts while a trailer is attached.** The SDK keeps reporting the assigned
job's `cargoMass` after you unhook, which read a bobtail as 39.8 t instead of 10.7 t and
corrupted every mass-scaled term (accel `weight_factor`, `gain_scale`, creep FF, this
baseline). `compute_estimated_mass_kg` drops cargo when `trailer_count == 0`.

**Measured brake plant, from 61 fitted braking episodes.** Fitting a first-order-plus-dead-
time build-up to the *speed* trace (never to differentiated decel, which amplifies the 20 Hz
physics staircase) gives tau 0.19 s median, 0.31 s p90, 0.38 s max, with 0.12 s dead time.
Build-up (dead + tau) is therefore 0.25 s median and 0.37 s p90. The observer's model taus
(0.25 solo / 0.50 trailer) sit above that, which is the intended safe side, and the fit
could not separate the load classes, so they are left alone. `stop_buffer_response_s` in AEB
is sized against this, not against the model. Full-pedal slams are faster than this (tau
0.08 s median, see the onset guard below), so the fit describes ordinary braking.

`brake_efficiency.nominal_max_brake_decel_ms2` now defers to `baseline_brake_ms2` instead of
carrying its own `11.5 * wheels/12 * 17000/mass`. That old form had the same `1/mass` error
and collapsed on a loaded rig: against the probe it read **45% low on the 54 t double** and
30% low at 40 t, which inverts the whole point of a degradation warning, since a healthy
truck looks like it is over-performing and real fade can never reach the ratio. It returns
decel *at brake=1.0*, so it scales the fitted asymptote by `brake_curve_fraction(1.0)`.

**Gas**: shape model `G(gear) = anchor * ratio^(_ANCHOR_GEAR - gear)` learned in log-space;
monotonic ratio clamp. Skipped after clutch, gear dwell, or moving pedal. Persisted to
`settings.json` on drift.

## AEB decel controller (`AEBDecelController` in `thread.py`)

Owns the brake pedal while `AEB_brake` is true (gas is cut in `main_pedal_thread`).
Feedforward is the inverse brake curve at the commanded decel; a disturbance observer
supplies the correction.

- **Observer**: a plant model (dead time then first-order lag) is driven by the brake
  actually written to the game, filtered with the same 0.12 s lag the measurement
  carries, and the residual against measured decel is the environment bias. Because
  model and measurement share the lag, the residual is bias rather than lag, so it
  settles with the plant instead of behind it. No integrator, so nothing winds up.
- **Load classes**: the plant model is keyed on trailer presence and sits at the measured
  lag: dead 0.10 s / tau 0.19 s solo (median of 61 braking episodes, dead 0.12), dead 0.12 s
  / tau 0.31 s with a trailer (the p90; the 2026-08-12 trailer probe read 0.22-0.30 and
  refuted the older ~0.65 s figure). The models used to sit at 0.25 / 0.50 s on purpose, so
  the observer leaned toward under-braking. Drivers paid for it: during the engagement slam
  the real truck ran far ahead of the slow model, the residual read as a large bias, and
  the pedal fell to near zero while the target was still 4-10 m/s2 (every no-driver stop in
  `brake_debug.csv`, 2026-10-04), which reads as "AEB braked for nothing". In the stop sim
  at 100% slider the lowest decel after the hit went 87% -> 94% of target and the residual
  gap +0.28 -> +0.46 m, p90 lag unchanged. `test_never_overshoots_the_target_decel` covers
  trailer plants up to 0.65 s; its 0.80 s case was dropped with Lukas's approval, since the
  measured maximum is 0.38 s. A fully median model (0.19 s both) reaches 100% but fails the
  0.65 s case and the "uses the truck it has" peak. With a single solo-tuned model, trailer
  plants overshot the target decel by up to 39%.
- **Onset guard** (`_AEB_ONSET_GUARD_S`, 1.0 s from engagement): the bias may add brake
  but not remove it. AEB always opens with a slam, and slams build fast: 17 full-pedal
  onsets in `brake_debug.csv` (2026-09-29 to 10-04, AEB and driver slams, solo and
  trailer) fit dead time 0.075-0.15 s and tau 0.08 s median (0.02-0.19), while gentle
  driver or cruise onsets fit 0.23-0.6 s. No single model fits both, and against the slower one the
  residual during build-up read as bias and cut the pedal: 64-77% of target 0.3-0.6 s
  after the hit at 100%, 155% and 300% alike. The stop sim put that dip at 0.49-0.74; the
  guard holds it at 0.90-0.97 with every gap equal or larger. A faster model was tried
  instead: it fixes fast plants but sends slow ones to 1.4-1.6x target and costs up to
  0.3 m. The price is paid only when the truck brakes harder than AEB believes: up to
  1.24x target for the first second on a truck 25% stronger, then 1.00x.
- **Slider axis** (`AebPedalAxis`): the controller's pedal is a point on its capacity's
  brake curve. Above `I = 1.1` the sending thread maps it to the sent pedal that delivers
  that decel and maps the sent pedal back for the observer. The tune range runs through the
  same remap as the driver's pedal; with a measured full pedal the extra between the tune
  top and a sent 1.0 is spread linearly over the rest of the travel, unmeasured that part is
  skipped. Full pedal is always a sent 1.0 so a slam reaches the slider's extra. Writing
  the pedal raw braked `I / 1.1` times harder than planned (2.7x at 300%), which the
  additive bias could only undo by cutting the pedal to ~0.06. At or below 1.1 the axis is
  the identity.
- **Measurement**: its own 0.12 s tracking differentiator. Do not point this at
  `_spd_smooth` (0.30 s), which `PedalCapacityTracker` and published telemetry depend on.
- The commanded decel is floored at `AEB_ff_decel_ms2` so a stale or zero published
  target cannot silence AEB, and the pedal merge stays a `max` so the driver can always
  out-brake it.
- **Saturation override**: when the uncapped `AEB_required_decel_ms2` reaches what pedal
  1.0 can deliver, the controller returns 1.0 immediately instead of inverting the curve
  at the capped target. AEB's `ego_decel_frac` (0.9) headroom is a tracking margin; once
  the threat needs more than the truck has there is nothing left to track, and holding
  back only costs metres. This matters most downhill, where `effective_max_decel` also
  subtracts the gravity term: on an 8% grade the capped target inverts to pedal 0.67.

Convergence is plant-limited, not filter-limited: solo reaches ~84% of target at 0.5 s
and ~96% at 0.8 s; a trailer cannot do better than its own ~0.65 s brake build-up. The
distance that build-up costs is paid for by `stop_buffer_response_s` in AEB, not here.

## Hold controller (`hold_controller.py`)

Single authority for hill rollback prevention. FSM: ROLLING / STOPPING / HOLDING / LAUNCHING.
Hold states combine slope feedforward (inverse brake curve) plus a rollback integrator that
only adds brake. LAUNCHING ramps feedforward down; integrator stays active; ramp retreats on
live rollback, not on stored integrator level (avoids steep-hill launch livelock).

ROLLING captures STOPPING below 2 km/h on any command at or below zero, except
while `crawl_follow` is set: ACC is keeping speed behind a lead it measures
moving (`commanded_crawl_follow` on the telemetry thread, passed in only while
the tracking commander owns the command). Then a mild decel is speed keeping, and
the capture still happens on a command at or below -0.3 m/s², under 1 km/h, on
any rollback, or when the measured (or commanded) decel plus 30 % of the uphill
grade's pull would reach zero within 1 s. Those four are what keep crawl follow
from ever rolling back further than the plain capture; do not relax one without
re-running the hill grid in `tests/test_hold_crawl_follow.py`. Rationale and
measurements: `core/acc/ACC_ARCHITECTURE.md` §10.3.

## Brake efficiency (`brake_efficiency.py`)

Optional cruise-only degradation warning via EMA of measured vs expected decel. Flat-road
gate; high-brake samples only. Expected decel follows the fitted brake curve, not a straight
line through the pedal: at the 0.70 sampling threshold the truck already makes 84% of full,
so the old `pedal * nominal` under-predicted by a fifth and read grip that much high.

**`BrakeEfficiencyTracker` is not wired to anything.** Nothing in `core/` constructs it; only
its three `Settings` flags exist. The model it references is now correct, but the warning
does not run, so fixing it changes no behaviour until something calls `update()`.

## Visualization bar (`visualization_bar.py`)

A 3 px always-on-top `Qt.Tool` strip along the bottom of the primary screen. Created on
the Qt main thread via `create_visualization_bar()`. It reads `aforward` / `abackward`
and flashes on `em_stop` / `AEB_warn`.

It must not call `raise_()` from its animation timer. A per-frame raise fights
`cc_panel` and can freeze Qt on Windows when the main window is minimised.
`WindowStaysOnTopHint` does not survive the NVIDIA overlay plus alt-tab: Qt still
reports the window visible, so `show()` never runs again. The main window puts it
back every 5 s with `SetWindowPos` (`ui/overlay_topmost.py`) without activating it.
The bar is `WA_ShowWithoutActivating` and `WindowDoesNotAcceptFocus` for the same reason.

## Brake intensity (`core/scs_profile/intensity.py`)

In-game **Braking intensity** (`g_brake_intensity`, 1/3 left, 1 centre, 3 right)
makes the same pedal brake harder. Mapper, AEB and ACC were tuned at **I = 1.1**.
The remap inverts it, last step before `SCSController.abackward`:

`sent = min(1, logical * 1.1 / I)`

**A given pedal must brake the same at every slider setting.** That is the point
of the remap and it is confirmed by feel (Lukas, 2026-09-29): a small brake input
has the same effect at either end of the slider. It applies to the driver's own
pedal as well as to cruise, deliberately; only AEB and an `em_stop` slam bypass it.
Keep it that way. `test_a_small_input_brakes_the_same_across_the_slider` pins it.

Do not restore a pedal power or treat UI 50/100/150 as the gain (150% is `I = 3`).
At `I = 1.1` the remap is identity. At `I = 1.0` it is `* 1.1`. Unreadable files
behave as `I = 1.0`.

### What the slider does, measured 2026-09-29

- Per unit of **logical** pedal the truck braked as it did before the remap
  existed (27-30 against 26-34 m/s2 per unit, lag-aligned ACC braking over nine
  days at `I = 2.158`). The linear invert holds partial braking constant.
- Full-pedal capacity rises far less than the slider. The in-game A/B (one
  12-wheel 17 t rig, 90 km/h, peak capacity as the curve asymptote) read 12.6 m/s2
  at 100%, 14.9 at 135% and 16.2 at 150%: x1.29 where the slider value alone
  predicts x3. The likely cause is traction: past ~15-16 m/s2 the tyres are at
  their grip limit and ABS caps the decel, while at 100% the rig is still
  brake-limited. At 100% it matched `baseline_brake_ms2` (12.57) to 0.1%.
- The September learner multiplied every measured decel by `1.1 / I` and divided by
  the curve at the sent pedal. At 135% that read the truck 51-85% as strong as it
  was, and full-pedal stops, capped by traction, read 0.51. `brake_scale` walked
  from 0.94 to 0.57, the mapper believed 7.2 m/s2 against a real ~12, and ACC's peak
  delivered/requested decel went to 1.21 (median, 182 stops). After the fix: 0.99
  (11 stops), against 1.02 before the remap existed.

`tools/brake_intensity_probe.py` re-runs these checks from `brake_debug.csv`.

What follows from it:

- **Learning** reads the sent pedal back through the same remap,
  `effective_brake_pedal` = `min(1, sent * I / 1.1)`, and never scales decel, so
  `brake_scale` and `max_brake_ms2` stay in tune units at any slider.
- **AEB capacity** comes from its own per-truck scale, see **AEB capacity per truck**
  below. The extra a high slider buys at full pedal is rig and load dependent, so it is
  credited only once measured on this truck, slider and load (**Full-pedal credit**).
  Unmeasured, never let AEB's capacity exceed `aeb_scale * baseline`: with the September
  learner recovered it would have believed up to 2x the truck, and on 09-21 it already
  believed ~19.7 m/s2 against stops of at most 17.5.
- **AEB and `em_stop`** still pass `full_authority=True` and write the logical
  pedal, so a slam writes 1.0 and saturation cannot be starved. AEB first puts its
  own pedal on the sent axis (`AebPedalAxis`, see the AEB decel controller).
- `brake_scale` persisted under the old learner is dropped once, keyed on
  `pedal_capacity_brake_model`.

Sub-engagement FF assist and cruise stay on the invert and on the tracker in tune
units, so a slider change does not retune ACC. `I < 1.0` cannot be fully
recovered: if AEB is enabled, warn once an hour.

`SendingThreadData.abackward` stays logical so the viz bar does not show the
remapped axis. `recent_brake_outputs` and the AEB observer use the sent value,
because `gameBrake` and the plant see that. CC's game-brake disengage compare
reads that ring buffer. `max_brake_ms2` on `SendingThreadData` is in tune units;
`aeb_max_brake_ms2` is what AEB reads.

### AEB capacity per truck (`aeb_capacity.py`)

AEB plans with `aeb_capacity_ms2(baseline, aeb_scale, I)`:

`baseline * min(aeb_scale * frac(min(1, I/1.1)) / frac(1), min(1, I/1.1))`

The second term is the slider cut AEB always used. At `I = 1.0` it is 1/1.1, and that
is not an error to remove: the model matched the truck at 100% (table below), so a clean
100% user engages at about 90% of what full pedal delivers, which is the room AEB needs
to brake harder or step down. The first term is what the learner's own pedal model says
full pedal gives at this slider. Taking the lower stops a low slider being counted twice:
the learned scale already absorbs part of the cut, and multiplying the cut on top read a
slider-minimum rig at 0.15 of the model while it stopped at about 0.45.

`aeb_scale` is learned per truck, keyed `game|truck_id|trailer_count`, and only from
firm braking: tune pedal at least 0.5, or a sent pedal of at least 0.9 on a slider too
low to reach that. It takes the tracker's settled sample (`last_settled_brake_sample`)
before the mapper's 1.35x cap gate, clamps it to 0.35-1.00 and moves at
`0.04 * firmness^3` per sample, so one full stop converges. An unmeasured truck starts at
0.95. At most 64 trucks are kept, least recently braked first out, persisted in
`aeb_brake_scales`. The mapper and ACC keep the global `brake_scale` unchanged.

Why it is separate, from 2026-10-03/04 data:

- Light braking reads low through the brake curve and road load: tune pedal under 0.3
  gave median candidates 0.69 at 135% and 0.47 at 100%, firm braking 0.96. Light samples
  carried about half the learning weight, so the shared scale only drifted down.
- One shared scale mixes trucks: a 19 min ATS drive moved the ETS2 FH's estimate from
  0.940 to 0.888, though that ATS truck stops at 0.97 of the model.
- Contributed v1.1.0 clips: 45 of 49 rig clusters ran AEB at or below 0.92 of the model,
  20 had learned down to 0.60-0.88.

Full-pedal stops 2026-10-04 (peak 0.5 s decel less road load, over `frac(1)`), divided
by `baseline_brake_ms2`:

| rig | 100% | 139% | 300% |
|---|---|---|---|
| FH bobtail 10.5 t | 1.03, 0.93 | 1.40 | 1.47 |
| FH single, 17.5 t | 1.00, 0.96, 0.89 | 1.25 | 1.44 |
| FH single, 25.7 t | 0.99 | 1.29 | 1.63-1.68 |
| FH single, 39.6 t | 1.15, 1.09 | 1.14 | |
| ATS 10-wheel, 36 t | 0.97, 0.97 | | |

The model holds to about 5% at 100%. Above it the gain runs from +3% to +65% by load,
which is why `aeb_scale` never carries it: a scale learned on the 25.7 t load at 139%
would over-read the 39.6 t load by about 13%. The ceiling stays 1.00 for the same
carry-over reason as `brake_scale`.

#### Full-pedal credit (`FullPedalStore`)

Peaks over-state what a stop delivers. Averaged over each settled full-pedal stop
instead, the FH bobtail at 135% stops at 1.27x the model (9 stops, 1.07-1.36) while AEB
planned with 0.95x, so it fired when the stop needed about 75% of the truck: "too early"
on 2026-10-04 even with the onset guard. The extra cannot be predicted (same trailer:
1.03 at 17.5 t, 1.19 at 25.7 t, 1.05 at 39.6 t, all at 139%), so it is measured.

- A **full-pedal stop** is a sent pedal of at least 0.97 on full authority (driver slam or
  long AEB saturation). `FullPedalRun` fits a least-squares slope to the speed trace from
  0.3 s after the slam while above 18 km/h, needs at least 0.5 s of it, and reports
  `(decel - road load) / (baseline x frac(1))` when the pedal comes off. It does not use
  the tracker's settled samples: during a slam the differentiated decel swings +-2 m/s2
  on the 20 Hz physics staircase, and on 2026-10-04 two full stops from 124 and 75 km/h
  produced one settled tick between them.
- Stops are kept per `game|truck|trailers|slider`, then per load within 10% of mass (four
  loads, last five stops each, 64 keys), persisted in `aeb_full_pedal`.
- **Credit** needs two stops at this load and uses the weakest of the last five. Capacity
  becomes `max(today, baseline x ratio / 1.1)`: the same 1/1.1 room a clean 100% user has,
  never lower than the uncredited value. No transfer across loads or sliders. The weakest,
  not a median, because the same truck and slider spread more than the room between
  drives: the FH at 135% read 1.26-1.38 in the morning and 1.07-1.16 that evening
  (2026-10-04, same speeds; cause not established, weather is the suspect).
- In the stop sim (`stop_buffer` 0.5) an exact measurement moves the bobtail's engagement
  at 80 km/h from 35.3 m to 30.2 m at 135% (28.2 m on the 25.7 t load at 150%) with every
  gap at or above +0.12 m,
  including extra-braking shapes the axis does not assume. A measurement 5% high lands at
  the bumper (-0.03 m) only with the p90 build-up, which no AEB slam has shown.
- Rain is not visible in telemetry. If wet grip cuts the extra, a dry measurement
  over-reads it; this was accepted on 2026-10-04 for a later, more precise trigger.

## Brake debug log (`debug_csv.py`)

With `debug` on in `config.json`, `brake_debug.csv` at the project root gets every
braking tick at 20 Hz plus a 2 s tail: speed, the 0.30 s and 0.12 s decel
measurements, road load, each brake source (user, mapper, hold, AEB), the pedal
before and after the intensity remap, the tune pedal learning inverts, AEB's
target and demand, the learned capacity, and which gate ended the learner's tick.
Off by default, so it never ships to drivers. A header change rotates the old file
aside, the same as `coast_debug.csv`.

## Main pedal thread

See `core/main_pedal_thread/README.md` for joystick, OPD, em_stop, and button capture.
