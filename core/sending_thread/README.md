# Sending thread and pedal I/O

> Maps commanded accel to game pedals via shared memory (`SCSController`).
> Longitudinal arbitration in `cruise_control_thread`; agent rules in root `AGENTS.md`.

## Sending thread (`thread.py`)

Opens SCS controls, runs `AccelToPedals`, hold FSM, pedal capacity learning, optional
visualization bar, hazard toggling, AEB decel assist, auto-neutral, creep compensation,
and commander merge (CC/ACC/limiter/AEB/user). Publishes `aforward` / `abackward` on
`SendingThreadData`.

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

## Pedal capacity (`pedal_capacity.py`)

Always-on brake decel and gas gain learning (replaces legacy brake efficiency tracker).

**Brake**: `update_brake` every tick; accept samples only when pedal and decel settle.
Candidate inverts the fitted brake curve; pedal³ weighting; underperformance drops estimate
2× faster than overperformance rises. Road load canceled before sampling. Fast EMA during
deep settled AEB braking. Candidates reject above `_BRAKE_CANDIDATE_MAX_FRACTION` (1.35) of
the load baseline.

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

`_UNDERPERFORM_MULT` (2.0) biases the settled estimate 4-10% low under symmetric candidate
noise, measured; at the observed scatter it is ~10%. That is deliberate — believing
degradation quickly is the safe asymmetry — and low is the safe direction, so it stays.

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
is sized against this, not against the model.

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
- **Load classes**: the plant model is keyed on trailer presence. Measured from the clip
  corpus, a solo tractor reaches t63 in ~0.22 s while a trailer's air brakes need ~0.65 s.
  Both model taus are set above the measured median deliberately: a model slower than the
  real plant biases the observer toward under-braking, and only over-braking is dangerous.
  With a single solo-tuned model, trailer plants overshot the target decel by up to 39%.
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

It must not call `raise_()` from its animation timer. `WindowStaysOnTopHint` already
keeps it above other applications; a per-frame raise fights `cc_panel` wherever they
overlap and can freeze Qt on Windows when the main window is minimised. The bar is
`WA_ShowWithoutActivating` and `WindowDoesNotAcceptFocus` for the same reason.

## Main pedal thread

See `core/main_pedal_thread/README.md` for joystick, OPD, em_stop, and button capture.
