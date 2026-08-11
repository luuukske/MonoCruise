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

- Road load is gravity along grade plus rolling resistance. Pitch comes from telemetry
  `rotationY` in degrees, converted with `math.radians` (same convention as AEB), scaled
  by the `mapper_rolling_resistance` setting.
- Adaptive accel/brake estimates learn from load-compensated accel (`raw + road_load`),
  so a slope cannot bias the learned full-pedal capability.
- Also holds the shared telemetry mass-estimate helper that `telemetry_thread` uses.

## Pedal capacity (`pedal_capacity.py`)

Always-on brake decel and gas gain learning (replaces legacy brake efficiency tracker).

**Brake**: `update_brake` every tick; accept samples only when pedal and decel settle.
Candidate inverts the fitted brake curve; pedal³ weighting; underperformance drops estimate
2× faster than overperformance rises. Road load canceled before sampling. Fast EMA during
deep settled AEB braking. Partial-pedal candidates reject above
`_BRAKE_CANDIDATE_MAX_FRACTION` (1.35) of the load baseline; settled high pedal may use the
nominal ceiling.

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
(`docs/aeb_high_speed_stop_overshoot.md`, clip ab291591).

**Cargo only counts while a trailer is attached.** The SDK keeps reporting the assigned
job's `cargoMass` after you unhook, which read a bobtail as 39.8 t instead of 10.7 t and
corrupted every mass-scaled term (accel `weight_factor`, `gain_scale`, creep FF, this
baseline). `compute_estimated_mass_kg` drops cargo when `trailer_count == 0`.

The `_BRAKE_CEILING_NOMINAL_MULT` path stays for now; it was a workaround for the broken
baseline, and retightening it needs in-game confirmation of the corrected values first.

`brake_efficiency.nominal_max_brake_decel_ms2` (`11.5 * wheels/12 * 17000/mass`) carries the
same `1/mass` error and is therefore badly low on loaded rigs. It only feeds the optional
degradation warning, so it is left alone for now, but do not reuse it as a capacity model.

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

Optional cruise-only degradation warning via EMA of measured vs expected decel (nominal max
scales with wheels and mass). Flat-road gate; high-brake samples only.

## Main pedal thread

See `core/main_pedal_thread/README.md` for joystick, OPD, em_stop, and button capture.
