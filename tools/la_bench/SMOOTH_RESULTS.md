# LA servo: smooth at target on any wheel, no calibration

Everything in this file is **simulated**. No real wheel was available. Branch `la-smooth-ffb`, cut from `v1.2` at
`7185d85`.

## Decision log

Newest last. Every measured effect is on simulated wheels.

1. **Plant: worse-wheel features as optional `WheelParams` fields** (`dwell_s`, `backlash` with `motor_share`,
   `stick_speed`, `stall_every_s` and `stall_s`, plus an `external` force for hands). Why: the brief's population
   needs them and the handoff listed them as the model's gaps. Effect: the five presets replay bit-identically
   (step and slow ramp, two designs each, angle traces and replays hashed before and after the change). Cost: a
   second integration path for backlash (motor and rim, stiff damped contact at 40 Hz).
2. **Population and metrics as a new module** (`tools/la_bench/population.py`): 200 wheels from a fixed seed, every
   fifth held out, the hold test and its four metrics. The bench, check, probe and design code are kept as they were:
   nothing is replaced, so the five presets still verify through `run_check` as before.
3. **Final error is measured beyond what the reading resolves** (decided before any tuning, after the baseline run):
   half a reading step plus the distance from the target to the nearest step. A servo that only sees a 5 mlock
   reading cannot park a wheel within 0.5 mlock of a target between two steps; parking it in the nearest step counts
   as on target.
4. **The learner lives in MonoCruise (`core/la/wheel_learner.py`), not in the plugin**, fed from the plugin's existing
   trace block (every FFB update). Why: the DLL cannot be rebuilt here (no MSVC, no mingw), the trace block already
   carries the reading, the game's force, what went to the wheel and the steering frame count, and Python is where
   the design and its bounds live. Cost: the trace block is now read every tick, not only with `debug` on; learning
   lags by one 20 Hz tick, which does not matter for slow estimates. No layout change, no DLL change.
5. **A target hold in MonoCruise (`core/la/hold.py`)**, acting only through the gains and pushes the layout 5 plugin
   already takes: once the wheel stands on the target, kp, kd, ki and the friction pushes fade to zero, so the
   plugin's integral stands still and keeps the force that parked the wheel. Why: with the true wheel's parameters
   the plugin law still hunted at the target on sticky and backlash wheels (about 20 sign flips per 2 s on a fifth
   of the population); a plugin-side hold would react at the FFB rate but cannot ship without a rebuild. Effect: on
   the tuning wheels that hunted with the reference design, 9 of 13 stopped hunting (for example 23 to 0 and 60 to 0
   flips per 2 s). Cost: it parks 0.15 s after the wheel stands and releases one tick (50 ms) after it is pushed off.
6. **Friction is learned from LA's own pushes, not from manual driving.** Why: with hands on the rim, manual driving
   shows the driver's torque, not the rim's friction; LA's pushes are known and the learner adds none. Two
   estimates: the force a standing wheel started on after a slow climb (a fast climb runs past the start before the
   reading shows it, which read 1127 on a 370 wheel), and the forces it stood against for 0.15 s. Cost: friction is
   unknown for the first minutes LA steers.
7. **Inertia and delay are not learned; a persistent design ladder replaces them.** Why: neither shows in normal
   driving with hands on the rim. What they cause, a servo hunting at a still target, does: each such hunt steps the
   design one rung gentler (bandwidth x0.7, design inertia x0.7, damping +0.2, damping first; floor 0.5 Hz on 150),
   ten minutes of servo time without one steps it back. Never stiffer than the configured design.
8. **Persisted in `config.json` (`la_wheel_learned`)**, as `la_saved_calibration` is: local, offline, bounded on read.
9. **`WheelTuning.hold` and `.learn` default off; the runners turn both on from `la_wheel_learning` (default on).**
   The pure-logic default is the old servo bit for bit, so every existing test still checks what it checked.
10. **No driving-condition table.** Tried: the game's spring at speed (15000 to 60000 per lock) on the population.
    In the model the game's spring acts with no delay on a light rim and rang with any servo design, the default
    overshooting 6x; that says more about the model than the servo, so the verdict stays on the bench condition the
    presets were fitted in.
11. **Owner decisions, 2026-11-08 (Lukas, asked when he was back):** the final-error limit is 1.5 mlock, the plugin's
    own breakaway hold-off (`FFB_BREAK_HOLD`), instead of 0.5: with the wheels' true parameters about 65 % of the
    tuning wheels parked 0.5 to 1.5 mlock off, inside the band where the layout 5 law leaves only the integral. And
    the plugin's C law may change behind a new status bit (the shipped DLL keeps working, Lukas rebuilds it).

## Feature merit check

Verdict: **ship with changes**.

Strongest case against, from the driver's seat:

- The wheel would change how it feels over the first minutes of a drive, and again after a different truck or wheel
  setting, with nothing on screen saying why. A servo that is fine on Monday and a little different on Tuesday is the
  kind of surprise a driver reports as "LA feels random".
- With hands on the rim, manual driving barely shows the rim's own inertia, friction or delay: the driver's arms
  dominate. A learner that claims to read them from normal driving can mislearn and quietly make the servo worse than
  the default, and the driver has no way to see or undo that.
- LA is off by default and hidden; few drivers get it. Any learner adds code and a persisted file for a small
  audience, and the existing optional wheel check already covers the testers who care.

Changes that make it worth shipping:

- Learn only what normal driving actually shows (how often and how finely the game reads the wheel, how the game's
  centring pushes it, when a standing wheel starts), and say so; never claim inertia or delay from data that cannot
  show them.
- Start from the current defaults and let learning only ever make the servo gentler at the target, never more
  aggressive than the default, so the first minutes and any mislearning are no worse than today.
- Bound every estimate, persist it locally (offline, no telemetry), and keep the wheel check as the optional fallback.

## Metric thresholds (set before tuning)

Steady state at target: the last 2 s of a 5 s hold after a 0.02 lock step, both up from rest and on the swing back
through zero (worst of the two). `F` is the wheel's breakaway force (its sliding friction when it has no breakaway).

| metric | definition | satisfactory |
| --- | --- | --- |
| jitter | RMS of the servo output change per FFB tick, over `F` | at most 0.05 |
| jumps | largest single-tick servo output change, over `F` | at most 0.5 |
| flips | sign changes of the servo output in the 2 s window, counted with a hysteresis of 0.1 `F` | at most 1 |
| final error | mean angle over the last 0.5 s against the target, less what the reading cannot resolve (half a reading step plus the target's distance from the nearest step; decision 3), in mlock | at most 1.5 (Lukas, 2026-11-08; was 0.5, decision 11) |
| overshoot | the step's peak past the target, share of the step | reported; 0.2 is the check's ACCEPT_UP |

A held-out wheel passes when all four limits hold. The set passes when every held-out wheel passes (reported as a
pass share too) and the five presets do not regress against the handoff baseline.

## Assumptions

- Hands off the rim in the step tests, as the bench runs them: the lightest rim, so the most prone to ring.
- The game's spring in the step tests is the wheel's standing spring (the bench's), not the stiffer one at speed.

## Results

(pending)
