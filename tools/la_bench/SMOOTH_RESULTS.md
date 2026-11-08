# LA servo: smooth at target on any wheel, no calibration

Everything in this file is **simulated**. No real wheel was available. Branch `la-smooth-ffb`, cut from `v1.2` at
`7185d85`.

## STOPPED

- **Trigger:** the stuck rule. After the last kept improvement (decision 18; decision 21 is a correctness fix that
  measured neutral) three different approaches to the remaining failures (wheels parked more than 1.5 mlock off, or
  never reaching the target) did not improve the tuning set, so no fourth was tried. Stopped at 13:45 UTC on
  2026-11-08, before the 15:00 deadline. Nothing is blocked and nothing unsafe was found; every change is committed
  and pushed (`la-smooth-ffb`), so no WIP commit was needed.
- **Tried, one line each** (tuning set, learned + settle law, against 84 of 160 at decision 18):
  - hunt detection at 6 flips in 2 s instead of 4: 63 of 160;
  - hunt detection at 3 flips: 71 of 160;
  - wheels without a learned friction push held at the fourth design step instead of the sixth: 81 of 160.
  - (before those, also dropped: a 1.1 margin on the top start, 69; an integral speed-up while stuck far, 73;
    lower damping per step, 70.)
- **Command and output tail:** `python -m tools.la_bench.smooth_eval tune --part k --of 3 --out <file>` (k = 0, 1,
  2), pass row: `| pass | 4/160 (2 %) | 69/160 (43 %) | 81/160 (51 %) | 77/160 (48 %) | 89/160 (56 %) |`.
- **Held-out state** (commit `cc54838`): 19 of 40 meet every limit with the settle law, 17 of 40 on today's plugin
  (0 of 40 with today's default); 32 and 26 of 40 are smooth at the target (jitter, jumps and flips all within the
  limits). Not satisfactory (all 40).
- **The one decision needed from Lukas:** whether the next step is the real wheel. The model cannot say more about the
  remaining failures without real data (they sit on the friction, delay and hands the model guesses at). If yes:
  rebuild the DLL with the settle law (`plugins/ffb/README.md`, Build), drive with `la_wheel_learning` on (the
  default) and send `la_wheel_learned` from `config.json` and the `la_ffb_debug.csv` trace (with `debug` on) of a few
  LA minutes; if not, say which of the remaining classes to chase in the model.

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
   design one rung gentler (bandwidth x0.7, design inertia x0.7, damping first; floor 0.5 Hz on 150, later 50),
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
12. **Ladder patience without a friction push.** Without a learned push the design stays at most two steps down (a
    gentle servo alone cannot move a sticky wheel), but two more hunts there take it further, one step per hunt.
    Why: wheels that rang at the second step never stood still long enough to learn their friction, so they rang
    for good. Effect (tuning subset of 60, full learning loop): rough wheels 22 to 11, pass share 0.40 either way.
    Cost: such a wheel can end parked short until its friction is learned.
13. **The friction push is 0.6 of the start force, the slide fit is no longer used.** Why: the steady-slide fit read
    a wheel sliding on 271 as 3 and on 212 as 123, which made the breakaway push climb from almost nothing to the
    top start and drop back every cycle (single-tick jumps of 0.5 to 1.0 of the breakaway). Effect: neutral on the
    pass share (0.383 to 0.40 on the subset), fewer jumps.
14. **A settle law in the plugin** (`plugins/ffb/ffb_signal.h`, servo flag bit 2, status bit 4096; no layout change,
    the 128-byte block keeps its fields). The breakaway push finishes the move down to 0.5 mlock plus half a reading
    step instead of standing off inside 1.5 mlock, climbs over 1 s from nothing near the target, lets go over 20 ms
    instead of in one tick, and parks a wheel standing in that band at the FFB rate. Why: on the tuning set the
    learned design left 95 of 160 wheels more than 1.5 mlock off, most of them in the 1.5 to 3 mlock zone where the
    layout 5 push is only partly on. Effect (saved learned models, tuning set): pass 0.362 to 0.487 (the release at
    30 ms gave 0.469, 60 ms 0.463, 12 ms 0.431). Cost: on the light, very sticky preset (`sticky`) each start throws
    the rim 8 to 15 mlock and the law hunts; hence decision 15. C law, Python mirror and parity tests changed together;
    the C test builds with gcc and passes. **The DLL is not rebuilt** (no MSVC here): `core/sdk_installer/own/` and
    its SHA in `own.py` are unchanged, the shipped layout 5 DLL ignores the flag bit, so nothing changes in game until
    Lukas rebuilds it.
15. **The learner turns the settle law off for a wheel that still hunts at its gentlest design** (rung 6, the hunt
    count's top). Ten minutes of calm servo time step it back, so it is tried again about every ten minutes.
16. **Fixes found on the tuning set, 2026-11-08.** (a) The learner skips samples no newer than the last (so LA and
    LKA can share one); the simulated drives restarted their clock at 0, which dropped most of the LA phase (no starts
    learned at all): the sim now continues the clock. (b) The forces a standing wheel "resisted" included short
    spikes it rode out by less than a reading step and forces from just before it stopped (832 read on a 480 wheel):
    now a 50 ms low-pass of the force, counted only after the last reading change, recorded every second a wheel
    stands (a wheel that never moves taught nothing before), and the top start is the most it resisted. (c) The
    quantum needs 60 reading steps, not 200: coarse encoders made too few in a minute (20 of 160 read as fine).
    (d) Two more design steps, down to a design inertia of 50 (light rims with 45 to 60 ms of delay rang at 150);
    settle off after eight hunts. Effect, tuning set, learned + settle law: 66 to 76 of 160; quantum right on 153 of
    160 (was 140).
17. **A hunt counts push flips past a tenth of the wheel's start force** (at least 30, as before), the flip metric's
    own hysteresis. Effect, tuning set, learned + settle law: 76 to 82 of 160, smooth 131 to 132; on the layout 5
    plugin 61 either way. Tried and dropped: the top start at 1.1 times the most a standing wheel resisted (69 of 160,
    more flips: an over-push) and four design steps instead of six (81, but 119 smooth).
18. **The design steps keep the configured damping ratio.** They raised it by 0.2 a step to 1.4; with the delay on a
    light rim the bigger damping term itself rang. Effect, tuning set: learned + settle law 82 to 84 of 160, layout 5
    61 to 69, smooth 132 and 107 to 132 and 116. Tried and dropped: lowering it 0.05 a step to 0.5 (70 and 57). Also
    tried and dropped: an integral eight times faster while a wheel stands more than 4 mlock off a still target for a
    second (73 with the settle law, and it would wind the integral faster against a driver holding the wheel).
19. **The lane keeping assist takes the hold and the settle flag, not the learned design.** LKA ships to drivers
    (Steer), and the design steps lower the servo's bandwidth after hunts at still targets, which could weaken a
    pulse's push back from the line; that was not measured, so LKA's design and pushes stay as configured. Its trace
    still feeds the shared learner. Cost: LKA does not get the friction push on a sticky wheel.
20. **Hunt detection stays at 4 flips in 2 s.** Measured on the tuning set with the final configuration: 6 flips gave
    63 of 160 with the settle law (93 smooth), 3 flips 71 (133 smooth), against 84 (132 smooth) at 4.
21. **The plugin's settle hold lets go when the target moves past its band** (C and mirror). Found by driving the C
    law and the Python mirror with the same scripted readings: both matched to the last bit over 480 ticks, and both
    kept a wheel parked with its old push when the target moved by less than the release margin (1 mlock plus the
    band). LA's target moves all the time, so this matters more in the game than in the step test: tuning set 82 of
    160 with the settle law, against 84 before (within run-to-run noise).

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
- Manual driving in the model: the driver's hands pull the rim toward a wanted angle (three slow sines, 0.05 to
  0.4 Hz, up to 40 mlock) with 10000 to 40000 units per lock and 100 to 400 of damping, servo off, 60 s per wheel.
  Real drives give the learner far more than a minute; 60 s is the short end.
- LA's part in the model: still targets 5 to 30 mlock off for 4 s each, hands off (the harsher case; resting hands
  would add friction, which the learner would read as the wheel's, as LA would meet it).
- The model reads the wheel on FFB ticks, so its reading rates are 120/n Hz (60, 40, 30); the learner is scored
  against that.
- `la_wheel_learning` defaults to on: nothing for the driver to do is the point. LA itself stays off by default; the
  setting off restores today's servo exactly (`tests/la/test_runner_learning.py`).
- "A small local JSON": the learned model is a `config.json` key, as the saved calibration already is (decision 8).
- The population: every wheel has a breakaway of 300 to 700 and slides on 40 to 95 % of it; inertia 30 to 800 and
  damping 100 to 1500 (log-uniform), spring 300 to 3000, delay 20 to 100 ms, reading 30 to 60 Hz, quantum 0 (a fifth)
  or 0.2 to 5 mlock, start spread up to 0.4; dwell stiction, backlash (1 to 2 mlock) and speed-dependent friction on
  about half, reading stalls (40 to 120 ms, every 1 to 4 s) on about 40 %.

## Results

All simulated, on the bench condition (standing, hands off). Run with `python -m tools.la_bench.smooth_eval held
--out <file>` (and `tune`, `presets`; `--part k --of n` splits a run). Each wheel gets a 60 s manual drive (hands on the
rim, servo off), then 30 LA holds of 4 s each (still targets 5 to 30 mlock off, the servo on and nothing else pushed),
the learner fed the plugin's trace samples throughout; then the 0.02 lock step test on what it learned.

Columns: **default** is what a driver gets today without the wheel check (2 Hz on 650, no push); **learned, layout 5**
is the learner and the target hold with the plugin as shipped; **learned + settle law** adds the plugin's new settle
law (decision 14, needs the rebuilt DLL); **reference** is the tuning grid's most robust design given the true rim's
friction, delay and damping (what a perfect wheel check would know), with the hold.

### Held-out set (40 wheels, every fifth of the 200), final configuration (commit `cc54838`)

| metric | default (today, no check) | learned, layout 5 plugin | learned + settle law | reference (true rim) + hold | reference + settle law |
|---|---|---|---|---|---|
| pass | 0/40 (0 %) | 17/40 (42 %) | 19/40 (48 %) | 17/40 (42 %) | 25/40 (62 %) |
| smooth (jitter, jumps, flips all in) | 0/40 | 26/40 | 32/40 | 29/40 | 32/40 |
| jitter > 0.05 | 40 | 14 | 8 | 11 | 8 |
| jumps > 0.5 | 39 | 10 | 4 | 7 | 4 |
| flips > 1 | 40 | 5 | 4 | 4 | 4 |
| final > 1.5 mlock | 33 | 21 | 21 | 21 | 13 |
| jitter median / p90 | 0.673 / 1.161 | 0.000 / 0.110 | 0.020 / 0.061 | 0.000 / 0.225 | 0.000 / 0.106 |
| jump median / p90 | 2.290 / 3.59 | 0.000 / 0.85 | 0.112 / 0.44 | 0.000 / 0.74 | 0.000 / 0.46 |
| flips median / p90 / worst | 6 / 19 / 42 | 0 / 2 / 4 | 0 / 1 / 4 | 0 / 0 / 32 | 0 / 1 / 16 |
| final mlock median / p90 | 138.34 / 825.6 | 1.54 / 20.0 | 1.75 / 20.6 | 1.53 / 3.8 | 0.77 / 3.3 |
| overshoot median / p90 | 19.14 / 66.41 | 0.00 / 0.81 | 0.06 / 1.30 | 0.00 / 1.15 | 0.03 / 1.46 |

Convergence (after the manual drive and that many LA holds; at 0 nothing about friction is known yet):

| LA holds | layout 5 | settle law |
|---|---|---|
| 0 | 3/40 | 2/40 |
| 8 | 16/40 | 18/40 |
| 16 | 19/40 | 24/40 |
| 30 | 17/40 | 19/40 |

The held-out set was run four times in the session, each time on the configuration the tuning set had chosen:
18, 19, 20 and 19 of 40 with the settle law, after decisions 16, 17, 18 and the settle hold fix. No choice was made
on it; the last is the one that ships.

### Tuning set (160 wheels), final configuration

| metric | default | learned, layout 5 | learned + settle law | reference + hold | reference + settle law |
|---|---|---|---|---|---|
| pass | 4/160 (2 %) | 69/160 (43 %) | 82/160 (51 %) | 77/160 (48 %) | 89/160 (56 %) |
| smooth (jitter, jumps, flips all in) | 4/160 | 116/160 | 131/160 | 138/160 | 124/160 |
| final > 1.5 mlock | 141 | 83 | 74 | 82 | 58 |

How the tuning set moved through the run (learned, settle law where it exists): 2 % default; 26 % with the first
learner and the hold at the 0.5 mlock limit; 38 % at the 1.5 mlock limit; 41 % with the settle law; 48 % after the
fixes of decision 16; 51 % after decision 17; 52 % after decision 18 (43 % on the layout 5 plugin); 51 % after the settle hold fix (decision 21).

### The five presets against the handoff baseline

The handoff baseline is the wheel check's own design for each preset (`run_check` on the model, full grid), held
without the hold.

| preset | handoff check design (no hold) | today's default | learned, layout 5 | learned + settle law |
|---|---|---|---|---|
| `r3` | verified, `bw1.44 z0.7 J598.845 dfirst ahead plan37ms`; pass: jitter 0.002, jump 0.01, flips 0, final 0.07, over 0.36 | fail: jitter 1.999, jump 6.43, flips 6, final 66.23, over 3.68 | pass: jitter 0.000, jump 0.00, flips 0, final 1.47, over 0.00 | pass: jitter 0.003, jump 0.02, flips 0, final 1.24, over 0.00 |
| `bare` | verified, `bw0.5 z1 J578.597 dfirst ahead plan33ms`; fail: jitter 0.000, jump 0.00, flips 0, final 1.66, over 0.00 | fail: jitter 2.023, jump 6.58, flips 5, final 251.31, over 25.87 | pass: jitter 0.002, jump 0.02, flips 0, final 1.19, over 0.00 | pass: jitter 0.001, jump 0.01, flips 0, final 0.68, over 0.00 |
| `geared` | not verified, `bw0.5 z1.4 J50 dfirst`; fail: jitter 0.000, jump 0.00, flips 0, final 19.00, over 0.00 | pass: jitter 0.000, jump 0.00, flips 0, final 0.00, over 0.55 | pass: jitter 0.000, jump 0.00, flips 0, final 0.04, over 0.00 | pass: jitter 0.000, jump 0.00, flips 0, final 0.00, over 0.00 |
| `sticky` | verified, `bw1.32 z0.7 J153.194 dfirst ahead break537 plan28ms`; pass: jitter 0.001, jump 0.00, flips 0, final 0.66, over 0.16 | fail: jitter 0.323, jump 1.79, flips 12, final 2.88, over 9.18 | pass: jitter 0.000, jump 0.00, flips 0, final 0.43, over 0.05 | fail: jitter 0.000, jump 0.00, flips 0, final 1.56, over 0.11 |
| `old` | not verified, `bw2.5 z0.7 J100.207 dfirst ahead break584 plan16ms`; pass: jitter 0.001, jump 0.00, flips 0, final 0.00, over 6.55 | fail: jitter 1.254, jump 2.87, flips 10, final 7.14, over 15.04 | pass: jitter 0.000, jump 0.00, flips 0, final 0.65, over 0.18 | fail: jitter 0.000, jump 0.00, flips 0, final 2.54, over 0.20 |

Against the handoff baseline, on the layout 5 plugin (what ships today): all five pass, smoother (no flips, no jitter,
no overshoot past 0.18); `bare` and `geared` pass where the check's design did not (the check did not verify `geared`
at all), `sticky` parks nearer (0.43 against 0.66 mlock), `old` passes where the check's design was not verified and
overshot 6.5 times the step, and `r3` parks further off (1.47 against 0.07 mlock, within the limit). With the settle
law the two sticky presets regress on final error: `sticky` 1.56 and `old` 2.54 mlock, just and clearly past the
limit, both still smooth. Per preset the result depends on the learning path: other seeds moved `old` between 0.13
and 3.8 mlock with the settle law.

### What the learner learns, and how fast (held-out set, simulated)

- quantum within 5 % (or both fine): 38/40; fine read as coarse 0; coarse read as fine 2
- reading rate within 5 % of the model's tick-quantised rate: 40/40
- game spring learned on 40/40, learned/true median 1.000, p10 0.995, p90 1.002
- start force learned on 39/40, learned/true breakaway median 0.76, p10 0.57, p90 0.97
- rung after the drive: {3: 8, 4: 9, 5: 4, 6: 9, 7: 3, 8: 7}
- tracking slow ramp 0.06 at 0.05/s: default rms median 27.2 mlock (21 ran away), learned median 26.0 mlock p90 41.2 (0 ran away)
- tracking slow sine 0.03 at 0.4 Hz: default rms median 35.2 mlock (20 ran away), learned median 16.6 mlock p90 24.2 (0 ran away)

- Reading rate and quantum: from the first minute of manual driving (60 reading steps). The game's centring: exact on
  the model, where it is a pure spring; in the game it depends on speed and cornering, and the existing calibration
  still sets the servo.
- Start force: from LA's own holds, ready after 8 to 16 holds of 4 s (about a minute of LA steering at still
  targets); it reads low (median 0.80 of the model's breakaway), the safe side: an under-push leaves the rest to the
  feedback.
- Design step: on the bench condition the default design hunts on almost every wheel, so most wheels step down
  within the first holds; the pass share peaks after 16 holds (24 of 40 with the settle law, 19 on the layout 5
  plugin) and the later steps cost some of it (19 and 17 at 30 holds).
- Inertia and delay: not learned (decision 7).

### Tracking cost

Error rms with the learned design (settle law) against the default on a slow ramp and a slow sine, the hold not
involved: the last two lines above. The default ran away (the bench aborted) on about half the wheels; the learned
design never did. Where both ran, its slow-ramp error is about the same (28.5 against 27 mlock rms median) and its
slow-sine error is half. Fast moves were not measured.

### Exit reason

STUCK rule, at 13:45 UTC (see STOPPED at the top), with the deadline at 15:00: unsatisfactory, 19 of 40 held-out
wheels meet every limit with the settle law and 17 of 40 on today's plugin (the brief asks for all of them). On
today's plugin the five presets pass and do not regress except `r3`'s final error (1.47 against 0.07 mlock, inside the
limit); with the settle law `sticky` and `old` regress on final error. No cap, threshold, default or safety check was
changed.

### Known limits (simulated)

- Final error: about half the held-out wheels park more than 1.5 mlock off, and a few never reach the target (p90
  about 20 mlock): a light, delayed rim that keeps ringing at every design step never stands still long enough to
  learn its friction, and without the friction push the gentle design cannot move a sticky wheel.
- On the light, very sticky preset (`sticky`) the settle law hunts with a fixed design; the learner turns it off after
  eight hunts.
- The model has no hands in the step test and a standing spring; the game's spring at speed was not modelled
  credibly (decision 10).
- Results per wheel depend on the learning path (seeds): read single-wheel results as noisy and the shares as the
  result.

### What only Lukas can confirm on the real wheel

1. Rebuild `plugins/ffb` with MSVC (`plugins/ffb/README.md`, Build), copy it to `core/sdk_installer/own/` and put its
   git blob SHA in `core/sdk_installer/own.py`; the plugin then reports status bit 4096 (settle). Until then the
   settle law is source only and MonoCruise's flag bit is ignored.
2. With `la_wheel_learning` on (default), after a drive `config.json` should carry `la_wheel_learned`: a reading rate
   near the game's frame rate, a quantum (0.00058 on the old wheel), a spring, and after some LA time a start force.
3. Whether the parked wheel feels right with hands resting: the hold freezes the push where the wheel stopped; a hand
   moving it more than about 1 mlock brings the servo back within 50 ms.
4. Whether the design ladder steps down on a good wheel in normal LA driving (log line "servo hunted at a still
   target"); if it does, the R3's feel changes, and `la_wheel_learning: false` restores today's servo exactly.
5. The lane keeping assist's pulses: they move the target, so the hold should rarely park during one; check that the
   pulses feel the same.
6. Tests: `python -m pytest tests/la tests/lane_departure tests/invariants tests/test_sdk_own_plugins.py -q` gave 524
   passed and the 3 known failures in `tests/lane_departure/test_side_traffic.py` (unchanged from the handoff); the C
   signal test builds with gcc and prints `ffb signal ok`.
