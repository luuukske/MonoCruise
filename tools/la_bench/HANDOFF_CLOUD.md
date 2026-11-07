# LA wheel servo: cloud handoff

For a long cloud run. Goal: the LA/LKA servo and the wheel check work on any wheel, with worse-quality wheels as the main target. Everything here is simulated until Lukas runs the check on a real wheel.

## Read first

1. `AGENTS.md`: repo rules. The hard ones are repeated below.
2. `core/la/README.md`: LA design, calibration, old-wheel notes.
3. `plugins/ffb/README.md`: game-side plugin (layout 5).
4. This file.

## Hard rules for this run

- Work on a new branch, e.g. `la-cloud-worse-wheels`, cut from the handoff commit. Never push to `stable`. Push only with Lukas's OK.
- No em dash anywhere: code, comments, docs, commit messages. Code comments max 2 lines.
- No absolute paths, usernames or machine names in committed files.
- Never weaken a test, safety check or watchdog. Never raise a ratchet baseline. Never change a cap (`la_servo_cap`, plugin servo caps, slew limits) or an ACCEPT threshold without Lukas: cap changes need approval.
- Plugin C changes: keep `plugins/ffb/ffb_signal.h` and `tests/la/servo_sim.py` constants in parity (`tests/la/test_sticky_wheel.py` checks this). The DLL needs MSVC (see `plugins/ffb/README.md`). If the box cannot build it, leave `core/sdk_installer/own/monocruise_ffb.dll` as is and say so. Its git blob SHA is in `core/sdk_installer/own.py` and must match the file.
- Commit style: terse, no filler. End commit messages with `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>` (keep this line as given).

## Where things are

| File | What |
| --- | --- |
| `plugins/ffb/ffb_signal.h` | Servo law: `ffb_servo_ff`, `ffb_servo_stuck`, breakaway ramp, rest push gate. Constants `FFB_STUCK_FROM 1.5`, `FFB_BREAK_RAMP_S 0.3`, `FFB_BREAK_BAND 0.003`, `FFB_BREAK_HOLD 0.0015` |
| `plugins/ffb/ffb_signal_test.c` | C tests. Prints `ffb signal ok` |
| `tests/la/servo_sim.py` | Python mirror of the plugin law |
| `tools/la_bench/plant.py` | `WheelParams`, `SimWheel`, `SIM_WHEELS`, `replay`, `fit`, `estimate_spring` |
| `tools/la_bench/probe.py` | `slide_probe` (start force), `damping_probe` (open-loop holds, least-squares fit of c, k, f), `fit_slides` |
| `tools/la_bench/check.py` | `run_check`, `match_verdicts`, `accepted()`, ladder constants |
| `tools/la_bench/design.py` | Design candidates, `BREAKAWAY_SHARES`, `breakaway_top`, `push_ahead_of` |
| `tools/la_bench/bench.py` | `Bench`, `run_stage`, `reading_trouble`, `RunInvalid`, other-writer guard |
| `tools/la_bench/speed.py` | Servo speed bench |
| `tools/la_servo_bench.py` | CLI: `capture`, `check`, `report`, `fit`, `predict`, `speed` |
| `tests/la/test_sticky_wheel.py`, `test_bench_guards.py`, `test_wheel_check.py` | Sticky-wheel, guard and check tests |

Recordings go to `tools/la_bench_out/` (gitignored, local only). Real-wheel data is not in the repo. If you need it, ask Lukas to attach the `la_check_*.json` files.

## State at handoff

- Tests: `python -m pytest tests/la tests/lane_departure tests/invariants tests/test_sdk_own_plugins.py -q`: 481 pass, 3 fail. The 3 failures are in `tests/lane_departure/test_side_traffic.py` (`test_an_oncoming_car_passing_beside_us_is_beside_until_it_is_about_to_leave`, `test_a_car_we_pass_about_to_drop_off_the_trailer_no_longer_counts`, `test_a_car_passing_us_about_to_clear_the_bumper_no_longer_counts`). They also fail on a clean checkout of the same base, so this work did not cause them. They come from side-traffic code (commit 6838d51). Do not edit those assertions to pass. Report them.
- `ruff check .`: clean. C signal test: `ffb signal ok`.
- Plugin: breakaway push ramps up while the wheel stands, holds off within 1.5 mlock of the target, climbs fully from 3 mlock. Layout 5. Installed DLL SHA: `c426d15c078f22c9361e7cf2c194346a0016c374`.
- Check: damping probe, held spring and damping (`damping_held`), breakaway ladder that uses the start spread, reading guard, other-writer guard, `ACCEPT_SHORT 0.6` (a step fails only when it stops more than 60 % short).

## The old wheel (ground truth for "worse")

Thrustmaster Advanced Mode Racer, cheap, old. Measured on the real wheel:

- Creep slide about 305. Breakaway from rest 311 to 526, varying trial to trial at one spot (spread about +/-20 % around 436).
- Angle quantum 0.00058 lock. Game reads the angle at 60 Hz. Light rim (about 30). Plugin-to-rim delay 42 to 50 ms.
- Game spring 533 to 692 (model uses 650).
- Damping probe at 21:45: c 424, k 330, f 435. With the game spring removed, fast slides run 500 to 545 at 0.2 to 0.46 lock/s: friction rises with speed.
- Near the target the wheel restarts on about 250 (dwell stiction or backlash). From rest it needs 401 to 480. The model has neither.

Runs that shaped the current code:

- 18:01: invalid (PC stalls, MonoCruise interference). Led to the reading and other-writer guards.
- 19:16: fixed push did not start the wheel. Led to the ramp, the damping probe and the held spring.
- 21:45: ramp got the big move going. Rest push flipped +/-250 within 1.5 mlock. The wheel hunted +/-1.5 mlock at about 4 Hz, the buzz guard cut the servo, and the partial gate stalled it 4.4 mlock short. Current fix: no push within 1.5 mlock, full climb from 3 mlock. Not yet tried on the wheel.

## Model gaps (main work)

1. No dwell-time stiction or lower restart force near the target. Model restarts at its one breakaway everywhere.
2. No backlash or play. The hunt may be backlash, not stiction. Test both.
3. No friction that rises with speed (`stick` is constant in `plant.py`).
4. No reading stalls or jitter. `reading_trouble` exists in the bench but the model never produces stalls.
5. Quantum: the old wheel is 0.00058. Coarse-encoder wheels could be 0.002 to 0.005. Test that range.

## Worse wheels to simulate

Add new `SIM_WHEELS` entries. Keep the existing five (`r3`, `bare`, `geared`, `sticky`, `old`) as regression. New features must be optional `WheelParams` fields whose defaults leave every existing wheel's replay bit-identical.

- `cheap_a`: old-like, plus dwell stiction (restart near target about half the breakaway, within 3 mlock) and 1 to 2 mlock backlash.
- `cheap_b`: quantum 0.004, reading 30 Hz, delay 0.08, breakaway 600, spread 0.4.
- `cheap_c`: spring 300 (weak centring), damping 900, stick 350, speed-dependent friction.
- `worst`: all of the above together.

Use several seeds for coverage. Keep `SPREAD_SEED` as is in the tests.

## Tasks, in order

1. Baseline: run the test set above. If the numbers differ from "State at handoff", stop and report.
2. Add the model features as optional fields with bit-identical defaults. Prove the defaults change nothing: the existing wheels' replays must match before and after.
3. Add the worse-wheel entries. Run `run_check` end to end on each sim wheel. Record per wheel: verdict, chosen design, overshoot, final error, sign flips at target in the last 2 s, whether the breakaway ramp fired, pass or fail.
4. Find where designs or verdicts fail on worse wheels. Likely: the design is too aggressive for backlash, and the breakaway gate hunts with backlash. Fix in `design.py` and `check.py` first, no C change. Change the C law only if a simulated result shows the C law is the cause. Then update `servo_sim.py` and the parity tests in the same commit.
5. Regression: all five existing sim wheels must still verify. Do not lower `accepted()` thresholds to pass.
6. Write `tools/la_bench/HANDOFF_RESULTS.md`: a before and after table per wheel, what changed, and what only the real wheel can confirm.
7. Commit on the branch in small commits. Do not push.

## Done means

- Every simulated worse wheel verifies, with no more than one sign flip per 2 s at the target. The five existing wheels still verify.
- Tests pass, except the 3 side-traffic failures above, which are named in the results file.
- The results file lists what needs the real wheel.

## Not verifiable in the cloud

- Any real wheel. Every number from this run is simulated.
- Game behaviour, MonoCruise loading the plugin, the real DLL build (unless the box builds it).
- To test on the wheel, Lukas: closes the game, starts MonoCruise once and closes it, starts the game, closes Roblox and overlays, then runs `python tools/la_servo_bench.py check --note "<wheel and settings>"`.

## Stop and ask Lukas when

- A fix needs a cap change, an ACCEPT threshold change, or a test assertion change.
- A fix needs anything that sends data off the machine or adds telemetry. LA is offline and stays offline.
- The plugin C changes and the DLL cannot be rebuilt.
- Anything would be pushed.
