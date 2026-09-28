# tools/acc_platoon

A closed-loop convoy: one scripted lead truck and ten MonoCruise clients behind it
on TruckersMP, every client following at ACC gap level 2. It exists to answer one
question: **does ACC absorb a disturbance, or amplify it into a phantom jam, and
is it still safe while it does?** `tests/acc/test_platoon.py` is built on it.

Nothing here ships or is imported by the app.

## What is real and what is modelled

| Layer | In the sim |
|---|---|
| Radar filter chain | **Real.** One `Vehicle` per remote rig per client, advanced with `update_from_last` on the physics-step clock, exactly as `TrafficReader` does: lag freeze, position mismatch, pose-jump guard, both speed chains. |
| ACC tracker | Replaced. On a straight road it would publish the nearest in-path parts; the stand-in publishes the trailer rear and the tractor rear of every rig within 150 m (TMP trucks enter the chain rear part first, `core/acc/ACC_ARCHITECTURE.md` §9.8), top 3 by distance, score pinned at `SCORE_MAX`, kinematics from the rig (the TMP trailer swap). |
| ACC controller | **Real.** `core.longitudinal.acc.AdaptiveCruiseController` wrapping the shipped `AdaptiveCruiseController`, fed through its own `_read_acc_snapshot`. |
| Cruise PID, arbitration | **Real.** `CruiseController` and `CruiseControlThread._arbitrate_named`. Settings the stack reads are pinned to their shipped defaults for the run. |
| AEB | **Real, opt-in.** The headless `AEBThread` from `core/aeb/clip_eval.py`, 30 Hz per client. AEB ships disabled, so scenarios run without it unless asked. The orchestrator's AEB-then-stop disarm is mirrored, plus a driver who taps resume. |
| Standstill hold | **Real.** `HoldController`. |
| Mapper and truck | Modelled: the command is tracked through a dead time and a first-order lag, inside engine power and brake capacity. Capacity (10.85 to 12.58 m/s²), dead time (0.12 s) and brake lag (0.19 to 0.31 s) are the fitted rigs of `tests/aeb/test_stop_distance_envelope.py`. |
| TruckersMP | Modelled, calibrated on the clip corpus. Below. |

Each client sees the others through its own TMP link, and a truck collides with the
truck ahead **as its own client draws it**, which is how TMP collides remote
trucks. The drawn gap is never larger than the true one while moving forward, so
it is the conservative collision measure. Trucks do not pass through each other:
contact clamps the follower to the rear it hit.

## The TruckersMP display model (`netcode.py`)

Real TMP streams show no position noise. A remote truck is drawn on its **own past
path**, replayed by a playback clock that runs a sync delay behind: constant-speed
segments of 0.1 to 0.2 s, pauses (byte-identical positions) followed by 1.2 to
1.3x catch-up, short backward runs, and a ~1 Hz wander from clock sync. So the
model is a playback clock on the sender's true trajectory:

* **Delay**: server `syncdelay` (TMP API: 200 ms on Simulation 1 and 2, 100 to
  350 ms across servers) plus half of each side's ping.
* **Segments**: every 0.10 to 0.20 s the clock picks a rate, `1 + err / 0.6 s`
  plus noise, where `err` is how far it is behind its target. The target moves
  every ~1 s by a clock-sync error.
* **Pauses**: Poisson, lognormal length (median 30 ms), both sides' rates added.
  Real pauses are mostly one or two frames.
* **Rewinds**: 0.15 to 0.25 s at -1 to -20 % of real time.
* **Pair roughness**: a pair is clean with probability `clean_share²`, otherwise
  every artefact is scaled by U(0.4, 1.8). The corpus is that bimodal: a quarter of
  streams are nearly perfect, the rest range from moderate to very rough.

`LAGGY` is a deliberately bad connection (250 ms ping, heavy pauses and rewinds),
harsher than the corpus tail. `CLEAN` is a perfect link one delay late, for A/B.

### Calibration

`python -m tools.acc_platoon --calibrate` measures both sides the same way, and
`tests/acc/test_platoon_calibration.py` holds the model to it. Measured 2026-09-28
against every clip in the local store (about 6400 TMP streams, 18 h of moving traffic),
model `NORMAL` as the convoy tests draw it:

| Raw drawn position | corpus | model |
|---|---|---|
| per-frame speed ratio p10 / p90 | 0.899 / 1.109 | 0.930 / 1.102 |
| residual RMS around a 2 s fit, p50 / p90 | 0.138 / 0.337 m | 0.172 / 0.342 m |
| pauses per minute, length p50 / p90 | 7.5, 0.070 / 0.10 s | 7.7, 0.067 / 0.12 s |
| rewinds per minute | 0.89 | 0.77 |

**Read the whole store.** Clips are bursty by session: the first calibration used a
150-clip sample and measured 2.9 pauses per minute, a 400-clip sample measured 8.6,
and quarters of the store range 5.4 to 9.8.

What the shipped `Vehicle` chain makes of an artefact on a steady lead, which is
the part that decides whether ACC brakes:

| Response within 2 s | corpus | model |
|---|---|---|
| ACC chain, phantom decel after a pause, p50 / p90 | 0.55 / 2.25 m/s² | 0.90 / 1.79 m/s² |
| ACC chain, phantom decel after a rewind, p50 | 3.26 m/s² | 1.72 m/s² |
| ACC chain, no artefact, p90 | 1.86 m/s² | 1.31 m/s² |
| ACC chain, \|`acc_accel`\| on steady stretches, p90 | 1.15 m/s² | 0.66 m/s² |
| AEB chain, speed dip after a pause, p50 / p90 | 0.92 / 4.64 m/s | 2.11 / 4.43 m/s |
| AEB chain, speed dip after a rewind, p50 | 2.97 m/s | 5.72 m/s |

**Read the second table before trusting a number.** On the ACC chain the model is
at or below reality at p90, so a phantom brake the sim shows is one the game will
show at least as often. On AEB's short window the model's rewinds bite about twice
as hard as real ones, so phantom AEB counts are an upper bound and no test asserts
on them. Rewinds cannot match both chains with one shape: real ones hit the long
window harder and the short window softer than any backward run tried.

## Scenarios (`scenarios.py`)

All at gap level 2, 80 km/h, set speed 90 km/h, a mixed fleet, ten followers.

| Name | What the lead does |
|---|---|
| `steady` | holds 80 km/h: whatever the followers do is the netcode |
| `human_lead` | holds 80 km/h on a keyboard, throttle and lift in bursts |
| `slowdown` | 80 to 60 km/h at 2 m/s² |
| `hard_brake` | 80 to 30 km/h at 6.5 m/s², then holds |
| `emergency_stop` | full-pedal stop from 80 km/h (8 m/s²), stands 15 s, drives off |
| `stationary_lock` | the emergency stop with AEB and nobody tapping resume |
| `queue_stop` | 50 km/h to a standstill at 2 m/s², stands 20 s, drives off |
| `stop_and_go` | 60 and 25 km/h in turns |
| `laggy_client` | holds 80 km/h, follower 5 is on `LAGGY` |
| `blackout_brake` | `hard_brake` while follower 1 gets no update for 1.0 s, the worst pause seen |
| `desync_snap` | holds 80 km/h, drawn 8 m closer to follower 1 for 0.3 s |

## What it found (2026-09-28)

Gap level 2, ten followers, seeds 1 to 3. AEB ships disabled, so "ACC alone" is
what a default install does.

| Scenario | ACC alone | ACC with AEB |
|---|---|---|
| `steady` | 4 to 11 brakes in 40 s with the lead never braking; speed σ from < 0.1 km/h at truck 1 to 1.8 to 10 km/h at the back | not asserted (upper bound, above) |
| `slowdown` | worst hop gain 1.29 to 1.34; up to 5 trucks stop; contacts at trucks 9 and 10 in two seeds | |
| `hard_brake` | 9 of 10 stop, 7 or 8 collide, truck 1 bottoms out 15 to 18 km/h under the lead | no contact (≥ 1.1 m); 8 or 9 stop and AEB disarms ACC on each |
| `emergency_stop` | 8 to 10 collide: an 8 m/s² stop is past ACC's own authority | no contact (0.1 to 1.7 m); 9 or 10 disarmed, and they stay put until each driver taps resume |
| `queue_stop` | no contact, no creep, but trucks bunch up to 1.1 to 1.4 m (target 5 m) and drive off 2 to 4 s apart | |
| `blackout_brake` | | no contact (≥ 0.7 m) |
| `laggy_client` | no contact, closest 2.4 cm (seed 1) | |
| `desync_snap` | truck 1 brakes 1.1 to 1.9 m/s², the TTC overlay does not trip | |

Against the gap level, seed 1, ACC alone:

| | level 2 | level 3 | level 4 |
|---|---|---|---|
| `steady`: brakes / speed σ at the back | 11 / 10.1 km/h | 11 / 4.9 km/h | 8 / 2.9 km/h |
| `slowdown`: truck 10's dip over the lead's | 2.38 | 0.73 | 0.49 |
| `hard_brake`: trucks stopped / collided | 9 / 7 | 7 / 0 | 0 / 0 |

Two separate mechanisms, and the clean link separates them:

* **Phantom brakes are netcode.** A pause or a rewind drops `acc_speed` and reads as
  a lead brake (the calibration table above). On a clean link there are none, at
  any level.
* **Amplification is perception lag.** TMP speed is a least-squares fit over about
  1.3 s of positions, so `acc_speed` runs ~0.9 s behind, and the truck adds ~0.45 s.
  With a delay τ ≈ 1.4 s the textbook `h ≥ 2τ` asks for ~2.8 s of headway; level 2
  is 1.1 s. A clean link amplifies as much as the TMP one does.

## The tests

`tests/acc/test_platoon.py` runs eleven scenarios once per session in spawned
worker processes: about 11 s on eight cores. Bounds come in two kinds. `TARGET_*`
is the requirement. `BASELINE_*` is the worst of seeds 1 to 3 at landing where the
stack is short of its target; lower it when ACC improves, never raise it, and once
it reaches its target assert the target instead (a test enforces that ordering).

Held as absolute requirements today: a clean link is calm; with AEB, no contact in
the harsh brake, the blackout and the full stop; no creep at a standstill; trucks
AEB switched off stay put; a stopped queue drives off again without the driver; a
desync does not trip the TTC overlay; no contact behind a laggy client.

The ratchets are known to bite: restoring the brake release and landing from
before `core/acc/ACC_ARCHITECTURE.md` §13.1 and §13.4 fails three of them (hop
gain 1.87, undershoot 28 km/h, relaunch hop 4.9 s).

## Running

```bash
python -m tools.acc_platoon --scenario hard_brake --seeds 1,2,3
```

```bash
python -m tools.acc_platoon --scenario all --aeb --json
```

```bash
python -m tools.acc_platoon --scenario slowdown --clean --gap-level 3
```

`--clean` removes every netcode artefact but keeps the delay, which separates what
the netcode does from what the speed estimate's own lag does. A run costs about
0.13 s per simulated second without AEB and twice that with it.

## Metrics (`metrics.py`)

| Metric | Meaning |
|---|---|
| dip gain | a truck's speed dip below the cruise speed over the lead truck's dip; above 1 is amplification |
| hop gain | a truck's dip over the dip of the truck directly ahead; string stability wants it at or below 1 |
| unprovoked brakes | brake episodes (command below -1 m/s²) a truck starts while the truck ahead has not braked for 3 s |
| contacts | followers whose drawn gap reached zero |
| stopped | followers that came to a standstill |
| disarmed | followers whose ACC an AEB stop switched off |

## Limits

Straight flat road, one lane, no gear shifts, no lateral motion and no cut-ins.
The tracker is bypassed, so lock latency and score flicker are absent. The mapper is
a lag model, not `AccelToPedals`. The lead is scripted and perfectly smooth unless
`lead_wobble_ms2` is set. Collision is contact only, with no crash physics. The
AEB-disarm resume is a model of a driver, not of the button FSM.
