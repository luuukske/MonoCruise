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
| ACC tracker | Replaced. On a straight road it would publish the nearest in-path parts; the stand-in publishes the trailer rear and the tractor rear of every rig within 150 m (TMP trucks enter the chain rear part first, `core/acc/ACC_ARCHITECTURE.md` §9.8), top 5 by distance (`TRACKER_LEADS`, mirroring the tracker's `PUBLISHED_LEADS`), score pinned at `SCORE_MAX`, kinematics from the rig (the TMP trailer swap). |
| ACC controller | **Real.** `core.longitudinal.acc.AdaptiveCruiseController` wrapping the shipped `AdaptiveCruiseController`, fed through its own `_read_acc_snapshot`. |
| Cruise PID, arbitration | **Real.** `CruiseController` and `CruiseControlThread._arbitrate_named`. Settings the stack reads are pinned to their shipped defaults for the run. |
| AEB | **Real, opt-in.** The headless `AEBThread` from `core/aeb/clip_eval.py`, 30 Hz per client. AEB ships disabled, so scenarios run without it unless asked. The orchestrator's AEB-then-stop disarm is mirrored, plus a driver who taps resume. |
| Standstill hold | **Real.** `HoldController`. |
| Mapper and truck | Modelled: the command is tracked through a dead time and a first-order lag, inside engine power and brake capacity. Capacity (10.85 to 12.58 m/s²) and dead time (0.12 s) are the fitted rigs of `tests/aeb/test_stop_distance_envelope.py`. Cruise braking lags 0.19 to 0.31 s (the gentle-braking fit); while AEB brakes the lag is `tau_slam_s` 0.15 s, the p90 of measured full-pedal slams. |
| Gearbox | Modelled, fitted on the mapper debug log: every truck, the lead included, shifts on the logged schedule and cuts its drive for each upshift. Below, "Gear shifts". |
| TruckersMP | Modelled, calibrated on the clip corpus. Below. |

Each client sees the others through its own TMP link, and a truck collides with the
truck ahead **as its own client draws it**, which is how TMP collides remote
trucks. Trucks do not pass through each other: contact clamps the follower to the
rear it hit. While a truck brakes TMP draws it further forward than it is (below),
so the drawn gap can be a few metres larger than the true one until TMP corrects it.

## The TruckersMP display model (`netcode.py`)

Two layers, each measured on real remote-truck streams.

**Playback clock.** A remote truck is drawn on its own past path, replayed a sync
delay behind (server `syncdelay`, 200 ms on Simulation 1 and 2, plus half of each
side's ping). The clock runs in constant-rate segments of 0.10 to 0.20 s, `1 + err /
0.6 s` plus jitter, and wanders ~1 Hz with clock sync. The jitter falls with speed:
below ~18 m/s the drawn speed is a sawtooth of about ±10 % (90 % of streams at 6 to
14 m/s, 10 % above 22 m/s); the ego truck's own telemetry shows a third of that at
those speeds, so it is TMP, not driveline. Stalls freeze the drawn truck
(byte-identical positions) and end in a catch-up:

* Rate falls with the sender's speed: per minute 2.0, 1.35, 0.9, 0.8, 0.4 at 1.5,
  5.5, 11.5, 18.5, 26 m/s in the mean session. The fall holds inside one stream.
* Length lognormal, median 90 ms, a tail to 1 s. After a stall the stream stays
  fragile for 1.5 s (hazard x5, at most 20 per minute).
* Hard acceleration and pulling away from rest stall every session alike.
* **The receiver's session sets the rate**, not the pair: 83 % of the spread in
  stall rate is between clips, 17 % between trucks in one clip. Sessions are
  lognormal (σ 1.4) around the mean: 38 % of clips show no stall at all, a few
  percent are ten times rougher. A client draws its session once and every truck it
  sees shares it.

**Braking overshoot.** TMP draws a braking truck with a speed that lags the
sender's (0.8 s), so it runs ahead of its playback point and the lead grows
roughly linearly. About 1.8 s after the lead passes 0.3 m TMP corrects it: it holds
the truck still or runs it backwards for ~0.2 s (rewinds dominate hard braking),
then the rest bleeds off over ~1.5 s while the drawn truck follows the sender. The
lead is held within 0.09 s²/m x speed², so a truck that stops sheds its overshoot
on the way down: corpus rewinds below 3 m/s are centimetres. This is where real TMP
rewinds come from: at steady speed they are 50x rarer than in braking, and their
rate rises roughly with deceleration squared.

`LAGGY` is about the 99th-percentile session on a 250 ms ping. `CLEAN` is a perfect
link one delay late, with no overshoot, for A/B.

### Calibration

`python -m tools.acc_platoon --calibrate` measures both sides the same way, and
`tests/acc/test_platoon_calibration.py` holds the model to it. The corpus is every
clip in both stores (local and contributed), every label including untagged and
ignore, deduplicated by clip id: 2041 clips whose replay rebuilt the physics-step
clock, 54 675 TMP streams. The model side is 240 receiver sessions of six trucks in
mixed traffic (`TrafficDriver` on the plant with its gearbox), so every speed and
acceleration bin the corpus fills is filled.

**Only clips on the step clock count.** In 12 % of clips the replay could not rebuild
it and kept wall time; there a repeated game frame reads as every truck stalling at
once (81 % of their stall frames froze every moving truck, 30 % froze the ego as
well). The live radar sees that frame's `simulatedTime` again and skips it as a
sub-frame. Those clips were 92 % of the "rough" sessions and made the earlier
calibration read 7.5 stalls per minute where the step-clock clips show 1.4.

Measured 2026-10-08 (`calibrate.py`, model `NORMAL`):

| Drawn position | corpus | model |
|---|---|---|
| per-frame speed ratio p10 / p90 | 0.905 / 1.099 | 0.925 / 1.087 |
| residual RMS around a 2 s fit, p50 / p90 | 0.134 / 0.344 m | 0.110 / 0.225 m |
| stalls / rewinds per minute above 8 m/s | 1.43 / 0.60 | 1.44 / 0.39 |
| steady stalls per minute at 0.5-3, 3-8, 8-15, 15-22, 22+ m/s | 5.6, 1.7, 0.78, 0.38, 0.16 | 4.2, 1.5, 0.70, 0.35, 0.10 |
| sessions without a stall; per-session rate p90 / p99 (x mean) | 0.59; 5.6 / 25 | 0.62; 5.3 / 17 |
| holds + rewinds per minute braking at 3-22 m/s, > 6, 3.5-6, 2-3.5, 0.8-2 m/s² | 38.9, 13.9, 6.5, 2.5 | 43.5, 29.7, 14.4, 1.8 |
| drawn lead at the correction, 2-3.5, 3.5-6, > 6 m/s² | 1.9, 3.4, 5.8 m | 1.5, 2.4, 4.2 m |
| radar frames 1 / 2 / 3 / 4 physics steps apart | 0.15 / 0.61 / 0.22 / 0.02 | the same, by construction |

The overshoot is read against a quadratic fitted 2 to 3.5 s either side of the
correction. On model streams that reads 2.2 m where the model's own lead is 2.0 m,
so the corpus numbers are real overshoot, and the model stays below them.

What the shipped `Vehicle` chain makes of it, on streams above 8 m/s:

| | corpus | model |
|---|---|---|
| abs(`acc_accel`) on steady stretches, p50 / p90 | 0.26 / 1.21 m/s² | 0.34 / 1.00 m/s² |
| `acc_speed` error on steady stretches, p50 / p90 | 0.18 / 0.59 m/s | 0.17 / 0.47 m/s |
| phantom decel within 2 s of a stall on a steady lead, p50 / p90 | 1.63 / 8.28 m/s² | 1.23 / 4.65 m/s² |
| phantom decel with no artefact, p90 | 2.14 m/s² | 1.28 m/s² |

**Read before trusting a number.** On the ACC chain the model is at or below the
game, so a phantom brake the sim shows is one the game shows at least as often. The
braking overshoot is also below the corpus, by 20 to 30 %. The one place the model is
harsher: holds and rewinds at 2 to 6 m/s² come about twice as often as in the
corpus; fixing that pushed the overshoot further below the corpus, so the test holds
the gap (`BRAKING_FACTOR`) instead of tuning it away.

### Gear shifts (`plant.py`, `shifts.py`)

Fitted on `accel_to_pedals_debug.csv` (1.65 M rows at 10 Hz, 6270 upshifts):

* **AMT** (the 12-, 13- and 14-speed boxes): drive below half for 0.90 / 1.25 / 1.55 s
  (p10 / p50 / p90, 2128 clean upshifts). Torque starts to fall 0.8 s before the
  gear number changes, is gone for ~0.3 s, and is back by ~1.3 s after. The model
  ramps it out over 0.6 s, holds it off 0.3 ± 0.25 s (drawn per shift) and ramps it
  back over 1.2 s, scaling whatever the engine was delivering.
* **Length does not depend on throttle, acceleration, gear or speed** (|corr| 0.06,
  0.14, 0.05). Hard acceleration shifts more often and loses more speed per shift,
  not longer. `test_shift_length_does_not_grow_with_throttle` pins it.
* **Torque-converter automatic** (the log's 6-speed drives): no cut, a dip to ~0.68
  of the pre-shift acceleration. `POWERSHIFT`, not in the default fleet.
* Schedule, 14-speed on its usual 4-6-8-10-11-12-13-14 path, median of 5014 upshifts:
  11.8, 19.7, 27.6, 43.0, 55.3, 69.8, 88.7 km/h, 3 % lower at no throttle and 3 %
  higher at full. Downshifts at 7.6 to 74.2 km/h; asked for 60 % of full power below
  the kickdown speed (67 km/h out of 13th) it kicks down. At 80 km/h a truck is in 13th.

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
| `slow_pull_away` | the convoy stands; the lead pulls away at 0.3 m/s² to 30 km/h |
| `creep` | the convoy stands; the lead inches off at 2 km/h, below where `acc_speed` reads non-zero, and never stops |
| `pull_through` | the convoy rolls at 30 km/h; the lead pulls through to 80 at full power, three upshifts |
| `laggy_client` | holds 80 km/h, follower 5 is on `LAGGY` |
| `blackout_brake` | `hard_brake` while follower 1 gets no update for 1.0 s, the worst pause seen |
| `desync_snap` | holds 80 km/h, drawn 8 m closer to follower 1 for 0.3 s |

## What it found

### 2026-10-08, with the corpus-wide netcode and the gearbox

Gap level 2, ten followers, seeds 1 to 3, ranges over the seeds.

| Scenario | ACC alone | ACC with AEB |
|---|---|---|
| `steady` | 0 to 4 brakes in 40 s; worst speed σ 1.0 to 8.1 km/h, 27.6 km/h behind one 99th-percentile session | |
| `slowdown` | worst hop gain 1.77 to 1.86; up to 6 trucks stop | |
| `hard_brake` | all 10 stop, 3 to 6 collide, truck 1 stops from 30 km/h | 1 to 6 collide, closest -0.43 m; 8 disarmed |
| `emergency_stop` | 5 to 10 collide | 4 to 9 collide; trucks behind locked ones creep up to 0.38 m |
| `queue_stop` (2 m/s²) | 2 to 6 collide, bunched to 0 m; drive-off hops 4.6 s, seed 3 never drives off | |
| `blackout_brake` | | 2 to 6 collide |
| `laggy_client` | no contact, 4 to 6 brakes | |
| `desync_snap` | truck 1 brakes 1.2 to 2.0 m/s², the TTC overlay does not trip | |
| `slow_pull_away` | first follower after 0.85 to 0.97 s, no hop over 1.85 s | |
| `creep` | 7 to 9 re-stops (was 14) | |
| `pull_through` | 1 to 4 brakes, all netcode: none on a clean link | |

Every new contact comes from the braking overshoot: with it switched off
(`dead_reckoning=False`) `queue_stop` and `hard_brake` with AEB have none. The drawn
lead runs metres ahead while it brakes, so ACC and AEB see the braking late, then
the correction lands the truck metres closer in 0.2 s. Gear shifts change little
here: switched off, the same scenarios give the same contacts.

### 2026-09-28, the first netcode model


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

Against the gap level, seed 1, ACC alone, first model:

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

`tests/acc/test_platoon.py` runs fifteen scenarios once per session in spawned
worker processes: about 20 s on eight cores. `tests/acc/test_platoon_gearbox.py`
holds the gearbox to the log constants (the log itself under `needs_clips`).
Bounds come in two kinds. `TARGET_*` is the requirement. `BASELINE_*` is the worst of seeds 1 to 3 at landing where the
stack is short of its target; lower it when ACC improves, never raise it, and once
it reaches its target assert the target instead (a test enforces that ordering).

Held as absolute requirements today: a clean link is calm and gear-shift dips alone
brake nobody; no creep behind a queue that stopped on its own; trucks AEB switched
off stay put; a stopped queue drives off again without the driver (seed 1; on seed
3 a truck that touched the one ahead never does); a desync does not trip the TTC
overlay; no contact behind a laggy client, in a pull-away, a creep or a pull-through.

Re-baselined 2026-10-08, with Lukas's approval, for the recalibrated netcode and the
gearbox. The braking overshoot turned four no-contact requirements into ratchets
with target 0: AEB in the harsh brake, the blackout and the full stop, and ACC alone
in the queue stop; the creep behind AEB-locked trucks is a ratchet too.

On the first netcode model the ratchets bit: restoring the brake release and
landing from before `core/acc/ACC_ARCHITECTURE.md` §13.1 and §13.4 failed three of
them (hop gain 1.87, undershoot 28 km/h, relaunch hop 4.9 s). Not re-measured on
the 2026-10-08 model.

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

Long queues: `Scenario(followers=100, visible_ahead=10, record_every=6)` draws only
the ten trucks ahead each client's 200 m traffic buffer can hold and keeps traces at
10 Hz; 100 trucks then cost about 1.5 s per simulated second.

## Convoys of 100 (2026-10-08)

`--followers 100` runs the same scenarios with 100 followers. A long convoy prints a decile
summary instead of a row per truck. Each run takes about a minute.

```bash
python -m tools.acc_platoon --scenario steady --followers 100 --gap-level 2 --seeds 1,2,3
```

Before and after `ACC_ARCHITECTURE.md` §9.9 (five leads, `ant_kv` 0.8, `ant_tau_s` 0.6), seeds 1 to 5,
levels 2 and 3, stopped / contacts / unprovoked brakes, summed over the seeds. Measured on the
earlier netcode model, before the step-clock recalibration and the gearbox:

| scenario | level | before | after |
|---|---|---|---|
| steady | 2 | 33 / 10 / 217 | 0 / 0 / 243 |
| steady | 3 | 0 / 0 / 234 | 0 / 0 / 232 |
| slowdown | 2 | 15 / 4 / 175 | 0 / 0 / 194 |
| slowdown | 3 | 0 / 0 / 183 | 0 / 0 / 183 |
| hard_brake | 2 | 62 / 47 / 180 | 37 / 0 / 190 |
| hard_brake | 3 | 36 / 0 / 181 | 0 / 0 / 180 |
| stop_and_go | 2 | 237 / 30 / 169 | 13 / 0 / 208 |
| stop_and_go | 3 | 31 / 0 / 208 | 0 / 0 / 208 |

Level 1 (seeds 1 and 2): stopped 668 to 381, contacts 431 to 7, unprovoked brakes 348 to 149.
What is left: a hard brake at level 2 still stops trucks at the back, and level 1 still jams. The
10-follower ratchets in `tests/acc/test_platoon.py` were lowered to the new worst-of-seeds values.

### Brake release 0.20 s (2026-10-08)

Release chase `J_RELEASE_TAU_S` 0.30 s → 0.20 s (`ACC_ARCHITECTURE.md` §13.1). Seed 1, 100 followers,
stopped / contacts / unprovoked brakes, and the steady convoy's minimum speed, on the earlier
netcode model. 0.15 s is shown because it was measured too and not shipped:

| scenario | level | 0.30 s | 0.20 s (shipped) | 0.15 s |
|---|---|---|---|---|
| hard_brake | 2 | 7 / 0 / 28 | 4 / 0 / 27 | 2 / 0 / 29 |
| hard_brake | 3 | 0 / 0 / 25 | 0 / 0 / 28 | 0 / 0 / 27 |
| hard_brake, no anticipation (`ma_max_leads` 1) | 2 | 43 / 34 / 31 | 26 / 19 / 28 | 18 / 15 / 29 |
| hard_brake, no anticipation | 3 | 9 / 8 / 33 | 8 / 6 / 36 | 8 / 6 / 33 |
| steady, minimum speed, unprovoked | 2 | 50.4 km/h, 44 | 53.2 km/h, 45 | 53.9 km/h, 49 |
| steady, minimum speed, unprovoked | 3 | 51.0 km/h, 39 | 53.1 km/h, 41 | 54.3 km/h, 40 |

Level 3 hard brake: dip gain 1.45 → 1.38, hop gain 1.47 → 1.40, slowest truck 5.1 → 8.6 km/h.
0.15 s recovered more but cost more under a lead that pumps the brake (`ACC_ARCHITECTURE.md` §13.1)
and at level 1, which is why 0.20 s shipped.

### Both on the step-clock netcode and gearbox (2026-10-08)

Ten followers, level 2, worst of seeds 1 to 3, the ACC before §9.9 against five leads with the
0.20 s release. This is what the ratchets in `tests/acc/test_platoon.py` were lowered to:

| metric | before | after |
|---|---|---|
| harsh brake, contacts without AEB | 6 | 1 |
| harsh brake with AEB: contacts / least room / ACC disarmed | 6 / −0.44 m / 8 | 0 / 2.29 m / 3 |
| blackout during a harsh brake, contacts with AEB | 6 | 0 |
| full-pedal stop, contacts without / with AEB | 10 / 9 | 5 / 2 |
| queue stop: contacts / least stop gap | 6 / 0.0 m | 0 / 3.35 m |
| slowdown: hop gain / trucks stopped | 1.86 / 6 | 1.78 / 0 |
| steady: worst speed spread / unprovoked brakes | 27.6 km/h / 4 | 18.0 km/h / 3 |
| laggy client: unprovoked brakes / slowest truck | 6 / 0.0 km/h | 8 / 35.7 km/h |

The laggy client brakes more often but far less deeply: one deep brake that stopped a truck on
seed 3 became several shallow ones. On seed 2 of the queue stop the back of the queue is still
rolling when the lead goes and stops again, so its relaunch hop reads 13.6 s; seeds 1 and 3 read
2.0 and 2.2 s.

## Metrics (`metrics.py`)

| Metric | Meaning |
|---|---|
| dip gain | a truck's speed dip below the cruise speed over the lead truck's dip; above 1 is amplification |
| hop gain | a truck's dip over the dip of the truck directly ahead; string stability wants it at or below 1 |
| unprovoked brakes | brake episodes (command below -1 m/s²) a truck starts while the truck ahead has not braked for 3 s |
| contacts | followers whose drawn gap reached zero |
| stopped | followers that came to a standstill |
| re-stops | times a truck came back to rest after rolling off; behind a lead that never stops, each one is a lurch |
| disarmed | followers whose ACC an AEB stop switched off |

## Limits

Straight flat road, one lane, no lateral motion and no cut-ins. The gearbox has no
manual (H-shifter) drivers: the log only holds AMT and a torque-converter box. Ego
pose is exact: clips also show a one-step ego pairing slip on a few percent of
frames, not modelled.
The tracker is bypassed, so lock latency and score flicker are absent. The mapper is
a lag model, not `AccelToPedals`. The lead is scripted and perfectly smooth unless
`lead_wobble_ms2` is set. Collision is contact only, with no crash physics. The
AEB-disarm resume is a model of a driver, not of the button FSM.
