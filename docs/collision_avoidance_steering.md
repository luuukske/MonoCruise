> **EXPLORATION, not a plan of record.** Nothing here is implemented. It maps what MonoCruise
> already has (AEB, radar, map, the LA/LKA steering actuator, the clip corpus) onto a future
> emergency steering layer and states where the data runs out. Numbers are quoted from the
> module READMEs and code as of 2026-11-04; anything inferred is marked (inferred).

# Collision avoidance by steering: can MonoCruise swerve for an absent driver?

## 0. Framing

The question: with the corpus and the live data MonoCruise already has, could it avoid a
collision by steering through the FFB plugin, assuming nobody (or nobody attentive) is at the
wheel, and how would it (1) pick a course among many moving vehicles and (2) choose between
braking and steering with the least annoyance and the most safety.

Two framing decisions fall out of the existing docs before any design:

- This is an **emergency manoeuvre layer** (automatic emergency steering, AES), not lane
  following and not self-driving. `core/la/README.md` states full self-driving is ETS2LA's job,
  LA keeps the lane only, and the lane departure work has standing decisions against pushing
  the wheel in ways the driver did not ask for. An AES tier that fires rarely and hands back
  on any grip fits inside that stance; a planner that drives the truck does not.
- **Braking stays the default.** The AEB chain is tuned on about 2600 clips with a measured
  stop-distance envelope. Steering adds a second escape only where braking alone fails or
  where a small offset is clearly cheaper. Every rule below is written so that a wrong steer
  is treated as far worse than a wrong brake.

## 1. What already exists

### 1.1 Perception (radar, `core/radar/README.md`)

- Up to 24 nearest bodies from the ETS2LA traffic buffer plus the parked buffer, 30 Hz radar
  over 60 Hz physics, ego pose paired to the same game frame (`scs_pose.py`).
- Per body: raw world position (the lateral coordinate is **not** smoothed), `_smooth_yaw`
  (EMA alpha 0.5), speed from a 20-sample (about 1.3 s) fit that reports speed near the window
  middle, acceleration from a 1.5 s slope, a 25-sample position history (about 1.7 s), a circle
  fit curvature, crash latch, TMP lag freeze and ramp. TMP drawn position sits 0.13 m p50 and
  0.34 m p90 off a 2 s fit and runs metres ahead of truth while the sender brakes (1.9 to
  5.8 m depending on decel).
- Not available for other vehicles: lane id, blinkers, brake lights, wheel angle, intent.
  Angular velocity is 0 for remote players.
- Ego: SDK wheel layout (`ego_geometry.py`, path origin at the grounded non-steered axle
  group, half width 1.265 m on the reference rig), steer-led curvature (`EgoPathModel`, gain
  0.09 to 0.24, lag-free: steer to curvature best shift 0 frames over 437 clips).
- Published "cannot hit" sets a planner must consume as ids and never re-derive:
  `off_surface_ids`, `ncz_ids` (TMP no-collision zones), and AEB's `OverlapGhostHold` set.

### 1.2 Prediction primitives

- `ArcPath`: constant curvature, horizon 2.5 to 3.0 s, clearance search to 5.0 s.
  `heading_at_dist` mirrors the capsule heading on curved arcs by convention; fixing it alone
  moved the corpus cost 256 to 630, so a steering planner must orient bodies its own way (as
  `TurnCompletionFilter` does) and not call that method for capsule orientation.
- Map-leaning path (`core/road_layout`, `map_path.py`): leaves the vehicle on its arc and
  smoothsteps onto its lane over 2.5 s of travel, trust zero past 2.4 m offset or 16 deg of
  heading. Measured lateral error at 3 s (`tools/map_target_eval.py`, 433 clips):

  | class | lane path p50 | lane path > 1.75 m off | plain arc > 1.75 m off |
  |---|---|---|---|
  | AI, straight | 0.02 m | 3.6 % | 11.0 % |
  | AI, bend | 0.07 m | 4.1 % | 62.2 % |
  | TMP, straight | 0.26 m | 6.9 % | 23.5 % |
  | TMP, bend | 0.56 m | 13.0 % | 81.2 % |

  There is no table past 3 s. A plan that depends on where another body is at 4 or 5 s is
  outside the measured envelope.
- ACC anticipation chain (`ACC_ARCHITECTURE.md` section 9): up to five in-lane leads coupled
  by time gap. Longitudinal only. Two lessons carry over: never drop a body that sits beside
  its predecessor (on the corpus those were mostly stopped cars the lead swerved round), and
  never skip trailers (trucks enter the chain rear first).

### 1.3 AEB (`core/aeb/README.md`)

- Clearance demand (`clearance.py`): per target, the minimum constant decel that keeps the
  bumper behind every body ego would share a corridor with, with `stop_buffer_response_s`
  (0.25 s solo, 0.30 s with trailer) applied as a time lag. Aggregated by max. Engage at 0.85
  of capacity or TTB under 0.5 s. From engagement to full pedal: 0.20 s median build-up,
  0.24 s p90.
- `EgoEvasionFilter`: the only existing "can ego steer round it" test. Two constant-curvature
  escape arcs at `delta_kappa = min(0.08 g / v^2, 0.016)`; a body either arc misses at margin
  0 is dropped, with bypasses for latched ids, in-lane closers, moving head-on targets.
- Avoidability gate: a drop that rests on a guess about ego's path stands only until that
  body's demand reaches the braking deadline. The measured reason this matters here: on
  `cb0154bb` and `ddf9b4fd` the body was avoidable by plain unwinding until 4.31 s and 4.24 s
  while demand passed full capacity at 4.25 s and 4.04 s. **The window in which steering still
  works but braking no longer does was 0.1 to 0.3 s wide.** A steer-limit gate built on a
  constant-curvature fan with the corpus grip envelope (p99 of `v^2 |kappa|`: 7.9 m/s^2 at
  6.5 m/s to 11.9 m/s^2 at 30 m/s) was measured and rejected: it under-counted
  turn-then-straighten escapes in yards and, with a 0.35 g budget on curvature change,
  overrode three last-second swerves the driver completed.
- `braking_worsens`: a braked hit whose closing speed exceeds the unbraked one by 0.5 m/s,
  or a faster vehicle behind. This is AEB's existing brake-versus-leave-it split.
- Corpus: schema 6 clips with raw buffers, ego pose, `userSteer` (the game steer scalar),
  blinkers, pedals, AEB decisions and suppression reasons. Labels are human only. Replay is
  open loop on the recorded trajectory; the pedal never moves the truck.

### 1.4 Steering actuator (`core/la/README.md`, `plugins/ffb/README.md`)

- `LaneDepartureThread` (20 Hz) writes `Local\MonoCruiseFfbCmd` layout 4: target wheel angle
  (lock, -1 to 1), rate, kp/ki/kd, cap, direction, cancel, `servo_only`. The plugin runs a
  damped PID at the FFB rate on top of the game's own force, target integrated at the sent
  rate for up to 0.15 s so 20 Hz updates are not steps.
- Caps the plugin enforces whatever MonoCruise asks: servo 3000, assist 4000, damping 1500,
  cancel 6000 of a 10000 full scale. The model puts 3000 at about 5 m/s^2 of cornering
  against the game's centring (about 590 units per m/s^2). **Past the cap the wheel stops
  and the truck runs wide.** A driver can always overpower it; that is by design.
- Steering gain `g_s` 0.184 to 0.187 1/m per lock, learned; servo lag `WheelActuator.lag_s`
  0.15 s at 80 to 90 km/h, 0.25 s at 100 to 130 km/h. Target walk capped at 0.2 lock/s
  (0.04 held a target through a prefab kink and hit a pole). The truck's path follows the
  wheel within a tick; correction to sideways motion took 0.25 to 0.3 s on drive 8.
- `path.WantedPath` (`offset_m`, `kappa`, `source`) is the hook the README reserves "so a
  later collision avoidance can hand in a wanted travel path". It has no heading, no time
  axis, `path.kappa` is not previewed a lag ahead, and stop-before-the-line and the lane
  bias still act on top of it. An evasive source needs that type extended (section 5).
- Guards with incidents behind them: ring guard, plugin buzz guard, weave guard, hands-off
  (15 s), override/insist (0.3 s still and 0.015 lock off target), `hand_push` (freeze at
  40 % of the pulse's push, give from 50 %, gone at 80 %), geometry hold, bend limit, route
  gates (branch, merge, junction lane, short route), AEB active and brake over 0.7 mute LA.
- Side awareness exists but does not steer: `side_traffic.py` (beside within 1 m behind to
  2 m ahead of the rig, same-way traffic within 3.0 s, oncoming within 2.5 s, crossers under
  `parallel_cos` 0.5) and `edges.beyond` (what lies 4.5 m past each line: `lane`, `oncoming`,
  `edge`, `unknown`). The live map knows carriageway lanes only: **no shoulders, barriers,
  signs or props** (the `tools/map_obstacle_*` probes parse them offline and are not on the
  map thread).

## 2. Question 1: estimating the best course among many moving vehicles

### 2.1 Represent the problem as space-time occupancy, not as paths to dodge

AEB already does the right primitive per target: `occupancy_profile` samples a body's capsule
along its predicted path and asks when it occupies ego's corridor. Steering generalises that
from one corridor to a small set of candidate corridors. For each candidate ego manoeuvre
\(m\) and each body \(j\) the question is "does ego's swept body intersect body \(j\)'s
predicted swept body, with uncertainty, inside the horizon". Everything else is cost.

Inputs per tick (30 Hz, same radar frame AEB consumes):

- Ego: pose, speed, steer, `EgoPathModel` curvature, body from `ego_geometry`, trailer body
  from `trailer_bodies_from_sdk` (AEB is tractor only today; a swerve must sweep the trailer).
- Bodies: every `Vehicle` and `trailer_vehicles` member not in the three exclusion sets, with
  the TMP rig link so a rig is swept as a unit.
- Per body prediction: the map-leaning path where the snapshot is fresh and trust is high,
  the plain arc otherwise, each with a lateral inflation that grows with time taken from the
  table in section 1.2 (AI about 0.1 m at 3 s, TMP 0.6 m and a 13 % chance of being a lane
  off; a crude first model is a per-class sigma that reaches those p50 values at 3 s and
  treats out-rate as a hard probability floor).
- Map: ego's `LaneRoute`, neighbour lanes and `edges.beyond` per side, `branch_s`, `merge_s`.

### 2.2 Candidate manoeuvre set: a small library, not a search

A continuous optimiser over trajectories is the wrong tool here: 30 Hz, Python, 24 bodies,
and the thing that failed on the corpus was a fan of constant-curvature arcs because real
escapes are turn-then-straighten. The practical set is a handful of **lateral offset
profiles** with a recoverable end state:

1. `stay`: hold lane, brake per AEB (today's behaviour, always a candidate).
2. `nudge_L` / `nudge_R`: in-lane offset of up to about 1.0 m (stay inside the lane lines
   less ego half width), bang-bang lateral accel at 1.5 to 2.5 m/s^2, ending parallel. Buys
   clearance past a body that only partly overlaps the corridor (a car half on the shoulder,
   a trailer corner, a cut-in that has not finished).
3. `shift_L` / `shift_R`: full move to the adjacent **same-direction** lane, lateral accel
   2 to 4 m/s^2 (servo cap sits near 5), ending centred and parallel.
4. Each of 2 and 3 combined with brake levels {0, AEB demand, full}.

Each candidate is an S-shaped lateral profile \(y(t)\) with \(\dot y = 0\) at both ends and a
longitudinal speed profile; the heading is \(\arctan(\dot y / v)\) and the curvature command is
\(\ddot y / v^2\) plus the lane's own curvature a servo lag ahead (what LA already does).
Add the actuator: a 0.15 to 0.25 s servo lag and the 0.2 lock/s walk. At `g_s` 0.186 that
walk is 0.037 1/m per second of curvature change, which at 25 m/s is 23 m/s^3 of lateral
jerk, so the walk is not the limit; the force cap and grip are (inferred from the numbers).

Minimum durations (bang-bang, lateral accel \(a_y\), shift \(y\), plus 0.2 s servo lag):

| manoeuvre | \(a_y\) | time | distance at 50 km/h | at 90 km/h | at 120 km/h |
|---|---|---|---|---|---|
| nudge 1.0 m | 2 m/s^2 | 1.6 s | 22 m | 40 m | 53 m |
| half lane 1.75 m | 3 m/s^2 | 1.7 s | 24 m | 43 m | 57 m |
| full lane 3.5 m | 3 m/s^2 | 2.4 s | 33 m | 60 m | 79 m |
| full lane 3.5 m | 5 m/s^2 | 1.9 s | 26 m | 47 m | 62 m |

Braking from the same speeds at 8 m/s^2 with the 0.25 s pad: 16 m, 45 m, 77 m. The table
is the core of the brake-versus-steer answer in section 3: for a truck limited to 3 m/s^2 a
full lane change does not beat braking until about 110 to 120 km/h, while a nudge beats it
from about 80 km/h up. Steering earns its keep on partial overlaps and at motorway speed, not
in town.

### 2.3 Scoring a candidate against temporal bodies

For each candidate, sweep ego's rig at 24 to 36 time samples out to the manoeuvre's end plus
one braking stop, and test every body's inflated capsule at the same times. Reject if any
sample intersects. Among survivors, cost terms (all measurable from existing signals):

- Minimum clearance over the horizon, with TMP bodies inflated more.
- Lateral accel and jerk asked (annoyance and trailer stability).
- Leaving the lane at all (large constant), entering a lane whose `edges.beyond` is
  `oncoming`, `edge` or `unknown` (forbidden, not a cost: unknown is a wall because the live
  map has no shoulders or barriers).
- End state: must be parallel, inside a lane, and from it a plain brake must still stop behind
  everything in the new corridor out to the clearance horizon (the "recoverable" test; it is
  what stops a swerve from trading one hit for another two seconds later).
- Bodies **behind and beside** in the target lane via `side_traffic` logic with its 3.0 s and
  2.5 s time margins; ETS2LA publishes bodies behind ego, so this is computable.
- Hysteresis: the candidate chosen last tick gets a bonus so the plan does not flip at 30 Hz
  when two candidates are near equal (LA's drives show every flip becomes a wheel swing the
  ring and buzz guards then punish).

Bodies past 3 s are not planned around; they only enter through the recoverable-end-state
brake test, which is a demand computation AEB already does.

### 2.4 Interactions and what is not modelled

Other drivers' reactions are not modelled, the same assumption AEB makes. Oncoming traffic
is treated as the other party keeping or widening its line (`OppositeLaneFilter` models the
other party's evasion; a planner must not count on it). A body with no track yet (first
sighting, TMP match dropout) blocks its lane. TMP bodies in a lag freeze block where they are
drawn and are never advanced on SDK velocity.

## 3. Question 2: brake or steer, least annoyance, still safest

### 3.1 The physics puts braking first for a truck

Section 2.2's table says it: a full lane change at the lateral accel a loaded truck and the
3000 force cap allow is slower than braking below motorway speed. Steering wins in two
places, (a) partial overlaps where a nudge suffices and (b) high speed. Both are narrow, and
the avoidability analysis on the corpus says the window between "braking still works" and
"only steering works" was 0.1 to 0.3 s on the clips that motivated it. A decision rule that
waits for braking to fail before it steers will almost always be too late; one that steers
early on a guess collects false swerves. The way out is cheap manoeuvres decided on certain
geometry, and expensive ones decided only when braking has already provably lost.

### 3.2 Tiered arbitration

Tier order; a higher tier is entered only when its conditions hold, and every tier keeps the
AEB brake it inherits.

| tier | action | entry | why this cost order |
|---|---|---|---|
| 0 | nothing / warn | AEB's own warn classes | unchanged |
| 1 | AEB brake | today's bar: demand at 0.85 of capacity or TTB under 0.5 s | measured envelope |
| 2 | brake + in-lane nudge | tier 1 engaged, target is `certain_geom` co-directional or stationary in ego's lane, a nudge clears it with margin, no body in the swept band, lateral accel under 2.5 m/s^2, driver not pushing | never crosses a line, so a wrong nudge is a 1 m wobble under braking, not a lane change |
| 3 | lane shift (plus partial brake) | demand exceeds capacity even with the pad (braking has lost), adjacent same-direction lane exists and is empty by `side_traffic` margins, no `branch_s` or `merge_s` inside the manoeuvre, speed above a floor (about 50 km/h, below it braking is always shorter), not TMP zone, end state recoverable, driver not pushing | the only case where a wrong steer is cheaper than the hit that is otherwise certain |
| any | hand back | `hand_push` beyond the servo, override/insist, blinker, driver `evade` | a grip counts as attention (Lukas, 2026-11-03) |

Things the tiers never do, each with a reason already written down in the tree:

- Never steer toward `oncoming`, `edge` or `unknown`. The map has no shoulder, barrier or
  prop knowledge live; what is off the carriageway is unknown.
- Never steer for a crosser, a head-on target, or anything past the 30 m lane-confidence
  range on oblique geometry. Those are the classes where AEB's own filters most often
  disagreed with the label (crossing scenes about 79 % `ignore` on the contributed corpus).
  A false brake for them is a known annoyance; a false swerve would be a new accident class.
- Never brake at capacity and steer at the same time. The grip envelope is one budget
  (p99 total 7.9 to 11.9 m/s^2 on the corpus); a nudge under hard braking stays under about
  2.5 m/s^2 lateral, a shift gives the brake back to about 5 m/s^2 while turning. ETS2's
  tyre model is not the real one, so the split must be measured in game (inferred).
- Never continue a manoeuvre the servo cannot complete: if the wheel sits off target for
  0.3 s (the override test) the plan is abandoned to brake-only, because past the cap the
  truck runs wide anyway.

### 3.3 Annoyance accounting

The driver's complaints, ranked by what the tree already knows about them:

1. A phantom swerve at speed. Must be near zero. Tier 3 therefore sits strictly behind
   "braking has already lost", where the alternative is a certain hit, and tier 2 is
   confined to inside the lane.
2. A phantom brake. Today's AEB false positive rate on the local store was 65 FP against
   430 TN on 960 clips at the last scorer change; the steering tiers inherit that and add
   nothing to it, because they never fire without tier 1.
3. A swerve that fights the driver. The hand-back column covers it; the force cap means a
   hand always wins, and `hand_push` releases from 50 % of the pulse's own push. LA's lesson
   is that detecting the driver from the 20 Hz view pushed back for up to a second, so the
   one-sided `servo_only` clip at the FFB rate matters for a nudge too: once the plan is a
   nudge to the left, the servo may never push right.
4. A manoeuvre that leaves the truck in the wrong lane. The end-state rule insists on
   centred and parallel; the actuator then hands to LA only if LA is enabled, otherwise it
   fades out over 1 s and the truck is where the driver left it, in a lane.

### 3.4 The absent driver

Nothing in the tree can prove a seat is empty. The signals that exist are negative ones:
no blinker, wheel not meeting `deliberate` or `intent`, `hand_push` near zero or only
cancelling the servo, and after 10 to 15 s of LA engagement the unmeasured `hands_off`
check. The honest design does not try: **the tiers act whenever braking alone cannot avoid
the hit, and a grip hands back.** An attentive driver who is already steering trips `evade`
or `hand_push` and gets nothing; an inattentive one gets the manoeuvre until they grab the
wheel. The one cost, already accepted for LKA, is that heavy resting hands read as a grip
and the truck runs wide after the hand-back.

## 4. What the corpus can and cannot do for this

### 4.1 Cannot: closed-loop evaluation by replay

Replay rides the recorded, already-braked trajectory and never moves the pedal; a steer
counterfactual needs a new ego pose sequence, and the other vehicles' recorded tracks do not
react to it. The existing `evasion.py` ghost is the measured ceiling for that kind of
counterfactual: credible to `CREDIBLE_S` 1.5 s, p90 drift 3.2 to 6.7 m at 2.0 s depending on
which axis is held. A lane-change manoeuvre lasts 1.9 to 2.4 s plus the stop, so clip replay
can at most judge the first half of one.

### 4.2 Can: mine the geometry to size the design

Directly from recorded fields (no map):

- The **brake-fails window** per labelled positive: run clearance on each tick to find when
  demand passed capacity, and the escape test (a nudge or shift profile, not a constant arc)
  to find when the last clean swerve started. The distribution of that gap is the single
  number that decides whether tier 3 is worth building. The two avoidability clips suggest
  0.1 to 0.3 s; the whole TP set may say otherwise.
- Lateral free space around ego at engagement from raw buffers (bodies in a fixed band one
  lane over, left and right), `lat_at_min_m` and `min_corridor_gap_m` are already in
  `features.py`.
- How often the driver swerved: the `driver_swerve` tag (`SWERVE_RATE` 1.2 per second,
  `SWERVE_DELTA` 0.09 on `userSteer`), with `steer_rate_peak` and the lateral accel it
  produced, split by TP/FP/`good_intervention`. This gives the lateral accel real drivers
  used in this sim and whether they steered, braked or both.
- Partial-overlap statistics: fraction of co-directional TPs where the body covers less than
  half of ego's corridor (the nudge case) versus the full lane.
- Grip envelope by speed (already measured once: 7.9 to 11.9 m/s^2 p99).

With the local map rebuild (`tools/map_replay.py`, matched on 52 to 90 % of clips by store):

- Whether an adjacent same-direction lane existed at engagement (`lanes_same_dir`, dossier
  `map_rel` `adjacent_left/right`), whether it held a body, and `branch_s`/`merge_s` inside
  the manoeuvre length. This is tier 3's feasibility rate on real events.
- `edges.beyond` class on each side at engagement, to see how often "unknown is a wall"
  forbids the only escape.
- Prediction inflation: `map_target_eval.py` already scores other vehicles' lane-following
  prediction against where they went; the per-class error table is the uncertainty model in
  section 2.1 and can be extended to 4 and 5 s on the same tool.

Under the rules: labels stay human only, dossiers may be read to analyse and never as tuning
input, map data never enters a clip, nothing leaves the machine.

### 4.3 What would be needed to test it properly

1. **Lateral closed-loop harness.** Copy `_Plant` and the gap integrator from
   `tests/aeb/test_stop_distance_envelope.py`, add a yaw and lateral state driven by the
   wheel model that `tests/la/test_controller.py` and `tests/lane_departure/test_lka.py`
   already use (inertia 650, game spring, servo PID, 0.2 to 0.3 s truck lag), and run the
   manoeuvre library against scripted bodies: stopped car half in lane, cut-in, slower lead
   with an empty and with an occupied neighbour lane, oncoming on the other side. Scenario
   sweeps over speed and load, like the envelope's four rigs at five speeds.
2. **Recorded-traffic ghost runs.** Fly the manoeuvre against the recorded tracks of
   labelled TPs for the first 1.5 s (the credible window), counting clean, grazed and
   infeasible. It cannot see reactions; it can say how often the geometry was there.
3. **Platoon sim extension.** `tools/acc_platoon` has the real radar chain and a calibrated
   TMP netcode model but no lateral state and no cut-ins. Adding a lateral degree of freedom
   to its plant and a scripted cut-in gives the only place where TMP jitter meets a swerve
   decision in closed loop.
4. **Live shadow mode first.** Compute the tier decision every tick, log it with the inputs,
   never actuate. Compare against what the driver did (`driver_swerve`) and what AEB did.
   This is how the AEB warn classes were graded, and it needs no clip schema change (a new
   clip field bumps `CONSENT_VERSION`).

## 5. Where it would plug in

- **Planner** as a new consumer of `RadarThread` and `MapThread` (own thread from the
  example template, 30 Hz, fails to "no manoeuvre" on any missing sibling), publishing a
  wanted path plus brake request. It reads AEB's published demand, colliding set and
  `engage_vetoed_ids`; it never recomputes AEB's decision and never feeds anything back into
  AEB (AEB's section 9 rules forbid a planner moving a latch, hold or warn).
- **Path type.** Extend `WantedPath` (or add a sibling) to carry a time-parametrised lateral
  profile: offset, heading and curvature at the preview time, plus a validity stamp. The
  controller needs to preview `path.kappa` one servo lag ahead the way it previews lane
  curvature, drop stop-before-the-line and the learned bias while an evasive source is
  active, and allow the source to outrank `blinker` and LA's own `evade` (the manoeuvre's
  wheel motion would otherwise read as a driver swerve). Keep two-sided servo for a shift
  (the truck must unwind onto the new line), one-sided for a nudge.
- **Keep every cap and guard the LA README marks untouchable**: the 3000/1500/4000/6000
  caps, override and insist, `hand_push` hand-back, ring guard, plugin buzz guard, sign
  lock, stale fades, geometry hold. Bypass only what pulse mode already bypasses (hands-off,
  weave guard, route gates) and only for the manoeuvre's few seconds.
- **Brake side.** The planner's brake request goes through AEB's `AebPedalAxis` path with
  `full_authority`, never a second mapper and never a new pedal slam.
- **TruckersMP.** LDW and LKA pulses were cleared by moderation (2026-11-02); continuous
  steering (LA) was not, `la_tmp_enabled` is false. A 2 s lane shift is LA class, not a
  pulse. Off on TMP until asked, and the TMP prediction errors (table in 1.2) argue for
  keeping even tier 2 SP-only at first.

## 6. Invariants this must not break

- Radar: consume `Vehicle` read-only, never open the buffer, never re-derive elevation,
  NCZ or overlap ghosts, never move a frozen TMP body on SDK velocity.
- AEB: no change to engagement, demand aggregation (max over targets, time lag pad), the
  latched set, the vetoes' reach, the 0.85 bar. The planner is downstream.
- LA/LKA: never push the wheel toward the line a pulse saves, never weaken ring, buzz,
  override, hands-off, caps. A grip counts as attention.
- Map: a stale snapshot or `map_data_enabled` off must leave the planner at tier 1 (brake
  only), bit identical to today for AEB and ACC.
- Privacy: no new recorded fields without a consent bump, no map data in clips, no player
  identity, nothing leaves the machine.

## 7. Verdict and open questions

Merit check (feature, from the driver's seat): **prototype in shadow mode; do not ship
actuation yet.** The strongest case against: a truck that steers by itself at 90 km/h is a
new accident class in a community that signed up for pedal assists, the live map cannot tell
a shoulder from a ditch, TMP positions are metres wrong exactly when the sender brakes, and
the one measurement that exists says the steer-only window is a few tenths of a second wide.
The strongest case for: tier 2 (an in-lane nudge under an AEB brake that is already
happening) is cheap, bounded by lane lines, hands back on any grip, and partial overlaps are
a real share of co-directional TPs (to be measured, 4.2).

Questions the data should answer before any code:

1. Across all labelled TPs, how wide is the brake-fails-but-steer-works window, and how
   often does a 1 m nudge rather than a lane shift close it?
2. On how many TPs did an empty same-direction lane exist at engagement, by the map rebuild?
3. What lateral accel did real drivers use when they swerved in these clips, and did a
   swerve plus brake beat brake alone on the recorded outcome?
4. In game, how does the tyre model split grip between braking and cornering on a loaded
   rig, and does the 3000 cap really hold 5 m/s^2 at 90 km/h on a real wheel?
5. Does TruckersMP moderation accept a 2 s automatic lane shift? If not, the feature is SP
   only and the TMP prediction table stops mattering for it.
