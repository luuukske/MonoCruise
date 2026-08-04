# MonoCruise ACC (in-lane vehicle tracker)

> ACC-specific logic. Coordinate system, Vehicle smoothing, ArcPath
> geometry, and the RadarThread snapshot shape are documented in
> `core/radar/README.md`: **read that first.** This file only covers
> what differs from the shared radar layer.
>
> Agent workflow and do-not-break rules: top-level `AGENTS.md`.

---

## 1. Scope

This module is an **in-lane vehicle tracker**, not a controller.

It watches the pre-filtered traffic stream from `RadarThread`, scores
each vehicle per-frame against the ego path, and publishes a ranked
list of in-lane leads on `ACCThread.data`. Consumers (currently the
cruise control thread, eventually a gap-control law) read that list
and decide what to do with it.

**No accel command, no accel cap, no gap control lives here.** That
all belongs in `core/cruise_control_thread/` when it gets written.
This module answers one question only: **"which vehicles are in
ego's lane right now, and how confidently?"**

---

## 2. Ego path: blended curvature

Tracking quality depends on an accurate "where is ego going" arc.
ACC blends two curvature sources (`core/acc/ego_path.py`):

| Source              | Where it comes from                    | Strength                    |
|---------------------|-----------------------------------------|-----------------------------|
| Steering-derived κ  | `userSteer * 0.17`                      | Leads actual trajectory:   |
|                     |                                         | correct below 15 km/h when  |
|                     |                                         | history-fit is collinear.   |
| History-derived κ   | `RadarData.ego_curvature`               | Accurate at speed:         |
|                     | (circumscribed-circle fit on last 25    | reflects real trajectory.   |
|                     | ego positions, from `RadarThread`).     |                             |

Blend weight is a linear ramp in `km/h`:

    ≤ 15 km/h  → 100 % steering
    ≥ 30 km/h  → 70 % history + 30 % steering
    linear between.

**AEB does not consume `RadarData.ego_curvature`**: see
`core/aeb/README.md`. AEB must react to the instantaneous yaw-rate
proxy. ACC can (and should) use the smoothed history fit because
tracking tolerates: and prefers: a little smoothing.

### Path half-width

    half = LANE_BASE_HALF_M + sin(min(|steer|·1.5, 1) · π/2) · LANE_FLARE_HALF_M

- `LANE_BASE_HALF_M = 1.25 m`: 2.5 m corridor on a straight road.
- `LANE_FLARE_HALF_M = 2.0 m`: up to +2 m per side under heavy steer.

---

## 3. Scoring (meter-native)

Four components per frame (see `core/acc/scoring.py`). The formulas
mirror legacy ETS2radar `SCORING_REFERENCE.md` faithfully: the only
semantic departure is dt-scaling so the loop can run at any cadence.

| Component  | Units         | Range (approx.) | Formula (summary)                                                                 |
|------------|---------------|-----------------|-----------------------------------------------------------------------------------|
| offset     | dimensionless | [-1.6, +1.5]    | Gaussian `2^(-(x/σ)²)` on arc-crossing lateral (σ = 2.25 m), × distance_amp,      |
|            |               |                 | clamped ±1, × outer `1.5·(angle_amp·0.4 + 0.6)`, + baseline.                      |
| yaw        | dimensionless | [-1.5, 0.0]     | `(2^(-(|Δyaw|/90°)^5) - 1) · 1.5`.                                                |
| path       | dimensionless | [-4.0, +5.0]    | `1.03^(-d_m) · slow_amp · (1 - b²·0.4)`; `min(·, 5)` in / `-min(·×0.6, 4)` out.     |
| angle      |:             | 0.0 (reserved)  | Arc-arrival angle in radians. Legacy had it disabled: still is.                  |

### offset: constants and baselines

    σ                = 2.25 m
    distance_amp(d)  = [2^(-d/100) + 8/(d+3) - 1] / 3 + 1
    angle_amp        = 2^(-(normalised_arc_angle / 0.06)²)   # from trail-arc fit
    baseline         = 0.0 on arc hit
                     = -0.40 when the arc was fit but didn't cross ego row
                     = -0.16 when position history was too short to fit

Blinker lateral bias is applied as a **scalar offset shift** on the
scored lateral: `offset_for_score = lat - blinker · 4.5 m`: not as
an ego-arc translation. Shifting the arc geometrically would distort
arc-arc hit tests, which we don't want.

### Trail-arc fit (`core/acc/trail_arc.py`)

Each target's `_position_history` is downsampled (≥ 1 m AND ≥ 0.05 s
between kept samples) and then **LS algebraic circle-fit** in place.
Centre + radius come from the positions alone: no smoothed-yaw
input: so yaw jitter no longer translates into crossing-lateral
jitter. A max-perpendicular-from-chord straightness pre-check
prevents position noise from being fit as a tight curve. The fitted
circle is intersected with the **ego row** (line through ego
perpendicular to ego heading); the crossing point gives `offset_m`,
and the tangent direction there gives `arc_angle` which feeds
`angle_amp = 2^(-(arc_angle / 0.06)²)`.

The downsampling lives inside this module: radar's
`_position_history`, `Vehicle.curvature_from_history`, the TMP raw-
speed LS fit, and AEB consumers all see the dense raw buffer
unchanged. Only ACC's per-target trail consumes the gated subset.

Three buckets drive baseline selection:

| Bucket       | Trigger                                                       | Baseline | offset_m fallback | angle_amp |
|--------------|---------------------------------------------------------------|----------|--------------------|-----------|
| `HIT`        | fit + arc intersects ego row                                  | 0.0      | crossing lateral   | from arc  |
| `NO_ARC_HIT` | fit but no intersection (target's circle too tight, off-side) | −0.40    | current lateral    | 1.0       |
| `NO_HISTORY` | fewer than 5 samples / chord < 0.5 m / curvature_from_history None | −0.16 | current lateral | 1.0       |

### Evidence: how much the trail is trusted

The bucket alone is not confidence. Before evidence gating, both failure buckets
set `angle_amp = 1.0`, the **maximum** amplitude, so a stationary target with no
trail at all scored better than a slow one whose noisy fit read a steep arrival
angle. On the clip corpus 98 % of stationary locks arrived through
`NO_HISTORY`, and in-corridor co-directional targets under 8 m/s sat at a median
`angle_amp` of 0.088 while stopped ones got 1.0. The ordering was inverted.

`evidence ∈ [0, 1]` now scales the whole offset term:

    motion   = |history[-1] − history[0]|          # net, not path length
    evidence = clamp((motion − 1.0 m) / (8.0 m − 1.0 m), 0, 1)
    amp      = evidence · angle_amp + (1 − evidence)
    offset   = evidence · (baseline + clamped · outer)

Two properties matter:

- **Evidence is observed motion, not fit success.** A slow target with too few
  downsampled samples to fit a circle has still shown which lane it is
  travelling, so it keeps partial credit. A target that never moved contributes
  nothing and the corridor geometry decides alone.
- **The angle penalty regresses with its own evidence.** A short trail measures
  its arrival angle badly, so a low-evidence steep angle must not read as a
  confident "this is crossing my lane". At full evidence the penalty is intact,
  which is what keeps oncoming and cross traffic rejected.

Net displacement rather than summed path length: 25 samples of a few centimetres
of position noise on a parked vehicle sum to over a metre of apparent travel.

#### The evidence floor, and why `path` is gated by it

`evidence = max(trail_evidence, road_weight, _ARC_EVIDENCE_FLOOR)`. The floor
exists because the scored lateral is a **blend**, and the ego arc in it is a
measurement in its own right, so position evidence is never actually zero.
Without the floor a stationary target with no road model zeroed its own offset
term and then scored on `path` alone, which awards a positive value for being in
the corridor regardless of how well the corridor is known. That is the same
absence-as-evidence inversion the evidence scaling exists to remove, arriving
through a different component: tightening road confidence made stationary
false-lock **rise** from 3.5 % to 4.3 %.

So `path_component` takes `evidence` too, and gates the in-corridor **reward**
only. The out-of-corridor penalty stays ungated, the same asymmetry as
`offset_component`: awarding "it is in my lane" needs to know where it is,
declining to is the conservative direction.

`_ARC_EVIDENCE_FLOOR` **must stay below** `_VALIDATED_STATIONARY_EVIDENCE`. At
0.6 they collided and the validation latch below stopped distinguishing a vehicle
watched into place from one that was always parked; `tests/acc` pins the
ordering.

### Stationary validation latch

Production ACC refuses to brake for stationary objects because a stationary
object carries no trajectory evidence about which lane it is in. MonoCruise can
do better than a real radar here: ids are stable and history never drops, so a
target that was **watched driving its own line** can be trusted after it stops.

`TrackState.moving_validated` latches once a target holds a `HIT` with evidence
≥ `_VALIDATE_MIN_EVIDENCE` while moving faster than `_VALIDATE_MIN_SPEED_MS`.
Once latched, a stopped target with no current fit keeps
`_VALIDATED_STATIONARY_EVIDENCE` (0.6) on its instantaneous lateral instead of
falling to zero. The latch is never revoked; a track that disappears loses it
with the `TrackState`.

Deliberately **not** done: never-validated stationary targets are not suppressed
from the lead list. Coming over a crest into an already-stopped queue is a real
case, and the corridor geometry must still be able to lock it. The latch raises
precision on validated targets; it does not gate recall on the rest.

Constants live in `core/acc/trail_arc.py`:
`_HISTORY_MIN_DIST_M = 1.0`, `_HISTORY_MIN_DT_S = 0.05` (per-fit
downsample), `_MIN_FIT_SAMPLES = 5`, `_MIN_PATH_LEN_M = 0.5`,
`_STRAIGHT_KAPPA_MAX = 1/2000` (below this κ the LS-fitted circle is
collapsed back to a straight line), `_MIN_SAGITTA_RATIO = 0.5` (the
observed perpendicular sagitta has to be ≥ 50 % of what the LS
radius would imply, otherwise the curve was fitted to noise),
`_ANGLE_AMP_SIGMA = 0.06` rad. Sweep direction (`sign`) is recovered
from the actual history (cross-product of prev-to-centre and
target-to-centre vectors), so it follows `ArcPath._sign` without
needing kappa to be signed up front: +1 ⇒ left turn ⇒ CW sweep
around the centre, per `max_sweep = -sign · arc_length / radius`.

The LS circle fit mean-centres `(x, z)` before solving the normal
equations so TMP world coordinates in the 10⁵ range do not collapse the
3×3 determinant (otherwise every trail reads as straight on real maps).

### path: slow_speed_amp and blinker reduction

    slow_amp          = 1.4 + (kmh / 100) · 4.1
    blinker_reduction = 1 - b² · 0.4       # b = signed blinker scalar; no clamp

At low speed the corridor effectively fattens because the path decay
is shallower relative to lane width. During a lane change the squared
blinker scalar reduces the path amplitude by up to 40 %, so targets
in the current lane stop pinning score while we're committing to the
move.

### Accumulation: `[-5, +8]` clamp

    weighted  = offset·1.5 + yaw + path·0.7 + angle
    delta     = weighted · speed_mult(v_target) · dt · 10
    score_new = clamp(score_prev + delta, -5, +8)

- Asymmetric clamp: fast lock on in-path vehicles, slow release on
  uncertain tracks (narrow negative floor).
- **The ceiling tracks the consumer, not the tracker.**
  `AdaptiveCruiseController` saturates its confidence ramp at score 5
  (`ANT_SCORE_FULL`), so score above that buys nothing except release
  latency. At the old ceiling of 20, half of all positive-score frames sat
  pinned at the clamp and a lead leaving the lane took a p90 of 2.45 s to
  fall back under confidence. At 8 that is 1.20 s with lock latency
  unchanged. Raising the ceiling again re-creates the hooking.
  Enforced by `tests/acc/test_scoring_evidence.py`.
- `speed_mult = max((|v_target| / 90 m/s)^0.8, 0.5)`: **target**
  vehicle speed, not ego's. The 0.5 floor applies at every realistic road
  speed, so in practice this term is the constant 0.5.
- The `· 10` multiplier keeps the per-frame delta matched to the legacy
  integer-tick maths. Higher or lower loop rates scale proportionally: the
  accumulation is frame-rate independent by design.
- **`_PATH_DECAY_BASE` is 1.022, not the legacy 1.03.** At 1.03 the path term is
  `1.03^-85 = 0.081` at 85 m, an eighth of its close-range value, so a correctly
  classified distant lead accumulated at ~1.5 score/s instead of ~12. The
  measured cost of locking in time was a lead that only reached confidence once
  ego was already close, which is what made AEB warn on approach. Going further
  (1.015, 1.01) buys more recall but puts stationary false-lock back up from
  1.6 % to 2.5-3.4 %, undoing §9's gain. Corpus at 1.022: moving in-lane recall
  56.1 -> 57.5 %, stationary false-lock unchanged at 1.6 %, moving false-lock
  2.3 -> 2.7 %.

### In-path hysteresis

A held target keeps `_IN_PATH_HYSTERESIS_M` (0.8 m) of extra corridor before it
is released. On winding roads a noisy lateral flipped the decision a median of 2
and up to 12 times per in-lane track, and every flip pushed the score back down.
Hysteresis cuts the flip p90 from 6 to 4 and raises `in_path` correctness at
45-70 m from 85.4 % to 88.1 %.

Worth knowing: hysteresis alone moved the **lock** rate by 0.1 points. Flicker
was real but it was not what kept leads unlocked; the distance weighting above
was. Do not reach for more hysteresis expecting lock latency to follow.

### In-path threshold

`score > 0` ⇒ in-path. Hard cut-off: only positive-score vehicles
appear in the published `leads` list.

---

## 9. Shared road model (`core/acc/road_model.py`)

Before this, two independent estimates answered "which lane": the ego arc
(constant curvature, extrapolated from behind ego) and each target's own trail.
They never spoke to each other, and both are measured **in the ego frame**, where
"the target moved sideways", "ego rotated" and "the road curved" are the same
observable. That ambiguity is structural, not a tuning problem.

One centreline is now fitted per frame from every source at once, and laterals
are measured against it.

### Parameterisation

    y(x) = c1·u + c2·u² + c3·u³        u = x / 100 m
    κ(x) = -(2·c2 + 6·c3·u) / 100²

`+x` is ego forward, `+y` is ego right, anchored so `y(0) = 0`. This is the
standard clothoid cubic: `c2` is curvature at ego and `c3` is **curvature rate**,
which is what the old constant-curvature arc lacked and why it lost in-lane leads
past ~40 m on curves.

### Heading prior: the centreline leaves ego along ego's heading

`y(0) = 0` is only a *positional* constraint. With `c1` free the road could also
**tilt** at ego, and a tilt biases every target's lateral by `c1·x/100`:
negligible at 20 m, about a metre at 100 m. That is a distance-proportional bias,
which reads exactly like "in-lane vehicles are being called adjacent" and is
structurally likely, because after offset elimination a far source's centred
samples only span its own trail window and have a large, ill-conditioned lever on
`c1`.

`_HEADING_PRIOR_WEIGHT` adds a ridge term pulling `c1` to zero, so the road
leaves ego along ego's heading. Measured on the corpus, the effect **saturates
immediately** (a prior of 100 and a hard pin give identical results), which
confirms `c1` was being set by a weak, noisy signal:

| | c1 free | with prior |
|---|---|---|
| moving in-lane recall, overall | 54.0 % | **56.1 %** |
| stationary false-lock, overall | 2.4 % | **1.6 %** |
| stationary false-lock, 20-45 m | 7.4 % | **4.6 %** |
| lateral error p50, 20-45 m | 0.43 m | **0.33 m** |

The ridge form is kept rather than a hard pin so the constraint stays soft: ego
is not always lane-aligned. Since the result saturates, the exact weight does not
matter; do not spend time tuning it.

### Base arc: why the centreline is not a polynomial alone

A parabola **undershoots** a circle. True offset is `R − √(R²−x²)`; `x²/2R`
always curves less, so a polynomial-only centreline sits toward the **outside**
of a bend, growing with `x²` and with tightness:

| radius | error at 60 m | error at 100 m |
|--------|---------------|----------------|
| R = 500 m | 0.02 m | 0.1 m |
| R = 200 m | 0.21 m | 1.8 m |
| R = 100 m | 2.0 m | (x beyond radius) |

That is invisible on motorway curves and over a lane wide on a tight bend,
which is exactly how it was reported from the driver's seat. `base_arc_lateral`
therefore carries ego's current curvature as an **exact circle** and the cubic
only describes the deviation from it, where the small-angle assumption holds.
`_BASE_ARC_MAX_FRAC` clamps evaluation inside the circle's forward extent,
because past that a forward-distance parameterisation has no unique answer.

Synthetic road fixtures must generate **circles, not parabolas**. The first
version of `tests/acc/test_road_model.py` generated parabolic roads and so
silently blessed the bias it existed to catch.

Corpus effect was +0.1 points of recall: the AEB clip corpus is nearly all
gentle highway geometry and contains very few R < 250 bends, so it cannot
demonstrate this fix. The unit tests pin it instead.

### Per-source trust

Sources earn road-model weight by agreeing with the fitted road over time
(`_ROAD_TRUST_TAU_UP_S` 0.5 s up, `_ROAD_TRUST_TAU_DOWN_S` 0.15 s down). A new
id enters at `_ROAD_TRUST_INITIAL` rather than at full weight, which is what
stops the centreline stepping as ids churn in busy traffic.

Measured on centreline jump, weighted by the confidence that actually reaches
the blend: p99 at 50 m 1.48 -> 1.33 m, max 31.1 -> 21.8 m; p99 at 100 m
3.83 -> 3.16 m. Recall and stationary false-lock both move about 0.3 points, in
opposite directions, so the jump numbers are what justify it.

**The trust loop must be able to bootstrap.** `fit_road_model` returns its
per-source residuals even when confidence is zero; without that, low trust means
no fit, no fit means no residuals, and no residuals means trust never rises.

### Centreline slew limit (`RoadSmoother`)

The fit is stateless, so a change in the source set steps the centreline. The
carry-over is done in **sample space**: the previous frame's nodes on `_NODE_X`
are transformed into the current ego frame and resampled before being combined,
so ego's own motion is removed rather than smoothed as if it were road change.
Coefficients are never filtered frame to frame (sloshing, see above).

The combination is a **rate limit, not a low-pass.** That distinction is the
whole result:

| | recall | stationary FP | jump @50 m p99 | @100 m p99 |
|---|---|---|---|---|
| none | 52.5 % | 1.3 % | 2.20 m | 4.10 m |
| EMA tau 0.25 s | 56.4 %* | 3.0 %* | 1.22 m* | 3.13 m* |
| EMA tau 0.70 s | 52.3 %* | 4.4 %* | 0.93 m* | 2.31 m* |
| **slew 28 m/s** | **52.4 %** | **1.3 %** | **1.42 m** | **2.84 m** |

(*measured on a 40-clip sample; the slew rows are the 80-clip sample. Compare
within a column group only.)

An EMA degrades recall and stationary false-lock monotonically with tau, because
it lags every frame in order to soften the few that step. The measured jump
distribution says why that is the wrong tool: p50 is 0.001 m and p90 is 0.12 m,
so the centreline is stable almost always and the problem is a **tail of
discrete events** (a source entering or leaving, or a bad fit). Averaging a bad
value with a good one yields a bad value that also lags. A rate limit passes
normal frame-to-frame change untouched and only clips the steps.

Do not tighten the limit chasing the tail. Below about 20 m/s the clipped
centreline diverges from the fit and then catches up in a rush, so **jumpiness
gets worse than no limit at all** (at 8 m/s, jump p99 at 50 m rises to 1.39 m
against 1.33 m unlimited) while recall collapses.

#### The budget is curvature, not metres

A flat metre-per-second cap was the wrong unit. A centreline's lateral offset is
`kappa·x²/2`, so a single change in road curvature moves a node at 100 m a
hundred times further than one at 10 m. Under the old flat 28 m/s the budget was
spent almost entirely on the far nodes, and the measured saturation was a clean
function of distance:

| node | 50 m | 100 m | 140 m |
|------|------|-------|-------|
| frames at the limit | 2.3 % | 7.9 % | **26.4 %** |

Worse, the quantity being limited is largely **ego's own steering wheel**, which
is a known input rather than an estimate. The base arc alone demanded a lateral
rate at 100 m of p50 5.4 m/s, p90 36.9 and p99 215, so in **14.5 % of frames the
base arc on its own exceeded the entire budget** and the far end of the
centreline was permanently lagging the near end and then catching up. That drag
is what reads from the driver's seat as the centreline sweeping between two arcs.

`node_slew_budget_ms(x)` therefore spends `_SMOOTH_MAX_KAPPA_RATE` of curvature
per second, which is `kappa_rate·x²/2` in metres, floored at
`_SMOOTH_MIN_RATE_MS` so the nodes nearest ego are not frozen. One curvature
change now costs the same fraction of the budget at every distance.

| | flat 28 m/s | curvature budget |
|---|---|---|
| slew saturation | 42.6 % | **17.6 %** |
| jump @50 m p90 | 0.925 | **0.674** |
| jump @50 m p99 | 0.988 | **0.709** |
| jump @50 m while turning, p99 | 1.244 | **0.997** |
| lateral error @70 m p90 | 9.83 | **9.40** |
| lateral error @100 m p90 | 19.42 | **18.47** |

Loosening the far field improved accuracy at every range as well as the jump,
which is the tell that the limiter had been the binding constraint rather than
the fit. **The floor is load-bearing**: at 2 m/s the near nodes are throttled
instead and error at 30 m p90 doubles to 3.25 m, a full lane at the range ACC
actually locks. 20 m/s beat both 2 and 10 on accuracy *and* on saturation.

#### Confidence is rate limited too

The fit is stateless, so `confidence` was recomputed each frame and could go
from 1.0 to 0.0 inside a single frame. That retargets every node at once between
the fitted cubic and the bare ego arc, which is the same sweep by another route.
`_CONF_RATE_UP_PER_S` / `_CONF_RATE_DOWN_PER_S` bound it, so the largest possible
change is `rate·dt` (0.05 at 30 Hz). While confidence decays, nodes the fit no
longer reaches **hold their carried shape**, fading to the base arc as it drops,
rather than snapping there the moment the last source is lost.

Do not slow the decay further to suppress the on/off count. At 0.8/s the flip
interval improves (3.69 s to 4.19 s) but stale shape survives long enough that
lateral error at 50 m p99 goes 10.41 m to **29.54 m**. The count of zero
crossings is the wrong target; the size of each step is the one that matters.

Nodes past `support_x_m` carry the base arc, not the extrapolated cubic. Feeding
untrusted extrapolation into the carry-over made the 100 m outliers three times
worse, because a bad value then persisted instead of being a single frame.

### Things measured and rejected

Recorded so they are not retried blind. All on the same 40-clip sample.

| Tried | Result |
|-------|--------|
| Raising the ego-sample weight (2, 4, 6) | **No effect at all.** Once the base arc carries ego's curvature exactly, ego's own samples are redundant with it. Left at parity. |
| Lateral cap on sources, per sample | Stationary false-lock 2.3 % -> 4.1 %. Truncating one source's trail biases its own mean, which the offset elimination feeds back as shape. |
| Lateral cap on sources, per source | Still 4.1 %. |
| Lateral cap measured against the base arc | 4.1 % and recall 56.9 -> 55.9 %. On an R200 bend an **in-lane** vehicle 120 m ahead is 40 m laterally in the straight frame, so any lateral cap preferentially deletes the far-field sources that carry the curve. Dropped entirely. |
| Plausibility clamp on cubic deviation | Rejects legitimate **curve entry**, where ego is still straight while the road ahead bends, so the deviation is genuinely large exactly when the model matters. Reverted. |
| Oncoming traffic as a road source | Recall unchanged to -0.3 pts, stationary false-lock 2.3 -> 1.6 %, but centreline jump p99 at 50 m 1.33 -> 1.79 m and at 100 m 3.16 -> 6.45 m. The reasoning holds (an oncoming vehicle's trail lies **beyond** it, so it is the only source sampling road further ahead than the furthest co-directional vehicle) but oncoming traffic passes quickly and churns the source set. Re-measured after the slew and confidence work: confident frames 49.8 -> 72.1 %, error at 70 m p90 12.78 -> 9.25 m, downstream lock and false-lock unchanged. Not yet enabled. |
| Rate limiting the fitted deviation instead of the absolute lateral | **Algebraically a no-op** while the nodes hold absolute laterals: the base arc appears in both the fresh value and the prior and cancels. Bit-identical output. Decoupling requires the nodes themselves to store deviation, and even then a curvature change is a genuine disagreement between the carried road and the new reference, not a bookkeeping artefact. |
| Restricting the cubic on the grid to `support_x_m` rather than `support + 30` | The discontinuity at that boundary is real and large (p50 7.1 m, p90 87 m, max 470 m), and the code gating on `confidence_at(x) > 0` did contradict this section. Fixing it moved nothing measurable: jump p90 0.820 -> 0.834. The far nodes are dominated by the base arc, not the cubic. |
| Gating the `path` in-corridor reward on evidence with no floor | Stationary false-lock 4.3 -> 2.6 % but moving recall 41.5 -> 39.0 %. Too blunt on its own; the arc evidence floor is what makes it affordable. |

Confidence must come from **traffic weight only**, never total weight including
ego. Ego anchors the fit but samples no road ahead, so counting it lets an empty
road report full confidence in a model with zero forward evidence.

### What this model cannot do

Beyond about 70 m **no available estimator resolves a lane width.** Measured
lateral error p50 against ego's own future path, moving targets with a usable
trail:

| band | ego-arc lateral | trail crossing | road model | blend (shipped) |
|------|-----------------|----------------|------------|-----------------|
| 20-45 m | 0.48 | 0.75 | 0.41 | **0.33** |
| 45-70 m | 1.36 | 1.52 | 1.04 | **0.78** |
| 70-100 m | 3.09 | 5.39 | 4.82 | **2.31** |
| 100-140 m | 5.70 | 8.58 | 10.05 | **3.48** |

The blend wins everywhere, but at 70-100 m its error is most of a lane, and
`in_path` is only 54.5 % correct there. That is an input limit, not a tuning
gap. Do not try to make ACC lock reliably past ~70 m; the uncertainty gate in
this section exists precisely to stop it pretending otherwise.

Note the trail crossing is the **worst** estimator at every range, which is not
obvious: it projects a circle fitted to ~40 m of trail back to the ego row, and
that extrapolation degrades faster than the ego arc does. It was measured as a
candidate replacement and rejected.

The fit is **stateless and refitted every frame** in sample space. Coefficients
are never filtered frame to frame: the clothoid parameter space is badly
conditioned (small measurement errors give wildly different `c2`, `c3` for nearly
identical road shapes), so fusing there produces parameter sloshing. The inputs
already span ~1.7 s of history per source, which is where the smoothing comes
from.

### Sources and per-source offset elimination

| Source | Describes | Weight |
|--------|-----------|--------|
| Ego's own recent path | road **behind** ego | 1.0, and it is the reference |
| Co-directional target histories | road **ahead** of ego | trail evidence, halved for TMP |

Only the targets carry preview. Ego's path and yaw rate describe the road at and
behind ego, which is exactly why extrapolating from them failed ahead.

A vehicle one lane over still describes the road's **shape**, just at its own
lateral offset. Each non-ego source therefore has its own offset eliminated
(weighted fixed-effects: its samples and basis are centred on the source's own
mean before the shared solve), so traffic in the next lane bends the model
without dragging it sideways. A source needs `_MIN_SOURCE_SAMPLES` to say
anything once its offset is gone.

### Rejecting vehicles that are not following the road

Using a target as a road sensor assumes it holds a constant offset to the centre.
A lane-changing target violates that, so two robust passes run:

- **Sample-level Huber** (`_HUBER_DELTA_M`) for position noise.
- **Source-level scaling** on each vehicle's own residual RMS
  (`_SOURCE_RESIDUAL_DELTA_M`). A manoeuvring vehicle is inconsistent as a whole,
  not sample by sample; sample-level rejection alone lets the cubic absorb its
  ramp.

### Confidence must measure agreement, not volume

The first version was `weight_term × residual_term`, and the residual term was
**dead**: it is measured after two robust passes have already down-weighted
whatever disagreed, so the surviving rows always fit well. Residual RMS came in
under 0.5 m in **99.8 %** of frames, pinning that term at 1.0. Confidence was
therefore a proxy for how much traffic happened to be nearby, which is exactly
the failure reported from the driver's seat: a car appears, so that must be where
the road goes.

Two consequences were measured, both on the 40-clip sample:

- Confidence sorted p50 and p90 error correctly but was **blind to the tail**.
  At confidence 1.0 the p99 error at 50 m was 19.21 m, against 17.25 m at
  confidence below 0.25. Being certain bought nothing where it mattered.
- With a **single** source the model was worse than the plain ego arc it
  overrides, out to 70 m (at 50 m, p90 3.06 against 2.41; at 70 m, 6.35 against
  5.37), and still reported confidence up to 0.42.

The single-source case is structural, not a tuning miss. Source-level rejection
compares each vehicle against the shared fit, so with one source there is nothing
to disagree with and `_source_scales` returns 1.0 whatever that vehicle does.
**Robust estimation needs a quorum, and nothing was checking for one.**

The signal that fixes it was already being computed and thrown away: the spread
of the per-source residual RMS.

| spread of `source_rms` | error @50 m p50 | p90 |
|---|---|---|
| 0.05-0.2 m | 0.47 | **2.41** |
| 0.2-0.5 m | 0.62 | 2.67 |
| 0.5 m and above | 1.34 | **13.38** |

`_CONF_SPREAD_GOOD_M` / `_CONF_SPREAD_BAD_M` scale confidence by it, and
`_CONF_SINGLE_SOURCE_CAP` bounds the unverifiable case. Effect where the model is
consumed: error at 50 m p99 **20.59 m → 10.41 m**, at 70 m p99 **57.67 → 28.50**.

Keep the band tight. Loosening it to 0.30-1.00 restores coverage (38.8 % to
43.6 %) but gives back the entire accuracy win (50 m p99 back to 20.85 m) for no
measurable downstream gain, so the coverage was never worth having.

The cost is real and deliberate: coverage falls from 49.8 % to 38.9 % of frames
and moving in-corridor recall from 43.2 % to 40.6 %. Those are frames where the
sources disagreed, which is where the estimate measured twice as bad.

### Confidence, and why it must fade with distance

`confidence` comes from total surviving weight and residual RMS.
`confidence_at(x)` additionally decays past `support_x_m`, the furthest sample:
a cubic fitted to samples ending at 60 m says nothing trustworthy at 140 m,
however well it fits what it has. Without that decay the raw model's long-range
error tail was **worse** than the ego arc (p90 35.99 m vs 20.23 m at 90-130 m);
with the confidence blend it is better (19.72 m).

### How the tracker consumes it

    w        = model.confidence_at(x_target)
    lateral  = w · model.offset_of(x, y) + (1 - w) · arc_offset

Blended at the **input**, not added as a fourth score component: the road model
and the arc measure the same physical quantity by two methods, so summing them
double-counts lateral position with two error models and, on a curve, makes half
a correct signal fight half a wrong one. At `w = 0` behaviour collapses to
exactly the pre-road-model tracker, which is what happens on an empty road.

### Lateral uncertainty gate

`lateral_sigma_m(x)` is the **measured** residual error of the blended estimate
against ego's own future path on the clip corpus: 0.65 m at 30-60 m, 2.0 m at
60-90 m, 3.7 m at 90-130 m. A target is only in-lane if its body still overlaps
the corridor after being shifted by sigma **both ways**:

    near = corridor_half - (body_lat_min + sigma)
    far  = (body_lat_max - sigma) + corridor_half
    in_path = near >= 0 and far >= 0

Sigma shrinks the overlap; it is **not** added to a corner lateral, which already
carries the body width. Adding it there rejects a vehicle centred in its own lane.

Only the target's **own trajectory evidence** shrinks sigma
(`sigma · (1 - trail_evidence)`), never the road model's confidence: sigma was
measured on the blended estimate, so the road model's help is already inside it.
Scaling by the blended evidence cancels the gate and put stationary locking back
up from 4.7 % to 8.7 % on the corpus.

Effect on the shipped decision (40 clips, ground truth = ego's own future path):

| | step 2 | step 3 |
|---|---|---|
| stationary false-lock, overall | 5.1 % | **2.4 %** |
| stationary false-lock, 45-70 m | 11.4 % | **1.9 %** |
| stationary false-lock, 70-100 m | 2.4 % | **0.0 %** |
| stationary false-lock, 20-45 m | 7.4 % | 7.4 % |
| moving in-lane recall, overall | 51.0 % | **54.0 %** |
| moving in-lane recall, 45-70 m | 67.2 % | **73.9 %** |

Close range is deliberately unchanged: an unconfirmed stopped vehicle 25 m ahead
must still lock. The gate only removes lane calls the estimate could not support.

### Position evidence vs heading evidence

The road model knows where a target **is** even when the target has no trail of
its own; a trail only ever spoke to where it was **going**. Those feed
`offset_component` separately:

- `evidence = max(trail_evidence, road_weight)` scales the whole offset term.
- `angle_evidence = trail_evidence` gates the arrival-angle penalty.

and the term is **asymmetric**: a positive (in-lane) contribution is scaled by
`angle_evidence`, a negative one is not. Calling a target in-lane needs to know
it is travelling the lane rather than crossing it; rejecting one only needs to
know where it is. Without that asymmetry, `amp = ang_ev·angle_amp + (1 - ang_ev)`
reaches its **maximum** at zero heading evidence, which reintroduced the original
inversion as soon as the road model started supplying position evidence.

---

## 4. Trailer → tractor swap

TMP trailers appear as separate `Vehicle` instances with their own
`position`, `speed`, `acceleration`. Their raw kinematics lag the
tractor (shared-memory pipeline) enough that any downstream gap
controller using them would command phantom braking.

`ACCTracker._top_leads` detects TMP-trailer leads and promotes the
pulling tractor's `speed` / `acceleration` into `LeadInfo.effective_*`.
The vehicle reference itself is unchanged so debug views still show
the trailer: only the kinematics exposed via `LeadInfo` get swapped.

### Nested trailers (road trains)

Only the tractor and the *first* trailer appear as top-level radar
vehicles. Every trailer behind the first is a nested `Trailer` on that
first trailer (AI trucks nest all of theirs the same way), so the
tracker: which iterates Vehicles: never saw them. On a multi-trailer
convoy that made the rearmost trailer, the one ego actually closes on,
invisible to scoring.

`RadarThread` now publishes those nested trailers wrapped as standalone
`Vehicle`s in `RadarData.trailer_vehicles` (see `core/radar/README.md`
§12). `ACCThread` scores them alongside `vehicles`: no tracker change,
they are just more entries in the list. Each carries its own position
history and smoothing (synthetic per-id continuity), so it scores and
locks exactly like a real vehicle, and a wrapped trailer lead still
goes through the trailer→tractor kinematic swap above.

TMP top-level trailers have no buffer parent link. `ACCTracker._resolve_tractor`
locks a tractor with a **strict** gate on first pick (longi 3–16 m, |lat| ≤ 1.5 m,
|Δyaw| ≤ 15°) and a **loose** gate when revalidating a cached pair (longi 1–25 m,
|lat| ≤ 4 m, |Δyaw| ≤ 60°). Among strict candidates, lowest
`|lat| + 0.05·|longi−10| + 0.2·|Δyaw|` wins. Wrapped nested trailers
(`id ≥ 1_000_000`) skip locking and use their own filtered kinematics.

---

## 5. Blinker lateral bias

Blinker state resolves to a signed scalar `b ∈ [-1, +1]` (`-1` full
left, `+1` full right):

- **Pinned at 1** on the indicated side while the blinker is held.
- **Cos decay to 0** over `_BLINKER_HOLD_S = 2.5 s` after release, so
  a short blink still covers the full manoeuvre. Implemented by
  stamping `_last_*_active` every active frame: on release, decay
  starts cleanly at `t = 0`.

The scalar is consumed in two places:
- **Scoring lateral shift**: `offset_for_score = lat - b · 4.5 m`.
  Targets in the indicated adjacent lane score near zero offset during
  the manoeuvre.
- **Path amplitude reduction**: `amp *= 1 - b² · 0.4`. Up to 40 %
  cut so the current-lane lead stops pinning score while we commit
  to the change.

On blinker **rising edge** at ego speed ≥ `_BLINKER_SCORE_RESET_KMH`
(65 km/h), all per-id scores are reset to 0 once. Legacy "highway
lane change" reset: clear the current lead completely so a new lead
can be picked up on the new side without inheriting residual score.

The ego arc itself is **not** translated: doing so would distort
arc-arc hit tests. Shift lives in scoring only.

---

## 6. Published data (`ACCData`)

    @dataclass
    class ACCData(ThreadData):
        enabled: bool            # Settings.acc_enabled & cc_mode == "Cruise control"
        has_lead: bool           # True when leads is non-empty
        lead_id: int             # leads[0].vehicle.id (or -1)
        lead_dist_m: float       # leads[0].dist_m
        lead_rel_speed_ms: float # leads[0].rel_speed_ms (lead - ego; neg = closing)
        lead_score: float        # leads[0].score
        leads: list[LeadInfo]    # top-3 nearest first (score breaks ties), post trailer-swap
        t_mono: float            # radar t_mono the snapshot is tied to
        _lock: threading.Lock

`LeadInfo`:

    @dataclass
    class LeadInfo:
        vehicle: Vehicle            # shared ref: READ ONLY
        score: float                # current accumulated score
        dist_m: float               # longitudinal distance along ego heading
        rel_speed_ms: float         # effective_speed - ego_speed (signed)
        effective_speed_ms: float   # tractor speed when vehicle is a TMP trailer
        effective_accel_ms2: float  # tractor accel when vehicle is a TMP trailer

Consumers always copy what they need under `_lock` and release it
before doing any further work:

    with acc.data._lock:
        leads = list(acc.data.leads)   # shallow list copy: refs are safe
        has_lead = acc.data.has_lead

---

## 7. Thread integration

### ACCThread (`core/acc/thread.py`)

- Runs at 30 Hz (radar cadence).
- Reads `RadarData` (vehicles + ego snapshot) and telemetry blinkers.
- Advances only when `t_mono` changed: paused / stale frames hold
  the previous lead list and skip tracker integration.
- Publishes to `self.data`.

### Registry names

    radar_thread    : produces RadarData.
    telemetry_thread: produces blinkerLeft / blinkerRight.
    acc_thread      : this module; exposes in-lane leads.
    cruise_control_thread: will consume acc_thread.data.leads
                            for gap-based accel control (not yet
                            implemented).

---

## 8. Critical rules

Agent-facing copy of these rules also lives in the top-level `AGENTS.md` (keep that in sync if you change them).

1. **Never mutate Vehicle instances from ACC.** They are shared
   references with AEB; `RadarThread` carries smoothing state forward
   across frames via `update_from_last`. A mutation here corrupts AEB
   next tick.
2. **AEB uses yaw-rate proxy, ACC uses history fit.** Codified in
   `core/aeb/README.md` and `core/radar/README.md` §11. Don't cross
   the streams.
3. **No control law lives in this module.** `core/cruise_control_thread`
   owns longitudinal decisions. If you're tempted to compute an
   accel cap here, stop and put it there instead.
4. **Scoring is meter-native.** New components / tweaks should be
   derived from physical quantities (metres, seconds, m/s), not
   pixels or empirical curves fitted to a specific resolution.
5. **A failed or weak trail fit must never read as high confidence.**
   Both fit-failure paths once set `angle_amp = 1.0`, which made a
   stationary target outrank a slow in-lane one. Scale the offset term by
   `evidence`; do not reintroduce a constant amplitude on the failure path.
6. **The score ceiling must stay near the consumer's confidence
   saturation.** See §3 accumulation. Headroom above `ANT_SCORE_FULL` is
   pure integrator windup and shows up as lead hooking.
7. **Measure changes on the clip corpus.** `tests/acc/` holds the replay
   harness and the recorded baselines. A tracker change with no corpus
   number attached is not reviewable.
8. **Never let road-model confidence shrink the lateral uncertainty gate.**
   `lateral_sigma_m` was measured on the blended estimate, so the road model's
   contribution is already inside it. Scaling the gate by the blended evidence
   cancels it and puts stationary locking back where it started (§9).
9. **The offset term is asymmetric on purpose.** A positive contribution is
   scaled by heading evidence, a negative one is not. Making it symmetric
   restores the original failure the moment any source supplies position
   evidence without heading evidence (§9).
10. **Do not filter road-model coefficients frame to frame.** The clothoid
    parameter space is ill-conditioned; fuse in sample space and refit (§9).

