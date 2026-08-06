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

**Evidence is bounded by how much trail the vehicle is allowed to keep**, which
is `_TRAIL_MAX_AGE_S` in `core/radar/traffic.py`, and at 2.0 s that bound was
the binding constraint on slow traffic: a vehicle at 5 km/h kept 3.5 m of trail
and earned evidence 0.35, so it scored weakly and latched slowly even when
squarely in the estimated lane. Reported from the driver's seat as slow traffic
reading as stationary. At 6.0 s the same vehicle keeps 8.8 m and earns 1.00.
The measurement, and why this overturns an earlier rejection, is in
`core/radar/README.md` §7.

Note `_VALIDATE_MIN_SPEED_MS` is 2.0 m/s, so a vehicle that never exceeds
7.2 km/h can never set `moving_validated` and has no route to the stationary
rescue. Below that speed its score rests entirely on `trail_evidence`, which is
why the retention cap governed it so completely.

#### Corroboration between slow vehicles (`core/acc/corroboration.py`)

`evidence = max(trail_ev, road_w, corroboration, _ARC_EVIDENCE_FLOOR)`. Measured
on the corpus, over half of all slow traffic ahead falls through to the floor,
meaning nothing knows where it is:

| where a slow vehicle's evidence came from | share of 20,031 |
|---|---|
| its own trail | 30.5 % |
| the road model at its position | 17.7 % |
| **`_ARC_EVIDENCE_FLOOR`, i.e. nothing** | **51.8 %** |

Of those floor-limited ones the road model was confident anywhere in only
25.4 %, which is the tension in one line: **context is worth most exactly where
the road model is silent, which is where you can least check the line is ego's
road.** 8.4 % of them are in ego's corridor, and 66.6 % of those have three or
more aligned slow neighbours, so the corroboration is usually available.

A group earns evidence only if it is aligned to the road tangent (25 deg),
mutually parallel (15 deg spread), holds one lane (2 m offset spread), spans
12 m, and has three members. Those gates are the ones measured at **90.8 %** of
accepted members genuinely on ego's road, against 8.9 % for stationary traffic
unfiltered. It fires in **4.3 %** of frames, closely matching the 4.8 % the same
filter reached when it was tried as a road source.

Three properties that make it affordable:

- **It raises how well a position is known, not how much it looks like a lead.**
  `offset_component` scales its whole range by evidence, negative included, so a
  corroborated vehicle 20 m off the centreline is rejected *harder*. The risk is
  confined to vehicles near the corridor, not the whole stationary population.
- **It never feeds the road fit.** Road-model error is bit-identical with it on
  and off. That is deliberate: a queue used to define the road, then scored
  against that road, is the circularity that sank the queue **source** (see the
  rejected table in §9). This consumes the same grouping without closing a loop.
- **It cannot outrank direct observation.** Capped at
  `_VALIDATED_STATIONARY_EVIDENCE` (0.75), and at 0.60 when no confident road
  model anchors it. Inference from other vehicles beating having watched *this*
  vehicle drive its own line would invert the evidence hierarchy.

Measured effect, slow (< 2 m/s) in-corridor targets: lock rate 30.7 -> 31.7 %,
latch p50 1.48 -> 1.40 s, p90 unchanged; ACC lock p90 overall 2.42 -> 2.15 s.
Cost is stationary false lock 4.2 -> 4.4 %. **That is a small benefit and the
corpus cannot show a larger one**: a queue member becoming the correct lead is
rare in highway clips, which is the same limit that stopped the queue source
being judged fairly. Town and junction clips would settle it.

There is deliberately **no penalty arm.** Lowering confidence for a vehicle that
disagrees with its neighbours inverts on the case that matters most: twenty
queued on the hard shoulder and one stopped in the live lane makes the hazard
the outlier. If it is ever added it must only reduce confidence for vehicles
already outside the corridor, which is close to a no-op.

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

### Accumulation: `[-5, +6]` clamp

    weighted  = offset·1.5 + yaw + path·0.7 + angle
    delta     = weighted · speed_mult(v_target) · dt · 10
    score_new = clamp(score_prev + delta, -5, +6)

- Asymmetric clamp: fast lock on in-path vehicles, slow release on
  uncertain tracks (narrow negative floor).
- **The ceiling tracks the consumer, not the tracker.**
  `AdaptiveCruiseController` saturates its confidence ramp at score 5
  (`ANT_SCORE_FULL`), so score above that buys nothing except release
  latency. At the old ceiling of 20, half of all positive-score frames sat
  pinned at the clamp and a lead leaving the lane took a p90 of 2.45 s to
  fall back under confidence. At 8 that was 1.30 s and at **6 it is 1.06 s**,
  with lock p50 and p90 both bit-identical and recall within 0.1 points, because
  the headroom was never doing anything but delaying release. Raising the ceiling
  again re-creates the hooking. Enforced by `tests/acc/test_scoring_evidence.py`.
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

    n(s) = c1·v + c2·v² + c3·v³ + c4·v⁴      v = s / 100 m
    κ(s) = base_kappa − (2·c2 + 6·c3·v + 12·c4·v²) / 100²
    centreline(s) = arc_point(base_kappa, s) + n(s)·arc_normal(base_kappa, s)

`s` is **arc length** along the base arc and `n` is offset along its normal,
positive to ego's right, anchored so `n(0) = 0`. `c2` is curvature at ego, `c3`
is **curvature rate**, which is what the old constant-curvature arc lacked and
why it lost in-lane leads past ~40 m on curves, and `c4` lets curvature **bend**
rather than only ramp. See the corner-entry section below for why the clothoid
cubic alone was not enough.

**It used to be `y(x)`, indexed by forward distance, and that is a graph.** A
graph cannot describe a road that turns more than 90 deg, because the road then
revisits `x` and every point has two answers. The failure did not wait for 90
deg either: see the angle-ceiling section below for the four separate walls that
produced it. Arc length has none of them, and it is a coordinate change only,
so the fit stays one weighted solve.

Callers holding a 2D point should use `road_coords(x, y)`, which returns
`(arc length, offset right of the centreline)` and is defined all the way round
a bend. `lateral_at(x)` still exists for drawing and tests and still saturates,
because a forward distance is what is ill-defined, not the model.

The one limit left is that a circle closes: arc length is periodic, so at
exactly `pi·R` the far end sits diametrically opposite ego and the sign of `s`
is genuinely ambiguous. `arc_span_limit` stops just short, which costs nothing
on a road and correctly gives almost no span to ego steering hard at a
standstill, where `blend_curvature` reports R = 7 m. **That guard is
load-bearing**: without it those samples aliased onto the near side of their own
circle and the fit answered with a 4 km deviation.

Measured against ego's own future path, 40 clips, forward distance to arc length:

| | forward `y(x)` | arc length `n(s)` |
|---|---|---|
| confident frames | 58.9 % | **62.2 %** |
| lateral error @50 m p99 | 24.86 m | **7.42 m** |
| @70 m p90 / p99 | 5.18 / 38.96 | **4.43 / 14.48** |
| @100 m p90 / p99 | 8.86 / 47.95 | **7.93 / 24.85** |
| slew saturation | 24.2 % | **20.7 %** |

At matched coverage it is better on the frames both reach (@70 m p99 18.69 ->
14.48) and, more to the point, the frames it **stops** being confident on were
the worst ones the old model had: median error 10.10 m at 50 m and 32.64 m at
100 m. The extra coverage it buys is ordinary by comparison, p50 1.34 m at 50 m.

Two things this cost, both real. Moving in-corridor recall 43.9 -> 42.7 %, and
lead release, because a centreline that is right holds a lead in corridor longer
than one that is wrong. `_PATH_OUT_GAIN` was raised 0.6 -> 1.0 to pay for the
second (hook p90 1.44 -> 1.20 s, lock p90 2.41 -> 2.45). The score ceiling is
**not** available for this: `SCORE_MAX >= ANT_SCORE_FULL` is a contract with the
cruise-control confidence ramp, and 5.0 is not low enough to help anyway.

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
carry-over is done in **sample space**: the previous frame's nodes on `_NODE_S`
are rebuilt as world points under the base arc they were fitted against, read
back under this frame's, and resampled before being combined, so ego's own
motion drops out rather than being smoothed as if it were road change.
A curvature change does not drop out, and should not: the nodes carry a
deviation, so a new base arc makes the carried road genuinely far from it,
and the rate limit walks it in.
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

`node_slew_budget_ms(s)` therefore spends `_SMOOTH_MAX_KAPPA_RATE` of curvature
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

Nodes past `support_s_m` carry the base arc, not the extrapolated cubic. Feeding
untrusted extrapolation into the carry-over made the 100 m outliers three times
worse, because a bad value then persisted instead of being a single frame.

#### Re-acquisition is not a step to suppress

A confident fit arriving on top of a carried estimate of *nothing* is the first
good answer, not a jump away from a good one. Rate limiting it publishes the
bare base arc while reporting the fit's confidence for as long as the slew takes.
Measured over the first half second after confidence returns, lateral error at
50 m p90 4.32 -> 3.93 m.

**The transient is not fixed, only reduced.** Error stays roughly twice steady
state for about two seconds after re-acquisition (p50 0.98 against 0.43), and
the tail is worse than that:

| since confidence returned | p50 | p90 | p99 |
|---|---|---|---|
| 0.0-0.5 s | 0.98 | 3.93 | **54.24** |
| 1.0-2.0 s | 0.64 | 2.05 | 5.90 |
| steady state | 0.43 | 2.26 | 5.39 |

That p99 rose from 15.08 m with the longer trail retention, and the rise is
**entirely** the retention: it is identical with and without the snap above.
The likely cause is a source arriving with 40 m of trail describing a line it
has already left, which one source alone can carry into the fit. Not chased
further; `_CONF_SINGLE_SOURCE_CAP` bounds what a lone source can claim, and the
bucket is about eight frames in 12,608.

### Things measured and rejected

Recorded so they are not retried blind. All on the same 40-clip sample.

| Tried | Result |
|-------|--------|
| Raising the ego-sample weight (2, 4, 6) | **No effect at all.** Once the base arc carries ego's curvature exactly, ego's own samples are redundant with it. Left at parity. |
| Lateral cap on sources, per sample | Stationary false-lock 2.3 % -> 4.1 %. Truncating one source's trail biases its own mean, which the offset elimination feeds back as shape. |
| Lateral cap on sources, per source | Still 4.1 %. |
| Lateral cap measured against the base arc | 4.1 % and recall 56.9 -> 55.9 %. On an R200 bend an **in-lane** vehicle 120 m ahead is 40 m laterally in the straight frame, so any lateral cap preferentially deletes the far-field sources that carry the curve. Dropped entirely. |
| Plausibility clamp on cubic deviation | Rejects legitimate **curve entry**, where ego is still straight while the road ahead bends, so the deviation is genuinely large exactly when the model matters. Reverted. |
| Oncoming traffic as a road source | Rejected on its first measurement (centreline jump p99 at 50 m 1.33 -> 1.79 m) and **that rejection was wrong**: the churn it was blamed for was the IRLS weight compounding below, not the oncoming traffic. Now enabled; see the oncoming subsection above. |
| Loosening the centreline slew limit to speed up reaction | Latch time is **byte-identical** at every setting from current to fully off, so the road smoothing does not gate reaction at all. Loosening flatters jump p90 (0.67 -> 0.48 m) while making p99 4.7x worse and the max 11x worse (3.36 -> 37.84 m). A limiter is judged on its tail. |
| Shortening the agreement low-pass, or removing the confidence rate limit, to speed up reaction | Both made latching **slower**: lock p90 2.68 -> 4.89 s at tau 0.05, and 2.68 -> 3.53 s with the rate limit off. |
| Speeding up the per-source trust ramp | Lock p50 identical at 0.81 s for every tau tried, and cut-in latch got worse (1.10 -> 1.29 s). Nothing to gain. |
| Raising `_PATH_OUT_GAIN` to release a departing lead faster | Works (hook p90 1.30 -> 1.13 s) but costs lock p90 (2.29 -> 2.36 s). Lowering the score ceiling does the same job for free. |
| Rate limiting the fitted deviation instead of the absolute lateral | **Algebraically a no-op** while the nodes hold absolute laterals: the base arc appears in both the fresh value and the prior and cancels. Bit-identical output. Decoupling requires the nodes themselves to store deviation, and even then a curvature change is a genuine disagreement between the carried road and the new reference, not a bookkeeping artefact. |
| Restricting the cubic on the grid to `support_x_m` rather than `support + 30` | The discontinuity at that boundary is real and large (p50 7.1 m, p90 87 m, max 470 m), and the code gating on `confidence_at(x) > 0` did contradict this section. Fixing it moved nothing measurable: jump p90 0.820 -> 0.834. The far nodes are dominated by the base arc, not the cubic. |
| Gating the `path` in-corridor reward on evidence with no floor | Stationary false-lock 4.3 -> 2.6 % but moving recall 41.5 -> 39.0 %. Too blunt on its own; the arc evidence floor is what makes it affordable. |
| Range (`max - min`) of per-source residuals as the agreement statistic | Zeroed confidence at five or more sources: corroboration made it worse. Shipped in 0180e4b and reported from the driver's seat as the prediction fading in and out. Replaced by a quantile. |
| Residual of the initial unrobust fit as the agreement statistic | Correct that post-IRLS statistics are laundered, but an RMS over all rows is not robust: one source drifting 1 m dragged the fit until every source read wrong, 0.77 -> 0.00 confidence. Best p90 at matched coverage, worst dissenter tolerance. |
| Weighted consensus share (fraction of source weight agreeing) | Count-stable and flicker-free, but measured **after** the robust passes, so it reads ~1.0 even on a fit captured by a minority. Raising it to a power changed nothing at all: a no-op, which is how the laundering was confirmed. |
| Share of source weight the robust passes discarded | Real discrimination (p50 error 0.57 -> 5.47 across its range) but strictly worse than the quantile at matched coverage, and redundant once IRLS stopped compounding weights. |
| Stationary vehicles as a road source, unfiltered | Only **8.9 %** of stationary traffic within 170 m sits within 5.5 m of the road ego goes on to drive, median offset **23.8 m**, and headings match the road tangent within 5 deg only 8.8 % of the time (median error **53 deg**). What the radar reports as stationary is mostly car parks, service areas, depots and side streets. |
| A queue as a road source, with all three filters | **Built and reverted.** A real queue *is* separable: requiring the group to be aligned with the road (25 deg), mutually parallel (15 deg spread) and collinear once the base arc is removed leaves **90.8 %** of members genuinely on ego's road, in 4.8 % of frames. It still does not work, and the reason is not the filter. Where a queue was accepted and confident, the road model including it was **worse than the bare ego arc it replaced** (error at 50 m p50 0.59 against 0.44, p90 1.25 against 0.96, better in only 55 % of frames): three to six points spanning 40 m is a far weaker shape constraint than one vehicle's 40 m trail, and the queue enters exactly where the model was already weakest. Every weight from 1 to 8, with and without keeping members' own trails, measured worse than off. The circularity also showed up as predicted, stationary false-lock 3.28 % -> 3.39-3.97 %, since a stopped vehicle used to define the road then scores as being on it. Restricting it to a pure fallback removed that cost but still added only worse-than-arc estimates. **Do not rebuild this without a corpus containing real jams**: only 6 of 12,608 frames had a queue as the sole traffic, so the corpus can measure this idea's cost but not its benefit. |
| Damping the re-base part-way toward ego's curvature | Strictly worse at 0.7 than committing fully: R < 80 coverage 49.7 -> 42.1 %, cut-in p90 2.87 -> 2.94 s, centreline jump p99 at 30 m 1.87 -> 4.42 m. Damping scales the vote's noise and its signal by the same factor, so it buys no stability, and the mismatch it deliberately leaves behind is exactly the coupling the quartic then has to spend its freedom undoing. The constant was removed rather than left at 1.0. |
| A ridge on `c4` | Its response is a cliff rather than a damper: after per-source offset elimination the `v⁴` diagonal is order 0.03, so every value from 0.03 to 8.0 pins the term outright and only exactly zero frees it, while `c4` reaches -40 to -165 on a real corner. Left free. Not retried on the re-based `base_kappa`, which is not fitted and so has no diagonal to ridge. |
| Median of per-source residuals | Too permissive: p99 at matched coverage ~15 m against 9.6-10.1 for the 75th percentile. The right amount of pessimism is an upper quantile, not a central one. |
| Correcting the agreement quantile index | `agreement_residual_m` indexes `int(0.75 · n)` clamped to `n − 1`, which **is the maximum** for every source count up to four, and that is most real scenes. Indexing `n − 1` instead makes it a genuine quantile that drops one dissenter from three or more. Measured on top of `c4`: coverage up again (R 80-200 48.4 -> 54.4 %, R < 80 29.0 -> 33.2 %) but cut-in lock p90 2.31 -> 2.81 s against a 3.0 bound and recall 45.4 -> 43.9 %. It costs more than `c4` did and buys less. **Reverted, and the docstring now says the index is deliberate**: with few sources the pessimistic reading is the protective one, and `_CONF_SINGLE_SOURCE_CAP` only covers `n = 1`. Retry only alongside something that improves far-field accuracy. |
| Ego curvature from yaw rate instead of the position circle fit | `kappa = dpsi/ds` regressed over the path is better *conditioned* than a three-point circle fit, and it is still **worse**, because it is a trailing estimate either way and the window needed to condition it costs more lag than it buys noise. Measured against a forward reference (heading change over the next 20 m of path), p90 error 0-20 km/h: chord fit **0.0586**, yaw regression 0.1032, distance-capped 40 m window 0.1301. The shipped fit wins in every speed bucket. Note a first pass measured against a *centred* reference and got the opposite answer: a trailing estimator matches a centred reference by lagging, so only a forward reference can rank these. |
| Distance-capping the ego position history instead of the 25-sample cap | Same table. Lengthening the window monotonically worsens the forward-reference error at every speed. Also moot below 15 km/h, where `blend_curvature` gives history curvature weight **0.00** and the corridor is `steer * 0.17` alone. |
| Letting the fit choose `base_kappa` instead of inheriting ego's | **Built and reverted**, patch kept. The mechanism is real: the implied curvature converges to the truth (0.0051 against 0.0050) once damped, and on a synthetic R200 corner entry at 60 deg of sweep it takes the estimate from 26.79 m error at zero confidence to 0.84 m at 0.85. On the corpus it is **exactly nothing** where it matters: at matched coverage the 30 m percentiles are identical and 50-100 m are within noise. All it adds is +5.8 points of coverage whose median error is *outside the model's own stated sigma* at the range ACC locks (0.77 m against 0.25 at 30 m, 1.30 against 0.955 at 50 m). It costs lead release, hook p90 **1.026 -> 1.338 s**, breaching the corpus bound, plus cut-in p90 2.212 -> 2.420 and 410 -> 1567 us of fit. Four gates were tried to separate the benefit from the cost (source count, cost margin, a confidence haircut on re-based fits, minimum curvature disagreement); **three are exact no-ops on hook p90** and the only setting that clears the bound clears it by 0.009 s while cutting the synthetic win from 8 of 16 cases to 5. The cost comes from the firings that are wanted, not from marginal ones. Part of its value is compensating for the forward-distance parameterisation, so retry it only *after* arc-length, and only against a corpus with real corners. **Superseded**: retried as instructed and shipped, but by raw trail geometry rather than by fitting. See the re-basing section. |

Confidence must come from **traffic weight only**, never total weight including
ego. Ego anchors the fit but samples no road ahead, so counting it lets an empty
road report full confidence in a model with zero forward evidence.

### The angle ceiling: why tight corners used to fail

Reported from the driver's seat as the estimate cutting off and losing tracking
in low-speed tight corners. There were four separate walls, all of them
consequences of indexing by forward distance, and all removed by the arc-length
parameterisation above. Kept because the diagnosis is what justifies that
change, and because anything reintroducing a forward-distance index gets them
all back.

**`_BASE_ARC_MAX_FRAC` clamps the arc at 0.95 of its radius, which is 71.8 deg
of heading change.** Past that `base_arc_lateral` returns a frozen value, a flat
line, with no confidence penalty. That injected residual is unexplainable, IRLS
reads it as a lying source, and confidence collapses. Fed a perfect circle with
`base_kappa` exactly right and one noiseless source:

| R | sweep | agreement rms | confidence |
|---|-------|---------------|-----------|
| 100 | 72 deg | 0.000 | 0.25 |
| 100 | 75 deg | 0.348 | 0.25 |
| 100 | **80 deg** | 0.963 | **0.00** |
| 25 | **85 deg** | 0.702 | **0.00** |

**`y(x)` is a graph, so past 90 deg the road revisits `x`.** Road points become
two-valued and the fit sees genuine geometry as disagreement. Worse,
`_ROAD_SAMPLE_MIN_X_M` is -30 m, so a vehicle 200 deg around a 25 m bend arrives
at `sx = -8.5` and is accepted as a sample *behind* ego. A half circle is not
representable at any tuning.

**`base_kappa` is ego's past.** With it correct the cubic is exact to 72 deg;
with it zero, which is what corner entry hands it, error at 60 deg of sweep is
already 0.53 m at R100 and 29.29 m at 90 deg. No trailing estimator fixes this
(see the rejected table above): the traffic has to carry it.

**Support was forward `x`, not arc length.** At R25 with 60 deg of road
visible, 26 m of road became a support of 21.6 m. The ratio is
`sin(theta)/theta`, so 0.64 at 90 deg and 0.30 at 135 deg, and confidence
faded 30 m past that. It is `support_s_m` now, in arc length.

One hypothesis measured and dropped: `_prior_on_grid` sorts by `x` before
resampling, which would scramble a folded prior. It never folds. The prior is a
graph by construction and one frame of yaw is 2.3 deg at R25, nowhere near the
87 deg tangent needed. The smoother's de-rotation is already correct.

### Corner entry: the fifth wall, and why `c4` exists

Arc length removed the four walls above but left a fifth, reported from the
driver's seat the same way: traffic round a bend stops being tracked at roughly
45 deg, with oncoming traffic present and confirming the road.

**A corner is a curvature step and the cubic's curvature is affine in `s`.**
With `c1` pinned by the heading prior the model has exactly two free terms to
describe the road ahead, which is one clothoid: curvature may ramp linearly and
nothing else. A real corner has a definite station where the bend starts. Best
achievable weighted residual on **noiseless** samples of a straight-then-R80
road, `c1` pinned, against `_CONF_RESIDUAL_BAD_M` of 0.60:

| visible bend | c2,c3 | c2,c3,c4 | c2..c5 |
|---|---|---|---|
| 30 deg | 0.341 | 0.126 | 0.125 |
| **45 deg** | **0.492** | 0.210 | 0.138 |
| 60 deg | 0.493 | 0.422 | 0.138 |
| 75 deg | 1.029 | 0.965 | 0.306 |

So at 45 deg the cubic's own bias already ate 80 % of the confidence budget with
nothing wrong with the evidence. Source-level IRLS then compounds it: the misfit
is largest where the road is most bent, so the sources reaching furthest round
the corner score worst and are the first discarded. Oncoming traffic is exactly
that source, which is why the case with the **most** corroboration failed. On a
noiseless R80 corner at 45 deg of sweep the co-directional lead read 0.248 and
the oncoming source 3.406, and the fit kept the lead and threw away the corner.

`c4` is the smallest term that buys a curvature step. Measured over 60 clips,
fraction of frames with any road confidence, bucketed by ego curvature:

| ego curvature | cubic | with `c4` |
|---|---|---|
| straight | 73.8 % | 76.9 % |
| R > 500 | 78.1 % | 82.9 % |
| R 200-500 | 65.6 % | 69.9 % |
| **R 80-200** | **35.7 %** | **48.4 %** |
| **R < 80** | **19.2 %** | **29.0 %** |

**It is not free, and the cost is lock latency, not accuracy.** Cut-in lock p90
2.08 -> 2.31 s (bound 3.0), fresh-id lock p90 0.98 -> 2.82 s, moving in-corridor
recall 46.6 -> 45.4 % (floor 38 %). Hook p90 is unmoved at 1.264, hook p50
improved 0.478 -> 0.410, and rank mismatches fell 279 -> 263. Every recorded
baseline stays green. The mechanism is coverage, not wobble: the model is
now confident in frames where it previously abstained, and in those frames it
overrides the ego arc, which at 70-100 m is the better estimator (see the table
in "What this model cannot do"). Tightening the slew limit was measured and
recovers the jump but **not** the latency, confirming the same insensitivity the
rejected table records.

**No ridge on `c4`**; see the rejected table.

`c4` roughly doubles the centreline's frame-to-frame jump (p90 at 30 m 0.126 ->
0.306 m, p99 1.07 -> 3.00 m); the p99 tail is re-acquisition, which bypasses the
slew limiter by design.

### Re-basing the arc onto the traffic

`c4` moved the wall from 45 deg to about 60, and left the fold beyond it: on
corner entry `base_kappa` is ego's trailing curvature, which is **zero**, so arc
length degenerates back to forward distance and every wall in the section above
comes back. Ego's curvature describes the road at ego. The road 100 m ahead is a
whole bend away from it.

`_rebased_kappa` lets the traffic choose the base arc. Each qualifying source
votes with the curvature of the least-squares circle through **its own** samples,
and the median wins. Three properties matter, and all three were arrived at by
getting them wrong first:

- **Its own trail, not a bearing from ego.** A chord from ego to a vehicle is
  biased by that vehicle's lane offset, so traffic sitting in the right-hand lane
  of a dead straight road reads as a right-hand bend. A source's own trail is
  parallel to the road, so the offset cancels; all that survives is the radius
  difference between lanes, 4 % at R80.
- **Least squares over the whole trail, not three points on it.** The
  three-point circle reads the road off two gaps, jitters frame to frame, and the
  base arc hands that jitter to every lateral at once.
- **Only sources the fit itself would accept.** Gated on `_MIN_SOURCE_SAMPLES`
  and on the source's own span, not its range: the vote is that trail's
  curvature, so a short trail is the noisy one wherever it sits.

This is the rejected table's "letting the fit choose `base_kappa`" retried after
arc-length, as that entry said to. It is **not** the same mechanism: nothing
searches for a base that minimises residual. That objective is degenerate, and
measurably so, since on a noiseless R80 corner at 45 deg of sweep the argmin
picks R18 while the geometry says R80. Raw geometry beats fitting here.

One thing had to go with it. **Ego's own samples are dropped whenever the arc is
re-based**, because they lie on the base arc only while the base is ego's own
curvature, and otherwise they fight the traffic for the quartic's freedom. They
cost nothing: `n(0) = 0` is structural, since the basis has no constant term, so
ego anchors nothing that the parameterisation was not already anchoring, and the
rejected table already records that ego's weight changes nothing. With them in,
a noiseless R80 corner at 75 deg still read 0.620 agreement and zero confidence.
With them out it reads 0.000 and 1.00.

Confidence on a noiseless straight-then-bend corner, ego at the bend start:

| visible bend | 30 deg | 45 deg | 60 deg | 75 deg | 90 deg | 120 deg |
|---|---|---|---|---|---|---|
| cubic | 0.59 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 |
| `c4` | 0.98 | 0.69 | 0.04 | 0.00 | 0.00 | 0.00 |
| `c4` + re-based | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

That holds at every radius from 20 m to 500 m. Corpus coverage, fraction of
frames with any road confidence:

| ego curvature | cubic | `c4` | `c4` + re-based |
|---|---|---|---|
| straight | 73.8 % | 76.9 % | 77.6 % |
| R > 500 | 78.1 % | 82.9 % | 83.4 % |
| R 200-500 | 65.6 % | 69.9 % | 72.2 % |
| **R 80-200** | **35.7 %** | 48.4 % | **57.9 %** |
| **R < 80** | **19.2 %** | 29.0 % | **49.7 %** |

The two are complementary rather than redundant, which was measured rather than
assumed: re-basing on a cubic gives 48.5 % and 40.1 % in those two buckets, so
each buys roughly independent coverage and the pair is additive. `c4` still earns
its place because a re-based arc carries a road that **is** one arc, and a bend
that arrives and then leaves is not.

Costs, against the cubic baseline: cut-in lock p90 2.08 -> **2.87 s** against a
3.0 bound, fresh-id p90 0.98 -> 2.41 s, moving recall 46.6 -> 42.3 % (floor 38).
Gains: hook p90 1.264 -> 1.224, hook p50 0.478 -> 0.339, rank mismatches 279 ->
243, zero-confidence overlay trips 4.06 -> 3.10 %. Every recorded baseline stays
green, but **cut-in p90 is now the binding one** with 0.13 s of headroom, where
it had 0.92 s. Anything that widens road-model coverage again has to buy that
back first.

Note this is the opposite sign to the reverted attempt, which cost hook p90
(1.026 -> 1.338, breaching the bound). Releasing a departing lead got *better*
here, and the likely reason is that the earlier version re-based by fitting, so
it was wrong in a way that dragged the corridor with it.

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
| Ego's own recent path | road **behind** ego | 1.0, and it is the reference, but only while the base arc is ego's own curvature (see re-basing) |
| Co-directional target histories | road **ahead** of ego | trail evidence, halved for TMP |
| Oncoming target histories | road **beyond** them | as above, times `_ROAD_SAMPLE_ONCOMING_WEIGHT` |

Only the targets carry preview. Ego's path and yaw rate describe the road at and
behind ego, which is exactly why extrapolating from them failed ahead.

Traffic between the two heading bands (`_ROAD_SAMPLE_CODIR_DEG` to
`_ROAD_SAMPLE_ONCOMING_DEG`) is turning off and describes no road ego will drive,
so it contributes nothing.

#### Oncoming traffic

An oncoming vehicle drives the same road in the other direction, and the sign of
its heading changes nothing about the geometry it carries: the per-source offset
elimination removes its lane offset exactly as it does for an adjacent lane. It
is also the **only** source whose trail lies further ahead than the vehicle
itself, so it reaches past the furthest co-directional target.

Without it, one vehicle ahead means one source, which the single-source cap holds
at 0.25, and a road full of opposite traffic contributed nothing at all. Fraction
of frames with a usable estimate at 50 m, by what traffic is present:

| scene | co-directional only | with oncoming |
|---|---|---|
| oncoming only, no co-directional | **0.2 %** | **40.2 %** |
| one co-directional, plus oncoming | 4.2 % | **37.9 %** |
| two or more co-directional | 53.6 % | **65.4 %** |

Accuracy improves at the same time rather than trading against availability: in
the one-co-directional case, error at 50 m p90 falls from 3.42 m to 1.40 m.

Weighted at parity with co-directional traffic. The reason to discount it would
be that on a divided highway the opposite carriageway is a **concentric arc at a
different radius**: offset elimination removes the separation but not the
curvature difference, about 1.7 m at 100 m for an R200 bend with a 15 m median.
That cost did not appear. At parity, against the co-directional-only model:

| | co-dir only | oncoming at parity |
|---|---|---|
| frames with any confidence | 41.1 % | **58.9 %** |
| lateral error @50 m p90 | 3.48 | **2.44** |
| lateral error @70 m p90 | 7.51 | **5.18** |
| lateral error @100 m p90 | 15.26 | **8.86** |
| moving in-corridor recall | 41.5 % | **43.9 %** |
| lock p50 | 0.99 s | **0.89 s** |

The cost is **source churn**: oncoming traffic passes quickly, so the source set
turns over faster and confidence moves more. Slew saturation 17.5 -> 24.2 % and
confidence peak-to-peak over 1 s p90 0.41 -> 0.61. Measured at half weight as
well, and the churn is the same there (p90 0.51), so it is the price of using
oncoming at all rather than of the weight. That is what the first attempt at this
source misread as a reason to reject it.

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

**Each pass must rescale the original weights, never the previous pass's.**
Compounding them (`w * scale` accumulated across passes) makes the down-weighting
monotone and irreversible, so a first fit dragged by one dissenter craters every
source and no later pass can undo it. Measured on four sources fitting a
synthetic road exactly, adding one vehicle drifting 0.5 m:

| | surviving source weight | confidence |
|---|---|---|
| compounding (was) | 24.00 → **1.48** | 0.77 → **0.00** |
| rescaling originals | 24.00 → 24.01 | 0.77 → 0.77 |

A 94 % weight collapse from a correctly-rejected dissenter, which drops total
weight under `_CONF_WEIGHT_MIN` and zeroes confidence. This was the dominant
cause of the estimate dropping out in traffic, ahead of the choice of agreement
statistic above. `tests/acc/test_road_model.py` pins it at four drift sizes.

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

The signal that fixes it was already being computed and thrown away: the
per-source residual RMS. **Which statistic you reduce it with is the whole
result**, and the first two choices were both wrong in instructive ways.

#### Never a range, never an RMS over all rows

The first version used the **range** (`max - min`) of the per-source residuals.
A range has a breakdown point of zero and its expected value grows with the
number of samples, so more corroborating traffic made it worse. Measured:

| sources | 2 | 3 | 4 | 5 | 7 | 9+ |
|---|---|---|---|---|---|---|
| confidence p50 | 0.42 | 0.77 | 0.60 | **0.00** | **0.00** | **0.00** |

Five vehicles in view zeroed the estimate. From the driver's seat that is the
prediction fading in and out exactly when the traffic to support it is there.

The second attempt used the **residual of the initial unrobust fit**, on the
argument that anything measured after the robust passes is laundered by them.
That argument is right, but an RMS over all rows is not robust either: one
dissenter drags the initial fit, and then *every* source reads as wrong. A
vehicle drifting 1 m over 120 m took confidence from 0.77 to 0.00.

What works is an **upper quantile** (`_CONF_AGREE_QUANTILE`, `agreement_residual_m`).
It is count-stable, and it survives a minority of dissenters while still rising
when the majority disagree, which is the capture case that matters. Compared at
**matched coverage** (see below), error p99 at 50 m:

| statistic | 25 % | 40 % | 55 % | 70 % |
|---|---|---|---|---|
| weight only | 11.22 | 15.81 | 15.50 | 15.19 |
| range | 9.93 | 10.13 | 10.10 | 10.22 |
| initial-fit RMS | 9.93 | 14.86 | 14.00 | 13.29 |
| median | 15.72 | 15.12 | 14.86 | 14.37 |
| **75th percentile** | **9.93** | **10.10** | **9.79** | **9.61** |

**Compare confidence signals only at matched coverage.** Percentiles conditioned
on `confidence > 0` are not comparable between candidates: a signal that fires
less often is scored on a smaller and more selective set, which flatters it for
free. Rank the frames by each candidate and take a fixed top fraction.

The quantile is low-passed (`_CONF_RESIDUAL_TAU_S`) before it reaches confidence.
Smooth the measurement, rate limit the decision: they are separate jobs, and
skipping the first put confidence reversals at p90 5.0/s.

### Confidence, and why it must fade with distance

`confidence` comes from total surviving weight and residual RMS.
`confidence_at(s)` additionally decays past `support_s_m`, the furthest sample:
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

