# AEB (Automatic Emergency Braking)

> AEB-specific reference. Shared fundamentals (coordinate system, rotation,
> world→ego transforms, the shared-memory buffer, Vehicle state/smoothing,
> ArcPath geometry, position-based curvature) live in
> `core/radar/README.md`: read that first.
>
> Agent workflow and do-not-break rules: top-level `AGENTS.md`.

---

## 1. Data pipeline

AEB is a consumer of `RadarThread`:

```python
rt = registry.get_thread("radar_thread")
with rt.data._lock:
    vehicles      = rt.data.vehicles
    tmp_session   = rt.data.tmp_session
    ego_x, ego_y, ego_z = rt.data.ego_x, rt.data.ego_y, rt.data.ego_z
    ego_yaw_rad   = rt.data.ego_yaw_rad
    ego_speed     = rt.data.ego_speed
    ego_pitch_rad = rt.data.ego_pitch_rad
    ego_steer     = rt.data.ego_steer
    ego_curvature = rt.data.ego_curvature      # may be None
    paused        = rt.data.paused
```

Vehicle instances are shared references: do not mutate them from AEB. All
smoothing, yaw EMA, position history, and curvature state is produced once
per frame by the radar thread.

### TMP trailer-as-vehicle kinematic swap

In TMP sessions, other players' trailers appear as separate top-level radar
vehicles (`is_tmp=True`, `is_trailer=True`). The shared-memory speed field for
those slots has no engine telemetry: it commonly reports 0: and the
position-history LS fit can also return 0 during transients (small chord,
stall, fresh spawn). Without correction, AEB sees a "stationary" obstacle
directly ahead of ego and triggers falsely.

`_swap_trailer_kinematics()` in `thread.py` walks the vehicle list once at the
top of each loop. For every `is_tmp` + `is_trailer` entry it locates the
nearest non-trailer TMP vehicle within 30 m (the pulling tractor) and returns
a shallow copy with `speed` and `acceleration` replaced by the tractor's
filtered values. The trailer's pose, yaw, curvature, and trailer-flag stay
its own: only the kinematics it cannot self-measure are inherited. ACC has
the same swap in `core/acc/tracker.py::_top_leads`.

Only the precompute and main collision iterations consume `vehicles_eff`. The
radar visualizer still reads the original `vehicles` so raw vs filtered speed
displays remain meaningful for debugging the source data.

### Ego curvature: yaw-rate proxy only

AEB **does not** consume `RadarData.ego_curvature` (the position-history
fit). The ego path must react instantly to steering input; a history-based
fit lags the truck through a transient steer and either smears the ego
corridor onto the outgoing lane (false positive after the corner clears)
or leaves it pointing at the previous lane (false negative into the
corner). The yaw-rate proxy has zero lag and zero smoothing:

```python
if ego_speed > 0.5:
    yaw_rate_rad_s = math.radians(steer * ego_speed * 12.0)
    ego_curvature  = yaw_rate_rad_s / ego_speed
else:
    ego_curvature  = 0.0
```

`RadarData.ego_curvature` is consumed by ACC, not AEB. Do not add an
"optional" history fallback to AEB: the reactivity loss is the problem,
not the transient-sample count.

### Target-vehicle curvature: two-source blend

Per-target arc curvature is computed by `_vehicle_curvature_blend()` in
`filters.py`. AEB-local two-source path prediction combines:

| Signal | Source | Role |
|--------|--------|------|
| `pos_kappa` | `ego_curvature_from_history` on the last `cal.aeb_pos_history_len = 6` samples of `v._position_history` (shorter than the full 25-sample fit used elsewhere) | **Smooth**: damps single-frame yaw noise |
| `yaw_kappa` | `radians(v.angular_velocity) / abs_v_speed`: per-frame yaw rate already maintained by the radar thread | **Responsive**: single-frame rotation signal |

```python
if pos_kappa is not None and yaw_kappa is not None:
    v_curvature = cal.aeb_yaw_blend * yaw_kappa + (1 - cal.aeb_yaw_blend) * pos_kappa
elif pos_kappa is not None:
    v_curvature = pos_kappa
elif yaw_kappa is not None:
    v_curvature = yaw_kappa
else:
    v_curvature = 0.0
```

`aeb_yaw_blend` (default `0.4`) controls the mix. This produces
`v_curvature`, the raw measured curvature used by same-curve and diverge
filters. Arcs built for prediction use `arc_curvature`, which is `v_curvature`
after Fix D over-rotation damping when that damping applies. Visualization and
collision must use the same `arc_curvature` so the debug view matches what AEB
actually evaluates. Call sites that go through the helper:

1. `thread.py::_build_vehicle_collision_data` (precompute path): collision tractor + trailer
2. `thread.py::loop` (per-vehicle else branch): visualization, deriving `arc_curvature` from `_dampen_turning_curvature(...)`, then passing it to `v.get_arc(...)` and trailer `build_arc(...)`
3. `filters.py::_build_vehicle_collision_data` (test harness path): collision tractor + trailer

`LaneClassifier.apply` does **not** recompute `v_curvature`: the upstream
call sites above populate it on the `FilterContext` once per frame and the
stage reads it as-is. Recomputing there would double-step the One-Euro state
(see below).

### One-Euro post-filter on `v_curvature`

The yaw+pos blend is fed through a per-vehicle One-Euro filter
(`VehicleCurvatureBlender` in `filters.py`) before reaching the pipeline.
Speed-adaptive low-pass: cutoff = `min_cutoff + beta · |dkappa/dt|`. Quiet
steady state → heavy smoothing (kills single-frame jitter). Genuine corner
entry → cutoff jumps with the derivative and the filter approaches
passthrough within a frame. Reference: Casiez et al., "1€ Filter", CHI 2012.

State lives on `AEBThread._curvature_blender`, keyed by `vehicle.id`. The
helper is stepped exactly once per vehicle per frame: first call site that
sees the vehicle (precompute or fallback else branch in the per-vehicle
loop) advances the filter; `LaneClassifier` reads the cached value from
`ctx.v_curvature`. The blender is `prune`d at the end of each loop against
the current `vehicles_eff` id set so disappearing targets release state.

Calibration:

| Knob | Default | Role |
|------|---------|------|
| `aeb_kappa_one_euro_min_cutoff` | 1.0 Hz | Smooth-floor cutoff at zero derivative |
| `aeb_kappa_one_euro_beta` | 200.0 | Slope of cutoff vs `|dkappa/dt|`: higher = snappier transient, lets more noise through |
| `aeb_kappa_one_euro_d_cutoff` | 1.0 Hz | Low-pass on the derivative estimate (rejects noise-driven cutoff swings) |
| `aeb_kappa_one_euro_beta_turn_scale` | 30.0 | Progressive beta attenuation with `\|kappa\|`: `beta_eff = beta / (1 + scale * \|x_prev\|)`. Counters in-turn cutoff inflation from magnitude-scaling noise (yaw_rate/v amplification, pos-fit numerical sensitivity). 0 disables |

When `_vehicle_curvature_blend()` is called without a blender (e.g. test
paths that don't carry filter state across frames), it returns the raw
blended value. Production paths in `AEBThread` always pass the blender.

Never call `v.get_arc()` from AEB without a `curvature_override`: the
fallback inside `traffic.py::get_arc` uses the full 25-sample
`curvature_from_history()` fit, which lags exiting a corner and leaves
the tractor arc curved long after the trailer arc has straightened.

Do not enlarge `aeb_pos_history_len` toward 25: the responsiveness gain
is the whole point, and `curvature_from_history()` on the full buffer is
what ACC uses.

---

## 2. Module layout

| Module | Role |
|--------|------|
| `core/aeb/calibration.py` | Frozen `AEBCalibration` dataclass: all tunable constants. `DEFAULT` singleton used by both `thread.py` and tests. |
| `core/aeb/lane_frame.py` | `Lane` enum, `project_to_ego_arc()`, `classify()`: arc-projected lane membership, replacing the old cross-product `lateral_offset`. |
| `core/aeb/filters.py` | Named filter pipeline: 12 stage classes + `FilterContext` + `build_pipeline()`. |
| `core/aeb/thread.py` | `AEBThread`: data acquisition, ego-arc construction, pipeline dispatch, TTB/state output. |

---

## 3. Filter pipeline

`build_pipeline(cal)` returns the ordered list of stage instances. Each stage
exposes `.apply(ctx) -> FilterResult`. The first stage that returns
`suppressed=True` short-circuits and the vehicle is skipped. If all stages
pass, the vehicle enters collision evaluation.

| Stage | Purpose |
|-------|---------|
| `RangeFilter` | Distance gate (`cal.max_range`) |
| `ElevationFilter` | Slope-aware Y check (`cal.elevation_margin`) |
| `TmpRelSpeedFilter` | TMP session relative-speed pre-filter |
| `LaneClassifier` | Populates `ctx` geometry fields; sets `ctx.lane` via `lane_frame` |
| `OppositeLaneFilter` | Oncoming vehicles in their own lane (collapses Fix A + Fix B) |
| `CoDirectionalDivergeFilter` | Co-directional arcs already diverging (Fix C + outer-lane same-turn) |
| `TurningCrossTrafficFilter` | Cross-traffic turning through intersection (Fix D absorbed) |
| `OutOfLaneParallelFilter` | Capsule lane-keeping adjacent / roadside traffic; also rear overtakers |
| `TmpCrossTrafficFilter` | TMP-only: straight snapshot uses centre closest-approach (T-bone vs body-graze), turning snapshot uses full-horizon endpoint lane |
| `SweepPassFilter` | Stationary cross-traffic ego turns through |
| `CornerEntryStationaryFilter` | Stationary at corner entry: out-of-lane oncoming/co-dir, or in-lane with arc consistency |
| `EgoEvasionFilter` | Ego can steer around target within 0.08 g (runs for `Lane.EGO` too) |

`FilterContext.d_miss` carries the measured CBDR miss for the vehicle, computed
once per vehicle per frame by the `los_miss` memo in `thread.py::loop`. It is
`None` when the track is too short; filters must fail open on `None`.

Fix labels A/B/C/D are retired: the logic now lives in the named stages above.

The legacy `RearOvertakerFilter` was retired in favour of a unified
"braking worsens" classification in the collision-evaluation step (see §5).
That check compares closing-speed magnitude under braked vs unbraked
trajectories and subsumes the rear-overtaker case along with cross-traffic
scenarios where braking parks ego in a target's path.

### `LaneClassifier`: canonical lane primitive

`LaneClassifier` is the first stage to populate all geometry fields. It uses
`project_to_ego_arc()` from `core/aeb/lane_frame.py` to compute the arc-projected
lateral offset. For curved ego arcs, the returned `d_abs` is the **maximum** of
the circle-offset and the straight-line heading offset. This prevents a tight ego
turn from projecting an opposite-lane vehicle into the EGO bucket.

Lane thresholds (`cal.lane_half_width=1.95 m`, `cal.lane_separation=3.9 m`):

| d_abs | Lane |
|-------|------|
| ≤ 1.95 m | `EGO` |
| 1.95–1.95 m | `ADJACENT` (empty range: no ADJACENT in practice) |
| 1.95–7.8 m | `OPPOSITE_OR_OUTER` |
| > 7.8 m | `OFF_ROAD` |

### `OppositeLaneFilter`

Applies to `head_on` vehicles (`fwd_dot < cal.head_on_dot=-0.7`), and to
`near_head_on` for the **body-separation** fast path only (collide can start
before `fd` crosses `head_on_dot`).

1. Lane check: `own_lane = ctx.lane in (OPPOSITE_OR_OUTER, OFF_ROAD)`
2. Body-separation fast path: if pose `d_abs >= clear_bar - oncoming_body_sep_soft_m`
   and the measured miss agrees, suppress (also when lane reads `EGO` under the
   soft bar: corner-pull adjacent). Skip when `oncoming_closing_into` (below)
   or when `max_evasion_lat_g` refuses (physically unavoidable closing). Opp
   arms max-g refuse at 7 m by default; targets ≥ `max_evasion_opp_fast_kmh`
   (60) use the lower `max_evasion_min_lat_m_opp_fast` (4.5) arm.
3. Evasion arc test (`head_on` only): for each target arc, build two curvature-offset arcs
   (`base_curvature ± delta_kappa_t`, `decel=0`). If either clears `ego_arc`, suppress.
   - `delta_kappa_t = min(evasion_g_oncoming / v², evasion_max_dkappa)`, scaled by
     `opposite_lane_kappa_scale` when `own_lane`.
   - Fix B: `delta_kappa_t = max(delta_kappa_t, min(|ego_curvature|, shared_turn_max_kappa))`
     when `own_lane` and ego is turning, **except** when CBDR miss is closing
     fast while ego turns (below).

**Turn-into-path (CBDR miss rate).** Ego steering into oncoming can inflate
arc `d_abs` while the measured miss shrinks (clip `e0fd28b3`: `d_miss` rate
about −5.6 m/s). Closing predicate: `|ego_curvature| >= turning_diverge_kappa`,
`d_miss_rate <= oncoming_closing_dmiss_rate_mps` (−1.5 m/s), straight-frame
`|lat| < oncoming_closing_lat_m` (0.85 m), and arc inflation
`d_abs >= |lat| * oncoming_closing_dabs_lat_ratio` (honest adjacent stays
body-sep; turn-into with inflated `d_abs` still skips). On that course skip
body-sep and Fix B κ expansion; also skip engagement-entry LOS / turn
extrapolation vetoes so warn can promote to brake. Evasion clearance still may
suppress. Colliding closing targets also earn `certain_geom` for instant engage.
`OppositeLaneFilterMirrored` stays `Lane.EGO` only.

### `CoDirectionalDivergeFilter`

Applies only to `co_directional` vehicles (`fwd_dot > cal.co_directional_dot=0.7`).
For each arc with `speed > 0.5 m/s` evaluate `_is_approaching` at the hit point
(lookahead `co_dir_diverge_lookahead_s=0.25 s`). **Suppress only when every
colliding body of the rig diverges** (the first approaching arc passes the rig):
a tractor pulling into the outer lane genuinely diverges, but if its trailer is
still closing in ego's lane the rig is a real rear-end course, and vetoing on the
cab arc (evaluated first) would drop the approaching trailer with it (crash clip
434f0401).
Extended lookahead (`dynamic_horizon × co_same_turn_lookahead_scale=0.5`) when
all four conditions hold: vehicle in outer lane, both curvatures above threshold,
same curvature sign.

**Trailer-in-lane rescue.** The lane primitive (`ctx.lane`) keys off the tractor
reference point only, so a long trailer swung across ego's lane while its tractor
rides the outer lane of a shared curve reads as EGO-lane-clear: the extended
lookahead then extrapolates the whole rig away as "diverging" and suppresses a
genuine rear-end until the cab crosses into ego's lane at contact (crash clip
434f0401). `_any_body_in_ego_lane` samples each arc's rigid body capsule
centreline (rear→front) at `t=0`; when any body is physically inside
`lane_half_width`, the same-turn extended lookahead is dropped and the in-lane
dip check is armed. The centreline (no body half-width) is used so a
corridor-grazing outer-lane body is not miscounted as in-lane
(`fp_co_directional_outer_lane` stays suppressed). Regression scenario:
`tp_trailer_in_lane_shared_curve`. The same primitive gates the equivalent
tractor-based lane checks in `OutOfLaneParallelFilter` and `EgoEvasionFilter`.

**Pass-through dip check (in-lane only).** The `_is_approaching` endpoint
comparison alone inverts for fast closers: the hit time is the capsule
contact moment (bumper gap ≈ the effective corridor margin since the
cap-alignment fix), so a modest closing speed already carries the `t + dt`
sample beyond the target, center distance grows again, and a lead about to
be rear-ended reads as "diverging" (FN clip fbc397b3: 97 km/h ego suppressed
a 20 km/h in-lane lead until impact, back when the padded contact distance
still put the inversion near ~70 km/h closing). Fix:
`_is_approaching` samples `cal.diverge_dip_samples` points across the
window; when `dip_active` and any sample's center distance drops below the
sum of the body half-widths (no corridor margin), the extrapolated bodies
overlap and the pair counts as approaching regardless of the endpoints.
`dip_active` is `ctx.lane == Lane.EGO` in `TurningCrossTrafficFilter`, and
`ctx.lane == Lane.EGO or in_lane_body` in this stage (see the trailer-in-lane
rescue above): for out-of-lane targets with no body in ego's lane an
extrapolated body overlap is usually a constant-curvature artifact (e.g.
overtaking a slower outer-lane vehicle in a shared turn:
`fp_co_directional_outer_lane`), which is exactly what the filter exists to
suppress. Regression scenario: `tp_fast_closing_lead` (80 km/h closing, in-lane).

### `OutOfLaneParallelFilter`

Suppresses co-directional / stationary traffic whose body stays out of ego's
lane: the capsule collision body registers a grazing corridor overlap for a
long vehicle driving or parked alongside ego that the point model mostly
missed. A target whose centre never enters ego's lane over the horizon
(arc-projected offset stays above `lane_half_width`) is lane-keeping adjacent
traffic or a roadside object, not a collision course. Head-on own-lane
oncoming is handled by `OppositeLaneFilter`; follow / latched threats and any
body sample already in the EGO band (trailer swung into path, clip 434f0401)
are exempt.

**Stationary adjacent straddle.** An angled parked body can put the reference
pose (or a shallow centreline graze) in `Lane.EGO` while the far end sits in
the adjacent lane (clip `82acb8e8`: samples ~1.0–3.5 m). When
`min(d_abs)` in `(stationary_ool_graze_min_m, stationary_ool_graze_max_m]` and
`max(d_abs)` in `[lane_half_width * stationary_ool_span_scale, lane_separation]`,
suppress as roadside. Stationary centre scans also use `v.position` instead of the
body-offset `arc.start`, which otherwise fakes an in-lane centre.

**Rear-overtaker early suppress.** Faster co-directional traffic approaching
from behind in another lane (`v·ego_fwd > ego_speed` and `dx·ego_fwd < 0`) is
suppressed *before* the predicted-centre scan. That scan is circular: it
projects onto the same bent ego arc that manufactures the phantom capsule
hit, so it leaks exactly on the wiggle ticks that create the FP (passing
clips f0b2ace6 / 02642609). A genuine cut-in brings its body into the EGO
band and is not suppressed here; braking_worsens covers the same class if
anything still reaches collision eval.

### `TmpCrossTrafficFilter`

TMP-only filter that absorbs MP-data uncertainty for routine intersection
maneuvers. TMP position/yaw/curvature snapshots are jittered enough that an
in-progress turn at a side road can briefly project an arc through ego's lane
even though the actual MP target is sweeping past. Applies to TMP vehicles
(`v.is_tmp=True`) that aren't co-directional and have non-trivial speed. For
each target arc with a collision hit, build a non-braked "sweep" arc from the
snapshot's `(start, yaw, curvature, speed)`, then branch on how trustworthy
the snapshot's motion is (`ctx.v_curvature`):

- **Straight snapshot** (`|v_curvature| < turning_diverge_kappa`): the motion
  is trustworthy, so decide on the geometry directly. Take the closest approach
  of the two reference **centres** over the horizon (ego arc vs sweep arc). If
  they genuinely meet (`d_min <= cal.tmp_cross_center_hit_dist`) it is a real
  T-bone → **pass** (brake). If the centres miss (the collision is only the
  target's long body grazing ego's corridor as it sweeps clear) → **suppress**.
- **Turning snapshot** (`|v_curvature| >= turning_diverge_kappa`): the jittered
  curvature makes the predicted centre path unreliable, so keep the full-horizon
  endpoint-lane test. Endpoint in `OPPOSITE_OR_OUTER` / `OFF_ROAD` → the target
  sweeps clear → **suppress**; endpoint in `Lane.EGO` → real continuing threat →
  **pass**.

The split fixes a false negative in the old design, which used the endpoint
test for **all** TMP cross-traffic. The endpoint answers "where does the target
*end up* at the 3 s horizon", not "does it occupy ego's lane *when ego
arrives*". A genuine straight perpendicular crosser on a dead-center collision
course always ends tens of metres past ego's lane at the endpoint, so it was
suppressed at every range (no warn, no brake in TMP sessions until the crosser's
measured speed dropped: corpus FN clip ffd29f9e). The centre-miss test answers
the correct question for the trustworthy straight case: a real collision brings
the reference centres to ~0 m, a body-only graze keeps them metres apart. The
turning branch is unchanged and still suppresses the mid-turn jitter phantom
(regression `fp_tmp_side_road_right_turn` phase 2, `fp_cross_traffic_completing_turn`).

Optional `tmp_cross_in_corridor_pass`: when enabled, do not graze-suppress if the
body is already inside `|lat| ≤ lane_half_width` ahead with a closing (or unknown)
miss rate. Default off: max_evasion fail-closed covers the hard TMP twins without
the side-road FP cost. `max_evasion_lat_g` also refuses TmpCross suppress when
straight-frame `|lat|` clears `max_evasion_min_lat_m_tmp_cross` (3 m; Opp
uses the higher 7 m arm) and required lateral accel exceeds the truck
budget (physically unavoidable closing).

No imminence / TTB floor is used: the turning phantom persists to `hit ≈ 0`
(the jittered vehicle sweeps through ego's lane right up to closest approach),
so any positive "never suppress below this hit time" floor would leak that
regression. A turning TMP target that is genuinely on a collision course and
close-in relies on the engagement-certainty gate + TTB slam net downstream, not
this stage.

Uses a freshly-built non-braking arc from the **undamped** `ctx.v_curvature`
(rather than `base_target_arc`) in both branches: the standard arc may be
truncated by target-side full-brake modeling at near-head-on angles, and its
Fix D over-rotation damping straightens the arc of a target genuinely sweeping
through a corner. Either one distorts the projected centre path and endpoint and
masks where the cross-traffic actually sweeps to.

Non-TMP targets bypass entirely: AI vehicles' arcs are deterministic and
already handled by `OppositeLaneFilter`, `TurningCrossTrafficFilter`, etc.

### `CornerEntryStationaryFilter`

Suppresses stationary targets (`|speed| < sweep_pass_max_target_speed`) at
corner *entry* (`|ego_curvature| < turning_diverge_kappa`) when their pose
implies they sit on a curved road continuation rather than blocking ego's
straight-line path.

- Symmetric road-bend formula: `road_bend = acos(|fwd_dot|)`: folds oncoming
  (`fwd_dot ≈ -1`) and co-directional (`fwd_dot ≈ +1`) cases into `[0, π/2]`.
- Implied curvature `road_bend / dist` must exceed `turning_diverge_kappa`.
- **Mode A** (out-of-lane): vehicle in `OPPOSITE_OR_OUTER`/`OFF_ROAD` → suppress.
  Covers the original "around the bend" oncoming case, now also co-directional.
- **Mode B** (in-lane): vehicle in `Lane.EGO` → require geometric consistency
  with a curved continuation. Yaw-rotation direction must match lateral-offset
  direction, and `|dist · sin(road_bend / 2) − |lat_signed||` must fall within
  `corner_entry_lateral_tol`. Catches MP stopped queues whose lead vehicle
  projects to ego's straight axis but whose pose only makes sense on a curve.

### `EgoEvasionFilter`

After all previous stages pass, checks if ego could steer around the target
within `evasion_g=0.08 g`. Uses `margin=0.0` for evasion arc checks (physical
body clearance, not padded corridor).

**`ctx.lane == Lane.EGO` never bypasses this stage.** Lane-EGO classification
is not evidence of danger: stationary shoulder vehicles and passing traffic on
wide or curved roads land in the EGO bucket routinely, and a target a 0.08 g
steer clears is not a threat whatever bucket it sits in. Geometry decides.
The three remaining bypasses are narrow:

| Bypass | Condition | Why |
|--------|-----------|-----|
| Head-on | `head_on` **and** target moving | Oncoming traffic belongs to `OppositeLaneFilter`, which runs its own evasion arcs at `evasion_g_oncoming`. A *stationary* target facing ego is a parked obstacle, not oncoming, so it runs the check. |
| Trailer swing | out-of-lane co-directional mover with a body inside `lane_half_width` | Genuine rear-end course; crash clip 434f0401 must never be evasion-suppressed. This is the one place the trailer-in-lane rescue is deliberately *stronger* than `Lane.EGO`. |
| Follow-threat | `v.id in follow_threat_ids` **and** `ref_kmh_for_filter <= tmp_filter_split_kmh` | Inside the low-speed TMP band `TmpRelSpeedFilter` drops targets unless relative speed exceeds `tmp_filter_rel_below_kmh`, which discards real low-speed dangers; the behavioral latch is the fallback there. Above the split the latch is ACC-shaped lead tracking, which does not establish danger, so it no longer shields a target. |

Corpus note: making `Lane.EGO` run the check costs clip `55848211` (fn, sev 4)
a TP → LATE and ~26 cost on the labelled corpus as of this change. Accepted
deliberately: the FP class it targets (shoulder and passing vehicles bucketed
as `Lane.EGO`) is not yet labelled, so the corpus cannot currently price the
upside. Re-check this trade once those clips are in.

---

## 4. TMP rel-speed pre-filter

When **any** slot in the frame has `is_tmp`, AEB pre-filters targets by
**‖v_ego − v_target‖** (km/h) vs a **reference ego speed**:

- ref **> 40 km/h** → threat only if rel **> 15 km/h**
- ref **≤ 40 km/h** → threat only if rel **> 40 km/h**

Reference is current ego speed unless **latched**: the first frame with
`AEBState ≥ WARN` or addressing brake (`_read_user_braking`: driver pedal or
CC/ACC program end brake) saves `ego_kmh`;
latch held until state drops below WARN **and** brake released; cleared when
the session is no longer TMP.

---

## 5. AEB Thread Interface

Read from other threads (acquire `data._lock` first):

```python
aeb = registry.get_thread("aeb_thread").data
aeb.AEB_warn                       # bool: UI/sound cue (warn fraction or TTB)
aeb.AEB_brake                      # bool: engagement latched and target > 0
aeb.AEB_target_decel_ms2           # float: rate-limited commanded decel (m/s²)
aeb.AEB_ff_decel_ms2               # float: always-on additive FF decel (m/s²); 0 when no threat
aeb.AEB_required_decel_ms2         # float: slope-corrected required decel
aeb.AEB_effective_max_decel_ms2    # float: slope-corrected capacity ceiling
aeb.AEB_realized_decel_ms2         # float: lead-compensated measured decel
aeb.time_to_brake                  # float: seconds (1e9 = no threat)
aeb.em_stop_requested              # bool: mirror of AEB_brake
aeb.snapshot                       # AEBSnapshot: full debug state
```

### Continuous-decel logic

1. Pipeline runs as before. First `suppressed=True` short-circuits.
2. For surviving targets: collision check yields `unbraked_hit` and
   `braked_hit`. Per-target `closing_speed` is the **vector magnitude** of
   the relative velocity in world frame (`|v_ego_vec − v_target_vec|`), not
   the axial projection onto ego's heading. The axial form clamps to zero
   for rear-overtakers and misses the rear-end-worsens case.
   - `braking_worsens` is set if either:
     - `braked_hit is not None` and (`v_target_along_ego > ego_speed`
       — pure rear-overtaker shortcut for imminent collisions where
       `t_braked` is too small for the comparison below — **or**
       `closing_braked > closing_unbraked + brake_worsens_hysteresis_ms`
       — cross-traffic where braking parks ego in target's path), **or**
     - `braked_hit is None` and `v_target_along_ego > ego_speed` and the
       target is behind ego (`dx·ego_fwd < 0`): full braking clears a
       rear-overtaker entirely, but feeding its closing speed into
       required_decel still reads an adjacent-lane pass as a frontal
       threat (passing FP clips f0b2ace6 / 02642609). Scoped behind ego
       so a faster crosser ahead, where braking *is* the avoidance, still
       contributes.
   - Targets flagged `braking_worsens` are added to `braking_worsens_ids`
     and excluded from `best_ttb` / `best_v_closing`. AEB engagement on
     these is forbidden.
   - Non-worsens targets set `ttb = unbraked_ttc` and contribute to
     `best_ttb`, `best_closing_distance`, `best_v_closing` (all from the
     lowest-`ttb` target).
3. Required decel for the worst target:
   ```
   d_rel             = closing_distance - stop_buffer - v_closing * stop_buffer_response_s
   required_decel    = v_closing² / (2 * d_rel)              # while d_rel > 0
   # d_rel <= 0 fallback (_required_decel_two_frame): the relative frame
   # degenerates when v_closing * ttc undershoots stop_buffer at near-zero
   # closing speed (hit exists only via the target's predicted decel: ego
   # creeping behind a slowing lead, clip 5a6050f5). Braking can never do
   # better than stopping ego before the contact point, so switch to the
   # consistent ego frame; imminent contacts are owned by the TTB slam.
   required_decel    = ego_speed² / (2 * max(ego_travel_to_hit - stop_buffer
                                             - ego_speed * stop_buffer_response_s, 1e-3))
   slope_accel       = g · sin(ego_pitch_rad)         # +ve = uphill (radar convention)
   downhill_offset   = max(−slope_accel, 0)           # gravity stealing brake force
   effective_max     = ego_decel_frac · capacity_estimate − downhill_offset
   effective_required= required_decel + downhill_offset
   ```
   `capacity_estimate` is read from `sending_thread.data.max_brake_ms2`
   (PedalCapacityTracker) with a fallback constant.

   `threat_present = required_decel > 0`. Slope modifies a threat-derived
   demand, it never sources one: `warn_by_decel` and the FF branch both
   require `threat_present`, so gravity alone can neither warn nor feed
   brake assist. Without that gate the offset raises `effective_required`
   and lowers `warn_threshold` (itself `0.5 · effective_max`) at the same
   time, so pitch alone crosses the bar once `downhill_offset ≥ 0.3 ·
   capacity_estimate`, about 17°. Public roads never reach that; a wreck,
   an embankment or an airborne truck does immediately, which was the
   phantom-warn-after-crash class (clip b530ea7b: 9 warn ticks and 92 FF
   ticks up to 3.16 m/s² with `colliding_ids` empty and `ttb`/`ttc` at
   infinity for the entire clip; 844 of 7199 corpus warn ticks were pure
   slope). `brake_ttb_active` is deliberately outside the gate: it owns the
   geometric threats where `required_decel` collapses to 0 by design.
4. Engagement hysteresis (slope-aware):
   - `brake_ttb_active = (time_to_brake < cal.brake_ttb + cal.brake_response_window_s)`
    : emergency criterion for path-crossing / arc-cross threats where
     `v_closing ≈ 0` collapses the `required_decel = v_closing²/2d` formula
     but the geometry still says full brake can't avoid intersection. The
     `brake_response_window_s` headroom compensates for actuator lag + the
     rate-limited brake ramp so the slam fires before impact rather than
     after the pedal has already needed to be at full. Without it, AEB
     stays in WARN forever in these scenarios.
   - Geometry-driven engagement latch: once engaged, hold engagement
     while any colliding target has `unbraked_ttc < disarm_hold_ttc_s`.
     A working brake pushes `effective_required` under the disarm
     threshold and headway over the distance latch by construction
     (`required ~ v²`, `headway = d/v`), so a narrow hold window releases
     mid-stop and the event pumps: release at ~30 km/h closing, coast,
     re-engage at 7 m (clip 29c8e7e0). The hold releases when the target
     clears laterally (not colliding), accelerates away (ttc grows), or
     ego stops (no closing → ttc ∞). Entry is untouched.
   - Latched-distance hold: see "Latched-threat hold" below. Adds a
     headway-driven engagement hold over targets that have been engaged
     on previously, independent of current `v_closing`.
   - Engage when `effective_required ≥ aeb_engage_frac · effective_max` **OR**
     `brake_ttb_active`, subject to the tiered entry certainty gate below.
   - **Tiered entry certainty gate**: a new engagement additionally requires
     one of: (a) `brake_ttb_engage_active` (imminent, full brake barely
     avoids: instant), (b) certain geometry: a colliding, non-LOS-vetoed
     target in `Lane.EGO` with `|fwd_dot| ≥ aeb_certain_fwd_dot` (aligned
     in-lane traffic: rear-end or wrong-way, whose collision prediction
     barely depends on arc extrapolation: instant), (c) continuity: a
     colliding target already in `_latched_threat_ids` (instant), or (d)
     qualification sustained for a geometry-graded confirm window (tracked
     by `AEBThread._engage_confirm`, an `OccupancyConfirm`; reset while
     engaged): `aeb_engage_confirm_s` when any qualifying colliding
     target is **near-certain** (in `Lane.EGO`, or aligned `|fwd_dot| ≥
     aeb_certain_fwd_dot` in any lane: one classification step from
     certain), else `aeb_engage_confirm_oblique_s` for **oblique
     out-of-lane** threats, the extrapolation-fragile class (corner
     sweeps, mutual-turn passes) whose corpus phantoms qualify ≤ ~0.15 s.
     **Lapse tolerance (occupancy window).** The confirm window is not a
     hard-reset timer. `OccupancyConfirm` (`core/aeb/confirm.py`) tracks the
     per-frame qualification over a trailing window and fires when the
     window has elapsed AND the qualified fraction reaches
     `aeb_confirm_occupancy`, dropping the streak only after more than
     `aeb_confirm_max_gap_frames` consecutive unqualified frames. This
     absorbs the per-frame detection flicker (the 36-sample collision time
     grid, TMP jitter walking the predicted course across coverage edges)
     that used to restart the whole window on a single missed frame, without
     letting sparse 1-tick blips accumulate: they never reach the occupancy
     threshold and a long gap resets. The instant paths (a)/(b)/(c) and the
     reset-while-engaged behaviour are unchanged.
     Known trade at 0.20 s: a genuine perpendicular crosser whose
     qualification sustains ~0.16 s (clip ffd29f9e) loses the brake: warn
     still fires and the TTB slam net catches a materializing threat, but
     the phantom and genuine distributions touch at ~0.16 s, so the window
     cannot separate them perfectly. Rationale: corpus analysis showed phantom engagements
     from extrapolation-fragile crossing arcs qualify for 1-2 ticks and
     vanish, while genuine threats sustain qualification 3-30 ticks; and
     the dominant genuine classes (in-lane aligned) skip the wait entirely,
     so the confirm window costs real rear-end/head-on events zero latency.
     Required-decel magnitude is NOT a certainty signal: crossing-arc
     phantoms enter at ~200% of max (collapsed d_remaining) while genuine
     rear-ends enter at 70-100%. Warn and the FF-assist layer are untouched
     by this gate.
   - **Per-target risk confirm** shares the same `OccupancyConfirm`
     mechanism (`AEBThread._risk_confirm`, keyed by vehicle id). A colliding
     target must sustain qualification for `risk_confirm_s`
     (`risk_confirm_oncoming_s` for head-on) before it contributes to the
     aggregates; an id survives up to `aeb_confirm_max_gap_frames` missed
     frames before being dropped, so a single collision-grid dropout no
     longer restarts its clock or evicts it from the tracking dict.
   - Disarm when `effective_required <  aeb_disarm_frac · effective_max` **AND
     NOT** `brake_ttb_active` **AND NOT** `geom_threat_latched` **AND NOT**
     `latched_distance_threat`.
5. Setpoint pipeline:
   - When `brake_ttb_active`: `target_raw = effective_max` (slam: required formula is unreliable).
   - Otherwise: `target_raw = clamp(effective_required, 0, effective_max)` while engaged,
     then floored at `cal.latched_min_decel_frac · effective_max` when `latched_distance_threat`.
   - **Deadband + rate-limit**: if `|Δ| < aeb_target_deadband_ms2` and the
     held value is younger than `aeb_target_refresh_min_s`, hold. Else move
     toward `target_raw` capped at `aeb_target_rate_ms3 · dt` (m/s² per tick).
   - The published value is `AEB_target_decel_ms2` consumed by sending_thread.
6. Flags:
   - `AEB_warn` rises on `effective_required ≥ aeb_warn_frac · effective_max`
     OR `time_to_brake < warn_ttb`, gated by warn persistence: certain /
     near-certain geometry, imminent TTB, latched threats, and active
     engagement warn instantly, while oblique out-of-lane threats must
     sustain the raw warn condition for the `aeb_warn_confirm_oblique_s`
     occupancy window (tracked by `AEBThread._warn_confirm`, an
     `OccupancyConfirm` observed every frame; the streak follows the raw
     condition, so the user-braking display suppression never resets it, and
     an isolated 1-2 frame dropout does not restart the window). Because warn
     qualifies on a superset of the engage-qualified frames, uses the same
     `aeb_confirm_occupancy` / `aeb_confirm_max_gap_frames`, and
     `aeb_warn_confirm_oblique_s ≤ aeb_engage_confirm_oblique_s − 0.1`, the
     warn streak confirms at least 0.1 s before an oblique engagement even
     through flicker: the driver's gas-override reaction window is preserved.
     Default `aeb_warn_confirm_oblique_s` is 0.30 s (paired with
     `aeb_warn_frac=0.60` and the vetoed window below): short clear-pass
     encounters that only flicker the raw condition may stay silent. That is
     intentional comfort trade; see TUNING.md TODO for residual false_warn.

     **Fully-vetoed out-of-lane persistence.** A second, longer occupancy
     window (`aeb_warn_confirm_vetoed_s`, `AEBThread._warn_vetoed_confirm`)
     applies on the frames where *every* colliding target is both in
     `engage_vetoed_ids` and outside ego's lane band
     (`ego_lane_colliding_ids`). That intersection is the extrapolation-phantom
     class: highway oncoming at tens of metres whose predicted contact the LOS
     or turn veto has already refused to engage on. The window is latency, not
     silence: a course that persists still warns, and the instant paths
     (engaged, TTB slam, latched) bypass it. A target only joins the vetoed set
     once its LOS track is long enough (`los_veto_min_samples`), so the opening
     frames of a short encounter still use the ordinary oblique timing.

     **Evidence-class persistence (oncoming / wide-lateral).** Two more
     all-targets windows sit alongside the vetoed one, because near-certain
     geometry alone was granting an instant warn to the corpus's two dominant
     phantom-beep classes. `aeb_warn_confirm_oncoming_s` applies while every
     colliding target is head-on or near-head-on: opposite-carriageway traffic
     is aligned (`|fwd_dot| >= aeb_certain_fwd_dot`), so it reached
     `nearcertain_geom_ids` and beeped on a single frame of corridor clip.
     `aeb_warn_confirm_wide_lat_s` applies while every colliding target
     projects more than `aeb_warn_wide_lat_m` off the ego arc, roughly a full
     lane over: slow or stopped vehicles ego is passing. Both are latency, not
     silence, and share the vetoed window's instant bypasses, with one
     exception: the TTB slam presumes an in-path target, so
     `aeb_warn_ttb_needs_narrow` makes an all-wide-lateral set clear the
     wide-lateral window even when `brake_ttb_active`. Separately,
     `aeb_warn_max_range_m` drops the raw warn when the nearest colliding
     target is past it: no corpus clip's genuine warn opens beyond ~80 m, and a
     beep about something further out is not actionable.

     **Class stickiness and the instant floor.** Two escapes closed the gate's
     back door. A target that shrinks under `aeb_warn_wide_lat_m` as it closes
     used to leave the wide class mid-approach and collect an instant warn the
     gate had been refusing a frame earlier, so the class is sticky across
     `aeb_warn_wide_lat_sticky_s` of lapse. The grace is deliberately short: it
     bridges a collision-grid dropout but lets a target that genuinely leaves
     and re-approaches start fresh, which is what keeps the one real trigger in
     `075d163a` while dropping its two phantom ones. Separately,
     `aeb_warn_instant_min_s` makes even certain geometry show a couple of
     frames of raw warn before the instant bypass fires, since a single-frame
     demand spike on a target that then vanishes is a tracking artefact that
     the 0.3 s state hold would otherwise stretch into an audible beep.

     Corpus effect of the whole gate: `false_warn` 43 -> 3, `TN` 286 -> 326,
     every other verdict unmoved. Three genuine clips lose their warn
     (`adf20ad7`, `4ba23e1c`, `505af174`, all one-to-two-frame blips). The
     dominant price is lead, not coverage: warn-before-brake clips 120 -> 59
     and clips with at least 0.3 s of lead 28 -> 14, almost all of it from
     `aeb_warn_instant_min_s`. Setting that knob to 0 restores lead (88
     warn-before-brake, 21 at >= 0.3 s) at the cost of one false_warn
     (`9fa4c844`). See TUNING.md before touching either.
   - `AEB_brake` is true while engagement is latched and the published target
     is above zero. Other subsystems (cruise/HMI) gate off this flag.
   - **User-braking suppression**: when `_read_user_braking()` is true and
     demand has not reached `aeb_warn_near_full_frac · effective_max`,
     `AEB_warn` is forced false: a driver already braking is not warned about a
     threat they are handling. This silences the cue only. Engagement, the
     published target, and `AEB_brake` are untouched. Two sources count, each
     compared against `_USER_BRAKE_LATCH_THRESHOLD` (0.03):
     - `main_pedal_thread.brakeval`, the physical brake axis.
     - `sending_thread.mapper_command_brake`, the CC/ACC/limiter command.

     The threshold is deliberately low: a light dab still carries braking force
     and signals that the driver saw the hazard, and AEB engagement remains the
     emergency override when the dab is not enough.

     `main_pedal_thread.opdbrakeval` is **excluded on purpose**. It sits below
     `brakeval` whenever the pedal is pressed (measured: 0.28 -> 0.21,
     0.17 -> 0.13), so it adds nothing there; its only unique contribution is
     the OPD coast-down floor, which is capped by `max_opd_brake_variable`
     (default 0.04) but still clears 0.03. Including it would read ordinary
     coasting, the default OPD state, as a deliberate danger response: measured
     at 518 such ticks over one session.

     Both taps are AEB-free by construction: `brakeval` is raw axis input, and
     `mapper_command_brake` is read upstream of the point where AEB's FF and
     slam merge into the pedal. **Never source this from `abackward` or
     `brake_output`**: both contain AEB's own brake, so AEB would read its own
     output back and silence its own warning. The old `if self._engaged:
     return False` guard existed only to paper over that contamination.
7. Hold semantics: warn/brake state holds for 0.3 s after a downgrade to
   suppress chatter, identical to the old WARN→STANDBY hold. The hold shapes
   `aeb_state` and `AEB_brake`, but must **not** re-assert `AEB_warn` on a tick
   where the user-braking suppression fired. Suppression is computed before the
   hold and `warn_suppressed` carries past it. Without that carry the hold
   reinstated the cue for up to 0.3 s after every warn, and since each
   legitimate re-warn re-arms the hold, a driver braking through a decaying
   threat heard near-continuous beeping. `stop_warning()` still replays the
   clip plus `_AEB_WARNING_STOP_EXTRA_REPLAYS`, so even a 2-frame spurious warn
   is audible: short suppression leaks are not cosmetic.
8. Head-on targets: modelled as also braking at `full_brake_decel (7.8 m/s²)`
   inside the collision pipeline (unchanged).

### LOS-rate engagement veto (CBDR)

New engagements evaluate a second, engagement-only aggregate chain that
excludes targets vetoed by measured line-of-sight drift. A genuine collision
course holds near-constant world-frame bearing while range shrinks (constant
bearing, decreasing range); corner cross-traffic whose extrapolated arc
phantom-intersects ego's corridor drifts its bearing consistently instead.
Per target, `_los_predicted_miss()` fits bearing/range slopes over the last
`los_veto_window_s` of raw positions (`AEBThread._los_tracks`, fed every
frame for every tracked vehicle, pruned with the blender) and estimates
`d_miss = |omega_los| * R^2 / |v_rel|`. It is computed at most once per
vehicle per frame (`los_miss` memo in `loop`) and reused by every consumer
below.

Veto fires only when the track has `los_veto_min_samples`, the target is
beyond a range floor, and `d_miss` exceeds a miss bar. `_los_veto_bar()`
picks the pair:

| Geometry | Range floor | Miss bar | Why |
|----------|-------------|----------|-----|
| Co-directional or crossing | `los_veto_min_range_m` (25 m) | `los_veto_miss_dist_m` (6.0 m) | Relative motion is not constant, so the estimate needs a wide margin. Corpus separation: genuine engagement edges 0.05-4.4 m, corner phantoms 6.8-12.3 m. |
| Head-on (`fwd_dot < head_on_dot`) | `los_veto_headon_min_range_m` (20 m) | `los_veto_headon_miss_dist_m` (2.8 m) | An antiparallel encounter lasts 1-3 s and is decided by lateral separation alone, so the bar is physical body clearance (`ego_hw + target_hw` is about 2.4 m) plus measurement margin. |
| Head-on but manoeuvring | falls back to the general pair | | `|v_curvature| >= los_veto_headon_max_kappa` (0.05, about R 20 m) at `abs_speed >= los_veto_headon_min_speed_ms`: a target turning that hard is not holding a straight line, so straight-line CBDR does not describe it. The speed floor is required because `kappa = yaw_rate / v` diverges near zero speed. |

The head-on branch exists because the dominant labelled false-positive class
was oncoming traffic that the arc model placed in `Lane.EGO`. On a bend of
1000-1500 m radius the lateral shift over 40-90 m of straight-line
extrapolation is 1-4 m, which is exactly a lane width, and the steer-derived
ego curvature cannot see a bend that gentle. The 2.8 m bar was derived on the
360-clip corpus, where the split at the engaging tick was 8 genuine head-on
engagements measuring 0.19-2.72 m of miss against 13 phantoms measuring
1.74-6.24 m; it held unchanged when the corpus grew to 514.

Scope is strictly engagement *entry*: warn, disarm, geometry latch, and
distance holds all keep the full aggregates, so a wrong veto costs latency on
one target, never silence. Vetoed ids are published in
`snapshot.los_vetoed_ids`.

### Extrapolation vetoes

`_extrapolation_veto()` bars engagement entry on two further classes where the
predicted contact rests on extrapolation the system cannot support. Both feed
the same engagement-only aggregate as the LOS veto, so warn and FF assist are
untouched. `ctx.lane == Lane.EGO` and `_any_body_in_ego_lane` are hard
exemptions: the trailer-in-lane rescue must survive both (clip 434f0401).
`extrap_veto_enabled` disables the whole helper.

- **Ego-turn extrapolation** (non-co-directional targets). A hit reachable only
  by holding the current steer for `turn_veto_min_ttc_s` (1.2 s) while
  `|ego_curvature| >= turn_veto_min_kappa` (0.012, about R 83 m) is vetoed.
  Real turns are transient (entry, apex, exit); a constant-curvature ego arc
  swept 20-30 degrees into a junction manufactures crossings with traffic on
  the road ego is turning onto. Near-term hits and straight-line driving are
  untouched, so the scope is junctions and roundabouts only. Corpus: the
  labelled phantoms sat at 17-31 degrees of required sweep with the target
  23-49 m off the ego arc, while every genuine engagement while turning had
  its hit within 0.5 s or its target inside the lane.
- **Matched-speed neighbour** (co-directional, out of lane). Vetoed when the
  axial closing speed `ego_speed - v_target_along_ego` is in
  `[0, codir_adjacent_veto_axial_ms)` (2.0 m/s) **and** the measured
  `d_miss >= codir_adjacent_veto_miss_m` (2.0 m). Braking removes at most the
  axial component, so at under 7 km/h of axial closure any predicted contact is
  lateral and brakes do not steer. The lower bound of the band matters: a
  *faster* target is an overtaker, which `braking_worsens` already owns and
  which the corpus labels as a genuine threat when it cuts in. The miss term
  matters too: a neighbour whose measured track is converging on ego is a real
  side contact, and with no measurement yet the veto stays off.

### Lane confidence range

`Lane.EGO` is treated as certainty only where pose can actually fix a lane.
For a non-co-directional target an unseen road bend displaces it by roughly
`kappa_road * s^2 / 2`, which reaches a lane half-width (1.95 m) at about 30 m
for the gentlest bend worth worrying about, so beyond
`lane_confidence_range_m` neither `certain_geom_ids` nor `nearcertain_geom_ids`
is populated and the target takes the oblique confirm window instead of an
instant engage. Engagement is not blocked, only its instant path: a stopped
obstacle 55 m ahead still brakes well outside the range, it just confirms
first (`test_far_obstacle_still_engages_through_the_confirm_window`).

**Co-directional targets are exempt** from both this range and the oblique
window, via `lane_trusted` and the `ctx.co_directional` term in
`nearcertain_geom_ids`. A pair travelling the same way shares whatever bend it
is on, so the bend's lateral error is common-mode and cancels; the pair is
simply not the extrapolation-fragile class the window exists for. Dropping
that exemption cost two true positives on the corpus (`cad4dae6`: a vehicle
merging in from the right and stopping dead ahead, 55 m to 15 m at constant
bearing, which read as "oblique" only because `fwd_dot` was 0.91 rather than
0.95).

A `lane_confidence_miss_m` clause used to re-earn the lane at any range when
the measured miss was near zero. It was **removed**: on the expanded corpus it
was wrong on every clip it touched (`44681ca9`, `4c18f4cd`, `d53658ef`,
`f0450e0f`, all labelled false positives). The reason is that `d_miss` scales
as `omega * R^2 / v_rel`, so at 40-120 m it is a short-baseline measurement
extrapolated over a long lever arm and cannot resolve the very bend that puts
the target in the wrong lane bucket. The measurement outranks pose only where
its own error is smaller than the pose error it is correcting, which for this
estimator means removing certainty (a *large* miss is robust to that noise),
not restoring it.

These mechanisms are a stopgap for not having a road model. When
`core/acc/road_model.py` is stable enough to consume here, the honest fix is to
project lane membership onto the estimated road instead of onto a
constant-curvature extrapolation, and these range and miss bars should be
re-derived against it rather than carried over.

### Geometry-graded engage fraction

`aeb_engage_frac` (0.85) is a hedge: only take the brake off the driver once
the situation needs most of the truck's capacity, because the geometry that
produced `required_decel` might be wrong. Where the geometry is certain that
hedge buys nothing, so the threshold is graded by the same
`certain_geom_ids` the confirm window already uses (aligned, in-lane,
lane-trusted, not engage-vetoed): `aeb_engage_frac_certain` (0.70) applies
when such a target is colliding, `aeb_engage_frac` otherwise.

This is the fix for the largest missed-positive class on the corpus: 21 of 79
misses were in-lane co-directional rear-ends the pipeline tracked as colliding
and warned on, whose demand peaked at 0.18-0.83 of capacity and so never
crossed a flat 0.85. On a 10 m/s² truck the graded bar engages at about
6.3 m/s² of required decel instead of 7.65, both of which are well past
comfortable braking (2-3 m/s²).

**This knob has no flat band.** Unlike the veto thresholds it is a pure
sensitivity trade, and every value buys true positives at a steady price in
false ones, so it should be re-priced against the corpus whenever the label
set changes rather than treated as settled:

| `aeb_engage_frac_certain` | TP | late | FN | FP | FP cost |
|---|---|---|---|---|---|
| 0.85 (ungraded) | 161 | 5 | 74 | 6 | 0.83 |
| 0.80 | 167 | 4 | 69 | 6 | 0.86 |
| 0.75 | 169 | 4 | 67 | 7 | 1.42 |
| **0.70** | **174** | **6** | **60** | **8** | **3.26** |
| 0.65 | 177 | 5 | 58 | 10 | 8.38 |
| 0.60 | 182 | 9 | 49 | 10 | 10.71 |

0.80 is free. 0.70 is the knee. Below it the price climbs sharply: 0.65 costs
two more false positives and 2.5x the comfort cost for three true positives.
0.60 is available if the miss rate matters more than comfort: it costs the
same clip count as 0.65 but brakes harder on them. Every clip the lower bar
newly brakes on is the same shape as the ones it rescues (a co-directional
slow or stopped lead in ego's lane at 30-60 m), so there is no geometric
scoping that separates them: this really is the sensitivity dial.

Soft crawl / matched-speed in-lane rear-ends whose `v²/2d` stays under the
graded bar still only engage via the ~0.50 s TTB slam (or wait until demand
climbs). A former `certain_engage_ttb` bridge that engaged whenever TTB was
under 1.30 s for certain geometry was removed: at crawl speed that was metres
of bumper gap and felt like AEB ignoring the collision boxes.

### Oncoming clearance: pose plus measurement

`OppositeLaneFilter`'s body-separation fast path suppresses a head-on target
outright when its arc-projected offset already exceeds `ego_hw + v_hw_coll`.
That is the same extrapolation-fragile quantity the engagement vetoes exist to
distrust, and it accounted for 18 of the 79 missed positives: on every one of
them the fast path fired, and on several the measured CBDR miss flatly
contradicted the pose (clip 9cc70333: pose 9.1 m of clearance, measurement
0.44 m).

The fast path now also requires `ctx.d_miss >= clear_bar *
oncoming_body_sep_miss_scale`, failing open when there is no track yet. The
scale is **0.25**, not 1.0: at parity the guard costs 5 false positives for
2 true positives, because the pose-clear and measurement-clear populations
overlap heavily. At 0.25 it only overrides the shortcut when the measurement
says the two will pass within about 0.6 m of centreline separation, which is a
dead-on course, and it costs nothing. The remaining head-on misses need
evidence this system does not have; they are the road model's to fix.

### Latched-threat hold

`AEBThread._latched_threat_ids: set[int]` keeps an engaged target attached
to the pipeline across frames so two effects can hold:

1. **TMP rel-speed pre-filter bypass**: `TmpRelSpeedFilter` (and the
   matching precompute prefilter in `thread.py::loop`) skip the rel-speed
   gate for any id in the latched set. Without this, ego matching a TMP
   convoy partner's speed under braking drops `rel_kmh` below the 15 / 40
   km/h threshold, the target leaves the pipeline, `colliding_ids` empties
   and AEB disarms while the gap may still be unsafe.
2. **Distance-based engagement hold + decel floor**: for every latched id
   still in `vehicle_collision_data`, compute
   `headway = max(dist − stop_buffer, 0) / max(ego_speed, 0.5)`. Release
   the id when it leaves `vehicles_eff`, drops out of `vehicle_collision_data`
   (range/elevation), its headway exceeds `cal.latched_release_headway_s`,
   or it falls out of **scope** (below).
   While any remaining latched id has `headway < cal.latched_min_headway_s`
   set `latched_distance_threat = True`:
   - The disarm gate gains `... and not latched_distance_threat`.
   - `target_raw` is floored at `cal.latched_min_decel_frac · effective_max_decel`
     so the published decel doesn't decay to zero when
     `required_decel = v_closing²/2d` collapses on speed-match.

**Scope release.** The hold exists for one scenario: a forward, in-lane lead
that ego has speed-matched (so it no longer registers as colliding while the
gap is still unsafe). Headway alone is euclidean and direction-blind: without
a geometry re-check, a target ego has evaded around, is driving beside, or a
crosser that swept clear keeps the brake floored at 70 % of max until it is
~1.5 s of *distance* away in any direction. Per frame, each latched id is
checked via `project_to_ego_arc(ego_arc, …)`: it is **in scope** when it is
still in `colliding_ids`, or when `s > 0` (forward of ego along the arc) and
`d_abs ≤ cal.lane_half_width` (EGO lane band). In-scope ids refresh
`AEBThread._latched_scope_ok_mono[vid]`; ids out of scope longer than
`cal.latched_scope_release_s` lose the latch. The grace absorbs
lane-classification flicker (curve transients, One-Euro settling) without
letting a cleared target hold engagement. Scope stamps travel with the clip
warm state (`AEBWarmState.latched_scope_ok_mono`); clips recorded before the
field default to grace-starts-at-window-start.

The set is populated every frame after the engagement state machine via
`self._latched_threat_ids.update(colliding_ids)` while `self._engaged` is
true (newly latched ids get their scope stamp at promotion). Cleared on
`teardown`.

| Knob | Default | Role |
|------|---------|------|
| `latched_min_headway_s` | 1.5 s | Headway below which latched-distance hold fires |
| `latched_release_headway_s` | 2.5 s | Headway above which a latched id is dropped |
| `latched_min_decel_frac` | 0.7 | Fraction of `effective_max_decel` as the `target_raw` floor under hold |
| `latched_scope_release_s` | 0.5 s | Grace before an out-of-scope (not colliding, not forward-in-lane) latched id is dropped |

### Follow-threat flag

Behavioral latch for a genuine slowing lead (co-directional, sustained closing
and own deceleration over `follow_threat_window_s`, then `follow_threat_hold_s`).
While the hold is active, the target must be in `Lane.EGO` **or** arc-projected
`d_abs` must be shrinking (lateral converge): a braking cut-in often decels
before its centre enters ego lane.

Flagged ids bypass `TmpRelSpeedFilter`, are exempt from
`CoDirectionalDivergeFilter`, are exempt from `EgoEvasionFilter` **only while
`ref_kmh_for_filter <= tmp_filter_split_kmh`**, and get follow-track decel
on collision arcs so a braking lead does not clip through on constant-speed
projection. Implemented in `AEBThread._update_follow_threats`.

The speed gate on the evasion exemption is deliberate. The flag is a
lead-following signal of the same shape ACC uses, and following something is
not evidence that it is dangerous, so above the split it must not be able to
wave a target past a geometric filter. Below the split it stays unconditional
because `TmpRelSpeedFilter` is at its most aggressive there
(`rel > tmp_filter_rel_below_kmh` to survive) and would otherwise discard real
low-speed dangers.

| Knob | Default | Role |
|------|---------|------|
| `follow_threat_window_s` | 0.6 s | Trailing kinematic window |
| `follow_threat_hold_s` | 2.0 s | Hold after kinematic qualification |
| `follow_threat_min_decel_ms2` | 0.8 m/s² | Min own-decel slope to qualify |

### Closed-loop coupling

`sending_thread` consumes `AEB_target_decel_ms2` via `AEBDecelController`:
- Feedforward pedal from the inverse brake curve (`_brake_pedal_from_decel`).
- Disturbance observer instead of a PI: a brake-plant model (dead time plus
  first-order lag, keyed on trailer presence) is driven by the pedal actually
  sent to the game, and the filtered residual against measured decel is the
  environment bias. That bias is subtracted from the target before the inverse
  curve, so grade, capacity error and curve error are compensated without an
  integrator to wind up. Both model taus are set slower than the measured
  median on purpose: modelling the plant slower than it is biases the estimate
  toward under-braking, and only over-braking is dangerous.
- Decel measurement for this loop uses its own 0.12 s differentiator, not the
  0.30 s `_spd_smooth` that capacity learning and published telemetry read.
- The merge stays `b = max(b, aeb_pedal)`, so a driver out-braking AEB wins,
  and `AEB_ff_decel_ms2` floors the commanded decel so a stale target cannot
  silence AEB.
- `AEB_required_decel_ms2` is published **uncapped** for this reason: when it
  reaches the decel pedal 1.0 can deliver, sending_thread slams rather than
  inverting the curve at the `ego_decel_frac`-capped target. Do not clamp that
  field to `effective_max_decel`, it is the saturation signal.
- Mapper's fast-PID trim is frozen while `AEB_brake` is true (via
  `AccelToPedals.step(..., freeze_trim=True)`) so two controllers don't
  fight on the brake.
- All three AEB→pedal paths (engagement slam in `main_pedal_thread`, FF
  additive in `sending_thread`, closed-loop controller in `sending_thread`)
  are gated by `gas_output / gasval >= 0.8`: full gas pedal is the user
  override and defeats AEB braking authority across every layer.

---

## 6. Elevation filter (slope-aware)

```python
rz          = _world_to_ego_forward(dx, dz, ego_yaw_rad)
expected_y  = ego_y + rz * math.tan(ego_pitch_rad)
if abs(v.position.y - expected_y) > cal.elevation_margin:
    continue
```

`ego_pitch_rad` is the radar-thread pitch snapshot: telemetry `rotationY`
is inverted before it reaches AEB, so use the published value as-is.

---

## 7. Calibration constants

The full constant-by-constant reference moved to `core/aeb/TUNING.md`. All tunables
live in `AEBCalibration` (frozen dataclass, `core/aeb/calibration.py`); `DEFAULT` is
the production singleton and tests pass a modified instance to `build_pipeline(cal)`
or `evaluate_frame(frame, cal)`.

## 8. Head-on lateral-gap activation

`cal.lane_separation = 3.9 m`: oncoming vehicles whose centerlines are this far
apart laterally are suppressed at the `arc_arc_collision` level
(`min_lateral_gap`). Passed only for `near_head_on` vehicles (`fwd_dot < -0.5`).

---

## 9. Critical Rules: Do Not Break (AEB-specific)

Agent-facing copy of these rules also lives in the top-level `AGENTS.md` (keep that in sync if you change them).

- **AEB is a consumer of `RadarThread`.** Do not open the traffic shared-memory buffer directly and do not mutate `Vehicle` instances.
- **AEB ego curvature is the yaw-rate proxy, full stop.** Do not read `RadarData.ego_curvature` from AEB.
- **Target-vehicle curvature is the `_vehicle_curvature_blend` helper** (sliced position fit blended with `angular_velocity`-derived yaw rate, weighted by `cal.aeb_yaw_blend`, then One-Euro filtered per-vehicle by `AEBThread._curvature_blender`). Do not call `v.curvature_from_history()` directly from AEB paths and do not enlarge `aeb_pos_history_len` toward 25. The blender must be stepped exactly **once per vehicle per frame**: any new call site must thread the existing `ctx.v_curvature` through rather than re-invoking `_vehicle_curvature_blend(...)` with the production blender.
- **`co_directional` must use `fwd_dot > 0.7`, not `abs(fwd_dot) > 0.7`.** The two flags must be mutually exclusive with `head_on`.
- **All tunable constants live in `AEBCalibration`.** Do not introduce new bare numeric literals in `thread.py` or `filters.py`. Add the constant to `calibration.py` first.
- **`lane_frame.project_to_ego_arc` is the canonical lane primitive.** Do not use cross-product `lateral_offset` for lane classification: it compresses on curved roads. The `max(d_arc, d_straight)` formula in `project_to_ego_arc` is the safety-critical fix.
- **`OppositeLaneFilter` body-separation check uses `ego_hw + v_hw_coll`, not corridor width.** The margin (`corridor_margin=0.5 m`) is for probabilistic corridor overlap; body separation uses only actual half-widths.
- **`EgoEvasionFilter` uses `margin=0.0` for evasion arc checks.** Physical body clearance, not padded corridors. Main collision detection still uses `cal.corridor_margin`.
- **Fix B has no ego_k guard.** The `own_lane` check is the only gate; `|ego_curvature|` expands `delta_kappa_t` only if it would actually increase it.
- **Fix D (target arc over-rotation damping) applies to `arc_curvature`, not `v_curvature`.** `v_curvature` is the raw measured value used by `same_curve` and `CoDirectionalDivergeFilter`. Collision and visualization arcs both use the damped `arc_curvature` when building predicted paths.
- **`LaneClassifier` must run before `OppositeLaneFilter`, `CoDirectionalDivergeFilter`, and `EgoEvasionFilter`**: those stages read `ctx.lane`, `ctx.fwd_dot`, `ctx.v_curvature` etc. populated by `LaneClassifier`.
- **TMP trailer-as-vehicles get tractor speed/accel via `_swap_trailer_kinematics`.** Buffer speed for trailer slots is unreliable (often 0). The swap is done on a shallow copy: never mutate the original Vehicle.
- **AEB pedal authority is two-layered, never binary-gated to zero.** AEB publishes `AEB_ff_decel_ms2` every tick when there is any real threat (`required_decel > 0`); sending_thread converts it to a brake pedal via the inverse FF curve and merges it as `b = max(b, aeb_ff_pedal)`. This is the **sub-engagement assist** layer: it adds force on top of user braking when the system warns but has not yet engaged. It is **ramped**, not gated: the assist weight rises linearly from 0 at `cal.ff_assist_ramp_lo` (0.03) to 1 at `cal.user_brake_latch` (0.12), and the merge is `b = max(b, b + (aeb_ff_pedal - b) * w)`. Below the ramp floor it contributes nothing, so it still cannot phantom-brake during normal manual cruising where routine lead-following yields a small non-zero `required_decel` (measured median FF pedal there: 0.004). At or above `user_brake_latch` the weight is 1 and the expression collapses to the original `max(b, aeb_ff_pedal)`, so behaviour above the old gate is unchanged. The ramp exists because a hard gate at 0.12 activated the assist only where the driver was already out-braking it (median jump **−0.187** pedal, i.e. inert) while blocking it across 0.03–0.12 where it would actually add force (median **+0.054**, 51% of those ticks carrying `ff_decel ≥ 2.0`). Dropping the gate outright instead was rejected: it left an 0.877-pedal worst-case jump off a 6% dab, which the ramp cuts to 0.476 and takes to zero above 0.60. When AEB engages (`AEB_brake == True`), main_pedal_thread cuts gas only; the brake is owned by `AEBDecelController` in sending_thread, which tracks `AEB_target_decel_ms2` (the **closed-loop** layer). All AEB pedal paths are gated by `gas_output >= 0.8` (full-gas user authority, the only override that can defeat AEB braking).

**History (2026-08-11):** main_pedal_thread used to slam `brake_output = 1.0` on engagement. Because sending_thread merges every AEB path with `max()`, that slam pinned the pedal at 1.0 for the whole engagement and `AEBDecelController` never influenced the output: `AEB_target_decel_ms2` and its rate limit were dead code. Measured over 32 engagement clips, realized decel was a median **2.25x** the published target, which left **5 to 6 m** of unused gap on 65 km/h stops (0.2 m at crawl, hence the speed-squared symptom). An earlier attempt to drop the slam in favour of *pure FF* was reverted because AEB felt silenced; that failed for two reasons now fixed: the target ramped from 0 at `aeb_target_rate_ms3` (0.8 s to reach the requirement, so the first bite was ~0.005 pedal), and there was no pad for brake build-up. Engagement now steps the target straight to the requirement, and `stop_buffer_response_s` covers the plant lag. Do not restore the slam without re-reading `docs/aeb_high_speed_stop_overshoot.md`: a `max()`-merged constant of 1.0 silently disables every layer beneath it.
- **Engagement-entry vetoes never touch warn timing beyond persistence, and never touch FF assist, disarm, or the holds.** `_los_veto_bar`, `_extrapolation_veto`, and the lane-confidence range all feed `engage_vetoed_ids`, which is subtracted from the engagement-only aggregate chain (`best_ttb_engage` and friends) and from the `certain_geom` instant path. The full aggregates still drive `AEB_warn`, `AEB_ff_decel_ms2`, the disarm gate, the geometry latch, and the latched-distance hold. The one permitted coupling is `aeb_warn_confirm_vetoed_s`: when every colliding target is vetoed **and** out of ego's lane, warn waits on a longer occupancy window. That is a delay a persisting course clears, not a suppression, and a vetoed target may never be removed from the warn aggregate outright. Keep it that way: a wrong veto must cost latency on one target, never silence. Measured on the labelled corpus, the vetoes left warn coverage on positive clips unchanged (135 of 160 clips, identical lead-time distribution) while cutting warn ticks on must-not-trigger clips by 11 %.
- **A measured miss may remove certainty, never grant it.** The vetoes exist because arc-projected lane membership is an extrapolation and the CBDR miss is a measurement, so a *large* measured miss removes certainty (head-on bar, matched-speed neighbour). The converse does not hold: `d_miss` scales as `omega * R^2 / v_rel`, so a small value at range is not evidence of danger, it is a short-baseline fit over a long lever arm. A `lane_confidence_miss_m` clause that restored certainty on a small miss was tried and removed after the corpus grew: it was wrong on all four clips it affected. Also do not let a veto fire with no measurement at all unless its own physics stands alone (the ego-turn branch does; the matched-speed branch deliberately does not).
- **The engage fraction is graded by certainty, and only by certainty.** `aeb_engage_frac_certain` applies when a colliding, non-engage-vetoed target is in `certain_geom_ids`, the same set that grants the instant confirm path. Do not widen it to `nearcertain_geom_ids` or to demand magnitude: required-decel size is not a certainty signal (see the tiered entry gate), and the corpus shows every clip the lower bar newly brakes on is geometrically identical to the ones it rescues. Unlike the veto thresholds it has no flat band, so re-price it against the corpus rather than assuming it still holds.
- **Co-directional targets are exempt from the lane-confidence range and the oblique confirm window.** Both exemptions rest on the same fact: a pair travelling the same way shares whatever bend it is on, so the bend's lateral error is common-mode. Removing either one costs true positives on vehicles merging in and stopping ahead, which read as oblique purely because `fwd_dot` lands just under `aeb_certain_fwd_dot`.
- **Warn suppression while already braking.** `aeb_warn` is suppressed when `_read_user_braking()` is true UNLESS `effective_required >= cal.aeb_warn_near_full_frac × effective_max_decel`. That helper is true for the driver's physical `brakeval` or the mapper's `sending_thread.mapper_command_brake` (CC/ACC/limiter), each above `_USER_BRAKE_LATCH_THRESHOLD` (0.03). Both taps are AEB-free by construction; never source it from `abackward` or `brake_output`, which carry AEB's own slam and FF and would silence the warn during engagement. See section 3 item 6 for why `opdbrakeval` is excluded. The driver / ACC already addressing the threat does not need a redundant alert: only surface it when AEB itself wants near-full brake.

---

## 10. Test suite

```
tests/aeb/
    harness.py          # Frame, EgoState, make_vehicle, evaluate_frame
    test_scenarios.py   # pytest parametrized over 12 scenarios
    report.py           # standalone human-readable table: python -m tests.aeb.report
    scenarios/
        tp_stopped_in_lane.py
        tp_slow_lead.py
        tp_lane_cutter.py
        tp_head_on_in_lane.py
        fp_oncoming_straight.py
        fp_oncoming_gentle_curve.py
        fp_oncoming_sharp_curve.py
        fp_corner_entry_stationary.py
        fp_side_road_uturn.py
        fp_overtaker.py
        fp_co_directional_outer_lane.py
        fp_parked_shoulder.py
```

`tests/aeb/test_engage_vetoes.py` covers the engagement-entry vetoes. The
scenario harness (`evaluate_frame`) stops at the filter pipeline and does not
run the engagement state machine, so those tests drive whole synthetic clips
through `run_headless` instead: ego and an oncoming vehicle on a shared 1200 m
bend with ego reporting zero steer, which is the mechanism the head-on bar
exists for. Only the oncoming lane offset changes between cases (0 m and 2 m
brake, 3 m does not and only warns).

Run with: `pytest tests/aeb -v`
Report: `python -m tests.aeb.report`

### Corpus result

Measured over the 514 labelled clips in the local store (272 must-not-trigger,
242 with a should-trigger window). "vetoes" is the engagement-entry work;
"graded" adds the geometry-graded engage fraction and the oncoming clearance
guard:

| | before | vetoes | + graded |
|---|---|---|---|
| False positives (must-not-trigger clips that engaged) | 46 | 6 | 8 |
| ... in SP | 14 | 4 | 4 |
| ... in TMP | 32 | 2 | 4 |
| Comfort cost of those engagements | 21.18 | 0.83 | 3.26 |
| True positives | 161 | 161 | 174 |
| Late | 5 | 5 | 7 |
| False negatives | 74 | 74 | 59 |
| Total corpus cost | 1038.3 | 1033.2 | 654.5 |

No clip regresses on the positive side against the original at any stage.

The positive side is bit-identical, and setting the new knobs back to their
pre-change values reproduces the old score exactly, so nothing else moved.

Every **veto** threshold sits inside a flat response band rather than on a
cliff, which is the check that those numbers are not fitted to individual
clips: `lane_confidence_range_m` is flat over 25-33 m,
`los_veto_headon_miss_dist_m` over 2.4-2.8 m, `turn_veto_min_kappa` over
0.008-0.018, `turn_veto_min_ttc_s` over 0.8-1.2 s, and
`codir_adjacent_veto_axial_ms` over 1.5-2.0 m/s. `turn_veto_min_kappa` 0.008
removes one more sev-1 false positive but sits one step from 0.006, which
costs a true positive; 0.012 keeps the margin instead.
`aeb_engage_frac_certain` is the exception and has no such band: see the
geometry-graded engage section for its trade curve.

The eight survivors are two stopped-vehicle-at-the-lane-edge clips
(`15eba13d`, `7add71c9`, both sev 1, sitting inside the distribution of
genuine stopped-lead engagements), two long-range oncoming clips (`1c25f5a1`
at 60 m, `27ba3683` at 94 m), two far off-lane clips (`4099ba36`, `6a9c94cd`)
whose ego curvature falls just under the turn veto, and the two slow-lead
clips the graded engage bar buys (`6d23fe39`, `82acb8e8`). Together they cost
3.26, against 21.18 before.

### Steer formula for scenarios

The AEB yaw-rate proxy: `kappa = radians(steer * speed * 12) / speed = radians(steer * 12)`.
Inverting: `steer = kappa * 180 / (12 * pi)` (speed cancels).
Do **not** include `speed` in the inverse formula.

---

## 11. Review and labelling tool

`python -m tools.aeb_review` (dev only, never shipped). `tools/aeb_review.py` holds
the window and the label form; `tools/aeb_review_widgets.py` holds the scene, the
timeline strip, and the background decoder.

### Decode happens off the GUI thread

`ClipStore.load` plus `replay_clip` costs about 0.5 s per clip, so a `ClipLoader`
worker does both on its own thread and the window keeps an LRU of four decoded
clips (~14 MB each). Selecting a row queues the next two, so a pass down the list
hits the cache and lands in ~0.1 s instead of ~0.5 s. The store rescan runs on the
same worker: 652 clips take ~1.1 s and the UI stays live throughout.

`ClipStore.peek_metadata` reads a 64 KB prefix before falling back to the whole
file. Metadata needs a median 14 KB of a 423 KB clip (max seen 20 KB), so a listing
touches a few percent of the store instead of all of it. The fallback matters:
`thumbnail_jpeg` lives in the metadata and can push it past the prefix.

### Desmoothing the recorded decel

`LiveAEB.target_decel_ms2` is what the tick **published**, after the deadband
(`aeb_target_deadband_ms2`, `aeb_target_refresh_min_s`) and the slew limit
(`aeb_target_rate_ms3`). On a real engagement it reads 0.26, 1.11, 1.40, 1.67 over
four ticks while the demand was already at full capacity (10.84 m/s²). Plotting the
published value puts the apparent moment of threat up to half a second late, which
is exactly the moment the reviewer is placing a window against.

`clip_replay.raw_target_decel` rebuilds the pre-slew `target_raw` from the recorded
tick (`engaged`, `time_to_brake` against `brake_ttb + brake_response_window_s`,
`required_decel_ms2` clamped to `effective_max_decel_ms2`) and `replay_clip` puts it
on every `ReviewFrame` as `raw_target_ms2`. It is exact except for the
latched-hold floor (`latched_min_decel_frac`), whose state is not recorded, so it
under-reads during a latched hold. It is never wrong about onset timing, which is
what it is drawn for. `required_decel_ms2` is already raw and needs no rebuild.

### Should-trigger window proposal

`recorded_band` proposes the span the live AEB reacted over: the warn-or-brake
ticks, or the ticks with a finite `time_to_collision` when it stayed silent. Over
the 133 labelled `tp` clips in the local store that band matches the human window
within 0.3 s at **both** ends for 66% of clips (median error 0.01 s start, 0.02 s
end).

The proposal is drawn dashed on the strip and **never applied on load**; `W`
commits it. That is deliberate. The corpus exists to judge AEB, so seeding ground
truth from AEB's own output and saving it unexamined would quietly encode "AEB was
right". `fn` clips are the case that proves it: the recorded band is wrong there by
definition, which is what makes them misses.

---

## 12. Screenshot capture

`core/aeb/screenshot.py::grab_thumbnail()` supplies the optional
`thumbnail_jpeg` clip field; `capture.py` calls it off-thread
(`AEBClipRecorder._start_thumbnail_grab`) so no control loop is touched. Two
properties are load-bearing, not incidental, and must not be relaxed without
updating the claims linked below.

### Game-window crop only

`FindWindowW(None, title)` is tried for "Euro Truck Simulator 2" then
"American Truck Simulator", in that order (`ctypes`, no pywin32 dependency).
The handle is cached and revalidated with `IsWindow` before every capture,
since the game can be restarted between clips. `GetWindowRect` gives the
bbox passed to `ImageGrab.grab(bbox=...)`.

**No game window found means `None`, never a full-monitor grab.** Before
this, `ImageGrab.grab()` took no bbox and captured the whole primary
display: a user with the game on a second monitor uploaded whatever sat on
the first one, and one sampled clip in the local store shows a strip of
desktop chrome along the top edge. Do not reintroduce an unconditional
fallback grab; it defeats the reason this code exists.

Non-Windows platforms never look up a window (`ctypes.windll` does not exist
there) and always return `None`. `_game_window_rect()` checks
`sys.platform` before any `ctypes.windll` access, so importing this module
on Linux CI stays safe; keep any new Windows-only call behind that guard.

### `_MAX_PX = 240`: text must not survive

At the previous `480x270`, sampled real clips had a readable speedometer,
speed-limit signs, a route HUD city name, job cargo/payment figures, and a
full in-game notification sentence. The clip-contribution consent prompt
tells contributors "text is not legible", and that sentence is only true at
240x135, so this constant backs a user-facing claim. Raising it makes the
claim false.

Resolution ladder on one sampled frame at quality 50: 480x270 (17.5 KB) text
readable; 360x203 (10.9 KB) marginal; 240x135 (5.7 KB) illegible with road
layout still clear; 160x90 (3.1 KB) vehicles too mushy to tag. 240x135 was
re-checked against the two hardest sampled frames, a TMP scene with nametags
and a frame with a notification popup, and no text survives either.

Existing clips in the store stay at 480x270; only newly captured ones drop
to 240x135. `tools/aeb_review.py` renders both sizes, aspect-correct, from
the original decoded pixmap each time rather than re-scaling a scaled copy.

### DPI caveat: unresolved, needs a scaled display to verify

`GetWindowRect` returns physical pixels. `ImageGrab.grab(bbox=...)` expects
virtual-screen coordinates. The two agree when the calling process is
per-monitor DPI aware and can disagree otherwise. No scaled secondary
display was available to reproduce the mismatch, so no numeric correction
is applied here: this is a known open item, not a silently-ignored one.

`_dpi_mismatch_note()` logs a debug line when `GetDpiForWindow(hwnd)`
(Windows 10 1607+) reports something other than 96, so a support log at
least carries the signal. That is detection only; the capture bbox is
unchanged either way.

Process-wide DPI awareness is deliberately not touched to fix this: the
Qt UI is not per-monitor DPI aware today, and flipping that process-wide
for one screenshot crop would rescale the entire settings panel and debug
windows. If a clip from a scaled display shows a wrong crop, the fix to
try first is a thread-local DPI awareness override
(`SetThreadDpiAwarenessContext`) around just the capture call, not a
process-wide flip.

---

## 13. Clip contribution intake policy

`core/aeb/intake_policy.py` holds the server's terms for the opt-in clip
sharing: whether intake is open at all, and the floors a clip has to clear.
Nothing uploads yet; this is the gate the uploader will consult.

`contribution_enabled()` also gates capture itself: `core/aeb/capture.py` starts
a recorder when `Settings.debug` **or** the opt-in is current. A contribute-only
user gets a 100 MB store instead of 500 MB, and `capture_tn=False` so the
`shadow_near` and `random` background triggers never fire. Someone who is both
debug and contributing keeps the debug behaviour on both counts.

### Fetched only for users who opted in

`start_policy_fetch()` returns `None` and starts no thread unless
`contribution_enabled()` is true, which requires both `Settings.aeb_contribute`
and a consent version at or above `CONSENT_VERSION`. A machine that never ticked
the box therefore makes **no request to the server at all**, which is what keeps
this a continuation of the consent the user already gave rather than a new
network behaviour. Do not move the fetch above that gate.

It is a plain daemon thread at boot, like `core/update_check`, so the Qt main
thread never waits on it. Failures are logged and swallowed.

### Fail closed

`upload_blocked_reason()` returns a reason string, or `None` to allow. Every
unknown is a refusal: no policy cached, a policy that cannot be parsed, an
unparseable version against a floor, and the dataclass defaults themselves all
refuse. `accepting: false` is the fleet kill switch and needs no client release.

The document is a static JSON file at the edge rather than an endpoint, so it
costs nothing to serve and cannot be knocked over.

### Cache

The **raw response text** is cached in `Settings.aeb_intake_policy_json`, not the
parsed fields, so a field added server-side survives a round trip through an
older client. `aeb_intake_checked` stamps the last successful fetch and is only
advanced after a response parses, so a bad response retries on the next boot
rather than being cached as a refusal for a full window. The refresh window comes
from the cached policy's own `refresh_hours`.

---

## 14. Clip upload

`core/aeb/upload.py` is the **only** module in this package that may send a clip
anywhere, and `tests/aeb/test_upload_egress.py` asserts that by scanning the
package rather than trusting the convention. `intake_policy.py` is the one other
module allowed to make a request at all, and it only fetches the kill switch.

### Consent is checked twice

Once in `capture._on_clip_written`, which is where a clip enters the upload path,
and once in `ClipUploader._handle` before the socket is opened. The first check
is not redundant: since the capture gate widened to `debug or
contribution_enabled()`, the writer callback also fires for debug testers who
never opted in, and their clips must never reach the queue. **A config flag is
never consent.** Both checks read the setting live, so unticking the box stops
uploads immediately rather than at the next boot.

### Eligibility is per clip, never per store

`clip_ineligible_reason()` judges each clip from its own metadata:

- `shadow_near` and `random` are background negatives and are never contributed.
- A thumbnail whose long side exceeds `screenshot._MAX_PX` predates the
  game-window crop and may show the whole monitor with legible text.
- An unreadable thumbnail, unreadable metadata, or a non-null `client_id` all
  refuse. Fail closed.

The capture-time TN exclusion does not cover this, because a user who is both
debug and contributing keeps `capture_tn=True` by design and their store holds
clips from many builds. Judge the image, not a version string.

### Responses

Sends are paced by `_MIN_SEND_GAP_S`. A clip is captured about once a minute, but
the queue holds up to `_QUEUE_MAX`, so a machine that was offline would otherwise
drain it back to back. An edge rate limit counts requests rather than intentions,
and a contributor tripping one on their own recovery traffic loses every clip
still queued, since a paused uploader drops rather than retries. Measured against
the live rule on 2026-08-10: it fires from about the fifth rapid request.

`accepted` and `duplicate` delete locally; `quota` and `closed` pause the whole
uploader for the server's `retry_after_s`; a bare `429` pauses on the status
alone, because an edge throttle answers with an HTML page carrying no `reason`
and would otherwise read as an ordinary refusal. Everything else is kept and not
retried. Network errors and 5xx retry four times with bounded backoff, waiting on
the stop event so shutdown stays prompt. **A debug user never deletes**, whatever
`aeb_delete_after_upload` says: that store is the working corpus.

### One notification per clip

Exactly one of "sent" or "saved" fires for a given clip. The uploader announces a
successful send, coalescing after the first into a summary once per
`_NOTIFY_COOLDOWN_S`, held while AEB is intervening (`note_intervention`, called
from the capture tick). When a clip stays on the machine the `on_kept` callback
fires instead, and `capture._notify_kept` shows the old "AEB clip saved" popup to
debug users only. Refusals and offline machines never claim a send.

`aeb_submissions.jsonl` beside the store records one line per attempt, capped at
5000 lines, carrying no coordinates and no image data. It is what makes
delete-on-ack defensible, and it is also what drives the retry pass below.

### Holdover retry, driven by the log and not by the store

A clip is offered once, at the moment it is written. Without a retry pass an
offline machine, or an hour of server downtime, silently costs every clip
captured in that window: a failed send is abandoned and no later run knows the
clip exists.

`_retry_pending()` runs on the uploader thread before the queue loop. It reads
`SubmissionLog.retryable_clip_ids()`, which returns ids whose **most recent**
entry is in `_RETRYABLE_RESULTS`, and re-offers up to `_RETRY_BATCH` of them
oldest-first through the normal `_handle()` path, so eligibility, consent and
pacing all still apply.

`paused` is one of those results, which is why the pause check sits **after** the
eligibility check rather than first: a clip held back by a pause needs a log
entry to be recoverable, while one that was never going to be sent must stay out
of the log entirely. Without that, hitting the daily cap at noon would lose every
clip captured for the rest of the day.

**A clip with no log entry is unreachable from here.** That is the safety
property, and it holds by construction rather than by a filter: a back
catalogue, anything captured before this feature shipped, and everything the
eligibility rules refuse have no entries, so a retry pass cannot reach them. Do
not replace the log lookup with a store scan.

---

*Source: `core/aeb/thread.py`, `core/aeb/filters.py`, `core/aeb/calibration.py`,
`core/aeb/lane_frame.py`, `core/radar/*`: LD-Tech / MonoCruise.*

