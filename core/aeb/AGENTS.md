# AGENTS.md: AEB (Automatic Emergency Braking)

> AEB-specific reference. Shared fundamentals (coordinate system, rotation,
> world→ego transforms, the shared-memory buffer, Vehicle state/smoothing,
> ArcPath geometry, position-based curvature) live in
> `core/radar/AGENTS.md`: read that first.

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
| `TmpCrossTrafficFilter` | TMP-only: target whose extrapolated arc lands outside ego lane |
| `SweepPassFilter` | Stationary cross-traffic ego turns through |
| `CornerEntryStationaryFilter` | Stationary at corner entry: out-of-lane oncoming/co-dir, or in-lane with arc consistency |
| `EgoEvasionFilter` | Ego can steer around target within 0.08 g |

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

Applies only to `head_on` vehicles (`fwd_dot < cal.head_on_dot=-0.7`).

1. Lane check: `own_lane = ctx.lane in (OPPOSITE_OR_OUTER, OFF_ROAD)`
2. Body-separation fast path: if `own_lane` and `d_abs >= ego_hw + v_hw_coll`, the
   vehicles already have physical body clearance: suppress directly.
3. Evasion arc test: for each target arc, build two curvature-offset arcs
   (`base_curvature ± delta_kappa_t`, `decel=0`). If either clears `ego_arc`, suppress.
   - `delta_kappa_t = min(evasion_g_oncoming / v², evasion_max_dkappa)`, scaled by
     `opposite_lane_kappa_scale` when `own_lane`.
   - Fix B: `delta_kappa_t = max(delta_kappa_t, min(|ego_curvature|, shared_turn_max_kappa))`
     when `own_lane` and ego is turning.

### `CoDirectionalDivergeFilter`

Applies only to `co_directional` vehicles (`fwd_dot > cal.co_directional_dot=0.7`).
For each arc with `speed > 0.5 m/s`: if `_is_approaching` returns False at the hit
point (with lookahead `co_dir_diverge_lookahead_s=0.25 s`), suppress.
Extended lookahead (`dynamic_horizon × co_same_turn_lookahead_scale=0.5`) when
all four conditions hold: vehicle in outer lane, both curvatures above threshold,
same curvature sign.

### `TmpCrossTrafficFilter`

TMP-only filter that absorbs MP-data uncertainty for routine intersection
maneuvers. TMP position/yaw/curvature snapshots are jittered enough that an
in-progress turn at a side road can briefly project an arc through ego's
lane even though the actual MP target is sweeping past. For TMP vehicles
(`v.is_tmp=True`) that aren't co-directional and have non-trivial speed:
build a non-braked "sweep" arc from the snapshot's `(start, yaw, curvature,
speed)` over the full horizon and check the endpoint's lane via
`project_to_ego_arc()`. If the arc terminates in `OPPOSITE_OR_OUTER` or
`OFF_ROAD`, the target sweeps clear of ego's lane → suppress. If the
endpoint is in `Lane.EGO`, this is a real continuing threat → fall through.

Uses a freshly-built non-braking arc from the **undamped** `ctx.v_curvature`
(rather than `base_target_arc`) for two reasons: the standard arc may be
truncated by target-side full-brake modeling at near-head-on angles, and its
Fix D over-rotation damping straightens the arc of a target genuinely
sweeping through a corner. Either one parks the projected endpoint inside
ego's lane and masks the true sweep destination, so the filter never
suppresses and the corner cross-traffic becomes a phantom brake.

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
body clearance, not padded corridor). Bypassed for head-on targets and moving
co-directional in-lane vehicles.

---

## 4. TMP rel-speed pre-filter

When **any** slot in the frame has `is_tmp`, AEB pre-filters targets by
**‖v_ego − v_target‖** (km/h) vs a **reference ego speed**:

- ref **> 40 km/h** → threat only if rel **> 15 km/h**
- ref **≤ 40 km/h** → threat only if rel **> 40 km/h**

Reference is current ego speed unless **latched**: the first frame with
`AEBState ≥ WARN` or driver brake > `user_brake_latch` saves `ego_kmh`;
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
     - `v_target_along_ego > ego_speed` (target faster than ego along
       ego's heading: pure rear-overtaker shortcut; handles imminent
       collisions where `t_braked` is too small for the comparison below
       to fire), **or**
     - `closing_braked > closing_unbraked + brake_worsens_hysteresis_ms`
       (cross-traffic where braking parks ego in target's path).
   - Targets flagged `braking_worsens` are added to `braking_worsens_ids`
     and excluded from `best_ttb` / `best_v_closing`. AEB engagement on
     these is forbidden.
   - Non-worsens targets set `ttb = unbraked_ttc` and contribute to
     `best_ttb`, `best_closing_distance`, `best_v_closing` (all from the
     lowest-`ttb` target).
3. Required decel for the worst target:
   ```
   d_remaining       = closing_distance - stop_buffer
   required_decel    = v_closing² / (2 * max(d_remaining, 1e-3))
   slope_accel       = g · sin(ego_pitch_rad)         # +ve = uphill (radar convention)
   downhill_offset   = max(−slope_accel, 0)           # gravity stealing brake force
   effective_max     = ego_decel_frac · capacity_estimate − downhill_offset
   effective_required= required_decel + downhill_offset
   ```
   `capacity_estimate` is read from `sending_thread.data.max_brake_ms2`
   (PedalCapacityTracker) with a fallback constant.
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
     while any colliding target has `unbraked_ttc < warn_ttb`. Prevents
     disarm mid-event as ego decelerates and `best_v_closing` collapses
     faster than `d_remaining`.
   - Latched-distance hold: see "Latched-threat hold" below. Adds a
     headway-driven engagement hold over targets that have been engaged
     on previously, independent of current `v_closing`.
   - Engage when `effective_required ≥ aeb_engage_frac · effective_max` **OR**
     `brake_ttb_active`.
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
     OR `time_to_brake < warn_ttb`.
   - `AEB_brake` is true while engagement is latched and the published target
     is above zero. Other subsystems (cruise/HMI) gate off this flag.
7. Hold semantics: warn/brake state holds for 0.3 s after a downgrade to
   suppress chatter, identical to the old WARN→STANDBY hold.
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
`d_miss = |omega_los| * R^2 / |v_rel|`.

Veto fires only when the track has `los_veto_min_samples`, the target is
beyond `los_veto_min_range_m`, and `d_miss > los_veto_miss_dist_m`
(corpus separation: genuine engagement edges 0.05-4.4 m, corner phantoms
6.8-12.3 m). Scope is strictly engagement *entry*: warn, disarm, geometry
latch, and distance holds all keep the full aggregates, so a wrong veto
costs latency on one target, never silence, and close-in turning threats
are protected by the range floor. Vetoed ids are published in
`snapshot.los_vetoed_ids`.

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
   (range/elevation), or its headway exceeds `cal.latched_release_headway_s`.
   While any remaining latched id has `headway < cal.latched_min_headway_s`
   set `latched_distance_threat = True`:
   - The disarm gate gains `... and not latched_distance_threat`.
   - `target_raw` is floored at `cal.latched_min_decel_frac · effective_max_decel`
     so the published decel doesn't decay to zero when
     `required_decel = v_closing²/2d` collapses on speed-match.

The set is populated every frame after the engagement state machine via
`self._latched_threat_ids.update(colliding_ids)` while `self._engaged` is
true. Cleared on `teardown`.

| Knob | Default | Role |
|------|---------|------|
| `latched_min_headway_s` | 1.5 s | Headway below which latched-distance hold fires |
| `latched_release_headway_s` | 2.5 s | Headway above which a latched id is dropped |
| `latched_min_decel_frac` | 0.7 | Fraction of `effective_max_decel` as the `target_raw` floor under hold |

### Closed-loop coupling

`sending_thread` consumes `AEB_target_decel_ms2` via `AEBDecelController`:
- Feedforward pedal from the inverse brake curve (`_brake_pedal_from_decel`).
- Small PI on lead-compensated measured decel; integrator freezes on
  pedal saturation so AEB never fights its own anti-windup.
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

## 7. Quick Reference: calibration constants

All constants live in `AEBCalibration` (frozen dataclass, `core/aeb/calibration.py`).
`DEFAULT = AEBCalibration()` is the production singleton. Tests can pass a modified
instance to `build_pipeline(cal)` or `evaluate_frame(frame, cal)`.

| Constant | Default | Role |
|----------|---------|------|
| `full_brake_decel` | 7.8 m/s² | Full brake deceleration |
| `warn_ttb` | 1.3 s | WARN threshold |
| `brake_ttb` | 0.2 s | BRAKE threshold |
| `ego_half_width` | 1.15 m | Ego arc corridor half-width |
| `lane_half_width` | 1.95 m | EGO lane boundary |
| `lane_separation` | 3.9 m | Road lane pitch |
| `head_on_dot` | -0.7 | `head_on` flag threshold |
| `co_directional_dot` | 0.7 | `co_directional` flag threshold |
| `evasion_g` | 0.08×9.81 | Ego evasion lateral accel |
| `evasion_g_oncoming` | 0.13×9.81 | Oncoming evasion lateral accel |
| `evasion_max_dkappa` | 0.008 /m | Max curvature offset for evasion arcs |
| `opposite_lane_kappa_scale` | 2.0 | Kappa multiplier when target in own lane |
| `turning_diverge_kappa` | 0.007 /m | Corner threshold for Fix-C/D conditions |
| `co_same_turn_lookahead_scale` | 0.5 | Extended lookahead fraction of horizon |
| `corner_entry_min_road_bend` | 0.10 rad | Min ego↔tangent angle for Mode-B suppression |
| `corner_entry_min_lateral` | 0.4 m | Min |lat_signed| to claim "off ego axis" (Mode B) |
| `corner_entry_lateral_tol` | 1.5 m | Chord-offset tolerance for arc-consistency check (Mode B) |

---

## 8. Head-on lateral-gap activation

`cal.lane_separation = 3.9 m`: oncoming vehicles whose centerlines are this far
apart laterally are suppressed at the `arc_arc_collision` level
(`min_lateral_gap`). Passed only for `near_head_on` vehicles (`fwd_dot < -0.5`).

---

## 9. Critical Rules: Do Not Break (AEB-specific)

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
- **AEB pedal authority is two-layered, never binary-gated to zero.** AEB publishes `AEB_ff_decel_ms2` every tick when there is any real threat (`required_decel > 0`); sending_thread converts it to a brake pedal via the inverse FF curve and merges it as `b = max(b, aeb_ff_pedal)`. This is the **sub-engagement assist** layer: it adds force on top of user braking when the system warns but has not yet engaged, and is gated on `brakeval > cal.user_brake_latch` so it does not phantom-brake during normal manual cruising when routine lead-following produces a small but non-zero `required_decel`. When AEB engages (`AEB_brake == True`), main_pedal_thread slams `brake_output = 1.0` (the **engagement slam** layer): by definition, engagement means the system has decided full braking is warranted, and the inverse FF curve at a modest required-decel would produce a pedal too soft to act on the threat. The engagement slam is independent of `brakeval` (it must fire even on a distracted driver). All AEB pedal paths are gated by `gas_output >= 0.8` (full-gas user authority, the only override that can defeat AEB braking). Reason: removing the engagement slam in favour of pure FF made AEB feel silenced on engagement because FF pedal for 3–5 m/s² is only ~0.14–0.34.
- **Warn suppression while user braking.** `aeb_warn` is suppressed when `brakeval > cal.user_brake_latch` UNLESS `effective_required >= cal.aeb_warn_near_full_frac × effective_max_decel`. The user does not need a redundant alert while addressing the threat: only surface it when AEB itself wants near-full brake.

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

Run with: `pytest tests/aeb -v`
Report: `python -m tests.aeb.report`

### Steer formula for scenarios

The AEB yaw-rate proxy: `kappa = radians(steer * speed * 12) / speed = radians(steer * 12)`.
Inverting: `steer = kappa * 180 / (12 * pi)` (speed cancels).
Do **not** include `speed` in the inverse formula.

---

*Source: `core/aeb/thread.py`, `core/aeb/filters.py`, `core/aeb/calibration.py`,
`core/aeb/lane_frame.py`, `core/radar/*`: LD-Tech / MonoCruise.*

