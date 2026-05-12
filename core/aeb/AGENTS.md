# AGENTS.md — AEB (Automatic Emergency Braking)

> AEB-specific reference. Shared fundamentals (coordinate system, rotation,
> world→ego transforms, the shared-memory buffer, Vehicle state/smoothing,
> ArcPath geometry, position-based curvature) live in
> `core/radar/AGENTS.md` — read that first.

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

Vehicle instances are shared references — do not mutate them from AEB. All
smoothing, yaw EMA, position history, and curvature state is produced once
per frame by the radar thread.

### TMP trailer-as-vehicle kinematic swap

In TMP sessions, other players' trailers appear as separate top-level radar
vehicles (`is_tmp=True`, `is_trailer=True`). The shared-memory speed field for
those slots has no engine telemetry — it commonly reports 0 — and the
position-history LS fit can also return 0 during transients (small chord,
stall, fresh spawn). Without correction, AEB sees a "stationary" obstacle
directly ahead of ego and triggers falsely.

`_swap_trailer_kinematics()` in `thread.py` walks the vehicle list once at the
top of each loop. For every `is_tmp` + `is_trailer` entry it locates the
nearest non-trailer TMP vehicle within 30 m (the pulling tractor) and returns
a shallow copy with `speed` and `acceleration` replaced by the tractor's
filtered values. The trailer's pose, yaw, curvature, and trailer-flag stay
its own — only the kinematics it cannot self-measure are inherited. ACC has
the same swap in `core/acc/tracker.py::_top_leads`.

Only the precompute and main collision iterations consume `vehicles_eff`. The
radar visualizer still reads the original `vehicles` so raw vs filtered speed
displays remain meaningful for debugging the source data.

### Ego curvature — two-source AEB-local path

AEB **does not** consume `RadarData.ego_curvature` (the 25-sample
circumscribed-circle position fit consumed by ACC). The ACC fit is
smoothed for stable long-horizon scoring; that same smoothing lags the
truck through transients and either smears the corridor onto the
outgoing lane (FP after the corner clears) or leaves it pointing at the
previous lane (FN into the corner).

AEB maintains its **own** short rolling histories and blends two
signals every frame:

| Signal | Source | Role |
|--------|--------|------|
| `yaw_kappa` | wrap-safe yaw delta over `cal.aeb_yaw_history_len = 3` samples, dt-floor `cal.aeb_yaw_min_dt = 0.03 s` | **Responsive** — picks up rotation within ~2 frames |
| `pos_kappa` | circumscribed-circle fit on `cal.aeb_pos_history_len = 6` ego positions (shared helper `ego_curvature_from_history`) | **Smooth** — damps single-frame yaw noise |

```python
yaw_kappa = _ego_curvature_from_yaw_history(self._ego_yaw_history, ego_speed, cal)
pos_kappa = ego_curvature_from_history(self._ego_position_history)

if yaw_kappa is not None and pos_kappa is not None:
    ego_curvature = cal.aeb_yaw_blend * yaw_kappa + (1 - cal.aeb_yaw_blend) * pos_kappa
elif yaw_kappa is not None:
    ego_curvature = yaw_kappa
elif pos_kappa is not None:
    ego_curvature = pos_kappa
elif ego_speed > 0.5:
    ego_curvature = math.radians(steer * ego_speed * cal.yaw_rate_steer_gain) / ego_speed
else:
    ego_curvature = 0.0
```

`aeb_yaw_blend = 0.6` weights yaw higher: position is a smoother on the
yaw signal, not a co-equal source. The steer proxy survives only as the
last-resort bootstrap before either history is populated.

Both AEB histories are independent of ACC's 25-sample one. Do not enlarge
`aeb_yaw_history_len` or `aeb_pos_history_len` toward ACC parity — the
reactivity loss is the problem, not the transient-sample count.

---

## 2. Module layout

| Module | Role |
|--------|------|
| `core/aeb/calibration.py` | Frozen `AEBCalibration` dataclass — all tunable constants. `DEFAULT` singleton used by both `thread.py` and tests. |
| `core/aeb/lane_frame.py` | `Lane` enum, `project_to_ego_arc()`, `classify()` — arc-projected lane membership, replacing the old cross-product `lateral_offset`. |
| `core/aeb/filters.py` | Named filter pipeline: 12 stage classes + `FilterContext` + `build_pipeline()`. |
| `core/aeb/thread.py` | `AEBThread` — data acquisition, ego-arc construction, pipeline dispatch, TTB/state output. |

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
| `RearOvertakerFilter` | Behind ego, faster — overtake pass |
| `LaneClassifier` | Populates `ctx` geometry fields; sets `ctx.lane` via `lane_frame` |
| `OppositeLaneFilter` | Oncoming vehicles in their own lane (collapses Fix A + Fix B) |
| `CoDirectionalDivergeFilter` | Co-directional arcs already diverging (Fix C + outer-lane same-turn) |
| `TurningCrossTrafficFilter` | Cross-traffic turning through intersection (Fix D absorbed) |
| `TmpCrossTrafficFilter` | TMP-only: target whose extrapolated arc lands outside ego lane |
| `SweepPassFilter` | Stationary cross-traffic ego turns through |
| `CornerEntryStationaryFilter` | Stationary at corner entry — out-of-lane oncoming/co-dir, or in-lane with arc consistency |
| `EgoEvasionFilter` | Ego can steer around target within 0.08 g |

Fix labels A/B/C/D are retired — the logic now lives in the named stages above.

### `LaneClassifier` — canonical lane primitive

`LaneClassifier` is the first stage to populate all geometry fields. It uses
`project_to_ego_arc()` from `core/aeb/lane_frame.py` to compute the arc-projected
lateral offset. For curved ego arcs, the returned `d_abs` is the **maximum** of
the circle-offset and the straight-line heading offset. This prevents a tight ego
turn from projecting an opposite-lane vehicle into the EGO bucket.

Lane thresholds (`cal.lane_half_width=1.95 m`, `cal.lane_separation=3.9 m`):

| d_abs | Lane |
|-------|------|
| ≤ 1.95 m | `EGO` |
| 1.95–1.95 m | `ADJACENT` (empty range — no ADJACENT in practice) |
| 1.95–7.8 m | `OPPOSITE_OR_OUTER` |
| > 7.8 m | `OFF_ROAD` |

### `OppositeLaneFilter`

Applies only to `head_on` vehicles (`fwd_dot < cal.head_on_dot=-0.7`).

1. Lane check: `own_lane = ctx.lane in (OPPOSITE_OR_OUTER, OFF_ROAD)`
2. Body-separation fast path: if `own_lane` and `d_abs >= ego_hw + v_hw_coll`, the
   vehicles already have physical body clearance — suppress directly.
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

Uses a freshly-built non-braking arc (rather than `base_target_arc`) because
the standard arc may be truncated by target-side full-brake modeling at
near-head-on angles, which would mask the true sweep destination.

Non-TMP targets bypass entirely — AI vehicles' arcs are deterministic and
already handled by `OppositeLaneFilter`, `TurningCrossTrafficFilter`, etc.

### `CornerEntryStationaryFilter`

Suppresses stationary targets (`|speed| < sweep_pass_max_target_speed`) at
corner *entry* (`|ego_curvature| < turning_diverge_kappa`) when their pose
implies they sit on a curved road continuation rather than blocking ego's
straight-line path.

- Symmetric road-bend formula: `road_bend = acos(|fwd_dot|)` — folds oncoming
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
aeb.AEB_warn                       # bool — UI/sound cue (warn fraction or TTB)
aeb.AEB_brake                      # bool — engagement latched and target > 0
aeb.AEB_target_decel_ms2           # float — rate-limited commanded decel (m/s²)
aeb.AEB_required_decel_ms2         # float — slope-corrected required decel
aeb.AEB_effective_max_decel_ms2    # float — slope-corrected capacity ceiling
aeb.AEB_realized_decel_ms2         # float — lead-compensated measured decel
aeb.time_to_brake                  # float — seconds (1e9 = no threat)
aeb.em_stop_requested              # bool — mirror of AEB_brake
aeb.snapshot                       # AEBSnapshot — full debug state
```

### Continuous-decel logic

1. Pipeline runs as before. First `suppressed=True` short-circuits.
2. For surviving targets: collision check yields TTB plus per-target
   `closing_distance` and `v_closing` (best by lowest TTB).
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
   - Engage when `effective_required ≥ aeb_engage_frac · effective_max`
   - Disarm when `effective_required <  aeb_disarm_frac  · effective_max`
5. Setpoint pipeline:
   - `target_raw = clamp(effective_required, 0, effective_max)` while engaged
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

### Closed-loop coupling

`sending_thread` consumes `AEB_target_decel_ms2` via `AEBDecelController`:
- Feedforward pedal from the inverse brake curve (`_brake_pedal_from_decel`).
- Small PI on lead-compensated measured decel; integrator freezes on
  pedal saturation so AEB never fights its own anti-windup.
- Mapper's fast-PID trim is frozen while `AEB_brake` is true (via
  `AccelToPedals.step(..., freeze_trim=True)`) so two controllers don't
  fight on the brake.

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

## 7. Quick Reference — calibration constants

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

`cal.lane_separation = 3.9 m` — oncoming vehicles whose centerlines are this far
apart laterally are suppressed at the `arc_arc_collision` level
(`min_lateral_gap`). Passed only for `near_head_on` vehicles (`fwd_dot < -0.5`).

---

## 9. Critical Rules — Do Not Break (AEB-specific)

- **AEB is a consumer of `RadarThread`.** Do not open the traffic shared-memory buffer directly and do not mutate `Vehicle` instances.
- **AEB ego curvature is a blend of AEB-local yaw-delta and AEB-local position-fit curvature (`aeb_yaw_blend` weight on yaw), with the steer proxy as bootstrap fallback.** Do not read `RadarData.ego_curvature` (ACC's 25-sample position fit) from AEB, and do not enlarge `aeb_yaw_history_len` / `aeb_pos_history_len` toward ACC parity.
- **`co_directional` must use `fwd_dot > 0.7`, not `abs(fwd_dot) > 0.7`.** The two flags must be mutually exclusive with `head_on`.
- **All tunable constants live in `AEBCalibration`.** Do not introduce new bare numeric literals in `thread.py` or `filters.py`. Add the constant to `calibration.py` first.
- **`lane_frame.project_to_ego_arc` is the canonical lane primitive.** Do not use cross-product `lateral_offset` for lane classification — it compresses on curved roads. The `max(d_arc, d_straight)` formula in `project_to_ego_arc` is the safety-critical fix.
- **`OppositeLaneFilter` body-separation check uses `ego_hw + v_hw_coll`, not corridor width.** The margin (`corridor_margin=0.5 m`) is for probabilistic corridor overlap; body separation uses only actual half-widths.
- **`EgoEvasionFilter` uses `margin=0.0` for evasion arc checks.** Physical body clearance, not padded corridors. Main collision detection still uses `cal.corridor_margin`.
- **Fix B has no ego_k guard.** The `own_lane` check is the only gate; `|ego_curvature|` expands `delta_kappa_t` only if it would actually increase it.
- **Fix D (target arc over-rotation damping) applies to `arc_curvature`, not `v_curvature`.** `v_curvature` is the raw measured value used by `same_curve` and `CoDirectionalDivergeFilter`. Only the curvature passed to `build_arc()` is scaled.
- **`LaneClassifier` must run before `OppositeLaneFilter`, `CoDirectionalDivergeFilter`, and `EgoEvasionFilter`** — those stages read `ctx.lane`, `ctx.fwd_dot`, `ctx.v_curvature` etc. populated by `LaneClassifier`.
- **TMP trailer-as-vehicles get tractor speed/accel via `_swap_trailer_kinematics`.** Buffer speed for trailer slots is unreliable (often 0). The swap is done on a shallow copy — never mutate the original Vehicle.

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
`core/aeb/lane_frame.py`, `core/radar/*` — LD-Tech / MonoCruise.*
