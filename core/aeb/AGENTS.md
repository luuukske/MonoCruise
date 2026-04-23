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

### Ego curvature — yaw-rate proxy only

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
"optional" history fallback to AEB — the reactivity loss is the problem,
not the transient-sample count.

---

## 2. TMP rel-speed pre-filter

When **any** slot in the frame has `is_tmp`, AEB pre-filters targets by
**‖v_ego − v_target‖** (km/h) vs a **reference ego speed**:

- ref **> 40 km/h** → threat only if rel **> 15 km/h**
- ref **≤ 40 km/h** → threat only if rel **> 40 km/h**

Reference is current ego speed unless **latched**: the first frame with
`AEBState ≥ WARN` or driver brake > `_USER_BRAKE_LATCH_THRESHOLD` saves
`ego_kmh`; latch held until state drops below WARN **and** brake released;
cleared when the session is no longer TMP.

---

## 3. AEB Thread Interface

Read from other threads (acquire `data._lock` first):

```python
aeb = registry.get_thread("aeb_thread").data
aeb.AEB_warn          # bool — TTB < 1.3 s
aeb.AEB_brake         # bool — TTB < 0.1 s
aeb.time_to_brake     # float — seconds (1e9 = no threat)
aeb.em_stop_requested # bool — mirror of AEB_brake
aeb.snapshot          # AEBSnapshot — full debug state
```

### TTB logic summary

1. Check `ego_arc` (constant speed) vs target → no hit = skip
1b. **Diverging co-directional suppression** — if co-directional and `speed > 0.5 m/s` and paths are already diverging at `t_hit` (`_is_approaching` returns False): skip. Prevents false triggers on overtaking or same-direction vehicles pulling away. **Fix C** — when the target is in the outer lane of the same corner (`lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN`, both curvatures meaningful and same sign), the lookahead is extended to `dynamic_horizon × _CO_SAME_TURN_LOOKAHEAD_SCALE (0.5)`. At the standard 0.25 s lookahead, inner/outer arc corridors overlap before their centerlines cross, so the vehicles still appear converging. The extended lookahead sees the post-crossing divergence.
1c. **Turning cross-traffic suppression** — if `not head_on` and `not co_directional` and `speed > 0.5 m/s` and `|target_curvature| > _TURNING_DIVERGE_CURVATURE (0.03/m)` and paths diverging at `t_hit`: skip. Prevents false triggers when a vehicle is turning through an intersection or corner that ego is entering straight. All five conditions must hold — any failure passes through to full collision evaluation.
2. **Fix A** — before evasion arcs are tested, `cross_zone_padding` is scaled down for
   near-head-on targets (`fwd_dot < -0.5`) that are clearly in their own lane
   (`lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN`). At near-head-on angles
   `sin(angle) ≈ 0.8`, producing ghost arcs ±4 m wide at 10 m/s that phantom-widen
   the target corridor and prevent the evasion filter from clearing. The scale-down
   (`× _NEAR_HEAD_ON_CROSS_SCALE = 0.3`) only fires when lateral displacement confirms
   own-lane placement. Applies to both main hit detection and evasion arc testing.
2a. Evasion filter (non-head-on): check `ego_evasion_left` and `ego_evasion_right` arcs
   (±0.1 g curvature offset) vs **the current target only** (not other vehicles) → if
   either misses the target, vehicle is evasion-filtered (corner/roadside) and skipped.
   Bypassed for moving co-directional and head-on targets. Uses `effective_cross_padding`
   (post Fix A scaling) so ghost arcs are already correctly sized before this runs.
2b. Oncoming evasion filter (head-on only): determines `own_lane` using `same_curve`-
   aware threshold (`_SAME_CURVE_OWN_LANE_LAT = 1.0 m` when both curvatures are
   same-sign and above `_TURNING_DIVERGE_CURVATURE`, else `_OPPOSITE_LANE_OFFSET =
   2.0 m`). For `own_lane` vehicles, builds two curvature-offset arcs with
   `decel=0.0` (vehicle follows road at speed) and scaled `delta_kappa_t`. **Fix B**
   further expands `delta_kappa_t` toward `ego_curvature` magnitude when `own_lane`.
   If either arc clears `ego_arc`, vehicle is suppressed. For non-own-lane vehicles
   uses `base_target_arc.decel` (full brake). Skipped when target speed ≤ 1 m/s.
3. Check `ego_braked_arc` (7.8 m/s² full brake) vs target
   - No hit → braking avoids; `TTB = max(unbraked_ttc - t_stop * buffer, 0)`
   - Hit → braking insufficient; `TTB = 0`
4. State: `TTB < 1.3 s` → WARN; `TTB < 0.1 s` → BRAKE
5. BRAKE latch: holds until `TTB >= 0.3 s`
6. Risk confirmation: vehicle must be continuously risky for `0.1 s` before contributing to TTB
7. Head-on targets (`fwd_dot < -0.7`): modelled as also braking at `_FULL_BRAKE_DECEL`

### Evasion filter (corner / roadside vehicle suppression)

After the unbraked ego arc detects a collision with a target, two additional
ego arcs are tested — one offset left and one offset right in curvature:

```python
delta_kappa = min(_EVASION_G_THRESHOLD / (ego_speed ** 2),
                  _EVASION_FILTER_MAX_DELTA_KAPPA)
left_kappa  = ego_curvature + delta_kappa
right_kappa = ego_curvature - delta_kappa

# Snap evasion paths back toward lane centre when they would cross it
# this is to prevent false activation when turning left into a lane.
if ego_curvature < 0 and left_kappa < 0:
    left_kappa /= 5.0
if ego_curvature > 0 and right_kappa > 0:
    right_kappa /= 5.0

ego_evasion_left  = build_arc(..., left_kappa,  ...)
ego_evasion_right = build_arc(..., right_kappa, ...)
```

- `_EVASION_G_THRESHOLD = 0.1 × 9.81` — the lateral acceleration a gentle
  steer would produce. `Δκ = a_lat / v²` gives the curvature offset at
  the current speed.
- `_EVASION_FILTER_MAX_DELTA_KAPPA = 0.008` — hard clamp so the filter
  arcs stay meaningful at low speed and avoid unrealistic curvature.

A vehicle must collide with **all three** ego paths (centre + left + right)
to be considered a genuine in-lane hazard. The two offset paths are tested
for collision with **that target vehicle only** (not with other traffic). If
either offset path misses the target, ego could steer around it within 0.1 g
— indicating a parked or corner vehicle rather than an obstacle truly
blocking the lane.

**Bypass conditions** — the filter is skipped for:
- Moving co-directional targets (`co_directional and speed > 0.5 m/s`) —
  these are genuinely sharing the lane.
- Head-on traffic (`fwd_dot < -0.7`) — evasion geometry is not meaningful
  when closing head-on.

**Fix A — ghost-arc padding reduction for near-head-on own-lane targets**

Before the evasion arcs are tested, `cross_zone_padding` is conditionally scaled:

```python
effective_cross_padding = cross_padding
if near_head_on and lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN:
    effective_cross_padding *= _NEAR_HEAD_ON_CROSS_SCALE
```

`_apply_cross_zone` uses `effective_cross_padding` rather than the raw value.
At near-head-on angles (`fwd_dot < _NEAR_HEAD_ON_DOT = -0.5`), `sin(angle) ≈ 0.8`,
so at 10 m/s the ghost arcs extend ±4 m along the target heading. This
phantom-widens the target's corridor so both evasion arcs still hit it even when
the vehicle is safely displaced 3+ m into its own lane. The scale-down
(`_NEAR_HEAD_ON_CROSS_SCALE = 0.3`) only fires when `lateral_offset >=
_NEAR_HEAD_ON_LATERAL_MIN (3.0 m)`, confirming own-lane placement.

Filtered vehicles are tracked in `evasion_filtered_ids` (debug only) and
drawn in cyan in the debug window. They do **not** contribute to TTB or
AEB state.

### Oncoming evasion filter (head-on vehicle suppression)

Mirrors the ego evasion filter, but from the oncoming vehicle's perspective.
After `ego_arc` detects a head-on hit, the filter determines whether the target
is genuinely in ego's lane or is a same-road vehicle that will naturally clear.

```python
# 1. Base delta_kappa_t
delta_kappa_t = min(_EVASION_G_THRESHOLD_ONCOMING / (abs_v_speed ** 2),
                    _EVASION_FILTER_MAX_DELTA_KAPPA)

# 2. same_curve: target is on the same curved road as ego
same_curve = (abs(v_curvature) >= _TURNING_DIVERGE_CURVATURE
              and ego_curvature * v_curvature > 0)

# 3. own_lane: target is laterally displaced into its own lane
lane_threshold = _SAME_CURVE_OWN_LANE_LAT if same_curve else _OPPOSITE_LANE_OFFSET
own_lane = lateral_offset >= lane_threshold

# 4. Lateral scaling
if own_lane:
    delta_kappa_t = min(delta_kappa_t * _OPPOSITE_LANE_KAPPA_SCALE,
                        _EVASION_FILTER_MAX_DELTA_KAPPA * _OPPOSITE_LANE_KAPPA_SCALE)

# 5. Fix B — road-following expansion (own_lane only, no ego_k guard)
if own_lane and abs(ego_curvature) >= _TURNING_DIVERGE_CURVATURE:
    delta_kappa_t = max(delta_kappa_t, min(abs(ego_curvature), _SHARED_TURN_MAX_KAPPA))

# 6. Evasion arc decel: zero for own-lane vehicles, full brake for in-lane threats
evasion_decel = 0.0 if own_lane else base_target_arc.decel

tgt_evasion_left  = build_arc(..., target_curvature + delta_kappa_t, ..., decel=evasion_decel)
tgt_evasion_right = build_arc(..., target_curvature - delta_kappa_t, ..., decel=evasion_decel)
```

If either arc clears `ego_arc` the vehicle is skipped and tracked in
`oncoming_evasion_filtered_ids`.

#### `own_lane` determination and `same_curve` threshold

The standard `_OPPOSITE_LANE_OFFSET (2.0 m)` is too high when both vehicles are
on the same curve. Ego's heading axis cuts diagonally across the road — the
cross-product lateral offset `abs(dx*ego_fwd_z - dz*ego_fwd_x)` compresses, and
a vehicle genuinely a full lane away reads as 1.0–1.5 m. The `same_curve` flag
detects this geometry: if the target has same-sign curvature as ego above
`_TURNING_DIVERGE_CURVATURE`, `lane_threshold` drops to `_SAME_CURVE_OWN_LANE_LAT
(1.0 m)`. A vehicle genuinely cutting into ego's lane on the same curve would be
< 1 m laterally displaced even accounting for heading-axis compression.

**Safety property of `same_curve`:** a vehicle drifting straight into ego's lane
has near-zero curvature (fails `abs(v_k) >= _TURNING_DIVERGE_CURVATURE`). A
vehicle cutting the corner in the wrong direction has opposite-sign curvature
(fails `ego_k * v_k > 0`). Both remain at the 2.0 m threshold.

#### Evasion arc decel for own-lane vehicles

The head-on `base_target_arc` carries `decel=_FULL_BRAKE_DECEL (7.8 m/s²)`.
Evasion arcs previously inherited this, stopping the arc in `v/d ≈ 1.3 s` —
right inside ego's curved forward path — causing both `left_clears` and
`right_clears` to be False even when the vehicle is multiple metres into its own
lane. For `own_lane=True`, `evasion_decel=0.0` so the arc runs at full speed
for the full horizon, correctly modelling "vehicle follows the road through the
corner without braking."

#### Fix B — road-following curvature expansion

When `delta_kappa_t` is too small to model the target following the intersection
road, Fix B expands it to `min(|ego_curvature|, _SHARED_TURN_MAX_KAPPA)`.

**ego_k guard removed:** the original guard `|ego_curvature| >= _TURNING_DIVERGE_CURVATURE`
was dropped. The yaw-rate proxy (`steer * speed * 12.0 / speed`) consistently
underestimates curvature on gentle corners — Fix B was silently blocked in
exactly the scenarios where it was most needed. The `own_lane` check is the only
gate required; Fix B still expands only if `|ego_curvature|` would actually
increase `delta_kappa_t`.

#### Lateral-offset kappa scaling

When `own_lane=True`, `delta_kappa_t` is multiplied by `_OPPOSITE_LANE_KAPPA_SCALE
(2.0)` before Fix B is applied. Capped at `_EVASION_FILTER_MAX_DELTA_KAPPA × scale`.

**Conditions:**
- Only runs for `head_on` targets (`fwd_dot < -0.7`). Mutually exclusive with
  the ego evasion filter (`if not head_on` vs `elif head_on`).
- Bypassed when target speed ≤ 1 m/s to avoid Δκ blow-up at near-zero speed.
- `evasion_decel` is `0.0` for `own_lane`, `base_target_arc.decel` otherwise.
- Checked against `ego_arc` directly — **not** against `ego_evasion_left/right`
  and **not** against cross arcs.

`oncoming_evasion_filtered_ids` is stored in `AEBSnapshot` (debug only) and
should be drawn in a distinct colour (e.g. orange) to differentiate from
cyan ego-evasion-filtered vehicles.

---

## 4. Head-on lateral-gap activation

`_LATERAL_LANE_SEPARATION = 3.9 m` — tuned for typical ETS2 2-lane roads so
oncoming-centre separation sits safely inside the threshold. AEB passes
`lateral_gap = _LATERAL_LANE_SEPARATION if head_on else 0.0` to
`_earliest_hit`, so the filter is **only active for head-on vehicles**. See
`core/radar/AGENTS.md` §8 for the underlying `min_lateral_gap` semantics.

---

## 5. Elevation filter (slope-aware)

```python
rz          = _world_to_ego_forward(dx, dz, ego_yaw_rad)
expected_y  = ego_y + rz * math.tan(ego_pitch_rad)
if abs(v.position.y - expected_y) > _ELEVATION_MARGIN_M:
    continue    # skip vehicles above / below the expected road level
```

`ego_pitch_rad` uses `rotationY` (positive = uphill). Slope projection
prevents false skips on vehicles in front of ego on a slope.

---

## 6. Quick Reference — AEB-specific constants / formulas

| Formula | Code / Notes |
|---------|-------------|
| Evasion filter Δκ | `min(0.1*9.81 / v², 0.008)` with additional centreline snap when evasion path would cross lane centre |
| Oncoming evasion filter Δκ | `min(0.13*9.81 / v², 0.008)`, scaled by `_OPPOSITE_LANE_KAPPA_SCALE` when `own_lane`, then Fix B expansion |
| Fix A ghost-arc scale | `_NEAR_HEAD_ON_CROSS_SCALE = 0.3` applied when `near_head_on` and `lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN (3.0 m)` |
| Fix B Δκ expansion | `max(delta_kappa_t, min(\|ego_curvature\|, _SHARED_TURN_MAX_KAPPA (0.05)))` when `own_lane`; no ego_k guard |
| Fix C co-same-turn lookahead | `dynamic_horizon × _CO_SAME_TURN_LOOKAHEAD_SCALE (0.5)` replaces 0.25 s when `lateral_offset >= 3.0 m`, both `|κ| >= _TURNING_DIVERGE_CURVATURE`, same curvature sign |
| `own_lane` determination | `lateral_offset >= lane_threshold`; `lane_threshold = _SAME_CURVE_OWN_LANE_LAT (1.0 m)` if `same_curve` else `_OPPOSITE_LANE_OFFSET (2.0 m)` |
| `same_curve` flag | `abs(v_curvature) >= _TURNING_DIVERGE_CURVATURE and ego_curvature * v_curvature > 0` |
| Oncoming evasion arc decel | `0.0` when `own_lane` (vehicle follows road at speed); `base_target_arc.decel` otherwise |
| Head-on lateral gap | `_LATERAL_LANE_SEPARATION = 3.9 m` (cross product of hit-point separation vs `a.fwd`) |
| Near-head-on threshold | `_NEAR_HEAD_ON_DOT = -0.5` — activates lateral gap; looser than `head_on` (-0.7) to catch shared-turn approach geometry |
| Opposite-lane offset | `_OPPOSITE_LANE_OFFSET = 2.0 m` — lateral distance from ego axis at which oncoming kappa scale activates |
| Opposite-lane kappa scale | `_OPPOSITE_LANE_KAPPA_SCALE = 2.0` — multiplier on `delta_kappa_t` for clearly displaced oncoming vehicles |
| Turning diverge curvature threshold | `_TURNING_DIVERGE_CURVATURE = 0.007 /m` (≈ 143 m radius) |
| Sweep-pass suppression | stationary target (`abs_v_speed < _SWEEP_PASS_MAX_TARGET_SPEED (1.0 m/s)`) + ego in real corner (`|ego_curvature| > _TURNING_DIVERGE_CURVATURE`); at `t_hit`, compute ego heading and position on arc; suppress if `dot(ego_fwd_at_hit, vehicle_pos − ego_pos_at_hit) <= 0` (vehicle behind ego's heading — arc swept through, not a real collision) |
| Corner-entry stationary suppression | stationary target + `fwd_dot < -0.3` + `|ego_curvature| < _TURNING_DIVERGE_CURVATURE` (corner entry, ego not yet turning) + `lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN (3.0 m)` + `dist > 1.0 m`; infer road curvature from target yaw: `implied_kappa = acos(-fwd_dot) / dist`; suppress if `implied_kappa > _TURNING_DIVERGE_CURVATURE`. `lateral_offset` gate preserves genuine in-lane threats (straight ahead → near-zero lateral offset → gate fails). Adds to `oncoming_evasion_filtered_ids`. |

---

## 7. Critical Rules — Do Not Break (AEB-specific)

- **AEB is a consumer of `RadarThread`.** Do not open the traffic shared-memory buffer directly and do not mutate `Vehicle` instances — the radar thread owns per-id smoothing state.
- **AEB ego curvature is the yaw-rate proxy, full stop.** Do not read `RadarData.ego_curvature` from AEB. The ego arc cannot tolerate smoothing or history lag — the proxy (`steer * speed * 12.0 / speed`) reacts instantly to driver input. `RadarData.ego_curvature` is reserved for ACC.
- **`co_directional` must use `fwd_dot > 0.7`, not `abs(fwd_dot) > 0.7`.** Using `abs` makes perfectly head-on vehicles (`fwd_dot = -1.0`) simultaneously `head_on=True` and `co_directional=True`. The two flags must be mutually exclusive.
- **`_LATERAL_LANE_SEPARATION` is 3.9 m.** Tuned for typical ETS2 2-lane roads so the oncoming vehicle's center sits safely inside the lateral-gap threshold, avoiding boundary misses on perfectly anti-parallel vehicles (`fwd_dot = -1.0`).
- **`near_head_on` (lateral gap activation) and `head_on` (evasion/decel model) are separate thresholds.** `head_on = fwd_dot < -0.7` governs target decel, evasion filter bypass, and risk confirm duration. `near_head_on = fwd_dot < _NEAR_HEAD_ON_DOT (-0.5)` governs only lateral gap activation. Do not unify them.
- **`lateral_offset` is computed once per vehicle** (`abs(dx*ego_fwd_z - dz*ego_fwd_x)`) and reused throughout the per-vehicle loop, including inside the oncoming evasion filter branch. Do not recompute it inside the `elif head_on` block.
- **Fix A applies to `effective_cross_padding`, not `cross_padding`.** `cross_padding` (raw value) is preserved. `effective_cross_padding` is the scaled value used by `_apply_cross_zone`. Never scale `cross_padding` in-place.
- **Fix B is not a blind suppression.** It expands `delta_kappa_t` so the evasion filter tests a road-following arc. The result must still pass `arc_arc_collision` — if the arc hits ego, the vehicle is not filtered.
- **Fix B has no ego_k guard.** The `own_lane` check is the only gate.
- **Oncoming evasion arc `decel=0.0` for own-lane vehicles.** `base_target_arc.decel` is `_FULL_BRAKE_DECEL` for all head-on targets. Evasion arcs must not inherit this for own-lane vehicles.
- **`same_curve` uses `ego_curvature * v_curvature > 0`.** Both must have the same curvature sign and be above `_TURNING_DIVERGE_CURVATURE`.
- **`_SAME_CURVE_OWN_LANE_LAT = 1.0 m` is tight by design.** On a shared curve, the cross-product lateral offset compresses due to heading-axis misalignment. Do not raise this without understanding the compression effect.
- **Fix C extends the co-directional diverge lookahead, not the curvature model.** The inner/outer lane arc overlap is a timing artifact — corridors overlap before centerlines cross. Do not relax the curvature-sign guard.
- **Sweep-pass suppression targets stationary cross-traffic only** (`abs_v_speed < 1.0 m/s`). It must not fire on moving vehicles. The ego curvature guard (`|ego_curvature| > _TURNING_DIVERGE_CURVATURE`) ensures it only fires during a real corner.
- **Corner-entry stationary suppression requires `lateral_offset >= _NEAR_HEAD_ON_LATERAL_MIN`.** This gate is safety-critical: a stationary vehicle blocking ego's own lane on a curve has near-zero lateral displacement from ego's current heading axis. Removing or loosening this gate risks suppressing real in-lane threats.

---

*Source: `core/aeb/thread.py`, `core/radar/*` — LD-Tech / MonoCruise.*
