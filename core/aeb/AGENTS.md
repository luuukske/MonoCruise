# AGENTS.md — ETS2 / MonoCruise Codebase Reference

> Authoritative reference for AI agents working on this codebase.
> Read this before touching any coordinate, rotation, or vehicle logic.

---

## 1. Coordinate System

ETS2 uses a right-handed 3D system. Ground plane = **XZ**. Y = elevation (filter only).

| Axis | Meaning | Increases toward |
|------|---------|-----------------|
| X | Lateral | East (right) |
| Y | Elevation | Up — **never used in 2D math** |
| Z | Longitudinal | South (forward at default orientation) |

Telemetry keys: `coordinateX`, `coordinateY`, `coordinateZ`.

---

## 2. Ego Truck Yaw (`rotationX`)

- Telemetry key is `rotationX` — this is **yaw**, not pitch. The name is wrong.
- Range: `0.0–1.0` (normalised full circle). `0.0 = South`, `0.25 = West`, `0.5 = North`, `0.75 = East`.
- Direction: counter-clockwise when increasing.

### Conversion — pick the right one

```python
# Radar / top-down view (ETS2radar.py) — +0.5 aligns 0 to North (screen-up)
yaw_rad = (yaw_norm + 0.5) * 2 * math.pi

# AEB / arc geometry (thread.py) — NO +0.5 offset
ego_yaw_rad = yaw_norm * 2.0 * math.pi
```

> **WARNING — do not mix these up.**
> Adding `+0.5` in the AEB context rotates the ego forward vector 180°, reversing the ego arc.
> The AEB forward vector `fwd = (-sin, -cos)` already points North at `yaw=0` without the offset.
> If only the ego arc points backward, the bug is in this conversion — not in `traffic.py`.

---

## 3. Traffic Vehicle Rotation (Quaternion)

```python
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = y   # intentional swap — compensates for ETS2 internal axis ordering
        self.y = x   # intentional swap
        self.z = z
```

> **Do not remove the x/y swap.** It is not a bug. Removing it breaks all traffic vehicle rotations.

`euler()` returns `(pitch, yaw, roll)` in degrees:
- Yaw range: `-180` to `+180`
- `yaw = 0` → South
- Positive = counter-clockwise (CCW)

In corner geometry (`get_corners`, `rotate_around_point`) yaw is **negated** (`-yaw`) to convert CCW→CW for screen conventions (numpy / OpenCV).

---

## 4. World → Ego Space

All rendering and scoring is in **ego-space** (ego = origin, forward = screen-up).

```python
dx = vehicle.position.x - ego_x
dz = vehicle.position.z - ego_z
rx, rz = rotate_point(-dx, dz, -yaw_rad)
# rz > 0 = in front of ego
# rx > 0 = to the right of ego
```

Both sign flips are mandatory:
- `-dx` corrects ETS2's leftward X convention.
- `-yaw_rad` rotates the world opposite to ego heading so ego always faces up.

---

## 5. Shared Memory Buffer (`Local\ETS2LATraffic`)

```python
_VEHICLE_FORMAT       = "ffffffffffffhhbb"   # 16 fields
_TRAILER_FORMAT       = "ffffffffff"          # 10 fields per trailer slot
_VEHICLE_OBJECT_FORMAT = _VEHICLE_FORMAT + _TRAILER_FORMAT * 3
_TOTAL_FORMAT         = "=" + _VEHICLE_OBJECT_FORMAT * 40
_BUF_SIZE             = 6960
_VEH_STRIDE           = 46  # fields per vehicle slot (16 + 3*10)
```

### Vehicle field layout (index → field)

| Idx | Field | Type | Notes |
|-----|-------|------|-------|
| 0 | position.x | float | World X |
| 1 | position.y | float | Elevation — never used in 2D |
| 2 | position.z | float | World Z |
| 3 | rotation.w | float | Quaternion W |
| 4 | rotation.x (input) | float | Stored as `self.y` after axis swap |
| 5 | rotation.y (input) | float | Stored as `self.x` after axis swap |
| 6 | rotation.z | float | Quaternion Z |
| 7 | size.width | float | metres |
| 8 | size.height | float | metres — unused in 2D |
| 9 | size.length | float | metres |
| 10 | speed | float | AI = use as-is from buffer (may be signed in singleplayer). TMP = LS fit of longitudinal motion over up to 10 full-frame positions, else single-interval Δ/dt; then speed-dependent EMA of raw speed (see §7). |
| 11 | acceleration | float | m/s² — AI = buffer as-is. TMP buffer value is **ignored**; `Vehicle.acceleration` is EMA of the time derivative of filtered TMP speed (see §7). Arcs use `accel_for_arc()` (= `acceleration`). |
| 12 | trailer_count | short | 0–3 |
| 13 | id | short | Per-frame continuity key |
| 14 | is_tmp | byte | `1` = TMP multiplayer, `0` = AI |
| 15 | is_trailer | byte | `1` = trailer record, `0` = tractor |

### Trailer slots (offsets 16, 26, 36)

Each trailer = 10 floats: `position.x/y/z`, `rotation.w/x/y/z`, `size.width/height/length`.
A slot is valid if `position` is non-zero.

**TMP trailer pivot fix** — TMP pivot is at the front coupler, not center. `correct_position()` shifts it backward:

```python
offset_x = (length / 2) * math.sin(yaw_rad)
offset_z = (length / 2) * math.cos(yaw_rad)
```

Non-TMP trailer positions are already centered — use `tr.position` directly.

---

## 6. Vehicle Corner Geometry

### TMP (multiplayer) — symmetric

```python
back_z  = position.z + length / 2
front_z = position.z - length / 2
left_x  = position.x - width  / 2
right_x = position.x + width  / 2
```

### AI (non-TMP) — asymmetric pivot correction

```python
back_z  = position.z + length * 0.82   # 82% behind pivot
front_z = position.z - length * 0.18   # 18% in front of pivot
```

> Using `0.5/0.5` shifts all AI polygons ~1–2 m forward from their true positions.

### Corner rotation

```python
pitch, yaw, roll = vehicle.rotation.euler()
corner = rotate_around_point(corner, ground_middle, pitch, -yaw, roll=0)
# roll=0 intentional — top-down view ignores banking
```

---

## 7. Vehicle State & Smoothing

### Fields — positions raw; TMP speed/accel filtered

| Field | Source | Use for |
|-------|--------|---------|
| `position.x/z` | **Unfiltered** shared-memory world coordinates | Arc start position, rendering, collision geometry |
| `_smooth_yaw` | Wrap-safe EMA of `rotation.euler()[1]` in radians (`_RAW_YAW_ALPHA = 0.5`, AI and TMP) | **Arc curvature. Never use `rotation.euler()` directly for arcs.** |
| `speed` | AI = buffer as-is. TMP = raw speed from position history (LS on s vs τ along `fwd`), then EMA with α = `_tmp_speed_ema_alpha(|prev.speed|)` | Arc direction, TTB |
| `acceleration` | AI = buffer as-is. TMP = EMA of `(filtered_speed − prev.filtered_speed) / dt` with `_tmp_accel_ema_alpha(|prev.speed|)` (not buffer field 11) | Arc decel/accel via `_accel_to_arc_params()` |
| `angular_velocity` | Degrees/s from rotation delta/dt | Arc curvature via `κ = ω_rad/speed` |

### TMP speed & acceleration — adaptive EMA (speed-dependent α)

World `position.x/z` are **not** low-pass filtered. Raw longitudinal speed is derived
from recent positions (LS fit), then smoothed with a plain EMA so the filtered
estimate tracks the true speed without a separate prediction path drifting ahead
or behind.

Raw speed each full frame (after yaw EMA): keep the last `_TMP_SPEED_HISTORY_LEN`
`(t, x, z)` samples from full updates (`_position_history`). With at least two
samples, fit `s ≈ v·τ` where `s = dot(p − p₀, fwd(smooth_yaw))` and `τ = t − t₀`
(least squares: `v = Σ(τ s)/Σ(τ²)`). If the first→last chord is below 0.025 m,
`raw_speed = 0`. With only one history sample, fall back to the single-interval
formula. Sub-frames still use instantaneous `Δraw/dt` for diagnostics only and do
not push the history.

```python
alpha      = _tmp_speed_ema_alpha(abs(prev.speed))   # 0.5 at rest → 0.15 at 90 km/h
speed      = alpha * raw_speed + (1 - alpha) * prev.speed
kin_accel  = (speed - prev.speed) / dt               # derivative of filtered speed
beta       = _tmp_accel_ema_alpha(abs(prev.speed))   # 0.5 at rest → 0.2 at 90 km/h
accel      = beta * kin_accel + (1 - beta) * prev_accel
```

On the first full frame after spawn, `accel` falls back to
`(raw_speed - prev_raw_speed) / dt` until a filtered speed exists.

`α` decreases with |speed| (more smoothing when fast). TMP never uses buffer
field 11 for physics.

Singleplayer (AI) vehicles skip this block entirely: `speed` and `acceleration`
stay buffer values, positions stay raw.

**Arc / collision** — `Vehicle.accel_for_arc()` is `return self.acceleration`.
TMP vehicles initialise `acceleration = 0` until the first `update_from_last`;
`get_arc()` and `thread.py` always use this field for `_accel_to_arc_params`.

### Lag / freeze detection (TMP vehicles only)

TMP vehicles derive speed from raw position delta.  If ETS2 stops sending
position updates for a vehicle (network lag), the same raw coordinates arrive
every frame, which would snap the derived speed to 0 and then back — causing
false speed readings and, worse, false AEB triggers.

**Detection criterion (per full-update frame):**

```python
raw_disp_sq    = (raw_x - prev._raw_x)² + (raw_z - prev._raw_z)²
expected_disp  = abs(prev.speed) * dt
is_lag         = (abs(prev.speed) > _LAG_MIN_SPEED_MS          # was moving
                  and raw_disp_sq < (expected_disp * _LAG_DISP_RATIO)²)
                 # raw moved less than 10 % of expected displacement
```

**Three-state machine:**

| Elapsed since first frozen frame | Action |
|----------------------------------|--------|
| 0 – `_LAG_FREEZE_DURATION` (0.3 s) | **Freeze** — hold last position; decay speed quadratically: `speed = prev_speed × (1 − frac²)` where `frac = elapsed / 0.3`; force `acceleration = 0`; return early. AEB sees the vehicle at its last known position decelerating toward 0. |
| ≥ 0.3 s | **Release** — set `lag_confirmed = True`, fall through to normal update. Speed falls to 0. AEB detects the stopped obstacle naturally via arc collision. |
| Raw position moves again | Reset `_lag_since = None`, `lag_confirmed = False`. |

`lag_confirmed` is a public flag on `Vehicle`. `thread.py` does not need to
read it: once released, the vehicle's speed = 0 and the existing AEB arc
collision logic handles the stationary obstacle without special-casing.

### Yaw EMA — wrap-safe, α = 0.5 (AI and TMP)

```python
diff       = (raw_yaw - smooth_yaw + math.pi) % (2 * math.pi) - math.pi
smooth_yaw = smooth_yaw + _RAW_YAW_ALPHA * diff   # _RAW_YAW_ALPHA = 0.5
```

### Speed sign detection

**AI (singleplayer):** Use buffer speed as-is. The buffer may already provide
signed speed (positive = forward, negative = reverse). Do not derive or flip
sign from displacement — that can make vehicles appear to move backwards.

**TMP (multiplayer):** Speed magnitude is not trusted from the buffer; on full
frames derive it from up to ten `(t, x, z)` samples (longitudinal LS along `fwd`),
with single-interval fallback and the same forward dot for sign on that path.

```python
# Fallback when history has one sample only:
dist = math.sqrt(disp_x**2 + disp_y**2 + disp_z**2)
direction = 1.0 if (disp_x*fwd_x + disp_z*fwd_z) >= 0.0 else -1.0
speed = direction * dist / dt
```

### Position mismatch (TMP only)

Detects out-of-order packets where the raw position jumps backward along the heading for a limited number of frames.

**Detection:** `dot(raw_disp, prev_smooth_fwd) < -_POS_MISMATCH_BACKWARD_THRESHOLD (-0.05 m)`

**Action:** Increment `_pos_mismatch_frames` counter; hold `_smooth_x/z`; carry `speed` and `acceleration` from prev; return early **after** yaw EMA and angular_velocity have run. Path, arc construction, and all other state are unaffected.

**Cap:** When `_POS_MISMATCH_MAX_FRAMES` is reached (10 frames), the flag is cleared and raw position is passed through on the next frame regardless.

### Crash detection (TMP only)

Fires when **both** rotation jerk (any axis) **and** sporadic position change occur in the same frame window. Runs before position-mismatch and lag early-returns so a crash-induced backward jump is not silently swallowed.

**Rotation jerk** — per-axis rate (deg/s) is computed each full frame from `rotation.euler()` (pitch/yaw/roll). Jerk = change in rate since previous frame. Fires if any axis exceeds its threshold:

| Axis | Jerk threshold |
|------|---------------|
| Pitch | 2 deg/s² |
| Yaw | 15 deg/s² |
| Roll | 2 deg/s² |

**Sporadic position** — fires on either:
- Vertical jump: `|ΔY| > 0.08 m`, or
- XZ direction reversal: `cos(prev_disp, cur_disp) < -0.3` when both displacement magnitudes exceed 0.025 m.

When both signals fire simultaneously: `_crash_since` starts. After `0.10 s`: `crash_confirmed = True`. `_crash_since` resets whenever either signal is absent.

**Effect of `crash_confirmed`:** disables the position-mismatch filter and the lag freeze for that vehicle. Position, speed, and acceleration are derived from raw data as normal. Any displacement — even tiny — passes through unfiltered. Speed and acceleration are **not** overridden; AEB evaluates the vehicle from live kinematics.

### Sub-frame pass (dt < 0.05 s)

**AI:** state unchanged; speed/accel from last full update; pose from `_smooth_x/z`.

**TMP:** pose is snapped to the **latest** buffer `position.x/z` every read (not held at
the last full tick). If `|Δraw| > 0.025 m` over `t_now − prev.time`, **`_raw_speed`** is
recomputed as `±|Δ|/dt` (same forward dot as full updates) for diagnostics; **`speed`**
stays the last full-tick filtered value until the next `dt ≥ 0.05 s` update. Skipped
during lag freeze (`_lag_since` inside `_LAG_FREEZE_DURATION`), position-mismatch hold
(`_pos_mismatch_frames > 0`), and `crash_confirmed`. Acceleration is still carried
from the last full update on sub-frames.

### Game pause

When the game is paused, wall-clock time advances but simulation state (raw positions) does not. On the first frame after unpause, `dt = t_now - prev.time` can be large; TMP `kin_accel` uses that `dt` while position history still constrains `raw_speed`, then the accel EMA softens the step. The TMP speed EMA blends the new raw sample with the previous filtered speed (no separate prediction step).

---

## 8. Arc Path Geometry

Each vehicle carries an `ArcPath` (circular arc or straight ray). Enables O(1) position lookups.

### Forward vector — FIXED, do not change

```python
fwd_x = -math.sin(yaw_rad)
fwd_z = -math.cos(yaw_rad)
```

If the ego arc points backward, the bug is in `thread.py`'s `rotationX → yaw_rad` conversion, not here.

### Key fields

| Field | Description |
|-------|-------------|
| `start_x / start_z` | From smoothed `position` |
| `yaw_rad` | From `_smooth_yaw` — never from `rotation.euler()` directly |
| `speed` | `build()` normalises to `abs` and flips `fwd` if originally negative |
| `curvature` | `κ = ω_rad_s / abs_speed`. Positive = left turn (CCW). |
| `half_width` | `size.width / 2` by default |
| `decel` | Ego braking arc, head-on target arc, or non-head-on target arc when the vehicle is decelerating. Derived via `_accel_to_arc_params()`. Mutually exclusive with `accel`. |
| `arc_length` | Accounts for decel/accel to stop |
| `is_straight` | True if `|κ| < 1e-6` or `speed < 0.001` |

### Arc center (curved only)

```python
# sign = +1 for left turn (κ > 0), -1 for right
center_x = start_x + sign * radius * fwd_z
center_z = start_z + sign * radius * (-fwd_x)
```

### Collision detection

`arc_arc_collision(a, b, margin, n_samples, min_lateral_gap=0.0)` returns `(time_s, hit_x, hit_z)` or `None`.

- Both straight, no decel/accel → closed-form quadratic O(1)
- Otherwise → time-synchronised sampling + 6-step bisection O(n)
- Corridor threshold = `a.half_width + b.half_width + margin`
- AEB narrows vehicle half_width by 0.1 m per side to reduce false positives from measurement noise

#### `min_lateral_gap` — head-on turn filter

When `min_lateral_gap > 0`, a candidate hit is suppressed if the perpendicular distance between the two arc centerlines (measured along `a`'s instantaneous heading at the hit point) is ≥ this value. This prevents false positives when ego and an oncoming vehicle both enter a curve — their arcs overlap in the forward dimension, but the vehicles remain in their own lanes laterally.

```python
# Lateral separation via cross product (2D):
lat = abs((bz - az) * fwd_x_a - (bx - ax) * fwd_z_a)
if lat >= min_lateral_gap:
    suppress hit
```

- Applied in both `_ray_ray_collision` and `_sampled_collision`
- In the sampled path, the lateral check runs at each coarse sample **before** entering bisection; during bisection, a failing lateral check advances `lo` rather than breaking, so the refiner keeps searching for a sample where lanes genuinely cross
- `_LATERAL_LANE_SEPARATION = 3.9 m` — tuned for typical ETS2 2-lane roads so oncoming-centre separation sits safely inside the threshold.
- AEB passes `lateral_gap = _LATERAL_LANE_SEPARATION if head_on else 0.0` to `_earliest_hit`, so the filter is **only active for head-on vehicles**

---

## 9. AEB Thread Interface

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

## 10. Forward Vector & Position Prediction

```python
yaw_rad   = math.radians(yaw_degrees)
forward_x = -math.sin(yaw_rad)   # lateral
forward_z = -math.cos(yaw_rad)   # longitudinal

# Future position (get_position_in)
x_new = position.x - speed * math.sin(yaw_rad)
z_new = position.z - speed * math.cos(yaw_rad)
```

Negative signs are required — without them, `yaw=0` (South) points the wrong way.

---

## 11. Yaw Alignment Scoring

```python
d        = vehicle_yaw_deg - ego_yaw_deg
yaw_diff = min(abs(d), abs(d + 360), abs(d - 360))
# ~0°   → same direction (co-directional, lane candidate)
# ~180° → oncoming traffic
# 45–135° → cross-traffic (AEB applies cross-zone padding)
```

---

## 12. Quick Reference — Formulas

| Formula | Code / Notes |
|---------|-------------|
| Ego yaw → rad (radar) | `(yaw + 0.5) * 2 * pi` |
| Ego yaw → rad (AEB) | `yaw_norm * 2 * pi` (no +0.5) |
| Ego yaw → degrees | `yaw * 360` |
| World → ego-space | `rotate_point(-dx, dz, -yaw_rad)` |
| 2D rotation | `rx = dx*cos(a) - dz*sin(a); rz = dx*sin(a) + dz*cos(a)` |
| Forward vector | `fwd_x = -sin(yaw); fwd_z = -cos(yaw)` |
| Future position | `x -= speed*sin(yaw); z -= speed*cos(yaw)` |
| Yaw wraparound diff | `min(\|d\|, \|d+360\|, \|d-360\|)` |
| TMP corner Z | `± length / 2` |
| AI corner Z | `+length*0.82` (rear), `-length*0.18` (front) |
| Braking distance | `v² / (2 × decel)` (implicit in `build()` when `t_stop < horizon`) |
| Arc accel→decel params | `_accel_to_arc_params(accel, override_decel)` → `(decel, accel)` |
| Quaternion euler yaw | `atan2(2*(y*z + w*x), w²-x²-y²+z²)` degrees |
| Arc curvature | `κ = omega_rad_s / abs_speed` |
| Arc center | `cx = x + sign*R*fwd_z; cz = z + sign*R*(-fwd_x)` |
| TMP raw speed | LS on longitudinal `(t,x,z)` history (max 10 full frames): `v = Σ(τ s)/Σ(τ²)`; else `Δraw/dt`, signed via forward dot |
| TMP smooth speed / accel | `α = _tmp_speed_ema_alpha` (0.5→0.15 @ 90 km/h); `speed = α*raw+(1-α)*prev`; `kin=(speed−prev)/dt`; `β = _tmp_accel_ema_alpha` (0.5→0.2 @ 90 km/h); `accel = β*kin+(1−β)*prev_accel`; buffer 11 unused |
| AI speed / accel | Buffer as-is; no TMP EMA |
| Positions | No EMA — always raw world coordinates |
| Lag detection | `raw_disp < 10 % of (prev_speed × dt)` AND `prev_speed > 2 m/s` → decay speed: `prev_speed × (1 − frac²)`, release after 0.3 s |
| Pos mismatch | `dot(raw_disp, prev_fwd) < -0.05 m` AND `is_tmp` AND `frames < 10` → hold smooth pos + speed, allow yaw |
| Crash detection | rotation jerk (pitch/yaw/roll deg/s²) AND (|ΔY| > 0.08 m OR XZ dir reversal cos < -0.3); both must fire → confirm after 0.10 s; disables pos-mismatch filter and lag freeze; speed/accel stay raw |
| Yaw EMA (wrap-safe) | `smooth += 0.5 * ((raw - smooth + π) % 2π - π)` |
| TMP trailer pivot fix | `pos.x += (len/2)*sin(yaw); pos.z += (len/2)*cos(yaw)` |
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

---

## 13. Position-Based Curvature

Traffic vehicles derive curvature from a circumscribed circle fit over `_position_history` (`curvature_from_history()`), falling back to `angular_velocity / speed` when fewer than 3 samples are available. Ego still uses the yaw-rate proxy (stub pending).

### Traffic vehicles — `Vehicle.curvature_from_history()` ✅

Averages κ over up to four `(oldest, mid, newest)` triples from `_position_history`. Sign from cross product of consecutive displacement vectors: positive = left turn (κ > 0). Returns `0.0` when near-stationary (chord < 5 cm); `None` when < 3 samples (caller falls back to yaw-rate). `_position_history` is populated for both TMP and AI vehicles during full-frame updates in `update_from_last()`.

Used in `get_arc()` and both `v_curvature` sites in `thread.py`:
```python
_hist_k = self.curvature_from_history()
curvature = _hist_k if _hist_k is not None else (math.radians(self.angular_velocity) / abs_speed if abs_speed > 0.5 else 0.0)
```

### Ego — `AEBThread._ego_curvature_from_history()` (stub)

Still returns `None`; falls back to yaw-rate proxy. When implemented, change the `or` fallback to `if ... is not None else` — `0.0` is falsy and would incorrectly fall back.

---

## 14. Critical Rules — Do Not Break

- **No long comments.** Do not write long comments to explain code. edit AGENTS.md if you need to explain something long, otherwise use small one-line comments.
- **Quaternion x/y swap is intentional.** Never remove it.
- **`rotationX` in telemetry is yaw.** The name is misleading.
- **Radar uses `+0.5` offset; AEB does not.** Do not mix them.
- **`-dx` and `-yaw_rad` in ego-space transform are both required.**
- **AI vehicles use asymmetric corner offsets (0.82 / 0.18), not 0.5 / 0.5.**
- **Always use `_smooth_yaw` for arc construction**, never `rotation.euler()` directly.
- **Never use `+0.5` yaw offset in `thread.py`.** It was a historical bug — see inline comment.
- **Y axis is never used in 2D math**, only for elevation filtering: vehicles below or above the expected road level (ego Y + slope × forward distance, ±margin) are not tracked; slope (`rotationY`, positive = uphill) avoids filtering vehicles in front on a slope.
- **AEB forward vector formula is `(-sin, -cos)`.** Do not flip signs or swap to `(sin, cos)`.
- **`co_directional` must use `fwd_dot > 0.7`, not `abs(fwd_dot) > 0.7`.** Using `abs` makes perfectly head-on vehicles (`fwd_dot = -1.0`) simultaneously `head_on=True` and `co_directional=True`. The two flags must be mutually exclusive — `co_directional` means same direction, `head_on` means opposite direction.
- **`_LATERAL_LANE_SEPARATION` is 3.9 m.** Tuned for typical ETS2 2-lane roads so the oncoming vehicle's center sits safely inside the lateral-gap threshold, avoiding boundary misses on perfectly anti-parallel vehicles (`fwd_dot = -1.0`).
- **`near_head_on` (lateral gap activation) and `head_on` (evasion/decel model) are separate thresholds.** `head_on = fwd_dot < -0.7` governs target decel, evasion filter bypass, and risk confirm duration. `near_head_on = fwd_dot < _NEAR_HEAD_ON_DOT (-0.5)` governs only lateral gap activation. Do not unify them — real-world ETS2 turn geometry means oncoming vehicles in a shared curve rarely reach -0.7 during the approach.
- **TMP speed EMA uses `_tmp_speed_ema_alpha(abs(prev.speed))`** — hyperbolic **0.5 at rest → 0.15 at 90 km/h** on raw speed. **`acceleration`** is an EMA of `(speed − prev.speed) / dt` with **`_tmp_accel_ema_alpha`** (**0.5 at rest → 0.2 at 90 km/h**). World positions are not low-pass filtered.
- **TMP `acceleration` is kinematic-only** — buffer field 11 is ignored; `accel_for_arc()` reads `self.acceleration` (smoothed derivative of filtered TMP speed).
- **TMP lag freeze holds position, filtered speed decay, and internal EMA state.** Do not advance position during a freeze — that would snap when updates resume.
- **Lag freeze speed decays quadratically: `prev_speed × (1 − frac²)`.** Never hold speed constant during lag — it keeps AEB informed while smoothly approaching 0.
- **`lag_confirmed` is set by `traffic.py`, not `thread.py`.** thread.py does not need to check it. A confirmed-stopped vehicle has speed = 0 and is detected as a stationary obstacle by the existing arc collision logic.
- **Position mismatch (TMP only) runs before lag detection.** It is mutually exclusive with lag: a backward jump is not near-stationary. The `not _skip_position_update` guard on the lag block enforces this.
- **Position mismatch is capped at `_POS_MISMATCH_MAX_FRAMES (10)`.** When the cap is reached, the next frame always passes raw position through. Without this cap, a genuine crash or prolonged backward event would be silently swallowed.
- **Crash detection does not override speed or acceleration.** It disables the pos-mismatch filter and lag freeze so raw position data passes through unfiltered. Speed and acceleration are derived from live kinematics as normal.
- **Crash detection runs before pos-mismatch and lag early-returns.** A crash-induced backward jump must not be silently swallowed by the pos-mismatch filter. Both signals (rotation jerk and sporadic position) must fire simultaneously; `_crash_since` resets whenever either is absent.
- **`crash_confirmed` and `lag_confirmed` are both handled in `traffic.py`.** Neither requires special-casing in `thread.py` — both produce `speed = 0` which AEB detects as a stationary obstacle naturally.
- **Vehicle longitudinal accel for arcs** — `Vehicle.accel_for_arc()` → `self.acceleration` (TMP = filtered kinematic; AI = buffer). Then `_accel_to_arc_params(accel, override_decel)`. Head-on override (`_FULL_BRAKE_DECEL`) takes priority.
- **AI (singleplayer) speed is used as-is from the buffer.** Do not derive/flip sign from displacement or turning vehicles can be misclassified as reversing.
- **`lateral_offset` is computed once per vehicle** (`abs(dx*ego_fwd_z - dz*ego_fwd_x)`) and reused throughout the per-vehicle loop, including inside the oncoming evasion filter branch. Do not recompute it inside the `elif head_on` block.
- **Fix A applies to `effective_cross_padding`, not `cross_padding`.** `cross_padding` (raw value) is preserved. `effective_cross_padding` is the scaled value used by `_apply_cross_zone`. Never scale `cross_padding` in-place — the raw value may be needed by debug output or future code.
- **Fix B is not a blind suppression.** It expands `delta_kappa_t` so the evasion filter tests a road-following arc. The result must still pass `arc_arc_collision` — if the arc hits ego, the vehicle is not filtered.
- **Fix B has no ego_k guard.** The original `|ego_curvature| >= _TURNING_DIVERGE_CURVATURE` guard was removed — the yaw-rate proxy underestimates curvature on gentle corners and silently blocked Fix B where it was most needed. The `own_lane` check is the only gate.
- **Oncoming evasion arc `decel=0.0` for own-lane vehicles.** `base_target_arc.decel` is `_FULL_BRAKE_DECEL` for all head-on targets. Evasion arcs must not inherit this for own-lane vehicles — a braking arc stops in ~1.3 s inside ego's curved path, causing both left and right clears to be False. Own-lane vehicles will follow the road at speed, not brake.
- **`same_curve` uses `ego_curvature * v_curvature > 0`.** Both must have the same curvature sign and be above `_TURNING_DIVERGE_CURVATURE`. A vehicle drifting across (zero curvature) or cutting the corner wrong (opposite sign) stays at the 2.0 m `_OPPOSITE_LANE_OFFSET` threshold and does not benefit from the reduced `_SAME_CURVE_OWN_LANE_LAT`.
- **`_SAME_CURVE_OWN_LANE_LAT = 1.0 m` is tight by design.** On a shared curve, the cross-product lateral offset compresses due to heading-axis misalignment — a full-lane-away vehicle reads as 1.0–1.5 m. A vehicle genuinely in ego's lane on the same curve would be < 1 m. Do not raise this without understanding the compression effect.
- **Ego arc starts at the front bumper** — `ego_front_x = ego_x + ego_half_l * fwd_x`. `_ARC_START_PCTG` applies to target vehicle arcs only; do not apply it to ego.
- **`_ego_curvature_from_history()` returns `None` (stub).** The `or` fallback in `ego_curvature = self._ego_curvature_from_history() or ego_curvature_yaw` depends on this. When the real implementation is ready, change to `if ... is not None else` — a `0.0` return from a straight-road estimate is falsy and would incorrectly fall back to the yaw-rate proxy.
- **`Vehicle.curvature_from_history()` is implemented.** Returns circumscribed-circle curvature from `_position_history`; `None` when < 3 samples (caller falls back to yaw-rate); `0.0` when near-stationary. Both TMP and AI vehicles populate `_position_history` in `update_from_last()`. Used in `get_arc()` and both `v_curvature` sites in `thread.py`.
- **Fix C extends the co-directional diverge lookahead, not the curvature model.** The inner/outer lane arc overlap is a timing artifact — corridors overlap before centerlines cross. The fix is a longer `_is_approaching` dt, not a wider arc. Do not relax the curvature-sign guard — without it, a vehicle genuinely drifting into ego's lane in a corner could be suppressed.
- **Sweep-pass suppression targets stationary cross-traffic only** (`abs_v_speed < 1.0 m/s`). It must not fire on moving vehicles — a slow-moving vehicle crossing ego's path is a real threat. The ego curvature guard (`|ego_curvature| > _TURNING_DIVERGE_CURVATURE`) ensures it only fires during a real corner, never on a straight road.

---

*Source: `traffic.py`, `thread.py`, `ETS2radar.py` (old code), `classes.py` (old code) — LD-Tech / MonoCruise — March 2026*