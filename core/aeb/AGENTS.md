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
| 10 | speed | float | AI = use as-is from buffer (may be signed in singleplayer); TMP = magnitude from buffer, sign from displacement dot product |
| 11 | acceleration | float | m/s² — converted to arc params via `_accel_to_arc_params()`: braking (< 0) → `decel = min(|accel|, 6.0)`; accelerating (≥ 0) → `accel = min(accel, 4.0)`; head-on override uses `_FULL_BRAKE_DECEL`. Crash-induced backward jump spikes are suppressed by the 6 m/s² cap. |
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

### Smoothed fields — use these, not raw values

| Field | Source | Use for |
|-------|--------|---------|
| `position.x/z` | Dynamic-alpha EMA in `update_from_last()` — alpha = 1.0 at 0 km/h, 0.15 at 90 km/h, further reduced by rolling noise | All rendering, arc start position |
| `_smooth_yaw` | Wrap-safe EMA of `rotation.euler()[1]` in radians | **Arc curvature. Never use `rotation.euler()` directly for arcs.** |
| `speed` | Signed m/s, set in `update_from_last()` | Arc direction, speed display |
| `angular_velocity` | Degrees/s from rotation delta/dt | Arc curvature via `κ = ω_rad/speed` |

### Position smoothing — prediction-corrected EMA with dynamic alpha

Pure EMA blends `raw` against `prev_smooth`, producing a steady-state lag of
`((1 − α) / α) × dt × speed` (≈ 4.4 m at 80 km/h with α = 0.20).

Instead, `prev_smooth` is replaced by a **kinematic prediction** of where the
vehicle should be this frame:

```python
pred_dist = speed * dt + 0.5 * clamp(acceleration, -6, 4) * dt²
pred_x    = smooth_x_prev + pred_dist * (-sin(smooth_yaw))
pred_z    = smooth_z_prev + pred_dist * (-cos(smooth_yaw))

alpha     = _compute_position_alpha(prev.speed, self._noise_est)
smooth_x  = alpha * raw_x + (1 - alpha) * pred_x
smooth_z  = alpha * raw_z + (1 - alpha) * pred_z
```

`alpha` follows a **hyperbolic (1/x) curve** implemented in `_compute_position_alpha`:

- Base component: 1.0 at rest → 0.15 at 90 km/h, using a pole `d` derived from `_ALPHA_SPEED_SCALE`.
- Noise modifier: large rolling residuals (`|raw − predicted|`) reduce the effective alpha, making the filter trust prediction more when the input is noisy.

The curve drops steeply at low speeds (noise matters more at short displacement) and flattens toward ~0.15 at highway speed (where lag spikes dominate). A minimum floor is always enforced so the filter never fully ignores new measurements.

`smooth_yaw` (not `rotation.euler()`) is used for the prediction forward vector —
it is already available from `prev._smooth_yaw` at this point in the update.
`speed` is already sign-corrected before this block runs.

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
| 0 – `_LAG_FREEZE_DURATION` (0.3 s) | **Freeze** — hold smooth position and `_noise_est`; decay speed quadratically: `speed = prev_speed × (1 − frac²)` where `frac = elapsed / 0.3`; force `acceleration = 0`; return early. AEB sees the vehicle at its last known position decelerating toward 0. |
| ≥ 0.3 s | **Release** — set `lag_confirmed = True`, fall through to normal update. Speed falls to 0. AEB detects the stopped obstacle naturally via arc collision. |
| Raw position moves again | Reset `_lag_since = None`, `lag_confirmed = False`. |

`lag_confirmed` is a public flag on `Vehicle`. `thread.py` does not need to
read it: once released, the vehicle's speed = 0 and the existing AEB arc
collision logic handles the stationary obstacle without special-casing.

### Yaw EMA (unchanged) — wrap-safe

```python
diff       = (raw_yaw - smooth_yaw + math.pi) % (2 * math.pi) - math.pi
smooth_yaw = smooth_yaw + yaw_alpha * diff
```

### Speed sign detection

**AI (singleplayer):** Use buffer speed as-is. The buffer may already provide
signed speed (positive = forward, negative = reverse). Do not derive or flip
sign from displacement — that can make vehicles appear to move backwards.

**TMP (multiplayer):** Speed magnitude is not trusted from the buffer; derive
from raw position delta / dt and use forward dot product for sign:

```python
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

Evidence accumulator driven by three independent signals. Once evidence holds above threshold for `_CRASH_CONFIRM_DURATION`, `crash_confirmed` is set. Position EMA continues normally; only speed and acceleration are filtered.

| Signal | Evidence added | Threshold |
|--------|---------------|-----------|
| Raw yaw rate | +0.35 | > 30 deg/s |
| Backward raw displacement | +0.30 | dot < -0.3 m |
| Micro-oscillation | +0.20 | `raw_disp_sq < 0.0225 m²` AND `speed > 1 m/s` |

Evidence decays by `× 0.7` each full frame. `_crash_since` starts when evidence first exceeds `0.75`. After `0.25 s`: `crash_confirmed = True`; speed decelerates at `10 m/s²` toward 0; `acceleration = 0`. AEB handles the stopped obstacle via arc collision.

### Sub-frame pass (dt < 0.05 s)

`update_from_last()` carries forward all smoothed state unchanged. Speed sign is preserved from the last full update. **Do not re-derive speed sign on sub-frame passes.**

### Game pause

When the game is paused, wall-clock time advances but simulation state (raw positions) does not. On the first frame after unpause, `dt = t_now - prev.time` can be large (e.g. tens of seconds). Using that full `dt` for the kinematic prediction would extrapolate motion and push smoothed positions far ahead of the (unchanged) raw positions.

For this reason, the prediction step uses a capped dt: `dt_pred = min(dt, _MAX_PREDICTION_DT)` only for computing the predicted offset (`_mid_yaw` and `_pred_dist`). Real `dt` is still used for the sub-frame check, angular velocity, and speed derivation so behaviour remains correct.

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
1b. **Diverging co-directional suppression** — if co-directional and `speed > 0.5 m/s` and paths are already diverging at `t_hit` (`_is_approaching` returns False): skip. Prevents false triggers on overtaking or same-direction vehicles pulling away.
1c. **Turning cross-traffic suppression** — if `not head_on` and `not co_directional` and `speed > 0.5 m/s` and `|target_curvature| > _TURNING_DIVERGE_CURVATURE (0.03/m)` and paths diverging at `t_hit`: skip. Prevents false triggers when a vehicle is turning through an intersection or corner that ego is entering straight. All five conditions must hold — any failure passes through to full collision evaluation.
2. Evasion filter (non-head-on): check `ego_evasion_left` and `ego_evasion_right` arcs
   (±0.1 g curvature offset) vs **the current target only** (not other vehicles) → if
   either misses the target, vehicle is evasion-filtered (corner/roadside) and skipped.
   Bypassed for moving co-directional and head-on targets.
2b. Oncoming evasion filter (head-on only): build two curvature-offset arcs for the
   *target* (same ±0.1 g Δκ) and test them against `ego_arc`. If either clears ego,
   the oncoming vehicle can steer around ego and is not a genuine head-on threat.
   Skipped when target speed ≤ 1 m/s.
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

Filtered vehicles are tracked in `evasion_filtered_ids` (debug only) and
drawn in cyan in the debug window. They do **not** contribute to TTB or
AEB state.

### Oncoming evasion filter (head-on vehicle suppression)

Mirrors the ego evasion filter, but from the oncoming vehicle's perspective.
After `ego_arc` detects a head-on hit, two curvature-offset arcs are built
for the **target** and tested against `ego_arc` only (not the ego evasion arcs):

```python
delta_kappa_t = min(_EVASION_G_THRESHOLD_ONCOMING / (abs_v_speed ** 2),
                    _EVASION_FILTER_MAX_DELTA_KAPPA)

lateral_offset = abs(dx*ego_fwd_z - dz*ego_fwd_x)
if lateral_offset >= _OPPOSITE_LANE_OFFSET:
    delta_kappa_t = min(
        delta_kappa_t * _OPPOSITE_LANE_KAPPA_SCALE,
        _EVASION_FILTER_MAX_DELTA_KAPPA * _OPPOSITE_LANE_KAPPA_SCALE,
    )

tgt_evasion_left  = build_arc(..., target_curvature + delta_kappa_t, ...)
tgt_evasion_right = build_arc(..., target_curvature - delta_kappa_t, ...)
```

If either arc clears `ego_arc` the oncoming vehicle has room to steer around
ego within 0.1 g — it is not a genuine collision course. The vehicle is
skipped and tracked in `oncoming_evasion_filtered_ids`.

#### Lateral-offset kappa scaling

When the oncoming vehicle's center is ≥ `_OPPOSITE_LANE_OFFSET (2.0 m)` from
ego's forward axis (cross product `|dx*ego_fwd_z - dz*ego_fwd_x|`), it is
clearly in its own lane. In this case `delta_kappa_t` is multiplied by
`_OPPOSITE_LANE_KAPPA_SCALE (2.5)` before building the evasion arcs — a
vehicle already displaced laterally needs much less curvature change to miss
ego, so the wider arcs make the filter more likely to correctly suppress it.
The scaled value is still capped at `_EVASION_FILTER_MAX_DELTA_KAPPA × scale`
to prevent degenerate arcs at low speed.  `abs()` is used on the lateral
offset so the filter works on both left- and right-hand traffic roads.

**Conditions:**
- Only runs for `head_on` targets (`fwd_dot < -0.7`). Mutually exclusive with
  the ego evasion filter (`if not head_on` vs `elif head_on`).
- Bypassed when target speed ≤ 1 m/s to avoid Δκ blow-up at near-zero speed.
- Target arcs inherit the head-on `decel=_FULL_BRAKE_DECEL` already set on
  `base_target_arc`, so the braking model remains consistent.
- Checked against `ego_arc` directly — **not** against `ego_evasion_left/right`
  and **not** against cross arcs. The question is purely whether the oncoming
  vehicle's path can miss the ego centre path with a gentle steer.

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
| TMP speed | `delta(raw_pos) / dt`, signed via forward dot |
| AI speed | `use buffer value as-is (may be signed); do not flip from displacement` |
| Position smooth | `alpha * raw + (1-alpha) * pred`; `alpha = _compute_position_alpha(speed, noise_est)` |
| Position alpha formula | See `_compute_position_alpha(speed_ms, noise_est)` — hyperbolic base (1.0 at rest → 0.15 at 90 km/h) times noise modifier |
| Lag detection | `raw_disp < 10 % of (prev_speed × dt)` AND `prev_speed > 2 m/s` → decay speed: `prev_speed × (1 − frac²)`, release after 0.3 s |
| Pos mismatch | `dot(raw_disp, prev_fwd) < -0.05 m` AND `is_tmp` AND `frames < 10` → hold smooth pos + speed, allow yaw |
| Crash evidence | decay `× 0.7`/frame; +0.35 raw yaw rate > 30 deg/s; +0.30 backward disp; +0.20 micro-osc; confirm after 0.25 s above 0.75 |
| Yaw EMA (wrap-safe) | `smooth += 0.20 * ((raw - smooth + π) % 2π - π)` |
| TMP trailer pivot fix | `pos.x += (len/2)*sin(yaw); pos.z += (len/2)*cos(yaw)` |
| Evasion filter Δκ | `min(0.1*9.81 / v², 0.008)` with additional centreline snap when evasion path would cross lane centre |
| Oncoming evasion filter Δκ | `min(0.13*9.81 / v², 0.008)` with `_OPPOSITE_LANE_OFFSET` / `_OPPOSITE_LANE_KAPPA_SCALE` scaling |
| Head-on lateral gap | `_LATERAL_LANE_SEPARATION = 3.9 m` (cross product of hit-point separation vs `a.fwd`) |
| Near-head-on threshold | `_NEAR_HEAD_ON_DOT = -0.5` — activates lateral gap; looser than `head_on` (-0.7) to catch shared-turn approach geometry |
| Opposite-lane offset | `_OPPOSITE_LANE_OFFSET = 2.0 m` — lateral distance from ego axis at which oncoming kappa scale activates |
| Opposite-lane kappa scale | `_OPPOSITE_LANE_KAPPA_SCALE = 2.0` — multiplier on `delta_kappa_t` for clearly displaced oncoming vehicles |
| Turning diverge curvature threshold | `_TURNING_DIVERGE_CURVATURE = 0.007 /m` (≈ 143 m radius) |

---

## 13. Critical Rules — Do Not Break

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
- **Position alpha follows a hyperbolic (1/x) curve: 1.0 at rest → 0.15 at 90 km/h, noise-modulated.** Never use fixed constants `_RAW_POSITION_ALPHA` or `_RAW_POSITION_ALPHA_TMP` — they have been removed. Call `_compute_position_alpha(prev.speed, self._noise_est)`.
- **TMP lag freeze holds ALL dynamic state (speed, accel, smooth position, noise_est).** Do not advance the smooth position via prediction during a freeze — that would cause a snap correction when the lag ends.
- **Lag freeze speed decays quadratically: `prev_speed × (1 − frac²)`.** Never hold speed constant during lag — it keeps AEB informed while smoothly approaching 0.
- **`lag_confirmed` is set by `traffic.py`, not `thread.py`.** thread.py does not need to check it. A confirmed-stopped vehicle has speed = 0 and is detected as a stationary obstacle by the existing arc collision logic.
- **Position mismatch (TMP only) runs before lag detection.** It is mutually exclusive with lag: a backward jump is not near-stationary. The `not _skip_position_update` guard on the lag block enforces this.
- **Position mismatch is capped at `_POS_MISMATCH_MAX_FRAMES (10)`.** When the cap is reached, the next frame always passes raw position through. Without this cap, a genuine crash or prolonged backward event would be silently swallowed.
- **Crash detection position EMA is never suppressed.** Only `speed` (decelerating at 10 m/s²) and `acceleration` (forced 0) are overridden on `crash_confirmed`. Position stays accurate.
- **`crash_confirmed` and `lag_confirmed` are both handled in `traffic.py`.** Neither requires special-casing in `thread.py` — both produce `speed = 0` which AEB detects as a stationary obstacle naturally.
- **Vehicle acceleration is never passed raw to AEB arc logic.** Always use `_accel_to_arc_params(accel, override_decel)`. Negative accel (braking) → `decel = min(|accel|, 6.0)` so arc_length uses the braking distance formula v²/(2d). Crash-induced backward jump artifacts produce large negative spikes that the 6 m/s² cap suppresses. Accelerating vehicles pass `accel = min(accel, 4.0)`. Head-on override (`_FULL_BRAKE_DECEL`) takes priority over both.
- **AI (singleplayer) speed is used as-is from the buffer.** Do not derive/flip sign from displacement or turning vehicles can be misclassified as reversing.

---

*Source: `traffic.py`, `thread.py`, `ETS2radar.py` (old code), `classes.py` (old code) — LD-Tech / MonoCruise — March 2026*