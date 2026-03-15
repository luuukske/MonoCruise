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
| 10 | speed | float | AI = unsigned from buffer; TMP = sign derived via dot product |
| 11 | acceleration | float | m/s² — clamped to [-6, +4] in AEB |
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
| `position.x/z` | EMA alpha=0.20 applied in `update_from_last()` | All rendering, arc start position |
| `_smooth_yaw` | Wrap-safe EMA of `rotation.euler()[1]` in radians | **Arc curvature. Never use `rotation.euler()` directly for arcs.** |
| `speed` | Signed m/s, set in `update_from_last()` | Arc direction, speed display |
| `angular_velocity` | Degrees/s from rotation delta/dt | Arc curvature via `κ = ω_rad/speed` |

### Position smoothing — prediction-corrected EMA

Pure EMA blends `raw` against `prev_smooth`, producing a steady-state lag of
`((1 − α) / α) × dt × speed` (≈ 4.4 m at 80 km/h with α = 0.20).

Instead, `prev_smooth` is replaced by a **kinematic prediction** of where the
vehicle should be this frame:

```python
pred_dist = speed * dt + 0.5 * clamp(acceleration, -6, 4) * dt²
pred_x    = smooth_x_prev + pred_dist * (-sin(smooth_yaw))
pred_z    = smooth_z_prev + pred_dist * (-cos(smooth_yaw))

smooth_x  = 0.20 * raw_x + 0.80 * pred_x
smooth_z  = 0.20 * raw_z + 0.80 * pred_z
```

When motion is predictable (`pred ≈ raw`), the `0.80` term contributes
near-zero error rather than lag. The `0.20` on raw still corrects for
sudden manoeuvres and measurement noise.

`smooth_yaw` (not `rotation.euler()`) is used for the prediction forward vector —
it is already available from `prev._smooth_yaw` at this point in the update.
`speed` is already sign-corrected before this block runs.

### Yaw EMA (unchanged) — wrap-safe

```python
diff       = (raw_yaw - smooth_yaw + math.pi) % (2 * math.pi) - math.pi
smooth_yaw = smooth_yaw + 0.20 * diff
```

### Speed sign detection

The buffer gives **unsigned** speed for AI vehicles. Sign is derived via forward dot product:

```python
fwd_x = -math.sin(yaw_rad)
fwd_z = -math.cos(yaw_rad)
if (disp_x * fwd_x + disp_z * fwd_z) < 0:
    speed = -speed   # reversing
```

TMP vehicles compute speed from raw position delta / dt, same sign logic.

### Sub-frame pass (dt < 0.05 s)

`update_from_last()` carries forward all smoothed state unchanged. Speed sign is preserved from the last full update. **Do not re-derive speed sign on sub-frame passes.**

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
| `decel` | Ego braking arc only. Mutually exclusive with `accel`. |
| `arc_length` | Accounts for decel/accel to stop |
| `is_straight` | True if `|κ| < 1e-6` or `speed < 0.001` |

### Arc center (curved only)

```python
# sign = +1 for left turn (κ > 0), -1 for right
center_x = start_x + sign * radius * fwd_z
center_z = start_z + sign * radius * (-fwd_x)
```

### Collision detection

`arc_arc_collision(a, b, margin, n_samples)` returns `(time_s, hit_x, hit_z)` or `None`.

- Both straight, no decel/accel → closed-form quadratic O(1)
- Otherwise → time-synchronised sampling + 6-step bisection O(n)
- Corridor threshold = `a.half_width + b.half_width + margin`
- AEB narrows vehicle half_width by 0.1 m per side to reduce false positives from measurement noise

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
ego_evasion_left  = build_arc(..., ego_curvature + delta_kappa, ...)
ego_evasion_right = build_arc(..., ego_curvature - delta_kappa, ...)
```

- `_EVASION_G_THRESHOLD = 0.1 × 9.81` — the lateral acceleration a gentle
  steer would produce. `Δκ = a_lat / v²` gives the curvature offset at
  the current speed.
- `_EVASION_FILTER_MAX_DELTA_KAPPA = 0.03` — hard clamp so the filter
  arcs stay meaningful at low speed.

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
delta_kappa_t = min(_EVASION_G_THRESHOLD / (abs_v_speed ** 2),
                    _EVASION_FILTER_MAX_DELTA_KAPPA)
tgt_evasion_left  = build_arc(..., target_curvature + delta_kappa_t, ...)
tgt_evasion_right = build_arc(..., target_curvature - delta_kappa_t, ...)
```

If either arc clears `ego_arc` the oncoming vehicle has room to steer around
ego within 0.1 g — it is not a genuine collision course. The vehicle is
skipped and tracked in `oncoming_evasion_filtered_ids`.

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

| Formula | Code |
|---------|------|
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
| Quaternion euler yaw | `atan2(2*(y*z + w*x), w²-x²-y²+z²)` degrees |
| Arc curvature | `κ = omega_rad_s / abs_speed` |
| Arc center | `cx = x + sign*R*fwd_z; cz = z + sign*R*(-fwd_x)` |
| TMP speed | `delta(raw_pos) / dt`, signed via forward dot |
| AI speed sign | `dot(disp, fwd) < 0 → negate speed` |
| Position smooth | `0.20 * raw + 0.80 * (prev + speed*dt*fwd + 0.5*accel*dt²*fwd)` |
| Yaw EMA (wrap-safe) | `smooth += 0.20 * ((raw - smooth + π) % 2π - π)` |
| TMP trailer pivot fix | `pos.x += (len/2)*sin(yaw); pos.z += (len/2)*cos(yaw)` |
| Evasion filter Δκ | `min(0.1*9.81 / v², 0.03)` |
| Oncoming evasion filter Δκ | same formula applied to target speed |

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

---

*Source: `traffic.py`, `thread.py`, `ETS2radar.py` (old code), `classes.py` (old code) — LD-Tech / MonoCruise — March 2026*