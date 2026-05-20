# AGENTS.md — MonoCruise Radar (shared fundamentals)

> Authoritative reference for **coordinate system, rotation, world→ego
> transforms, the shared-memory traffic buffer, Vehicle state/smoothing,
> and ArcPath geometry**. Read this before touching any code that
> consumes ETS2 telemetry, traffic, or predicted paths.
>
> AEB- and ACC-specific logic live in `core/aeb/AGENTS.md` and
> `core/acc/AGENTS.md` respectively — both of them build on the concepts
> defined here.

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

# AEB / ACC / arc geometry — NO +0.5 offset
ego_yaw_rad = yaw_norm * 2.0 * math.pi
```

> **WARNING — do not mix these up.**
> Adding `+0.5` in the arc-geometry context rotates the ego forward
> vector 180°, reversing the ego arc. The forward vector
> `fwd = (-sin, -cos)` already points North at `yaw=0` without the offset.
> If only the ego arc points backward, the bug is in this conversion —
> not in `traffic.py`.

`RadarThread` publishes both forms (`ego_yaw_norm` and `ego_yaw_rad`)
under the data lock; consumers should prefer `ego_yaw_rad`.

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

Defined in `core/radar/reader.py`; consumed by `RadarThread` only — AEB
and ACC receive ready-made ``Vehicle`` instances from radar data.

```python
_VEHICLE_FORMAT        = "ffffffffffffhhbb"   # 16 fields
_TRAILER_FORMAT        = "ffffffffff"         # 10 fields per trailer slot
_VEHICLE_OBJECT_FORMAT = _VEHICLE_FORMAT + _TRAILER_FORMAT * 3
_TOTAL_FORMAT          = "=" + _VEHICLE_OBJECT_FORMAT * 40
_BUF_SIZE              = 6960
_VEH_STRIDE            = 46  # fields per vehicle slot (16 + 3*10)
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
| 10 | speed | float | AI = use as-is from buffer (may be signed in singleplayer). TMP = LS fit of longitudinal motion over up to `_TMP_SPEED_HISTORY_LEN` position-history samples, else single-interval Δ/dt; then speed-dependent EMA of raw speed (see §7). |
| 11 | acceleration | float | m/s² — AI = buffer as-is. TMP buffer value is **ignored**; `Vehicle.acceleration` is EMA of the time derivative of filtered TMP speed (see §7). Arcs use `accel_for_arc()` (= `acceleration`). |
| 12 | trailer_count | short | 0–3 |
| 13 | id | short | Per-frame continuity key |
| 14 | is_tmp | byte | `1` = TMP multiplayer (ETS2LA); `0` = AI. Consumer threads may apply TMP-specific filters (e.g. AEB's rel-speed split — see `core/aeb/AGENTS.md`). |
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

### Fields — positions raw; speed/accel filtered (AI + TMP)

| Field | Source | Use for |
|-------|--------|---------|
| `position.x/z` | **Unfiltered** shared-memory world coordinates | Arc start position, rendering, collision geometry |
| `_smooth_yaw` | Wrap-safe EMA of `rotation.euler()[1]` in radians (`_RAW_YAW_ALPHA = 0.5`, AI and TMP) | **Arc curvature. Never use `rotation.euler()` directly for arcs.** |
| `speed` | Accel-corrected smoothed speed (`speed_corr`) — see filter chain below. AI + TMP | AEB arc direction, TTB |
| `acc_speed` | Extra-smoothed speed (`speed_acc`) — plain EMA of `speed`. AI + TMP | ACC following-distance only |
| `acceleration` | Nonlinear EMA of `d(speed_ema)/dt` — see filter chain below. AI + TMP (buffer field 11 unused) | Arc decel/accel via `_accel_to_arc_params()` |
| `angular_velocity` | Degrees/s from rotation delta/dt | Arc curvature via `κ = ω_rad/speed` |
| `_position_history` | `(t, x, z)` tuples appended each full update (AI + TMP); capped at `_POSITION_HISTORY_LEN = 25` | TMP raw-speed LS fit (uses last `_TMP_SPEED_HISTORY_LEN = 10`), `curvature_from_history`, ACC trail arcs |
| `_speed_ema_history` | `(t, speed_ema)` tuples appended each full update (AI + TMP); capped at `_SPEED_EMA_HISTORY_LEN` | LS-slope fits: `accel` over `_ACCEL_FIT_WINDOW_S`, `accel_ultra` over `_ACCEL_ULTRA_FIT_WINDOW_S` |

### Speed & acceleration — filter chain (AI + TMP)

World `position.x/z` are **not** low-pass filtered. `update_from_last()` runs the
same 4-signal chain for AI and TMP (`_smooth_vehicle_kinematics()` in
`traffic.py`); only the raw-speed source differs:

- **AI** raw speed = buffer field 10 as-is.
- **TMP** raw speed = LS fit of longitudinal motion over the last
  `_TMP_SPEED_HISTORY_LEN` `(t, x, z)` position-history samples — fit `s ≈ v·τ`
  with `s = dot(p − p₀, fwd(smooth_yaw))`, `τ = t − t₀`, `v = Σ(τ s)/Σ(τ²)`.
  Chord below 0.025 m → `raw_speed = 0`; one sample → single-interval `Δraw/dt`.
  Buffer fields 10/11 are never used for TMP physics.

```python
# 1. speed_ema  — plain EMA of raw speed (no lag compensation)
alpha      = _tmp_speed_ema_alpha(|avg(prev_speed_ema, raw_speed)|)   # 1.0 rest → 0.25 @ 90 km/h
speed_ema  = alpha * raw_speed + (1 - alpha) * prev_speed_ema
# 2. accel  — LS slope of the speed_ema history over _ACCEL_FIT_WINDOW_S, light EMA
accel_raw  = least_squares_slope( (t, speed_ema) samples within _ACCEL_FIT_WINDOW_S )
accel      = prev_accel + _ACCEL_EMA_ALPHA * (accel_raw - prev_accel)
# 3. speed_corr  — lag-compensated; τ is the step-1 EMA settling time
speed_corr = speed_ema + clamp(accel * dt*(1-alpha)/alpha, ±_SPEED_CORR_CLAMP_MS)
# 4. speed_acc  — ACC ultra-smooth: EMA of speed_corr (speed-dependent α), re-corrected
alpha_a       = _tmp_acc_speed_ema_alpha(|speed_corr|)   # 0.35 rest → 0.08 @ 90 km/h
speed_acc_ema = alpha_a * speed_corr + (1 - alpha_a) * prev_speed_acc_ema
accel_ultra   = LS slope of speed_ema history over _ACCEL_ULTRA_FIT_WINDOW_S (1 s)
speed_acc     = speed_acc_ema + clamp(accel_ultra * dt*(1-alpha_a)/alpha_a, ±_SPEED_CORR_CLAMP_MS)
```

Step 4's `accel_ultra·τ` term re-corrects the lag the ultra-smooth EMA would
otherwise re-introduce, so ACC tracks accel/decel without trailing behind.
`accel_ultra` is a **1 s** LS slope of `speed_ema` — decoupled from
`speed_acc_ema` (deriving the correction from its own output is a
self-referential lead that rings) and long-windowed, so it carries only the
trend, not the wiggle. `α` is heavier (more smoothing) at high speed, lighter
at low speed.

Exposed as `self.speed = speed_corr` (AEB), `self.acc_speed = speed_acc` (ACC),
`self.acceleration = accel` (shared); `speed_ema` is the internal `_speed_ema`
intermediate. On the first full frame after spawn every signal initialises to
`raw_speed` with `accel = 0`. `α` decreases with |speed| (more smoothing when fast).

**Arc / collision** — `Vehicle.accel_for_arc()` is `return self.acceleration`.
TMP vehicles initialise `acceleration = 0` until the first `update_from_last`;
`get_arc()` and all arc callers use this field for `_accel_to_arc_params`.

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

`lag_confirmed` is a public flag on `Vehicle`. Consumer threads do not need
to read it: once released, the vehicle's speed = 0 and the existing AEB arc
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

If the ego arc points backward, the bug is in the `rotationX → yaw_rad` conversion, not here.

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

When `min_lateral_gap > 0`, a candidate hit is suppressed if the perpendicular distance between the two arc centerlines (measured along `a`'s instantaneous heading at the hit point) is ≥ this value. This prevents false positives when ego and an oncoming vehicle both enter a curve — their arcs overlap in the forward dimension, but the vehicles remain in their own lanes laterally. AEB owns the activation policy; see `core/aeb/AGENTS.md`.

```python
# Lateral separation via cross product (2D):
lat = abs((bz - az) * fwd_x_a - (bx - ax) * fwd_z_a)
if lat >= min_lateral_gap:
    suppress hit
```

- Applied in both `_ray_ray_collision` and `_sampled_collision`
- In the sampled path, the lateral check runs at each coarse sample **before** entering bisection; during bisection, a failing lateral check advances `lo` rather than breaking, so the refiner keeps searching for a sample where lanes genuinely cross

---

## 9. Forward Vector & Position Prediction

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

## 10. Yaw Alignment Scoring

```python
d        = vehicle_yaw_deg - ego_yaw_deg
yaw_diff = min(abs(d), abs(d + 360), abs(d - 360))
# ~0°   → same direction (co-directional, lane candidate)
# ~180° → oncoming traffic
# 45–135° → cross-traffic
```

---

## 11. Position-Based Curvature

Traffic vehicles derive curvature from a circumscribed circle fit over `_position_history` (`Vehicle.curvature_from_history()`), falling back to `angular_velocity / speed` when fewer than 3 samples are available.

Ego uses the same math in `core/radar/ego_path.py::ego_curvature_from_history`, fed by `RadarThread._ego_position_history`. `RadarThread` publishes the result as `RadarData.ego_curvature` (may be `None`).

**Consumer policy — the history fit is for ACC only.**

- **ACC** uses `RadarData.ego_curvature` for in-path scoring. The
  smoothed, geometry-based value matches the smoothing applied to target
  vehicles, so scoring stays consistent across long horizons.
- **AEB** does **not** read `RadarData.ego_curvature`. It computes the
  yaw-rate proxy `steer * speed * 12.0 / speed` inline every frame. The
  ego arc must react instantly to driver input — a history-based fit
  lags the truck through transients and produces corridor-misalignment
  false positives / negatives during and after corners.

Sign convention: positive = left turn (κ > 0), matching `ArcPath`.

---

## 12. RadarThread Interface

Registry name: `radar_thread`. Runs at 30 Hz.

```python
rt = registry.get_thread("radar_thread")
with rt.data._lock:
    vehicles      = rt.data.vehicles          # list[Vehicle] — shared refs, do not mutate
    tmp_session   = rt.data.tmp_session       # True if any vehicle has is_tmp
    ego_x         = rt.data.ego_x
    ego_y         = rt.data.ego_y
    ego_z         = rt.data.ego_z
    ego_yaw_rad   = rt.data.ego_yaw_rad
    ego_speed     = rt.data.ego_speed         # m/s
    ego_pitch_rad = rt.data.ego_pitch_rad     # inverted from telemetry rotationY
    ego_steer     = rt.data.ego_steer
    ego_has_trailer = rt.data.ego_has_trailer
    ego_curvature = rt.data.ego_curvature     # None → fall back to yaw-rate proxy
    paused        = rt.data.paused
    t_mono        = rt.data.t_mono            # snapshot time (monotonic)
```

When `paused` is True the vehicle list and `t_mono` are held at their last
values; consumer threads should treat an unchanged `t_mono` as "no new frame".

---

## 13. Quick Reference — Formulas (shared)

| Formula | Code / Notes |
|---------|-------------|
| Ego yaw → rad (radar render) | `(yaw + 0.5) * 2 * pi` |
| Ego yaw → rad (arcs) | `yaw_norm * 2 * pi` (no +0.5) |
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
| TMP raw speed | LS on longitudinal `(t,x,z)` history (max `_TMP_SPEED_HISTORY_LEN` full frames): `v = Σ(τ s)/Σ(τ²)`; else `Δraw/dt`, signed via forward dot |
| Speed / accel filter (AI + TMP) | 4-signal chain in `_smooth_vehicle_kinematics()`: `speed_ema` (EMA of raw) → `accel` (LS slope of `speed_ema` history over `_ACCEL_FIT_WINDOW_S`, light EMA) → `speed_corr = speed_ema + accel·τ` (`self.speed`) → `speed_acc` (speed-dependent-α EMA of `speed_corr`, re-corrected by `accel_ultra·τ` where `accel_ultra` is a 1 s LS slope of `speed_ema`, `self.acc_speed`) |
| AI vs TMP raw speed | AI = buffer field 10; TMP = position-history LS fit. Filter chain identical after that |
| Positions | No EMA — always raw world coordinates |
| Lag detection | `raw_disp < 10 % of (prev_speed × dt)` AND `prev_speed > 2 m/s` → decay speed: `prev_speed × (1 − frac²)`, release after 0.3 s |
| Pos mismatch | `dot(raw_disp, prev_fwd) < -0.05 m` AND `is_tmp` AND `frames < 10` → hold smooth pos + speed, allow yaw |
| Crash detection | rotation jerk (pitch/yaw/roll deg/s²) AND (\|ΔY\| > 0.08 m OR XZ dir reversal cos < -0.3); both must fire → confirm after 0.10 s; disables pos-mismatch filter and lag freeze; speed/accel stay raw |
| Yaw EMA (wrap-safe) | `smooth += 0.5 * ((raw - smooth + π) % 2π - π)` |
| TMP trailer pivot fix | `pos.x += (len/2)*sin(yaw); pos.z += (len/2)*cos(yaw)` |

---

## 14. Critical Rules — Do Not Break

- **No long comments.** Do not write long comments to explain code. Edit AGENTS.md if you need to explain something long, otherwise use small one-line comments.
- **Quaternion x/y swap is intentional.** Never remove it.
- **`rotationX` in telemetry is yaw.** The name is misleading.
- **Radar render uses `+0.5` offset; arc-geometry threads do not.** Do not mix them.
- **`-dx` and `-yaw_rad` in ego-space transform are both required.**
- **AI vehicles use asymmetric corner offsets (0.82 / 0.18), not 0.5 / 0.5.**
- **Always use `_smooth_yaw` for arc construction**, never `rotation.euler()` directly.
- **Y axis is never used in 2D math**, only for elevation filtering.
- **Arc forward vector formula is `(-sin, -cos)`.** Do not flip signs or swap to `(sin, cos)`.
- **Speed/accel filtering runs for AI and TMP** via `_smooth_vehicle_kinematics()` — the 4-signal chain `speed_ema → accel → speed_corr → speed_acc`. `self.speed` is the accel-corrected `speed_corr`; `self.acc_speed` is the lag-corrected ultra-smooth `speed_acc` (ACC only); `self.acceleration` is the LS-slope `accel`. World positions are not low-pass filtered.
- **`acceleration` is kinematic-only** — buffer field 11 is ignored for AI and TMP; `accel_for_arc()` reads `self.acceleration` (least-squares slope of the `speed_ema` history, light-EMA smoothed).
- **`acc_speed` is ACC-only.** AEB and arc geometry use `self.speed`; never swap them.
- **TMP lag freeze holds position, filtered speed decay, and internal EMA state.** Do not advance position during a freeze — that would snap when updates resume.
- **Lag freeze speed decays quadratically: `prev_speed × (1 − frac²)`.** Never hold speed constant during lag — it keeps downstream threads informed while smoothly approaching 0.
- **`lag_confirmed` is set by `traffic.py`, not by consumer threads.** A confirmed-stopped vehicle has speed = 0 and is detected as a stationary obstacle by the existing arc collision logic.
- **Position mismatch (TMP only) runs before lag detection.** It is mutually exclusive with lag: a backward jump is not near-stationary. The `not _skip_position_update` guard on the lag block enforces this.
- **Position mismatch is capped at `_POS_MISMATCH_MAX_FRAMES (10)`.** When the cap is reached, the next frame always passes raw position through. Without this cap, a genuine crash or prolonged backward event would be silently swallowed.
- **Crash detection does not override speed or acceleration.** It disables the pos-mismatch filter and lag freeze so raw position data passes through unfiltered.
- **Crash detection runs before pos-mismatch and lag early-returns.** Both signals (rotation jerk and sporadic position) must fire simultaneously; `_crash_since` resets whenever either is absent.
- **Vehicle longitudinal accel for arcs** — `Vehicle.accel_for_arc()` → `self.acceleration` (TMP = filtered kinematic; AI = buffer). Then `_accel_to_arc_params(accel, override_decel)`.
- **AI (singleplayer) speed is used as-is from the buffer.** Do not derive/flip sign from displacement or turning vehicles can be misclassified as reversing.
- **`Vehicle.curvature_from_history()` is the curvature source.** Returns circumscribed-circle curvature from `_position_history`; `None` when < 3 samples (caller falls back to yaw-rate); `0.0` when near-stationary. Both TMP and AI vehicles populate `_position_history` in `update_from_last()`.
- **Consumer threads must not open the traffic shared-memory buffer.** Read vehicles from `registry.get_thread("radar_thread").data.vehicles` under the data lock. Mutating Vehicle instances from consumer threads corrupts the per-id smoothing state carried forward by the reader.

---

*Source: `core/radar/traffic.py`, `core/radar/reader.py`, `core/radar/thread.py`, `core/radar/ego_path.py` — LD-Tech / MonoCruise.*
