# ACC Gap-Control Architecture

This document describes the acceleration calculation used by the old `MonoCruiseACC`
class once a lead vehicle has already been identified.  Input is a smoothed
`LeadVehicleData` (distance m, speed m/s, accel m/s²) and ego speed (m/s).

---

| Mode | Condition | Action |
|---|---|---|
| Normal follow | default | PD + FF (`_calculate_accel`) |
| Stopping blend | `lead.speed < 0.5 m/s` OR (`lead.accel < -1 m/s²` AND ego closing) | 30 % PD+FF + 70 % kinematic stopping decel |

---

Desired following gap grows linearly with ego speed:

```
desired_gap = MIN_GAP + TIME_GAP_FOLLOW × ego_speed
            = 3.0 m  + 1.5 s × ego_speed_m/s
```

At 30 m/s (108 km/h) → 48 m.  At standstill → 3 m physical minimum.

---

## 3. PD + feed-forward control law

```
gap_error   = lead.distance − desired_gap          (positive → too far, negative → too close)
speed_error = lead.speed − ego_speed               (positive → lead pulling away)

p_term  = K_P × gap_error    (K_P = 0.06)
d_term  = K_D × speed_error  (K_D = 0.46)
ff_term = K_F × lead.accel × acc_amp × acceleration_amp

controller_accel = closeness_amp × (p_term + d_term) + ff_term
```

**D-term** is the relative speed (time-derivative of gap error), not a numeric
difference of past gaps.  This is exact by construction.

**Feed-forward** mirrors the lead vehicle's acceleration, scaled by two factors:

- `acceleration_amp = max(-(time_gap/2)³ + 1, 0.2)` — reduces the FF contribution
  at short gaps (where over-reacting is dangerous) and when the vehicle is far away.
- `acc_amp = 1.0` (config scalar, tuneable).

**Closeness amplitude** is a non-linear gain that amplifies the PD terms when the
actual time-gap is small:

```
closeness_amp = 0.8^(actual_time_gap × 10 − 3) + 0.6
```

When ego is decelerating (`speed_error < 0.1`), the amplitude is softened with a
0.8 power law to avoid a harsh jerk during gentle braking.  When accelerating, it is
clamped to `[0.7, 1.2]`.

**Low-speed creep suppression** (applied after the controller output):

```
slow_speed_increase = ((1 − ((|speed|+0.1)/5.5)^2.22) × (−0.6/(|speed|+0.2)+1)) / 0.715 × 0.10
target_accel −= slow_speed_increase   (floor 0)
```

Subtracts a small positive bias at very low speeds to counteract engine-idle creep
before the brakes can catch up.

---

## 4. Stopping decel (kinematic branch)

Separate kinematic decel computed and blended in at 70 % weight.  Three sub-cases:

| Sub-case | Condition | Formula |
|---|---|---|
| `stopped` | `lead.speed < 0.5 m/s` | `−ego_speed² / (2 × target_stop_pos)` |
| `decel` | `lead.accel < −0.1 m/s²` | Stop behind lead's predicted stop point |
| `closing` | otherwise | `a_lead − rel_speed² / (2 × distance_to_target)` — kinematic speed-match |

`closing` solves 1D kinematics to match ego speed to lead speed at `target_stop_pos`,
accounting for lead acceleration.  If target position < `min_safe_distance`,
`EMERGENCY_DECEL` (−8 m/s²) fires immediately.

`dynamic_gap` during stopping:

```
dynamic_gap = STOPPING_GAP − 1 / (max(0.3, ego_speed) × 10)
            = 4.5 m minus a small speed-dependent correction
```

Result multiplied by 1.1 (slightly over-brakes at start so ego never undershoots
the gap), then clamped to `[MAX_DECEL, 0]`.

---

## 5. Output pipeline

```
target_accel
    ──► np.clip([MAX_DECEL, MAX_ACCEL])          (−6.55 … +1.5 m/s²)
    ──► low-pass filter (α = 0.6)                bypass if value < MAX_DECEL (emergency)
    ──► zero clamp below 5 m/s if 0 ≤ accel < 0.2
    ──► final_accel (m/s²)
```

The low-pass filter formula is:

```
filtered = α × new + (1−α) × prev      (α = 0.6 → current sample dominates slightly)
```

Emergency decel bypasses the filter so hard braking reaches the actuator without lag.

---

## 6. Error handling

Any exception returns −1.0 m/s² (gentle braking) as fail-safe, logs the error, and
dumps `debug_info`.
