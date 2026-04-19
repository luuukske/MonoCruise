# `accel_to_pedals.py` — Architecture Rework Plan

## Goal

Replace the current dual-controller (separate gas PID + brake FF+trim PI) with a
unified effort controller that eliminates the gas/brake state machine. The result
should have smooth, continuous transitions between gas and brake with no dead zones,
no oscillation, and no integral state blowup across transitions.

`pedal_capacity.py` is **not touched**.

---

## Problems with the current system

- Gas and brake are controlled by two completely separate controllers with their own
  integrals. Switching between them requires freezing/decaying state on both sides,
  which causes suppression artifacts (throttle delayed after brake release, etc.).
- The gas integral is frozen during braking and resumes stale — this suppresses
  throttle for seconds after brake release.
- The brake trim integral decays when not braking, so it has to re-learn every
  braking event.
- Road load feedforward (gravity + rolling) is computed from telemetry that isn't
  perfectly accurate. Neither controller has a clean way to absorb the persistent
  bias from this inaccuracy.
- The brake multiplier EMA (slow curve scaling) was trying to solve the same problem
  as `pedal_capacity.py` — it is now redundant and should be removed.

---

## New architecture overview

A single **effort** value in normalized pedal units `[-1.0, 1.0]` is computed each
frame. Positive effort → gas pedal. Negative effort → brake pedal. No mode switch,
no threshold, no state machine.

```
effort = feedforward(wanted_smooth, effective_road_load) + fast_pid(error)

gas   = clamp( effort, 0.0, 1.0)
brake = clamp(-effort, 0.0, 1.0)
```

Before that combined command is converted to pedals, the mapper now runs a final
adaptive EMA in m/s² space. Small deltas stay more damped, while larger deltas
raise alpha with a cubic response so strong command moves stay reactive.

The road load is corrected by a slow integral that lives in m/s² space alongside it,
before any pedal conversion.

---

## Controller pipeline (step by step)

### 1. Smooth inputs

No change from current implementation.

```python
wanted_smooth = EMA(wanted_accel_ms2, tau=0.05 s)
raw_smooth    = EMA(raw_accel_ms2,    tau=0.10 s)
```

`raw_smooth` is **frozen** (EMA update skipped) during the gearshift block period
— see Gearshift section.

---

### 2. Road load

No change from current implementation.

```python
road_load_ms2, grade_unc_rad, grade_rad = _road_load_accel_ms2(...)
# road_load_ms2 = slope_accel + rolling_accel (positive = resisting forward motion)
```

---

### 3. Slow integral (road load bias correction)

The slow integral corrects for the portion of road load that the telemetry gets wrong.
It lives in **m/s² space** and is added directly to `road_load_ms2` before the FF
is computed. This means it shifts the zero-effort crossover point symmetrically —
if the truck is on a sustained slope the telemetry is underreporting, the slow integral
learns that offset and pushes it into both the gas and brake FF equally.

```python
error_ms2 = wanted_smooth - raw_smooth

slow_integral += KI_SLOW * error_ms2 * dt
slow_integral  = clamp(slow_integral, -SLOW_I_CLAMP_MS2, SLOW_I_CLAMP_MS2)

effective_road_load = road_load_ms2 + slow_integral
```

**No decay on the slow integral.** It holds its learned value. It only moves when
there is a sustained tracking error.

Suggested starting constants:
```python
KI_SLOW          = 0.03   # m/s² per (m/s²·s) — tune up if too slow to learn
SLOW_I_CLAMP_MS2 = 2.0    # ±2 m/s² — covers realistic road load errors
```

---

### 4. Feedforward

Uses `effective_road_load` (road load + slow integral). Determines the expected pedal
to achieve the wanted acceleration purely from known physics + learned capacity.

```python
combined = wanted_smooth + effective_road_load

if combined >= 0.0:
    # Gas side — linear
    ff = combined / max(max_accel_ms2, 0.1)
else:
    # Brake side — inverse of fitted curve (existing _brake_pedal_from_decel)
    decel_needed = -combined   # positive
    ff = -_brake_pedal_from_decel(decel_needed)
```

`_brake_pedal_from_decel` is **unchanged** from the current implementation. The brake
curve constants `_BRAKE_CURVE_RATE` and `_BRAKE_CURVE_POWER` are unchanged.

The conditional here is pure math, not a persistent mode — it has no state.

---

### 5. Fast PID

Handles transient tracking error on top of the feedforward. Small integral clamp
means it cannot wind up — it is a trim, not the primary controller.

Derivative is on **measurement only** (not on the error setpoint), to avoid spikes
when `wanted_smooth` steps.

```python
error_ms2 = wanted_smooth - raw_smooth

# Proportional
fast_p = KP_FAST * error_ms2

# Integral (small clamp — trim only)
fast_integral += KI_FAST * error_ms2 * dt
fast_integral  = clamp(fast_integral, -FAST_I_CLAMP, FAST_I_CLAMP)

# Derivative on measurement
deriv_raw        = (raw_smooth - prev_raw_smooth) / dt
deriv_smooth     = EMA(deriv_smooth, deriv_raw, tau=0.12 s)
fast_d           = -KD_FAST * deriv_smooth

# Convert to pedal units
capacity = max_accel_ms2 if error_ms2 >= 0 else max_brake_ms2
fast_out = (fast_p + fast_integral + fast_d) / max(capacity, 0.1)
fast_out = clamp(fast_out, -FAST_OUT_CLAMP, FAST_OUT_CLAMP)
```

The fast integral and derivative are also **frozen** during the gearshift block
(same freeze as `raw_smooth`).

Suggested starting constants:
```python
KP_FAST        = 0.25
KI_FAST        = 0.25
KD_FAST        = 0.15
FAST_I_CLAMP   = 0.10   # pedal units — intentionally small
FAST_OUT_CLAMP = 0.30   # total fast PID contribution cap
```

---

### 6. Combine and map to pedals

```python
effort = ff + fast_out   # normalized pedal units [-1, 1]

gas_cmd   = clamp( effort, 0.0, 1.0)
brake_cmd = clamp(-effort, 0.0, 1.0)
```

No state machine. No threshold. The transition through zero is continuous.

Apply rate limit to gas only (brake response must be immediate):
```python
if prev_gas_cmd is not None:
    gas_cmd = clamp(gas_cmd, prev_gas_cmd - GAS_RATE_LIMIT*dt,
                              prev_gas_cmd + GAS_RATE_LIMIT*dt)
prev_gas_cmd = gas_cmd
```

Zero gas in neutral (gear == 0), same as current.

---

## Gearshift handling (reworked)

The current implementation only reacts to the leading edge of the clutch. The new
implementation tracks the full clutch duration and applies a proper freeze + ramp.

### State needed

```python
_clutch_active:       bool  = False
_clutch_release_mono: float = -inf   # monotonic time clutch was released
_frozen_raw_smooth:   float = 0.0    # raw_smooth value at moment of freeze
```

### Logic (called at the top of each step, before EMA updates)

```python
CLUTCH_THRESHOLD   = 0.05
BLOCK_DURATION_S   = 0.5
RAMP_DURATION_S    = 1.0

clutch_pressed = clutch_applied > CLUTCH_THRESHOLD

if clutch_pressed and not _clutch_active:
    # Leading edge — start freeze
    _clutch_active     = True
    _frozen_raw_smooth = raw_smooth  # snapshot current value

elif not clutch_pressed and _clutch_active:
    # Trailing edge — start block countdown
    _clutch_active       = False
    _clutch_release_mono = now

# Determine freeze / ramp factor
time_since_release = now - _clutch_release_mono

if clutch_pressed:
    gearshift_factor = 0.0   # fully frozen
elif time_since_release < BLOCK_DURATION_S:
    gearshift_factor = 0.0   # still in hard block
elif time_since_release < BLOCK_DURATION_S + RAMP_DURATION_S:
    t = (time_since_release - BLOCK_DURATION_S) / RAMP_DURATION_S
    gearshift_factor = t     # linear ramp 0→1
else:
    gearshift_factor = 1.0   # fully live
```

### Applying the factor

When `gearshift_factor < 1.0`:

- **Skip the `raw_smooth` EMA update** entirely. Use `_frozen_raw_smooth` as the
  effective `raw_smooth` for all error computations.
- **Freeze `fast_integral`** — do not accumulate during the block.
- **Freeze `slow_integral`** — do not accumulate during the block.
- During the ramp: interpolate between `_frozen_raw_smooth` and the live EMA value
  using `gearshift_factor` as the blend weight.

This prevents the torque spikes during gear changes from contaminating any integrator
or derivative state.

---

## What to remove

The following are deleted entirely — no replacement needed:

| Removed | Reason |
|---|---|
| `_brake_multiplier`, `_BRAKE_MULTIPLIER_*`, `_update_brake_multiplier()` | Replaced by `pedal_capacity.py` |
| `_BRAKE_LEAD_TAU_S`, `_BRAKE_LEAD_DERIV_SMOOTH_TAU_S` | Dropped (causes more problems than it solves) |
| `_wanted_deriv_smooth`, `_prev_wanted_smooth` | Only used for brake lead |
| `_BRAKE_ACTIVATION_MS2` | No longer a threshold — effort sign drives brake |
| `_BRAKE_TRIM_DECAY_TAU_S`, `_BRAKE_KP_TRIM`, `_BRAKE_KI_TRIM`, `_BRAKE_KI_TRIM_CLAMP` | Separate brake trim PI removed |
| `_brake_step()` | Replaced by unified controller |
| `_gas_step()` | Replaced by unified fast PID |
| `_STATE_GAS`, `_STATE_BRAKE`, `_STATE_NAMES` | No state machine |
| `_GAS_KP`, `_GAS_KI`, `_GAS_KD`, `_GAS_KI_CLAMP` | Replaced by `KP/KI/KD_FAST` |
| `_GAS_INTEGRAL_BRAKE_DECAY_TAU_S` | No decay needed in unified model |
| `_gas_integral` (as gas-only state) | Replaced by `_fast_integral` |
| `_gearshift_start_mono`, `_integral_block_end_mono` | Replaced by new gearshift state |
| `braking` boolean / if-else block in `step()` | No mode switch |

---

## What to keep (unchanged)

| Kept | Notes |
|---|---|
| `_brake_pedal_from_decel()` | Brake curve inverse — unchanged |
| `_BRAKE_CURVE_RATE`, `_BRAKE_CURVE_POWER` | Fitted constants — do not change |
| `_road_load_accel_ms2()`, `_road_grade_from_norm()` | Road physics — unchanged |
| `_ema_step()`, `_ema_alpha()`, `_motion_sign()` | Utility helpers |
| `baseline_accel_ms2()`, `baseline_brake_ms2()` | Capacity baseline helpers |
| `_WANTED_SMOOTHING_TAU_S`, `_RAW_SMOOTHING_TAU_S` | Input smoothing unchanged |
| `_GAS_RATE_LIMIT_PER_S` | Rate limit kept on gas only |
| `_GAS_DERIVATIVE_TAU_S` | Reused for fast PID derivative smoothing |
| `PedalTargets` dataclass | Keep all fields; `pedal_state` can be removed if desired |
| Debug logging | Update field names to match new internals |

---

## New state variables

```python
# Unified fast PID
_fast_integral:      float = 0.0
_fast_deriv_smooth:  float = 0.0
_prev_raw_smooth:    float = 0.0
_prev_gas_cmd:       float | None = None

# Slow road load correction integral
_slow_integral:      float = 0.0

# Gearshift freeze
_clutch_active:        bool  = False
_clutch_release_mono:  float = -math.inf
_frozen_raw_smooth:    float = 0.0
```

---

## Implementation order

1. Add new state variables to `__init__` and `reset_smoothing`.
2. Implement the new gearshift logic as a standalone method `_gearshift_factor(now, clutch) -> (factor, frozen_raw)`.
3. Implement the fast PID as `_fast_pid_step(dt, error_ms2, raw_smooth, factor) -> float`.
4. Rewrite `step()`:
   - Compute road load.
   - Compute slow integral (freeze during gearshift).
   - Compute FF using `effective_road_load`.
   - Compute fast PID (freeze during gearshift).
   - Combine → effort → gas/brake.
5. Remove all deleted methods and constants.
6. Update debug log fields to match new internals.
7. Tune constants on a test drive — start with the suggested values above.
