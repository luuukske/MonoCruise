# Longitudinal controllers

> Set-speed PID, speed limiter, and ACC cap are children of `LongitudinalController`.
> Orchestration, mode flip, disengage, and arbitration live in `cruise_control_thread`.
> Agent invariants (CC/limiter exclusion, continuous limiter tracker, winner label,
> one mapper): root `AGENTS.md`.

## Stack

Each child `step(LongCtx)` returns `LongOutput(wanted_ms2, active)`. The orchestrator
builds `LongCtx` once per tick, runs children, and publishes `min(...)` of active bids
to telemetry for the single `AccelToPedals` instance in `sending_thread`.

| Child | Role |
|-------|------|
| `CruiseController` | Set-speed PID, target EMA, output EMA, gearshift D-freeze |
| `SpeedLimiter` | Global cap PID; continuous tracker (always `active=True` while enabled) |
| `AdaptiveCruiseController` | Wraps IIDM+CAH cap from `acc_controller.py` |

## CruiseController (`cc.py`)

Buttons, disengage, and global-limit clamp on target are orchestrator-owned. This class
owns PID state only.

- Target clamped every tick via `global_speed_limit_kmh`.
- D-term uses EMA-smoothed speed; D gain zeroed during clutch + post-release block/ramp
  (mirrors mapper gearshift machine).
- Output EMA bypassed when game throttle > 0.1 and CC wants brake.
- Integral freezes and leaks (`_HOLD_INTEGRAL_LEAK_TAU_S`) while `sending_thread.data.hold_active`
  so standstill does not wind the integrator to its clamp.
- Speed error only. Grade feed-forward is `AccelToPedals` road load, not this PID
  (see root `AGENTS.md`).
- Positive bids are capped by the acceleration envelope below, then rate-limited on
  the way up by the selected profile's jerk. Negative bids keep the plain
  `Settings.cc_accel_min_ms2` clamp and are never rate-limited.

## Acceleration envelope (`accel_envelope.py`)

`Settings.cc_accel_max_ms2` used to be the accel ceiling: a flat 1.0 m/s² at every
speed. With `cc_kp = 0.5` any speed error above ~8 km/h saturated it, so MonoCruise
commanded the same acceleration at 15 km/h as at 85 km/h. That is wrong at both ends,
which is what users reported: weak off the line, a shove at highway speed.

The replacement is two terms combined with `min()`. Neither is sufficient alone.

```
shape = a_launch                                  v <= v_knee
      = max(a_floor, a_launch * (v_knee / v)**p)  v > v_knee

a_env = min(shape, HEADROOM_FRAC * a_cap)   when a_cap is usable
      = shape                               otherwise
```

`Settings.cc_accel_max_ms2` survives as a hard rail applied on top, not as the shaper.
Its default moved from 1.0 to 2.5 and the old value is remapped on load.

### The shape term

The comfort law: what a driver wants to feel at a given speed, following a
power-limited vehicle (strong off the line, tapering with speed). It is **absolute**,
because comfort has nothing to do with how much engine is available. An empty tractor
with 3.5 m/s² on tap must not be handed 3.5 m/s².

| Profile | a_launch | v_knee | p | a_floor | rise jerk |
|---|---|---|---|---|---|
| Efficiency | 1.05 | 12 km/h | 0.50 | 0.30 | 0.5 m/s³ |
| Normal | 2.00 | 26.25 km/h | 1.00 | 0.45 | 1.3 m/s³ |
| Sport | 2.50 | 40 km/h | 0.45 | 1.20 | 2.5 m/s³ |

Normal is the default. Three things about that table are deliberate and easy to
undo by accident:

- **Normal's `a_launch * v_knee` is pinned.** At `p = 1.0` that product *is* the
  whole tail (`a = a_launch * v_knee / v`), so raising the launch and lowering the
  knee in step leaves every value above 35 km/h bit identical to the 1.50 / 35 km/h
  curve it replaced. That is what "more off the line, unchanged at speed" means here.
  Pinned by `test_normal_raised_the_launch_without_touching_the_highway_tail`.
- **Efficiency's knee sits just above the launch itself.** A flat plateau held out to
  30 km/h asked for more pedal than the speed warranted. The early knee plus a shallow
  `p` drops the post-launch band (0.81 at 20 km/h against 1.15 before) while leaving
  the highway tail near where it already was (0.39 at 85 km/h against 0.41).
- **Sport is meant never to bind on flat ground.** Capability at gas=1.0 tops out near
  1.4 m/s² above 40 km/h even on a light rig, and Sport's shape stays above that at
  every speed, so the capability guard is the binding term everywhere the truck is
  moving. The driver meets this ceiling only where the engine runs out, which in
  practice means steep inclines. `a_launch` equals the default `cc_accel_max_ms2`
  rail on purpose, so the rail is never the sole binder below the shape.

### The capability term

`a_cap` is the truck's real acceleration at gas=1.0 in the current gear and at the
current mass, read from `sending_thread.data.mapper_est_max_accel_ms2`.

**`HEADROOM_FRAC` is not a comfort knob and must not become per-profile.** The obvious
reading of "respect capability" is a per-mode share, say Normal = 0.70 x capability.
Work the case with `tools/accel_envelope_probe.py --rig loaded`: a 40 t rig at
85 km/h has 0.71 m/s² at gas=1.0. A 0.70 share commands 0.50, while today's
unachievable 1.0 request rails the pedal and the truck delivers its full 0.71. The
share would make loaded trucks *slower than no envelope at all*, which is the exact
complaint this work exists to fix.

So the fraction is a single control constant at 0.95, whose only job is to keep the
pedal just off the rail so the mapper's fast PI keeps trim authority and
`ff_saturated` stops firing in steady cruise.

Which term binds, on a 40 t rig with the cold-start capacity anchor: Efficiency and
Normal are shape-bound at every speed, and Sport is capability-bound above roughly
30 km/h. So the capability guard is in practice a Sport-only limiter, and the two
calmer profiles never see it. On a steep incline the mapper's grade feed-forward
pushes the pedal past the rail and the truck falls short of even the capability
figure, which is the one place a Sport driver notices a ceiling at all.

### Reading the capability estimate

- Fetched with the same defensive registry pattern as `_read_hold_active_safe`. A
  missing or crashed `sending_thread` yields unknown, never an exception.
- Non-finite or `<= 0.05` counts as **unknown**, and the envelope falls back to `shape`
  alone. The publisher legitimately zeroes the field while disconnected or idle, so
  this path is normal rather than exceptional.
- The published value is per-gear and steps by the learned ratio (~1.27) at every
  shift. `_CAPACITY_EMA_TAU_S` smooths it, frozen while `_gearshift_d_factor()` is 0
  so no mid-shift value is ingested, and held rather than cleared when the estimate
  goes unknown for a tick.

### Rise limiting

Comfort is dominated by jerk, not peak acceleration, so each profile rate-limits how
fast its bid may climb. Two details are load-bearing:

- The limit is measured against `max(prev_bid, 0)`, not the raw previous bid. Against a
  raw -0.5 a bid moving to +0.3 would be held negative for several ticks and read as
  phantom braking.
- It is bypassed entirely while `gear_dashboard == 0`. Neutral has no torque path to
  the wheels, so the limit buys no comfort there, and auto-neutral decides to shift
  back to drive off this bid crossing `_AUTONEUTRAL_WANTED_ACCEL_ON_MS2` (0.25 m/s²).
  Ramping from zero in neutral delayed every launch by up to half a second.

### Why this is not in the mapper

`AccelToPedals` is a closed loop on measured acceleration
(`error_ms2 = wanted_smooth - raw_smooth`) with a fast PI and a slow integrator. Gas
inflated there is observed as excess acceleration and integrated back out within a
second or two, so it is self-cancelling. Worse, `PedalCapacityTracker` would learn the
inflated gain as truck capability. The shaping belongs to the request. ACC inherits it
for free through the orchestrator `min()`, so the measured gap law is untouched.

The envelope is never applied to `SpeedLimiter`: that controller's positive bid caps
the **user's own pedal** through the min-merge in `SendingThread`, so shaping it would
throttle a driver flooring it below the cap.

## SpeedLimiter (`limiter.py`)

Lifecycle via orchestrator; no disengage logic here. Gains: `Settings.limiter_*`.

### Continuous tracker

PID runs every tick while enabled; returns `LongOutput(wanted, True)` even below the cap
so the mapper tightens the gas pedal progressively (see `AGENTS.md`).

### Overshoot protection

The kp term already bids strong decel for small overshoots; recovery time is mostly set
by the **floor** clamp, not gain above it. `_overshoot_floor` adds a cubic extra decel
term (`_OVERSHOOT_CUBIC_K * excess_ms³`), capped at `|accel_min|`, so overshoot protection
at most doubles the user-tuned floor.

- **Deadband** (`_OVERSHOOT_DEADBAND_MS`, 0.15 m/s ≈ 0.5 km/h): protection contributes
  nothing until ego is past the limit by more than the deadband, so ACC can sit at the
  cap without the protection floor winning the orchestrator `min()` on every small drift.
  The cubic runs off `overshoot_ms - deadband`, keeping zero slope at the engagement
  boundary. This is not the forbidden "only when over the limit" gate on the limiter bid:
  the continuous tracker still runs and bids every tick, only the extra floor is deadbanded.
- Cubic: negligible at the engagement boundary, meaningful only for real overshoot.
  Saturates at the doubled floor around 1.75 m/s (~6.3 km/h) over.
- **Engagement gate**: cubic scales by shortfall vs external decel only, measured against
  the **full** overshoot (`overshoot_ms / _OVERSHOOT_CLEAR_S`) since the recovery target is
  the limit itself, not the deadband edge. The protection's own commanded decel is
  subtracted before comparing measured decel to that requirement. Prevents self-defeating
  leak when the protection is doing the braking.
- Asymmetric engage/leak taus (`_CUBIC_ENGAGE_TAU_S`, `_CUBIC_LEAK_TAU_S`): bias toward
  not stacking brake when the mapper or driver is already slowing ego.
- kp boost when overshooting: blended over `_OVERSHOOT_BOOST_BAND_MS` to avoid chatter at zero error.
- Measured accel for the gate: tracking differentiator on speed (`_ACCEL_TRACK_TAU_S`), shorter
  than the mapper's accel tau so external braking is seen quickly. Not `lv_accelerationX`: that
  telemetry field is lateral (truck-local right/left), not longitudinal, and reads nonzero from
  cornering alone (see root `AGENTS.md` domain invariants).

## ACC child (`acc.py`)

Following-distance cap only; set-speed PID stays in `CruiseController`. Inner law:
`core/cruise_control_thread/acc_controller.py` and `core/acc/ACC_ARCHITECTURE.md`.
