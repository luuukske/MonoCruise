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
