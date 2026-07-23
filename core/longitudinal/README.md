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

## SpeedLimiter (`limiter.py`)

Lifecycle via orchestrator; no disengage logic here. Gains: `Settings.limiter_*`.

### Continuous tracker

PID runs every tick while enabled; returns `LongOutput(wanted, True)` even below the cap
so the mapper tightens the gas pedal progressively (see `AGENTS.md`).

### Overshoot recovery envelope

The kp term already bids strong decel for small overshoots; recovery time is mostly set
by the **floor** clamp, not gain above it. `_overshoot_floor` adds a cubic extra decel
term (`_OVERSHOOT_CUBIC_K * overshoot_ms³`), capped at `|accel_min|`, so the envelope
at most doubles the user-tuned floor.

- Cubic: negligible at the limit boundary, meaningful only for real overshoot.
- **Engagement gate**: cubic scales by shortfall vs external decel only. The envelope's
  own commanded decel is subtracted before comparing measured decel to the clear-window
  requirement (`overshoot_ms / _OVERSHOOT_CLEAR_S`). Prevents self-defeating leak when
  the envelope is doing the braking.
- Asymmetric engage/leak taus (`_CUBIC_ENGAGE_TAU_S`, `_CUBIC_LEAK_TAU_S`): bias toward
  not stacking brake when the mapper or driver is already slowing ego.
- kp boost when overshooting: blended over `_OVERSHOOT_BOOST_BAND_MS` to avoid chatter at zero error.
- Measured accel for the gate: tracking differentiator on speed (`_ACCEL_TRACK_TAU_S`), shorter
  than the mapper's accel tau so external braking is seen quickly.

## ACC child (`acc.py`)

Following-distance cap only; set-speed PID stays in `CruiseController`. Inner law:
`core/cruise_control_thread/acc_controller.py` and `core/acc/ACC_ARCHITECTURE.md`.
