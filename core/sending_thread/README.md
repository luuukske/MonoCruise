# Sending thread and pedal I/O

> Maps commanded accel to game pedals via shared memory (`SCSController`).
> Longitudinal arbitration in `cruise_control_thread`; agent rules in root `AGENTS.md`.

## Sending thread (`thread.py`)

Opens SCS controls, runs `AccelToPedals`, hold FSM, pedal capacity learning, optional
visualization bar, hazard toggling, AEB decel assist, auto-neutral, creep compensation,
and commander merge (CC/ACC/limiter/AEB/user). Publishes `aforward` / `abackward` on
`SendingThreadData`.

## AccelToPedals (`accel_to_pedals.py`)

Single mapper instance for the process. Converts wanted m/s² to gas/brake with smoothing,
leaky integral, road-load feed-forward (pitch via `rotationY`, rolling resistance setting),
adaptive full-pedal accel/brake estimates, gearshift integrator freeze, and tuning CSV rows
when high-demand estimates underperform.

## Pedal capacity (`pedal_capacity.py`)

Always-on brake decel and gas gain learning (replaces legacy brake efficiency tracker).

**Brake**: `update_brake` every tick; accept samples only when pedal and decel settle.
Candidate inverts the fitted brake curve; pedal³ weighting; underperformance drops estimate
2× faster than overperformance rises. Road load canceled before sampling. Fast EMA during
deep settled AEB braking. Hard bounds on estimate (`_ESTIMATE_UPPER_BOUND` protects AEB
engagement from inflated decel). Reject contaminated candidates above `_BRAKE_CANDIDATE_MAX_FRACTION`.

**Gas**: shape model `G(gear) = anchor * ratio^(_ANCHOR_GEAR - gear)` learned in log-space;
monotonic ratio clamp. Skipped after clutch, gear dwell, or moving pedal. Persisted to
`settings.json` on drift.

## Hold controller (`hold_controller.py`)

Single authority for hill rollback prevention. FSM: ROLLING / STOPPING / HOLDING / LAUNCHING.
Hold states combine slope feedforward (inverse brake curve) plus a rollback integrator that
only adds brake. LAUNCHING ramps feedforward down; integrator stays active; ramp retreats on
live rollback, not on stored integrator level (avoids steep-hill launch livelock).

## Brake efficiency (`brake_efficiency.py`)

Optional cruise-only degradation warning via EMA of measured vs expected decel (nominal max
scales with wheels and mass). Flat-road gate; high-brake samples only.

## Main pedal thread

See `core/main_pedal_thread/README.md` for joystick, OPD, em_stop, and button capture.
