# Cruise control orchestrator

> ACC in-lane tracking: `core/acc/README.md`. IIDM+CAH formulas: `core/acc/ACC_ARCHITECTURE.md`.
> Longitudinal children: `core/longitudinal/README.md`. Root `AGENTS.md` for CC/limiter
> mutual exclusion, disengage scope, user gas override, and mapper arbitration.

## Thread responsibilities

Each tick: snapshot telemetry and pedal, run CC button FSM and ACC gap FSM, build `LongCtx`,
handle mode-flip PID reset, CC-only disengage, dispatch by `Settings.cc_mode`, publish
`wanted_accel_ms2` and `active_controller` for `sending_thread` and UI.

CC-only disengage: raw/game brake thresholds, park/neutral/reverse, disarm-on-stop, crash
speed drop. Limiter path never sees these.

User OPD gas override (cruise mode, limiter active): latch excludes CC/ACC bids so the
limiter caps the user pedal until gas releases or ego falls below CC target minus margin.

## Adaptive cruise (`acc_controller.py`)

Publishes an upper bound on commanded accel (m/s²). Orchestrator uses `min(speed_pid, cap)`.

### Anticipation and chain tuning

Multi-lead anticipation weights pairwise time gaps (cosine ramp `ANT_GAP_FULL_S` to
`ANT_GAP_ZERO_S`), multiplies down the chain, and EMA-filters the total delta (`ANT_TAU_S`).
Score confidence ramps (`ANT_SCORE_MIN` to `ANT_SCORE_FULL`); per-vehicle confidence EMA
is fast up / slow down. Primary-lead ghost hold (`PRIMARY_GHOST_HOLD_S`) fades braking
when the immediate lead vanishes. Accel-side lift (`ANT_KV`, `ANT_KA`, TTC ramp) fades when
decel anticipation binds. Stationary immediate lead disables anticipation (speed ramp
`ANT_LEAD_MOVING_MIN_MS` to `ANT_LEAD_MOVING_FULL_MS`).

### Lead-loss grace

Brief empty-chain ticks reuse the last chain for `LEAD_LOSS_GRACE_S` so IIDM state does not
step between brake command and `NO_LEAD_CEILING_MS2`.

### No-lead path

Ceiling routed through the same jerk and output filter as real commands so `_prev_cmd_ms2`
stays continuous (see ACC architecture §15).
