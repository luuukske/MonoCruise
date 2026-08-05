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

### Braking authority vs tracker confidence

The tracker publishes leads at `score > 0`; this controller only starts trusting
one at `score > ANT_SCORE_MIN` (0.5). Everything in that gap used to receive full
braking authority anyway, because the TTC overlay ran on `chain_raw[0]` before the
confidence blend. Measured on the clip corpus, 22.9 % of overlay trips came from a
lead below the confidence floor and 8.2 % from one at exactly zero confidence,
nearly all of them stationary vehicles at 6 to 24 m. The least certain leads were
getting the most violent response.

`ANT_SCORE_MIN` is the **only** knob measured to change latch time: it was
lowered from 1.0 to 0.5 for lock p50 0.99 -> 0.81 s and cut-in lock 1.26 -> 1.10 s.
Loosening any of the road-model smoothing left latch time bit-identical (see
`core/acc/README.md` §9 rejected table). The floor also gates the TTC overlay
below, so lowering it buys reaction time at the cost of some braking exposure:
hard-brake trips on stopped vehicles rose 147 -> 154 on the clip corpus, which is
an upper bound because that count includes legitimate stopped traffic. Do not
lower it further; at 0.25 the median gained 0.02 s while cut-in latch got worse.

Two rules now:

- The **TTC overlay** (`ttc_hard_s` to `max_decel_ms2`) requires
  `_score_conf(...) > 0`. Below that the lead still goes through the normal
  IIDM+CAH law with the confidence blend, so it decelerates, just not at full
  authority. The **close-range emergency overlay** (`d_emergency_m`) stays
  ungated: at that distance certainty is irrelevant.
- Low confidence may **soften braking, never invert it**. With a single-lead
  chain the blend's alternative is `NO_LEAD_CEILING_MS2` (+1.5 m/s²), so a
  zero-confidence lead used to blend all the way to full acceleration toward the
  very target the lead law wanted to brake for. When the pre-blend lead law is
  negative the blended command is now clamped at 0, i.e. coast at worst.

Enforced by `tests/acc/test_overlay_confidence_gate.py`.

### Lead-loss grace

Brief empty-chain ticks reuse the last chain for `LEAD_LOSS_GRACE_S` so IIDM state does not
step between brake command and `NO_LEAD_CEILING_MS2`.

### No-lead path

Ceiling routed through the same jerk and output filter as real commands so `_prev_cmd_ms2`
stays continuous (see ACC architecture §15).
