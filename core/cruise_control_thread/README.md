# Cruise control orchestrator

> ACC in-lane tracking: `core/acc/README.md`. IIDM+CAH formulas: `core/acc/ACC_ARCHITECTURE.md`.
> Longitudinal children: `core/longitudinal/README.md`. Root `AGENTS.md` for CC/limiter
> mutual exclusion, disengage scope, user gas override, and mapper arbitration.

## Thread responsibilities

Each tick: snapshot telemetry and pedal, run CC button FSM and ACC gap FSM, build `LongCtx`,
handle mode-flip PID reset, CC-only disengage, dispatch by `Settings.cc_mode`, publish
`wanted_accel_ms2` and `active_controller` for `sending_thread` and UI.

CC-only disengage: raw/game brake thresholds, park/reverse, disarm-on-stop, crash
speed drop. Neutral does not disengage: positive CC/ACC bids are clamped to 0 while
`gear_dashboard == 0`, with a popup after 2 s continuous N. Skip the clamp while
`auto_neutral_holding` so auto-neutral can see the launch bid and shift to drive.
Limiter path never sees these.

User OPD gas override (cruise mode, limiter active): latch excludes CC/ACC bids so the
limiter caps the user pedal until gas releases or ego falls below CC target minus margin.

## Button presses (`press_counter.py`)

Short presses fire per press **counted**, not per press **observed**. This thread
samples `main_pedal_thread`'s published button level on its own clock, and both
run at `max(Settings.polling_rate, 10)`. At 10 Hz that is a 100 ms tick, so a tap
can begin and end entirely between two samples and be lost. Measured in the wild:
6 of 22 rapid taps dropped, while slow presses were never affected.

Counting has to happen where the edges are actually visible. `button_device_thread`
(100 Hz, drains every HID report) and `keyboard_thread` (OS hook, no polling)
each publish their own press counts; `resolve_press_count()` in
`core/input_bindings.py` reads them. `main_pedal_thread` consumes those deltas
into `cc_button_press_counts`, seeded at zero for all five bindings so a consumer
baselining on first sight cannot swallow the first press, and falls back to its
own edge detection only for joystick bindings, which have no source counter.
Counting in the pedal thread alone was not enough: it polls at the same 10 Hz and
missed the tap before its own counter ever saw it. `PressCounter.take_short`
returns how many counted presses still owe an action, excluding one still in
progress (it may yet become a long press). A long press calls `consume_one` so the
same press never also fires a short action.

Long presses still come from the level, since they need continuous hold. Only the
short-press path is count-driven.

Presses counted while input is gated are **discarded**, never queued: `discard()`
runs whenever telemetry is disconnected, the game is paused, the pedal device is
lost, the buttons are unassigned, or park/reverse blocks increase. Without that,
unpausing would replay a burst of speed changes at once.

`main_pedal_thread` also latches each press for `_BUTTON_MIN_HOLD_S`, which keeps
the level representative for anything reading it; correctness no longer depends on
that latch.

## ACC gap buttons (`acc_distance.py`)

One button assigned cycles the level and wraps; two assigned step and clamp, and
are suppressed while both are held. Same count-driven short press as above.

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

### Blinker arbitration

Tracker publishes `indicated_lead`, `blinker_b_eff` and `blinker_committed`
separately from `leads[]`. This controller applies R5–R8 (freer-lane pass in two
stages, tighter-lane min, hysteresis, TTC floor). Anticipation still reads
`leads[]` only. Candidacy and the intent model: `core/acc/README.md` §5.

**The indicated lead carries no collision authority.** `_pick_indicated_lead`
publishes only vehicles that are *not* in ego's corridor, so by construction
there is no collision path from ego's front to one. `_indicated_accel` therefore
bounds its demand at `-b_comfort` and the indicated-only path never runs
`_safety_overlays` and never returns `is_emergency`.

It used to do all three. A vehicle in the next lane could return
`emergency_decel_ms2` (-8.0) or `max_decel_ms2` (-6.55) with the emergency flag
set, and that flag is precisely what bypasses `_jerk_limit` and `_output_filter`,
so the command stepped there in a single tick. Combined with the one-sided R4
window (`core/acc/README.md` §5) it fired while ego was mid-overtake alongside a
slower vehicle: measured -6.55 m/s² unfiltered for a car 2 m off ego's front
bumper in the adjacent lane, with ego's own lane clear and asking for +1.21.
Unfiltered full brake for a vehicle ego cannot hit is its own rear-end risk for
whoever is behind ego. If merging behind a candidate needs more than comfort
braking, the merge is not happening and lane policy should own the tick; the
bound means the driver feels "not helping you in there" instead of a slam.
`leads[]` and AEB remain the collision backstop for the moment ego's body
actually enters the lane, and both keep full authority.

Stage 2 (`pass`) releases the lane being left rather than taking a min against
it. How much is released is `release_fraction`, ramped on whether the gap the
old lead still holds is realistic: full authority below `BLINKER_RELEASE_TTC_MIN_S`
or the speed-scaled gap floor, none above the full marks. So a committed overtake
with room lifts the cap entirely, and one that runs out of room gets the old lead
back proportionally, before `_safety_overlays` (TTC 1.5 s) has to catch it.

**The release outlives `b_eff`.** R11 collapses the blinker scalar on merge
completion, but the vehicle ego just left is still published for as long as its
score takes to decay (release p90 1.06 s). Ending the release at the collapse put
that vehicle back in full command for exactly that window, and since ego had been
accelerating under the release it was now closing faster than before, so the lane
law demanded a brake. Measured as a **-2.2 m/s² step** in the command, felt from
the seat as a brake tap just as ego clears the old lane, which is what
`test_no_blip_across_the_whole_collapse_window` pins.

`released_vid` therefore latches the vacated vehicle during stage 2 and keeps
releasing it after `b_eff` is gone, until it leaves the chain, another vehicle
becomes the primary, or `BLINKER_RELEASE_HOLD_S` (2.0 s) expires. The target is
`_chain_tail_accel`, the command the chain would give with that vehicle removed,
which is what "let it go" means once there is no indicated lane left to aim at.
Two guards keep the hold honest: it only latches once `lane_offset_m` has reached
`BLINKER_RELEASE_HOLD_MIN_M` (2.0 m), so an aborted lane change never carries a
release past its intent, and `release_fraction` is still evaluated every frame, so
closing on the vacated vehicle hands authority straight back.

Two bugs made stage 2 unreachable and are fixed here:

- **The ghost fade in `arbitrate` was dead code.** It branched on `ghost_age_s`,
  which is non-`None` only when `_ghost_vid == primary.vid`; but
  `_apply_primary_ghost` runs first and clears `_ghost_vid` whenever it is in the
  chain, and the primary always is. So the fade never ran and the two ghost
  mechanisms overwrote each other every frame. The blinker path no longer touches
  `_ghost_vid`; it suppresses the ghost while committed and nothing else.
- **`mode_mono` was stamped every frame**, including frames where the mode did
  not change, so `now - mode_mono` was one frame time and the `BLINKER_HYST_S`
  dwell test always passed. The hysteresis was permanent rather than 0.4 s:
  `soften`/`pass` could never fall back to `lane`, and `pass` could never yield
  to a tighter indicated lane. `_set_mode` stamps only on a real change.

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
