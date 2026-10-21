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

The replacement is a comfort shape per profile. `Settings.cc_accel_max_ms2` survives
as a hard rail on top, default 4.0, above every shape; the old shipped 1.0 and 2.5
are remapped to it on load, and a value a driver set to something else is left alone.

### The shape

```
a(v) = a_cruise + (a_launch - a_cruise) / (1 + (v / v_half)²)
```

Flat at rest (zero slope, so a launch does not start on a jerk step), halfway to the
cruise value at `v_half`, and smooth everywhere: there is no knee.

| Profile | a_launch | v_half | a_cruise | rise jerk | 20 km/h | 50 km/h | 85 km/h |
|---|---|---|---|---|---|---|---|
| Efficiency | 1.50 | 32 km/h | 0.35 | 1.0 m/s³ | 1.18 | 0.68 | 0.49 |
| Normal | 2.20 | 45 km/h | 0.50 | 2.0 m/s³ | 1.92 | 1.26 | 0.87 |
| Sport | none | | | none | | | |

Normal is the default. Efficiency is the same curve at 55 to 70 % of Normal at every
speed, so it feels like a gentler Normal rather than a different character.

Why this family (2026-10-19): a flat launch to a knee at 10 to 12 km/h, then a power
taper, asked Efficiency for 0.14 m/s² at 50 km/h and Normal for 0.29 at 85. The pull
was gone before the truck left town, and a highway resume crawled.
`test_the_pull_does_not_fall_away_through_town_speeds` pins the fix: no 10 km/h step
may cost more than a fifth of the pull.

Sport has no shape. The speed error is the bid, with no `cc_accel_max_ms2` rail and
no rise limit, so whenever the bid is more than the gear can do the pedal is on the
floor. That is a standing driver decision (2026-10-19): Sport is always uncapped.

`tools/accel_envelope_probe.py` prints the table, 0-50 / 0-90 times, and where a rig
model cannot deliver the request.

### No capability term

Removed 2026-10-19. The envelope used to be `min(shape, 0.95 * a_cap)`, with `a_cap`
the mapper's published `mapper_est_max_accel_ms2`. That estimate is not good enough
to cap a request with. In the 2026-10-19 mapper log it read 1.36 m/s² in 4th to 8th
gear while the truck pulled 1.0 to 1.6 on 20 to 50 % gas (the mapper's own
`StrengthScale` knew; the published estimate does not carry it), so a town pull was
held at 1.29 m/s². At 85 km/h it read 2.05 where full pedal gave 1.33, so it never
bound there. It could only hurt.

A truck that really cannot meet the shape now rails the pedal, which is what a driver
does and what Sport does on purpose. The mapper is built for that: its fast integrator
is back-calculated while the pedal is saturated, and its slow integrator does not wind
against a railed pedal.

Do not bring it back as a per-profile share either. Normal = 0.70 x capability was
costed and rejected before the estimate was even in question: a 40 t rig at 85 km/h
has 0.71 m/s² at gas=1.0, so the share commands 0.50 where railing delivers 0.71.

### ACC with no lead, or a lead far ahead

ACC's cap is min-merged with the cruise bid. With no lead it used to bid a flat
`NO_LEAD_CEILING_MS2` (1.5 m/s²), which capped every profile on an empty road: in the
2026-10-19 log Sport held exactly 1.50 to 90 km/h. Behind any tracked lead (tracker range
150 m) the IIDM law did the same thing through its free-road term, `a_max (1 - (v/v0)^δ)`
with `a_max` 1.5, even when the lead was far away and pulling away.

The orchestrator now hands CC's bid to the ACC child, which turns it into a free-road
target (`free_road_target_ms2` in `core/longitudinal/acc.py`): the bid clamped to
`[NO_LEAD_CEILING_MS2, a_max + j_onset_dead]`, 1.5 to 2.5 m/s². With no lead that target
is ACC's cap. With a lead, `core/cruise_control_thread/free_road_lift.py` raises the law
toward it while the law already grants nearly its whole free-road accel, and not at all
once the lead binds (`core/cruise_control_thread/README.md`, "Free-road lift"). Both go
through ACC's own jerk limiter and output filter. Below 1.5 nothing changed.

The 2.5 bound is load-bearing. From it, a lead that starts to bind (law near 1.4) is met
at the plain 2.5 m/s³ jerk rate, because the drop stays inside the onset chase's 1.0 m/s²
dead band. A 4.0 bound stepped the command down through the onset chase every time a car
came into range. Normal's strongest bid (2.2) is under it, so ACC never caps Normal or
Efficiency on a free road; Sport with ACC on is held at 2.5 there, which no truck exceeds
outside the lowest gears. With ACC off Sport has no ceiling at all.

Measured behind a faster lead 80 m ahead (closed loop, 3.0 m/s² truck): 0 to 40 km/h took
7.8 s on the law alone, 5.4 s on Normal and 4.8 s on Sport, with the gap above 115 m.
Closing on a slower or stopped lead from 150 m, minimum gap and peak decel were identical
with and without the lift and top speed rose by at most 0.9 km/h, because closing speed
takes the law out of the band within a second or two. The platoon sim is bit-identical
with the lift on and off, on Normal and on Sport. Enforced by
`tests/acc/test_no_lead_follow.py` and `tests/acc/test_free_road_lift.py`.

### Rise limiting

Comfort is dominated by jerk, not peak acceleration, so each comfort profile
rate-limits how fast its bid may climb. Two details are load-bearing:

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

The one exception is a telemetry disconnect. The speed it would track is stale, so the
limiter bids nothing and resets its PID state, while the enabled flag and target stay
set so the cap is back on the first connected tick. The zeroed pedals in
`sending_thread`'s disconnected branch used to be the only defence against that stale
bid.

### D-term on low-passed speed

The D-term differentiates the same power-shaped error as the P-term, but on speed through a
0.10 s low-pass (`_KD_SPEED_SMOOTH_TAU_S`). Telemetry speed only moves at the game's ~20 Hz
physics rate while the limiter ticks at the polling rate (100 Hz), so the tick-to-tick
derivative was zero for four ticks and five times too large on the fifth, and the bid jumped
by up to 0.4 m/s² between 10 Hz log samples (2026-10-21). Gains, the overshoot boost and how
fast the cap reacts are unchanged: on a 20 Hz staircase at 0.25 m/s² the low-pass cuts the
D-term ripple from 0.23 to 0.02 m/s² with the same mean. Do not smooth the output instead: the
limiter's positive bid caps the driver's own pedal, and a slower cap lets the truck through
the limit. Enforced by `tests/longitudinal/test_limiter_d_term.py`.

### Overshoot protection

The kp term already bids strong decel for small overshoots; recovery time is mostly set
by the **floor** clamp, not gain above it. `_overshoot_floor` adds a cubic extra decel
term (`_OVERSHOOT_CUBIC_K * excess_ms³`), capped at `|accel_min|`, so overshoot protection
at most doubles the user-tuned floor. The published bid is then raised to
`_LIMITER_MAX_DECEL_MS2` (1.0 m/s²) when the floor plus cubic is deeper than that.
A limit set far below current speed eases the truck back. It does not haul it down
at the old doubled floor (2.0 at the old default, 3.0 at a saved floor of -1.5).
Do not remove the ceiling. Grade is still the mapper's feed-forward, so this cap
is extra closing rate, not the brake that holds a hill.

- **Deadband** (`_OVERSHOOT_DEADBAND_MS`, 0.15 m/s ≈ 0.5 km/h): protection contributes
  nothing until ego is past the limit by more than the deadband, so ACC can sit at the
  cap without the protection floor winning the orchestrator `min()` on every small drift.
  The cubic runs off `overshoot_ms - deadband`, keeping zero slope at the engagement
  boundary. This is not the forbidden "only when over the limit" gate on the limiter bid:
  the continuous tracker still runs and bids every tick, only the extra floor is deadbanded.
- Cubic: negligible at the engagement boundary, meaningful only for real overshoot.
  Its own term still saturates at `|accel_min|`, but the published bid stops at the ceiling.
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

## Panic bypass (`limiter_override.py`)

A limit set lower than the driver meant has two outs. The brake ceiling above is the
passive one. The other is a pedal gesture, owned by `CruiseControlThread` because
the limiter child never sees the pedal.

Holding the gas at the floor keeps the limiter in command. That is the normal
"floor it until the cap" case, and it must not get past the limit. The bypass is
a fast stab while speed is already within `_BIND_UNDER_KMH` of the cap or over it.
How far the pedal moves is only a minimum (`_GAS_FLOOR` to `_GAS_RELEASE`). A
slow lift of any size does not qualify. The speed gates are `_QUICK_RELEASE_S`
on the way off and `_BLIP_WINDOW_S` on the way back.

1. Gas has been at `_GAS_FLOOR` for `_ARM_HOLD_S` in one press. Time on the floor
   counts on the way up, so a driver who has been pressing into the cap can stab
   as soon as it binds. A short press does not arm.
2. Gas falls through `_GAS_RELEASE` within `_QUICK_RELEASE_S` of leaving the floor.
   That interval is the speed of the lift. A roll-off drops the arm.
3. Gas is back at the floor inside `_BLIP_WINDOW_S`, and the lift lasted at least
   `_RELEASE_MIN_S` so a one-sample dip does not count.

While latched, the orchestrator publishes no CC, ACC, or limiter bid. The driver's
pedal is the only longitudinal command. AEB is untouched. Lifting off does not
end it. The latch ends only when speed falls below `_REARM_ABOVE_KMH` above the
cap, and only after the bypass has already been at least that far over, so a
stab at the cap can still pull away. The next approach needs a new stab. The rising edge emits a warning popup
at priority 2 with a short message (`PopupWindow.emit`, not the log-handler notice).

Do not treat a held floor as the bypass. That was the case this exists to refuse.

## ACC child (`acc.py`)

Following-distance cap only; set-speed PID stays in `CruiseController`. Inner law:
`core/cruise_control_thread/acc_controller.py` and `core/acc/ACC_ARCHITECTURE.md`.
