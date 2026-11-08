# MonoCruise FFB plugin (prototype)

An ETS2 / ATS SDK plugin that adds a short zero-mean rumble (`core/lane_departure/README.md`)
and the LA's steering (`core/la/README.md`) to the game's own constant-force
effect while MonoCruise asks for them. It is the probe in `plugins/ffb_probe` grown a command
channel. The steering axis is never written and with no request the game's force passes through
untouched. MonoCruise ships it and the SDK installer puts it in the game once the driver picks Warn or
Steer under Lane departure (or LA is on), never before (`core/sdk_installer/README.md`, MonoCruise's own
plugins). Set back to None it stays, idle, and is kept current. Remove the probe DLL if it is installed
(two plugins cannot both patch the effect), and delete the DLL to remove it.

## How it works

`scs_telemetry_init` opens DirectInput, finds the force-feedback device the game uses and
patches the `SetParameters` slot of its constant-force effect. Each call it reads
`Local\MonoCruiseFfbCmd` and adds to the game's magnitude, clamped to plus or minus 10000:

- a rumble, if requested: a 12 Hz sine under a slow envelope (0.25 s in, 0.30 s out);
- a torque, if requested: capped at 3000, low-passed (0.06 s) and slope-limited (6000 per second
  building, 12000 per second releasing). The LA sends none since layout 4; the test tool uses it;
- the steering servo (layout 5): a PID on the wheel angle, added to the game's force, `kp` on the error,
  `kd` on the rate error (the wheel's speed low-passed at 6 ms, the damping capped at half the servo cap: 1500 at
  the default 3000) and `ki` on the error over time (no windup while the output is pinned, at most the cap). The
  target is moved on at the rate and acceleration MonoCruise sends for at most 0.15 s after each new one
  (`servo_stamp`), so its 20 Hz updates never reach the rim as steps. Faded in 0.3 s and out 0.2 s, slope-limited
  at 150000 per second, capped, and halved when it buzzes:
  four swings within 0.5 s, a swing being the wheel turning back by 0.002 of lock or more from its last extreme
  with the push (past 300) flipped since the swing before. It comes back at 0.25 per second, 4 s from its 0.1
  floor. Until drive 12 (2026-11-02) it counted flips of the push alone and recovered over 20 s: the push
  flipping around zero on a still wheel, as a lane keeping pulse turned the truck back, cut 10 of 14 pulses,
  7 of them to 0.1, and the next pulses started weak. Lukas approved the change. While the plugin's own
  rumble plays (and fades) the guard counts nothing and starts over: the 12 Hz rumble at the line shakes the
  rim by about 0.06 lock/s, which the damping turned into a push flipping about 24 times a second, and that
  is what tripped the old count at the start of every pulse (Lukas, drive 12);
- one way only, if asked (`servo_only` at offset 47, +1 counterclockwise, -1 clockwise, 0 both): the PID's
  output the other way, damping included, is clipped to nothing and its integral never builds that way, at
  the FFB rate. The lane keeping assist's pulses use it so they never push the wheel back toward the line
  (Lukas, 2026-11-02). A plugin that knows it sets status bit 128; MonoCruise keeps LKA off without it;
- damping first, if asked (`servo_flags` bit 0 at offset 90, status bit 256 says the plugin knows it): P and I get only
  the room the cap leaves after D, so a servo pinned at its cap still brakes a wheel swinging through the target.
  The caps are the same (the total 3000, D 1500). Off, D is added to a pinned P and hidden behind it. Off by default:
  MonoCruise does not set it yet, `tools/la_servo_bench.py --law` does (`core/la/README.md`, Bench);
- the push ahead, if asked (layout 5, status bit 512, 0 by default): the force the rim itself needs to move as the
  target moves, its inertia times the target's acceleration, its damping times the target's rate and its dry friction
  the way it moves, added to P within the cap. The rim is the wheel check's fit (`la_ff_*` keys). With it the
  standing R3's model follows a ramp at 0.8 lock per second where the feedback alone fell behind at 0.4. With the
  plan (`servo_flags` bit 1, status bit 1024) it moves on the rate and acceleration MonoCruise sends at offset 116
  instead: where the target will be a wheel delay from now, so the rim starts and stops on time. On the 60 Hz model
  of the R3 that took a half lock at 2 lock/s from 23 to 5 % overshoot. At rest (the push's rate under about 0.02
  lock/s) the friction push turns toward the target instead, full from 0.002 lock of error: standing, the game holds
  about 200 against the way the wheel last moved (its steering friction, about 550 while it moves, 2026-11-06), and
  without the push the wheel parked a few mlock short or past and crept in over most of a second. With the plan
  the feedback also stops chasing what the reading shows late: the damping compares the wheel's rate with the
  target's lagged by half the interval between steering readings (measured by the plugin) plus the 6 ms rate filter,
  at most 0.05 s, and the integral holds while the target or the plan moves faster than 0.3 lock/s (LA's target never
  does). On the wheel the integral wound up to about 650 during a fast half lock on the reading's lag and pushed
  the wheel 6 mlock past the stop for over a second; in the model the lagged damping halved the swing after a stop;
- the breakaway push, if asked (offset 124, status bit 2048, 0 by default): while the wheel stands the friction push
  climbs from the sliding friction to this ceiling over 0.3 s (`FFB_BREAK_RAMP_S`), and drops back the moment the
  reading changes. "Stands" is read off the angle, not its rate: the climb starts once 1.5 readings came without a
  change, so a coarse encoder's slow slide (steps with still readings between) does not count as standing. It climbs
  only while the target moves or is more than 1.5 mlock away (`FFB_BREAK_HOLD`, all of it from 3 mlock,
  `FFB_BREAK_BAND`), and within 1.5 mlock the rest push toward the target is off too, so a wheel stopped near its
  target is not restarted into a hunt (the old wheel hunted ±1.5 mlock at 4 Hz on the rest push alone, 21:45). The old wheel (a Thrustmaster, 2026-11-06) slides on about
  305 but starts on anything from 311 to 526 from one try to the next, with angle steps of 0.0006 lock: without a
  push the stick-slip flipped the push and the buzz guard cut every slow ramp; a fixed push (370 at 19:16) left it
  standing with the servo at 440, or lurched it where it started easily. With the climb to just above its highest
  start, on its model (each stop drawing its own start force) a 0.5 Hz design holds every stage, slow tracking 5 to
  10 mlock rms where the wheel stood at 37 to 48;
- the settle law, if asked (`servo_flags` bit 2, status bit 4096, off by default; MonoCruise asks for it with
  `la_wheel_learning`): for a wheel with a breakaway push, the push finishes the move down to a band round the
  target (`FFB_SETTLE_BAND`, 0.5 mlock plus half the reading's step, which the plugin reads off the smallest recent
  change) instead of standing off inside 1.5 mlock; within 3 mlock it climbs over `FFB_SETTLE_RAMP_S` (1 s) from
  nothing rather than from the sliding friction, and lets go over `FFB_SETTLE_DROP_S` (20 ms) when the reading
  changes, not in one tick. A wheel standing within the band of a still target parks: the output is held as it was
  and the integral stands until the error leaves the band by its step (at least 1 mlock) or the target moves. Within
  the cap like every other term. Simulated only (`tools/la_bench/SMOOTH_RESULTS.md`): on the model population it took
  the learned design from 36 to 49 % of wheels holding their target within the limits; on the light, very sticky
  model wheel (`sticky`) it hunts, which is why MonoCruise turns it off for a wheel that still hunts at its gentlest
  design. **Not in the shipped DLL yet**: the source has it, the DLL in `core/sdk_installer/own/` is the layout 5
  build and ignores the bit;
- the lead, if asked (layout 5, 0 by default): the wheel and the target are read that many seconds ahead (the wheel
  on at its rate, plus the age of the steering reading, which comes at the game's frame rate, 60 Hz on the R3),
  together at most 0.08 s: where both are when the force reaches the rim. In the model it cut a small step's
  overshoot (25 to 16 %) but overshot sharp-cornered ramps; untested on the wheel;
- the cancel, only if asked (`servo_cancel`, 0 by default): that share of the game's steady force (its
  magnitude low-passed at 1.5 s, so turn-in, corrections and bumps still pass) is subtracted while the
  servo holds, faded with the servo and capped at 6000.

Why the servo moved into the plugin: fed from MonoCruise it saw the wheel a frame late with no damping and
rang at 3 to 4 Hz, its 20 Hz target arrived in steps, and holding the wheel on target against the game's
centring needed a model of the game's force that changed from drive to drive (`core/la/README.md`).
The servo's integral now carries the bend instead, and the game's own force stays on the wheel.

The wheel angle comes from the SDK channel `truck.input.steering`, registered every frame. The plugin
writes it back with its rate, the game's own magnitude (low-passed at 0.1 s, MonoCruise calibrates on
it), what it is applying, the servo's integral, the cancelled force and the game's FFB update rate. The patch
is undone at shutdown.

## Command block (`ffb_cmd.h`, 128 bytes, written by `core/lane_departure/command.py`)

Steering values are `truck.input.steering` units: -1 to 1 of lock, counterclockwise positive.

| offset | field | writer |
|---|---|---|
| 0 | version, 5 (1 rumble only, 2 added the torque, both 32 bytes; 3 a servo without damping, 64 bytes; 4 the damped servo, 96 bytes); 0 means no plugin | plugin |
| 4 | heartbeat, +1 per write, never 0 | MonoCruise |
| 8 | command (0 none, 1 rumble), side, intensity (u16, peak out of 10000) | MonoCruise |
| 12 | event counter | MonoCruise |
| 16 | torque (i16, out of 10000, signed like the game's magnitude) | MonoCruise |
| 18 | game force (i16, the game's magnitude low-passed at 0.1 s) | plugin |
| 20 | status bits: hooked, force seen, rumbling, dead, torque, servo, steering channel, one-sided servo (128), damping first (256), push ahead and lead (512), the planned push ahead (1024), the breakaway push (2048), the settle law (4096) | plugin |
| 24 | count of constant-force updates seen | plugin |
| 28 | assist applied now, torque plus servo (i16) | plugin |
| 30 | servo part applied now (i16) | plugin |
| 32 | servo target (f32), kp (f32), ki (f32, per lock per second), cap (u16), direction (i8: torque sign that turns the wheel counterclockwise, 0 off), only (i8 at 47: +1 or -1 the one way it may push, counterclockwise positive, 0 both) | MonoCruise |
| 48 | wheel angle (f32), its rate (f32), steering channel updates (u32) | plugin |
| 60 | buzz guard scale (f32) | plugin |
| 64 | target rate (f32, lock per second), kd (f32, per lock per second of rate error), cancel (f32, 0 to 1, share of the game's steady force) | MonoCruise |
| 76 | servo stamp (u32, bumped with each new request) | MonoCruise |
| 80 | servo integral now (f32), the game's FFB update rate (f32, Hz) | plugin |
| 88 | game force cancelled now (i16) | plugin |
| 90 | servo flags (u8): bit 0 damping first, bit 1 the push ahead moves on the plan at 116, bit 2 the settle law | MonoCruise |
| 92 | target acceleration (f32, lock/s^2), the rim's inertia, damping and friction to push ahead (f32 each, 0 none), lead (f32, s, at most 0.08) | MonoCruise |
| 112 | push ahead applied now (i16, the wheel's torque sign) | plugin |
| 116 | the plan the push ahead moves on with flag bit 1: the target's rate (f32, lock/s) and acceleration (f32, lock/s^2) a wheel delay from now, moved on at that acceleration for at most 0.15 s | MonoCruise |
| 124 | the breakaway push (f32, the ceiling the friction push climbs to while the wheel stands, used when above the friction at 104; 0 none; reserved before status bit 2048) | MonoCruise |

MonoCruise still rumbles through a version 1, 2 or 3 plugin (it falls back to the smaller block), and
steers through version 4 (without the push ahead and the lead) and 5. MonoCruise opens the 128-byte block first:
started before the game, it creates it and a version 4 plugin maps its own 96 bytes of it. A version 5 plugin
under a MonoCruise older than layout 5 that created the 96-byte block first finds no room and stays off; the
SDK installer puts the matching DLL back the next time the game is closed.

Change `ffb_cmd.h` and the Python module together; `tests/lane_departure/test_command.py`
compares them.

## Trace block (`ffb_trace.h`, 6336 bytes, read by `core/lane_departure/ffb_trace.py`)

`Local\MonoCruiseFfbTrace`, separate from the command block so that block's layout and version did not move.
The plugin writes it and MonoCruise only reads it (with `debug` on, into `la_ffb_debug.csv`); nothing in it
steers. A plugin without it simply has none.

| offset | field |
|---|---|
| 0 | version, 1; 0 means no trace |
| 4 | capacity, 256 samples (about 2 s at 120 Hz) |
| 8 | head: samples written in all; sample `i` sits in slot `i % 256`, written before the head moves past it |
| 12 | effect kinds the game made, bit `n` for `GUID_ConstantForce + n` (7 spring, 8 damper, 9 inertia, 10 friction), bit 31 any other |
| 16 | effects created, parameter writes on effects that are neither the constant force nor a condition, the constant force's own gain |
| 64 | spring, damper, inertia, friction (32 bytes each): offset, positive and negative coefficient, saturations, dead band, gain, writes seen |
| 192 | samples (24 bytes each): time (f64, performance counter seconds), wheel angle the servo used (f32), the game's force, what went to the wheel, the servo's part (i16 each), steering frame count (u16), flags (servo on 1, rumble 2, live command block 4, driving 8) |

The plugin only reads the game's effects (their kind on creation and enumeration, condition parameters on every
write); it changes none but its constant-force path. The game log names the kinds once (`ffb: game effects: ...`)
and each condition's first parameters. A reader copies the block between two reads of the head and drops the
slots that may have been rewritten meanwhile; a head that went back is a restarted plugin.
`tests/lane_departure/test_ffb_trace.py` compares the header and the Python module.

## Safety

- The wave is zero-mean and ramps in and out, so it vibrates the wheel and does not steer it.
  Peak is 1600 of 10000, and the plugin caps any request at 3000 whatever MonoCruise sends.
- The assist steers, so it is capped whatever MonoCruise sends: torque 3000, servo 3000 by default and never more than
  6000 (LA's `la_servo_cap`, Lukas 2026-11-06), both together 6500 of 10000, the servo's damping half the servo cap
  and its force slew 150000 per second (both Lukas 2026-11-06, for fast moves: 1500 and 30000 before; the slow slew
  made the loop ring at cap 5500 in the model), any push ahead within the cap, any cancelled game force 6000 and never more than the
  game's own. MonoCruise lets go when the driver holds the wheel against it, uses a blinker or lets go of
  the wheel (`core/la/README.md`). A hand on the rim always wins.
- A steering channel older than 0.15 s turns the servo off, and with it any cancel.
- A heartbeat that has not moved for 0.25 s, a missing or unversioned block, or a game that
  is not driving all mean silence: the rumble fades out and the torque ramps out at the release
  rate (3000 to 0 in a quarter second). Killing MonoCruise mid-push ends it in about half a second.
- Every hook body runs under SEH. The first fault sets `dead`: hooks become pass-through,
  the slots are restored from the next telemetry frame and a line goes to the game log.
- Known limit: DirectInput holds the last force it was given. If the game is paused in the
  middle of a rumble or a push, that last sample (rumble and torque at most 3000 of 10000 each)
  stays on the wheel until the game sends its next force.
- TruckersMP: moderation answered that the risk is the user's responsibility, so the rumble
  runs on TMP too. Not yet tested in game there. Moderation gave the go-ahead for steering too
  (2026-11-02): the lane keeping assist runs on TMP (`lka_tmp_enabled`, on by default). LA, the test
  prototype, stays off there (`la_tmp_enabled`).

## Build

From a VS 2022 x64 developer prompt, in this folder:

```
cl /nologo /std:c11 /W3 ffb_signal_test.c && ffb_signal_test.exe
cl /nologo /std:c11 /O2 /W3 /MT /LD ffb.c /link /DEF:ffb.def /Brepro dinput8.lib dxguid.lib
```

The second command writes `ffb.dll`; rename it `monocruise_ffb.dll` and copy it to the game's
`bin/win_x64/plugins`. Imports are `DINPUT8.dll` and `KERNEL32.dll` only. The tools expect a
local copy in `plugins/ffb/build/` (ignored by git). To ship a change, copy it to
`core/sdk_installer/own/monocruise_ffb.dll` and put its git-blob SHA in `core/sdk_installer/own.py`;
`tests/test_sdk_own_plugins.py` fails until they match. `/Brepro` makes a rebuild of the same source
byte-identical.

## Try it

1. Pick Warn under Lane departure in the settings (it installs the DLL) or install it by hand, start
   the game, drive for a few seconds. The game log should say `ffb: constant force hooked`.
2. `python tools/ldw_rumble_test.py` rumbles the wheel for 2 s (the game must be driving,
   not paused). MonoCruise may stay open.
3. With MonoCruise closed, set `"ldw_enabled": true` in config.json, start it and drift
   slowly toward a lane line at 60 km/h or more on a straight road. Add
   `"ldw_test_mode": true` to make any crossing warn whatever the steering looks like.
4. `python tools/la_torque_test.py` holds a torque one way, then the other, with no lane data.
   The LA itself is `"la_enabled": true` (`core/la/README.md`, Testing); the
   game log says `ffb: steering servo on` while it steers.
