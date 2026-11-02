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
- the steering servo (layout 4): a PID on the wheel angle, added to the game's force, `kp` on the error,
  `kd` on the rate error (the wheel's speed low-passed at 6 ms, the damping capped at 1500) and `ki` on the
  error over time (no windup while the output is pinned, at most the cap). The target is moved on at the rate
  MonoCruise sends for at most 0.15 s after each new one (`servo_stamp`), so its 20 Hz updates never
  reach the rim as steps. Faded in 0.3 s and out 0.2 s, slope-limited, capped, and halved when it buzzes:
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

## Command block (`ffb_cmd.h`, 96 bytes, written by `core/lane_departure/command.py`)

Steering values are `truck.input.steering` units: -1 to 1 of lock, counterclockwise positive.

| offset | field | writer |
|---|---|---|
| 0 | version, 4 (1 rumble only, 2 added the torque, both 32 bytes; 3 a servo without damping, 64 bytes); 0 means no plugin | plugin |
| 4 | heartbeat, +1 per write, never 0 | MonoCruise |
| 8 | command (0 none, 1 rumble), side, intensity (u16, peak out of 10000) | MonoCruise |
| 12 | event counter | MonoCruise |
| 16 | torque (i16, out of 10000, signed like the game's magnitude) | MonoCruise |
| 18 | game force (i16, the game's magnitude low-passed at 0.1 s) | plugin |
| 20 | status bits: hooked, force seen, rumbling, dead, torque, servo, steering channel, one-sided servo (128) | plugin |
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

MonoCruise still rumbles through a version 1, 2 or 3 plugin (it falls back to the smaller block), and
steers only through version 4.

Change `ffb_cmd.h` and the Python module together; `tests/lane_departure/test_command.py`
compares them.

## Safety

- The wave is zero-mean and ramps in and out, so it vibrates the wheel and does not steer it.
  Peak is 1600 of 10000, and the plugin caps any request at 3000 whatever MonoCruise sends.
- The assist steers, so it is capped whatever MonoCruise sends: torque 3000, servo 3000, both
  together 4000 of 10000, the servo's damping 1500, any cancelled game force 6000 and never more than the
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
