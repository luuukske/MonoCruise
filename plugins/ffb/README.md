# MonoCruise FFB plugin (prototype)

An ETS2 / ATS SDK plugin that adds a short zero-mean rumble to the game's own
constant-force effect while MonoCruise asks for one (`core/lane_departure/README.md`).
It is the probe in `plugins/ffb_probe` grown a command channel. The steering axis is never
written and with no request the game's force passes through untouched. It is not part of
the SDK installer: copy the DLL by hand, remove the probe DLL if it is installed (two
plugins cannot both patch the effect), and delete the DLL to remove it.

## How it works

`scs_telemetry_init` opens DirectInput, finds the force-feedback device the game uses and
patches the `SetParameters` slot of its constant-force effect. Each call it reads
`Local\MonoCruiseFfbCmd` and, if a rumble is requested, adds a 12 Hz sine under a slow
envelope (0.25 s in, 0.30 s out) to the game's magnitude, clamped to plus or minus 10000.
The patch is undone at shutdown.

## Command block (`ffb_cmd.h`, 32 bytes, written by `core/lane_departure/command.py`)

| offset | field | writer |
|---|---|---|
| 0 | version, 1; 0 means no plugin | plugin |
| 4 | heartbeat, +1 per write, never 0 | MonoCruise |
| 8 | command (0 none, 1 rumble), side, intensity (u16, peak out of 10000) | MonoCruise |
| 12 | event counter | MonoCruise |
| 16 | torque (i16), reserved for a lane keeping aid, ignored | none |
| 20 | status bits: hooked, force seen, rumbling, dead | plugin |
| 24 | count of constant-force updates seen | plugin |

Change `ffb_cmd.h` and the Python module together; `tests/lane_departure/test_command.py`
compares them.

## Safety

- The wave is zero-mean and ramps in and out, so it vibrates the wheel and does not steer it.
  Peak is 1600 of 10000, and the plugin caps any request at 3000 whatever MonoCruise sends.
- A heartbeat that has not moved for 0.25 s, a missing or unversioned block, or a game that
  is not driving all mean silence. Killing MonoCruise mid-rumble ends it in about half a second.
- Every hook body runs under SEH. The first fault sets `dead`: hooks become pass-through,
  the slots are restored from the next telemetry frame and a line goes to the game log.
- Known limit: DirectInput holds the last force it was given. If the game is paused in the
  middle of a rumble (at most 2 s), that last sample, at most 3000 of 10000, stays on the wheel
  until the game sends its next force.
- Unknown: whether TruckersMP's client tolerates patching DirectInput function pointers. Use
  singleplayer until that is answered.

## Build

From a VS 2022 x64 developer prompt, in this folder:

```
cl /nologo /std:c11 /W3 ffb_signal_test.c && ffb_signal_test.exe
cl /nologo /std:c11 /O2 /W3 /MT /LD ffb.c /link /DEF:ffb.def /Brepro dinput8.lib dxguid.lib
```

The second command writes `ffb.dll`; rename it `monocruise_ffb.dll` and copy it to the game's
`bin/win_x64/plugins`. Imports are `DINPUT8.dll` and `KERNEL32.dll` only.

## Try it

1. Install the DLL, start the game, drive for a few seconds. The game log should say
   `ffb: constant force hooked`.
2. `python tools/ldw_rumble_test.py` rumbles the wheel for 2 s (the game must be driving,
   not paused). MonoCruise may stay open.
3. With MonoCruise closed, set `"ldw_enabled": true` in config.json, start it and drift
   slowly toward a lane line at 60 km/h or more on a straight road. Add
   `"ldw_test_mode": true` to make any crossing warn whatever the steering looks like.
