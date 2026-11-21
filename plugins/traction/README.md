# MonoCruise traction probe (experimental)

An ETS2 / ATS SDK telemetry plugin (`monocruise_traction.dll`) that publishes what the game knows about grip
under each wheel: the surface (substance), the wheel's slip at frame rate and the game's own road wetness,
which the SDK does not expose. It only reads: no game memory is written, no input is sent. Not shipped and
not installed by the SDK installer; copy it to `bin/win_x64/plugins` by hand and delete it to remove it.
Meant to be merged later with the FFB plugin and the TruckersMP plugin into one MonoCruise plugin (see Merging).

`python tools/traction_probe.py` shows it live, `--csv drive.csv` logs every wheel at 20 Hz.

## What the game models (1.62.3.3s)

- **Road wetness** is one global value, `rain_wetness`, kept by the weather system. Rain raises it toward the
  sun profile's `rain_max_wetness` (about 0.55 to 0.70 in the profiles seen) at the climate's `wetting_factor`
  (0.1) and dry weather lowers it at `drying_factor` (0.01), both per game minute (`def/climate.sii`). It is
  saved as `stored_rain_wetness`. A forced change (`g_set_weather 1 f`) sets it to the profile maximum at once,
  `g_set_weather 0 f` sets it to 0; an interpolated change (`i`) moves it gradually.
- **Tyres** carry a `wet_grip` (and a `grip_factor`, unused by the stock defs): 0.9 for the base game's EU label
  D tyres, 1.0 for C, 1.05 / 1.15 for B (Goodyear, Michelin, Krone DLC tyres). So rain is meant to cut grip,
  less on better tyres. **How much is not known yet**: the formula is in code, not data (see Open).
- **Surface** per wheel comes from the SDK (`*.wheel.substance`): `road`, `road_snow`, `snow`, `ice`, `dirt`,
  `road_dirt`, `grass`, `gravel`, `concrete`, `rumble_stripe` and others. Prefab ground (dealer lots, yards,
  likely many junctions) reads `static` whatever is painted on it: at a dealer with the right wheels on grass,
  every truck wheel read `static`; the trailer's rear axle, on map terrain, read `road` left and `grass` right.
  Per-substance friction is in `game_substance_u` (`friction`, static and dynamic), not found in the def files.
- **Snow** as weather exists (`snow_intensity`, snow particles, `interior_glass_config_snow.sii`). Whether
  falling snow changes grip beyond `rain_wetness` is untested. Snow and ice on the ground are map substances.

## Per-wheel traction: what is possible

| input | per wheel? | source |
|---|---|---|
| surface (substance) | yes, but blind on prefab ground (`static`) | SDK |
| slip (wheel speed against ground speed at that wheel) | yes, at frame rate | SDK, computed here |
| suspension deflection (a load proxy) | yes | SDK |
| road wetness | no, one value for the world | game memory, this plugin |
| tyre `wet_grip` | per axle (front, drive, trailer) | the fitted tyre, in the save, not the SDK |
| friction per substance and the wet formula | | game code, not read yet |

So a per-wheel guess is `surface(wheel) x wet(wetness, tyre)`, with wetness global. Measured slip only says
something near the limit (spin on pull-away, ABS on a hard stop); at ordinary loads the slip of a brush-model
tyre barely depends on the friction limit.

## How the wetness is found

At start the plugin scans the game's `.text` for the save code that copies the live weather into the save
object (`mov rcx,[rdi+0x998]` ... `mov eax,[rcx+0x45e8]` ... `mov [rsi+0x5e8],eax`) and takes every offset
from that instruction sequence, plus the global from the `mov rdi,[rip+...]` that starts the function. The
signature must match exactly once or the weather stays off (status bit 1 clear). Reads go through `__try`, are
range checked (weather index under 64, wetness 0 to 2) and stop after repeated faults. 1.62.3.3s resolved to:
global RVA 0x3714FB0, world +0x998, current weather +0x3E78, target weather +0x3E7C, change timer +0x4578,
wetness +0x45E8. A game update can break the signature; the plugin then keeps publishing everything else.

## Block (`traction_shm.h`, `Local\MonoCruiseTraction`, 2172 bytes)

Header: version (1; 0 no writer), seq (odd while writing, even when consistent), status bits (1 weather found,
2 weather read this frame, 4 substances, 8 truck config, 16 trailer config, 32 paused, 64 trailer channels),
frames, current and target weather, wetness, weather timer, speed, truck and trailer local velocity and
angular velocity, wheel counts, substance count, the resolved RVA and offsets, game version. Then 32 substance
names of 16 bytes and 32 wheels of 48 bytes (truck 0 to 15, trailer 16 to 31): flags (on ground, powered,
steerable, simulated, configured, trailer), substance index, radius, angular speed (rot/s), rim speed, ground
speed along the wheel, slip ((rim - ground) / |ground|, + spin, - lock, 0 under 1 m/s), slip peak held 1 s,
deflection, position, steering. Change the header and `tools/traction_probe.py` together.

## Build

From a VS 2022 x64 developer prompt, in this folder:

```
cl /nologo /std:c11 /O2 /W4 /MT /LD traction.c /Fobuild\ /Febuild\monocruise_traction.dll /link /DEF:traction.def /Brepro kernel32.lib
```

Imports are `KERNEL32.dll` only. The game log shows `traction: weather at rva ...` and `traction: ready`.

## Merging

The plugin keeps its own block so the FFB and TruckersMP layouts never move. A merged DLL exports
`scs_telemetry_init` once and calls this file's registration from it, and also exports `truckersmp_init` for
the TruckersMP part; the SDK events and channels here do not collide with the FFB plugin's (`truck.input.steering`
and the frame events can be shared). Keep the signature scan behind its own guard there too.

## Measured wet grip (2026-11-18)

Twelve full stops by Lukas, same road, Volvo FH6 6x4 with a curtain trailer, brake intensity 1.39 (115 %),
about 70 km/h unless noted. Every stop locked or reversed the rear and trailer wheels 20 to 31 % of the time, so
all are grip limited. Plateau decel is the linear fit between 90 % and 15 % of the start speed.

| wetness | stops | plateau decel (m/s^2) | of dry |
|---|---|---|---|
| 0 | 4 (one from 42 km/h) | 11.84 to 13.32, mean 12.29 | 100 % |
| 0.20 to 0.33 (drying, rain fading) | 3 | 10.71 to 10.92 | 88 % |
| 0.39 to 0.42 (forced rain) | 5 | 9.69 to 10.77 | 83 % |

Fit: grip = dry x (1 - k x wetness), k = 0.42 over all stops (0.35 without the hardest dry stop), 2 to 4 % rms.
The 42 km/h dry stop matched the 70 km/h ones, so no speed term was visible. Wetness above 0.42 was not driven:
forced rain reaches the time-of-day profile maximum (0.42 in the evening, 0.70 seen in the afternoon), and 1.0
extrapolates to 58 to 65 % of dry, untested. One truck and one tyre set (its `wet_grip` unknown).

Wheel speeds chatter at lock (rim speed swinging to -50 m/s within a frame or two), so a single slip sample means
nothing past the limit; count locked or reversed frames over a window instead.

## Used by MonoCruise

`core/traction/reader.py` maps the whole block (a smaller block created first would leave the plugin unable to
map its own), reads the wetness while status bit 2 is set and `seq` keeps moving (stale after 2 s), and gives
`wet_grip_factor` = 1 - 0.42 x wetness. The sending thread uses it for AEB's planning capacity and to keep
every brake capacity learner dry (AEB's and the mapper's `brake_scale`) (`core/sending_thread/README.md`, Full-pedal credit). Without the plugin every
factor is 1.0. The SDK installer does not ship the plugin yet.

## Open

- More wetness levels (afternoon rain for 0.7), other tyres and loads, and snow mods (they may change substances).
- Per-substance friction values, and whether `static` hides real surfaces that grip differently.
- Snowfall's effect on grip.
- TruckersMP: the same exe, so the signature should hold; untested there.
