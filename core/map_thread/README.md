# Map worker

`MapThread` (`thread.py`) reads lane geometry from the installed game's own map
files and publishes, per ego pose, a `MapSnapshot`: the lane window around ego
and ego's route along its lane. The geometry, the router and the fused path
live in `core/road_layout/` (read its README first).

## What it does each pass

| State | Work |
|---|---|
| `off` | `Settings.map_data_enabled` is false. Nothing is opened. |
| `waiting` | No game connected, or no ego pose from the radar thread. |
| `opening` | `GameMapOpener` opens archives for about 20 ms per pass (107 on ETS2 with DLCs, about 130 ms in total: the HashFS index is a sorted hash list, entries decode on lookup). The game is the telemetry game id (1 ETS2, 2 ATS). |
| `loading` | Sectors for the first window parse one at a time inside the same budget. |
| `ready` | `EgoRouter.step` on the current window; a new window (ego +-600 m) is built once ego comes within 300 m of the old one's edge. |
| `no_map` | No install found, or its archives have no map. Logged once. |

The loop never blocks for long: one archive or one sector per budget slice, and
building a window from parsed sectors takes about 5 to 30 ms. Measured passes on
the installed ETS2: 178 ms once at start (install lookup), about 100 ms for the
catalog and the densest sector, 0.3 ms per pass once ready. It also routes every
moving vehicle within 120 m and not more than 10 m behind ego, nearest first,
inside a 15 ms budget, and publishes them as `target_routes`: AEB bends the arcs of
targets its road path puts in ego's lane. Status changes log
once at info level; nothing here raises a popup, because the driver cannot act
on it. Logs name sectors and counts, never world coordinates.

## TruckersMP map additions

TruckersMP's launcher keeps its own map changes as `.mp` files next to the game's
(`data/<game>/mods` under its install folder, found through `launcher-options.json`).
Each is HashFS v1 with zlib entries or a plain zip: nothing is encrypted, so they
open like the game's own archives (`core/road_layout/archive.py`, `tmp_mods.py`).

- On while the radar's TruckersMP plugin flag says connected or TruckersMP vehicles
  are in the traffic buffer, and for 60 s after the last sign of either. Off, ego
  and every target see the base map exactly as before. `Settings.map_tmp_enabled`
  (config.json only) turns it off for good, `map_data_enabled` still turns off everything.
- Only the mods the launcher currently lists in its `filecache.dat` count (names
  and hashes, no player data). Event mods stay on disk long after the event:
  58 `.mp` files sat in the ETS2 folder, 20 were listed, and only 2 of those
  (`halloween2026`, `hq_final`) carry map sectors. Applying the leftovers would
  swap in roads the server is not using (`christmas_2025_map` alone drops 2,894 base
  items). A missing or unreadable list means no TruckersMP map at all, never a guess
  by file age. The listed archives layer over the base map, newest file winning.
- A listed sector the parser cannot read reads as no lanes there, never the stale
  base roads. The parser reads the version 907 layout and the older versions TruckersMP
  mods use (896, 898, 900, 901, 903, 905, 906: terrain, sign and curve items differ,
  `core/road_layout/sector.py`); a version nobody checked is refused. Checked on the
  installed mods: 113 of the 116 sectors any mod replaces parse, and for the
  versions 898 to 907, 92 to 98 % of the roads shared with the base game are
  identical (same look, nodes and length). The 3 that fail are ProMods and Silverstone
  sectors in version 896.
- TruckersMP builds some junctions on top of existing roads without removing them:
  the old road stays in the sector 1 to 2 m under the new surface (confirmed in
  free cam at the intersection that prompted this). A lane that lies 0.8 to 3.5 m
  under a lane of an item the mod added, for 30 m and at least 30 % of its length,
  is dropped from the window (`core/road_layout/buried.py`); a truck needs about
  4 m to pass under another lane, so such a lane cannot be driven. Only lanes
  under something the mod added count: on the whole base map the same rule would
  hit 15 of 578,000 lanes, and it is not applied there. The router's own 4.5 m
  height gate stays for real stacked roads.
- The map snapshot's `tmp` flag says which map is serving; the AEB debug window's
  legend shows it ("Map: game + TMP"). The standalone lane map
  (`tools/road_layout_view.py`) keeps its own loader and follows the plugin's
  connection flag the same way. Replays and clip viewers rebuild the map locally
  (`tools/map_replay.py`); a TruckersMP clip gets the additions the launcher lists
  today, so a clip from before those mods were current can differ from what ran live.

## Map mods

ProMods and other map mods are the same sector files, shipped as `.scs`, `.zip`,
or a plain folder. The game mounts only the packages enabled for this session.
TruckersMP does that itself on a ProMods server and leaves the single-player
mod list unused, so the list in `profile.sii` is not the source. `game.log.txt`
is: each `mounted to mod pool` line, in order, after the latest `New profile
selected`. The file stays on disk after the game exits, which is what clip
review uses.

- Only a path under that user directory's `mod` folder, or the Steam workshop
  folder for ETS2 or ATS, is opened. Anything else in the log is ignored. The
  path is not logged.
- The layer is built when one mounted package contains the map. The other
  mounted packages are read with it, so a definition package beside the map
  still supplies road looks and prefabs. A package with no map in the set is
  left out.
- A package is read like a game archive. A directory package is read in place.
  A `base/` directory inside the package is used when the map is not at the
  package root (the game always mounts that part). `dlc_*` parts are not: the
  game mounts those only for an owned DLC, and this reader does not guess which.
- The packages layer under TruckersMP's `.mp` files. That keeps a listed
  TruckersMP fix on top. A ProMods-server log was not available to confirm the
  game mounts them in that order.
- A sector the package replaces and the parser cannot read has no lanes there.
  The buried-lane drop stays on the TruckersMP layer only. It was measured on
  TruckersMP junctions, not on map mods.
- Clip replay (`tools/map_replay.py`) uses these packages only when the stock
  map, including TruckersMP's additions for a TruckersMP clip, does not follow
  the driven path and the modded map does. A clip that already fits the stock
  map is left on it, so today's log does not redraw an older drive.

## What consumers may do

- Read `map_snapshot()` (fresh `ready` snapshot or None). Never open archives,
  never parse, never mutate the snapshot. A missing worker, a stale snapshot or
  `None` is "no map", and the consumer's output must then be exactly what it was
  before the map existed.
- Build their own `PathContext` on their own frame and their own curvature: the
  route is in world coordinates and stays valid as ego moves along it.
- AEB: only for ego's road path and the targets it puts in ego's lane
  (core/aeb/README.md, "Road-bent trajectories"). Never to drop a target or move
  a decision AEB makes without the map, never the FF, disarm, holds or latches.
- ACC: laterals for scoring and the corridor, nested over the road model and
  the arc. Never the gap law, never the geometric lead failsafe.

## What it must not do

- Persist anything about the map or where the user drove, or send it anywhere.
- Put map data into clips (it would widen the consented data). Replays rebuild
  the map from the local install (`tools/map_replay.py`).
- Read mod or TruckersMP archives it cannot open. TMP's `.mp` files are never
  decrypted.

Tests: `tests/road_layout/test_map_thread.py` (states with a stub map).
