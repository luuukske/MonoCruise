# Road layout

First slice of the map reader in `docs/ROAD_LAYOUT_PLAN.md`. It loads road
centerlines from the game's 4 km sectors and can check them against a truck
pose. Nothing here is wired into ACC or AEB. Until it is, those systems behave
as they do today.

Parsed geometry is not written anywhere and is not shipped.

## Where the sectors are

The map is a grid of 4000 m squares on telemetry X and Z. The square is
`floor(coord / 4000)`. The file inside a HashFS `.scs` archive is

```
map/<name>/sec+0003-0001.base
```

`<name>` is the `.mbd` under `map/` (`europe` for ETS2, `usa` for ATS). The
sign is part of the four-digit field, so column 0 row -2 is `sec+0000-0002.base`.

`base_map.scs` holds the base map. A DLC archive that contains the same path
replaces that sector. Archives are opened in filename order and the last copy
wins. Map mods in the user folder are not consulted yet.

Paths are hashed with CityHash64 from CityHash 1.0.3. The 1.1 hash that current
PyPI builds ship does not match these archives. The empty string is the one
value both revisions share.

## What a centerline is

A `.base` file is binary, template version 907 on the current game. A road item
is two node ids plus a road-look token. Node positions are integers in 1/256 m.
X and Z are the ground plane (east and south), Y is up, the same frame as
`coordinateX/Y/Z`.

The centerline is a cubic Hermite spline between the two nodes. The tangent at
each node is that node's ahead direction times the XZ distance between them.
Ahead is local -Z of the node quaternion:

```
yaw = atan2(-qy, qw) * 2 - pi/2
ahead = (cos yaw, sin yaw)
```

On one 3x3 window that ahead vector agrees with the chord at both ends (mean
dot 0.96). Y along the spline is linear between the node heights. Lane lines
and the asphalt edge are this spline offset by the road-look, which this slice
does not read. Prefab junctions live in `.ppd` and are not in these polylines.

A road can reference a node stored in the neighboring sector, so a pose check
loads the truck's square and the ring around it (3x3 at radius 1).

## Checked against clip poses

`tools/road_layout_check.py` reads the newest AEB clips, takes the mid frame,
and measures XZ distance to the nearest centerline. On this machine that is
`europe` from three archives (base map, Going East, Scandinavia), 40 clips:

| | |
|---|---|
| within 4 m | 21 / 40 |
| within 6 m | 27 / 40 |
| xz, all clips, p50 / p90 / max | 3.6 / 25 / 147 m |
| xz, the ones within 6 m, p50 / p90 | 2.8 / 4.7 m |
| height error \|dy\| p50 / p90 / max | 0.08 / 0.39 / 1.21 m |

The close hits sit about a lane off the spline, which is what a truck in the
right-hand lane looks like when the spline is the carriageway center. Height
tracks the road surface. The far misses are poses with no road item nearby
(a junction, a yard, or off the asphalt). A node can still be close there,
because the node table also holds prefab origins and props. Those curves are
the prefab step, not this one.

```
python tools/road_layout_check.py
python tools/road_layout_check.py --samples 40
python tools/road_layout_check.py --x 3296 --y 5 --z 19626
```
