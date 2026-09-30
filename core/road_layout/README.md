# Road layout

First slice of the map reader in `docs/ROAD_LAYOUT_PLAN.md`. It loads lane
centerlines from the game's 4 km sectors, including the driving lines through
junctions and roundabouts, and can check them against a truck pose. Nothing
here is wired into ACC or AEB. Until it is, those systems behave as they do
today.

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
is two node ids plus a road-look token. A prefab item is a model token, the
node that anchors it, and which of the prefab's own nodes that anchor is.
Node positions are integers in 1/256 m.
X and Z are the ground plane (east and south), Y is up, the same frame as
`coordinateX/Y/Z`.

The centerline of the carriageway is a cubic Hermite spline between the two
nodes. Each lane is that spline shifted sideways to the middle of the lane.
The look lists the lanes on the left and the right. A lane is 4.5 m wide, and
`road_offset` is how far each side's inner edge sits from the spline. A one-way
road has lanes on only one side. An even count is centered on the spline. An
odd count sits half a lane to the left, so the spline is the right edge of the
middle lane. A look with only left lanes is the mirror of that.
A per-lane shift from the look is added when the template asks for one.
Positive is to the right of travel, from the start node toward the end node.

Junctions, roundabouts and merges are not road items. Each driving line
through them is a nav curve in the prefab's `.ppd`, already one lane. The
sector names the world node the prefab's origin sits on, and the curve is
rotated by the difference of those two headings.

The tangent at each node is that node's ahead direction times the XZ chord
length. Ahead is local -Z of the node quaternion:

```
yaw = atan2(-qy, qw) * 2 - pi/2
ahead = (cos yaw, sin yaw)
```

On one 3x3 window that ahead vector agrees with the chord at both ends (mean
dot 0.96). Y along a road lane is linear between the node heights.

A road can reference a node stored in the neighboring sector, so a pose check
loads the truck's square and the ring around it (3x3 at radius 1).

## Checked against poses

On a live stop in a junction the carriageway center was 22.8 m away and the
nearest prefab lane was 0.3 m away. On recent clips, a pose that sat 2 to 4 m
off the carriageway center (the right-hand lane) lands within about 2 m of a
lane center, often under 1 m. A pose 16 m off every road center, the junction
gap, landed 0.1 m from a prefab curve.

## Which lane ego is in

The live window keeps one lane for the truck. How far the truck sits from a
centerline is the main term. Direction still counts: a sharp branch does not
win by passing only a little closer, and a branch that still shares a tangent
is judged by where it goes next. A few metres of offset beats a modest angle.
Parallel lanes, which head the same way, take the nearer center.

The bright line follows that lane forward until just before the sixth prefab,
or 200 m, whichever comes first. A lane that ends beside the one it joins
continues on that lane. Curves that belong to the same prefab count as one.

## Checked against clip poses

`tools/road_layout_check.py` uses the mid frame of a saved clip, or the
coordinates you pass. It prints the distance to the carriageway center and to
the nearest lane, whether that lane is a road or a prefab.

`tools/road_layout_view.py` is a separate window for driving. North is up and
the truck stays centred. Road looks and the prefab index load once. A sector
is parsed when the view can reach it (about 1.4 km, so a road's far node is
included) and dropped once the truck is a full sector past that. The picture
updates from telemetry the whole time. A longer road is drawn with more points
than a short one, about one every 10 m. It is not part of the ACC debug window.
The bright line is the lane chosen above. The pink line is the road model
fitted from that lane: its direction, with the truck's sideways offset held.
Map objects out to 400 m are drawn in their own color. Pointing at one
shows the model path and the hex color. Road paint, power lines, low curbs,
forest bands, plant circles, and the ground surface are left out.

```
python tools/road_layout_check.py
python tools/road_layout_check.py --samples 40
python tools/road_layout_view.py
```
