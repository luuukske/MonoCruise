# Road layout

The map reader. It loads lane centerlines from the game's own 4 km sectors,
including the driving lines through junctions and roundabouts, follows ego's
lane forward, and builds the predicted path ACC and AEB use (ego and other
vehicles). `core/map_thread/` runs it live; `core/map_thread/README.md` has the
worker and what consumers may and may not do with it.

The map is a prior, never truth. Every consumer falls back to today's
behaviour when there is no map, when the map is distrusted, or when it is
switched off (`Settings.map_data_enabled`).

Parsed geometry is not written anywhere, is not shipped, and never goes into a
clip. Replay tools rebuild it from the local install at view time.

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
wins. Map mods in the user folder are not consulted. TruckersMP's `.mp` files are
layered on top while connected to a server (HashFS v1 or zip, `tmp_mods.py`,
core/map_thread/README.md).

Paths are hashed with CityHash64 from CityHash 1.0.3. The 1.1 hash that current
PyPI builds ship does not match these archives. The empty string is the one
value both revisions share.

## What a centerline is

A `.base` file is binary, template version 907 on the current game (TruckersMP mod
sectors are older, see the module docstring of `sector.py`). A road item
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

The tangent at each node is that node's ahead direction times the road's
stored length (`MapRoad.length`), or the XZ chord when that length is under
0.99x or over 3x the chord. Ahead is local -Z of the node quaternion:

```
yaw = atan2(-qy, qw) * 2 - pi/2
ahead = (cos yaw, sin yaw)
```

On one 3x3 window that ahead vector agrees with the chord at both ends (mean
dot 0.96). Y along a road lane is linear between the node heights.

### The tangent is the stored length, not the chord

SCS stores the length of a spline whose tangent magnitude is that length. A
chord-length tangent is shorter and cuts every long bend inwards, by up to a
lane mid-span. AI traffic drives the nav lanes, so AI trails are the arbiter:
`tools/map_lane_audit.py`, 3929 clips, about 470k AI samples, mid-span of
curved roads, |offset| from our lane line:

| bend | chord tangent p50 / p99 | stored length p50 / p99 |
|---|---|---|
| 60-120 deg, chord 150-400 m | 1.58 / 3.40 m | 0.06 / 0.67 m |
| 60-120 deg, chord 50-150 m | 1.00 / 3.15 m | 0.22 / 1.99 m |
| 30-60 deg, chord 150-400 m | 0.30 / 2.01 m | 0.10 / 1.92 m |

So the "drifts a lane off on long curves" artifact was our reconstruction.
Prefab curves keep the chord: a circle-matching tangent changed nothing for
AI on them (p50 0.097 m either way). `tests/road_layout/test_centerline.py`
pins the stored-length rule.

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

## Ego's route and where it stops being certain (`ego_route.py`)

`EgoRouter` keeps ego's lane across frames: the score above, plus a hold
margin so the lane only changes when another scores clearly better, a height
gate so a bridge over the road is never ego's lane, and a reset after a pose
jump. The route follows that lane up to 250 m.

Wherever another lane leaves a span tip (a junction, an exit, a temporary
divider) the route records how far that branch drifts away by each distance
(`divergence`). Lanes ego could equally be on right now (overlapping junction
curves that fan out later) count the same way, compared at ego's held offset,
so two parallel lanes cost nothing. Consumers stop trusting the route where
branches part; they never guess which branch the driver takes.

`extend_lane` returns a whole joined line in stored order, which can run
backwards; the route orients each span from the previous tip. Without that an
S-bend route folded back on itself.

## The predicted path (`map_path.py`, `path_context.py`)

Ego frame, x forward, y right. The path leaves ego on the consumer's own arc
(its heading and its curvature: AEB's steer-led `EgoPathModel`, ACC's blend)
and bends onto the lane, at ego's current offset from it, over
`blend_s` (2.5 s) of travel, 15 to 60 m. The smoothstep keeps the start's
heading and curvature exactly those of the arc, so steering always leads.

Trust per distance: full while ego sits within 1.2 m of its lane centre and
8 deg of its heading, zero by 2.4 m or 16 deg (a lane change or a driver off
the lane), and zero once a branch is 1.5 m away from the route.

`tools/map_path_eval.py`, 876 clips, ego's true future position against each
path, lane changes excluded; "out" is more than 1.75 m off (outside the lane):

| | steer arc out | map path p50 | map path out |
|---|---|---|---|
| straight, 50 / 100 m | 22.6 / 61.2 % | 0.28 / 0.42 m | 8.2 / 16.0 % |
| gentle, 50 / 100 m | 45.3 / 80.1 % | 0.40 / 0.51 m | 12.0 / 18.2 % |
| bend, 50 / 100 m | 68.2 / 92.4 % | 0.55 / 0.57 m | 15.0 / 16.9 % |
| sharp, 50 / 100 m | 91.3 / 97.1 % | 0.99 / 0.74 m | 37.0 / 27.1 % |

At 25 m the path is no worse than the arc anywhere (straight 1.4 % vs 2.9 %
out). Trust ranks failure: at 50 m, 6.6 % of paths with trust >= 0.95 were
out, against 41 % below 0.05 (the arc was out 43 % and 56 % in those frames).
The remaining failures are mostly lane changes the evaluator could not flag
and junction turns, which is what the branch trust is for.

`PathContext` places one path in the world for a frame and answers the
queries consumers need: offset from the path, the arc's own lateral, distance
to the nearest lane centreline, and a trust-weighted blend of a lateral.

## Other vehicles (`target_path.py`)

The same router and path, one per tracked vehicle, fed its own positions,
heading and measured curvature. AI traffic follows the nav lanes almost
exactly; TruckersMP players drive the painted road, so they sit further off.
Consumers must weigh them apart, and never replace measured motion with it.

`tools/map_target_eval.py`, 433 clips, lateral error of each vehicle's real
position 1 to 3 s later from its predicted path; "out" is more than 1.75 m:

| | AEB's arc out | lane path p50 | lane path out | out at trust >= 0.9 |
|---|---|---|---|---|
| AI, straight, 3 s | 11.0 % | 0.02 m | 3.6 % | 1.6 % |
| AI, bend, 3 s | 62.2 % | 0.07 m | 4.1 % | 1.2 % |
| TMP, straight, 3 s | 23.5 % | 0.26 m | 6.9 % | 2.9 % |
| TMP, bend, 3 s | 81.2 % | 0.56 m | 13.0 % | 6.3 % |
| TMP, bend, 1 s | 37.6 % | 0.89 m | 20.4 % | 18.1 % |

AEB uses a softer form of this (`road_bend.py`), and only while a vehicle follows
its lane: the path starts on its own curvature and blends into the lane's over
about a second, and a corner it is too fast for is ignored from where it starts.
AEB judges on it only targets the road puts in ego's lane; see core/aeb/README.md,
"Road-bent trajectories".

## Two lanes becoming one

`build_route` marks a merge when a lane ego could equally be on (a sibling) starts
at least 1.5 m off the route and comes within 0.75 m of it inside 150 m:
`merge_s` is the route arc length there, `merge_index` the sibling's entry in
`alternatives`, `merge_offset_m` ego's offset from it. Such a lane is not a branch
(it is left out of the divergence), the ACC map path fades its held offset out by
`merge_s` and takes trust from the nearer of the two offsets, and AEB's road path
fades its offset the same way on either lane. Found live at a two-lane roundabout
exit merging into one lane: ego between the lanes, the router on the right one,
and both paths holding a 2.2 m offset over the centre line.

## Lane travel direction

`LaneLine.travel` is +1 when traffic runs in point order, -1 against it, 0 when
unknown. Right lanes of a road run start to end and left lanes back, except on a
road with left lanes only, which is one-way and runs forward; prefab curves run
start to end. Audited against ego's heading on 500 clips: 97 % of road-lane and
93 % of prefab samples agree (the rest is overtaking and junction curves).
Routing does not use it yet: a join penalty for wrong-way lanes changed 0.5 % of
routes, as many for the worse as the better.

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
The bright line is the lane chosen above. The pink line is the fused map path
(steering near the truck, the lane further out, the truck's offset held).
Map objects out to 400 m are drawn in their own color. Pointing at one
shows the model path and the hex color. Road paint, power lines, low curbs,
forest bands, plant circles, and the ground surface are left out.

```
python tools/road_layout_check.py
python tools/road_layout_check.py --samples 40
python tools/road_layout_view.py
```

Measuring tools (all offline, all local, output gitignored):

| Tool | Question |
|---|---|
| `tools/map_lane_audit.py` | How far do AI, TMP and ego trails sit from our lane lines, per spline variant? |
| `tools/map_path_eval.py` | How well does each ego path predict where the truck went? |
| `tools/map_target_eval.py` | Same for other vehicles, AI and TMP apart. |
| `tools/map_path_render.py` | One frame: lanes, route, fused path, steer arc and the truth. |
| `tools/map_aeb_ab.py` | AEB with and without the map over the corpus. |
| `tools/map_replay.py` | The map snapshot for a clip, for any replay (`install()`). |
