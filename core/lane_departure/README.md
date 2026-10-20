# Lane departure rumble (prototype)

Rumbles the wheel's force feedback when the truck drifts out of its lane without the
driver meaning to. Stage 1 is the warning only. A lane keeping aid would reuse the same
lane data and the same plugin, and does not exist yet. Off by default (`Settings.ldw_enabled`,
config.json only, no UI).

```
radar_thread + telemetry_thread + map_thread
        |  LaneDepartureThread (20 Hz)  ->  detector.py
        v
Local\MonoCruiseFfbCmd  (command.py)  ->  plugins/ffb  ->  game's constant-force effect
```

`detector.py` is pure logic. `thread.py` reads siblings (a missing one reads as no data),
`command.py` writes the shared-memory block, and `plugins/ffb/README.md` covers the
game side. Nothing here touches ACC, AEB, CC or the pedals, and the map stays a prior:
no map, a stale snapshot or `map_data_enabled` off means silence.

## What is measured

The lane is the centre polyline of `MapSnapshot.route` plus or minus half a lane
(`half_lane_m`, 2.25 m: the map's own 4.5 m lane spacing in `core/road_layout/looks.py`; the
map has no paint, shoulder or curb data). Ego is the rectangle of `RadarData.ego_geometry`
(`core/radar/README.md` section 17): its width is the SDK's outermost wheel plus the tyre, and
its ends are the bumper and the rear end. The gap per side is from the outer corner, front or
rear whichever is farther out, to that lane edge. The rear corner keeps the rumble going while
the cab swings back. Without readable wheels the reference rig is used, and the thread logs which
body it has once. The trailer is not included.

There are two triggers on the same lane data, and they differ in what they sense.

- **Line warning (full rumble).** Position only: how far the body is from the lane centre.
  It starts when the corner reaches the warning line, `warn_overlap_m` past the edge (-0.1, so
  0.1 m before the tyre touches the line) plus the bend allowance on the inside of a bend
  (below). Lateral speed and heading play no part, so a fast and a slow drift warn at the same
  place.
- **Approach cue (softer rumble, 60 % strength).** The relative speed: "you are getting close to
  the lane". It starts when the time to reach the warning line is 0.8 s or less, from the
  corner's lateral speed and a damped lateral acceleration (half the regression of the heading
  over 0.6 s, capped at 1.5 m/s^2), and the truck is drifting toward the edge at 0.4 m/s or more.
  It only exists near the edge: the corner must be within 0.4 m of the warning line, so a fast
  move in the middle of the lane is ignored. It escalates to the full rumble when the line is
  reached. A slow wander (under 0.4 m/s) gets no
  cue, only the line warning. On the inside of a bend (curvature toward that edge of 1/1000 or
  more) the cue also needs the truck pointing at the edge by 3 degrees or more, because cutting
  a corner steers that way; the outside edge, where a truck with a still wheel runs out, keeps
  the plain rule. Sharp bends stay silent as a whole (`bend_sharp`), and the heading is read from
  the smoothed lane tangent so a coarse polyline cannot fake a closing speed.

Both are muted while the truck moves toward the lane centre at 0.2 m/s or more (`returning`),
so a driver who is already steering back is not rumbled at, even if the corner is still over the
warning line.

A rumble always lasts at least 1 s once started, and at most 4 s per level. After the minimum it
ends when the line warning's corner is back inside the warning line, the truck moves toward the
centre, a blinker or deliberate steering appears, or the corner is 1.5 m past the edge (the truck
is out for good); the cue ends when it no longer closes on the line. Each end is held 0.3 s. A
gate in the table below (AEB, pause, stale data, no route, speed) ends it at once, even inside
the minimum.

Re-arming never needs a recentre. After a rumble the side is disarmed until the corner has been
0.15 m inside the warning line for 0.3 s and 1 s has passed. The cue and the line warning are
armed separately: a cue that ends never disarms the line warning, so a truck that hovers just
short of the line and then crosses it is still warned. A truck that keeps straddling a line gets
one warning, not a train of them. At most 12 episodes per minute.

A truck centre more than three half lanes from the route is `off_route`.

## When it stays silent

| Reason | Meaning |
|---|---|
| `no_map`, `route_kind`, `short_route`, `branch`, `merge`, `off_route` | no route, a lane that is neither a road lane nor a straight prefab lane (below), route ends or forks within `max(40 m, 2 s)`, truck far off the route |
| `speed`, `reverse`, `paused`, `stale`, `aeb`, `brake`, `tmp_map` | below 60 km/h (disarm 50), reversing, game paused, radar older than 0.25 s, AEB cue, brake 0.7 or more (hard braking only), TruckersMP map overlay |
| `lane_settle` | first 1 s on a lane after a lane switch of the router, a junction, no map or stale data (a fork or fast bend ahead does not restart it) |
| `bend_sharp` | lane curvature of 1/90 m or more |
| `blinker` | that side's lamp (or hazards) is on, or went off under 2 s ago |
| `deliberate` | the wheel moved toward that edge by 0.06 of lock or more over the last 2 s, or at 0.3 per second or more over 0.3 s; held 1 s |
| `intent` | the heading toward that edge is 1.2 degrees or more and the wheel moved toward it (0.015 of lock net over 2 s, or 0.06 per second); held 2 s |
| `evade` | a hard wheel move in either direction (0.15 of lock net over 2 s, or 0.6 per second over 0.3 s) mutes both sides for 2 s: a swerve and its counter-steer |
| `latched` | reached the warning line while blinking or steering deliberately: silent until back inside it |
| `returning` | moving toward the lane centre at 0.2 m/s or more; mutes the line warning (the cue needs a drift toward the edge anyway) |
| `bend_inner` | not a mute: the tyre is over the inside line but within the bend allowance (reported for tuning) |

Every number above is a placeholder in `Tuning`, set by reasoning and not by driving.

## Clear intent to cross is ignored

A lane change without a blinker is told from a drift by whether the wheel made the heading: the
truck points at the edge by 1.2 degrees or more (about 0.5 m/s at motorway speed) and the wheel
moved toward that edge. Gentle steering is enough, because a motorway lane change is gentle.
A drift has a still wheel, or a wheel that only follows the road, so it warns. The heading
condition keeps a slow creep of the wheel with a slow drift (under 1.2 degrees) in the warned
group, and the wheel condition keeps a fast run-out with a still wheel in it. Bend-following
steering is not counted (below), which keeps a truck that follows a curve and slides slightly into
the other lane warned.

## Hard steering is ignored, except for the bend itself

An evasive move is hard steering, so it silences the rumble (`evade`, both sides, 2 s, on top of
the per-side `deliberate`). In a tight bend (lane curvature of 1/500 or more toward a side)
steering that way is the bend, not intent, and is not counted: a driver who turns hard into a
corner and still runs wide, or sleeps through it, is warned for the outside edge. Steering
against the bend still counts. The inside edge of a bend is then decided by position and the bend
allowance alone. Gentler curves get no exemption, so a lane change on a motorway sweep is muted
as on a straight.

## Prefab lanes

A prefab lane (a junction or a road piece built from a prefab) is used like a road lane when it
is straight and does not fork or join: no branch or merge within the look-ahead, and lane
curvature below 1/400 from 10 m behind to `max(30 m, 1.5 s)` ahead. A curved prefab lane (a
junction turn, a roundabout) stays `route_kind`. Prefab lane widths are not known, so the same
half lane is assumed. On the clip corpus about three quarters of the 60 km/h-plus seconds that
were silent for `route_kind` were straight enough to qualify (228 s down to 54 s of 4790 s).

## Intent comes from the wheel

Closing speed says nothing about intent, so it never mutes a side as deliberate (it is a cue
trigger and the `returning` mute, not an intent signal): a truck running out with a still wheel (asleep,
distracted, no reaction in a bend) is the case to warn on, and its lateral speed grows fast.
So only the steering history of the last 2 s mutes a side, and only a clear movement toward
that edge does. Smooth, slow or no steering toward the edge warns. Evasive swerves and lane
changes with a pulse of steering mute, and so does a blinker. On a bend the wheel simply
stays straight when nobody reacts, the lane curves away, and the truck reaches the outside
edge (the oncoming lane on a right-hand bend in right-hand traffic), which has no allowance.

Measured on the clip corpus (600 clips, `userSteer`, 2 s window, positive toward the blinker
side): 30 blinker lane changes at 60 km/h or more moved the wheel by a median 0.022 of lock
(p75 0.039, p90 0.109) with a median peak rate of 0.056 per second. Ordinary driving moved it
by 0.011 (p90 0.056). Highway lane changes are gentle, so steering size separates intent only
roughly: most lane changes made without a blinker will rumble. The first thresholds (0.03 and
0.08 per second) read as deliberate in 23 % (net) and about 30 % (rate) of ordinary driving
moments, either direction, and silently ate warnings; 0.06 and 0.3 per second cut that to about
10 % and under 10 %; an evasive swerve should still clear them (not measured). The log line of each warning
carries `steer_net` and `steer_peak` so they can be tuned on real drives.

## Curves

Cutting a corner puts the tyre over the inside line; running out of a bend goes to the
outside. So only the edge a bend turns toward gets extra room on top of the warning line,
`bend_gain_m2 * kappa` (80 m^2) capped at `bend_cap_m` (0.3 m): 0.08 m at R 1000, 0.27 m at
R 300, 0.3 m from R 267 down. With the warning line 0.1 m inside the edge, the tyre may reach
at most 0.2 m past the inside line. The outside edge and straights get none. The curvature is
the peak toward that edge from 10 m behind ego to `max(30 m, 1.5 s)` ahead, so turning in
before the bend already counts. The cap is what keeps a real departure to the inside warned.

Measured on the clip corpus (58 usable clips at 60 km/h or more, no blinker, no clear
steering toward the edge, front-axle tyre): the p90 overlap past the inside line was -0.08 m on
straights and 0.15 to 0.5 m on bends from R 1000 to R 90. So a driver who cuts bends hard
will feel it. Thin data from AEB clips, so the numbers are placeholders like the rest.

Cost: on a left bend (right-hand traffic) the inside edge is the oncoming lane, which gets the
same allowance and no cue. Trailer off-tracking is not modelled.

On straights 8 of 21 clips had the truck's median position more than 0.5 m from the map's lane
centre (3 more than 1 m). Whether that is map error or how those drivers drive is open; a road
where the map line sits off the painted one gives one warning, then the side stays disarmed
until the truck is back inside the warning line.

## Logs

One `info` line per cue or warning with the features (side, gap, allowance, time to line, drift
speed, heading, curvature, speed, steering) and `debug` lines for muted close calls, at most one a second. Every 10 s
an `info` line says where the time went (seconds per silence reason, closest edge gap, closest
warning line), skipped while the truck is only below the speed gate. No coordinates or
sector ids. Plugin state changes (`present`, `hooked`, `dead`) log once each.

## Testing

`python tools/ldw_rumble_test.py` rumbles the wheel with no lane data, to check the plugin
path alone; it may run while MonoCruise is open, because MonoCruise writes its request only
when it changes. `ldw_test_mode` (config.json) makes every close call warn, ignoring
blinker and deliberate steering, so crossing a line by hand proves the whole
chain; the speed, route and AEB gates still apply. Unit tests: `tests/lane_departure/` (synthetic straight and curved lanes, every
gate, the shared-memory layout against `plugins/ffb/ffb_cmd.h`).

## Not done

Side-coded cues, edge class (oncoming lane or not), trailer swept path, UI, installer
integration, ATS validation, TruckersMP (see `plugins/ffb/README.md`), and the lane keeping
aid. The command block reserves a torque field for it that the plugin ignores.
