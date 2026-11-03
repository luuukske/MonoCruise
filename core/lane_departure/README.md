# Lane departure rumble and lane keeping assist (prototype)

Rumbles the wheel's force feedback when the truck drifts out of its lane without the
driver meaning to (LDW, `Settings.ldw_enabled`), and, as a separate switch, steers it back with a
short pulse (LKA, `Settings.lka_enabled`, see "Lane keeping assist" below). Both off by default. The
settings panel's Lane departure dropdown (under Emergency Braking, BETA) sets them together: None, Warn (the
rumble) or Steer (the rumble and the pulses); a pick other than None installs the FFB plugin
(`plugins/ffb/README.md`). LA (`core/la/README.md`) reuses the same lane data, the same plugin and
this thread, which owns the command block for all three; each is switched on separately.

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

`LaneDepartureData.watch_reason` (the detector's reason while LDW or LKA watches) and `firing_side` (LDW rumbling, or
the line an LKA pulse steers from) feed the CC panel's lane lines (`ui/cc_panel/README.md`) and the bottom bar's red corners
(`core/sending_thread/README.md`) through `display.py`; display only.

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
  reached. A slow wander (under 0.4 m/s) gets no cue, only the line warning. The cue follows the
  change of curvature, not the curvature itself, over the road from ego to `max(30 m, 3 s)`
  ahead, each zone held 2 s after it clears:
  - **Steady slight bend** (curvature varying by less than 1/1000 across that road, gentler than
    R 400): the cue works as on a straight, both edges.
  - **Curvature changing** (entry, exit, wavy roads; `bend_change`): a drift there is mostly
    positioning and timing, turning in late or unwinding late. The cue is kept for the driver who
    is not paying attention: it is muted while the wheel has moved away from that edge in the
    last 1.5 s (0.008 of lock net, or 0.03 per second), a driver already correcting, and a still
    wheel or one moving toward the edge cues as on a straight.
  - **Tight corner** (R 400 or tighter anywhere from 10 m behind to 3 s ahead; `bend_tight`):
    never a cue, and a cue already running ends. Only the line warning is left.

  Sharp bends stay silent as a whole (`bend_sharp`), and the heading is read from the smoothed
  lane tangent so a coarse polyline cannot fake a closing speed.

Lateral speed is read at the rear axle group (`EgoGeometry.path_origin_m`, `core/radar/README.md`
section 17), the point that rolls along the heading: the heading is compared with the lane tangent
there, and a corner's speed adds the heading rate times its distance ahead of that axle. Read at
the placement origin, 2.1 m ahead of it on the reference rig, a steady bend read as drift toward
the outside of `v * kappa * d` (0.35 m/s at 90 km/h on R 150), which cued trucks holding a line
near the outer marking and muted inside-edge warnings as `returning`. Without readable wheels the
reference rig's axle is used.

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
one warning, not a train of them. There is no per-minute limit (Lukas, 2026-11-02: it was 12; a driver
who is not paying attention feels each rumble).

A truck centre more than three half lanes from the route is `off_route`.

## When it stays silent

| Reason | Meaning |
|---|---|
| `no_map`, `route_kind`, `short_route`, `branch`, `merge`, `off_route` | no route, a lane that is neither a road lane nor a straight prefab lane (below), route ends or forks within `max(40 m, 2 s)`, truck far off the route |
| `speed`, `reverse`, `paused`, `stale`, `aeb`, `brake` | below 60 km/h (disarm 50), reversing, game paused, radar older than 0.25 s, AEB cue, brake 0.7 or more (hard braking only) |
| `lane_settle` | first 1 s on a lane after a lane switch of the router, a junction, no map or stale data (a fork or fast bend ahead does not restart it) |
| `bend_sharp` | lane curvature of 1/90 m or more |
| `bend_change` | approach cue only: lane curvature varies by 1/1000 or more from ego to `max(30 m, 3 s)` ahead (`dkappa` in the log), held 2 s, and the wheel moved away from that edge in the last 1.5 s (`steer_away`); the line warning still applies |
| `bend_tight` | approach cue only: curvature of 1/400 or more from 10 m behind to `max(30 m, 3 s)` ahead, held 2 s; the line warning still applies |
| `blinker` | that side's lamp (or hazards) is on, or went off under 2 s ago; not toward an unsafe lane (Side traffic) |
| `deliberate` | the wheel moved toward that edge by 0.06 of lock or more over the last 2 s, or at 0.3 per second or more over 0.3 s; held 1 s |
| `intent` | the heading toward that edge is 1.2 degrees or more and the wheel moved toward it (0.015 of lock net over 2 s, or 0.06 per second); held 2 s |
| `evade` | a hard wheel move in either direction (0.15 of lock net over 2 s, or 0.6 per second over 0.3 s) mutes both sides for 2 s: a swerve and its counter-steer |
| `latched` | reached the warning line while blinking or steering deliberately: silent until back inside it. `deliberate`, `intent` and `latched` do not mute toward an unsafe lane either (Side traffic) |
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
same allowance, and no cue in a tight corner. Trailer off-tracking is not modelled.

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

## Lane keeping assist

A short steer back when the detector above catches an unintended drift out of the lane, into the next
lane, off the road or into the oncoming lane. It reads the detector and never changes it: the rumble
warns exactly as it does without it (`test_lane_departure_warns_exactly_the_same_with_it_on`), and with
`ldw_enabled` off the detector still runs for it, silently. It steers through LA's core (`core/la/`, the
lane fit, path tracking and servo) in its pulse mode; LA outranks it when both are on. Pure logic in
`lka.py` (when to start and stop) and `edges.py` (what lies past each line), settings and logs in
`lka_runner.py`.

**When it starts.** On the detector's own episode on a side, the approach cue or the line warning, with
the same mutes: blinker on that side, deliberate steering, a wheel that made the heading, a swerve,
`latched`, `returning`, and every gate (speed, AEB, hard braking, stale data, junctions, forks, merges,
sharp bends). The detector mutes its approach cue in bends; there LKA still starts on the same test
(0.8 s to the warning line, drifting at 0.4 m/s or more, within 0.4 m of it) on a straight or a bend's
outer edge: running wide is caught early. On a bend's inner edge it does so only for a run at 0.8 m/s or
more; a slower corner cut keeps to the line warning, as in LDW (on 2026-11-02 a 1.5 m/s run into the
oncoming lane on the inside of an R 90 waited for the line and went 0.5 m over). An approach start, this
test or the detector's own cue, also needs the truck itself drifting toward the line at 0.2 m/s or more by
LA's lane fit (`drift` in the skips): the detector reads its heading off the route polyline, which in fast
bends showed up to 0.8 m/s toward the outer line while the truck was heading home. On 2026-11-02 that was 17
of 47 pulse starts, among them every 0.2 s stutter and the pulse the driver held at the servo's cap at
20:59:55. A start at the line warning stays position only. It needs no swerve in the last 5 s (the
detector's own hold is 2 s, and a pulse must not steer back onto what the driver swerved round), no
override still holding (it used to start a pulse with no force), and LA's calibration: about a minute of bends on
the first drive, then the saved one once the drive confirms the wheel's direction (`core/la/README.md`,
Calibration). Straight prefab lanes count like roads, as in LDW: drive 12 had departures there that warned
and got no pulse.

**What it does.** Authority in over 0.15 s. The target is LA's: bend feedforward, the stop before the
line, and the correction, which in a pulse steers to a line 0.5 m inside the edge it saves instead of
the lane centre. It lets go once the body is 0.15 m back inside the warning line (so the detector re-arms)
and no longer drifting toward it, by the detector and by LA's lane fit (0.05 m/s at most: the detector's
drift alone flickered home in fast bends and let go 0.2 s in), fading out over 0.5 s. It also ends on any
driver mute on that side, an
override (a wheel held short of the target, toward the line), the body 1.5 m past the edge (out for good),
a gate, or after 3 s. It rides
through the detector's silences for a fork, a join, a route end or a junction lane ahead (`branch`,
`merge`, `short_route`, `route_kind`), judging the line from LA's own lane fit and the blinker meanwhile:
on drive 12 those silences ended 11 of 14 pulses mid-correction, one 0.64 s in with the truck 0.23 m over.
Every other silence (no map, stale data, a lane switch, speed, AEB, the brake, a sharp bend) ends it.

**One way only (Lukas, 2026-11-02).** A pulse never pushes the wheel toward the line it saves, not even to
take back its own steering: the plugin's servo is one-sided for it (`servo_only` in the command block,
`plugins/ffb/README.md`), so at the game's FFB rate any push toward the line, the damping's included, is
clipped to nothing and its integral cannot build that way. A driver steering home harder than the pulse
wants feels no push back at all, and the moment the wheel is short of what the drift needs it pushes again,
at once and from where the wheel is. A first try that detected the driver from MonoCruise's 20 Hz view
still pushed back for 0.3 to 1 s. The pulse needs a plugin that reports it can (status bit 128, built on
2026-11-02, after 15:17): with an older one LKA stays off (`plugin`) and says so once.

Because a pulse cannot straighten the wheel it turned, the game's centring and the driver's hands do, and a
truck swinging home can reach the other line. For 2.5 s after a pulse the other line therefore starts its
own pulse on the detector's tests even where the detector reads the turned wheel as the driver's intent
(`deliberate`, `intent`, `latched`), and that pulse is not ended by them; a blinker or a swerve still is. It
too pushes only away from its own line, so the two catch the swing between them.

The pulse's own line gets the same 2.5 s while the truck still drifts out at 0.4 m/s (and 0.2 by the lane
fit): released short of home, the turned wheel comes back toward that line, the detector read it as intent
and latched, or kept the side disarmed after its warning, and nothing fired again. There, and in any pulse,
`deliberate`, `intent` and `latched` count as the driver only once the wheel has turned toward the line past
where it was as the first pulse began (0.01 of lock). In the model, a sleeping driver holding 85 % of an
R 250 on a truck that answers its wheel within 0.05 to 0.1 s ran 64 to 66 m off the road with no pulse and
no rumble; now it stays within 0.4 m of the line.

**What it keeps and drops (Lukas, 2026-11-02).** Kept: the servo's cap (3000 of 10000, a hand always
overpowers it), LA's override (a wheel held still off the target for 0.3 s, or held against the servo
near its cap; yields in 0.4 s) and the hand-back to the driver's push (below), the plugin's own buzz guard (since drive 12 a real ring of the wheel only,
see `plugins/ffb/README.md`), and every detector mute and gate. Dropped
for a pulse, which lasts a second or three: LA's ring guard, weave guard, hands-off check, the 2 s rise
and 1 s re-arm, LA's blinker and swerve mutes (the detector's per-side ones decide) and the lane carry.

**Handing back to the driver's push (Lukas, 2026-11-02 and 03).** The driver's estimated push against a pulse
(`core/la/README.md`, "The driver's push") is read as a share of the pulse's own push, smoothed over 0.15 s,
while the pulse pushes 500 or more and is not turning the wheel home (resting hands' damping and friction resist
only a moving wheel). From 40 % the pulse pushes no harder: its cap is frozen at its push then, and the plugin
clamps the servo's integral to it. From 50 % its authority falls toward what the share leaves (nothing at 80 %),
at the driver yield rate (0.4 s for all of it), and at 98 % it hands back: the core yields and holds off 2 s,
without the rule that waits for a lane centre. Freeze and hand-back are latched for the pulse. Why: the servo's
integral wound the force up to 3000 in about 0.7 s against a held wheel, faster than the push showed (a median
2500 when the driver's share first held 50 %), and the old test, 75 % of the pulse's push for 0.3 s and then a
0.45 s fade, let go 0.9 to 1.5 s into a fight. A slower integral was measured instead and cost a sleeping driver
0.35 m past the line at R 400 (0.03 m now). The cost of this one, by decision: in the model a sleeping driver
whose resting hands hold 90 % of an R 400 is handed back once, the detector then stays disarmed past the line,
and the truck runs 18.7 m wide in 12 s against 0.03 m when the pulse ignored the push
(`test_heavy_resting_hands_in_a_bend_count_as_a_grip`, a ratchet). A free wheel never reads as a push.

**What lies past the line.** At each start the map is asked what lies one lane width past that line,
half a second ahead of the bumper: a parallel lane whose traffic runs ego's way (`lane`), the other way
(`oncoming`), none (`edge`: the road's edge, or a median to the other carriageway), or `unknown` (no map
window, or a lane of unknown travel). It is logged with each pulse; nothing acts on it yet.

**In the model** (`tests/lane_departure/test_lka.py`, the LA servo model at 90 km/h, truck 0.3 s behind
its wheel): a still-wheel drift on a straight of 0.3 to 1.2 m/s, wheel free, stays within 0.1 m of either
line over 16 s, a pulse at the line and a gentler one at the other as the swing home is caught, each 1.8
to 2.3 s; a driver holding 80 to 90 % of a bend's steering goes 0.03 to 0.06 m past at R 400 and 0.38 to
0.43 m at R 250. Euro NCAP's lane keeping bar is 0.3 m past the line. With resting hands whose friction
(350) holds the wheel where a pulse left it, the truck swings from line to line and the swings grow (2 to
15 m out within 16 s, at the model's spring and at the game's): nothing straightens the wheel. These are
the model's numbers, not the game's, whose centring at speed is about twice the model's.

**Against real systems** (a Haiku review, 2026-11-02, of UN R79, Euro NCAP and maker manuals; summaries,
not the standards' text). Known gaps, open:

- No hands-on check: real systems nag and stand by without hands; here a sleeping driver in a long bend
  gets a pulse every few seconds, with the rumble only if LDW is on. There is no per-minute limit (Lukas,
  2026-11-02: it was 8; a driver gripping the wheel is paying attention, and one who is not feels the pulls).
- With LDW off, or in a bend where LDW mutes its cue, the pulse comes with no warning first (R79 asks
  for one at the latest when it intervenes).
- The cap is a share of the wheel's own peak: about 1 Nm on a 4 Nm wheel, 6 Nm or more on a direct
  drive. R79 caps override effort at 50 N at the rim; `lka_strength` is the lever.
- Like real systems it yields on the driver's torque, here an estimate (handing back, above). The still-wheel
  override alone held the full 3000 for up to 3.5 s on 2026-11-02: it reads the servo net of the game's
  centring (2280, under its 2400 bar) or a wheel 0.013 to 0.016 short of its 0.015 bar. A driver correcting
  harder than the pulse is not resisted (one way only), but a pulse cannot straighten the wheel it turned
  either; real systems do both.
- Resting hands that hold whatever angle the wheel was left at make pulses swing the truck from line to
  line in the model (above). A gentle pull back toward the lane's angle, weaker than any hand, would stop
  it but goes against "never push back".
- The lane is the map's centre plus 2.25 m; where the map sits off the paint, a pulse can come at the
  wrong place. No trailer: the cab can be inside while the trailer crosses.
- The intent thresholds it inherits from the detector are still set by reasoning, not by driving.

**Config**: `lka_enabled` (false, Steer in the dropdown), `lka_strength` (1.0, 0 to 2, scales the servo like
`la_strength`), `lka_tmp_enabled` (true: TruckersMP moderation gave the go-ahead for steering, 2026-11-02;
config.json only, false keeps it off there). The
arm speed is `ldw_min_speed_kmh`; the wheel inertia and the fixed calibration are LA's keys
(`la_wheel_inertia`, `la_force_per_ms2`, `la_kappa_per_steer`), one wheel and one truck. `ldw_test_mode`
reaches it too, since it reads the detector: every close call starts a pulse, blinker or not.

**Logs.** `LKA on/off` (and `waiting while LA steers`), its settings, the shared calibration as for LA,
one line per pulse start (side, what lies past the line, gap, drift, time to line, speed, and the core's
state: `pulse`, or a mute such as `calibrating`) and end (why, how long, closest gap, and the driver's estimated
push against it: `core/la/README.md`, "The driver's push"; `tools/lka_pushback.py` lists it per pulse from
`lka_debug.csv`), one per departure the
detector caught that got no pulse (`skips`: a recent swerve, an override, a truck not
drifting out by the lane fit), and every 10 s
while it pulsed the LA summary. With `debug` on, `lka_debug.csv` gets LA's columns at 20 Hz during a
pulse and 2 s after.

## Side traffic

`side_traffic.py`, dev only for Lukas so far, always on with LDW or LKA. Per side, the vehicle that makes moving
over a lane unsafe right now. It does two things (Lukas, 2026-11-03), and nothing else:

- **Intent stops counting toward an unsafe lane.** The thread hands the sides to the detector
  (`Frame.unsafe_left` / `unsafe_right`), and on such a side the blinker, `deliberate`, `intent` and `latched`
  mute nothing, so a driver who signals and moves over calmly into a car is warned, and LKA pulses (its pulse
  mode reads the same edges; the ride through a fork ignores the lamp there too). A swerve (`evade`) still mutes,
  and so does every gate. Once the lane is clear again the mutes are back, so a lane change finishes as before.
- **The warning says so.** LDW or LKA acting shows on the CC panel's lane line and the bottom bar's corner in
  steady dark orange; acting toward an unsafe lane blinks hard red (`firing_alert`), on both. An episode that is
  an alert at any time stays one to its end, and both are held 1 s after LDW and LKA stop (`display.LaneLines`).
  Side traffic alone, with LDW and LKA not acting, shows nothing and never moves the wheel
  (`test_side_traffic_alone_lights_nothing_and_never_moves_the_wheel`).

Every number below is a placeholder in `SideTuning`, set by reasoning.

- **The frame.** Ego's lane from the map: the route ahead (240 m) and `route_behind` (120 m, rebuilt every 10 m
  or on a lane change; a straight line back where the map has none). Each body is placed by its centre's arc
  length and offset from the lane centre, its corners on the tangent there, so bends need no prediction.
- **The rig.** The tractor (`EgoGeometry`) and every attached trailer from the SDK (`TrailerBody`,
  `core/radar/README.md` §17, ends estimated). Its rear and front are the farthest corners along the lane.
- **Beside**: a body that reaches from 1 m behind the rig to 2 m ahead of it, entirely clear of the rig on that
  side and centred within 7 m of the lane centre, oncoming or not. A body that overlaps the rig is a ghost. A
  body that leaves that zone within 0.2 s at its speed relative to the rig no longer counts (Lukas, 2026-11-04: a
  car we pass dropping off a long trailer's rear, or one that passed us clearing the bumper as we pull out), and
  is released at once rather than held; one at our speed never leaves.
- **Everything else is a time lookahead** (Lukas, 2026-11-03), one lane over (centre 2.6 to 7 m off), within
  200 m, at today's closing speed; distance alone never counts, so a car at our speed just behind or ahead stays
  dark. **Behind** (we are being overtaken): travelling our way and reaching the rig's rear within 3 s.
  **Ahead** (we overtake it): travelling our way, slower, and reached by our bumper within 3 s. **Oncoming**:
  travelling the other way and reaching the bumper within 2.5 s.
- **Left out**: parked vehicles, radar's off-surface and no-collision-zone ids, crossing traffic (more than 60
  degrees off the lane), anything below 30 km/h, and a body one lane over where the map has no lane past that
  line (`edges.beyond` says `edge`: a median or a shoulder). Unknown or no map window keeps it.
- **Held** 0.3 s after it clears; a gate (speed, no map, stale or paused data, the rig off the lane frame) drops
  it at once.

`LaneDepartureData.side_left` / `side_right` carry the kind (`beside`, `behind`, `ahead`, `oncoming`, or ""). Logs: one
`info` line per side lighting up or changing kind (AI or TMP, length, offset, closing speed, gap and time, what
the map has past the line), the rig once it changes (trailers, length, kingpin to fifth wheel gap), and every
60 s how long each side was lit with onsets per kind, and one line per LDW/LKA episode that turns into an alert.
No coordinates and no player identity.

## Not done

Side-coded cues, acting on the edge class (a firmer or earlier pulse toward an oncoming lane or the road
edge), side traffic for players (thresholds from drives, a switch), a hands-on check for LKA, trailer swept path
and ATS validation. LA, the continuous lane assist prototype, is in `core/la/`.
