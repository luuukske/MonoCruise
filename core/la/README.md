# LA: lane assist (prototype, hidden)

LA is the codename for the lane assist prototype, so it is not confused with the lane keeping assist
(LKA) being added to lane departure, which is a separate feature. Steers the truck along its current
lane, bends included, by turning the wheel through its force feedback. The driver keeps their hands
on the wheel and always wins. It keeps the lane only: it never changes lanes, never takes a
junction, fork or merge, hands sharp bends back, and lets go when the hands leave the wheel. Full
self driving is ETS2LA's job and stays out of scope. Off by default and config only (`la_enabled`,
no UI): it is for testers who were told how to switch it on. `la_follow_all` is a test switch that
keeps it on through junctions, forks and merges (see "When it is off").

The lane keeping assist (LKA, `core/lane_departure/README.md`) runs this core in its pulse mode
(`AssistTuning.pulse`): standby at no authority, in over 0.15 s while the lane departure detector asks for a
pulse on a side, the correction steered to a line 0.5 m inside that edge, out over 0.5 s with the correction
dropped. The pulse mode skips the ring guard, the weave guard, the hands-off check, the 2 s rise and re-arm and
LA's own blinker and swerve mutes (Lukas, 2026-11-02: it lasts a second or three); the caps and the override
stay. A pulse's servo is one-sided: it never pushes toward the line it saves (`servo_only`, see the LKA section
there), so LA's override counts only a wheel held short of its target during a pulse.
Evasive steering support, much later, is meant to plug in as a path source ("Adding a path source").

```
radar_thread + telemetry_thread + map_thread
        |  LaneDepartureThread (20 Hz) -> AssistRunner (runner.py)
        |      path source: WantedPath (path.py), the lane follower passes LANE
        |      path tracking: LaneAssist (controller.py) on the lane fit (lane_model.py) -> target angle, rate
        |      wheel actuator: WheelActuator (actuator.py) -> servo request
        |      calibration: ForceCalibration (calibration.py) -> Gains
        v
Local\MonoCruiseFfbCmd v4 (core/lane_departure/command.py)
        -> plugins/ffb: damped PID on the wheel angle at the game's FFB rate, on top of the game's own force
```

`controller.py`, `actuator.py`, `path.py`, `lane_model.py` and `calibration.py` are pure logic, `runner.py` reads settings and
keeps the logs. The lane departure thread hosts it because one writer owns the command block; the rumble and the
assist run side by side and are switched separately. The route arrives in the lane departure detector's frame
(`core/lane_departure/README.md`); offset, heading and curvature come from the assist's own lane fit.
Nothing here touches ACC, AEB, CC or the pedals.

## How it turns the wheel

Two loops, each where it can run fast enough:

- **Path tracking** (MonoCruise, 20 Hz) turns the lane, plus any wanted path, into a target wheel angle and the
  rate it is moving at.
- **The wheel servo** (the plugin, at the game's FFB rate) holds the wheel on that target with a PID on the
  wheel angle, added to the game's own force: `kp` on the error, `kd` on the rate error, `ki` on the error over
  time. The target is moved on at its rate between MonoCruise's updates (for at most 0.15 s), so the wheel never
  sees a 20 Hz staircase. The game's force is kept whole by default, so the wheel still feels like the game: its
  weight, its turn-in, its bumps. In a steady bend the servo's integral carries what the driver's arms would.

The servo is designed as a second-order response on the wheel's inertia (rim with hands, `la_wheel_inertia`,
650 force units per lock per s^2 by default): 2 Hz, 0.7 of critical damping, the integral at 0.3 of the
bandwidth. The game's own centring answers the wheel angle almost at once (57000 per lock within 50 ms at
97 km/h on the third drive, close to the calibrated spring), so it counts toward that stiffness: the servo adds
only what the game's spring at the current speed does not give, never less than a quarter, and damping and
integral are sized on the total. That keeps the wheel equally stiff and damped at every speed. `la_strength`
scales the total. Caps: the servo 3000 of 10000 (up to the plugin's hard 6000 when `la_servo_cap` asks, LA only), its
damping half that (1500 at 3000), its integral the cap. The plugin may also push the rim ahead from the target's rate
and acceleration and read the wheel ahead by its delay (`la_ff_*`, `la_servo_lead_s`, off by default; Bench, Speed
benchmark). LA's target itself still moves on at its rate only (moving it on with the follower's acceleration
swung a slower truck's return further in the model); the push ahead moves on the follower's rate and acceleration
`la_ff_plan_s` from now (`actuator.follower_ahead`), sent with plugin flag bit 1.

`la_cancel_share` (0 by default) removes a share of the game's steady force while the servo holds: the
magnitude low-passed at 1.5 s, so turn-in weight, corrections and bumps still come through, faded with the
servo, at most 6000. It is there for a wheel whose servo runs out of room in hard bends; in the model the lane
is kept as well without it (worst 0.19 m after 20 s against 0.17 m with a full cancel).

### Why it is built this way (drives of 2026-11-01, Moza R3, light in-game FFB)

1. The first versions added a correction or a "hold" on top of the game's centring, sized by the game's own
   force. That centring was 100 to 400 of 10000 while assisting: too light to move a wheel against its own
   friction and the driver's resting hands, so it could not take a bend.
2. A stiff pull on the wheel angle (4 to 8 times the game's spring) took the bends but buzzed at 3 to 4 Hz, the
   plugin's buzz guard at a median 0.67: a stiffness with no damping, fed back a frame late, rings. The rim's
   inertia (about 650) was read off that ring: 4 times the spring at 90 km/h rang at 3.3 Hz.
3. The hold needed a model of the game's force, and that model would not stay put: one drive it flattened in
   harder bends (`593 * a^0.72`, 777 per m/s^2 at 0.5 against 431 at 3.5), the next it was nearly straight
   (`540 * a^0.93`). With a straight line the hold was 28 to 47 % too strong in bends and the truck cut every one.
4. MonoCruise's target moved in 20 Hz steps (up to 200 units of pull each, p95), the "jumps".

So the plugin now holds the wheel with its own damped servo on top of the game's force, its integral replacing
the model of that force, and moves the target on between updates. A first v4 build cancelled the game's slow
force outright; that took the game's feel away while assisting, and is now an option. In the design study (`tests/la/servo_sim.py`, 60 Hz, a frame plus 8 ms of
delay) the 2 Hz design stays stable from half to twice the design inertia, with or without 350 units of dry
friction and hands, and tracks a moving target within 0.001 of lock; 2.5 Hz rang at half the inertia and 3 Hz at
the design inertia itself. Damping on the old 40 ms low-passed wheel speed made ringing worse on the bench; the
servo reads the speed through a 6 ms low-pass.

The **ring guard** stays as the backstop for a wheel much lighter than designed: it cuts the servo to 0.8 each
time the plugin's buzz guard fires (once per drop), halves it when the wheel swings through the target past
0.004 of lock four times within 2 s, and cuts it to zero on a swing past 0.02 that comes within 0.5 s of the one
before (a ring at 1 Hz or more, caught half a swing later than before; Lukas, 2026-11-01). On drives 7 to 9 a
single swing past 0.02 fired 11 times, always 0.7 to 5 s after the previous one: the slow overshoot of a big return
or a lane crossing. Each took the servo to zero for 30 to 120 s, and at 8 to 30 % of its stiffness the truck then
swayed up to 2.4 m through its lane at 0.15 Hz. It comes back over 30 s;
a cut within a minute of the last doubles that, up to 120 s. A rim four times lighter than designed, with no
hands, still rings in the model; set `la_wheel_inertia` lower for such a wheel. Swings within 1 s after the wheel
sat still for 0.3 s or more, 0.008 of lock or more off the target, do not count (Lukas, 2026-11-01): on drive 8 a
hand or a stalled reading held it 2 to 4 s while the servo pushed, the wheel snapped 0.03 to 0.06 past the target
when let go, and that one swing took the servo to zero for up to two minutes. A ringing wheel is never still that
long. Nor does a swing count when the target moved more than the swing itself since the one before (Lukas,
2026-11-01): a ring swings round a still target, while on drive 10 a roundabout's lanes swung the target and the
wheel chasing it read as a ring, halved the servo and then cut it to zero; the truck swayed up to 2.5 m for two
minutes with no force to speak of. Replayed on drives 7 to 10 that leaves only the buzz cuts; the light rim with no
hands still rings round its target in the model and is still cut.

A cut takes the friction push and the breakaway push down with the gains (2026-11-09). Before, they stayed whole
while kp, kd and ki came back from zero, and a friction push with next to no servo under it pushes toward the
target like a relay. On a model fitted to the old wheel's drive that day (rim 104, damping 315, sliding 77,
breakaway 139, 21 ms, steps of 0.00058 at 60 Hz, the game's centring at 8000 to 20000 per lock) the push alone kept a
rim with 30 % less friction swinging 0.05 of lock at 1.6 Hz until it reached full lock; with the pushes cut, none
of the 7 runs that did that ran away, and the rest tracked as before. The push ahead on the target's own motion
(`ff_inertia`, `ff_damping`) is not cut: it pushes only while the target moves.

A cut leaves the integral whole down to `ring_ki_full_from` (0.3) and fades it to nothing below (Lukas approved
2026-11-09); kp and kd are cut as before. The game's centring pulls toward straight ahead, not the target, so the
integral carries every angle off centre and the wheel trails by about the game's spring over ki (`lag_s`). Cut with
the stiffness, ki fell as its 1.5th power: on the old wheel's second drive that day four ring cuts left 16 to 35 %,
ki went from 166000 to about 36000, the wheel trailed its target 0.4 to 0.7 s, and the lane wove 1.5 to 3 m either
side at 0.3 Hz; the weave guard fired 18 times and could not stop it, since the stop before the line drove the
swing. In a closed-loop model of that wheel (lane, truck, LA with its guards, plugin mirror, the fitted rim, game
centring at 90 km/h, 15 runs each) a cut to 0.3 with light hands swapped sides 64 times with the cut integral and
none with it whole, and hands off with no forced cut 8 runs went past 2 m against 1. The ring guard's own cuts
stayed as often as before. Do not leave the integral whole all the way down: with kp and kd cut to zero it steered
alone and 10 of 15 hands-off runs ran away. Predicting the correction a wheel lag ahead and cancelling half the
game's force were tried as well: the first helped less on its own, the second made the weave worse.

## The target angle

```
kappa_cmd = bend scale * lane curvature one servo lag ahead  +  wanted path curvature  +  correction / v^2
target    = -kappa_cmd / g_s                                 g_s = curvature per unit of steering, learned
```

- **Bend feedforward**: the lane's own curvature, from the lane fit, is what makes it take bends on its own. It is
  read as far ahead as the wheel trails its target: the game's spring the servo's integral has to carry, over that
  integral's gain (`WheelActuator.lag_s`, 0.05 to 0.4 s). That lag grows with speed: drives 5 and 6 measured
  0.15 s at 80 to 90 km/h and 0.25 s at 100 to 130 km/h, as predicted, while the truck's path follows the wheel
  within a tick. A fixed 0.4 s led the wheel by up to 0.25 s: the truck turned in early, cut inside, unwound early
  and left every bend 0.4 to 1.35 m wide (drive 6). In the model, on a straight-bend-straight lane at R 150 and the
  game's spring at 90 km/h, that was 0.30 m inside and 0.34 m wide; read one lag ahead it is 0.01 m either way.
- **Bend scale**: in steady bends (curvature past 1/700, engaged, drift under 0.1 m/s) the command is what the
  truck needs, so its ratio to the map curvature is learned over 15 s of such bends, bounded 0.8 to 1.2. On the
  drives the map curvature and the steering gain disagreed by 7 % in steady bends and about 12 % in tight ones
  (0.20 against the calibrated 0.186 per lock), enough to cut every bend. In the model a steering gain 12 % off
  either way now settles within 0.08 m of the centre in a bend.
- **Correction** toward the learned centre plus the wanted path's offset: 1.0 m/s^2 per metre past a 0.05 m soft
  free zone, 2.0 per m/s of drift, smoothly capped at 2 m/s^2, plus a slow integral (0.05 per m per s, bounded at
  0.8 m/s^2) built only while fully engaged and steady, leaking over 20 s otherwise. It was 0.6 and 1.6 with a
  0.1 m zone; the precise servo made the stronger centring stable in the model. The cap was 1 m/s^2 until drive 6,
  where a 0.6 m/s drift at 130 km/h sat on it (halved by the weave guard) for 3 s and reached the next lane; at 2 a
  1 m/s drift is caught 0.15 m sooner and back within 0.2 m in half the time, and near the centre nothing changes.
  Stiffer gains (1.3 to 2.5 per metre) overshot and wove in the model, because of the servo's lag.
- **Stop before the line**: drifting toward a lane line faster than 0.3 m/s, the correction is at least the
  lateral decel that stops the drift before the body reaches the line, with the room counted from where the truck
  will be one servo lag plus 0.2 s later, up to 5 m/s^2. In the model it takes a 2 m/s drift 0.5 m less far
  (1.27 m against 1.75); the wheel's own lag and the servo's cap are what limit it beyond that. It fades out from
  the centre to a metre past it on the far side: as a hard floor it also forbade any pull toward the far line,
  which is the way home from beyond the other one, so on drive 8 the truck came back at about 0.3 m/s.
- **Heading** against the lane is filtered at 0.05 s from the lane fit, 0.15 s from the polyline fallback. The loop from
  correction to the truck's sideways motion measured 0.25 to 0.3 s on drive 8 and the filter added to it: in the
  model with the truck's path 0.2 s behind its wheel, a 1.2 m/s kick overshot 0.39 m and a 1.8 m/s one 0.87 m with
  five `weave` cuts at 0.15 s, against 0.18 m and 0.39 m with one at 0.05 s.
- **Drift ahead**: the correction acts on the offset and drift 0.15 s on (`lat_predict_s`), from the wheel's own
  angle: the curvature it gives through the learned steering gain against the lane's through the bend scale, so it
  is zero in a steady bend. The lane shows a correction about that late (0.075 s of measurement plus the heading
  filter, drive 8), and acting on the late drift overshot every return. The wheel's part is capped at the
  correction's ceiling, so a stalled steering reading (drive 8) moves the drift by 0.3 m/s at most. In the model a
  1.2 m/s kick with the truck 0.3 s behind its wheel overshot 0.36 m and settled in 7 s, and 0.14 m in 2 s with it;
  0.2 s behind, 0.18 m against 0.02 m, with no `weave` cut either way. A longer horizon stalls the return when the truck answers at once; 0.15 s
  holds from 0.05 to 0.3 s of lag. It does not replace the weave guard: counting only flips across the centre, the
  heavy, self-aligning wheel with resting hands still swung out of a bend in the model, with prediction or without.
- **Bumpless and smooth**: the target starts at the wheel and follows the wanted angle with a critically damped
  second-order follower at 3 Hz, at most 0.2 of lock per second; starting from rest, it does not yank when it
  engages. Its rate is smooth and goes with the target. It was a 0.12 s low-pass, whose tick-to-tick rate
  zigzagged and fed the servo's damping (over 300 units of it 3 % of the time on the fifth drive); 2 Hz added
  enough lag to graze the line in the model. The limit was 0.04 of lock per second, sized for engaging: on drive 7
  a prefab's right kink into a left one at 100 km/h held the target on it for over a second and the truck hit a
  pole 3 m out. In the model that kink went 3.4 m out at 0.04 and 0.5 m at 0.1 or more (a sharper one 7.3 m and
  0.8 m).

## The lane under the truck

The map rebuilds ego's route 20 times a second from ego's foot point on the lane, interpolated on the chord
between the lane's own vertices (10 m apart on roads). Read straight off that polyline, bends went wrong on the
fifth drive:

- the heading against the lane jumped at every rebuild, because the first span and its midpoint moved with the
  foot: the drift jittered 0.12 m/s tick to tick (p90 0.25) above 2.5 m/s^2 of cornering against 0.005 on
  straights. The correction followed it, the `weave` guard cut its gain (0.68 median), and the target's rate sat
  on its limit;
- a chord sags `L^2 / 8R` inside the lane (0.08 m for 10 m at R 150), so a truck on the lane read as outside it
  and was steered inside: 0.1 to 0.2 m median toward the inside of bends.

`LaneModel` (`lane_model.py`) keeps the lane's own vertices, everything past the foot, across rebuilds while the
new route continues on the same vertices, up to 40 m behind ego, and fits a quadratic in ego's frame weighted
over 25 m each side. Offset, heading and curvature at the rear axle come from that fit, the feedforward's
curvature from a second fit 0.4 s ahead. On a new lane, with nothing behind ego yet, the fit reaches 50 m ahead
instead; with too few vertices the polyline is used as before. In the model, on routes rebuilt the map's way at
R 150, the drift jittered 0.29 m/s tick to tick off the polyline and 0.02 off the fit, and the target's rate
fell from its 0.04 limit to 0.004. The polyline still decides `off_route`, forks, merges and the bend peaks.
`lane_fit_m = 0` (code only) goes back to the polyline.

The route is the lane the truck is in, so crossing a line moves it to the next lane: the offset jumps by a lane
width. Normally that starts the lane over (`lane_settle`). With `la_always_on` the lane it left stays the
reference instead: the jump is carried as a shift on the offset (event `carry`), so the pull home and the stop
before the line keep working across the line, until the route is back on that lane or 10 s pass (not while the
truck is still heading home), after which the new lane is taken. A gap in the geometry (any gate) ends the carry and
the next jump after it starts the lane over, and so does a jump or a shift past 6 m (a parallel road, a re-plan).
The bias, the bend scale and the correction's integral do not learn while carrying. On drive 8, 13 crossings each
started the lane over, with no correction for 1.5 s, and then centred the truck in the next lane. The lane keeping assist will need the same carry with a rule for when the
driver meant the change (blinker, steering through it).

A lane that moves under the truck faster than a truck moves (more than 6 m/s sideways that its drift does not
explain, or its drift changing faster than 20 m/s^2) is the geometry, not the truck: a roundabout's or junction's
lanes trading places, often over two ticks so that each stays under the 1.5 m jump. The target then holds where it
is for 0.5 s after the last such tick (the wheel stays put, the driver can still steer) and the lane starts over
(event `geometry`). On drive 10, at 70 km/h in a roundabout, the lane went 1.9 m out and back in half a second and
read as 11 m/s of drift. Normal driving sits far below either line (p99 1.6 m/s and 5 m/s^2 on drives 7 to 10);
those drives cross them 10 times, all at junction speeds or far off the lane.

## Calibration

`ForceCalibration` fits two lines on steady cornering samples (40 km/h or more, lateral accel within
0.3 m/s^2 over 0.8 s, game force under 9000, 120 s forgetting, 20 s of samples to start):

| fit | used for | ready when |
|---|---|---|
| game force per m/s^2 of cornering (`k_a`) | its sign is the wheel's torque direction; the logs show the game's spring | spread 0.25 m/s^2, r 0.7, 100 to 30000 |
| curvature per unit of steering (`g_s`) | the target angle | spread 0.004, r 0.7, positive, 0.05 to 0.8 per metre |

Both keep learning while the assist steers: the game's own force is read before the plugin changes it. Each fit
locks its first sign (a later flip is a `conflict` and keeps the assist off until restart) and, once ready,
holds its last good value through a weak stretch (`held`). `la_force_per_ms2` and `la_kappa_per_steer` fix them
by hand. Drives so far: +480 to +735 per m/s^2 (r 0.94 to 0.99) and 0.184 to 0.187 per metre per lock (r 1.00)
within a minute.

Live fits ready on both are kept for the next start (`la_saved_calibration`, at most once a minute and on a 3 %
change), because learning took the first one to two minutes of every drive: on 2026-11-02 the first six lane
keeping pulses had no force. A saved calibration is used only once the drive confirms the wheel's direction,
since a wrong sign steers toward the line: the game's force against the wheel angle at 40 km/h or more (bends
or not, 2 s of it, r 0.6, 0.002 of lock of spread) must have the saved sign, and that drive's force against
the angle was -41900 per lock at r -0.96 (the cornering fit's sign, as the angle here is counterclockwise).
Confirmed, it fills in until the live fits are ready, and its sign is locked, so a live fit that later reads the
other way is a `conflict`. Disagreeing, it is dropped with a warning and the calibration learns afresh. The
steering gain carried to another truck is off by a few percent at most on the drives so far; the bend scale and
the correction take that up.

## Wheel learning and the target hold

Simulated only so far (`tools/la_bench/SMOOTH_RESULTS.md`); nothing here has run on a real wheel. With
`la_wheel_learning` on (the default) the servo needs no wheel check: `core/la/wheel_learner.py` learns the wheel from
the plugin's trace block (every FFB update: the reading, the game's force, what went to the wheel, the servo's part,
the steering frame count) while the driver drives, and `core/la/hold.py` keeps the servo quiet at the target.

What it learns, and from what (nothing extra is ever pushed; the wheel only moves as the driver or LA moves it):

| estimate | from | used for |
|---|---|---|
| reading rate | the steering frame count, always | reported |
| reading step (quantum) | the reading's changes with the servo off (a hunting servo reads in big steps only) | the hold's band |
| game's centring | the game's force against the angle, servo off | reported (the calibration still sets the servo) |
| start force, its top | the force a standing wheel started on after a slow climb, or the forces (low-passed at 50 ms) it stood against for 0.15 s, while LA steers (hands included, which is what the servo must move); the top at most 1.5 times the middle | the friction push and the breakaway push |
| design step (`rung`) | the hold: a still target the servo's push flipped round 4 times in 2 s without parking, each flip past a tenth of the start force (at least 30) | one step gentler each time, back one step after 10 minutes of servo time without a hunt; at the eighth, the plugin's settle law off |

Not learned: the rim's own inertia and the delay. With hands on the rim they do not show in normal driving, so the
design steps down on the hunt they cause instead. Every estimate starts unknown, which leaves the servo exactly as
configured (the defaults or the wheel check's keys); a step only ever lowers the bandwidth and the design inertia, damping
first, the damping ratio as configured (`learned_overrides`), down to 0.5 Hz on 50 (six steps), and below the second step only once the friction push
is known or after two more hunts there, since a gentle servo alone cannot move a sticky wheel (but one that keeps
ringing never stands still long enough to learn its friction). The friction push is 0.6 of the start force (an
under-push leaves the rest to the feedback), climbing to 1.2 of the way to the top start while the wheel stands; a
configured push (`la_ff_*`) wins. The model is bounded on every field, saved to `la_wheel_learned` at most once a
minute, and shared by LA and the lane keeping assist (the lane keeping assist takes the hold and the settle flag only:
its design and pushes stay as configured).

The target hold runs at MonoCruise's 20 Hz on the plugin's reading. A wheel whose reading stays within half a reading
step plus 0.25 mlock of a still target for 0.15 s (after 1.5 s within 1.5 mlock, the plugin's own breakaway hold-off),
or rattles over at most 3 mlock round it for 0.5 s (a motor bouncing in its play), has its gains and friction pushes
faded to nothing over 0.15 s: the plugin's integral then stands still and keeps the force that parked the wheel, so the
output neither jitters nor flips. The reading or the target leaving the band brings the servo back over 0.05 s. It only
ever lowers the gains the actuator asks for, so the caps, the slew and the buzz and ring guards are untouched, and it
works with the layout 5 plugin as it is (no DLL change). Its limit, in the model: the last 1.5 mlock is the
integral's alone (the plugin's breakaway push stands off there), so a sticky wheel with a fine encoder is often parked
1 to 2 mlock short. The final-error limit is 1.5 mlock (Lukas, 2026-11-08).

A plugin that has the settle law (`plugins/ffb/README.md`, status bit 4096) takes the last mlocks too: MonoCruise
sets servo flag bit 2 whenever `la_wheel_learning` is on (a layout 5 plugin ignores it), and the breakaway push then
finishes the move to 0.5 mlock plus half a reading step and parks the wheel there at the FFB rate. Not in the shipped
DLL yet; simulated only.

## Wheel calibration (in progress)

Learning the wheel while driving went wrong on the old Thrustmaster three drives running (2026-11-08/09): the hunt
check read its 0.58 mlock reading steps as hunting and stepped the design to rung 2 or 3, and the soft design that
left trailed its target by 0.5 s at 60 km/h (the game's spring over the integral) and the lane wove 3 m. The
calibration measures the rim instead, standing, in about 14 s: `core/la/wheel_calibration.py`, pure logic fed the
plugin's trace samples, asking for one open-loop torque per 20 Hz step.

Truck parked, hands off the rim, the wheel within 0.25 lock of centre. Six trials, their ways alternating so the
wheel ends near where it started: the torque climbs at 400 per second until the wheel moves 2 mlock (its start),
then holds half that push (to see whether it keeps sliding), then 1.25 and 1.6 times it, switched 25 % up and down
every step (a smooth hold hid the delay on a heavy rim: the R3 model read 8 ms for 36). Each hold ends after 0.4 s or
0.12 lock of travel. A hold of more than the start that stops within 10 mlock is play in the drive, taken up: the
push climbs again from there. One least-squares fit over every trial's readings from its start until the wheel stops
or turns back, push = inertia * accel + damping * rate + sliding friction + spring * angle, integrated twice so the
readings are never differentiated, with one start rate per trial, scanned over the delay from 0 to 80 ms: inertia,
damping, sliding friction, spring and delay. The push is the trace's `out`, the game's own force included, so the
game's standing friction (about 550 while the wheel moves, 2026-11-06) does not count as the rim's; the profile keeps
the calibration's own push at the start beside it. A fit outside its bounds, or off by more than 25 %, is no fit.

Model wheels (`tests/la/test_wheel_calibration.py`): the old, sticky, drive-fitted old (inertia 104, damping 315,
sliding 77, breakaway 139, 21 ms), R3 read at 60 Hz and bare rim with 4 mlock of play all come back within 10 % and
9 ms, in 10 to 13 s, the wheel ending within 0.08 lock of where it began. Two model features it does not fit: a
friction that rises with speed reads as damping (R3 model: 1636 for 630, 60 ms for 36), and a geared wheel's teeth
leave a rough fit (inertia 479 for 600, damping 1606 for 900). The bench recordings of 2026-11-06 hold the assist
only, not the game's force, so they cannot test it.

On the old wheel (2026-11-09, two runs) the game's own force while the wheel moved was small (at most 85) and the fit
was off by 0.5 to 0.7 %, against well under 0.1 % on the models even with 0 to 8 ms of reading jitter: lumps of 2 to 6
mlock the model has no term for. Damping (705, 766) and sliding friction (213, 189) held across runs and every delay
and force smoothing tried, but the error only slid toward one end of the delay scan, so inertia (41 to 139) and delay
were not pinned. A delay is now taken only from a clear dip inside the scan (5 % below both ends); otherwise the fit
at 20 ms gives the damping and friction and leaves inertia and delay unknown (`WheelProfile.pinned`). The drive's
closed-loop fit read damping 315 and sliding 77 on the same wheel, so standing and driving still disagree. The
reading rate is the median gap between readings: the trace still held samples from before the game came back to the
front, and the span read 26 and 37 for 62.

So a relay ends the run (Lukas, 2026-11-09): the plugin's servo saturated round the resting angle (P only, full
push from 0.5 mlock of error, its target 2 mlock off so the first push starts the swing) at 1.3 times the median
start push, for 1 s or until the wheel is 30 mlock from where it stood. It swings at the rate where the delay costs
the most phase, and its readings join the fit in 0.25 s windows, the friction following the way the wheel moved
reading to reading, the stretches where it stuck left out. The plugin's buzz guard halves it after four swings; the
fit reads the force that went to the wheel, so that costs only swing size. On the model wheels with 0 to 8 ms of
reading jitter it gives the fit a sharp dip (the real-like model, inertia 104, damping 750, sliding 200, breakaway
460, 25 ms, small cogs: off by 0.54 % at no delay, 0.21 % at 30 ms, 0.75 % at 80 ms), and the delay comes out 2 to 7
ms long (the jitter's own 4 ms), the inertia within 5 %, the swing 11 to 38 mlock from peak to peak (the sticky model
61: the travel is checked at 20 Hz). At twice the start the old wheel's model swung 76 mlock; at 1.1 times some
wheels never started. A wheel with play leaves the relay out of the fit, since every turn crosses the play (the bare
rim with 4 mlock fit off by 0.54 % with it); play is the middle of the holds over the start, since one hold that
stuck and climbed again (2.9 mlock on the old wheel) left the relay out the first time.

The relay's readings in the fit were not enough on the wheel: over four runs on the old wheel (2026-11-09 18:14 to
18:18) the fit's dip sat at 24, 44, 44 and 52 ms, with inertia 105 down to 69, sliding along the same inertia against
delay ridge, while the relay itself repeated (half swings 0.13 to 0.20 s, 8 to 30 mlock, peak push 600 to 700, 50 to
75 ms from each push flip to the swing's peak). So the delay now comes from the relay's swings: for each delay of the
scan the fit's rim at that delay runs the relay in a model (`simulate_relay`: the plugin's saturated P at its FFB rate
faded in over 0.3 s, readings held at the game's rate in the wheel's steps, the rim starting on the trials' breakaway),
and the delay whose model swings as the wheel did over the 0.65 s after the first push flip wins, when it matches 1.5
times better than either end of the scan. The four runs then read inertia 105, 99, 87 and 101, delay 24, 24, 34 and 24
ms (the drive's fit 104 and 21), damping 760, 789, 896 and 561, sliding 180, 257, 240 and 282, the relay matching 1.8
to 4.4 times better than at the ends (`tests/la/data/old_wheel_calibrations_2026-11-09.json`). On the model wheels
with reading jitter the delay comes out within 7 ms and the inertia within 10 %, and a 40 ms model reads 38. Damping
and friction stay two to three times the drive's fit, as on every standing run.

`python tools/la_servo_bench.py calibrate` runs it on the wheel (the same setup as the wheel check: truck standing,
LA, the lane keeping assist and the rumble off) and saves every trace sample to `tools/la_bench_out/`;
`calibrate --replay FILE` analyses a saved run again, `--simulate NAME` runs a model wheel. Nothing leaves the
machine.

The design: a pinned profile becomes a model rim (`calibrate.rim_of`: its inertia, damping, sliding friction,
breakaway and its spread, spring, delay, reading rate and steps) and the wheel check's search (`design.choose_design`,
push ahead and breakaway push on, the friction treated as measured) picks the fastest design that holds on it and its
neighbours, about two and a half minutes. `calibrate --save` writes its `la_*` keys and the profile
(`la_wheel_profile`) to config.json and starts the learner's design steps over; while a pinned profile is saved the
learner never steps the design (`learned_overrides(calibrated=True)`; a hunt is logged, nothing more), since the
steps took the old wheel's design to rung 2 or 3 on three drives. The hold, the reading step and the settle law stay.
Delete `la_wheel_profile` to go back to learning.

On the old wheel's four runs (median rim: inertia 100, damping 760, sliding 240, breakaway 420, spread 15 %, 26 ms) the
search picked 1.75 Hz on 150, damping 0.7, push ahead inertia 100, damping 760, friction 240 a delay early, breakaway
push up to 580. In LA's lane model at the game's spring for 90 and 60 km/h, kicked every 12 s by a drift of 0.4 or 1.2
m/s, hands off and light, on the calibrated rim and on the drive's fit (damping 315, sliding 77), 2026-11-09:

| design | worst offset, calibrated rim | worst, drive's rim | sign flips | ring cuts | weave cuts |
|---|---|---|---|---|---|
| the config of that day (2.3 Hz on 184, friction push 150) | 0.92 m | 3.54 m | 11 | 13 | 13 |
| the same at rung 2 (the drive that ran off) | ran away | 5.38 m | 90 | 0 | 25 |
| the calibration's design | 0.93 m | 0.86 m | 0 | 12 | 2 |
| the design for the drive's fit (2.0 Hz on 104, friction 77) | 2.30 m | 1.05 m | 150 | 0 | 59 |

The calibration's ring cuts all come on the drive's rim, where a friction push of 240 over-pushes one sliding on 77;
a push of 0.6 times the measured friction cut them to 2 but the calibrated rim then flipped sides 8 times, so the
design keeps the measured push. Which rim is the truth while driving is still open. Not done: the lane correction
slowed to the servo's lag at the game's spring, and the in-app start (parked, hands off, saved per wheel, for the lane
keeping assist too).

On the wheel (2026-11-09 18:48 and 18:50, speed bench, standing) the `--save` run's design (it read 16 ms where the
other runs read 24 to 34, so 2.5 Hz on 107) and the 1.75 Hz design both did better than any design before (Lukas)
but jittered on a moving target and near rest: the 0.1 lock/s ramp tracked 4.5 and 3.4 mlock rms, the wheel stopping
and jumping 10 mlock ahead; the 0.02 step overshot 104 and 71 %, ringing about 10 mlock either side at 4 Hz, the
push swinging between +1000 and -1000 (P 360 and a breakaway push climbed to 580 at the start), until the buzz guard
halved it on the 0.2 lock/s ramp and the step; then the wheel stood 3 to 5 mlock off while the integral built to 500.
The calibrated rim in the model gave 0 to 19 % on that step and 1.0 mlock on that ramp, and a closed-loop fit of the
eight bench runs (inertia 142, damping 1648, sliding 121, 34 ms) still gave 11 to 32 % and 1.1 to 2.1 mlock: the wheel
does something no rim of this model does. A force deadband (small pushes doing nothing, which would read as friction
standing and vanish under the game's centring at speed) was fitted beside the friction on the five relay runs and
read anywhere from -275 to +324: not separable from this data. Which part of the push makes the overreaction is the
next thing to find on the wheel: the breakaway push, the friction push, or the feedback alone.

It is the friction push. Three short benches at 1.75 Hz on 150 (2026-11-09 20:35 to 20:37, ramps 0.1 to 0.4 lock/s,
steps 0.02 and 0.05, half lock at 0.5 lock/s):

| push ahead | ramps | 0.02 step | buzz guard cuts |
|---|---|---|---|
| friction 240, breakaway push 580 (18:50) | 0.2 not followed | 71 % over, rang | ramp, step, half lock |
| friction 240, no breakaway push | 0.1 not followed | 90 % over, rang | every run |
| friction 120, no breakaway push | all followed, at most 1 % over | 17 % over, clean, 90 % in 0.42 s | half lock only |
| none (feedback alone) | all followed, 8 to 9 % over, 0.1 lock/s 6.2 mlock rms against 4.1 | 8 % over, 0.56 s | none |

The calibration reads the friction on fast slides (0.3 to 0.5 lock/s); at slow moves the wheel slides on less (the
drive's fit read 77), and the full push over-pushed at every slow start and stop. The design now pushes half the
measured sliding friction and no breakaway push (`calibrate.FRICTION_SHARE`, `choose_design(friction_share=0.5,
breakaway=False)`); the wheel check keeps the full push. Every run still stood 1 to 3.5 mlock off the target after
a small step with the settle law off, as the bench runs; LA's hold and the settle law handle that part on a drive.

The Moza R3 (2026-11-09 21:52, simulated inertia as Lukas drives it) calibrated to inertia 426, damping 512, sliding
137, 20 ms, close to its earlier bench fits, but the design picked (1.75 Hz on 638) felt crunchy, "as if driving over
gravel", and shook coming to the target; it rang on the 0.4 lock/s ramp, the 0.02 step (27 %) and half lock. The
crunch is force chatter: the servo's force changed by a median 148 and 241 from one reading to the next on the 0.1 and
0.2 lock/s ramps, against 42 and 85 for the 1.25 Hz on 528 design Lukas found the smoothest on 2026-11-06. The model
showed none of it (3 to 23) because it reads the wheel perfectly; with the game's reading up to 4 ms older than the
FFB update reports (`WheelParams.reading_jitter_s`, uniform) it gives 120 and 235 for the calibrated design and 68 and
129 for the smooth one, the same order as the wheel. The calibration's design now models that jitter and rejects a
design whose median force change per reading on a 0.2 lock/s ramp passes 150 on the calibrated rim
(`design.chatter`, `choose_design(chatter_max=...)`; the wheel check keeps neither). It then picks 1.25 Hz on 426 for
the R3 and 2.0 Hz on 105 (friction push 90) for the old wheel's 18:14 run. On the old wheel every bench design read
108 to 127 at 0.2 lock/s on the wheel: its trouble was the friction push, not chatter.

### Any wheel (first test, 2026-11-09)

Three numbers above were fitted to these two wheels (half the friction, 4 ms of jitter) and the design search took two
and a half minutes, so the calibration now measures what it assumed and checks its design on the wheel (Lukas:
"read all information needed for smooth operation on ANY wheel", the long search "a tiny issue for a release"):

1. **Slow friction (creep).** After the six trials a gentle servo (1.5 Hz on the trials' rough inertia, at least
   100, critically damped, a friction push of 0.6 of the fast slides' friction so a sticky wheel moves at all) walks
   the target 0.06 lock out at 0.05 lock/s, holds 0.3 s and walks back: the push it took while moving, both ways
   averaged so a steady offset cancels, less the damping, is the friction a slow move meets (`slow_friction`).
   Model wheels read it within 20 % of their friction at 0.05 lock/s (the drive-fitted old wheel 73 for 77, the R3
   201 to 221 for 190, one whose friction rises with speed 132 for 146), where the fast slides read up to 1.6 times it.
2. **Reading jitter.** From the trials' slides: each reading against a quadratic through its six neighbours, the
   reading steps' own share taken out, over the squared rate (`jitter_s`). Model wheels read back their jitter within
   1.5 ms; the real R3 reads 6.8 ms and the old wheel 7 to 12 (its lumps count in), where 4 ms reproduced the R3's
   chatter: the model starts a little cautious there, which the check makes up for.
3. **Design in seconds.** The model rim (`calibrate.rim_of`) has the measured jitter and a friction that rises from
   the creep's to the fast slides' (`stick_speed`); the push ahead is the rim's inertia and damping and the creep's
   friction a delay early, no breakaway push. `design.fast_design` takes the fastest bandwidth on the rim's own inertia
   at damping 0.7 whose model chatter stays under 150 and that holds on the rim and its neighbours: 1.2 to 2.4 s.
4. **Design check on the wheel** (`core/la/design_check.py`, about 3 s each): out 0.06 lock at 0.2 lock/s and back,
   a 0.02 step and back, holds between. It fails on force chatter over 130 (the median change per reading on the
   slow moves; the R3 crunched at 241, was smooth at 73), overshoot past the step over 35 %, or three swings past
   it; the model's design goes 20 % gentler while it fails and 25 % faster while it passes with room (chatter under
   0.7 of the limit, overshoot under 15 %), at most four checks, and the fastest that passed is the design. If none
   passed, nothing is saved. How far off it comes to rest is reported, not judged: a sticky wheel rests further off
   at a lower bandwidth, so stepping down cannot fix it (the sticky model wheel rested 10 mlock off).

About 15 s of calibration, 2 s of design and 3 to 12 s of checks. On the model wheels: the R3 1.25 Hz, clean at once;
the old wheel's model 2.5 Hz in the model, 44 % over on the check, 2.0 Hz clean; the sticky model nothing in the
model, 0.5 and 0.62 Hz clean, 0.78 Hz 58 % over. The chatter limit is the one number from feel (Lukas on the R3,
force units are a share of each wheel's own full force).

First runs on the R3 (2026-11-09 22:31 and 22:32): the jitter read 10.3 and 4.6 ms, so the model started at 0.5 and
1.25 Hz, and the checks ended at 0.78 Hz (resting 7 to 8 mlock off) and 1.0 Hz (1.25 Hz failed on chatter 166, where
the speed bench had measured 73 for that design). Two fixes. The jitter is now the median scatter, not the mean,
scaled to read the model wheels back (`jitter_scale`): the R3's three runs read 2.1 to 2.7 ms (4.6 to 10.3 by the
mean), the old wheel's five 3.7 to 6.6. The check's ramps were 0.06 lock with sharp corners, and their stops counted
in the chatter: they are now the speed bench's, 0.15 lock at 0.2 lock/s with 0.1 s rounded corners, and in the model
the check reads within 10 % of the bench (92 against 99 at 1.0 Hz, 111 against 127 at 1.25 Hz). A check now takes
about 4 s.

The chatter limit was then set on the wheel (2026-11-09 22:43 to 22:47, the R3, Lukas away, the game left running):
the check run three times each on the design Lukas felt smooth (1.25 Hz on 426), on the crunchy one (1.75 Hz on 638)
and on the calibration's 1.5 Hz. Per reading the force change scattered too much to judge on (smooth 73 to 113, 1.5 Hz
102 to 142, crunchy 150 to 250); per FFB update, what the motor gets, it repeats and separates: smooth 53 to 56, 1.5
Hz 79 to 86, 1.88 Hz 95, crunchy 116 to 137. The check now measures it per FFB update with the limit at 90. Two
calibrations after it ended at 1.6 Hz (chatter 83, 3 % over, resting 0.1 mlock off) and 1.5 Hz (76, 0 %, 0.4) from
rims that read inertia 441 and 443 but damping 283 and 496 and delays of 12 and 14 ms (the runs before read 343 to 359,
792 to 873 and 36 ms): the R3's fit still wanders from run to run, and the check keeps the design steady anyway.
The speed bench on that 1.5 Hz design: every ramp to 0.8 lock/s followed (at most 4 % over), steps of 0.02, 0.05 and
0.1 lock to 90 % in 0.21, 0.20 and 0.23 s (7, 17 and 14 % over; the smooth 1.25 Hz took 0.71, 0.31 and 0.27 s), half
lock at 2 lock/s in 0.37 to 0.39 s, no buzz cut, chatter per reading 46 and 93 on the 0.1 and 0.2 lock/s ramps (1.25
Hz 36 and 73, crunchy 149 and 241). Lukas felt it: "nice, not perfect, but good enough for a release on the R3".

Then the old wheel (2026-11-09 22:55 to 23:12, the agent alone). The first run designed 1.4 Hz on its rim of 89 with
a friction push of 264, and the check passed it; a grid of 16 checks showed why that was wrong: below a stiffness of
about 16000 per lock (1.4 Hz on 89 or 150, 1.75 Hz on 89) the wheel never moved for the 0.02 step at all, so it
"overshot 0 %" and the check called it clean, and with 180 or more of friction push at 1.75 Hz on 150 it overshot 27
to 69 % and rang. On a sticky wheel a small step needs the feedback's push (kp times the step) and the friction push to
beat the breakaway; below that it waits for the integral and then lurches, the jitter Lukas felt. So:

- the friction push is cut on a sticky wheel by its sliding over its breaking force (`calibrate.push_of`): the creep
  reads the push a slow move takes (250 to 500 on the old wheel, sliding or stuck), most of which is stick-slip;
- the design inertia is raised to the stiffness that moves a 0.02 step past the breakaway with that push
  (`kp_floor_of`, `fast_design(kp_floor=...)`), and kept there when the check goes gentler (`_floored`);
- the check runs steps of 0.02 and 0.05, and fails a design that follows the small step less than 70 % of the way:
  that one goes stiffer (half again on its design inertia), not gentler, unless it rings;
- the old wheel's fitted sliding friction read 210 to 372 from run to run, so once a design passes the check runs it
  again with half the friction push and keeps the half if it overshoots at least 5 points less.

Its last runs kept 2.0 Hz on 70 with a push of 126 (the half: 5 % against 14 %) and 2.0 Hz on 97 with 146 (the half
overshot more). The speed bench on 2.0 Hz on 97, push 146: ramps of 0.1, 0.2 and 0.4 lock/s followed (at most 2 %
over), the 0.02 step 15 % over in 0.29 s, half lock at 0.5 lock/s followed, no buzz cut in any run (the hand-found
1.75 Hz on 150 with a push of 120 cut on half lock); the 0.05 step's swing through zero overshot 49 % (25 % by hand),
a move the check does not run. Not felt by Lukas yet.

### Fast on the road (2026-11-11)

Lukas: since the calibration the lane keeping pulses cross the line they used to barely touch. `lka_debug.csv` per
day: the pulse's kp fell from 25661 to 8028 and its integral gain from about 400000 to 118429 per second, its peak push
from a median 1300 to 1700 (the cap, 3000, at the 90th percentile) to 470 to 590. The cap never changed. The cause is
the design the calibration saved, 1.4 Hz on 415 for 2.0 Hz on 650: a design stiffness (`inertia * (2 pi f)^2`) of
32100 for 102600, and the lane keeping assist reads the same `la_*` keys.

Why a design that checks out standing is slow on the road: standing, the servo is the whole stiffness on the rim. At
speed the game's own centring is most of it (about 122 per lock per (m/s)^2 on the test truck: 34000 at 60 km/h, 76000
at 90, 159000 at 130), the servo keeps a quarter of its design as its own push (`own_floor`) and the integral has to
carry the rest, so the wheel trails its target by the game's spring over the integral gain (`WheelActuator.lag_s`).
On the calibrated rim in the model (`tools/la_bench/road.py`, pulses as `pulse_eval`, hands off and resting, drifts of
0.4 and 1.2 m/s):

| design | trails by, 60 / 90 / 130 km/h | pulses, worst past a line | standing step |
|---|---|---|---|
| 2.0 Hz on 650 (before the calibration) | 0.09 / 0.20 / 0.29 s | 0.10 to 0.52 m, one runaway hands off at 130 | 99 % over on a neighbour rim |
| 1.4 Hz on 415 (the calibration's) | 0.29 / 0.64 / 1.35 s | 0.26 to 11 m | holds, 19 % over |
| 1.75 Hz on 415 | 0.21 / 0.33 / 0.69 s | 0.19 to 0.83 m | holds, 26 % over |
| 2.0 Hz on 415 | 0.14 / 0.22 / 0.46 s | 0.12 to 0.67 m, on the edge of ringing hands off at 130 | holds, 36 % over |
| 3.0 Hz on 200, damping 1.0 | 0.08 / 0.14 / 0.28 s | 0.11 to 0.34 m | 55 % over standing, 42 % at 30 km/h, 10 % at 60 |

A 0.02 step at 90 km/h took 0.34 s to 90 % on 2.0 Hz on 415 and never got there in the stage's hold on 1.4 Hz; at
130 km/h no design of the first four reached it in under 0.65 s, with no overshoot and almost no chatter. So at speed
every design has stability and smoothness to spare and is slow, and standing it is at its limit: one design for every
speed cannot be fast on the road and well damped standing. On the model wheels with less rim inertia it is worse (the
bare rim's standing pick, 1.75 Hz on 148, trails 0.92 s at 90 km/h): the stiffness the road needs is set by the game's
spring, not by the rim.

Measured and not built: letting the integral follow the game's spring whole (it is capped at the design's stiffness
in `_design`) brought 1.4 Hz on 415 to 0.13 to 0.33 m past at 90 km/h and rang hands off at 130 (9 to 12 wheel
reversals in a pulse, 3 to 9 m), from about 660000 per second of integral gain. A design with less design inertia at
a higher bandwidth (the last row) is fast everywhere on the road with modest kp and kd, and overshoots below 60 km/h.

**The road design (Lukas, 2026-11-11: "build it").** The servo has a second design, `la_road_inertia`,
`la_road_bandwidth_hz` and `la_road_damping` (`WheelTuning.road_*`, 0 is none). `WheelActuator.gains` works out both
designs' gains at the game's spring and moves from the standing design's to the road design's as that spring grows from
15000 to 34000 per lock (`road_from`, `road_full`: the test truck's 40 and 60 km/h), eased, for LA and the lane keeping
assist alike; a pulse, which starts at 60 km/h, is on it throughout. On the spring, not the speed, so a truck with a
weaker centring gets it later. Standing and below the blend the gains are exactly the standing design's, which is all
the wheel check and the design check ever vouch for. Strength and the ring guard scale it as they do the standing
design, the caps are the same, a wheel whose learner steps its design down loses it with the first step, and it has no
push ahead of its own (those are the rim's).

The calibration picks it in the model (`road.road_design`, after the standing design passed its check on the wheel;
`--no-road` skips it): bandwidths of 3.0, 2.5 and 2.0 Hz on 0.35 to 1.0 of the rim's inertia at damping 1.0 and 0.7,
those that trail at least a fifth less than the standing design at 90 km/h, fastest first. One is taken when a 0.02
step overshoots at most 25 % at 45, 50, 60, 90 and 130 km/h of the game's spring (and at 90 and 130 with a third more
delay) and lane keeping pulses on the blended gains stay within 1 m of a
line with at most 8 wheel reversals at 60, 90 and 130 km/h, hands off and resting, drifts of 0.4 and 1.2 m/s, the
target hold on as the app runs them (without it every design hunted round a still target and none passed). Where none
of the first eight holds, the one that runs least far past a line is taken if that is under half the standing design's
alone and it rings no more. Half a minute to two. Its chatter is printed and not judged (Lukas, 2026-11-12: the push
comes first).

Two calibrations on the R3 after it was built (2026-11-12) showed how much the standing fit decides. At 01:56 the
reading jitter read 11.2 ms (2.1 to 3.8 on five other runs): the model's chatter then failed every design above 1.0 Hz
and every road candidate, and the run saved 1.0 Hz on 964 with no road design, trailing 0.67 s at 90 km/h. The jitter
the model uses is now capped at 5 ms (`calibrate.JITTER_MAX_S`). At 02:41 it read 3.8 ms and a delay of 32 ms (18 and
20 before) and saved 1.95 Hz on 405, which trails 0.13, 0.22 and 0.45 s at 60, 90 and 130 km/h, as fast as before the
calibration at 90; no road candidate a fifth faster passed its step on that rim, which is no fault: `calibrate` now
says whether the standing design alone is fast enough or soft.

On the calibrated R3 (414, 20 ms) beside the standing 1.79 Hz it is 3.0 Hz on 145, damping 1.0, the fifth tried (3.0 on
207 and 2.5 on 290 rang hands off at 130 km/h, 15 reversals):

| | trails by, 60 / 90 / 130 km/h | pulses hands off | pulses hands resting |
|---|---|---|---|
| 1.4 Hz on 415 (saved on 2026-11-10) | 0.29 / 0.64 / 1.35 s | 0.30 to 3.75 m | 0.73 to 11.9 m |
| standing 1.79 Hz alone | 0.19 / 0.31 / 0.64 s | 0.16 to 0.40 m | 0.23 to 0.69 m |
| with the road design | 0.12 / 0.19 / 0.39 s | 0.11 to 0.33 m | 0.17 to 0.53 m |
| 2.0 Hz on 650 (before the calibration) | 0.09 / 0.20 / 0.29 s | 0.10 to 0.17 m, 5.16 m at 130 with a 1.2 m/s drift | 0.16 to 0.52 m |

On the slower model wheels nothing holds every case and the fallback takes a clear improvement: the R3 preset (374,
38 ms) 2.0 Hz on 261, 2.75 m at worst for 20.4 m on its standing 1.25 Hz alone; the bare rim (148, 32 ms) 2.0 Hz on
148, 10.5 m for 25.6 m. Those rims run wide at 130 km/h hands off on every design, and the real R3's fitted delay has
read anywhere from 12 to 36 ms from run to run, so which of these the wheel is decides what it gets. All of it is the
model: nothing here has run on a wheel, and the first drive on a new road design is the test.

What else the calibration does differently since (model only, not run on a wheel):

- **Chatter on the road gains.** The model's chatter limit and the check's slow ramps run on the gains the design has
  at 60 km/h of the game's centring (`fast_design(road_spring=...)`, `DesignCheck.road_spring`); the steps keep the
  standing gains, so overshoot and ringing are still judged at their worst. Standing chatter held 2.0 Hz back (172 for
  a limit of 150) where it reads 117 at 60 km/h and 53 at 90. Cost: LA below about 60 km/h can crunch on a design this
  lets through.
- **The middle.** Between the fastest design that passed the check and a faster one that failed, the middle gets one
  more run (the calibrated rim in the model: 2.0 Hz 40 % over, 1.6 Hz clean, 1.79 Hz tried).
- **The road report.** `calibrate` prints how far the wheel trails at 60, 90 and 130 km/h on the standing design
  alone and with the road design, and says when that is soft for lane keeping (over 0.35 s at 90 km/h). The game's
  centring is read from `la_saved_calibration` (read only), else the test truck's 122. `--save` writes the road keys
  with the rest, zeros when there is no road design, so an older one never stays behind.

## Adding a path source

The lane follower is one path source. A later one (lane departure prevention, collision avoidance with a wanted
travel path) plugs in at `LaneAssist.update(..., path=WantedPath(...))`:

- `offset_m`: where it wants the rear axle, metres right of the lane centre; the correction steers there
  instead of the learned centre alone (they add).
- `kappa`: extra rightward path curvature on top of the lane's, the feedforward for a swerve shape.
- `source`: its name, carried on `AssistOutput.source` and into the logs.

Path tracking (target angle, bumpless walk, override, hands-on check) and the wheel actuator (servo, caps, ring
guard) stay the same for every source, and so does the calibration. What a new source has to decide for itself,
and is not done here: whether its own request may outrank the driver mutes (blinker, override, evade), how fast
`target_rate` may move for it (0.2 of lock per second covers the lane's own kinks; the servo itself follows
faster targets), and a cap on the offset it may ask for. It must never weaken the driver override
or the caps.

## The driver always wins

| Reason | Meaning |
|---|---|
| `override` | the wheel held still (under 0.05 of lock per second) more than 0.015 of lock from the target for 0.3 s, or held still for 0.5 s with the servo pushing at 80 % of its cap net of the game's force (firm hands holding against it). In a bend the servo carries the game's centring on its own near its cap; counting the servo alone handed three bends back on the fifth drive. A lane keeping pulse also stops pushing harder and then hands back as the driver's estimated push against it grows ("The driver's push"; `core/lane_departure/README.md`, handing back): a driver gripping the wheel is paying attention (Lukas, 2026-11-02). Fades out in 0.4 s, holds 2 s, and rejoins only within 0.6 m of a lane centre, so a lane change without a blinker is not pulled back |
| `blinker` | either lamp or hazards, held 2 s after it goes off |
| `evade` | a hard wheel move (0.15 of lock in 1 s, or 0.6 per second); 3 s |
| `hands_off` | the wheel follows the target with no tremor: the servo error's ripple (its deviation from a 1 s mean, rms over 3 s) under 0.0008 of lock. Rumble after 10 s, lets go after 15 s, back as soon as the wheel moves at 0.02 of lock per second |

A released wheel swinging past the target is not an override: it is moving. A wheel on its way to the target
with the servo at its cap is not one either. The servo's damping resists a fast hand by at most 1500 of 10000.
The hands-on threshold is a placeholder with no measurement behind it; the 10 s summary logs the ripple's median.

## The driver's push

`hand_push.py` estimates the driver's torque on the rim every tick from what the plugin reports: what the wheel's
motion needs (`la_wheel_inertia` times its angular acceleration, from the plugin's wheel rate) less what the game
and the assist push (the game's force with the plugin's 0.1 s low-pass undone, the assist applied, any cancelled
force). Counterclockwise positive like the wheel, in the plugin's units out of 10000, smoothed at 0.1 s. A lane
keeping pulse yields to it (`override`, below); LA itself only logs it (Lukas, 2026-11-02: a pulse pushed back
harder than wanted while avoiding an accident).

Most of it is the driver holding the wheel against the game's centring: in standby on the drive of 2026-11-02 it
was the game's force negated, up to 3300 in bends. So during a pulse what counts is the change since the pulse
began, toward the line it saves (`against`), set against the pulse's own push away from that line (`push`).
Passive hands, the wheel's friction and its damping can at most cancel the push; only a driver steering toward the
line goes past it (`beyond`). Re-estimated from that drive's `lka_debug.csv` (38 pulses):

- four of the five pulses that turned the wheel home by 0.03 lock or more met at most 560 against; the fifth, a
  lane change at 2.4 m/s that ended `out`, met 2680 against its 3000;
- the override at 20:59:54 met 2470 against 3000: the driver held the wheel still at the servo's cap for 0.4 s
  before the override noticed;
- five went past the pulse's own push by 930 to 1730, 0.5 to 1.3 s in: the driver steering through it (one of
  them 1.5 m over the line already). Pulses whose wheel moved under 0.01 lock read up to 500 past it, hence the
  500 the logs count from.

What it does not know: the wheel's own damping and friction read as hand (they only resist motion, so never past
the push); the wheel rate is low-passed at 0.04 s in the plugin and read at 20 Hz, so fast swings are off by a few
hundred units; the plugin's own rumble shakes the rim (`rumbling` in the CSV). In the model
(`tests/la/test_hand_push.py`) a free wheel reads under 40 % of a ramped push on the way and under 100 once
settled, a held wheel within 5 % of the push. `python tools/la_torque_test.py` checks it on the real wheel with the truck
standing: hands off it should read about 0, the wheel held still about minus the torque. If it is too rough, the
plugin could estimate it at the game's FFB rate from the raw force (offset 90 of the block is free).

`la_debug.csv` and `lka_debug.csv` carry `hand`, `steer_accel`, `pulse_push`, `hand_against`, `servo_only` and
`rumbling`. Each LKA pulse end line says how hard the pulse pushed, how hard the driver pushed against it and when
the driver first went 500 past the pulse's own push. `python tools/lka_pushback.py` lists every pulse of a drive,
re-estimating files written before the columns existed.

## Smooth on and off

Every authority change is a ramp under a smoothstep: in over 2 s and only after 1 s without any mute,
out over 1 s on a gate and 0.4 s when the driver acts. The servo's gains, its cap and any cancel all follow that
ramp, and the plugin fades the servo in over 0.3 s and out over 0.2 s on top.
Forks, merges, route ends and junction lanes count after lasting 0.5 s. Bends taper the authority from 1/200 to
none at 1/100 (`bend`, `bend_sharp`). If the lateral correction itself keeps flipping sides while engaged (three
times in 8 s past 0.3 m/s^2), its slope halves and comes back over 60 s (`weave`); its 2 m/s^2 ceiling and the
stop before the line stay, so a cut never weakens a big correction (drive 6 sat on a halved ceiling into the next
lane).

## Caps

The servo 3000 of 10000 by default, and up to 6000 (the plugin's hard maximum, the whole assist 6500) only when
`la_servo_cap` asks for it: config only, LA only, the lane keeping assist never reads it (Lukas, 2026-11-06, to hold
corners past what 3000 can). Its damping half its cap (1500 at 3000) and its force slew 150000 per second (Lukas,
2026-11-06, for fast moves; 1500 and 30000 before), its integral its cap, any push ahead within the cap, any
cancelled game force 6000 and never more than the game's own, whatever MonoCruise sends. A stale heartbeat (0.25 s), a steering channel
older than 0.15 s, or a game that is not driving fades everything out, the cancel included.

## Model results

`tests/la/test_controller.py`, closed loop at 90 km/h on four rims (the design inertia, twice it, and
four times lighter; viscous friction 0.1 to 0.3 of critical), with 350 units of dry friction unless noted,
starting 0.5 m off the line:

- hands resting, R 300, the game's force on the wheel angle or on the truck's cornering: never out of the lane
  on any rim, within 0.4 m after 20 s;
- hands off, the three rims near the design, R 800, 300 and 200: never out of the lane, no false override;
- a 0.4 m/s drift or a 0.8 m offset on a straight: caught, overshoots at most 0.45 m, within 0.45 m;
- a steering gain 15 % off either way, hands resting: settles in the lane; on the way in it can graze the
  line by up to 0.1 m;
- the rim four times lighter, no hands, no friction: rings, the guard cuts it, the shake stays under 0.12 of lock;
- R 150 to 300 on routes rebuilt the map's way every tick, 10 m vertices, no dry friction: drift jitter under
  0.04 m/s tick to tick and centred within 0.05 m. With 350 units of dry friction and resting hands the wheel
  sticks and slips and the truck sways about 0.15 m every 4 s in a bend, fit or polyline alike.

What the model does not know: the real rim inertia, friction and firmware damping, the game's real frame rate
and delay, and how hands load a wheel. These are the model's numbers, not the game's.

## Where the driver keeps the truck

The map has no paint, and on straights the driven line sat over 0.5 m off the map's lane centre in 8 of 21
clips. The centre the assist steers to follows the driver: a 20 s low-pass of the offset, learned only
while steady, on straights and gentle curves (under 1/1000), without blinker, swerve or lane settle,
bounded at 0.3 m. It starts over on `no_map`, `off_route` and `stale`; a lane change keeps it.

## When it is off

| Reason | Meaning |
|---|---|
| `no_data`, `no_map`, `paused`, `stale`, `reverse` | as in lane departure |
| `speed` | below `la_min_speed_kmh` (60), disarm 10 below |
| `tmp` | TruckersMP map active and `la_tmp_enabled` off (the default: moderation cleared the rumble, not steering) |
| `aeb`, `brake` | AEB cue, brake 0.7 or more |
| `off_route` | more than three half lanes from the route |
| `short_route`, `branch`, `merge`, `route_kind` | after 0.5 s: route ends, forks or joins within `max(40 m, 2 s)`, or a junction lane |
| `bend`, `bend_sharp` | authority tapered by the bend, none from 1/100 |
| `calibrating` | no usable `k_a` or `g_s` yet |
| `plugin` | no version 4 plugin, not hooked, dead, or its steering channel stopped |
| `lane_settle` | first 1.5 s on a lane |

`la_follow_all` (test only) drops `short_route`, `branch`, `merge` and `route_kind`, and tapers bends by
the lateral acceleration the lane asks for now and 0.4 s ahead (`v^2 * kappa` from 4 to none at 6 m/s^2), not
by the tightest bend in the next seconds. That demand is held while it eases (falling 1 m/s^2 per second), so the
authority does not bounce with it inside a bend. A lane switch (`lane_settle`) keeps the authority and steers on the bend
alone for 1.5 s instead of letting go. At a fork it follows whichever branch the route picked, the one needing the
least change of curvature that the truck is following, so it can pull toward a branch the driver did not mean;
steering against it or a blinker hands back as usual. Every driver mute, the speed gate, AEB, brake and
`off_route` stay.

`la_always_on` (test only) is for judging the servo through tight bends without it letting go: it implies
`la_follow_all`, removes the bend limit, and never hands back to the driver: no `override`, `evade`, `blinker` or
`hands_off`, and no hands-on rumble. The speed gate, `no_map`, `off_route`, `stale`, `paused`, `reverse`, AEB and
the brake gate stay, and so do the caps, the ring guard, the plugin's buzz guard and the weave guard. The servo is
capped at 3000 of 10000, so a hand on the rim still overpowers it; with the switch on it just keeps pushing. A
line crossed keeps the lane it left as the reference (see the lane under the truck).

## Config (config.json only)

| Key | Default | |
|---|---|---|
| `la_enabled` | false | on/off |
| `la_strength` | 1.0 | 0 to 2: scales the servo's stiffness (damping follows). 0 is no servo |
| `la_wheel_inertia` | 0 (650) | the rim with hands the servo is designed on; lower for a light wheel that buzzes, higher for a heavy one that feels slow |
| `la_cancel_share` | 0 | share of the game's steady FFB force removed while it steers, 0 to 1; 0 keeps the game's feel |
| `la_ff_inertia`, `la_ff_damping`, `la_ff_friction` | 0 | the plugin pushes the rim ahead: its inertia times the target's acceleration, its damping times its rate, its dry friction the way it moves. Take them from the wheel check's fit, then confirm with the speed benchmark (`speed --push-ahead`), which prints them when the run was clean. LA and the lane keeping assist both use them: they are the wheel's |
| `la_servo_lead_s` | 0 | reads the wheel and the target that many seconds ahead (the wheel's delay), at most 0.08. Buzzed ramps in the model; leave it at 0 |
| `la_ff_plan_s` | 0 | the push ahead moves the rim on the target's motion that many seconds from now (the follower's own, propagated), so it starts and stops on time; the wheel check's delay (26 ms on the R3). At most 0.08 |
| `la_ff_breakaway` | 0 | the ceiling the friction push climbs to while the wheel stands, for a wheel that needs more to start than to slide (the old wheel: 311 to 526 against 305); used when above `la_ff_friction`, needs the plugin's status bit 2048. The wheel check writes it for a sticky wheel; LA and the lane keeping assist both use it |
| `la_servo_cap` | 0 (3000) | LA's servo cap out of 10000, 1000 to 6000. The game pushes back about 560 units per m/s^2 of cornering, so 8 m/s^2 needs about 4500 plus friction. A hand overpowers it; with `la_always_on` nothing hands back |
| `la_bandwidth_hz`, `la_damping` | 0 (2 Hz, 0.7) | the servo's design on that rim, from the wheel check; bounded to 0.3 to 3 Hz and 0.3 to 2 |
| `la_road_inertia`, `la_road_bandwidth_hz`, `la_road_damping` | 0 (none) | the second design, for the road, from the wheel calibration ("Fast on the road"): blended in as the game's centring grows, all of it from about 60 km/h; LA and the lane keeping assist. 0 on the inertia or the bandwidth is none |
| `la_damping_first` | false | damping first under the cap (`plugins/ffb/README.md`); needs the plugin that knows it. LA and the lane keeping assist both read the three |
| `la_min_speed_kmh` | 60 | arm speed |
| `la_force_per_ms2`, `la_kappa_per_steer` | 0 | 0 learns them; any other value is used as is, sign included |
| `la_saved_calibration` | {} | written by MonoCruise: the last live calibration, used at the next start once the drive confirms its sign (Calibration) |
| `la_wheel_learning` | true | learn the wheel from normal driving and park the servo at a still target (Wheel learning). LA and the lane keeping assist; false runs the servo exactly as configured |
| `la_wheel_learned` | {} | written by MonoCruise: what the wheel learner knows, kept for the next start; delete it to learn afresh |
| `la_tmp_enabled` | false | allow on TruckersMP |
| `la_follow_all` | false | test: keep it on through junctions, forks, merges and lane switches |
| `la_always_on` | false | test: never hand back to the driver and no bend limit; hands-on check off |
| `debug` | false | also writes `la_debug.csv` and `la_ffb_debug.csv` (Logs) |

## Logs

`LA on/off`; the settings when they change (and at start while `la_follow_all` is on); the
calibration on every state change (at most once per 10 s) and every 60 s while usable; one line per yield,
swerve, weave cut, ring cut or hands-off with offset, centre, drift, target, wheel and the hands ripple; every
10 s the seconds per reason, applied torque, the game's own centring and what is left of it with the assist
added (while assisting), the spring at the current
speed, learned centre, weave gain, the servo's ring scale (`stiffness`), the bend scale, the plugin's buzz scale,
the game's FFB rate and the hands ripple median. No coordinates.

With `debug` on, `la_debug.csv` in the project folder gets a row every 50 ms while the assist is not
idle (and 2 s after): reason, authority, offset, centre, drift, bend feedforward, correction, target and its
rate, the plugin's wheel angle and speed, the game's force, the servo and the cancelled force, the integral,
gains, cap, every guard's scale, the bend scale, the FFB rate, whether the lane fit was in use, and the plugin's
FFB and steering frame counters. No coordinates, local only, rotated at 50 MB.

`la_ffb_debug.csv` (`lka_ffb_debug.csv` for the lane keeping assist) gets every constant-force update of the
plugin's trace block (`plugins/ffb/README.md`, Trace block) while the game drives, the assist idle or not: the
game's own force before anything is added, what went to the wheel, the servo's part and the wheel angle it used,
each with the 20 Hz tick's speed, reason, target, gains and the calibration, and the gain and spring and damper
coefficients of any condition effect the game made. It is there to fit the game's own centring, as each user has it
set up, at the rate the servo works at: the 20 Hz `game_force` is low-passed at 0.1 s and the calibration's spring
(`Gains.spring`, cornering force only) read 25 times too low under 25 km/h on the 2026-11-06 drive. The log says
once which effects the game made (`LA FFB trace: the game made ... effects`). Local only, rotated at 300 MB
(a few hours of driving).

## Testing

1. Build and install the version 5 plugin (`plugins/ffb/README.md`); the game must be closed to replace it.
2. `python tools/la_torque_test.py` holds +800 then -800 with no lane data: note which way each pushes.
3. `"la_enabled": true`, drive bends at 40 km/h or more until `LA calibration` shows the cornering
   and steering fits ready (about a minute). A positive spring must match "+800 pushed left".
4. On a road with gentle bends at 60 km/h or more, hands resting lightly: the wheel should turn into bends
   by itself and hold the lane. Steer against it, use a blinker, and let go for 15 s to see each hand-back.
   A `ring cut` line or a buzzing wheel means the servo is too stiff for that wheel: lower
   `la_wheel_inertia` (or `la_strength`).

## Bench

`python tools/la_servo_bench.py capture` tests the servo on the standing wheel, with no lane data and no game restart per
change: gains, target and rate are runtime fields of the command block, so only a change to the plugin's C needs the
game closed. Truck standing (driving, not paused), hands off the rim, LA, LKA and the rumble idle (they would
overwrite the requests). It finds which torque sign turns the wheel left, then runs each design through steps
(0.01 and 0.02 lock), two ramps and two sines, sent at 20 Hz with the target's rate as MonoCruise sends them, and
prints rise time, overshoot, settling, ripple, tracking error and lag per run. The recording lands in
`tools/la_bench_out/` (gitignored). Steps stay small because standing there is no game spring to catch a ringing
wheel: past 0.4 lock the stage ends and the servo is asked off.

- `report FILE` prints the same numbers again. `capture --simulate` is the whole flow on a model wheel, a rehearsal.
- `fit FILE` fits the rim (inertia in `la_wheel_inertia` units, viscous damping, dry friction, spring, delay) to the
  recording with the plugin's servo mirrored from `tests/la/servo_sim.py`. Runs whose servo reached 90 % of its cap are
  fitted on the size of their swing, not the wheel angle (a pinned servo with a delay is a relay, its limit cycle
  chaotic in phase but not in size), and that is what pins the delay: the small runs fit 4 to 20 ms equally well.
  The delay is scanned first. `--no-ring` leaves the pinned runs out.
  On a model rim the fit gave back inertia, damping, friction, spring and delay within 5 %; on the real wheel the
  numbers are the first measurement of what "What the model does not know" lists.
- `predict FILE --bandwidth 1.5 2 2.5` runs other designs on the fitted rim, so a change is tried in the model first
  and only then captured on the wheel.

First capture on the Moza R3 (2026-11-05, three takes that agree): the rim is light, inertia about 290 against the 650
the servo is designed on, damping 640 per lock/s, dry friction 180, the game's standing spring 1500, 9 to 20 ms of
delay. The 1 Hz design is clean but slow (a 0.02 step takes 1.2 s: the friction holds the wheel until the integral
builds). The 2 Hz design rings: from a 0.02 step the servo pins at 3000 and the wheel swings 0.3 lock either side
at about 1 Hz; the model on the fitted rim does the same. Cause: the damping force is capped at 1500, reached at
0.13 lock/s of rate error, and a pinned P hides it, so a light rim that overshoots is never braked. Two fixes in the
model, both on the fitted rim: `la_wheel_inertia` near 300 (config only), and damping first (`--law
damping-first`, the plugin's `servo_flags`): rise 0.075 s, overshoot 22 %, settled in 0.3 s at the 650 design. Damping
first also cut the ripple of a servo pinned against 57000 per lock of game spring under force noise to a third; the
inertia alone did not, and pinned servos are what LA's vibration in sharp bends at speed looks like. The noise is
an assumption: confirm on a drive.

Recording notes: the wheel has no centring standing and rests where it was left, so the bench walks it back to its
starting angle before each stage (`go_home`) and aborts only if that start is 0.25 lock off centre. `--note` keeps
the game's steering settings in the file; take every run with the settings you drive with.

### Wheel check

`python tools/la_servo_bench.py check` (truck standing, hands off the rim, the same setup as `capture`; `--simulate
r3|bare|geared` rehearses it on a model wheel) fits this wheel and writes the `la_*` design keys for it. About two
minutes on the wheel:

1. A ladder of damping-first designs at 0.7, 1.0, 1.4 and 2.0 Hz (cap 2000, designed on 650) runs a 0.01 and a 0.02
   step each and stops at the first that rings (an abort, a buzz cut, 100 % overshoot, ripple, bad tracking). The
   first rung also runs the slow ramp and sine. The wheel is never asked for more than the rung before it held.
2. The rim is fitted on those runs (the swing size of a run that rang pins the delay).
3. A design is chosen for it in the model: damping first, designed on the fitted inertia or up to 3 times it (a
   stiffer, better damped servo: on the R3 a design on 650 was the fastest that held, one on the fitted 300 half as
   stiff), the fastest bandwidth and damping ratio that holds up on the fitted rim (overshoot under 40 %) and four
   neighbours (delay 1.3 and 0.8 times, inertia 0.75 times, inertia 1.6 times with hands) with no abort, no buzz cut and
   overshoot under 60 %. `la_wheel_inertia` is then the design inertia, which is also the rim-with-hands the driver's
   push is estimated on.
4. That design is run on the wheel (step, slow ramp, slow sine, fast ramp, fast sine). If it rings or overshoots more
   than 20 % up or 35 % on the swing, that run goes back into the fit (a ring is what pins the delay) and a new design
   is chosen, up to three rounds, never a design as fast as one that rang; then the gentlest rung is tried. If none
   holds, the check says so and writes no keys: leave the defaults.
5. A design that held is tried 15 % faster, up to four times, and the fastest the wheel accepts is kept. Without this
   the bare R3 stopped at a model-cautious 1.5 Hz (56 ms of fitted delay): two steps up cut the 0.02 step's rise from
   0.45 to 0.33 s and the follow-run lag from 180 to 80 ms with no overshoot. With simulated inertia on, 1.75 Hz held
   (48 % swing under the earlier 50 % limit, hence the tighter one) and 2.0 Hz pinned the servo with 270 % overshoot:
   the edge is a cliff, so the limits leave margin. A wheel that cannot be walked back to its start ends the check
   early; what was recorded is kept and reported.

With a plugin that pushes ahead on a plan (status bit 1024) the check designs with it, for whatever wheel it
measures (2026-11-06): it counts how often the game reads the wheel (the steering frame count, 60 Hz on the R3) and
fits the rim at that rate, designs with the push ahead from that fit (inertia, damping, friction, the plan its delay
early) on the fitted rim and its neighbours, a rim with half the friction among them (where a friction push over-pushes
and could hunt), and verifies on the wheel with a rounded 0.4 lock/s ramp added. When nothing holds in the model, as
on a wheel the fit cannot describe (gear teeth, backlash, a coarse encoder: the geared model wheel fit inertia 159 and
67 ms where it is 600 and 25), it tries push-ahead designs on the wheel itself, 0.5, 0.7, 1.0 and 1.4 Hz on 650 at 1.0
of critical damping, and keeps the fastest that holds (the geared model wheel held to 1.4 Hz). It then writes the
`la_ff_*` and `la_ff_plan_s` keys too. The model wheels on 2026-11-06, read at 60 Hz: the R3 1.2 Hz, the bare R3 1.25
Hz, the geared wheel 1.4 Hz chosen on the wheel, all with the push ahead.

The old wheel (a Thrustmaster, 2026-11-06) got nothing from that check: every rung and every design rang. It needs
about 460 to start and slides on about 200 (the R3 starts at about 106, under its fitted friction), its angle reads
in steps of 0.0006 lock (the R3's 0.00002), and its rim is light. Each slow move stuck, jumped 5 to 12 mlock once the
push built up, and the damping, braking the jump late, kicked the stopped wheel back until the buzz guard cut it. The
fit read inertia 152 and damping 1457 there, because it matches angle traces and a limit cycle's trace is matched best
by a heavily damped rim, and the model then held 2 Hz where the wheel rang at 4.3 Hz. Since then:

- the breakaway (the median largest force in the readings before each start of a still wheel) and the angle's steps
  are read off the wheel and given to the model (`plant.breakaway_of`, `quantum_of`); a rim is sticky when it starts
  at over 1.5 times its fitted friction;
- the fit is held to the wheel's verdicts: the runs are run again on the model, and if it does not ring where the
  wheel rang and hold where it held, rims near the fit (a quarter and half the inertia, half the damping, 15 ms less
  or 10 ms more delay) are tried and the one that agrees on most runs wins, then the closest peak force
  (`check.match_verdicts`). A sticky rim is always scanned: many rims agree on a few stick-slip runs (a short fit of the
  sticky model wheel read 132 and 57 ms, the scan 66 and 32 ms, the truth 55 and 30). On the old wheel's runs rims of
  40 to 70 agreed on all ten. A model that agrees on under 80 % of the runs is not used: designs go on the wheel;
- a wheel that rings on the first rung climbs again on a design of 160 (the bare R3 rang at 1.0 Hz on 650 too);
- for a sticky rim the design search adds the breakaway push (`la_ff_breakaway`, `plugins/ffb/README.md`): the
  plugin's push climbs to a ceiling of the sliding friction plus 1.1 or 1.4 times the gap to the highest start the
  slide probe saw (1.2 on the wheel ladder), the slow ramp among the stages, a neighbour with a quarter less breakaway (where the push
  over-pushes at every start) and design inertia up to four times the rim; its neighbours may track up to 30 mlock rms
  (they are there to catch rings: the fast ramp tracked 18 to 24 on the rim with hands); the fitted rim still 15;
- the designs tried on the wheel itself are on four times the rim (150 to 650), with the breakaway push, and on twice
  it when even the gentlest of them rings;
- a wheel that stands still over 45 % of the climb's slow moves (the old wheel 53 to 85 %, the real R3 2.5 %; the
  R3 and bare model wheels 29 to 31 %, the geared, sticky and old-wheel ones 67 to 78 %) gets the slide probe
  (`tools/la_bench/probe.py`), and only then is the breakaway given to the model: open-loop
  torque ramps up at 400 per second until the wheel starts (its breakaway, read 5 to 10 % high on a light rim, more
  on a heavy one), then holds 30, 50 or 70 % of that for 0.4 s, both ways round, the wheel walked home between: still
  sliding at the end, the sliding friction is below the hold. The second run of the old wheel (2026-11-06) agreed as
  well with a rim sliding on 150 (which has a design that holds) as with one on 358 (which has none), because the
  wheel barely slid in it; on those two model rims the probe read 72 to 144 and 332 to 474. The scan then takes its
  sliding frictions from that bracket, and of the rims that agree as well the design is made on the lowest friction
  and must hold on the others (an under-push leaves the rest to the feedback, an over-push hunts at every stop);
- the bench waits when the game's force stops while its steering goes on (the plugin's count of force updates stands
  still for 0.5 s: another window in front of the game, 2026-11-06, left the servo pushing nothing for whole stages),
  and runs that stage again once the force is back; after two minutes, or three stops in one stage, it gives up;
- a stage or slide probe trial whose wheel readings stalled for 0.2 s, or came in bunches (over 5 % of the gaps 2.5
  times the median), runs again too, and three in a row stop the check with nothing saved. The old wheel's 18:01 run
  (2026-11-06) read in bursts 36 to 40 ms apart, with stalls of 0.3 to 2.4 s, while another program starved the PC
  (MonoCruise's own threads froze on file opens in the same minutes): the fit moved to an inertia of 233 and a spring
  of 3200, and the designs it chose rang. The rule flags all 9 of its stages and none of the 175 in every other saved
  run, R3 and old wheel alike;
- the check stops at once when another program writes to the servo (`RumbleChannel.foreign`: a heartbeat in the
  block that is not its own). MonoCruise with lane departure on ran beside that run, and its thread, restarted by the
  watchdog, turned the bench's servo off mid-stage: a fresh channel writes "servo off" once;
- slide probe trials that disagree (a hold that stopped the wheel above one it kept sliding at: 276 stopped, 244 slid
  at 18:01) bracket both, `mixed` in the probe, and the design is made on the lowest friction as before;
- the clean 19:16 run (2026-11-06) showed the fixed breakaway push failing: the design it chose (0.5 Hz on 150,
  push 371) sat at 400 to 440 for 1.5 s, the integral adding 27 a second, and the wheel never moved; 650 moved it and
  stick-slipped into the buzz guard. Its starts ran 348 to 516. The model wheel now draws each stop's breakaway
  within `breakaway_spread` (the probe's `start_spread`) and reproduces both; the plugin's push climbs instead (above);
- the damping probe (`probe.damping_probe`, after the slide probe) starts the wheel on the ramp and holds 1.05, 1.2
  and 1.35 of that start for 0.3 s each, both ways; push = damping * rate + spring * angle + friction over the steady
  slides. On the old wheel's model the push ahead holds only with a damping 0.7 to 1.2 times the real one, and the
  closed-loop fits read 250 to 2473 on the wheel. It reads the model's 595 at 561 to 569 and 1200 at 1158 to 1286; a
  wheel that slides too fast for six steady slides reads nothing and the fit keeps it. A measured damping is held in
  the fit and the verdict scan, and so is the spring the game reports on a wheel with a breakaway (the model's 1000
  fitted at 8064; the old wheel's game read 533 to 692 against fits of 889 to 1163);
- a step that stops more than 60 % short no longer counts as held (19:16 stopped 88 % short and passed; the geared
  model stops exactly halfway on its 0.002 steps). On the old wheel's model the check now verifies 0.5 Hz on 150,
  push ahead on the measured damping, ceiling 529;
- the 21:45 run with the climbing push (2026-11-06, "almost smooth") was cut by the buzz guard on both steps: within
  1.5 mlock of the target the rest push (the friction toward the target) flipped ±250 and the wheel hunted there at
  4 Hz, though it started there on about 250 (its starts from rest ran 401 to 480), and on the swing the push climbed
  only halfway 4 mlock short and stalled. With the breakaway push the plugin now pushes nothing within 1.5 mlock
  (`FFB_BREAK_HOLD`) and climbs all the way from 3 mlock. The model cannot show that hunt (its wheel restarts only on
  its drawn breakaway, never on 250), so only the wheel can confirm it. The damping probe read 424, spring 330 and
  sliding 435 there: with the game's measured spring (about 690) its slides took 500 to 545 at 0.2 to 0.46 lock/s
  whatever the speed, more friction than the 305 it creeps on;
- the model wheel read the angle at 40 Hz where the game reads 60 whenever the FFB rate was a little over 120 (two
  ticks of 120.6 Hz fall 0.1 ms short of 1/60 s): every fit and design on a real recording (120.4 to 120.8 Hz) ran on
  a slower wheel than the real one, and a design that held at 120.0 rang at 120.6. It now reads within half a tick;
- the walk home does not start for an offset under 0.03 lock (the stages run from where the wheel rests), and a walk
  that does not settle goes once more on a gentle design (0.5 Hz on 160): its servo is designed on 650, and on the old
  wheel it hunted and let go 0.3 lock off centre, which ended three checks;
- a design that holds every stage but the 0.4 lock/s ramp the push ahead adds is kept when nothing holds them all,
  with a note and not tried faster: LA's own target moves at most 0.2 lock/s. On the model rim sliding on 358 the
  check found nothing before; now 0.5 Hz on 333 with the push ahead. On the one sliding on 150 it verified 1.25 Hz
  with the breakaway push at 383 (two 2.5 Hz designs rang first and went back into the fit).

On a model of the old wheel (`--simulate sticky`: inertia 55, sliding 200, breakaway 480, steps of 0.00058, 30 ms) the
check now reads the breakaway at 501, fits inertia 65, agrees on 6 of 6 runs and verifies a 2 Hz design with the
breakaway push at 301: the slow ramp tracks 4 mlock rms (the gentle rung 15, cut by the buzz guard), a 0.02 step
rises in 0.07 s with 4 % overshoot. Expect small steps on slow moves all the same: each start overshoots by about
the difference of the two frictions over the loop's 40 ms, whatever the servo does. Untested on the wheel.

The output is responsiveness (the verified bandwidth and the rise of a 0.02 step), smoothness and the keys. Smoothness
is steppy when the breakaway force is over 12 % of the servo's push, the wheel stands still over 25 % of the time
the target moves slowly, the angle reads in steps over 0.0002 lock, or slow tracking is off by over 0.01 lock rms.
The one real wheel so far, the R3 with its simulated inertia on, reads 6 %, 2.5 %, 0.00002 and 0.0026: smooth. The
thresholds and the geared model wheel are guesses until a geared wheel has been through it. Designing on the bare
rim's inertia is on purpose: hands only add inertia, so a design that holds on the bare rim holds with them, and the
reverse is how the 2 Hz design rang. `la_wheel_inertia` also sizes the driver's push estimate, which this makes
truer. Nothing leaves the machine; the recording goes to `tools/la_bench_out/` like the others.

The wheel's own settings change the result a lot: on 2026-11-05 the same 1.25 and 1.5 Hz damping-first designs were
clean with the Moza software's simulated inertia on and rang hard with it off. Run the check again after changing
any wheel setting, and keep one set of keys per setting.

### Speed benchmark

`python tools/la_servo_bench.py speed --rim tools/la_bench_out/la_check_<stamp>.json` (same setup as `check`; the
design comes from the `la_*` keys in config.json, read only, or `--bandwidth`, `--inertia`, `--cap`, `--law`) measures
how fast the servo moves the standing wheel: ramps of 0.15 lock at 0.1 to 3.2 lock per second (followed while the
wheel trails by at most 0.1 s and overshoots the end by at most 10 %), then steps of 0.02 to 0.2 lock (time to 90 %,
the swing through zero, overshoot). Either ladder stops at the first run that rings. Beside each step it prints what
the fitted rim allows at the same cap: the bang-bang move a perfect servo would make, with and without the plugin's
force slew, and the rim's top rate. `--simulate rim` runs it on the fitted rim, `--simulate r3` on a model wheel.
`--push-ahead` pushes the `--rim` fit ahead and leads by its delay (`--lead 0` drops the lead), and prints the `la_ff_*`
keys when no step rang and it followed ramps of at least 0.4 lock/s (twice LA's own target limit; a step's overshoot is
the feedback's, the push ahead does not act on a step). A ramp reaches its rate over 0.1 s (`--corner`), or half its move at the fastest, and
is also sent where its acceleration changes: a commanding feature that smooths its target plans when it brakes, and
one that changes its acceleration between 20 Hz ticks gets up to 50 ms of late braking (with the push ahead, 18 %
overshoot at 0.8 lock/s in the model against 7 %). Standing only: at speed the game's centring and damping slow every number.

On the R3's fit (simulated inertia on, model, design 1.75 Hz on 528, damping first, cap 5500), 2026-11-06:

| plugin | ramps followed up to | 0.1 lock step to 90 % |
|---|---|---|
| slew 30000, damping 1500 | 0.2 lock/s (90 deg/s), the 0.4 ramp and the 0.02 step rang | rang |
| slew 150000, damping half the cap | 0.4 lock/s | 0.23 s (small steps 25 % over) |
| and the push ahead | 0.8 lock/s (360 deg/s) | 0.23 s |
| and the lead (26 ms) | 0.2 lock/s (the 0.4 ramp's corners buzzed) | 0.24 s (small steps 16 to 18 % over) |

The rim allows 0.18 s for that step at slew 150000. 2.0 and 2.5 Hz designs did no better on the new plugin.

On the wheel (2026-11-06, same design, layout 5) it did worse than that model: ramps followed to 0.2 lock/s with or
without the push ahead (which cut the 0.2 ramp's overshoot from 8 to 1 % and the 0.02 step's from 22 to 8 %), and a
ring near 4.5 Hz after fast stops tripped the buzz guard on the 0.4 ramp (with the push ahead), the 0.8 ramp and the
0.05 step. The model had read the wheel at every FFB tick; the game reads it at its frame rate, every 16.5 ms on the
R3. With `WheelParams.reading_hz = 60` the model reproduces those runs (the 0.8 ramp: error rms 0.025 and a buzz cut,
the wheel 0.026 and a buzz cut), and in it the 1.75 Hz design rings on the 0.1 step and the slowest half-lock ramp,
while 1.25 to 1.5 Hz with the push ahead held everything up to a half lock at 0.5 lock/s. The lead cut small steps'
overshoot there too but buzzed the 0.4 ramp at every value tried (4 to 20 ms).

The bench centres the wheel: before the first stage and between stages it walks it to the centre on a rounded move
(0.3 s corners, at most 0.3 lock/s, damping first and the run's push ahead, sent every 20 ms). Until then it walked
back to wherever the wheel rested at the start, which on 2026-11-06 was 0.09 to 0.16 lock off centre. Last in the
benchmark is half lock: out from the centre to 0.5 lock and back, then to the other side and back, on ramps at 0.5, 1
and 2 lock/s that speed up and slow down at 70 % of what the fitted rim does at the cap (8 lock/s^2 without a fit,
`--half-accel`), then at once. Its angle limit is the move plus 0.15. At once is slow on purpose: damping first gives
the damping up to half the cap, which holds a wheel flung at a far target to about half the cap over kd (0.34 lock/s
on the 1.75 Hz design); a feature that wants a fast big move sends it as a ramp. In the 60 Hz model half lock took
0.54 s at 1 lock/s (10 to 12 % overshoot) and 0.36 s at 2 lock/s (about 20 %): the stop starts a delay late.

On the wheel (2026-11-06, push ahead, cap 5500), 1.25 Hz was the first run with no buzz cut at all and Lukas's
smoothest so far: ramps followed to 0.4 lock/s (0.8 at 11 % overshoot), the 0.02 step slow (0.62 s to 90 %, 5 %),
half lock at 0.5, 1 and 2 lock/s done (7, 16 and 12 % overshoot, 0.96, 0.51 and 0.37 s). The same run at 1.75 Hz
buzzed on the 0.4 ramp and the first half lock, as the 60 Hz model said. What was left was the wheel's delay: slow
ramps trailed 25 to 30 ms and every stop started that late. The push ahead on the plan a wheel delay early
(`--plan`, by default the `--rim` delay with `--push-ahead`; `la_ff_plan_s`) fixes that in the 60 Hz model: at 1.25
Hz ramps followed to 0.8 lock/s (3 %) and half lock at 2 lock/s in 0.34 s with 4 % overshoot (0.36 s is the rim's
own limit at 5500), with 1.5 Hz close behind. `--simulate rim` now reads at 60 Hz (`--reading-hz`).

On the wheel with the plan (2026-11-06): ramps followed to 0.8 lock/s (1 %), half lock at 0.5, 1 and 2 lock/s with 2 %
overshoot each (0.96, 0.51, 0.36 s). Lukas felt three things left: stopping short, stopping late, and a slow creep
onto the target. The recording shows the wheel parking 3 to 15 mlock short of or past the target and taking 0.4 to 0.8 s
(some holds never) to come within 3 mlock. Standing, the game's own force on the wheel is its steering friction, about
550 while the wheel moves and about 200 held against the way it last moved once it stops (`c_ff_fricton` 0.4); P on
the 1.25 Hz design only beats the rim's 164 at 5 mlock, and the integral is slow to make up the rest. The plugin's
friction push now turns toward the target at rest (full from 2 mlock): in the 60 Hz model ramps came within 3 mlock in
0.33 to 0.70 s instead of 0.72 to 0.81, small steps in 0.21 to 0.28 s instead of never, half lock at 0.5 lock/s in
about 0.6 s instead of 0.75. Comparing the wheel with the target as it was when the reading was taken (the reading
is up to 16 ms old) did not help in the model and was dropped. The model's rebound after a fast half lock (30 mlock)
comes from the fit's standing spring, which on the wheel is the game's held friction: the wheel rebounded 10 to 27.

With the rest push on the wheel (2026-11-06) moves settled in 0.01 to 0.34 s after slow ramps and 0.3 to 0.8 s after
steps (from 0.4 to 0.8 s and never), but Lukas still found the endings slow and jittery. At the end of a 2 lock/s
half lock the wheel stopped 3 mlock short, rebounded to 11 short, swung 9 past and sat 6 past for over a second: the
damping braked again on the rate the reading showed late, after the plan had already braked on time, and the
integral had wound up to about 650 on the reading's lag during the move and kept pushing past. With the plan the
plugin now lags the damping's target rate by the reading's own lag and holds the integral while the target moves
faster than 0.3 lock/s (Plugin README, the push ahead); in the 60 Hz model the swing after a stop went from 36 to 21
mlock on the R3 and from 56 to 36 on the geared wheel.

What the bench cannot see is the game's own force at speed. The design counts the game's spring in (the servo adds
only what is missing), which is the part to check on a drive with `la_debug.csv`. The plugin caps the damping force at
half the servo cap (1500 at 3000, reached at 0.18 lock per second of rate error on the 1.75 Hz design), so damping is
at full strength only for slow, small moves: look at the overshoot of the 0.02 step first.

Unit tests: `tests/la/` (closed loop, every gate and mute, calibration, the thread end to end, the bench),
`tests/lane_departure/test_command.py` (layout against `plugins/ffb/ffb_cmd.h`), and the plugin's
`ffb_signal_test.c`.

## Not done

Road test 2026-11-06 (`la_always_on`, design 1.75 Hz on 528, cap 3000): the servo sat on its cap 4 % of the time above
40 km/h, but 21 % when the lane asked 5 to 6 m/s^2, 49 % for 6 to 8 and 57 % above 8, with the steering error
growing from 0.02 to 0.08 lock. The shake was not at the cap: the steering ripple 1 s after the servo pinned (0.0017
lock) was no higher than 1 s before (0.0020), long pinned stretches were slow (0.6 Hz), and the three ring cuts came
on straights at 38, 85 and 91 km/h with the lane asking under 1 m/s^2. Above about 86 km/h the game's spring
(50 to 114k per lock) is more than the design's total stiffness, so the loop is stiffer than the standing bench
tested; the damping is sized on at most 1.25 times the total.
The servo's cap against the game's centring: the game pushes back about 590 units per m/s^2 of cornering, so
the cap bounds the bend it can hold at any speed: 2500 held about 4.2 m/s^2 (in the model at the game's spring
at 130 km/h, an R 150 at 25 m/s ran off), the plugin's own 3000 (Lukas, 2026-11-01, for the tight-bend tests)
about 5 m/s^2. Past that the wheel stops on the cap and the truck runs wide; `la_cancel_share` is the other
lever. The cap is the force a driver has to overpower.
The plugin's buzz guard: on the fifth drive it fired 22 times, nearly all on straights with light hands, in
bursts that halved the servo three times within 0.6 s down to 0.13, so it stayed weak for 10 to 20 s (below 1
for 21 % of the engaged time). The target's rate zigzag explains part of it. Since drive 12 it counts only swings
of the wheel itself with the push flipping between them, and recovers in 4 s (`plugins/ffb/README.md`): the old
count of push flips alone tripped on a still wheel. Whether that also ends LA's straight-road cuts is not driven yet.
Measuring the wheel's inertia in the plugin instead of assuming it, the hands-on threshold from measurement,
departure prevention, persisting the bend scale between sessions, trailer, ATS validation,
TruckersMP (moderation), UI. Every number above is a starting point set by reasoning and the model, not by driving.
