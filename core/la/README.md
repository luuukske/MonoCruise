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
| `la_servo_cap` | 0 (3000) | LA's servo cap out of 10000, 1000 to 6000. The game pushes back about 560 units per m/s^2 of cornering, so 8 m/s^2 needs about 4500 plus friction. A hand overpowers it; with `la_always_on` nothing hands back |
| `la_bandwidth_hz`, `la_damping` | 0 (2 Hz, 0.7) | the servo's design on that rim, from the wheel check; bounded to 0.3 to 3 Hz and 0.3 to 2 |
| `la_damping_first` | false | damping first under the cap (`plugins/ffb/README.md`); needs the plugin that knows it. LA and the lane keeping assist both read the three |
| `la_min_speed_kmh` | 60 | arm speed |
| `la_force_per_ms2`, `la_kappa_per_steer` | 0 | 0 learns them; any other value is used as is, sign included |
| `la_saved_calibration` | {} | written by MonoCruise: the last live calibration, used at the next start once the drive confirms its sign (Calibration) |
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
