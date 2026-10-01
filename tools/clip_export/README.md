# tools/clip_export

Turns a recorded AEB clip into clean top-down footage for trailers, docs and
socials. Dev only, never shipped. It replays the same recorded geometry the review
tool draws, but it is a presentation layer: `python -m tools.aeb_agent render`
(the full debug view) stays the tool for diagnosing a clip.

```bash
python -m tools.clip_export 3e1f9422
```

```bash
python -m tools.clip_export 3e1f9422 --vertical --speed 0.5 --start 7.0 --end 10.0
```

```bash
python -m tools.clip_export 3e1f9422 --still 8.3
```

Output lands in `tools/aeb_corpus_run/exports/` (gitignored) unless `--out` is
given. Default is 1920x1080 at 60 fps, H.264 through `ffmpeg` on `PATH`; `--png`
writes a frame sequence instead and needs no ffmpeg. `--no-hud` drops the state
label for a clean plate. The window defaults to 3.5 s before the first warn or
brake through 2 s after the last one, but never starts before the threat fits the
shot (see Reframing). `--recorded` shows the live recording's AEB decisions instead
of re-running the current pipeline (see Decisions).

| File | Role |
|---|---|
| `timeline.py` | Replay resampled onto the video clock: pose smoothing, track splitting, threat choice, per-tick state, the ego trailer follower. No Qt. |
| `camera.py` | Auto-framing. No Qt. |
| `painter.py` | One `QImage` per frame. |
| `export.py` | Orchestration, the AEB re-run, the open-loop warning, and the ffmpeg pipe. |
| `cli.py` | Clip lookup through the agent index, and the contributed-clip gate. |

## What the debug view draws, and what survives

The look is flat on purpose: solid fills and solid edges, no glows, window bands or
vignette, and nothing fades in time. The one gradient is along each predicted path,
fading from the vehicle out to the far end, so distance reads at a glance. Every
visual change is a hard cut on a tick.

Kept, restyled:

- Ground dots (10 m and 100 m lattice), through the same `draw_ground_markers`
  as the debug view, drawn darker. They are the only world-anchored thing on
  screen, so they are what makes speed, braking and turning readable.
- Ego body and predicted corridor. Ego always points straight up, as in the debug view.
- Traffic bodies as plain rounded rectangles, each with its predicted path in one
  standard gray. Parked vehicles have a zero-length arc and draw none.
- Threats: any vehicle in `colliding_ids`, path and body in amber, red on ticks where
  AEB brakes.
- One HUD element: a state label in the top-left, the debug view's corner, reading
  `STANDBY`, `AEB WARN` or `AEB BRAKE` from the tick. It is always shown and
  sized for the longest label, so it never resizes on a state change.

Dropped, because they are internals that read as noise or as bugs to a viewer:

- Range rings, crosshair axes and their metre labels: meaningless once the zoom moves.
- Per-vehicle text (AI/TMP tag, speed, `kin:` source, ACC score).
- The AEB HUD text dump (speed, tracked, suppressed, TTC/TTB counts) and the legend.
- The ACC panel, ACC lead and candidate colours, trail arcs, the road-model line.
- Suppressed, braking-worsens and evasion-filtered colour classes, and the evasion arcs.
- The hit cross. In replay it sits on the threat's own position, so it repeated the highlight.
- The heading tick on vehicle bodies.

Tried in the first pass and removed after review: a speed readout, an AEB brake bar,
cab window bands, glow halos, a vignette and brake tint, vehicle
fade-in and fade-out, eased colour transitions, and a landscape layout with ego driving
left to right.

## Reframing

The debug view is ego-locked at a fixed 5.5 px/m with ego at 75% height, so most of
the frame is empty road. `camera.solve` instead frames a subject box per frame and
smooths the result offline:

1. **Subjects.** Ego, its trailer and a lookahead point
   (`max(16 m, 1.6 s of travel)`), always. The primary threat counts in full from its
   **entry**, the first time ego and threat together fit the widest allowed shot
   (170 m on the short axis, inside the margins), until 1.5 s after AEB last tracked
   it, when it ramps out. The auto window never starts before entry, so the first
   frame already holds both and the camera only ever zooms in toward the event. The
   first version framed ego alone and then zoomed out to find the threat, which read
   oddly; on four test clips the auto window now accumulates zero zoom-out. With an
   explicit `--start` before entry the threat ramps in over 1.5 s from entry instead.
2. **Orientation.** Locked to ego's smoothed heading, so ego points straight up and
   a turn swings the world around it. The frame centre is smoothed in ego's own frame
   for the same reason: smoothed in world axes it lagged the rotation and slid ego
   sideways through every turn.
3. **Zoom.** Log-scale, clamped to a short-axis span of 34..170 m. Margins are 14%
   horizontal and 10% vertical. A 0.8 s sliding
   minimum runs before smoothing, so the camera widens ahead of the need and
   tightens after it.
4. **Containment.** The smoothed centre can drift off a fast subject, so a second
   pass takes the zoom each frame needs to hold every subject inside the frame at
   that centre and folds it back in. Threat points enter this pass pulled toward
   ego by `1 - weight`. An earlier version added them as a step at weight 0.35,
   which put a one-frame zoom kink of 0.046 (log ppm per frame squared) into
   `3e1f9422`; the continuous pull brought it under 0.001.

Every smoother is a Gaussian-weighted local linear fit (`timeline.local_linear`), not
a kernel average: it passes a constant velocity through unbiased where the kernel
goes one-sided at the ends of the window, which would otherwise park the camera
metres ahead of a moving ego on the first and last frames.

The camera looks ahead in time. That is framing only: highlight, colours and the
state label are all driven by the tick nearest each frame, so nothing on screen
claims AEB knew something before it did.

### Heading-up costs zoom in landscape

Forward depth lands on the short axis of a 16:9 frame, so in-lane events frame
small. Measured mean zoom over the tracked span, 1920x1080:

| Clip | Heading-up, 17% vertical margin | Heading-up, 10% | Ego left-to-right, 17% (removed) |
|---|---|---|---|
| `3e1f9422` crosser | 13.2 px/m | 16.0 px/m | 23.9 px/m |
| `66dffd1a` head-on | 8.8 px/m | 10.6 px/m | not measured |

The 10% vertical margin is what the state label moving to the corner paid for. For a
rear-end or head-on clip, `--vertical` puts that depth on the long axis.

## Motion smoothing

Ego pose comes from the telemetry poll and traffic from the radar buffer, both
quantized to 60 Hz physics ticks, so a ~30 Hz clip advances 3 ticks then 1 tick
while its timestamps stay evenly spaced. Drawn per tick, a straight 90 km/h drive
steps 1.26 m, 0.42 m, 1.26 m.

Each entity's x, z and unwrapped yaw run through `local_linear` at
`POSE_SIGMA_S = 0.06` at the output time. On `3e1f9422`, the per-frame second
difference of ego position at 60 fps went from p50 0.51 m / p99 2.30 m (nearest
tick) to p50 0.002 m / p99 0.018 m, and traffic from p99 1.15 m to 0.014 m. The TMP
clip `13445b50` measured the same, p99 1.14 m to 0.012 m. Shape error on a curve is
about `sigma^2 v^2 kappa / 2`, a couple of centimetres at highway speed.

A vehicle id is split into separate tracks on a gap over `TRACK_GAP_S` or a jump
further than `TRACK_JUMP_M` plus 45 m/s of travel, so an id the game reuses never
slides across the map. Vehicles appear and disappear on the tick the radar does.

The threat highlight is the `colliding_ids` of the nearest tick, with one
exception: `threat_ticks` bridges a dropout of up to `THREAT_GAP_TICKS` (2) ticks
between two flagged ticks, so a one-tick collision-grid miss does not blink the
vehicle grey. It fills interior gaps only, never the ends, so a highlight starts and
stops exactly where the recording does.

Vehicle paths are the arcs `replay_clip` rebuilds for the nearest tick, moved
rigidly from that tick's raw pose to the smoothed pose, so their shape is exactly
what AEB evaluated. The ego corridor is rebuilt with `build_arc` from the smoothed pose and
the smoothed recorded curvature and horizon.

## Decisions: re-run by default

By default `build_timeline` runs `core/aeb/clip_eval.run_headless` over the clip at
the working tree's constants and swaps its warn, brake and `colliding_ids` into every
frame (`timeline.with_rerun_decisions`), so the video shows how today's AEB treats
the recorded traffic. `--recorded` keeps the live decisions. The export result
reports how many ticks changed state and both first-brake times.

**The re-run is open loop.** Traffic, ego position and ego speed all come from the
recording, and nothing the re-run decides can move the truck. Two consequences:

- If the recorded AEB braked, every frame after that brake shows a truck slowing
  from the old decision. A re-run that brakes later, or not at all, then shows the
  truck braking under a `STANDBY` label.
- After the recorded brake the re-run sees an ego that is already slowing, so its
  own demand reads low. Its decisions past that point are biased toward not braking.
  Only the stretch before the recorded brake is a clean comparison.

`export.open_loop_warning` prints a warning whenever the re-run's first brake lands
more than 0.1 s after the recording's. Measured on the working tree of 2026-09-14:

| Clip | Recorded brake | Re-run brake | Ticks changed |
|---|---|---|---|
| `3e1f9422` crosser | 7.99 s | 10.52 s | 31 |
| `66dffd1a` head-on | 8.20 s | 8.89 s | 50 |
| `e87c2d97` turn-in | 8.68 s | 8.05 s | 10 |
| `13445b50` TMP | 4.49 s | 4.43 s | 2 |

The first two are exactly the warning case: their footage needs `--recorded`, or a
clip where the recorded AEB did not act. Closing the loop would need a model of the
truck's response to re-drive ego speed; that is not built.

## Ego trailer

Clips record whether ego has a trailer (`ego_has_trailer` or `trailer_count`), not
where it is. The debug view draws it rigidly in line with the tractor, which looks
wrong in any turn, so this draws it from a kinematic follower instead: the axle
point is pulled along behind a kingpin 2.4 m behind the tractor centre, at a
10.2 m wheelbase, integrated at 120 Hz from the clip start. It is a visual
reconstruction of a generic 13.6 m semi-trailer, not recorded data, and it never
feeds framing decisions beyond the subject box.

## Contributed clips

`cli.py` refuses a clip that only exists in the contributed store unless
`--contributed-ok` is passed, and then always burns "anonymous contributed clip"
into the frame. The acceptance recorded in `tools/aeb_agent/README.md` covers the
v1.1 trailer only, so each further use needs its own decision; the flag exists to
make that a deliberate step rather than a default.
