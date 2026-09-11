# tools/

Offline probes and build helpers. Nothing here ships in the app or is imported
by it, but `tests/invariants/` does scan this directory, so the AGENTS.md
hygiene rules apply to every file in it.

| File | What it does |
|---|---|
| `acc_response_map.py` | Axes, statistics, ASCII/text/JSON reports, rendering and the CLI. Documented below. |
| `acc_transition_probe.py` | Is the command smooth as the lead barely brakes? Sweeps `a_lead` through zero for the gain steps, then runs closed loop against noisy telemetry. Renders before/after in one process. |
| `acc_probe_rig.py` | Measurement half: loads a checkout, publishes a synthetic lead, reads the cap. Import this directly to build a new ACC probe without the map's presentation. |
| `aeb_clearance_probe.py`, `aeb_fetch.py`, `aeb_review.py`, `aeb_review_widgets.py` | AEB clip corpus tooling. See `core/aeb/README.md`. |
| `aeb_filter_trace.py`, `aeb_filter_charts.py` | The `C` window in the review tool: what the radar speed / accel / lag filters did to one vehicle over a clip. Documented below. |
| `aeb_agent/` | Headless clip review for an agent: text dossiers instead of watching a clip, scenario tags, mistag audit, and a validated propose/apply/revert path for labels. Read `tools/aeb_agent/README.md` first. |
| `accel_envelope_probe.py` | What does the CC accel ceiling command at each speed, and how long is 0-50 / 0-90? Prints the per-profile table plus a capability-limited rig model (`--rig loaded`) so the light and loaded regimes can be compared. |
| `plot_coast.py` | Coast-fit plots for the mapper. |
| `release.py`, `tune_visualizer.py` | Release packaging and live tuning UI. |

---

## acc_response_map.py

Answers "what does the ACC gap law actually command, and what did my change do
to it", without needing the game running.

### What it runs

Not a reimplementation of the law. It registers a stub `acc_thread` in the
registry, publishes one synthetic lead per query, patches `time.monotonic` to a
deterministic clock, and calls the real
`AdaptiveCruiseController.accel_cap_ms2`. Safety overlays, the confidence
blend, the jerk limiter and the output EMA are all in the loop.

Validated: off-overlay the probe reproduces `lead_law` to `0.000e+00` across
3785 random states, and TTC-overlay cells land exactly on `max_decel_ms2`. Every
run prints a settle residual (the change from quadrupling the tick count); it is
`0.0` for a single constant lead, because every EMA seeds to its input on the
first tick. A non-zero residual means something stateful is not converging and
the numbers should not be trusted.

`core.settings` is redirected at a throwaway directory before anything reads it,
the same way `conftest.py` does it, so no run can touch the live `config.json`.

### Coordinates

* **X, closing speed** `v_ego - v_lead`, positive = ego catching up. This is the
  sign the controller reasons in (`v_close`).
* **Y, lead deceleration**, positive = braking harder, so `a_lead = -y`.
* **Value**, the accel cap in m/s². It is a *cap*: cruise takes
  `min(speed_pid, cap)`, so wherever the surface sits at the no-lead ceiling the
  ACC is not binding at all. Those cells are hatched in the PNG.
* Cells where `v_lead` would be negative are masked, and drawn grey.
* Gap is the **published** `lead.dist_m`. The controller subtracts
  `ego_front_offset_m` (2.5 m) internally.

### Structural limits of this instrument

Read these before drawing a conclusion from a plot.

* **One lead, so multi-vehicle anticipation never runs.** It needs two or more
  chain members. Anything the virtual lead or the per-lead decel side would have
  contributed is absent by construction.
* **A state-space slice, not a trajectory.** Each cell is an instantaneous
  state. A real lead that brakes also changes `v_lead`, which walks you across
  the map. Use `--trace` for the time-domain question.
* **Steady state.** The map cannot show lag. `--trace` can.
* **A change that only fires on a measure-zero set is invisible here.** The
  strict CAH branch test in `b404efc` only differs at `v_lead = 0` *and*
  `a_lead = 0`, which is one point per panel. Use `--probe` for those.

### Agent recipes

Default to `--report text`. A PNG costs far more context than it returns, and
the text report carries the numbers a decision actually turns on.

Is lead braking reaching the command at all? The `cap span over the whole decel
axis` line is the answer as a single number per panel:

```bash
python tools/acc_response_map.py --scenario matched-speed --report text --no-ascii
```

Exact values at named states, no grid, no image, fastest mode:

```bash
python tools/acc_response_map.py --probe "ego=80,gap=40,closing=0,decel=4" --probe "ego=80,gap=40,closing=4,decel=0"
```

How long after the lead brakes does the cap respond, closed loop:

```bash
python tools/acc_response_map.py --scenario brake-onset
```

Machine-readable summary for a script to diff:

```bash
python tools/acc_response_map.py --scenario highway --report text --json -
```

### Comparing revisions

Every map run writes a `.npz` next to the PNG. Check the old revision out into a
worktree, point `--repo` at it, then diff. `--delta-only` renders just the change
panels, which is the small figure to reach for when the absolute surfaces are
already known.

```bash
git worktree add --detach /tmp/mc-base <rev>
```

```bash
python tools/acc_response_map.py --repo /tmp/mc-base --label base --report png --out out/base.png
```

```bash
python tools/acc_response_map.py --label head --compare out/base.npz --delta-only --report both --out out/change.png
```

Axes and conditions are recorded in the `.npz` and checked on load, so a
mismatched pair fails loudly instead of plotting a meaningless delta. The
headway of both runs is printed in the figure subtitle, and flagged in capitals
if they differ, because a gap-level mismatch silently turns the change column
into two effects added together.

`--repo` works on any revision back to `121f885` (the IIDM+CAH rewrite). The
loader tries several module paths and requires a class exposing
`accel_cap_ms2`, so it survives the file moves in between. Remove the worktree
when done, or `git worktree prune` after the temp directory is cleaned up.

### Scenarios

`--scenario NAME` sets defaults that any explicit flag overrides. `--help` lists
them with one-line descriptions.

| Scenario | Kind | Purpose |
|---|---|---|
| `highway` | map | one speed, three gaps around the level-3 wanted gap |
| `highway-pair` | map | two speeds by three gaps, the default comparison grid |
| `town` | map | low speed, short gaps |
| `cutin` | map | inside the wanted gap at low closing speed |
| `matched-speed` | map | zoom on the band where lead braking should matter |
| `brake-onset` | trace | closed-loop response lag |
| `stopped-lead` | probe | cap against a stopped lead, by speed and gap |

### Reading the text report

```
panel 80 km/h, gap 40 m
  cap -6.55 .. +1.33 m/s^2, zero at closing -0.2 m/s
  cap span over the whole decel axis   [closing 0: 2.06, closing 5: 4.23]
  cap span over the whole closing axis [decel 0: 5.71, decel 4: 6.93]
  at decel clamp 19.7%, at ceiling 0.3%
```

`cap span over the whole decel axis` is how much the command moves when the
lead's deceleration is swept across the entire Y range at a fixed closing speed.
It is the direct measure of whether `a_lead` is reaching the command. It read
near zero at `closing 0` before the §8.6 feedforward landed, which is the
regression to watch: a panel back near zero there means `a_lead` has lost its
path to the command again. `core/acc/ACC_ARCHITECTURE.md` §8.5 has the before
numbers, §8.6 the after.

The ASCII field is on by default in text mode, at about 15 lines per panel.
`--no-ascii` drops it when only the statistics are wanted, which puts a
six-panel report at roughly 40 lines total.

---

## acc_transition_probe.py

The response map is a steady-state slice, so it cannot see a filter that
switches time constants, and its default 0.06 m/s^2 grid is too coarse to
resolve a gain step near `a_lead = 0`. This probe covers both.

```bash
python tools/acc_transition_probe.py --report text
```

`worst gain jump` is the headline: the largest adjacent change in
`d(cap)/d(a_lead)` across a fine sweep. A hard clamp shows up as a near
vertical line in the gain panel of the PNG.

The `before` column is not a different revision. Every softening knob restores
the old behaviour at 0, and `rig.overrides` applies them to a fresh controller,
so both variants run against the working tree. The accel-side nudge (§8.9) is a
feature rather than a softening knob and stays live in both columns.

Chatter statistics average over `--seeds` runs (12 by default). Do not read a
single seed: it ranks the variants wrongly, which is how a floor band of 0.12
first looked better than 0.09.

### Building a new probe on the rig

`acc_probe_rig.Rig` is the reusable part. It exposes one primitive,
`rig.cap(ego_ms, dist_m, v_lead_ms, a_lead_ms2)`, returning the steady-state cap
for a constant lead state, plus `rig.mod` and `rig.cls` for the loaded revision's
module and controller class. Wrap calls in `patched_clock(rig.clock)` and call
`rig.cleanup()` when finished.

```python
from acc_probe_rig import Rig, patched_clock

rig = Rig(repo, gap_level=3, dt=1 / 30, settle=8, score=5.0)
with patched_clock(rig.clock):
    cap = rig.cap(80 / 3.6, dist_m=40.0, v_lead_ms=22.22, a_lead_ms2=-4.0)
rig.cleanup()
```

Pick a different cut when the map's two axes hide what you are after. Distance
on X is the obvious next one: it puts a whole stopped-lead approach on one panel
instead of one pixel.

### Trace output

`--trace` runs a closed loop with a **perfect actuator** (`ego accel = min(0,
cap)`), so the printed onset times are a floor. Mapper lag, brake build-up and
the ~20 Hz physics tick all add on top. `clamp` is the time the cap first
reaches `max_decel_ms2`; a value there means the smooth law did not get the job
done and the TTC overlay took over.

---

## aeb_filter_charts.py

Answers "why did this vehicle's speed, acceleration or freeze state look like
that", against a clip already in the corpus. Press `C` in `tools/aeb_review.py`.

It is not a simulator. Every number is read off the `Vehicle` objects that
`decode_radar_stream` already produced for the replay, so what is drawn is what
the filter did on that clip, at the constants in the working tree. Changing a
constant and reloading the clip redraws it.

### Where the numbers come from

`aeb_filter_trace.build_trace` walks one sample per radar frame per vehicle
within `TRACE_RANGE_M`, plus any id the live AEB ever tracked or suppressed.
Most signals are attributes (`speed`, `acc_speed`, `_speed_ema`, `acceleration`,
`_raw_speed`, the state flags). The rest are rebuilt with the production helpers
from `core/radar/traffic.py`, never re-derived by hand:

* `raw_long`, `brake_floor`, `accel_trend`, `accel_long`, `accel_win`, `lag_raw_recent`,
  `lag_raw_decay` and `lag_freeze_dur` all call the same private helper the
  filter calls.
* The step 4 gates (`ramp`, `consistency`, `ff_gate`, `accel_factor`,
  `speed_factor`, `tau`) have no such helper: `_acc_speed_step` returns only the
  speed. They are the one place the arithmetic is restated. `_RESIDUAL` carries
  the rebuilt `acc_speed` minus the recorded one on every frame, and
  `test_step4_rebuild_reproduces_the_recorded_acc_speed` fails the moment they
  disagree. Measured bit-exact over 50,119 samples on 25 corpus clips; a 1%
  change to the rebuilt `tau` fails the test.

### Which frames carry which signals

Sub-frames (`dt < _LOCATION_UPDATE_FREQUENCY`), clock re-anchors, lag freezes,
position-mismatch holds and a track's first full update all skip the chain. Half
the radar frames in a typical clip are sub-frames, so this is the normal case,
not an edge:

* `st_subframe` marks a copy, `st_bypassed` marks a real early return. They are
  never both set.
* Step 4 internals are **held** across sub-frames, because the filter genuinely
  still holds them, and left blank on a bypass.
* The lag gates only evaluate on full frames, which is why a trace is drawn
  through a hole up to `_MAX_GAP_S` rather than broken at every gap. Breaking at
  every gap left the lane empty: alternate samples were `NaN` so no two adjacent
  points ever existed.

`_thin` exists for the same reason. About half of consecutive samples are exact
repeats, so plotting every one draws a staircase whose treads are the radar rate
rather than anything the vehicle did. A run of equal values contributes its first
sample, plus its last when the run outlasted `_HOLD_MIN_S`: a one or two frame
copy becomes a line straight from update to update, while a genuine hold still
reads flat with a steep exit. Curves are drawn antialiased at float coordinates;
the grid, the state ticks and the decision band stay on integer pixels because
they are single-pixel verticals that antialiasing only blurs.

### Reading the lanes

| Lane | Axis | What it answers |
|---|---|---|
| speed chain | m/s, auto | Where does AEB's `speed` sit against ACC's `acc_speed`, and how far behind the raw input is each. |
| accel chain | m/s², symmetric | Does `acc_accel` reach the lead's real deceleration, and did the hard-brake floor engage. `accel` and `acc_accel` fit the speed-scaled window; `accel_trend` and `accel_long` are the step-4 gate inputs and keep fixed windows. |
| step 4 gates | 0..1, `tau` at half scale, `accel_win` at 0.4 | Which term is setting the ACC filter's time constant right now, plus the live accel fit window in seconds. |
| lag entry gates | log2 of gate over threshold | Which of the four entry gates is holding a freeze open or shut. |
| filter state | one row per flag | Freeze, short window, mismatch, crash, standstill, sub-frame, bypass. |

Above the lanes sit two decision bands. **rec** is what the clip recorded live.
**now** is the same clip re-run through `clip_eval.run_headless` at the working
tree's constants, with a cyan tick wherever the two disagree. Everything below is
drawn at the current constants, so a recorded-only band could not be compared
against it; the second row is what makes "my change moved this" legible on the
same page. The re-run costs about 0.8 s a clip, more than the whole rest of the
load, so it is a **separate job the review window only asks for while the chart
window is open**, and it arrives after the clip is already on screen. The row
reads `now recomputing...` until it lands.

The lag lane needs its axis explained. The four gates run between roughly 0.1x
and 13x their own thresholds, so no single linear range shows all four. Each is
plotted as `log2(gate / its own threshold)`: **zero is the threshold**, +1 is
twice it, -1 is half it, clipped at ±3. A gate crossing zero is a gate changing
its answer.

The auto ranges use a percentile over the trace **past `_SETTLE_S`**. A fresh
track's least-squares slope runs through two or three points and reaches tens of
m/s², which is an artefact of the fit rather than of the vehicle; ranging on the
whole trace put every real signal flat against the axis.

### Workflow

The chart window is a separate top level, so it belongs on a second monitor. It
shares the review window's clock: scrubbing either moves both, and every review
binding still works while the chart window has focus, so tagging never needs a
click back. The vehicle picker follows the labelled target by default and
releases that link the moment it is used by hand.

`ClipLoader` builds the trace on its own thread from the decode the replay
already paid for, so opening the charts costs about 0.1 s per clip in the
background and nothing on the GUI thread.
