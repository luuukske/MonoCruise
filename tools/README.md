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
