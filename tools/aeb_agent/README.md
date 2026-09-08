# tools/aeb_agent

Headless AEB clip corpus tooling for an agent. Dev only, never shipped, never
imported by the app. Everything runs without Qt and without the game.

`tools/aeb_review.py` is the human tool: it plays the clip, draws the scene and
lets a person mark a window in a couple of seconds. It is still the arbiter. This
package exists because the corpus now grows faster than a person can watch it, so
an agent can do the reading, the sorting and the first pass at a verdict, and the
person spends their time confirming rather than searching.

```bash
python -m tools.aeb_agent --help
```

## The one rule that is not negotiable

**Never take a should-trigger window from AEB's own output.** The corpus exists to
judge AEB. A window copied from the ticks AEB reacted on encodes "AEB was right"
into the ground truth, and every later measurement made against it is circular.
`recorded band` is printed in every dossier because it is useful context, and
`apply` refuses a window that matches it to within 0.05 s unless you pass
`--allow-recorded-window` and say in the rationale what you checked.

`fn` clips are the case that proves it: the recorded band is wrong there by
definition, which is what makes them misses.

## Workflow

```bash
python -m tools.aeb_agent index --features       # one-time, a few minutes
python -m tools.aeb_agent stats                  # what is in the corpus
python -m tools.aeb_agent list --unlabeled --limit 40
python -m tools.aeb_agent show <id> --replay     # read one clip
python -m tools.aeb_agent propose <id> --class fp --severity 2 --rationale "..."
python -m tools.aeb_agent apply                  # dry run, prints what would change
python -m tools.aeb_agent apply --commit         # writes, and journals every change
```

Nothing before `apply --commit` touches a clip file. `journal` lists what was
written and `revert <id> --commit` puts the previous label back, exactly.

For a batch pass, `triage` writes one dossier per clip into the workspace plus a
`worklist.json`, so the reading is file-by-file instead of one giant transcript:

```bash
python -m tools.aeb_agent triage --unlabeled --scenario crossing --limit 40
```

Then append one JSON object per line to `proposals.jsonl` directly (the same
shape `propose` writes: `clip_id`, `label_class`, `severity`, `window`,
`rationale`, `confidence`, `reviewer`, `mtime`, `size_bytes`) and run `apply`.
That is far cheaper than one `propose` invocation per clip. Copy `mtime` and
`size_bytes` from the worklist so the staleness check can do its job.

## Commands

| Command | What it is for |
|---|---|
| `index` | Build the metadata index; `--features` also builds the scene cache (a process pool, about 0.25 s per clip) |
| `list` | One line per clip: label, speed, corridor gap, ttc, scenario tags |
| `stats` | Counts by class, origin, session, trigger, severity, scenario |
| `show` | The full dossier for a clip. This is what you read instead of watching it |
| `scene` | ASCII top-down maps at chosen timestamps |
| `triage` | Write one dossier file per clip plus a worklist, for a batch pass |
| `audit` | Rank labelled clips whose label looks wrong |
| `clearance` | A/B a calibration flag per clip; defaults to the 9eee3dd clearance rewrite |
| `thumbs` | Export clip screenshots so a human can flip through a shortlist |
| `render` | Export a clip's debug view as a PNG sequence, for video work |
| `propose` | Append one proposed label change to `proposals.jsonl` |
| `apply` | Validate the proposals; write only with `--commit` |
| `journal` / `revert` | What was applied, and how to undo it |
| `score` | The corpus objective under the working tree, per store |

Filters compose on `list`, `stats` and `audit`: `--label`, `--scenario`,
`--session`, `--origin`, `--trigger`, `--sev-min/--sev-max`, `--since/--until`,
`--unlabeled`, `--labelled`, `--inconsistent`, `--flagged`, `--ids`, `--limit`.
Add `--json` for machine output, `--fast` to skip the feature cache.

## Reading a dossier

`show` prints six blocks. What each one is worth:

- **CLIP / LABEL** are facts about the file and the existing verdict.
- **SCENE** is the scenario tag set, derived from geometry. **FLAGS** are
  data-quality problems: paused frames, a short capture, AEB switched off.
- **RECORDED AEB** is what the shipped build did *at capture time*, under whatever
  tuning was current then. It is evidence about the past, not about the code you
  are testing, and it is never ground truth.
- **TARGETS** is the measurement. `corrGap` is the closest the body came while
  inside ego's straight-ahead corridor, which is the rear-end proximity; `minRange`
  counts vehicles passing alongside too. `tgeom` is corridor gap over closing rate.
  `dot` is the cosine between ego heading and target heading: `+1` co-directional,
  `-1` oncoming, near `0` perpendicular. `lat` is positive to the right.
- **TIMELINE** is per tick around the action, tracking the primary target.
- **SCENES** are ASCII maps, forward up the page, `:` marking lane separation and
  an uppercase glyph marking a vehicle AEB currently calls colliding.

`--replay` adds what the *working tree* does on the clip now. Use it when you are
asking "is this still a miss", not when you are asking "was this label right".

## The classes

| Class | Means | Window |
|---|---|---|
| `tp` | AEB braked and should have | required |
| `good_intervention` | AEB acted usefully, short of a full stop | required |
| `fn` | AEB should have braked and did not | required, and it must be your own judgement |
| `fp` | AEB reacted and should not have | must be absent |
| `tn` | AEB stayed silent and should have | must be absent |
| `ignore` | The clip cannot arbitrate anything | either |

`ignore` is not a failure to decide. Junk captures, TMP desync, a scene the driver
resolved themselves and paused or truncated recordings all belong there, and a
clip left in the corpus under a class it cannot support does more harm than a clip
dropped from it.

## The counterfactual: what the driver prevented

The corpus question is "would this have been a collision if the driver had done
nothing", never "did they collide". Drivers are good, so those answers differ
constantly, and judging a clip on what actually happened labels every
driver-rescued threat a true negative. A swerve is the hard case: it leaves a
recorded geometry that reads as a comfortable pass.

`evasion.py` finds the driver's action near the clip's action point (a steering
spike, a brake stamp, a throttle lift) and then flies a ghost ego from that fork
against the traffic's own recorded future. The ghost replaces **only what the
driver changed**: a swerve keeps the recorded speed profile and gets the pre-fork
heading rate, a brake keeps the recorded heading and gets the pre-fork speed.

### The verdict is graded because the ghost drifts

Measured against the recorded path on 292 to 340 no-intervention clips, p90 drift:

| horizon | heading held | speed held | both held |
|---|---|---|---|
| 0.5 s | 0.66 m | 0.66 m | 0.68 m |
| 1.0 s | 1.28 m | 1.09 m | 1.35 m |
| 1.5 s | 2.74 m | 2.13 m | 3.38 m |
| 2.0 s | 6.48 m | 3.20 m | 6.68 m |

At 2 s the error is wider than a lane, so a boolean "would have hit" would be
dishonest. The verdict is graded against the drift band at the moment of closest
approach: `collides` (overlap deeper than the band), `likely` (overlap inside
it), `close`, `clear`. Anything past **1.5 s** is marked not credible and must not
be used as evidence. `CREDIBLE_S` is that bound.

### Degenerate cases

If bodies already overlap at the fork, every projection "collides". That is a TMP
spawn pile or a decode artefact, not a threat, and it is reported as
`degenerate`. The guard removed 81 clips and cut the flagged set from 73 to 39,
so it is load-bearing: without it the rule mostly found spawn piles.

### What it changes

Over the corpus, 39 clips carrying `fp`, `tn` or `ignore` have a credible
counterfactual saying the driver's action is what avoided contact, 22 of them at
the confident tier. `rule_evasion_rescued` surfaces them, and
`rule_window_after_evasion` catches positive labels whose window opens after the
swerve, which describes the rescue rather than the threat.

Three older rules now consult it before firing, because each of them read "the
corridor stayed empty" as evidence of no threat: `positive_no_target`,
`crossing_cleared` and `driver_handled`.

```bash
python -m tools.aeb_agent list --label ignore,fp,tn --counterfactual collides --credible-cf
```

### Guards, each of which cost a real false positive

Found by reviewing flagged clips rather than by reasoning, so do not remove one
without a replacement:

| Guard | Why |
|---|---|
| `degenerate` on fork overlap | Bodies already overlapping make every projection collide. Removed 81 clips. |
| `MAX_HELD_OMEGA` 0.35 rad/s | A yard manoeuvre held for the horizon draws a spiral. Only applies when the ghost holds yaw. |
| `MIN_GHOST_MS` 4.0 m/s | Below ~15 km/h a contact is a parking scrape, not an AEB scenario. |
| closing test | A body the ghost is departing from is not a threat. |
| lag guard in `evasion_rescued` | A desync ghost must not drive a rescue claim. |
| `clear` never demotes | See below. |

### `clear` is one-directional, on purpose

There is no rule that turns a positive label negative on a `clear` verdict. A
wrong `clear` silently deletes a real threat from the corpus; a wrong `collides`
only sends a clip back for review. This mirrors `core/aeb/README.md` on measured
misses: they may remove certainty, never grant it.

### What it cannot do

Traffic keeps its recorded motion, so a target that reacted to the real swerve is
not re-simulated. The ghost is a constant-rate projection, so a driver mid-corner
gets a worse error bar than one on a straight, which is the p90 tail in the table
above. Neither is a reason to distrust the graded verdict; both are reasons the
verdict is graded.

Known open gaps: the counterfactual does not consult the elevation gate, and the
`heading` mode is most fragile exactly where drivers steer most, on bends. When a
verdict looks wrong, check `mode` first: it says which axis the ghost replaced,
and a `speed`-mode ghost keeps the recorded heading, so a large
`yaw_rate_before` in the intervention line is not being used.

## A/B-ing one commit over the corpus

`clearance` flips a boolean on `AEBCalibration` and reports which clips decide
differently. It defaults to `clearance_required_enabled`, the flag commit
`9eee3dd` added, so it answers "which clips does the clearance rewrite actually
move". Results are cached on file identity like the feature rows.

```bash
python -m tools.aeb_agent clearance --unlabeled --material
```

Direction is `engages` (brakes where the old model did not), `silences`,
`advances` / `delays` (first brake moved), or `reshapes`. `--material` keeps only
the clips where the outcome flipped, the first brake moved by 0.20 s or more, or
at least 10 ticks differ; the rest is tick-level jitter. Over the untagged set,
345 of 734 clips change at all and 198 are material, which is why the bar exists.

`show --ab` adds the block to one dossier, and `triage` includes it by default
(`--no-ab` to skip) along with `clearance_material` and `clearance_tag` in the
worklist. `--flag` points the same machinery at any other boolean.

## Audit rules

`audit` ranks suspicion, it does not decide. Every hit prints the numbers behind
it so you can disagree from the same evidence.

| Rule | Fires when |
|---|---|
| `class_window` | Class and should-trigger window contradict each other |
| `positive_no_target` | A positive label with nothing that ever entered ego's corridor |
| `driver_handled` | Positive label, AEB never braked, and the driver braked hard |
| `lag_positive` | Positive label whose primary target is a TMP desync or teleport suspect |
| `lag_negative` | `fp`/`tn` charged against the tuning when the target was a desync ghost |
| `crossing_cleared` | An `fn` on a crosser that cleared the corridor long before ego arrived |
| `negative_looks_real` | `fp`/`tn` with an in-lane target at very low geometric ttc |
| `severity_outlier` | Severity that the geometry does not support in either direction |
| `window_bounds` | Empty, reversed, out-of-range or implausibly wide window |
| `window_is_recorded` | An `fn` window identical to AEB's own reaction band |
| `target_vid_absent` | `target_vid` is not among the tracked vehicles |
| `data_quality` | Paused, truncated, or AEB-disabled capture under a real label |
| `duplicate_clip` | The same clip_id in both stores, so a hand edit can go stale |

The lag and crossing rules are the two the corpus most needs. A frozen TMP body
looks exactly like a stationary obstacle appearing at range, and perpendicular
traffic that clears the corridor is a true negative however alarming it looked.

### Stream stalls are not a desync signal

`stall_ticks` and `stall_run_max` count ticks where a body reported speed the raw
stream did not move it by. That looks like the ideal desync detector and is not
one: measured over the corpus it fires on 141 of the positive clips, including
clip `3920236e`, which is a correct `tp` where AEB stops from 94 km/h behind a
genuine stationary queue with 18 m to spare. What it actually measures is the TMP
traffic update rate against a ~30 Hz AEB tick, which the radar smoothing exists to
absorb. It stays in the dossier as a diagnostic a human can weigh, and **no audit
rule may key on it**. `lag_confirmed` from `core/radar/traffic.py` is the gated
signal, and its entry conditions (rotation liveness, raw-window speed, decay
ratio) are what separate a real stall from a slow update. See
`core/radar/README.md` section 7.

## What the tool refuses

`apply` skips a proposal, rather than writing it, when:

- the class is not one of the six, or severity is outside 1..5 on a real class;
- the class and window contradict each other;
- the window is empty, reversed, or outside the clip;
- the window matches AEB's recorded reaction band (see the rule above);
- there is no rationale;
- confidence is below `--min-confidence`;
- the clip file changed on disk since the proposal was made.

A skip is printed with its reason and the run continues. Nothing partially applies.

## Where things live

The workspace holds `index.json`, `features.json`, `proposals.jsonl` and
`journal.jsonl`. It defaults to `tools/aeb_corpus_run/agent/`, which is gitignored,
and `MONOCRUISE_AEB_AGENT_DIR` overrides it. Clip files themselves live in the two
stores that `core/aeb/clip_store.py` defines and are never copied into the repo.

The feature cache is keyed on file mtime and size, so relabelling a clip
invalidates only that clip. `index --features --refresh-features` forces a full
rebuild, which is only needed when `FEATURES_VERSION` changes.

## Prompts

Two ready prompts live beside this file:

- `PROMPT_untagged_pass.md` labels every untagged clip, in batches, until none
  remain.
- `PROMPT_recheck_context.md` rechecks existing labels against the driver and
  counterfactual block, tiered so the 934 clips with no intervention are not
  re-read for nothing.
- `PROMPT_trailer_shortlist.md` builds a bucketed shortlist of AEB saves worth
  showing in a release video. Local clips only, for the reason below.

## Rendering a clip as footage

`render` replays a clip through the same top-down view the review UI draws and
writes one PNG per tick, so a recorded event can become video without re-driving
it. It prefers the native Qt platform: under `offscreen` the font database can
come up empty and every HUD string renders as tofu, so `_ensure_fonts` registers
system fonts as a fallback. Frames come out at the display's device pixel ratio,
so a `--width 1280` request can land at 2240 px wide, which is a bonus for video
but means the flags are not exact. The ACC panel and the legend are always drawn;
there is no toggle for them short of changing `core/aeb/debug_window.py`.

## Publishing contributed clips

`ui/main_window/assets/clip_contribution.md` is what contributors agreed to, and
it says clips are used "to tune AEB and ACC. Never sold, never shared." Using a
contributed clip in a release video is outside that wording. Lukas reviewed the
risk and accepted it for the v1.1 trailer, under two conditions:

- only the **debug view** is published, never the stored screenshot, so no game
  imagery, place names or player names leave the machine;
- every contributed clip carries an on-screen credit reading **"anonymous
  contributed clip"**.

`render` enforces the second by writing a `CREDIT.txt` beside the frames of any
contributed clip and echoing `credit_required` in its result. That decision
covers this trailer; it is not a general licence to publish contributed data, and
a future use should be decided on its own.

## Reading further

- `core/aeb/README.md` section 11 for the human review tool and the desmoothed
  decel trace, section 5 for clearance-based required decel, section 9 for the
  invariants any label reasoning has to respect.
- `core/radar/README.md` section 7 for TMP lag and freeze detection, which is what
  the lag rules key on.
- `AGENTS.md` for the repo rules that apply to any change made here.

## Where the numbers in this README came from

The corpus at the time of writing: 1852 clips (1068 local, 784 pulled), 1117
labelled, 735 unreviewed. `audit` over it returns 84 suspicions across 76 clips,
12 of them high. Those counts move as the corpus grows; re-run rather than
quoting them.
