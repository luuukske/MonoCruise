# Prompt: label every untagged AEB clip

Paste the block below into Cursor as the task. It is written for a capable model
working unattended over a long run. Everything it needs is in this repo; it must
not ask for the game, a GUI, or a human mid-run.

---

You are labelling the AEB clip corpus for MonoCruise. There are ~734 untagged
clips across the local and contributed stores. Your job is to get through **all of
them**, in batches, without stopping until zero remain.

## Before you start

Read `tools/aeb_agent/README.md` end to end. Then read `core/aeb/README.md`
sections 5 (clearance-based required decel), 9 (invariants) and 11 (review tool).
Do not skip this: the rules there decide most of the calls you are about to make.

Verify the caches are warm, then note the starting count:

```bash
python -m tools.aeb_agent index --features
python -m tools.aeb_agent list --unlabeled --fast | tail -1
```

## The loop

Repeat until `list --unlabeled` reports zero clips. Do not stop early, do not ask
whether to continue, do not summarise progress instead of continuing. Batches of
25 keep each dossier readable:

1. `python -m tools.aeb_agent triage --unlabeled --limit 25 --out batch`
2. Read `worklist.json` in the printed directory, then read every dossier file it
   names. One clip at a time. Do not label from the worklist alone: the worklist
   has tags, the dossier has the evidence.
3. Append one JSON object per line to `proposals.jsonl` in the workspace, one per
   clip. Copy `clip_id`, `mtime` and `size_bytes` straight from the worklist so
   the staleness check works.
4. `python -m tools.aeb_agent apply` (dry run). Fix anything it skips, then
   `python -m tools.aeb_agent apply --commit`.
5. Delete the batch directory and go back to step 1.

If `apply` skips a proposal, read the reason and fix that proposal. Never work
around a skip by loosening a flag you were not told to use.

## Read the driver first, the geometry second

The question a label answers is **"would this have been a collision if the driver
had done nothing"**, not "was there a collision". Those differ constantly, because
the driver is good at their job. A swerve leaves a recorded geometry that reads as
a comfortable pass, and taking that at face value is the single biggest source of
wrong labels in the last pass.

Every dossier has a `DRIVER AND COUNTERFACTUAL` block. **Read it before the
TARGETS table.** It reports what the driver did near the action (swerve, brake,
lift, with the steering trace and blinker state) and then flies a ghost ego from
that moment on the heading rate and speed the driver was already holding, against
the traffic's own recorded future.

Its verdict is graded, not boolean, because the ghost drifts:

- `COLLIDES` - the ghost overlaps a body by more than the model's own error bar.
  Treat this as a real threat that the driver removed.
- `LIKELY` - overlap, but inside the error bar. Real enough to weigh; say so in
  the notes and drop confidence.
- `CLOSE` - passes within the error bar. Suggestive, not evidence.
- `CLEAR` - holding course was safe. The recorded near miss was the actual margin.
- `DEGENERATE` - bodies already overlapped at the fork, usually a TMP spawn pile.
  Proves nothing. Frequently an `ignore`.

The block also prints whether the verdict is inside the 1.5 s the ghost is
trustworthy over. **Past that, do not treat it as evidence**, whatever it says.

What this changes:

- A `COLLIDES` or `LIKELY` verdict on a clip where AEB stayed silent is an `fn`,
  not a `tn`, and the driver's rescue is the proof. Do not label it `tn` because
  nothing touched.
- Do not use "nothing ever entered ego's corridor" as a reason for a negative
  class when the driver swerved. The corridor is empty *because* they swerved.
- `driver-handled` reasoning only applies when the counterfactual is `CLEAR`. If
  the driver braked and holding course would have hit, AEB missed it.
- If the driver's action was routine, not evasive (a lane change with a blinker,
  ordinary cornering), say so and judge on the geometry as normal.

## How to decide a class

Every clip gets exactly one of `tp`, `good_intervention`, `fp`, `tn`, `fn`,
`ignore`. Decide in this order:

**First ask: can this clip arbitrate anything at all?** If not, it is `ignore`,
severity 0. Use `ignore` for:

- lag or desync: the dossier shows `lag_confirmed` ticks or raw teleports on the
  target that matters, so the "threat" is a network artefact and not a scene;
- junk capture: `FLAGS` reports paused frames, a truncated clip, very few AEB
  ticks, or AEB disabled for much of the clip;
- no target: nothing entered ego's corridor and nothing came near;
- the driver resolved it **and** the counterfactual says `CLEAR`: the scene was
  never going to be a collision, so it says nothing about AEB. If the
  counterfactual says `COLLIDES` or `LIKELY`, this is an `fn`, not an `ignore`;
- the counterfactual is `DEGENERATE` and the targets sit at zero range: a TMP
  spawn pile is not a scene.

Be generous with `ignore`. A clip left in under a class it cannot support poisons
every corpus measurement made afterwards. A clip dropped costs nothing.

**Then, for the clips that can arbitrate:**

- AEB braked and should have: `tp`, needs a window.
- AEB acted usefully but short of a full stop, and that was right:
  `good_intervention`, needs a window.
- AEB braked or warned and should not have: `fp`, no window.
- AEB stayed silent and that was correct: `tn`, no window.
- AEB should have braked and did not: `fn`, needs a window.

Severity 1-5 on everything except `ignore`. Judge it by what the outcome would
have been, not by how alarming the scene looked: 1 is a minor comfort event, 5 is
a collision or a near miss that only geometry avoided.

## Windows: the rule you must not break

A `should_trigger` window is **your judgement of when braking became necessary**,
read off the geometry in the `TARGETS` table and the `TIMELINE`. It is not the
band AEB reacted over.

The dossier prints `reaction band` because it is useful context. Copying it is
forbidden and `apply` will refuse a window that matches it within 0.05 s. This
matters most on `fn` clips, where AEB's band is wrong by definition.

Set `window_source` to `judged` always. If you genuinely cannot place a window
from the geometry, still assign the class, set `confidence` at or below 0.4, and
put `needs-human-window` in the tags so it can be filtered out later.

**A window must open before the driver had to act.** If the driver swerved at
t=6.4, a window starting at 6.8 describes the rescue, not the threat. Open it
where the geometry made braking necessary, which is at or before the swerve.

## Notes format

Every proposal needs `notes` shaped like this, and a `rationale` that names the
numbers you used:

```
[tag tag tag] one sentence describing the scene and why this class.
```

Keep the sentence under about 20 words. Tags are lowercase, hyphenated, space
separated, and drawn from this vocabulary. Add a tag only when it is true.

**Scenario:** `oncoming` `crossing` `codirectional` `overtaker` `stationary`
`in-lane` `adjacent-lane` `cut-in` `queue` `merge` `junction` `roundabout`
`curve` `ego-turning` `ego-stopped` `low-speed` `parked` `shoulder`
`trailer-target` `ego-trailer` `tmp` `sp`

**Why it is ignored:** `lag` `desync` `teleport` `junk-capture` `paused`
`truncated` `driver-handled` `aeb-disabled` `no-target` `duplicate`

**Driver and counterfactual:** `driver-swerve` `driver-brake` `driver-lift`
`would-have-collided` `evasion-close` `counterfactual-clear`
`counterfactual-degenerate` `routine-manoeuvre`

**Debugging interest:** `late-brake` `early-brake` `phantom-brake` `warn-only`
`no-warn` `hard-decel` `capacity-limited` `suppressed-by-filter`
`needs-human-window`

Name the filter when one is load-bearing, e.g. `suppressed-by-filter` plus the
stage name in the sentence ("OppositeLaneFilter held it for 196 ticks").

## The 9eee3dd tag, which is the point of this pass

Commit `9eee3dd` replaced "stop at the intersection point" with clearance-based
required decel. Every clip it materially moves must carry a tag, so the effect
can be filtered later.

**Do not judge this yourself.** The dossier has a `FLAG A/B` block computed by
flipping `clearance_required_enabled`. Use exactly what it says:

- block says `MATERIAL` and direction `engages` -> add `clearance-engages`
- `MATERIAL` + `silences` -> add `clearance-silences`
- `MATERIAL` + `advances` -> add `clearance-earlier`
- `MATERIAL` + `delays` -> add `clearance-later`
- `MATERIAL` + `reshapes` -> add `clearance-reshapes`
- block says `minor`, or there is no block -> add nothing

198 of the untagged clips are material, so roughly one in four should carry one of
these. If a whole batch comes back with none, you are reading the wrong block;
stop and check before continuing.

When a clip carries a clearance tag, say in the sentence what moved: "9eee3dd
brakes 0.4 s later on this crosser" is the useful form.

## Proposal shape

```json
{"clip_id": "...", "label_class": "fp", "severity": 2, "window": null,
 "target_vid": 716, "notes": "[oncoming tmp curve suppressed-by-filter] Oncoming truck in its own lane through a bend.",
 "rationale": "lat -3.7 m the whole approach, dot -1.00, corridor gap 6.8 m, OppositeLaneFilter held it 196 ticks",
 "confidence": 0.85, "reviewer": "grok-4.5", "window_source": "judged",
 "mtime": 1786371655.4721155, "size_bytes": 399370}
```

`window` is `[from_t, to_t]` for `tp`, `good_intervention` and `fn`, and `null`
for `fp`, `tn` and `ignore`.

## Confidence, honestly

`confidence` is how sure you are of the class, not how sure you are that you read
the file. Use the range. Clips you would bet on are 0.85 or above. Clips where two
classes are defensible sit near 0.5, and the notes must say which two and why you
picked one. Do not put 0.9 on everything: the point of the number is that the
human can filter to the ones worth re-checking, and a flat 0.9 makes it useless.

## When you are done

Report:

- how many clips you labelled, broken down by class;
- how many had a driver intervention, and how the counterfactual verdicts split;
- how many you called `fn` **because** of the counterfactual, listed by id, since
  those are the ones worth a human spot check;
- how many carry each `clearance-*` tag;
- how many are `ignore` and the top reasons;
- how many carry `needs-human-window`;
- `python -m tools.aeb_agent score` before and after, and what moved;
- `python -m tools.aeb_agent audit` after, and any suspects your own labels
  created.

Then confirm `list --unlabeled` reports zero.

Do not stop before that.
