# Prompt: recheck labels against driver context

A follow-up to `PROMPT_untagged_pass.md`. The first pass judged clips on what
happened. The dossier now also reports what the driver did and what would have
happened if they had not, so labels decided without that evidence need rechecking.

Paste the block below into Cursor.

---

You labelled this corpus judging clips on their recorded geometry. That geometry
is the result of the driver's actions, not of the scene alone: when a driver
swerves or brakes out of a threat, the record shows a comfortable pass, and
reading that as "no threat" is wrong.

The dossier now carries a `DRIVER AND COUNTERFACTUAL` block. Recheck your labels
against it and correct the ones that were decided blind.

## What the new block says

It reports the driver's action near the clip's action point (swerve, brake, lift,
with the steering trace and blinker), then flies a ghost ego from that fork on the
heading rate and speed the driver was already holding, against the traffic's own
recorded future.

The verdict is graded because the ghost drifts:

- `COLLIDES` - overlap deeper than the model's own error bar. A real threat the
  driver removed.
- `LIKELY` - overlap inside the error bar. Weigh it, say so, lower confidence.
- `CLOSE` - passes within the error bar. Suggestive only.
- `CLEAR` - holding course was safe. The recorded margin was the real margin.
- `DEGENERATE` - bodies already overlapped at the fork, usually a TMP spawn pile.
  Proves nothing.

The block states whether the verdict falls inside the 1.5 s the ghost is
trustworthy over. **Past 1.5 s it is not evidence**, whatever it says.

## Scope, in priority order

Do these tiers in order and do not stop until each is finished.

**Tier 1, 95 clips.** The counterfactual is credible and contradicts the label:

```bash
python -m tools.aeb_agent list --label ignore,fp,tn --counterfactual collides,likely --credible-cf
python -m tools.aeb_agent list --label tp,fn,good_intervention --counterfactual clear,degenerate --credible-cf
python -m tools.aeb_agent audit --rule evasion_rescued,window_after_evasion
```

**Tier 2, ~250 clips.** Credible counterfactual that agrees with the label. Skim
these to confirm, do not rewrite what is already right:

```bash
python -m tools.aeb_agent list --credible-cf --labelled
```

**Tier 3, ~660 clips.** A driver intervention exists but the counterfactual is
not credible. Read the driver block for intent, judge on geometry as before, and
change a label only if the driver's intent genuinely reframes the scene.

**Tier 4, 934 clips.** No intervention was detected. **These carry no new
evidence**: the recorded path already is the no-action path, so your original
reasoning still stands. Do not re-read their dossiers. Spot check about 20 to
confirm the detector did not miss an obvious swerve, and move on.

Work Tiers 1 to 3 in batches of 25 with `triage` (drop `--unlabeled`, these are
already labelled), then `apply` as before. Everything stays revertible.

## What to change, and what not to

Change a label when:

- it is `fp`, `tn` or `ignore`, AEB stayed silent or was called wrong, and a
  credible `COLLIDES` or `LIKELY` says the driver is the only reason nothing was
  hit. That is an `fn`, and the rescue is the evidence;
- you called it `ignore` for "driver handled it" but the counterfactual is not
  `CLEAR`. Driver-handled only excuses AEB when holding course was safe anyway;
- you used "nothing entered ego's corridor" to justify a negative class on a clip
  where the driver swerved. The corridor is empty because of the swerve;
- the window on a positive label opens after the driver acted. Reopen it where
  the geometry made braking necessary, at or before the intervention;
- the counterfactual is `DEGENERATE` with targets at zero range and you called it
  a real class. A spawn pile is an `ignore`.

Leave it alone when:

- the counterfactual is `CLEAR`, or not credible, and your original call was
  sound. Confirming costs nothing and rewriting costs accuracy;
- the driver's action was routine rather than evasive: a signalled lane change,
  ordinary cornering, easing off in traffic. Say `routine-manoeuvre` in the tags
  if it is worth recording, and keep the label;
- the clip carries no intervention at all.

## Rules that still hold

Everything from the first prompt still applies. In particular: never copy AEB's
recorded reaction band into a window (`apply` refuses it); every proposal needs a
rationale naming the numbers; confidence must use its range rather than sitting
at 0.9; and `ignore` remains a real answer.

Add `context-recheck` to the tags of every clip you change in this pass, plus the
relevant one of `would-have-collided`, `counterfactual-clear`,
`counterfactual-degenerate` or `routine-manoeuvre`, so this pass can be filtered
out later.

## Caveats you should carry

The ghost is a constant-rate projection and traffic keeps its recorded motion, so
a target that reacted to the real swerve is not re-simulated, and a driver
mid-corner gets a wider error bar than one on a straight. When the verdict is
marginal, say so in the notes and lower confidence rather than picking a side.

## When you are done

Report, per tier: how many you rechecked, how many you changed, and the class
transitions (for example `ignore -> fn: 14`). List every clip you turned into an
`fn` on counterfactual evidence, since those are the ones worth a human spot
check. Then run `python -m tools.aeb_agent audit` and `score`, and say what moved.

Note separately any clip where you think the counterfactual itself is wrong. The
detector is new and its failure modes are not fully known; a disagreement you can
argue from the numbers is more useful than a label you changed against your
judgement.
