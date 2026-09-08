# Prompt: find AEB saves worth putting in the v1.1 trailer

Builds a ranked, bucketed shortlist of clips whose scenarios would carry a release
video. Not a labelling pass: nothing here changes a label.

Paste the block below into Cursor.

---

Build a shortlist of AEB events from the clip corpus for the v1.1 release trailer.
The target is a montage board of saves: low speed, high speed, and above all
scenarios that *look* dangerous, plus a few honest failures.

## Two things to be clear about before you start

**The footage is the debug view, not game capture.** A clip holds radar frames,
telemetry, the AEB decision stream and one 240x135 screenshot, so no game video
can be cut from it. What *can* be produced directly is the top-down AEB view, as
a PNG sequence:

```bash
python -m tools.aeb_agent render <id> --start 6.5 --end 10.0
```

That is the footage this shortlist feeds. Rank on what the geometry will look
like rendered: how the corridor sweeps, whether the threat box turns red, how far
the arc reaches. Do not rank on the screenshot, which is illegible by design.

**Both stores are in scope, and contributed clips need a credit.** Search the
local and contributed stores. Every contributed clip that reaches the shortlist
carries an on-screen credit reading **"anonymous contributed clip"** when it is
published, which is the condition this footage is used under.

Record `contributed: yes` on every such entry, and group the contributed picks
into their own list at the end of the shortlist so the credits are easy to apply
in the edit. `render` writes a `CREDIT.txt` next to the frames of any contributed
clip; do not delete it.

## The buckets

Aim for this shape. If a bucket is thin, say so rather than padding it with weak
candidates.

| Bucket | Target | What it is |
|---|---|---|
| High-speed save | 6-8 | 80 km/h and up, hard AEB brake, big speed drop |
| Full stop from speed | 4-6 | Comes to a complete standstill from 40 km/h or more |
| Head-on | 4-6 | Oncoming vehicle on ego's side of the road |
| Junction / red-light runner | 4-6 | Crossing traffic entering ego's path |
| Stopped obstacle | 3-5 | Stationary vehicle or queue tail appearing ahead |
| Lead vehicle crashes | 3-5 | The lead itself collides or stops violently |
| Low-speed save | 3-4 | Under 35 km/h, town or queue |
| Honest failure | 2-3 | AEB braked and contact happened anyway |

The honest failures matter. A trailer that shows only perfect outcomes reads as
marketing; one that shows a save it could not make reads as confident. Pick
failures where the physics were plainly impossible, not where AEB was late.

## Getting the data

Everything you need is in the feature cache. Pull JSON and rank it yourself:

```bash
python -m tools.aeb_agent list --json > /tmp/all.json
python -m tools.aeb_agent list --label tp,good_intervention --json
python -m tools.aeb_agent list --scenario oncoming --json
python -m tools.aeb_agent list --scenario crossing --json
python -m tools.aeb_agent list --trigger auto_crash --json
```

For a candidate you are seriously considering, read the full dossier:

```bash
python -m tools.aeb_agent show <id> --scenes 3
```

The `TIMELINE` is what tells you whether it reads as dramatic: how fast the gap
closed, how hard the brake came on, whether the truck stopped.

## What makes a clip trailer-worthy

Rank on these, in roughly this order:

1. **The save is visible.** AEB braked (`brake ticks` well above zero), the peak
   target decel is high, and ego's speed actually collapses. A clip where AEB
   warns and nothing else happens shows nothing.
2. **The threat is legible in one second.** A viewer gets no context. A truck
   appearing across a junction reads instantly; a subtle lane-edge case does not.
3. **Speed.** Higher is more dramatic, and the speed drop is the payoff.
4. **Closing geometry.** Low `tgeom`, small corridor gap, high closing rate.
5. **Scene is clean.** One clear threat beats six vehicles where nobody can tell
   what the brake was for.

## What to exclude, without exception

- **Lag or desync.** The dossier shows `lag_confirmed` ticks or teleports on the
  target. It will look like a bug on video, because it is one.
- **Degenerate counterfactuals and spawn piles.** Targets at zero range from the
  first frame.
- **Paused, truncated, or very short captures.** Check `FLAGS`.
- **Anything labelled `ignore`** for a data-quality reason.
- **Phantom brakes.** Anything labelled `fp` stays out, however dramatic. Putting
  a false positive in a trailer is a promise you will be held to.

TMP clips are fine. The debug view draws no player names and no game imagery, so
a TMP scene renders the same as a singleplayer one.

## Deliverable

Write `tools/aeb_corpus_run/agent/trailer_shortlist.md`, grouped by bucket. One
entry per clip:

```
### 84cdd6b3  high-speed save
speed        144 -> 3 km/h, peak AEB decel 8.9 m/s2, stopped
threat       stationary queue tail, corridor gap 6.2 m, tgeom 1.1 s
why it reads well  one clean target, huge speed drop, nothing else in frame
session      SP
contributed  no
render       --start 6.4 --end 10.2  (brake onset 7.9)
caveats      none
```

Give every entry a suggested `render` window: open about 1.5 s before the brake
so the approach reads, and close once the truck is stopped or the threat is past.

Then export the screenshots as a contact sheet for a human to flip through:

```bash
python -m tools.aeb_agent thumbs --ids <comma list> --out trailer
```

The screenshots are deliberately low resolution and no text is legible in them.
They are for judging layout only. Do not try to read a speedometer off one, and
do not describe anything you think you see in an image: work from the numbers.

## Report

Per bucket: how many candidates you found, the top picks, and any bucket that came
up short. State how many picks are contributed and therefore need the credit. Then list the five you would open the trailer with, and say why in one
line each. Flag any candidate whose drama depends on something you are not certain
of from the numbers.
