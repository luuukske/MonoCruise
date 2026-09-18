# ACC tracker tests

Two layers. Everything except `test_corpus_baseline.py` runs in CI.

| File | Needs clips | Covers |
|------|-------------|--------|
| `test_trail_arc_geometry.py` | no | Trail fit + ego-row crossing against synthetic straight and curved roads; arrival angle against the road and its range ramp |
| `test_road_model.py` | no | Shared centreline fit: curvature recovery, per-source offset elimination, lane-change rejection, fallbacks |
| `test_scoring_evidence.py` | no | Evidence gating of the offset term, score clamp vs consumer confidence |
| `test_tracker_validation.py` | no | Tracker on synthetic traffic: lock, release, stationary validation latch, in-lane lock past a bend |
| `test_overlay_confidence_gate.py` | no | ACC controller braking authority vs tracker confidence |
| `test_standstill_hold.py` | no | Standstill hold released by wanted accel, not lead speed: margin, rest, crawl, snap and raw-gap rules |
| `test_blinker_offset.py` | no | Blinker candidacy (R0-R4/R9-R15) and arbitration (R5-R8) fixtures |
| `test_lead_failsafe.py` | no | Geometric rescue gates, reasons, ramp, and survival of a poisoned road model |
| `test_corpus_baseline.py` | **yes** | Replay of the local AEB clip store, bounded against recorded metrics |

## Harness

`harness.py` drives `TrafficReader` + `ACCTracker` over recorded AEB clips,
mirroring `RadarThread.loop` and `ACCThread.loop`: same 25-sample ego position
history, same `ego_curvature_from_history`, same reanchor-on-unpause, same dt
clamp. Older clips lack blinker fields (default false); newer clips replay
the recorded stalk. Corpus baselines stay bit-identical while `b_eff` is 0.

`make_vehicle()` builds a synthetic `Vehicle` with a seeded position history for
the non-clip tests. Pass `history_speed` with `speed=0` to model a vehicle that
drove in and then stopped, which is the case the validation latch exists for.

## Metrics

| Metric | Definition |
|--------|------------|
| `lock_s` | Corridor entry (`in_path` rising edge) to `score > 1` |
| `cutin_lock_s` | Same, for tracks known at least 1 s before entering the corridor |
| `fresh_lock_s` | Same, for tracks that entered the corridor within 1 s of first sight |
| `hook_s` | Corridor exit while locked, to `score <= 1` |
| `moving_lock_rate` | Fraction of moving in-corridor frames at `score > 1` |
| `stationary_lock_rate` | Fraction of sub-1 m/s tracked frames at `score > 1` |
| `saturated_rate` | Fraction of positive-score frames pinned at `SCORE_MAX` |
| `zero_conf_overlay_rate` | Fraction of overlay-tripping frames whose lead scores at or below the confidence floor |

The lock split matters. A cut-in has been tracked while alongside, so it already
carries full trail evidence and locks on geometry alone; a fresh id has almost
no history, so it must accumulate evidence first. Aggregate `moving_lock_rate`
mixes the two and moves whenever fresh-id latency changes, which makes it a poor
headline number on its own.

## Recorded baselines

Bounds in `test_corpus_baseline.py` come from a measured run over the oldest 60
clips in the local store, not from theory. Tighten them when the tracker
improves. Never loosen one to go green: that is the AGENTS.md rule for every
baselined invariant in this repo.

Measured before and after the evidence-gating work, same 60 clips:

| Metric | Before | After |
|--------|--------|-------|
| cut-in lock p50 / p90 | 1.03 / 2.78 s | 0.94 / 2.17 s |
| fresh-id lock p50 / p90 | 0.07 / 0.34 s | 0.65 / 1.12 s |
| hook p50 / p90 | 0.44 / 2.45 s | 0.65 / 1.20 s |
| moving in-corridor locked | 52.6 % | 46.9 % |
| stationary locked | 7.1 % | 4.5 % |
| score saturated | 44.9 % | 52.4 % |

Fresh-id locks got slower on purpose: the old tracker locked them in about two
frames on a fabricated offset. Saturation rose because the ceiling dropped from
20 to 8, so reaching it is cheap and no longer buys release latency.

### Step 3: road model and uncertainty gate

`moving_lock_rate` stops being comparable here, because the gate changes which
frames count as in-corridor and so moves the denominator. The comparable measure
is the tracker's shipped `score > 1` decision scored against **ego's own future
path** (`gateeval`, 40 clips):

| | step 2 | step 3 |
|---|---|---|
| stationary false-lock, overall | 5.1 % | **2.4 %** |
| stationary false-lock, 45-70 m | 11.4 % | **1.9 %** |
| stationary false-lock, 70-100 m | 2.4 % | **0.0 %** |
| stationary false-lock, 20-45 m | 7.4 % | 7.4 % |
| moving in-lane recall, overall | 51.0 % | **54.0 %** |
| moving in-lane recall, 45-70 m | 67.2 % | **73.9 %** |
| moving in-lane recall, 70-100 m | 17.4 % | **20.8 %** |
| moving false-lock, overall | 2.5 % | **2.0 %** |

Harness metrics over the same 60 clips: hook p90 1.20 -> 0.85 s, stationary
locked 4.7 -> 4.4 %, lock p50 0.88 -> 0.73 s. Fresh-id lock p90 rose 1.30 -> 3.55
s on n around 45; that tail is the gate declining to call a distant unconfirmed
target in-lane, which is the intended behaviour, so the recorded bound guards the
median instead.

### Step 3b: heading prior, hysteresis, distance weighting

Same 40 clips, `score > 1` against ego's future path:

| | before | after |
|---|---|---|
| moving in-lane recall, overall | 54.0 % | **57.5 %** |
| stationary false-lock, overall | 2.4 % | **1.6 %** |
| stationary false-lock, 20-45 m | 7.4 % | **4.6 %** |
| in-lane recall, 20-45 m | 90.2 % | **93.3 %** |
| in-lane recall, 45-70 m | 73.9 % | **78.5 %** |
| moving false-lock | 2.0 % | 2.7 % |

**Classification vs lock** is the diagnostic that mattered. For genuinely in-lane
moving targets, `in_path` correctness and `score > 1` diverge sharply with range:

| band | in_path | score > 1 | gap |
|------|---------|-----------|-----|
| 20-45 m | 96.9 % | 93.3 % | 3.5 pts |
| 45-70 m | 88.1 % | 78.5 % | 9.6 pts |
| 70-100 m | 54.5 % | 22.6 % | 31.8 pts |

A large gap means the tracker classified correctly and then failed to accumulate;
a low `in_path` means it never classified at all. Split them before tuning
anything: hysteresis fixes the second and does almost nothing to the first, while
the path decay base fixes the first and nothing else does.

**Ground truth without labels.** A first attempt used each vehicle's lateral
offset at closest approach. It produced **zero** positive samples, because ego
never passes the vehicle it is following, so closest-approach only ever selects
overtaken and shoulder traffic. The working version measures the minimum distance
from a target's position to the polyline ego actually drove afterwards, and only
scores samples where ego's path got past the target. A clip that ends with ego
stopped behind a lead contributes nothing rather than a false "shoulder".

### Step 4: arrival angle read against the road at range

`core/acc/README.md` §9 has the mechanism and the hindsight-lane numbers (1062
clips). Harness metrics over the test sample, before -> after:

| | before | after |
|---|---|---|
| moving in-corridor locked | 40.7 % | 57.6 % |
| stationary locked | 3.1 % | 5.0 % |
| score saturated | 67.6 % | 74.1 % |
| cut-in lock p90 | 3.06 s | 1.87 s |
| lock p50 / p90 | 0.61 / 2.87 s | 0.51 / 1.60 s |
| hook p90 | 1.20 s | 1.23 s |

The saturation bound was raised to 0.76 with this change, the one loosened bound:
in-lane traffic at range stopped being rejected and now reaches the ceiling, and
the release latency that bound exists for is pinned directly by the hook test,
which still passes. The recall floor and the cut-in bound were tightened.

### Step 5: curvature prior and dropping a dead carried shape

`core/acc/README.md` §9 has both mechanisms and the corpus numbers. Harness
metrics over the test sample, before -> after:

| | before | after |
|---|---|---|
| moving in-corridor locked | 57.6 % | 59.2 % |
| stationary locked | 5.0 % | 5.0 % |
| score saturated | 74.1 % | 73.5 % |
| cut-in lock p90 | 1.87 s | 1.60 s |
| lock p50 / p90 | 0.51 / 1.60 s | 0.51 / 1.40 s |
| hook p90 | 1.23 s | 1.20 s |

### Step 6: trust replaces confidence as the blend weight

`core/acc/README.md` §9 has the rule, the corpus numbers and the rejected
raw-publish variant. Harness metrics over the test sample, before -> after:

| | before | after |
|---|---|---|
| moving in-corridor locked | 59.2 % | 58.9 % |
| stationary locked | 5.0 % | 4.8 % |
| score saturated | 73.5 % | 73.6 % |
| cut-in lock p90 | 1.60 s | 1.84 s |
| lock p50 / p90 | 0.51 / 1.40 s | 0.58 / 1.57 s |
| hook p90 | 1.20 s | 1.03 s |

The sample metrics barely move because this change acts at range, where the test
sample has few labelled targets; the 1974-clip in-path measurement in §9 is the
one that shows it. Cut-in p90 moves 0.24 s here against 0.006 s [-0.145, +0.125]
over all clips, which is the n = 60 noise the README warns about.

## Running

```bash
python -m pytest tests/acc -q
```

The corpus test needs `%LOCALAPPDATA%/MonoCruise/aeb_clips` and skips cleanly
without it under the `needs_clips` marker.
