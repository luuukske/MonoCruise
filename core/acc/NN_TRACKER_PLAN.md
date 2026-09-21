# ACC neural in-lane tracker: v1.2 plan

Status: research done, not started. Target implementer: a capable coding agent, later.
Read `core/acc/README.md` and `core/acc/ACC_ARCHITECTURE.md` first; `AGENTS.md` invariants still apply.

## 0. Verdict

Ship with changes. Replace the **geometry estimator** (road model, smoother, trail arcs,
ego-arc blend, corroboration, score integrator) with one learned per-vehicle in-path
probability. Keep everything that is a contract or a safety net: `ACCData`/`LeadInfo`,
geometric failsafe (`failsafe.py`), blinker overlay (`blinker.py`), trailer lock, elevation
gate consumption, the gap law in cruise. "Fully end to end" up to the pedal is rejected:
the gap law is measured and pinned by dozens of tests; the failure is in lane estimation only.

Root cause being fixed: past ~45-60 deg of bend the road model's confidence collapses,
the blend falls back to a trailing ego arc, score decays, lead leaves `leads[]`
(README section 9). Corpus at R<80 m is still only ~50% confident.

## 1. Data (measured on the local corpus, 2231 clips, 677k live frames)

- Clip: ~11 s, ~330 radar frames at 30 Hz, up to 24 vehicles, trails on a 0.5 m grid.
  Replay via `tests/acc/harness.py` pattern (`replay_frames` + `TrafficReader.replay_frame`
  + elevation gate). Full single-thread replay: 15.6 min; cache once to `.npz`, use
  multiprocessing per clip.
- Curvature is not rare: 21% of frames |kappa|>1/80, 43% of clips hold a >=1 s bend.
- Hindsight label = lateral offset of the vehicle's **current** pose from ego's **future**
  XZ polyline in the same clip (5 m longitudinal slack, pause breaks the path).
  Yield: 4.77M ahead pairs, 63% uncensored, 548k positives (|lat|<1.25 m), 261k frames
  with >=1 positive. Positives thin out on bends (10% of uncensored vs 18% straight).
- Ahead vehicles/frame mean 7, p90 15. 60% of pairs have a trail >=10 points.

Label rules (`tools/acc_nn/labels.py`):
- positive |lat|<1.25 m, negative |lat|>2.5 m, ignore 1.25-2.5 m. Also emit soft target
  `lat_m` for an auxiliary regression.
- censored pair (clip ends before ego reaches the vehicle's arc position): mask out.
- ego lane change guard: positive only if vehicle yaw is within 20 deg of the future-path
  tangent at the reach point, and only for co-directional movers; parked and stopped bodies
  keep the raw geometric label.
- exclude `is_trailer` bodies (tractor gets the label; trailer lock resolves at runtime).
- never feed ego's future into the input. Trail dropout 30% at train time so the net works
  on fresh ids without a trail.
- splits by **clip**, never by frame. Hold out a curvature-stratified set: 15% of clips,
  with high-kappa clips held out at the same rate as they occur.

## 2. Model (VectorNet-lite + DeepSets, ~15-25k params)

Ego frame: translate to ego, rotate so ego yaw = 0, left-right flip and small yaw jitter
as augmentation (flip Y, yaw, steer, blinkers together).

Inputs per frame:
- ego token: speed, steer, history kappa (`ego_curvature_from_history`, 0 + valid bit when
  None), blinker L/R lamps, optional previous-frame speed and steer (2-frame stack, no RNN).
- per vehicle (pad to 24, validity mask): x, z, rel yaw (sin, cos), speed, rel speed,
  length, width, `is_parked`, `is_tmp`, `has_trail`, dist.
- per vehicle trail: up to 32 vectors `[xs, zs, xe, ze]` in ego frame + mask.

Blocks: trail vector MLP 4->32->32, masked max-pool -> trail embedding 32.
Agent MLP (own feats ++ trail emb) -> 64. Global masked max-pool of agent embs ++ ego
token -> 64, broadcast back, per-agent head 64->32->2 (logit in-path, lat_m).
No attention in v1; add one masked self-attention layer only if cut-in latency fails the
gate below.

Loss: masked BCE on in-path + 0.2 * Huber on `lat_m` (uncensored pairs only).
Sample weight x3 for frames with |kappa|>1/80. AdamW, weight decay 1e-3, early stop on
held-out clip NLL. Temperature-scale on held-out clips afterwards (one scalar).

## 3. Training tooling (`tools/acc_nn/`, dev-only deps)

- `dataset.py`: corpus -> per-clip `.npz` cache (features, labels, masks, kappa bins).
- `train.py`: torch, CPU is fine at this size. Writes `weights.npz` (<200 KB) + `meta.json`
  (feature normalisation, temperature, git sha, corpus size, metrics).
- `eval.py`: per-curvature-bin PR / top-1 / ECE, time-to-lock on cut-in, time-to-drop on
  cut-out, adjacent-lane FP rate on bends, id-switch rate. Same replay path as
  `tests/acc/harness.py` so numbers are comparable to the geometric tracker.
- torch goes in a `requirements-dev.txt`, never in `requirements.txt`.

## 4. Runtime (`core/acc/nn_tracker.py`)

- Forward pass hand-written in **numpy** from `weights.npz`: a handful of `matmul`s on
  `24 x d` matrices, ~1e6 MACs, well under the 2 ms budget at 30 Hz. No torch, no
  onnxruntime. Pure Python is too slow at this MAC count.
- numpy is a **new runtime dependency** (~30 MB in the one-folder build): add to
  `requirements.txt`, `hiddenimports` in `monocruise.spec`, weights + meta as `datas`.
  Load via importlib resources, never an absolute path.
- Post-processing per track id: logit EMA (tau ~0.3 s), hysteresis enter 0.6 / exit 0.3,
  map smoothed probability onto the existing score range [-5, +6] so `LeadInfo.score`
  and the controller's 0.5..5.0 saturation keep their meaning. Fresh ids may enter
  quickly; the EMA is seeded at the first logit, not at 0.
- Keep unchanged and downstream of the NN: `failsafe.py` (floors published score only),
  `blinker.py` indicated-lane candidacy and commit, `trailer_lock.resolve_tractor`,
  `off_surface_ids` filtering, `IN_PATH_THRESHOLD`, closest-first top-3, never mutating
  `Vehicle`.
- Any exception in the NN path (missing weights, numpy import, shape error) logs once and
  falls back to the geometric tracker for the session. ACC must never publish nothing
  because a model file is broken.

## 5. Integration and rollout

1. `Settings.acc_tracker = "geometric" | "neural" | "shadow"`. Default `shadow` for one
   preview: geometric drives, NN runs alongside, disagreement rate and NN latency go to
   `ACCData` debug dicts and the AEB debug overlay. No popups.
2. Flip default to `neural` after shadow numbers are clean in-game on bends.
3. Delete road model / smoother / trail arc / corroboration one release later, together
   with their tests and README sections 4-9. Until then both trackers stay behind the flag.

## 6. Acceptance gates (all must hold before the default flips)

- `tests/acc/test_corpus_baseline.py` bounds hold on the NN: moving lock >=50%,
  stationary lock <=6%, saturation <=76%, cut-in p90 <=2.5 s, fresh-id p50 <=1.0 s,
  hook p90 <=1.3 s. Do not widen a bound to pass.
- New: on held-out clips with |kappa|>1/80, in-path top-1 accuracy and lock retention are
  at least as good as the geometric tracker's on the same clips, and lead loss events
  inside a bend (positive label present, `leads[]` empty for >0.5 s) drop by >=50%.
- Adjacent-lane FP rate on bends not worse than geometric.
- p99 forward-pass time <2 ms on the CI runner; ACC loop never blocks >0.5 s.
- `pytest -m "not needs_clips"` green without the corpus: unit tests use a tiny fixture
  weights file and synthetic frames; corpus tests carry `needs_clips`.
- Repo hygiene: no absolute paths, no em dashes, comment budget, torch not imported
  anywhere under `core/`.

## 7. Risks, ordered

1. Label bias: ego's future path encodes the driver's intent, not the lane. Mitigated by
   the lane-change guard, blinkers as input, and the blinker overlay staying a policy.
2. 37% censoring biases toward near targets. Mask, do not guess; consider longer
   `post_s` capture in a later release if far-target recall is weak.
3. Bends have fewer positives (38k frames). Reweighting first; if per-bin metrics still
   lag, record bend-heavy clips deliberately (capture already exists).
4. numpy as a shipped dependency: size and one more native wheel for AV heuristics.
   Conventional, but flag it in release notes and the AGENTS.md packaging section.
5. Learned model regressions are silent. Shadow mode plus per-bin eval is the guard;
   keep the geometric fallback code for one release.
