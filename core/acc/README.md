# MonoCruise ACC (in-lane vehicle tracker)

> ACC-specific logic. Coordinate system, Vehicle smoothing, ArcPath
> geometry, and the RadarThread snapshot shape are documented in
> `core/radar/README.md`: **read that first.** This file only covers
> what differs from the shared radar layer.
>
> Agent workflow and do-not-break rules: top-level `AGENTS.md`.

---

## 1. Scope

This module is an **in-lane vehicle tracker**, not a controller.

It watches the pre-filtered traffic stream from `RadarThread`, scores
each vehicle per-frame against the ego path, and publishes a ranked
list of in-lane leads on `ACCThread.data`. Consumers (currently the
cruise control thread, eventually a gap-control law) read that list
and decide what to do with it.

**No accel command, no accel cap, no gap control lives here.** That
all belongs in `core/cruise_control_thread/` when it gets written.
This module answers one question only: **"which vehicles are in
ego's lane right now, and how confidently?"**

---

## 2. Ego path: blended curvature

Tracking quality depends on an accurate "where is ego going" arc.
ACC blends two curvature sources (`core/acc/ego_path.py`):

| Source              | Where it comes from                    | Strength                    |
|---------------------|-----------------------------------------|-----------------------------|
| Steering-derived κ  | `userSteer * 0.17`                      | Leads actual trajectory:   |
|                     |                                         | correct below 15 km/h when  |
|                     |                                         | history-fit is collinear.   |
| History-derived κ   | `RadarData.ego_curvature`               | Accurate at speed:         |
|                     | (circumscribed-circle fit on last 25    | reflects real trajectory.   |
|                     | ego positions, from `RadarThread`).     |                             |

Blend weight is a linear ramp in `km/h`:

    ≤ 15 km/h  → 100 % steering
    ≥ 30 km/h  → 70 % history + 30 % steering
    linear between.

**AEB does not consume `RadarData.ego_curvature`**: see
`core/aeb/README.md`. AEB must react to the instantaneous yaw-rate
proxy. ACC can (and should) use the smoothed history fit because
tracking tolerates: and prefers: a little smoothing.

### Path half-width

    half = LANE_BASE_HALF_M + sin(min(|steer|·1.5, 1) · π/2) · LANE_FLARE_HALF_M

- `LANE_BASE_HALF_M = 1.25 m`: 2.5 m corridor on a straight road.
- `LANE_FLARE_HALF_M = 2.0 m`: up to +2 m per side under heavy steer.

---

## 3. Scoring (meter-native)

Four components per frame (see `core/acc/scoring.py`). The formulas
mirror legacy ETS2radar `SCORING_REFERENCE.md` faithfully: the only
semantic departure is dt-scaling so the loop can run at any cadence.

| Component  | Units         | Range (approx.) | Formula (summary)                                                                 |
|------------|---------------|-----------------|-----------------------------------------------------------------------------------|
| offset     | dimensionless | [-1.6, +1.5]    | Gaussian `2^(-(x/σ)²)` on arc-crossing lateral (σ = 2.25 m), × distance_amp,      |
|            |               |                 | clamped ±1, × outer `1.5·(angle_amp·0.4 + 0.6)`, + baseline.                      |
| yaw        | dimensionless | [-1.5, 0.0]     | `(2^(-(|Δyaw|/90°)^5) - 1) · 1.5`.                                                |
| path       | dimensionless | [-4.0, +5.0]    | `1.03^(-d_m) · slow_amp · (1 - b²·0.4)`; `min(·, 5)` in / `-min(·×0.6, 4)` out.     |
| angle      |:             | 0.0 (reserved)  | Arc-arrival angle in radians. Legacy had it disabled: still is.                  |

### offset: constants and baselines

    σ                = 2.25 m
    distance_amp(d)  = [2^(-d/100) + 8/(d+3) - 1] / 3 + 1
    angle_amp        = 2^(-(normalised_arc_angle / 0.06)²)   # from trail-arc fit
    baseline         = 0.0 on arc hit
                     = -0.40 when the arc was fit but didn't cross ego row
                     = -0.16 when position history was too short to fit

Blinker lateral bias is applied as a **scalar offset shift** on the
scored lateral: `offset_for_score = lat - blinker · 4.5 m`: not as
an ego-arc translation. Shifting the arc geometrically would distort
arc-arc hit tests, which we don't want.

### Trail-arc fit (`core/acc/trail_arc.py`)

Each target's `_position_history` is downsampled (≥ 1 m AND ≥ 0.05 s
between kept samples) and then **LS algebraic circle-fit** in place.
Centre + radius come from the positions alone: no smoothed-yaw
input: so yaw jitter no longer translates into crossing-lateral
jitter. A max-perpendicular-from-chord straightness pre-check
prevents position noise from being fit as a tight curve. The fitted
circle is intersected with the **ego row** (line through ego
perpendicular to ego heading); the crossing point gives `offset_m`,
and the tangent direction there gives `arc_angle` which feeds
`angle_amp = 2^(-(arc_angle / 0.06)²)`.

The downsampling lives inside this module: radar's
`_position_history`, `Vehicle.curvature_from_history`, the TMP raw-
speed LS fit, and AEB consumers all see the dense raw buffer
unchanged. Only ACC's per-target trail consumes the gated subset.

Three buckets drive baseline selection:

| Bucket       | Trigger                                                       | Baseline | offset_m fallback | angle_amp |
|--------------|---------------------------------------------------------------|----------|--------------------|-----------|
| `HIT`        | fit + arc intersects ego row                                  | 0.0      | crossing lateral   | from arc  |
| `NO_ARC_HIT` | fit but no intersection (target's circle too tight, off-side) | −0.40    | current lateral    | 1.0       |
| `NO_HISTORY` | fewer than 5 samples / chord < 0.5 m / curvature_from_history None | −0.16 | current lateral | 1.0       |

Constants live in `core/acc/trail_arc.py`:
`_HISTORY_MIN_DIST_M = 1.0`, `_HISTORY_MIN_DT_S = 0.05` (per-fit
downsample), `_MIN_FIT_SAMPLES = 5`, `_MIN_PATH_LEN_M = 0.5`,
`_STRAIGHT_KAPPA_MAX = 1/2000` (below this κ the LS-fitted circle is
collapsed back to a straight line), `_MIN_SAGITTA_RATIO = 0.5` (the
observed perpendicular sagitta has to be ≥ 50 % of what the LS
radius would imply, otherwise the curve was fitted to noise),
`_ANGLE_AMP_SIGMA = 0.06` rad. Sweep direction (`sign`) is recovered
from the actual history (cross-product of prev-to-centre and
target-to-centre vectors), so it follows `ArcPath._sign` without
needing kappa to be signed up front: +1 ⇒ left turn ⇒ CW sweep
around the centre, per `max_sweep = -sign · arc_length / radius`.

The LS circle fit mean-centres `(x, z)` before solving the normal
equations so TMP world coordinates in the 10⁵ range do not collapse the
3×3 determinant (otherwise every trail reads as straight on real maps).

### path: slow_speed_amp and blinker reduction

    slow_amp          = 1.4 + (kmh / 100) · 4.1
    blinker_reduction = 1 - b² · 0.4       # b = signed blinker scalar; no clamp

At low speed the corridor effectively fattens because the path decay
is shallower relative to lane width. During a lane change the squared
blinker scalar reduces the path amplitude by up to 40 %, so targets
in the current lane stop pinning score while we're committing to the
move.

### Accumulation: `[-5, +15]` clamp

    delta     = total(components) · speed_mult(v_target) · dt · 30
    score_new = clamp(score_prev + delta, -5, +15)

- Asymmetric clamp: fast lock on in-path vehicles (wide positive
  ceiling), slow release on uncertain tracks (narrow negative floor).
- `speed_mult = max((|v_target| / 90 m/s)^0.8, 0.5)`: **target**
  vehicle speed, not ego's. The 0.5 floor applies at all realistic
  road speeds; only above ~60 m/s does it lift.
- The `· 30` multiplier exists so at dt = 1/30 s (legacy tick) the
  per-frame delta matches the legacy integer-tick maths. Higher or
  lower loop rates scale proportionally: the accumulation is
  frame-rate independent by design.

### In-path threshold

`score > 0` ⇒ in-path. Hard cut-off: only positive-score vehicles
appear in the published `leads` list.

---

## 4. Trailer → tractor swap

TMP trailers appear as separate `Vehicle` instances with their own
`position`, `speed`, `acceleration`. Their raw kinematics lag the
tractor (shared-memory pipeline) enough that any downstream gap
controller using them would command phantom braking.

`ACCTracker._top_leads` detects TMP-trailer leads and promotes the
pulling tractor's `speed` / `acceleration` into `LeadInfo.effective_*`.
The vehicle reference itself is unchanged so debug views still show
the trailer: only the kinematics exposed via `LeadInfo` get swapped.

### Nested trailers (road trains)

Only the tractor and the *first* trailer appear as top-level radar
vehicles. Every trailer behind the first is a nested `Trailer` on that
first trailer (AI trucks nest all of theirs the same way), so the
tracker: which iterates Vehicles: never saw them. On a multi-trailer
convoy that made the rearmost trailer, the one ego actually closes on,
invisible to scoring.

`RadarThread` now publishes those nested trailers wrapped as standalone
`Vehicle`s in `RadarData.trailer_vehicles` (see `core/radar/README.md`
§12). `ACCThread` scores them alongside `vehicles`: no tracker change,
they are just more entries in the list. Each carries its own position
history and smoothing (synthetic per-id continuity), so it scores and
locks exactly like a real vehicle, and a wrapped trailer lead still
goes through the trailer→tractor kinematic swap above.

TMP top-level trailers have no buffer parent link. `ACCTracker._resolve_tractor`
locks a tractor with a **strict** gate on first pick (longi 3–16 m, |lat| ≤ 1.5 m,
|Δyaw| ≤ 15°) and a **loose** gate when revalidating a cached pair (longi 1–25 m,
|lat| ≤ 4 m, |Δyaw| ≤ 60°). Among strict candidates, lowest
`|lat| + 0.05·|longi−10| + 0.2·|Δyaw|` wins. Wrapped nested trailers
(`id ≥ 1_000_000`) skip locking and use their own filtered kinematics.

---

## 5. Blinker lateral bias

Blinker state resolves to a signed scalar `b ∈ [-1, +1]` (`-1` full
left, `+1` full right):

- **Pinned at 1** on the indicated side while the blinker is held.
- **Cos decay to 0** over `_BLINKER_HOLD_S = 2.5 s` after release, so
  a short blink still covers the full manoeuvre. Implemented by
  stamping `_last_*_active` every active frame: on release, decay
  starts cleanly at `t = 0`.

The scalar is consumed in two places:
- **Scoring lateral shift**: `offset_for_score = lat - b · 4.5 m`.
  Targets in the indicated adjacent lane score near zero offset during
  the manoeuvre.
- **Path amplitude reduction**: `amp *= 1 - b² · 0.4`. Up to 40 %
  cut so the current-lane lead stops pinning score while we commit
  to the change.

On blinker **rising edge** at ego speed ≥ `_BLINKER_SCORE_RESET_KMH`
(65 km/h), all per-id scores are reset to 0 once. Legacy "highway
lane change" reset: clear the current lead completely so a new lead
can be picked up on the new side without inheriting residual score.

The ego arc itself is **not** translated: doing so would distort
arc-arc hit tests. Shift lives in scoring only.

---

## 6. Published data (`ACCData`)

    @dataclass
    class ACCData(ThreadData):
        enabled: bool            # Settings.acc_enabled & cc_mode == "Cruise control"
        has_lead: bool           # True when leads is non-empty
        lead_id: int             # leads[0].vehicle.id (or -1)
        lead_dist_m: float       # leads[0].dist_m
        lead_rel_speed_ms: float # leads[0].rel_speed_ms (lead - ego; neg = closing)
        lead_score: float        # leads[0].score
        leads: list[LeadInfo]    # top-3 ordered by score, post trailer-swap
        t_mono: float            # radar t_mono the snapshot is tied to
        _lock: threading.Lock

`LeadInfo`:

    @dataclass
    class LeadInfo:
        vehicle: Vehicle            # shared ref: READ ONLY
        score: float                # current accumulated score
        dist_m: float               # longitudinal distance along ego heading
        rel_speed_ms: float         # effective_speed - ego_speed (signed)
        effective_speed_ms: float   # tractor speed when vehicle is a TMP trailer
        effective_accel_ms2: float  # tractor accel when vehicle is a TMP trailer

Consumers always copy what they need under `_lock` and release it
before doing any further work:

    with acc.data._lock:
        leads = list(acc.data.leads)   # shallow list copy: refs are safe
        has_lead = acc.data.has_lead

---

## 7. Thread integration

### ACCThread (`core/acc/thread.py`)

- Runs at 30 Hz (radar cadence).
- Reads `RadarData` (vehicles + ego snapshot) and telemetry blinkers.
- Advances only when `t_mono` changed: paused / stale frames hold
  the previous lead list and skip tracker integration.
- Publishes to `self.data`.

### Registry names

    radar_thread    : produces RadarData.
    telemetry_thread: produces blinkerLeft / blinkerRight.
    acc_thread      : this module; exposes in-lane leads.
    cruise_control_thread: will consume acc_thread.data.leads
                            for gap-based accel control (not yet
                            implemented).

---

## 8. Critical rules

Agent-facing copy of these rules also lives in the top-level `AGENTS.md` (keep that in sync if you change them).

1. **Never mutate Vehicle instances from ACC.** They are shared
   references with AEB; `RadarThread` carries smoothing state forward
   across frames via `update_from_last`. A mutation here corrupts AEB
   next tick.
2. **AEB uses yaw-rate proxy, ACC uses history fit.** Codified in
   `core/aeb/README.md` and `core/radar/README.md` §11. Don't cross
   the streams.
3. **No control law lives in this module.** `core/cruise_control_thread`
   owns longitudinal decisions. If you're tempted to compute an
   accel cap here, stop and put it there instead.
4. **Scoring is meter-native.** New components / tweaks should be
   derived from physical quantities (metres, seconds, m/s), not
   pixels or empirical curves fitted to a specific resolution.

