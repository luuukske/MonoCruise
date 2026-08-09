# AEB calibration reference

> Lookup table for `AEBCalibration`. Pipeline architecture, filter behaviour and
> engagement rationale stay in `core/aeb/README.md`.

## Constants

All constants live in `AEBCalibration` (frozen dataclass, `core/aeb/calibration.py`).
`DEFAULT = AEBCalibration()` is the production singleton. Tests can pass a modified
instance to `build_pipeline(cal)` or `evaluate_frame(frame, cal)`.

| Constant | Default | Role |
|----------|---------|------|
| `full_brake_decel` | 7.8 m/s² | Full brake deceleration |
| `warn_ttb` | 1.3 s | WARN threshold |
| `brake_ttb` | 0.2 s | BRAKE threshold |
| `certain_engage_ttb` | 1.30 s | Certain-geom soft rear-end engage OR: `best_ttb_engage` under this qualifies (between the 0.50 s slam and warn_ttb); see README certain-TTB bridge |
| `ego_half_width` | 1.15 m | Ego arc corridor half-width |
| `ego_half_length` | 3.0 m | Ego capsule half-length (body extents via `capsule_extents`; collision segments are cap-aligned: extents minus half_width, see `core/radar/README.md` §8) |
| `corridor_margin` | 0.5 m | Corridor padding for crossing-path sample uncertainty |
| `stop_buffer_response_s` | 0.10 s | Response-lag distance in required-decel gap (`v_closing * this`). Last-point envelope, not corpus timing optimum; rejected cap/tiering in calibration comments |
| `disarm_hold_ttc_s` | 3.0 s | Geometry latch window while engaged: hold the event while any colliding target's unbraked ttc is inside it (anti-pumping; was `warn_ttb`) |
| `capsule_parallel_margin_scale` | 0.3 | Near-parallel capsule contacts use `margin * scale` blended by heading sine toward full margin at perpendicular; kills adjacent-lane side-graze FPs (ab524f87 / 29bf31b8). 1.0 disables |
| `lane_half_width` | 1.95 m | EGO lane boundary |
| `lane_separation` | 3.9 m | Road lane pitch |
| `out_of_lane_scan_samples` | 10 | OutOfLaneParallelFilter horizon lane scan count |
| `stationary_ool_graze_min_m` | 0.90 m | Stationary straddle: closest centreline sample must stay above this |
| `stationary_ool_graze_max_m` | 1.50 m | Stationary straddle: closest sample must stay at or below this |
| `stationary_ool_span_scale` | 1.5 | Stationary straddle: farthest sample ≥ `lane_half_width ×` this |
| `head_on_dot` | -0.7 | `head_on` flag threshold |
| `co_directional_dot` | 0.7 | `co_directional` flag threshold |
| `evasion_g` | 0.08×9.81 | Ego evasion lateral accel |
| `oncoming_body_sep_miss_scale` | 0.25 | OppositeLane body-sep also needs measured miss ≥ clear_bar × this |
| `oncoming_body_sep_soft_m` | 0.80 m | Pose clear when `d_abs ≥ clear_bar − soft` (EGO lane allowed) |
| `oncoming_closing_dmiss_rate_mps` | −1.5 m/s | Turn-into-path: miss closing this fast with ego turning |
| `oncoming_closing_lat_m` | 0.85 m | Turn-into-path: straight `|lat|` must collapse under this |
| `oncoming_closing_dabs_lat_ratio` | 10.0 | Turn-into also needs `d_abs ≥ \|lat\| × ratio` (adjacent vs inflated) |
| `max_evasion_lat_g` | 0.35×9.81 | Refuse Opp/TmpCross suppress when required `a_lat` exceeds this and `|lat|` clears the stage arm |
| `max_evasion_min_lat_m` | 7.0 | OppositeLane arm: `|lat|` must reach this (or `clear_bar`) before max-g refuse |
| `max_evasion_min_lat_m_opp_fast` | 4.5 | Opp arm when target ≥ `max_evasion_opp_fast_kmh`. Provisional n=1 (`0af8aedb`); corpus ablation finds no twins |
| `max_evasion_opp_fast_kmh` | 60.0 | Target speed for opp_fast arm. Provisional; do not retune as if corpus-fit |
| `max_evasion_min_lat_m_tmp_cross` | 3.0 | TmpCross arm (lower): cca-class TMP never hits Opp head_on |
| `tmp_cross_in_corridor_pass` | false | TmpCross: pass when `|lat| ≤ lane_half_width` ahead (and miss closing) |
| `evasion_max_dkappa` | 0.008 /m | Max curvature offset for evasion arcs |
| `opposite_lane_kappa_scale` | 2.0 | Kappa multiplier when target in own lane |
| `turning_diverge_kappa` | 0.007 /m | Corner threshold for Fix-C/D conditions; also the straight/turning split in `TmpCrossTrafficFilter` |
| `tmp_cross_center_hit_dist` | 2.5 m | `TmpCrossTrafficFilter` straight-snapshot genuine-crosser threshold: centre closest-approach at/below this is a real T-bone (pass), above is a body-graze clear (suppress) |
| `co_same_turn_lookahead_scale` | 0.5 | Extended lookahead fraction of horizon |
| `diverge_dip_samples` | 8 | `_is_approaching` window samples for the in-lane pass-through dip check |
| `aeb_engage_confirm_s` | 0.06 s | Sustained-qualification wait for near-certain engagement entries (3rd tick at 30 Hz) |
| `aeb_engage_confirm_oblique_s` | 0.40 s | Sustained-qualification wait for oblique out-of-lane entries (extrapolation-fragile class); 0.40 silences three FPs on the labelled corpus for +2 FN |
| `aeb_warn_confirm_oblique_s` | 0.30 s | Warn persistence for oblique out-of-lane threats; keeps ≥ 0.1 s warn lead ahead of an oblique engagement. 0.30 silences short clear-pass flicker (the 2 s head-on bend unit case no longer warns). Pair with `aeb_warn_confirm_vetoed_s` and `aeb_warn_frac` |
| `aeb_warn_confirm_vetoed_s` | 1.00 s | Warn persistence when **every** colliding target is engage-vetoed **and** outside ego's lane band: the extrapolation-phantom class. Latency, never silence. Saturates at 1.0 s |
| `aeb_warn_frac` | 0.60 | Fraction of `effective_max` at which `AEB_warn` rises via demand. Raised from 0.50 with the persistence windows to cut highway oncoming beeps |
| `aeb_warn_near_full_frac` | 0.85 | Demand fraction above which the warn cue survives the user-braking suppression. Currently equal to `aeb_engage_frac`, so any engagement warns even while the driver brakes; raising it restores a quiet band but silences the cue on under-braking drivers |
| `user_brake_latch` | 0.12 | Top of the FF-assist ramp: at or above this the sub-engagement assist applies at full weight |
| `ff_assist_ramp_lo` | 0.03 | Bottom of the FF-assist ramp; below it the assist contributes nothing. Matches `_USER_BRAKE_LATCH_THRESHOLD` so one notion of "the driver is braking" governs both the warn cue and the assist |
| `aeb_confirm_occupancy` | 0.6 | Min qualified fraction over the trailing confirm window for the three `OccupancyConfirm` streaks (risk / engage / warn) to fire |
| `aeb_confirm_max_gap_frames` | 2 | Max consecutive unqualified frames tolerated before a confirm streak drops; absorbs isolated collision-grid / TMP-jitter dropouts |
| `aeb_certain_fwd_dot` | 0.90 | `|fwd_dot|` above which an in-lane colliding target is "certain" and skips the confirm wait |
| `aeb_warn_confirm_oncoming_s` | 2.00 s | Warn occupancy while *every* colliding target is head-on / near-head-on (opposite-carriageway phantom class) |
| `aeb_warn_confirm_wide_lat_s` | 0.50 s | Warn occupancy while *every* colliding target projects past `aeb_warn_wide_lat_m` off the ego arc |
| `aeb_warn_wide_lat_m` | 4.0 m | Arc-lateral offset above which a colliding target counts as "a full lane over" for the warn gate |
| `aeb_warn_wide_lat_sticky_s` | 0.20 s | How long the wide class survives a lapse, so a target closing under the bar cannot buy back the instant warn |
| `aeb_warn_instant_min_s` | 0.05 s | Raw-warn floor under the certain-geometry instant bypass; kills single-frame demand spikes. Costs warn lead, see below |
| `aeb_warn_ttb_needs_narrow` | True | An all-wide-lateral set clears the wide-lateral window even under the TTB slam, which presumes an in-path target |
| `aeb_warn_max_range_m` | 90 m | Raw warn is dropped when the nearest colliding target is past this; no genuine corpus warn opens beyond ~80 m |
| `corner_entry_min_road_bend` | 0.10 rad | Min ego↔tangent angle for Mode-B suppression |
| `corner_entry_min_lateral` | 0.4 m | Min |lat_signed| to claim "off ego axis" (Mode B) |
| `corner_entry_lateral_tol` | 1.5 m | Chord-offset tolerance for arc-consistency check (Mode B) |

**Warn comfort status (residual false_warn 3 on the 610-clip store):** the gate
took `false_warn` 43 -> 3 with no verdict other than false_warn moving. The
three left (`075d163a`, `6f5a1555`, `75f2969c`) were reviewed and accepted:
`075d163a` now fires only its one reasonable trigger, the other two are
persistent converging geometry no warn knob reaches.

`aeb_warn_instant_min_s` is the knob to revisit first. It buys exactly one
false_warn (`9fa4c844`) and costs two extra genuine clips plus a third of the
meaningful warn lead (clips with >= 0.3 s lead, 21 -> 14). Set it to 0 to trade
back. `aeb_warn_confirm_wide_lat_s` must not go past 0.6: at 0.7 it drops
`88f8223d`, a real converging crosser. Full frontier, per-clip traces, and the
rejected levers (demand floors, slope gate, speed-ratio carve-outs,
co-directional out-of-lane window): `tools/aeb_corpus_run/progress.md`.

---
