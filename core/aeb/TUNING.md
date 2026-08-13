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
| `aeb_reserve_release_s` | 0.0 (held) | How fast the build-up reserve latched at engagement bleeds off. **0 holds it for the event**, which is what removes the fade: on clip e9fb04c9 the command fell 10.68 to 0.41 m/s² across 2.4 s with the warn still sounding, because a reserve recomputed from live `v_closing` hands its metres back as ego slows. Held, the command holds, the event is 0.7-0.9 s shorter and the residual gap grows from 3.1 m to 5.0 m at 100 km/h. **The cost is headroom**: the command then sits at the cap for ~80% of the event, so a lead that suddenly brakes harder gets no increase, AEB is already at maximum. Setting 0.35 (the measured build-up) frees that headroom and answers a lead step in 0.16 s, but spends the whole margin: the p90-lag stop lands on the obstacle (−0.02 m) and so does the loaded double against its measured 4% model over-read. Paying for it with a larger `stop_buffer` was tried at 2.5 m and rejected: it re-introduces braking while creeping up to a stopped queue (`test_soft_crawl_rear_end_does_not_engage_without_slam`). Partial release does not reconcile them either, the two requirements have no overlap: responsiveness needs ≥0.75 of the reserve gone, the over-read case needs ≤0.5. That conflict is set by the engage bar, not by this knob |
| `aeb_engage_frac` | 0.85 | Fraction of `capability_decel` (not `effective_max`) at which a new engagement fires. Re-swept on the stop simulator after the base change: 0.85 is the knee. 0.90 leaves 0.07 m of residual at p90 brake lag and 0.95 collides, while 0.80 costs 2.2 m of trigger distance to buy 0.3 m of margin. The corpus sensitivity table under "Geometry-graded engage fraction" was priced against `effective_max` and no longer applies to these numbers |
| `ego_decel_frac` | 0.9 | Tracking headroom on the **command** only. It is not part of the entry bar: `engage_threshold` and `aeb_warn_near_full_frac` run off `capability_decel` (`capacity − downhill`), while the target cap, `aeb_disarm_frac` and `aeb_warn_frac` run off `effective_max`. Folding it into entry made the real bar 0.765 of capacity and pushed the 100 km/h engage point 3.7 m further out on a 13.89 m/s² double |
| `warn_ttb` | 1.3 s | WARN threshold |
| `brake_ttb` | 0.2 s | BRAKE threshold |
| `ego_half_width` | 1.265 m | Ego arc corridor half-width (flush trailer standoff 2026-08-11) |
| `ego_half_length` | 3.333 m | Ego capsule half-length (flush trailer standoff 2026-08-11; body extents via `capsule_extents`; collision segments are cap-aligned: extents minus half_width, see `core/radar/README.md` §8) |
| `corridor_margin` | 0.5 m | Corridor padding for crossing-path sample uncertainty |
| `stop_buffer_response_s` | 0.30 s | Brake build-up distance for a solo tractor (`v_closing * this`). Since the entry bar moved off `ego_decel_frac` this is the only entry margin, so it is sized on the measured plant: build-up is 0.25 s median and 0.37 s p90 over 61 fitted episodes, and 0.16 was under even the median. 0.42 (the p90) was simulated and rejected: the pad is a fixed *time*, so at crawl speed it eats most of a queue gap and AEB grabs the brake creeping up to stopped traffic. At 0.30 the weakest rig (bobtail) still stops clear under median lag and grazes 2 cm at p90 lag; at 0.16 it grazed under median lag. 0.36 would clear p90 too, but it puts the crawl case within 1.5% of engaging (`test_soft_crawl_rear_end_does_not_engage_without_slam`), which is too close to ship. The bobtail is the tight rig here: its capacity is low enough that the entry bar sits near the command cap, and no pad value fixes both ends. Rejected cap/tiering in calibration comments |
| `stop_buffer_response_trailer_s` | 0.40 s | Same term with a trailer attached, and the term that dominates trigger distance at speed (`0.40 * v` is 11 m at 100 km/h). Was 0.50, chosen when the plant was believed to be t63 ~0.65 s; fitting 61 recorded episodes puts build-up at 0.25 s median / 0.37 s p90 with no separable load split, so 0.50 was well past p90. At 0.40 every trailer rig still stops clear under p90 lag (worst residual 0.70 m) and the 100 km/h trigger comes in 2.8 m closer; 0.35 leaves 0.14 m and 0.30 collides. Kept above the solo value because the fit could not rule a split out |
| `aeb_target_rate_engaged_ms3` | 30 m/s³ | Target slew while engaged. The engagement edge itself is exempt and steps straight to the requirement: ramping from zero used to cost the entire brake build-up window |
| `aeb_engage_frac_certain` | 0.85 | **In-game trial from 2026-08-11**, was 0.70. Now equal to `aeb_engage_frac`, so the geometry grading is flat and aligned in-lane traffic no longer skips the uncertainty hedge (`tests/aeb/test_engage_sensitivity.py::test_graded_bar_brakes_earlier_on_an_in_lane_obstacle` fails by design while this holds). Note this knob does not soften braking: required decel is recomputed as the gap closes, so engaging later means engaging at a *higher* demand. On clip `ac6b48b4` the engage-tick command rises 6.26 to 7.61 m/s², and headroom below `effective_max` for the loop to recover a build-up shortfall drops from 30% to 15% |
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
