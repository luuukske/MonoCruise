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
| `aeb_reserve_release_s` | 0.35 | How fast the build-up reserve latched at engagement bleeds off. **0 holds it for the event**, which is what removes the fade: on clip e9fb04c9 the command fell 10.68 to 0.41 m/s² across 2.4 s with the warn still sounding, because a reserve recomputed from live `v_closing` hands its metres back as ego slows. Held, the command holds, the event is 0.7-0.9 s shorter and the residual gap grows from 3.1 m to 5.0 m at 100 km/h. **The cost is headroom**: the command then sits at the cap for ~80% of the event, so a lead that suddenly brakes harder gets no increase, AEB is already at maximum. Setting 0.35 (the measured build-up) frees that headroom and answers a lead step in 0.16 s, but spends the whole margin: the p90-lag stop lands on the obstacle (−0.02 m) and so does the loaded double against its measured 4% model over-read. Paying for it with a larger `stop_buffer` was tried at 2.5 m and rejected: it re-introduces braking while creeping up to a stopped queue (parked in `_soft_crawl_rear_end_should_not_engage_without_slam`). Partial release does not reconcile them either, the two requirements have no overlap: responsiveness needs ≥0.75 of the reserve gone, the over-read case needs ≤0.5. That conflict is set by the engage bar, not by this knob |
| `aeb_engage_frac` | 0.85 | Fraction of `capability_decel` (not `effective_max`) at which a new engagement fires. Since the demand is tracked to the stop buffer, this is also how hard AEB brakes. Lukas asked for 90% of the truck on 2026-10-06, confirmed it in game the same evening, and set 0.85 for margin: on the stop sim it engages 2-3.5 m further out at 100-120 km/h and still ends on the buffer, and it leaves room for a truck braking up to ~10% weaker than believed. A 0.50 bar shipped in 1.1.2-preview.1 after a corpus reprice (the corpus cannot score timing) and was reverted the same day: it held about half the truck, 6 m/s² on a trailer rig that stops at 14. On today's stop sim (slam-fitted plant, bumper fixed, `stop_buffer` 0.5) every rig at 40-120 km/h ends 0.50 m or more short at p90 build-up; the older numbers below predate both. Re-swept on the stop simulator after the base change: 0.85 is the knee. 0.90 leaves 0.07 m of residual at p90 brake lag and 0.95 collides, while 0.80 costs 2.2 m of trigger distance to buy 0.3 m of margin. The corpus sensitivity table under "Geometry-graded engage fraction" was priced against `effective_max` and no longer applies to these numbers |
| `ego_decel_frac` | 0.9 | Tracking headroom on the **command** only. It is not part of the entry bar: `engage_threshold` and `aeb_warn_near_full_frac` run off `capability_decel` (`capacity − downhill`), while the target cap, `aeb_disarm_frac` and `aeb_warn_frac` run off `effective_max`. Folding it into entry made the real bar 0.765 of capacity and pushed the 100 km/h engage point 3.7 m further out on a 13.89 m/s² double |
| `warn_ttb` | 1.3 s | WARN threshold |
| `brake_ttb` | 0.2 s | BRAKE threshold |
| `stop_buffer` | 0.5 m | Gap the clearance demand plans to stop short by. Was 0.7 while ego's body was the reference rig's for every truck; lowered once the SDK wheels size the body (`core/radar/README.md` §17, confirmed in game 2026-10-04). Closed-loop stop sim (`tests/aeb/test_stop_distance_envelope.py`, 4 rigs x 5 speeds), shipped release, median / worst residual: median lag +0.07 / -0.04 -> -0.01 / -0.03 m, p90 lag -0.01 / -0.03 -> -0.02 / -0.06 m. The released reserve already spends most of the buffer, so the sim moves far less than 0.2 m |
| `ego_half_width` | 1.265 m | Ego arc corridor half-width (flush trailer standoff 2026-08-11). Fitted on `vehicle.volvo.fh_2024` 6x4; live sessions now size every truck from its SDK wheels with this as the reference, and use it only when the layout is unreadable (`core/radar/README.md` §17) |
| `ego_half_length` | 3.333 m | Ego capsule half-length (flush trailer standoff 2026-08-11; body extents via `capsule_extents`; collision segments are cap-aligned: extents minus half_width, see `core/radar/README.md` §8). Same reference role as `ego_half_width` |
| `corridor_margin` | 0.5 m | Corridor padding for crossing-path sample uncertainty |
| `cross_zone_base` | 1.0 m | Along-track pad in front of and behind a 90 deg target (`|sin(heading diff)|`). Collision and the stop demand both see it. 0 disables |
| `cross_zone_speed` | 0.3 s | Extra along-track length (`this * speed`) on top of `cross_zone_base` |
| `cross_zone_radial` | 0.7 m | Halo around the body at 90 deg, so a required stop clears more than the skin |
| `stop_buffer_response_s` | 0.25 s | Brake build-up **lag** for a solo tractor. The clearance model applies it as a time (ego rolls at `v0` for this long before decel starts), which is why the co-directional limit still comes out as `dv^2 / (2 * (gap - dv * this))`. Since the entry bar moved off `ego_decel_frac` this is the only entry margin, and once engaged it is released over `aeb_reserve_release_s`, so an oversized pad reads as "fires early, then lets go". Sized on AEB's own response: AEB always slams, and 17 full-pedal slams (AEB and driver, solo and trailer, 2026-10-04) build up in 0.20 s median / 0.24 s p90. The old 0.30 came from 61 mostly gentle episodes (0.25 / 0.37 s). At 0.25 every stop-distance case stops clear on the slam plant; 0.20 reaches the bumper at slam p90. A larger pad trips crawl engage in a queue (`_soft_crawl_rear_end_should_not_engage_without_slam`) |
| `stop_buffer_response_trailer_s` | 0.30 s | Same term with a trailer attached, and the term that dominates trigger distance at speed. Trailer slams build up no slower than solo ones, so the stop-distance envelope would take 0.25, but in the TMP convoy sim (`tools/acc_platoon`, lead stops dead from 80 km/h, three seeds) 0.25 put two followers on the truck ahead: there the pad also pays for seeing a braking lead through netcode. 0.30 is contact-free with 0.29-0.63 m left, 0.35 leaves 0.56-0.79 m, the old 0.40 0.91-1.05 m (all on the slam plant). Was 0.50, then 0.40 against the gentle-braking fit |
| `aeb_target_rate_engaged_ms3` | 30 m/s³ | Target slew while engaged. The engagement edge itself is exempt and steps straight to the requirement: ramping from zero used to cost the entire brake build-up window |
| `aeb_engage_frac_certain` | 0.85 | **In-game trial from 2026-08-11**, was 0.70. Now equal to `aeb_engage_frac`, so the geometry grading is flat and aligned in-lane traffic no longer skips the uncertainty hedge (`tests/aeb/test_engage_sensitivity.py::test_graded_bar_brakes_earlier_on_an_in_lane_obstacle` fails by design while this holds). Note this knob does not soften braking: required decel is recomputed as the gap closes, so engaging later means engaging at a *higher* demand. On clip `ac6b48b4` the engage-tick command rises 6.26 to 7.61 m/s², and headroom below `effective_max` for the loop to recover a build-up shortfall drops from 30% to 15% |
| `avoidability_gate_enabled` | True | Lets the four guess stages (`OutOfLaneParallelFilter`, `SweepPassFilter`, both corner-entry stages) drop a parked body only until its clearance demand reaches the braking deadline (README, Avoidability gate). False restores the bare stages; that is how the corpus is A/B'd. No threshold to tune: the deadline is the engage test at fraction 1 |
| `clearance_required_enabled` | True | Clearance demand model. False restores the relative-frame `_required_decel_two_frame` / `_codir_required_cap` path; that is how the corpus is A/B'd. It reverts the **formula only**, not the `max`-over-targets aggregation, which is unconditional: OFF therefore differs from the pre-change tree on exactly the multi-target clips (2 on the store as of 2026-08-23, `e66a2827` and `e9aaf92b`, both `false_negative -> true_positive`) |
| `clearance_horizon_s` | 5.0 s | How far the occupancy profile is sampled. Independent of `arc_horizon_max`: collision detection keeps `dynamic_horizon` so no filter verdict moves. Past the window a still-occupying target is linearly extrapolated to the closed-form peak, so this is not a demand cliff |
| `clearance_samples` | 24 | Dense grid over `[0, dynamic_horizon]`, where the hit actually lives. With the golden-section refinement the effective resolution is ~10 ms, which is why the co-directional equivalence is exact rather than approximate |
| `clearance_far_samples` | 8 | Sparse tail out to `clearance_horizon_s`. The cheap lever if a tick ever crowds the loop budget: 0 costs only the demand on distant slow closers, all of it under ~1.2 m/s2 |
| `clearance_refine_steps` | 4 | Golden-section steps inside the argmax bracket. 0 leaves grid error on a lead whose demand peaks at a short `t*` |
| `clearance_clear_margin_s` | 0.30 s | Hold the last conflict position this long past the frame a crosser vacates it, so ego does not arrive on its tail. Only applies when occupancy ends inside the window, so a co-directional lead never sees it (pinned by `test_the_clear_margin_never_fires_for_a_co_directional_lead`) |
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
| `oncoming_shared_bend_ratio` | 0.5 | Turn-into exits when target `\|κ\| ≥ ego \|κ\| ×` this: both on one bend, so `\|lat\|` sweeping the nose is a pass, not a turn-into. Magnitude only, no sign test |
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
| `aeb_warn_frac` | 0.60 | Fraction of `effective_max` at which `AEB_warn` rises via demand. Raised from 0.50 with the persistence windows to cut highway oncoming beeps (`63538d5`), then silently reset to 0.50 by `f39d262`, which drivers reported as a beep before they would start braking for stopped traffic. Restored 2026-10-02. At 90 km/h toward a stopped car on a 17 t rig (capacity 11.8, trailer lag 0.4 s) warn opens at ~60 m / 2.4 s TTC against ~69 m / 2.8 s at 0.50; engagement is ~40 m / 1.6 s. Labelled corpus (1312 clips): `false_warn` 42 -> 32, every brake verdict unchanged. Cost: warn lead of 0.5 s or more before an AEB brake on 65 of 675 braked clips, against 98 at 0.50. 0.70 measured `false_warn` 23 but only 30 clips keep a 0.5 s lead, so it was not taken |
| `aeb_warn_near_full_frac` | 0.85 | Demand fraction of `capability_decel` above which the warn cue survives the user-braking suppression. Equal to `aeb_engage_frac` (pinned by `test_near_full_warn_bar_matches_the_engage_bar`), so any engagement warns even while the driver brakes; raising it restores a quiet band but silences the cue on under-braking drivers. `f39d262` raised the engage bar to 0.90 and left this at 0.85, which beeped at a braking driver for demand between the two with no AEB brake behind it; realigned 2026-10-02 (corpus: one `false_warn` to `true_negative`, no other verdict moved) |
| `user_brake_latch` | 0.12 | Top of the FF-assist ramp: at or above this the sub-engagement assist applies at full weight |
| `ff_assist_ramp_lo` | 0.03 | Bottom of the FF-assist ramp; below it the assist contributes nothing. Matches `_USER_BRAKE_LATCH_THRESHOLD`, the physical-pedal warn deadzone. OPD and mapper warn suppression use any value above zero, not this floor |
| `aeb_confirm_occupancy` | 0.6 | Min qualified fraction over the trailing confirm window for the three `OccupancyConfirm` streaks (risk / engage / warn) to fire |
| `aeb_confirm_max_gap_frames` | 2 | Max consecutive unqualified frames tolerated before a confirm streak drops; absorbs isolated collision-grid / TMP-jitter dropouts |
| `aeb_certain_fwd_dot` | 0.90 | `|fwd_dot|` above which an in-lane colliding target is "certain" and skips the confirm wait |
| `aeb_warn_confirm_oncoming_s` | 2.00 s | Warn occupancy while *every* colliding target is head-on / near-head-on (opposite-carriageway phantom class) |
| `aeb_warn_clear_class` | True | Clear threat (README, Warn classes): ego arc and measured CBDR line both within `aeb_warn_clear_band_m` of ego's path for `aeb_warn_clear_hold_s`, driving ego's way or stopped. Gets the look-ahead warn and is never oncoming. False restores the old classes for A/B |
| `aeb_warn_clear_band_m` | 1.0 m | The test-track trailer sat at 0.3-0.9 m by both measures when straight ahead; queues across a bend at 1.2-1.9 m. Trailers parked 1-2 m off the path get a shorter lead. Against the 0.85 build both warn classes take the labelled corpus from 31 to 25 false warns (crossers -8, clear class +2, no brake verdict moved); a 1.5 m band nets 30, the lane band (1.95 m) 32 |
| `aeb_warn_clear_hold_s` | 0.50 s | How long the body must stay in the band first; the timer runs before the collision horizon reaches it. 0.3 s let one-tick bend flickers through, which the 0.3 s state hold stretches into a beep |
| `aeb_warn_lead_s` | 1.1 s | A clear threat warns when the clearance demand with this much extra build-up lag reaches the engage bar: AEB would have to brake this much later if nothing changed. 1.1 leaves a full second after the two-tick confirm. 0 disables |
| `aeb_warn_crossers_with_brake` | True | When every colliding target is a moving crosser (`abs(fwd_dot) < aeb_warn_crosser_dot`, 0.5), the cue starts with the brake. Their pre-brake warns were 9 of the corpus false warns, and their lead before a real brake was about zero |
| `aeb_warn_confirm_wide_lat_s` | 0.60 s | Warn occupancy while *every* colliding target projects past `aeb_warn_wide_lat_m` off the ego arc |
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
