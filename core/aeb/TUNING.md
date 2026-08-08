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
| `certain_engage_ttb` | 1.25 s | Certain-geom soft rear-end engage OR: `best_ttb_engage` under this qualifies (between the 0.50 s slam and warn_ttb); see README certain-TTB bridge |
| `ego_half_width` | 1.15 m | Ego arc corridor half-width |
| `ego_half_length` | 3.0 m | Ego capsule half-length (body extents via `capsule_extents`; collision segments are cap-aligned: extents minus half_width, see `core/radar/README.md` §8) |
| `corridor_margin` | 0.5 m | Corridor padding for crossing-path sample uncertainty |
| `stop_buffer_response_s` | 0.10 s | Response-lag distance in required-decel gap (`v_closing * this`). Last-point envelope, not corpus timing optimum; rejected cap/tiering in calibration comments |
| `disarm_hold_ttc_s` | 3.0 s | Geometry latch window while engaged: hold the event while any colliding target's unbraked ttc is inside it (anti-pumping; was `warn_ttb`) |
| `capsule_parallel_margin_scale` | 0.3 | Near-parallel capsule contacts use `margin * scale` blended by heading sine toward full margin at perpendicular; kills adjacent-lane side-graze FPs (ab524f87 / 29bf31b8). 1.0 disables |
| `lane_half_width` | 1.95 m | EGO lane boundary |
| `lane_separation` | 3.9 m | Road lane pitch |
| `head_on_dot` | -0.7 | `head_on` flag threshold |
| `co_directional_dot` | 0.7 | `co_directional` flag threshold |
| `evasion_g` | 0.08×9.81 | Ego evasion lateral accel |
| `oncoming_body_sep_miss_scale` | 0.25 | OppositeLane body-sep also needs measured miss ≥ clear_bar × this |
| `oncoming_closing_dmiss_rate_mps` | −1.5 m/s | Turn-into-path: miss closing this fast with ego turning |
| `oncoming_closing_lat_m` | 0.85 m | Turn-into-path: straight `|lat|` must collapse under this |
| `evasion_max_dkappa` | 0.008 /m | Max curvature offset for evasion arcs |
| `opposite_lane_kappa_scale` | 2.0 | Kappa multiplier when target in own lane |
| `turning_diverge_kappa` | 0.007 /m | Corner threshold for Fix-C/D conditions; also the straight/turning split in `TmpCrossTrafficFilter` |
| `tmp_cross_center_hit_dist` | 2.5 m | `TmpCrossTrafficFilter` straight-snapshot genuine-crosser threshold: centre closest-approach at/below this is a real T-bone (pass), above is a body-graze clear (suppress) |
| `co_same_turn_lookahead_scale` | 0.5 | Extended lookahead fraction of horizon |
| `diverge_dip_samples` | 8 | `_is_approaching` window samples for the in-lane pass-through dip check |
| `aeb_engage_confirm_s` | 0.06 s | Sustained-qualification wait for near-certain engagement entries (3rd tick at 30 Hz) |
| `aeb_engage_confirm_oblique_s` | 0.20 s | Sustained-qualification wait for oblique out-of-lane entries (extrapolation-fragile class) |
| `aeb_warn_confirm_oblique_s` | 0.10 s | Warn persistence for oblique out-of-lane threats; keeps ≥ 0.1 s warn lead ahead of an oblique engagement |
| `aeb_warn_frac` | 0.5 | Fraction of `effective_max` at which `AEB_warn` rises |
| `aeb_warn_near_full_frac` | 0.85 | Demand fraction above which the warn cue survives the user-braking suppression. Currently equal to `aeb_engage_frac`, so any engagement warns even while the driver brakes; raising it restores a quiet band but silences the cue on under-braking drivers |
| `user_brake_latch` | 0.12 | Top of the FF-assist ramp: at or above this the sub-engagement assist applies at full weight |
| `ff_assist_ramp_lo` | 0.03 | Bottom of the FF-assist ramp; below it the assist contributes nothing. Matches `_USER_BRAKE_LATCH_THRESHOLD` so one notion of "the driver is braking" governs both the warn cue and the assist |
| `aeb_confirm_occupancy` | 0.6 | Min qualified fraction over the trailing confirm window for the three `OccupancyConfirm` streaks (risk / engage / warn) to fire |
| `aeb_confirm_max_gap_frames` | 2 | Max consecutive unqualified frames tolerated before a confirm streak drops; absorbs isolated collision-grid / TMP-jitter dropouts |
| `aeb_certain_fwd_dot` | 0.90 | `|fwd_dot|` above which an in-lane colliding target is "certain" and skips the confirm wait |
| `corner_entry_min_road_bend` | 0.10 rad | Min ego↔tangent angle for Mode-B suppression |
| `corner_entry_min_lateral` | 0.4 m | Min |lat_signed| to claim "off ego axis" (Mode B) |
| `corner_entry_lateral_tol` | 1.5 m | Chord-offset tolerance for arc-consistency check (Mode B) |

---
