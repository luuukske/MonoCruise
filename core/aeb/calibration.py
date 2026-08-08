"""AEB calibration: single source of truth for all tunable constants."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AEBCalibration:
    # Brake / TTB
    full_brake_decel: float = 7.8
    ego_decel_frac: float = 0.9
    warn_ttb: float = 1.3
    brake_ttb: float = 0.2
    brake_release_ttb: float = 0.5
    risk_confirm_s: float = 0.05
    risk_confirm_oncoming_s: float = 0.10
    brake_response_window_s: float = 0.30
    brake_worsens_hysteresis_ms: float = 0.5

    # Geometry / corridor
    ego_half_width: float = 1.15
    ego_half_length: float = 3.0
    corridor_margin: float = 0.5
    # Near-parallel capsule contacts: margin * scale at parallel; see core/aeb/README.md.
    capsule_parallel_margin_scale: float = 0.3
    stop_buffer: float = 0.2
    # Response-lag gap term (v_closing * this); last-point envelope, not corpus timing optimum.
    stop_buffer_response_s: float = 0.10
    # Rejected 2026-07-19: response distance cap, threat-age tiering, engage 0.9 (README §7).
    elevation_margin: float = 5.0
    max_range: float = 200.0
    arc_start_pctg: float = 0.2
    collision_samples: int = 36

    # Arc horizon
    arc_horizon_min: float = 2.5
    arc_horizon_max: float = 3.0

    # Lane-frame thresholds
    lane_half_width: float = 1.95
    lane_separation: float = 3.9

    # Yaw / fwd-dot regimes
    head_on_dot: float = -0.7
    near_head_on_dot: float = -0.5
    co_directional_dot: float = 0.7

    # Curvature / dynamics
    turning_diverge_kappa: float = 0.007
    shared_turn_max_kappa: float = 0.05
    turn_complete_curvature_scale: float = 3.0
    evasion_g: float = 0.08 * 9.81
    evasion_g_oncoming: float = 0.13 * 9.81
    evasion_max_dkappa: float = 0.008
    yaw_rate_steer_gain: float = 12.0
    # Target path: pos slice + yaw rate blend; see core/aeb/README.md (target curvature).
    aeb_pos_history_len: int = 10
    aeb_yaw_blend: float = 0.7

    # One-Euro on blended kappa; knobs in README §7 and target-curvature section.
    aeb_kappa_one_euro_min_cutoff: float = 1.0
    aeb_kappa_one_euro_beta: float = 100.0
    aeb_kappa_one_euro_d_cutoff: float = 0.3
    aeb_kappa_one_euro_beta_turn_scale: float = 20.0

    # Co-directional diverge
    co_dir_diverge_lookahead_s: float = 0.25
    co_same_turn_lookahead_scale: float = 0.5
    # In-lane pass-through dip samples in _is_approaching; CoDirectionalDivergeFilter (README).
    diverge_dip_samples: int = 8

    # Sweep-pass / corner-entry stationary
    sweep_pass_max_target_speed: float = 1.0
    corner_entry_min_distance: float = 1.0
    # Mode B (in-lane geometric consistency): MP queue at corner entry
    corner_entry_min_road_bend: float = 0.10        # rad (~5.7°)
    corner_entry_min_lateral: float = 0.4           # m
    corner_entry_lateral_tol: float = 1.5           # m

    # TMP rel-speed filter
    tmp_filter_split_kmh: float = 50.0
    tmp_filter_rel_above_kmh: float = 5.0
    tmp_filter_rel_below_kmh: float = 50.0
    user_brake_latch: float = 0.12
    # FF assist ramps in from here to user_brake_latch (README pedal authority).
    ff_assist_ramp_lo: float = 0.03

    # TmpCrossTrafficFilter straight-snapshot center hit threshold (README TmpCrossTrafficFilter).
    tmp_cross_center_hit_dist: float = 2.5

    # Retired ghost-arc padding fields; kept for name compatibility only.
    cross_zone_base: float = 2.0
    cross_zone_speed: float = 0.3

    # OutOfLaneParallelFilter horizon lane scan count; see README OutOfLaneParallelFilter.
    out_of_lane_scan_samples: int = 10
    # Stationary angled adjacent: shallow graze (min d_abs) with far end out of lane.
    stationary_ool_graze_min_m: float = 0.90
    stationary_ool_graze_max_m: float = 1.50
    stationary_ool_span_scale: float = 1.5

    # Oncoming evasion kappa scaling
    opposite_lane_kappa_scale: float = 2.0
    # OppositeLaneFilter body-separation fast path: the measured miss must also
    # clear this multiple of the body bar. 0 trusts pose alone (README).
    oncoming_body_sep_miss_scale: float = 0.25
    # Turn-into-path: CBDR miss closing this fast (m/s) with ego turning skips
    # OppositeLane body-sep / Fix B (README turn-into-path). e0fd ~ -5.6 m/s.
    oncoming_closing_dmiss_rate_mps: float = -1.5
    # Straight-frame |lat| must collapse under this (m) with the miss-rate gate.
    # 0.85 recovers e0fd confirm span; 7c635440 id162 floors ~0.72 but stays TN
    # via remaining evasion (corpus sweep).
    oncoming_closing_lat_m: float = 0.85
    # Closing skip requires arc inflation vs |lat|; kills false closing on
    # adjacent head-on (6a35 engage d_abs/|lat|~9.3) while e0fd (~30) still skips.
    oncoming_closing_dabs_lat_ratio: float = 10.0
    # Soft pose clearance under clear_bar for OppositeLane body-sep (m).
    # 0.80 locks 887 engage (d_abs 1.58 vs clear ~2.32) and 280 (1.8 vs ~2.31).
    oncoming_body_sep_soft_m: float = 0.80
    # Fail-closed: do not suppress Opp/TmpCross when required a_lat exceeds this.
    # Only when straight |lat| >= clear_bar (wide pre-collapse); soft adjacent stays.
    max_evasion_lat_g: float = 0.35 * 9.81
    # TmpCross: pass when body already in ego lane band. Off: +FP without new FN win
    # once max_evasion recovers cca/e09.
    tmp_cross_in_corridor_pass: bool = False

    # Fix A: near-head-on ghost-arc reduction
    near_head_on_cross_scale: float = 0.3
    near_head_on_lateral_min: float = 3.0

    # Continuous-decel AEB (rate-limit, deadband, slope-aware engagement)
    aeb_target_deadband_ms2: float = 0.4
    aeb_target_refresh_min_s: float = 0.20
    aeb_target_rate_ms3: float = 8.0
    # aeb_engage_frac 0.85 from 2026-07-19 scan; 0.80 rejected (full corpus +8 FP).
    aeb_engage_frac: float = 0.85
    # Geometry-graded: aligned in-lane traffic skips the uncertainty hedge.
    # Pure sensitivity trade, no flat band; see README geometry-graded engage.
    aeb_engage_frac_certain: float = 0.70
    # Certain-geom soft rear-ends: warn via ttb < warn_ttb while v^2/2d stays
    # below the 0.70 bar and outside the 0.50 s slam (README certain-TTB bridge).
    certain_engage_ttb: float = 1.30
    aeb_disarm_frac: float = 0.45
    # Geometry latch while colliding unbraked ttc inside window (anti-pumping; README).
    disarm_hold_ttc_s: float = 3.0
    aeb_warn_frac: float = 0.5
    aeb_warn_near_full_frac: float = 0.85
    brake_actuator_lag_s: float = 0.10
    # New engagements only fire when |ego_speed| is above this threshold.
    aeb_min_engage_speed_kmh: float = 5.0
    # Tiered engage confirm windows; brake-TTB slam exempt (README continuous-decel).
    aeb_engage_confirm_s: float = 0.06
    aeb_engage_confirm_oblique_s: float = 0.40
    aeb_certain_fwd_dot: float = 0.90
    # Oblique warn must lead oblique engage by >= 0.1 s (README warn persistence).
    aeb_warn_confirm_oblique_s: float = 0.10

    # OccupancyConfirm lapse tolerance; shared by risk/engage/warn (README).
    aeb_confirm_occupancy: float = 0.6
    aeb_confirm_max_gap_frames: int = 2

    # CBDR LOS veto on engagement entry only; crash-confirmed ids exempt (README LOS veto).
    los_veto_enabled: bool = True
    los_veto_window_s: float = 0.6
    los_veto_min_samples: int = 12
    los_veto_min_range_m: float = 25.0
    los_veto_miss_dist_m: float = 6.0
    # Head-on branch: antiparallel pairs clear each other on measured lateral
    # gap alone, so the miss bar drops to body clearance (README LOS veto).
    los_veto_headon_miss_dist_m: float = 2.8
    los_veto_headon_min_range_m: float = 20.0
    # Straight-line CBDR is void for a manoeuvring target; kappa is only
    # trustworthy above this speed (kappa = yaw_rate / v blows up near zero).
    los_veto_headon_max_kappa: float = 0.05
    los_veto_headon_min_speed_ms: float = 5.0

    # Engagement-entry vetoes for hits that rest on unsupportable extrapolation
    # (README engagement-entry vetoes). Warn and FF assist are untouched.
    extrap_veto_enabled: bool = True
    turn_veto_min_kappa: float = 0.012
    turn_veto_min_ttc_s: float = 1.2
    codir_adjacent_veto_axial_ms: float = 2.0
    codir_adjacent_veto_miss_m: float = 2.0
    # Range past which an unseen bend moves a non-co-directional target by more
    # than a lane, so its lane stops being evidence (README lane confidence).
    lane_confidence_range_m: float = 30.0

    # Follow-threat: kinematic hold + lane/converge gate; README follow-threat section.
    follow_threat_window_s: float = 0.6
    follow_threat_min_span_s: float = 0.45
    follow_threat_min_samples: int = 8
    follow_threat_min_decel_ms2: float = 0.8
    follow_threat_min_closing_ms: float = 0.5
    follow_threat_min_lat_converge_ms: float = 0.3
    follow_threat_hold_s: float = 2.0
    follow_threat_max_range_m: float = 80.0

    # Latched-threat TMP bypass + headway hold; scope release grace (README latched-threat).
    latched_min_headway_s: float = 1.5
    latched_release_headway_s: float = 2.5
    latched_min_decel_frac: float = 0.7
    latched_scope_release_s: float = 0.5


DEFAULT = AEBCalibration()

