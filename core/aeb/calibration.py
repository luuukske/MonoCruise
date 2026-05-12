"""AEB calibration — single source of truth for all tunable constants."""

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

    # Geometry / corridor
    ego_half_width: float = 1.15
    ego_half_length: float = 3.0
    corridor_margin: float = 0.5
    stop_buffer: float = 1.6
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
    rear_dot: float = -0.5

    # Curvature / dynamics
    turning_diverge_kappa: float = 0.007
    shared_turn_max_kappa: float = 0.05
    turn_complete_curvature_scale: float = 3.0
    evasion_g: float = 0.08 * 9.81
    evasion_g_oncoming: float = 0.13 * 9.81
    evasion_max_dkappa: float = 0.008
    yaw_rate_steer_gain: float = 12.0
    overtake_speed_margin: float = 2.0
    # AEB-local target-vehicle path prediction — two sources blended every
    # frame: a sliced position fit (last `aeb_pos_history_len` samples of the
    # vehicle's own _position_history, shorter than the full 25-sample fit)
    # and the per-frame yaw rate from `angular_velocity`.  Combined as
    # `yaw_blend * yaw_kappa + (1 - yaw_blend) * pos_kappa`; yaw weighted
    # higher for responsiveness, position included for smoothing.
    aeb_pos_history_len: int = 6
    aeb_yaw_blend: float = 0.4

    # Co-directional diverge
    co_dir_diverge_lookahead_s: float = 0.25
    co_same_turn_lookahead_scale: float = 0.5

    # Sweep-pass / corner-entry stationary
    sweep_pass_max_target_speed: float = 1.0
    corner_entry_min_distance: float = 1.0
    # Mode B (in-lane geometric consistency) — MP queue at corner entry
    corner_entry_min_road_bend: float = 0.10        # rad (~5.7°)
    corner_entry_min_lateral: float = 0.4           # m
    corner_entry_lateral_tol: float = 1.5           # m

    # TMP rel-speed filter
    tmp_filter_split_kmh: float = 40.0
    tmp_filter_rel_above_kmh: float = 15.0
    tmp_filter_rel_below_kmh: float = 40.0
    user_brake_latch: float = 0.12

    # Cross-zone (ghost-arc) padding — legacy, kept for behaviour parity
    cross_zone_base: float = 2.0
    cross_zone_speed: float = 0.3

    # Oncoming evasion kappa scaling
    opposite_lane_kappa_scale: float = 2.0

    # Fix A — near-head-on ghost-arc reduction
    near_head_on_cross_scale: float = 0.3
    near_head_on_lateral_min: float = 3.0

    # Continuous-decel AEB (rate-limit, deadband, slope-aware engagement)
    aeb_target_deadband_ms2: float = 0.4
    aeb_target_refresh_min_s: float = 0.20
    aeb_target_rate_ms3: float = 8.0
    aeb_engage_frac: float = 0.7
    aeb_disarm_frac: float = 0.45
    aeb_warn_frac: float = 0.5
    brake_actuator_lag_s: float = 0.18


DEFAULT = AEBCalibration()
