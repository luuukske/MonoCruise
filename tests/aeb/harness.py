"""Scenario builder and frame evaluator for AEB unit tests. No radar thread, no shared memory.
Builds synthetic Vehicle instances with seeded position history and runs the filter pipeline +
collision evaluator directly."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from core.radar.elevation import ElevationGate, build_surface
from core.radar.traffic import (
    Vehicle, Position, Quaternion, Size, Trailer,
    ArcPath, build_arc, arc_arc_collision, _accel_to_arc_params,
    capsule_extents,
)
from core.aeb.calibration import AEBCalibration, DEFAULT as CAL_DEFAULT
from core.aeb.filters import (
    FilterContext, FilterResult,
    _build_vehicle_collision_data, _world_to_ego_forward,
    _apply_cross_zone, _earliest_hit, build_pipeline,
)
from core.aeb.lane_frame import Lane, project_to_ego_arc, classify
from core.aeb.thread import AEBState, _INF


_DT = 1.0 / 30.0  # 30 Hz frame interval
_HISTORY_SEED_FRAMES = 15

# MP-style position-jitter sigma (metres). Real filtered TMP data drifts a
# few centimetres frame-to-frame; tests opt in by passing `noise_seed=i`.
_NOISE_SIGMA = 0.05


def _noise_offset(vid: int, frame_idx: int) -> tuple[float, float]:
    """Deterministic small (dx, dz) Gaussian jitter, seeded by (vid, frame)."""
    rng = random.Random((vid * 1_000_003) ^ (frame_idx * 2_654_435_761))
    return rng.gauss(0.0, _NOISE_SIGMA), rng.gauss(0.0, _NOISE_SIGMA)


@dataclass
class EgoState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_norm: float = 0.5  # 0.5 = North (forward = -Z direction)
    speed: float = 0.0  # m/s
    steer: float = 0.0  # degrees (ETS2 steer)
    pitch_deg: float = 0.0


@dataclass
class Frame:
    ego: EgoState
    vehicles: list[Vehicle]
    t: float = 0.0


@dataclass
class EvalResult:
    aeb_state: AEBState
    time_to_brake: float
    time_to_collision: float
    suppression_reasons: dict[int, list[FilterResult]]
    colliding_ids: set[int]
    suppressed_ids: set[int]
    evasion_filtered_ids: set[int]
    oncoming_evasion_filtered_ids: set[int]


def _make_quaternion_from_yaw_deg(yaw_deg: float) -> Quaternion:
    """Build a Quaternion from yaw in degrees (ETS2 convention, x/y swap applied)."""
    yaw_rad = math.radians(yaw_deg)
    w = math.cos(yaw_rad / 2.0)
    # In ETS2 convention the yaw axis is Z
    z = math.sin(yaw_rad / 2.0)
    # Quaternion x/y swap makes pure-yaw fragile; callers set Vehicle._smooth_yaw instead.
    return Quaternion(1.0, 0.0, 0.0, 0.0)


def make_vehicle(
    vid: int,
    x: float,
    z: float,
    yaw_deg: float,
    speed: float,
    curvature: float = 0.0,
    width: float = 2.5,
    length: float = 10.0,
    is_tmp: bool = False,
    acceleration: float = 0.0,
    y: float = 0.0,
    noise_seed: int | None = None,
) -> Vehicle:
    """Synthetic Vehicle for AEB filter tests (seeded history, optional TMP noise)."""
    if noise_seed is not None:
        nx, nz = _noise_offset(vid, noise_seed)
        x = x + nx
        z = z + nz
    pos = Position(x, y, z)
    rot = _make_quaternion_from_yaw_deg(yaw_deg)
    sz = Size(width, 1.5, length)

    v = Vehicle(
        position=pos,
        rotation=rot,
        size=sz,
        speed=speed,
        acceleration=acceleration,
        trailer_count=0,
        trailers=[],
        id=vid,
        is_tmp=is_tmp,
        is_trailer=False,
    )

    yaw_rad = math.radians(yaw_deg)
    v._smooth_yaw = yaw_rad
    v._smooth_speed = speed
    v._smooth_accel = acceleration
    v._raw_speed = speed
    v.speed = speed
    v.acceleration = acceleration
    v.angular_velocity = math.degrees(curvature * speed) if speed > 0.5 else 0.0

    # Seed position history: backdated positions at 30 Hz, moving at `speed` with `curvature`.
    fwd_x = -math.sin(yaw_rad)
    fwd_z = -math.cos(yaw_rad)
    t_now = time.monotonic()
    history: list[tuple[float, float, float]] = []

    for i in range(_HISTORY_SEED_FRAMES, 0, -1):
        dt_back = i * _DT
        t_hist = t_now - dt_back
        if curvature == 0.0 or speed < 0.1:
            hx = x - speed * dt_back * fwd_x
            hz = z - speed * dt_back * fwd_z
        else:
            radius = 1.0 / abs(curvature)
            sign = 1.0 if curvature > 0 else -1.0
            cx = x + sign * radius * fwd_z
            cz = z + sign * radius * (-fwd_x)
            angle0 = math.atan2(z - cz, x - cx)
            arc_back = speed * dt_back
            sweep = -sign * arc_back / radius
            angle = angle0 - sweep  # go backward
            hx = cx + radius * math.cos(angle)
            hz = cz + radius * math.sin(angle)
        history.append((t_hist, hx, hz))

    history.append((t_now, x, z))
    v._position_history = history
    v._curvature_cache_valid = False

    return v


def evaluate_frame(
    frame: Frame,
    calibration: AEBCalibration = CAL_DEFAULT,
) -> EvalResult:
    """Run AEB filters on one frame (no radar thread). Returns EvalResult."""
    ego = frame.ego
    ego_yaw_rad = ego.yaw_norm * 2.0 * math.pi
    ego_speed = ego.speed
    cal = calibration

    if ego_speed > 0.5:
        yaw_rate_rad_s = math.radians(ego.steer * ego_speed * cal.yaw_rate_steer_gain)
        ego_curvature = yaw_rate_rad_s / ego_speed
    else:
        ego_curvature = 0.0

    ego_hw = cal.ego_half_width
    ego_half_l = cal.ego_half_length
    effective_decel = cal.ego_decel_frac * cal.full_brake_decel

    t_stop = ego_speed / effective_decel if effective_decel > 0 else 0.0
    dynamic_horizon = min(max(cal.arc_horizon_min, t_stop * 2.0), cal.arc_horizon_max)

    ego_yaw_rad_val = ego_yaw_rad
    fwd_x = -math.sin(ego_yaw_rad_val)
    fwd_z = -math.cos(ego_yaw_rad_val)
    body_offset = (cal.arc_start_pctg - 0.5) * (2.0 * ego_half_l)
    ego_front_x = ego.x + body_offset * fwd_x
    ego_front_z = ego.z + body_offset * fwd_z
    ego_cap_fwd, ego_cap_back = capsule_extents(ego_half_l, ego_half_l, body_offset)

    ego_arc = build_arc(
        ego_front_x, ego_front_z, ego_yaw_rad_val, ego_speed,
        ego_curvature, ego_hw, dynamic_horizon,
        fwd_len=ego_cap_fwd, back_len=ego_cap_back,
        parallel_margin_scale=cal.capsule_parallel_margin_scale,
    )
    ego_braked_arc = build_arc(
        ego_front_x, ego_front_z, ego_yaw_rad_val, ego_speed,
        ego_curvature, ego_hw, dynamic_horizon, decel=effective_decel,
        fwd_len=ego_cap_fwd, back_len=ego_cap_back,
        parallel_margin_scale=cal.capsule_parallel_margin_scale,
    )

    ego_evasion_left = ego_evasion_right = None
    if ego_speed > 1.0:
        delta_kappa = min(
            cal.evasion_g / (ego_speed * ego_speed), cal.evasion_max_dkappa,
        )
        left_kappa = ego_curvature + delta_kappa
        if ego_curvature < 0 and left_kappa < 0:
            left_kappa /= 1.5
        right_kappa = ego_curvature - delta_kappa
        if ego_curvature > 0 and right_kappa > 0:
            right_kappa /= 1.5
        ego_evasion_left = build_arc(
            ego_front_x, ego_front_z, ego_yaw_rad_val, ego_speed,
            left_kappa, ego_hw, dynamic_horizon,
            fwd_len=ego_cap_fwd, back_len=ego_cap_back,
        )
        ego_evasion_right = build_arc(
            ego_front_x, ego_front_z, ego_yaw_rad_val, ego_speed,
            right_kappa, ego_hw, dynamic_horizon,
            fwd_len=ego_cap_fwd, back_len=ego_cap_back,
        )

    ego_fwd_x = ego_arc.fwd_x
    ego_fwd_z = ego_arc.fwd_z
    ego_pitch_rad = math.radians(ego.pitch_deg)
    max_range_sq = cal.max_range ** 2

    # Single-frame surface: no ego history, so grade comes from pitch alone.
    _surface = build_surface(ego.y, ego_pitch_rad, None)
    off_surface_ids = ElevationGate().step(
        list(frame.vehicles), _surface, ego.x, ego.z, ego_yaw_rad_val,
    )

    pipeline = build_pipeline(cal)

    colliding_ids: set[int] = set()
    suppressed_ids: set[int] = set()
    evasion_filtered_ids: set[int] = set()
    oncoming_evasion_filtered_ids: set[int] = set()
    suppression_reasons: dict[int, list[FilterResult]] = {}
    best_ttb = _INF
    best_ttc = _INF

    for v in frame.vehicles:
        dx = v.position.x - ego.x
        dz = v.position.z - ego.z
        dist_sq = dx * dx + dz * dz
        dist = math.sqrt(dist_sq)

        # Precompute collision arcs for vehicles in range
        if dist_sq <= max_range_sq:
            (all_target_arcs, cross_padding, precomputed_cross_arcs,
             v_yaw_rad, abs_v_speed, veh_fwd_x, veh_fwd_z, v_curvature,
             ) = _build_vehicle_collision_data(
                v, dynamic_horizon, ego_yaw_rad_val, ego_fwd_x, ego_fwd_z, cal,
            )
        else:
            all_target_arcs = []
            cross_padding = 0.0
            precomputed_cross_arcs = None
            _, v_yaw_deg, _ = v.rotation.euler()
            v_yaw_rad = math.radians(v_yaw_deg)
            abs_v_speed = abs(v.speed)
            veh_fwd_x = -math.sin(v_yaw_rad)
            veh_fwd_z = -math.cos(v_yaw_rad)
            v_curvature = 0.0

        _, v_yaw_deg_c, _ = v.rotation.euler()

        ctx = FilterContext(
            v=v,
            ego_arc=ego_arc,
            ego_braked_arc=ego_braked_arc,
            ego_evasion_left=ego_evasion_left,
            ego_evasion_right=ego_evasion_right,
            ego_x=ego.x, ego_y=ego.y, ego_z=ego.z,
            ego_yaw_rad=ego_yaw_rad_val,
            ego_speed=ego_speed,
            ego_pitch_rad=ego_pitch_rad,
            ego_curvature=ego_curvature,
            ego_fwd_x=ego_fwd_x,
            ego_fwd_z=ego_fwd_z,
            ego_hw=ego_hw,
            dynamic_horizon=dynamic_horizon,
            tmp_traffic_session=False,
            ref_kmh_for_filter=ego_speed * 3.6,
            cal=cal,
            dx=dx, dz=dz,
            dist_sq=dist_sq, dist=dist,
            v_yaw_rad=v_yaw_rad,
            abs_v_speed=abs_v_speed,
            veh_fwd_x=veh_fwd_x,
            veh_fwd_z=veh_fwd_z,
            v_curvature=v_curvature,
            all_target_arcs=all_target_arcs,
            precomputed_cross_arcs=precomputed_cross_arcs,
            cross_padding=cross_padding,
            off_surface_ids=off_surface_ids,
        )

        suppression_reasons[v.id] = []
        for stage in pipeline:
            res = stage.apply(ctx)
            if res.suppressed:
                suppression_reasons[v.id].append(res)
                reason = res.reason or ""
                if reason in ("OppositeLaneFilter", "EgoEvasionFilter"):
                    if ctx.head_on:
                        oncoming_evasion_filtered_ids.add(v.id)
                    else:
                        evasion_filtered_ids.add(v.id)
                elif reason == "CornerEntryStationaryFilter":
                    oncoming_evasion_filtered_ids.add(v.id)
                else:
                    suppressed_ids.add(v.id)
                break
        else:
            if not all_target_arcs:
                continue
            fwd_dot = ctx.fwd_dot
            head_on = ctx.head_on
            near_head_on = ctx.near_head_on
            lateral_gap = ctx.lateral_gap
            own_lane_for_fix_a = ctx.lane in (Lane.OPPOSITE_OR_OUTER, Lane.OFF_ROAD)
            fix_a_active = ctx.near_head_on and own_lane_for_fix_a
            effective_cross_padding = (
                cross_padding * cal.near_head_on_cross_scale if fix_a_active else cross_padding
            )

            for arc_idx, base_target_arc in enumerate(all_target_arcs):
                if fix_a_active or precomputed_cross_arcs is None:
                    cross_arcs = _apply_cross_zone(base_target_arc, effective_cross_padding)
                else:
                    cross_arcs = precomputed_cross_arcs[arc_idx]

                unbraked_hit = _earliest_hit(
                    ego_arc, cross_arcs, cal.corridor_margin, cal.collision_samples,
                    lateral_gap,
                )
                if unbraked_hit is None:
                    continue

                colliding_ids.add(v.id)
                unbraked_ttc = unbraked_hit[0]
                if unbraked_ttc < best_ttc:
                    best_ttc = unbraked_ttc

                braked_hit = _earliest_hit(
                    ego_braked_arc, cross_arcs,
                    cal.corridor_margin,
                    cal.collision_samples,
                    lateral_gap,
                )
                v_ego_x = ego_speed * ego_fwd_x
                v_ego_z = ego_speed * ego_fwd_z
                v_t_x = v.speed * ctx.veh_fwd_x
                v_t_z = v.speed * ctx.veh_fwd_z
                closing_unbraked = math.hypot(v_ego_x - v_t_x, v_ego_z - v_t_z)
                v_target_along_ego = v_t_x * ego_fwd_x + v_t_z * ego_fwd_z
                braking_worsens = False
                if braked_hit is not None:
                    if v_target_along_ego > ego_speed:
                        braking_worsens = True
                    else:
                        t_braked = braked_hit[0]
                        v_ego_braked = max(0.0, ego_speed - effective_decel * t_braked)
                        v_egb_x = v_ego_braked * ego_fwd_x
                        v_egb_z = v_ego_braked * ego_fwd_z
                        closing_braked = math.hypot(v_egb_x - v_t_x, v_egb_z - v_t_z)
                        if closing_braked > closing_unbraked + cal.brake_worsens_hysteresis_ms:
                            braking_worsens = True
                if braking_worsens:
                    continue
                ttb = unbraked_ttc
                if ttb < best_ttb:
                    best_ttb = ttb

    # Determine state
    state = AEBState.STANDBY
    if best_ttb < _INF:
        if best_ttb < cal.warn_ttb:
            state = AEBState.WARN
        if best_ttb < cal.brake_ttb:
            state = AEBState.BRAKE

    return EvalResult(
        aeb_state=state,
        time_to_brake=best_ttb,
        time_to_collision=best_ttc,
        suppression_reasons=suppression_reasons,
        colliding_ids=colliding_ids,
        suppressed_ids=suppressed_ids,
        evasion_filtered_ids=evasion_filtered_ids,
        oncoming_evasion_filtered_ids=oncoming_evasion_filtered_ids,
    )


def build_scenario_frames(
    ego_states: list[EgoState],
    vehicle_builders: list[Any],
) -> list[Frame]:
    """Build frames from ego sequence and per-index vehicle builder callables."""
    frames = []
    for i, ego in enumerate(ego_states):
        t = i * _DT
        vehicles = []
        for builder in vehicle_builders:
            v = builder(i, t)
            if v is not None:
                vehicles.append(v)
        frames.append(Frame(ego=ego, vehicles=vehicles, t=t))
    return frames

