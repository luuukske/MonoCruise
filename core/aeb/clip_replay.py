"""Clip review: replay radar poses, live_aeb decisions, rebuilt arcs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.aeb.calibration import DEFAULT as _CAL
from core.aeb.clip_schema import Clip, ConsumedContext, LiveAEB
from core.aeb.filters import VehicleCurvatureBlender, _vehicle_curvature_blend
from core.aeb.thread import (
    AEBSnapshot, AEBState, _INF, _dampen_turning_curvature,
    _swap_trailer_kinematics,
)
from core.radar.reader import TrafficReader
from core.radar.traffic import ArcPath, Vehicle, build_arc

# Suppression stages the debug window colours as "evasion filtered" (cyan)
# rather than hard-suppressed (grey); mirrors the classification in thread.py.
_EVASION_STAGES = {
    "OppositeLaneFilter", "OppositeLaneFilterMirrored", "EgoEvasionFilter",
    "CornerEntryStationaryFilter", "CornerEntryStationaryFilterMirrored",
}


@dataclass
class ReviewFrame:
    """One scrubber step: a renderable snapshot plus the recorded decision."""

    t_rel: float                 # seconds from clip start
    t_mono: float
    snapshot: AEBSnapshot
    live_aeb: LiveAEB
    consumed: ConsumedContext
    raw_target_ms2: float = 0.0  # pre-slew demand, see raw_target_decel


def raw_target_decel(live: LiveAEB, cal: dict) -> float:
    """Pre-slew decel demand rebuilt from a recorded tick (README §review desmoothing).

    ``LiveAEB.target_decel_ms2`` is deadbanded and rate limited, so it lags the moment
    a threat became real by up to several ticks. This mirrors ``target_raw`` in
    ``AEBThread.loop`` before that limiter. The latched-hold floor is not recorded, so
    this under-reads on hold ticks; it is never wrong about onset timing.
    """
    if not live.engaged:
        return 0.0
    cap = live.effective_max_decel_ms2
    ttb_gate = cal.get("brake_ttb", 0.2) + cal.get("brake_response_window_s", 0.30)
    if live.time_to_brake < ttb_gate:
        return cap
    return max(0.0, min(live.required_decel_ms2, cap))


def _ego_curvature(steer: float, speed: float) -> float:
    if speed > 0.5:
        return math.radians(steer * speed * _CAL.yaw_rate_steer_gain) / speed
    return 0.0


def _veh_yaw(v: Vehicle) -> float:
    if getattr(v, "_smooth_yaw", None) is not None:
        return v._smooth_yaw
    return math.radians(v.rotation.euler()[1])


def _vehicle_dict(v: Vehicle) -> dict:
    yaw = _veh_yaw(v)
    trailers = []
    for tr in v.trailers:
        _, tr_yaw_deg, _ = tr.rotation.euler()
        trailers.append({
            "x": tr.position.x, "z": tr.position.z,
            "yaw": math.radians(tr_yaw_deg),
            "half_w": tr.size.width / 2.0,
            "length": tr.size.length,
            "is_tmp": tr.is_tmp,
            "speed_kmh": abs(v.speed) * 3.6,
        })
    return {
        "vid": v.id,
        "x": v.position.x, "z": v.position.z,
        "yaw": yaw,
        "half_w": v.size.width / 2.0,
        "length": v.size.length,
        "is_tmp": v.is_tmp,
        "is_trailer": getattr(v, "is_trailer", False),
        "kinematics_swapped": getattr(v, "_debug_kinematics_swapped", False),
        "speed_kmh": abs(v.speed) * 3.6,
        "trailers": trailers,
    }


def _arc_curvature(v: Vehicle, ego_fwd_x: float, ego_fwd_z: float,
                   horizon: float, blender: VehicleCurvatureBlender,
                   now: float) -> float:
    """Replay arc curvature; One-Euro stepped once per vehicle per tick."""
    abs_v_speed = abs(v.speed)
    v_curvature = _vehicle_curvature_blend(v, abs_v_speed, _CAL, blender, now)
    v_yaw = _veh_yaw(v)
    veh_fwd_x = -math.sin(v_yaw)
    veh_fwd_z = -math.cos(v_yaw)
    fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
    return _dampen_turning_curvature(
        v_curvature, fwd_dot,
        ego_fwd_x, ego_fwd_z, veh_fwd_x, veh_fwd_z,
        abs_v_speed, abs_v_speed * horizon, _CAL,
    )


def _vehicle_arc_list(v: Vehicle, arc_curvature: float,
                      horizon: float) -> list[ArcPath]:
    """Tractor arc plus one arc per trailer, as ``AEBSnapshot.vehicle_arcs`` holds."""
    arcs = [v.get_arc(
        horizon,
        arc_start_pctg=_CAL.arc_start_pctg,
        curvature_override=arc_curvature,
    )]
    is_reversing = v.speed < -1e-3
    effective_p = (1.0 - _CAL.arc_start_pctg) if is_reversing else _CAL.arc_start_pctg
    for tr in v.trailers:
        _, tr_yaw_deg, _ = tr.rotation.euler()
        tr_yaw = math.radians(tr_yaw_deg)
        tr_fwd_x = -math.sin(tr_yaw)
        tr_fwd_z = -math.cos(tr_yaw)
        body_offset = (effective_p - 0.5) * tr.size.length
        arcs.append(build_arc(
            tr.position.x + body_offset * tr_fwd_x,
            tr.position.z + body_offset * tr_fwd_z,
            tr_yaw, v.speed, arc_curvature,
            tr.size.width / 2.0, horizon,
        ))
    return arcs


def _build_snapshot(ego, vehicles: list[Vehicle], live: LiveAEB,
                    consumed: ConsumedContext,
                    blender: VehicleCurvatureBlender, now: float) -> AEBSnapshot:
    ego_x = ego.coordinateX
    ego_z = ego.coordinateZ
    ego_yaw = ego.rotationX * 2.0 * math.pi
    ego_speed = ego.speed
    ego_hw = _CAL.ego_half_width
    ego_hl = _CAL.ego_half_length

    capacity = max(consumed.max_brake_ms2, 1.0)
    t_stop = ego_speed / (_CAL.ego_decel_frac * capacity) if capacity > 0 else 0.0
    horizon = min(max(_CAL.arc_horizon_min, t_stop * 2.0), _CAL.arc_horizon_max)
    curv = _ego_curvature(ego.userSteer, ego_speed)

    fwd_x = -math.sin(ego_yaw)
    fwd_z = -math.cos(ego_yaw)
    body_offset = (_CAL.arc_start_pctg - 0.5) * (2.0 * ego_hl)
    ego_arc = build_arc(
        ego_x + body_offset * fwd_x, ego_z + body_offset * fwd_z,
        ego_yaw, ego_speed, curv, ego_hw, horizon,
    )

    vehicle_arcs = {
        v.id: _vehicle_arc_list(
            v, _arc_curvature(v, fwd_x, fwd_z, horizon, blender, now), horizon,
        )
        for v in vehicles
    }
    blender.prune({v.id for v in vehicles})

    colliding = {int(i) for i in live.colliding_ids}
    suppressed = {int(i) for i in live.suppressed_ids}
    worsens = {int(i) for i in live.braking_worsens_ids}
    evasion = {
        int(vid) for vid, stages in live.suppression_reasons.items()
        if any(s in _EVASION_STAGES for s in stages)
    }

    if live.aeb_brake:
        state = AEBState.BRAKE
    elif live.aeb_warn:
        state = AEBState.WARN
    else:
        state = AEBState.STANDBY

    hit_x = hit_z = 0.0
    ttc = live.time_to_collision
    threat = [v for v in vehicles if v.id in colliding]
    if threat:
        t = min(threat, key=lambda v: (v.position.x - ego_x) ** 2 + (v.position.z - ego_z) ** 2)
        hit_x, hit_z = t.position.x, t.position.z
    elif state >= AEBState.WARN:
        ttc = _INF   # no threat vehicle to mark, suppress the hit cross

    return AEBSnapshot(
        ego_x=ego_x, ego_z=ego_z, ego_yaw=ego_yaw,
        ego_speed=ego_speed, ego_half_w=ego_hw, ego_half_l=ego_hl,
        ego_arc=ego_arc, ego_braked_arc=None,
        ego_has_trailer=bool(ego.ego_has_trailer),
        vehicles=[_vehicle_dict(v) for v in vehicles],
        vehicle_arcs=vehicle_arcs,
        colliding_ids=colliding, suppressed_ids=suppressed,
        braking_worsens_ids=worsens,
        evasion_filtered_ids=evasion,
        oncoming_evasion_filtered_ids=set(),
        aeb_state=state,
        time_to_collision=ttc,
        time_to_brake=live.time_to_brake,
        hit_x=hit_x, hit_z=hit_z,
        suppression_reasons={int(k): v for k, v in live.suppression_reasons.items()},
        tmp_traffic_session=any(v.is_tmp for v in vehicles),
    )


def decode_radar_stream(clip: Clip):
    """TrafficReader-smoothed vehicles per radar frame; (veh_by_t, ego_by_t, frame_t)."""
    reader = TrafficReader()
    frames = sorted(clip.radar_frames, key=lambda f: f.t_mono)
    veh_by_t: dict[float, list[Vehicle]] = {}
    last_vehs: list[Vehicle] = []
    was_paused = False
    for f in frames:
        if f.traffic_buf is None or f.ego.paused:
            was_paused = True
            veh_by_t[f.t_mono] = list(last_vehs)
            continue
        if was_paused:
            was_paused = False
            reader.request_reanchor()
        res = reader.replay_frame(
            f.traffic_buf, f.parked_buf,
            f.ego.coordinateX, f.ego.coordinateY, f.ego.coordinateZ, f.ego.speed,
            f.t_wall,
        )
        last_vehs = list(res[0]) if res is not None else []
        veh_by_t[f.t_mono] = last_vehs
    ego_by_t = {f.t_mono: f.ego for f in frames}
    return veh_by_t, ego_by_t, sorted(veh_by_t)


def nearest_frame_t(frame_t: list[float], t: float) -> float | None:
    """Frame timestamp closest to ``t`` (exact match preferred)."""
    if not frame_t:
        return None
    if t in frame_t:
        return t
    return min(frame_t, key=lambda ft: abs(ft - t))


def replay_clip(clip: Clip) -> list[ReviewFrame]:
    """Decode + smooth the radar stream and build one ReviewFrame per AEB tick."""
    if not clip.aeb_ticks and not clip.radar_frames:
        return []

    veh_by_t, ego_by_t, frame_t = decode_radar_stream(clip)
    frames = sorted(clip.radar_frames, key=lambda f: f.t_mono)

    t_candidates = [f.t_mono for f in frames] + [tk.t_mono for tk in clip.aeb_ticks]
    t0 = min(t_candidates) if t_candidates else 0.0

    # One blender for the whole clip: its One-Euro state must carry tick to tick
    # the way it does across live AEB loops, or the drawn arcs are unsmoothed.
    blender = VehicleCurvatureBlender(_CAL)

    cal_rec = clip.metadata.calibration or {}

    out: list[ReviewFrame] = []
    for tk in sorted(clip.aeb_ticks, key=lambda x: x.t_mono):
        ft = nearest_frame_t(frame_t, tk.radar_t_mono)
        vehicles = veh_by_t.get(ft, []) if ft is not None else []
        ego = ego_by_t.get(ft) if ft is not None else None
        if ego is None and frames:
            ego = frames[0].ego
        if ego is None:
            continue
        vehicles_eff = _swap_trailer_kinematics(vehicles)
        snap = _build_snapshot(
            ego, vehicles_eff, tk.live_aeb, tk.consumed, blender, tk.t_mono,
        )
        out.append(ReviewFrame(
            tk.t_mono - t0, tk.t_mono, snap, tk.live_aeb, tk.consumed,
            raw_target_decel(tk.live_aeb, cal_rec),
        ))
    return out


def clip_duration(clip: Clip) -> float:
    ts = [f.t_mono for f in clip.radar_frames] + [t.t_mono for t in clip.aeb_ticks]
    return (max(ts) - min(ts)) if ts else 0.0
