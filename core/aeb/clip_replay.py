"""Reconstruct renderable AEB scenes from a captured clip.

Replays the raw radar byte streams through the real ``TrafficReader`` smoothing
to recover ``Vehicle`` poses per frame, then merges each AEB tick with the radar
snapshot it consumed (by ``radar_t_mono``) and builds an ``AEBSnapshot`` per tick
that the existing ``AEBDebugWindow`` renderer can draw.

The AEB *decision* (threat / suppressed ids, warn / brake state, TTC / TTB) comes
from the recorded ``live_aeb`` parity oracle, not a re-run of the pipeline: this
shows exactly what the program decided, which is what the tagger judges.
Predicted per-vehicle arcs are not stored in a clip, so vehicle corridors are
omitted; the ego corridor is rebuilt from ego geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.aeb.calibration import DEFAULT as _CAL
from core.aeb.clip_schema import Clip, ConsumedContext, LiveAEB
from core.aeb.thread import AEBSnapshot, AEBState, _INF
from core.radar.reader import TrafficReader
from core.radar.traffic import Vehicle, build_arc

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
        "kinematics_swapped": False,
        "speed_kmh": abs(v.speed) * 3.6,
        "trailers": trailers,
    }


def _build_snapshot(ego, vehicles: list[Vehicle], live: LiveAEB,
                    consumed: ConsumedContext) -> AEBSnapshot:
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
        vehicle_arcs={},
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


def replay_clip(clip: Clip) -> list[ReviewFrame]:
    """Decode + smooth the radar stream and build one ReviewFrame per AEB tick."""
    if not clip.aeb_ticks and not clip.radar_frames:
        return []

    reader = TrafficReader()
    frames = sorted(clip.radar_frames, key=lambda f: f.t_mono)
    veh_by_t: dict[float, list[Vehicle]] = {}
    for f in frames:
        res = reader.replay_frame(
            f.traffic_buf, f.parked_buf,
            f.ego.coordinateX, f.ego.coordinateY, f.ego.coordinateZ, f.ego.speed,
            f.t_wall,
        )
        veh_by_t[f.t_mono] = list(res[0]) if res is not None else []

    frame_t = sorted(veh_by_t)
    ego_by_t = {f.t_mono: f.ego for f in frames}

    def nearest_frame_t(t: float) -> float | None:
        if t in veh_by_t:
            return t
        if not frame_t:
            return None
        return min(frame_t, key=lambda ft: abs(ft - t))

    t_candidates = [f.t_mono for f in frames] + [tk.t_mono for tk in clip.aeb_ticks]
    t0 = min(t_candidates) if t_candidates else 0.0

    out: list[ReviewFrame] = []
    for tk in sorted(clip.aeb_ticks, key=lambda x: x.t_mono):
        ft = nearest_frame_t(tk.radar_t_mono)
        vehicles = veh_by_t.get(ft, []) if ft is not None else []
        ego = ego_by_t.get(ft) if ft is not None else None
        if ego is None and frames:
            ego = frames[0].ego
        if ego is None:
            continue
        snap = _build_snapshot(ego, vehicles, tk.live_aeb, tk.consumed)
        out.append(ReviewFrame(tk.t_mono - t0, tk.t_mono, snap, tk.live_aeb, tk.consumed))
    return out


def clip_duration(clip: Clip) -> float:
    ts = [f.t_mono for f in clip.radar_frames] + [t.t_mono for t in clip.aeb_ticks]
    return (max(ts) - min(ts)) if ts else 0.0
