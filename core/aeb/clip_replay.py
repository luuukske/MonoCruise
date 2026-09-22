"""Clip review: replay radar poses, live_aeb decisions, rebuilt arcs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.aeb.calibration import DEFAULT as _CAL, ego_path_params
from core.aeb.clip_schema import Clip, ConsumedContext, LiveAEB, RadarFrameRecord
from core.aeb.clip_timebase import decode_buffers, replay_frames
from core.aeb.filters import VehicleCurvatureBlender, _vehicle_curvature_blend, travel_sign
from core.aeb.thread import (
    AEBSnapshot, AEBState, _INF, _dampen_turning_curvature,
    _swap_trailer_kinematics,
)
from core.radar.ego_path_model import EgoPathModel, EgoPathState, warm_gain
from core.radar.elevation import ElevationGate, EgoElevationTrack, build_surface
from core.radar.reader import TrafficReader
from core.radar.traffic import (
    ArcPath, Vehicle, build_arc, is_sub_frame, _raw_speed_from_position_history,
)

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


def ego_path_replay(clip: Clip, cal=_CAL, frames=None) -> tuple[dict[float, float], float]:
    """Sim-clock time per radar frame, plus the steer gain a live session would hold.

    Clips never record the learned gain (a new clip field bumps CONSENT_VERSION),
    so replay re-derives it from the whole clip: a live truck has been learning
    far longer than an 11 second window, and starting from the prior every time
    would replay a colder model than the one that made the recorded decisions.
    ``frames`` reuses a ``replay_frames`` result so the gain sees that clock.
    """
    if frames is None:
        frames = replay_frames(clip)
    tkin_by_t = {f.t_mono: f.t_wall for f in frames}
    samples = [
        (f.t_wall, f.ego.rotationX * 2.0 * math.pi, f.ego.speed, f.ego.userSteer)
        for f in frames
        if not f.ego.paused
    ]
    return tkin_by_t, warm_gain(samples, ego_path_params(cal))


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
    sign = travel_sign(v.speed, _CAL)
    veh_fwd_x = -sign * math.sin(v_yaw)
    veh_fwd_z = -sign * math.cos(v_yaw)
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
                    blender: VehicleCurvatureBlender, now: float,
                    ego_path=None) -> AEBSnapshot:
    ego_x = ego.coordinateX
    ego_z = ego.coordinateZ
    ego_yaw = ego.rotationX * 2.0 * math.pi
    ego_speed = ego.speed
    ego_hw = _CAL.ego_half_width
    ego_hl = _CAL.ego_half_length

    capacity = max(consumed.max_brake_ms2, 1.0)
    t_stop = ego_speed / (_CAL.ego_decel_frac * capacity) if capacity > 0 else 0.0
    horizon = min(max(_CAL.arc_horizon_min, t_stop * 2.0), _CAL.arc_horizon_max)
    path = ego_path if ego_path is not None else EgoPathState()
    curv = path.kappa_path

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
        ego_kappa_steer=path.kappa_steer,
        ego_kappa_meas=path.kappa_meas,
        ego_steer_gain=path.gain,
        ego_path_saturated=path.saturated,
    )


# Buffer positions read ahead of the clip start to measure a first-sighting
# speed: 4 samples give the 3 intervals the LS fit needs to reject one bad step.
_SEED_SAMPLES: int = 4
_SEED_MAX_SPAN_S: float = 1.0


def cold_start_speeds(
    clip: Clip, frames: list[RadarFrameRecord] | None = None,
) -> dict[int, float]:
    """Signed m/s per vehicle id at the clip's first frame, from later frames.

    A clip window is an arbitrary cut of a continuous stream, so a vehicle in the
    first frame was already moving. Live radar cannot know that and starts the
    speed chain cold; offline the samples are right there, so read ahead and run
    the same estimator the live chain uses. See core/aeb/README.md. ``frames``
    reuses a ``replay_frames`` result.
    """
    if frames is None:
        frames = replay_frames(clip)
    samples: dict[int, list[tuple[float, float, float]]] = {}
    yaws: dict[int, float] = {}
    first_ids: set[int] | None = None
    t_start: float | None = None
    for f in frames:
        if f.traffic_buf is None or f.ego.paused:
            continue
        if t_start is None:
            t_start = f.t_wall
        elif f.t_wall - t_start > _SEED_MAX_SPAN_S:
            break
        vehicles = decode_buffers(f.traffic_buf, f.parked_buf)
        if vehicles is None:
            continue
        if first_ids is None:
            first_ids = {int(v.id) for v in vehicles}
        for v in vehicles:
            vid = int(v.id)
            if vid not in first_ids:
                continue
            hist = samples.setdefault(vid, [])
            if len(hist) >= _SEED_SAMPLES:
                continue
            # Same cadence as a live full update, so the fit sees the same samples.
            if hist and is_sub_frame(f.t_wall - hist[-1][0]):
                continue
            if not hist:
                yaws[vid] = math.radians(v.rotation.euler()[1])
            hist.append((f.t_wall, v.position.x, v.position.z))
        if first_ids and all(
            len(samples.get(vid, ())) >= _SEED_SAMPLES for vid in first_ids
        ):
            break

    out: dict[int, float] = {}
    for vid, hist in samples.items():
        yaw = yaws.get(vid, 0.0)
        speed = _raw_speed_from_position_history(
            hist, -math.sin(yaw), -math.cos(yaw),
        )
        if speed is not None:
            out[vid] = speed
    return out


def decode_radar_stream(clip: Clip, as_recorded: bool = False, *,
                        frames=None, with_elevation: bool = True):
    """Smoothed vehicles + elevation gate per radar frame, as RadarThread.loop runs them.

    Frames come from ``replay_frames``: legacy ego poses re-paired, simulated clock.
    ``frames`` reuses that result. ``with_elevation`` defaults on.
    Returns ``(veh_by_t, ego_by_t, frame_t, off_by_t)``.
    """
    if frames is None:
        frames = replay_frames(clip, as_recorded=as_recorded)
    reader = TrafficReader()
    reader.set_cold_start_speeds(cold_start_speeds(clip, frames))
    veh_by_t: dict[float, list[Vehicle]] = {}
    off_by_t: dict[float, frozenset[int]] = {}
    elev_track = EgoElevationTrack() if with_elevation else None
    elev_gate = ElevationGate() if with_elevation else None
    last_vehs: list[Vehicle] = []
    last_off: frozenset[int] = frozenset()
    was_paused = False
    for f in frames:
        if f.traffic_buf is None or f.ego.paused:
            was_paused = True
            veh_by_t[f.t_mono] = list(last_vehs)
            off_by_t[f.t_mono] = last_off
            continue
        if was_paused:
            was_paused = False
            reader.request_reanchor()
            if elev_track is not None:
                elev_track.clear()
                elev_gate.clear()
        ego = f.ego
        if elev_track is not None:
            elev_track.push(ego.coordinateX, ego.coordinateZ, ego.coordinateY)
        res = reader.replay_frame(
            f.traffic_buf, f.parked_buf,
            ego.coordinateX, ego.coordinateY, ego.coordinateZ, ego.speed,
            f.t_wall,
        )
        last_vehs = list(res[0]) if res is not None else []
        if elev_track is not None:
            trailers = list(res[1]) if res is not None else []
            pitch_norm = (ego.rotationY + 0.5) % 1.0 - 0.5
            surface = build_surface(
                ego.coordinateY, -pitch_norm * 2.0 * math.pi, elev_track,
            )
            last_off = elev_gate.step(
                last_vehs + trailers, surface,
                ego.coordinateX, ego.coordinateZ, ego.rotationX * 2.0 * math.pi,
            )
        veh_by_t[f.t_mono] = last_vehs
        off_by_t[f.t_mono] = last_off
    ego_by_t = {f.t_mono: f.ego for f in frames}
    return veh_by_t, ego_by_t, sorted(veh_by_t), off_by_t


def nearest_frame_t(frame_t: list[float], t: float) -> float | None:
    """Frame timestamp closest to ``t`` (exact match preferred)."""
    if not frame_t:
        return None
    if t in frame_t:
        return t
    return min(frame_t, key=lambda ft: abs(ft - t))


def clip_t0(clip: Clip) -> float:
    """Clip-relative time origin. Every t_rel in the review tools is measured from it."""
    ts = [f.t_mono for f in clip.radar_frames] + [tk.t_mono for tk in clip.aeb_ticks]
    return min(ts) if ts else 0.0


def replay_clip(clip: Clip, *, stream=None, radar_frames=None) -> list[ReviewFrame]:
    """Decode + smooth the radar stream and build one ReviewFrame per AEB tick.

    ``stream`` and ``radar_frames`` reuse one ``replay_frames`` result. Passing
    ``stream`` alone rebuilds the timebase for the ego-path gain.
    """
    if not clip.aeb_ticks and not clip.radar_frames:
        return []

    if stream is None:
        if radar_frames is None:
            radar_frames = replay_frames(clip)
        stream = decode_radar_stream(clip, frames=radar_frames)
    elif radar_frames is None:
        radar_frames = replay_frames(clip)

    veh_by_t, ego_by_t, frame_t, _off_by_t = stream
    frames = sorted(clip.radar_frames, key=lambda f: f.t_mono)

    # Same ego path model the live loop runs, stepped on the same frames.
    tkin_by_t, warm = ego_path_replay(clip, frames=radar_frames)
    ego_path = EgoPathModel(params=ego_path_params(_CAL), gain=warm)

    t0 = clip_t0(clip)

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
        t_kin = tkin_by_t.get(ft, 0.0) if ft is not None else 0.0
        if t_kin > 0.0 and not ego.paused:
            path_state = ego_path.step(
                t_kin, ego.rotationX * 2.0 * math.pi, ego.speed, ego.userSteer,
            )
        else:
            path_state = ego_path.state
        snap = _build_snapshot(
            ego, vehicles_eff, tk.live_aeb, tk.consumed, blender, tk.t_mono,
            ego_path=path_state,
        )
        out.append(ReviewFrame(
            tk.t_mono - t0, tk.t_mono, snap, tk.live_aeb, tk.consumed,
            raw_target_decel(tk.live_aeb, cal_rec),
        ))
    return out


def clip_duration(clip: Clip) -> float:
    ts = [f.t_mono for f in clip.radar_frames] + [t.t_mono for t in clip.aeb_ticks]
    return (max(ts) - min(ts)) if ts else 0.0
