"""Scene features from a decoded clip: per-target tracks, scenario tags, data flags.

Purely observational. Everything here comes from the recorded radar stream and the
recorded live decision, never from a re-run of the current AEB code, so a cached
feature row stays valid when the AEB tuning changes. Ask `replay` for the current
code's opinion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from core.aeb.calibration import DEFAULT as _CAL
from core.aeb.clip_replay import decode_radar_stream, nearest_frame_t
from core.aeb.clip_schema import Clip

from tools.aeb_agent import evasion

FEATURES_VERSION = 8

TTC_INF = 100.0
# A target further than this was never a candidate for anything, so a positive
# label on a clip with nothing inside it is a mistag signal.
NEAR_RANGE_M = 60.0
CLOSE_GAP_M = 12.0
# Heading dot bands: cos of the angle between ego forward and target forward.
CODIR_DOT = 0.6
ONCOMING_DOT = -0.6
STOPPED_MS = 0.5
USER_BRAKE_MIN = 0.15
# Reported speed above this with no matching raw displacement: the traffic
# stream did not update this body on this tick.
STALL_SPEED_MS = 3.0
STALL_NEAR_S = 2.0


def world_to_ego(wx: float, wz: float, ex: float, ez: float,
                 ego_yaw: float) -> tuple[float, float]:
    """(forward, right) metres in the ego frame. Mirrors `_w2e` in debug_window."""
    dx = wx - ex
    dz = wz - ez
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    rx = (-dx) * c - dz * s
    rz = (-dx) * s + dz * c
    return -rz, -rx


@dataclass
class TargetTrack:
    """One vehicle across the clip, measured in the ego frame."""

    vid: int
    ticks: int = 0
    is_tmp: bool = False
    is_trailer: bool = False
    has_trailer: bool = False
    length_m: float = 0.0
    half_w_m: float = 0.0
    min_range_m: float = 1e9
    min_gap_m: float = 1e9
    t_min_range: float = 0.0
    fwd_at_min_m: float = 0.0
    lat_at_min_m: float = 0.0
    speed_at_min_kmh: float = 0.0
    heading_dot_at_min: float = 0.0
    # Closest the body came while inside ego's straight-ahead corridor. This is
    # the rear-end proximity; min_gap_m also counts vehicles passing alongside.
    min_corridor_gap_m: float = 1e9
    t_min_corridor_gap: float = 0.0
    corridor_ticks: int = 0
    min_geom_ttc_s: float = TTC_INF
    max_closing_ms: float = 0.0
    ahead_frac: float = 0.0
    in_lane_frac: float = 0.0
    speed_min_kmh: float = 1e9
    speed_max_kmh: float = 0.0
    stopped_frac: float = 0.0
    lag_ticks: int = 0
    teleport_ticks: int = 0
    # Stream did not move this body although it reported speed. Diagnostic only,
    # no rule may key on it: see "Stream stalls" in this package's README.
    stall_ticks: int = 0
    stall_run_max: int = 0
    stall_near_action: int = 0
    _stall_run: int = 0
    max_jump_m: float = 0.0
    colliding_ticks: int = 0
    suppressed_ticks: int = 0
    suppression_stages: dict = field(default_factory=dict)

    @property
    def lag_suspect(self) -> bool:
        """Radar's own confirmed lag, or raw teleports. Never the stall counters."""
        return self.lag_ticks >= 3 or self.teleport_ticks >= 2

    def kind(self) -> str:
        """Coarse relative-motion class at closest approach."""
        dot = self.heading_dot_at_min
        if self.speed_max_kmh < 2.0:
            return "stationary"
        if dot >= CODIR_DOT:
            return "codirectional" if self.ahead_frac >= 0.5 else "overtaker"
        if dot <= ONCOMING_DOT:
            return "oncoming"
        return "crossing"


@dataclass
class ClipFeatures:
    """Everything the agent needs about a clip without decoding it again."""

    version: int = FEATURES_VERSION
    clip_id: str = ""
    path: str = ""
    duration_s: float = 0.0
    tick_count: int = 0
    ego_speed_min_kmh: float = 0.0
    ego_speed_max_kmh: float = 0.0
    ego_speed_at_action_kmh: float = 0.0
    ego_stopped_frac: float = 0.0
    ego_max_abs_steer: float = 0.0
    ego_max_kappa: float = 0.0
    ego_mass_kg: float | None = None
    ego_has_trailer: bool = False
    paused_frac: float = 0.0
    user_brake_max: float = 0.0
    user_brake_frac: float = 0.0
    user_gas_max: float = 0.0
    program_brake_max: float = 0.0
    aeb_disabled_frac: float = 0.0
    max_brake_ms2: float = 0.0
    warn_ticks: int = 0
    brake_ticks: int = 0
    engaged_ticks: int = 0
    first_warn_t: float | None = None
    first_brake_t: float | None = None
    peak_target_ms2: float = 0.0
    peak_raw_target_ms2: float = 0.0
    peak_required_ms2: float = 0.0
    min_ttc_s: float = TTC_INF
    min_ttb_s: float = TTC_INF
    recorded_band: list | None = None
    action_t: float = 0.0
    n_vehicles: int = 0
    primary_vid: int | None = None
    intervention: dict = field(default_factory=dict)
    counterfactual: dict = field(default_factory=dict)
    targets: list = field(default_factory=list)
    scenario: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def primary(self) -> dict | None:
        for t in self.targets:
            if t.get("vid") == self.primary_vid:
                return t
        return self.targets[0] if self.targets else None


def _ego_kappa(steer: float) -> float:
    """AEB's yaw-rate proxy curvature. Speed cancels, see core/aeb/README.md."""
    return math.radians(steer * _CAL.yaw_rate_steer_gain)


def _veh_yaw(v) -> float:
    smooth = getattr(v, "_smooth_yaw", None)
    if smooth is not None:
        return smooth
    return math.radians(v.rotation.euler()[1])


def _raw_motion(v, prev, dt: float) -> tuple[float, bool, bool]:
    """Raw displacement this tick, plus the teleport and stream-stall verdicts.

    A stall is a tick where the body reported speed the raw stream did not move
    it by. That is ordinary at TMP update rates, so it is reported and never
    acted on; `lag_confirmed` is the gated signal that discriminates.
    """
    rx = getattr(v, "_raw_x", None)
    px = getattr(prev, "_raw_x", None)
    if rx is None or px is None or dt <= 1e-6 or dt > 1.0:
        return 0.0, False, False
    jump = math.hypot(rx - px, getattr(v, "_raw_z", 0.0) - getattr(prev, "_raw_z", 0.0))
    expected = abs(v.speed) * dt
    teleport = jump > max(4.0, 3.0 * expected + 2.0)
    stalled = abs(v.speed) > STALL_SPEED_MS and jump < 0.2 * expected
    return jump, teleport, stalled


def _tick_context(clip: Clip):
    """Recorded ticks sorted, plus t0 and the recorded-band helpers."""
    ticks = sorted(clip.aeb_ticks, key=lambda x: x.t_mono)
    all_t = [f.t_mono for f in clip.radar_frames] + [t.t_mono for t in ticks]
    t0 = min(all_t) if all_t else 0.0
    return ticks, t0


def _live_summary(feats: ClipFeatures, ticks, t0: float) -> None:
    warn_t, brake_t, tracked_t = [], [], []
    for tk in ticks:
        la = tk.live_aeb
        t_rel = tk.t_mono - t0
        if la.aeb_warn:
            warn_t.append(t_rel)
        if la.aeb_brake:
            brake_t.append(t_rel)
        if la.engaged:
            feats.engaged_ticks += 1
        if la.time_to_collision < TTC_INF:
            tracked_t.append(t_rel)
            feats.min_ttc_s = min(feats.min_ttc_s, la.time_to_collision)
        if la.time_to_brake < TTC_INF:
            feats.min_ttb_s = min(feats.min_ttb_s, la.time_to_brake)
        feats.peak_target_ms2 = max(feats.peak_target_ms2, la.target_decel_ms2)
        feats.peak_required_ms2 = max(
            feats.peak_required_ms2, min(la.required_decel_ms2, 50.0))
        con = tk.consumed
        feats.user_brake_max = max(feats.user_brake_max, con.brakeval)
        feats.user_gas_max = max(feats.user_gas_max, con.gasval)
        feats.program_brake_max = max(feats.program_brake_max, con.program_brake)
        feats.max_brake_ms2 = max(feats.max_brake_ms2, con.max_brake_ms2)
        if con.brakeval > USER_BRAKE_MIN:
            feats.user_brake_frac += 1.0
        if not con.aeb_enabled:
            feats.aeb_disabled_frac += 1.0

    n = max(len(ticks), 1)
    feats.user_brake_frac /= n
    feats.aeb_disabled_frac /= n
    feats.warn_ticks = len(warn_t)
    feats.brake_ticks = len(brake_t)
    feats.first_warn_t = warn_t[0] if warn_t else None
    feats.first_brake_t = brake_t[0] if brake_t else None
    band = warn_t + brake_t
    if band:
        feats.recorded_band = [min(band), max(band)]
    elif tracked_t:
        feats.recorded_band = [tracked_t[0], tracked_t[-1]]
    feats.action_t = (
        feats.first_brake_t if feats.first_brake_t is not None
        else feats.first_warn_t if feats.first_warn_t is not None
        else (feats.recorded_band[0] if feats.recorded_band else 0.0)
    )


def _accumulate(tracks: dict[int, TargetTrack], v, ego, ego_yaw: float,
                t_rel: float, prev_v, dt: float, live,
                action_t: float = 0.0) -> None:
    tr = tracks.get(v.id)
    if tr is None:
        tr = TargetTrack(vid=int(v.id))
        tracks[v.id] = tr
    fwd, right = world_to_ego(v.position.x, v.position.z,
                              ego.coordinateX, ego.coordinateZ, ego_yaw)
    rng = math.hypot(fwd, right)
    half_len = v.size.length / 2.0
    gap = rng - _CAL.ego_half_length - half_len
    speed_kmh = abs(v.speed) * 3.6

    tr.ticks += 1
    tr.is_tmp = bool(v.is_tmp)
    tr.is_trailer = bool(getattr(v, "is_trailer", False))
    tr.has_trailer = tr.has_trailer or bool(v.trailers)
    tr.length_m = v.size.length
    tr.half_w_m = v.size.width / 2.0
    if rng < tr.min_range_m:
        tr.min_range_m = rng
        tr.min_gap_m = gap
        tr.t_min_range = t_rel
        tr.fwd_at_min_m = fwd
        tr.lat_at_min_m = right
        tr.speed_at_min_kmh = speed_kmh
        tr.heading_dot_at_min = math.cos(ego_yaw - _veh_yaw(v))
    if fwd > 0.0:
        tr.ahead_frac += 1.0
    if abs(right) <= _CAL.lane_half_width:
        tr.in_lane_frac += 1.0
    tr.speed_max_kmh = max(tr.speed_max_kmh, speed_kmh)
    tr.speed_min_kmh = min(tr.speed_min_kmh, speed_kmh)
    if abs(v.speed) < STOPPED_MS:
        tr.stopped_frac += 1.0
    if getattr(v, "lag_confirmed", False):
        tr.lag_ticks += 1

    corridor = _CAL.ego_half_width + v.size.width / 2.0 + _CAL.corridor_margin
    in_corridor = abs(right) <= corridor and fwd > 0.0
    fwd_gap = fwd - _CAL.ego_half_length - half_len
    if in_corridor:
        tr.corridor_ticks += 1
        if fwd_gap < tr.min_corridor_gap_m:
            tr.min_corridor_gap_m = fwd_gap
            tr.t_min_corridor_gap = t_rel

    if prev_v is not None:
        jump, teleported, stalled = _raw_motion(v, prev_v, dt)
        tr.max_jump_m = max(tr.max_jump_m, jump)
        if teleported:
            tr.teleport_ticks += 1
        if stalled:
            tr.stall_ticks += 1
            tr._stall_run += 1
            tr.stall_run_max = max(tr.stall_run_max, tr._stall_run)
            if abs(t_rel - action_t) <= STALL_NEAR_S:
                tr.stall_near_action += 1
        else:
            tr._stall_run = 0
        prev_fwd, prev_right = world_to_ego(
            prev_v.position.x, prev_v.position.z,
            ego.coordinateX, ego.coordinateZ, ego_yaw)
        prev_rng = math.hypot(prev_fwd, prev_right)
        if dt > 1e-6:
            closing = (prev_rng - rng) / dt
            tr.max_closing_ms = max(tr.max_closing_ms, closing)
            if in_corridor and closing > 0.5 and fwd_gap > 0.0:
                tr.min_geom_ttc_s = min(tr.min_geom_ttc_s, fwd_gap / closing)
    if v.id in live.colliding_ids:
        tr.colliding_ticks += 1
    if v.id in live.suppressed_ids:
        tr.suppressed_ticks += 1
    for stage in live.suppression_reasons.get(str(v.id), []) or \
            live.suppression_reasons.get(v.id, []):
        tr.suppression_stages[stage] = tr.suppression_stages.get(stage, 0) + 1


def _pick_primary(tracks: dict[int, TargetTrack], label_vid: int | None) -> int | None:
    """Closest genuine threat by geometry.

    Deliberately not "the id AEB flagged most": that picks the target AEB was
    looking at, which is the opinion this corpus exists to judge. Colliding ticks
    are a tiebreak only.
    """
    if not tracks:
        return None
    corridor = [t for t in tracks.values() if t.corridor_ticks > 0]
    if corridor:
        return min(corridor,
                   key=lambda t: (t.min_corridor_gap_m, -t.colliding_ticks)).vid
    if label_vid is not None and label_vid in tracks:
        return int(label_vid)
    return min(tracks.values(),
               key=lambda t: (t.min_range_m, -t.colliding_ticks)).vid


def _tags(feats: ClipFeatures, primary: TargetTrack | None) -> None:
    scen, flags = feats.scenario, feats.flags
    if feats.ego_speed_max_kmh < 30.0:
        scen.append("low_speed")
    if feats.ego_stopped_frac > 0.5:
        scen.append("ego_stopped")
    if feats.ego_max_kappa > 0.012:
        scen.append("ego_turning")
    if feats.user_brake_frac > 0.05 or feats.user_brake_max > 0.4:
        scen.append("user_braking")
    inter = feats.intervention or {}
    cf = feats.counterfactual or {}
    if inter.get("found"):
        kind = inter.get("kind", "")
        if "swerve" in kind:
            scen.append("driver_swerve")
        if "brake" in kind:
            scen.append("driver_brake")
        if "lift" in kind:
            scen.append("driver_lift")
    verdict = cf.get("verdict", "unknown")
    if cf.get("ran") and verdict in ("collides", "likely"):
        scen.append("would_have_collided")
        flags.append(
            f"driver evasion changed the outcome ({verdict}): holding course "
            f"comes within {cf.get('min_separation_m', 0):+.1f} m of vid "
            f"{cf.get('target_vid')} at {cf.get('dt_at_min_s', 0):.2f} s past the "
            f"fork. The recorded near miss is not evidence there was no threat")
    elif cf.get("ran") and verdict == "close":
        scen.append("evasion_close")
    if feats.ego_has_trailer:
        scen.append("ego_trailer")
    if primary is None:
        scen.append("no_target")
    else:
        scen.append(primary.kind())
        if primary.min_range_m > NEAR_RANGE_M:
            scen.append("far_only")
        if primary.corridor_ticks == 0:
            scen.append("never_in_corridor")
        elif primary.min_corridor_gap_m < CLOSE_GAP_M:
            scen.append("close_approach")
        if primary.in_lane_frac > 0.25:
            scen.append("in_lane")
        if primary.has_trailer or primary.is_trailer:
            scen.append("trailer_target")
        if primary.is_tmp:
            scen.append("tmp_target")
        if primary.lag_suspect:
            scen.append("lag_suspect")
            flags.append(
                f"primary lag_confirmed on {primary.lag_ticks} ticks, "
                f"{primary.teleport_ticks} raw teleports, max jump "
                f"{primary.max_jump_m:.1f} m")

    if feats.paused_frac > 0.10:
        flags.append(f"paused for {feats.paused_frac:.0%} of frames")
    if feats.duration_s < 3.0:
        flags.append(f"short clip ({feats.duration_s:.1f} s)")
    if feats.tick_count < 20:
        flags.append(f"only {feats.tick_count} AEB ticks")
    if feats.aeb_disabled_frac > 0.2:
        flags.append(f"AEB disabled for {feats.aeb_disabled_frac:.0%} of ticks")
    if feats.n_vehicles == 0:
        flags.append("no traffic decoded at all")
    if primary is not None and primary.min_range_m > NEAR_RANGE_M:
        flags.append(f"nothing closer than {primary.min_range_m:.0f} m all clip")
    if primary is not None and primary.corridor_ticks == 0:
        flags.append("no target ever entered ego's corridor")
    if feats.user_brake_max > 0.5:
        flags.append(f"driver braked to {feats.user_brake_max:.2f}")


class _PedalFrame:
    """A radar frame with the pedal state of its nearest AEB tick attached.

    `evasion.detect` needs steering and pedals on one timeline; the schema keeps
    steering on radar frames and pedals on AEB ticks.
    """

    __slots__ = ("t_mono", "ego", "brakeval", "gasval")

    def __init__(self, frame, brakeval: float, gasval: float) -> None:
        self.t_mono = frame.t_mono
        self.ego = frame.ego
        self.brakeval = brakeval
        self.gasval = gasval


def _pedal_frames(frames, ticks) -> list:
    """Radar frames carrying the pedals of the tick that consumed them."""
    by_radar: dict[float, object] = {}
    for tk in ticks:
        by_radar.setdefault(tk.radar_t_mono, tk.consumed)
    tick_times = sorted(by_radar)
    out = []
    for f in frames:
        con = by_radar.get(f.t_mono)
        if con is None and tick_times:
            near = min(tick_times, key=lambda t: abs(t - f.t_mono))
            con = by_radar[near] if abs(near - f.t_mono) < 0.25 else None
        out.append(_PedalFrame(f, getattr(con, "brakeval", 0.0),
                               getattr(con, "gasval", 0.0)))
    return out


def _vehicle_dicts(vehicles) -> list[dict]:
    """Minimal body dicts the counterfactual needs, tractor plus trailers."""
    out = []
    for v in vehicles:
        trailers = []
        for tr in v.trailers:
            trailers.append({
                "x": tr.position.x, "z": tr.position.z,
                "yaw": math.radians(tr.rotation.euler()[1]),
                "length": tr.size.length, "half_w": tr.size.width / 2.0,
            })
        out.append({
            "vid": int(v.id), "x": v.position.x, "z": v.position.z,
            "yaw": _veh_yaw(v), "length": v.size.length,
            "half_w": v.size.width / 2.0, "speed_kmh": abs(v.speed) * 3.6,
            "trailers": trailers,
        })
    return out


def _counterfactual(feats: ClipFeatures, clip: Clip, frames, ticks,
                    t0: float, veh_by_t, frame_t) -> None:
    """Detect the driver's intervention and fly the no-action ghost past it."""
    if not frames:
        return
    pedal = _pedal_frames(frames, ticks)
    inter = evasion.detect(pedal, t0, feats.action_t)
    feats.intervention = asdict(inter)
    if not inter.found:
        feats.counterfactual = asdict(evasion.Counterfactual(
            reason="no intervention, the recorded path is the no-action path"))
        return

    cache: dict[float, list] = {}

    def veh_at(t_mono: float) -> list:
        got = cache.get(t_mono)
        if got is None:
            ft = nearest_frame_t(frame_t, t_mono)
            got = _vehicle_dicts(veh_by_t.get(ft, []) if ft is not None else [])
            cache[t_mono] = got
        return got

    feats.counterfactual = asdict(evasion.run(inter, pedal, veh_at, t0))


def extract(clip: Clip, path: str = "") -> ClipFeatures:
    """Decode the radar stream and summarize the scene, the driver and live AEB."""
    feats = ClipFeatures(clip_id=clip.metadata.clip_id, path=path)
    ticks, t0 = _tick_context(clip)
    feats.tick_count = len(ticks)
    feats.ego_mass_kg = None
    frames = sorted(clip.radar_frames, key=lambda f: f.t_mono)
    if frames:
        feats.duration_s = max(
            frames[-1].t_mono - t0,
            (ticks[-1].t_mono - t0) if ticks else 0.0)
        feats.ego_mass_kg = frames[0].ego.estimated_total_mass_kg
        feats.ego_has_trailer = bool(frames[0].ego.ego_has_trailer)
        feats.paused_frac = sum(1 for f in frames if f.ego.paused) / len(frames)
    if not ticks:
        _tags(feats, None)
        return feats

    _live_summary(feats, ticks, t0)

    veh_by_t, ego_by_t, frame_t, _off = decode_radar_stream(clip)
    tracks: dict[int, TargetTrack] = {}
    prev_by_vid: dict[int, object] = {}
    prev_ft: float | None = None
    speeds: list[float] = []
    stopped = 0
    for tk in ticks:
        ft = nearest_frame_t(frame_t, tk.radar_t_mono)
        ego = ego_by_t.get(ft) if ft is not None else None
        if ego is None:
            continue
        ego_yaw = ego.rotationX * 2.0 * math.pi
        speeds.append(ego.speed * 3.6)
        if ego.speed < 0.5:
            stopped += 1
        feats.ego_max_abs_steer = max(feats.ego_max_abs_steer, abs(ego.userSteer))
        feats.ego_max_kappa = max(feats.ego_max_kappa, abs(_ego_kappa(ego.userSteer)))
        dt = (ft - prev_ft) if (prev_ft is not None and ft is not None) else 0.0
        for v in veh_by_t.get(ft, []):
            _accumulate(tracks, v, ego, ego_yaw, tk.t_mono - t0,
                        prev_by_vid.get(v.id), dt, tk.live_aeb, feats.action_t)
            prev_by_vid[v.id] = v
        prev_ft = ft

    if speeds:
        feats.ego_speed_min_kmh = min(speeds)
        feats.ego_speed_max_kmh = max(speeds)
        feats.ego_stopped_frac = stopped / len(speeds)
        idx = min(range(len(ticks)), key=lambda i: abs(
            (ticks[i].t_mono - t0) - feats.action_t))
        feats.ego_speed_at_action_kmh = speeds[min(idx, len(speeds) - 1)]

    for tr in tracks.values():
        if tr.ticks:
            tr.ahead_frac /= tr.ticks
            tr.in_lane_frac /= tr.ticks
            tr.stopped_frac /= tr.ticks
        if tr.speed_min_kmh > 1e8:
            tr.speed_min_kmh = 0.0
    feats.n_vehicles = len(tracks)
    label = clip.metadata.label
    feats.primary_vid = _pick_primary(tracks, label.target_vid if label else None)
    ordered = sorted(tracks.values(),
                     key=lambda t: (t.min_corridor_gap_m, t.min_range_m,
                                    -t.colliding_ticks))
    feats.targets = [asdict(t) for t in ordered[:12]]
    _counterfactual(feats, clip, frames, ticks, t0, veh_by_t, frame_t)
    _tags(feats, tracks.get(feats.primary_vid) if feats.primary_vid is not None else None)
    return feats
