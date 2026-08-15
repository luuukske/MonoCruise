"""Headless ACC replay: drives TrafficReader + ACCTracker over recorded AEB clips.

Mirrors RadarThread.loop and ACCThread.loop closely enough that tracker metrics
measured here match the live thread. See tests/acc/README.md for the metric
definitions and how the recorded baselines were produced."""
from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from core.acc.tracker import ACCTracker
from core.radar.ego_path import EGO_POSITION_HISTORY_LEN, ego_curvature_from_history
from core.radar.elevation import ElevationGate, EgoElevationTrack, build_surface
from core.radar.reader import TrafficReader

# Imported, not copied: retuning the controller must not leave these metrics
# silently measuring the old threshold while reporting the new behaviour.
from core.cruise_control_thread.acc_controller import ANT_SCORE_FULL, ANT_SCORE_MIN

CONF_MIN = ANT_SCORE_MIN
CONF_FULL = ANT_SCORE_FULL

# A track counts as stationary below this speed, moving above it.
STATIONARY_MS = 1.0

# ACC controller overlay constants mirrored for exposure accounting.
EGO_FRONT_OFFSET_M = 2.5
D_EMERGENCY_M = 1.5
TTC_HARD_S = 1.5
TTC_MIN_VCLOSE_MS = 0.3
STANDSTILL_SPEED_MS = 0.4


def clip_root() -> Path | None:
    """Local AEB clip store, or None when it is not present on this machine."""
    base = os.environ.get("LOCALAPPDATA")
    if base:
        p = Path(base) / "MonoCruise" / "aeb_clips"
        return p if p.is_dir() else None
    p = Path.home() / ".monocruise" / "aeb_clips"
    return p if p.is_dir() else None


def load_clips(limit: int):
    """Oldest-first slice of the local clip store so the sample is stable."""
    from core.aeb.clip_store import ClipStore

    root = clip_root()
    if root is None:
        return []
    store = ClipStore(root=root)
    infos = sorted(store.list_clips(), key=lambda i: i.path.name)[:limit]
    out = []
    for info in infos:
        clip = store.load(info.path)
        if clip is not None:
            out.append(clip)
    return out


def baseline_tag(baseline: float) -> str:
    """Trail-fit bucket from the offset baseline the tracker recorded."""
    if abs(baseline) < 0.05:
        return "HIT"
    if abs(baseline + 0.40) < 0.05:
        return "NO_ARC"
    return "NO_HIST"


@dataclass
class Metrics:
    """Aggregated tracker behaviour over a set of clips."""

    lock_s: list[float] = field(default_factory=list)
    # Locks split by prior track age at the corridor entry. See tests/acc/README.md.
    cutin_lock_s: list[float] = field(default_factory=list)
    fresh_lock_s: list[float] = field(default_factory=list)
    hook_s: list[float] = field(default_factory=list)
    buckets: dict[str, int] = field(default_factory=dict)
    moving_in_frames: int = 0
    moving_in_locked: int = 0
    stationary_frames: int = 0
    stationary_locked: int = 0
    positive_frames: int = 0
    saturated_frames: int = 0
    lead_frames: int = 0
    overlay_trips: int = 0
    overlay_trips_zero_conf: int = 0
    rank_mismatch: int = 0

    def bump(self, key: str) -> None:
        self.buckets[key] = self.buckets.get(key, 0) + 1

    @property
    def moving_lock_rate(self) -> float:
        return self.moving_in_locked / max(self.moving_in_frames, 1)

    @property
    def stationary_lock_rate(self) -> float:
        return self.stationary_locked / max(self.stationary_frames, 1)

    @property
    def saturated_rate(self) -> float:
        return self.saturated_frames / max(self.positive_frames, 1)

    @property
    def zero_conf_overlay_rate(self) -> float:
        return self.overlay_trips_zero_conf / max(self.overlay_trips, 1)

    def bucket_rate(self, key: str) -> float:
        return self.buckets.get(key, 0) / max(sum(self.buckets.values()), 1)

    def summary(self) -> str:
        def stat(name, vals):
            if not vals:
                return f"  {name:24s} (none)"
            return (f"  {name:24s} n={len(vals):5d} p50={statistics.median(vals):.2f}s "
                    f"p90={_pct(vals, 0.90):.2f}s")
        lines = [
            stat("lock in_path->s>1", self.lock_s),
            stat("  lock cut-in", self.cutin_lock_s),
            stat("  lock fresh id", self.fresh_lock_s),
            stat("hook out-edge->s<=1", self.hook_s),
            f"  moving in-corridor locked  {100 * self.moving_lock_rate:5.1f}% "
            f"(n={self.moving_in_frames})",
            f"  stationary locked          {100 * self.stationary_lock_rate:5.1f}% "
            f"(n={self.stationary_frames})",
            f"  score saturated            {100 * self.saturated_rate:5.1f}% "
            f"(of {self.positive_frames} positive)",
            f"  zero-conf overlay trips    {100 * self.zero_conf_overlay_rate:5.1f}% "
            f"(of {self.overlay_trips})",
        ]
        for key in ("HIT", "NO_ARC", "NO_HIST"):
            lines.append(f"  baseline {key:8s}          {100 * self.bucket_rate(key):5.1f}%")
        return "\n".join(lines)


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _conf(score: float) -> float:
    return max(0.0, min(1.0, (score - CONF_MIN) / (CONF_FULL - CONF_MIN)))


# A track known this long before entering the corridor counts as a cut-in
# rather than a vehicle that just appeared in radar range.
CUTIN_PRIOR_TRACK_S = 1.0


class _TrackWindows:
    """Per-id edge bookkeeping for the lock and hook latency metrics."""

    def __init__(self) -> None:
        self.in_since: dict[int, float] = {}
        self.locked: dict[int, bool] = {}
        self.prev_in: dict[int, bool] = {}
        self.hook_since: dict[int, float] = {}
        self.first_seen: dict[int, float] = {}
        self.was_cutin: dict[int, bool] = {}

    def step(self, vid: int, st, now: float, m: Metrics) -> None:
        self.first_seen.setdefault(vid, now)
        if st.in_path:
            if vid not in self.in_since:
                self.in_since[vid] = now
                self.locked[vid] = st.score > CONF_MIN
                self.was_cutin[vid] = (
                    now - self.first_seen[vid] >= CUTIN_PRIOR_TRACK_S
                )
            elif not self.locked.get(vid) and st.score > CONF_MIN:
                self.locked[vid] = True
                latency = now - self.in_since[vid]
                m.lock_s.append(latency)
                bucket = m.cutin_lock_s if self.was_cutin.get(vid) else m.fresh_lock_s
                bucket.append(latency)
            self.hook_since.pop(vid, None)
        else:
            self.in_since.pop(vid, None)
            if self.prev_in.get(vid) and st.score > CONF_MIN:
                self.hook_since[vid] = now
            elif vid in self.hook_since and st.score <= CONF_MIN:
                m.hook_s.append(now - self.hook_since.pop(vid))
        self.prev_in[vid] = st.in_path


def replay_clip(clip, metrics: Metrics, tracker: ACCTracker | None = None) -> ACCTracker:
    """Run one clip through a fresh reader + tracker, accumulating ``metrics``."""
    reader = TrafficReader()
    tracker = tracker or ACCTracker()
    windows = _TrackWindows()
    ego_hist: list[tuple[float, float, float]] = []
    elev_track = EgoElevationTrack()
    elev_gate = ElevationGate()
    last_hist_t = -1.0
    last_t = 0.0
    was_paused = False

    for frame in sorted(clip.radar_frames, key=lambda f: f.t_mono):
        ego = frame.ego
        if frame.traffic_buf is None or ego.paused:
            was_paused = True
            continue
        if was_paused:
            was_paused = False
            reader.request_reanchor()
            ego_hist = []
            last_hist_t = -1.0
            elev_track.clear()
            elev_gate.clear()

        t_kin = frame.t_wall
        if t_kin > last_hist_t:
            ego_hist.append((t_kin, ego.coordinateX, ego.coordinateZ))
            last_hist_t = t_kin
            ego_hist = ego_hist[-EGO_POSITION_HISTORY_LEN:]
        ego_kappa = ego_curvature_from_history(ego_hist)
        elev_track.push(ego.coordinateX, ego.coordinateZ, ego.coordinateY)

        decoded = reader.replay_frame(
            frame.traffic_buf, frame.parked_buf,
            ego.coordinateX, ego.coordinateY, ego.coordinateZ, ego.speed, t_kin,
        )
        if decoded is None:
            continue
        targets = decoded[0] + decoded[1]

        dt = (frame.t_mono - last_t) if last_t > 0 else 1.0 / 30.0
        dt = max(1e-3, min(dt, 0.2))
        last_t = frame.t_mono

        pitch_norm = (ego.rotationY + 0.5) % 1.0 - 0.5
        ego_yaw_rad = ego.rotationX * 2.0 * math.pi
        surface = build_surface(
            ego.coordinateY, -pitch_norm * 2.0 * math.pi, elev_track,
        )
        off_ids = elev_gate.step(
            targets, surface, ego.coordinateX, ego.coordinateZ, ego_yaw_rad,
        )
        leads = tracker.update(
            now_mono=frame.t_mono, dt=dt, vehicles=targets,
            ego_x=ego.coordinateX, ego_z=ego.coordinateZ,
            ego_yaw_rad=ego_yaw_rad,
            ego_speed_ms=ego.speed, ego_steer=ego.userSteer,
            ego_history_kappa=ego_kappa,
            blinker_left=bool(getattr(ego, "blinkerLeft", False)),
            blinker_right=bool(getattr(ego, "blinkerRight", False)),
            off_surface_ids=off_ids,
        )
        _accumulate(metrics, windows, tracker, targets, leads, ego, frame.t_mono)
    return tracker


def _accumulate(m: Metrics, windows, tracker, targets, leads, ego, now: float) -> None:
    by_id = {v.id: v for v in targets}
    for vid, st in tracker.tracks.items():
        if not st.last_seen_this_frame:
            continue
        m.bump(baseline_tag(st.last_baseline))
        vehicle = by_id.get(vid)
        speed = abs(vehicle.speed) if vehicle is not None else 0.0
        if speed < STATIONARY_MS:
            m.stationary_frames += 1
            if st.score > CONF_MIN:
                m.stationary_locked += 1
        elif st.in_path:
            m.moving_in_frames += 1
            if st.score > CONF_MIN:
                m.moving_in_locked += 1
        if st.score > 0.0:
            m.positive_frames += 1
            from core.acc.scoring import SCORE_MAX
            if st.score >= SCORE_MAX - 0.1:
                m.saturated_frames += 1
        windows.step(vid, st, now, m)

    if not leads:
        return
    m.lead_frames += 1
    if max(leads, key=lambda lead: lead.score).vehicle.id != leads[0].vehicle.id:
        m.rank_mismatch += 1
    primary = leads[0]
    dist = max(0.0, primary.dist_m - EGO_FRONT_OFFSET_M)
    if dist <= 0.0:
        return
    v_close = max(0.0, ego.speed) - primary.effective_speed_ms
    ttc = dist / max(v_close, TTC_MIN_VCLOSE_MS) if v_close > STANDSTILL_SPEED_MS else math.inf
    if dist <= D_EMERGENCY_M or ttc < TTC_HARD_S:
        m.overlay_trips += 1
        if _conf(primary.score) <= 0.0:
            m.overlay_trips_zero_conf += 1


def replay_corpus(clips) -> Metrics:
    """Fresh reader + tracker per clip; metrics summed across the corpus."""
    metrics = Metrics()
    for clip in clips:
        replay_clip(clip, metrics)
    return metrics


def make_vehicle(vid: int, x: float, z: float, speed: float, *, yaw_rad: float = 0.0,
                 history_speed: float | None = None, samples: int = 25,
                 dt: float = 1.0 / 15.0, width: float = 2.4, length: float = 5.0):
    """Synthetic straight-line Vehicle with a seeded position history.

    ``history_speed`` defaults to ``speed``; pass a non-zero value with
    ``speed=0`` to model a vehicle that drove in and then stopped."""
    from core.radar.traffic import Position, Quaternion, Size, Vehicle

    vehicle = Vehicle(
        Position(x, 0.0, z), Quaternion(1.0, 0.0, 0.0, 0.0),
        Size(width, 3.0, length), speed, 0.0, 0, [], vid, False, False,
    )
    vehicle._smooth_yaw = yaw_rad
    vehicle._smooth_x = x
    vehicle._smooth_z = z
    vehicle.acc_speed = speed
    fwd_x = -math.sin(yaw_rad)
    fwd_z = -math.cos(yaw_rad)
    trail_speed = speed if history_speed is None else history_speed
    history = []
    for i in range(samples):
        back = (samples - 1 - i) * trail_speed * dt
        history.append((i * dt, x - back * fwd_x, z - back * fwd_z))
    vehicle._position_history = history
    # Mirror the reader: the geometry trail is retained on a distance grid.
    trail: list[tuple[float, float, float]] = []
    for sample in history:
        if trail:
            _, lx, lz = trail[-1]
            if math.hypot(sample[1] - lx, sample[2] - lz) < 0.5:
                continue
        trail.append(sample)
    vehicle._trail_history = trail
    return vehicle
