"""Clip replay resampled onto a constant video clock, with pose jitter removed.

Recorded poses are exact positions sampled at the wrong instants: ego and traffic
are both quantized to 60 Hz physics ticks, so a ~30 Hz stream alternates 3-tick and
1-tick steps and a straight drive plays back as a stutter. Every pose here goes
through a Gaussian-weighted local linear fit instead. `tools/clip_export/README.md`
has the numbers.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field, replace

from core.aeb.calibration import DEFAULT as _CAL
from core.aeb.clip_replay import ReviewFrame
from core.radar.traffic import build_arc

POSE_SIGMA_S = 0.06
ARC_SIGMA_S = 0.12
SPEED_SIGMA_S = 0.10
TRACK_JUMP_M = 12.0
TRACK_MAX_SPEED_MS = 45.0
TRACK_GAP_S = 0.35
THREAT_GAP_TICKS = 2
CORRIDOR_SAMPLES = 24
ARC_HOLD_S = 0.5

# Ego trailer is drawn from a kinematic follower: clips record no ego trailer pose.
TRAILER_LEN_M = 13.6
TRAILER_HALF_W_M = 1.3
KINGPIN_BACK_M = 2.4
KINGPIN_INSET_M = 1.2
AXLE_FROM_REAR_M = 2.2
TRAILER_SIM_DT_S = 1.0 / 120.0

Corridor = tuple[list[tuple[float, float]], list[tuple[float, float]]]


@dataclass(frozen=True)
class Body:
    """A rectangle in world metres; forward is ``(-sin yaw, -cos yaw)``."""

    x: float
    z: float
    yaw: float
    half_w: float
    length: float

    def corners(self) -> list[tuple[float, float]]:
        fx, fz = -math.sin(self.yaw), -math.cos(self.yaw)
        rx, rz = fz, -fx
        hl = self.length / 2.0
        return [
            (self.x + fx * sl * hl + rx * sw * self.half_w,
             self.z + fz * sl * hl + rz * sw * self.half_w)
            for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]


@dataclass
class VehicleView:
    vid: int
    body: Body
    trailers: list[Body]
    threat: bool


@dataclass
class FrameState:
    """Everything the painter and the camera need for one output frame."""

    t: float
    ego: Body
    ego_trailer: Body | None
    ego_speed_ms: float
    ego_corridor: Corridor | None
    vehicles: list[VehicleView]
    corridors: dict[int, list[Corridor]]
    state: int
    primary_vid: int | None

    def vehicle(self, vid: int | None) -> VehicleView | None:
        if vid is None:
            return None
        return next((v for v in self.vehicles if v.vid == vid), None)


def local_linear(ts: list[float], vs: list[float], t: float, sigma: float) -> float:
    """Gaussian-weighted local linear fit of ``vs(ts)`` evaluated at ``t``.

    Unlike a kernel average it passes constant velocity through unbiased, which is
    what keeps a moving body on its line where the kernel goes one-sided at a track end.
    """
    lo = bisect_left(ts, t - 3.0 * sigma)
    hi = bisect_right(ts, t + 3.0 * sigma)
    if hi <= lo:
        i = bisect_left(ts, t)
        if i >= len(ts) or (i > 0 and t - ts[i - 1] < ts[i] - t):
            i -= 1
        return vs[max(i, 0)]
    inv = 0.5 / (sigma * sigma)
    sw = st = stt = sv = stv = 0.0
    for i in range(lo, hi):
        d = ts[i] - t
        w = math.exp(-d * d * inv)
        sw += w
        st += w * d
        stt += w * d * d
        sv += w * vs[i]
        stv += w * d * vs[i]
    det = sw * stt - st * st
    if det <= 1e-6 * sw * sw * sigma * sigma:
        return sv / sw
    return (stt * sv - st * stv) / det


def unwrap(prev: float, angle: float) -> float:
    """``angle`` shifted by whole turns to sit within half a turn of ``prev``."""
    return angle + 2.0 * math.pi * round((prev - angle) / (2.0 * math.pi))


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class _Series:
    ts: list[float] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    zs: list[float] = field(default_factory=list)
    yaws: list[float] = field(default_factory=list)

    def push(self, t: float, x: float, z: float, yaw: float) -> None:
        if self.yaws:
            yaw = unwrap(self.yaws[-1], yaw)
        self.ts.append(t)
        self.xs.append(x)
        self.zs.append(z)
        self.yaws.append(yaw)

    def pose(self, t: float, sigma: float) -> tuple[float, float, float]:
        t = min(max(t, self.ts[0]), self.ts[-1])
        return (local_linear(self.ts, self.xs, t, sigma),
                local_linear(self.ts, self.zs, t, sigma),
                local_linear(self.ts, self.yaws, t, sigma))


@dataclass
class _Track:
    vid: int
    series: _Series
    half_w: float
    length: float
    trailers: list[tuple[_Series, float, float]] = field(default_factory=list)

    @property
    def t0(self) -> float:
        return self.series.ts[0]

    @property
    def t1(self) -> float:
        return self.series.ts[-1]


def pick_primary_threat(frames: list[ReviewFrame], label_vid: int | None) -> int | None:
    """The vehicle the story is about: labelled target, else what AEB acted on."""
    present = {int(v["vid"]) for f in frames for v in f.snapshot.vehicles}
    if label_vid is not None and int(label_vid) in present:
        return int(label_vid)
    for active_only in (True, False):
        counts = Counter(
            int(i) for f in frames
            if not active_only or f.live_aeb.aeb_warn or f.live_aeb.aeb_brake
            for i in f.snapshot.colliding_ids
        )
        if counts:
            return counts.most_common(1)[0][0]
    return None


def threat_span(frames: list[ReviewFrame], vid: int | None) -> tuple[float, float] | None:
    """Clip time over which AEB tracked ``vid`` on a collision course."""
    if vid is None:
        return None
    hits = [f.t_rel for f in frames if vid in f.snapshot.colliding_ids]
    if not hits:
        hits = [f.t_rel for f in frames if f.live_aeb.aeb_warn or f.live_aeb.aeb_brake]
    return (hits[0], hits[-1]) if hits else None


def threat_ticks(frames: list[ReviewFrame], max_gap: int = THREAT_GAP_TICKS) -> list[set[int]]:
    """Recorded ``colliding_ids`` per tick, with dropouts of ``max_gap`` ticks bridged.

    Only interior gaps are filled, so a highlight never starts earlier or ends later than
    the recording; it just stops blinking on a one-tick collision-grid miss.
    """
    out = [{int(i) for i in f.snapshot.colliding_ids} for f in frames]
    last_seen: dict[int, int] = {}
    for j, ids in enumerate(out):
        for vid in list(ids):
            k = last_seen.get(vid)
            if k is not None and 1 < j - k <= max_gap + 1:
                for m in range(k + 1, j):
                    out[m].add(vid)
            last_seen[vid] = j
    return out


def with_rerun_decisions(frames: list[ReviewFrame], ticks) -> list[ReviewFrame]:
    """Replay frames carrying ``clip_eval.run_headless`` decisions in place of the recorded ones.

    Geometry and ego motion stay recorded: the re-run is open loop, it cannot move the truck.
    Frames the re-run skipped are dropped rather than left showing the old decision.
    """
    by_t = {tk.t_mono: tk for tk in ticks}
    out: list[ReviewFrame] = []
    for f in frames:
        tk = by_t.get(f.t_mono)
        if tk is None:
            continue
        live = replace(
            f.live_aeb, aeb_warn=tk.aeb_warn, aeb_brake=tk.aeb_brake, engaged=tk.engaged,
            target_decel_ms2=tk.target_decel_ms2, required_decel_ms2=tk.required_decel_ms2,
            time_to_brake=tk.time_to_brake, time_to_collision=tk.time_to_collision,
            colliding_ids=sorted(int(i) for i in tk.colliding_ids),
        )
        snap = replace(f.snapshot, colliding_ids={int(i) for i in tk.colliding_ids})
        out.append(replace(f, snapshot=snap, live_aeb=live))
    return out


def aeb_state(live) -> int:
    """0 standby, 1 warn, 2 brake, straight from the recorded tick."""
    return 2 if live.aeb_brake else (1 if live.aeb_warn else 0)


def auto_window(frames: list[ReviewFrame], *, lead_s: float = 3.5,
                tail_s: float = 2.0) -> tuple[float, float]:
    """Export span around the event: warn or brake if any, else tracked threats."""
    first, last = frames[0].t_rel, frames[-1].t_rel
    events = [f.t_rel for f in frames if f.live_aeb.aeb_warn or f.live_aeb.aeb_brake]
    if not events:
        events = [f.t_rel for f in frames if f.snapshot.colliding_ids]
    if not events:
        return first, last
    return max(first, events[0] - lead_s), min(last, events[-1] + tail_s)


class Timeline:
    """Smoothed, resampled view of one replayed clip."""

    def __init__(self, frames: list[ReviewFrame], *, ego_has_trailer: bool,
                 label_vid: int | None = None) -> None:
        if not frames:
            raise ValueError("clip replayed to no frames")
        self.frames = frames
        self.tick_ts = [f.t_rel for f in frames]
        self.primary_vid = pick_primary_threat(frames, label_vid)
        self.primary_span = threat_span(frames, self.primary_vid)
        first = frames[0].snapshot
        # Coordinates are re-based on the first ego pose so painting never sees 1e5 m values.
        self.origin = (first.ego_x, first.ego_z)
        self._build_ego()
        self._build_tracks()
        self._trailer = self._simulate_trailer() if ego_has_trailer else None
        self._threats = threat_ticks(frames)
        self._arc_cache: dict[tuple[int, int], list[Corridor]] = {}
        self.decisions = "recorded"
        self.changed_ticks = 0
        self.recorded_brake_t: float | None = None
        self.shown_brake_t: float | None = None

    @property
    def t_first(self) -> float:
        return self.tick_ts[0]

    @property
    def t_last(self) -> float:
        return self.tick_ts[-1]

    def _build_ego(self) -> None:
        ox, oz = self.origin
        self._ego = _Series()
        self._ego_speed: list[float] = []
        self._ego_curv: list[float] = []
        self._ego_horizon: list[float] = []
        for f in self.frames:
            s = f.snapshot
            self._ego.push(f.t_rel, s.ego_x - ox, s.ego_z - oz, s.ego_yaw)
            self._ego_speed.append(s.ego_speed)
            arc = s.ego_arc
            self._ego_curv.append(arc.curvature if arc is not None else 0.0)
            self._ego_horizon.append(arc.horizon if arc is not None else 0.0)
        self.ego_half_w = first_or(self.frames, "ego_half_w", _CAL.ego_half_width)
        self.ego_half_l = first_or(self.frames, "ego_half_l", _CAL.ego_half_length)

    def _build_tracks(self) -> None:
        ox, oz = self.origin
        self._tracks: list[_Track] = []
        self._present: list[list[int]] = []
        open_tracks: dict[int, int] = {}
        for f in self.frames:
            here: list[int] = []
            for v in f.snapshot.vehicles:
                vid = int(v["vid"])
                x, z = v["x"] - ox, v["z"] - oz
                idx = open_tracks.get(vid)
                if idx is not None and not self._continues(self._tracks[idx], f.t_rel, x, z):
                    idx = None
                if idx is None:
                    idx = len(self._tracks)
                    self._tracks.append(_Track(vid, _Series(), v["half_w"], v["length"]))
                    open_tracks[vid] = idx
                track = self._tracks[idx]
                track.series.push(f.t_rel, x, z, v["yaw"])
                for k, tr in enumerate(v.get("trailers", [])):
                    if k >= len(track.trailers):
                        track.trailers.append((_Series(), tr["half_w"], tr["length"]))
                    track.trailers[k][0].push(f.t_rel, tr["x"] - ox, tr["z"] - oz, tr["yaw"])
                here.append(idx)
            self._present.append(here)

    @staticmethod
    def _continues(track: _Track, t: float, x: float, z: float) -> bool:
        dt = t - track.t1
        jump = math.hypot(x - track.series.xs[-1], z - track.series.zs[-1])
        return dt <= TRACK_GAP_S and jump <= TRACK_JUMP_M + TRACK_MAX_SPEED_MS * dt

    def ego_body(self, t: float) -> Body:
        x, z, yaw = self._ego.pose(t, POSE_SIGMA_S)
        return Body(x, z, yaw, self.ego_half_w, 2.0 * self.ego_half_l)

    def ego_speed(self, t: float) -> float:
        t = min(max(t, self.t_first), self.t_last)
        return local_linear(self.tick_ts, self._ego_speed, t, SPEED_SIGMA_S)

    def _kingpin(self, t: float) -> tuple[float, float]:
        b = self.ego_body(t)
        return (b.x + math.sin(b.yaw) * KINGPIN_BACK_M,
                b.z + math.cos(b.yaw) * KINGPIN_BACK_M)

    def _simulate_trailer(self) -> tuple[list[float], list[float], list[float]]:
        """Trailer axle as a follower of the kingpin, integrated from the clip start."""
        wheelbase = TRAILER_LEN_M - KINGPIN_INSET_M - AXLE_FROM_REAR_M
        b0 = self.ego_body(self.t_first)
        kx, kz = self._kingpin(self.t_first)
        ax, az = kx + math.sin(b0.yaw) * wheelbase, kz + math.cos(b0.yaw) * wheelbase
        ts, axs, azs = [], [], []
        n = int((self.t_last - self.t_first) / TRAILER_SIM_DT_S) + 2
        for i in range(n):
            t = self.t_first + i * TRAILER_SIM_DT_S
            kx, kz = self._kingpin(t)
            dx, dz = ax - kx, az - kz
            d = math.hypot(dx, dz)
            if d > 1e-6:
                ax, az = kx + dx / d * wheelbase, kz + dz / d * wheelbase
            ts.append(t)
            axs.append(ax)
            azs.append(az)
        return ts, axs, azs

    def ego_trailer_body(self, t: float) -> Body | None:
        if self._trailer is None:
            return None
        ts, axs, azs = self._trailer
        i = min(max(bisect_left(ts, t), 1), len(ts) - 1)
        f = min(max((t - ts[i - 1]) / (ts[i] - ts[i - 1]), 0.0), 1.0)
        ax = axs[i - 1] + (axs[i] - axs[i - 1]) * f
        az = azs[i - 1] + (azs[i] - azs[i - 1]) * f
        kx, kz = self._kingpin(t)
        yaw = math.atan2(-(kx - ax), -(kz - az))
        back = TRAILER_LEN_M / 2.0 - KINGPIN_INSET_M
        return Body(kx + math.sin(yaw) * back, kz + math.cos(yaw) * back,
                    yaw, TRAILER_HALF_W_M, TRAILER_LEN_M)

    def ego_corridor(self, t: float, body: Body) -> Corridor | None:
        tc = min(max(t, self.t_first), self.t_last)
        horizon = local_linear(self.tick_ts, self._ego_horizon, tc, ARC_SIGMA_S)
        if horizon <= 0.0:
            return None
        curv = local_linear(self.tick_ts, self._ego_curv, tc, ARC_SIGMA_S)
        off = (_CAL.arc_start_pctg - 0.5) * body.length
        fx, fz = -math.sin(body.yaw), -math.cos(body.yaw)
        arc = build_arc(body.x + off * fx, body.z + off * fz, body.yaw,
                        self.ego_speed(t), curv, body.half_w, horizon)
        return arc.sample_corridor(CORRIDOR_SAMPLES)

    def nearest_tick(self, t: float) -> int:
        i = bisect_left(self.tick_ts, t)
        if i >= len(self.tick_ts):
            return len(self.tick_ts) - 1
        if i > 0 and t - self.tick_ts[i - 1] < self.tick_ts[i] - t:
            return i - 1
        return i

    def _vehicle_view(self, idx: int, t: float, threat: bool) -> VehicleView:
        track = self._tracks[idx]
        x, z, yaw = track.series.pose(t, POSE_SIGMA_S)
        trailers = [
            Body(*series.pose(t, POSE_SIGMA_S), hw, length)
            for series, hw, length in track.trailers
            if series.ts[0] - 1e-6 <= t <= series.ts[-1] + TRACK_GAP_S
        ]
        return VehicleView(track.vid, Body(x, z, yaw, track.half_w, track.length),
                           trailers, threat)

    def _corridors(self, tick: int, view: VehicleView) -> list[Corridor]:
        """Recorded predicted corridors, carried rigidly from the tick pose to the smoothed one."""
        vid = view.vid
        src = tick
        arcs = self.frames[tick].snapshot.vehicle_arcs.get(vid)
        if arcs is None:
            for j in range(tick - 1, -1, -1):
                if self.tick_ts[tick] - self.tick_ts[j] > ARC_HOLD_S:
                    break
                arcs = self.frames[j].snapshot.vehicle_arcs.get(vid)
                if arcs is not None:
                    src = j
                    break
        if arcs is None:
            return []
        raw = next((v for v in self.frames[src].snapshot.vehicles if int(v["vid"]) == vid), None)
        if raw is None:
            return []
        key = (src, vid)
        if key not in self._arc_cache:
            ox, oz = self.origin
            arc_list = arcs if isinstance(arcs, list) else [arcs]
            self._arc_cache[key] = [
                ([(x - ox, z - oz) for x, z in left], [(x - ox, z - oz) for x, z in right])
                for left, right in (a.sample_corridor(CORRIDOR_SAMPLES) for a in arc_list)
            ]
        px, pz = raw["x"] - self.origin[0], raw["z"] - self.origin[1]
        rot = raw["yaw"] - view.body.yaw
        c, s = math.cos(rot), math.sin(rot)
        bx, bz = view.body.x, view.body.z

        def move(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
            return [(bx + (x - px) * c - (z - pz) * s, bz + (x - px) * s + (z - pz) * c)
                    for x, z in pts]

        return [(move(left), move(right)) for left, right in self._arc_cache[key]]

    def states(self, times: list[float], *, corridors: bool = True) -> list[FrameState]:
        """One state per output time. Discrete signals come from the nearest tick, no easing."""
        out: list[FrameState] = []
        for t in times:
            tick = self.nearest_tick(t)
            threats = self._threats[tick]
            vehicles = [
                self._vehicle_view(idx, t, self._tracks[idx].vid in threats)
                for idx in self._present[tick]
            ]
            paths = {}
            for v in vehicles if corridors else ():
                found = [c for c in self._corridors(tick, v) if len(c[0]) >= 2]
                if found:
                    paths[v.vid] = found
            ego = self.ego_body(t)
            out.append(FrameState(
                t=t, ego=ego, ego_trailer=self.ego_trailer_body(t),
                ego_speed_ms=self.ego_speed(t), ego_corridor=self.ego_corridor(t, ego),
                vehicles=vehicles, corridors=paths,
                state=aeb_state(self.frames[tick].live_aeb), primary_vid=self.primary_vid,
            ))
        return out


def first_or(frames: list[ReviewFrame], attr: str, default: float) -> float:
    value = getattr(frames[0].snapshot, attr, None)
    return float(value) if value else default
