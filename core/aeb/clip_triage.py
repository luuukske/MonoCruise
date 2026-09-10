"""Pre-upload triage: does a contributed clip still carry information.

Runs on the uploader thread, never on a control loop, and judges a clip from its
own recorded streams rather than from the build that is running now. See
``core/aeb/README.md`` section 15 for the measured rates behind every threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from core.aeb.calibration import DEFAULT as _CAL
from core.aeb.clip_schema import Clip

logger = logging.getLogger(__name__)

# Above this a recorded time-to-collision means "nothing was being tracked",
# matching the sentinel the AEB thread publishes when it sees no threat.
TTC_NONE_S: float = 100.0

# A target that never came this close was never a candidate for anything.
NEAR_RANGE_M: float = 60.0
# Paired with "AEB never braked": far in time and unacted on is not an event.
QUIET_TTC_S: float = 4.0
# Rear-end geometry below this is the corpus's most repeated scene, and the
# filtering it exercises is already covered many times over.
STRAIGHT_MAX_KMH: float = 40.0
# Send one in this many of the straight sub-40 clips anyway, so a regression in
# that class still shows up between versions.
STRAIGHT_SAMPLE_EVERY: int = 10
# Below this a target is parked or stopped, not a co-directional lead.
MOVING_MIN_KMH: float = 2.0
# Fractions of a target's tracked ticks: ahead of ego, and inside the lane band.
AHEAD_FRAC_MIN: float = 0.5
IN_LANE_FRAC_MIN: float = 0.25


@dataclass
class _Track:
    """One vehicle across the clip, in the ego frame."""

    vid: int
    ticks: int = 0
    min_range_m: float = 1e9
    heading_dot_at_min: float = 0.0
    speed_max_kmh: float = 0.0
    ahead_ticks: int = 0
    in_lane_ticks: int = 0
    corridor_ticks: int = 0
    min_corridor_gap_m: float = 1e9
    colliding_ticks: int = 0


@dataclass
class SceneSummary:
    """The scalars the triage rules read. Everything else about a clip is ignored."""

    n_vehicles: int = 0
    nearest_range_m: float = 1e9
    min_ttc_s: float = TTC_NONE_S
    brake_ticks: int = 0
    ego_speed_at_action_kmh: float = 0.0
    straight_in_lane: bool = False
    # False when the radar stream could not be replayed. Every geometry rule is
    # skipped in that case, so a decode failure can never refuse a clip.
    decoded: bool = False


def _world_to_ego(wx: float, wz: float, ex: float, ez: float,
                  ego_yaw: float) -> tuple[float, float]:
    """(forward, right) metres in the ego frame. Mirrors ``_w2e`` in debug_window."""
    dx = wx - ex
    dz = wz - ez
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    rx = (-dx) * c - dz * s
    rz = (-dx) * s + dz * c
    return -rz, -rx


def _veh_yaw(v) -> float:
    smooth = getattr(v, "_smooth_yaw", None)
    if smooth is not None:
        return smooth
    return math.radians(v.rotation.euler()[1])


def _live_scalars(clip: Clip) -> tuple[list, float, float, int, float]:
    """Sorted ticks, t0, min TTC, brake ticks and the action time, from live AEB only."""
    ticks = sorted(clip.aeb_ticks, key=lambda x: x.t_mono)
    all_t = [f.t_mono for f in clip.radar_frames] + [t.t_mono for t in ticks]
    t0 = min(all_t) if all_t else 0.0
    min_ttc = TTC_NONE_S
    brake_ticks = 0
    first_warn: float | None = None
    first_brake: float | None = None
    tracked: list[float] = []
    for tk in ticks:
        la = tk.live_aeb
        t_rel = tk.t_mono - t0
        if la.aeb_warn and first_warn is None:
            first_warn = t_rel
        if la.aeb_brake:
            brake_ticks += 1
            if first_brake is None:
                first_brake = t_rel
        if la.time_to_collision < TTC_NONE_S:
            tracked.append(t_rel)
            min_ttc = min(min_ttc, la.time_to_collision)
    action_t = first_brake
    if action_t is None:
        action_t = first_warn
    if action_t is None:
        action_t = tracked[0] if tracked else 0.0
    return ticks, t0, min_ttc, brake_ticks, action_t


def _primary(tracks: dict[int, _Track]) -> _Track | None:
    """Closest genuine threat by geometry, mirroring the offline feature tool.

    Corridor targets win on gap; with none, the nearest body wins and colliding
    ticks only break a tie. Deliberately not "the id AEB flagged most".
    """
    if not tracks:
        return None
    corridor = [t for t in tracks.values() if t.corridor_ticks > 0]
    if corridor:
        return min(corridor, key=lambda t: (t.min_corridor_gap_m, -t.colliding_ticks))
    return min(tracks.values(), key=lambda t: (t.min_range_m, -t.colliding_ticks))


def _accumulate(tracks: dict[int, _Track], v, ego, ego_yaw: float, live) -> None:
    tr = tracks.get(v.id)
    if tr is None:
        tr = tracks[v.id] = _Track(vid=int(v.id))
    fwd, right = _world_to_ego(v.position.x, v.position.z,
                               ego.coordinateX, ego.coordinateZ, ego_yaw)
    rng = math.hypot(fwd, right)
    tr.ticks += 1
    if rng < tr.min_range_m:
        tr.min_range_m = rng
        tr.heading_dot_at_min = math.cos(ego_yaw - _veh_yaw(v))
    if fwd > 0.0:
        tr.ahead_ticks += 1
    if abs(right) <= _CAL.lane_half_width:
        tr.in_lane_ticks += 1
    tr.speed_max_kmh = max(tr.speed_max_kmh, abs(v.speed) * 3.6)

    corridor = _CAL.ego_half_width + v.size.width / 2.0 + _CAL.corridor_margin
    if abs(right) <= corridor and fwd > 0.0:
        tr.corridor_ticks += 1
        fwd_gap = fwd - _CAL.ego_half_length - v.size.length / 2.0
        tr.min_corridor_gap_m = min(tr.min_corridor_gap_m, fwd_gap)
    if v.id in live.colliding_ids:
        tr.colliding_ticks += 1


def summarize(clip: Clip) -> SceneSummary:
    """Fold a decoded clip down to the triage scalars. Never raises."""
    out = SceneSummary()
    ticks, t0, out.min_ttc_s, out.brake_ticks, action_t = _live_scalars(clip)
    if not ticks:
        return out

    try:
        from core.aeb.clip_replay import decode_radar_stream, nearest_frame_t

        veh_by_t, ego_by_t, frame_t, _off = decode_radar_stream(clip)
    except Exception:
        logger.debug("could not decode a clip for upload triage", exc_info=True)
        return out
    out.decoded = True

    tracks: dict[int, _Track] = {}
    seen_ego = False
    speed_at_action = 0.0
    best_dt = 1e9
    for tk in ticks:
        ft = nearest_frame_t(frame_t, tk.radar_t_mono)
        ego = ego_by_t.get(ft) if ft is not None else None
        if ego is None:
            continue
        seen_ego = True
        dt_action = abs((tk.t_mono - t0) - action_t)
        if dt_action < best_dt:
            best_dt = dt_action
            speed_at_action = ego.speed * 3.6
        ego_yaw = ego.rotationX * 2.0 * math.pi
        for v in veh_by_t.get(ft, []):
            _accumulate(tracks, v, ego, ego_yaw, tk.live_aeb)

    out.n_vehicles = len(tracks)
    out.ego_speed_at_action_kmh = speed_at_action if seen_ego else 0.0
    primary = _primary(tracks)
    if primary is None:
        return out
    out.nearest_range_m = primary.min_range_m
    seen = max(primary.ticks, 1)
    out.straight_in_lane = (
        primary.speed_max_kmh >= MOVING_MIN_KMH
        and primary.heading_dot_at_min >= _CAL.co_directional_dot
        and primary.ahead_ticks / seen >= AHEAD_FRAC_MIN
        and primary.in_lane_ticks / seen > IN_LANE_FRAC_MIN
    )
    return out


def triage_reason(summary: SceneSummary) -> str | None:
    """Why this clip carries nothing worth sending, or None when it does.

    The straight sub-40 class is not a reason here: it is sampled rather than
    refused, so ``is_straight_slow`` reports it separately.
    """
    if not summary.decoded:
        return None
    if summary.n_vehicles == 0:
        return "no traffic in the clip"
    if summary.nearest_range_m > NEAR_RANGE_M:
        return f"nothing came within {NEAR_RANGE_M:.0f} m"
    if summary.min_ttc_s >= QUIET_TTC_S and summary.brake_ticks == 0:
        return f"no approach inside {QUIET_TTC_S:.0f} s and no brake"
    return None


def is_straight_slow(summary: SceneSummary) -> bool:
    """True for the repeated rear-end-in-lane scene below ``STRAIGHT_MAX_KMH``."""
    return (
        summary.decoded
        and summary.straight_in_lane
        and summary.ego_speed_at_action_kmh < STRAIGHT_MAX_KMH
    )


def sample_keeps(position: int, every: int = STRAIGHT_SAMPLE_EVERY) -> bool:
    """True when this occurrence is the one kept, counting the first as kept."""
    if every <= 1:
        return True
    return position % every == 0
