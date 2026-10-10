"""Pre-upload triage: does a contributed clip still carry information.

Runs on the uploader thread, never on a control loop, and judges a clip from its
own recorded streams rather than from the build that is running now. See
``core/aeb/README.md`` section 15 for the measured rates behind every threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

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

# Scene classes, judged on who caused the encounter. See README section 15.
SCENE_OTHER_DRIVER: str = "other_driver"
SCENE_EGO_RECKLESS: str = "ego_reckless"
SCENE_STANDARD: str = "standard"
SCENE_UNCLASSIFIED: str = ""
# Standard braking at any speed, and ego-caused scenes, are sampled one in N.
STANDARD_SAMPLE_EVERY: int = 4
RECKLESS_SAMPLE_EVERY: int = 10
# Clips per local day after sampling. Other-driver scenes and crash clips are exempt:
# crash clips hold 11 of the 14 contributed misses.
DAILY_BUDGET: int = 25
# Above the TMP 110 cap and any truck limit; the clips past it were nearly all ignore.
RECKLESS_SPEED_KMH: float = 115.0
# Who moved: each vehicle's heading change over the window before a lane entry.
ATTRIBUTION_WINDOW_S: float = 3.0
MOVER_DEG: float = 2.5
EGO_HELD_DEG: float = 2.0
TARGET_HELD_DEG: float = 1.5
EGO_MOVED_DEG: float = 3.0
# Both headings turning together is the road bending, not anyone changing lane.
CURVE_REL_DEG: float = 2.0
# Past this either vehicle is turning at a junction, not holding a lane.
TURNING_DEG: float = 20.0
# Larger heading jumps inside the window are TMP rotation flips, not steering.
YAW_GLITCH_DEG: float = 150.0
# Lane entry: from beside the lane band to inside it, ahead and this close.
ENTRY_GAP_M: float = 40.0
ENTRY_LOOKAHEAD_M: float = 60.0
ENTRY_SEARCH_BEFORE_S: float = 5.0
ENTRY_SEARCH_AFTER_S: float = 0.5
ADJACENT_LAT_MARGIN_M: float = 0.9
IN_LANE_LAT_MARGIN_M: float = 0.3
# Another driver's move in front of a crawling or parked truck threatens nothing.
OTHER_DRIVER_EGO_MIN_KMH: float = 20.0
CUT_IN_TARGET_MIN_KMH: float = 5.0
# Ego changing lane into less than this much time gap.
DIVE_IN_HEADWAY_S: float = 1.0
# Range kept for per-target history; nothing further out can enter the lane.
HISTORY_RANGE_M: float = 120.0


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
    reversing: bool = False
    length_m: float = 0.0
    # (t, fwd, right, world yaw, speed m/s), for lane-entry attribution.
    history: list = field(default_factory=list)


@dataclass
class SceneSummary:
    """The scalars the triage rules read. Everything else about a clip is ignored."""

    n_vehicles: int = 0
    nearest_range_m: float = 1e9
    min_ttc_s: float = TTC_NONE_S
    brake_ticks: int = 0
    ego_speed_at_action_kmh: float = 0.0
    straight_in_lane: bool = False
    # One of the SCENE_* names, and the short reason behind it.
    scene: str = SCENE_UNCLASSIFIED
    scene_why: str = ""
    crash_trigger: bool = False
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


def _accumulate(tracks: dict[int, _Track], v, ego, ego_yaw: float, live,
                t_rel: float = 0.0) -> None:
    tr = tracks.get(v.id)
    if tr is None:
        tr = tracks[v.id] = _Track(vid=int(v.id))
    fwd, right = _world_to_ego(v.position.x, v.position.z,
                               ego.coordinateX, ego.coordinateZ, ego_yaw)
    rng = math.hypot(fwd, right)
    if rng <= HISTORY_RANGE_M:
        tr.history.append((t_rel, fwd, right, _veh_yaw(v), v.speed))
    tr.length_m = v.size.length
    if v.speed < -_CAL.reversing_speed_ms:
        tr.reversing = True
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


def _deg(rad: float) -> float:
    """An angle in radians, wrapped to +-pi, in degrees."""
    return math.degrees((rad + math.pi) % (2.0 * math.pi) - math.pi)


def _heading_devs(hist: list, ego_yaw_by_t: dict, t_from: float,
                  t_to: float) -> tuple[float, float, float] | None:
    """Peak heading change of the target, of ego, and of target relative to ego."""
    seg = [h for h in hist if t_from <= h[0] <= t_to and h[0] in ego_yaw_by_t]
    if len(seg) < 2:
        return None
    t_yaw0 = seg[0][3]
    e_yaw0 = ego_yaw_by_t[seg[0][0]]
    dev_t = dev_e = rel = 0.0
    for h in seg:
        d_t = h[3] - t_yaw0
        d_e = ego_yaw_by_t[h[0]] - e_yaw0
        dev_t = max(dev_t, abs(_deg(d_t)))
        dev_e = max(dev_e, abs(_deg(d_e)))
        rel = max(rel, abs(_deg(d_t - d_e)))
    return dev_t, dev_e, rel


def _who_moved(devs: tuple[float, float, float] | None) -> str:
    """"target", "ego", or "" when neither clearly caused the lateral change."""
    if devs is None:
        return ""
    dev_t, dev_e, rel = devs
    if dev_t > YAW_GLITCH_DEG or rel < CURVE_REL_DEG:
        return ""
    if dev_t >= MOVER_DEG and dev_e < EGO_HELD_DEG:
        return "target"
    if dev_e >= EGO_MOVED_DEG and dev_t < TARGET_HELD_DEG:
        return "ego"
    return ""


@dataclass
class _LaneEntry:
    vid: int
    t: float
    gap_m: float
    ego_kmh: float
    target_kmh: float
    mover: str


def _lane_entries(tracks: dict[int, _Track], ego_by_t: dict,
                  action_t: float) -> list[_LaneEntry]:
    """Vehicles that crossed from beside ego's lane into it, ahead, near the action."""
    in_lane = _CAL.lane_half_width - IN_LANE_LAT_MARGIN_M
    adjacent = _CAL.lane_half_width + ADJACENT_LAT_MARGIN_M
    yaw_by_t = {t: e[0] for t, e in ego_by_t.items()}
    out: list[_LaneEntry] = []
    for tr in tracks.values():
        hist = tr.history
        for i in range(1, len(hist)):
            t, fwd, right = hist[i][0], hist[i][1], hist[i][2]
            if not (action_t - ENTRY_SEARCH_BEFORE_S <= t <= action_t + ENTRY_SEARCH_AFTER_S):
                continue
            if abs(right) > in_lane or not (0.0 < fwd <= ENTRY_LOOKAHEAD_M):
                continue
            if abs(hist[i - 1][2]) <= in_lane:
                continue
            prior = [h for h in hist[:i] if t - ATTRIBUTION_WINDOW_S <= h[0] < t]
            if not prior or max(abs(h[2]) for h in prior) < adjacent:
                continue
            ego = ego_by_t.get(t)
            if ego is None:
                break
            out.append(_LaneEntry(
                vid=tr.vid, t=t,
                gap_m=fwd - _CAL.ego_half_length - tr.length_m / 2.0,
                ego_kmh=ego[1] * 3.6, target_kmh=abs(hist[i][4]) * 3.6,
                mover=_who_moved(_heading_devs(hist, yaw_by_t, prior[0][0], t)),
            ))
            break
    return out


def _kind(tr: _Track) -> str:
    """Relative motion at closest approach, as the offline feature tool names it."""
    if tr.speed_max_kmh < MOVING_MIN_KMH:
        return "stationary"
    if tr.heading_dot_at_min >= _CAL.co_directional_dot:
        return "codirectional"
    if tr.heading_dot_at_min <= -_CAL.co_directional_dot:
        return "oncoming"
    return "crossing"


def _classify_scene(tracks: dict[int, _Track], primary: _Track | None,
                    ego_history: list, action_t: float,
                    ego_kmh: float) -> tuple[str, str]:
    """Who caused the encounter: another driver, ego, or nobody unusual."""
    if ego_kmh >= RECKLESS_SPEED_KMH:
        return SCENE_EGO_RECKLESS, f"ego above {RECKLESS_SPEED_KMH:.0f} km/h"
    ego_by_t = {h[0]: (h[1], h[2]) for h in ego_history}
    entries = _lane_entries(tracks, ego_by_t, action_t)
    close = [e for e in entries if e.gap_m <= ENTRY_GAP_M]
    for e in close:
        if (e.mover == "target" and e.gap_m >= 0.0
                and e.ego_kmh >= OTHER_DRIVER_EGO_MIN_KMH
                and e.target_kmh >= CUT_IN_TARGET_MIN_KMH):
            return SCENE_OTHER_DRIVER, "cut-in"
    if primary is None:
        return SCENE_UNCLASSIFIED, ""

    kind = _kind(primary)
    entered = next((e for e in entries if e.vid == primary.vid), None)
    t_ref = entered.t if entered is not None else action_t
    yaw_by_t = {t: e[0] for t, e in ego_by_t.items()}
    devs = _heading_devs(primary.history, yaw_by_t, t_ref - ATTRIBUTION_WINDOW_S, t_ref)
    dev_t, dev_e, rel = devs if devs is not None else (0.0, 0.0, 0.0)
    if dev_t > YAW_GLITCH_DEG:
        dev_t = 0.0
    moving = ego_kmh >= OTHER_DRIVER_EGO_MIN_KMH
    if moving and primary.reversing:
        return SCENE_OTHER_DRIVER, "reversing target"
    if moving and kind == "crossing" and dev_e < TURNING_DEG:
        return SCENE_OTHER_DRIVER, "crossing target"
    if moving and kind in ("codirectional", "crossing") and dev_t >= TURNING_DEG \
            and dev_e < TURNING_DEG / 2.0:
        return SCENE_OTHER_DRIVER, "target turning across"
    if kind == "oncoming":
        mover = _who_moved(devs)
        if moving and mover == "target":
            return SCENE_OTHER_DRIVER, "oncoming drifting in"
        if mover == "ego":
            return SCENE_EGO_RECKLESS, "ego in the oncoming lane"

    dives = [e for e in close if e.mover == "ego"]
    if dives:
        dive = min(dives, key=lambda e: e.gap_m)
        if dive.gap_m / max(dive.ego_kmh / 3.6, 1.0) < DIVE_IN_HEADWAY_S:
            return SCENE_EGO_RECKLESS, "ego changed lane into a short gap"

    seen = max(primary.ticks, 1)
    in_lane_ahead = (primary.ahead_ticks / seen >= AHEAD_FRAC_MIN
                     and primary.in_lane_ticks / seen > IN_LANE_FRAC_MIN)
    if in_lane_ahead and entered is None:
        if kind == "stationary" and dev_e < TURNING_DEG:
            return SCENE_STANDARD, "stopped vehicle in lane"
        if kind == "codirectional" and rel < MOVER_DEG:
            return SCENE_STANDARD, "lead in lane"
    return SCENE_UNCLASSIFIED, ""


def summarize(clip: Clip) -> SceneSummary:
    """Fold a decoded clip down to the triage scalars. Never raises."""
    out = SceneSummary()
    out.crash_trigger = clip.metadata.trigger_source == "auto_crash"
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
    ego_history: list[tuple[float, float, float]] = []
    seen_ego = False
    speed_at_action = 0.0
    best_dt = 1e9
    for tk in ticks:
        ft = nearest_frame_t(frame_t, tk.radar_t_mono)
        ego = ego_by_t.get(ft) if ft is not None else None
        if ego is None:
            continue
        seen_ego = True
        t_rel = tk.t_mono - t0
        dt_action = abs(t_rel - action_t)
        if dt_action < best_dt:
            best_dt = dt_action
            speed_at_action = ego.speed * 3.6
        ego_yaw = ego.rotationX * 2.0 * math.pi
        ego_history.append((t_rel, ego_yaw, ego.speed))
        for v in veh_by_t.get(ft, []):
            _accumulate(tracks, v, ego, ego_yaw, tk.live_aeb, t_rel)

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
    try:
        out.scene, out.scene_why = _classify_scene(
            tracks, primary, ego_history, action_t, out.ego_speed_at_action_kmh)
    except Exception:
        logger.debug("could not classify a clip for upload triage", exc_info=True)
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


def sample_class(summary: SceneSummary) -> tuple[str, int] | None:
    """The sampled class this clip falls in and its one-in-N, or None when it always goes.

    Other-driver scenes are never sampled and never count toward the daily budget.
    """
    if not summary.decoded or summary.scene == SCENE_OTHER_DRIVER:
        return None
    if summary.scene == SCENE_EGO_RECKLESS:
        return "reckless", RECKLESS_SAMPLE_EVERY
    if is_straight_slow(summary):
        return "straight", STRAIGHT_SAMPLE_EVERY
    if summary.scene == SCENE_STANDARD:
        return "standard", STANDARD_SAMPLE_EVERY
    return None


def is_priority(summary: SceneSummary) -> bool:
    """Another driver caused the encounter: always sent, outside the daily budget."""
    return summary.decoded and summary.scene == SCENE_OTHER_DRIVER


def counts_toward_budget(summary: SceneSummary) -> bool:
    """False for the classes the daily budget must never drop, and for undecoded clips."""
    return summary.decoded and not (is_priority(summary) or summary.crash_trigger)
