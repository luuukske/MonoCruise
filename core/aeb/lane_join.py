"""Lane join: a mover joining the lane beside ego's from further out, able to stop on its centre, is let go.

Its sideways speed comes from its own position track against that lane, never its curvature.
The last stage. No map, no verdict. See ``core/aeb/README.md``, "Lane join".
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from core.aeb.calibration import AEBCalibration
from core.aeb.filters import _PASS, FilterResult, _suppress
from core.aeb.lane_keep import LaneKeepFilter
from core.aeb.lane_track import LaneTrackHold
from core.aeb.target_follow import TargetFollowFilter
from core.road_layout.road_bend import _curve_heading, fast_offset

if TYPE_CHECKING:
    from core.aeb.filters import FilterContext
    from core.aeb.road_paths import RoadPaths

# A track restarts after a gap this long, or a step across the lane this large between frames.
_MAX_GAP_S = 0.3
_MAX_STEP_M = 1.0


def lane_join_frame(road_paths: "RoadPaths", x: float, z: float,
                    cal: AEBCalibration) -> list | None:
    """``[ego's lane, ego s on it, side -> the lane beside it, side ego moves to or None, ego right]``.

    Ego's lane is the one it keeps, or, moving across at ``lane_join_change_drift_ms`` or more,
    the one it can still settle in at ``lane_join_ego_settle_ms2``. None otherwise.
    """
    keep = road_paths.keep_lane
    if keep is None or not road_paths.active:
        return None
    curve, keep_s, ego_right, _reach, drift, heading, accel = keep[:7]
    moving = None
    # Below the bar ego keeps its own lane: a lane change bound at 0.2 m/s flipped the lane read
    # every few frames and restarted each body's hold (478d7717).
    if abs(drift) >= cal.lane_join_change_drift_ms:
        if abs(heading) > math.radians(cal.lane_change_max_heading_deg):
            return None
        moving = 1.0 if drift > 0.0 else -1.0
        # Short of its own lane's centre it settles there; past it, in the next lane on that side.
        lane, remaining = curve, -moving * ego_right
        if remaining <= 0.0:
            lane = road_paths.neighbour_lane(curve, keep_s, moving, cal)
            if lane is None:
                return None
            remaining = -moving * fast_offset(lane, x, z)[1]
        # Its own settle test, not lane change clear's 1 m/s^2: ego crossing at 3.5 m/s (ffb1688a),
        # allowed to settle a little past the centre toward the lane it moves to.
        if drift * drift > 2.0 * cal.lane_join_ego_settle_ms2 * (remaining + cal.lane_join_ego_overshoot_m):
            return None
        # Near that centre a lane change eases off: still pushing toward the side, ego swerves on (7f002208).
        if remaining < cal.lane_join_ego_push_m and moving * accel > cal.lane_join_ego_push_ms2:
            return None
    elif (abs(ego_right) <= cal.lane_keep_max_ego_offset_m
          and abs(heading) <= math.radians(cal.lane_keep_max_ego_heading_deg)):
        lane = curve
    else:
        return None
    ego_s, right = fast_offset(lane, x, z)
    return [lane, ego_s, {}, moving, right] if math.isfinite(ego_s) else None


def lane_join_sight(road_paths: "RoadPaths", frame: list, v, yaw_rad: float, travel: float,
                    ego_hw: float, cal: AEBCalibration) -> tuple | None:
    """``(side, metres beyond the next lane's centre, metres ahead, cosine along it, lane spacing,
    ego changing lanes toward it, metres it may still come past that centre)``. The next lane is
    the one beside ego's lane on the body's side.
    """
    lane, ego_s, nexts, moving, ego_right = frame
    x, z = v.position.x, v.position.z
    s, right = fast_offset(lane, x, z)
    ahead = s - ego_s
    if not math.isfinite(right) or not 0.0 < ahead <= cal.lane_join_range_m:
        return None
    side = 1.0 if right > 0.0 else -1.0
    # While ego changes lanes only the side it moves to is judged: the lane it leaves is not
    # beside it yet (b13f5c5c braked for its own lead).
    if moving is not None and side != moving:
        return None
    if side not in nexts:
        nexts[side] = road_paths.neighbour_lane(lane, ego_s, side, cal, ahead=0.0)
    nxt = nexts[side]
    if nxt is None:
        return None
    s_n, right_n = fast_offset(nxt, x, z)
    h = _curve_heading(nxt, s_n) if 0.0 <= s_n <= nxt.length else None
    if not math.isfinite(right_n) or h is None:
        return None
    # Where the next lane has merged into ego's, the body is in ego's lane (6457d079).
    spacing = abs(right - right_n)
    if spacing < cal.lane_change_lane_min_m:
        return None
    along = travel * (-math.sin(yaw_rad) * math.cos(h) - math.cos(yaw_rad) * math.sin(h))
    # Its body must stop lane_join_clear_m short of where ego's ends up: ego's offset toward it,
    # or past its lane's centre by the overshoot while moving into it.
    ego_toward = cal.lane_join_ego_overshoot_m if moving is not None else max(side * ego_right, 0.0)
    stop = spacing - ego_toward - ego_hw - 0.5 * v.size.width - cal.lane_join_clear_m
    return side, side * right_n, ahead, along, spacing, moving is not None, stop


def map_lane_stages(cal: AEBCalibration) -> list:
    """The map lane rules that run last, in order: lane keep, target follow, lane join."""
    return [LaneKeepFilter(cal), TargetFollowFilter(cal), LaneJoinFilter(cal)]


def _slope(track) -> float:
    """Least-squares rate of ``(t, metres)`` samples over the window, metres per second."""
    n = len(track)
    mt = sum(t for t, _ in track) / n
    my = sum(y for _, y in track) / n
    var = sum((t - mt) ** 2 for t, _ in track)
    return sum((t - mt) * (y - my) for t, y in track) / var if var > 1e-9 else 0.0


def _fit(track, now: float) -> tuple[float, float]:
    """Rate and acceleration now of ``(t, metres)`` samples: a least-squares quadratic."""
    sums = [0.0] * 5
    rhs = [0.0] * 3
    for t, y in track:
        u = t - now
        p = 1.0
        for k in range(5):
            if k < 3:
                rhs[k] += p * y
            sums[k] += p
            p *= u
    a = [[sums[i + j] for j in range(3)] for i in range(3)]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
           - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    if abs(det) <= 1e-12:
        return 0.0, 0.0
    c = []
    for col in (1, 2):
        m = [row[:] for row in a]
        for i in range(3):
            m[i][col] = rhs[i]
        c.append((m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                  - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / det)
    return c[0], 2.0 * c[1]


def settles(beyond: float, rate: float, cal: AEBCalibration, accel: float = 0.0) -> bool:
    """Moving toward ego's lane at ``rate``, ``beyond`` metres from where it must stop, it still can:
    slower than the bypass, and the lateral decel that needs under the bar."""
    if accel > 0.0:
        # Still pushing toward ego, it is judged where its push leaves it a reaction later: a swerve
        # (02e3a612, 0 to 2 m/s across in 0.2 s) is no lane change.
        lead = cal.lane_join_accel_lead_s
        beyond -= rate * lead + 0.5 * accel * lead * lead
        rate += accel * lead
    room = beyond
    if room <= 0.0 or rate >= cal.lane_join_max_rate_ms:
        return False
    return rate <= 0.0 or rate * rate / (2.0 * room) < cal.lane_join_max_settle_ms2


class LaneJoinTracker:
    """Seconds each body has been able to settle in the lane beside ego's, without a break."""

    def __init__(self) -> None:
        # vid -> (side, deque of (t, metres beyond the next lane's centre)).
        self._tracks: dict[int, tuple[float, deque]] = {}
        # vid -> (side, last time its centre sat beyond the next lane's far edge).
        self._far: dict[int, tuple[float, float]] = {}
        # vid -> (time, metres beyond) of its last reading, kept across frames with no sight.
        self._last: dict[int, tuple[float, float]] = {}
        self._hold = LaneTrackHold()

    def step(self, vid: int, sight, speed: float, now: float, cal: AEBCalibration,
             accel: float = 0.0) -> float:
        """``sight`` is ``lane_join_sight``'s answer for this frame, or None; ``accel`` along travel."""
        return self._hold.step(vid, self._settles(vid, sight, speed, now, cal, accel), now)

    def _settles(self, vid: int, sight, speed: float, now: float, cal: AEBCalibration,
                 accel: float = 0.0) -> bool:
        if sight is None:
            self._tracks.pop(vid, None)
            return False
        side, beyond, _ahead, along, spacing, changing, stop = sight
        held = self._tracks.get(vid)
        stepped = held is not None and abs(beyond - held[1][-1][1]) > _MAX_STEP_M
        if held is None or held[0] != side or now - held[1][-1][0] > _MAX_GAP_S or stepped:
            held = self._tracks[vid] = (side, deque())
        # A jump further than it could move, across a gap too, is another lane read (ffb1688a): the
        # origin is unknown, as it is after a change of side.
        last = self._last.get(vid)
        jumped = last is not None and abs(beyond - last[1]) > (
            _MAX_STEP_M + cal.lane_join_max_rate_ms * max(now - last[0], 0.0))
        self._last[vid] = (now, beyond)
        if stepped or jumped or self._far.get(vid, (side,))[0] != side:
            self._far.pop(vid, None)
        if beyond > 0.5 * spacing:
            self._far[vid] = (side, now)
        track = held[1]
        track.append((now, beyond))
        while now - track[0][0] > cal.lane_join_fit_s:
            track.popleft()
        # Only a body joining the next lane from beyond its far edge: one already in it is a cut-in.
        # While ego moves into its lane, any body beyond the next lane's centre is outside it (ffb1688a).
        far = self._far.get(vid)
        if not changing and (far is None or now - far[1] > cal.lane_join_from_s):
            return False
        # Braking hard it is not changing lanes calmly (e0638fb2, 7.6 m/s^2 beside ego's lane).
        if (speed < cal.lane_join_min_speed_ms or along < cal.lane_join_min_along
                or accel < -cal.lane_join_max_brake_ms2):
            return False
        if len(track) < 4 or now - track[0][0] < 0.5 * cal.lane_join_fit_s:
            return False
        # Toward ego's lane is the distance beyond the next lane's centre shrinking.
        slope, curve = _fit(track, now) if cal.lane_join_rate_now else (_slope(track), 0.0)
        # Nor crossing steeply: 3.5 m/s across at 6 m/s along (39d96205).
        if beyond <= 0.0 or -slope > cal.lane_join_max_cross_ratio * speed:
            return False
        return settles(beyond + max(stop, 0.0), -slope, cal, -curve)

    def prune(self, live_ids) -> None:
        for held in (self._tracks, self._far, self._last):
            for vid in list(held):
                if vid not in live_ids:
                    del held[vid]
        self._hold.prune(live_ids)

    def clear(self) -> None:
        self._tracks.clear()
        self._far.clear()
        self._last.clear()
        self._hold.prune(())


class LaneJoinFilter:
    name = "LaneJoinFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: "FilterContext") -> FilterResult:
        cal = self._cal
        if not cal.lane_join_enabled or ctx.lane_join_s < cal.lane_join_hold_s:
            return _PASS
        # A body AEB already brakes for is never let go on the map's word.
        if ctx.v.id in ctx.latched_threat_ids or getattr(ctx.v, "crash_confirmed", False):
            return _PASS
        return _PASS if ctx.head_on else _suppress(self.name)
