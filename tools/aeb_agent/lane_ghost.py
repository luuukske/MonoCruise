"""The no-intervention ghost, driven along ego's map lane instead of a held yaw rate.

`evasion.py` flies the ghost on the heading rate it held at the fork, which leaves
the road on any bend within a second or two. Here the ghost follows ego's lane
route at its held offset, at the speed the mode keeps. Tagging evidence only: it
needs the local map and never runs in the app.

`measure` replays this against ego's real path on clips with no intervention, so
the error bar is measured the same way as the constant-rate one in the README.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from core.aeb.calibration import DEFAULT as _CAL

from tools.aeb_agent.context import _Sample, _vdict, _yaw, body_gap, cumulative, project

# Follow the lane only while ego is on it at the fork, like the map path's full trust.
MAX_OFFSET_M = 1.2
MAX_HEADING_DEG = 8.0
HORIZON_S = 3.0
DRIFT_AT_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
CLOSING_EPS = 0.02
MIN_GHOST_MS = 4.0


@dataclass
class LaneGhost:
    """Ghost poses ``(t, x, z, yaw)`` from the fork, or why there are none."""

    ok: bool = False
    reason: str = ""
    poses: list = field(default_factory=list)
    stops_at_branch: bool = False


def _point(pts, cum, s: float):
    """``(x, z, tx, tz)`` at arc length `s`, extended straight past either end."""
    j = max(1, min(bisect.bisect_left(cum, s), len(pts) - 1))
    ax, az = pts[j - 1]
    bx, bz = pts[j]
    seg = math.hypot(bx - ax, bz - az)
    if seg < 1e-9:
        return ax, az, 0.0, -1.0
    tx, tz = (bx - ax) / seg, (bz - az) / seg
    a = s - cum[j - 1]
    return ax + tx * a, az + tz * a, tx, tz


def follow(route, x0: float, z0: float, times: list[float], speeds: list[float],
           horizon_s: float = HORIZON_S) -> LaneGhost:
    """Ghost along `route` from the fork pose, moving `speeds[k]` over each `times` step."""
    g = LaneGhost()
    if route is None or len(route.pts) < 2:
        g.reason = "no lane route at the fork"
        return g
    if abs(route.offset_m) > MAX_OFFSET_M or abs(math.degrees(route.heading_err_rad)) > MAX_HEADING_DEG:
        g.reason = "ego was not following its lane at the fork"
        return g
    pts = list(route.pts)
    cum = cumulative(pts)
    hit = project(pts, cum, x0, z0)
    if hit is None:
        g.reason = "fork pose does not project on the route"
        return g
    s, offset = hit[0], hit[1]
    limit = route.branch_s if route.branch_s is not None else route.reach_m
    t0 = times[0]
    for k, t in enumerate(times):
        if t - t0 > horizon_s:
            break
        if k:
            s += max(speeds[k - 1], 0.0) * (t - times[k - 1])
        if s > limit:
            g.stops_at_branch = route.branch_s is not None
            break
        x, z, tx, tz = _point(pts, cum, s)
        # Right of travel is (-tz, tx), the same convention as ego_route.
        gx, gz = x - tz * offset, z + tx * offset
        g.poses.append((t, gx, gz, math.atan2(-tx, -tz)))
    g.ok = len(g.poses) >= 2
    if not g.ok and not g.reason:
        g.reason = "route ends before the ghost moves"
    return g


def constant_rate(x0, z0, yaw0, omega, times, speeds, hold_yaw: bool, recorded_yaw=None,
                  horizon_s: float = HORIZON_S) -> list:
    """`evasion.run`'s ghost, for the side-by-side drift measurement."""
    out = []
    x, z, yaw = x0, z0, yaw0
    t0 = times[0]
    for k, t in enumerate(times):
        if t - t0 > horizon_s:
            break
        if k:
            h = t - times[k - 1]
            v = max(speeds[k - 1], 0.0)
            x += -math.sin(yaw) * v * h
            z += -math.cos(yaw) * v * h
            yaw = yaw + omega * h if hold_yaw else (recorded_yaw[k] if recorded_yaw else yaw)
        out.append((t, x, z, yaw))
    return out


def yaw_rate(samples: list[_Sample], k: int, span_s: float = 0.5) -> float:
    start = k
    while start > 0 and samples[k].t - samples[start].t < span_s:
        start -= 1
    if start >= k:
        return 0.0
    total = 0.0
    for a, b in zip(samples[start:k], samples[start + 1:k + 1]):
        total += (_yaw(b.ego) - _yaw(a.ego) + math.pi) % (2.0 * math.pi) - math.pi
    dt = samples[k].t - samples[start].t
    return total / dt if dt > 1e-6 else 0.0


def drift_rows(samples: list[_Sample], k_fork: int, route) -> list[dict]:
    """Ghost against ego's real path from one fork on a clip where nobody intervened."""
    fut = samples[k_fork:]
    times = [s.t for s in fut]
    rec = [s.ego.speed for s in fut]
    held = [fut[0].ego.speed] * len(fut)
    e0 = fut[0].ego
    if e0.speed < MIN_GHOST_MS:
        return []
    omega = yaw_rate(samples, k_fork)
    real = {round(s.t, 3): (s.ego.coordinateX, s.ego.coordinateZ) for s in fut}
    rec_yaw = [_yaw(s.ego) for s in fut]
    ghosts = {
        "const_heading": constant_rate(e0.coordinateX, e0.coordinateZ, _yaw(e0), omega, times, rec, True),
        "const_speed": constant_rate(e0.coordinateX, e0.coordinateZ, _yaw(e0), omega, times, held, False, rec_yaw),
        "const_both": constant_rate(e0.coordinateX, e0.coordinateZ, _yaw(e0), omega, times, held, True),
    }
    lane_rec = follow(route, e0.coordinateX, e0.coordinateZ, times, rec)
    lane_held = follow(route, e0.coordinateX, e0.coordinateZ, times, held)
    if lane_rec.ok:
        ghosts["lane_recorded_speed"] = lane_rec.poses
    if lane_held.ok:
        ghosts["lane_held_speed"] = lane_held.poses
    rows = []
    for name, poses in ghosts.items():
        for t, x, z, _yaw_g in poses:
            dt = t - times[0]
            target = next((d for d in DRIFT_AT_S if abs(dt - d) <= 0.02), None)
            if target is None:
                continue
            rx, rz = real[round(t, 3)]
            rows.append({"mode": name, "dt": target, "err": math.hypot(x - rx, z - rz)})
    return rows


FORK_BEFORE_ACTION_S = 3.5


def measure_path(path_str: str, action_t: float) -> list[dict]:
    """Worker: drift rows for one clip, forked early enough that AEB has not acted yet."""
    from pathlib import Path

    from core.aeb.clip_store import deserialize_clip
    from core.road_layout.ego_route import EgoRouter

    from tools.aeb_agent.context import _nearest, _timeline
    from tools.map_replay import match_clip

    try:
        clip = deserialize_clip(Path(path_str).read_bytes())
        samples = _timeline(clip)
        if len(samples) < 20 or action_t < FORK_BEFORE_ACTION_S:
            return []
        _game, window = match_clip(clip)
        if window is None:
            return []
        k_fork = _nearest(samples, action_t - FORK_BEFORE_ACTION_S)
        router = EgoRouter()
        route = None
        for s in samples[:k_fork + 1]:
            e = s.ego
            yaw = _yaw(e)
            route = router.step(window, e.coordinateX, e.coordinateY, e.coordinateZ,
                                -math.sin(yaw), -math.cos(yaw))
        rows = drift_rows(samples, k_fork, route)
        turn = abs(math.degrees(_yaw(samples[min(k_fork + 60, len(samples) - 1)].ego)
                                - _yaw(samples[k_fork].ego)))
        turn = min(turn, 360.0 - turn)
        cat = "straight" if turn < 5 else "gentle" if turn < 15 else "bend"
        for r in rows:
            r["cat"] = cat
        return rows
    except Exception:
        return []


# p90 drift against ego's real path on clips with no intervention (`python -m
# tools.aeb_agent drift`); README table "Lane-following ghost" has the source run.
LANE_DRIFT_P90 = {
    "lane_recorded_speed": ((0.0, 0.0), (0.5, 0.84), (1.0, 1.03), (1.5, 1.32),
                            (2.0, 1.47), (2.5, 1.81), (3.0, 2.00)),
    "lane_held_speed": ((0.0, 0.0), (0.5, 0.87), (1.0, 1.32), (1.5, 2.27),
                        (2.0, 3.41), (2.5, 4.81), (3.0, 6.64)),
}
# Credible while the drift band stays inside the 2.74 m the constant-rate ghost allows at 1.5 s.
CREDIBLE_DRIFT_M = 2.75


def lane_drift(dt: float, mode: str) -> float:
    table = LANE_DRIFT_P90.get(mode, LANE_DRIFT_P90["lane_held_speed"])
    prev_t, prev_v = table[0]
    for t, v in table[1:]:
        if dt <= t:
            frac = (dt - prev_t) / (t - prev_t) if t > prev_t else 0.0
            return prev_v + frac * (v - prev_v)
        prev_t, prev_v = t, v
    return prev_v


def grade(cf: dict) -> dict:
    """`verdict`, `drift_m` and `credible` for a stored lane counterfactual, like evasion's."""
    if not cf or not cf.get("ran"):
        return {"verdict": "not_run", "credible": False, "drift_m": 0.0}
    dt = float(cf.get("dt_at_min_s", 0.0))
    gap = float(cf.get("min_separation_m", 1e9))
    sigma = lane_drift(dt, cf.get("mode", ""))
    if gap <= 0.0 and dt < 0.25:
        return {"verdict": "degenerate", "credible": False, "drift_m": round(sigma, 2)}
    verdict = ("collides" if gap <= -sigma else "likely" if gap <= 0.0
               else "close" if gap <= sigma else "clear")
    return {"verdict": verdict, "credible": sigma <= CREDIBLE_DRIFT_M, "drift_m": round(sigma, 2)}


@dataclass
class LaneCounterfactual:
    """Nearest approach of the lane ghost to any vehicle, past the fork."""

    ran: bool = False
    reason: str = ""
    mode: str = ""
    min_separation_m: float = 1e9
    dt_at_min_s: float = 0.0
    target_vid: int | None = None
    horizon_s: float = 0.0
    stops_at_branch: bool = False


def counterfactual(samples: list[_Sample], k_fork: int, route, hold_speed: bool) -> LaneCounterfactual:
    """Fly the lane ghost from the fork and find its nearest closing approach."""
    cf = LaneCounterfactual(mode="lane_held_speed" if hold_speed else "lane_recorded_speed")
    fut = samples[k_fork:]
    if len(fut) < 4:
        cf.reason = "fewer than four ticks after the fork"
        return cf
    if fut[0].ego.speed < MIN_GHOST_MS:
        cf.reason = "ego under 15 km/h at the fork"
        return cf
    times = [s.t for s in fut]
    speeds = ([fut[0].ego.speed] * len(fut)) if hold_speed else [s.ego.speed for s in fut]
    g = follow(route, fut[0].ego.coordinateX, fut[0].ego.coordinateZ, times, speeds)
    if not g.ok:
        cf.reason = g.reason
        return cf
    cf.ran = True
    cf.stops_at_branch = g.stops_at_branch
    cf.horizon_s = round(g.poses[-1][0] - times[0], 2)
    per: dict[int, float] = {}
    by_t = {round(s.t, 3): s for s in fut}
    for t, x, z, yaw in g.poses:
        s = by_t.get(round(t, 3))
        if s is None:
            continue
        for v in s.vehicles:
            gap = body_gap((x, z, yaw), _vdict(v))
            fwd = (v.position.x - x) * -math.sin(yaw) + (v.position.z - z) * -math.cos(yaw)
            if fwd < -_CAL.ego_half_length - 1.0:
                continue
            vid = int(v.id)
            prev = per.get(vid)
            closing = prev is None or gap < prev - CLOSING_EPS
            if prev is None or gap < prev:
                per[vid] = gap
            if closing and gap < cf.min_separation_m:
                cf.min_separation_m = round(gap, 2)
                cf.dt_at_min_s = round(t - times[0], 2)
                cf.target_vid = vid
    return cf
