"""Scene context for one clip: where ego really went, where its road went, and where each target sat on both.

Two evidence sources, kept apart so a reader can weigh them:

- Hindsight: ego's recorded future path. No map needed. A target ego later drove
  through was in its path, whatever the straight-ahead `lat` column says.
- Map lanes from the local game install (`tools/map_replay.py`). Tagging evidence
  only: nothing here feeds AEB, `core/`, or a clip file.

Every quantity is a measurement of the recording. None of it is AEB's opinion.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass, field

from core.aeb.calibration import DEFAULT as _CAL

from tools.aeb_agent.features import world_to_ego

CONTEXT_VERSION = 2

# Target centre this close to ego's path centreline shares ego's lane.
SAME_LANE_M = 2.0
ADJACENT_MAX_M = 8.0
OPPOSITE_MAX_M = 12.0
ALIGNED_DOT = 0.5
# Further than this from every lane centre is a yard, a car park or a shoulder.
OFF_LANE_AI_M = 3.0
OFF_LANE_TMP_M = 3.5
LANE_SEARCH_M = 10.0
LEVEL_GAP_M = 5.0
# Target ahead of ego by more than this at the reference moment is not in the table.
TARGET_RANGE_M = 120.0
MAX_TARGETS = 8
APPROACH_LEAD_S = 1.5
# Hindsight demand: the decel that stops ego's closing on a target in its path.
DEMAND_MIN_GAP_M = 0.5
DEMAND_CAP_MS2 = 30.0
# Route heading samples for the bend description.
BEND_AT_M = (50.0, 100.0)


@dataclass
class Placement:
    """One target at one moment, measured against ego's real path and ego's lane route."""

    t: float = 0.0
    fwd: float = 0.0
    lat: float = 0.0
    path_s: float | None = None
    path_d: float | None = None
    path_reached: bool = False
    path_short_m: float = 0.0
    t_arrive: float | None = None
    map_rel: str = "unknown"
    map_s: float | None = None
    map_d: float | None = None
    map_dot: float | None = None
    lane_dist_m: float | None = None
    lane_kind: str = ""
    against_travel: bool = False
    past_branch: bool = False
    level_gap_m: float = 0.0


@dataclass
class TargetContext:
    vid: int
    is_tmp: bool = False
    is_trailer: bool = False
    length_m: float = 0.0
    half_w_m: float = 0.0
    speed_kmh: float = 0.0
    approach: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    real_min_sep_m: float = 1e9
    t_real_min: float = 0.0
    in_path: str = "unknown"


@dataclass
class SceneContext:
    """Everything the dossier says about the road and the hindsight geometry."""

    version: int = CONTEXT_VERSION
    clip_id: str = ""
    path: str = ""
    error: str = ""
    map_matched: bool = False
    game: str = ""
    t_approach: float = 0.0
    t_action: float = 0.0
    ego: dict = field(default_factory=dict)
    ego_travel_m: float = 0.0
    ego_end_speed_kmh: float = 0.0
    targets: list = field(default_factory=list)
    hindsight_primary: int | None = None
    demand: list = field(default_factory=list)
    demand_vid: int | None = None
    # Ego frame (fwd, right) polylines at each reference moment, for the ASCII scenes.
    overlays: dict = field(default_factory=dict)
    # Raw lane-following counterfactual; graded at read time against lane_ghost's table.
    lane_cf: dict = field(default_factory=dict)


def _yaw(ego) -> float:
    return ego.rotationX * 2.0 * math.pi


def _veh_yaw(v) -> float:
    smooth = getattr(v, "_smooth_yaw", None)
    yaw = smooth if smooth is not None else math.radians(v.rotation.euler()[1])
    return yaw + math.pi if v.speed < 0.0 else yaw


def _unit(dx: float, dz: float) -> tuple[float, float] | None:
    n = math.hypot(dx, dz)
    if n < 1e-9:
        return None
    return dx / n, dz / n


def cumulative(pts) -> list[float]:
    out = [0.0]
    for a, b in zip(pts, pts[1:]):
        out.append(out[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return out


def project(pts, cum, x: float, z: float) -> tuple[float, float, int] | None:
    """``(s, right offset, segment index)`` of a point on a polyline.

    The end segments extend past the ends, so `s` can be negative or exceed the length.
    """
    if len(pts) < 2:
        return None
    best = None
    last = len(pts) - 2
    for i in range(len(pts) - 1):
        ax, az = pts[i]
        bx, bz = pts[i + 1]
        seg = _unit(bx - ax, bz - az)
        if seg is None:
            continue
        tx, tz = seg
        length = cum[i + 1] - cum[i]
        along = (x - ax) * tx + (z - az) * tz
        if i > 0:
            along = max(along, 0.0)
        if i < last:
            along = min(along, length)
        px, pz = ax + tx * along, az + tz * along
        dist = math.hypot(x - px, z - pz)
        if best is None or dist < best[0]:
            right = (x - ax) * (-tz) + (z - az) * tx
            best = (dist, cum[i] + along, right, i)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _tangent(pts, i: int) -> tuple[float, float] | None:
    i = max(0, min(i, len(pts) - 2))
    return _unit(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])


def _body_points(x: float, z: float, yaw: float, half_len: float) -> list:
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    return [(x + f * half_len * fx, z + f * half_len * fz) for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]


def body_gap(ego_xz_yaw, veh: dict) -> float:
    """Closest gap between ego's tractor body and a vehicle with its trailers."""
    ex, ez, eyaw = ego_xz_yaw
    ego_pts = _body_points(ex, ez, eyaw, _CAL.ego_half_length)
    best = 1e9
    bodies = [(veh["x"], veh["z"], veh["yaw"], veh["length"] / 2.0, veh["half_w"])]
    for tr in veh.get("trailers", []) or []:
        bodies.append((tr["x"], tr["z"], tr["yaw"], tr["length"] / 2.0, tr["half_w"]))
    for bx, bz, byaw, hl, hw in bodies:
        for tx, tz in _body_points(bx, bz, byaw, hl):
            for px, pz in ego_pts:
                gap = math.hypot(px - tx, pz - tz) - _CAL.ego_half_width - hw
                if gap < best:
                    best = gap
    return best


def bend_deg(route_pts, ego_x: float, ego_z: float, ego_yaw: float, at_m: float) -> float | None:
    """Heading change of the route from its start to `at_m` along it; positive turns right."""
    cum = cumulative(route_pts)
    if cum[-1] < at_m * 0.6:
        return None
    local = [world_to_ego(x, z, ego_x, ego_z, ego_yaw) for x, z in route_pts]
    j = bisect.bisect_left(cum, min(at_m, cum[-1])) - 1
    j = max(0, min(j, len(local) - 2))
    h0 = math.atan2(local[1][1] - local[0][1], local[1][0] - local[0][0])
    h1 = math.atan2(local[j + 1][1] - local[j][1], local[j + 1][0] - local[j][0])
    return math.degrees((h1 - h0 + math.pi) % (2.0 * math.pi) - math.pi)


def classify(map_d: float, dot: float, moving: bool) -> str:
    """Relation of a target to ego's lane route from its offset and relative heading."""
    side = "right" if map_d > 0.0 else "left"
    if abs(map_d) <= SAME_LANE_M:
        if moving and dot <= -ALIGNED_DOT:
            return "ego_lane_wrong_way"
        if moving and abs(dot) < ALIGNED_DOT:
            return "across_ego_lane"
        return "ego_lane"
    if dot >= ALIGNED_DOT and abs(map_d) <= ADJACENT_MAX_M:
        return f"adjacent_{side}"
    if dot <= -ALIGNED_DOT and abs(map_d) <= OPPOSITE_MAX_M:
        return f"oncoming_{side}"
    if abs(dot) < ALIGNED_DOT and moving:
        return f"crossing_{side}"
    return f"beside_road_{side}"


def _lane_hit(window, x: float, z: float, yaw: float):
    """``(distance, kind, against travel)`` to the nearest lane centre within reach."""
    near = getattr(window, "index", None)
    lanes = getattr(window, "lanes", ())
    if near is None or not lanes:
        return None, "", False
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    best = None
    for idx in near.around(x, z, LANE_SEARCH_M):
        lane = lanes[idx]
        pts = lane.pts
        for i in range(len(pts) - 1):
            ax, az = pts[i]
            bx, bz = pts[i + 1]
            seg = _unit(bx - ax, bz - az)
            if seg is None:
                continue
            length = math.hypot(bx - ax, bz - az)
            along = max(0.0, min(length, (x - ax) * seg[0] + (z - az) * seg[1]))
            dist = math.hypot(x - (ax + seg[0] * along), z - (az + seg[1] * along))
            if best is None or dist < best[0]:
                dot = fx * seg[0] + fz * seg[1]
                against = bool(lane.travel) and abs(dot) > 0.7 and dot * lane.travel < 0.0
                best = (dist, lane.kind, against)
    return best if best is not None else (None, "", False)


@dataclass
class _Sample:
    t: float
    ego: object
    vehicles: list


def _timeline(clip):
    """Per AEB tick: (t_rel, decoded ego, decoded vehicles), as features.extract pairs them."""
    from core.aeb.clip_replay import decode_radar_stream

    ticks = sorted(clip.aeb_ticks, key=lambda x: x.t_mono)
    all_t = [f.t_mono for f in clip.radar_frames] + [t.t_mono for t in ticks]
    t0 = min(all_t) if all_t else 0.0
    veh_by_t, ego_by_t, frame_t, _off = decode_radar_stream(clip, with_elevation=False)
    frame_t = sorted(frame_t)
    out: list[_Sample] = []
    for tk in ticks:
        if not frame_t:
            break
        j = bisect.bisect_left(frame_t, tk.radar_t_mono)
        cands = [frame_t[k] for k in (j - 1, j) if 0 <= k < len(frame_t)]
        ft = min(cands, key=lambda f: abs(f - tk.radar_t_mono))
        ego = ego_by_t.get(ft)
        if ego is None or ego.paused:
            continue
        out.append(_Sample(tk.t_mono - t0, ego, veh_by_t.get(ft, [])))
    return out


def _vdict(v) -> dict:
    trailers = [{
        "x": tr.position.x, "z": tr.position.z,
        "yaw": math.radians(tr.rotation.euler()[1]),
        "length": tr.size.length, "half_w": tr.size.width / 2.0,
    } for tr in v.trailers]
    return {"vid": int(v.id), "x": v.position.x, "y": v.position.y, "z": v.position.z,
            "yaw": _veh_yaw(v), "length": v.size.length, "half_w": v.size.width / 2.0,
            "speed": abs(v.speed), "trailers": trailers, "is_tmp": bool(v.is_tmp),
            "is_trailer": bool(getattr(v, "is_trailer", False))}


def _nearest(samples: list[_Sample], t: float) -> int:
    times = [s.t for s in samples]
    j = bisect.bisect_left(times, t)
    cands = [k for k in (j - 1, j) if 0 <= k < len(samples)]
    return min(cands, key=lambda k: abs(samples[k].t - t))


def _ego_path(samples: list[_Sample], start: int):
    """Ego's recorded future from `start`, de-duplicated so standstill adds no points."""
    pts, times = [], []
    for s in samples[start:]:
        p = (s.ego.coordinateX, s.ego.coordinateZ)
        if pts and math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) < 0.05:
            continue
        pts.append(p)
        times.append(s.t)
    return pts, times


def _extend(pts, route, reach_m: float = 150.0):
    """Ego's real path, continued along its lane route (or straight) past where it stopped."""
    if len(pts) < 2:
        return list(pts)
    out = list(pts)
    end = out[-1]
    if route is not None and len(route.pts) >= 2:
        cum = cumulative(route.pts)
        hit = project(route.pts, cum, end[0], end[1])
        if hit is not None:
            s0 = hit[0]
            for p, s in zip(route.pts, cum):
                if s > s0 + 1.0:
                    out.append(p)
            return out
    tan = _unit(out[-1][0] - out[-2][0], out[-1][1] - out[-2][1])
    if tan is None:
        return out
    out.append((end[0] + tan[0] * reach_m, end[1] + tan[1] * reach_m))
    return out


def _place(veh: dict, sample: _Sample, path, path_cum, path_len, times,
           route, window) -> Placement:
    ego = sample.ego
    eyaw = _yaw(ego)
    fwd, lat = world_to_ego(veh["x"], veh["z"], ego.coordinateX, ego.coordinateZ, eyaw)
    pl = Placement(t=round(sample.t, 2), fwd=round(fwd, 1), lat=round(lat, 1))
    pl.level_gap_m = round(veh.get("y", 0.0) - ego.coordinateY, 1)
    hit = project(path, path_cum, veh["x"], veh["z"]) if len(path) >= 2 else None
    if hit is not None:
        s, d, _i = hit
        pl.path_s, pl.path_d = round(s, 1), round(d, 2)
        pl.path_reached = 0.0 <= s <= path_len
        pl.path_short_m = round(max(0.0, s - path_len), 1)
        if pl.path_reached and times:
            k = min(bisect.bisect_left(path_cum, s), len(times) - 1)
            pl.t_arrive = round(times[k], 2)
    if route is not None:
        rcum = cumulative(route.pts)
        rh = project(route.pts, rcum, veh["x"], veh["z"])
        if rh is not None:
            s, d, i = rh
            d_rel = d
            tan = _tangent(route.pts, i)
            vf = (-math.sin(veh["yaw"]), -math.cos(veh["yaw"]))
            dot = (vf[0] * tan[0] + vf[1] * tan[1]) if tan is not None else 0.0
            pl.map_s, pl.map_d, pl.map_dot = round(s, 1), round(d_rel, 2), round(dot, 2)
            moving = veh["speed"] > 1.0
            rel = classify(d_rel, dot, moving)
            if s < -_CAL.ego_half_length:
                rel = "behind_" + rel
            elif s > route.reach_m + 5.0:
                rel = "beyond_route"
            pl.map_rel = rel
            pl.past_branch = route.branch_s is not None and s > route.branch_s
    if window is not None:
        dist, kind, against = _lane_hit(window, veh["x"], veh["z"], veh["yaw"])
        pl.lane_dist_m = None if dist is None else round(dist, 2)
        pl.lane_kind = kind
        pl.against_travel = against
        limit = OFF_LANE_TMP_M if veh.get("is_tmp") else OFF_LANE_AI_M
        if dist is None or dist > limit:
            pl.map_rel = "off_lane" if pl.map_rel == "unknown" else pl.map_rel + "+off_lane"
    return pl


def _in_path(pl: Placement, half_w: float) -> str:
    """Overlap of the target with ego's real (or extended) path, by lateral offset."""
    if pl.path_d is None or pl.path_s is None or pl.path_s < 0.0:
        return "unknown"
    reach = _CAL.ego_half_width + half_w
    off = abs(pl.path_d)
    if off <= reach:
        return "in_path"
    if off <= reach + 1.0:
        return "brushing"
    return "clear"


def _demand(samples, vid: int, start: int, path, path_cum):
    """Per tick, the constant decel that stops ego closing on `vid` along its real path."""
    out = []
    for s in samples[start:]:
        veh = next((v for v in s.vehicles if int(v.id) == vid), None)
        if veh is None:
            continue
        ego = s.ego
        eh = project(path, path_cum, ego.coordinateX, ego.coordinateZ)
        th = project(path, path_cum, veh.position.x, veh.position.z)
        if eh is None or th is None:
            continue
        tan = _tangent(path, th[2])
        vyaw = _veh_yaw(veh)
        along = 0.0
        if tan is not None:
            along = abs(veh.speed) * (-math.sin(vyaw) * tan[0] - math.cos(vyaw) * tan[1])
        gap = th[0] - eh[0] - _CAL.ego_half_length - veh.size.length / 2.0
        closing = ego.speed - along
        if closing <= 0.0:
            req = 0.0
        elif gap <= DEMAND_MIN_GAP_M:
            req = DEMAND_CAP_MS2
        else:
            req = min(DEMAND_CAP_MS2, closing * closing / (2.0 * gap))
        out.append((round(s.t, 2), round(req, 2), round(gap, 1), round(ego.speed * 3.6, 1)))
    return out


def _ego_block(samples, k_act: int, route, window, router_keys) -> dict:
    ego = samples[k_act].ego
    out: dict = {"on_lane": route is not None}
    if route is None:
        return out
    lane = window.lanes[route.lane_index] if 0 <= route.lane_index < len(window.lanes) else None
    out["lane_kind"] = route.kind
    out["offset_m"] = round(route.offset_m, 2)
    out["heading_err_deg"] = round(math.degrees(route.heading_err_rad), 1)
    if lane is not None and lane.kind == "road":
        same = [ln for ln in window.lanes if ln.kind == "road" and ln.uid == lane.uid]
        out["lanes_total"] = len(same)
        out["lanes_same_dir"] = sum(1 for ln in same if ln.travel == lane.travel) if lane.travel else None
    out["route_reach_m"] = round(route.reach_m, 0)
    out["branch_s"] = None if route.branch_s is None else round(route.branch_s, 0)
    out["merge_s"] = None if route.merge_s is None else round(route.merge_s, 0)
    for at in BEND_AT_M:
        b = bend_deg(route.pts, ego.coordinateX, ego.coordinateZ, _yaw(ego), at)
        out[f"bend_{int(at)}_deg"] = None if b is None else round(b, 0)
    t_act = samples[k_act].t
    changes = [t for t, prev, cur in router_keys
               if prev is not None and cur is not None and prev[0] == cur[0] == "road"
               and prev[1] == cur[1] and prev[2] != cur[2] and t_act - 3.0 <= t <= t_act + 1.0]
    out["lane_change_t"] = round(changes[0], 2) if changes else None
    return out


def build(clip, feats, path: str = "", map_provider=None) -> SceneContext:
    """Context for one decoded clip. `feats` is its ClipFeatures (or the cached dict)."""
    get = feats.get if isinstance(feats, dict) else (lambda k, d=None: getattr(feats, k, d))
    ctx = SceneContext(clip_id=clip.metadata.clip_id, path=path)
    samples = _timeline(clip)
    if len(samples) < 5:
        ctx.error = "too few usable ticks"
        return ctx
    action_t = float(get("action_t", 0.0) or 0.0)
    k_act = _nearest(samples, action_t)
    k_app = _nearest(samples, max(samples[0].t, action_t - APPROACH_LEAD_S))
    ctx.t_action, ctx.t_approach = samples[k_act].t, samples[k_app].t
    inter = get("intervention", {}) or {}
    k_fork = _nearest(samples, float(inter.get("t", 0.0))) if inter.get("found") else None

    game = window = None
    if map_provider is None:
        from tools.map_replay import match_clip
        map_provider = match_clip
    try:
        game, window = map_provider(clip)
    except Exception as exc:  # a broken sector must not lose the hindsight half
        ctx.error = f"map: {type(exc).__name__}"
    ctx.map_matched = window is not None
    ctx.game = game or ""

    routes: dict[int, object] = {}
    router_keys = []
    if window is not None:
        from core.road_layout.ego_route import EgoRouter
        router = EgoRouter()
        prev_key = None
        for k, s in enumerate(samples):
            e = s.ego
            yaw = _yaw(e)
            r = router.step(window, e.coordinateX, e.coordinateY, e.coordinateZ,
                            -math.sin(yaw), -math.cos(yaw))
            if k in (k_app, k_act, k_fork):
                routes[k] = r
            key = None
            if r is not None and 0 <= r.lane_index < len(window.lanes):
                ln = window.lanes[r.lane_index]
                key = (ln.kind, ln.uid, ln.lane_index)
            router_keys.append((s.t, prev_key, key))
            prev_key = key
        ctx.ego = _ego_block(samples, k_act, routes.get(k_act), window, router_keys)
        if k_fork is not None:
            from tools.aeb_agent.lane_ghost import counterfactual
            kind = inter.get("kind") or ""
            lane_cf = counterfactual(samples, k_fork, routes.get(k_fork),
                                     hold_speed="brake" in kind or "lift" in kind)
            ctx.lane_cf = asdict(lane_cf)

    last = samples[-1].ego
    ctx.ego_end_speed_kmh = round(last.speed * 3.6, 1)
    full, _t = _ego_path(samples, 0)
    ctx.ego_travel_m = round(cumulative(full)[-1] if len(full) >= 2 else 0.0, 1)

    tracks: dict[int, TargetContext] = {}
    wanted = [int(t["vid"]) for t in (get("targets", []) or [])][:MAX_TARGETS]
    for k_ref, label in ((k_app, "approach"), (k_act, "action")):
        sample = samples[k_ref]
        real, times = _ego_path(samples, k_ref)
        route = routes.get(k_ref)
        path = _extend(real, route)
        pcum = cumulative(path) if len(path) >= 2 else [0.0]
        real_len = cumulative(real)[-1] if len(real) >= 2 else 0.0
        ego = sample.ego
        ctx.overlays[label] = {
            "t": round(sample.t, 2),
            "path": _local(real, ego),
            "route": _local(route.pts, ego) if route is not None else [],
        }
        near = []
        for v in sample.vehicles:
            fwd, lat = world_to_ego(v.position.x, v.position.z, ego.coordinateX,
                                    ego.coordinateZ, _yaw(ego))
            if -20.0 <= fwd <= TARGET_RANGE_M and abs(lat) <= 40.0:
                near.append((int(v.id) not in wanted, math.hypot(fwd, lat), v))
        near.sort(key=lambda x: (x[0], x[1]))
        for _rank, _dist, v in near[:MAX_TARGETS + 4]:
            vd = _vdict(v)
            tc = tracks.get(vd["vid"])
            if tc is None:
                tc = TargetContext(vid=vd["vid"], is_tmp=vd["is_tmp"],
                                   is_trailer=vd["is_trailer"], length_m=round(vd["length"], 1),
                                   half_w_m=round(vd["half_w"], 2),
                                   speed_kmh=round(vd["speed"] * 3.6, 1))
                tracks[vd["vid"]] = tc
            pl = _place(vd, sample, path, pcum, real_len, times, route, window)
            setattr(tc, label, asdict(pl))
            if label == "approach" or tc.in_path == "unknown":
                tc.in_path = _in_path(pl, vd["half_w"])

    for tc in tracks.values():
        for s in samples:
            veh = next((v for v in s.vehicles if int(v.id) == tc.vid), None)
            if veh is None:
                continue
            gap = body_gap((s.ego.coordinateX, s.ego.coordinateZ, _yaw(s.ego)), _vdict(veh))
            if gap < tc.real_min_sep_m:
                tc.real_min_sep_m = round(gap, 2)
                tc.t_real_min = round(s.t, 2)

    rank = {"in_path": 0, "brushing": 1, "clear": 2, "unknown": 3}
    ordered = sorted(tracks.values(), key=lambda t: (
        rank[t.in_path], _ahead_s(t) if _ahead_s(t) is not None else 1e9, t.real_min_sep_m))
    ctx.targets = [asdict(t) for t in ordered[:MAX_TARGETS]]
    ahead = [t for t in ordered if t.in_path == "in_path" and _ahead_s(t) is not None]
    if ahead:
        ctx.hindsight_primary = ahead[0].vid
        real, _times = _ego_path(samples, 0)
        path = _extend(real, routes.get(k_act))
        if len(path) >= 2:
            ctx.demand = _demand(samples, ahead[0].vid, 0, path, cumulative(path))
            ctx.demand_vid = ahead[0].vid
    return ctx


def _local(pts, ego, step_m: float = 2.0, reach_m: float = 150.0) -> list:
    """World polyline to ego-frame (fwd, right) points about every `step_m`, up to `reach_m`."""
    out: list = []
    travelled = 0.0
    last = None
    yaw = _yaw(ego)
    for p in pts:
        if last is not None:
            travelled += math.hypot(p[0] - last[0], p[1] - last[1])
            if travelled > reach_m:
                break
        if out and last is not None and travelled - out[-1][2] < step_m:
            last = p
            continue
        fwd, right = world_to_ego(p[0], p[1], ego.coordinateX, ego.coordinateZ, yaw)
        out.append((round(fwd, 1), round(right, 1), travelled))
        last = p
    return [[f, r] for f, r, _s in out]


def _ahead_s(tc: TargetContext) -> float | None:
    """Distance along ego's path at the earliest placement, when the target was ahead."""
    ref = tc.approach or tc.action or {}
    s = ref.get("path_s")
    return s if s is not None and s >= 0.0 else None


def to_dict(ctx: SceneContext) -> dict:
    return asdict(ctx)
