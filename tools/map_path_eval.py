"""How well each ego path predicts where the truck actually went, on the clip corpus.

Ground truth is ego's own future path in the clip. Every method starts at the
rear-axle path origin, as ACC and AEB do. Methods:

- ``arc_aeb``: constant curvature from AEB's steer-led ``EgoPathModel``.
- ``arc_acc``: ACC's steer/history blend.
- ``map_<b>``: the fused map path with a steering-led stretch of ``b`` seconds.
- ``fused_<b>``: that path blended into ``arc_aeb`` by its own trust, which is
  what a consumer that falls back on low trust actually gets.

    python tools/map_path_eval.py --sample 400 --workers 16
    python tools/map_path_eval.py --report tools/map_lane_audit_out/path_eval.json

Output holds residuals and categories only, never world coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DISTANCES = (25.0, 50.0, 75.0, 100.0, 150.0)
BLENDS = (0.0, 1.5, 2.5, 4.0)
_EVAL_EVERY = 8
_MIN_SPEED_MS = 5.0
_OUT = ROOT / "tools" / "map_lane_audit_out"
_MAPS: dict = {}


def _game_map(game: str):
    if game not in _MAPS:
        from core.road_layout.game_map import GameMap
        try:
            _MAPS[game] = GameMap.open_installed(game)
        except (FileNotFoundError, OSError):
            _MAPS[game] = None
    return _MAPS[game]


def _turn_cat(dpsi_deg: float) -> str:
    a = abs(dpsi_deg)
    if a < 3.0:
        return "straight"
    if a < 15.0:
        return "gentle"
    if a < 45.0:
        return "bend"
    return "sharp"


def _pick_game(ego_pts, box):
    from core.road_layout.ego_route import EgoRouter

    for game in ("ets2", "ats"):
        gm = _game_map(game)
        if gm is None:
            continue
        window = gm.lanes_in_box(*box)
        router = EgoRouter()
        hits = 0
        for x, y, z, fx, fz in ego_pts[:: max(1, len(ego_pts) // 30)]:
            route = router.step(window, x, y, z, fx, fz)
            if route is not None and abs(route.offset_m) < 3.0:
                hits += 1
        if hits >= 0.6 * len(ego_pts[:: max(1, len(ego_pts) // 30)]):
            return game, window
    return None, None


def eval_clip(path_str: str) -> dict:
    from core.acc.ego_path import blend_curvature
    from core.acc.road_model import arc_coords
    from core.aeb.calibration import DEFAULT as CAL, ego_path_params
    from core.aeb.clip_replay import ego_path_replay
    from core.aeb.clip_store import ClipStore
    from core.aeb.clip_timebase import replay_frames
    from core.radar.ego_geometry import calibration_geometry
    from core.radar.ego_path import EGO_POSITION_HISTORY_LEN, ego_curvature_from_history
    from core.radar.ego_path_model import EgoPathModel
    from core.road_layout.ego_route import EgoRouter
    from core.road_layout.map_path import PathParams, build_map_path

    path = Path(path_str)
    clip = ClipStore(path.parent).load(path)
    if clip is None or not clip.radar_frames:
        return {"error": "unreadable"}
    cid = (clip.metadata.clip_id or path.name)[:8]
    try:
        frames = [f for f in replay_frames(clip) if not f.ego.paused]
        tkin_by_t, warm = ego_path_replay(clip, frames=frames)
    except Exception as exc:  # an odd clip must not stop the run
        return {"clip": cid, "error": f"replay {type(exc).__name__}"}
    if len(frames) < 50:
        return {"clip": cid, "error": "short"}
    geo = frames[0].ego.geometry() or calibration_geometry(
        CAL.ego_half_length, CAL.ego_half_width, CAL.arc_start_pctg,
    )
    origin = geo.path_origin_m
    model = EgoPathModel(params=ego_path_params(CAL), gain=warm)
    rows = []
    hist: list[tuple[float, float, float]] = []
    ego = []
    for f in frames:
        e = f.ego
        yaw = e.rotationX * 2.0 * math.pi
        fx, fz = -math.sin(yaw), -math.cos(yaw)
        t_kin = tkin_by_t.get(f.t_mono, 0.0)
        state = model.step(t_kin, yaw, e.speed, e.userSteer) if t_kin > 0.0 else model.state
        hist.append((f.t_mono, e.coordinateX, e.coordinateZ))
        if len(hist) > EGO_POSITION_HISTORY_LEN:
            del hist[0]
        k_hist = ego_curvature_from_history(hist)
        k_acc = blend_curvature(e.userSteer, k_hist, e.speed)
        ox, oz = e.coordinateX + origin * fx, e.coordinateZ + origin * fz
        ego.append((e.coordinateX, e.coordinateY, e.coordinateZ, fx, fz, ox, oz, e.speed,
                    state.kappa_path, k_acc))
    xs = [p[0] for p in ego]
    zs = [p[2] for p in ego]
    box = (min(xs) - 300, max(xs) + 300, min(zs) - 300, max(zs) + 300)
    game, window = _pick_game([(p[0], p[1], p[2], p[3], p[4]) for p in ego], box)
    if game is None:
        return {"clip": cid, "error": "off map"}
    cum = [0.0]
    for i in range(1, len(ego)):
        cum.append(cum[-1] + math.hypot(ego[i][5] - ego[i - 1][5], ego[i][6] - ego[i - 1][6]))
    router = EgoRouter()
    params = {b: PathParams(blend_s=b, blend_min_m=0.01 if b == 0.0 else 15.0) for b in BLENDS}
    offsets: dict[int, float] = {}
    for i, (x, y, z, fx, fz, ox, oz, speed, k_aeb, k_acc) in enumerate(ego):
        route = router.step(window, x, y, z, fx, fz)
        if route is not None:
            offsets[i] = route.offset_m
        if i % _EVAL_EVERY or abs(speed) < _MIN_SPEED_MS:
            continue
        paths = {}
        if route is not None:
            for b, prm in params.items():
                paths[b] = build_map_path(route, x, z, fx, fz, k_aeb, speed, prm, origin_m=origin)
        j_ptr = i
        for d in DISTANCES:
            target = cum[i] + d
            while j_ptr < len(cum) - 1 and cum[j_ptr] < target:
                j_ptr += 1
            if cum[j_ptr] < target:
                break
            j = j_ptr
            tx, tz = ego[j][5], ego[j][6]
            dx, dz = tx - ox, tz - oz
            px, py = dx * fx + dz * fz, dx * (-fz) + dz * fx
            fjx, fjz = ego[j][3], ego[j][4]
            dpsi = math.degrees(math.atan2(fx * fjz - fz * fjx, fx * fjx + fz * fjz))
            lane_change = (
                i in offsets and j in offsets and abs(offsets[j] - offsets[i]) > 1.5
            ) or (i in offsets and j in offsets and abs(offsets[j] - offsets[i]) > 1.0 and abs(dpsi) < 3.0)
            e_aeb = arc_coords(k_aeb, px, py)[1]
            e_acc = arc_coords(k_acc, px, py)[1]
            row = {"d": d, "cat": _turn_cat(dpsi), "lc": bool(lane_change),
                   "arc_aeb": round(e_aeb, 3), "arc_acc": round(e_acc, 3), "i": i,
                   "v": round(speed, 1)}
            if route is not None:
                row["kind"] = route.kind
                row["off"] = round(route.offset_m, 2)
                row["branch"] = None if route.branch_s is None else round(route.branch_s, 1)
            for b, mp in paths.items():
                if mp is None:
                    continue
                hit = mp.coords(px, py)
                if hit is None:
                    continue
                s_at, err = hit
                w = mp.trust_at(s_at)
                row[f"map_{b}"] = round(err, 3)
                row[f"fused_{b}"] = round(w * err + (1.0 - w) * e_aeb, 3)
                if b == 2.5:
                    row["trust"] = round(w, 3)
            rows.append(row)
    return {"clip": cid, "game": game, "session": clip.metadata.session_kind, "rows": rows}


def _clip_paths(store: str) -> list[str]:
    from core.aeb.clip_store import ClipStore, contributed_clip_root, default_clip_root

    roots = []
    if store in ("local", "both"):
        roots.append(default_clip_root())
    if store in ("contributed", "both"):
        roots.append(contributed_clip_root())
    out: list[str] = []
    for root in roots:
        out.extend(str(info.path) for info in ClipStore(root).list_clips())
    return out


def _q(values, p):
    return values[min(len(values) - 1, int(p * len(values)))]


def report(results: list[dict], session: str | None = None) -> None:
    ok = [r for r in results if "rows" in r and (session is None or r.get("session") == session)]
    print(f"clips {len(results)} usable {len(ok)} session {session or 'all'}")
    methods = ["arc_aeb", "arc_acc"] + [f"map_{b}" for b in BLENDS] + [f"fused_{b}" for b in BLENDS]
    for lc in (False, True):
        print(f"\n=== {'lane changes' if lc else 'no lane change'} ===")
        for cat in ("straight", "gentle", "bend", "sharp"):
            for d in DISTANCES:
                rows = [row for r in ok for row in r["rows"] if row["d"] == d and row["cat"] == cat and row["lc"] == lc]
                if len(rows) < 30:
                    continue
                cover = sum(1 for row in rows if "map_2.5" in row) / len(rows)
                print(f"{cat:8s} d={d:5.0f} n={len(rows):6d} map coverage {cover:.0%}")
                both = [row for row in rows if "map_2.5" in row]
                for m in methods:
                    vals = sorted(abs(row[m]) for row in both if m in row)
                    if not vals:
                        continue
                    print(f"   {m:10s} p50 {_q(vals, .5):6.2f}  p90 {_q(vals, .9):6.2f}  p99 {_q(vals, .99):6.2f}  >1.75m {sum(v > 1.75 for v in vals) / len(vals):.1%}")


def calibration(results: list[dict]) -> None:
    """Share of map paths outside a lane (> 1.75 m) by trust bin: trust must rank failure."""
    ok = [r for r in results if "rows" in r]
    print()
    print("=== trust calibration, map_2.5, no lane change ===")
    for d in (50.0, 100.0):
        rows = [row for r in ok for row in r["rows"] if row["d"] == d and not row["lc"] and "trust" in row]
        for lo, hi in ((0.0, 0.05), (0.05, 0.5), (0.5, 0.95), (0.95, 1.01)):
            sub = [row for row in rows if lo <= row["trust"] < hi]
            if not sub:
                continue
            fail = sum(abs(row["map_2.5"]) > 1.75 for row in sub) / len(sub)
            arc = sum(abs(row["arc_aeb"]) > 1.75 for row in sub) / len(sub)
            print(f"  d={d:4.0f} trust {lo:.2f}-{hi:.2f} n={len(sub):6d} map fail {fail:6.1%}  arc fail {arc:6.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default="both", choices=("local", "contributed", "both"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=str(_OUT / "path_eval.json"))
    parser.add_argument("--report")
    parser.add_argument("--session", choices=("SP", "TMP"))
    args = parser.parse_args()
    if args.report:
        loaded = json.loads(Path(args.report).read_text(encoding="utf-8"))
        report(loaded, args.session)
        calibration(loaded)
        return 0
    paths = _clip_paths(args.store)
    if args.sample and args.sample < len(paths):
        random.Random(args.seed).shuffle(paths)
        paths = paths[:args.sample]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(eval_clip, p) for p in paths]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"error": f"worker {type(exc).__name__}"})
            if n % 100 == 0:
                print(f"{n}/{len(paths)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results), encoding="utf-8")
    report(results, args.session)
    calibration(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
