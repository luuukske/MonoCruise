"""How well lane-following predicts other vehicles, against AEB's constant-curvature arcs.

For every moving tracked vehicle, the predicted path is compared with where the
vehicle actually was 1, 2 and 3 s later (lateral error from the predicted path,
so speed errors do not count). AI and TruckersMP traffic are reported apart.

    python tools/map_target_eval.py --sample 300 --workers 16
    python tools/map_target_eval.py --report tools/map_lane_audit_out/target_eval.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HORIZONS = (1.0, 2.0, 3.0)
_EVAL_EVERY = 10
_MIN_SPEED_MS = 3.0
_MAX_RANGE_M = 150.0
_OUT = ROOT / "tools" / "map_lane_audit_out"


def _yaw(v) -> float:
    if v._smooth_yaw is not None:
        return v._smooth_yaw
    return math.radians(v.rotation.euler()[1])


def _turn_cat(dpsi: float) -> str:
    a = abs(dpsi)
    return "straight" if a < 3 else "gentle" if a < 15 else "bend" if a < 45 else "sharp"


def eval_clip(path_str: str) -> dict:
    from core.acc.road_model import arc_coords
    from core.aeb.calibration import DEFAULT as CAL
    from core.aeb.clip_replay import decode_radar_stream
    from core.aeb.clip_store import deserialize_clip
    from core.aeb.clip_timebase import replay_frames
    from core.aeb.filters import VehicleCurvatureBlender, _vehicle_curvature_blend
    from core.road_layout.target_path import TargetRouter, target_path
    from tools.map_replay import match_clip

    try:
        clip = deserialize_clip(Path(path_str).read_bytes())
    except Exception:
        return {"error": "unreadable"}
    cid = (clip.metadata.clip_id or "")[:8]
    game, window = match_clip(clip)
    if game is None:
        return {"clip": cid, "error": "off map"}
    try:
        frames = replay_frames(clip)
        veh_by_t, ego_by_t, frame_t, _off = decode_radar_stream(clip, frames=frames, with_elevation=False)
    except Exception as exc:  # an odd clip must not stop the run
        return {"clip": cid, "error": f"replay {type(exc).__name__}"}
    tracks: dict[int, list] = {}
    for t in frame_t:
        for v in veh_by_t.get(t, []):
            if v.is_trailer or v.is_parked or v.id < 0:
                continue
            tracks.setdefault(v.id, []).append((t, v.position.x, v.position.z))
    blender = VehicleCurvatureBlender(CAL)
    router = TargetRouter()
    rows = []
    for k, t in enumerate(frame_t):
        ego = ego_by_t.get(t)
        vehicles = [v for v in veh_by_t.get(t, []) if not v.is_trailer and not v.is_parked and v.id >= 0]
        router.prune(v.id for v in vehicles)
        for v in vehicles:
            speed = abs(v.speed)
            kappa = _vehicle_curvature_blend(v, speed, CAL, blender, t)
            yaw = _yaw(v)
            fx, fz = -math.sin(yaw), -math.cos(yaw)
            if v.speed < 0.0:
                fx, fz = -fx, -fz
            route = router.route(v.id, window, v.position.x, v.position.y, v.position.z, fx, fz)
            if k % _EVAL_EVERY or speed < _MIN_SPEED_MS or ego is None:
                continue
            if math.hypot(v.position.x - ego.coordinateX, v.position.z - ego.coordinateZ) > _MAX_RANGE_M:
                continue
            path = target_path(route, v.position.x, v.position.z,
                               yaw if v.speed >= 0.0 else yaw + math.pi, kappa, speed)
            track = tracks[v.id]
            times = [p[0] for p in track]
            for h in HORIZONS:
                j = bisect.bisect_left(times, t + h)
                if j >= len(track) or abs(track[j][0] - (t + h)) > 0.1:
                    break
                dx = track[j][1] - v.position.x
                dz = track[j][2] - v.position.z
                px, py = dx * fx + dz * fz, dx * (-fz) + dz * fx
                if math.hypot(px, py) < 1.0:
                    break
                jj = max(0, j - 3)
                tx, tz = track[j][1] - track[jj][1], track[j][2] - track[jj][2]
                dpsi = math.degrees(math.atan2(fx * tz - fz * tx, fx * tx + fz * tz)) if math.hypot(tx, tz) > 0.5 else 0.0
                row = {"h": h, "tmp": bool(v.is_tmp), "cat": _turn_cat(dpsi),
                       "arc": round(arc_coords(kappa, px, py)[1], 3)}
                if path is not None:
                    hit = path.coords(px, py)
                    if hit is not None:
                        row["lane"] = round(hit[1], 3)
                        row["trust"] = round(path.trust_at(hit[0]), 3)
                        row["off"] = round(route.offset_m, 2)
                rows.append(row)
    return {"clip": cid, "game": game, "session": clip.metadata.session_kind, "rows": rows}


def report(results: list[dict]) -> None:
    ok = [r for r in results if "rows" in r]
    print(f"clips {len(results)} usable {len(ok)}")

    def q(vals, p):
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    for tmp in (False, True):
        print(f"\n=== {'TMP players' if tmp else 'AI traffic'} ===")
        for cat in ("straight", "gentle", "bend", "sharp"):
            for h in HORIZONS:
                rows = [row for r in ok for row in r["rows"] if row["h"] == h and row["tmp"] == tmp and row["cat"] == cat]
                if len(rows) < 30:
                    continue
                cover = sum("lane" in row for row in rows) / len(rows)
                both = [row for row in rows if "lane" in row]
                trusted = [row for row in both if row["trust"] >= 0.9]
                parts = []
                for name, sel, key in (("arc", both, "arc"), ("lane", both, "lane"), ("lane@trust", trusted, "lane")):
                    vals = sorted(abs(row[key]) for row in sel)
                    if vals:
                        parts.append(f"{name} p50 {q(vals, .5):5.2f} p90 {q(vals, .9):5.2f} out {sum(v > 1.75 for v in vals) / len(vals):5.1%}")
                print(f"{cat:8s} {h:.0f}s n={len(rows):6d} cover {cover:4.0%} | " + " | ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default="both", choices=("local", "contributed", "both"))
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=str(_OUT / "target_eval.json"))
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.report:
        report(json.loads(Path(args.report).read_text(encoding="utf-8")))
        return 0
    from tools.map_aeb_ab import _init_worker, _paths

    paths = _paths(args.store)
    if args.sample and args.sample < len(paths):
        random.Random(13).shuffle(paths)
        paths = paths[:args.sample]
    results = []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(eval_clip, p) for p in paths]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"error": f"worker {type(exc).__name__}"})
            if n % 50 == 0:
                print(f"{n}/{len(paths)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results), encoding="utf-8")
    report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
