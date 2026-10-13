"""ACC corridor with and without the map path, against hindsight lanes.

Truth for a tracked vehicle is its distance to ego's own later driven path at
the point it is passed: in lane up to 1.5 m, adjacent 3 to 6 m, outer beyond 6 m,
5 to 150 m ahead. The tracker runs the way ``tests/acc/harness.replay_clip``
runs it, once without a map and once with ``tools/map_replay``.

    python tools/map_acc_eval.py --sample 400 --workers 12
    python tools/map_acc_eval.py --report tools/map_lane_audit_out/acc_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_OUT = ROOT / "tools" / "map_lane_audit_out"
_EVERY = 3
_BENDS = ((0, 5), (5, 15), (15, 30), (30, 360))


def _replay(clip, clip_map):
    """Per sampled frame: ``(t, ego xz, [(vid, x, z, speed, in_path, locked), ...])``."""
    from core.acc.tracker import ACCTracker
    from core.aeb.clip_timebase import replay_frames
    from core.cruise_control_thread.acc_controller import ANT_SCORE_MIN
    from core.radar.ego_path import EGO_POSITION_HISTORY_LEN, ego_curvature_from_history
    from core.radar.elevation import EgoElevationTrack, ElevationGate, build_surface
    from core.radar.reader import TrafficReader

    reader = TrafficReader()
    tracker = ACCTracker()
    elev_track = EgoElevationTrack()
    elev_gate = ElevationGate()
    hist: list[tuple[float, float, float]] = []
    last_t = 0.0
    last_hist_t = -1.0
    out = []
    for k, frame in enumerate(replay_frames(clip)):
        ego = frame.ego
        if frame.traffic_buf is None or ego.paused:
            continue
        t_kin = frame.t_wall
        if t_kin > last_hist_t:
            hist.append((t_kin, ego.coordinateX, ego.coordinateZ))
            hist = hist[-EGO_POSITION_HISTORY_LEN:]
            last_hist_t = t_kin
        elev_track.push(ego.coordinateX, ego.coordinateZ, ego.coordinateY)
        decoded = reader.replay_frame(
            frame.traffic_buf, frame.parked_buf,
            ego.coordinateX, ego.coordinateY, ego.coordinateZ, ego.speed, t_kin,
        )
        if decoded is None:
            continue
        targets = decoded[0] + decoded[1]
        dt = max(1e-3, min((frame.t_mono - last_t) if last_t > 0 else 1.0 / 30.0, 0.2))
        last_t = frame.t_mono
        yaw = ego.rotationX * 2.0 * math.pi
        pitch = (ego.rotationY + 0.5) % 1.0 - 0.5
        surface = build_surface(ego.coordinateY, -pitch * 2.0 * math.pi, elev_track)
        off_ids = elev_gate.step(targets, surface, ego.coordinateX, ego.coordinateZ, yaw)
        snap = clip_map(ego) if clip_map is not None else None
        tracker.update(
            now_mono=frame.t_mono, dt=dt, vehicles=targets,
            ego_x=ego.coordinateX, ego_z=ego.coordinateZ, ego_yaw_rad=yaw,
            ego_speed_ms=ego.speed, ego_steer=ego.userSteer,
            ego_history_kappa=ego_curvature_from_history(hist),
            blinker_left=bool(getattr(ego, "blinkerLeft", False)),
            blinker_right=bool(getattr(ego, "blinkerRight", False)),
            off_surface_ids=off_ids, ego_geometry=ego.geometry(), map_snapshot=snap,
        )
        if k % _EVERY:
            continue
        by_id = {v.id: v for v in targets}
        rows = []
        for vid, st in tracker.tracks.items():
            v = by_id.get(vid)
            if not st.last_seen_this_frame or v is None:
                continue
            rows.append((vid, v.position.x, v.position.z, abs(v.speed), bool(st.in_path),
                         bool(st.score > ANT_SCORE_MIN)))
        out.append((t_kin, ego.coordinateX, ego.coordinateZ, abs(ego.speed), rows))
    return out


def _truth_path(clip):
    from core.aeb.clip_timebase import replay_frames

    pts = []
    for f in replay_frames(clip):
        if f.ego.paused:
            continue
        p = (f.t_wall, f.ego.coordinateX, f.ego.coordinateZ)
        if not pts or math.hypot(p[1] - pts[-1][1], p[2] - pts[-1][2]) >= 1.0:
            pts.append(p)
    return pts


def _classify(path, t, ex, ez, x, z):
    """``(class, bend deg)`` from ego's later path, or None when the truth does not reach."""
    start = None
    for i, (pt, px, pz) in enumerate(path):
        if pt >= t:
            start = i
            break
    if start is None or start + 2 >= len(path):
        return None
    best = None
    walked = 0.0
    for i in range(start, len(path) - 1):
        ax, az = path[i][1], path[i][2]
        bx, bz = path[i + 1][1], path[i + 1][2]
        sx, sz = bx - ax, bz - az
        seg = math.hypot(sx, sz)
        if seg < 1e-6:
            continue
        u = max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / (seg * seg)))
        d = math.hypot(x - ax - u * sx, z - az - u * sz)
        if best is None or d < best[0]:
            best = (d, walked + u * seg, i, u)
        walked += seg
    if best is None:
        return None
    d, s, i, u = best
    if not 5.0 <= s <= 150.0 or (i == len(path) - 2 and u >= 0.999):
        return None
    h0 = math.atan2(path[start + 1][2] - path[start][2], path[start + 1][1] - path[start][1])
    h1 = math.atan2(path[i + 1][2] - path[i][2], path[i + 1][1] - path[i][1])
    bend = abs(math.degrees(math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))))
    cls = "lane" if d <= 1.5 else "adjacent" if 3.0 <= d <= 6.0 else "outer" if d > 6.0 else None
    return None if cls is None else (cls, bend)


def eval_clip(path_str: str) -> dict:
    from core.aeb.clip_store import deserialize_clip
    from tools.map_replay import ClipMap

    try:
        clip = deserialize_clip(Path(path_str).read_bytes())
    except Exception:
        return {"error": "unreadable"}
    cid = (clip.metadata.clip_id or "")[:8]
    clip_map = ClipMap(clip)
    if clip_map.window is None:
        return {"clip": cid, "error": "off map"}
    try:
        truth = _truth_path(clip)
        runs = {"base": _replay(clip, None), "map": _replay(clip, clip_map)}
    except Exception as exc:  # an odd clip must not stop the run
        return {"clip": cid, "error": f"replay {type(exc).__name__}"}
    counts: dict[str, int] = {}
    for name, frames in runs.items():
        for t, ex, ez, speed, rows in frames:
            if speed < 3.0:
                continue
            for vid, x, z, vs, in_path, locked in rows:
                cls = _classify(truth, t, ex, ez, x, z)
                if cls is None:
                    continue
                motion = "moving" if vs >= 1.0 else "stationary"
                bend = next(f"{lo}-{hi}" for lo, hi in _BENDS if lo <= cls[1] < hi)
                key = f"{name}|{cls[0]}|{motion}|{bend}"
                counts[key + "|n"] = counts.get(key + "|n", 0) + 1
                counts[key + "|in"] = counts.get(key + "|in", 0) + int(in_path)
                counts[key + "|lock"] = counts.get(key + "|lock", 0) + int(locked)
    return {"clip": cid, "session": clip.metadata.session_kind, "counts": counts}


def report(results: list[dict]) -> None:
    ok = [r for r in results if "counts" in r]
    print(f"clips {len(results)} usable {len(ok)}")
    total: dict[str, int] = {}
    for r in ok:
        for k, v in r["counts"].items():
            total[k] = total.get(k, 0) + v
    for cls, what in (("lane", "recall"), ("adjacent", "false lock"), ("outer", "false lock")):
        for motion in ("moving", "stationary"):
            print(f"\n{cls} {motion} ({what}): in_path% / locked% , base -> map")
            for lo, hi in _BENDS:
                bend = f"{lo}-{hi}"
                parts = []
                for name in ("base", "map"):
                    key = f"{name}|{cls}|{motion}|{bend}"
                    n = total.get(key + "|n", 0)
                    if not n:
                        parts.append("n=0")
                        continue
                    parts.append(f"{100 * total[key + '|in'] / n:5.1f} / {100 * total[key + '|lock'] / n:5.1f} (n={n})")
                print(f"  bend {bend:7s} " + "  ->  ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default="both", choices=("local", "contributed", "both"))
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default=str(_OUT / "acc_eval.json"))
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.report:
        report(json.loads(Path(args.report).read_text(encoding="utf-8")))
        return 0
    from tools.map_aeb_ab import _init_worker, _paths

    paths = _paths(args.store)
    if args.sample and args.sample < len(paths):
        random.Random(17).shuffle(paths)
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
