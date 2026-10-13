"""How far real vehicles drive from our lane centerlines, per spline variant.

AI traffic drives the game's nav lanes, so an AI trail that sits off our line
is our reconstruction's error. TMP players and ego follow the painted road,
which can differ from the nav lane: comparing the two separates our bugs from
SCS's own data. Clips carry world poses; the map comes from the local install.

    python tools/map_lane_audit.py --sample 300 --workers 8
    python tools/map_lane_audit.py --report tools/map_lane_audit_out/audit.jsonl

Writes one JSON line per clip to ``--out`` and prints residual tables. Output
holds clip ids and residuals only, never world coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = {
    "chord": ("chord", "chord"),
    "stored": ("stored", "chord"),
    "circle": ("circle", "circle"),
}
_SAMPLE_STEP_M = 5.0
_MIN_SPEED_MS = 2.0
_MATCH_M = 3.5
_MATCH_ANG = math.radians(25.0)
_CELL_M = 25.0
_OFF_MAP_M = 6.0
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


def _trails(clip):
    """Ego and per-vehicle trails: ``(t, x, z, speed)``; vehicles also carry ``is_tmp``."""
    from core.aeb.clip_replay import decode_radar_stream
    from core.aeb.clip_timebase import replay_frames

    frames = replay_frames(clip)
    veh_by_t, ego_by_t, frame_t, _off = decode_radar_stream(clip, frames=frames, with_elevation=False)
    ego: list[tuple[float, float, float, float]] = []
    cars: dict[int, list] = defaultdict(list)
    tmp_ids: set[int] = set()
    for t in frame_t:
        e = ego_by_t.get(t)
        if e is None or e.paused:
            continue
        ego.append((t, e.coordinateX, e.coordinateZ, abs(e.speed)))
        for v in veh_by_t.get(t, []):
            if v.is_trailer or v.is_parked or v.id < 0:
                continue
            cars[v.id].append((t, v.position.x, v.position.z, abs(v.speed)))
            if v.is_tmp:
                tmp_ids.add(v.id)
    return ego, dict(cars), tmp_ids


def _thin(trail):
    """Points about one sample step apart, with travel direction, moving only."""
    out = []
    last = None
    for i, (t, x, z, speed) in enumerate(trail):
        if speed < _MIN_SPEED_MS:
            continue
        if last is not None and math.hypot(x - last[0], z - last[1]) < _SAMPLE_STEP_M:
            continue
        heading = _direction(trail, i)
        if heading is None:
            continue
        out.append((x, z, heading[0], heading[1]))
        last = (x, z)
    return out


def _direction(trail, i):
    lo = max(0, i - 6)
    hi = min(len(trail) - 1, i + 6)
    dx = trail[hi][1] - trail[lo][1]
    dz = trail[hi][2] - trail[lo][2]
    n = math.hypot(dx, dz)
    if n < 2.0:
        return None
    return dx / n, dz / n


class _Index:
    """Lane segments in a grid, each with its line's metadata."""

    def __init__(self, lines, road_meta) -> None:
        self.grid: dict[tuple[int, int], list] = defaultdict(list)
        for line in lines:
            pts = line.points
            meta = road_meta.get(line.uid) if line.kind == "road" else None
            n = len(pts) - 1
            for i in range(n):
                a = pts[i]
                b = pts[i + 1]
                seg = (a[0], a[2], b[0], b[2], line.kind, meta, (i + 0.5) / max(n, 1), line.lane_count)
                for key in self._cells(a[0], a[2], b[0], b[2]):
                    self.grid[key].append(seg)

    @staticmethod
    def _cells(ax, az, bx, bz):
        x0, x1 = sorted((ax, bx))
        z0, z1 = sorted((az, bz))
        for ix in range(math.floor(x0 / _CELL_M), math.floor(x1 / _CELL_M) + 1):
            for iz in range(math.floor(z0 / _CELL_M), math.floor(z1 / _CELL_M) + 1):
                yield ix, iz

    def match(self, x, z, hx, hz):
        """Signed offset (right of travel) to the nearest lane heading the same way."""
        best = None
        cx, cz = math.floor(x / _CELL_M), math.floor(z / _CELL_M)
        for ix in (cx - 1, cx, cx + 1):
            for iz in (cz - 1, cz, cz + 1):
                for seg in self.grid.get((ix, iz), ()):
                    ax, az, bx, bz = seg[0], seg[1], seg[2], seg[3]
                    sx, sz = bx - ax, bz - az
                    length2 = sx * sx + sz * sz
                    if length2 < 1e-9:
                        continue
                    length = math.sqrt(length2)
                    tx, tz = sx / length, sz / length
                    dot = tx * hx + tz * hz
                    if abs(dot) < math.cos(_MATCH_ANG):
                        continue
                    t = max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / length2))
                    px, pz = ax + t * sx, az + t * sz
                    dist = math.hypot(x - px, z - pz)
                    if best is not None and dist >= best[0]:
                        continue
                    right = (x - px) * (-hz) + (z - pz) * hx
                    best = (dist, right, seg)
        return best


def _road_meta(game_map, window):
    """Per road uid: (turn deg, chord m, stored length m)."""
    from core.road_layout.centerline import node_forward_xz

    meta = {}
    for key in window.sectors:
        part = game_map.sector(key)
        nodes = part.nodes
        for road in part.roads:
            start = nodes.get(road.start_uid)
            end = nodes.get(road.end_uid)
            if start is None or end is None:
                continue
            f0 = node_forward_xz(start)
            f1 = node_forward_xz(end)
            turn = abs(math.degrees(math.atan2(f0[0] * f1[1] - f0[1] * f1[0], f0[0] * f1[0] + f0[1] * f1[1])))
            chord = math.hypot(end.x - start.x, end.z - start.z)
            meta[road.uid] = (turn, chord, road.length)
    return meta


def audit_clip(path_str: str) -> dict:
    from core.aeb.clip_store import ClipStore

    path = Path(path_str)
    clip = ClipStore(path.parent).load(path)
    if clip is None or not clip.radar_frames:
        return {"clip": path.name[:60], "error": "unreadable"}
    cid = (clip.metadata.clip_id or path.name)[:8]
    session = clip.metadata.session_kind
    try:
        ego, cars, tmp_ids = _trails(clip)
    except Exception as exc:  # replay of an odd clip must not stop the run
        return {"clip": cid, "error": f"replay {type(exc).__name__}"}
    ego_pts = _thin(ego)
    if len(ego_pts) < 3:
        return {"clip": cid, "error": "ego not moving"}
    xs = [p[0] for p in ego_pts] + [p[1] for tr in cars.values() for p in tr]
    zs = [p[1] for p in ego_pts] + [p[2] for tr in cars.values() for p in tr]
    box = (min(p[0] for p in ego_pts) - 250, max(p[0] for p in ego_pts) + 250,
           min(p[1] for p in ego_pts) - 250, max(p[1] for p in ego_pts) + 250)
    del xs, zs
    out = {"clip": cid, "session": session, "game": None, "variants": {}}
    for game in ("ets2", "ats"):
        game_map = _game_map(game)
        if game_map is None:
            continue
        window = game_map.lanes_in_box(*box, road_tangent="chord")
        index = _Index(window.lines, {})
        dists = [m[0] if (m := index.match(*p)) else 99.0 for p in ego_pts]
        if statistics.median(dists) <= _OFF_MAP_M:
            out["game"] = game
            break
    if out["game"] is None:
        out["error"] = "off map"
        return out
    game_map = _game_map(out["game"])
    car_pts = {vid: _thin(tr) for vid, tr in cars.items()}
    for name, (road_mode, prefab_mode) in VARIANTS.items():
        window = game_map.lanes_in_box(*box, road_tangent=road_mode, prefab_tangent=prefab_mode)
        index = _Index(window.lines, _road_meta(game_map, window))
        rows = []
        for who, pts in [("ego", ego_pts)] + [
            ("tmp" if vid in tmp_ids else "ai", p) for vid, p in car_pts.items()
        ]:
            for x, z, hx, hz in pts:
                hit = index.match(x, z, hx, hz)
                if hit is None or hit[0] > _MATCH_M:
                    rows.append((who, None))
                    continue
                dist, right, seg = hit
                meta = seg[5]
                rows.append((who, (
                    round(right, 3), seg[4],
                    None if meta is None else round(meta[0], 1),
                    None if meta is None else round(meta[1], 1),
                    round(seg[6], 2), seg[7],
                )))
        out["variants"][name] = rows
    return out


def _clip_paths(store: str) -> list[str]:
    from core.aeb.clip_store import ClipStore, contributed_clip_root, default_clip_root

    roots = []
    if store in ("local", "both"):
        roots.append(default_clip_root())
    if store in ("contributed", "both"):
        roots.append(contributed_clip_root())
    paths: list[str] = []
    for root in roots:
        paths.extend(str(info.path) for info in ClipStore(root).list_clips())
    return paths


def _quantiles(values):
    if not values:
        return "n=0"
    values = sorted(values)

    def q(p):
        return values[min(len(values) - 1, int(p * len(values)))]

    return f"n={len(values):6d}  p50 {q(0.5):.3f}  p90 {q(0.9):.3f}  p95 {q(0.95):.3f}  p99 {q(0.99):.3f}"


def _turn_bin(turn):
    for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 120), (120, 181)):
        if lo <= turn < hi:
            return f"{lo}-{hi}"
    return "?"


def _chord_bin(chord):
    for lo, hi in ((0, 50), (50, 150), (150, 400), (400, 10_000)):
        if lo <= chord < hi:
            return f"{lo}-{hi}"
    return "?"


def report(results: list[dict]) -> None:
    ok = [r for r in results if not r.get("error")]
    errors = defaultdict(int)
    for r in results:
        if r.get("error"):
            errors[r["error"]] += 1
    games = defaultdict(int)
    for r in ok:
        games[(r["game"], r["session"])] += 1
    print(f"clips {len(results)}  usable {len(ok)}  errors {dict(errors)}  by game/session {dict(games)}")
    for name in VARIANTS:
        print(f"\n== variant {name} ==")
        by = defaultdict(list)
        signed = defaultdict(list)
        unmatched = defaultdict(int)
        total = defaultdict(int)
        for r in ok:
            for who, row in r["variants"].get(name, []):
                total[who] += 1
                if row is None:
                    unmatched[who] += 1
                    continue
                right, kind, turn, chord, pos, _count = row
                by[(who, kind)].append(abs(right))
                if kind == "road" and turn is not None and turn >= 10 and 0.25 <= pos <= 0.75:
                    by[(who, "road", _turn_bin(turn), _chord_bin(chord))].append(abs(right))
                    signed[(who, _turn_bin(turn))].append(right)
        for who in ("ai", "tmp", "ego"):
            share = unmatched[who] / total[who] if total[who] else 0.0
            print(f"{who:4s} unmatched {share:.1%}")
            for kind in ("road", "prefab"):
                print(f"  {who:4s} {kind:6s} |res| {_quantiles(by[(who, kind)])}")
        print("  mid-span on curved roads, |res| by turn x chord (ai | ego):")
        for tb in ("10-30", "30-60", "60-120", "120-181"):
            for cb in ("0-50", "50-150", "150-400", "400-10000"):
                ai = by.get(("ai", "road", tb, cb), [])
                eg = by.get(("ego", "road", tb, cb), [])
                if not ai and not eg:
                    continue
                print(f"    turn {tb:7s} chord {cb:9s} ai {_quantiles(ai)}")
                print(f"    {'':7s}      {'':9s} ego {_quantiles(eg)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default="both", choices=("local", "contributed", "both"))
    parser.add_argument("--sample", type=int, default=0, help="random clips, 0 = all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(_OUT / "audit.jsonl"))
    parser.add_argument("--report", help="only print the tables for an existing jsonl")
    args = parser.parse_args()
    if args.report:
        rows = [json.loads(line) for line in Path(args.report).read_text(encoding="utf-8").splitlines() if line]
        report(rows)
        return 0
    paths = _clip_paths(args.store)
    if args.sample and args.sample < len(paths):
        random.Random(args.seed).shuffle(paths)
        paths = paths[:args.sample]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out_path.open("w", encoding="utf-8") as fh, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(audit_clip, p) for p in paths]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as exc:
                res = {"clip": "?", "error": f"worker {type(exc).__name__}"}
            results.append(res)
            fh.write(json.dumps(res) + "\n")
            if n % 50 == 0:
                print(f"{n}/{len(paths)}", flush=True)
    report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
