"""Check map centerlines against a truck pose.

Reads the 4 km sector under the pose and the ring around it, from the game's
own archives. Prints how far the pose sits from the nearest road centerline.
Does not change ACC or AEB.

Examples:
  python tools/road_layout_check.py
  python tools/road_layout_check.py --samples 40
  python tools/road_layout_check.py --x 3296 --y 5 --z 19626
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.aeb.clip_store import ClipStore
from core.road_layout.archive import HashFs
from core.road_layout.centerline import (
    nearest_centerline,
    node_forward_xz,
    roads_missing_a_node,
)
from core.road_layout.grid import (
    load_sectors,
    map_folder,
    neighborhood,
    sector_index,
)
from core.sdk_installer.game_paths import find_game_installations


def _open_map(game: Path) -> tuple[str, list[HashFs]]:
    opened: list[HashFs] = []
    for path in sorted(game.glob("*.scs")):
        try:
            opened.append(HashFs(path))
        except (OSError, ValueError):
            continue
    map_name = None
    for archive in opened:
        map_name = map_folder(archive)
        if map_name:
            break
    if map_name is None:
        raise SystemExit("no map folder in the game archives")
    kept = [
        archive for archive in opened
        if map_folder(archive) == map_name or archive.listdir(f"map/{map_name}") is not None
    ]
    if not kept:
        raise SystemExit("map archives have no sectors")
    return map_name, kept


def _pose_from_clip(index: int) -> tuple[float, float, float, str]:
    infos = ClipStore().list_clips()
    if not infos:
        raise SystemExit("no AEB clips, pass --x --y --z")
    if index < 0 or index >= len(infos):
        raise SystemExit(f"clip index {index} outside 0..{len(infos) - 1}")
    clip = ClipStore().load(infos[index].path)
    if clip is None or not clip.radar_frames:
        raise SystemExit("clip has no frames")
    ego = clip.radar_frames[len(clip.radar_frames) // 2].ego
    label = (clip.metadata.clip_id or infos[index].name)[:8]
    return ego.coordinateX, ego.coordinateY, ego.coordinateZ, label


def _report_one(mesh, x: float, y: float, z: float, label: str) -> None:
    hit = nearest_centerline(mesh, x, z)
    ix, iz = sector_index(x, z)
    print(
        f"{label} sector {ix:+d} {iz:+d}  files {mesh.sectors}  "
        f"roads {len(mesh.roads)}  missing-node {roads_missing_a_node(mesh)}"
    )
    print(f"  pose x {x:.1f}  z {z:.1f}  y {y:.2f}")
    if hit is None:
        print("  no centerline")
        return
    print(
        f"  centerline {hit.distance_m:.2f} m  road y {hit.road_y:.2f}  "
        f"dy {y - hit.road_y:+.2f}  look {hit.look}"
    )


def _report_samples(archives, map_name: str, count: int) -> None:
    infos = ClipStore().list_clips()
    if not infos:
        raise SystemExit("no AEB clips")
    dists: list[float] = []
    dys: list[float] = []
    far = 0
    agree_mesh = None
    for info in infos[:count]:
        clip = ClipStore().load(info.path)
        if clip is None or not clip.radar_frames:
            continue
        ego = clip.radar_frames[len(clip.radar_frames) // 2].ego
        ix, iz = sector_index(ego.coordinateX, ego.coordinateZ)
        mesh = load_sectors(archives, map_name, neighborhood(ix, iz))
        if agree_mesh is None:
            agree_mesh = mesh
        hit = nearest_centerline(mesh, ego.coordinateX, ego.coordinateZ)
        if hit is None:
            continue
        dists.append(hit.distance_m)
        dys.append(abs(ego.coordinateY - hit.road_y))
        if hit.distance_m > 8.0:
            far += 1
    if not dists or agree_mesh is None:
        raise SystemExit("no clip poses")
    start_dot, end_dot, n_roads = _forward_agreement(agree_mesh)
    dists.sort()
    dys.sort()
    print(
        f"{len(dists)} clips, mid frame, 3x3 sectors, map {map_name}, "
        f"{len(archives)} archives"
    )
    print(
        f"  xz m  p50 {statistics.median(dists):.2f}  "
        f"p90 {_pct(dists, 0.90):.2f}  max {dists[-1]:.2f}  over 8 m {far}"
    )
    print(
        f"  |dy| m  p50 {statistics.median(dys):.2f}  "
        f"p90 {_pct(dys, 0.90):.2f}  max {dys[-1]:.2f}"
    )
    print(
        f"  node ahead vs chord  start {start_dot:+.3f}  end {end_dot:+.3f}  "
        f"roads {n_roads}"
    )


def _forward_agreement(mesh) -> tuple[float, float, int]:
    start_dots: list[float] = []
    end_dots: list[float] = []
    for road in mesh.roads:
        start = mesh.nodes.get(road.start_uid)
        end = mesh.nodes.get(road.end_uid)
        if start is None or end is None:
            continue
        dx = end.x - start.x
        dz = end.z - start.z
        chord = math.hypot(dx, dz)
        if chord < 1.0:
            continue
        hx, hz = dx / chord, dz / chord
        sx, sz = node_forward_xz(start)
        ex, ez = node_forward_xz(end)
        start_dots.append(sx * hx + sz * hz)
        end_dots.append(ex * hx + ez * hz)
    if not start_dots:
        return 0.0, 0.0, 0
    return (
        sum(start_dots) / len(start_dots),
        sum(end_dots) / len(end_dots),
        len(start_dots),
    )


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a truck pose against map centerlines.")
    parser.add_argument("--game", choices=("ets2", "ats"), default="ets2")
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)
    parser.add_argument("--clip", type=int, default=0, help="Newest-first clip index")
    parser.add_argument("--samples", type=int, default=0, help="Summarise this many newest clips")
    parser.add_argument("--radius", type=int, default=1, help="Sectors around the pose, 1 is 3x3")
    args = parser.parse_args(argv)
    installs = find_game_installations(args.game)
    if not installs:
        raise SystemExit(f"no {args.game} install found")
    map_name, archives = _open_map(installs[0])
    if args.samples > 0:
        _report_samples(archives, map_name, args.samples)
        return 0
    if (args.x is None) != (args.z is None):
        raise SystemExit("pass both --x and --z")
    if args.x is None:
        x, y, z, label = _pose_from_clip(args.clip)
        if args.y is not None:
            y = args.y
    else:
        x, z = args.x, args.z
        y = 0.0 if args.y is None else args.y
        label = "pose"
    ix, iz = sector_index(x, z)
    mesh = load_sectors(archives, map_name, neighborhood(ix, iz, args.radius))
    print(f"map {map_name}  archives {len(archives)}")
    _report_one(mesh, x, y, z, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
