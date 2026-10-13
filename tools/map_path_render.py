"""Draw one clip frame: map lanes, ego's route, the fused path, the steer arc, the truth.

Ego-centred, ego forward up. For inspecting where the map path goes wrong.

    python tools/map_path_render.py <clip id prefix> --frame 400 --out frame.png
    python tools/map_path_render.py <clip id prefix> --frames 100,300,500

Images stay local (``tools/map_lane_audit_out/`` by default, gitignored).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_OUT = ROOT / "tools" / "map_lane_audit_out"


def find_clip(prefix: str):
    from core.aeb.clip_store import ClipStore, contributed_clip_root, default_clip_root

    for root in (default_clip_root(), contributed_clip_root()):
        store = ClipStore(root)
        for info in store.list_clips():
            if prefix in info.name:
                return store.load(info.path)
    raise SystemExit(f"no clip matching {prefix}")


def render(clip, frame_indices, out_dir: Path, view_m: float = 160.0, game: str | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.aeb.calibration import DEFAULT as CAL, ego_path_params
    from core.aeb.clip_replay import ego_path_replay
    from core.aeb.clip_timebase import replay_frames
    from core.radar.ego_geometry import calibration_geometry
    from core.radar.ego_path_model import EgoPathModel
    from core.road_layout.ego_route import EgoRouter
    from core.road_layout.game_map import GameMap
    from core.road_layout.map_path import build_map_path, world_points

    frames = [f for f in replay_frames(clip) if not f.ego.paused]
    tkin_by_t, warm = ego_path_replay(clip, frames=frames)
    geo = frames[0].ego.geometry() or calibration_geometry(
        CAL.ego_half_length, CAL.ego_half_width, CAL.arc_start_pctg,
    )
    origin = geo.path_origin_m
    xs = [f.ego.coordinateX for f in frames]
    zs = [f.ego.coordinateZ for f in frames]
    box = (min(xs) - 300, max(xs) + 300, min(zs) - 300, max(zs) + 300)
    gm = GameMap.open_installed(game or "ets2")
    window = gm.lanes_in_box(*box)
    model = EgoPathModel(params=ego_path_params(CAL), gain=warm)
    router = EgoRouter()
    wanted = set(frame_indices)
    written: list[Path] = []
    cid = (clip.metadata.clip_id or "clip")[:8]
    for i, f in enumerate(frames):
        e = f.ego
        yaw = e.rotationX * 2.0 * math.pi
        fx, fz = -math.sin(yaw), -math.cos(yaw)
        t_kin = tkin_by_t.get(f.t_mono, 0.0)
        state = model.step(t_kin, yaw, e.speed, e.userSteer) if t_kin > 0.0 else model.state
        route = router.step(window, e.coordinateX, e.coordinateY, e.coordinateZ, fx, fz)
        if i not in wanted:
            continue
        ox, oz = e.coordinateX + origin * fx, e.coordinateZ + origin * fz

        def to_screen(wx, wz, ox=ox, oz=oz, fx=fx, fz=fz):
            dx, dz = wx - ox, wz - oz
            fwd = dx * fx + dz * fz
            right = dx * (-fz) + dz * fx
            return right, fwd

        fig, ax = plt.subplots(figsize=(7, 9), dpi=110)
        ax.set_facecolor("#111118")
        for lane in window.lanes:
            pts = [to_screen(x, z) for x, z in lane.pts]
            if not any(abs(a) < view_m and -40 < b < view_m * 1.4 for a, b in pts):
                continue
            color = "#e09a45" if lane.kind == "road" else "#6aa8e0"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=0.8, alpha=0.8)
        if route is not None:
            pts = [to_screen(x, z) for x, z in route.pts]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#bfeb82", lw=2.2, alpha=0.8, label="route")
            path = build_map_path(route, e.coordinateX, e.coordinateZ, fx, fz, state.kappa_path, e.speed,
                                  origin_m=origin)
            if path is not None:
                wp = world_points(path, e.coordinateX, e.coordinateZ, fx, fz, origin)
                pts = [to_screen(x, z) for x, z in wp]
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=list(path.trust), cmap="plasma",
                           vmin=0, vmax=1, s=6, zorder=5, label="fused path (trust)")
        kappa = state.kappa_path
        arc = []
        for k in range(0, 61):
            s = k * 2.5
            if abs(kappa) < 1e-6:
                arc.append((0.0, s))
            else:
                arc.append((-(1 - math.cos(kappa * s)) / kappa, math.sin(kappa * s) / kappa))
        ax.plot([p[0] for p in arc], [p[1] for p in arc], color="#46aaff", lw=1.2, ls="--", label="steer arc")
        future = []
        for f2 in frames[i:]:
            e2 = f2.ego
            yaw2 = e2.rotationX * 2.0 * math.pi
            future.append(to_screen(e2.coordinateX - origin * math.sin(yaw2), e2.coordinateZ - origin * math.cos(yaw2)))
        ax.plot([p[0] for p in future], [p[1] for p in future], color="white", lw=1.4, label="truth")
        ax.set_xlim(-view_m * 0.6, view_m * 0.6)
        ax.set_ylim(-30, view_m * 1.3)
        ax.set_aspect("equal")
        title = f"{cid} frame {i} v {e.speed * 3.6:.0f} km/h k {kappa:.4f}"
        if route is not None:
            title += f"\noffset {route.offset_m:.2f} m  {route.kind}  branch {route.branch_s}"
        ax.set_title(title, fontsize=9)
        ax.legend(loc="lower right", fontsize=7)
        out = out_dir / f"{cid}_{i:05d}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--frames", default="")
    parser.add_argument("--out-dir", default=str(_OUT / "frames"))
    parser.add_argument("--game", choices=("ets2", "ats"))
    args = parser.parse_args()
    indices = [args.frame] if args.frame is not None else [int(v) for v in args.frames.split(",") if v]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in render(find_clip(args.clip), indices, out_dir, game=args.game):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
