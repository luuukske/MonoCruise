"""Ego body and path origin from the SDK wheel layout. See core/radar/README.md §17.

    python tools/ego_geometry_probe.py            live truck (game running)
    python tools/ego_geometry_probe.py --json     same, machine-readable
    python tools/ego_geometry_probe.py --clips    path-origin evidence from the local clip store

Never imports core.settings.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.radar.ego_geometry import (  # noqa: E402
    REF_HALF_LENGTH_M, REF_HALF_WIDTH_M, EgoGeometry, geometry_from_sdk,
)

# Reference rig's rear-wheel mean, used for the clip evidence when no game is up.
_REF_PATH_ORIGIN_M = -2.093
_SPEED_BANDS = ((2, 5), (5, 10), (10, 20), (20, 40))
_MIN_KAPPA = 0.004


def _live() -> tuple[dict, EgoGeometry | None] | None:
    try:
        import truck_telemetry
        truck_telemetry.init()
        raw = truck_telemetry.get_data()
    except Exception as exc:
        print(f"SDK not readable ({exc}); is the game running?", file=sys.stderr)
        return None
    return raw, geometry_from_sdk(raw)


def _print_live(raw: dict, g: EgoGeometry | None, as_json: bool) -> None:
    n = int(raw.get("truckWheelCount", 0) or 0)
    wheels = [
        {
            "x": raw["truckWheelPositionX"][i], "z": raw["truckWheelPositionZ"][i],
            "radius": raw["truckWheelRadius"][i], "steerable": raw["truckWheelSteerable"][i],
            "powered": raw["truckWheelPowered"][i], "lift": raw["truck_wheelLift"][i],
        }
        for i in range(n)
    ]
    if as_json:
        print(json.dumps({
            "truck_id": raw.get("truckId", ""), "wheels": wheels,
            "geometry": None if g is None else {
                "front_m": g.front_m, "rear_m": g.rear_m, "length_m": g.front_m + g.rear_m,
                "half_width_m": g.half_width_m, "path_origin_m": g.path_origin_m,
                "front_delta_m": g.front_delta_m,
            },
        }, indent=2))
        return
    print(f"truck {raw.get('truckId', '')}  wheels {n}  (vehicle space: x right, z back)")
    for i, w in enumerate(wheels):
        print(f"  {i:>2}  x {w['x']:+.3f}  z {w['z']:+.3f}  r {w['radius']:.3f}"
              f"  steer {int(w['steerable'])}  powered {int(w['powered'])}  lift {w['lift']:.2f}")
    if g is None:
        print("layout unreadable: AEB uses the calibration body, ACC its legacy arc")
        return
    print(f"length {g.front_m + g.rear_m:.3f} m  (front {g.front_m:.3f}, rear {g.rear_m:.3f};"
          f" calibration {2 * REF_HALF_LENGTH_M:.3f})")
    print(f"width  {2 * g.half_width_m:.3f} m  (calibration {2 * REF_HALF_WIDTH_M:.3f})")
    print(f"path origin {-g.path_origin_m:.3f} m behind placement; ACC gap reference"
          f" {g.front_delta_m:+.3f} m vs the reference rig")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _poses(path: str) -> list:
    try:
        d = json.loads(gzip.decompress(Path(path).read_bytes()))
    except Exception:
        return []
    out, last = [], None
    for f in sorted(d.get("radar_frames", []), key=lambda f: f.get("t_mono", 0.0)):
        e = f.get("ego", {})
        if e.get("paused"):
            out.append(None)
            last = None
            continue
        p = (float(e.get("coordinateX", 0.0)), float(e.get("coordinateZ", 0.0)),
             float(e.get("rotationX", 0.0)) * 2.0 * math.pi, float(e.get("speed", 0.0)),
             float(e.get("userSteer", 0.0)), float(f.get("t_mono", 0.0)))
        if last is None or p[:3] != last[:3]:
            out.append(p)
            last = p
    return out


def _clip_rows(args: tuple[str, float]) -> tuple[list, list]:
    """(sideslip rows, ACC-arc miss rows) for one clip."""
    from core.acc.ego_path import blend_curvature
    from core.acc.tracker import ACCTracker
    from core.radar.ego_path import ego_curvature_from_history
    from core.radar.traffic import build_arc

    path, origin = args
    poses = _poses(path)
    slip, miss, hist = [], [], []
    for i, a in enumerate(poses):
        if a is None:
            hist = []
            continue
        hist = (hist + [(a[5], a[0], a[1])])[-25:]
        if 0 < i < len(poses) - 1 and poses[i - 1] and poses[i + 1] and a[3] >= 2.0:
            b, c = poses[i - 1], poses[i + 1]
            dx, dz = c[0] - b[0], c[1] - b[1]
            ds = math.hypot(dx, dz)
            if 0.05 < ds < 6.0:
                fx, fz = -math.sin(a[2]), -math.cos(a[2])
                slip.append((a[3], _wrap(c[2] - b[2]) / ds, (fx * dz - fz * dx) / ds))
        if i % 3 or a[3] < 2.0 or len(hist) < 10:
            continue
        kappa = blend_curvature(a[4], ego_curvature_from_history(hist), a[3])
        s_acc, prev = 0.0, a
        for b in poses[i + 1:]:
            if b is None or math.hypot(b[0] - prev[0], b[1] - prev[1]) > 6.0:
                break
            s_acc += math.hypot(b[0] - prev[0], b[1] - prev[1])
            prev = b
            if s_acc >= 10.0:
                lats = []
                for f in (0.0, origin):
                    sx, sz = a[0] - f * math.sin(a[2]), a[1] - f * math.cos(a[2])
                    arc = build_arc(sx, sz, a[2], max(a[3], 0.1), kappa, 1.25, 2.5)
                    px, pz = b[0] - f * math.sin(b[2]), b[1] - f * math.cos(b[2])
                    lats.append(ACCTracker._project_onto_arc(
                        arc, px, pz, -math.sin(a[2]), -math.cos(a[2]))[1])
                miss.append((a[3], _wrap(b[2] - a[2]) / s_acc, lats))
                break
    return slip, miss


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else math.nan


def _clips(origin: float) -> None:
    from core.aeb.clip_store import default_clip_root

    paths = [str(p) for p in default_clip_root().glob("*.json.gz")]
    slip, miss = [], []
    with ProcessPoolExecutor() as ex:
        for s, m in ex.map(_clip_rows, [(p, origin) for p in paths], chunksize=8):
            slip.extend(s)
            miss.extend(m)
    print(f"{len(paths)} clips; path origin under test {-origin:.3f} m behind placement")
    print("zero-sideslip point (m ahead of placement, median) by speed:")
    for lo, hi in _SPEED_BANDS:
        sel = [r[2] / r[1] for r in slip if lo <= r[0] < hi and abs(r[1]) >= _MIN_KAPPA]
        print(f"  {lo:>2}-{hi:<2} m/s  n {len(sel):>6}  {_median(sel):+.3f}")
    print("ACC arc, 10 m ahead in bends, origin -> path origin:")
    print("  (bias: median signed miss, sign flipped with the bend direction)")
    for lo, hi in _SPEED_BANDS:
        sel = [r for r in miss if lo <= r[0] < hi and abs(r[1]) >= _MIN_KAPPA]
        miss_o = _median([abs(r[2][0]) for r in sel])
        miss_p = _median([abs(r[2][1]) for r in sel])
        bias_o = _median([r[2][0] * (1.0 if r[1] > 0 else -1.0) for r in sel])
        bias_p = _median([r[2][1] * (1.0 if r[1] > 0 else -1.0) for r in sel])
        print(f"  {lo:>2}-{hi:<2} m/s  n {len(sel):>6}  |miss| {miss_o:.3f} -> {miss_p:.3f} m"
              f"  bias {bias_o:+.3f} -> {bias_p:+.3f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--clips", action="store_true")
    args = ap.parse_args()
    if args.clips:
        live = _live()
        g = live[1] if live else None
        _clips(g.path_origin_m if g is not None else _REF_PATH_ORIGIN_M)
        return
    live = _live()
    if live is None:
        sys.exit(1)
    _print_live(*live, as_json=args.json)


if __name__ == "__main__":
    main()
