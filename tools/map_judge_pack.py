"""Evidence pack for an independent safety/comfort review of AEB with road-bent paths.

For each clip whose decisions changed in ``tools/map_aeb_ab.py``, replays AEB
without and with the local map, keeps every tick's snapshot, and writes:

- ``<clip>/summary.json``: label, verdicts, first warn/brake per variant, ticks.
- ``<clip>/t<time>.png``: two panels at a moment the variants disagree: left today,
  right road-bent. Lanes, ego's predicted path, every vehicle body, the colliding
  ones in red, warn/brake state, and ego's real later path in white.
- ``<clip>/thumb.jpg``: the game screenshot at the trigger, when the clip has one.

    python tools/map_judge_pack.py tools/map_lane_audit_out/aeb_bend_ab.json --out tools/map_lane_audit_out/judge
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_VIEW_M = 90.0


def _init_worker() -> None:
    logging.getLogger("ui.popup.popup_window").setLevel(logging.CRITICAL)


def _capture(clip, with_map: bool):
    """Per tick ``(t_rel, warn, brake, required, snapshot)`` for one replay."""
    import core.aeb.clip_eval as ce
    from core.aeb.thread import AEBThread
    from tools.map_replay import ClipMap

    rows = []
    original = AEBThread.loop

    def spy(self):
        original(self)
        rows.append((bool(self.data.AEB_warn), bool(self.data.AEB_brake),
                     float(self.data.AEB_required_decel_ms2), self.data.snapshot))

    AEBThread.loop = spy
    try:
        ce.REPLAY_MAP_FACTORY = ClipMap if with_map else None
        ticks = ce.run_headless(clip)
    finally:
        ce.REPLAY_MAP_FACTORY = None
        AEBThread.loop = original
    return [(tk.t_rel, w, b, req, snap) for tk, (w, b, req, snap) in zip(ticks, rows)]


def _moments(base, mapped, burn):
    """Up to three tick indices where the variants first disagree, or first act."""
    picks = []
    for i, (b, m) in enumerate(zip(base, mapped)):
        if b[0] < burn:
            continue
        if (b[1], b[2]) != (m[1], m[2]):
            picks.append(i)
            break
    for rows in (base, mapped):
        first = next((i for i, r in enumerate(rows) if r[0] >= burn and (r[1] or r[2])), None)
        if first is not None:
            picks.append(first)
    out = []
    for i in sorted(set(picks)):
        if all(abs(i - j) > 8 for j in out):
            out.append(i)
    return out[:3]


def _panel(ax, snap, warn, brake, req, title, window, truth, lanes: bool):
    ex, ez, ey = snap.ego_x, snap.ego_z, snap.ego_yaw
    fx, fz = -math.sin(ey), -math.cos(ey)

    def local(wx, wz):
        dx, dz = wx - ex, wz - ez
        return dx * (-fz) + dz * fx, dx * fx + dz * fz

    ax.set_facecolor("#111118")
    if window is not None and lanes:
        for lane in window.lanes:
            pts = [local(x, z) for x, z in lane.pts]
            if not any(abs(a) < _VIEW_M and -30 < b < _VIEW_M * 1.6 for a, b in pts):
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=":",
                    color="#e09a45" if lane.kind == "road" else "#6aa8e0", lw=0.6, alpha=0.6)
            if lane.kind == "road":
                for side in (-2.25, 2.25):
                    edge = _offset(lane.pts, side)
                    ep = [local(x, z) for x, z in edge]
                    ax.plot([p[0] for p in ep], [p[1] for p in ep], color="#8a8fa8", lw=0.6, alpha=0.7)
    pts = [local(x, z) for x, z in truth]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="white", lw=1.4, label="ego real later path")
    if snap.ego_arc is not None:
        left, right = snap.ego_arc.sample_corridor(40)
        for side in (left, right):
            sp = [local(x, z) for x, z in side]
            ax.plot([p[0] for p in sp], [p[1] for p in sp], color="#46aaff", lw=1.0)
    road = getattr(snap, "ego_road_arc", None)
    if road is not None:
        rp = [local(*road.position_at_dist(road.arc_length * i / 40.0)) for i in range(41)]
        ax.plot([p[0] for p in rp], [p[1] for p in rp], color="#ebd75a", lw=1.2, ls="--")
    for v in snap.vehicles:
        vx, vz, yaw = v["x"], v["z"], v["yaw"]
        hl, hw = v["length"] / 2.0, v["half_w"]
        vfx, vfz = -math.sin(yaw), -math.cos(yaw)
        corners = []
        for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw), (hl, hw)):
            corners.append(local(vx + a * vfx - b * vfz, vz + a * vfz + b * vfx))
        hit = v["vid"] in snap.colliding_ids
        ax.plot([c[0] for c in corners], [c[1] for c in corners],
                color="#ff4040" if hit else "#50d282", lw=1.6 if hit else 1.0)
        ax.text(corners[0][0], corners[0][1], f'{v["vid"]} {v.get("speed_kmh", 0):.0f}',
                color="#ff4040" if hit else "#a0e0b0", fontsize=6, clip_on=True)
    state = "BRAKE" if brake else ("WARN" if warn else "quiet")
    ax.set_title(f"{title}: {state}, demand {req:.1f} m/s2", fontsize=8,
                 color="#ff6060" if brake else ("#ffc040" if warn else "#d0d0d0"))
    ax.set_xlim(-_VIEW_M * 0.6, _VIEW_M * 0.6)
    ax.set_ylim(-25, _VIEW_M * 1.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def _offset(pts, right: float):
    out = []
    for i in range(len(pts)):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, len(pts) - 1)]
        hx, hz = b[0] - a[0], b[1] - a[1]
        n = math.hypot(hx, hz)
        out.append(pts[i] if n < 1e-9 else (pts[i][0] - right * hz / n, pts[i][1] + right * hx / n))
    return out


def pack_clip(path_str: str, out_root: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.aeb.clip_store import deserialize_clip
    from tools.map_replay import match_clip

    clip = deserialize_clip(Path(path_str).read_bytes())
    meta = clip.metadata
    cid = (meta.clip_id or "")[:8]
    out = Path(out_root) / cid
    out.mkdir(parents=True, exist_ok=True)
    _game, window = match_clip(clip)
    base = _capture(clip, False)
    mapped = _capture(clip, True)
    burn = meta.burn_in_s
    frames = sorted((f for f in clip.radar_frames if not f.ego.paused), key=lambda f: f.t_mono)
    t0 = min([f.t_mono for f in clip.radar_frames] + [tk.t_mono for tk in clip.aeb_ticks])

    def first(rows, k):
        return next((round(r[0], 2) for r in rows if r[0] >= burn and r[k]), None)

    label = meta.label
    # The reviewer sees the scene only; the human label goes to a separate file for scoring.
    private = {
        "clip": cid,
        "label": label.class_ if label else None,
        "label_window": label.should_trigger if label else None,
        "label_severity": label.severity if label else None,
        "label_target": label.target_vid if label else None,
        "label_notes": (label.notes or "")[:300] if label else "",
    }
    summary = {
        "clip": cid, "session": meta.session_kind,
        "speed_kmh_at_start": round(frames[0].ego.speed * 3.6) if frames else None,
        "today": {"first_warn": first(base, 1), "first_brake": first(base, 2),
                  "brake_ticks": sum(r[2] for r in base if r[0] >= burn),
                  "warn_ticks": sum(r[1] for r in base if r[0] >= burn)},
        "road_bent": {"first_warn": first(mapped, 1), "first_brake": first(mapped, 2),
                      "brake_ticks": sum(r[2] for r in mapped if r[0] >= burn),
                      "warn_ticks": sum(r[1] for r in mapped if r[0] >= burn)},
        "panels": [],
    }
    for i in _moments(base, mapped, burn):
        t_rel = base[i][0]
        t_abs = t0 + t_rel
        truth = [(f.ego.coordinateX, f.ego.coordinateZ) for f in frames if f.t_mono >= t_abs - 0.05][:400]
        fig, axes = plt.subplots(1, 2, figsize=(10, 7.5), dpi=100)
        _panel(axes[0], base[i][4], base[i][1], base[i][2], base[i][3], f"today t={t_rel:.2f}s",
               window, truth, True)
        _panel(axes[1], mapped[i][4], mapped[i][1], mapped[i][2], mapped[i][3], f"road-bent t={t_rel:.2f}s",
               window, truth, True)
        fig.suptitle(f"{cid} {meta.session_kind} ego {base[i][4].ego_speed * 3.6:.0f} km/h. Blue: predicted ego "
                     f"corridor. Yellow dashed: road path. Red: flagged colliding. White: where ego really "
                     f"went. Grey: lane edges. Labels: id and km/h", fontsize=8, color="#d0d0d0")
        name = f"t{t_rel:05.2f}.png"
        fig.savefig(out / name, bbox_inches="tight", facecolor="#1a1a22")
        plt.close(fig)
        summary["panels"].append(name)
    if meta.thumbnail_jpeg:
        (out / "thumb.jpg").write_bytes(base64.b64decode(meta.thumbnail_jpeg))
        summary["thumbnail"] = "thumb.jpg"
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (Path(out_root) / "_labels").mkdir(exist_ok=True)
    (Path(out_root) / "_labels" / f"{cid}.json").write_text(json.dumps(private, indent=1), encoding="utf-8")
    return summary


def _select(rows: list[dict], cap: int) -> list[str]:
    """Changed clips, ordered: braking removed or later, verdict flips, braking added, earlier, rest."""
    changed = [r for r in rows if "base" in r and r["base"] != r["map"]]

    def rank(r):
        b, m = r["base"], r["map"]
        bb, mb = b["first_brake"], m["first_brake"]
        if bb is not None and (mb is None or mb > bb + 0.05):
            return 0
        if r.get("base_verdict") != r.get("map_verdict"):
            return 1
        if bb is None and mb is not None:
            return 2
        if bb is not None and mb is not None and mb < bb - 0.3:
            return 3
        return 4

    ordered = sorted(changed, key=lambda r: (rank(r), r["clip"]))
    return [r["clip"] for r in (ordered[:cap] if cap else ordered)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ab_json")
    parser.add_argument("--out", default=str(ROOT / "tools" / "map_lane_audit_out" / "judge"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max", type=int, default=0, help="cap the pack, most safety-relevant first")
    args = parser.parse_args()
    rows = json.loads(Path(args.ab_json).read_text(encoding="utf-8"))
    wanted = set(_select(rows, args.max))
    from tools.map_aeb_ab import _paths

    paths = [p for p in _paths("both") if any(cid in Path(p).name for cid in wanted)]
    done = []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(pack_clip, p, args.out) for p in paths]
        for fut in as_completed(futures):
            try:
                done.append(fut.result())
            except Exception as exc:
                print("failed:", type(exc).__name__, exc)
    index = sorted(done, key=lambda d: d["clip"])
    Path(args.out, "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"packed {len(done)} clips into {Path(args.out).name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
