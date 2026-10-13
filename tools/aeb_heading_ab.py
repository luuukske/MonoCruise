"""AEB before and after the curved-arc heading fix, over the clip corpus (no map).

``ArcPath.heading_at_dist`` used to add the swept angle, which turns opposite to
the yaw convention, so every body on a curved arc was mirrored. The base run
patches the old formula back in; the candidate is the tree as it is.

    python tools/aeb_heading_ab.py --workers 20
    python tools/aeb_heading_ab.py --report tools/map_lane_audit_out/heading_ab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.map_aeb_ab import _init_worker, _paths, _summary, report  # noqa: E402

_OUT = ROOT / "tools" / "map_lane_audit_out"


def _old_heading(self, dist: float) -> float:
    if self.is_straight:
        return self.yaw_rad
    dist = max(0.0, min(dist, self.arc_length))
    frac = dist / self.arc_length if self.arc_length > 0 else 0.0
    return self.yaw_rad + frac * self.max_sweep


def ab_clip(path_str: str) -> dict:
    import core.aeb.clip_eval as clip_eval
    from core.aeb.clip_score import score_clip
    from core.aeb.clip_store import deserialize_clip
    from core.radar.traffic import ArcPath

    path = Path(path_str)
    try:
        clip = deserialize_clip(path.read_bytes())
    except Exception:
        return {"error": "unreadable"}
    meta = clip.metadata
    out = {"clip": (meta.clip_id or path.name)[:8], "session": meta.session_kind,
           "label": meta.label.class_ if meta.label else None, "game": "-", "fires": 0}
    fixed = ArcPath.heading_at_dist
    try:
        ArcPath.heading_at_dist = _old_heading
        base = clip_eval.run_headless(clip)
        base_score = score_clip(clip)
        ArcPath.heading_at_dist = fixed
        cand = clip_eval.run_headless(clip)
        cand_score = score_clip(clip)
    except Exception as exc:  # an odd clip must not stop the run
        out["error"] = f"replay {type(exc).__name__}"
        return out
    finally:
        ArcPath.heading_at_dist = fixed
    burn = meta.burn_in_s
    out["base"] = _summary(base, burn)
    out["map"] = _summary(cand, burn)
    out["fires"] = int(out["base"] != out["map"])
    if base_score is not None and cand_score is not None:
        out.update(base_verdict=base_score.verdict, map_verdict=cand_score.verdict,
                   base_cost=base_score.cost, map_cost=cand_score.cost)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=str(_OUT / "heading_ab.json"))
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.report:
        report(json.loads(Path(args.report).read_text(encoding="utf-8")))
        return 0
    rows = []
    paths = _paths("both")
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(ab_clip, p) for p in paths]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:
                rows.append({"error": f"worker {type(exc).__name__}"})
            if n % 200 == 0:
                print(f"{n}/{len(paths)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows), encoding="utf-8")
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
