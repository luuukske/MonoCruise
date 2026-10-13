"""AEB with and without the local game map, over the clip corpus.

Runs ``run_headless`` twice per clip, the second time with ``tools/map_replay``
installed, and reports verdict and cost changes on labelled clips plus warn and
brake ticks the map removed or added everywhere. The corpus score is a
regression veto, not the objective (AGENTS.md, AEB measurement).

    python tools/map_aeb_ab.py --workers 16
    python tools/map_aeb_ab.py --report tools/map_lane_audit_out/aeb_ab.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_OUT = ROOT / "tools" / "map_lane_audit_out"


def _init_worker(overrides: dict | None = None) -> None:
    import logging

    logging.getLogger("ui.popup.popup_window").setLevel(logging.CRITICAL)
    if overrides:
        from core.aeb.calibration import DEFAULT

        # Every module holds the one DEFAULT instance, so set fields on it in place.
        for key, value in overrides.items():
            object.__setattr__(DEFAULT, key, value)


def _parse_set(items: list[str]) -> dict:
    from core.aeb.calibration import DEFAULT

    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        current = getattr(DEFAULT, key)
        if isinstance(current, bool):
            out[key] = raw.lower() in ("1", "true", "yes")
        elif isinstance(current, (int, float)):
            out[key] = type(current)(float(raw))
        else:
            out[key] = raw
    return out


def _score(clip, ticks):
    """Score a replay already run, without replaying it again inside the scorer."""
    import core.aeb.clip_eval as clip_eval
    from core.aeb.clip_score import score_clip

    original = clip_eval.run_headless
    clip_eval.run_headless = lambda *_a, **_k: ticks
    try:
        return score_clip(clip)
    finally:
        clip_eval.run_headless = original


def _summary(ticks, burn_in):
    evs = [e for e in ticks if e.t_rel >= burn_in]
    brake = [e.t_rel for e in evs if e.aeb_brake]
    warn = [e.t_rel for e in evs if e.aeb_warn]
    return {
        "brake_ticks": len(brake), "warn_ticks": len(warn),
        "first_brake": brake[0] if brake else None, "first_warn": warn[0] if warn else None,
    }


def ab_clip(path_str: str) -> dict:
    import core.aeb.clip_eval as clip_eval
    from core.aeb.clip_store import deserialize_clip
    from tools.map_replay import ClipMap, match_clip

    path = Path(path_str)
    try:
        clip = deserialize_clip(path.read_bytes())
    except Exception:
        return {"error": "unreadable"}
    meta = clip.metadata
    out = {"clip": (meta.clip_id or path.name)[:8], "session": meta.session_kind,
           "label": meta.label.class_ if meta.label else None}
    game, _window = match_clip(clip)
    out["game"] = game
    if game is None:
        out["error"] = "off map"
        return out
    burn = meta.burn_in_s
    try:
        clip_eval.REPLAY_MAP_FACTORY = None
        base = clip_eval.run_headless(clip)
        base_score = _score(clip, base)
        clip_eval.REPLAY_MAP_FACTORY = ClipMap
        mapped = clip_eval.run_headless(clip)
        map_score = _score(clip, mapped)
    except Exception as exc:  # an odd clip must not stop the run
        out["error"] = f"replay {type(exc).__name__}: {exc}"
        return out
    finally:
        clip_eval.REPLAY_MAP_FACTORY = None
    out["base"] = _summary(base, burn)
    out["map"] = _summary(mapped, burn)
    out["fires"] = int(out["base"] != out["map"])
    removed = 0
    added = 0
    for b, m in zip(base, mapped):
        removed += len(b.colliding_ids - m.colliding_ids)
        added += len(m.colliding_ids - b.colliding_ids)
    out["colliding_removed"] = removed
    out["colliding_added"] = added
    if base_score is not None and map_score is not None:
        out["base_verdict"] = base_score.verdict
        out["map_verdict"] = map_score.verdict
        out["base_cost"] = base_score.cost
        out["map_cost"] = map_score.cost
    return out


def _paths(store: str) -> list[str]:
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


def _labelled(paths: list[str]) -> list[str]:
    from core.aeb.clip_store import ClipStore

    out = []
    for p in paths:
        try:
            meta = ClipStore(Path(p).parent).peek_metadata(Path(p))
        except Exception:
            continue
        if meta is not None and meta.label is not None and meta.label.class_ != "ignore":
            out.append(p)
    return out


def report(rows: list[dict]) -> None:
    ok = [r for r in rows if "base" in r]
    print(f"clips {len(rows)} replayed {len(ok)} errors {Counter(r.get('error') for r in rows if 'error' in r)}")
    print(f"by session {Counter(r['session'] for r in ok)}  game {Counter(r['game'] for r in ok)}")
    changed = [r for r in ok if r["base"] != r["map"]]
    print(f"decision stream changed on {len(changed)} clips")
    scored = [r for r in ok if "base_verdict" in r]
    base_cost = sum(r["base_cost"] for r in scored)
    map_cost = sum(r["map_cost"] for r in scored)
    print(f"labelled {len(scored)}: cost {base_cost:.2f} -> {map_cost:.2f} (lower is better)")
    flips = Counter((r["base_verdict"], r["map_verdict"]) for r in scored if r["base_verdict"] != r["map_verdict"])
    for (a, b), n in flips.most_common():
        print(f"  {a:15s} -> {b:15s} {n}")
    print("\nchanged clips:")
    for r in sorted(changed, key=lambda r: (r.get("label") or "~", r["clip"])):
        b, m = r["base"], r["map"]
        print(f"  {r['clip']} {r['session']:3s} {str(r.get('label')):6s} "
              f"brake {b['brake_ticks']:3d}->{m['brake_ticks']:3d} warn {b['warn_ticks']:3d}->{m['warn_ticks']:3d} "
              f"first_brake {b['first_brake']}->{m['first_brake']} "
              f"{r.get('base_verdict', '')}->{r.get('map_verdict', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default="both", choices=("local", "contributed", "both"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=str(_OUT / "aeb_ab.json"))
    parser.add_argument("--report")
    parser.add_argument("--only", help="json from an earlier run: replay only the clips the stage fired on")
    parser.add_argument("--ids", help="text file of clip id prefixes to replay")
    parser.add_argument("--labelled", action="store_true", help="replay labelled clips only")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="override an AEBCalibration field for the run (repeatable)")
    args = parser.parse_args()
    overrides = _parse_set(args.set)
    if args.report:
        report(json.loads(Path(args.report).read_text(encoding="utf-8")))
        return 0
    paths = _paths(args.store)
    if args.only:
        fired = {r["clip"] for r in json.loads(Path(args.only).read_text(encoding="utf-8")) if r.get("fires")}
        paths = [p for p in paths if any(cid in Path(p).name for cid in fired)]
    if args.ids:
        wanted = Path(args.ids).read_text(encoding="utf-8").split()
        paths = [p for p in paths if any(cid in Path(p).name for cid in wanted)]
    if args.labelled:
        paths = _labelled(paths)
    if args.sample and args.sample < len(paths):
        random.Random(5).shuffle(paths)
        paths = paths[:args.sample]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(overrides,)) as pool:
        futures = [pool.submit(ab_clip, p) for p in paths]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:
                rows.append({"error": f"worker {type(exc).__name__}"})
            if n % 100 == 0:
                print(f"{n}/{len(paths)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows), encoding="utf-8")
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
