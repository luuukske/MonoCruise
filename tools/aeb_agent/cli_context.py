"""Scene-context commands for the agent CLI: hindsight and map evidence, ghost drift.

Read-only analysis, split out of cli.py to keep it readable. Never labels a clip.
"""

from __future__ import annotations

import json
import sys


def _note(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _select(args, *, need_features: bool):
    from tools.aeb_agent.cli import _select as select
    return select(args, need_features=need_features)


def cmd_context(args) -> int:
    """Build the scene context (hindsight + local map) and report map coverage."""
    from tools.aeb_agent.context_cache import context_for

    _index, rows, feats = _select(args, need_features=True)
    ctx = context_for(rows, feats, workers=args.workers,
                      rebuild=args.rebuild_context, progress=_note)
    tally: dict[str, dict[str, int]] = {}
    agree = differ = 0
    for row in rows:
        c = ctx.get(row.clip_id)
        if c is None:
            continue
        group = tally.setdefault(f"{row.origin}/{row.session_kind}",
                                 {"clips": 0, "map": 0, "on_lane": 0, "errors": 0})
        group["clips"] += 1
        group["map"] += int(bool(c.get("map_matched")))
        group["on_lane"] += int(bool((c.get("ego") or {}).get("on_lane")))
        group["errors"] += int(bool(c.get("error")))
        f = feats.get(row.clip_id)
        hp = c.get("hindsight_primary")
        if f is not None and hp is not None:
            agree += int(hp == f.primary_vid)
            differ += int(hp != f.primary_vid)
    print(json.dumps({"by_store_session": tally,
                      "hindsight_primary_vs_corridor_primary":
                          {"same": agree, "different": differ}}, indent=1))
    return 0


def cmd_drift(args) -> int:
    """p90 ghost drift, constant-rate against lane-following, on clips with no intervention."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tools.aeb_agent.feature_cache import default_workers
    from tools.aeb_agent.lane_ghost import measure_path

    _index, rows, feats = _select(args, need_features=True)
    jobs = [(r.path, feats[r.clip_id].action_t) for r in rows
            if r.clip_id in feats and not (feats[r.clip_id].intervention or {}).get("found")]
    _note(f"measuring drift on {len(jobs)} clips with no intervention")
    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers or default_workers()) as pool:
        futs = [pool.submit(measure_path, p, t) for p, t in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            out += fut.result()
            if i % 200 == 0:
                _note(f"  {i}/{len(jobs)}")
    table: dict = {}
    for cat in ("all", "straight", "gentle", "bend"):
        for mode in sorted({r["mode"] for r in out}):
            for dt in sorted({r["dt"] for r in out}):
                errs = sorted(r["err"] for r in out if r["mode"] == mode and r["dt"] == dt
                              and (cat == "all" or r["cat"] == cat))
                if len(errs) >= 20:
                    table.setdefault(cat, {}).setdefault(mode, {})[str(dt)] = {
                        "n": len(errs), "p50": round(errs[len(errs) // 2], 2),
                        "p90": round(errs[int(0.9 * len(errs))], 2)}
    print(json.dumps(table, indent=1))
    return 0


def register(sub, add_filters) -> None:
    cx = sub.add_parser("context",
                        help="build hindsight + map scene context, report map coverage")
    add_filters(cx)
    cx.add_argument("--rebuild-context", action="store_true", dest="rebuild_context")
    cx.set_defaults(func=cmd_context)

    dr = sub.add_parser("drift", help="ghost drift: constant-rate against lane-following")
    add_filters(dr)
    dr.set_defaults(func=cmd_drift)
