"""Agent-facing AEB corpus CLI: python -m tools.aeb_agent <command>. Dev only.

Read tools/aeb_agent/README.md before using this. Every command is read-only
except `apply` and `revert`, and both of those are dry runs without --commit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

logging.getLogger("ui.popup.popup_window").setLevel(logging.CRITICAL)

from tools.aeb_agent import ab, audit, digest, labels  # noqa: E402
from tools.aeb_agent.corpus import (  # noqa: E402
    build_index, composition, filter_rows, load_clip,
)
from tools.aeb_agent.feature_cache import features_for, prune_cache  # noqa: E402
from tools.aeb_agent.paths import workspace  # noqa: E402


def _note(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _select(args, *, need_features: bool):
    """Index, filter and (optionally) feature-load the clips a command works on.

    A feature-dependent filter overrides `--fast`: silently skipping the filter
    would return a wrong answer rather than a cheap one.
    """
    feature_filter = bool(getattr(args, "scenario", "")
                          or getattr(args, "flagged", False)
                          or getattr(args, "counterfactual", "")
                          or getattr(args, "credible_cf", False))
    if feature_filter and getattr(args, "fast", False):
        _note("--fast ignored: the filters you asked for need the feature cache")
        args.fast = False
        need_features = True
    index = build_index(args.store, rebuild=getattr(args, "rebuild", False),
                        progress=_note if args.verbose else None)
    rows = filter_rows(
        index.rows,
        label=args.label, origin=args.origin, session=args.session,
        trigger=args.trigger, unlabeled=args.unlabeled,
        labelled_only=args.labelled, sev_min=args.sev_min, sev_max=args.sev_max,
        since=args.since, until=args.until, notes=args.notes,
        version=args.version, inconsistent=args.inconsistent,
    )
    if args.ids:
        wanted = {i.strip().lower() for i in args.ids.split(",") if i.strip()}
        rows = [r for r in rows
                if any(r.clip_id.lower().startswith(w) for w in wanted)]
    feats = {}
    # Limit early unless a feature filter still has to run, so `--limit 5` on a
    # cold cache extracts five clips rather than the whole selection.
    needs_feature_filter = feature_filter
    if args.limit and not needs_feature_filter:
        rows = rows[:args.limit]
    if need_features:
        feats = features_for(rows, workers=args.workers,
                             rebuild=getattr(args, "refresh_features", False),
                             progress=_note)
        if args.scenario:
            wanted = {s.strip() for s in args.scenario.split(",") if s.strip()}
            rows = [r for r in rows
                    if wanted & set(feats.get(r.clip_id).scenario
                                    if r.clip_id in feats else [])]
        if args.flagged:
            rows = [r for r in rows
                    if r.clip_id in feats and feats[r.clip_id].flags]
        if args.counterfactual:
            wanted = {v.strip() for v in args.counterfactual.split(",") if v.strip()}
            rows = [r for r in rows
                    if r.clip_id in feats
                    and (feats[r.clip_id].counterfactual or {}).get("verdict")
                    in wanted]
        if args.credible_cf:
            rows = [r for r in rows
                    if r.clip_id in feats
                    and (feats[r.clip_id].counterfactual or {}).get("credible")]
    if args.limit:
        rows = rows[:args.limit]
    return index, rows, feats


def cmd_list(args) -> int:
    _index, rows, feats = _select(args, need_features=not args.fast)
    if args.json:
        payload = [
            {"clip_id": r.clip_id, "short": r.short, "origin": r.origin,
             "session": r.session_kind, "trigger": r.trigger_source,
             "captured_at": r.captured_at, "class": r.label_class,
             "severity": r.severity, "window": r.window, "notes": r.notes,
             "scenario": (feats[r.clip_id].scenario if r.clip_id in feats else []),
             "flags": (feats[r.clip_id].flags if r.clip_id in feats else [])}
            for r in rows
        ]
        print(json.dumps(payload, indent=1))
        return 0
    for r in rows:
        print(digest.one_line(r, feats.get(r.clip_id)))
    print(f"\n{len(rows)} clips", file=sys.stderr)
    return 0


def cmd_stats(args) -> int:
    index, rows, feats = _select(args, need_features=not args.fast)
    stats = composition(rows)
    if feats:
        scen: dict[str, int] = {}
        for f in feats.values():
            for tag in f.scenario:
                scen[tag] = scen.get(tag, 0) + 1
        stats["by_scenario"] = dict(sorted(scen.items(), key=lambda kv: -kv[1]))
        stats["with_flags"] = sum(1 for f in feats.values() if f.flags)
    stats["index_total"] = len(index.rows)
    print(json.dumps(stats, indent=1))
    return 0


def cmd_show(args) -> int:
    index = build_index(args.store)
    for ident in args.clip:
        row = index.resolve(ident)
        clip = load_clip(row)
        if clip is None:
            print(f"{ident}: failed to load", file=sys.stderr)
            continue
        result = None
        if args.ab:
            result = ab.results_for([row], progress=_note).get(row.clip_id)
        print(digest.build(row, clip, scenes=args.scenes,
                           timeline=not args.no_timeline, replay=args.replay,
                           ab_result=result))
        print()
    return 0


def cmd_scene(args) -> int:
    from core.aeb.clip_replay import replay_clip

    from tools.aeb_agent.features import extract
    from tools.aeb_agent.scene import render_series

    index = build_index(args.store)
    row = index.resolve(args.clip)
    clip = load_clip(row)
    frames = replay_clip(clip)
    if not frames:
        print("no frames", file=sys.stderr)
        return 1
    if args.t is not None:
        times = [args.t]
    else:
        action = extract(clip, row.path).action_t
        times = [action + off for off in (-2.0, -1.0, 0.0, 1.0)][:args.count]
    print(render_series(frames, times, span_fwd=args.span,
                        span_side=args.side))
    return 0


def cmd_triage(args) -> int:
    """Write one dossier file per clip plus a worklist, for a batch review pass."""
    _index, rows, feats = _select(args, need_features=True)
    abs_by_id = {}
    if not args.no_ab:
        abs_by_id = ab.results_for(rows, workers=args.workers, progress=_note)
    out_dir = workspace() / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    worklist = []
    for i, row in enumerate(rows, 1):
        clip = load_clip(row)
        if clip is None:
            continue
        text = digest.build(row, clip, feats=feats.get(row.clip_id),
                            scenes=args.scenes, replay=args.replay,
                            ab_result=abs_by_id.get(row.clip_id))
        dest = out_dir / f"{row.short}.txt"
        dest.write_text(text, encoding="utf-8")
        f = feats.get(row.clip_id)
        a = abs_by_id.get(row.clip_id)
        worklist.append({
            "clip_id": row.clip_id, "short": row.short, "dossier": dest.name,
            "current_class": row.label_class, "severity": row.severity,
            "window": row.window,
            "scenario": (f.scenario if f else []), "flags": (f.flags if f else []),
            "clearance_changed": bool(a and a.changed),
            "clearance_material": bool(a and a.material),
            "clearance_tag": (a.tag if a else ""),
            "clearance_direction": (a.direction if a else "unknown"),
            "mtime": row.mtime, "size_bytes": row.size_bytes,
        })
        if args.verbose or i % 25 == 0:
            _note(f"  {i}/{len(rows)} {row.short}")
    index_file = out_dir / "worklist.json"
    index_file.write_text(json.dumps(worklist, indent=1), encoding="utf-8")
    print(json.dumps({"clips": len(worklist), "dir": str(out_dir),
                      "worklist": index_file.name,
                      "next": "read each dossier, then append to proposals.jsonl "
                              "and run: python -m tools.aeb_agent apply"}, indent=1))
    return 0


def cmd_thumbs(args) -> int:
    """Write each selected clip's screenshot out so a human can flip through them.

    The images are 240x135 by design and text is not legible in them; they are for
    judging layout and framing, not for reading anything off the HUD.
    """
    import base64

    _index, rows, feats = _select(args, need_features=not args.fast)
    out_dir = workspace() / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    written = missing = 0
    for row in rows:
        clip = load_clip(row)
        blob = clip.metadata.thumbnail_jpeg if clip is not None else None
        if not blob:
            missing += 1
            continue
        (out_dir / f"{row.short}.jpg").write_bytes(base64.b64decode(blob))
        written += 1
    print(json.dumps({"written": written, "no_screenshot": missing,
                      "dir": str(out_dir)}, indent=1))
    return 0


def cmd_render(args) -> int:
    """Export a clip's debug view as a PNG sequence for video editing."""
    from tools.aeb_agent.render import render_frames, write_credit

    index = build_index(args.store)
    row = index.resolve(args.clip)
    clip = load_clip(row)
    if clip is None:
        print("failed to load", file=sys.stderr)
        return 1
    out_dir = workspace() / args.out / row.short
    result = render_frames(
        clip, out_dir, width=args.width, height=args.height,
        start_t=args.start, end_t=args.end,
        vehicle_paths=not args.no_paths, progress=_note)
    if row.origin != "local":
        write_credit(out_dir, row.clip_id)
        result["credit_required"] = "anonymous contributed clip"
        result["credit_file"] = "CREDIT.txt"
    print(json.dumps(result, indent=1))
    return 0


def cmd_clearance(args) -> int:
    """A/B one calibration flag over the selection and report which clips move."""
    _index, rows, _feats = _select(args, need_features=False)
    res = ab.results_for(rows, flag=args.flag, workers=args.workers,
                         rebuild=args.rebuild_ab, progress=_note)
    moved = [(r, res[r.clip_id]) for r in rows
             if r.clip_id in res
             and (res[r.clip_id].material if args.material
                  else res[r.clip_id].changed)]
    if args.json:
        print(json.dumps({
            "flag": args.flag, "scanned": len(res), "changed": len(moved),
            "clips": [{"clip_id": r.clip_id, "short": r.short,
                       "class": r.label_class, "direction": a.direction,
                       "material": a.material, "tag": a.tag,
                       "summary": a.summary(), **a.__dict__}
                      for r, a in moved]}, indent=1))
        return 0
    for row, a in sorted(moved, key=lambda x: x[1].direction):
        print(f"{row.short}  {row.label_class or '-':<17} "
              f"{(a.tag or 'minor'):<20} {a.summary()}")
    errs = sum(1 for a in res.values() if a.error)
    print(f"\n{len(moved)} of {len(res)} clips change decision under "
          f"{args.flag} ({errs} errors)", file=sys.stderr)
    return 0


def cmd_audit(args) -> int:
    _index, rows, feats = _select(args, need_features=True)
    suspects = audit.run(rows, feats, only=args.rule)
    if args.json:
        print(json.dumps({"summary": audit.summarize(suspects),
                          "suspects": [s.__dict__ for s in suspects]}, indent=1))
        return 0
    for s in suspects:
        print(s.line())
    print("\n" + json.dumps(audit.summarize(suspects)), file=sys.stderr)
    return 0


def cmd_index(args) -> int:
    index = build_index(args.store, rebuild=args.rebuild, progress=_note)
    dropped = 0
    if args.features:
        features_for(index.rows, workers=args.workers,
                     rebuild=args.refresh_features, progress=_note)
        dropped = prune_cache(index.rows)
    print(json.dumps({"clips": len(index.rows), "workspace": str(workspace()),
                      "features": bool(args.features),
                      "stale_feature_rows_dropped": dropped}, indent=1))
    return 0


def cmd_propose(args) -> int:
    index = build_index(args.store)
    row = index.resolve(args.clip)
    window = None
    if args.window:
        parts = [float(x) for x in args.window.replace(",", " ").split()]
        if len(parts) != 2:
            print("--window needs two numbers: 'from to'", file=sys.stderr)
            return 2
        window = parts
    prop = labels.Proposal(
        clip_id=row.clip_id, label_class=args.set_class, severity=args.severity,
        window=window, target_vid=args.target_vid, notes=args.notes,
        rationale=args.rationale, confidence=args.confidence,
        reviewer=args.reviewer, window_source=args.window_source,
        mtime=row.mtime, size_bytes=row.size_bytes,
    )
    path = labels.append_proposal(prop, labels.proposals_path(args.file))
    print(json.dumps({"appended": row.short, "file": str(path)}))
    return 0


def cmd_apply(args) -> int:
    index = build_index(args.store)
    props = labels.read_proposals(labels.proposals_path(args.file))
    if not props:
        print("no proposals found", file=sys.stderr)
        return 1
    feats = {}
    rows = [index.resolve(p.clip_id) for p in props]
    feats = features_for(rows, workers=args.workers, progress=_note)
    checks = [labels.validate(p, index, feats, min_confidence=args.min_confidence,
                              allow_recorded_window=args.allow_recorded_window)
              for p in props]
    for check in checks:
        print(check.line())
    result = labels.apply_checks(checks, index, commit=args.commit)
    print(json.dumps(result), file=sys.stderr)
    if not args.commit:
        print("dry run: pass --commit to write", file=sys.stderr)
    return 0


def cmd_journal(args) -> int:
    entries = labels.read_journal()
    if args.json:
        print(json.dumps(entries[-args.limit:] if args.limit else entries, indent=1))
        return 0
    for entry in (entries[-args.limit:] if args.limit else entries):
        before = entry.get("before", {}) or {}
        after = entry.get("after", {}) or {}
        print(f"{entry['at']}  {entry['clip_id'][:8]}  "
              f"{before.get('class') or '-'} -> {after.get('class')}  "
              f"({entry.get('reviewer')}) {entry.get('rationale', '')[:70]}")
    print(f"{len(entries)} journal entries", file=sys.stderr)
    return 0


def cmd_revert(args) -> int:
    index = build_index(args.store)
    result = labels.revert(args.clip, index, commit=args.commit)
    print(json.dumps(result))
    if not args.commit:
        print("dry run: pass --commit to write", file=sys.stderr)
    return 0


def cmd_score(args) -> int:
    from core.aeb.clip_score import format_corpus
    from core.aeb.clip_store import ClipStore

    from tools.aeb_agent.paths import stores

    from tools.aeb_corpus_run._parallel_score import (
        default_workers, score_store_parallel,
    )
    totals = []
    for origin, store in stores(args.store):
        paths = [i.path for i in store.list_clips()]
        cs = score_store_parallel(
            ClipStore(store.root),
            workers=args.workers or default_workers(), paths=paths)
        totals.append((origin, cs))
        if args.per_clip:
            print(format_corpus(cs, f"SCORE {origin}"))
    payload = {origin: {"total_cost": cs.total_cost, "scored": cs.scored,
                        "verdicts": dict(cs.verdict_counts)}
               for origin, cs in totals}
    print(json.dumps(payload, indent=1))
    return 0


def _add_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", default="both", choices=("both", "local", "remote"))
    p.add_argument("--label", default="", help="comma list of label classes")
    p.add_argument("--origin", default="", help="local or remote")
    p.add_argument("--session", default="", help="SP or TMP")
    p.add_argument("--trigger", default="", help="substring of trigger_source")
    p.add_argument("--scenario", default="", help="comma list of scenario tags")
    p.add_argument("--ids", default="", help="comma list of clip id prefixes")
    p.add_argument("--notes", default="", help="substring of the label notes")
    p.add_argument("--version", default="", help="substring of client_version")
    p.add_argument("--since", default="", help="captured_at lower bound")
    p.add_argument("--until", default="", help="captured_at upper bound")
    p.add_argument("--sev-min", type=int, default=None, dest="sev_min")
    p.add_argument("--sev-max", type=int, default=None, dest="sev_max")
    p.add_argument("--unlabeled", action="store_true", help="only unreviewed clips")
    p.add_argument("--labelled", action="store_true", help="only reviewed clips")
    p.add_argument("--inconsistent", action="store_true",
                   help="only clips whose class and window disagree")
    p.add_argument("--flagged", action="store_true",
                   help="only clips carrying a data-quality flag")
    p.add_argument("--counterfactual", default="",
                   help="comma list of verdicts: collides, likely, close, "
                        "clear, degenerate")
    p.add_argument("--credible-cf", action="store_true", dest="credible_cf",
                   help="only clips whose counterfactual is inside the "
                        "trustworthy 1.5 s window")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--rebuild", action="store_true", help="re-peek all metadata")
    p.add_argument("--refresh-features", action="store_true", dest="refresh_features")
    p.add_argument("--fast", action="store_true",
                   help="skip feature extraction (metadata only)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tools.aeb_agent",
                                description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    lst = sub.add_parser("list", help="one line per clip")
    _add_filters(lst)
    lst.set_defaults(func=cmd_list)

    st = sub.add_parser("stats", help="corpus composition")
    _add_filters(st)
    st.set_defaults(func=cmd_stats)

    show = sub.add_parser("show", help="full text dossier for one or more clips")
    show.add_argument("clip", nargs="+")
    show.add_argument("--store", default="both", choices=("both", "local", "remote"))
    show.add_argument("--scenes", type=int, default=3)
    show.add_argument("--no-timeline", action="store_true", dest="no_timeline")
    show.add_argument("--replay", action="store_true",
                      help="also run the working-tree AEB over the clip")
    show.add_argument("--ab", action="store_true",
                      help="also A/B the 9eee3dd clearance flag on this clip")
    show.set_defaults(func=cmd_show)

    scene = sub.add_parser("scene", help="ASCII top-down maps for one clip")
    scene.add_argument("clip")
    scene.add_argument("--store", default="both", choices=("both", "local", "remote"))
    scene.add_argument("--t", type=float, default=None, help="single timestamp")
    scene.add_argument("--count", type=int, default=4)
    scene.add_argument("--span", type=float, default=80.0)
    scene.add_argument("--side", type=float, default=20.0)
    scene.set_defaults(func=cmd_scene)

    tri = sub.add_parser("triage",
                         help="write a dossier per clip plus a worklist for batch review")
    _add_filters(tri)
    tri.add_argument("--out", default="triage", help="subdirectory of the workspace")
    tri.add_argument("--scenes", type=int, default=2)
    tri.add_argument("--replay", action="store_true")
    tri.add_argument("--no-ab", action="store_true", dest="no_ab",
                     help="skip the clearance-flag A/B block")
    tri.set_defaults(func=cmd_triage)

    th = sub.add_parser("thumbs",
                        help="export clip screenshots for a human to flip through")
    _add_filters(th)
    th.add_argument("--out", default="thumbs", help="subdirectory of the workspace")
    th.set_defaults(func=cmd_thumbs)

    rn = sub.add_parser("render",
                        help="export a clip's debug view as a PNG sequence")
    rn.add_argument("clip")
    rn.add_argument("--store", default="both", choices=("both", "local", "remote"))
    rn.add_argument("--out", default="render")
    rn.add_argument("--width", type=int, default=1280)
    rn.add_argument("--height", type=int, default=720)
    rn.add_argument("--start", type=float, default=None, help="t_rel seconds")
    rn.add_argument("--end", type=float, default=None, help="t_rel seconds")
    rn.add_argument("--no-paths", action="store_true", dest="no_paths",
                    help="hide per-vehicle predicted corridors")
    rn.set_defaults(func=cmd_render)

    clr = sub.add_parser(
        "clearance",
        help="A/B a calibration flag; default is the 9eee3dd clearance rewrite")
    _add_filters(clr)
    clr.add_argument("--flag", default=ab.CLEARANCE_FLAG,
                     help="boolean field on AEBCalibration to flip")
    clr.add_argument("--material", action="store_true",
                     help="only clips the flag moves enough to matter")
    clr.add_argument("--rebuild-ab", action="store_true", dest="rebuild_ab")
    clr.set_defaults(func=cmd_clearance)

    aud = sub.add_parser("audit", help="rank clips whose label looks wrong")
    _add_filters(aud)
    aud.add_argument("--rule", default="",
                     help=f"comma list of {', '.join(audit.RULE_NAMES)}")
    aud.set_defaults(func=cmd_audit)

    idx = sub.add_parser("index", help="build or refresh the caches")
    idx.add_argument("--store", default="both", choices=("both", "local", "remote"))
    idx.add_argument("--features", action="store_true")
    idx.add_argument("--rebuild", action="store_true")
    idx.add_argument("--refresh-features", action="store_true", dest="refresh_features")
    idx.add_argument("--workers", type=int, default=None)
    idx.set_defaults(func=cmd_index)

    prop = sub.add_parser("propose", help="append one label proposal")
    prop.add_argument("clip")
    prop.add_argument("--store", default="both", choices=("both", "local", "remote"))
    prop.add_argument("--class", dest="set_class", required=True)
    prop.add_argument("--severity", type=int, default=3)
    prop.add_argument("--window", default="", help="'from to' in seconds")
    prop.add_argument("--target-vid", type=int, default=None, dest="target_vid")
    prop.add_argument("--notes", default="")
    prop.add_argument("--rationale", required=True,
                      help="why, in one sentence; refused if empty")
    prop.add_argument("--confidence", type=float, default=0.0)
    prop.add_argument("--reviewer", default="agent")
    prop.add_argument("--window-source", default="judged", dest="window_source",
                      choices=("judged", "recorded", "unchanged"))
    prop.add_argument("--file", default="", help="proposals file name")
    prop.set_defaults(func=cmd_propose)

    app = sub.add_parser("apply", help="validate proposals, write with --commit")
    app.add_argument("--store", default="both", choices=("both", "local", "remote"))
    app.add_argument("--file", default="")
    app.add_argument("--commit", action="store_true")
    app.add_argument("--workers", type=int, default=None)
    app.add_argument("--min-confidence", type=float, default=0.0,
                     dest="min_confidence")
    app.add_argument("--allow-recorded-window", action="store_true",
                     dest="allow_recorded_window")
    app.set_defaults(func=cmd_apply)

    jr = sub.add_parser("journal", help="applied label changes, newest last")
    jr.add_argument("--limit", type=int, default=40)
    jr.add_argument("--json", action="store_true")
    jr.set_defaults(func=cmd_journal)

    rev = sub.add_parser("revert", help="restore pre-change labels from the journal")
    rev.add_argument("clip", nargs="+")
    rev.add_argument("--store", default="both", choices=("both", "local", "remote"))
    rev.add_argument("--commit", action="store_true")
    rev.set_defaults(func=cmd_revert)

    sc = sub.add_parser("score", help="corpus objective under the working tree")
    sc.add_argument("--store", default="both", choices=("both", "local", "remote"))
    sc.add_argument("--workers", type=int, default=None)
    sc.add_argument("--per-clip", action="store_true", dest="per_clip")
    sc.set_defaults(func=cmd_score)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
