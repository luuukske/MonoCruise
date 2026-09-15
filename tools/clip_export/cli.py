"""python -m tools.clip_export: showcase video of one AEB clip. Dev only, never shipped."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.clip_export.export import CREDIT_LINE, ExportOptions, export_clip, render_still

_CONTRIBUTED_REFUSAL = (
    "{short} is a contributed clip. Publishing one needs its own decision: see "
    "'Publishing contributed clips' in tools/aeb_agent/README.md. Re-run with "
    "--contributed-ok to export it with the credit burned in."
)


def _note(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m tools.clip_export",
        description="Export an AEB clip as a clean top-down showcase video.")
    ap.add_argument("clip", help="clip id, short id prefix, or file name fragment")
    ap.add_argument("--store", default="both", choices=("both", "local", "remote"))
    ap.add_argument("--out", default=None,
                    help="output .mp4 (or folder with --png); default under the agent workspace")
    ap.add_argument("--start", type=float, default=None, help="t_rel seconds, default auto")
    ap.add_argument("--end", type=float, default=None, help="t_rel seconds, default auto")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--vertical", action="store_true", help="1080x1920 for shorts")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--speed", type=float, default=1.0, help="0.5 = half-speed slow motion")
    ap.add_argument("--recorded", action="store_true",
                    help="show the AEB decisions recorded live instead of re-running the current pipeline")
    ap.add_argument("--no-hud", action="store_true", dest="no_hud",
                    help="clean plate: no state label")
    ap.add_argument("--png", action="store_true", help="PNG sequence instead of mp4")
    ap.add_argument("--crf", type=int, default=16, help="x264 quality, lower is better")
    ap.add_argument("--still", type=float, default=None, metavar="T",
                    help="write one PNG at clip time T instead of a video")
    ap.add_argument("--contributed-ok", action="store_true", dest="contributed_ok",
                    help="allow a contributed clip; the credit line is always burned in")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from tools.aeb_agent.corpus import build_index, load_clip
    from tools.aeb_agent.paths import workspace
    from tools.aeb_agent.render import _app

    try:
        row = build_index(args.store, progress=_note).resolve(args.clip)
    except KeyError as exc:
        _note(str(exc))
        return 1
    contributed = row.origin != "local"
    if contributed and not args.contributed_ok:
        _note(_CONTRIBUTED_REFUSAL.format(short=row.short))
        return 2
    clip = load_clip(row)
    if clip is None:
        _note("failed to load " + row.short)
        return 1

    width, height = (1080, 1920) if args.vertical else (1920, 1080)
    opts = ExportOptions(
        width=args.width or width, height=args.height or height, fps=args.fps,
        speed=args.speed, start=args.start, end=args.end, hud=not args.no_hud,
        png=args.png, crf=args.crf, recompute=not args.recorded,
    )
    credit = CREDIT_LINE if contributed else None
    _app()
    exports = workspace().parent / "exports"
    try:
        if args.still is not None:
            out = Path(args.out) if args.out else exports / f"{row.short}_{args.still:.2f}.png"
            result = render_still(clip, args.still, out, opts, credit=credit)
        else:
            suffix = "" if opts.png else ".mp4"
            tag = "_vertical" if args.vertical else ""
            out = Path(args.out) if args.out else exports / f"{row.short}{tag}{suffix}"
            result = export_clip(clip, out, opts, credit=credit, progress=_note)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        _note(str(exc))
        return 1
    if result.get("warning"):
        _note("warning: " + result["warning"])
    print(json.dumps(result, indent=1))
    return 0
