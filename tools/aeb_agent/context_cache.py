"""Process-pool scene context over the corpus, cached by file identity like the features.

A context row costs about 1 to 2 s (decode, map match, lane routing), so it lives in
its own cache file and is only built for the clips a command asks for.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from core.aeb.clip_store import deserialize_clip

from tools.aeb_agent.context import CONTEXT_VERSION, build
from tools.aeb_agent.corpus import ClipRow
from tools.aeb_agent.feature_cache import cache_key, default_workers
from tools.aeb_agent.paths import workspace

_CACHE_NAME = "context.json"


def _init_worker() -> None:
    logging.getLogger("ui.popup.popup_window").setLevel(logging.CRITICAL)
    logging.getLogger("core.road_layout").setLevel(logging.ERROR)


def context_path(path_str: str, feats: dict) -> dict | None:
    """Worker entry: decode one clip and build its context row."""
    path = Path(path_str)
    try:
        clip = deserialize_clip(path.read_bytes())
    except Exception:
        return None
    try:
        return asdict(build(clip, feats, path_str))
    except Exception as exc:
        return {"version": CONTEXT_VERSION, "path": path_str,
                "error": f"context raised {type(exc).__name__}: {exc}"}


def _cache_file() -> Path:
    return workspace() / _CACHE_NAME


def load_cache() -> dict[str, dict]:
    path = _cache_file()
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if int(blob.get("version", 0)) != CONTEXT_VERSION:
        return {}
    return {str(k): v for k, v in blob.get("rows", {}).items()}


def save_cache(rows: dict[str, dict]) -> None:
    payload = {"version": CONTEXT_VERSION, "rows": rows}
    _cache_file().write_text(json.dumps(payload), encoding="utf-8")


def context_for(rows: list[ClipRow], feats: dict, *, workers: int | None = None,
                rebuild: bool = False, progress=None) -> dict[str, dict]:
    """Context rows keyed by clip_id. `feats` maps clip_id to ClipFeatures."""
    cache = {} if rebuild else load_cache()
    todo = [r for r in rows if cache_key(r) not in cache and r.clip_id in feats]
    if todo:
        n_workers = default_workers() if workers is None else max(1, int(workers))
        if progress is not None:
            progress(f"building scene context for {len(todo)} clips ({n_workers} workers)")
        args = [(r, asdict(feats[r.clip_id])) for r in todo]
        if n_workers <= 1:
            for i, (row, fd) in enumerate(args):
                got = context_path(row.path, fd)
                if got is not None:
                    cache[cache_key(row)] = got
                if progress is not None and (i + 1) % 25 == 0:
                    progress(f"  {i + 1}/{len(todo)}")
        else:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
                futs = {pool.submit(context_path, row.path, fd): row for row, fd in args}
                for done, fut in enumerate(as_completed(futs), 1):
                    row = futs[fut]
                    got = fut.result()
                    if got is not None:
                        cache[cache_key(row)] = got
                    if progress is not None and (done % 100 == 0 or done == len(todo)):
                        progress(f"  {done}/{len(todo)}")
                        save_cache(cache)
        save_cache(cache)
    out = {}
    for row in rows:
        got = cache.get(cache_key(row))
        if got is not None:
            out[row.clip_id] = _current(got)
    return out


def _current(ctx: dict) -> dict:
    """Rows built before the labeller was dropped still carry a window proposal: drop it."""
    out = {k: v for k, v in ctx.items() if k not in ("window_proposal", "window_target")}
    if "demand_vid" not in out:
        out["demand_vid"] = ctx.get("window_target")
    return out
