"""Process-pool feature extraction over the corpus, cached by file identity.

Extraction costs about 0.25 s per clip, so a full corpus pass is a pool job. The
cache key is (path, mtime, size) plus FEATURES_VERSION, which means a labelling
edit invalidates only the clip it touched.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from core.aeb.clip_store import deserialize_clip

from tools.aeb_agent.corpus import ClipRow
from tools.aeb_agent.features import FEATURES_VERSION, ClipFeatures, extract
from tools.aeb_agent.paths import workspace

_CACHE_NAME = "features.json"


def _init_worker() -> None:
    logging.getLogger("ui.popup.popup_window").setLevel(logging.CRITICAL)


def extract_path(path_str: str) -> dict | None:
    """Worker entry: decode one clip file and return its feature dict."""
    path = Path(path_str)
    try:
        clip = deserialize_clip(path.read_bytes())
    except Exception:
        return None
    try:
        return asdict(extract(clip, path_str))
    except Exception:
        return {"version": FEATURES_VERSION, "path": path_str,
                "flags": ["feature extraction raised"], "scenario": ["broken"]}


def default_workers() -> int:
    n = os.cpu_count() or 4
    # Same cap as tools/aeb_corpus_run/_parallel_score.py: Windows spawn RAM.
    return max(1, min(16, n - 2 if n > 4 else n))


def _cache_path() -> Path:
    return workspace() / _CACHE_NAME


def load_cache() -> dict[str, dict]:
    path = _cache_path()
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if int(blob.get("version", 0)) != FEATURES_VERSION:
        return {}
    return {str(k): v for k, v in blob.get("rows", {}).items()}


def save_cache(rows: dict[str, dict]) -> None:
    payload = {"version": FEATURES_VERSION, "rows": rows}
    _cache_path().write_text(json.dumps(payload), encoding="utf-8")


def _key(row: ClipRow) -> str:
    return f"{row.path}|{row.mtime:.6f}|{row.size_bytes}"


def features_for(rows: list[ClipRow], *, workers: int | None = None,
                 rebuild: bool = False, progress=None) -> dict[str, ClipFeatures]:
    """Feature rows keyed by clip_id, filling and persisting the cache as needed."""
    cache = {} if rebuild else load_cache()
    todo = [r for r in rows if _key(r) not in cache]
    if todo:
        n_workers = default_workers() if workers is None else max(1, int(workers))
        if progress is not None:
            progress(f"extracting features for {len(todo)} clips "
                     f"({n_workers} workers)")
        if n_workers <= 1:
            for i, row in enumerate(todo):
                got = extract_path(row.path)
                if got is not None:
                    cache[_key(row)] = got
                if progress is not None and (i + 1) % 50 == 0:
                    progress(f"  {i + 1}/{len(todo)}")
        else:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_worker) as pool:
                futs = {pool.submit(extract_path, r.path): r for r in todo}
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    row = futs[fut]
                    got = fut.result()
                    if got is not None:
                        cache[_key(row)] = got
                    if progress is not None and (done % 50 == 0 or done == len(todo)):
                        progress(f"  {done}/{len(todo)}")
        save_cache(cache)

    out: dict[str, ClipFeatures] = {}
    for row in rows:
        raw = cache.get(_key(row))
        if raw is None:
            continue
        out[row.clip_id] = ClipFeatures(**raw)
    return out


def prune_cache(rows: list[ClipRow]) -> int:
    """Drop cache entries whose file identity is no longer in the index."""
    cache = load_cache()
    live = {_key(r) for r in rows}
    stale = [k for k in cache if k not in live]
    for k in stale:
        cache.pop(k, None)
    if stale:
        save_cache(cache)
    return len(stale)
