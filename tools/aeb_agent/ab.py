"""Per-clip A/B of one calibration flag, cached like the feature rows.

Built for `clearance_required_enabled` (commit 9eee3dd, clearance-based required
decel), which ships enabled: flipping it off restores the pre-clearance
relative-frame demand, so a clip whose decision stream moves between the two is a
clip that commit is load-bearing on. The mechanism is generic, so any boolean on
`AEBCalibration` can be diffed the same way.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from core.aeb.calibration import DEFAULT as _CAL_DEFAULT
from core.aeb.clip_eval import run_headless
from core.aeb.clip_store import deserialize_clip

from tools.aeb_agent.corpus import ClipRow
from tools.aeb_agent.paths import workspace

AB_VERSION = 1
CLEARANCE_FLAG = "clearance_required_enabled"

# A few ticks of jitter is not "this commit changed the clip". Material means the
# outcome flipped, the brake moved by this much, or the stream differs this long.
MATERIAL_SHIFT_S = 0.20
MATERIAL_DIFF_TICKS = 10


@dataclass
class AbResult:
    """How one clip's decision stream differs between flag on and flag off."""

    clip_id: str = ""
    flag: str = ""
    changed: bool = False
    brake_ticks_on: int = 0
    brake_ticks_off: int = 0
    warn_ticks_on: int = 0
    warn_ticks_off: int = 0
    first_brake_on: float | None = None
    first_brake_off: float | None = None
    peak_required_on: float = 0.0
    peak_required_off: float = 0.0
    diff_ticks: int = 0
    error: str = ""

    @property
    def direction(self) -> str:
        """Which way the flag moved this clip, in plain terms."""
        if not self.changed:
            return "unchanged"
        if self.brake_ticks_off and not self.brake_ticks_on:
            return "silences"
        if self.brake_ticks_on and not self.brake_ticks_off:
            return "engages"
        if (self.first_brake_on is not None and self.first_brake_off is not None):
            delta = self.first_brake_on - self.first_brake_off
            if delta > 0.05:
                return "delays"
            if delta < -0.05:
                return "advances"
        return "reshapes"

    @property
    def material(self) -> bool:
        """True when the flag moves this clip enough to be worth a debugging tag."""
        if not self.changed or self.error:
            return False
        if self.direction in ("engages", "silences"):
            return True
        if self.direction in ("delays", "advances"):
            if self.first_brake_on is None or self.first_brake_off is None:
                return True
            return abs(self.first_brake_on - self.first_brake_off) >= MATERIAL_SHIFT_S
        return self.diff_ticks >= MATERIAL_DIFF_TICKS

    @property
    def tag(self) -> str:
        """Note tag for the direction, or an empty string when immaterial."""
        if not self.material:
            return ""
        return {
            "engages": "clearance-engages",
            "silences": "clearance-silences",
            "delays": "clearance-later",
            "advances": "clearance-earlier",
        }.get(self.direction, "clearance-reshapes")

    def summary(self) -> str:
        if self.error:
            return f"error: {self.error}"
        if not self.changed:
            return "no decision change"
        on = ("-" if self.first_brake_on is None
              else f"{self.first_brake_on:.2f}s")
        off = ("-" if self.first_brake_off is None
               else f"{self.first_brake_off:.2f}s")
        weight = "MATERIAL" if self.material else "minor"
        return (f"{self.direction} ({weight}): brake ticks "
                f"{self.brake_ticks_off} -> {self.brake_ticks_on}, first brake "
                f"{off} -> {on}, {self.diff_ticks} ticks differ")


def _init_worker() -> None:
    logging.getLogger("ui.popup.popup_window").setLevel(logging.CRITICAL)


def _stream(evs) -> tuple[int, int, float | None, float]:
    braked = [e for e in evs if e.aeb_brake]
    warned = [e for e in evs if e.aeb_warn]
    peak = max((min(e.required_decel_ms2, 99.0) for e in evs), default=0.0)
    return len(braked), len(warned), (braked[0].t_rel if braked else None), peak


def diff_one(path_str: str, flag: str = CLEARANCE_FLAG) -> dict:
    """Worker: run the clip with the flag on and off, report where they differ."""
    path = Path(path_str)
    try:
        clip = deserialize_clip(path.read_bytes())
    except Exception as exc:
        return {"clip_id": "", "flag": flag, "error": f"load failed: {exc!r}"[:120]}
    res = AbResult(clip_id=clip.metadata.clip_id, flag=flag)
    try:
        on = run_headless(clip, cal=_CAL_DEFAULT)
        off = run_headless(clip, cal=replace(_CAL_DEFAULT, **{flag: False}))
    except Exception as exc:
        res.error = f"replay failed: {exc!r}"[:120]
        return res.__dict__
    (res.brake_ticks_on, res.warn_ticks_on,
     res.first_brake_on, res.peak_required_on) = _stream(on)
    (res.brake_ticks_off, res.warn_ticks_off,
     res.first_brake_off, res.peak_required_off) = _stream(off)
    res.diff_ticks = sum(
        1 for a, b in zip(on, off)
        if a.aeb_brake != b.aeb_brake or a.aeb_warn != b.aeb_warn)
    res.changed = res.diff_ticks > 0
    return res.__dict__


def default_workers() -> int:
    n = os.cpu_count() or 4
    # Same cap as the feature pool: two headless runs per clip, Windows spawn RAM.
    return max(1, min(16, n - 2 if n > 4 else n))


def _cache_path(flag: str) -> Path:
    return workspace() / f"ab_{flag}.json"


def load_cache(flag: str) -> dict[str, dict]:
    path = _cache_path(flag)
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if int(blob.get("version", 0)) != AB_VERSION:
        return {}
    return {str(k): v for k, v in blob.get("rows", {}).items()}


def save_cache(flag: str, rows: dict[str, dict]) -> None:
    _cache_path(flag).write_text(
        json.dumps({"version": AB_VERSION, "flag": flag, "rows": rows}),
        encoding="utf-8")


def _key(row: ClipRow) -> str:
    return f"{row.path}|{row.mtime:.6f}|{row.size_bytes}"


def results_for(rows: list[ClipRow], *, flag: str = CLEARANCE_FLAG,
                workers: int | None = None, rebuild: bool = False,
                progress=None) -> dict[str, AbResult]:
    """A/B every row, filling the cache. Keyed on file identity like features."""
    cache = {} if rebuild else load_cache(flag)
    todo = [r for r in rows if _key(r) not in cache]
    if todo:
        n_workers = default_workers() if workers is None else max(1, int(workers))
        if progress is not None:
            progress(f"A/B {flag} over {len(todo)} clips ({n_workers} workers)")
        if n_workers <= 1:
            for i, row in enumerate(todo):
                cache[_key(row)] = diff_one(row.path, flag)
                if progress is not None and (i + 1) % 25 == 0:
                    progress(f"  {i + 1}/{len(todo)}")
        else:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_worker) as pool:
                futs = {pool.submit(diff_one, r.path, flag): r for r in todo}
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    cache[_key(futs[fut])] = fut.result()
                    if progress is not None and (done % 25 == 0
                                                 or done == len(todo)):
                        progress(f"  {done}/{len(todo)}")
        save_cache(flag, cache)

    out: dict[str, AbResult] = {}
    for row in rows:
        raw = cache.get(_key(row))
        if raw is None:
            continue
        known = {f for f in AbResult.__dataclass_fields__}
        out[row.clip_id] = AbResult(**{k: v for k, v in raw.items() if k in known})
    return out
