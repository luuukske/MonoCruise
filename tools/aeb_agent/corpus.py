"""Metadata index over both clip stores: cached peek, short ids, filtering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.aeb.clip_schema import ClipMetadata
from core.aeb.clip_score import class_window_warning
from core.aeb.clip_store import ClipStore

from tools.aeb_agent.paths import stores, workspace

_INDEX_NAME = "index.json"
_INDEX_VERSION = 1
SHORT_LEN = 8
LABEL_CLASSES = ("tp", "good_intervention", "fp", "fn", "tn", "ignore")


@dataclass
class ClipRow:
    """One clip as the index knows it: file identity plus peeked metadata."""

    clip_id: str
    short: str
    path: str
    origin: str
    mtime: float
    size_bytes: int
    captured_at: str = ""
    session_kind: str = ""
    trigger_source: str = ""
    client_version: str = ""
    brake_reached: bool = False
    frame_count: int = 0
    tick_count: int = 0
    has_thumbnail: bool = False
    label_class: str = ""
    severity: int = 0
    window: list | None = None
    target_vid: int | None = None
    desired_peak_decel_ms2: float | None = None
    notes: str = ""

    @property
    def labelled(self) -> bool:
        return bool(self.label_class)

    @property
    def window_span(self) -> float:
        if not self.window:
            return 0.0
        return float(self.window[1]) - float(self.window[0])

    def label_warning(self) -> str | None:
        if not self.label_class:
            return None
        return class_window_warning(self.label_class, self.window is not None)


def _row_from_meta(info, meta: ClipMetadata | None, origin: str) -> ClipRow:
    clip_id = (meta.clip_id if meta else "") or info.path.stem
    lbl = meta.label if meta else None
    window = None
    if lbl is not None and lbl.should_trigger:
        window = [float(lbl.should_trigger.get("from_t", 0.0)),
                  float(lbl.should_trigger.get("to_t", 0.0))]
    return ClipRow(
        clip_id=clip_id,
        short=clip_id[:SHORT_LEN],
        path=str(info.path),
        origin=origin,
        mtime=info.mtime,
        size_bytes=info.size_bytes,
        captured_at=(meta.captured_at if meta else ""),
        session_kind=(meta.session_kind if meta else ""),
        trigger_source=(meta.trigger_source if meta else ""),
        client_version=(meta.client_version if meta else ""),
        brake_reached=bool(meta.brake_reached) if meta else False,
        frame_count=(meta.frame_count if meta else 0),
        tick_count=(meta.tick_count if meta else 0),
        has_thumbnail=bool(meta.thumbnail_jpeg) if meta else False,
        label_class=(lbl.class_ if lbl else ""),
        severity=(int(lbl.severity) if lbl else 0),
        window=window,
        target_vid=(lbl.target_vid if lbl else None),
        desired_peak_decel_ms2=(lbl.desired_peak_decel_ms2 if lbl else None),
        notes=(lbl.notes if lbl else ""),
    )


@dataclass
class Index:
    """Every clip in the selected stores, keyed by path, with a short-id map."""

    rows: list[ClipRow] = field(default_factory=list)

    def by_short(self) -> dict[str, list[ClipRow]]:
        out: dict[str, list[ClipRow]] = {}
        for row in self.rows:
            out.setdefault(row.short, []).append(row)
        return out

    def resolve_all(self, ident: str) -> list[ClipRow]:
        """Every file for one clip. A pulled clip can also sit in the local store."""
        key = ident.strip().lower()
        for pick in (lambda r: r.clip_id.lower() == key,
                     lambda r: r.clip_id.lower().startswith(key),
                     lambda r: key in Path(r.path).name.lower()):
            hits = [r for r in self.rows if pick(r)]
            ids = {r.clip_id for r in hits}
            if len(ids) == 1:
                return hits
            if len(ids) > 1:
                listed = ", ".join(sorted(r.short for r in hits)[:8])
                raise KeyError(f"{ident!r} matches {len(ids)} clips: {listed}")
        raise KeyError("no clip matches " + repr(ident))

    def resolve(self, ident: str) -> ClipRow:
        """One clip by clip_id, short id prefix, or file name fragment."""
        return self.resolve_all(ident)[0]


def _cache_path() -> Path:
    return workspace() / _INDEX_NAME


def _load_cache() -> dict[str, dict]:
    path = _cache_path()
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if int(blob.get("version", 0)) != _INDEX_VERSION:
        return {}
    return {str(k): v for k, v in blob.get("rows", {}).items()}


def _save_cache(rows: list[ClipRow]) -> None:
    payload = {"version": _INDEX_VERSION, "rows": {r.path: asdict(r) for r in rows}}
    _cache_path().write_text(json.dumps(payload), encoding="utf-8")


def build_index(which: str = "both", *, rebuild: bool = False,
                progress=None) -> Index:
    """Peek every clip's metadata, reusing cached rows whose mtime and size match."""
    cache = {} if rebuild else _load_cache()
    rows: list[ClipRow] = []
    fresh = 0
    for origin, store in stores(which):
        infos = store.list_clips()
        for i, info in enumerate(infos):
            hit = cache.get(str(info.path))
            if (hit and abs(float(hit.get("mtime", -1)) - info.mtime) < 1e-6
                    and int(hit.get("size_bytes", -1)) == info.size_bytes):
                rows.append(ClipRow(**hit))
                continue
            rows.append(_row_from_meta(info, store.peek_metadata(info.path), origin))
            fresh += 1
            if progress is not None and fresh % 100 == 0:
                progress(f"  peeked {fresh} new ({origin} {i + 1}/{len(infos)})")
    rows.sort(key=lambda r: (r.captured_at, r.clip_id))
    _save_cache(rows)
    return Index(rows)


def load_clip(row: ClipRow):
    """Full decode of one indexed clip."""
    return ClipStore(Path(row.path).parent).load(Path(row.path))


def _in_range(value: float, lo: float | None, hi: float | None) -> bool:
    if lo is not None and value < lo:
        return False
    return not (hi is not None and value > hi)


def filter_rows(rows: list[ClipRow], *, label: str = "", origin: str = "",
                session: str = "", trigger: str = "", unlabeled: bool = False,
                labelled_only: bool = False, sev_min: int | None = None,
                sev_max: int | None = None, since: str = "", until: str = "",
                notes: str = "", version: str = "",
                inconsistent: bool = False) -> list[ClipRow]:
    """Apply the metadata-only filters. Feature filters live in features.py."""
    wanted = {c.strip() for c in label.split(",") if c.strip()}
    out = []
    for r in rows:
        if wanted and r.label_class not in wanted:
            continue
        if unlabeled and r.labelled:
            continue
        if labelled_only and not r.labelled:
            continue
        if origin and r.origin != origin:
            continue
        if session and r.session_kind.upper() != session.upper():
            continue
        if trigger and trigger not in r.trigger_source:
            continue
        if not _in_range(r.severity, sev_min, sev_max):
            continue
        if since and r.captured_at < since:
            continue
        if until and r.captured_at > until:
            continue
        if notes and notes.lower() not in (r.notes or "").lower():
            continue
        if version and version not in r.client_version:
            continue
        if inconsistent and r.label_warning() is None:
            continue
        out.append(r)
    return out


def composition(rows: list[ClipRow]) -> dict:
    """Counts by label class, origin, session kind, trigger source and severity."""
    def tally(key):
        acc: dict[str, int] = {}
        for r in rows:
            name = key(r) or "(none)"
            acc[name] = acc.get(name, 0) + 1
        return dict(sorted(acc.items(), key=lambda kv: -kv[1]))

    return {
        "total": len(rows),
        "labelled": sum(1 for r in rows if r.labelled),
        "by_class": tally(lambda r: r.label_class),
        "by_origin": tally(lambda r: r.origin),
        "by_session": tally(lambda r: r.session_kind),
        "by_trigger": tally(lambda r: r.trigger_source),
        "by_severity": tally(lambda r: str(r.severity) if r.labelled else ""),
    }
