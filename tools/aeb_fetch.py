"""Dev-only: pull contributed AEB clips into the local contributed store. Not shipped.

    MONOCRUISE_PULL_TOKEN=... python -m tools.aeb_fetch --list
    MONOCRUISE_PULL_TOKEN=... python -m tools.aeb_fetch --since 2026-08-01

Writes into `contributed_clip_root()`, never the local capture store, so a pull
can never evict clips recorded on this machine.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from core.aeb.clip_store import (
    ClipStore, contributed_clip_root, default_clip_root, deserialize_clip,
)

BASE_URL = "https://ld-tech.org/api/v1/aeb_pull.php"
_TOKEN_ENV = "MONOCRUISE_PULL_TOKEN"
_TIMEOUT = 60


@dataclass(frozen=True)
class PullResult:
    """Outcome of one pull_missing run (also used by the review UI)."""

    listed: int = 0
    already: int = 0
    saved: int = 0
    failed: int = 0
    landed: int = 0
    root: str = ""
    error: str | None = None


def _session(token: str):
    import requests

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def list_clips(http, base: str, **filters) -> list[dict]:
    """Index rows matching the filters. Not named `session`: that is a filter."""
    params = {"op": "list", **{k: v for k, v in filters.items() if v}}
    resp = http.get(base, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return list(resp.json().get("clips", []))


def fetch_clip(http, base: str, clip_id: str) -> bytes:
    resp = http.get(base, params={"op": "clip", "clip_id": clip_id}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def local_clip_ids(store: ClipStore) -> set[str]:
    """clip_ids already in the store, so a re-run only fetches what is missing."""
    out: set[str] = set()
    for info in store.list_clips():
        meta = store.peek_metadata(info.path)
        if meta is not None and meta.clip_id:
            out.add(meta.clip_id)
    return out


def safe_pull_root(store_root: Path | str | None = None) -> Path:
    """Target directory for a pull. Never the local capture store."""
    root = Path(store_root) if store_root is not None else contributed_clip_root()
    try:
        if root.resolve() == default_clip_root().resolve():
            return contributed_clip_root()
    except OSError:
        pass
    return root


def pull_missing(
    store: ClipStore,
    *,
    token: str | None = None,
    base_url: str = BASE_URL,
    since: str = "",
    until: str = "",
    trigger: str = "",
    session: str = "",
    limit: int = 500,
    on_progress: Callable[[str], None] | None = None,
) -> PullResult:
    """Download clips the store does not yet have. Never raises."""
    def note(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    tok = (token if token is not None else os.environ.get(_TOKEN_ENV, "")).strip()
    if not tok:
        return PullResult(root=str(store.root), error=f"set {_TOKEN_ENV} to the pull token")

    http = _session(tok)
    try:
        rows = list_clips(
            http, base_url,
            since=since, until=until,
            trigger=trigger, session=session, limit=limit,
        )
    except Exception as exc:
        return PullResult(root=str(store.root), error=f"list failed: {exc}")

    have = local_clip_ids(store)
    todo = [r for r in rows if r.get("clip_id") and r["clip_id"] not in have]
    note(f"{len(rows)} on server, {len(todo)} to fetch")

    saved = failed = 0
    for i, row in enumerate(todo, start=1):
        clip_id = row["clip_id"]
        note(f"fetching {i}/{len(todo)}: {clip_id[:8]}")
        try:
            blob = fetch_clip(http, base_url, clip_id)
            # Decode before writing: a truncated download must not land in the
            # store looking like a real clip, and this gets the store's naming.
            clip = deserialize_clip(blob)
            if store.write(clip) is None:
                raise OSError("store refused the write")
        except Exception:
            failed += 1
            continue
        saved += 1

    # Count what actually landed rather than trusting the loop. Store filenames
    # carry only 8 characters of the clip_id, so a collision would overwrite a
    # different clip while write() still reported success.
    landed = len(local_clip_ids(store) - have)
    return PullResult(
        listed=len(rows), already=len(have), saved=saved, failed=failed,
        landed=landed, root=str(store.root),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull contributed AEB clips.")
    parser.add_argument("--root", default=None, help="target store (default: contributed root)")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--since", default="", help="received on or after, YYYY-MM-DD")
    parser.add_argument("--until", default="", help="received on or before, YYYY-MM-DD")
    parser.add_argument("--trigger", default="", help="e.g. auto_engagement")
    parser.add_argument("--session", default="", help="SP or TMP")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--list", action="store_true", help="show what is there, download nothing")
    args = parser.parse_args(argv)

    token = os.environ.get(_TOKEN_ENV, "").strip()
    if not token:
        print(f"set {_TOKEN_ENV} to the pull token from the server config", file=sys.stderr)
        return 2

    root = safe_pull_root(args.root)
    store = ClipStore(root=root)
    http = _session(token)

    if args.list:
        try:
            rows = list_clips(
                http, args.base_url,
                since=args.since, until=args.until,
                trigger=args.trigger, session=args.session, limit=args.limit,
            )
        except Exception as exc:
            print(f"list failed: {exc}", file=sys.stderr)
            return 1
        print(f"{len(rows)} clip(s) on the server")
        for row in rows:
            print("  {clip_id}  {received_at}  {trigger_source:<16} {session_kind:<4} "
                  "{bytes:>8} B  v{client_version}".format(
                      clip_id=row.get("clip_id", "?"),
                      received_at=row.get("received_at", "?"),
                      trigger_source=row.get("trigger_source", "?"),
                      session_kind=row.get("session_kind", "?"),
                      bytes=row.get("bytes", 0),
                      client_version=row.get("client_version", "?")))
        return 0

    print(f"pulling into {root}")
    result = pull_missing(
        store, token=token, base_url=args.base_url,
        since=args.since, until=args.until,
        trigger=args.trigger, session=args.session, limit=args.limit,
        on_progress=print,
    )
    if result.error:
        print(result.error, file=sys.stderr)
        return 1

    print(f"{result.already} already local, saved {result.saved}, failed {result.failed}")
    if result.landed != result.saved:
        print(f"  {result.saved - result.landed} clip(s) did not survive the write, likely a "
              f"filename collision in {root}", file=sys.stderr)
        return 1
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
