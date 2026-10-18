"""Dev-only client for the review endpoint (claims, batches, labels). Not shipped.

Lukas claims with the pull token and sends only "done" for his saves, so his
labels never leave this machine. A reviewer uses their own token:

    MONOCRUISE_REVIEWER_TOKEN=... python -m tools.aeb_review --reviewer
    MONOCRUISE_PULL_TOKEN=... python -m tools.aeb_review_remote --seed-done

See `core/aeb/README.md` section 11, reviewers.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from core.aeb.clip_schema import ClipMetadata
from core.aeb.clip_store import (
    ClipStore, contributed_clip_root, default_clip_root, deserialize_clip,
)

logger = logging.getLogger(__name__)

REVIEW_URL = "https://ld-tech.org/api/v1/aeb_review.php"
REVIEWER_TOKEN_ENV = "MONOCRUISE_REVIEWER_TOKEN"
ADMIN_TOKEN_ENV = "MONOCRUISE_PULL_TOKEN"
_TIMEOUT = 30
_IDS_PER_CALL = 2000


def reviewer_clip_root() -> Path:
    """Batch clips a reviewer downloads, apart from every other store."""
    return default_clip_root().with_name("aeb_clips_review")


def reviews_root() -> Path:
    """Outbox, the label journal Lukas pulls, backups and disputes."""
    return default_clip_root().with_name("aeb_reviews")


def safe_review_root(root: Path | str | None = None) -> Path:
    """Target for batch downloads. Never the capture store or the contributed store."""
    target = Path(root) if root is not None else reviewer_clip_root()
    try:
        resolved = target.resolve()
        if resolved in (default_clip_root().resolve(), contributed_clip_root().resolve()):
            return reviewer_clip_root()
    except OSError:
        pass
    return target


def review_stores(*, root: str | None, contributed_only: bool,
                  reviewer: bool = False, own: bool = False) -> list[ClipStore]:
    """Stores the review tool opens. Default merges local + contributed; flags narrow it.

    A reviewer sees their own capture store (``own``) or their batch store, never both.
    """
    if reviewer:
        return [ClipStore(root=default_clip_root() if own else reviewer_clip_root())]
    if root:
        return [ClipStore(root=root)]
    if contributed_only:
        return [ClipStore(root=contributed_clip_root())]
    return [ClipStore(root=default_clip_root()), ClipStore(root=contributed_clip_root())]


@dataclass(frozen=True)
class Claim:
    """Claim state as the server shows it to the caller."""

    state: str = ""        # "open" or "done"
    by: str = ""           # lukas or a handle; reviewers see you / lukas / reviewer
    mine: bool = False

    @property
    def blocks(self) -> bool:
        """Someone else holds it or tagged it: skip it when moving on."""
        return bool(self.state) and not self.mine

    def describe(self) -> str:
        if self.by == "intake":
            return "already tagged"
        who = "Lukas" if self.by == "lukas" else ("another reviewer" if self.by == "reviewer" else self.by)
        if self.mine:
            return "claimed by you" if self.state == "open" else "tagged by you"
        return f"being tagged by {who}" if self.state == "open" else f"already tagged by {who}"


@dataclass(frozen=True)
class ClaimResult:
    granted: bool
    claim: Claim
    error: str | None = None


class ReviewClient:
    """Thin wrapper over the endpoint. Network failures raise; refusals come back as data."""

    def __init__(self, token: str, *, admin: bool, base_url: str = REVIEW_URL, http=None) -> None:
        self.admin = admin
        self.base_url = base_url
        if http is None:
            import requests

            http = requests.Session()
        http.headers.update({"Authorization": f"Bearer {token}"})
        self._http = http

    @classmethod
    def from_env(cls, *, admin: bool, base_url: str = REVIEW_URL) -> "ReviewClient | None":
        token = os.environ.get(ADMIN_TOKEN_ENV if admin else REVIEWER_TOKEN_ENV, "").strip()
        return cls(token, admin=admin, base_url=base_url) if token else None

    def _get(self, op: str, **params):
        return self._http.get(self.base_url, params={"op": op, **params}, timeout=_TIMEOUT)

    def _post(self, op: str, payload: dict | None = None, *, data: bytes | None = None, **params):
        if data is not None:
            return self._http.post(self.base_url, params={"op": op, **params}, data=data,
                                   headers={"Content-Type": "application/gzip"}, timeout=_TIMEOUT)
        return self._http.post(self.base_url, params={"op": op, **params},
                               json=payload or {}, timeout=_TIMEOUT)

    @staticmethod
    def _json(resp) -> dict:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        return body if isinstance(body, dict) else {}

    def claim_of(self, row: dict | None) -> Claim:
        row = row or {}
        by = str(row.get("by") or "")
        return Claim(state=str(row.get("state") or ""), by=by,
                     mine=by == "you" or (self.admin and by == "lukas"))

    def claim(self, clip_id: str) -> ClaimResult:
        resp = self._post("claim", {"clip_id": clip_id})
        body = self._json(resp)
        if resp.status_code != 200:
            return ClaimResult(False, Claim(), error=str(body.get("error") or resp.status_code))
        return ClaimResult(bool(body.get("granted")), self.claim_of(body))

    def release(self, clip_id: str) -> bool:
        resp = self._post("release", {"clip_id": clip_id})
        return resp.status_code == 200 and bool(self._json(resp).get("released"))

    def states(self, ids: list[str]) -> dict[str, Claim]:
        """Claims on these clips. Lukas reads the whole table in one call."""
        out: dict[str, Claim] = {}
        if self.admin:
            wanted = set(ids)
            now = time.time()
            for cid, row in self.claims_table().items():
                live = row.get("state") == "done" or float(row.get("until") or 0) > now
                if cid in wanted and live:
                    out[cid] = self.claim_of(row)
            return out
        for i in range(0, len(ids), _IDS_PER_CALL):
            resp = self._post("status", {"ids": ids[i:i + _IDS_PER_CALL]})
            resp.raise_for_status()
            for cid, row in (self._json(resp).get("states") or {}).items():
                out[cid] = self.claim_of(row)
        return out

    def claims_table(self) -> dict[str, dict]:
        """Lukas only: every claim with its real holder."""
        resp = self._get("claims")
        resp.raise_for_status()
        table = self._json(resp).get("claims") or {}
        return {str(k): v for k, v in table.items() if isinstance(v, dict)}

    def done(self, ids: list[str]) -> int:
        n = 0
        for i in range(0, len(ids), _IDS_PER_CALL):
            resp = self._post("done", {"ids": ids[i:i + _IDS_PER_CALL]})
            resp.raise_for_status()
            n += int(self._json(resp).get("done") or 0)
        return n

    def batch(self) -> list[str]:
        resp = self._get("batch")
        resp.raise_for_status()
        return [str(c) for c in self._json(resp).get("clips") or []]

    def fetch_clip(self, clip_id: str) -> bytes:
        resp = self._get("clip", clip_id=clip_id)
        resp.raise_for_status()
        return resp.content

    def submit_label(self, clip_id: str, label: dict) -> tuple[int, dict]:
        resp = self._post("label", {"clip_id": clip_id, "label": label})
        return resp.status_code, self._json(resp)

    def upload_clip(self, clip_id: str, blob: bytes) -> tuple[int, dict]:
        resp = self._post("upload", data=blob, clip_id=clip_id)
        return resp.status_code, self._json(resp)

    def reviews(self) -> list[dict]:
        resp = self._get("reviews")
        resp.raise_for_status()
        return [r for r in self._json(resp).get("records") or [] if isinstance(r, dict)]


def unlabelled_blob(path: Path | str) -> bytes:
    """A local clip as the server must receive it: same bytes layout, label removed."""
    data = json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))
    data["label"] = None
    return gzip.compress(json.dumps(data).encode("utf-8"), compresslevel=6)


def submit_payload(label_json: dict) -> dict:
    """What a reviewer sends: the label fields only. The server attaches who sent it."""
    keep = ("class", "severity", "should_trigger", "target_vid", "desired_peak_decel_ms2", "notes")
    return {k: label_json.get(k) for k in keep}


class Outbox:
    """Labels the server has not acknowledged yet, on disk so an outage loses none."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else reviews_root() / "outbox.json"

    def pending(self) -> list[dict]:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def _save(self, rows: list[dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(rows), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            logger.exception("could not save the review outbox")

    def add(self, clip_id: str, label_json: dict, clip_path: str) -> None:
        """Queue a label; a later save of the same clip replaces the earlier one."""
        rows = [r for r in self.pending() if r.get("clip_id") != clip_id]
        rows.append({"clip_id": clip_id, "label": submit_payload(label_json), "path": str(clip_path)})
        self._save(rows)

    def flush(self, client: ReviewClient) -> tuple[int, list[str]]:
        """Send what is queued. Returns (sent, problems). Stops at the first network failure."""
        rows = self.pending()
        sent, problems, left = 0, [], []
        for i, row in enumerate(rows):
            cid = str(row.get("clip_id") or "")
            try:
                status, body = client.submit_label(cid, row.get("label") or {})
                if status == 200 and body.get("accepted") and body.get("need_clip"):
                    up_status, up = client.upload_clip(cid, unlabelled_blob(row["path"]))
                    if up_status != 200:
                        problems.append(f"{cid[:8]}: clip upload refused ({up.get('error') or up_status})")
            except Exception as exc:
                logger.debug("review submission failed", exc_info=True)
                problems.append(f"offline: {exc.__class__.__name__}")
                left.extend(rows[i:])
                break
            if status == 200 and body.get("accepted"):
                sent += 1
            elif status == 409:
                claim = client.claim_of(body)
                problems.append(f"{cid[:8]}: {claim.describe() if claim.state else 'not claimed by you'}")
            elif status in (429, 500, 503):
                problems.append(f"{cid[:8]}: server busy ({body.get('error') or status})")
                left.append(row)
            else:
                problems.append(f"{cid[:8]}: refused ({body.get('error') or status})")
        self._save(left)
        return sent, problems


@dataclass(frozen=True)
class BatchResult:
    held: int = 0
    saved: int = 0
    failed: int = 0
    pruned: int = 0
    error: str | None = None


def peek_all(store: ClipStore) -> list[tuple[Path, ClipMetadata | None]]:
    """Every clip in the store with its metadata, read in parallel."""
    paths = [info.path for info in store.list_clips()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(zip(paths, pool.map(store.peek_metadata, paths)))


def _ids_by_path(store: ClipStore) -> dict[str, Path]:
    return {meta.clip_id: path for path, meta in peek_all(store) if meta is not None and meta.clip_id}


def pull_batch(
    client: ReviewClient,
    store: ClipStore,
    *,
    outbox: Outbox | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> BatchResult:
    """Drop finished clips, then download the open batch into ``store``. Never raises."""
    def note(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    if safe_review_root(store.root) != store.root:
        return BatchResult(error="refusing to download a batch into a corpus store")
    try:
        local = _ids_by_path(store)
        queued = {r.get("clip_id") for r in (outbox or Outbox()).pending()}
        pruned = 0
        # A clip the server has as tagged by you is finished: keep no copy of it.
        for cid, claim in client.states(sorted(local)).items():
            if claim.mine and claim.state == "done" and cid not in queued and store.delete(local[cid]):
                pruned += 1
                del local[cid]
        ids = client.batch()
    except Exception as exc:
        return BatchResult(error=f"batch failed: {exc}")
    todo = [cid for cid in ids if cid not in local]
    saved = failed = 0
    for i, cid in enumerate(todo, start=1):
        note(f"fetching {i}/{len(todo)}: {cid[:8]}")
        try:
            clip = deserialize_clip(client.fetch_clip(cid))
            if clip.metadata.clip_id != cid or store.write(clip) is None:
                raise OSError("clip did not land")
        except Exception:
            logger.debug("batch clip %s failed", cid[:8], exc_info=True)
            failed += 1
            continue
        saved += 1
    return BatchResult(held=len(ids), saved=saved, failed=failed, pruned=pruned)


def labelled_ids(stores: list[ClipStore]) -> list[str]:
    """Clips Lukas tagged himself (no reviewer handle), across these stores."""
    out: set[str] = set()
    for store in stores:
        for _path, meta in peek_all(store):
            if meta is not None and meta.clip_id and meta.label is not None and not meta.label.by:
                out.add(meta.clip_id)
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AEB review claims (Lukas's side).")
    parser.add_argument("--base-url", default=REVIEW_URL)
    parser.add_argument("--seed-done", action="store_true",
                        help="mark every clip you tagged as done, so no reviewer is handed it")
    parser.add_argument("--claims", action="store_true", help="summarise the claims table")
    args = parser.parse_args(argv)

    client = ReviewClient.from_env(admin=True, base_url=args.base_url)
    if client is None:
        print(f"set {ADMIN_TOKEN_ENV} to the pull token", file=sys.stderr)
        return 2
    if args.seed_done:
        stores = [ClipStore(root=default_clip_root()), ClipStore(root=contributed_clip_root())]
        ids = labelled_ids(stores)
        print(f"{len(ids)} tagged clip(s) found; marking done")
        try:
            print(f"server marked {client.done(ids)} done")
        except Exception as exc:
            print(f"done failed: {exc}", file=sys.stderr)
            return 1
    if args.claims:
        try:
            table = client.claims_table()
        except Exception as exc:
            print(f"claims failed: {exc}", file=sys.stderr)
            return 1
        counts = Counter((row.get("by", "?"), row.get("state", "?")) for row in table.values())
        for (by, state), n in sorted(counts.items()):
            print(f"  {by:<12} {state:<5} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
