"""Dev-only: bring enrolled reviewers' labels into the local clip stores. Not shipped.

    MONOCRUISE_PULL_TOKEN=... python -m tools.aeb_review_import            # dry run
    MONOCRUISE_PULL_TOKEN=... python -m tools.aeb_review_import --apply
    python -m tools.aeb_review_import --remove-reviewer r_7f3a --apply

The only code allowed to move a reviewer's label into a clip (AGENTS.md, labels
are human-only). It never overwrites a label, every label it writes carries
the reviewer's handle, and its one bulk operation removes one reviewer's labels.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from core.aeb.clip_schema import ClipMetadata, Label
from core.aeb.clip_store import ClipStore, contributed_clip_root, default_clip_root
from tools.aeb_review_remote import (
    ADMIN_TOKEN_ENV, REVIEW_URL, ReviewClient, peek_all, reviews_root,
)

REVIEWER_RE = re.compile(r"^r_[a-z0-9]{2,16}$")
_CLASSES = {"tp", "good_intervention", "fp", "fn", "tn", "ignore"}


@dataclass
class ImportPlan:
    write: list[tuple[Path, dict]] = field(default_factory=list)   # path, label json with "by"
    kept: int = 0                                                  # clip already carries a label
    missing: list[str] = field(default_factory=list)               # no local copy of the clip
    disputed: dict[str, list[dict]] = field(default_factory=dict)  # reviewers disagree on class
    rejected: int = 0                                              # malformed record


def valid_record(rec: dict) -> bool:
    """A record the server could have produced: a reviewer handle and a sane label."""
    label = rec.get("label")
    if not isinstance(rec.get("clip_id"), str) or not isinstance(label, dict):
        return False
    if not REVIEWER_RE.match(str(rec.get("by") or "")):
        return False
    sev = label.get("severity")
    return label.get("class") in _CLASSES and isinstance(sev, int) and 1 <= sev <= 5


def latest_per_reviewer(records: list[dict]) -> dict[str, dict[str, dict]]:
    """clip_id -> by -> newest record. A reviewer's later save replaces their earlier one."""
    out: dict[str, dict[str, dict]] = {}
    for rec in sorted(records, key=lambda r: str(r.get("received_at") or "")):
        out.setdefault(rec["clip_id"], {})[rec["by"]] = rec
    return out


def plan_import(records: list[dict], local: dict[str, tuple[Path, ClipMetadata | None]]) -> ImportPlan:
    """What an import would do. Pure: reads nothing and writes nothing."""
    plan = ImportPlan()
    good = [r for r in records if isinstance(r, dict) and valid_record(r)]
    plan.rejected = len(records) - len(good)
    for clip_id, by_reviewer in latest_per_reviewer(good).items():
        hit = local.get(clip_id)
        if hit is None:
            plan.missing.append(clip_id)
            continue
        path, meta = hit
        if meta is None or meta.label is not None:
            plan.kept += 1
            continue
        recs = sorted(by_reviewer.values(), key=lambda r: str(r.get("received_at") or ""))
        if len({r["label"]["class"] for r in recs}) > 1:
            plan.disputed[clip_id] = recs
            continue
        first = recs[0]
        plan.write.append((path, {**first["label"], "by": first["by"]}))
    return plan


def _backup(path: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)


def apply_plan(plan: ImportPlan, store: ClipStore, backup_dir: Path) -> int:
    """Write the planned labels. Re-reads each clip first: a label set meanwhile is kept."""
    written = 0
    for path, label_json in plan.write:
        meta = store.peek_metadata(path)
        if meta is None or meta.label is not None or not REVIEWER_RE.match(label_json.get("by", "")):
            continue
        _backup(path, backup_dir)
        if store.write_label(path, Label.from_json(label_json)):
            written += 1
    return written


def removal_plan(local: dict[str, tuple[Path, ClipMetadata | None]], reviewer: str) -> list[Path]:
    """Every clip whose label this reviewer set. Never Lukas's: "" is not a handle."""
    if not REVIEWER_RE.match(reviewer):
        return []
    return [path for path, meta in local.values()
            if meta is not None and meta.label is not None and meta.label.by == reviewer]


def apply_removal(paths: list[Path], reviewer: str, store: ClipStore, backup_dir: Path) -> int:
    """Clear those labels. Re-reads each clip: a label someone else set since is kept."""
    if not REVIEWER_RE.match(reviewer):
        return 0
    removed = 0
    for path in paths:
        meta = store.peek_metadata(path)
        if meta is None or meta.label is None or meta.label.by != reviewer:
            continue
        _backup(path, backup_dir)
        if store.write_label(path, None):
            removed += 1
    return removed


def scan_local(stores: list[ClipStore]) -> dict[str, tuple[Path, ClipMetadata | None]]:
    """clip_id -> (path, metadata) over the given stores."""
    out: dict[str, tuple[Path, ClipMetadata | None]] = {}
    for store in stores:
        for path, meta in peek_all(store):
            if meta is not None and meta.clip_id:
                out.setdefault(meta.clip_id, (path, meta))
    return out


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import reviewer labels.")
    parser.add_argument("--base-url", default=REVIEW_URL)
    parser.add_argument("--apply", action="store_true", help="write; without it, only report")
    parser.add_argument("--remove-reviewer", default="", help="clear every label this handle set")
    args = parser.parse_args(argv)

    stores = [ClipStore(root=default_clip_root()), ClipStore(root=contributed_clip_root())]
    local = scan_local(stores)
    backup_dir = reviews_root() / "backup" / _stamp()

    if args.remove_reviewer:
        if not REVIEWER_RE.match(args.remove_reviewer):
            print("a reviewer handle looks like r_7f3a", file=sys.stderr)
            return 2
        paths = removal_plan(local, args.remove_reviewer)
        print(f"{len(paths)} label(s) by {args.remove_reviewer}")
        if args.apply:
            n = apply_removal(paths, args.remove_reviewer, stores[0], backup_dir)
            print(f"removed {n}; originals in {backup_dir}")
        return 0

    client = ReviewClient.from_env(admin=True, base_url=args.base_url)
    if client is None:
        print(f"set {ADMIN_TOKEN_ENV} to the pull token", file=sys.stderr)
        return 2
    try:
        records = client.reviews()
    except Exception as exc:
        print(f"reviews failed: {exc}", file=sys.stderr)
        return 1
    root = reviews_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "reviews.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    plan = plan_import(records, local)
    print(f"{len(records)} record(s): {len(plan.write)} to write, {plan.kept} already labelled, "
          f"{len(plan.disputed)} disputed, {len(plan.missing)} not local, {plan.rejected} rejected")
    if plan.missing:
        print("  not local: run 'Update from server' in the review tool, then import again")
    if plan.disputed:
        (root / "disputed.json").write_text(json.dumps(plan.disputed, indent=1), encoding="utf-8")
        for cid, recs in plan.disputed.items():
            print(f"  disputed {cid[:8]}: " + ", ".join(f"{r['by']}={r['label']['class']}" for r in recs))
    if args.apply and plan.write:
        n = apply_plan(plan, stores[0], backup_dir)
        print(f"wrote {n}; originals in {backup_dir}")
    elif plan.write:
        print("dry run: add --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
