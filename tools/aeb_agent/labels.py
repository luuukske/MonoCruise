"""Proposal, validation and journalled application of label changes.

An agent never writes a clip directly. It appends proposals to a JSONL file; a
separate `apply` step validates each one, refuses stale or malformed ones, and
records the previous label in an append-only journal so any change can be undone.
Applying is opt-in: without `--commit` the step is a dry run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.aeb.clip_schema import Label
from core.aeb.clip_score import class_window_warning
from core.aeb.clip_store import ClipStore

from tools.aeb_agent.corpus import LABEL_CLASSES, ClipRow, Index
from tools.aeb_agent.paths import workspace

PROPOSALS_NAME = "proposals.jsonl"
JOURNAL_NAME = "journal.jsonl"
# Windows within this of AEB's own reaction band are refused unless the caller
# says it checked, see core/aeb/README.md on seeding ground truth from AEB.
RECORDED_WINDOW_EPS = 0.05


@dataclass
class Proposal:
    """One requested label change, as an agent writes it."""

    clip_id: str
    label_class: str
    severity: int = 3
    window: list | None = None
    target_vid: int | None = None
    desired_peak_decel_ms2: float | None = None
    notes: str = ""
    rationale: str = ""
    confidence: float = 0.0
    reviewer: str = "agent"
    proposed_at: str = ""
    window_source: str = "judged"
    mtime: float = 0.0
    size_bytes: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Proposal":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Check:
    """Validation outcome for one proposal."""

    proposal: Proposal
    ok: bool
    problems: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    before: dict | None = None

    def line(self) -> str:
        state = "OK  " if self.ok else "SKIP"
        p = self.proposal
        win = "none" if not p.window else f"{p.window[0]:.2f}..{p.window[1]:.2f}"
        detail = "; ".join(self.problems or self.warnings)
        return (f"{state} {p.clip_id[:8]}  {p.label_class:<17} sev{p.severity} "
                f"win {win:<14} conf {p.confidence:.2f}  {detail}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def proposals_path(name: str = "") -> Path:
    return workspace() / (name or PROPOSALS_NAME)


def journal_path() -> Path:
    return workspace() / JOURNAL_NAME


def read_proposals(path: Path) -> list[Proposal]:
    if not path.is_file():
        return []
    out: list[Proposal] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(Proposal.from_json(json.loads(line)))
    return out


def append_proposal(prop: Proposal, path: Path | None = None) -> Path:
    """Append one proposal, stamping the time if the caller left it blank."""
    target = path or proposals_path()
    if not prop.proposed_at:
        prop.proposed_at = utc_now()
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(prop.to_json()) + "\n")
    return target


def _label_from(prop: Proposal) -> Label:
    window = None
    if prop.window:
        window = {"from_t": float(prop.window[0]), "to_t": float(prop.window[1])}
    return Label(
        class_=prop.label_class,
        severity=int(prop.severity),
        should_trigger=window,
        target_vid=prop.target_vid,
        desired_peak_decel_ms2=prop.desired_peak_decel_ms2,
        notes=prop.notes,
    )


def _stamp(prop: Proposal) -> str:
    """Notes carry who proposed it and how the window was arrived at."""
    tag = f"[{prop.reviewer} {prop.proposed_at or utc_now()} " \
          f"conf {prop.confidence:.2f} window:{prop.window_source}]"
    body = prop.notes.strip()
    if prop.rationale.strip():
        body = f"{body} {prop.rationale.strip()}".strip()
    return f"{tag} {body}".strip()


def validate(prop: Proposal, index: Index, feats_by_id: dict,
             *, min_confidence: float = 0.0,
             allow_recorded_window: bool = False) -> Check:
    problems: list[str] = []
    warnings: list[str] = []
    try:
        row: ClipRow | None = index.resolve(prop.clip_id)
    except KeyError as exc:
        return Check(prop, False, [str(exc)])

    if prop.label_class not in LABEL_CLASSES:
        problems.append(f"class {prop.label_class!r} not in {LABEL_CLASSES}")
    if prop.label_class != "ignore" and not 1 <= int(prop.severity) <= 5:
        problems.append(f"severity {prop.severity} outside 1..5")
    if prop.confidence < min_confidence:
        problems.append(f"confidence {prop.confidence:.2f} below "
                        f"{min_confidence:.2f}")
    if not prop.rationale.strip():
        problems.append("no rationale given")

    warn = class_window_warning(prop.label_class, bool(prop.window))
    if warn:
        problems.append(warn)

    feats = feats_by_id.get(row.clip_id)
    if prop.window:
        lo, hi = float(prop.window[0]), float(prop.window[1])
        if hi <= lo:
            problems.append(f"window {lo:.2f}..{hi:.2f} is empty or reversed")
        if feats is not None:
            if lo < -0.01 or hi > feats.duration_s + 0.5:
                problems.append(f"window outside the clip (0..{feats.duration_s:.2f})")
            band = feats.recorded_band
            if (band and abs(lo - band[0]) <= RECORDED_WINDOW_EPS
                    and abs(hi - band[1]) <= RECORDED_WINDOW_EPS):
                msg = ("window equals AEB's own reaction band; seeding ground "
                       "truth from AEB output encodes 'AEB was right'")
                if allow_recorded_window:
                    warnings.append(msg + " (allowed by flag)")
                else:
                    problems.append(msg)

    stale = (abs(row.mtime - prop.mtime) > 1e-6 or row.size_bytes != prop.size_bytes)
    if prop.mtime and stale:
        problems.append("clip changed on disk since the proposal was made")
    elif not prop.mtime:
        warnings.append("proposal carries no file stamp, staleness unchecked")

    if row.labelled and row.label_class != prop.label_class:
        warnings.append(f"overwrites existing class {row.label_class}")

    before = {
        "class": row.label_class, "severity": row.severity, "window": row.window,
        "target_vid": row.target_vid, "notes": row.notes,
    }
    return Check(prop, not problems, problems, warnings, before)


def apply_checks(checks: list[Check], index: Index, *, commit: bool = False) -> dict:
    """Write the passing proposals and journal each change. Dry run unless commit."""
    applied = skipped = failed = 0
    entries = []
    for check in checks:
        if not check.ok:
            skipped += 1
            continue
        targets = index.resolve_all(check.proposal.clip_id)
        if not commit:
            applied += 1
            continue
        prop = check.proposal
        prop.notes = _stamp(prop)
        label = _label_from(prop)
        written = []
        for row in targets:
            store = ClipStore(Path(row.path).parent)
            if store.write_label(Path(row.path), label):
                written.append(row)
        if not written:
            failed += 1
            continue
        applied += 1
        entries.append({
            "at": utc_now(),
            "clip_id": targets[0].clip_id,
            # Every copy written, so a revert can find them all again.
            "files": [Path(r.path).name for r in written],
            "reviewer": prop.reviewer,
            "before": check.before,
            "after": {
                "class": prop.label_class, "severity": prop.severity,
                "window": prop.window, "target_vid": prop.target_vid,
                "notes": prop.notes,
            },
            "rationale": prop.rationale,
            "confidence": prop.confidence,
        })
    if entries:
        with open(journal_path(), "a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")
    return {"applied": applied, "skipped": skipped, "failed": failed,
            "committed": commit}


def read_journal() -> list[dict]:
    path = journal_path()
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def revert(clip_ids: list[str], index: Index, *, commit: bool = False) -> dict:
    """Restore the pre-change label for each clip from its newest journal entry."""
    entries = read_journal()
    wanted = {c.strip() for c in clip_ids if c.strip()}
    newest: dict[str, dict] = {}
    for entry in entries:
        cid = entry["clip_id"]
        if wanted and not any(cid.startswith(w) for w in wanted):
            continue
        newest[cid] = entry
    done = failed = 0
    for cid, entry in newest.items():
        rows = index.resolve_all(cid)
        before = entry.get("before") or {}
        label = None
        if before.get("class"):
            win = before.get("window")
            label = Label(
                class_=before["class"],
                severity=int(before.get("severity") or 0),
                should_trigger=({"from_t": win[0], "to_t": win[1]} if win else None),
                target_vid=before.get("target_vid"),
                notes=before.get("notes", ""),
            )
        if not commit:
            done += 1
            continue
        ok = False
        for row in rows:
            store = ClipStore(Path(row.path).parent)
            ok = store.write_label(Path(row.path), label) or ok
        if ok:
            done += 1
        else:
            failed += 1
    return {"reverted": done, "failed": failed, "committed": commit}
