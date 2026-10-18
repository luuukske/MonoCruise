"""Sync with server, the review window's one network button. Not shipped.

Reviewer: send the queued labels, then drop finished clips and download the
open batch. Lukas: send the ids of clips he tagged (never a label), pull new
contributed clips, then report what a reviewer-label import would write; it
writes only after he confirms. See `core/aeb/README.md` section 11, reviewers.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from core.aeb.clip_schema import ClipMetadata
from core.aeb.clip_store import ClipStore
from tools.aeb_review_import import (
    ImportPlan, apply_plan, new_backup_dir, plan_import, save_journal,
)
from tools.aeb_review_remote import BatchResult, DoneResult, Outbox, ReviewClient, pull_batch
from tools.aeb_review_widgets import pull_landed_in_view, pull_status_text

logger = logging.getLogger(__name__)

# Between two syncs. A batch fetch spends the reviewer's daily clip cap before the clip is read.
MIN_INTERVAL_S = 30.0

REVIEWER_TIP = ("Sends your saved labels, then removes clips the server has as finished and "
                "downloads your open batch. Labels only go up: a saved label can no longer be "
                "edited after the sync that sends it.")
LUKAS_TIP = ("Sends the ids of clips you tagged, never your labels. Pulls new contributed clips, "
             "then shows which reviewer labels an import would write. Reviewer labels come down "
             "only when you confirm the import.")


@dataclass(frozen=True)
class ReviewsOutcome:
    records: int = 0
    plan: ImportPlan | None = None
    error: str = ""


@dataclass(frozen=True)
class ApplyOutcome:
    planned: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()    # open in the window while the import ran
    written: int = 0
    backup: str = ""
    error: str = ""


def local_index(stores: list[ClipStore], known: dict) -> dict[str, tuple[Path, ClipMetadata | None]]:
    """clip_id -> (path, metadata). Reuses the window's metadata for files that did not change."""
    out: dict[str, tuple[Path, ClipMetadata | None]] = {}
    for store in stores:
        for info in store.list_clips():
            hit = known.get(str(info.path))
            if hit is not None and hit[0] == info.mtime and hit[1] == info.size_bytes:
                meta = hit[2]
            else:
                meta = store.peek_metadata(info.path)
            if meta is not None and meta.clip_id:
                out.setdefault(meta.clip_id, (info.path, meta))
    return out


def review_plan(client: ReviewClient, stores: list[ClipStore], known: dict) -> ReviewsOutcome:
    """Every reviewer record, and what an import would do with it. A dry run: writes no clip."""
    try:
        records = client.reviews()
    except Exception as exc:
        logger.debug("reading reviewer records failed", exc_info=True)
        return ReviewsOutcome(error=exc.__class__.__name__)
    plan = plan_import(records, local_index(stores, known))
    try:
        save_journal(records, plan)
    except OSError:
        logger.warning("could not save the reviewer label journal")
    return ReviewsOutcome(records=len(records), plan=plan)


def apply_reviewer_labels(plan: ImportPlan, store: ClipStore, *, skip) -> ApplyOutcome:
    """The confirmed import, through the import module's own apply. Never raises."""
    skipped: list[str] = []

    def leave(path) -> bool:
        if skip(path):
            skipped.append(str(path))
            return True
        return False

    planned = tuple(str(path) for path, _label in plan.write)
    backup = new_backup_dir()
    try:
        written = apply_plan(plan, store, backup, skip=leave)
    except Exception as exc:
        logger.warning("reviewer label import stopped: %s", exc.__class__.__name__)
        return ApplyOutcome(planned=planned, skipped=tuple(skipped), backup=backup.name,
                            error=exc.__class__.__name__)
    return ApplyOutcome(planned=planned, skipped=tuple(skipped), written=written, backup=backup.name)


def done_text(res: DoneResult | None, pending: int, err: str = "") -> str:
    if res is None:
        return f"ids: not sent ({err or 'failed'}), {pending} kept for the next sync"
    if not res.done and not res.skipped:
        return "ids: nothing to send"
    text = f"ids: {res.done} marked done"
    if res.skipped:
        text += f", {res.skipped} skipped because a reviewer holds or tagged them"
    if res.held:
        text += f" ({len(res.held)} sent again later)"
    return text


def flush_text(sent: int, problems: list[str], left: int) -> str:
    if not (sent or problems or left):
        return "labels: nothing queued"
    text = f"labels: {sent} sent"
    if left:
        text += f", {left} kept for the next sync"
    if problems:
        text += "; " + "; ".join(problems[:3])
    return text


def batch_text(res: BatchResult) -> str:
    if res.error:
        return f"batch: {res.error}"
    text = (f"batch: {res.held} held, {res.saved} downloaded, {res.failed} failed, "
            f"{res.pruned} finished removed")
    if res.capped:
        text += f"; daily clip limit reached, {res.left} left for tomorrow"
    elif res.left:
        text += f"; stopped, {res.left} not fetched"
    return text


def reviews_text(out: ReviewsOutcome) -> str:
    if out.error or out.plan is None:
        return f"reviewer labels: not read ({out.error or 'failed'})"
    plan = out.plan
    text = (f"reviewer labels: {out.records} record(s), {len(plan.write)} to import, "
            f"{plan.kept} already labelled, {len(plan.disputed)} disputed (left alone), "
            f"{len(plan.missing)} not in view, {plan.rejected} rejected")
    return text + ("; dry run, nothing written" if plan.write else "")


class SyncWorker(QObject):
    """The slow sync steps on their own thread, so claims keep moving while a batch downloads."""

    result = Signal(str, object)  # op, payload

    def __init__(self, client: ReviewClient) -> None:
        super().__init__()
        self._client = client
        self.stopping = threading.Event()

    @Slot(str, object)
    def run(self, op: str, arg) -> None:
        try:
            if op == "batch":
                res = pull_batch(self._client, ClipStore(root=arg), outbox=Outbox(),
                                 on_progress=lambda msg: self.result.emit("progress", msg),
                                 should_stop=self.stopping.is_set)
            elif op == "reviews":
                res = review_plan(self._client, *arg)
            elif op == "apply":
                plan, store, skip = arg
                res = apply_reviewer_labels(
                    plan, store, skip=lambda p: self.stopping.is_set() or skip(p))
            else:
                return
        except Exception as exc:
            logger.debug("sync step %s failed", op, exc_info=True)
            self.result.emit("failed", (op, exc.__class__.__name__))
            return
        self.result.emit(op, res)


class _SyncBridge(QObject):
    """GUI-thread side: sends steps to the worker, lands its results."""

    request = Signal(str, object)

    def __init__(self, win) -> None:
        super().__init__(win)
        self._win = win

    @Slot(str, object)
    def landed(self, op: str, payload) -> None:
        on_result(self._win, op, payload)


def sync_setup(win, client: ReviewClient | None) -> None:
    """After `claims_setup`. The worker gets its own HTTP session: a Session is not thread-safe."""
    win._sync_started = float("-inf")
    win._sync_lines = []
    win._import_plan = None
    win._importing = False
    win._stale = set()
    win._sync_thread = None
    win._update_btn.setToolTip(REVIEWER_TIP if win._reviewer else LUKAS_TIP)
    if client is None:
        return
    win._sync_thread = QThread(win)
    win._sync_worker = SyncWorker(client.fork() if isinstance(client, ReviewClient) else client)
    win._sync_worker.moveToThread(win._sync_thread)
    win._sync_bridge = _SyncBridge(win)
    win._sync_bridge.request.connect(win._sync_worker.run)
    win._sync_worker.result.connect(win._sync_bridge.landed)
    win._sync_thread.start()


def stop_sync(win) -> None:
    thread = getattr(win, "_sync_thread", None)
    if thread is None:
        return
    win._sync_worker.stopping.set()
    thread.quit()
    thread.wait(5000)


def _ask(win, op: str, arg) -> None:
    if win._sync_thread is None:
        win._sync_lines.append(f"{op}: no review client")
        _finish(win)
        return
    win._sync_bridge.request.emit(op, arg)


def _finish(win) -> None:
    win._pulling = False
    win._update_btn.setEnabled(True)
    win._import_btn.setEnabled(True)
    win._status.setText("\n".join(win._sync_lines))


def _drop_import(win) -> None:
    win._import_plan = None
    win._import_btn.setVisible(False)


def start(win) -> None:
    """The button. One sync at a time, and none within MIN_INTERVAL_S of the last."""
    if win._pulling:
        return
    since = time.monotonic() - win._sync_started
    if since < MIN_INTERVAL_S:
        win._status.setText(f"synced {since:.0f} s ago; next sync in {MIN_INTERVAL_S - since:.0f} s")
        return
    if win._reviewer and win._claim_client is None:
        win._status.setText("no reviewer token: set MONOCRUISE_REVIEWER_TOKEN")
        return
    win._sync_started = time.monotonic()
    win._pulling = True
    win._update_btn.setEnabled(False)
    _drop_import(win)
    win._sync_lines = []
    if win._claim_client is None:
        _pull(win)
        return
    win._status.setText("sync: sending labels..." if win._reviewer else "sync: sending tagged clip ids...")
    # Step 1 runs on the claims thread, the only writer of the outbox and the pending ids.
    win.remote_requested.emit("sync_flush" if win._reviewer else "sync_done", None)


def on_queue_sent(win, payload, err: str = "") -> None:
    """Step 1 landed. Only then the prune: a label upload may still read the local clip."""
    if not win._pulling:
        return
    if win._reviewer:
        win._sync_lines.append(flush_text(*payload) if payload else f"labels: not sent ({err}), queue kept")
        win._status.setText("sync: getting batch...")
        _ask(win, "batch", str(win._store.root))
    else:
        win._sync_lines.append(done_text(*payload) if payload else done_text(None, 0, err))
        _pull(win)


def _pull(win) -> None:
    root = win._pull_target()
    win._status.setText(f"sync: pulling contributed clips into {root.name}...")
    win.pull_requested.emit(root)


def on_pulled(win, result) -> None:
    """Lukas's step 2 landed: a tools.aeb_fetch.PullResult."""
    text = pull_status_text(result)
    if result.landed and pull_landed_in_view(result, win._stores):
        win._refresh_clips()
    if not win._pulling:
        win._status.setText(text)
        return
    win._sync_lines.append(f"clips: {text}")
    if win._claim_client is None:
        win._sync_lines.append("reviewer labels: not read, no pull token when the window opened")
        _finish(win)
        return
    win._status.setText("sync: reading reviewer labels...")
    _ask(win, "reviews", (list(win._stores), dict(win._meta_cache)))


def on_result(win, op: str, payload) -> None:
    if op == "progress":
        win._status.setText(f"sync: {payload}")
    elif op == "batch":
        win._sync_lines.append(batch_text(payload))
        win._refresh_clips()
        _finish(win)
    elif op == "reviews":
        win._sync_lines.append(reviews_text(payload))
        if payload.plan is not None and payload.plan.write:
            win._import_plan = payload.plan
            win._import_btn.setText(f"Import {len(payload.plan.write)} reviewer label(s)...")
            win._import_btn.setVisible(True)
        _finish(win)
    elif op == "apply":
        _applied(win, payload)
    elif op == "failed":
        step, err = payload
        win._sync_lines.append(f"{step}: failed ({err})")
        win._importing = False
        _finish(win)


def ask_to_apply(win, plan: ImportPlan) -> bool:
    """Lukas's explicit yes. Disputed clips and labelled clips are never in the plan."""
    reply = QMessageBox.question(
        win, "Import reviewer labels",
        f"Write {len(plan.write)} reviewer label(s) into local clips?\n\n"
        "Each keeps its reviewer's handle. A clip labelled by then is left alone, "
        f"{len(plan.disputed)} disputed clip(s) are left alone, and the clip open in this "
        "window is skipped. Originals are backed up under aeb_reviews/backup/.",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
    )
    return reply == QMessageBox.Yes


def import_labels(win) -> None:
    """The import button. Writes the dry run's labels only after an explicit yes."""
    plan = win._import_plan
    if plan is None or win._pulling or win._sync_thread is None:
        return
    if not ask_to_apply(win, plan):
        win._status.setText("import cancelled, nothing written")
        return
    win._pulling = win._importing = True
    win._update_btn.setEnabled(False)
    win._import_btn.setEnabled(False)
    win._sync_lines = []
    win._status.setText(f"importing {len(plan.write)} reviewer label(s)...")
    _ask(win, "apply", (plan, win._stores[0], lambda path: str(path) == str(win._path)))


def forget(win, paths) -> None:
    """Drop decoded clips and metadata of rewritten files. A queued decode has not read its file yet."""
    for path in paths:
        win._cache.pop(path, None)
        win._meta_cache.pop(path, None)
        if path == win._inflight:
            win._stale.add(path)


def _applied(win, out: ApplyOutcome) -> None:
    win._importing = False
    touched = [p for p in out.planned if p not in out.skipped]
    forget(win, touched)
    current = str(win._path) if win._path else ""
    if current in touched:
        # Opened while the import ran, so the form shows the file from before it.
        win._form_path = None
        win._awaiting = current
        win._request(current, urgent=True)
    win._refresh_clips()
    text = f"imported {out.written} reviewer label(s), originals in aeb_reviews/backup/{out.backup}"
    if out.error:
        text = f"import stopped ({out.error}); what it rewrote is backed up in aeb_reviews/backup/{out.backup}"
    if out.skipped:
        text += f"; {len(out.skipped)} skipped while open in this window, sync again after moving on"
    win._sync_lines.append(text)
    _drop_import(win)
    _finish(win)
