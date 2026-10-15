"""Claims and reviewer submission for the AEB clip review window. Not shipped.

Selecting a clip claims it on the server so nobody tags it twice; saving marks
it done. Lukas's saves send the clip id only. A reviewer's save queues the
label in an on-disk outbox. See `core/aeb/README.md` section 11, reviewers.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from core.aeb.clip_store import ClipStore
from tools.aeb_review_remote import Claim, Outbox, ReviewClient, pull_batch, reviews_root

logger = logging.getLogger(__name__)

_REFRESH_MS = 120_000
_BLOCKED_STYLE = "color:#e0a020;"
_MINE_STYLE = "color:#50d282;"
_QUIET_STYLE = "color:#999;"


class _PendingDone:
    """Clip ids Lukas saved that the server has not marked done yet."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else reviews_root() / "done_pending.json"

    def load(self) -> list[str]:
        try:
            ids = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [str(i) for i in ids] if isinstance(ids, list) else []

    def save(self, ids: list[str]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(sorted(set(ids))), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            logger.exception("could not save pending done ids")


class ClaimsWorker(QObject):
    """Every network call off the GUI thread. One op at a time, in order."""

    result = Signal(str, object)  # op, payload

    def __init__(self, client: ReviewClient) -> None:
        super().__init__()
        self._client = client
        self._outbox = Outbox()
        self._pending_done = _PendingDone()

    @Slot(str, object)
    def run(self, op: str, arg) -> None:
        try:
            self._run(op, arg)
        except Exception as exc:
            logger.debug("review %s failed", op, exc_info=True)
            self.result.emit("error", (op, exc.__class__.__name__))

    def _run(self, op: str, arg) -> None:
        client = self._client
        if op == "claim":
            path, cid = arg
            self.result.emit("claim", (path, cid, client.claim(cid)))
        elif op == "release":
            client.release(arg)
        elif op == "done":
            ids = self._pending_done.load() + ([arg] if arg else [])
            self._pending_done.save(ids)
            if ids:
                client.done(ids)
                self._pending_done.save([])
        elif op == "submit":
            cid, label_json, path = arg
            self._outbox.add(cid, label_json, path)
            self.result.emit("submitted", self._outbox.flush(client))
        elif op == "flush":
            if self._outbox.pending():
                self.result.emit("submitted", self._outbox.flush(client))
        elif op == "states":
            self.result.emit("states", client.states(arg))
        elif op == "batch":
            res = pull_batch(client, ClipStore(root=arg), outbox=self._outbox,
                             on_progress=lambda msg: self.result.emit("progress", msg))
            self.result.emit("batch", res)


class _ClaimsBridge(QObject):
    """GUI-thread landing for the worker. A plain function would run on the worker."""

    def __init__(self, win) -> None:
        super().__init__(win)
        self._win = win

    @Slot(str, object)
    def result(self, op: str, payload) -> None:
        on_result(self._win, op, payload)


def claims_setup(win, client: ReviewClient | None, *, reviewer: bool) -> None:
    """Claim state, hooks the window calls, and the worker thread behind them."""
    win._claims = {}
    win._claim_client = client
    win._reviewer = reviewer
    win._claimed_id = None
    win._saved_ids = set()
    win._claim_select = lambda path: on_select(win, path)
    win._claim_saved = lambda path, label: on_saved(win, path, label)
    win._claim_of = lambda meta: claim_of(win, meta)
    win._claims_refresh = lambda: refresh(win)
    win._claims_batch = lambda root: request_batch(win, root)
    win._stop_claims = lambda: stop_claims(win)
    win._claims_thread = None
    if client is None:
        win._claim_lbl.setText("" if not reviewer else "no reviewer token: set MONOCRUISE_REVIEWER_TOKEN")
        return
    win._claims_thread = QThread(win)
    win._claims_worker = ClaimsWorker(client)
    win._claims_worker.moveToThread(win._claims_thread)
    win._claims_bridge = _ClaimsBridge(win)
    win.remote_requested.connect(win._claims_worker.run)
    win._claims_worker.result.connect(win._claims_bridge.result)
    win._claims_thread.start()
    win._claims_timer = QTimer(win)
    win._claims_timer.timeout.connect(lambda: refresh(win))
    win._claims_timer.start(_REFRESH_MS)
    win.remote_requested.emit("flush" if reviewer else "done", None if reviewer else "")


def stop_claims(win) -> None:
    thread = getattr(win, "_claims_thread", None)
    if thread is None:
        return
    if win._claimed_id and win._claimed_id not in win._saved_ids:
        win.remote_requested.emit("release", win._claimed_id)
    thread.quit()
    thread.wait(3000)


def claim_of(win, meta) -> Claim | None:
    if meta is None or not meta.clip_id:
        return None
    return win._claims.get(meta.clip_id)


def _clip_id(win, path: str) -> str:
    cached = win._meta_cache.get(str(path))
    meta = cached[2] if cached else None
    return meta.clip_id if meta is not None else ""


def _show(win, text: str, style: str) -> None:
    win._claim_lbl.setText(text)
    win._claim_lbl.setStyleSheet(style)


def on_select(win, path: str) -> None:
    """Release the clip left behind unsaved, then claim the new one."""
    if win._claim_client is None:
        return
    cid = _clip_id(win, path)
    prev = win._claimed_id
    if prev and prev != cid and prev not in win._saved_ids:
        win.remote_requested.emit("release", prev)
    win._claimed_id = None
    if not cid:
        _show(win, "", _QUIET_STYLE)
        return
    known = win._claims.get(cid)
    if known is not None and known.blocks:
        _show(win, f"{known.describe()}: you can look, someone else has it", _BLOCKED_STYLE)
    else:
        _show(win, "claiming...", _QUIET_STYLE)
    win.remote_requested.emit("claim", (path, cid))


def on_saved(win, path: str, label) -> None:
    """Lukas: mark the clip done (id only). Reviewer: queue the label for the server."""
    if win._claim_client is None:
        return
    cid = _clip_id(win, path)
    if not cid:
        return
    win._saved_ids.add(cid)
    win._claims[cid] = Claim(state="done", by="you" if win._reviewer else "lukas", mine=True)
    if win._reviewer:
        win.remote_requested.emit("submit", (cid, label.to_json(), str(path)))
        _show(win, "sending label...", _QUIET_STYLE)
    else:
        win.remote_requested.emit("done", cid)
        _show(win, "tagged by you", _MINE_STYLE)
    update_row(win, path)


def refresh(win) -> None:
    """Re-read claims for every listed clip, and retry anything queued."""
    if win._claim_client is None:
        return
    ids = sorted({meta.clip_id for _info, meta, _o in win._entries if meta is not None and meta.clip_id})
    if ids:
        win.remote_requested.emit("states", ids)
    win.remote_requested.emit("flush" if win._reviewer else "done", None if win._reviewer else "")


def request_batch(win, root) -> None:
    if win._claim_client is None:
        win._status.setText("no reviewer token: set MONOCRUISE_REVIEWER_TOKEN")
        win._pulling = False
        win._update_btn.setEnabled(True)
        return
    win.remote_requested.emit("batch", str(root))


def update_row(win, path: str) -> None:
    """Redraw one list row after its claim changed, without rebuilding the list."""
    from tools.aeb_review_widgets import _clip_item

    row = win._row_of(str(path))
    if row < 0:
        return
    for info, meta, origin in win._entries:
        if str(info.path) == str(path):
            fresh = _clip_item(info, meta, origin, claim_of(win, meta))
            win._list.item(row).setText(fresh.text())
            return


def on_result(win, op: str, payload) -> None:
    if op == "claim":
        path, cid, res = payload
        current = _clip_id(win, win._path) if win._path else ""
        if res.error:
            if cid == current:
                _show(win, f"claim failed ({res.error}); tag anyway, it may be tagged twice", _QUIET_STYLE)
            return
        if res.claim.state:
            win._claims[cid] = res.claim
        if not res.granted:
            if cid == current:
                _show(win, f"{res.claim.describe()}: you can look, someone else has it", _BLOCKED_STYLE)
        elif cid != current and cid not in win._saved_ids:
            win.remote_requested.emit("release", cid)
        else:
            win._claimed_id = cid
            _show(win, res.claim.describe() if res.claim.state else "claimed by you", _MINE_STYLE)
        update_row(win, path)
    elif op == "states":
        if payload != win._claims:
            win._claims = dict(payload)
            win._apply_filter()
    elif op == "submitted":
        sent, problems = payload
        text = f"sent {sent} label(s)" if sent else ""
        if problems:
            text = (text + "; " if text else "") + "; ".join(problems[:3])
        if text:
            win._status.setText(text)
            _show(win, "label sent" if sent and not problems else text,
                  _MINE_STYLE if sent and not problems else _BLOCKED_STYLE)
    elif op == "progress":
        win._status.setText(str(payload))
    elif op == "batch":
        win._pulling = False
        win._update_btn.setEnabled(True)
        if payload.error:
            win._status.setText(payload.error)
            return
        win._status.setText(
            f"batch: {payload.held} held, {payload.saved} downloaded, "
            f"{payload.failed} failed, {payload.pruned} finished removed")
        win._refresh_clips()
    elif op == "error":
        what, err = payload
        if what == "batch":
            win._pulling = False
            win._update_btn.setEnabled(True)
        if what in ("claim", "batch", "submit"):
            win._status.setText(f"server unreachable ({what}: {err}); work is kept and retried")
