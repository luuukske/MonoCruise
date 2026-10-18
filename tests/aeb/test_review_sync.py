"""Sync with server: order of steps, caps, what survives a failure, and the import gate."""
from __future__ import annotations

import threading
import time

import pytest

from PySide6.QtWidgets import QApplication

from core.aeb.clip_schema import Label
from core.aeb.clip_store import ClipStore, serialize_clip
from tests.aeb.test_clip_capture import _make_clip
from tests.aeb.test_fetch_tool import _FakeSession
from tools import aeb_fetch, aeb_review_claims, aeb_review_import, aeb_review_remote, aeb_review_sync
from tools.aeb_review import ReviewWindow
from tools.aeb_review_claims import ClaimsWorker, _PendingDone
from tools.aeb_review_remote import (
    Claim, ClaimResult, DoneResult, Outbox, ReviewClient, pull_batch,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _private_reviews(tmp_path, monkeypatch):
    """Outbox, pending ids, journal and backups never land in the real aeb_reviews directory."""
    for module in (aeb_review_claims, aeb_review_remote, aeb_review_import):
        monkeypatch.setattr(module, "reviews_root", lambda: tmp_path / "reviews")


class _HttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.response = type("R", (), {"status_code": status})()


class _Server:
    """Duck-typed ReviewClient over an in-memory claims table; records every call."""

    claim_of = ReviewClient.claim_of

    def __init__(self, *, admin: bool, batch=(), clips=None, records=(), offline=False) -> None:
        self.admin = admin
        self.calls: list[str] = []
        self.fetched: list[str] = []
        self.done_by_me: set[str] = set()
        self.batch_ids = list(batch)
        self.clips = clips or {}
        self.records = list(records)
        self.offline = offline
        self.fetch_errors: dict[str, Exception] = {}
        self.gate: threading.Event | None = None
        self.fetching = threading.Event()

    def _hit(self, what: str) -> None:
        self.calls.append(what)
        if self.offline:
            raise ConnectionError("down")

    def claim(self, cid):
        self._hit("claim")
        return ClaimResult(True, Claim(state="open", by="lukas" if self.admin else "you", mine=True))

    def release(self, cid):
        self._hit("release")
        return True

    def done(self, ids):
        self._hit("done")
        return DoneResult(done=len(ids))

    def states(self, ids):
        self._hit("states")
        by = "lukas" if self.admin else "you"
        return {c: Claim(state="done", by=by, mine=True) for c in ids if c in self.done_by_me}

    def submit_label(self, cid, label):
        self._hit("label")
        self.done_by_me.add(cid)
        return 200, {"accepted": True, "need_clip": False}

    def upload_clip(self, cid, blob):
        self._hit("upload")
        return 200, {"accepted": True}

    def batch(self):
        self._hit("batch")
        return list(self.batch_ids)

    def fetch_clip(self, cid):
        self._hit("fetch")
        self.fetched.append(cid)
        self.fetching.set()
        if self.gate is not None:
            self.gate.wait(10)
        if cid in self.fetch_errors:
            raise self.fetch_errors[cid]
        return self.clips[cid]

    def reviews(self):
        self._hit("reviews")
        return list(self.records)


def _wait(qapp, cond, timeout_s: float = 10.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def _store(tmp_path, *ids: str, name: str = "clips"):
    store = ClipStore(root=tmp_path / name)
    return store, {cid: store.write(_make_clip(clip_id=cid)) for cid in ids}


def _rec(cid: str, by: str, cls: str = "fp") -> dict:
    return {"clip_id": cid, "by": by, "received_at": "2026-10-15T10:00:00Z",
            "label": {"class": cls, "severity": 2, "should_trigger": None, "target_vid": 7,
                      "desired_peak_decel_ms2": None, "notes": "ghost"}}


def _settled(qapp, win, server) -> None:
    """Startup scan and its claims refresh have landed; the call log starts clean."""
    assert _wait(qapp, lambda: bool(win._entries)), "scan never landed"
    assert _wait(qapp, lambda: "states" in server.calls), "claims refresh never ran"
    server.calls.clear()


def _no_pull(monkeypatch):
    monkeypatch.setenv("MONOCRUISE_PULL_TOKEN", "a-token")
    monkeypatch.setattr(aeb_fetch, "_session", lambda token: _FakeSession([], {}))


# Hazard: the batch prune deletes every clip the server shows as "done, mine".

def test_the_pull_token_never_runs_the_batch_prune(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    server = _Server(admin=True, batch=["bbbb0002"])
    server.done_by_me.add("aaaa0001")
    res = pull_batch(server, store, outbox=Outbox(tmp_path / "outbox.json"))
    assert res.error and res.pruned == 0
    assert paths["aaaa0001"].exists()
    assert server.calls == []


def test_a_lukas_sync_keeps_every_clip_he_tagged(tmp_path, qapp, monkeypatch):
    _no_pull(monkeypatch)
    store, paths = _store(tmp_path, "aaaa0001", "bbbb0002")
    store.write_label(paths["aaaa0001"], Label(class_="tp", severity=3))
    server = _Server(admin=True, batch=["cccc0003"])
    server.done_by_me.update({"aaaa0001", "bbbb0002"})
    win = ReviewWindow(store, client=server, reviewer=False)
    try:
        _settled(qapp, win, server)
        win._update_btn.click()
        assert _wait(qapp, lambda: not win._pulling), "sync never finished"
        assert "batch" not in server.calls and "fetch" not in server.calls
        assert all(p.exists() for p in paths.values())
        text = win._status.text()
        assert "ids:" in text and "clips:" in text and "reviewer labels:" in text
    finally:
        win.close()


# Hazard: the prune must come after the flush, and see labels queued while it ran.

def test_a_reviewer_sync_sends_labels_before_it_prunes(tmp_path, qapp):
    store, paths = _store(tmp_path, "aaaa0001", "bbbb0002")
    server = _Server(admin=False, batch=["cccc0003"],
                     clips={"cccc0003": serialize_clip(_make_clip(clip_id="cccc0003"))})
    win = ReviewWindow(store, client=server, reviewer=True)
    try:
        _settled(qapp, win, server)
        Outbox().add("aaaa0001", {"class": "fp", "severity": 2}, str(paths["aaaa0001"]))
        win._update_btn.click()
        assert _wait(qapp, lambda: not win._pulling), "sync never finished"
        assert server.calls.index("label") < server.calls.index("states") < server.calls.index("batch")
        assert not paths["aaaa0001"].exists() and paths["bbbb0002"].exists()
        ids = {store.peek_metadata(i.path).clip_id for i in store.list_clips()}
        assert ids == {"bbbb0002", "cccc0003"}
        text = win._status.text()
        assert "labels: 1 sent" in text and "1 downloaded" in text and "1 finished removed" in text
    finally:
        win.close()


def test_a_label_queued_while_the_prune_reads_states_keeps_its_clip(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    outbox = Outbox(tmp_path / "outbox.json")
    server = _Server(admin=False)

    def states(ids):
        # The claims thread queues and sends a label between the store scan and this answer.
        outbox.add("aaaa0001", {"class": "fp", "severity": 2}, str(paths["aaaa0001"]))
        return {"aaaa0001": Claim(state="done", by="you", mine=True)}

    server.states = states
    assert pull_batch(server, store, outbox=outbox).pruned == 0
    assert paths["aaaa0001"].exists()


# Hazard: a batch download must not hold up claims.

def test_claims_answer_while_a_batch_downloads(tmp_path, qapp):
    store, paths = _store(tmp_path, "aaaa0001")
    server = _Server(admin=False, batch=["cccc0003"],
                     clips={"cccc0003": serialize_clip(_make_clip(clip_id="cccc0003"))})
    server.gate = threading.Event()
    win = ReviewWindow(store, client=server, reviewer=True)
    try:
        _settled(qapp, win, server)
        win._update_btn.click()
        assert _wait(qapp, lambda: server.fetching.is_set()), "batch never started fetching"
        win._path = str(paths["aaaa0001"])
        win._claim_select(str(paths["aaaa0001"]))
        assert _wait(qapp, lambda: "claimed by you" in win._claim_lbl.text()), "claim queued behind the batch"
        assert win._pulling and not server.gate.is_set()
        server.gate.set()
        assert _wait(qapp, lambda: not win._pulling), "sync never finished"
    finally:
        server.gate.set()
        win.close()


# Hazard: repeated presses burn the reviewer's daily caps.

def test_the_button_is_off_while_syncing_and_rate_limited_after(tmp_path, qapp):
    store, _paths = _store(tmp_path, "aaaa0001")
    server = _Server(admin=False, batch=["cccc0003"],
                     clips={"cccc0003": serialize_clip(_make_clip(clip_id="cccc0003"))})
    server.gate = threading.Event()
    win = ReviewWindow(store, client=server, reviewer=True)
    try:
        _settled(qapp, win, server)
        win._update_btn.click()
        assert _wait(qapp, lambda: server.fetching.is_set())
        assert not win._update_btn.isEnabled()
        aeb_review_sync.start(win)
        server.gate.set()
        assert _wait(qapp, lambda: not win._pulling), "sync never finished"
        assert win._update_btn.isEnabled()
        win._update_btn.click()
        qapp.processEvents()
        assert "next sync in" in win._status.text()
        assert server.calls.count("batch") == 1
    finally:
        server.gate.set()
        win.close()


def test_a_transient_fetch_failure_is_retried_once(tmp_path):
    store = ClipStore(root=tmp_path / "review")
    server = _Server(admin=False, batch=["cccc0003"])
    server.fetch_errors["cccc0003"] = ConnectionError("reset")
    res = pull_batch(server, store, outbox=Outbox(tmp_path / "outbox.json"))
    assert server.fetched == ["cccc0003", "cccc0003"]
    assert (res.saved, res.failed) == (0, 1)


def test_a_refused_fetch_is_not_retried(tmp_path):
    store = ClipStore(root=tmp_path / "review")
    server = _Server(admin=False, batch=["cccc0003"])
    server.fetch_errors["cccc0003"] = _HttpError(409)
    pull_batch(server, store, outbox=Outbox(tmp_path / "outbox.json"))
    assert server.fetched == ["cccc0003"]


def test_the_daily_clip_cap_stops_the_batch(tmp_path):
    store = ClipStore(root=tmp_path / "review")
    ids = ["cccc0003", "dddd0004", "eeee0005"]
    server = _Server(admin=False, batch=ids,
                     clips={c: serialize_clip(_make_clip(clip_id=c)) for c in ids})
    server.fetch_errors["dddd0004"] = _HttpError(429)
    res = pull_batch(server, store, outbox=Outbox(tmp_path / "outbox.json"))
    assert server.fetched == ["cccc0003", "dddd0004"]
    assert (res.saved, res.failed, res.left, res.capped) == (1, 1, 1, True)
    assert "daily clip limit" in aeb_review_sync.batch_text(res)


# Hazard: offline is the normal case; nothing queued may be lost.

def test_an_offline_reviewer_sync_keeps_the_queue_and_reports_each_step(tmp_path, qapp):
    store, paths = _store(tmp_path, "aaaa0001")
    server = _Server(admin=False)
    win = ReviewWindow(store, client=server, reviewer=True)
    try:
        _settled(qapp, win, server)
        server.offline = True
        Outbox().add("aaaa0001", {"class": "fp", "severity": 2}, str(paths["aaaa0001"]))
        win._update_btn.click()
        assert _wait(qapp, lambda: not win._pulling), "sync never finished"
        text = win._status.text()
        assert "labels: 0 sent, 1 kept" in text and "batch: batch failed" in text
        assert len(Outbox().pending()) == 1 and paths["aaaa0001"].exists()
    finally:
        win.close()


def test_lukas_ids_survive_an_offline_sync(tmp_path, qapp, monkeypatch):
    _no_pull(monkeypatch)
    _PendingDone().save(["aaaa0001"])
    store, _paths = _store(tmp_path, "aaaa0001")
    server = _Server(admin=True, offline=True)
    win = ReviewWindow(store, client=server, reviewer=False)
    try:
        assert _wait(qapp, lambda: bool(win._entries))
        win._update_btn.click()
        assert _wait(qapp, lambda: not win._pulling), "sync never finished"
        text = win._status.text()
        assert "ids: not sent (ConnectionError), 1 kept" in text and "clips:" in text
        assert "reviewer labels: not read (ConnectionError)" in text
        assert _PendingDone().load() == ["aaaa0001"]
    finally:
        win.close()


def _worker(server) -> tuple[ClaimsWorker, list]:
    worker = ClaimsWorker(server)
    got: list = []
    worker.result.connect(lambda op, payload: got.append((op, payload)))
    return worker, got


def test_held_ids_stay_pending_and_finished_ones_leave(qapp):
    server = _Server(admin=True)
    server.done = lambda ids: DoneResult(done=1, skipped=2, held=("bbbb0002",))
    _PendingDone().save(["aaaa0001", "bbbb0002", "cccc0003"])
    worker, got = _worker(server)
    worker.run("sync_done", None)
    res, pending, err = got[-1][1]
    assert got[-1][0] == "sync_pushed" and err == "" and pending == 1
    assert _PendingDone().load() == ["bbbb0002"]
    text = aeb_review_sync.done_text(res, pending)
    assert "1 marked done" in text and "2 skipped" in text and "1 sent again later" in text


def test_the_done_reply_carries_skipped_and_held():
    class _Http:
        headers: dict = {}

        def post(self, url, params=None, json=None, timeout=None):
            body = {"done": 1, "skipped": 1, "held": ["bbbb0002"]}
            return type("R", (), {"status_code": 200, "json": lambda self: body,
                                  "raise_for_status": lambda self: None})()

    client = ReviewClient("a-token-long-enough", admin=True, http=_Http())
    assert client.done(["aaaa0001", "bbbb0002"]) == DoneResult(done=1, skipped=1, held=("bbbb0002",))
    fork = client.fork()
    assert fork._http is not client._http and fork.admin
    assert fork._http.headers["Authorization"] == "Bearer a-token-long-enough"


# The import: a dry run, then a write only after an explicit yes.

def _import_window(tmp_path, qapp, monkeypatch):
    _no_pull(monkeypatch)
    store, paths = _store(tmp_path, "aaaa0001", "bbbb0002", "cccc0003", "dddd0004")
    store.write_label(paths["dddd0004"], Label(class_="tp", severity=4))
    records = [_rec("aaaa0001", "r_aa"), _rec("bbbb0002", "r_aa", "tn"),
               _rec("cccc0003", "r_aa", "fp"), _rec("cccc0003", "r_bb", "tp"),
               _rec("dddd0004", "r_aa"), _rec("eeee0005", "r_aa")]
    server = _Server(admin=True, records=records)
    win = ReviewWindow(store, client=server, reviewer=False)
    assert _wait(qapp, lambda: len(win._entries) == 4), "scan never landed"
    win._update_btn.click()
    assert _wait(qapp, lambda: not win._pulling), "sync never finished"
    return win, store, paths


def _labels(store, paths):
    return {cid: store.peek_metadata(p).label for cid, p in paths.items()}


def test_reviewer_labels_are_a_dry_run_until_confirmed(tmp_path, qapp, monkeypatch):
    win, store, paths = _import_window(tmp_path, qapp, monkeypatch)
    try:
        text = win._status.text()
        assert "2 to import" in text and "1 disputed" in text and "1 already labelled" in text
        assert "1 not in view" in text and "nothing written" in text
        assert not win._import_btn.isHidden() and "Import 2" in win._import_btn.text()
        before = _labels(store, paths)
        assert before["aaaa0001"] is None and before["bbbb0002"] is None
        monkeypatch.setattr(aeb_review_sync, "ask_to_apply", lambda win, plan: False)
        win._import_btn.click()
        assert "nothing written" in win._status.text()
        assert _labels(store, paths) == before
    finally:
        win.close()


def test_a_confirmed_import_skips_the_open_clip_and_refreshes_the_caches(tmp_path, qapp, monkeypatch):
    win, store, paths = _import_window(tmp_path, qapp, monkeypatch)
    try:
        asked = []
        monkeypatch.setattr(aeb_review_sync, "ask_to_apply",
                            lambda win, plan: asked.append(len(plan.write)) or True)
        open_path, other = str(paths["aaaa0001"]), str(paths["bbbb0002"])
        win._path = open_path
        win._cache[other] = object()
        win._import_btn.click()
        assert asked == [2]
        assert _wait(qapp, lambda: not win._pulling), "import never finished"
        labels = _labels(store, paths)
        assert labels["aaaa0001"] is None
        assert (labels["bbbb0002"].class_, labels["bbbb0002"].by) == ("tn", "r_aa")
        assert labels["cccc0003"] is None
        assert (labels["dddd0004"].class_, labels["dddd0004"].by) == ("tp", "")
        assert other not in win._cache
        assert "imported 1" in win._status.text() and "1 skipped" in win._status.text()
        assert win._import_btn.isHidden()
        assert list((tmp_path / "reviews" / "backup").glob("*/" + paths["bbbb0002"].name))
        assert _wait(qapp, lambda: (win._meta_cache.get(other) or (0, 0, None))[2] is not None
                     and win._meta_cache[other][2].label is not None), "rescan never saw the import"
    finally:
        win.close()


def test_save_waits_while_an_import_runs(tmp_path, qapp):
    store, paths = _store(tmp_path, "aaaa0001")
    win = ReviewWindow(store)
    try:
        win._path = str(paths["aaaa0001"])
        win._importing = True
        assert win._save() is False
        assert store.peek_metadata(paths["aaaa0001"]).label is None
    finally:
        win.close()


def test_a_clip_rewritten_while_it_decoded_is_read_again(tmp_path, qapp):
    store, paths = _store(tmp_path, "aaaa0001")
    path = str(paths["aaaa0001"])
    win = ReviewWindow(store)
    try:
        win._inflight = path
        win._awaiting = path
        aeb_review_sync.forget(win, [path])
        win._on_loaded(path, store.load(paths["aaaa0001"]), [], None, None)
        assert path not in win._cache
        assert win._inflight == path or path in win._queue
    finally:
        win.close()
