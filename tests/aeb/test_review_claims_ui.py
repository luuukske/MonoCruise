"""Review window claims: nobody tags a clip twice, and Lukas's saves send an id only."""
from __future__ import annotations

import time

import pytest

from PySide6.QtWidgets import QApplication

from core.aeb.clip_schema import Label
from core.aeb.clip_store import ClipStore
from tests.aeb.test_clip_capture import _make_clip
from tools import aeb_review_claims, aeb_review_remote
from tools.aeb_review import ReviewWindow
from tools.aeb_review_remote import Claim, ClaimResult, ReviewClient


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _private_outbox(tmp_path, monkeypatch):
    """Outbox and pending ids must never land in the real aeb_reviews directory."""
    monkeypatch.setattr(aeb_review_claims, "reviews_root", lambda: tmp_path / "reviews")
    monkeypatch.setattr(aeb_review_remote, "reviews_root", lambda: tmp_path / "reviews")


class _FakeClient:
    claim_of = ReviewClient.claim_of

    def __init__(self, *, admin: bool, granted: bool = True, holder: str = "") -> None:
        self.admin = admin
        self.granted = granted
        self.holder = holder
        self.done_ids: list[str] = []
        self.labels: list[tuple[str, dict]] = []
        self.claimed: list[str] = []

    def claim(self, cid):
        self.claimed.append(cid)
        if self.granted:
            return ClaimResult(True, Claim(state="open", by="lukas" if self.admin else "you", mine=True))
        return ClaimResult(False, self.claim_of({"state": "open", "by": self.holder}))

    def release(self, cid):
        return True

    def done(self, ids):
        self.done_ids.extend(ids)
        return len(ids)

    def states(self, ids):
        return {}

    def submit_label(self, cid, label):
        self.labels.append((cid, label))
        return 200, {"accepted": True, "need_clip": False}

    def upload_clip(self, cid, blob):
        return 200, {"accepted": True}


def _window(tmp_path, client, *, reviewer: bool):
    store = ClipStore(root=tmp_path / "clips")
    for cid in ("aaaa0001", "bbbb0002"):
        store.write(_make_clip(clip_id=cid))
    win = ReviewWindow(store, client=client, reviewer=reviewer)
    win._on_scanned([(i, store.peek_metadata(i.path)) for i in store.list_clips()])
    paths = {store.peek_metadata(i.path).clip_id: str(i.path) for i in store.list_clips()}
    return win, store, paths


def _wait(qapp, cond, timeout_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_a_clip_someone_else_holds_is_skipped_and_marked(tmp_path, qapp):
    win, _store, paths = _window(tmp_path, _FakeClient(admin=False), reviewer=True)
    try:
        win._claims = {"aaaa0001": Claim(state="open", by="lukas")}
        win._apply_filter()
        assert not win._is_untagged(paths["aaaa0001"])
        assert win._is_untagged(paths["bbbb0002"])
        row = win._row_of(paths["aaaa0001"])
        assert "being tagged by Lukas" in win._list.item(row).text()
    finally:
        win.close()


def test_a_refused_claim_says_who_has_it(tmp_path, qapp):
    client = _FakeClient(admin=True, granted=False, holder="r_aa")
    win, _store, paths = _window(tmp_path, client, reviewer=False)
    try:
        win._path = paths["aaaa0001"]
        win._claim_select(paths["aaaa0001"])
        assert _wait(qapp, lambda: "r_aa" in win._claim_lbl.text())
        assert win._claims["aaaa0001"].blocks
    finally:
        win.close()


def test_lukas_save_sends_the_clip_id_and_no_label(tmp_path, qapp):
    client = _FakeClient(admin=True)
    win, _store, paths = _window(tmp_path, client, reviewer=False)
    try:
        win._claim_saved(paths["aaaa0001"], Label(class_="tp", severity=3, notes="mine"))
        assert _wait(qapp, lambda: client.done_ids == ["aaaa0001"])
        assert client.labels == []
    finally:
        win.close()


def test_reviewer_save_sends_the_label_without_a_handle(tmp_path, qapp):
    client = _FakeClient(admin=False)
    win, _store, paths = _window(tmp_path, client, reviewer=True)
    try:
        win._claim_saved(paths["bbbb0002"], Label(class_="fp", severity=2, by="r_spoof"))
        assert _wait(qapp, lambda: len(client.labels) == 1)
        cid, payload = client.labels[0]
        assert cid == "bbbb0002" and payload["class"] == "fp" and "by" not in payload
        assert client.done_ids == []
    finally:
        win.close()
