"""Review client: what leaves a reviewer's machine, and where batches land."""
from __future__ import annotations

import gzip
import json

from core.aeb.clip_schema import Label
from core.aeb.clip_store import ClipStore, contributed_clip_root, default_clip_root, serialize_clip
from tests.aeb.test_clip_capture import _make_clip
from tools.aeb_review_remote import (
    Claim, Outbox, ReviewClient, pull_batch, reviewer_clip_root, safe_review_root,
    submit_payload, unlabelled_blob,
)


class _FakeClient:
    """Duck-typed ReviewClient: records calls, answers from canned replies."""

    admin = False
    claim_of = ReviewClient.claim_of

    def __init__(self, *, label_reply=(200, {"accepted": True, "need_clip": False}),
                 offline=False, states=None, batch=(), clips=None) -> None:
        self.label_reply = label_reply
        self.offline = offline
        self._states = states or {}
        self._batch = list(batch)
        self._clips = clips or {}
        self.labels: list[tuple[str, dict]] = []
        self.uploads: list[tuple[str, bytes]] = []

    def submit_label(self, cid, label):
        if self.offline:
            raise ConnectionError("down")
        self.labels.append((cid, label))
        return self.label_reply

    def upload_clip(self, cid, blob):
        self.uploads.append((cid, blob))
        return 200, {"accepted": True}

    def states(self, ids):
        return {cid: c for cid, c in self._states.items() if cid in ids}

    def batch(self):
        return list(self._batch)

    def fetch_clip(self, cid):
        return self._clips[cid]


def _labelled_clip_path(tmp_path, cid="aaaa0001"):
    store = ClipStore(root=tmp_path / "own")
    path = store.write(_make_clip(clip_id=cid))
    store.write_label(path, Label(class_="fp", severity=2, notes="mine", by="r_aa"))
    return store, path


def test_an_uploaded_clip_carries_no_label_and_is_otherwise_unchanged(tmp_path):
    store, path = _labelled_clip_path(tmp_path)
    sent = json.loads(gzip.decompress(unlabelled_blob(path)))
    local = json.loads(gzip.decompress(path.read_bytes()))
    assert sent["label"] is None and local["label"] is not None
    assert {k: v for k, v in sent.items() if k != "label"} == {k: v for k, v in local.items() if k != "label"}
    assert list(sent).index("label") == list(sent).index("radar_frames") - 1


def test_a_submission_carries_no_handle():
    payload = submit_payload({"class": "tp", "severity": 3, "by": "r_aa", "notes": "x"})
    assert "by" not in payload and payload["class"] == "tp"


def test_claims_read_as_mine_for_the_right_caller():
    lukas = _FakeClient()
    lukas.admin = True
    assert lukas.claim_of({"state": "open", "by": "lukas"}).mine
    reviewer = _FakeClient()
    assert reviewer.claim_of({"state": "done", "by": "you"}).mine
    taken = reviewer.claim_of({"state": "done", "by": "lukas"})
    assert taken.blocks and taken.describe() == "already tagged by Lukas"
    arrived = reviewer.claim_of({"state": "done", "by": "intake"})
    assert arrived.blocks and arrived.describe() == "already tagged"
    assert not Claim().blocks


def test_batches_never_land_in_a_corpus_store(tmp_path):
    assert safe_review_root(default_clip_root()) == reviewer_clip_root()
    assert safe_review_root(contributed_clip_root()) == reviewer_clip_root()
    assert safe_review_root(tmp_path) == tmp_path
    result = pull_batch(_FakeClient(batch=["x"]), ClipStore(root=default_clip_root()))
    assert result.error and result.saved == 0


def test_a_clip_the_server_lacks_is_uploaded_without_its_label(tmp_path):
    _store, path = _labelled_clip_path(tmp_path)
    outbox = Outbox(tmp_path / "outbox.json")
    outbox.add("aaaa0001", {"class": "fp", "severity": 2, "by": "r_aa"}, str(path))
    client = _FakeClient(label_reply=(200, {"accepted": True, "need_clip": True}))
    sent, problems = outbox.flush(client)
    assert (sent, problems) == (1, [])
    assert json.loads(gzip.decompress(client.uploads[0][1]))["label"] is None
    assert outbox.pending() == []


def test_labels_are_kept_while_offline(tmp_path):
    outbox = Outbox(tmp_path / "outbox.json")
    outbox.add("aaaa0001", {"class": "fp", "severity": 2}, "unused")
    sent, problems = outbox.flush(_FakeClient(offline=True))
    assert sent == 0 and problems and len(outbox.pending()) == 1


def test_a_label_for_a_clip_lukas_tagged_is_dropped_and_said(tmp_path):
    outbox = Outbox(tmp_path / "outbox.json")
    outbox.add("aaaa0001", {"class": "fp", "severity": 2}, "unused")
    client = _FakeClient(label_reply=(409, {"accepted": False, "state": "done", "by": "lukas"}))
    sent, problems = outbox.flush(client)
    assert sent == 0 and "already tagged by Lukas" in problems[0]
    assert outbox.pending() == []


def test_a_later_save_replaces_the_queued_one(tmp_path):
    outbox = Outbox(tmp_path / "outbox.json")
    outbox.add("aaaa0001", {"class": "fp", "severity": 2}, "p")
    outbox.add("aaaa0001", {"class": "tp", "severity": 4}, "p")
    rows = outbox.pending()
    assert len(rows) == 1 and rows[0]["label"]["class"] == "tp"


def test_a_batch_downloads_new_clips_and_drops_finished_ones(tmp_path):
    store = ClipStore(root=tmp_path / "review")
    store.write(_make_clip(clip_id="aaaa0001"))
    client = _FakeClient(
        states={"aaaa0001": Claim(state="done", by="you", mine=True)},
        batch=["bbbb0002"],
        clips={"bbbb0002": serialize_clip(_make_clip(clip_id="bbbb0002"))},
    )
    result = pull_batch(client, store, outbox=Outbox(tmp_path / "outbox.json"))
    assert (result.error, result.saved, result.pruned) == (None, 1, 1)
    ids = {store.peek_metadata(i.path).clip_id for i in store.list_clips()}
    assert ids == {"bbbb0002"}


def test_a_finished_clip_still_queued_is_kept(tmp_path):
    store = ClipStore(root=tmp_path / "review")
    store.write(_make_clip(clip_id="aaaa0001"))
    outbox = Outbox(tmp_path / "outbox.json")
    outbox.add("aaaa0001", {"class": "fp", "severity": 2}, "p")
    client = _FakeClient(states={"aaaa0001": Claim(state="done", by="you", mine=True)})
    assert pull_batch(client, store, outbox=outbox).pruned == 0
    assert len(store.list_clips()) == 1
