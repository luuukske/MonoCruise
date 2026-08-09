"""aeb_fetch: skip what is already local, validate before storing, stay off the local store."""
from __future__ import annotations

import pytest

from core.aeb.clip_store import ClipStore, contributed_clip_root, default_clip_root, serialize_clip
from tools import aeb_fetch

from tests.aeb.test_clip_review import _build_replayable_clip


class _FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeSession:
    """Stands in for requests.Session; records every call it is given."""

    def __init__(self, rows, blobs):
        self._rows = rows
        self._blobs = blobs
        self.headers: dict[str, str] = {}
        self.fetched: list[str] = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        if params.get("op") == "list":
            return _FakeResponse(payload={"clips": self._rows, "count": len(self._rows)})
        clip_id = params["clip_id"]
        self.fetched.append(clip_id)
        return _FakeResponse(content=self._blobs[clip_id])


def _clip_blob(clip_id: str) -> bytes:
    clip = _build_replayable_clip()
    clip.metadata.clip_id = clip_id
    return serialize_clip(clip)


@pytest.fixture()
def wired(monkeypatch, tmp_path):
    monkeypatch.setenv("MONOCRUISE_PULL_TOKEN", "a-token")
    ids = ["11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"]
    rows = [{"clip_id": i, "received_at": "2026-08-09T10:00:00Z", "trigger_source": "auto_engagement",
             "session_kind": "SP", "bytes": 100, "client_version": "1.1.0"} for i in ids]
    session = _FakeSession(rows, {i: _clip_blob(i) for i in ids})
    monkeypatch.setattr(aeb_fetch, "_session", lambda token: session)
    return session, ids, tmp_path


def test_the_contributed_root_is_not_the_local_store():
    """A pull must never be able to evict clips recorded on this machine."""
    assert contributed_clip_root() != default_clip_root()
    assert contributed_clip_root().name == "aeb_clips_contributed"


def test_a_missing_token_refuses_to_run(monkeypatch, tmp_path):
    monkeypatch.delenv("MONOCRUISE_PULL_TOKEN", raising=False)
    assert aeb_fetch.main(["--root", str(tmp_path)]) == 2


def test_clips_are_fetched_into_the_given_root(wired):
    session, ids, tmp_path = wired
    assert aeb_fetch.main(["--root", str(tmp_path)]) == 0

    stored = {ClipStore(root=tmp_path).peek_metadata(c.path).clip_id
              for c in ClipStore(root=tmp_path).list_clips()}
    assert stored == set(ids)
    assert sorted(session.fetched) == sorted(ids)


def test_a_second_run_downloads_nothing(wired):
    session, _ids, tmp_path = wired
    aeb_fetch.main(["--root", str(tmp_path)])
    session.fetched.clear()

    assert aeb_fetch.main(["--root", str(tmp_path)]) == 0
    assert session.fetched == []


def test_list_only_downloads_nothing(wired):
    session, _ids, tmp_path = wired
    assert aeb_fetch.main(["--root", str(tmp_path), "--list"]) == 0
    assert session.fetched == []
    assert ClipStore(root=tmp_path).list_clips() == []


def test_a_clobbered_clip_is_reported_not_counted(monkeypatch, tmp_path):
    """Store filenames carry 8 characters of the clip_id; a collision loses one."""
    monkeypatch.setenv("MONOCRUISE_PULL_TOKEN", "a-token")
    # Same captured_at and the same first 8 characters means the same filename.
    ids = ["deadbeef-1111-4111-8111-aaaaaaaaaaaa", "deadbeef-2222-4222-8222-bbbbbbbbbbbb"]
    rows = [{"clip_id": i, "received_at": "2026-08-09T10:00:00Z"} for i in ids]
    session = _FakeSession(rows, {i: _clip_blob(i) for i in ids})
    monkeypatch.setattr(aeb_fetch, "_session", lambda token: session)

    assert aeb_fetch.main(["--root", str(tmp_path)]) == 1
    assert len(ClipStore(root=tmp_path).list_clips()) == 1


def test_a_corrupt_download_is_not_stored(wired, monkeypatch):
    """A truncated body must not land in the store looking like a real clip."""
    session, ids, tmp_path = wired
    monkeypatch.setattr(aeb_fetch, "fetch_clip", lambda *a, **k: b"not a gzipped clip")

    assert aeb_fetch.main(["--root", str(tmp_path)]) == 1
    assert ClipStore(root=tmp_path).list_clips() == []
