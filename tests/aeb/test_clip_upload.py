"""Uploader: consent gating, per-clip eligibility, response handling, notifications."""
from __future__ import annotations

import base64
import io
import json
import logging

import pytest

from core.aeb import upload as upload_mod
from core.aeb.clip_schema import Clip, ClipMetadata
from core.aeb.clip_store import ClipStore
from core.aeb.upload import ClipUploader, SubmissionLog, clip_ineligible_reason


def _thumbnail(px: int) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (px, max(1, px // 2))).save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clip(**meta) -> Clip:
    base = dict(clip_id="c" * 8, client_version="1.1.0", trigger_source="auto_engagement",
                captured_at="2026-08-10T12:00:00Z")
    base.update(meta)
    return Clip(metadata=ClipMetadata(**base), radar_frames=[], aeb_ticks=[])


class _Transport:
    """Records requests and replays scripted responses."""

    def __init__(self, *responses):
        self.responses = list(responses) or [(200, {"accepted": True, "clip_id": "x"})]
        self.calls: list[tuple[bytes, dict]] = []

    def __call__(self, url, data, headers):
        self.calls.append((data, headers))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class _Boom(_Transport):
    def __call__(self, url, data, headers):
        self.calls.append((data, headers))
        raise OSError("no route to host")


@pytest.fixture()
def opted_in(monkeypatch):
    monkeypatch.setattr(upload_mod, "contribution_enabled", lambda: True)
    monkeypatch.setattr(upload_mod, "cached_policy", lambda: object())
    monkeypatch.setattr(upload_mod, "upload_blocked_reason", lambda *a, **k: None)


def _uploader(tmp_path, transport, **kwargs):
    store = ClipStore(root=tmp_path / "clips")
    kwargs.setdefault("log", SubmissionLog(tmp_path / "log.jsonl"))
    return store, ClipUploader(store, transport=transport, **kwargs)


def _write(store, clip: Clip):
    path = store.write(clip)
    assert path is not None
    return path


# -- eligibility ----------------------------------------------------------

@pytest.mark.parametrize("source", ["shadow_near", "random"])
def test_background_negatives_are_never_contributed(source):
    assert clip_ineligible_reason(ClipMetadata(trigger_source=source)) == "background negative"


def test_an_old_whole_monitor_thumbnail_is_refused():
    """480 px clips predate the game-window crop and may show legible text."""
    meta = ClipMetadata(trigger_source="auto_engagement", thumbnail_jpeg=_thumbnail(480))
    assert clip_ineligible_reason(meta) == "oversized thumbnail"


def test_a_cropped_thumbnail_is_accepted():
    meta = ClipMetadata(trigger_source="auto_engagement", thumbnail_jpeg=_thumbnail(240))
    assert clip_ineligible_reason(meta) is None


def test_a_clip_without_a_thumbnail_is_accepted():
    assert clip_ineligible_reason(ClipMetadata(trigger_source="auto_crash")) is None


def test_an_unreadable_thumbnail_is_refused():
    """Fail closed: an image we cannot measure is one we cannot vouch for."""
    meta = ClipMetadata(trigger_source="manual", thumbnail_jpeg="not base64 at all")
    assert clip_ineligible_reason(meta) == "unreadable thumbnail"


def test_a_clip_carrying_an_identifier_is_refused():
    meta = ClipMetadata(trigger_source="manual", client_id="someone")
    assert clip_ineligible_reason(meta) == "carries an identifier"


def test_unreadable_metadata_is_refused():
    assert clip_ineligible_reason(None) == "unreadable metadata"


# -- consent gating -------------------------------------------------------

def test_nothing_is_sent_without_consent(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_mod, "contribution_enabled", lambda: False)
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip()))
    assert transport.calls == []


def test_nothing_is_sent_when_the_policy_blocks(tmp_path, monkeypatch, opted_in):
    monkeypatch.setattr(upload_mod, "upload_blocked_reason", lambda *a, **k: "intake closed")
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip()))
    assert transport.calls == []


def test_a_tn_clip_is_not_sent_even_when_opted_in(tmp_path, opted_in):
    """The capture gate cannot cover this: debug users keep capturing them."""
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip(trigger_source="shadow_near")))
    assert transport.calls == []


def test_an_old_thumbnail_is_not_sent_even_when_opted_in(tmp_path, opted_in):
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip(thumbnail_jpeg=_thumbnail(480))))
    assert transport.calls == []


def test_an_eligible_clip_is_sent(tmp_path, opted_in):
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    path = _write(store, _clip(thumbnail_jpeg=_thumbnail(240)))
    expected = path.read_bytes()
    up._handle(path)
    assert len(transport.calls) == 1
    body, headers = transport.calls[0]
    assert body == expected                      # sent byte for byte, no re-encoding
    assert headers["Content-Encoding"] == "gzip"
    assert headers["X-MonoCruise-Version"] == "1.1.0"
    assert headers["User-Agent"] == "MonoCruise/1.1.0"


def test_the_payload_carries_no_identifier(tmp_path, opted_in):
    import gzip

    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip()))
    sent = json.loads(gzip.decompress(transport.calls[0][0]).decode("utf-8"))
    assert sent["client_id"] is None
    assert not any("id" in k.lower() for k in transport.calls[0][1])


# -- response handling ----------------------------------------------------

def test_an_accepted_clip_is_deleted(tmp_path, opted_in):
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})))
    path = _write(store, _clip())
    up._handle(path)
    assert not path.exists()


def test_a_duplicate_is_deleted_too(tmp_path, opted_in):
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": False, "reason": "duplicate"})))
    path = _write(store, _clip())
    up._handle(path)
    assert not path.exists()


@pytest.mark.parametrize("reason", ["quota", "closed", "unwanted", "too_large", "bad_schema"])
def test_a_refused_clip_stays_on_disk(tmp_path, opted_in, reason):
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": False, "reason": reason})))
    path = _write(store, _clip())
    up._handle(path)
    assert path.exists()


def test_a_debug_user_never_loses_a_clip(tmp_path, opted_in):
    """That store is the working corpus, not a staging area."""
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})), delete_after=False)
    path = _write(store, _clip())
    up._handle(path)
    assert path.exists()


@pytest.mark.parametrize("reason", ["quota", "closed"])
def test_a_quota_refusal_pauses_further_uploads(tmp_path, opted_in, reason):
    transport = _Transport((200, {"accepted": False, "reason": reason, "retry_after_s": 3600}))
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip(clip_id="first")))
    up._handle(_write(store, _clip(clip_id="second", captured_at="2026-08-10T12:00:01Z")))
    assert len(transport.calls) == 1


def test_a_network_error_retries_then_gives_up(tmp_path, opted_in, monkeypatch):
    monkeypatch.setattr(upload_mod, "_BACKOFF_BASE_S", 0.0)
    transport = _Boom()
    store, up = _uploader(tmp_path, transport)
    path = _write(store, _clip())
    up._handle(path)
    assert len(transport.calls) == upload_mod._MAX_ATTEMPTS
    assert path.exists()


def test_a_5xx_retries_then_gives_up(tmp_path, opted_in, monkeypatch):
    monkeypatch.setattr(upload_mod, "_BACKOFF_BASE_S", 0.0)
    transport = _Transport((503, {}))
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip()))
    assert len(transport.calls) == upload_mod._MAX_ATTEMPTS


def test_a_4xx_is_not_retried(tmp_path, opted_in):
    transport = _Transport((400, {"accepted": False, "reason": "bad_schema"}))
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip()))
    assert len(transport.calls) == 1


# -- queue ----------------------------------------------------------------

def test_the_queue_drops_the_oldest_rather_than_blocking(tmp_path):
    _store, up = _uploader(tmp_path, _Transport(), queue_max=2)
    assert up.submit(tmp_path / "a.json.gz") is True
    assert up.submit(tmp_path / "b.json.gz") is True
    assert up.submit(tmp_path / "c.json.gz") is True      # would block if unbounded
    assert up._queue.qsize() == 2


# -- submission log -------------------------------------------------------

def test_the_log_records_one_line_per_attempt(tmp_path, opted_in):
    log_path = tmp_path / "log.jsonl"
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})),
                          log=SubmissionLog(log_path))
    up._handle(_write(store, _clip(thumbnail_jpeg=_thumbnail(240))))
    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["result"] == "accepted"
    assert entry["trigger"] == "auto_engagement"
    assert entry["had_thumbnail"] is True
    assert entry["bytes"] > 0


def test_the_log_holds_no_coordinates_or_image_data(tmp_path, opted_in):
    log_path = tmp_path / "log.jsonl"
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})),
                          log=SubmissionLog(log_path))
    up._handle(_write(store, _clip(thumbnail_jpeg=_thumbnail(240))))
    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert set(entry) == {"t", "clip_id", "trigger", "bytes", "result", "had_thumbnail"}


def test_the_log_is_capped(tmp_path):
    log = SubmissionLog(tmp_path / "log.jsonl", max_lines=5)
    for i in range(12):
        log.append({"t": "now", "clip_id": str(i)})
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert json.loads(lines[-1])["clip_id"] == "11"


# -- notification ---------------------------------------------------------

def _popups(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if getattr(r, "popup", False)]


def test_the_first_send_notifies_immediately(tmp_path, opted_in, caplog):
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})))
    with caplog.at_level(logging.INFO, logger="core.aeb.upload"):
        up._handle(_write(store, _clip()))
    assert _popups(caplog) == ["AEB clip sent"]


def test_later_sends_coalesce_into_one_summary(tmp_path, opted_in, caplog):
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})),
                          notify_cooldown_s=600.0)
    with caplog.at_level(logging.INFO, logger="core.aeb.upload"):
        for i in range(4):
            up._handle(_write(store, _clip(clip_id=f"c{i}",
                                           captured_at=f"2026-08-10T12:00:0{i}Z")))
        assert _popups(caplog) == ["AEB clip sent"]      # three held
        up.notify_cooldown_s = 0.0
        up._flush_notice()
    assert _popups(caplog) == ["AEB clip sent", "3 AEB clips sent"]


def test_a_refusal_never_notifies(tmp_path, opted_in, caplog):
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": False, "reason": "quota"})))
    with caplog.at_level(logging.INFO, logger="core.aeb.upload"):
        up._handle(_write(store, _clip()))
    assert _popups(caplog) == []


def test_notifications_are_held_during_an_intervention(tmp_path, opted_in, caplog):
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})))
    up.set_intervening(True)
    with caplog.at_level(logging.INFO, logger="core.aeb.upload"):
        up._handle(_write(store, _clip()))
        assert _popups(caplog) == []
        up.set_intervening(False)
    assert _popups(caplog) == ["AEB clip sent"]


def test_notifications_can_be_silenced(tmp_path, opted_in, caplog):
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})), notify=False)
    with caplog.at_level(logging.INFO, logger="core.aeb.upload"):
        up._handle(_write(store, _clip()))
    assert _popups(caplog) == []


def test_a_kept_clip_reports_back_once(tmp_path, opted_in):
    """Exactly one of sent/kept fires, so a debug contributor gets one popup."""
    kept: list = []
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": False, "reason": "unwanted"})),
                          on_kept=kept.append)
    path = _write(store, _clip())
    up._handle(path)
    assert kept == [path]


def test_a_sent_clip_does_not_also_report_kept(tmp_path, opted_in):
    kept: list = []
    store, up = _uploader(tmp_path, _Transport((200, {"accepted": True})), on_kept=kept.append)
    up._handle(_write(store, _clip()))
    assert kept == []
