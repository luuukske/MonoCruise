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
    # Unpaced by default so the rest of the suite is not waiting on the clock.
    kwargs.setdefault("min_send_gap_s", 0.0)
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


def test_a_clip_held_back_by_a_pause_is_recoverable(tmp_path, opted_in):
    """Otherwise a day at the cap loses every clip captured after it was hit."""
    log_path = tmp_path / "log.jsonl"
    transport = _Transport((200, {"accepted": False, "reason": "quota", "retry_after_s": 3600}))
    store, up = _uploader(tmp_path, transport, log=SubmissionLog(log_path))
    up._handle(_write(store, _clip(clip_id="triggers-the-pause")))
    up._handle(_write(store, _clip(clip_id="held-back-by-it",
                                   captured_at="2026-08-10T12:00:01Z")))
    assert up._log.retryable_clip_ids() == {"triggers-the-pause", "held-back-by-it"}


def test_a_pause_does_not_log_a_clip_that_was_never_eligible(tmp_path, opted_in):
    """A background negative must not enter the log and become retryable."""
    transport = _Transport((200, {"accepted": False, "reason": "quota", "retry_after_s": 3600}))
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip(clip_id="triggers-the-pause")))
    up._handle(_write(store, _clip(clip_id="a-tn-clip-9999", trigger_source="shadow_near",
                                   captured_at="2026-08-10T12:00:01Z")))
    assert up._log.retryable_clip_ids() == {"triggers-the-pause"}


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


def test_an_edge_throttle_pauses_even_without_a_reason(tmp_path, opted_in):
    """Cloudflare answers a rate limit with an HTML page, not the endpoint's JSON.
    Without the status-only branch every later clip would hammer a shut door."""
    transport = _Transport((429, {"retry_after_s": "60"}))
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip(clip_id="first")))
    up._handle(_write(store, _clip(clip_id="second", captured_at="2026-08-10T12:00:01Z")))
    assert len(transport.calls) == 1


def test_an_edge_throttle_without_a_retry_header_still_pauses(tmp_path, opted_in):
    transport = _Transport((429, {}))
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip(clip_id="first")))
    up._handle(_write(store, _clip(clip_id="second", captured_at="2026-08-10T12:00:01Z")))
    assert len(transport.calls) == 1


def test_an_http_date_retry_header_does_not_crash_the_pause(tmp_path, opted_in):
    """Retry-After may be a date; an unparseable one must fall back, not raise."""
    store, up = _uploader(tmp_path, _Transport(
        (429, {"retry_after_s": "Wed, 21 Oct 2026 07:28:00 GMT"})))
    up._handle(_write(store, _clip()))
    assert up._paused_until > 0.0


def test_a_4xx_is_not_retried(tmp_path, opted_in):
    transport = _Transport((400, {"accepted": False, "reason": "bad_schema"}))
    store, up = _uploader(tmp_path, transport)
    up._handle(_write(store, _clip()))
    assert len(transport.calls) == 1


# -- holdover retry -------------------------------------------------------

def _log_lines(path, *entries) -> SubmissionLog:
    log = SubmissionLog(path)
    for e in entries:
        log.append(e)
    return log


def test_only_transient_failures_come_back(tmp_path):
    log = _log_lines(
        tmp_path / "log.jsonl",
        {"clip_id": "a", "result": "network_error"},
        {"clip_id": "b", "result": "server_error"},
        {"clip_id": "c", "result": "quota"},
        {"clip_id": "d", "result": "closed"},
        {"clip_id": "e", "result": "http_429"},
        {"clip_id": "j", "result": "paused"},
        {"clip_id": "f", "result": "accepted"},
        {"clip_id": "g", "result": "duplicate"},
        {"clip_id": "h", "result": "unwanted"},
        {"clip_id": "i", "result": "bad_schema"},
    )
    assert log.retryable_clip_ids() == {"a", "b", "c", "d", "e", "j"}


def test_a_later_success_retires_an_earlier_failure(tmp_path):
    log = _log_lines(
        tmp_path / "log.jsonl",
        {"clip_id": "a", "result": "network_error"},
        {"clip_id": "a", "result": "accepted"},
    )
    assert log.retryable_clip_ids() == set()


def test_a_clip_that_failed_again_stays_eligible(tmp_path):
    log = _log_lines(
        tmp_path / "log.jsonl",
        {"clip_id": "a", "result": "accepted"},
        {"clip_id": "a", "result": "network_error"},
    )
    assert log.retryable_clip_ids() == {"a"}


def test_a_held_over_clip_is_re_offered(tmp_path, opted_in):
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    path = _write(store, _clip(clip_id="held-over-1234"))
    up._log.append({"clip_id": "held-over-1234", "result": "network_error"})
    assert up._retry_pending() == 1
    assert len(transport.calls) == 1
    assert not path.exists()


def test_a_clip_that_was_never_offered_is_never_swept_in(tmp_path, opted_in):
    """The whole safety property: no log entry means unreachable from a retry."""
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    _write(store, _clip(clip_id="never-offered-1"))
    _write(store, _clip(clip_id="never-offered-2", captured_at="2026-08-10T12:00:01Z"))
    assert up._retry_pending() == 0
    assert transport.calls == []


def test_a_refused_clip_is_not_resurrected_by_a_retry(tmp_path, opted_in):
    """A TN clip logged under an old build must not come back through this door."""
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    path = _write(store, _clip(clip_id="tn-clip-9999", trigger_source="shadow_near"))
    up._log.append({"clip_id": "tn-clip-9999", "result": "network_error"})
    up._retry_pending()
    assert transport.calls == []
    assert path.exists()


def test_the_retry_batch_is_bounded(tmp_path, opted_in):
    transport = _Transport()
    store, up = _uploader(tmp_path, transport)
    for i in range(6):
        cid = f"holdover{i:04d}"
        _write(store, _clip(clip_id=cid, captured_at=f"2026-08-10T12:00:0{i}Z"))
        up._log.append({"clip_id": cid, "result": "network_error"})
    assert up._retry_pending(limit=2) == 2
    assert len(transport.calls) == 2


def test_a_missing_file_is_skipped_rather_than_retried(tmp_path, opted_in):
    transport = _Transport()
    _store, up = _uploader(tmp_path, transport)
    up._log.append({"clip_id": "gone-forever-01", "result": "network_error"})
    assert up._retry_pending() == 0


# -- pacing ---------------------------------------------------------------

def _record_waits(up) -> list[float]:
    """Capture what the uploader asks to wait for, without waiting for it."""
    waits: list[float] = []

    def _wait(timeout=None):
        waits.append(timeout)
        return False

    up._stop.wait = _wait
    return waits


def test_the_first_send_is_not_delayed(tmp_path, opted_in):
    store, up = _uploader(tmp_path, _Transport(), min_send_gap_s=2.0)
    waits = _record_waits(up)
    up._handle(_write(store, _clip()))
    assert waits == []


def test_a_queue_drain_is_paced_rather_than_bursted(tmp_path, opted_in):
    """Cloudflare counts requests, not intentions: 16 queued clips must not
    arrive as a flood and trip a rate limit on the contributor's own traffic."""
    store, up = _uploader(tmp_path, _Transport(), min_send_gap_s=2.0)
    waits = _record_waits(up)
    for i in range(3):
        up._handle(_write(store, _clip(clip_id=f"c{i}",
                                       captured_at=f"2026-08-10T12:00:0{i}Z")))
    assert len(waits) == 2                      # every send after the first
    assert all(0 < w <= 2.0 for w in waits), waits


def test_pacing_never_delays_a_clip_that_is_not_sent(tmp_path, opted_in):
    """A refused clip costs no request, so it must not cost a gap either."""
    store, up = _uploader(tmp_path, _Transport(), min_send_gap_s=2.0)
    waits = _record_waits(up)
    up._handle(_write(store, _clip(trigger_source="shadow_near")))
    up._handle(_write(store, _clip(clip_id="real", captured_at="2026-08-10T12:00:01Z")))
    assert waits == []


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
