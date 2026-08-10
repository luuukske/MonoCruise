"""Opt-in AEB clip upload to ld-tech.org.

The only module in ``core.aeb`` allowed to make an outbound request. Consent is
checked twice: once where a clip enters the queue (``core/aeb/capture.py``) and
once here before the socket is opened. See ``core/aeb/README.md`` section 13 and
``docs/aeb_clip_contribution_plan.md`` section 5.6.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

from core.aeb.clip_schema import SCHEMA_VERSION, ClipMetadata, utc_now_iso
from core.aeb.clip_store import ClipStore
from core.aeb.intake_policy import cached_policy, contribution_enabled, upload_blocked_reason
from core.aeb.recorder import _TN_SOURCES
from core.aeb.screenshot import _MAX_PX as _MAX_THUMBNAIL_PX

logger = logging.getLogger(__name__)

ENDPOINT = "https://ld-tech.org/api/v1/aeb_reports.php"

# Connect, read. A clip is a few hundred KB on a domestic uplink.
_REQUEST_TIMEOUT: tuple[float, float] = (10.0, 30.0)
_QUEUE_MAX: int = 16
_MAX_ATTEMPTS: int = 4
_BACKOFF_BASE_S: float = 2.0
# Cap on the total sleep one clip may cost, so a dead server cannot wedge the
# queue behind a single item.
_MAX_BACKOFF_S: float = 30.0
# Smallest gap between two POSTs. An edge rate limit counts requests, not
# intentions, so a queue drain has to look like traffic rather than a flood.
_MIN_SEND_GAP_S: float = 2.0
_NOTIFY_COOLDOWN_S: float = 600.0
_LOG_MAX_LINES: int = 5000
_LOG_NAME: str = "aeb_submissions.jsonl"

# Server verdicts. Anything unrecognised is treated as "do not retry", because a
# reason this client does not understand is not one it can act on.
_KEEP_AND_STOP: frozenset[str] = frozenset({"quota", "closed"})
_DELETE_REASONS: frozenset[str] = frozenset({"duplicate"})


def _post(url: str, data: bytes, headers: dict) -> tuple[int, dict]:
    """POST and normalise the response. Seam for tests; nothing else may call it."""
    import requests

    resp = requests.post(url, data=data, headers=headers, timeout=_REQUEST_TIMEOUT)
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    # An edge throttle answers with an HTML page and a Retry-After header, not
    # the endpoint's JSON, so carry the header through as the same field.
    if "retry_after_s" not in body and resp.headers.get("Retry-After"):
        body = {**body, "retry_after_s": resp.headers["Retry-After"]}
    return resp.status_code, body


def _thumbnail_long_side(b64: str | None) -> int | None:
    """Long side in pixels: 0 when there is no thumbnail, None when unreadable."""
    if not b64:
        return 0
    try:
        from PIL import Image

        raw = base64.b64decode(b64, validate=True)
        with Image.open(io.BytesIO(raw)) as img:
            return max(img.size)
    except Exception:
        logger.debug("could not measure a clip thumbnail", exc_info=True)
        return None


def clip_ineligible_reason(meta: ClipMetadata | None) -> str | None:
    """Why this clip may not be contributed, or None when it may.

    Judged per clip and from the clip itself, never from the store it sits in or
    the build that is running now: a store outlives many releases, and a user who
    is both debug and contributing keeps capturing the classes a contributor
    skips.
    """
    if meta is None:
        return "unreadable metadata"
    if meta.trigger_source in _TN_SOURCES:
        return "background negative"
    if meta.client_id is not None:
        return "carries an identifier"
    long_side = _thumbnail_long_side(meta.thumbnail_jpeg)
    if long_side is None:
        return "unreadable thumbnail"
    if long_side > _MAX_THUMBNAIL_PX:
        # Captured before the game-window crop landed, so it may be a grab of
        # the whole monitor with legible text.
        return "oversized thumbnail"
    return None


class SubmissionLog:
    """Append-only record of what left the machine. Never raises."""

    def __init__(self, path: Path, *, max_lines: int = _LOG_MAX_LINES) -> None:
        self.path = Path(path)
        self.max_lines = max_lines
        self._lock = threading.Lock()
        self._lines: int | None = None

    def _count_lines(self) -> int:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0

    def _trim_locked(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                kept = fh.readlines()[-self.max_lines:]
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.writelines(kept)
            self._lines = len(kept)
        except OSError:
            logger.debug("could not trim the AEB submission log", exc_info=True)

    def append(self, entry: dict) -> None:
        """One line per attempt. Never contains coordinates or image data."""
        with self._lock:
            if self._lines is None:
                self._lines = self._count_lines()
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
                self._lines += 1
            except OSError:
                logger.debug("could not write the AEB submission log", exc_info=True)
                return
            if self._lines > self.max_lines:
                self._trim_locked()


class ClipUploader:
    """Queue + daemon uploader. Mirrors AsyncClipWriter: off every control loop."""

    def __init__(
        self,
        store: ClipStore,
        *,
        transport: Callable[[str, bytes, dict], tuple[int, dict]] = _post,
        delete_after: bool = True,
        notify: bool = True,
        on_kept: Callable[[Path], None] | None = None,
        log: SubmissionLog | None = None,
        queue_max: int = _QUEUE_MAX,
        notify_cooldown_s: float = _NOTIFY_COOLDOWN_S,
        min_send_gap_s: float = _MIN_SEND_GAP_S,
    ) -> None:
        self._store = store
        self._transport = transport
        self.delete_after = delete_after
        self.notify_enabled = notify
        # Called when a clip stays on this machine, so the caller can announce
        # the save instead of the send. Exactly one of the two fires per clip.
        self._on_kept = on_kept
        self._log = log if log is not None else SubmissionLog(
            store.root.parent / _LOG_NAME
        )
        self._queue: "queue.Queue[Path]" = queue.Queue(maxsize=queue_max)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.min_send_gap_s = min_send_gap_s
        self._paused_until: float = 0.0
        # -inf so the first send of a session is never delayed.
        self._last_send_mono: float = float("-inf")
        self._intervening: bool = False
        self._pending_notice: int = 0
        # -inf so the first notice is never held back by the cooldown, however
        # long the process has been up.
        self._last_notice_mono: float = float("-inf")
        self.notify_cooldown_s = notify_cooldown_s

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="aeb_clip_upload", daemon=True)
        self._thread.start()
        logger.debug("AEB clip uploader started")

    def stop(self, timeout: float = 3.0) -> None:
        """Signal stop, flush any held notification, and join the worker."""
        self._stop.set()
        # Force past the cooldown: a held summary must not die with the process.
        self._flush_notice(force=True)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.debug("AEB clip uploader stopped")

    def submit(self, path: Path) -> bool:
        """Enqueue a written clip. Returns False when it was not queued."""
        try:
            self._queue.put_nowait(Path(path))
            return True
        except queue.Full:
            # Drop the oldest: a fresh clip is worth more than a stale one, and
            # blocking here would sit on the writer thread.
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                logger.debug("AEB upload queue full; dropped %s", dropped.name)
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(Path(path))
                return True
            except queue.Full:
                return False

    def set_intervening(self, active: bool) -> None:
        """Hold notifications while AEB is acting. Called from the AEB loop."""
        was = self._intervening
        self._intervening = bool(active)
        if was and not self._intervening:
            self._flush_notice()

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._flush_notice()
                continue
            try:
                self._handle(path)
            except Exception:
                logger.exception("AEB clip upload raised")
            finally:
                self._queue.task_done()

    def _handle(self, path: Path) -> None:
        # Consent is re-read per clip rather than captured at construction, so
        # unticking the box stops uploads immediately rather than next boot.
        if not contribution_enabled():
            return self._kept(path)
        if time.monotonic() < self._paused_until:
            logger.debug("AEB upload paused; keeping %s locally", path.name)
            return self._kept(path)

        meta = self._store.peek_metadata(path)
        reason = clip_ineligible_reason(meta)
        if reason is not None:
            logger.debug("not contributing %s: %s", path.name, reason)
            return self._kept(path)

        try:
            blob = path.read_bytes()
        except OSError:
            logger.debug("could not read %s for upload", path.name, exc_info=True)
            return self._kept(path)

        blocked = upload_blocked_reason(
            cached_policy(),
            clip_bytes=len(blob),
            client_version=meta.client_version,
            schema_version=meta.schema_version,
        )
        if blocked is not None:
            logger.debug("not uploading %s: %s", path.name, blocked)
            return self._kept(path)

        self._send(path, blob, meta)

    def _kept(self, path: Path) -> None:
        """The clip stays here. Never raises into the worker loop."""
        if self._on_kept is None:
            return
        try:
            self._on_kept(path)
        except Exception:
            logger.debug("AEB upload kept-callback raised", exc_info=True)

    def _headers(self, meta: ClipMetadata) -> dict:
        version = meta.client_version or ""
        return {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "X-MonoCruise-Version": version,
            "X-MonoCruise-Schema": str(meta.schema_version or SCHEMA_VERSION),
            "User-Agent": f"MonoCruise/{version}" if version else "MonoCruise",
        }

    def _pace(self) -> None:
        """Space POSTs out so a queue drain never looks like a flood.

        A clip is captured every minute or so, but the queue holds up to
        `_QUEUE_MAX`, so a machine that was offline would otherwise empty it back
        to back and trip an edge rate limit on its own traffic. Waits on the stop
        event rather than sleeping, so shutdown stays prompt.
        """
        gap = self.min_send_gap_s - (time.monotonic() - self._last_send_mono)
        if gap > 0:
            self._stop.wait(gap)
        self._last_send_mono = time.monotonic()

    def _send(self, path: Path, blob: bytes, meta: ClipMetadata) -> None:
        headers = self._headers(meta)
        self._pace()
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            if self._stop.is_set():
                return
            try:
                status, body = self._transport(ENDPOINT, blob, headers)
            except Exception:
                logger.debug("AEB upload attempt %d failed", attempt, exc_info=True)
                if self._backoff(attempt):
                    continue
                self._record(path, meta, len(blob), "network_error")
                return self._kept(path)

            if 500 <= status < 600:
                logger.debug("AEB upload got HTTP %d on attempt %d", status, attempt)
                if self._backoff(attempt):
                    continue
                self._record(path, meta, len(blob), "server_error")
                return self._kept(path)

            self._apply(path, meta, len(blob), status, body)
            return

    def _backoff(self, attempt: int) -> bool:
        """Sleep before the next attempt. False when attempts are exhausted."""
        if attempt >= _MAX_ATTEMPTS:
            return False
        delay = min(_BACKOFF_BASE_S * (2 ** (attempt - 1)), _MAX_BACKOFF_S)
        # Waiting on the stop event rather than sleeping keeps shutdown prompt.
        return not self._stop.wait(delay)

    def _apply(self, path: Path, meta: ClipMetadata, size: int, status: int, body: dict) -> None:
        accepted = bool(body.get("accepted"))
        reason = str(body.get("reason") or "")

        if accepted:
            self._record(path, meta, size, "accepted")
            self._delete(path)
            self._note_sent()
            return

        if reason in _DELETE_REASONS:
            # The server already holds it, so a local copy buys nothing. No
            # notification either: the first offer of this clip already gave one.
            self._record(path, meta, size, reason)
            self._delete(path)
            return

        # 429 without a reason is an edge throttle rather than this endpoint, so
        # the status has to be enough on its own. Treating it as an ordinary
        # refusal would keep every later clip hammering a door already shut.
        if reason in _KEEP_AND_STOP or status == 429:
            self._pause(body)

        self._record(path, meta, size, reason or f"http_{status}")
        self._kept(path)

    def _pause(self, body: dict) -> None:
        try:
            wait = float(body.get("retry_after_s") or 0.0)
        except (TypeError, ValueError):
            wait = 0.0
        wait = max(60.0, min(wait, 24 * 3600.0))
        self._paused_until = time.monotonic() + wait
        logger.debug("AEB intake refused; pausing uploads for %.0f s", wait)

    def _delete(self, path: Path) -> None:
        if not self.delete_after:
            return
        self._store.delete(path)

    def _record(self, path: Path, meta: ClipMetadata, size: int, result: str) -> None:
        self._log.append({
            "t": utc_now_iso(),
            "clip_id": meta.clip_id,
            "trigger": meta.trigger_source,
            "bytes": size,
            "result": result,
            "had_thumbnail": bool(meta.thumbnail_jpeg),
        })
        logger.debug("AEB clip %s upload result: %s", path.name, result)

    # -- notification ------------------------------------------------------

    def _note_sent(self) -> None:
        """One notice for the first send, then a coalesced summary (plan 5.9)."""
        if not self.notify_enabled:
            return
        self._pending_notice += 1
        # The first send shows immediately, because a silent background uploader
        # is what the notification exists to disprove. Then the cooldown coalesces.
        self._flush_notice()

    def _flush_notice(self, *, force: bool = False) -> None:
        if not self.notify_enabled or self._pending_notice <= 0:
            return
        if self._intervening:
            return
        if not force and time.monotonic() - self._last_notice_mono < self.notify_cooldown_s:
            return
        count = self._pending_notice
        self._pending_notice = 0
        self._last_notice_mono = time.monotonic()
        if count == 1:
            logger.info("AEB clip sent", extra={"popup": True})
        else:
            logger.info("%d AEB clips sent", count, extra={"popup": True})
