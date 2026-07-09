"""Local, on-disk store for AEB capture clips (plan 3.4 / 11).

Clips are gzipped JSON written under ``%LOCALAPPDATA%\\MonoCruise\\aeb_clips``,
deliberately outside the install root so a client update (which overwrites the
install dir) never evicts them. Writes are atomic (temp file + ``os.replace``);
a rotation cap evicts the oldest clips once the store exceeds ``max_clips`` or
``max_bytes``.

``ClipStore`` is synchronous: every method touches the disk and may block, so
threads on the 30 Hz loops must not call it directly. ``AsyncClipWriter`` wraps
it on a background daemon fed by a queue, so a slow disk never stalls a
real-time loop. The writer is a plain daemon rather than a ``BaseThread``: it is
a best-effort I/O sink with no physical output and no real-time deadline, and it
blocks on the queue (which a watchdog-paced loop must not do). Losing a queued
clip on shutdown is acceptable and never a safety concern.

Capture stays local: nothing here uploads. Submission to the server is a
separate, explicit, consent-gated action built later (plan 11).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import queue
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.aeb.clip_schema import Clip, ClipMetadata, Label

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CLIPS: int = 300
_DEFAULT_MAX_BYTES: int = 500 * 1024 * 1024
_CLIP_SUFFIX: str = ".json.gz"
_TMP_SUFFIX: str = ".tmp"


def default_clip_root() -> Path:
    """Resolve the clip directory at runtime (never hardcoded / persisted).

    Uses ``%LOCALAPPDATA%`` on Windows; falls back to the user home on other
    platforms or when the variable is unset (dev / CI).
    """
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "MonoCruise" / "aeb_clips"
    return Path.home() / ".monocruise" / "aeb_clips"


def serialize_clip(clip: Clip) -> bytes:
    """Clip -> gzipped UTF-8 JSON bytes."""
    payload = json.dumps(clip.to_json_dict()).encode("utf-8")
    return gzip.compress(payload, compresslevel=6)


def deserialize_clip(blob: bytes) -> Clip:
    """Gzipped JSON bytes -> Clip."""
    return Clip.from_json_dict(json.loads(gzip.decompress(blob).decode("utf-8")))


@dataclass
class ClipInfo:
    """Lightweight directory listing entry (no decompression)."""

    path: Path
    name: str
    size_bytes: int
    mtime: float


def _safe_stamp(captured_at: str) -> str:
    """Compact, filename-safe form of an ISO timestamp, e.g. 20260617T120000Z."""
    return re.sub(r"[^0-9A-Za-z]", "", captured_at) or "clip"


class ClipStore:
    """Synchronous gzipped-JSON clip store with size/count rotation."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_clips: int = _DEFAULT_MAX_CLIPS,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self.root: Path = Path(root) if root is not None else default_clip_root()
        self.max_clips = max_clips
        self.max_bytes = max_bytes

    def _ensure_root(self) -> bool:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            logger.exception("could not create AEB clip directory")
            return False

    def _filename(self, clip: Clip) -> str:
        meta = clip.metadata
        stamp = _safe_stamp(meta.captured_at)
        cid = (meta.clip_id or "clip")[:8]
        return f"{stamp}_{meta.trigger_source}_{cid}{_CLIP_SUFFIX}"

    def write(self, clip: Clip) -> Path | None:
        """Serialize + atomically write one clip, then prune. Returns its path.

        Never raises: a disk failure is logged and swallowed so a caller on the
        capture path is never disturbed. Logs the basename only, never the full
        path (privacy: no machine-specific paths in logs).
        """
        if not self._ensure_root():
            return None

        name = self._filename(clip)
        final = self.root / name
        try:
            blob = serialize_clip(clip)
        except Exception:
            logger.exception("failed to serialize AEB clip %s", name)
            return None

        if not self._atomic_write(final, blob):
            return None

        logger.debug("saved AEB clip %s (%d bytes)", name, len(blob))
        self.prune()
        return final

    def _atomic_write(self, final: Path, blob: bytes) -> bool:
        """Write bytes to *final* via a temp file + os.replace. Never raises."""
        name = final.name
        fd, tmp_name = tempfile.mkstemp(prefix=name + ".", suffix=_TMP_SUFFIX, dir=str(self.root))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(blob)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(str(tmp_path), str(final))
            return True
        except Exception:
            logger.exception("failed to write AEB clip %s", name)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return False

    def peek_metadata(self, path: Path) -> ClipMetadata | None:
        """Decode only a clip's metadata (skips base64-decoding the streams).

        Cheap enough to scan the whole store for a list view / tagged badge.
        """
        try:
            blob = Path(path).read_bytes()
            top = json.loads(gzip.decompress(blob).decode("utf-8"))
            return ClipMetadata.from_json(top)
        except Exception:
            logger.exception("failed to read AEB clip metadata %s", Path(path).name)
            return None

    def write_label(self, path: Path, label: Label | None) -> bool:
        """Attach (or clear) a label on an existing clip, rewriting it in place.

        Keeps the same filename (label does not affect it) so tagging a clip
        never spawns a duplicate. Never raises.
        """
        clip = self.load(path)
        if clip is None:
            return False
        clip.metadata.label = label
        try:
            blob = serialize_clip(clip)
        except Exception:
            logger.exception("failed to serialize labelled clip %s", Path(path).name)
            return False
        ok = self._atomic_write(Path(path), blob)
        if ok:
            logger.debug("labelled AEB clip %s (class=%s)",
                         Path(path).name, label.class_ if label else None)
        return ok

    def list_clips(self) -> list[ClipInfo]:
        """List stored clips, newest first. Never raises."""
        out: list[ClipInfo] = []
        try:
            entries = list(self.root.glob(f"*{_CLIP_SUFFIX}"))
        except OSError:
            return out
        for p in entries:
            try:
                st = p.stat()
            except OSError:
                continue
            out.append(ClipInfo(path=p, name=p.name, size_bytes=st.st_size, mtime=st.st_mtime))
        out.sort(key=lambda c: c.mtime, reverse=True)
        return out

    def load(self, path: Path) -> Clip | None:
        """Read and decode one clip file. Never raises."""
        try:
            blob = Path(path).read_bytes()
            return deserialize_clip(blob)
        except Exception:
            logger.exception("failed to load AEB clip %s", Path(path).name)
            return None

    def delete(self, path: Path) -> bool:
        """Delete one clip we own (must live in the store dir with our suffix)."""
        p = Path(path)
        try:
            if p.parent != self.root or not p.name.endswith(_CLIP_SUFFIX):
                logger.warning("refusing to delete non-clip path %s", p.name)
                return False
            p.unlink()
            return True
        except OSError:
            logger.exception("failed to delete AEB clip %s", p.name)
            return False

    def prune(self) -> int:
        """Evict oldest clips until within the count and size caps. Returns count removed."""
        clips = self.list_clips()
        total = sum(c.size_bytes for c in clips)
        removed = 0
        # Oldest last (list is newest-first): pop from the tail.
        while clips and (len(clips) > self.max_clips or total > self.max_bytes):
            victim = clips.pop()
            if self.delete(victim.path):
                total -= victim.size_bytes
                removed += 1
        if removed:
            logger.debug("pruned %d old AEB clip(s)", removed)
        return removed


class AsyncClipWriter:
    """Background daemon that serializes + stores clips off a queue.

    ``submit`` is O(1) and non-blocking, safe to call from a 30 Hz loop. If the
    queue is full (disk backed up) the clip is dropped and logged rather than
    blocking the caller. ``notify`` receives the saved clip's basename on
    success and is the hook for the "AEB clip saved" popup / sound (plan 3.3).
    """

    def __init__(
        self,
        store: ClipStore | None = None,
        *,
        notify: Callable[[str], None] | None = None,
        queue_max: int = 32,
    ) -> None:
        self._store = store if store is not None else ClipStore()
        self._notify = notify
        self._queue: "queue.Queue[Clip]" = queue.Queue(maxsize=queue_max)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def store(self) -> ClipStore:
        return self._store

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="aeb_clip_writer", daemon=True,
        )
        self._thread.start()
        logger.debug("AEB clip writer started")

    def submit(self, clip: Clip) -> bool:
        """Enqueue a clip for background writing. Returns False if dropped."""
        try:
            self._queue.put_nowait(clip)
            return True
        except queue.Full:
            logger.warning("AEB clip writer queue full; dropping clip")
            return False

    def _write_one(self, clip: Clip) -> None:
        try:
            path = self._store.write(clip)
        except Exception:
            logger.exception("AEB clip write raised")
            return
        if path is not None and self._notify is not None:
            try:
                self._notify(path.name)
            except Exception:
                logger.exception("AEB clip notify callback raised")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                clip = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_one(clip)
            finally:
                self._queue.task_done()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal stop, drain any queued clips, and join the worker."""
        self._stop.set()
        # Best-effort drain so a clean shutdown does not silently drop pending clips.
        while True:
            try:
                clip = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._write_one(clip)
            finally:
                self._queue.task_done()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.debug("AEB clip writer stopped")
