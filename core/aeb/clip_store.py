"""Gzipped JSON AEB clips under LOCALAPPDATA; ClipStore sync, AsyncClipWriter for loops."""

from __future__ import annotations

import gzip
import json
import logging
import os
import queue
import re
import tempfile
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.aeb.clip_schema import Clip, ClipMetadata, Label

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES: int = 500 * 1024 * 1024
_CLIP_SUFFIX: str = ".json.gz"
_TMP_SUFFIX: str = ".tmp"


def default_clip_root() -> Path:
    """LOCALAPPDATA/MonoCruise/aeb_clips on Windows; ~/.monocruise/aeb_clips elsewhere."""
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


# Metadata prefix ends at first serialized stream key (see Clip.to_json_dict).
_STREAM_MARKER: bytes = b'"radar_frames"'
_PEEK_CHUNK: int = 65536


def deserialize_metadata(blob: bytes) -> ClipMetadata:
    """Gunzip only through metadata prefix (stops at ``radar_frames`` key)."""
    dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)  # 16 => gzip framing
    out = bytearray()
    scanned = 0
    for i in range(0, len(blob), _PEEK_CHUNK):
        out += dobj.decompress(blob[i:i + _PEEK_CHUNK])
        idx = out.find(_STREAM_MARKER, max(0, scanned - len(_STREAM_MARKER)))
        if idx != -1:
            head = bytes(out[:idx]).rstrip()
            if head.endswith(b","):
                head = head[:-1]
            return ClipMetadata.from_json(json.loads(head + b"}"))
        scanned = len(out)
    # No stream key found: the whole payload is metadata (e.g. an empty clip).
    return ClipMetadata.from_json(json.loads(bytes(out)))


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
    """Synchronous gzipped-JSON clip store with size rotation (count cap optional)."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_clips: int | None = None,
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
        """Serialize, atomic write, prune. Never raises; logs basename only."""
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
        """Metadata-only decode for list views; partial gunzip with full fallback."""
        try:
            blob = Path(path).read_bytes()
        except OSError:
            logger.exception("failed to read AEB clip %s", Path(path).name)
            return None
        try:
            return deserialize_metadata(blob)
        except Exception:
            pass
        try:
            top = json.loads(gzip.decompress(blob).decode("utf-8"))
            return ClipMetadata.from_json(top)
        except Exception:
            logger.exception("failed to read AEB clip metadata %s", Path(path).name)
            return None

    def write_label(self, path: Path, label: Label | None) -> bool:
        """Rewrite clip in place with label; same filename. Never raises."""
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
        """Evict oldest clips until within the size cap (and count cap, if set). Returns count removed."""
        clips = self.list_clips()
        total = sum(c.size_bytes for c in clips)
        removed = 0
        # Oldest last (list is newest-first): pop from the tail.
        while clips and (
            (self.max_clips is not None and len(clips) > self.max_clips)
            or total > self.max_bytes
        ):
            victim = clips.pop()
            if self.delete(victim.path):
                total -= victim.size_bytes
                removed += 1
        if removed:
            logger.debug("pruned %d old AEB clip(s)", removed)
        return removed


class AsyncClipWriter:
    """Queue + daemon writer; submit is non-blocking, drops when full."""

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
