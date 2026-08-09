"""Shared AEB clip recorder, writer and uploader; sole Settings touchpoint for capture."""

from __future__ import annotations

import atexit
import logging
import threading
from pathlib import Path

from core.aeb.clip_store import AsyncClipWriter, ClipStore
from core.aeb.recorder import AEBClipRecorder
from core.aeb.upload import ClipUploader

logger = logging.getLogger(__name__)

# Contribute-only stores stage clips for upload rather than holding a corpus,
# so they get a fifth of the debug cap.
_CONTRIBUTOR_MAX_BYTES: int = 100 * 1024 * 1024

_lock = threading.Lock()
_recorder: AEBClipRecorder | None = None
_writer: AsyncClipWriter | None = None
_uploader: ClipUploader | None = None
_debug: bool = False
_initialized: bool = False


def _notify_kept(path: Path) -> None:
    """Announce a clip that stays on this machine. Debug users only."""
    if not _debug:
        return
    # One user-facing confirmation per capture (plan 3.3); details to the log file.
    logger.info("AEB clip saved", extra={"popup": True})
    logger.debug("AEB clip kept: %s", path.name)


def _on_clip_written(path: Path) -> None:
    """Writer callback: hand the clip to the uploader, or announce the save.

    This is the boundary where a clip enters the upload path, so consent is
    checked here and not only at the socket. Since the capture gate widened to
    ``debug or contribution_enabled()``, this callback also fires for debug
    testers who never opted in, and their clips must never reach the queue.
    """
    logger.debug("AEB clip saved: %s", path.name)
    if _uploader is not None and _contribution_enabled():
        _uploader.submit(path)
        return
    _notify_kept(path)


def _contribution_enabled() -> bool:
    try:
        from core.aeb.intake_policy import contribution_enabled

        return contribution_enabled()
    except Exception:
        logger.debug("could not read the clip contribution opt-in", exc_info=True)
        return False


def get_recorder() -> AEBClipRecorder | None:
    """Shared recorder or None (capture gate); never raises."""
    global _initialized
    if _initialized:
        return _recorder
    with _lock:
        if _initialized:
            return _recorder
        _init_locked()
        _initialized = True
        return _recorder


def get_uploader() -> ClipUploader | None:
    """Shared uploader, or None when nobody opted in. Never starts the recorder."""
    return _uploader


def note_intervention(active: bool) -> None:
    """Hold upload notifications while AEB is acting (plan 5.9 rule 4)."""
    up = _uploader
    if up is not None:
        up.set_intervening(active)


def _init_locked() -> None:
    global _recorder, _writer, _uploader, _debug
    debug = False
    contributing = False
    try:
        from core.settings import Settings

        debug = bool(Settings.debug)
        contributing = _contribution_enabled()
    except Exception:
        logger.debug("could not read the AEB capture gate", exc_info=True)
        return
    if not (debug or contributing):
        return
    _debug = debug
    provider = None
    try:
        from core.settings import Settings

        if bool(getattr(Settings, "aeb_capture_screenshots", True)):
            from core.aeb.screenshot import grab_thumbnail

            provider = grab_thumbnail
    except Exception:
        logger.debug("could not set up AEB screenshot provider", exc_info=True)

    try:
        # A contributor keeps a smaller store: it is a staging area, not the
        # working corpus a debug user tags from.
        store = ClipStore() if debug else ClipStore(max_bytes=_CONTRIBUTOR_MAX_BYTES)
        _uploader = _build_uploader(store, debug) if contributing else None
        writer = AsyncClipWriter(store, notify=_on_clip_written)
        writer.start()
        _writer = writer
        _recorder = AEBClipRecorder(
            writer, enabled=True, capture_tn=debug, screenshot_provider=provider,
        )
        atexit.register(_shutdown)
        logger.info("AEB clip recorder active (debug=%s, contributing=%s, screenshots=%s)",
                    debug, contributing, provider is not None)
    except Exception:
        logger.exception("failed to start AEB clip recorder; capture disabled")
        _recorder = None
        _writer = None
        _uploader = None


def _build_uploader(store: ClipStore, debug: bool) -> ClipUploader | None:
    """Start the uploader for an opted-in user. Never raises."""
    try:
        from core.settings import Settings

        # A debug user's store is the working corpus, so nothing is deleted from
        # it whatever the flag says.
        delete_after = bool(getattr(Settings, "aeb_delete_after_upload", True)) and not debug
        uploader = ClipUploader(
            store,
            delete_after=delete_after,
            notify=bool(getattr(Settings, "aeb_contribute_notify", True)),
            on_kept=_notify_kept,
        )
        uploader.start()
        logger.info("AEB clip uploader active (delete_after=%s)", delete_after)
        return uploader
    except Exception:
        logger.exception("failed to start the AEB clip uploader; nothing will be sent")
        return None


def _shutdown() -> None:
    global _recorder
    if _recorder is not None:
        try:
            _recorder.reset()
        except Exception:
            pass
    if _writer is not None:
        try:
            _writer.stop()
        except Exception:
            pass
    if _uploader is not None:
        try:
            _uploader.stop()
        except Exception:
            pass
