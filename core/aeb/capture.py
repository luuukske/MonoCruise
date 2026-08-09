"""Debug-gated shared AEB clip recorder + writer; sole Settings touchpoint for capture."""

from __future__ import annotations

import atexit
import logging
import threading

from core.aeb.clip_store import AsyncClipWriter, ClipStore
from core.aeb.recorder import AEBClipRecorder

logger = logging.getLogger(__name__)

# Contribute-only stores stage clips for upload rather than holding a corpus,
# so they get a fifth of the debug cap.
_CONTRIBUTOR_MAX_BYTES: int = 100 * 1024 * 1024

_lock = threading.Lock()
_recorder: AEBClipRecorder | None = None
_writer: AsyncClipWriter | None = None
_initialized: bool = False


def _notify_saved(name: str) -> None:
    # One user-facing confirmation per capture (plan 3.3); details to the log file.
    logger.info("AEB clip saved", extra={"popup": True})
    logger.debug("AEB clip saved: %s", name)


def get_recorder() -> AEBClipRecorder | None:
    """Shared recorder or None (Settings.debug gate); never raises."""
    global _initialized
    if _initialized:
        return _recorder
    with _lock:
        if _initialized:
            return _recorder
        _init_locked()
        _initialized = True
        return _recorder


def _init_locked() -> None:
    global _recorder, _writer
    debug = False
    contributing = False
    try:
        from core.aeb.intake_policy import contribution_enabled
        from core.settings import Settings

        debug = bool(Settings.debug)
        contributing = contribution_enabled()
    except Exception:
        logger.debug("could not read the AEB capture gate", exc_info=True)
        return
    if not (debug or contributing):
        return
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
        writer = AsyncClipWriter(store, notify=_notify_saved)
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
