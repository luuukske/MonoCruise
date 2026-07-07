"""Process-wide owner of the AEB clip recorder + background writer.

Debug-gated: :func:`get_recorder` returns ``None`` unless ``Settings.debug`` is
on, so in normal builds the radar / AEB loops fetch ``None`` once and touch no
capture code. When active, one recorder feeds one background ``AsyncClipWriter``;
the writer daemon starts on first use and is drained at process exit.

The radar and AEB threads call :func:`get_recorder` each loop (cheap after the
first call) and push their records; a returned ``None`` means "capture off, do
nothing". This module is the only place that reads ``Settings`` so the recorder
core stays settings-free and unit-testable.
"""

from __future__ import annotations

import atexit
import logging
import threading

from core.aeb.clip_store import AsyncClipWriter, ClipStore
from core.aeb.recorder import AEBClipRecorder

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_recorder: AEBClipRecorder | None = None
_writer: AsyncClipWriter | None = None
_initialized: bool = False


def _notify_saved(name: str) -> None:
    # One user-facing confirmation per capture (plan 3.3); details to the log file.
    logger.info("AEB clip saved", extra={"popup": True})
    logger.debug("AEB clip saved: %s", name)


def get_recorder() -> AEBClipRecorder | None:
    """Return the shared recorder, or ``None`` when capture is disabled.

    Decided once, from ``Settings.debug``, on the first call. Never raises: any
    failure to start capture leaves it disabled so a recorder problem can never
    disturb the radar / AEB loops.
    """
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
    enabled = False
    try:
        from core.settings import Settings

        enabled = bool(Settings.debug)
    except Exception:
        logger.debug("could not read Settings.debug for capture gate", exc_info=True)
        enabled = False
    if not enabled:
        return
    try:
        store = ClipStore()
        writer = AsyncClipWriter(store, notify=_notify_saved)
        writer.start()
        _writer = writer
        _recorder = AEBClipRecorder(writer, enabled=True)
        atexit.register(_shutdown)
        logger.info("AEB clip recorder active (debug mode)")
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
