"""Log records written by one background thread, so a stalled disk never holds a caller.

A ``FileHandler`` flushes under its own lock on every record. When the disk stalls, every
thread that logs waits behind that flush, the pedal thread included (2026-11-20, README).
``QueuedLogging`` puts records on a bounded queue instead; a full queue drops the record.
"""

from __future__ import annotations

import logging
import queue
import threading

_STOP = object()


class _DroppingQueueHandler(logging.Handler):
    """Formats nothing and waits for nothing: hands the record to the writer thread."""

    def __init__(self, records: queue.Queue) -> None:
        super().__init__(logging.DEBUG)
        self._records = records
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._records.put_nowait(self._prepare(record))
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.handleError(record)

    @staticmethod
    def _prepare(record: logging.LogRecord) -> logging.LogRecord:
        # Render args and traceback now: they may change or die before the writer gets there.
        copy = logging.makeLogRecord(record.__dict__)
        copy.msg = record.getMessage()
        copy.args = None
        if record.exc_info and not record.exc_text:
            copy.exc_text = logging.Formatter().formatException(record.exc_info)
        copy.exc_info = None
        return copy


class QueuedLogging:
    """Owns the queue, the root-side handler and the writer thread for *handlers*."""

    def __init__(self, handlers: list[logging.Handler], max_records: int = 50_000) -> None:
        self._records: queue.Queue = queue.Queue(maxsize=max_records)
        self._handlers = list(handlers)
        self.handler = _DroppingQueueHandler(self._records)
        self._thread = threading.Thread(target=self._run, name="log_writer", daemon=True)

    @property
    def dropped(self) -> int:
        return self.handler.dropped

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> bool:
        """Write what is queued, then end the thread. False if the disk did not let it finish in time."""
        if not self._thread.is_alive():
            return True
        try:
            self._records.put(_STOP, timeout=timeout)
        except queue.Full:
            return False
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            record = self._records.get()
            if record is _STOP:
                return
            for handler in self._handlers:
                if record.levelno < handler.level:
                    continue
                try:
                    handler.handle(record)
                except Exception:
                    handler.handleError(record)
