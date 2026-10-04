"""Local user feedback for replay review; never changes brake calibration."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import queue
import threading
from datetime import datetime, timezone

from core import settings

_log = logging.getLogger(__name__)


class FeedbackJournal:
    def __init__(self, path: Path | None = None):
        self.path = path or settings.CONFIG_PATH.parent / "maps" / "aeb-feedback.jsonl"
        self._queue = queue.Queue(maxsize=32)
        self._lock = threading.Lock()
        self._worker = None
        self.status = "Feedback stays local; no automatic brake tuning"

    def submit(self, kind, context):
        if kind not in {"false_brake", "missed_hazard"}:
            raise ValueError("Unknown feedback kind")
        record = {"schema": 1, "time": datetime.now(timezone.utc).isoformat(),
                  "label": kind, "context": context}
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, daemon=True)
                self._worker.start()
        try:
            self.status = "Saving feedback..."
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            self.status = "Feedback queue full; try again"
            return False

    def _write(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _run(self):
        while True:
            record = self._queue.get()
            try:
                self._write(record)
                self.status = "Feedback saved locally for review; brake tuning unchanged"
            except Exception:
                _log.warning("Could not save local AEB feedback")
                self.status = "Feedback could not be saved"
            finally:
                self._queue.task_done()


journal = FeedbackJournal()
