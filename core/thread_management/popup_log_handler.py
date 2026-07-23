"""Logging handler: records with extra popup=True go to PopupWindow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.popup.popup_window import PopupWindow


class PopupLogHandler(logging.Handler):
    """Forwards popup-flagged log records to PopupWindow.emit_message()."""

    _LEVEL_TO_TYPE: dict[int, str] = {
        logging.CRITICAL: "e",
        logging.ERROR:    "e",
        logging.WARNING:  "w",
        logging.INFO:     "n",
        logging.DEBUG:    "n",
    }

    _LEVEL_TO_TITLE: dict[int, str] = {
        logging.CRITICAL: "Critical Error",
        logging.ERROR:    "Error",
        logging.WARNING:  "Warning",
        logging.INFO:     "Notice",
        logging.DEBUG:    "Notice",
    }

    @staticmethod
    def _friendly_source_name(logger_name: str) -> str:
        """Short title source from logger name (e.g. core.telemetry_thread.thread -> Telemetry)."""
        base = logger_name

        # Prefer the most specific part of a dotted path.
        if "." in logger_name:
            parts = logger_name.split(".")
            # Paths ending in .thread: use the segment before "thread".
            if len(parts) >= 2 and parts[-1] == "thread":
                base = parts[-2]
            else:
                base = parts[-1]

        # Normalise common thread-style names: "telemetry_thread" → "Telemetry".
        base = base.replace("_thread", "")
        base = base.replace("_", " ").strip()

        if not base:
            return logger_name

        return base.title()

    def __init__(
        self,
        popup: "PopupWindow",
        min_level: int = logging.WARNING,
        duration_ms: int = 6000,
        priority: int = 10,
    ) -> None:
        super().__init__(level=min_level)
        self._popup = popup
        self._duration_ms = duration_ms
        self._priority = priority

    def _priority_for(self, record: logging.LogRecord) -> int:
        """CRITICAL=10, ERROR=1, else handler base priority."""
        if record.levelno >= logging.CRITICAL:
            return 10
        if record.levelno >= logging.ERROR:
            return 1
        return self._priority

    def emit(self, record: logging.LogRecord) -> None:
        if not getattr(record, "popup", False):
            return
        try:
            msg_type = self._LEVEL_TO_TYPE.get(record.levelno, "e")
            source_name = self._friendly_source_name(record.name)
            title = f"{source_name} {self._LEVEL_TO_TITLE.get(record.levelno, 'Error')}"
            body = record.getMessage()
            priority = self._priority_for(record)
            
            self._popup.emit_message(
                title=title,
                message=body,
                message_type=msg_type,
                duration_ms=self._duration_ms,
                priority=priority,
            )
        except Exception:
            self.handleError(record)

