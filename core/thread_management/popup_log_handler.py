"""
popup_log_handler.py — logging.Handler that routes ERROR+ records to PopupWindow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.popup.popup_window import PopupWindow


class PopupLogHandler(logging.Handler):
    """
    Attach to any logger to forward records at or above `min_level`
    to a PopupWindow via emit_message().

    Level → message_type mapping:
        CRITICAL / ERROR  → 'e'  (error style)
        WARNING           → 'w'  (warning style)
        INFO              → 'n'  (notice style)
        DEBUG             → 'n'  (notice style)
    """

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
        """
        Derive a short, human-friendly source name from a logger name.

        Examples:
            "core.telemetry_thread.thread"      → "Telemetry"
            "telemetry_thread"                  → "Telemetry"
            "core.thread_management.watchdog"   → "Watchdog"
            "main"                              → "Main"
        """
        base = logger_name

        # Prefer the most specific part of a dotted path.
        if "." in logger_name:
            parts = logger_name.split(".")
            # If the module path ends with ".thread", take the segment before it
            # (e.g. "core.telemetry_thread.thread" → "telemetry_thread").
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
        """
        Compute queue priority for a given log record.

        CRITICAL records always use priority 10.
        ERROR records always use priority 1.
        All other levels fall back to the handler's base priority.
        """
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
