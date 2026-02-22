"""
Test: logging integration with popup notifications.
Forwards errors and warnings from any thread to the popup safely.
"""
import logging
import os
import sys
import threading
import time

# Ensure popup package imports resolve when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "popup"))

from PySide6.QtWidgets import QApplication

from popup.popup_window import PopupWindow


def _level_to_message_type(level: int) -> str:
    """Map logging level to popup message type."""
    if level >= logging.ERROR:
        return "e"
    if level >= logging.WARNING:
        return "w"
    if level >= logging.INFO:
        return "n"
    return "n"  # DEBUG


def _level_to_title(level: int) -> str:
    """Map logging level to popup title."""
    if level >= logging.CRITICAL:
        return "Critical"
    if level >= logging.ERROR:
        return "Error"
    if level >= logging.WARNING:
        return "Warning"
    if level >= logging.INFO:
        return "Info"
    return "Debug"


class PopupLoggingHandler(logging.Handler):
    """
    Thread-safe handler that forwards log records to the popup.
    Safe to use from any thread; emit_message uses Qt signals.
    """

    def __init__(self, popup_getter):
        super().__init__()
        self._popup_getter = popup_getter

    def emit(self, record: logging.LogRecord):
        try:
            popup = self._popup_getter()
            if popup is None:
                return
            msg = self.format(record)
            title = _level_to_title(record.levelno)
            msg_type = _level_to_message_type(record.levelno)
            popup.emit_message(title=title, message=msg, message_type=msg_type)
        except Exception:
            self.handleError(record)


def main():
    app = QApplication(sys.argv)

    popup = PopupWindow()
    popup.set_scale(1)
    popup_ref = [popup]  # Mutable ref so handler can read it

    handler = PopupLoggingHandler(lambda: popup_ref[0] if popup_ref else None)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    logger = logging.getLogger("test")

    def worker_error():
        time.sleep(1.5)
        logger.error("Worker A: Connection failed!\nRetrying in 5s...")

    def worker_warning():
        time.sleep(2.5)
        logger.warning("Worker B: Low memory (85% used)")

    def worker_critical():
        time.sleep(3.5)
        logger.critical("Worker C: Disk full - data may be lost!")

    def worker_info():
        time.sleep(0.5)
        logger.info("Worker D: Task started")
        time.sleep(4)
        logger.info("Worker D: Task completed")

    def main_thread_warning():
        time.sleep(4.5)
        logger.warning("Main: Session timeout approaching")

    threads = [
        threading.Thread(target=worker_error, daemon=True),
        threading.Thread(target=worker_warning, daemon=True),
        threading.Thread(target=worker_critical, daemon=True),
        threading.Thread(target=worker_info, daemon=True),
        threading.Thread(target=main_thread_warning, daemon=True),
    ]
    for t in threads:
        t.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
