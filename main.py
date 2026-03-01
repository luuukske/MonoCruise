"""
main.py — application entry point.

Responsibilities:
  1. Load settings (once).
  2. Configure logging.
  3. Instantiate and register all threads.
  4. Start watchdog (+ monitor if debug).
  5. Block until KeyboardInterrupt, then shut everything down cleanly.
"""

from __future__ import annotations

import psutil

def is_process_running(name: str) -> bool:
    try:
        for p in psutil.process_iter(['name']):
            if p.info.get('name', '').lower() == name.lower():
                return True
    except (psutil.Error, KeyError):
        pass
    return False

if is_process_running("MonoCruise.exe"):
    print("MonoCruise is already running")
    exit()

import logging
import signal
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from core.settings import Settings
from core.thread_management.registry import registry
from core.thread_management.watchdog import Watchdog
from core.thread_management.monitor  import Monitor
from core.thread_management.popup_log_handler import PopupLogHandler

from core.test_thread.thread import TestThread  # remove once real workers exist

from ui.popup.popup_window import PopupWindow

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(settings: Settings) -> None:
    fmt     = "%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    _fmt = formatter.format
    formatter.format = lambda r: _fmt(r).replace("\r", " ").replace("\n", " ")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("monocruise.log", encoding="utf-8", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def _attach_popup_handler(popup: PopupWindow) -> None:
    handler = PopupLogHandler(
        popup=popup
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)


# ---------------------------------------------------------------------------
# Thread factories
# ---------------------------------------------------------------------------

def _factory_for(thread):
    cls = type(thread)
    return cls


def _make_test_thread() -> TestThread:
    return TestThread()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def _stop_all() -> None:
    log = logging.getLogger("main")
    for t in reversed(registry.all_threads()):
        if t.is_alive():
            log.debug("stopping %s", t.name)
            t.stop()

    for t in registry.all_threads():
        if t.is_alive():
            t.join(timeout=3.0)
            if t.is_alive():
                log.warning("%s did not stop cleanly", t.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # QApplication must be created before any Qt objects (including PopupWindow)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    settings = Settings.load()
    _configure_logging(settings)
    log = logging.getLogger("main")
    log.info("starting — debug=%s", settings.debug)

    # ── Popup window ──────────────────────────────────────────────────────────
    popup = PopupWindow()
    popup.show()
    _attach_popup_handler(popup)

    # ── Watchdog ──────────────────────────────────────────────────────────────
    watchdog = Watchdog(auto_restart=settings.auto_restart)

    # ── Register workers ──────────────────────────────────────────────────────
    workers = [
        _make_test_thread(),
        # TelemetryThread(settings),
        # UIThread(settings),
    ]

    for w in workers:
        registry.register(w)
        watchdog.register_factory(w.name, _factory_for(w))

    registry.register(watchdog)

    # ── Monitor (debug only) ──────────────────────────────────────────────────
    monitor: Monitor | None = None
    if settings.debug:
        monitor = Monitor(watchdog=watchdog)
        registry.register(monitor)

    # ── Start workers ─────────────────────────────────────────────────────────
    for w in workers:
        w.start()
        log.info("started: %s", w.name)

    watchdog.start()
    log.info("started: watchdog")

    if monitor:
        monitor.start()
        log.info("started: monitor")

    # ── Poll thread liveness via QTimer (keeps Qt event loop free) ───────────
    def _check_threads() -> None:
        if not any(t.is_alive() for t in registry.all_threads()):
            log.info("all threads stopped — exiting")
            _stop_all()
            app.quit()

    poll_timer = QTimer()
    poll_timer.setInterval(250)
    poll_timer.timeout.connect(_check_threads)
    poll_timer.start()

    # ── Signal handling ───────────────────────────────────────────────────────
    def _shutdown(sig: int, _frame) -> None:
        log.info("signal %s received — shutting down", signal.Signals(sig).name)
        _stop_all()
        app.quit()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep Python's signal handler alive (Qt blocks SIGINT otherwise)
    sigint_timer = QTimer()
    sigint_timer.setInterval(200)
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()