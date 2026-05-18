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


def is_process_running(name: str) -> bool:
    import psutil
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
import re
import signal
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from core.settings import Settings, CONFIG_PATH
from core.version import __version__
from core.thread_management.registry import registry
from core.thread_management.watchdog import Watchdog
from core.thread_management.monitor  import Monitor
from core.thread_management.popup_log_handler import PopupLogHandler

from core.telemetry_thread.thread import TelemetryThread
from core.main_pedal_thread.thread import MainPedalThread
from core.keyboard_thread.thread import KeyboardThread
from core.button_device_thread.thread import ButtonDeviceThread
from core.cruise_control_thread.thread import CruiseControlThread
from core.sending_thread.thread import SendingThread, create_visualization_bar
from core.radar.thread import RadarThread
from core.aeb.thread import AEBThread
from core.acc.thread import ACCThread
from core.aeb.debug_window import AEBDebugWindow

from ui.main_window import create_main_window
from ui.popup.popup_window import PopupWindow

# Path redaction
# This is to protect user information from being shared online.

_REDACTED_PATH_PLACEHOLDER = "#:/####-redacted-absolute-path-####/"
# Windows: C:\... or C:/... (exclude :// such as in URLs)
_WIN_PATH_RE = re.compile(r'[A-Za-z]:(?!//)[\\\/][^\s"\'<>|*?\x00-\x1f]*')
# Linux: /home/..., /root/..., /opt/..., /usr/..., /var/..., /tmp/..., /etc/..., /mnt/..., /srv/..., /run/...
_LIN_PATH_RE = re.compile(r'(?<!\w)/(?:home|root|opt|usr|var|tmp|etc|mnt|srv|run)/[^\s"\'<>|*?\x00-\x1f]*')


def _shorten_path(match: re.Match) -> str:
    """Keep only the last two components (parent/filename) and redact the rest."""
    parts = re.split(r'[/\\]+', match.group(0).rstrip('/\\'))
    parts = [p for p in parts if p]
    tail = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1] if parts else ""
    return f"{_REDACTED_PATH_PLACEHOLDER}{tail}"


def _redact_paths(text: str) -> str:
    text = _WIN_PATH_RE.sub(_shorten_path, text)
    text = _LIN_PATH_RE.sub(_shorten_path, text)
    return text


class _RedactingFormatter(logging.Formatter):
    """Formatter that strips absolute paths from every log line after formatting."""

    def format(self, record: logging.LogRecord) -> str:
        return _redact_paths(super().format(record))


# Logging

def _configure_logging() -> None:
    fmt     = "%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s"
    datefmt = "%H:%M:%S"

    def _strip_newlines(formatter: logging.Formatter) -> None:
        _fmt = formatter.format
        formatter.format = lambda r: _fmt(r).replace("\r", " ").replace("\n", " ")

    # File log is what may be shared — must redact paths.
    file_formatter = _RedactingFormatter(fmt, datefmt=datefmt)
    _strip_newlines(file_formatter)

    # Console log stays on the user's machine — skip the regex passes for speed.
    stream_formatter = logging.Formatter(fmt, datefmt=datefmt)
    _strip_newlines(stream_formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    file_handler = logging.FileHandler("monocruise.log", encoding="utf-8", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def _attach_popup_handler(popup: PopupWindow) -> None:
    handler = PopupLogHandler(
        popup=popup,
        min_level=logging.INFO
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)


# Thread factories

def _factory_for(thread):
    """Return a no-arg callable that creates a new instance of the same type (for watchdog restart)."""
    cls = type(thread)
    return cls


# Shutdown

def _stop_all() -> None:
    log = logging.getLogger("main")
    for t in reversed(registry.all_threads()):
        if t.is_alive():
            log.debug("stopping %s", t.name)
            t.stop()
            t.join(timeout=3.0)

    for t in registry.all_threads():
        if t.is_alive():
            t.stop(force=True)
            t.join(timeout=3.0)
            if t.is_alive():
                log.warning("%s did not stop cleanly", t.name)


# Main
def main() -> None:
    # QApplication must be created before any Qt objects (including PopupWindow)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    settings = Settings()
    settings.load()
    _configure_logging()
    log = logging.getLogger("main")
    log.info("starting — debug=%s", settings.debug)

    # Auto-refresh settings in debug mode (lets you edit config.json while running).
    # Runs on the Qt main thread, so it doesn't block any worker thread loops.
    if settings.debug:
        settings_log = logging.getLogger("settings")
        _settings_last_mtime_ns: int | None = None

        def _refresh_settings() -> None:
            nonlocal _settings_last_mtime_ns
            try:
                mtime_ns = CONFIG_PATH.stat().st_mtime_ns if CONFIG_PATH.exists() else None
                if mtime_ns is not None and mtime_ns == _settings_last_mtime_ns:
                    return
                _settings_last_mtime_ns = mtime_ns
                Settings.load()
            except Exception:
                # Don't ever crash the main loop because of a malformed config during debug tinkering.
                settings_log.exception("failed to auto-refresh settings")

        settings_timer = QTimer()
        settings_timer.setInterval(1000)
        settings_timer.timeout.connect(_refresh_settings)
        settings_timer.start()

    # Popup window
    popup = PopupWindow()
    popup.show()
    _attach_popup_handler(popup)

    # Main window (lives on the main thread — no separate thread needed)
    window = create_main_window(settings, version=f"v{__version__}")
    registry.register_object("main_window", window)
    window.window_closed.connect(lambda: (_stop_all(), app.quit()))

    aeb_debug = AEBDebugWindow()
    aeb_debug.show()
    registry.register_object("aeb_debug", aeb_debug)

    # Watchdog
    watchdog = Watchdog()

    # Register workers (instantiate on the spot; _factory_for(worker) gives watchdog the class for restart)
    workers = [
        TelemetryThread(),
        MainPedalThread(),
        KeyboardThread(),
        ButtonDeviceThread(),
        CruiseControlThread(),
        SendingThread(),
        RadarThread(),
        AEBThread(),
        ACCThread(),
    ]

    for w in workers:
        registry.register(w)
        watchdog.register_factory(w.name, _factory_for(w))

    registry.register(watchdog)

    # Monitor (debug only)
    monitor: Monitor | None = None
    if settings.debug:
        monitor = Monitor(watchdog=watchdog)
        registry.register(monitor)

    # Start workers
    for w in workers:
        w.start()
        log.info("started: %s", w.name)

    watchdog.start()
    log.info("started: watchdog")

    if monitor:
        monitor.start()
        log.info("started: monitor")

    window.apply_startup_visibility()

    # Pedal visualization bar (shows aforward/abackward + em_stop; lives on main thread)
    _visualization_bar = create_visualization_bar()

    # Poll thread liveness via QTimer (keeps Qt event loop free)
    def _check_threads() -> None:
        try:
            telemetry = registry.get_thread("telemetry_thread")
            if telemetry.data.request_quit:
                log.info("shutdown requested — exiting")
                _stop_all()
                app.quit()
                return
        except (KeyError, AttributeError):
            pass
        if not any(t.is_alive() for t in registry.all_threads()):
            log.info("all threads stopped — exiting")
            _stop_all()
            app.quit()

    poll_timer = QTimer()
    poll_timer.setInterval(250)
    poll_timer.timeout.connect(_check_threads)
    poll_timer.start()

    # Signal handling
    def _shutdown(sig: int, _frame) -> None:
        log.info("signal %s received — shutting down", signal.Signals(sig).name)
        try:
            telemetry = registry.get_thread("telemetry_thread")
            with telemetry.data._lock:
                telemetry.data.request_quit = True
        except KeyError:
            pass
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