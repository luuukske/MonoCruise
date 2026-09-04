"""
main.py: application entry point.

Responsibilities:
  1. Load settings (once).
  2. Configure logging.
  3. Instantiate and register all threads.
  4. Start watchdog (+ monitor if debug).
  5. Block until KeyboardInterrupt, then shut everything down cleanly.
"""

from __future__ import annotations

import sys

_INSTANCE_MUTEX = None


def _acquire_single_instance() -> bool:
    """Windows named mutex; also used by checker to detect a running instance."""
    global _INSTANCE_MUTEX
    if sys.platform != "win32":
        return True
    import ctypes
    kernel32 = ctypes.windll.kernel32
    _INSTANCE_MUTEX = kernel32.CreateMutexW(None, False, "MonoCruiseSingleInstance")
    return kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


if not _acquire_single_instance():
    print("MonoCruise is already running")
    raise SystemExit(0)


_SHUTDOWN_EVENT = None


def _create_shutdown_event():
    """Named auto-reset event for updater-requested shutdown (Windows)."""
    global _SHUTDOWN_EVENT
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateEventW.restype = ctypes.c_void_p
        _SHUTDOWN_EVENT = kernel32.CreateEventW(None, False, False, "MonoCruiseShutdownRequest")
        return _SHUTDOWN_EVENT
    except Exception:
        return None


def _shutdown_requested(handle) -> bool:
    """True when the updater has signalled the shutdown event (clears it)."""
    if not handle:
        return False
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    return kernel32.WaitForSingleObject(handle, 0) == 0  # WAIT_OBJECT_0

import logging
import re
import signal
import time

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
# Linux install roots: /home, /root, /opt, /usr, /var, /tmp, /etc, /mnt, /srv, /run.
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

    # File log is what may be shared: must redact paths.
    file_formatter = _RedactingFormatter(fmt, datefmt=datefmt)
    _strip_newlines(file_formatter)

    # Console log stays on the user's machine: skip the regex passes for speed.
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


# Version marker

def _write_version_marker() -> None:
    """Plain-text installed_version.txt for the standalone updater (best-effort)."""
    try:
        (CONFIG_PATH.parent / "installed_version.txt").write_text(
            __version__, encoding="utf-8"
        )
    except Exception:
        logging.getLogger("main").debug("could not write version marker", exc_info=True)


def _read_version_marker() -> str:
    """Previous-run version from marker (read before _write_version_marker)."""
    try:
        return (CONFIG_PATH.parent / "installed_version.txt").read_text(
            encoding="utf-8"
        ).strip()
    except Exception:
        return ""


def _sync_channel_to_build(settings: Settings, previous_version: str) -> None:
    """After a version change, set update_channel to match stable vs preview build."""
    if previous_version == __version__:
        return
    desired = "preview" if "preview" in __version__.lower() else "stable"
    if settings.update_channel != desired:
        logging.getLogger("main").info(
            "version changed %s -> %s; matching update_channel to %s build",
            previous_version or "(none)", __version__, desired,
        )
        Settings.save(values={"update_channel": desired})


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
    log.info("starting: debug=%s", settings.debug)
    previous_version = _read_version_marker()
    _write_version_marker()
    _sync_channel_to_build(settings, previous_version)

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
                # Malformed debug config must not crash the main loop.
                settings_log.exception("failed to auto-refresh settings")

        settings_timer = QTimer()
        settings_timer.setInterval(1000)
        settings_timer.timeout.connect(_refresh_settings)
        settings_timer.start()

    # Popup window
    popup = PopupWindow()
    popup.show()
    _attach_popup_handler(popup)

    # SDK boot check on a daemon thread; auto-install via _auto_install_sdk when needed.
    from core.sdk_installer import SdkCheckResult, start_boot_check

    def _auto_install_sdk(result: SdkCheckResult) -> None:
        """Boot-thread SDK apply; never closes game; defers loaded outdated DLLs."""
        from core.sdk_installer import get_manager

        try:
            results = get_manager().apply(
                result.games_needing_action,
                close_running=False,
                allow_running_missing=True,
            )
        except Exception:
            log.exception("SDK auto-install failed")
            PopupWindow.emit(
                "Game plugin",
                "MonoCruise could not install the game plugin. Open settings "
                "and use Reinstall SDK.",
                "w",
                duration_ms=9000,
            )
            return

        changed = [r for r in results if (r.installed or r.disabled) and not r.errors]
        cleaned = [r for r in changed if r.disabled]
        deferred = [r for r in results if r.deferred_running and not r.errors]
        unsupported = [r for r in results if r.unsupported]
        failed = [r for r in results if r.errors]

        if changed:
            games = ", ".join(r.game_type.upper() for r in changed)
            if cleaned:
                log.info(
                    "SDK auto-install: disabled superseded plugins for %s",
                    ", ".join(r.game_type.upper() for r in cleaned),
                )
                PopupWindow.emit(
                    "Old game plugin disabled",
                    f"MonoCruise found a plugin from an older version for {games} and "
                    f"disabled it. Restart the game to activate pedal control.",
                    "c",
                    duration_ms=9000,
                )
            else:
                log.info("SDK auto-install: installed the plugin for %s", games)
                PopupWindow.emit(
                    "Game plugin installed",
                    f"MonoCruise installed the game plugin for {games}. Restart the "
                    f"game to activate pedal control.",
                    "c",
                    duration_ms=9000,
                )
        if deferred:
            games = ", ".join(r.game_type.upper() for r in deferred)
            log.info("SDK auto-install: update deferred, %s is running", games)
            PopupWindow.emit(
                "Game plugin update pending",
                f"{games} is running an outdated plugin. Close the game, then use "
                f"Reinstall SDK in settings to finish updating.",
                "n",
                duration_ms=9000,
            )
        if unsupported:
            _warn_unsupported_versions(unsupported)
        if failed:
            games = ", ".join(r.game_type.upper() for r in failed)
            log.warning("SDK auto-install failed for %s", games)
            PopupWindow.emit(
                "Game plugin install failed",
                f"MonoCruise could not install the game plugin for {games}. Open "
                f"settings and use Reinstall SDK.",
                "w",
                duration_ms=9000,
            )

    def _warn_unsupported_versions(entries) -> None:
        """No plugin exists for this game version, so ACC and AEB stay blind."""
        detail = "; ".join(
            f"{e.game_type.upper()}: {e.unsupported_reason}" for e in entries
        )
        log.warning("SDK check: %s", detail)
        PopupWindow.emit(
            "Unsupported game version",
            f"{detail}. Adaptive cruise control and emergency braking will not "
            f"see other vehicles until a matching plugin is available.",
            "e",
            duration_ms=12000,
        )

    def _on_sdk_result(result: SdkCheckResult) -> None:
        if not result.found_games:
            log.info("SDK check: no ETS2/ATS installation found")
        elif result.needs_action:
            games = ", ".join(g.game_type.upper() for g in result.games_needing_action)
            log.info("SDK check: action needed for %s; applying", games)
            _auto_install_sdk(result)
        elif result.unsupported_games:
            _warn_unsupported_versions(result.unsupported_games)
        else:
            log.info("SDK check: SDK DLLs present and up to date")

    try:
        start_boot_check(_on_sdk_result)
    except Exception:
        log.debug("could not start SDK boot check", exc_info=True)

    # Update check daemon; popup here, banner/tint via update_is_pending (see update_check README).
    from core.update_check import UpdateCheckResult, mark_popup_shown, popup_throttled, start_update_check

    def _on_update_result(result: UpdateCheckResult) -> None:
        if not result.update_available:
            return
        if not getattr(settings, "notify_for_updates", True):
            return  # user opted out of popups; banner + tint still signal it
        if popup_throttled():
            return  # already prompted recently; banner + tint still signal it
        mark_popup_shown()
        PopupWindow.emit(
            "Update available",
            f"MonoCruise {result.latest_version} is ready. Open settings to update.",
            "n",
            duration_ms=8000,
        )

    try:
        start_update_check(_on_update_result)
    except Exception:
        log.debug("could not start update check", exc_info=True)

    # Clip-contribution intake policy. No-op unless the user opted in, so a
    # machine that never ticked the box makes no request (see core/aeb/README.md).
    try:
        from core.aeb.intake_policy import start_policy_fetch

        start_policy_fetch()
    except Exception:
        log.debug("could not start the AEB intake policy fetch", exc_info=True)

    # Main window (lives on the main thread: no separate thread needed)
    window = create_main_window(settings, version=f"v{__version__}")
    registry.register_object("main_window", window)
    window.window_closed.connect(lambda: (_stop_all(), app.quit()))
    # Early minimized taskbar presence (checker launches); final visibility from telemetry poll.
    window.apply_startup_visibility()

    # Pedal visualization bar (shows aforward/abackward + em_stop; lives on main thread)
    _visualization_bar = create_visualization_bar()
    registry.register_object("visualization_bar", _visualization_bar)

    # AEB debug view: developer tooling, only shown in debug mode.
    if settings.debug:
        aeb_debug = AEBDebugWindow()
        aeb_debug.show()
        registry.register_object("aeb_debug", aeb_debug)

    # Watchdog
    watchdog = Watchdog()

    # Register workers; watchdog restart uses _factory_for(worker).
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

    # Updater shutdown event; window.close() for same path as the title-bar X.
    shutdown_event = _create_shutdown_event()

    # Apply updater_pending swap after updater exits (shared/updater_swap.py).
    from shared.updater_swap import apply_pending_updater_swap, pending_swap_staged
    from core.update_check import update_is_pending
    swap_root = str(CONFIG_PATH.parent)

    # Auto-close waits for the live viz bar to decay to center before quitting.
    # Timestamp of first seeing request_quit; None when no quit is pending.
    _quit_requested_at: float | None = None
    _QUIT_VIZ_TIMEOUT_S = 3.0

    def _visualization_settled() -> bool:
        try:
            bar = registry.get("visualization_bar")
        except KeyError:
            return True
        try:
            return bool(bar.is_settled_at_center())
        except Exception:
            return True

    # Poll thread liveness via QTimer (keeps Qt event loop free)
    def _check_threads() -> None:
        nonlocal _quit_requested_at
        if _shutdown_requested(shutdown_event):
            log.info("shutdown requested by updater: closing")
            window.close()
            return
        if pending_swap_staged(swap_root):
            apply_pending_updater_swap(swap_root)
        try:
            telemetry = registry.get_thread("telemetry_thread")
            if telemetry.data.request_quit:
                # Keep open when the user has restored the window, or when an
                # update prompt must stay visible.
                if getattr(window, "is_open_on_taskbar", False):
                    log.info("auto-close cancelled: window is open")
                    telemetry.stay_open_after_disconnect()
                    _quit_requested_at = None
                    return
                if update_is_pending():
                    log.info("auto-close cancelled: update pending; showing window")
                    telemetry.stay_open_after_disconnect()
                    _quit_requested_at = None
                    # Surface banner / update tint so the user can act on it.
                    window.show_normally()
                    return
                now = time.monotonic()
                if _quit_requested_at is None:
                    _quit_requested_at = now
                    log.info("auto-close: waiting for live visualization to settle")
                settled = _visualization_settled()
                timed_out = (now - _quit_requested_at) >= _QUIT_VIZ_TIMEOUT_S
                if not settled and not timed_out:
                    return
                if timed_out and not settled:
                    log.info("auto-close: visualization settle timed out; exiting")
                else:
                    log.info("shutdown requested: exiting")
                _stop_all()
                app.quit()
                return
            _quit_requested_at = None
        except (KeyError, AttributeError):
            pass
        if not any(t.is_alive() for t in registry.all_threads()):
            log.info("all threads stopped: exiting")
            _stop_all()
            app.quit()

    poll_timer = QTimer()
    poll_timer.setInterval(250)
    poll_timer.timeout.connect(_check_threads)
    poll_timer.start()

    # Signal handling
    def _shutdown(sig: int, _frame) -> None:
        log.info("signal %s received: shutting down", signal.Signals(sig).name)
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
