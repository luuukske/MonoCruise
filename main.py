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

import logging
import signal
import sys
import time

from core.settings  import Settings
from core.thread_management.registry  import registry
from core.thread_management.watchdog  import Watchdog
from core.thread_management.monitor   import Monitor

# ── Worker imports ────────────────────────────────────────────────────────────
# from core.telemetry.thread import TelemetryThread
# from core.ui.thread        import UIThread
# ---------------------------------------------------------------------------
# Placeholder workers for demonstration
# ---------------------------------------------------------------------------
from core.example_thread.thread import MyThread  # remove once real workers exist


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)-20s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Thread factories
# For watchdog restarts: factory must be a zero-arg callable → BaseThread.
# ---------------------------------------------------------------------------

def _make_my_thread() -> MyThread:
    return MyThread()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    settings = Settings.load()
    _configure_logging(settings)
    log = logging.getLogger("main")
    log.info("starting — debug=%s", settings.debug)

    # ── Watchdog ─────────────────────────────────────────────────────────────
    watchdog = Watchdog(auto_restart=settings.auto_restart)

    # ── Register workers ──────────────────────────────────────────────────────
    workers = [
        _make_my_thread(),
        # Add more threads here:
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

    # ── Start ─────────────────────────────────────────────────────────────────
    for w in workers:
        w.start()
        log.info("started: %s", w.name)

    watchdog.start()
    log.info("started: watchdog")

    if monitor:
        monitor.start()
        log.info("started: monitor")

    # ── Block ─────────────────────────────────────────────────────────────────
    def _shutdown(sig: int, _frame) -> None:  # type: ignore[type-arg]
        log.info("signal %s received — shutting down", signal.Signals(sig).name)
        _stop_all()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while any(t.is_alive() for t in registry.all_threads()):
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_all()

    log.info("all threads stopped — goodbye")


def _factory_for(thread):
    """Return a zero-arg factory that re-creates the same thread type."""
    cls = type(thread)
    return cls  # works when __init__ takes no required args


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


if __name__ == "__main__":
    main()
