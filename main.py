from __future__ import annotations

import logging
import sys
from typing import Optional

from monitor import MonitorData, MonitorThread
from registry import Registry
from settings import Settings
from telemetry_thread import TelemetryThread
from cruise_thread import CruiseThread
from updater_thread import UpdaterThread
from ui_thread import UiThread
from watchdog import WatchdogThread


def _setup_logging() -> None:
    """
    Configure the process-wide logger.

    The format includes the thread name automatically so that all worker
    logs can be correlated in real time.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
    )


def _build_registry(settings: Settings) -> tuple[Registry, Optional[MonitorThread]]:
    registry = Registry()
    auto_restart = settings.auto_restart_workers

    # Core worker threads
    registry.register_thread(
        "telemetry",
        lambda r: TelemetryThread(
            r,
            loop_interval=settings.telemetry_loop_interval,
        ),
        auto_start=True,
        auto_restart=auto_restart,
        heartbeat_timeout=settings.heartbeat_timeout,
    )

    registry.register_thread(
        "cruise",
        lambda r: CruiseThread(
            r,
            loop_interval=settings.cruise_loop_interval,
        ),
        auto_start=True,
        auto_restart=auto_restart,
        heartbeat_timeout=settings.heartbeat_timeout,
    )

    registry.register_thread(
        "updater",
        lambda r: UpdaterThread(
            r,
            loop_interval=settings.updater_loop_interval,
        ),
        auto_start=True,
        auto_restart=False,  # updates are best-effort
        heartbeat_timeout=settings.updater_loop_interval * 2.0,
    )

    registry.register_thread(
        "ui",
        lambda r: UiThread(
            r,
            loop_interval=settings.ui_loop_interval,
        ),
        auto_start=True,
        auto_restart=auto_restart,
        heartbeat_timeout=settings.heartbeat_timeout,
    )

    # Infrastructure threads
    registry.register_thread(
        "watchdog",
        lambda r: WatchdogThread(
            r,
            loop_interval=settings.watchdog_loop_interval,
            default_heartbeat_timeout=settings.heartbeat_timeout,
        ),
        auto_start=True,
        auto_restart=False,
    )

    registry.register_thread(
        "monitor",
        lambda r: MonitorThread(
            r,
            loop_interval=settings.monitor_loop_interval,
        ),
        auto_start=True,
        auto_restart=False,
        heartbeat_timeout=settings.monitor_loop_interval * 2.0,
    )

    registry.start_all()

    monitor_thread = registry.get_thread("monitor")
    return registry, monitor_thread if isinstance(monitor_thread, MonitorThread) else None


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  help / h           - Show this help\n"
        "  threads / ls       - List registered threads\n"
        "  status             - Show basic status for all threads\n"
        "  perf               - Show last performance snapshot (from monitor)\n"
        "  stop <name>        - Stop a thread\n"
        "  start <name>       - Start (or restart) a thread\n"
        "  restart <name>     - Restart a thread\n"
        "  quit / exit        - Stop all threads and exit\n"
    )


def _print_status(registry: Registry) -> None:
    snapshot = registry.status_snapshot()
    print("\nThread status:")
    for entry in snapshot:
        name = entry["name"]
        state = entry["state"]
        hb_age = entry["last_heartbeat_age"]
        exc = entry["last_exception"]
        iter_count = entry["iteration_count"]
        loop_dur = entry["last_loop_duration"]
        print(
            f"  {name:10s} state={state:10s} "
            f"hb_age={hb_age!r} iter={iter_count} last_loop={loop_dur!r} "
            f"exc={'yes' if exc else 'no'}"
        )
    print()


def _print_perf(monitor: Optional[MonitorThread]) -> None:
    if monitor is None:
        print("Monitor thread not available.\n")
        return

    data: MonitorData = monitor.data.snapshot()  # type: ignore[assignment]
    print(f"\nPerformance snapshot (ts={data.last_refresh_ts:.3f}):")
    for name, metrics in sorted(data.threads.items()):
        print(
            f"  {name:10s} state={metrics.state:10s} "
            f"alive={metrics.alive} iter={metrics.iteration_count} "
            f"last_loop={metrics.last_loop_duration!r} "
            f"hb_age={metrics.last_heartbeat_age!r}"
        )
    print()


def _cli_loop(registry: Registry, monitor: Optional[MonitorThread]) -> None:
    _print_help()
    while True:
        try:
            line = input("monocruise> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in {"help", "h"}:
            _print_help()
        elif cmd in {"threads", "ls"}:
            names = [spec.name for spec in registry.iter_specs()]
            print("Threads:", ", ".join(names))
        elif cmd == "status":
            _print_status(registry)
        elif cmd == "perf":
            _print_perf(monitor)
        elif cmd == "stop" and args:
            registry.stop_thread(args[0])
        elif cmd == "start" and args:
            registry.start_thread(args[0])
        elif cmd == "restart" and args:
            registry.restart_thread(args[0])
        elif cmd in {"quit", "exit"}:
            print("Stopping all threads and exiting...")
            break
        else:
            print(f"Unknown command: {cmd!r}. Type 'help' for a list of commands.")


def main(argv: list[str] | None = None) -> int:
    _setup_logging()

    settings = Settings.load("settings.json")
    logging.getLogger("monocruise").info("Settings loaded: %s", settings)

    registry, monitor = _build_registry(settings)

    try:
        _cli_loop(registry, monitor)
    finally:
        registry.stop_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

