from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict

from base_thread import BaseThread, SnapshotMixin
from registry import Registry


logger = logging.getLogger("monocruise")


@dataclass
class ThreadMetrics(SnapshotMixin):
    name: str = ""
    state: str = "unknown"
    alive: bool = False
    iteration_count: int = 0
    last_loop_duration: float | None = None
    last_heartbeat_age: float | None = None
    last_exception: str | None = None


@dataclass
class MonitorData(SnapshotMixin):
    last_refresh_ts: float = 0.0
    threads: Dict[str, ThreadMetrics] = field(default_factory=dict)


class MonitorThread(BaseThread[MonitorData]):
    """
    Periodically snapshots the Registry status into a typed dataclass so
    that the CLI (and other components) can inspect thread performance
    without recomputing metrics each time.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        name: str = "monitor",
        loop_interval: float = 1.0,
    ) -> None:
        self._registry = registry
        super().__init__(name=name, loop_interval=loop_interval, auto_restart=False)

    def create_initial_data(self) -> MonitorData:
        return MonitorData()

    # ------------------------------------------------------------------
    # BaseThread hooks
    # ------------------------------------------------------------------

    def on_startup(self) -> None:
        logger.info("Monitor starting")

    def loop(self) -> None:
        snapshot = self._registry.status_snapshot()
        now = time.monotonic()

        metrics_map: Dict[str, ThreadMetrics] = {}
        for entry in snapshot:
            metrics_map[entry["name"]] = ThreadMetrics(
                name=entry["name"],
                state=entry["state"],
                alive=bool(entry["alive"]),
                iteration_count=int(entry["iteration_count"]),
                last_loop_duration=entry["last_loop_duration"],
                last_heartbeat_age=entry["last_heartbeat_age"],
                last_exception=entry["last_exception"],
            )

        self.data.update(last_refresh_ts=now, threads=metrics_map)

    def on_shutdown(self) -> None:
        logger.info("Monitor stopping")


__all__ = ["MonitorThread", "MonitorData", "ThreadMetrics"]

