from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List

from base_thread import BaseThread, SnapshotMixin
from registry import Registry, ThreadSpec


logger = logging.getLogger("monocruise")


@dataclass
class WatchdogData(SnapshotMixin):
    last_check_ts: float = 0.0
    last_problems: List[str] = field(default_factory=list)


class WatchdogThread(BaseThread[WatchdogData]):
    """
    Supervises all registered threads:

    - Detects threads that have stopped or whose heartbeat is stale.
    - Logs problems.
    - Optionally auto-restarts threads whose ThreadSpec.auto_restart is True.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        name: str = "watchdog",
        loop_interval: float = 0.2,
        default_heartbeat_timeout: float = 1.0,
    ) -> None:
        self._registry = registry
        self._default_timeout = float(default_heartbeat_timeout)
        super().__init__(name=name, loop_interval=loop_interval, auto_restart=False)

    def create_initial_data(self) -> WatchdogData:
        return WatchdogData()

    # ------------------------------------------------------------------
    # BaseThread hooks
    # ------------------------------------------------------------------

    def on_startup(self) -> None:
        logger.info("Watchdog starting")

    def loop(self) -> None:
        now = time.monotonic()
        problems: List[str] = []

        for spec in self._registry.iter_specs():
            self._check_thread(spec, now, problems)

        # Record a concise summary for monitoring / CLI.
        self.data.update(last_check_ts=now, last_problems=list(problems))

    def on_shutdown(self) -> None:
        logger.info("Watchdog stopping")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_thread(self, spec: ThreadSpec, now: float, problems: List[str]) -> None:
        name = spec.name

        # Skip self – watchdog health is implicit.
        if name == self.name:
            return

        t = spec.thread
        if t is None:
            return

        timeout = spec.heartbeat_timeout or getattr(t, "heartbeat_timeout", self._default_timeout)

        if not t.is_alive():
            msg = f"{name}: thread is not alive"
            problems.append(msg)
            logger.warning(msg)
            if spec.auto_restart:
                logger.info("Watchdog restarting dead thread %s", name)
                self._registry.restart_thread(name)
            return

        hb = t.last_heartbeat
        if hb == 0.0:
            # Thread has not yet produced a heartbeat – probably still starting.
            return

        age = now - hb
        if age > timeout:
            msg = f"{name}: heartbeat stale (age={age:.3f}s > timeout={timeout:.3f}s)"
            problems.append(msg)
            logger.error(msg)
            if spec.auto_restart:
                logger.info("Watchdog restarting stalled thread %s", name)
                self._registry.restart_thread(name)


__all__ = ["WatchdogThread", "WatchdogData"]

