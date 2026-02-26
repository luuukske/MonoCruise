from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from base_thread import BaseThread, SnapshotMixin
from registry import Registry


logger = logging.getLogger("monocruise")


@dataclass
class TemplateData(SnapshotMixin):
    """
    Example per-thread dataclass.

    - Owners (this thread) may freely assign to fields.
    - Other threads read primitive fields directly, or call snapshot()
      when they need multiple fields to be consistent.
    """

    counter: int = 0
    last_update_ts: float = 0.0


class TemplateThread(BaseThread[TemplateData]):
    """
    Minimal example worker for coworkers to copy.

    Usage pattern:
    - Add your own typed dataclass fields to TemplateData.
    - Implement loop() with your I/O-bound work.
    - Use registry.get_thread(\"other\") to read other threads' data
      (e.g. telemetry.data.speed) and registry.publish() for events.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        name: str = "template",
        loop_interval: float = 0.5,
    ) -> None:
        self._registry = registry
        super().__init__(name=name, loop_interval=loop_interval)

    def create_initial_data(self) -> TemplateData:
        return TemplateData()

    # ------------------------------------------------------------------
    # BaseThread hooks
    # ------------------------------------------------------------------

    def on_startup(self) -> None:
        logger.info("TemplateThread starting")

    def loop(self) -> None:
        # Example: simple heartbeat-style work.
        new_counter = self.data.counter + 1
        now = time.monotonic()
        self.data.update(counter=new_counter, last_update_ts=now)

        # Example: publish an event occasionally.
        if new_counter % 10 == 0:
            self._registry.publish(
                "template:tick",
                payload={"counter": new_counter, "ts": now},
                sender=self.name,
            )

    def on_shutdown(self) -> None:
        logger.info("TemplateThread stopping")


__all__ = ["TemplateThread", "TemplateData"]

