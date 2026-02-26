from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from base_thread import BaseThread, SnapshotMixin
from registry import Registry


logger = logging.getLogger("monocruise.telemetry")


@dataclass
class TelemetryData(SnapshotMixin):
    """
    Public, typed view of game telemetry.

    In the real implementation this will be populated from the game
    process (shared memory, sockets, etc.).  For now we simulate a
    changing speed so that downstream threads can be exercised.
    """

    speed_kmh: float = 0.0
    brake_pressed: bool = False
    timestamp: float = 0.0


class TelemetryThread(BaseThread[TelemetryData]):
    """
    Sole reader of game data.

    - Writes only to its own TelemetryData instance.
    - Other threads read telemetry via:
        tele = registry.get_thread("telemetry")
        speed = tele.data.speed_kmh
    """

    def __init__(
        self,
        registry: Registry,
        *,
        name: str = "telemetry",
        loop_interval: float = 0.05,
    ) -> None:
        self._registry = registry
        super().__init__(name=name, loop_interval=loop_interval)

    def create_initial_data(self) -> TelemetryData:
        return TelemetryData()

    def on_startup(self) -> None:
        logger.info("TelemetryThread starting (simulated data source)")
        self._start_ts = time.monotonic()

    def loop(self) -> None:
        """
        Simulate some changing telemetry.

        Replace this body with real game integration – the public
        contract is that self.data is the authoritative source of
        telemetry for other threads.
        """
        now = time.monotonic()
        elapsed = now - self._start_ts

        # Simple repeatable speed curve: 0–120 km/h sine wave.
        speed = 60.0 * (1.0 + math.sin(elapsed / 5.0))
        brake = (elapsed % 15.0) > 12.0

        self.data.update(speed_kmh=speed, brake_pressed=brake, timestamp=now)

    def on_shutdown(self) -> None:
        logger.info("TelemetryThread stopping")


__all__ = ["TelemetryThread", "TelemetryData"]

