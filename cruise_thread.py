from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from base_thread import BaseThread, SnapshotMixin
from registry import Registry
from telemetry_thread import TelemetryThread, TelemetryData


logger = logging.getLogger("monocruise.cruise")


@dataclass
class CruiseData(SnapshotMixin):
    """
    Public state of the cruise-control logic.
    """

    engaged: bool = False
    target_speed_kmh: float = 0.0
    last_update_ts: float = 0.0


class CruiseThread(BaseThread[CruiseData]):
    """
    Reads telemetry.data and runs control logic, publishing events such
    as 'cruise:engaged' and 'cruise:disengaged'.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        name: str = "cruise",
        loop_interval: float = 0.05,
    ) -> None:
        self._registry = registry
        super().__init__(name=name, loop_interval=loop_interval)

    def create_initial_data(self) -> CruiseData:
        return CruiseData()

    def on_startup(self) -> None:
        logger.info("CruiseThread starting")

    def loop(self) -> None:
        telemetry = self._registry.get_thread("telemetry")
        if not isinstance(telemetry, TelemetryThread):
            # Telemetry not yet available – nothing to do this round.
            return

        tele_data: TelemetryData = telemetry.data

        # Single-field reads are safe without extra locking.
        speed = tele_data.speed_kmh
        brake = tele_data.brake_pressed

        cur = self.data
        now = time.monotonic()

        engaged = cur.engaged
        target = cur.target_speed_kmh

        # Very simple example logic:
        # - Engage when speed crosses 50 km/h and brake not pressed.
        # - Disengage whenever brake is pressed or speed drops below 20 km/h.
        if not engaged and speed >= 50.0 and not brake:
            engaged = True
            target = round(speed)
            logger.info("Cruise engaged at %.1f km/h", speed)
            self._registry.publish(
                "cruise:engaged",
                payload={"target_speed_kmh": target},
                sender=self.name,
            )
        elif engaged and (brake or speed < 20.0):
            engaged = False
            logger.info("Cruise disengaged (speed=%.1f, brake=%s)", speed, brake)
            self._registry.publish(
                "cruise:disengaged",
                payload={"speed_kmh": speed, "brake_pressed": brake},
                sender=self.name,
            )

        self.data.update(
            engaged=engaged,
            target_speed_kmh=target,
            last_update_ts=now,
        )

    def on_shutdown(self) -> None:
        logger.info("CruiseThread stopping")


__all__ = ["CruiseThread", "CruiseData"]

