from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from base_thread import BaseThread, SnapshotMixin
from registry import Registry


logger = logging.getLogger("monocruise.updater")


@dataclass
class UpdaterData(SnapshotMixin):
    """
    State of the update-check subsystem.

    The updater thread is fully isolated: it does not depend on any
    other thread.  It publishes high-level events like
    'updater:update_available' and then idles.
    """

    last_checked_ts: float = 0.0
    last_result_ok: bool | None = None
    update_available: bool = False
    latest_version: str | None = None


class UpdaterThread(BaseThread[UpdaterData]):
    """
    Periodically checks for updates and publishes the result.

    Replace the _check_for_update() stub with the real network call
    (e.g. GitHub API).  Any exceptions stay inside the thread and are
    logged via the shared logger.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        name: str = "updater",
        loop_interval: float = 60.0,
    ) -> None:
        self._registry = registry
        super().__init__(name=name, loop_interval=loop_interval)

    def create_initial_data(self) -> UpdaterData:
        return UpdaterData()

    def on_startup(self) -> None:
        logger.info("UpdaterThread starting")

    def loop(self) -> None:
        now = time.monotonic()
        ok, available, version = self._check_for_update()

        self.data.update(
            last_checked_ts=now,
            last_result_ok=ok,
            update_available=available,
            latest_version=version,
        )

        if ok and available:
            logger.info("Update available: %s", version)
            self._registry.publish(
                "updater:update_available",
                payload={"version": version},
                sender=self.name,
            )

    def on_shutdown(self) -> None:
        logger.info("UpdaterThread stopping")

    # ------------------------------------------------------------------
    # Stub I/O – replace with real updater implementation
    # ------------------------------------------------------------------

    def _check_for_update(self) -> tuple[bool, bool, str | None]:
        """
        Return (ok, update_available, version).

        This stub simply fakes an update every 5 checks.
        """
        d = self.data
        count = (d.last_checked_ts and int(d.last_checked_ts)) or 0
        # Fake: every 5th check yields a new "version".
        fake_version = f"0.0.{count + 1}"
        available = (count % 5) == 0
        ok = True
        return ok, available, fake_version if available else d.latest_version


__all__ = ["UpdaterThread", "UpdaterData"]

