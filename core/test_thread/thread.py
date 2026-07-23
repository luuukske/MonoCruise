"""Debug worker for error popup and watchdog testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import threading
import time

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry    import registry

logger = logging.getLogger(__name__)


@dataclass
class TestThreadData(ThreadData):
    pass


class TestThread(BaseThread):
    loop_interval = 0.05
    max_restarts  = 1

    def __init__(self) -> None:
        super().__init__(name="test_thread")
        self.data = TestThreadData()
        self._settings = None
        self.i = 0

    def setup(self) -> None:
        """Runs once before the loop. Raise to abort startup."""
        logger.debug("setup complete")

    def loop(self) -> None:
        """Loop body; no time.sleep (BaseThread paces)."""

        self.i += 1
        if self.i >= 100:
            while True:
                self.i = self.i + 1
                if self.i >= 100000000000000:
                    break

    def teardown(self) -> None:
        """Runs once after loop exits. Exceptions are suppressed by base."""
        logger.debug("teardown complete")

