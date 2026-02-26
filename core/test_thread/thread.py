"""
Test thread for debugging purposes and error popup testing.
"""

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
    max_restarts  = 2

    def __init__(self) -> None:
        super().__init__(name="test_thread")
        self.data = TestThreadData()
        self._settings = None
        self.i = 0

    def setup(self) -> None:
        """Runs once before the loop. Raise to abort startup."""
        logger.debug("setup complete")

    def loop(self) -> None:
        """
        Main work unit. Called every `loop_interval` seconds.
        Do NOT call time.sleep() here — the base class handles pacing.
        Raise any exception to trigger watchdog handling.
        """

        self.i += 1
        if self.i >= 100:
            raise Exception("test error")

    def teardown(self) -> None:
        """Runs once after loop exits. Exceptions are suppressed by base."""
        logger.debug("teardown complete")
