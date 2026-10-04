"""
TEMPLATE THREAD: copy this directory and rename it for your worker.

Convention:
  core/<feature_name>/thread.py   ← worker lives here
  core/<feature_name>/__init__.py ← keep empty

Steps:
  1. Copy core/example_thread/ → core/<your_name>/
  2. Rename MyThread / MyThreadData to match your feature.
  3. Implement setup(), loop(), teardown().
  4. Register + start in main.py.
  5. Other threads access data via:
       registry.get_thread("my_thread").data.my_field
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import threading
import time

from core.settings import Settings

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry    import registry

logger = logging.getLogger(__name__)

just_enabled = False

# Typed data container: other threads read fields directly (GIL-safe)

@dataclass
class AutoSpeedLimitData(ThreadData):
    # Add your typed fields here:
    #value: float = 0.0
    #label: str   = ""

    # For consistent multi-field reads, expose snapshot():
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class AutoSpeedLimit(BaseThread):
    loop_interval = 0.5   # seconds: rate-limit your loop here
    max_restarts  = 5

    def __init__(self) -> None:
        super().__init__(name="auto_speed_limit")
        self.data = AutoSpeedLimitData()
        self._settings = None   # inject via constructor or module import

    # lifecycle

    def setup(self) -> None:
        """Runs once before the loop. Raise to abort startup."""
        pass

    def loop(self) -> None:
        """
        Main work unit. Called every `loop_interval` seconds.
        Do NOT call time.sleep() here: the base class handles pacing.
        Raise any exception to trigger watchdog handling.
        """
        global just_enabled
        if Settings.autospeedlimit_variable:
            just_enabled = True
            speedlimit = registry.get_thread("telemetry_thread").data.speedLimit * 3.6
            if speedlimit == 0:
                speedlimit = None
            else:
                speedlimit = int(speedlimit)
            Settings.save({"global_speed_limit_kmh": speedlimit})
        elif not Settings.autospeedlimit_variable:
            if just_enabled: # The global speed limit will not go back to the value set in the box after turning this checkbox off
                Settings.save({"global_speed_limit_kmh": None}) # imo this is better (less confusing) than the global limit staying at the latest ingame limit
                just_enabled = False


    def teardown(self) -> None:
        """Runs once after loop exits. Exceptions are suppressed by base."""
        pass

