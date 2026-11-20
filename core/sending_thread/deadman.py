"""Pedal dead man: when the sending thread stops writing, release the gas and pass the driver's brake.

The game keeps the last pedals the sending thread wrote. On 2026-11-20 it froze four times on a
stalled disk with the driver's gas at 95 % and no way to brake (README, pedal dead man). This
thread does no file I/O and no blocking logging, so it keeps running through such a stall.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from core.scs_profile.intensity import apply_brake_intensity, last_brake_intensity
from core.settings import Settings
from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry

from .scscontroller import SCSController

logger = logging.getLogger(__name__)

STALE_S: float = 0.30            # sending heartbeat older than this, in 3 of its own loops at least
_STALE_LOOPS: float = 3.0
_OPEN_RETRY_S: float = 5.0

FRESH, STALE, IDLE = "fresh", "stale", "idle"


@dataclass
class PedalDeadmanData(ThreadData):
    engaged: bool = False
    engagements: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


def sending_state(sending, connected: bool, seen_fresh: bool, now: float) -> str:
    """FRESH while the sending thread writes, STALE once it stopped after having written, else IDLE."""
    if not connected:
        return IDLE
    if sending is not None:
        heartbeat = float(getattr(sending, "heartbeat_at", 0.0) or 0.0)
        interval = float(getattr(sending, "loop_interval", 0.0) or 0.0)
        limit = max(STALE_S, _STALE_LOOPS * interval)
        alive = bool(sending.is_alive()) and bool(getattr(sending, "running", False))
        if alive and heartbeat > 0.0 and now - heartbeat <= limit:
            return FRESH
    return STALE if seen_fresh else IDLE


def driver_brake_sent(brake_output: float, brakeval: float, em_stop: bool, brake_exp: float,
                      intensity: float | None) -> float:
    """The driver's pedal as the sending thread would send it, without its hold, creep or AEB layers."""
    try:
        logical = max(float(brake_output), max(float(brakeval), 0.0) ** float(brake_exp))
    except (TypeError, ValueError, OverflowError):
        logical = max(float(brake_output or 0.0), float(brakeval or 0.0))
    return apply_brake_intensity(logical, intensity, full_authority=bool(em_stop))


class PedalDeadmanThread(BaseThread):
    loop_interval = 0.02
    max_restarts = 5

    def __init__(self) -> None:
        super().__init__(name="pedal_deadman")
        self.data = PedalDeadmanData()
        self._controller: SCSController | None = None
        self._open_retry_at = 0.0
        self._seen_fresh = False
        self._held_brake = 0.0
        self._engaged_at: float | None = None

    def setup(self) -> None:
        logger.debug("pedal dead man ready (stale after %.2f s)", STALE_S)

    def loop(self) -> None:
        if not self.running:
            return
        now = time.monotonic()
        sending = self._sibling("sending_thread")
        state = sending_state(sending, self._connected(), self._seen_fresh, now)
        if state == IDLE:
            self._release(now)
            return
        controller = self._open(now)
        if controller is None:
            return
        if state == FRESH:
            self._seen_fresh = True
            self._release(now)
            try:
                self._held_brake = float(controller.abackward)
            except (AttributeError, OSError, TypeError, ValueError):
                self._held_brake = 0.0
            return
        self._take_over(controller, sending, now)

    def _take_over(self, controller: SCSController, sending, now: float) -> None:
        if self._engaged_at is None:
            self._engaged_at = now
            with self.data._lock:
                self.data.engaged = True
                self.data.engagements += 1
            logger.warning("pedal output stalled: gas released, driver brake passed through")
        # A frozen sending thread may resume with its hold or AEB brake, so keep that; a dead one never will.
        held = self._held_brake if sending is not None and sending.is_alive() else 0.0
        try:
            controller.aforward = 0.0
            controller.abackward = float(max(held, self._driver_brake()))
        except (AttributeError, OSError, TypeError, ValueError):
            logger.debug("dead man write failed", exc_info=True)
            self._close()

    def _driver_brake(self) -> float:
        pedal = self._sibling("main_pedal_thread")
        if pedal is None:
            return 0.0
        try:
            with pedal.data._lock:
                brake_output = pedal.data.brake_output
                brakeval = pedal.data.brakeval
                em_stop = pedal.data.em_stop
        except (AttributeError, RuntimeError):
            return 0.0
        brake_exp = Settings.brake_exponent_variable or 1.0
        return driver_brake_sent(brake_output, brakeval, em_stop, brake_exp, last_brake_intensity())

    def _release(self, now: float) -> None:
        if self._engaged_at is None:
            return
        logger.info("pedal output back after %.2f s", now - self._engaged_at)
        self._engaged_at = None
        with self.data._lock:
            self.data.engaged = False

    @staticmethod
    def _sibling(name: str):
        try:
            return registry.get_thread(name)
        except KeyError:
            return None

    def _connected(self) -> bool:
        tel = self._sibling("telemetry_thread")
        if tel is None:
            return False
        try:
            return bool(tel.is_alive()) and bool(tel.data.is_connected)
        except AttributeError:
            return False

    def _open(self, now: float) -> SCSController | None:
        if self._controller is not None:
            return self._controller
        if now < self._open_retry_at:
            return None
        try:
            self._controller = SCSController()
        except Exception:
            self._open_retry_at = now + _OPEN_RETRY_S
            logger.debug("dead man could not open the controls block", exc_info=True)
        return self._controller

    def _close(self) -> None:
        if self._controller is not None:
            try:
                self._controller.close()
            except Exception:
                pass
        self._controller = None

    def teardown(self) -> None:
        self._close()
        logger.debug("pedal dead man teardown complete")
