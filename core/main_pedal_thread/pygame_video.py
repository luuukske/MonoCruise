"""Bring pygame's video up so joystick events can be pumped. See README.md."""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from typing import Any

logger = logging.getLogger(__name__)

_DUMMY_DRIVER = "dummy"
_VIDEO_DOWN = "video system not initialized"


def video_ready(pg: Any) -> bool:
    """True when pygame.event.get() will not raise for a missing display."""
    try:
        return bool(pg.get_init() and pg.display.get_init())
    except Exception:
        return False


def ensure_pygame_video(
    pg: Any, environ: MutableMapping[str, str] | None = None
) -> bool:
    """Init pygame. True when the event pump is safe to call."""
    env = os.environ if environ is None else environ
    if video_ready(pg):
        _init_joystick(pg)
        return True
    passed, failed = _init_modules(pg)
    if video_ready(pg):
        _log_ready(pg, passed, failed)
        return True
    # pygame.init() counts a failed subsystem and does not raise, so the SDL
    # reason only surfaces from a direct display init. See the README.
    logger.warning(
        "pygame video init failed (%s passed, %s failed, SDL_VIDEODRIVER=%r)",
        passed,
        failed,
        env.get("SDL_VIDEODRIVER", ""),
    )
    _try_display_init(pg)
    if video_ready(pg):
        _log_ready(pg, passed, failed)
        return True
    env["SDL_VIDEODRIVER"] = _DUMMY_DRIVER
    logger.warning("retrying pygame video with the dummy driver")
    _try_display_init(pg)
    _init_joystick(pg)
    if video_ready(pg):
        _log_ready(pg, passed, failed)
        return True
    logger.warning(
        "pygame video still down after the dummy driver (%s passed, %s failed)",
        passed,
        failed,
    )
    return False


def publish_released(data: Any) -> None:
    """Publish released pedals and clear button holds."""
    with data._lock:
        data.gasval = 0.0
        data.brakeval = 0.0
        data.gas_output = 0.0
        data.brake_output = 0.0
        data.opdgasval = 0.0
        data.opdbrakeval = 0.0
        data.em_stop = False
        data.cc_start_held = False
        data.cc_inc_held = False
        data.cc_dec_held = False
        data.acc_dist_inc_held = False
        data.acc_dist_dec_held = False


class PygameVideo:
    """One pedal thread's video bring-up. A failure sticks for this instance."""

    def __init__(self) -> None:
        self.ok = False
        self._gave_up = False
        self._announced = False
        self._released = False
        self._pump_logged = False

    @property
    def failed(self) -> bool:
        return self._gave_up

    def start(self, pg: Any, environ: MutableMapping[str, str] | None = None) -> bool:
        if self._gave_up:
            return False
        self.ok = ensure_pygame_video(pg, environ)
        if not self.ok:
            self._gave_up = True
        return self.ok

    def read(self, pg: Any, environ: MutableMapping[str, str] | None = None) -> list:
        """Pending events. Empty when the pump cannot run."""
        if self._gave_up:
            return []
        if not self.ok or not video_ready(pg):
            if not self.start(pg, environ):
                return []
        try:
            return list(pg.event.get())
        except Exception as exc:
            if not self._pump_logged:
                self._pump_logged = True
                logger.warning("pygame event pump failed: %s", exc)
            if not video_ready(pg) or _VIDEO_DOWN in str(exc):
                self.ok = False
                self._gave_up = True
            return []

    def take_failure(self) -> bool:
        """True once after video has given up, for a single popup."""
        if not self._gave_up or self._announced:
            return False
        self._announced = True
        return True

    def release(self, pg: Any) -> None:
        """Close joysticks once, so a blind thread does not keep them open."""
        if not self._gave_up or self._released:
            return
        self._released = True
        try:
            joystick = pg.joystick
            if joystick.get_init():
                joystick.quit()
        except Exception as exc:
            logger.warning("pygame joystick release failed: %s", exc)


def _init_modules(pg: Any) -> tuple[int, int]:
    result = pg.init()
    _init_joystick(pg)
    if isinstance(result, tuple) and len(result) == 2:
        return int(result[0]), int(result[1])
    return 0, 0


def _init_joystick(pg: Any) -> None:
    joystick = pg.joystick
    if not joystick.get_init():
        joystick.init()


def _try_display_init(pg: Any) -> None:
    try:
        pg.display.init()
    except Exception as exc:
        logger.warning("pygame display init failed: %s", exc)


def _driver_name(pg: Any) -> str:
    try:
        name = pg.display.get_driver()
    except Exception:
        return ""
    return name or ""


def _log_ready(pg: Any, passed: int, failed: int) -> None:
    logger.info(
        "pygame ready (%s passed, %s failed, video %s)",
        passed,
        failed,
        _driver_name(pg),
    )
