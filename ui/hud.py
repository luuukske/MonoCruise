from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger("monocruise.ui.hud")


@dataclass
class HUDState:
    cruise_engaged: bool = False
    cruise_target_speed_kmh: float = 0.0


class HUD:
    """
    Plain, non-threaded HUD facade.

    The concrete rendering technology (PySide6 overlays, etc.) can be
    plugged in behind this API later.  For now it only logs changes so
    that the threading framework can be exercised without a GUI.
    """

    def __init__(self) -> None:
        self._state = HUDState()
        logger.info("HUD initialised")

    # ------------------------------------------------------------------
    # Public API owned by UiThread
    # ------------------------------------------------------------------

    def set_cruise_state(self, engaged: bool, target_speed_kmh: float) -> None:
        self._state.cruise_engaged = engaged
        self._state.cruise_target_speed_kmh = target_speed_kmh
        logger.info(
            "HUD cruise state: engaged=%s target=%.0f km/h",
            engaged,
            target_speed_kmh,
        )

    def shutdown(self) -> None:
        logger.info("HUD shutdown")


__all__ = ["HUD", "HUDState"]

