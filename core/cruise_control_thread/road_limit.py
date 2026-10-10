"""Road speed limit from the SDK: optional global cap and set-speed follow. See core/cruise_control_thread/README.md."""

from __future__ import annotations

import logging
import math

from core.longitudinal.cc import CruiseController
from core.settings import Settings
from core.speed_units import display_from_ms, kmh_from_display, quantize_speed_kmh

logger = logging.getLogger(__name__)

# The SDK reports 0 where the road has no posted limit.
_NO_LIMIT_BELOW_MS = 0.5


def road_limit_kmh(speed_limit_ms: float) -> float | None:
    """Posted limit in km/h, rounded in the driver's unit; None where the road has none."""
    try:
        v = float(speed_limit_ms)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < _NO_LIMIT_BELOW_MS:
        return None
    return kmh_from_display(display_from_ms(v))


class RoadLimit:
    """Feeds the posted limit to the CC clamp and, when enabled, to the set speed."""

    def __init__(self) -> None:
        self._followed_kmh: float | None = None

    def step(self, cc: CruiseController, speed_limit_ms: float) -> None:
        limit = road_limit_kmh(speed_limit_ms)
        cc.set_road_limit_kmh(limit if Settings.autospeedlimit_variable else None)
        if not Settings.autospeedtarget_variable:
            self._followed_kmh = None
            return
        # No posted limit keeps the set speed: disabling CC there would silently drop ACC.
        if limit is None or limit == self._followed_kmh:
            return
        self._followed_kmh = limit
        cc.set_target_kmh(quantize_speed_kmh(limit / 3.6, cc.global_limit_kmh))
        logger.debug("set speed follows road limit: %.1f km/h", cc.target_speed_kmh)
