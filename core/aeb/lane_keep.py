"""Lane keep: a mover following its own map lane, clear of ego's track, is not braked for.

The last stage, so it only judges what every other stage let through. No map, no
verdict. See ``core/aeb/README.md``, "Lane keep".
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.aeb.calibration import AEBCalibration
from core.aeb.filters import _PASS, FilterResult, _suppress

if TYPE_CHECKING:
    from core.aeb.filters import FilterContext


class LaneKeepFilter:
    name = "LaneKeepFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: "FilterContext") -> FilterResult:
        keeps = ctx.lane_keep_fn
        if keeps is None or not self._cal.lane_keep_enabled:
            return _PASS
        # A body AEB already brakes for is never let go on the map's word.
        if ctx.v.id in ctx.latched_threat_ids or getattr(ctx.v, "crash_confirmed", False):
            return _PASS
        return _suppress(self.name) if keeps(ctx.d_miss_rate) else _PASS
