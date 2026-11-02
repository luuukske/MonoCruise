"""What a path source asks the LA to follow, on top of the lane it keeps.

The lane follower asks for nothing extra. A later source (collision avoidance) can ask for a lateral offset from
the lane centre and extra path curvature; path tracking and the wheel actuator stay the same. See README.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WantedPath:
    """``offset_m`` right of the lane centre at the rear axle, ``kappa`` extra rightward curvature, from ``source``."""

    offset_m: float = 0.0
    kappa: float = 0.0
    source: str = "lane"


LANE = WantedPath()
