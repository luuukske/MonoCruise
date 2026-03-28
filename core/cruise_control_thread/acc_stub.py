"""
Adaptive cruise layout stub — real lead-vehicle logic will plug in later.

When ACC is enabled, the cruise thread will clamp commanded acceleration using
:meth:`AdaptiveCruiseControllerStub.accel_cap_ms2`. The stub returns a very
large cap so behaviour matches plain cruise until radar / gap control exists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ACConfig:
    """Tunable placeholders for future ACC."""

    max_accel_cap_ms2: float = 1.0e6


class AdaptiveCruiseControllerStub:
    def __init__(self, config: ACConfig | None = None) -> None:
        self.config = config or ACConfig()

    def accel_cap_ms2(self) -> float:
        """Upper bound on longitudinal accel (m/s²) from ACC; stub = no effective limit."""
        return float(self.config.max_accel_cap_ms2)
