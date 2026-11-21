"""How the gas pedal's slope changes from gear to gear. See core/sending_thread/README.md.

`PedalCapacityTracker` learns the level (the anchor) and the zero-gas offset; this holds the
shape. With the SDK's gear ratios the slope goes as `(R / REF_TOTAL) ** exponent`, R the gear
x axle ratio; without them as `step ** (ANCHOR_GEAR - gear)`, one learned step per gear.
"""

from __future__ import annotations

import math
from typing import Sequence

RATIO_INIT: float = 1.27            # per-gear step until cross-gear samples settle it; also
                                    # the multiplier that seeds the anchor from the legacy scalar
RATIO_MIN: float = 1.05             # absolute bounds: outside is unphysical for any
RATIO_MAX: float = 1.45             # common truck transmission
RATIO_RATE_PER_S: float = 0.125     # log-space shape rate per second at gas=1.0, a
                                    # quarter of the anchor's as with the old alphas
ANCHOR_GEAR: int = 6                # mid-stack reference gear of the step shape
LEGACY_SEED_GEAR: int = 8           # legacy max_accel_ms2 was mostly learned in top
                                    # cruise gears; project from here when seeding

# The anchor is quoted at this gear x axle ratio, and the exponent starts at a force through it.
REF_TOTAL: float = 4.0
EXP_INIT: float = 1.0
EXP_MIN: float = 0.4
EXP_MAX: float = 1.3
# 1: the anchor is quoted at REF_TOTAL; 0 (older): at ANCHOR_GEAR, migrated once.
GEAR_MODEL_VERSION: int = 1


def total_ratios(forward: Sequence[float], differential: float) -> tuple[float, ...] | None:
    """Gear x axle ratio per forward gear, or None unless every one is a usable number."""
    if not forward or not (math.isfinite(differential) and differential > 0.0):
        return None
    totals = []
    for ratio in forward:
        try:
            r = float(ratio)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(r) and r > 0.0):
            return None
        totals.append(r * differential)
    return tuple(totals)


class GearShape:
    """The gas slope's shape over gears. Ratios that arrive unusable keep the last good set."""

    def __init__(self) -> None:
        self.step: float = RATIO_INIT
        self.exponent: float = EXP_INIT
        self.model: int = GEAR_MODEL_VERSION
        self.ratios: tuple[float, ...] | None = None
        self._input: tuple[tuple[float, ...], float] | None = None
        self._saved: tuple[float, float] = (self.step, self.exponent)

    def load(self, step: float, exponent: float, model: int) -> None:
        self.step = min(max(step if step > 1.0 else RATIO_INIT, RATIO_MIN), RATIO_MAX)
        self.exponent = min(max(exponent if exponent > 0.0 else EXP_INIT, EXP_MIN), EXP_MAX)
        self.model = model
        self._saved = (self.step, self.exponent)

    @property
    def needs_requote(self) -> bool:
        """An anchor quoted at a gear number while ratios are known."""
        return self.ratios is not None and self.model != GEAR_MODEL_VERSION

    def set_ratios(self, forward: Sequence[float], differential: float) -> None:
        try:
            key = (tuple(forward), float(differential))
        except (TypeError, ValueError):
            return
        if key == self._input:
            return
        self._input = key
        totals = total_ratios(key[0], key[1])
        if totals is not None:
            self.ratios = totals

    def total_ratio(self, gear: int) -> float | None:
        ratios = self.ratios
        if ratios is None or gear <= 0:
            return None
        return ratios[min(gear, len(ratios)) - 1]

    def lever(self, gear: int) -> tuple[float, float, bool]:
        """(lever, log of the shape factor, from the SDK's ratios) for *gear*."""
        total = self.total_ratio(gear) if self.model == GEAR_MODEL_VERSION else None
        if total is None:
            x = float(ANCHOR_GEAR - gear)
            return x, x * math.log(self.step), False
        x = math.log(total / REF_TOTAL)
        return x, self.exponent * x, True

    def factor(self, gear: int) -> float:
        """Slope in *gear* over the anchor."""
        _, log_shape, from_ratios = self.lever(gear)
        return math.exp(log_shape) if from_ratios else self.step ** (ANCHOR_GEAR - gear)

    def learn(self, x: float, from_ratios: bool, residual: float, weight: float, dt: float) -> None:
        # Normalized by the lever arm: unnormalized, top gear learned it 8x faster and
        # walked the (anchor, shape) pair along its degenerate direction. See README.
        delta = min(RATIO_RATE_PER_S * weight * dt, 0.5) * x * residual / (1.0 + x * x)
        if from_ratios:
            self.exponent = min(max(self.exponent + delta, EXP_MIN), EXP_MAX)
        else:
            self.step = min(max(math.exp(math.log(self.step) + delta), RATIO_MIN), RATIO_MAX)

    def requote(self, anchor: float) -> tuple[float, int]:
        """(anchor quoted at REF_TOTAL, gear kept) so the anchor gear keeps its slope."""
        ratios = self.ratios or (REF_TOTAL,)
        gear = min(ANCHOR_GEAR, len(ratios))
        self.exponent = EXP_INIT
        self.model = GEAR_MODEL_VERSION
        slope = anchor * self.step ** (ANCHOR_GEAR - gear)
        return slope * (REF_TOTAL / ratios[gear - 1]) ** self.exponent, gear

    def seed_anchor(self, full_pedal_ms2: float) -> float:
        """Anchor from the legacy full-pedal scalar, learned mostly around LEGACY_SEED_GEAR."""
        return full_pedal_ms2 * self.step ** (LEGACY_SEED_GEAR - ANCHOR_GEAR)

    def drifted(self, threshold: float) -> bool:
        step, exponent = self._saved
        return (abs(self.step - step) / max(step, 0.01) >= threshold
                or abs(self.exponent - exponent) >= threshold)

    def mark_saved(self) -> None:
        self._saved = (self.step, self.exponent)

    def persisted(self) -> dict[str, float | int]:
        return {
            "pedal_capacity_accel_ratio_step": round(self.step, 4),
            "pedal_capacity_accel_ratio_exponent": round(self.exponent, 4),
            "pedal_capacity_accel_gear_model": self.model,
        }
