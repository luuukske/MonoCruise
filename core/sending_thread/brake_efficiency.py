"""Tracks braking efficiency during cruise control to detect performance degradation. See `core/sending_thread/README.md`."""

from __future__ import annotations

import logging
import math
import time

from core.settings import Settings

from .accel_to_pedals import baseline_brake_ms2, brake_curve_fraction

logger = logging.getLogger(__name__)

_MIN_SAMPLE_SPEED_MS: float = 5.0      # ~18 km/h
_BRAKE_HIGH_THRESHOLD: float = 0.70    # only high-brake samples for baseline/EMA max
_BRAKE_OUTPUT_FLOOR: float = 0.15      # decel-per-pedal normalization floor
_MIN_DECEL_MS2: float = 0.5            # filter near-zero decel noise
_MAX_SLOPE_RAD: float = 0.03           # ~1.7°: skip sloped road to avoid gravity bias
_BASELINE_REQUIRED: int = 15           # high-brake samples before baseline is locked
_WARN_COOLDOWN_S: float = 30.0         # minimum seconds between popup warnings

_REF_WHEELS: int = 12
_REF_MASS_KG: float = 17000.0


def nominal_max_brake_decel_ms2(wheels_on_ground: int, mass_kg: float) -> float:
    """Expected deceleration at brake=1.0 for this rig. See `core/sending_thread/README.md`."""
    w = wheels_on_ground if wheels_on_ground > 0 else _REF_WHEELS
    m = float(mass_kg) if mass_kg > 0.0 else _REF_MASS_KG
    # Asymptote of the fitted brake curve, so scale by what the pedal reaches.
    return baseline_brake_ms2(m, w > 6, w) * brake_curve_fraction(1.0)


class BrakeEfficiencyTracker:
    """Estimates vehicle braking performance and detects degradation via EMA. See `core/sending_thread/README.md`."""

    def __init__(self) -> None:
        self._ema_max_decel_ms2: float = 0.0
        self._baseline_max_decel_ms2: float = 0.0
        self._baseline_samples: list[float] = []
        self._baseline_locked: bool = False
        self._efficiency_ratio: float = 1.0
        self._last_warn_mono: float = 0.0

    @property
    def efficiency_ratio(self) -> float:
        """Current brake efficiency ratio (1.0 = nominal, <1.0 = degraded)."""
        return self._efficiency_ratio

    @property
    def baseline_locked(self) -> bool:
        return self._baseline_locked

    def reset(self) -> None:
        """Reset all state (e.g. after a long stop or conditions change)."""
        self._ema_max_decel_ms2 = 0.0
        self._baseline_max_decel_ms2 = 0.0
        self._baseline_samples = []
        self._baseline_locked = False
        self._efficiency_ratio = 1.0
        logger.debug("brake_efficiency_tracker: state reset")

    def update(
        self,
        brake_output: float,
        measured_decel_ms2: float,
        speed_ms: float,
        slope_rad: float = 0.0,
        wheels_on_ground: int = 0,
        mass_kg: float = 0.0,
    ) -> None:
        """Feed one braking sample. Call only when cruise is actively commanding brake. See `core/sending_thread/README.md`."""
        if not Settings.brake_efficiency_learning:
            self._efficiency_ratio = 1.0
            return

        if speed_ms < _MIN_SAMPLE_SPEED_MS:
            return
        if measured_decel_ms2 < _MIN_DECEL_MS2:
            return
        if brake_output < _BRAKE_HIGH_THRESHOLD:
            return
        if abs(slope_rad) > _MAX_SLOPE_RAD:
            return

        alpha = max(1e-4, min(1.0, float(Settings.brake_efficiency_alpha)))
        nominal_max = nominal_max_brake_decel_ms2(wheels_on_ground, mass_kg)
        # Pedal to decel is the fitted curve, not a straight line: at 0.7 pedal
        # the truck already makes 84% of full, so linear read grip 20% high.
        pedal = max(float(brake_output), _BRAKE_OUTPUT_FLOOR)
        expected_decel = (
            nominal_max * brake_curve_fraction(pedal) / brake_curve_fraction(1.0)
        )
        grip = measured_decel_ms2 / expected_decel
        if not math.isfinite(grip) or grip <= 0.0:
            return

        if self._ema_max_decel_ms2 <= 0.0:
            self._ema_max_decel_ms2 = grip
        else:
            self._ema_max_decel_ms2 = (
                (1.0 - alpha) * self._ema_max_decel_ms2 + alpha * grip
            )

        if not self._baseline_locked:
            self._baseline_samples.append(grip)
            if len(self._baseline_samples) >= _BASELINE_REQUIRED:
                sorted_samples = sorted(self._baseline_samples)
                top_half = sorted_samples[_BASELINE_REQUIRED // 2:]
                self._baseline_max_decel_ms2 = sum(top_half) / len(top_half)
                self._baseline_locked = True
                logger.info(
                    "brake_efficiency: baseline locked at %.2f× nominal decel (%d samples)",
                    self._baseline_max_decel_ms2,
                    len(self._baseline_samples),
                )

        if self._baseline_locked and self._baseline_max_decel_ms2 > 0.0:
            self._efficiency_ratio = max(
                0.1,
                min(1.0, self._ema_max_decel_ms2 / self._baseline_max_decel_ms2),
            )

    def check_warnings(self) -> None:
        """Emit popup warning if efficiency has degraded below the configured threshold."""
        if not self._baseline_locked:
            return

        warn_ratio = float(Settings.brake_efficiency_warn_ratio)
        now = time.monotonic()
        if (
            self._efficiency_ratio < warn_ratio
            and now - self._last_warn_mono > _WARN_COOLDOWN_S
        ):
            self._last_warn_mono = now
            logger.warning(
                "Braking efficiency at %.0f%%. Check road conditions or tire wear.",
                self._efficiency_ratio * 100.0,
                extra={"popup": True},
            )
            logger.debug(
                "brake_efficiency: ratio=%.3f ema_norm=%.2f baseline_norm=%.2f "
                "warn_threshold=%.2f",
                self._efficiency_ratio,
                self._ema_max_decel_ms2,
                self._baseline_max_decel_ms2,
                warn_ratio,
            )

