"""
Tracks estimated maximum brake deceleration and gas acceleration.

Replaces brake_efficiency.py with a simpler, always-on system:

  Brake learning
  - Samples whenever any brake pedal is applied (user, cruise, or any future source).
  - Each sample is weighted by brake_output² so light braking has tiny influence
    while heavy braking drives the estimate strongly.
  - Underperformance (measured decel < expected) drops the estimate 3× faster
    than overperformance rises it — safety bias for emergency stops.
  - Gravity and rolling resistance are canceled via road_load_ms2 before sampling.

  Gas learning
  - Same approach for the gas pedal / max acceleration.
  - Skips samples for 0.5 s after the clutch was last pressed to avoid
    contamination from gear-change transients.
  - Gravity and rolling resistance are canceled via road_load_ms2 before sampling.

  Both estimates are persisted to settings.json so the next session starts
  from the last known good value instead of the cold baseline.
"""

from __future__ import annotations

import logging
import math
import time

from core.settings import Settings

logger = logging.getLogger(__name__)

_MIN_BRAKE_SPEED_MS: float = 5.0    # ~18 km/h
_MIN_ACCEL_SPEED_MS: float = 3.0    # ~11 km/h
_BRAKE_PEDAL_FLOOR: float = 0.05    # skip samples below this pedal level
_ACCEL_PEDAL_FLOOR: float = 0.05
_MIN_DECEL_MS2: float = 0.3         # ignore near-zero decel (coasting / noise)
_MIN_ACCEL_MS2: float = 0.2         # ignore near-zero accel
_MAX_SLOPE_RAD: float = 0.15        # ~8.6° — skip extreme slopes (sensor uncertainty)
_WEIGHT_POWER: float = 3.0          # alpha ∝ pedal^4 — small inputs almost ignored, full pedal dominates
_BRAKE_BASE_ALPHA: float = 0.15     # EMA alpha at full brake pedal, no underperformance
_ACCEL_BASE_ALPHA: float = 0.08     # EMA alpha at full gas pedal, no underperformance
_UNDERPERFORM_MULT: float = 2.0     # drop estimate this much faster when below expectation
_CLUTCH_GUARD_S: float = 0.5        # seconds after clutch to skip gas learning
_CLUTCH_ACTIVE_THRESHOLD: float = 0.05
_SAVE_THRESHOLD: float = 0.1        # save when drift exceeds 10% of saved value
_SAVE_COOLDOWN_S: float = 30.0      # min seconds between successive writes
_ESTIMATE_LOWER_BOUND: float = 0.35 # fraction of baseline — hard floor
_ESTIMATE_UPPER_BOUND: float = 2.0  # fraction of baseline — hard ceiling


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PedalCapacityTracker:
    """
    Estimates vehicle max brake deceleration and max gas acceleration via
    weighted EMA, always active whenever any pedal is applied.

    Gravity and rolling resistance are canceled from each sample using
    road_load_ms2 (= slope_accel + rolling_accel, positive = uphill forward).

    Reads live from Settings:
      pedal_capacity_max_brake_ms2  — persisted brake estimate (0 = use baseline)
      pedal_capacity_max_accel_ms2  — persisted accel estimate (0 = use baseline)
    """

    def __init__(self) -> None:
        self._max_brake_ms2: float = 0.0   # 0 = not yet initialised
        self._max_accel_ms2: float = 0.0
        self._saved_brake: float = 0.0
        self._saved_accel: float = 0.0
        self._last_save_mono: float = 0.0
        self._last_clutch_mono: float = -math.inf

    @property
    def max_brake_ms2(self) -> float:
        """Current best estimate of max deceleration at brake=1.0 (m/s²)."""
        return self._max_brake_ms2

    @property
    def max_accel_ms2(self) -> float:
        """Current best estimate of max acceleration at gas=1.0 (m/s²)."""
        return self._max_accel_ms2

    def load_persisted(self, baseline_brake: float, baseline_accel: float) -> None:
        """Seed estimates from persisted settings at startup.

        Args:
            baseline_brake: Fallback baseline if no persisted value exists (m/s²).
            baseline_accel: Fallback baseline if no persisted value exists (m/s²).
        """
        b = _safe_float(Settings.pedal_capacity_max_brake_ms2)
        a = _safe_float(Settings.pedal_capacity_max_accel_ms2)
        self._max_brake_ms2 = b if b > 0.0 else baseline_brake
        self._max_accel_ms2 = a if a > 0.0 else baseline_accel
        self._saved_brake = self._max_brake_ms2
        self._saved_accel = self._max_accel_ms2
        logger.debug(
            "pedal_capacity loaded: brake=%.2f m/s² accel=%.2f m/s²",
            self._max_brake_ms2,
            self._max_accel_ms2,
        )

    def update_brake(
        self,
        brake_output: float,
        measured_decel_ms2: float,
        speed_ms: float,
        slope_rad: float,
        baseline_ms2: float,
        road_load_ms2: float = 0.0,
    ) -> None:
        """Feed one braking sample.

        Call whenever any brake pedal is applied regardless of source.

        Args:
            brake_output: Actual brake pedal sent to the game [0–1].
            measured_decel_ms2: Positive measured deceleration (m/s²).
            speed_ms: Current speed (m/s).
            slope_rad: Road pitch (rad, positive = uphill) — used only for slope filter.
            baseline_ms2: Baseline max decel for clamping bounds.
            road_load_ms2: slope_accel + rolling_accel (positive = uphill forward).
                           Subtracted from measured_decel to isolate pure brake force.
        """
        if self._max_brake_ms2 <= 0.0:
            self._max_brake_ms2 = baseline_ms2

        if speed_ms < _MIN_BRAKE_SPEED_MS:
            return
        if brake_output < _BRAKE_PEDAL_FLOOR:
            return
        if abs(slope_rad) > _MAX_SLOPE_RAD:
            return

        corrected_decel = measured_decel_ms2 - road_load_ms2
        if corrected_decel < _MIN_DECEL_MS2:
            return

        candidate = corrected_decel / max(brake_output, _BRAKE_PEDAL_FLOOR)

        weight = brake_output ** _WEIGHT_POWER
        alpha = _BRAKE_BASE_ALPHA * weight
        if candidate < self._max_brake_ms2:
            alpha *= _UNDERPERFORM_MULT
        alpha = min(alpha, 1.0)

        self._max_brake_ms2 += alpha * (candidate - self._max_brake_ms2)
        self._max_brake_ms2 = _clamp(
            self._max_brake_ms2,
            baseline_ms2 * _ESTIMATE_LOWER_BOUND,
            baseline_ms2 * _ESTIMATE_UPPER_BOUND,
        )
        self._maybe_save(time.monotonic())

    def update_accel(
        self,
        gas_output: float,
        measured_accel_ms2: float,
        speed_ms: float,
        slope_rad: float,
        baseline_ms2: float,
        game_clutch: float,
        road_load_ms2: float = 0.0,
    ) -> None:
        """Feed one acceleration sample.

        Skips learning for 0.5 s after the clutch was last pressed.

        Args:
            gas_output: Actual gas pedal sent to the game [0–1].
            measured_accel_ms2: Positive measured acceleration (m/s²).
            speed_ms: Current speed (m/s).
            slope_rad: Road pitch (rad, positive = uphill) — used only for slope filter.
            baseline_ms2: Mass/config adjusted baseline max accel for bounding.
            game_clutch: Current clutch position [0–1].
            road_load_ms2: slope_accel + rolling_accel (positive = uphill forward).
                           Added to measured_accel to recover pure engine contribution.
        """
        now = time.monotonic()

        if game_clutch > _CLUTCH_ACTIVE_THRESHOLD:
            self._last_clutch_mono = now
        if now - self._last_clutch_mono < _CLUTCH_GUARD_S:
            return

        if self._max_accel_ms2 <= 0.0:
            self._max_accel_ms2 = baseline_ms2

        if speed_ms < _MIN_ACCEL_SPEED_MS:
            return
        if gas_output < _ACCEL_PEDAL_FLOOR:
            return
        if abs(slope_rad) > _MAX_SLOPE_RAD:
            return

        corrected_accel = measured_accel_ms2 + road_load_ms2
        if corrected_accel < _MIN_ACCEL_MS2:
            return

        candidate = corrected_accel / max(gas_output, _ACCEL_PEDAL_FLOOR)

        weight = gas_output ** _WEIGHT_POWER
        alpha = _ACCEL_BASE_ALPHA * weight
        if candidate < self._max_accel_ms2:
            alpha *= _UNDERPERFORM_MULT
        alpha = min(alpha, 1.0)

        self._max_accel_ms2 += alpha * (candidate - self._max_accel_ms2)
        self._max_accel_ms2 = _clamp(
            self._max_accel_ms2,
            baseline_ms2 * _ESTIMATE_LOWER_BOUND,
            baseline_ms2 * _ESTIMATE_UPPER_BOUND,
        )
        self._maybe_save(now)

    def _maybe_save(self, now: float) -> None:
        if now - self._last_save_mono < _SAVE_COOLDOWN_S:
            return
        brake_drift = abs(self._max_brake_ms2 - self._saved_brake) / max(self._saved_brake, 0.01)
        accel_drift = abs(self._max_accel_ms2 - self._saved_accel) / max(self._saved_accel, 0.01)
        if brake_drift < _SAVE_THRESHOLD and accel_drift < _SAVE_THRESHOLD:
            return
        try:
            Settings.save(values={
                "pedal_capacity_max_brake_ms2": round(self._max_brake_ms2, 3),
                "pedal_capacity_max_accel_ms2": round(self._max_accel_ms2, 3),
            })
            self._saved_brake = self._max_brake_ms2
            self._saved_accel = self._max_accel_ms2
            self._last_save_mono = now
            logger.debug(
                "pedal_capacity saved: brake=%.3f accel=%.3f",
                self._max_brake_ms2,
                self._max_accel_ms2,
            )
        except Exception:
            logger.debug("pedal_capacity save failed", exc_info=True)


def _safe_float(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0
