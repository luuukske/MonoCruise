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

  Gas learning (per-gear)
  - The gas pedal -> acceleration gain is strongly gear-dependent (low gears
    multiply engine torque far more), so a separate weighted EMA is kept per
    transmission gear instead of one speed-mixed scalar.
  - Skips samples for 0.5 s after the clutch was last pressed and while the
    gas pedal is still moving (settle gate), so gear-change and launch-ramp
    transients do not contaminate the per-gear estimate.
  - Gravity and rolling resistance are canceled via road_load_ms2 before sampling.

  The brake estimate and the per-gear gain map are persisted to settings.json
  so the next session starts from the last known good values.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Deque

from core.settings import Settings

from .accel_to_pedals import weight_factor

logger = logging.getLogger(__name__)

_MIN_BRAKE_SPEED_MS: float = 5.0    # ~18 km/h
_MIN_ACCEL_SPEED_MS: float = 1.0    # lowered so gear 1 (used ~0-8 km/h) can learn
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

# Per-gear gas gain learning. The pedal->accel gain is binned by transmission
# gear because the gearbox ratio is the dominant cause of its speed dependence.
# Bounds are absolute (m/s² at gas=1.0): low gears legitimately produce far
# more acceleration per pedal than high gears, so a baseline-relative ceiling
# would clamp them down and reattach the low-speed overshoot bug.
_ACCEL_GAIN_MIN_MS2: float = 0.5
_ACCEL_GAIN_MAX_MS2: float = 8.0
# Per-gear-step boost when extrapolating a gain for an unlearned gear that
# sits below all learned gears: lower gears have higher gain, so bias the
# estimate upward (higher gain -> less pedal -> undershoot, the safe side).
_ACCEL_GEAR_EXTRAP_BOOST: float = 1.2

# Gas settled-pedal gate — skip learning while the gas pedal is still moving.
# During a launch ramp the speed differentiator lags real acceleration;
# sampling mid-transient biases the per-gear EMA.
_GAS_SETTLE_WINDOW_S: float = 0.20
_GAS_SETTLE_TOLERANCE: float = 0.03
_GAS_STEP_THRESHOLD: float = 0.05
_GAS_STEP_GUARD_S: float = 0.25
_GAS_PEDAL_HISTORY_LIMIT: int = 64

# Brake settled-pedal gate — skip learning while the brake pedal is still
# moving. Brake hydraulics have ~150–250 ms of lag; sampling during the
# transient drives the EMA below the steady-state truth.
_BRAKE_SETTLE_WINDOW_S: float = 0.20
_BRAKE_SETTLE_TOLERANCE: float = 0.03
_BRAKE_STEP_THRESHOLD: float = 0.05
_BRAKE_STEP_GUARD_S: float = 0.25
_BRAKE_PEDAL_HISTORY_LIMIT: int = 64


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PedalCapacityTracker:
    """
    Estimates vehicle max brake deceleration (single scalar) and gas
    acceleration gain (one weighted EMA per transmission gear) from samples
    taken whenever a pedal is applied.

    Gravity and rolling resistance are canceled from each sample using
    road_load_ms2 (= slope_accel + rolling_accel, positive = uphill forward).

    Persisted in Settings:
      pedal_capacity_max_brake_ms2       — brake estimate (0 = use baseline)
      pedal_capacity_max_accel_ms2       — global accel scalar; cold-start
                                           fallback before any gear is learned
      pedal_capacity_accel_gain_by_gear  — {gear: gain} learned per-gear map,
                                           stored mass-normalized
    """

    def __init__(self) -> None:
        self._max_brake_ms2: float = 0.0   # 0 = not yet initialised
        self._saved_brake: float = 0.0
        # Per-gear gas gain, mass-normalized (each sample multiplied by
        # weight_factor before learning, so the map holds only the
        # load-invariant gear-ratio shape). Only gears that have received at
        # least one sample appear here.
        self._accel_gain_by_gear: dict[int, float] = {}
        self._saved_accel_by_gear: dict[int, float] = {}
        # Cold-start fallback, used until a gear (or a neighbour) is learned.
        self._global_accel_scalar: float = 0.0
        self._last_save_mono: float = 0.0
        self._last_clutch_mono: float = -math.inf
        self._brake_pedal_history: Deque[tuple[float, float]] = deque(
            maxlen=_BRAKE_PEDAL_HISTORY_LIMIT
        )
        self._last_brake_step_mono: float = -math.inf
        self._gas_pedal_history: Deque[tuple[float, float]] = deque(
            maxlen=_GAS_PEDAL_HISTORY_LIMIT
        )
        self._last_gas_step_mono: float = -math.inf

    @property
    def max_brake_ms2(self) -> float:
        """Current best estimate of max deceleration at brake=1.0 (m/s²)."""
        return self._max_brake_ms2

    def load_persisted(self, baseline_brake: float, baseline_accel: float) -> None:
        """Seed estimates from persisted settings at startup.

        Args:
            baseline_brake: Fallback baseline if no persisted value exists (m/s²).
            baseline_accel: Fallback baseline if no persisted value exists (m/s²).
        """
        b = _safe_float(Settings.pedal_capacity_max_brake_ms2)
        a = _safe_float(Settings.pedal_capacity_max_accel_ms2)
        self._max_brake_ms2 = b if b > 0.0 else baseline_brake
        self._saved_brake = self._max_brake_ms2
        self._global_accel_scalar = a if a > 0.0 else baseline_accel

        # Per-gear map. JSON object keys are strings on disk — convert to int
        # gears and drop anything invalid or out of bounds.
        self._accel_gain_by_gear = {}
        raw_map = getattr(Settings, "pedal_capacity_accel_gain_by_gear", None)
        if isinstance(raw_map, dict):
            for key, value in raw_map.items():
                try:
                    gear = int(key)
                except (TypeError, ValueError):
                    continue
                gain = _safe_float(value)
                if gear > 0 and gain > 0.0:
                    self._accel_gain_by_gear[gear] = _clamp(
                        gain, _ACCEL_GAIN_MIN_MS2, _ACCEL_GAIN_MAX_MS2
                    )
        self._saved_accel_by_gear = dict(self._accel_gain_by_gear)
        logger.debug(
            "pedal_capacity loaded: brake=%.2f m/s² accel_gears=%s",
            self._max_brake_ms2,
            {g: round(v, 2) for g, v in sorted(self._accel_gain_by_gear.items())},
        )

    def accel_gain_for_gear(self, gear: int) -> float:
        """Mass-normalized gas gain for *gear* (m/s² at gas=1.0).

        The value is in the per-gear map's normalized space — the mapper
        divides it by weight_factor to recover the current-mass gain.

        A learned gear returns its EMA directly. An unlearned gear borrows
        from its nearest learned neighbours: geometric interpolation when
        bracketed; the nearest lower gear when above all learned gears (a
        lower gear has higher gain -> conservative undershoot); an
        upward-boosted nearest higher gear when below all learned gears (so a
        low gear is never given a too-low gain, which would overshoot). With
        nothing learned yet, the persisted global scalar is used.
        """
        fallback = (
            self._global_accel_scalar
            if self._global_accel_scalar > 0.0
            else _ACCEL_GAIN_MIN_MS2
        )
        try:
            g = int(gear)
        except (TypeError, ValueError):
            return fallback
        if g <= 0:
            return fallback

        gains = self._accel_gain_by_gear
        if g in gains:
            return gains[g]
        if not gains:
            return fallback

        learned = sorted(gains)
        lower = max((k for k in learned if k < g), default=None)
        upper = min((k for k in learned if k > g), default=None)
        if lower is not None and upper is not None:
            lo, hi = gains[lower], gains[upper]
            t = (g - lower) / (upper - lower)
            value = lo * (hi / lo) ** t
        elif lower is not None:
            value = gains[lower]
        else:
            value = gains[upper] * (_ACCEL_GEAR_EXTRAP_BOOST ** (upper - g))
        return _clamp(value, _ACCEL_GAIN_MIN_MS2, _ACCEL_GAIN_MAX_MS2)

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

        now = time.monotonic()
        history = self._brake_pedal_history
        if history:
            prev_pedal = history[-1][1]
            if abs(brake_output - prev_pedal) >= _BRAKE_STEP_THRESHOLD:
                self._last_brake_step_mono = now
        history.append((now, brake_output))
        # Keep the newest sample at or before the window start so the retained
        # history spans the FULL settle window. Popping everything strictly
        # older than cutoff would leave history[0] just *inside* the window,
        # making the span check below impossible to satisfy (it would reject
        # every sample).
        cutoff = now - _BRAKE_SETTLE_WINDOW_S
        while len(history) > 2 and history[1][0] <= cutoff:
            history.popleft()

        if speed_ms < _MIN_BRAKE_SPEED_MS:
            return
        if brake_output < _BRAKE_PEDAL_FLOOR:
            return
        if abs(slope_rad) > _MAX_SLOPE_RAD:
            return

        if now - self._last_brake_step_mono < _BRAKE_STEP_GUARD_S:
            return

        oldest_in_window = history[0][1]
        if (history[-1][0] - history[0][0] < _BRAKE_SETTLE_WINDOW_S
                or abs(brake_output - oldest_in_window) > _BRAKE_SETTLE_TOLERANCE):
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
        self._maybe_save(now)

    def update_accel(
        self,
        gas_output: float,
        measured_accel_ms2: float,
        speed_ms: float,
        slope_rad: float,
        game_clutch: float,
        gear: int,
        total_mass_kg: float,
        has_trailer: bool,
        road_load_ms2: float = 0.0,
    ) -> None:
        """Feed one acceleration sample into the per-gear gain map.

        Call whenever any gas pedal is applied regardless of source. The
        sample is routed to the bin for *gear*; neutral and reverse are
        skipped. Learning is skipped for 0.5 s after the clutch was last
        pressed and while the gas pedal is still moving (settle gate).

        The gain is stored mass-normalized (sample × weight_factor) so the
        per-gear map holds only the load-invariant gear-ratio shape and a
        load/unload does not force a relearn.

        Args:
            gas_output: Actual gas pedal sent to the game [0–1].
            measured_accel_ms2: Positive measured acceleration (m/s²).
            speed_ms: Current speed (m/s).
            slope_rad: Road pitch (rad, positive = uphill) — used only for slope filter.
            game_clutch: Current clutch position [0–1].
            gear: Current transmission gear (negative = reverse, 0 = neutral).
            total_mass_kg: Current truck + cargo + fuel mass (kg).
            has_trailer: Whether a trailer is attached.
            road_load_ms2: slope_accel + rolling_accel (positive = uphill forward).
                           Added to measured_accel to recover pure engine contribution.
        """
        now = time.monotonic()

        history = self._gas_pedal_history
        if history:
            prev_pedal = history[-1][1]
            if abs(gas_output - prev_pedal) >= _GAS_STEP_THRESHOLD:
                self._last_gas_step_mono = now
        history.append((now, gas_output))
        # Keep the newest sample at or before the window start so the retained
        # history spans the FULL settle window. Popping everything strictly
        # older than cutoff would leave history[0] just *inside* the window,
        # making the span check below impossible to satisfy (it would reject
        # every sample).
        cutoff = now - _GAS_SETTLE_WINDOW_S
        while len(history) > 2 and history[1][0] <= cutoff:
            history.popleft()

        if game_clutch > _CLUTCH_ACTIVE_THRESHOLD:
            self._last_clutch_mono = now
        if now - self._last_clutch_mono < _CLUTCH_GUARD_S:
            return

        try:
            g = int(gear)
        except (TypeError, ValueError):
            return
        if g <= 0:
            return

        if speed_ms < _MIN_ACCEL_SPEED_MS:
            return
        if gas_output < _ACCEL_PEDAL_FLOOR:
            return
        if abs(slope_rad) > _MAX_SLOPE_RAD:
            return

        if now - self._last_gas_step_mono < _GAS_STEP_GUARD_S:
            return

        oldest_in_window = history[0][1]
        if (history[-1][0] - history[0][0] < _GAS_SETTLE_WINDOW_S
                or abs(gas_output - oldest_in_window) > _GAS_SETTLE_TOLERANCE):
            return

        corrected_accel = measured_accel_ms2 + road_load_ms2
        if corrected_accel < _MIN_ACCEL_MS2:
            return

        candidate = corrected_accel / max(gas_output, _ACCEL_PEDAL_FLOOR)
        # Normalize out the current load: weight_factor scales the sample to a
        # reference mass so the per-gear EMA learns only the gear-ratio shape.
        # The mapper divides by weight_factor again at use time.
        candidate *= weight_factor(total_mass_kg, has_trailer)
        candidate = _clamp(candidate, _ACCEL_GAIN_MIN_MS2, _ACCEL_GAIN_MAX_MS2)

        current = self._accel_gain_by_gear.get(g)
        if current is None:
            # Seed a new gear from the best available estimate so it starts
            # near reality and the EMA only has to refine it.
            current = self.accel_gain_for_gear(g)

        weight = gas_output ** _WEIGHT_POWER
        alpha = _ACCEL_BASE_ALPHA * weight
        if candidate < current:
            alpha *= _UNDERPERFORM_MULT
        alpha = min(alpha, 1.0)

        updated = current + alpha * (candidate - current)
        self._accel_gain_by_gear[g] = _clamp(
            updated, _ACCEL_GAIN_MIN_MS2, _ACCEL_GAIN_MAX_MS2
        )
        self._maybe_save(now)

    def _maybe_save(self, now: float) -> None:
        if now - self._last_save_mono < _SAVE_COOLDOWN_S:
            return
        brake_drift = abs(self._max_brake_ms2 - self._saved_brake) / max(self._saved_brake, 0.01)
        if brake_drift < _SAVE_THRESHOLD and self._accel_gear_drift() < _SAVE_THRESHOLD:
            return
        try:
            Settings.save(values={
                "pedal_capacity_max_brake_ms2": round(self._max_brake_ms2, 3),
                "pedal_capacity_accel_gain_by_gear": {
                    str(g): round(v, 3)
                    for g, v in sorted(self._accel_gain_by_gear.items())
                },
            })
            self._saved_brake = self._max_brake_ms2
            self._saved_accel_by_gear = dict(self._accel_gain_by_gear)
            self._last_save_mono = now
            logger.debug(
                "pedal_capacity saved: brake=%.3f accel_gears=%s",
                self._max_brake_ms2,
                {g: round(v, 2) for g, v in sorted(self._accel_gain_by_gear.items())},
            )
        except Exception:
            logger.debug("pedal_capacity save failed", exc_info=True)

    def _accel_gear_drift(self) -> float:
        """Max relative drift of any per-gear gain since the last save.

        A newly learned gear (absent from the saved snapshot) counts as full
        drift so the first sample for a gear always triggers a save.
        """
        worst = 0.0
        for gear, gain in self._accel_gain_by_gear.items():
            saved = self._saved_accel_by_gear.get(gear)
            if saved is None:
                return 1.0
            worst = max(worst, abs(gain - saved) / max(saved, 0.01))
        return worst


def _safe_float(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0
