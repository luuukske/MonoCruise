"""
Tracks estimated maximum brake deceleration and gas acceleration.

Replaces brake_efficiency.py with a simpler, always-on system:

  Brake learning
  - Samples whenever any brake pedal is applied (user, cruise, or any future source).
  - Each sample is weighted by brake_output² so light braking has tiny influence
    while heavy braking drives the estimate strongly.
  - Underperformance (measured decel < expected) drops the estimate 3× faster
    than overperformance rises it: safety bias for emergency stops.
  - Gravity and rolling resistance are canceled via road_load_ms2 before sampling.

  Gas learning (shape-function: learned anchor + learned ratio)
  - Pedal -> accel gain is dominated by gearbox ratio, but the real shape
    isn't perfectly geometric (engine torque curve, splitter/range steps,
    clutch slip at launch). Two scalars parameterize the whole gear curve:
        G(gear) = anchor * ratio^(_ANCHOR_GEAR - gear)
    Both are learned online via log-space linear regression: each sample's
    residual nudges anchor and ratio toward the best fit, with the ratio's
    update weighted by the sample's gear distance from the anchor (samples
    far from the anchor carry more slope information; samples at the anchor
    only refine the amplitude).
  - Monotonic by construction (ratio is clamped > 1). One bad low-gear sample
    cannot invert against well-known top-gear samples.
  - Skipped after a clutch press (0.5 s), inside the per-gear dwell window
    after a gear change (the speed differentiator lags real accel through the
    launch ramp), and while the gas pedal is still moving (settle gate).
  - Gravity and rolling resistance are canceled via road_load_ms2 before
    sampling; samples are multiplied by weight_factor so the stored anchor
    is mass-normalized (load/unload does not force relearn).

  The brake estimate, the anchor gain, and the learned ratio step are
  persisted to settings.json so the next session starts from the last known
  good values.
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
_MAX_SLOPE_RAD: float = 0.15        # ~8.6°: skip extreme slopes (sensor uncertainty)
_WEIGHT_POWER: float = 3.0          # alpha scaled by pedal^3: full pedal dominates
_ACCEL_BASE_ALPHA: float = 0.08     # EMA alpha at full gas pedal, no underperformance
_UNDERPERFORM_MULT: float = 2.0     # drop estimate this much faster when below expectation
_CLUTCH_GUARD_S: float = 0.5        # seconds after clutch to skip gas learning
_CLUTCH_ACTIVE_THRESHOLD: float = 0.05
_SAVE_THRESHOLD: float = 0.1        # save when drift exceeds 10% of saved value
_SAVE_COOLDOWN_S: float = 30.0      # min seconds between successive writes
_ESTIMATE_LOWER_BOUND: float = 0.35 # fraction of baseline: hard floor
# Hard ceiling as a fraction of the mass-adjusted baseline. The old 2.0 allowed
# estimates near 16-18 m/s2 (over 1.6 g), which no truck foot brake can do.
# AEB divides required decel by this estimate to decide engagement, so an
# inflated value silently disables emergency braking (crash clips ddc0cdf7 /
# 0fe85c88, 2026-07-10). 1.3 leaves headroom for an unloaded truck
# outperforming the loaded baseline without letting contamination compound.
_ESTIMATE_UPPER_BOUND: float = 1.3  # fraction of baseline: hard ceiling
# Reject any single brake sample implying more than this fraction of baseline.
# candidate = decel / pedal extrapolates to full pedal; when the measured decel
# includes retarder or engine brake (no telemetry field exposes them), a light
# steady pedal yields a candidate far above what the foot brake can deliver.
# Such samples are contaminated, not information: drop them instead of
# averaging them in. Slightly above the ceiling so legitimate samples near the
# clamp still register.
_BRAKE_CANDIDATE_MAX_FRACTION: float = 1.35
# Two-speed brake learning: game brake force is progressive in pedal, so
# decel/pedal from gentle presses under-reads full-pedal capacity and
# routine driving dragged the estimate 9 -> 4 m/s2 (truck measured 10-12;
# clips 1b277e63 / fa70013c). Normal driving drifts the estimate slowly;
# AEB events (deep, honest presses) re-teach it fast.
_BRAKE_ALPHA_NORMAL: float = 0.02   # EMA alpha at full pedal, normal driving
_BRAKE_ALPHA_AEB: float = 0.15      # EMA alpha at full pedal during AEB braking

# Shape-function model for per-gear gas gain. Two scalars parameterize the
# whole curve via
#     G(gear) = anchor * ratio^(_ANCHOR_GEAR - gear)
# Both `anchor` and `ratio` are learned online via log-space regression on
# every accepted sample (see update_accel). The model is monotonic by
# construction (ratio is clamped > 1), so gear g+1 can never end up with a
# higher gain than gear g: the inversion that independent per-gear EMAs
# were prone to.
_RATIO_INIT: float = 1.27            # default until enough cross-gear samples settle the
                                     # regression: also the seed multiplier used to
                                     # project the legacy max_accel scalar to the anchor
_RATIO_MIN: float = 1.05             # absolute bounds: outside is unphysical for any
_RATIO_MAX: float = 1.45             # common truck transmission
_RATIO_BASE_ALPHA: float = 0.02      # log-space ratio learning rate at gas=1.0. Lower
                                     # than the anchor's because each sample's update
                                     # is already amplified by its gear distance from
                                     # the anchor (the regression's leverage term).
_ANCHOR_GEAR: int = 6                # mid-stack reference gear
_LEGACY_SEED_GEAR: int = 8           # legacy max_accel_ms2 was mostly learned in top
                                     # cruise gears; project from here when seeding

# Anchor-gain bounds (m/s² at gas=1.0, weight-normalized, evaluated at
# _ANCHOR_GEAR). Per-gear values derived from this anchor span a much wider
# range: the model handles the per-gear shape, the anchor handles amplitude.
_ACCEL_ANCHOR_MIN_MS2: float = 0.5
_ACCEL_ANCHOR_MAX_MS2: float = 8.0

# Gear-dwell gate. After a gear change the speed differentiator
# (sending_thread, τ=0.30 s) still lags real acceleration, and engine torque
# is settling through clutch engagement: samples taken here bias the gain
# low. Low gears (1–3) are engaged so briefly per launch that the full
# dwell would block them from ever sampling, so the gate uses a shorter
# threshold there.
_GEAR_DWELL_S: float = 0.30
_LOW_GEAR_DWELL_S: float = 0.10
_LOW_GEAR_DWELL_MAX_GEAR: int = 3

# Gas settled-pedal gate: skip learning while the gas pedal is still moving.
# Catches gas hunting during active control, independent of gear changes.
_GAS_SETTLE_WINDOW_S: float = 0.20
_GAS_SETTLE_TOLERANCE: float = 0.03
_GAS_STEP_THRESHOLD: float = 0.05
_GAS_STEP_GUARD_S: float = 0.25
_GAS_PEDAL_HISTORY_LIMIT: int = 64

# Brake settled-pedal gate: skip learning while the brake pedal is still
# moving. Brake hydraulics have ~150-250 ms of lag and the speed
# differentiator (tau=0.30 s) lags real decel further, so samples taken
# before both converge are biased in either direction. Longer than the gas
# window because brake samples feed AEB's engagement denominator: a
# contaminated sample here is costlier than a lost one.
_BRAKE_SETTLE_WINDOW_S: float = 0.50
_BRAKE_SETTLE_TOLERANCE: float = 0.03
_BRAKE_STEP_THRESHOLD: float = 0.05
_BRAKE_STEP_GUARD_S: float = 0.25
_BRAKE_PEDAL_HISTORY_LIMIT: int = 64


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PedalCapacityTracker:
    """
    Estimates vehicle max brake deceleration (single scalar) and gas
    acceleration gain (single shape-function anchor; per-gear values derived
    by geometric projection) from samples taken whenever a pedal is applied.

    Gravity and rolling resistance are canceled from each sample using
    road_load_ms2 (= slope_accel + rolling_accel, positive = uphill forward).

    Persisted in Settings:
      pedal_capacity_max_brake_ms2           : brake estimate (0 = use baseline)
      pedal_capacity_max_accel_ms2           : legacy scalar; cold-start seed
                                                source for the anchor
      pedal_capacity_accel_anchor_gain_ms2   : shape-function anchor (m/s² at
                                                gas=1.0, mass-normalized, at
                                                _ANCHOR_GEAR)
      pedal_capacity_accel_ratio_step        : learned per-gear-step ratio
                                                (1.0 = flat, >1.0 = lower gear
                                                has more gain)
    """

    def __init__(self) -> None:
        self._max_brake_ms2: float = 0.0   # 0 = not yet initialised
        self._saved_brake: float = 0.0
        # Shape-function anchor (m/s² at gas=1.0, mass-normalized, at
        # _ANCHOR_GEAR). Every gear's gain is derived by geometric
        # projection: see accel_gain_for_gear / update_accel.
        self._accel_anchor_gain_ms2: float = 0.0
        self._saved_accel_anchor: float = 0.0
        # Learned per-gear-step ratio. Starts at the in-code default; the
        # regression in update_accel adjusts it as cross-gear samples arrive.
        self._accel_ratio_step: float = _RATIO_INIT
        self._saved_accel_ratio: float = _RATIO_INIT
        # Legacy scalar kept as a fallback before the anchor is seeded.
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
        # Gear-change tracking for the dwell gate.
        self._prev_gear: int = 0
        self._last_gear_change_mono: float = -math.inf

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
        # Clamp the persisted value to the same bounds update_brake enforces:
        # a config poisoned by pre-guard contamination self-heals at startup
        # instead of carrying an unphysical estimate into AEB for a session.
        self._max_brake_ms2 = _clamp(
            b if b > 0.0 else baseline_brake,
            baseline_brake * _ESTIMATE_LOWER_BOUND,
            baseline_brake * _ESTIMATE_UPPER_BOUND,
        )
        self._saved_brake = self._max_brake_ms2

        # Legacy fallback (used only before the anchor is seeded).
        legacy = _safe_float(Settings.pedal_capacity_max_accel_ms2)
        self._global_accel_scalar = legacy if legacy > 0.0 else baseline_accel

        # Ratio first (the anchor seed below depends on it). Persisted value
        # is used when present; otherwise start at the in-code default.
        ratio = _safe_float(getattr(Settings, "pedal_capacity_accel_ratio_step", 0.0))
        self._accel_ratio_step = _clamp(
            ratio if ratio > 1.0 else _RATIO_INIT, _RATIO_MIN, _RATIO_MAX,
        )
        self._saved_accel_ratio = self._accel_ratio_step

        # Shape-function anchor. If a persisted anchor is present, use it.
        # Otherwise seed from the legacy max-accel scalar: that value was
        # learned mostly during top-gear cruise, so treat it as G(_LEGACY_SEED_GEAR)
        # and project geometrically to _ANCHOR_GEAR. Result: a sensible shape
        # across all gears on first launch; the anchor refines from new samples.
        anchor = _safe_float(getattr(Settings, "pedal_capacity_accel_anchor_gain_ms2", 0.0))
        if anchor <= 0.0:
            seed_source = legacy if legacy > 0.0 else baseline_accel
            anchor = seed_source * (self._accel_ratio_step ** (_LEGACY_SEED_GEAR - _ANCHOR_GEAR))
        self._accel_anchor_gain_ms2 = _clamp(
            anchor, _ACCEL_ANCHOR_MIN_MS2, _ACCEL_ANCHOR_MAX_MS2
        )
        self._saved_accel_anchor = self._accel_anchor_gain_ms2
        logger.debug(
            "pedal_capacity loaded: brake=%.2f m/s² accel_anchor=%.2f m/s² (at gear %d) ratio=%.3f",
            self._max_brake_ms2, self._accel_anchor_gain_ms2, _ANCHOR_GEAR,
            self._accel_ratio_step,
        )

    def accel_gain_for_gear(self, gear: int) -> float:
        """Mass-normalized gas gain for *gear* (m/s² at gas=1.0).

        Geometric projection from the learned anchor:
            G(gear) = anchor * ratio^(_ANCHOR_GEAR - gear)
        Lower gear = higher gain (more torque to wheels). The mapper divides
        by weight_factor to recover the current-mass gain. Monotonic by
        construction: independent per-gear EMAs let brief, noisy low-gear
        samples invert against well-known top-gear samples; this model can't.
        """
        anchor = self._accel_anchor_gain_ms2
        if anchor <= 0.0:
            # Fallback before the anchor is seeded (shouldn't happen after
            # load_persisted, but keep a defensive path).
            return (
                self._global_accel_scalar
                if self._global_accel_scalar > 0.0
                else _ACCEL_ANCHOR_MIN_MS2
            )
        try:
            g = int(gear)
        except (TypeError, ValueError):
            return anchor
        if g <= 0:
            return anchor
        return anchor * (self._accel_ratio_step ** (_ANCHOR_GEAR - g))

    def update_brake(
        self,
        brake_output: float,
        measured_decel_ms2: float,
        speed_ms: float,
        slope_rad: float,
        baseline_ms2: float,
        road_load_ms2: float = 0.0,
        aeb_active: bool = False,
    ) -> None:
        """Feed one braking sample.

        Call whenever any brake pedal is applied regardless of source.

        Args:
            brake_output: Actual brake pedal sent to the game [0–1].
            measured_decel_ms2: Positive measured deceleration (m/s²).
            speed_ms: Current speed (m/s).
            slope_rad: Road pitch (rad, positive = uphill): used only for slope filter.
            baseline_ms2: Baseline max decel for clamping bounds.
            road_load_ms2: slope_accel + rolling_accel (positive = uphill forward).
                           Subtracted from measured_decel to isolate pure brake force.
            aeb_active: True while AEB commands the brake: enables the fast
                        learning alpha (see the two-speed note above).
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
        if candidate > baseline_ms2 * _BRAKE_CANDIDATE_MAX_FRACTION:
            return

        weight = brake_output ** _WEIGHT_POWER
        base = _BRAKE_ALPHA_AEB if aeb_active else _BRAKE_ALPHA_NORMAL
        alpha = base * weight
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
        """Feed one acceleration sample into the shape-function anchor EMA.

        Every sample, regardless of which gear it came from, is projected
        back to the anchor gear and contributes to the single anchor_gain
        scalar. The geometric model gives monotonic per-gear gains by
        construction: brief noisy low-gear samples cannot invert against
        well-known top-gear samples.

        Skipped when:
          * within 0.5 s of the last clutch press;
          * within the per-gear dwell window after the most recent gear
            change (the speed differentiator τ=0.30 s lags real accel
            through a launch ramp; sampling too soon biases the anchor low);
          * gas pedal is still moving (settle gate).

        Args:
            gas_output: Actual gas pedal sent to the game [0–1].
            measured_accel_ms2: Positive measured acceleration (m/s²).
            speed_ms: Current speed (m/s).
            slope_rad: Road pitch (rad, positive = uphill): used only for slope filter.
            game_clutch: Current clutch position [0–1].
            gear: Current transmission gear (negative = reverse, 0 = neutral).
            total_mass_kg: Current truck + cargo + fuel mass (kg).
            has_trailer: Whether a trailer is attached.
            road_load_ms2: slope_accel + rolling_accel (positive = uphill forward).
                           Added to measured_accel to recover pure engine contribution.
        """
        now = time.monotonic()

        # Gear-change tracking: unconditional so the dwell timer is correct
        # even when the prior sample was rejected for another reason.
        g_now = 0
        try:
            g_now = int(gear)
        except (TypeError, ValueError):
            pass
        if g_now != self._prev_gear:
            self._prev_gear = g_now
            self._last_gear_change_mono = now

        history = self._gas_pedal_history
        if history:
            prev_pedal = history[-1][1]
            if abs(gas_output - prev_pedal) >= _GAS_STEP_THRESHOLD:
                self._last_gas_step_mono = now
        history.append((now, gas_output))
        # Keep the newest sample at or before the window start so the retained
        # history spans the FULL settle window. Popping everything strictly
        # older than cutoff would leave history[0] just *inside* the window,
        # making the span check below impossible to satisfy.
        cutoff = now - _GAS_SETTLE_WINDOW_S
        while len(history) > 2 and history[1][0] <= cutoff:
            history.popleft()

        if game_clutch > _CLUTCH_ACTIVE_THRESHOLD:
            self._last_clutch_mono = now
        if now - self._last_clutch_mono < _CLUTCH_GUARD_S:
            return

        g = g_now
        if g <= 0:
            return

        # Gear-dwell gate. Low gears (1–3) are engaged so briefly per launch
        # that a 0.3 s threshold would block them from ever sampling, so the
        # threshold scales with gear.
        dwell = _LOW_GEAR_DWELL_S if g <= _LOW_GEAR_DWELL_MAX_GEAR else _GEAR_DWELL_S
        if now - self._last_gear_change_mono < dwell:
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

        # Per-pedal gain at current mass, then mass-normalized.
        candidate = corrected_accel / max(gas_output, _ACCEL_PEDAL_FLOOR)
        candidate *= weight_factor(total_mass_kg, has_trailer)

        # Log-space linear regression on the model
        #     log(G(g)) = log(anchor) + x · log(ratio),   x = _ANCHOR_GEAR - g
        # so the residual is log(measured) - (log(anchor) + x · log(ratio)).
        # Gradient descent on r²/2:
        #     Δlog(anchor) = lr_α · r
        #     Δlog(ratio)  = lr_β · x · r
        # The ratio update naturally weights samples by their leverage |x|
        # (samples at the anchor gear give zero ratio info; samples far from
        # it carry most of the slope). Both lrs scale by pedal^3 so weak
        # inputs barely move the estimate. An underperform safety bias is
        # preserved: when measured gain is below predicted (the overshoot-
        # prone direction, also the failure mode the user reports: gear 1/2
        # undershoot means real low-gear gain is below the projection), both
        # updates run faster so the model adapts quickly.
        x = _ANCHOR_GEAR - g
        log_ratio = math.log(self._accel_ratio_step)
        log_m = math.log(candidate)

        if self._accel_anchor_gain_ms2 <= 0.0:
            # First valid sample: seed the anchor from this single sample
            # using the current ratio. Same as projecting the sample to the
            # anchor gear via the geometric model.
            seed = math.exp(log_m - x * log_ratio)
            self._accel_anchor_gain_ms2 = _clamp(
                seed, _ACCEL_ANCHOR_MIN_MS2, _ACCEL_ANCHOR_MAX_MS2,
            )
            self._maybe_save(now)
            return

        log_anchor = math.log(self._accel_anchor_gain_ms2)
        residual = log_m - (log_anchor + x * log_ratio)

        weight = gas_output ** _WEIGHT_POWER
        lr_anchor = _ACCEL_BASE_ALPHA * weight
        lr_ratio = _RATIO_BASE_ALPHA * weight
        if residual < 0.0:
            lr_anchor *= _UNDERPERFORM_MULT
            lr_ratio *= _UNDERPERFORM_MULT
        lr_anchor = min(lr_anchor, 1.0)
        lr_ratio = min(lr_ratio, 0.5)

        log_anchor += lr_anchor * residual
        log_ratio += lr_ratio * x * residual

        self._accel_anchor_gain_ms2 = _clamp(
            math.exp(log_anchor), _ACCEL_ANCHOR_MIN_MS2, _ACCEL_ANCHOR_MAX_MS2,
        )
        self._accel_ratio_step = _clamp(
            math.exp(log_ratio), _RATIO_MIN, _RATIO_MAX,
        )
        self._maybe_save(now)

    def _maybe_save(self, now: float) -> None:
        if now - self._last_save_mono < _SAVE_COOLDOWN_S:
            return
        brake_drift = abs(self._max_brake_ms2 - self._saved_brake) / max(self._saved_brake, 0.01)
        anchor_drift = (
            abs(self._accel_anchor_gain_ms2 - self._saved_accel_anchor)
            / max(self._saved_accel_anchor, 0.01)
        )
        ratio_drift = (
            abs(self._accel_ratio_step - self._saved_accel_ratio)
            / max(self._saved_accel_ratio, 0.01)
        )
        if (brake_drift < _SAVE_THRESHOLD
                and anchor_drift < _SAVE_THRESHOLD
                and ratio_drift < _SAVE_THRESHOLD):
            return
        try:
            Settings.save(values={
                "pedal_capacity_max_brake_ms2": round(self._max_brake_ms2, 3),
                "pedal_capacity_accel_anchor_gain_ms2": round(self._accel_anchor_gain_ms2, 3),
                "pedal_capacity_accel_ratio_step": round(self._accel_ratio_step, 4),
            })
            self._saved_brake = self._max_brake_ms2
            self._saved_accel_anchor = self._accel_anchor_gain_ms2
            self._saved_accel_ratio = self._accel_ratio_step
            self._last_save_mono = now
            logger.debug(
                "pedal_capacity saved: brake=%.3f accel_anchor=%.3f ratio=%.4f",
                self._max_brake_ms2, self._accel_anchor_gain_ms2,
                self._accel_ratio_step,
            )
        except Exception:
            logger.debug("pedal_capacity save failed", exc_info=True)


def _safe_float(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0

