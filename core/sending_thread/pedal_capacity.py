"""Online brake/gas capacity learning. See core/sending_thread/README.md."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Deque

from core.settings import Settings

from .accel_to_pedals import brake_curve_fraction, weight_factor

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
# Learned correction on the rig baseline, not an absolute m/s2. Bounds are
# asymmetric on purpose: under-delivery must be believable, over-reading is not.
_BRAKE_SCALE_MIN: float = 0.35
_BRAKE_SCALE_MAX: float = 1.00
# Reject partial-pedal extrapolations above this fraction of the load baseline.
_BRAKE_CANDIDATE_MAX_FRACTION: float = 1.35
# Two-speed brake learning. Routine presses are shallow (pedal³ weight) and
_BRAKE_ALPHA_NORMAL: float = 0.02   # EMA alpha at full pedal, normal driving
_BRAKE_ALPHA_AEB: float = 0.15      # EMA alpha at full pedal during AEB braking

# Shape-function model for per-gear gas gain. Two scalars parameterize the
_RATIO_INIT: float = 1.27            # default until enough cross-gear samples settle the
                                     # regression: also the seed multiplier used to
                                     # project the legacy max_accel scalar to the anchor
_RATIO_MIN: float = 1.05             # absolute bounds: outside is unphysical for any
_RATIO_MAX: float = 1.45             # common truck transmission
_RATIO_BASE_ALPHA: float = 0.02      # log-space ratio learning rate at gas=1.0. Lower
                                     # than the anchor's because each sample's update
_ANCHOR_GEAR: int = 6                # mid-stack reference gear
_LEGACY_SEED_GEAR: int = 8           # legacy max_accel_ms2 was mostly learned in top
                                     # cruise gears; project from here when seeding

# Anchor-gain bounds (m/s² at gas=1.0, weight-normalized, evaluated at
_ACCEL_ANCHOR_MIN_MS2: float = 0.5
_ACCEL_ANCHOR_MAX_MS2: float = 8.0

# Gear-dwell gate. After a gear change the driveline is still restoring
# torque, so accel is depressed for reasons that have nothing to do with gas.
# Measured recovery is about 1.5 s; the old 0.30 s opened mid-recovery and let
# the anchor learn the sag as weakness. See `core/sending_thread/README.md`.
_GEAR_DWELL_S: float = 1.50
_LOW_GEAR_DWELL_S: float = 0.30
_LOW_GEAR_DWELL_MAX_GEAR: int = 3

# Gas settled-pedal gate: skip learning while the gas pedal is still moving.
# Catches gas hunting during active control, independent of gear changes.
_GAS_SETTLE_WINDOW_S: float = 0.20
_GAS_SETTLE_TOLERANCE: float = 0.03
_GAS_STEP_THRESHOLD: float = 0.05
_GAS_STEP_GUARD_S: float = 0.25
_GAS_PEDAL_HISTORY_LIMIT: int = 64

# Brake settled-pedal gate: skip learning while the brake pedal is still
_BRAKE_SETTLE_WINDOW_S: float = 0.50
_BRAKE_SETTLE_TOLERANCE: float = 0.05
_BRAKE_STEP_THRESHOLD: float = 0.05
_BRAKE_STEP_GUARD_S: float = 0.25
_BRAKE_PEDAL_HISTORY_LIMIT: int = 64
_BRAKE_PEDAL_SMOOTH_TAU_S: float = 0.10

# Accel settled gate: the gas counterpart of the decel gate below. A settled
# pedal is not enough, the accel signal itself has to have stopped moving.
_ACCEL_SETTLE_WINDOW_S: float = 0.50
_ACCEL_SETTLE_ABS_MS2: float = 0.15
_ACCEL_SETTLE_FRAC: float = 0.10
_ACCEL_HISTORY_LIMIT: int = 64

# Decel settled gate: the decel signal trails the pedal through the game's
_DECEL_SETTLE_WINDOW_S: float = 0.40
_DECEL_SETTLE_ABS_MS2: float = 0.5
_DECEL_SETTLE_FRAC: float = 0.12
_DECEL_HISTORY_LIMIT: int = 64


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _brake_candidate_cap_ms2(baseline_ms2: float) -> float:
    """Reject cap for one sample: this far above the rig baseline is contamination."""
    return baseline_ms2 * _BRAKE_CANDIDATE_MAX_FRACTION


class PedalCapacityTracker:
    """Estimates vehicle max brake deceleration (single scalar) and gas See `core/sending_thread/README.md`."""

    def __init__(self) -> None:
        # Learned correction on baseline_brake_ms2; 1.0 = the model is right.
        self._brake_scale: float = 1.0
        self._saved_brake_scale: float = 1.0
        self._max_brake_ms2: float = 0.0   # scale x baseline, resolved per tick
        # Shape-function anchor (m/s² at gas=1.0, mass-normalized, at
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
        self._brake_pedal_smooth: float | None = None
        self._last_brake_call_mono: float | None = None
        self._decel_history: Deque[tuple[float, float]] = deque(
            maxlen=_DECEL_HISTORY_LIMIT
        )
        self._gas_pedal_history: Deque[tuple[float, float]] = deque(
            maxlen=_GAS_PEDAL_HISTORY_LIMIT
        )
        self._accel_history: Deque[tuple[float, float]] = deque(
            maxlen=_ACCEL_HISTORY_LIMIT
        )
        self._last_accel_call_mono: float | None = None
        self._last_gas_step_mono: float = -math.inf
        # Gear-change tracking for the dwell gate.
        self._prev_gear: int = 0
        self._last_gear_change_mono: float = -math.inf

    @property
    def max_brake_ms2(self) -> float:
        """Current best estimate of max deceleration at brake=1.0 (m/s²)."""
        return self._max_brake_ms2

    @property
    def brake_scale(self) -> float:
        """Learned correction on the rig brake baseline (1.0 = model believed)."""
        return self._brake_scale

    def load_persisted(self, baseline_brake: float, baseline_accel: float) -> None:
        """Seed estimates from persisted settings at startup. See `core/sending_thread/README.md`."""
        scale = _safe_float(Settings.pedal_capacity_brake_scale)
        self._brake_scale = _clamp(
            scale if scale > 0.0 else 1.0, _BRAKE_SCALE_MIN, _BRAKE_SCALE_MAX,
        )
        self._saved_brake_scale = self._brake_scale
        self._max_brake_ms2 = self._brake_scale * baseline_brake

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
        """Mass-normalized gas gain for *gear* (m/s² at gas=1.0). See `core/sending_thread/README.md`."""
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
        """Feed one braking tick. See `core/sending_thread/README.md`."""
        # Re-resolve against this tick's rig: hooking a trailer must move the
        # estimate in the same tick, so only the correction is carried over.
        if baseline_ms2 > 0.0:
            self._max_brake_ms2 = self._brake_scale * baseline_ms2

        now = time.monotonic()

        # Smoothed pedal for gating and the ratio (see the settle-gate note
        last = self._last_brake_call_mono
        self._last_brake_call_mono = now
        smooth = self._brake_pedal_smooth
        if smooth is None or last is None or now - last > _BRAKE_SETTLE_WINDOW_S:
            smooth = brake_output
            self._brake_pedal_history.clear()
            self._decel_history.clear()
        else:
            a_s = 1.0 - math.exp(
                -max(now - last, 1e-4) / _BRAKE_PEDAL_SMOOTH_TAU_S
            )
            smooth += a_s * (brake_output - smooth)
        self._brake_pedal_smooth = smooth

        history = self._brake_pedal_history
        if history:
            prev_pedal = history[-1][1]
            if abs(smooth - prev_pedal) >= _BRAKE_STEP_THRESHOLD:
                self._last_brake_step_mono = now
        history.append((now, smooth))
        # Keep the newest sample at or before the window start so the retained
        cutoff = now - _BRAKE_SETTLE_WINDOW_S
        while len(history) > 2 and history[1][0] <= cutoff:
            history.popleft()

        corrected_decel = measured_decel_ms2 - road_load_ms2
        d_hist = self._decel_history
        d_hist.append((now, corrected_decel))
        d_cutoff = now - _DECEL_SETTLE_WINDOW_S
        while len(d_hist) > 2 and d_hist[1][0] <= d_cutoff:
            d_hist.popleft()

        if speed_ms < _MIN_BRAKE_SPEED_MS or baseline_ms2 <= 0.0:
            return
        if smooth < _BRAKE_PEDAL_FLOOR:
            return
        if abs(slope_rad) > _MAX_SLOPE_RAD:
            return

        if now - self._last_brake_step_mono < _BRAKE_STEP_GUARD_S:
            return

        pedal_values = [p for _, p in history]
        if (history[-1][0] - history[0][0] < _BRAKE_SETTLE_WINDOW_S
                or max(pedal_values) - min(pedal_values) > _BRAKE_SETTLE_TOLERANCE):
            return

        # Decel settled: the ratio below is only meaningful once the measured
        if d_hist[-1][0] - d_hist[0][0] < _DECEL_SETTLE_WINDOW_S:
            return
        decel_values = [d for _, d in d_hist]
        d_max = max(decel_values)
        if d_max - min(decel_values) > max(
            _DECEL_SETTLE_ABS_MS2, _DECEL_SETTLE_FRAC * d_max
        ):
            return

        mean_decel = sum(decel_values) / len(decel_values)
        if mean_decel < _MIN_DECEL_MS2:
            return

        # Window means on both sides: zero-mean dither and telemetry ripple
        mean_pedal = max(
            sum(pedal_values) / len(pedal_values), _BRAKE_PEDAL_FLOOR
        )
        candidate = mean_decel / brake_curve_fraction(mean_pedal)
        if candidate > _brake_candidate_cap_ms2(baseline_ms2):
            return

        # Learn in baseline-relative units so the sample stays meaningful after
        # the rig changes underneath it.
        candidate_scale = candidate / baseline_ms2
        weight = mean_pedal ** _WEIGHT_POWER
        base = _BRAKE_ALPHA_AEB if aeb_active else _BRAKE_ALPHA_NORMAL
        alpha = base * weight
        if candidate_scale < self._brake_scale:
            alpha *= _UNDERPERFORM_MULT
        alpha = min(alpha, 1.0)

        self._brake_scale = _clamp(
            self._brake_scale + alpha * (candidate_scale - self._brake_scale),
            _BRAKE_SCALE_MIN,
            _BRAKE_SCALE_MAX,
        )
        self._max_brake_ms2 = self._brake_scale * baseline_ms2
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
        """Feed one acceleration sample into the shape-function anchor EMA. See `core/sending_thread/README.md`."""
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

        # A gap in the call stream means the pedal was released or the gearbox
        # went through neutral, so both windows restart rather than straddle it.
        last_call = self._last_accel_call_mono
        self._last_accel_call_mono = now
        if last_call is None or now - last_call > _ACCEL_SETTLE_WINDOW_S:
            self._gas_pedal_history.clear()
            self._accel_history.clear()

        history = self._gas_pedal_history
        if history:
            prev_pedal = history[-1][1]
            if abs(gas_output - prev_pedal) >= _GAS_STEP_THRESHOLD:
                self._last_gas_step_mono = now
        history.append((now, gas_output))
        # Keep the newest sample at or before the window start so the retained
        cutoff = now - _GAS_SETTLE_WINDOW_S
        while len(history) > 2 and history[1][0] <= cutoff:
            history.popleft()

        corrected_accel = measured_accel_ms2 + road_load_ms2
        a_hist = self._accel_history
        a_hist.append((now, corrected_accel))
        a_cutoff = now - _ACCEL_SETTLE_WINDOW_S
        while len(a_hist) > 2 and a_hist[1][0] <= a_cutoff:
            a_hist.popleft()

        if game_clutch > _CLUTCH_ACTIVE_THRESHOLD:
            self._last_clutch_mono = now
        if now - self._last_clutch_mono < _CLUTCH_GUARD_S:
            return

        g = g_now
        if g <= 0:
            return

        # Gear-dwell gate. Low gears (1–3) are engaged so briefly per launch
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

        # Accel settled: a gain read while the accel signal is still moving
        # measures the disturbance, not the truck.
        if a_hist[-1][0] - a_hist[0][0] < _ACCEL_SETTLE_WINDOW_S:
            return
        accel_values = [a for _, a in a_hist]
        a_max = max(accel_values)
        if a_max - min(accel_values) > max(
            _ACCEL_SETTLE_ABS_MS2, _ACCEL_SETTLE_FRAC * abs(a_max)
        ):
            return

        # Window means on both sides, so telemetry ripple averages out instead
        # of picking whichever tick the gate happened to open on.
        mean_accel = sum(accel_values) / len(accel_values)
        if mean_accel < _MIN_ACCEL_MS2:
            return
        pedal_values = [p for _, p in history]
        mean_pedal = max(
            sum(pedal_values) / len(pedal_values), _ACCEL_PEDAL_FLOOR
        )

        # Per-pedal gain at current mass, then mass-normalized.
        candidate = mean_accel / mean_pedal
        candidate *= weight_factor(total_mass_kg, has_trailer)

        # Log-space linear regression on the model
        x = _ANCHOR_GEAR - g
        log_ratio = math.log(self._accel_ratio_step)
        log_m = math.log(candidate)

        if self._accel_anchor_gain_ms2 <= 0.0:
            # First valid sample: seed the anchor from this single sample
            seed = math.exp(log_m - x * log_ratio)
            self._accel_anchor_gain_ms2 = _clamp(
                seed, _ACCEL_ANCHOR_MIN_MS2, _ACCEL_ANCHOR_MAX_MS2,
            )
            self._maybe_save(now)
            return

        log_anchor = math.log(self._accel_anchor_gain_ms2)
        residual = log_m - (log_anchor + x * log_ratio)

        weight = mean_pedal ** _WEIGHT_POWER
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
        brake_drift = (
            abs(self._brake_scale - self._saved_brake_scale)
            / max(self._saved_brake_scale, 0.01)
        )
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
                "pedal_capacity_brake_scale": round(self._brake_scale, 4),
                "pedal_capacity_accel_anchor_gain_ms2": round(self._accel_anchor_gain_ms2, 3),
                "pedal_capacity_accel_ratio_step": round(self._accel_ratio_step, 4),
            })
            self._saved_brake_scale = self._brake_scale
            self._saved_accel_anchor = self._accel_anchor_gain_ms2
            self._saved_accel_ratio = self._accel_ratio_step
            self._last_save_mono = now
            logger.debug(
                "pedal_capacity saved: brake_scale=%.4f accel_anchor=%.3f ratio=%.4f",
                self._brake_scale, self._accel_anchor_gain_ms2,
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

