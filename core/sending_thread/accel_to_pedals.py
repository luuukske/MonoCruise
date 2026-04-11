from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.settings import Settings

logger = logging.getLogger(__name__)

FUEL_KG_PER_LITER: float = 0.832
GRAVITY_MS2: float = 9.81

_IDLE_CREEP_SPEED_SKIP_MS: float = 2.0

_REFERENCE_MASS_TONS: float = 20.0
_WEIGHT_SPAN_TONS: float = 12.7
_WEIGHT_STRENGTH: float = 0.27
_WEIGHT_MIN_FACTOR: float = 0.55
_WEIGHT_MAX_FACTOR: float = 1.85
_TRAILER_WEIGHT_BIAS: float = 1.02

_WANTED_SMOOTHING_TAU_S: float = 0.05
_RAW_SMOOTHING_TAU_S: float = 0.10
_INTEGRAL_LEAK_TAU_S: float = 8.0
_INTEGRAL_FAST_LEAK_TAU_S: float = 0.60
_DERIVATIVE_SMOOTHING_TAU_S: float = 0.12
_IDLE_CORRECTION_DECAY_TAU_S: float = 0.12

_ESTIMATE_DROP_TAU_S: float = 0.10
_ESTIMATE_RISE_TAU_S: float = 0.60
_MAX_ESTIMATE_SLOPE_RAD: float = 0.2
_MIN_SAMPLE_PEDAL: float = 0.35
_SAMPLE_PEDAL_FLOOR: float = 0.25

_MIN_ACCEL_SAMPLE_MS2: float = 1.2
_MIN_BRAKE_SAMPLE_MS2: float = 2.2
_MIN_ACCEL_SAMPLE_SPEED_MS: float = 3.0
_MIN_BRAKE_SAMPLE_SPEED_MS: float = 5.0
_MIN_ACCEL_SAMPLE_FRACTION: float = 0.35
_MIN_BRAKE_SAMPLE_FRACTION: float = 0.45

_MIN_ACCEL_ESTIMATE_MS2: float = 0.8
_MAX_ACCEL_ESTIMATE_MS2: float = 5.0
_MIN_BRAKE_ESTIMATE_MS2: float = 2.0
_MAX_BRAKE_ESTIMATE_MS2: float = 10.0

_TUNING_LOG_NAME: str = "accel_to_pedals_tuning.csv"
_INACCURACY_LOG_THRESHOLD_MS2: float = 0.75
_INACCURACY_LOG_COMMAND_THRESHOLD: float = 0.45
_INACCURACY_LOG_COOLDOWN_S: float = 0.75
_GAME_CLUTCH_ACTIVE_THRESHOLD: float = 0.05
_GAME_THROTTLE_MARGIN: float = 0.10
_ROAD_LOAD_SPEED_EPSILON_MS: float = 0.2
# Same convention as AEB elevation filter: telemetry rotationY is degrees (positive = uphill).
_MAX_ROAD_GRADE_RAD: float = 0.35  # ~20° — clamp only pathological game values

# Brake: hardware fit is acceleration magnitude vs pedal — y = 11.4596·(1 − e^(−2.4277·x^0.8518)),
# x = brake pedal [0, 1], y = |accel|. Code applies the inverse: pedal from normalized brake demand.
_BRAKE_MAP_RATE: float = 2.4277
_BRAKE_MAP_POWER: float = 0.8518

# Gas: same inverse when _GAS_MAP_RATE > 0 (pedal vs |accel| fit); rate 0 keeps linear until fitted.
_GAS_MAP_RATE: float = 0.0
_GAS_MAP_POWER: float = 1.0

# Pedal sensitivity learning — slow EMA tracks measured/expected response ratio independently
# per pedal.  Only sampled on flat roads, adequate speed, and while the standard integral is
# small (which guards against slope / mass transients polluting the signal).
_SENSITIVITY_TAU_S: float = 10.0
_MIN_SENSITIVITY: float = 0.60
_MAX_SENSITIVITY: float = 1.65
_SENSITIVITY_SAMPLE_PEDAL: float = 0.25      # min raw pedal to qualify for a sample
_SENSITIVITY_MIN_EXPECTED_MS2: float = 0.30  # ignore samples where expected response is tiny
_SENSITIVITY_MAX_INTEGRAL_GUARD: float = 0.15  # skip update when integral is actively correcting
_SENSITIVITY_SAVE_THRESHOLD: float = 0.04   # save immediately when value drifts this far
_SENSITIVITY_SAVE_COOLDOWN_S: float = 30.0  # minimum seconds between successive writes
_SENSITIVITY_STABLE_PERIOD_S: float = 60.0  # seconds between stability snapshots
_SENSITIVITY_STABLE_DELTA: float = 0.01     # max drift over stable period to count as settled


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _finite_or_zero(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def compute_estimated_mass_kg(
    unit_mass_kg: float,
    cargo_mass_kg: float,
    fuel_litres: float,
    fuel_kg_per_liter: float = FUEL_KG_PER_LITER,
    trailer_count: int = 0,
) -> float:
    """Tractor + cargo + fuel mass from telemetry (kg)."""
    fuel_kg = max(0.0, float(fuel_litres)) * float(fuel_kg_per_liter)
    trailer_mass_kg = max(0, int(trailer_count)) * 7000.0
    return max(0.0, float(unit_mass_kg)) + cargo_mass_kg + fuel_kg + trailer_mass_kg


def idle_creep_brake(speed_ms: float, gear_dashboard: int) -> float:
    """Light brake at very low speed to cancel idle creep. Disabled — to be replaced."""
    # Constant low-speed brake to offset idle creep / idle power (restore when successor exists).
    # if int(gear_dashboard) == 0:
    #     return 0.0
    # speed_abs = abs(float(speed_ms))
    # if speed_abs > _IDLE_CREEP_SPEED_SKIP_MS:
    #     return 0.0
    #
    # value = (
    #     1.0 - ((speed_abs + 0.1) / 5.5) ** 2.22
    # ) * (-0.6 / (speed_abs + 0.2) + 1.0)
    # num = max((value / 0.715) * 0.1, 0.0)
    # return max((num / 7.0) ** 2.5, 0.0)
    return 0.0


@dataclass(slots=True)
class PedalTargets:
    gas: float
    brake: float
    command_gas: float = 0.0
    command_brake: float = 0.0
    # Road grade from telemetry rotationY: unclamped radians (see AEB ego_pitch_rad).
    slope_input_rad: float = 0.0
    # Grade radians after clamp, used for sin/cos in road load.
    effective_slope_rad: float = 0.0
    measured_control_ms2: float = 0.0
    road_load_ms2: float = 0.0
    control_wanted_ms2: float = 0.0
    wanted_smooth: float = 0.0
    raw_smooth: float = 0.0
    integral_correction: float = 0.0
    estimated_max_accel_ms2: float = 0.0
    estimated_max_brake_ms2: float = 0.0
    accel_limited: bool = False
    accel_sensitivity: float = 1.0
    brake_sensitivity: float = 1.0


class AccelToPedals:
    def __init__(self) -> None:
        self._wanted_smooth: float = 0.0
        self._raw_smooth: float = 0.0
        self._integral_correction: float = 0.0
        self._prev_error_ms2: float = 0.0
        self._error_deriv_smooth: float = 0.0
        self._estimated_max_accel_ms2: float | None = None
        self._estimated_max_brake_ms2: float | None = None
        self._prev_mono: float | None = None
        self._log_file = None
        self._log_writer = None
        self._project_root = Path(__file__).resolve().parents[2]
        self._last_accel_log_mono: float = 0.0
        self._last_brake_log_mono: float = 0.0

        # Pedal sensitivity — loaded from persisted settings; default 1.0 (neutral).
        _sa = _finite_or_zero(Settings.mapper_accel_sensitivity)
        _sb = _finite_or_zero(Settings.mapper_brake_sensitivity)
        self._accel_sensitivity: float = _sa if _MIN_SENSITIVITY <= _sa <= _MAX_SENSITIVITY else 1.0
        self._brake_sensitivity: float = _sb if _MIN_SENSITIVITY <= _sb <= _MAX_SENSITIVITY else 1.0
        self._saved_accel_sensitivity: float = self._accel_sensitivity
        self._saved_brake_sensitivity: float = self._brake_sensitivity
        self._last_sensitivity_save_mono: float = 0.0
        self._sensitivity_stable_check_mono: float | None = None
        self._sensitivity_stable_snap_accel: float = self._accel_sensitivity
        self._sensitivity_stable_snap_brake: float = self._brake_sensitivity

    def close(self) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.close()
        except OSError:
            pass
        self._log_file = None
        self._log_writer = None

    def reset_smoothing(self) -> None:
        self._wanted_smooth = 0.0
        self._raw_smooth = 0.0
        self._integral_correction = 0.0
        self._prev_error_ms2 = 0.0
        self._error_deriv_smooth = 0.0
        self._prev_mono = None

    def _weight_factor(self, total_mass_kg: float, has_trailer: bool) -> float:
        if not Settings.weight_adjustment:
            return 1.0

        current_tons = max(0.0, _finite_or_zero(total_mass_kg)) / 1000.0
        factor = 1.0 + (
            ((current_tons - _REFERENCE_MASS_TONS) / _WEIGHT_SPAN_TONS) * _WEIGHT_STRENGTH
        )
        factor = _clamp(factor, _WEIGHT_MIN_FACTOR, _WEIGHT_MAX_FACTOR)
        if has_trailer:
            factor = min(_WEIGHT_MAX_FACTOR, factor * _TRAILER_WEIGHT_BIAS)
        return factor

    def _baseline_accel_estimate(self, total_mass_kg: float, has_trailer: bool) -> float:
        base = max(_MIN_ACCEL_ESTIMATE_MS2, _finite_or_zero(Settings.mapper_accel_scale_ms2))
        return _clamp(
            base / self._weight_factor(total_mass_kg, has_trailer),
            _MIN_ACCEL_ESTIMATE_MS2,
            _MAX_ACCEL_ESTIMATE_MS2,
        )

    def _baseline_brake_estimate(self, total_mass_kg: float, has_trailer: bool) -> float:
        base = max(_MIN_BRAKE_ESTIMATE_MS2, _finite_or_zero(Settings.mapper_brake_scale_ms2))
        return _clamp(
            base / self._weight_factor(total_mass_kg, has_trailer),
            _MIN_BRAKE_ESTIMATE_MS2,
            _MAX_BRAKE_ESTIMATE_MS2,
        )

    @staticmethod
    def _motion_sign(speed_ms: float, wanted_accel_ms2: float, gear_dashboard: int) -> float:
        if abs(speed_ms) > _ROAD_LOAD_SPEED_EPSILON_MS:
            return 1.0 if speed_ms >= 0.0 else -1.0
        if abs(wanted_accel_ms2) > 1e-4:
            return 1.0 if wanted_accel_ms2 >= 0.0 else -1.0
        if gear_dashboard < 0:
            return -1.0
        return 1.0

    def _road_grade_from_telemetry_deg(self, pitch_deg: float) -> tuple[float, float]:
        """rotationY in degrees (AEB convention) → (unclamped_rad, clamped_rad)."""
        theta = math.radians(_finite_or_zero(pitch_deg))
        return theta, _clamp(theta, -_MAX_ROAD_GRADE_RAD, _MAX_ROAD_GRADE_RAD)

    def _road_load_accel_ms2(
        self,
        speed_ms: float,
        wanted_accel_ms2: float,
        pitch_deg: float,
        gear_dashboard: int,
    ) -> tuple[float, float, float]:
        motion_sign = self._motion_sign(speed_ms, wanted_accel_ms2, gear_dashboard)
        grade_unc_rad, grade_rad = self._road_grade_from_telemetry_deg(pitch_deg)
        rolling_coeff = max(0.0, _finite_or_zero(Settings.mapper_rolling_resistance))
        rolling_accel = motion_sign * rolling_coeff * GRAVITY_MS2 * math.cos(grade_rad)
        slope_accel = motion_sign * GRAVITY_MS2 * math.sin(grade_rad)
        return rolling_accel + slope_accel, grade_unc_rad, grade_rad

    @staticmethod
    def _measured_control_accel_ms2(raw_accel_ms2: float, road_load_ms2: float) -> float:
        return raw_accel_ms2 + road_load_ms2

    @staticmethod
    def _ema_step(current: float, sample: float, alpha: float) -> float:
        return current + alpha * (sample - current)

    @staticmethod
    def _estimate_alpha(dt: float, tau_s: float) -> float:
        return 1.0 - math.exp(-dt / max(tau_s, 1e-6))

    @staticmethod
    def _pedal_from_linear_saturating_response(linear: float, rate: float, power: float) -> float:
        """Pedal position from normalized demand [0, 1] for y/A = 1 − e^(−rate·pedal^power).

        The fit has pedal on the x-axis and |accel| on the y-axis. Demand d maps to the same
        fraction of full response as at pedal = 1, i.e. d·(1−e^(−rate)) = 1−e^(−rate·pedal^power).
        ``rate <= 0`` selects linear passthrough.
        """
        if linear <= 0.0:
            return 0.0
        d = min(1.0, max(0.0, linear))
        if rate <= 0.0:
            return d
        pw = power if power > 0.0 else 1.0
        one_minus_e = 1.0 - math.exp(-rate)
        if one_minus_e <= 0.0:
            return d
        inner = 1.0 - d * one_minus_e
        if inner >= 1.0:
            return 0.0
        if inner <= 0.0:
            return 1.0
        arg = -math.log(inner) / rate
        if arg <= 0.0:
            return 0.0
        return min(1.0, arg ** (1.0 / pw))

    @staticmethod
    def _is_accel_control_limited(
        command_gas: float,
        game_throttle: float,
        game_clutch: float,
    ) -> bool:
        if command_gas < _MIN_SAMPLE_PEDAL:
            return False
        if game_clutch > _GAME_CLUTCH_ACTIVE_THRESHOLD:
            return True
        return game_throttle + _GAME_THROTTLE_MARGIN < command_gas

    def _ensure_log(self) -> None:
        if self._log_file is not None:
            return

        try:
            path = self._project_root / _TUNING_LOG_NAME
            new_file = not path.exists()
            self._log_file = path.open("a", newline="", encoding="utf-8")
            self._log_writer = csv.writer(self._log_file)
            if new_file:
                self._log_writer.writerow(
                    [
                        "utc",
                        "mode",
                        "speed_ms",
                        "mass_kg",
                        "wanted_ms2",
                        "raw_ms2",
                        "measured_control_ms2",
                        "gas_cmd",
                        "brake_cmd",
                        "slope_rad",
                        "effective_slope_rad",
                        "road_load_ms2",
                        "est_accel_ms2",
                        "est_brake_ms2",
                        "game_throttle",
                        "game_clutch",
                    ]
                )
                self._log_file.flush()
        except OSError:
            self._log_file = None
            self._log_writer = None
            logger.debug("accel_to_pedals tuning log unavailable", exc_info=True)

    def _log_inaccuracy(
        self,
        *,
        now: float,
        mode: str,
        speed_ms: float,
        total_mass_kg: float,
        command_gas: float,
        command_brake: float,
        grade_unc_rad: float,
        grade_rad: float,
        road_load_ms2: float,
        game_throttle: float,
        game_clutch: float,
    ) -> None:
        if mode == "accel":
            if now - self._last_accel_log_mono < _INACCURACY_LOG_COOLDOWN_S:
                return
            self._last_accel_log_mono = now
        else:
            if now - self._last_brake_log_mono < _INACCURACY_LOG_COOLDOWN_S:
                return
            self._last_brake_log_mono = now

        self._ensure_log()
        if self._log_writer is None:
            return

        try:
            self._log_writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    mode,
                    f"{speed_ms:.2f}",
                    f"{total_mass_kg:.0f}",
                    f"{self._wanted_smooth:.3f}",
                    f"{self._raw_smooth:.3f}",
                    f"{self._measured_control_accel_ms2(self._raw_smooth, road_load_ms2):.3f}",
                    f"{command_gas:.3f}",
                    f"{command_brake:.3f}",
                    f"{grade_unc_rad:.4f}",
                    f"{grade_rad:.4f}",
                    f"{road_load_ms2:.3f}",
                    f"{(self._estimated_max_accel_ms2 or 0.0):.3f}",
                    f"{(self._estimated_max_brake_ms2 or 0.0):.3f}",
                    f"{game_throttle:.3f}",
                    f"{game_clutch:.3f}",
                ]
            )
            self._log_file.flush()
        except OSError:
            logger.debug("accel_to_pedals tuning log write failed", exc_info=True)

    def _adapt_estimate(
        self,
        estimate: float,
        candidate: float,
        baseline: float,
        dt: float,
        minimum: float,
        maximum: float,
    ) -> float:
        low = max(minimum, baseline * 0.45)
        high = min(maximum, max(baseline * 1.85, low + 0.1))
        candidate = _clamp(candidate, low, high)
        tau_s = _ESTIMATE_DROP_TAU_S if candidate < estimate else _ESTIMATE_RISE_TAU_S
        alpha = self._estimate_alpha(dt, tau_s)
        return estimate + alpha * (candidate - estimate)

    def _maybe_update_accel_sensitivity(
        self,
        dt: float,
        speed_ms: float,
        grade_unc_rad: float,
        road_load_ms2: float,
        raw_gas: float,
    ) -> None:
        """Update accel sensitivity EMA from pre-sensitivity gas pedal vs measured response."""
        if abs(grade_unc_rad) > _MAX_ESTIMATE_SLOPE_RAD:
            return
        if speed_ms < _MIN_ACCEL_SAMPLE_SPEED_MS:
            return
        if raw_gas < _SENSITIVITY_SAMPLE_PEDAL:
            return
        if self._wanted_smooth < _MIN_ACCEL_SAMPLE_MS2 * 0.5:
            return
        if abs(self._integral_correction) > _SENSITIVITY_MAX_INTEGRAL_GUARD:
            return
        if self._estimated_max_accel_ms2 is None:
            return
        measured_accel = max(
            0.0, self._measured_control_accel_ms2(self._raw_smooth, road_load_ms2)
        )
        expected_accel = raw_gas * self._estimated_max_accel_ms2
        if expected_accel < _SENSITIVITY_MIN_EXPECTED_MS2:
            return
        ratio = _clamp(measured_accel / expected_accel, _MIN_SENSITIVITY, _MAX_SENSITIVITY)
        alpha = self._estimate_alpha(dt, _SENSITIVITY_TAU_S)
        self._accel_sensitivity += alpha * (ratio - self._accel_sensitivity)
        self._accel_sensitivity = _clamp(self._accel_sensitivity, _MIN_SENSITIVITY, _MAX_SENSITIVITY)

    def _maybe_update_brake_sensitivity(
        self,
        dt: float,
        speed_ms: float,
        grade_unc_rad: float,
        road_load_ms2: float,
        raw_brake: float,
    ) -> None:
        """Update brake sensitivity EMA from pre-sensitivity brake pedal vs measured response."""
        if abs(grade_unc_rad) > _MAX_ESTIMATE_SLOPE_RAD:
            return
        if speed_ms < _MIN_BRAKE_SAMPLE_SPEED_MS:
            return
        if raw_brake < _SENSITIVITY_SAMPLE_PEDAL:
            return
        if self._wanted_smooth > -_MIN_BRAKE_SAMPLE_MS2 * 0.5:
            return
        if abs(self._integral_correction) > _SENSITIVITY_MAX_INTEGRAL_GUARD:
            return
        if self._estimated_max_brake_ms2 is None:
            return
        measured_decel = max(
            0.0, -self._measured_control_accel_ms2(self._raw_smooth, road_load_ms2)
        )
        expected_decel = raw_brake * self._estimated_max_brake_ms2
        if expected_decel < _SENSITIVITY_MIN_EXPECTED_MS2:
            return
        ratio = _clamp(measured_decel / expected_decel, _MIN_SENSITIVITY, _MAX_SENSITIVITY)
        alpha = self._estimate_alpha(dt, _SENSITIVITY_TAU_S)
        self._brake_sensitivity += alpha * (ratio - self._brake_sensitivity)
        self._brake_sensitivity = _clamp(self._brake_sensitivity, _MIN_SENSITIVITY, _MAX_SENSITIVITY)

    def _maybe_save_sensitivity(self, now: float) -> None:
        """Persist sensitivity values to Settings when a large change or stability is detected."""
        if not math.isfinite(now):
            return
        # Bootstrap: record initial snapshot without triggering a save.
        if self._sensitivity_stable_check_mono is None:
            self._sensitivity_stable_check_mono = now
            self._sensitivity_stable_snap_accel = self._accel_sensitivity
            self._sensitivity_stable_snap_brake = self._brake_sensitivity
            return

        large_change = (
            abs(self._accel_sensitivity - self._saved_accel_sensitivity) > _SENSITIVITY_SAVE_THRESHOLD
            or abs(self._brake_sensitivity - self._saved_brake_sensitivity) > _SENSITIVITY_SAVE_THRESHOLD
        )
        stable = False
        if now - self._sensitivity_stable_check_mono >= _SENSITIVITY_STABLE_PERIOD_S:
            stable = (
                abs(self._accel_sensitivity - self._sensitivity_stable_snap_accel)
                <= _SENSITIVITY_STABLE_DELTA
                and abs(self._brake_sensitivity - self._sensitivity_stable_snap_brake)
                <= _SENSITIVITY_STABLE_DELTA
            )
            self._sensitivity_stable_snap_accel = self._accel_sensitivity
            self._sensitivity_stable_snap_brake = self._brake_sensitivity
            self._sensitivity_stable_check_mono = now

        anything_new = (
            abs(self._accel_sensitivity - self._saved_accel_sensitivity) > 1e-5
            or abs(self._brake_sensitivity - self._saved_brake_sensitivity) > 1e-5
        )
        if (
            anything_new
            and (large_change or stable)
            and now - self._last_sensitivity_save_mono >= _SENSITIVITY_SAVE_COOLDOWN_S
        ):
            try:
                Settings.save(values={
                    "mapper_accel_sensitivity": round(self._accel_sensitivity, 4),
                    "mapper_brake_sensitivity": round(self._brake_sensitivity, 4),
                })
            except Exception:
                logger.debug("pedal sensitivity save failed", exc_info=True)
            self._saved_accel_sensitivity = self._accel_sensitivity
            self._saved_brake_sensitivity = self._brake_sensitivity
            self._last_sensitivity_save_mono = now
            logger.debug(
                "pedal sensitivity saved: accel=%.3f brake=%.3f",
                self._accel_sensitivity,
                self._brake_sensitivity,
            )

    def _maybe_update_accel_estimate(
        self,
        dt: float,
        speed_ms: float,
        grade_unc_rad: float,
        road_load_ms2: float,
        wanted_ms2: float,
        command_gas: float,
        baseline_accel_ms2: float,
    ) -> None:
        estimate = self._estimated_max_accel_ms2
        if estimate is None:
            return
        if abs(grade_unc_rad) > _MAX_ESTIMATE_SLOPE_RAD:
            return
        if speed_ms < _MIN_ACCEL_SAMPLE_SPEED_MS:
            return
        if command_gas < _MIN_SAMPLE_PEDAL:
            return
        if wanted_ms2 < max(_MIN_ACCEL_SAMPLE_MS2, estimate * _MIN_ACCEL_SAMPLE_FRACTION):
            return

        measured_accel = max(
            0.0,
            self._measured_control_accel_ms2(self._raw_smooth, road_load_ms2),
        )
        if measured_accel <= 0.0:
            return

        candidate = measured_accel / max(command_gas, _SAMPLE_PEDAL_FLOOR)
        self._estimated_max_accel_ms2 = self._adapt_estimate(
            estimate,
            candidate,
            baseline_accel_ms2,
            dt,
            _MIN_ACCEL_ESTIMATE_MS2,
            _MAX_ACCEL_ESTIMATE_MS2,
        )

    def _maybe_update_brake_estimate(
        self,
        dt: float,
        speed_ms: float,
        grade_unc_rad: float,
        road_load_ms2: float,
        wanted_ms2: float,
        command_brake: float,
        baseline_brake_ms2: float,
    ) -> None:
        estimate = self._estimated_max_brake_ms2
        if estimate is None:
            return
        if abs(grade_unc_rad) > _MAX_ESTIMATE_SLOPE_RAD:
            return
        if speed_ms < _MIN_BRAKE_SAMPLE_SPEED_MS:
            return
        if command_brake < _MIN_SAMPLE_PEDAL:
            return
        if wanted_ms2 < max(_MIN_BRAKE_SAMPLE_MS2, estimate * _MIN_BRAKE_SAMPLE_FRACTION):
            return

        measured_brake = max(
            0.0,
            -self._measured_control_accel_ms2(self._raw_smooth, road_load_ms2),
        )
        if measured_brake <= 0.0:
            return

        candidate = measured_brake / max(command_brake, _SAMPLE_PEDAL_FLOOR)
        self._estimated_max_brake_ms2 = self._adapt_estimate(
            estimate,
            candidate,
            baseline_brake_ms2,
            dt,
            _MIN_BRAKE_ESTIMATE_MS2,
            _MAX_BRAKE_ESTIMATE_MS2,
        )

    def step(
        self,
        wanted_accel_ms2: float,
        raw_accel_ms2: float,
        speed_ms: float,
        total_mass_kg: float,
        has_trailer: bool,
        *,
        cruise_commanding: bool = False,
        road_pitch_deg: float = 0.0,
        gear_dashboard: int = 0,
        game_throttle: float = 0.0,
        game_clutch: float = 0.0,
    ) -> PedalTargets:
        gear_dash = int(_finite_or_zero(gear_dashboard))
        speed = max(0.0, _finite_or_zero(speed_ms))
        pitch_deg = _finite_or_zero(road_pitch_deg)
        throttle_applied = _clamp(_finite_or_zero(game_throttle), 0.0, 1.0)
        clutch_applied = _clamp(_finite_or_zero(game_clutch), 0.0, 1.0)
        now = math.nan
        try:
            now = time.monotonic()
        except Exception:
            pass

        if self._prev_mono is None or not math.isfinite(now):
            dt = 0.02
        else:
            dt = _clamp(now - self._prev_mono, 1e-4, 0.5)
        self._prev_mono = now if math.isfinite(now) else None

        wanted_alpha = self._estimate_alpha(dt, _WANTED_SMOOTHING_TAU_S)
        raw_alpha = self._estimate_alpha(dt, _RAW_SMOOTHING_TAU_S)
        wanted = _finite_or_zero(wanted_accel_ms2) if cruise_commanding else 0.0
        raw = _finite_or_zero(raw_accel_ms2)
        road_load_accel, grade_unc_rad, grade_rad = self._road_load_accel_ms2(
            speed, wanted, pitch_deg, gear_dash
        )

        self._wanted_smooth = self._ema_step(self._wanted_smooth, wanted, wanted_alpha)
        self._raw_smooth = self._ema_step(self._raw_smooth, raw, raw_alpha)

        baseline_accel_ms2 = self._baseline_accel_estimate(total_mass_kg, has_trailer)
        baseline_brake_ms2 = self._baseline_brake_estimate(total_mass_kg, has_trailer)

        if not self._estimated_max_accel_ms2 or not math.isfinite(self._estimated_max_accel_ms2):
            self._estimated_max_accel_ms2 = baseline_accel_ms2
        if not self._estimated_max_brake_ms2 or not math.isfinite(self._estimated_max_brake_ms2):
            self._estimated_max_brake_ms2 = baseline_brake_ms2

        if not cruise_commanding:
            decay = math.exp(-dt / _IDLE_CORRECTION_DECAY_TAU_S)
            self._integral_correction *= decay
            base_signed = 0.0
            control_wanted = 0.0
        else:
            control_wanted = self._wanted_smooth + road_load_accel
            if control_wanted >= 0.0:
                base_signed = _clamp(
                    control_wanted / max(self._estimated_max_accel_ms2, 1e-6),
                    0.0,
                    1.0,
                )
            else:
                base_signed = -_clamp(
                    (-control_wanted) / max(self._estimated_max_brake_ms2, 1e-6),
                    0.0,
                    1.0,
                )

        error_ms2 = self._raw_smooth - self._wanted_smooth
        if not math.isfinite(error_ms2):
            error_ms2 = 0.0

        deriv_alpha = self._estimate_alpha(dt, _DERIVATIVE_SMOOTHING_TAU_S)
        if cruise_commanding:
            error_deriv_raw = (error_ms2 - self._prev_error_ms2) / max(dt, 1e-6)
            self._prev_error_ms2 = error_ms2
            self._error_deriv_smooth = self._ema_step(
                self._error_deriv_smooth, error_deriv_raw, deriv_alpha
            )
        else:
            self._prev_error_ms2 = error_ms2
            self._error_deriv_smooth = self._ema_step(self._error_deriv_smooth, 0.0, deriv_alpha)
        deriv_coeff = _finite_or_zero(Settings.mapper_derivative_coeff)
        derivative_correction = deriv_coeff * self._error_deriv_smooth if cruise_commanding else 0.0

        integral_coeff = _finite_or_zero(Settings.mapper_integral_coeff)
        integral_clamp = max(0.0, _finite_or_zero(Settings.mapper_integral_clamp))
        # Nonlinear integral: tanh compresses the integrand so the correction is
        # highly sensitive near zero error but accumulates more slowly when far off.
        # For |error| << scale behaviour is indistinguishable from linear;
        # for |error| >> scale the contribution is capped at ±scale per second.
        ni_scale = max(_finite_or_zero(Settings.mapper_integral_nonlinear_scale), 0.05)
        integral_input = math.tanh(error_ms2 / ni_scale) * ni_scale

        sign_mismatch = (self._integral_correction > 0.0 and error_ms2 < 0.0) or (
            self._integral_correction < 0.0 and error_ms2 > 0.0
        )
        leak_tau = _INTEGRAL_FAST_LEAK_TAU_S if sign_mismatch else _INTEGRAL_LEAK_TAU_S
        leak = math.exp(-dt / leak_tau)
        accel_limited = cruise_commanding and base_signed > 0.0 and self._is_accel_control_limited(
            base_signed,
            throttle_applied,
            clutch_applied,
        )
        self._integral_correction *= leak
        if cruise_commanding and not accel_limited:
            self._integral_correction += integral_coeff * integral_input * dt
        self._integral_correction = _clamp(
            self._integral_correction,
            -integral_clamp,
            integral_clamp,
        )

        drive_cmd = _clamp(
            base_signed - self._integral_correction - derivative_correction,
            -1.0,
            1.0,
        )
        if gear_dash == 0 and drive_cmd > 0.0:
            drive_cmd = 0.0

        # Raw pedal demand (before sensitivity correction) — used as the sensitivity error signal.
        raw_gas = self._pedal_from_linear_saturating_response(
            max(0.0, drive_cmd),
            _GAS_MAP_RATE,
            _GAS_MAP_POWER,
        )
        raw_brake = self._pedal_from_linear_saturating_response(
            max(0.0, -drive_cmd),
            _BRAKE_MAP_RATE,
            _BRAKE_MAP_POWER,
        )

        # Update sensitivity integrals from pre-correction pedal vs measured response.
        if cruise_commanding:
            if not accel_limited:
                self._maybe_update_accel_sensitivity(
                    dt, speed, grade_unc_rad, road_load_accel, raw_gas
                )
            self._maybe_update_brake_sensitivity(
                dt, speed, grade_unc_rad, road_load_accel, raw_brake
            )
        self._maybe_save_sensitivity(now if math.isfinite(now) else 0.0)

        # Apply sensitivity: scale up pedal when response is weaker than modelled,
        # scale down when stronger.  Capability estimates use the corrected pedal
        # so they stay consistent with what the game actually receives.
        command_gas = _clamp(raw_gas / max(self._accel_sensitivity, _MIN_SENSITIVITY), 0.0, 1.0)
        command_brake = _clamp(raw_brake / max(self._brake_sensitivity, _MIN_SENSITIVITY), 0.0, 1.0)

        if cruise_commanding:
            if not accel_limited:
                self._maybe_update_accel_estimate(
                    dt,
                    speed,
                    grade_unc_rad,
                    road_load_accel,
                    max(0.0, self._wanted_smooth),
                    command_gas,
                    baseline_accel_ms2,
                )
            self._maybe_update_brake_estimate(
                dt,
                speed,
                grade_unc_rad,
                road_load_accel,
                max(0.0, -self._wanted_smooth),
                command_brake,
                baseline_brake_ms2,
            )

        if (
            cruise_commanding
            and speed >= _MIN_ACCEL_SAMPLE_SPEED_MS
            and abs(grade_unc_rad) <= _MAX_ESTIMATE_SLOPE_RAD
            and command_gas >= _INACCURACY_LOG_COMMAND_THRESHOLD
            and self._wanted_smooth > 0.0
            and (self._wanted_smooth - self._raw_smooth) > _INACCURACY_LOG_THRESHOLD_MS2
            and not accel_limited
        ):
            self._log_inaccuracy(
                now=now if math.isfinite(now) else 0.0,
                mode="accel",
                speed_ms=speed,
                total_mass_kg=_finite_or_zero(total_mass_kg),
                command_gas=command_gas,
                command_brake=command_brake,
                grade_unc_rad=grade_unc_rad,
                grade_rad=grade_rad,
                road_load_ms2=road_load_accel,
                game_throttle=throttle_applied,
                game_clutch=clutch_applied,
            )
        if (
            cruise_commanding
            and speed >= _MIN_BRAKE_SAMPLE_SPEED_MS
            and abs(grade_unc_rad) <= _MAX_ESTIMATE_SLOPE_RAD
            and command_brake >= _INACCURACY_LOG_COMMAND_THRESHOLD
            and self._wanted_smooth < 0.0
            and (self._raw_smooth - self._wanted_smooth) > _INACCURACY_LOG_THRESHOLD_MS2
        ):
            self._log_inaccuracy(
                now=now if math.isfinite(now) else 0.0,
                mode="brake",
                speed_ms=speed,
                total_mass_kg=_finite_or_zero(total_mass_kg),
                command_gas=command_gas,
                command_brake=command_brake,
                grade_unc_rad=grade_unc_rad,
                grade_rad=grade_rad,
                road_load_ms2=road_load_accel,
                game_throttle=throttle_applied,
                game_clutch=clutch_applied,
            )

        creep = idle_creep_brake(speed, gear_dash)
        return PedalTargets(
            gas=command_gas,
            brake=min(1.0, command_brake + creep),
            command_gas=command_gas,
            command_brake=command_brake,
            slope_input_rad=grade_unc_rad,
            effective_slope_rad=grade_rad,
            measured_control_ms2=self._measured_control_accel_ms2(
                self._raw_smooth, road_load_accel
            ),
            road_load_ms2=road_load_accel,
            control_wanted_ms2=control_wanted if cruise_commanding else 0.0,
            wanted_smooth=self._wanted_smooth,
            raw_smooth=self._raw_smooth,
            integral_correction=self._integral_correction,
            estimated_max_accel_ms2=self._estimated_max_accel_ms2,
            estimated_max_brake_ms2=self._estimated_max_brake_ms2,
            accel_limited=accel_limited,
            accel_sensitivity=self._accel_sensitivity,
            brake_sensitivity=self._brake_sensitivity,
        )
