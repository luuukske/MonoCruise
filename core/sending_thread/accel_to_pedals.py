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

# Weight / mass helpers
_REFERENCE_MASS_KG: float = 20_000.0
_WEIGHT_SPAN_TONS: float = 12.7
_WEIGHT_STRENGTH: float = 0.27
_WEIGHT_MIN_FACTOR: float = 0.55
_WEIGHT_MAX_FACTOR: float = 1.85
_TRAILER_WEIGHT_BIAS: float = 1.02

# Smoothing time constants
_WANTED_SMOOTHING_TAU_S: float = 0.05
_RAW_SMOOTHING_TAU_S: float = 0.10
_IDLE_CORRECTION_DECAY_TAU_S: float = 0.12

# Gas PID defaults (Settings can override Kp/Ki/Kd)
_GAS_KP: float = 0.25
_GAS_KI: float = 0.25
_GAS_KD: float = 0.15
_GAS_KI_CLAMP: float = 0.5          # pedal units
_GAS_DERIVATIVE_TAU_S: float = 0.12  # measurement derivative smoothing
_GAS_INTEGRAL_BRAKE_DECAY_TAU_S: float = 1.0  # integral leak while braking

# Brake feedforward curve constants — fitted from collected data
# DO NOT CHANGE WITHOUT VALID COLLECTED DATA
# y = 11.4596 * (1 - e^(-2.4277 * x^0.8518))
# where x = brake pedal [0,1], y = |decel| in m/s².
# Code uses the inverse: pedal from desired deceleration.
_BRAKE_CURVE_AMPLITUDE: float = 11.4596
_BRAKE_CURVE_RATE: float = 2.4277
_BRAKE_CURVE_POWER: float = 0.8518

# Brake trim PI (small corrections on top of feedforward)
_BRAKE_KP_TRIM: float = 0.03
_BRAKE_KI_TRIM: float = 0.08
_BRAKE_KI_TRIM_CLAMP: float = 0.12     # pedal units
_BRAKE_TRIM_DECAY_TAU_S: float = 0.8   # integral leak when not braking

# Slow brake multiplier — single EMA tracking truck-specific curve scaling
_BRAKE_MULTIPLIER_TAU_S: float = 8.0
_BRAKE_MULTIPLIER_MIN: float = 0.70
_BRAKE_MULTIPLIER_MAX: float = 1.40
_BRAKE_MULTIPLIER_SAMPLE_PEDAL: float = 0.08
_BRAKE_MULTIPLIER_SAMPLE_SPEED_MS: float = 5.0
_BRAKE_MULTIPLIER_MAX_SLOPE_RAD: float = 0.15

# Brake activation threshold — brake only engages for real deceleration
# demands, not CC hunting noise. Gas PID handles coasting by outputting ~0.
_BRAKE_ACTIVATION_MS2: float = 0.00  # wanted must be below -this to engage brake

# Road load
_ROAD_LOAD_SPEED_EPSILON_MS: float = 0.2
_MAX_ROAD_GRADE_RAD: float = 0.35  # ~20 deg — clamp pathological game values

# Gearshift handling
_GAME_CLUTCH_ACTIVE_THRESHOLD: float = 0.05
_GEARSHIFT_BLOCK_DURATION_S: float = 0.5
_GEARSHIFT_RECOVERY_TAU_S: float = 0.5

# Rate limiting
_GAS_RATE_LIMIT_PER_S: float = 3.0

# Capacity estimates (static baselines; capacity tracker can override)
_MIN_ACCEL_ESTIMATE_MS2: float = 0.8
_MAX_ACCEL_ESTIMATE_MS2: float = 5.0
_MIN_BRAKE_ESTIMATE_MS2: float = 2.0
_MAX_BRAKE_ESTIMATE_MS2: float = 10.0

# Debug logging
_DEBUG_LOG_NAME: str = "accel_to_pedals_debug.csv"
_DEBUG_LOG_HEADER_ROW: list[str] = [
    "t_s",
    "utc",
    "speed_ms",
    "gear",
    "gearshift_active",
    "pedal_state",
    "wanted_ms2",
    "wanted_smooth",
    "raw_ms2",
    "raw_smooth",
    "error_ms2",
    "road_load_ms2",
    "control_wanted_ms2",
    # Gas PID
    "gas_p",
    "gas_i",
    "gas_d",
    "gas_cmd",
    # Brake feedforward + trim
    "brake_ff",
    "brake_trim_p",
    "brake_trim_i",
    "brake_cmd",
    "brake_multiplier",
    # Meta
    "gain_scale",
    "game_throttle",
    "game_clutch",
    "est_accel_ms2",
    "est_brake_ms2",
    "slope_rad",
]
_DEBUG_LOG_INTERVAL_S: float = 0.10  # 10 Hz


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


def _weight_factor(total_mass_kg: float, has_trailer: bool) -> float:
    if not Settings.weight_adjustment:
        return 1.0
    current_tons = max(0.0, _finite_or_zero(total_mass_kg)) / 1000.0
    ref_tons = _REFERENCE_MASS_KG / 1000.0
    factor = 1.0 + (
        ((current_tons - ref_tons) / _WEIGHT_SPAN_TONS) * _WEIGHT_STRENGTH
    )
    factor = _clamp(factor, _WEIGHT_MIN_FACTOR, _WEIGHT_MAX_FACTOR)
    if has_trailer:
        factor = min(_WEIGHT_MAX_FACTOR, factor * _TRAILER_WEIGHT_BIAS)
    return factor


def baseline_accel_ms2(total_mass_kg: float, has_trailer: bool) -> float:
    """Expected max acceleration (m/s^2) at gas=1.0, adjusted for mass/trailer."""
    base = max(_MIN_ACCEL_ESTIMATE_MS2, _finite_or_zero(Settings.mapper_accel_scale_ms2))
    return _clamp(
        base / max(_weight_factor(total_mass_kg, has_trailer), 1e-6),
        _MIN_ACCEL_ESTIMATE_MS2,
        _MAX_ACCEL_ESTIMATE_MS2,
    )


def baseline_brake_ms2(total_mass_kg: float, has_trailer: bool) -> float:
    """Expected max deceleration (m/s^2) at brake=1.0, adjusted for mass/trailer."""
    base = max(_MIN_BRAKE_ESTIMATE_MS2, _finite_or_zero(Settings.mapper_brake_scale_ms2))
    return _clamp(
        base / max(_weight_factor(total_mass_kg, has_trailer), 1e-6),
        _MIN_BRAKE_ESTIMATE_MS2,
        _MAX_BRAKE_ESTIMATE_MS2,
    )


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
    return 0.0


# Pedal state labels (for debug logging only)
_STATE_GAS: int = 1
_STATE_BRAKE: int = 2
_STATE_NAMES: dict[int, str] = {_STATE_GAS: "GAS", _STATE_BRAKE: "BRAKE"}


@dataclass(slots=True)
class PedalTargets:
    gas: float
    brake: float
    command_gas: float = 0.0
    command_brake: float = 0.0
    slope_input_rad: float = 0.0
    effective_slope_rad: float = 0.0
    measured_control_ms2: float = 0.0
    road_load_ms2: float = 0.0
    control_wanted_ms2: float = 0.0
    wanted_smooth: float = 0.0
    raw_smooth: float = 0.0
    integral_correction: float = 0.0
    estimated_max_accel_ms2: float = 0.0
    estimated_max_brake_ms2: float = 0.0
    # New diagnostic fields
    gas_p: float = 0.0
    gas_i: float = 0.0
    gas_d: float = 0.0
    brake_ff: float = 0.0
    brake_trim_p: float = 0.0
    brake_trim_i: float = 0.0
    brake_multiplier: float = 1.0
    gain_scale: float = 1.0
    pedal_state: int = _STATE_GAS


class AccelToPedals:
    def __init__(self) -> None:
        # Smoothed signals
        self._wanted_smooth: float = 0.0
        self._raw_smooth: float = 0.0
        self._prev_mono: float | None = None

        # Gas PID state
        self._gas_integral: float = 0.0
        self._prev_raw_smooth: float = 0.0
        self._gas_deriv_smooth: float = 0.0
        self._prev_gas_cmd: float | None = None

        # Brake trim PI state
        self._brake_trim_integral: float = 0.0

        # Brake multiplier — slow EMA tracking truck-specific curve scaling
        self._brake_multiplier: float = 1.0

        # Capacity estimates
        self._estimated_max_accel_ms2: float | None = None
        self._estimated_max_brake_ms2: float | None = None

        # Gearshift state
        self._gearshift_start_mono: float | None = None
        self._integral_block_end_mono: float | None = None

        # Debug logging
        self._project_root = Path(__file__).resolve().parents[2]
        self._debug_log_file = None
        self._debug_log_writer = None
        self._last_debug_log_mono: float = 0.0
        self._debug_log_start_mono: float | None = None

    # Lifecycle

    def close(self) -> None:
        if self._debug_log_file is not None:
            try:
                self._debug_log_file.close()
            except OSError:
                pass
            self._debug_log_file = None
            self._debug_log_writer = None

    def reset_smoothing(self) -> None:
        self._wanted_smooth = 0.0
        self._raw_smooth = 0.0
        self._gas_integral = 0.0
        self._prev_raw_smooth = 0.0
        self._gas_deriv_smooth = 0.0
        self._prev_gas_cmd = None
        self._brake_trim_integral = 0.0
        self._prev_mono = None
        self._gearshift_start_mono = None
        self._integral_block_end_mono = None

    # Helpers — shared

    @staticmethod
    def _ema_step(current: float, sample: float, alpha: float) -> float:
        return current + alpha * (sample - current)

    @staticmethod
    def _ema_alpha(dt: float, tau_s: float) -> float:
        return 1.0 - math.exp(-dt / max(tau_s, 1e-6))

    @staticmethod
    def _motion_sign(speed_ms: float, wanted_accel_ms2: float, gear_dashboard: int) -> float:
        if abs(speed_ms) > _ROAD_LOAD_SPEED_EPSILON_MS:
            return 1.0 if speed_ms >= 0.0 else -1.0
        if abs(wanted_accel_ms2) > 1e-4:
            return 1.0 if wanted_accel_ms2 >= 0.0 else -1.0
        if gear_dashboard < 0:
            return -1.0
        return 1.0

    @staticmethod
    def _road_grade_from_norm(pitch: float) -> tuple[float, float]:
        """Convert normalized full circle [0.0, 1.0] to radians."""
        val = _finite_or_zero(pitch)
        # Handle wrap-around (e.g., 0.99 is a slight negative angle)
        val = (val + 0.5) % 1.0 - 0.5
        theta = val * 2.0 * math.pi
        return theta, _clamp(theta, -_MAX_ROAD_GRADE_RAD, _MAX_ROAD_GRADE_RAD)

    def _road_load_accel_ms2(
        self,
        speed_ms: float,
        wanted_accel_ms2: float,
        pitch: float,
        gear_dashboard: int,
    ) -> tuple[float, float, float]:
        motion_sign = self._motion_sign(speed_ms, wanted_accel_ms2, gear_dashboard)
        grade_unc_rad, grade_rad = self._road_grade_from_norm(pitch)
        rolling_coeff = max(0.0, _finite_or_zero(Settings.mapper_rolling_resistance))
        rolling_accel = motion_sign * rolling_coeff * GRAVITY_MS2 * math.cos(grade_rad)
        slope_accel = motion_sign * GRAVITY_MS2 * math.sin(grade_rad)
        return rolling_accel + slope_accel, grade_unc_rad, grade_rad

    @staticmethod
    def _measured_control_accel_ms2(raw_accel_ms2: float, road_load_ms2: float) -> float:
        return raw_accel_ms2 + road_load_ms2

    # Brake feedforward — inverse of the fitted curve

    @staticmethod
    def _brake_pedal_from_decel(decel_ms2: float) -> float:
        """Inverse of y = A * (1 - e^(-rate * x^power)) -> pedal x from decel y.

        Returns pedal in [0, 1].
        """
        if decel_ms2 <= 0.0:
            return 0.0
        A = _BRAKE_CURVE_AMPLITUDE
        rate = _BRAKE_CURVE_RATE
        power = _BRAKE_CURVE_POWER
        # Clamp to curve max
        ratio = min(decel_ms2 / A, 1.0 - 1e-9)
        inner = 1.0 - ratio
        if inner <= 0.0:
            return 1.0
        arg = -math.log(inner) / rate
        if arg <= 0.0:
            return 0.0
        return min(1.0, arg ** (1.0 / power))

    # Gearshift detection

    def _gearshift_update_factor(self, now: float, clutch: float) -> float:
        """Returns 0.0 during gearshift block, ramps to 1.0 after recovery."""
        now_safe = now if math.isfinite(now) else 0.0
        if clutch > _GAME_CLUTCH_ACTIVE_THRESHOLD:
            self._integral_block_end_mono = now_safe + _GEARSHIFT_BLOCK_DURATION_S
            self._gearshift_start_mono = now_safe
        elif self._gearshift_start_mono is not None:
            # Clutch released — block_end already set
            self._gearshift_start_mono = None

        if self._integral_block_end_mono is None:
            return 1.0
        if now_safe < self._integral_block_end_mono:
            return 0.0
        time_since_unblock = now_safe - self._integral_block_end_mono
        factor = 1.0 - math.exp(-time_since_unblock / _GEARSHIFT_RECOVERY_TAU_S)
        if factor > 0.99:
            self._integral_block_end_mono = None
            return 1.0
        return factor

    # Gas path — PID on acceleration error, outputs pedal directly

    def _gas_step(
        self,
        dt: float,
        wanted_smooth: float,
        raw_smooth: float,
        road_load_ms2: float,
        gain_scale: float,
        shift_factor: float,
        gear_dash: int,
    ) -> tuple[float, float, float, float]:
        """Returns (gas_pedal, p_term, i_term, d_term)."""
        # Acceleration error
        error_ms2 = wanted_smooth - raw_smooth

        # Effective gains scaled by mass ratio
        kp = _GAS_KP * gain_scale
        ki = _GAS_KI * gain_scale
        kd = _GAS_KD * gain_scale
        ki_clamp = _GAS_KI_CLAMP

        # Gravity feedforward: pedal offset to cancel road load on gas side.
        # road_load_ms2 is positive when resisting forward motion (uphill/rolling).
        # This provides the baseline throttle so the integral only handles trim —
        # making it react faster and stay closer to zero.
        ff_gravity = 0.0
        if self._estimated_max_accel_ms2 and self._estimated_max_accel_ms2 > 0.1:
            ff_gravity = (wanted_smooth + road_load_ms2) / self._estimated_max_accel_ms2

        # Proportional — zeroed during gearshift
        p_term = kp * error_ms2 * shift_factor

        # Integral — blocked during gearshift
        self._gas_integral += ki * error_ms2 * shift_factor * dt
        self._gas_integral = _clamp(self._gas_integral, -ki_clamp, ki_clamp)
        i_term = self._gas_integral

        # Derivative on measurement only (not error) to avoid setpoint kick
        deriv_alpha = self._ema_alpha(dt, _GAS_DERIVATIVE_TAU_S)
        deriv_raw = (raw_smooth - self._prev_raw_smooth) / max(dt, 1e-6)
        self._gas_deriv_smooth = self._ema_step(self._gas_deriv_smooth, deriv_raw, deriv_alpha)
        # Negate: if accel is increasing (positive deriv), we want less gas
        d_term = -kd * self._gas_deriv_smooth * shift_factor

        # Combine
        gas_pedal = ff_gravity + p_term + i_term + d_term
        gas_pedal = _clamp(gas_pedal, 0.0, 1.0)

        # Rate limit
        if self._prev_gas_cmd is not None:
            max_delta = _GAS_RATE_LIMIT_PER_S * dt
            gas_pedal = _clamp(
                gas_pedal,
                self._prev_gas_cmd - max_delta,
                self._prev_gas_cmd + max_delta,
            )
        self._prev_gas_cmd = gas_pedal

        # No gas in neutral
        if gear_dash == 0:
            gas_pedal = 0.0

        return gas_pedal, p_term, i_term, d_term

    # Brake path — feedforward from fitted curve + trim PI

    def _brake_step(
        self,
        dt: float,
        wanted_smooth: float,
        raw_smooth: float,
        road_load_ms2: float,
        shift_factor: float,
    ) -> tuple[float, float, float, float]:
        """Returns (brake_pedal, ff_pedal, trim_p, trim_i)."""
        # Desired braking effort after compensating physical road load.
        # Negative road load (downhill) increases required brake effort, while
        # positive road load (uphill/rolling drag) reduces it.
        wanted_decel = max(0.0, -(wanted_smooth + road_load_ms2))

        # Feedforward from fitted curve (with truck-specific multiplier)
        ff_pedal = self._brake_pedal_from_decel(wanted_decel) * self._brake_multiplier

        # Trim PI: correct small errors from curve inaccuracy / truck variance
        # Error: positive means we're not decelerating enough (need more brake)
        measured_decel = max(0.0, -(raw_smooth + road_load_ms2))
        brake_error = wanted_decel - measured_decel

        trim_p = _BRAKE_KP_TRIM * brake_error * shift_factor
        self._brake_trim_integral += _BRAKE_KI_TRIM * brake_error * shift_factor * dt
        self._brake_trim_integral = _clamp(
            self._brake_trim_integral, -_BRAKE_KI_TRIM_CLAMP, _BRAKE_KI_TRIM_CLAMP
        )
        trim_i = self._brake_trim_integral

        brake_pedal = _clamp(ff_pedal + trim_p + trim_i, 0.0, 1.0)
        return brake_pedal, ff_pedal, trim_p, trim_i

    # Brake multiplier learning

    def _update_brake_multiplier(
        self,
        dt: float,
        speed_ms: float,
        grade_unc_rad: float,
        road_load_ms2: float,
        brake_pedal: float,
        raw_smooth: float,
    ) -> None:
        """Slowly adapt brake curve multiplier from measured vs expected response."""
        if speed_ms < _BRAKE_MULTIPLIER_SAMPLE_SPEED_MS:
            return
        if brake_pedal < _BRAKE_MULTIPLIER_SAMPLE_PEDAL:
            return
        if abs(grade_unc_rad) > _BRAKE_MULTIPLIER_MAX_SLOPE_RAD:
            return

        # What the curve predicts for this pedal: forward function
        # y = A * (1 - e^(-rate * x^power))
        x = _clamp(brake_pedal / max(self._brake_multiplier, 0.01), 0.0, 1.0)
        predicted_decel = _BRAKE_CURVE_AMPLITUDE * (
            1.0 - math.exp(-_BRAKE_CURVE_RATE * (x ** _BRAKE_CURVE_POWER))
        )
        if predicted_decel < 0.5:
            return

        measured_decel = max(0.0, -raw_smooth + road_load_ms2)
        if measured_decel < 0.3:
            return

        ratio = measured_decel / predicted_decel
        ratio = _clamp(ratio, _BRAKE_MULTIPLIER_MIN, _BRAKE_MULTIPLIER_MAX)
        alpha = self._ema_alpha(dt, _BRAKE_MULTIPLIER_TAU_S)
        self._brake_multiplier = self._ema_step(self._brake_multiplier, ratio, alpha)
        self._brake_multiplier = _clamp(
            self._brake_multiplier, _BRAKE_MULTIPLIER_MIN, _BRAKE_MULTIPLIER_MAX
        )

    # Debug logging

    def _ensure_debug_log(self) -> None:
        if self._debug_log_file is not None:
            return
        path = self._project_root / _DEBUG_LOG_NAME
        write_header = False
        try:
            if path.exists():
                size = path.stat().st_size
                if size == 0:
                    write_header = True
                elif size > 0:
                    with path.open("r", encoding="utf-8-sig", newline="") as rf:
                        first = rf.readline()
                        tail = rf.read()
                    if first and not first.lstrip("\ufeff").startswith("t_s,"):
                        with path.open("w", newline="", encoding="utf-8") as wf:
                            w = csv.writer(wf)
                            w.writerow(_DEBUG_LOG_HEADER_ROW)
                            wf.write(first)
                            wf.write(tail)
            else:
                write_header = True

            self._debug_log_file = path.open("a", newline="", encoding="utf-8")
            self._debug_log_writer = csv.writer(self._debug_log_file)
            if write_header:
                self._debug_log_writer.writerow(_DEBUG_LOG_HEADER_ROW)
                self._debug_log_file.flush()
        except OSError:
            self._debug_log_file = None
            self._debug_log_writer = None
            logger.debug("accel_to_pedals debug log unavailable", exc_info=True)

    def _log_debug_step(
        self,
        *,
        now: float,
        speed_ms: float,
        gear: int,
        gearshift_active: bool,
        pedal_state: int,
        wanted_ms2: float,
        raw_ms2: float,
        error_ms2: float,
        road_load_ms2: float,
        control_wanted_ms2: float,
        gas_p: float,
        gas_i: float,
        gas_d: float,
        gas_cmd: float,
        brake_ff: float,
        brake_trim_p: float,
        brake_trim_i: float,
        brake_cmd: float,
        brake_multiplier: float,
        gain_scale: float,
        game_throttle: float,
        game_clutch: float,
        slope_rad: float,
    ) -> None:
        if now - self._last_debug_log_mono < _DEBUG_LOG_INTERVAL_S:
            return
        self._last_debug_log_mono = now
        if self._debug_log_start_mono is None:
            self._debug_log_start_mono = now
        t_s = now - self._debug_log_start_mono

        self._ensure_debug_log()
        if self._debug_log_writer is None:
            return
        try:
            self._debug_log_writer.writerow([
                f"{t_s:.3f}",
                datetime.now(timezone.utc).isoformat(),
                f"{speed_ms:.2f}",
                gear,
                int(gearshift_active),
                _STATE_NAMES.get(pedal_state, "?"),
                f"{wanted_ms2:.3f}",
                f"{self._wanted_smooth:.3f}",
                f"{raw_ms2:.3f}",
                f"{self._raw_smooth:.3f}",
                f"{error_ms2:.3f}",
                f"{road_load_ms2:.3f}",
                f"{control_wanted_ms2:.3f}",
                f"{gas_p:.4f}",
                f"{gas_i:.4f}",
                f"{gas_d:.4f}",
                f"{gas_cmd:.3f}",
                f"{brake_ff:.3f}",
                f"{brake_trim_p:.4f}",
                f"{brake_trim_i:.4f}",
                f"{brake_cmd:.3f}",
                f"{brake_multiplier:.3f}",
                f"{gain_scale:.3f}",
                f"{game_throttle:.3f}",
                f"{game_clutch:.3f}",
                f"{(self._estimated_max_accel_ms2 or 0.0):.3f}",
                f"{(self._estimated_max_brake_ms2 or 0.0):.3f}",
                f"{slope_rad:.4f}",
            ])
            self._debug_log_file.flush()
        except OSError:
            logger.debug("accel_to_pedals debug log write failed", exc_info=True)

    # Main step — called once per frame by sending_thread

    def step(
        self,
        wanted_accel_ms2: float,
        raw_accel_ms2: float,
        speed_ms: float,
        total_mass_kg: float,
        has_trailer: bool,
        *,
        max_accel_ms2: float = 0.0,
        max_brake_ms2: float = 0.0,
        cruise_commanding: bool = False,
        road_pitch: float = 0.0,
        gear_dashboard: int = 0,
        game_throttle: float = 0.0,
        game_clutch: float = 0.0,
    ) -> PedalTargets:
        gear_dash = int(_finite_or_zero(gear_dashboard))
        speed = max(0.0, _finite_or_zero(speed_ms))
        pitch = _finite_or_zero(road_pitch)
        throttle_applied = _clamp(_finite_or_zero(game_throttle), 0.0, 1.0)
        clutch_applied = _clamp(_finite_or_zero(game_clutch), 0.0, 1.0)
        now = math.nan
        try:
            now = time.monotonic()
        except Exception:
            pass

        # Time step
        if self._prev_mono is None or not math.isfinite(now):
            dt = 0.02
        else:
            dt = _clamp(now - self._prev_mono, 1e-4, 0.5)
        self._prev_mono = now if math.isfinite(now) else None

        # Smooth inputs
        wanted_alpha = self._ema_alpha(dt, _WANTED_SMOOTHING_TAU_S)
        raw_alpha = self._ema_alpha(dt, _RAW_SMOOTHING_TAU_S)
        wanted = _finite_or_zero(wanted_accel_ms2) if cruise_commanding else 0.0
        raw = _finite_or_zero(raw_accel_ms2)

        self._wanted_smooth = self._ema_step(self._wanted_smooth, wanted, wanted_alpha)
        self._raw_smooth = self._ema_step(self._raw_smooth, raw, raw_alpha)

        # Road load
        road_load_accel, grade_unc_rad, grade_rad = self._road_load_accel_ms2(
            speed, wanted, pitch, gear_dash
        )

        # Capacity estimates
        bl_accel = baseline_accel_ms2(total_mass_kg, has_trailer)
        bl_brake = baseline_brake_ms2(total_mass_kg, has_trailer)

        if not self._estimated_max_accel_ms2 or not math.isfinite(self._estimated_max_accel_ms2):
            self._estimated_max_accel_ms2 = bl_accel
        if not self._estimated_max_brake_ms2 or not math.isfinite(self._estimated_max_brake_ms2):
            self._estimated_max_brake_ms2 = bl_brake

        if max_accel_ms2 > 0.0 and math.isfinite(max_accel_ms2):
            self._estimated_max_accel_ms2 = max_accel_ms2 / max(
                _weight_factor(total_mass_kg, has_trailer), 1e-6
            )
        if max_brake_ms2 > 0.0 and math.isfinite(max_brake_ms2):
            self._estimated_max_brake_ms2 = max_brake_ms2

        # Gain scheduling by mass ratio
        mass_kg = max(1.0, _finite_or_zero(total_mass_kg))
        gain_scale = _REFERENCE_MASS_KG / mass_kg

        # Gearshift detection
        shift_factor = self._gearshift_update_factor(now, clutch_applied)
        gearshift_active = clutch_applied > _GAME_CLUTCH_ACTIVE_THRESHOLD

        # Control wanted (with road load compensation)
        control_wanted = self._wanted_smooth + road_load_accel if cruise_commanding else 0.0

        # Error for logging
        error_ms2 = self._raw_smooth - self._wanted_smooth
        if not math.isfinite(error_ms2):
            error_ms2 = 0.0

        # Default outputs
        gas_cmd = 0.0
        brake_cmd = 0.0
        gas_p = 0.0
        gas_i = 0.0
        gas_d = 0.0
        brake_ff = 0.0
        brake_trim_p = 0.0
        brake_trim_i = 0.0

        if cruise_commanding:
            # Brake only for real deceleration demands — not CC hunting noise.
            # Gas PID handles mild negatives by outputting 0 (coast).
            braking = control_wanted < -_BRAKE_ACTIVATION_MS2

            # Gas PID runs continuously — output clamped [0,1].
            # When wanted is slightly negative, PID naturally outputs ~0 gas
            # (negative P overwhelms FF → clamps at 0). This IS coasting, but
            # the integral and derivative stay continuous. No transition spike.
            #
            # Freeze gas integral during braking: gas output is zeroed while
            # braking, so integrating only accumulates stale negative offset
            # that suppresses gas for seconds after brake release.
            saved_integral = self._gas_integral
            gas_cmd, gas_p, gas_i, gas_d = self._gas_step(
                dt, self._wanted_smooth, self._raw_smooth,
                road_load_accel, gain_scale, shift_factor, gear_dash,
            )
            if braking:
                self._gas_integral = saved_integral * math.exp(-dt / _GAS_INTEGRAL_BRAKE_DECAY_TAU_S)
                gas_i = self._gas_integral

            if braking:
                brake_cmd, brake_ff, brake_trim_p, brake_trim_i = self._brake_step(
                    dt, self._wanted_smooth, self._raw_smooth,
                    road_load_accel, shift_factor,
                )
                self._update_brake_multiplier(
                    dt, speed, grade_unc_rad, road_load_accel,
                    brake_cmd, self._raw_smooth,
                )
                # Don't send gas while actively braking
                gas_cmd = 0.0
            else:
                # Decay brake trim integral when not braking
                decay = math.exp(-dt / _BRAKE_TRIM_DECAY_TAU_S)
                self._brake_trim_integral *= decay

            pedal_state = _STATE_BRAKE if braking else _STATE_GAS

        else:
            # Not commanding — decay all state
            gas_decay = math.exp(-dt / _IDLE_CORRECTION_DECAY_TAU_S)
            brake_decay = math.exp(-dt / _BRAKE_TRIM_DECAY_TAU_S)
            self._gas_integral *= gas_decay
            self._brake_trim_integral *= brake_decay
            self._prev_gas_cmd = None
            pedal_state = _STATE_GAS

        # Update prev_raw_smooth for derivative (always, even when not commanding)
        self._prev_raw_smooth = self._raw_smooth

        # Debug logging
        if cruise_commanding and math.isfinite(now):
            self._log_debug_step(
                now=now,
                speed_ms=speed,
                gear=gear_dash,
                gearshift_active=gearshift_active,
                pedal_state=pedal_state,
                wanted_ms2=wanted,
                raw_ms2=raw,
                error_ms2=error_ms2,
                road_load_ms2=road_load_accel,
                control_wanted_ms2=control_wanted,
                gas_p=gas_p,
                gas_i=gas_i,
                gas_d=gas_d,
                gas_cmd=gas_cmd,
                brake_ff=brake_ff,
                brake_trim_p=brake_trim_p,
                brake_trim_i=brake_trim_i,
                brake_cmd=brake_cmd,
                brake_multiplier=self._brake_multiplier,
                gain_scale=gain_scale,
                game_throttle=throttle_applied,
                game_clutch=clutch_applied,
                slope_rad=grade_unc_rad,
            )

        creep = idle_creep_brake(speed, gear_dash)
        return PedalTargets(
            gas=gas_cmd,
            brake=min(1.0, brake_cmd + creep),
            command_gas=gas_cmd,
            command_brake=brake_cmd,
            slope_input_rad=grade_unc_rad,
            effective_slope_rad=grade_rad,
            measured_control_ms2=self._measured_control_accel_ms2(
                self._raw_smooth, road_load_accel
            ),
            road_load_ms2=road_load_accel,
            control_wanted_ms2=control_wanted,
            wanted_smooth=self._wanted_smooth,
            raw_smooth=self._raw_smooth,
            integral_correction=self._gas_integral,
            estimated_max_accel_ms2=self._estimated_max_accel_ms2,
            estimated_max_brake_ms2=self._estimated_max_brake_ms2,
            gas_p=gas_p,
            gas_i=gas_i,
            gas_d=gas_d,
            brake_ff=brake_ff,
            brake_trim_p=brake_trim_p,
            brake_trim_i=brake_trim_i,
            brake_multiplier=self._brake_multiplier,
            gain_scale=gain_scale,
            pedal_state=pedal_state,
        )
