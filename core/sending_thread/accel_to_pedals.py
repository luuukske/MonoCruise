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

# Fast PID (unified effort trim)
_KP_FAST: float = 0.25
_KI_FAST: float = 0.25
_KD_FAST: float = 0.15
_FAST_I_CLAMP: float = 0.10         # pedal units — intentionally small
_FAST_OUT_CLAMP: float = 0.30       # total fast PID contribution cap
_FAST_DERIV_TAU_S: float = 0.12     # measurement derivative smoothing

# Slow integral (road load bias correction in m/s² space)
_KI_SLOW: float = 0.03
_SLOW_I_CLAMP_MS2: float = 2.0

# Brake feedforward curve constants — fitted from collected data
# DO NOT CHANGE WITHOUT VALID COLLECTED DATA
# y = A * (1 - e^(-rate * x^power))
# where x = brake pedal [0,1], y = |decel| in m/s².
# Code uses the inverse: pedal from desired deceleration.
_BRAKE_CURVE_RATE: float = 2.4277
_BRAKE_CURVE_POWER: float = 0.8518

# Road load
_ROAD_LOAD_SPEED_EPSILON_MS: float = 0.2
_MAX_ROAD_GRADE_RAD: float = 0.35  # ~20 deg — clamp pathological game values
# Aerodynamic drag coefficient fitted from coast-down data.
# decel_aero = _AERO_DRAG_ACCEL_PER_V2 * v^2  [m/s² per (m/s)²]
# Fit on ~41 t truck gave ~4.9e-5. Close enough for any mass — slow integral
# absorbs mass-driven residual.
_AERO_DRAG_ACCEL_PER_V2: float = 4.9e-5

# Gearshift handling
_GAME_CLUTCH_ACTIVE_THRESHOLD: float = 0.05
_GEARSHIFT_BLOCK_DURATION_S: float = 0.5
_GEARSHIFT_RAMP_DURATION_S: float = 1.0

# Rate limiting (gas only)
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
    "gearshift_factor",
    "pedal_state",
    "wanted_ms2",
    "wanted_smooth",
    "raw_ms2",
    "raw_smooth",
    "error_ms2",
    "road_load_ms2",
    "slow_integral_ms2",
    "effective_road_load_ms2",
    "ff",
    "fast_p",
    "fast_i",
    "fast_d",
    "fast_out",
    "effort",
    "gas_cmd",
    "brake_cmd",
    "gain_scale",
    "game_throttle",
    "game_clutch",
    "est_accel_ms2",
    "est_brake_ms2",
    "slope_rad",
    "capacity_used_ms2",
    "error_pedal",
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


# Pedal state labels (for debug logging / telemetry only — derived from effort sign)
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
    # Diagnostics — field names preserved for external consumers.
    # Semantic mapping to unified controller:
    #   gas_p/gas_i/gas_d → fast PID terms (used for gas OR brake, unified)
    #   brake_ff          → brake-side feedforward pedal magnitude (0 when effort>=0)
    #   brake_trim_p      → slow_integral in m/s² space (repurposed; no trim P anymore)
    #   brake_trim_i      → fast integral in pedal units (same as gas_i, kept for compat)
    #   brake_multiplier  → fixed 1.0 (deprecated; pedal_capacity.py handles this)
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

        # Unified fast PID state
        self._fast_integral: float = 0.0
        self._fast_deriv_smooth: float = 0.0
        self._prev_raw_smooth: float = 0.0
        self._prev_gas_cmd: float | None = None

        # Slow road load correction integral (m/s² space)
        self._slow_integral: float = 0.0

        # Capacity estimates
        self._estimated_max_accel_ms2: float | None = None
        self._estimated_max_brake_ms2: float | None = None

        # Gearshift freeze state
        self._clutch_active: bool = False
        self._clutch_release_mono: float = -math.inf
        self._frozen_raw_smooth: float = 0.0

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
        self._fast_integral = 0.0
        self._fast_deriv_smooth = 0.0
        self._prev_raw_smooth = 0.0
        self._prev_gas_cmd = None
        self._slow_integral = 0.0
        self._prev_mono = None
        self._clutch_active = False
        self._clutch_release_mono = -math.inf
        self._frozen_raw_smooth = 0.0

    # Helpers

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
        aero_accel = motion_sign * _AERO_DRAG_ACCEL_PER_V2 * speed_ms * speed_ms
        return rolling_accel + slope_accel + aero_accel, grade_unc_rad, grade_rad

    @staticmethod
    def _measured_control_accel_ms2(raw_accel_ms2: float, road_load_ms2: float) -> float:
        return raw_accel_ms2 + road_load_ms2

    # Brake feedforward — inverse of the fitted curve

    def _brake_pedal_from_decel(self, decel_ms2: float) -> float:
        """Inverse of y = A * (1 - e^(-rate * x^power)) -> pedal x from decel y.

        A = current estimated max brake capacity (m/s²). Returns pedal in [0, 1].
        """
        if decel_ms2 <= 0.0:
            return 0.0
        A = self._estimated_max_brake_ms2 or 0.0
        if A <= 0.1:
            return 0.0
        rate = _BRAKE_CURVE_RATE
        power = _BRAKE_CURVE_POWER
        ratio = min(decel_ms2 / A, 1.0 - 1e-9)
        inner = 1.0 - ratio
        if inner <= 0.0:
            return 1.0
        arg = -math.log(inner) / rate
        if arg <= 0.0:
            return 0.0
        return min(1.0, arg ** (1.0 / power))

    # Gearshift freeze/ramp

    def _gearshift_factor(self, now: float, clutch: float, raw_smooth_live: float) -> tuple[float, float]:
        """Update clutch freeze state. Returns (factor, effective_raw_smooth).

        factor: 0.0 during hard block/clutch, ramps 0→1 after release.
        effective_raw_smooth: frozen value (blended during ramp) to use for error.
        """
        now_safe = now if math.isfinite(now) else 0.0
        clutch_pressed = clutch > _GAME_CLUTCH_ACTIVE_THRESHOLD

        if clutch_pressed and not self._clutch_active:
            # Leading edge
            self._clutch_active = True
            self._frozen_raw_smooth = raw_smooth_live
        elif not clutch_pressed and self._clutch_active:
            # Trailing edge
            self._clutch_active = False
            self._clutch_release_mono = now_safe

        if clutch_pressed:
            return 0.0, self._frozen_raw_smooth

        time_since_release = now_safe - self._clutch_release_mono
        if time_since_release < _GEARSHIFT_BLOCK_DURATION_S:
            return 0.0, self._frozen_raw_smooth
        if time_since_release < _GEARSHIFT_BLOCK_DURATION_S + _GEARSHIFT_RAMP_DURATION_S:
            t = (time_since_release - _GEARSHIFT_BLOCK_DURATION_S) / _GEARSHIFT_RAMP_DURATION_S
            blended = self._frozen_raw_smooth + t * (raw_smooth_live - self._frozen_raw_smooth)
            return t, blended
        return 1.0, raw_smooth_live

    # Unified fast PID — trim on top of feedforward

    def _fast_pid_step(
        self,
        dt: float,
        error_ms2: float,
        raw_smooth_eff: float,
        gain_scale: float,
        factor: float,
    ) -> tuple[float, float, float, float, float]:
        """Returns (fast_out, p_pedal, i_pedal, d_pedal, capacity_used_ms2).

        Integrator is stored in pedal units directly so sign flips in error_ms2
        don't cause discontinuous output. Capacity only affects the m/s² → pedal
        conversion rate, not the stored state.
        """
        kp = _KP_FAST * gain_scale
        ki = _KI_FAST * gain_scale
        kd = _KD_FAST * gain_scale

        # Capacity picked by error sign — determines how much pedal per m/s² of error.
        if error_ms2 >= 0.0:
            capacity = self._estimated_max_accel_ms2 or 0.0
        else:
            capacity = self._estimated_max_brake_ms2 or 0.0
        capacity = max(capacity, 0.1)

        # Convert error to pedal-equivalent ONCE; all PID terms live in pedal space.
        error_pedal = error_ms2 / capacity

        p_pedal = kp * error_pedal * factor

        # Integral in pedal units — no representation shift across sign flips.
        self._fast_integral += ki * error_pedal * factor * dt
        self._fast_integral = _clamp(self._fast_integral, -_FAST_I_CLAMP, _FAST_I_CLAMP)
        i_pedal = self._fast_integral

        # Derivative on measurement (smoothed), converted to pedal units.
        deriv_alpha = self._ema_alpha(dt, _FAST_DERIV_TAU_S)
        deriv_raw = (raw_smooth_eff - self._prev_raw_smooth) / max(dt, 1e-6)
        self._fast_deriv_smooth = self._ema_step(self._fast_deriv_smooth, deriv_raw, deriv_alpha)
        d_pedal = -kd * (self._fast_deriv_smooth / capacity) * factor

        fast_out = _clamp(p_pedal + i_pedal + d_pedal, -_FAST_OUT_CLAMP, _FAST_OUT_CLAMP)
        return fast_out, p_pedal, i_pedal, d_pedal, capacity

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
        gearshift_factor: float,
        pedal_state: int,
        wanted_ms2: float,
        raw_ms2: float,
        error_ms2: float,
        road_load_ms2: float,
        slow_integral_ms2: float,
        effective_road_load_ms2: float,
        ff: float,
        fast_p: float,
        fast_i: float,
        fast_d: float,
        fast_out: float,
        effort: float,
        gas_cmd: float,
        brake_cmd: float,
        gain_scale: float,
        game_throttle: float,
        game_clutch: float,
        slope_rad: float,
        capacity_used_ms2: float,
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
                f"{gearshift_factor:.3f}",
                _STATE_NAMES.get(pedal_state, "?"),
                f"{wanted_ms2:.3f}",
                f"{self._wanted_smooth:.3f}",
                f"{raw_ms2:.3f}",
                f"{self._raw_smooth:.3f}",
                f"{error_ms2:.3f}",
                f"{road_load_ms2:.3f}",
                f"{slow_integral_ms2:.3f}",
                f"{effective_road_load_ms2:.3f}",
                f"{ff:+.4f}",
                f"{fast_p:+.4f}",
                f"{fast_i:+.4f}",
                f"{fast_d:+.4f}",
                f"{fast_out:+.4f}",
                f"{effort:+.4f}",
                f"{gas_cmd:.3f}",
                f"{brake_cmd:.3f}",
                f"{gain_scale:.3f}",
                f"{game_throttle:.3f}",
                f"{game_clutch:.3f}",
                f"{(self._estimated_max_accel_ms2 or 0.0):.3f}",
                f"{(self._estimated_max_brake_ms2 or 0.0):.3f}",
                f"{slope_rad:.4f}",
                f"{capacity_used_ms2:.3f}",
                f"{(error_ms2 / max(capacity_used_ms2, 0.1)):+.4f}",
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

        # Smooth wanted (always)
        wanted_alpha = self._ema_alpha(dt, _WANTED_SMOOTHING_TAU_S)
        wanted = _finite_or_zero(wanted_accel_ms2) if cruise_commanding else 0.0
        raw = _finite_or_zero(raw_accel_ms2)
        self._wanted_smooth = self._ema_step(self._wanted_smooth, wanted, wanted_alpha)

        # Compute live raw EMA candidate (but do not commit until we know freeze state)
        raw_alpha = self._ema_alpha(dt, _RAW_SMOOTHING_TAU_S)
        raw_smooth_live = self._ema_step(self._raw_smooth, raw, raw_alpha)

        # Gearshift freeze: determine factor and effective raw_smooth to use this step
        factor, raw_smooth_eff = self._gearshift_factor(now, clutch_applied, raw_smooth_live)

        # Commit raw_smooth: frozen/blended during gearshift, live otherwise
        if factor >= 1.0:
            self._raw_smooth = raw_smooth_live
        else:
            self._raw_smooth = raw_smooth_eff

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

        # Defaults
        effort = 0.0
        ff = 0.0
        fast_out = 0.0
        fast_p = 0.0
        fast_i = 0.0
        fast_d = 0.0
        capacity_used = 0.0
        gas_cmd = 0.0
        brake_cmd = 0.0
        effective_road_load = road_load_accel

        if cruise_commanding:
            error_ms2 = self._wanted_smooth - self._raw_smooth
            if not math.isfinite(error_ms2):
                error_ms2 = 0.0

            # Slow integral — m/s² space, frozen during gearshift, no decay
            self._slow_integral += _KI_SLOW * error_ms2 * factor * dt
            self._slow_integral = _clamp(
                self._slow_integral, -_SLOW_I_CLAMP_MS2, _SLOW_I_CLAMP_MS2
            )
            effective_road_load = road_load_accel + self._slow_integral

            # Feedforward (stateless)
            combined = self._wanted_smooth + effective_road_load
            if combined >= 0.0:
                max_a = max(self._estimated_max_accel_ms2 or 0.1, 0.1)
                ff = combined / max_a
            else:
                decel_needed = -combined
                ff = -self._brake_pedal_from_decel(decel_needed)

            # Fast PID
            fast_out, fast_p, fast_i, fast_d, capacity_used = self._fast_pid_step(
                dt, error_ms2, self._raw_smooth, gain_scale, factor,
            )

            effort = _clamp(ff + fast_out, -1.0, 1.0)

            gas_cmd = _clamp(effort, 0.0, 1.0)
            brake_cmd = _clamp(-effort, 0.0, 1.0)

            # Rate limit on gas only (brake must be immediate)
            if self._prev_gas_cmd is not None:
                max_delta = _GAS_RATE_LIMIT_PER_S * dt
                gas_cmd = _clamp(
                    gas_cmd,
                    self._prev_gas_cmd - max_delta,
                    self._prev_gas_cmd + max_delta,
                )
            self._prev_gas_cmd = gas_cmd

            # No gas in neutral
            if gear_dash == 0:
                gas_cmd = 0.0

        else:
            # Not commanding — hold slow integral, reset fast trim, clear prev gas
            self._fast_integral = 0.0
            self._fast_deriv_smooth = 0.0
            self._prev_gas_cmd = None

        # Update prev_raw_smooth for derivative (always)
        self._prev_raw_smooth = self._raw_smooth

        pedal_state = _STATE_BRAKE if effort < 0.0 else _STATE_GAS

        # Debug logging
        if cruise_commanding and math.isfinite(now):
            self._log_debug_step(
                now=now,
                speed_ms=speed,
                gear=gear_dash,
                gearshift_factor=factor,
                pedal_state=pedal_state,
                wanted_ms2=wanted,
                raw_ms2=raw,
                error_ms2=self._wanted_smooth - self._raw_smooth,
                road_load_ms2=road_load_accel,
                slow_integral_ms2=self._slow_integral,
                effective_road_load_ms2=effective_road_load,
                ff=ff,
                fast_p=fast_p,
                fast_i=fast_i,
                fast_d=fast_d,
                fast_out=fast_out,
                effort=effort,
                gas_cmd=gas_cmd,
                brake_cmd=brake_cmd,
                gain_scale=gain_scale,
                game_throttle=throttle_applied,
                game_clutch=clutch_applied,
                slope_rad=grade_unc_rad,
                capacity_used_ms2=capacity_used,
            )

        creep = idle_creep_brake(speed, gear_dash)
        # brake_ff diagnostic: magnitude of brake-side feedforward pedal
        brake_ff_diag = -ff if ff < 0.0 else 0.0

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
            control_wanted_ms2=self._wanted_smooth + effective_road_load if cruise_commanding else 0.0,
            wanted_smooth=self._wanted_smooth,
            raw_smooth=self._raw_smooth,
            integral_correction=self._fast_integral,
            estimated_max_accel_ms2=self._estimated_max_accel_ms2,
            estimated_max_brake_ms2=self._estimated_max_brake_ms2,
            gas_p=fast_p,
            gas_i=fast_i,
            gas_d=fast_d,
            brake_ff=brake_ff_diag,
            brake_trim_p=self._slow_integral,
            brake_trim_i=fast_i,
            brake_multiplier=1.0,
            gain_scale=gain_scale,
            pedal_state=pedal_state,
        )
