from __future__ import annotations

"""Commanded m/s² to gas/brake mapping. See core/sending_thread/README.md."""

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

# Weight / mass helpers
_REFERENCE_MASS_KG: float = 20_000.0
_WEIGHT_SPAN_TONS: float = 12.7
_WEIGHT_STRENGTH: float = 0.27
_WEIGHT_MIN_FACTOR: float = 0.55
_WEIGHT_MAX_FACTOR: float = 1.85
_TRAILER_WEIGHT_BIAS: float = 1.02
# Brake capability, fitted to full-pedal stops across four rigs. Mass exponent
# is 0.25, not 1: air brakes are load-sensed (README has the numbers).
_BRAKE_FIT_COEFF: float = 70.8
_BRAKE_WHEEL_EXP: float = 0.52
_BRAKE_MASS_EXP: float = 0.31
# Fallbacks, used only when the wheel count is unavailable.
_BRAKE_LOAD_MULT_SOLO: float = 1.47
_BRAKE_LOAD_MULT_TRAILER: float = 2.18

# Smoothing time constants
_WANTED_SMOOTHING_TAU_S: float = 0.05
_RAW_SMOOTHING_TAU_S: float = 0.10
_OUTPUT_SMOOTHING_TAU_S: float = 0.08
_OUTPUT_SMOOTHING_DELTA_REF_MS2: float = 1.0

# Fast PID (unified trim, m/s² space): injected into `combined` upstream of the FF
_KP_FAST: float = 0.25
_KI_FAST: float = 0.25
_KD_FAST: float = 0.15
_FAST_I_CLAMP_MS2: float = 1.5      # m/s²
_FAST_DERIV_TAU_S: float = 0.12     # measurement derivative smoothing

# Low-gear D-term attenuation. At gear 1-2 the engine has a 300-400 ms response
_LOW_GEAR_KD_SCALE: float = 0.5
_LOW_GEAR_KD_MAX_GEAR: int = 2

# Slow integral (road load bias correction in m/s² space).
_KI_SLOW: float = 0.15
# Measured road-load error is around 0.5 m/s2, so +-1.0 is double the headroom
# the correction actually needs. The old +-2.0 mostly bounded windup, not bias.
_SLOW_I_CLAMP_MS2: float = 1.0
# Anti-windup. A railed pedal has no authority left, so tracking error that
_SLOW_I_SAT_EFFORT: float = 0.95
# Unwinding runs faster than winding: bias written in during a saturated climb
_KI_SLOW_UNWIND_MULT: float = 3.0
# Initial bias: overestimates resistance so the limiter approaches the set speed
# conservatively on first engagement. The integral corrects this within a few seconds.
_SLOW_I_INIT_BIAS_MS2: float = -0.4
# Stationary gate: at speeds below this, freeze slow_integral negative
_STATIONARY_SLOW_I_GATE_SPEED_MS: float = 0.5

# When the hold FSM owns standstill (freeze_slow_i=True), bleed slow_integral
_HOLD_SLOW_I_LEAK_TAU_S: float = 10.0

# Brake feedforward curve constants: fitted from collected data
_BRAKE_CURVE_RATE: float = 2.4277
_BRAKE_CURVE_POWER: float = 0.8518

# Road load
_ROAD_LOAD_SMOOTH_TAU_S: float = 1.2   # slow EMA for small bumps
_ROAD_LOAD_DELTA_REF_MS2: float = 0.25  # above this → fast tracking
_ROAD_LOAD_SPEED_EPSILON_MS: float = 0.2
_MAX_ROAD_GRADE_RAD: float = 0.35  # clamp pathological game values
# Aerodynamic drag coefficient fitted from coast-down data.
_AERO_DRAG_ACCEL_PER_V2: float = 4.9e-5

# Idle-creep feedforward. With the driveline closed the idle governor
_CREEP_MAX_REF_MS2: float = 2.343   # m/s² at 20 t, below the knee
_CREEP_REVERSE_SCALE: float = 0.5   # R1 magnitude / D1 magnitude (tunable)
_CREEP_KNEE_SPEED_MS: float = 0.55  # governor saturation below this
_CREEP_ZERO_SPEED_MS: float = 2.54  # idle-match speed: creep reaches 0

# Gearshift handling. While the clutch is open (plus a short block after
_GAME_CLUTCH_ACTIVE_THRESHOLD: float = 0.05
_GEARSHIFT_BLOCK_DURATION_S: float = 0.2
_GEARSHIFT_RAMP_DURATION_S: float = 0.5

# Gas-capacity glide across gearshifts. The per-gear capacity from
_CAPACITY_GLIDE_RISE_TAU_S: float = 1.0
_CAPACITY_GLIDE_FALL_TAU_S: float = 0.3

# Rate limiting (gas only)
_GAS_RATE_LIMIT_PER_S: float = 3.0

# Capacity estimates (static baselines; capacity tracker can override)
_MIN_ACCEL_ESTIMATE_MS2: float = 0.8
_MAX_ACCEL_ESTIMATE_MS2: float = 5.0
_MIN_BRAKE_ESTIMATE_MS2: float = 2.0
_MAX_BRAKE_ESTIMATE_MS2: float = 20.0

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


def weight_factor(total_mass_kg: float, has_trailer: bool) -> float:
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
        base / max(weight_factor(total_mass_kg, has_trailer), 1e-6),
        _MIN_ACCEL_ESTIMATE_MS2,
        _MAX_ACCEL_ESTIMATE_MS2,
    )


def baseline_brake_ms2(
    total_mass_kg: float, has_trailer: bool, wheels_on_ground: int = 0,
) -> float:
    """Expected max deceleration (m/s^2) at brake=1.0 for this rig. See README."""
    mass = _finite_or_zero(total_mass_kg)
    if wheels_on_ground > 0 and mass > 1000.0:
        # Braked axles vs mass. Never `weight_factor`: that is the acceleration
        # model, where more mass means less accel. Braking runs the other way.
        return _clamp(
            _BRAKE_FIT_COEFF
            * wheels_on_ground ** _BRAKE_WHEEL_EXP
            * mass ** -_BRAKE_MASS_EXP,
            _MIN_BRAKE_ESTIMATE_MS2,
            _MAX_BRAKE_ESTIMATE_MS2,
        )
    base = max(_MIN_BRAKE_ESTIMATE_MS2, _finite_or_zero(Settings.mapper_brake_scale_ms2))
    mult = _BRAKE_LOAD_MULT_TRAILER if has_trailer else _BRAKE_LOAD_MULT_SOLO
    return _clamp(base * mult, _MIN_BRAKE_ESTIMATE_MS2, _MAX_BRAKE_ESTIMATE_MS2)


def brake_curve_fraction(pedal: float) -> float:
    """Fraction of max brake capacity the game delivers at *pedal* [0-1]. See `core/sending_thread/README.md`."""
    x = _clamp(_finite_or_zero(pedal), 0.0, 1.0)
    if x <= 0.0:
        return 0.0
    return 1.0 - math.exp(-_BRAKE_CURVE_RATE * x ** _BRAKE_CURVE_POWER)


def compute_estimated_mass_kg(
    unit_mass_kg: float,
    cargo_mass_kg: float,
    fuel_litres: float,
    fuel_kg_per_liter: float = FUEL_KG_PER_LITER,
    trailer_count: int = 0,
) -> float:
    """Tractor + cargo + fuel mass from telemetry (kg).

    Cargo only counts while a trailer is attached: the SDK keeps reporting the
    assigned job's cargoMass after you drop the trailer, which made a bobtail
    read 39.8 t instead of ~10.7 t (see core/sending_thread/README.md).
    """
    trailers = max(0, int(trailer_count))
    fuel_kg = max(0.0, float(fuel_litres)) * float(fuel_kg_per_liter)
    trailer_mass_kg = trailers * 7000.0
    carried_cargo = max(0.0, float(cargo_mass_kg)) if trailers > 0 else 0.0
    return max(0.0, float(unit_mass_kg)) + carried_cargo + fuel_kg + trailer_mass_kg


def creep_accel_ms2(
    speed_ms: float,
    gear_dashboard: int,
    game_clutch: float,
    gain_scale: float,
) -> float:
    """Idle-creep along-gear accel magnitude (m/s²) with the driveline closed. See `core/sending_thread/README.md`."""
    gear = int(gear_dashboard)
    if abs(gear) != 1:
        return 0.0
    v = abs(_finite_or_zero(speed_ms))
    if v >= _CREEP_ZERO_SPEED_MS:
        return 0.0
    if v <= _CREEP_KNEE_SPEED_MS:
        shape = 1.0
    else:
        shape = (_CREEP_ZERO_SPEED_MS - v) / (_CREEP_ZERO_SPEED_MS - _CREEP_KNEE_SPEED_MS)
    clutch_closed = _clamp(1.0 - _finite_or_zero(game_clutch), 0.0, 1.0)
    reverse_scale = _CREEP_REVERSE_SCALE if gear < 0 else 1.0
    return (
        _CREEP_MAX_REF_MS2
        * shape
        * clutch_closed
        * max(_finite_or_zero(gain_scale), 0.0)
        * reverse_scale
    )


# Pedal state labels (for debug logging / telemetry only: derived from effort sign)
_STATE_GAS: int = 1
_STATE_BRAKE: int = 2
_STATE_NAMES: dict[int, str] = {_STATE_GAS: "GAS", _STATE_BRAKE: "BRAKE"}


@dataclass(slots=True)
class MapperSharedState:
    """State shared across multiple AccelToPedals instances. See `core/sending_thread/README.md`."""
    # Time
    prev_mono: float | None = None

    # Effective measured-accel EMA: frozen/blended across gearshifts.
    # This is what error computations and external consumers see.
    raw_smooth: float = 0.0
    # Always-live measured-accel EMA (never frozen): gearshift snapshot
    # source, blend target after the shift, and D-term derivative source.
    raw_smooth_live: float = 0.0
    # Previous live EMA value: D-term derivative source.
    prev_raw_smooth: float = 0.0

    # Road load EMA (physics)
    road_load_smooth: float = 0.0

    # Slow road-load bias correction integral (m/s² space).
    slow_integral: float = _SLOW_I_INIT_BIAS_MS2

    # Output m/s² EMA: continuity across handover.
    output_smooth_ms2: float | None = None

    # Gas rate-limit memory: continuity across handover.
    prev_gas_cmd: float | None = None

    # Signed effort committed on the previous tick (pedal units, gas positive).
    prev_effort: float = 0.0

    # Gas-capacity glide state (m/s² at gas=1.0): the capacity actually used
    accel_capacity_glide_ms2: float | None = None

    # Capacity estimates (single source of truth, fed in from PedalCapacityTracker).
    estimated_max_accel_ms2: float | None = None
    estimated_max_brake_ms2: float | None = None

    # Gearshift freeze (truck-level).
    clutch_active: bool = False
    clutch_release_mono: float = -math.inf
    frozen_raw_smooth: float = 0.0


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
    # Diagnostics: field names preserved for external consumers.
    gas_p: float = 0.0
    gas_i: float = 0.0
    gas_d: float = 0.0
    brake_ff: float = 0.0
    brake_trim_p: float = 0.0
    brake_trim_i: float = 0.0
    brake_multiplier: float = 1.0
    gain_scale: float = 1.0
    pedal_state: int = _STATE_GAS
    # Idle-creep FF term (m/s², >= 0). Reported even when not commanding so
    # the capacity tracker can subtract it from manual-driving gas samples.
    creep_ms2: float = 0.0
    # True when the pure-FF gas pedal alone saturates: the commanded accel
    ff_saturated: bool = False


class AccelToPedals:
    def __init__(self, shared: MapperSharedState | None = None) -> None:
        # Shared state: owned externally when multiple mappers must share
        self._shared: MapperSharedState = shared if shared is not None else MapperSharedState()

        # Per-instance smoothed signals (controller-specific)
        self._wanted_smooth: float = 0.0

        # Per-instance fast PID state (this controller's tracking error trim)
        self._fast_integral: float = 0.0
        self._fast_deriv_smooth: float = 0.0

        # One-shot: commander handover re-seeds the wanted EMA on the next
        # step instead of slewing across two commanders' unrelated setpoints.
        self._pending_wanted_snap: bool = False

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

    def set_shared_state(self, shared: MapperSharedState) -> None:
        """Swap in an externally-owned shared state (for multi-mapper handover)."""
        self._shared = shared

    def reset_smoothing(self) -> None:
        # Per-instance
        self._wanted_smooth = 0.0
        self._fast_integral = 0.0
        self._fast_deriv_smooth = 0.0
        # Shared
        s = self._shared
        s.raw_smooth = 0.0
        s.raw_smooth_live = 0.0
        s.prev_raw_smooth = 0.0
        s.output_smooth_ms2 = None
        s.prev_gas_cmd = None
        s.prev_effort = 0.0
        s.accel_capacity_glide_ms2 = None
        s.slow_integral = _SLOW_I_INIT_BIAS_MS2
        s.prev_mono = None
        s.road_load_smooth = 0.0
        s.clutch_active = False
        s.clutch_release_mono = -math.inf
        s.frozen_raw_smooth = 0.0

    def handover_reseed(self, applied_gas: float | None) -> None:
        """Commander handover (cc <-> limiter <-> none): one-shot bumpless See `core/sending_thread/README.md`."""
        self._pending_wanted_snap = True
        if applied_gas is None:
            self._shared.prev_gas_cmd = None
        else:
            self._shared.prev_gas_cmd = _clamp(_finite_or_zero(applied_gas), 0.0, 1.0)

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

    def _adaptive_output_ema_step(self, current: float | None, sample: float, dt: float) -> float:
        if current is None or not math.isfinite(current):
            return sample
        base_alpha = self._ema_alpha(dt, _OUTPUT_SMOOTHING_TAU_S)
        delta_ratio = _clamp(
            abs(sample - current) / max(_OUTPUT_SMOOTHING_DELTA_REF_MS2, 1e-6),
            0.0,
            1.0,
        )
        alpha = base_alpha + (1.0 - base_alpha) * (delta_ratio ** 3)
        return self._ema_step(current, sample, alpha)

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

    # Brake feedforward: inverse of the fitted curve

    def _brake_pedal_from_decel(
        self,
        decel_ms2: float,
        max_brake_ms2_override: float | None = None,
    ) -> float:
        """Inverse of y = A * (1 - e^(-rate * x^power)) -> pedal x from decel y. See `core/sending_thread/README.md`."""
        if decel_ms2 <= 0.0:
            return 0.0
        if max_brake_ms2_override is not None:
            A = max_brake_ms2_override
        else:
            A = self._shared.estimated_max_brake_ms2 or 0.0
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

    def brake_pedal_from_decel(
        self,
        decel_ms2: float,
        max_brake_ms2_override: float | None = None,
    ) -> float:
        """Public wrapper around the inverse brake curve. See `core/sending_thread/README.md`."""
        if max_brake_ms2_override is None and not (
            self._shared.estimated_max_brake_ms2
            and math.isfinite(self._shared.estimated_max_brake_ms2)
        ):
            max_brake_ms2_override = max(
                _MIN_BRAKE_ESTIMATE_MS2,
                _finite_or_zero(Settings.mapper_brake_scale_ms2),
            )
        return self._brake_pedal_from_decel(decel_ms2, max_brake_ms2_override)

    def brake_decel_from_pedal(
        self,
        pedal: float,
        max_brake_ms2_override: float | None = None,
    ) -> float:
        """Forward brake curve: decel (m/s²) the game produces at *pedal*. See `core/sending_thread/README.md`."""
        x = _clamp(_finite_or_zero(pedal), 0.0, 1.0)
        if x <= 0.0:
            return 0.0
        if max_brake_ms2_override is not None:
            A = max_brake_ms2_override
        elif (
            self._shared.estimated_max_brake_ms2
            and math.isfinite(self._shared.estimated_max_brake_ms2)
        ):
            A = self._shared.estimated_max_brake_ms2
        else:
            A = max(
                _MIN_BRAKE_ESTIMATE_MS2,
                _finite_or_zero(Settings.mapper_brake_scale_ms2),
            )
        if A <= 0.1:
            return 0.0
        return A * brake_curve_fraction(x)

    # Gearshift freeze/ramp

    def _gearshift_factor(
        self,
        now: float,
        clutch: float,
        raw_smooth_live: float,
        learn: bool,
    ) -> tuple[float, float]:
        """Track clutch state. Returns (factor, effective_raw_smooth). See `core/sending_thread/README.md`."""
        s = self._shared
        now_safe = now if math.isfinite(now) else 0.0
        clutch_pressed = clutch > _GAME_CLUTCH_ACTIVE_THRESHOLD

        if learn:
            if clutch_pressed and not s.clutch_active:
                # Leading edge: snapshot the live measurement so error keeps
                s.clutch_active = True
                s.frozen_raw_smooth = raw_smooth_live
            elif not clutch_pressed and s.clutch_active:
                # Trailing edge
                s.clutch_active = False
                s.clutch_release_mono = now_safe

        if clutch_pressed:
            if not s.clutch_active:
                # Preview-only caller mid-clutch, snapshot never armed: the
                # stored frozen value is stale, use live instead.
                return 0.0, raw_smooth_live
            return 0.0, s.frozen_raw_smooth

        time_since_release = now_safe - s.clutch_release_mono
        if time_since_release < _GEARSHIFT_BLOCK_DURATION_S:
            return 0.0, s.frozen_raw_smooth
        if time_since_release < _GEARSHIFT_BLOCK_DURATION_S + _GEARSHIFT_RAMP_DURATION_S:
            t = (time_since_release - _GEARSHIFT_BLOCK_DURATION_S) / _GEARSHIFT_RAMP_DURATION_S
            blended = s.frozen_raw_smooth + t * (raw_smooth_live - s.frozen_raw_smooth)
            return t, blended
        return 1.0, raw_smooth_live

    # Unified fast PID: trim on top of feedforward

    def _fast_pid_compute(
        self,
        dt: float,
        error_ms2: float,
        raw_live: float,
        prev_raw_live: float,
        gain_scale: float,
        factor: float,
        active: bool,
        prev_fast_integral: float,
        prev_fast_deriv_smooth: float,
        kd_scale: float = 1.0,
    ) -> tuple[float, float, float, float, float, float]:
        """Pure compute: returns (fast_trim_ms2, p_ms2, i_ms2, d_ms2, See `core/sending_thread/README.md`."""
        if not active:
            return 0.0, 0.0, 0.0, 0.0, prev_fast_integral, prev_fast_deriv_smooth

        kp = _KP_FAST * gain_scale
        ki = _KI_FAST * gain_scale
        kd = _KD_FAST * gain_scale * kd_scale

        p_ms2 = kp * error_ms2

        new_fast_integral = _clamp(
            prev_fast_integral + ki * error_ms2 * factor * dt,
            -_FAST_I_CLAMP_MS2, _FAST_I_CLAMP_MS2,
        )
        i_ms2 = new_fast_integral

        new_deriv_smooth = prev_fast_deriv_smooth
        if factor >= 1.0:
            deriv_alpha = self._ema_alpha(dt, _FAST_DERIV_TAU_S)
            deriv_raw = (raw_live - prev_raw_live) / max(dt, 1e-6)
            new_deriv_smooth = self._ema_step(prev_fast_deriv_smooth, deriv_raw, deriv_alpha)
        d_ms2 = -kd * new_deriv_smooth

        return p_ms2 + i_ms2 + d_ms2, p_ms2, i_ms2, d_ms2, new_fast_integral, new_deriv_smooth

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
        wanted_smooth: float,
        raw_ms2: float,
        raw_smooth: float,
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
        est_accel_ms2: float,
        est_brake_ms2: float,
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
                f"{wanted_smooth:.3f}",
                f"{raw_ms2:.3f}",
                f"{raw_smooth:.3f}",
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
                f"{est_accel_ms2:.3f}",
                f"{est_brake_ms2:.3f}",
                f"{slope_rad:.4f}",
                f"{capacity_used_ms2:.3f}",
                f"{(error_ms2 / max(capacity_used_ms2, 0.1)):+.4f}",
            ])
            self._debug_log_file.flush()
        except OSError:
            logger.debug("accel_to_pedals debug log write failed", exc_info=True)

    # Main step: called once per frame by sending_thread

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
        learn: bool = True,
        freeze_trim: bool = False,
        freeze_slow_i: bool = False,
        cap_mode: bool = False,
    ) -> PedalTargets:
        """Compute pedal targets for one tick. See `core/sending_thread/README.md`."""
        s = self._shared

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

        # Time step (read from shared)
        if s.prev_mono is None or not math.isfinite(now):
            dt = 0.02
        else:
            dt = _clamp(now - s.prev_mono, 1e-4, 0.5)
        new_prev_mono = now if math.isfinite(now) else None

        # === Compute everything to local "new_*" vars; commit at end if learn ===

        # Wanted smooth (per-instance)
        wanted_alpha = self._ema_alpha(dt, _WANTED_SMOOTHING_TAU_S)
        wanted = _finite_or_zero(wanted_accel_ms2) if cruise_commanding else 0.0
        raw = _finite_or_zero(raw_accel_ms2)
        if self._pending_wanted_snap:
            new_wanted_smooth = wanted
            self._pending_wanted_snap = False
        else:
            new_wanted_smooth = self._ema_step(self._wanted_smooth, wanted, wanted_alpha)

        # Live raw EMA (shared, never frozen): physics truth. The gearshift
        raw_alpha = self._ema_alpha(dt, _RAW_SMOOTHING_TAU_S)
        new_raw_smooth_live = self._ema_step(s.raw_smooth_live, raw, raw_alpha)

        # Gearshift: truck-level physics tracking, edges always committed
        factor, new_raw_smooth = self._gearshift_factor(
            now, clutch_applied, new_raw_smooth_live, learn=True,
        )

        # Road load: adaptive EMA: slow for small bumps, fast for steep hills
        road_load_raw, grade_unc_rad, grade_rad = self._road_load_accel_ms2(
            speed, wanted, pitch, gear_dash
        )
        rl_base_alpha = self._ema_alpha(dt, _ROAD_LOAD_SMOOTH_TAU_S)
        rl_delta_ratio = _clamp(
            abs(road_load_raw - s.road_load_smooth) / max(_ROAD_LOAD_DELTA_REF_MS2, 1e-6),
            0.0, 1.0,
        )
        rl_alpha = rl_base_alpha + (1.0 - rl_base_alpha) * (rl_delta_ratio ** 2)
        new_road_load_smooth = self._ema_step(s.road_load_smooth, road_load_raw, rl_alpha)
        road_load_accel = new_road_load_smooth

        # Capacity estimates
        bl_accel = baseline_accel_ms2(total_mass_kg, has_trailer)
        bl_brake = baseline_brake_ms2(total_mass_kg, has_trailer)

        if s.estimated_max_accel_ms2 and math.isfinite(s.estimated_max_accel_ms2):
            new_max_accel = s.estimated_max_accel_ms2
        else:
            new_max_accel = bl_accel
        if s.estimated_max_brake_ms2 and math.isfinite(s.estimated_max_brake_ms2):
            new_max_brake = s.estimated_max_brake_ms2
        else:
            new_max_brake = bl_brake

        # The caller passes a per-gear accel gain stored in mass-normalized
        if max_accel_ms2 > 0.0 and math.isfinite(max_accel_ms2):
            new_max_accel = max_accel_ms2 / max(
                weight_factor(total_mass_kg, has_trailer), 1e-6
            )
        if max_brake_ms2 > 0.0 and math.isfinite(max_brake_ms2):
            new_max_brake = max_brake_ms2

        # Gain scheduling by mass ratio
        mass_kg = max(1.0, _finite_or_zero(total_mass_kg))
        gain_scale = _REFERENCE_MASS_KG / mass_kg

        # Idle-creep FF term: along-gear accel the engine provides for free in
        creep_ms2 = creep_accel_ms2(speed, gear_dash, clutch_applied, gain_scale)

        # Defaults
        effort = 0.0
        ff = 0.0
        ff_saturated = False
        fast_out = 0.0
        fast_p = 0.0
        fast_i = 0.0
        fast_d = 0.0
        capacity_used = 0.0
        gas_cmd = 0.0
        brake_cmd = 0.0
        effective_road_load = road_load_accel

        # Local copies of stateful quantities: start from current, mutate locally
        new_slow_integral = s.slow_integral
        new_fast_integral = self._fast_integral
        new_fast_deriv_smooth = self._fast_deriv_smooth
        new_output_smooth_ms2 = s.output_smooth_ms2
        new_prev_gas_cmd = s.prev_gas_cmd
        new_accel_capacity_glide = s.accel_capacity_glide_ms2

        if cruise_commanding:
            error_ms2 = new_wanted_smooth - new_raw_smooth
            if not math.isfinite(error_ms2):
                error_ms2 = 0.0

            # Slow integral: m/s² space, frozen during gearshift, no decay.
            if freeze_slow_i:
                leak = math.exp(-dt / max(_HOLD_SLOW_I_LEAK_TAU_S, 1e-6))
                new_slow_integral = new_slow_integral * leak
            else:
                slow_i_gate = (
                    0.0
                    if (speed < _STATIONARY_SLOW_I_GATE_SPEED_MS and error_ms2 < 0.0)
                    else 1.0
                )
                # Anti-windup: the previous tick's pedal was railed, so error
                if (
                    (s.prev_effort >= _SLOW_I_SAT_EFFORT and error_ms2 > 0.0)
                    or (s.prev_effort <= -_SLOW_I_SAT_EFFORT and error_ms2 < 0.0)
                ):
                    slow_i_gate = 0.0
                # Asymmetric rate: error opposing the stored bias sheds it at
                ki_slow = _KI_SLOW
                if new_slow_integral * error_ms2 < 0.0:
                    ki_slow *= _KI_SLOW_UNWIND_MULT
                new_slow_integral = _clamp(
                    new_slow_integral + ki_slow * error_ms2 * factor * dt * slow_i_gate,
                    -_SLOW_I_CLAMP_MS2, _SLOW_I_CLAMP_MS2,
                )
            # Creep subtracts from the effective road load: the brake FF must
            # overcome it and the gas FF must not double-provide it.
            effective_road_load = road_load_accel + new_slow_integral - creep_ms2

            # Fast trim in m/s² space. Output stays continuous through
            kd_scale = (
                _LOW_GEAR_KD_SCALE
                if 1 <= gear_dash <= _LOW_GEAR_KD_MAX_GEAR
                else 1.0
            )
            fast_trim_ms2, fast_p, fast_i, fast_d, new_fast_integral, new_fast_deriv_smooth = (
                self._fast_pid_compute(
                    dt,
                    error_ms2,
                    raw_live=new_raw_smooth_live,
                    prev_raw_live=s.prev_raw_smooth,
                    gain_scale=gain_scale,
                    factor=factor,
                    active=not freeze_trim,
                    prev_fast_integral=new_fast_integral,
                    prev_fast_deriv_smooth=new_fast_deriv_smooth,
                    kd_scale=kd_scale,
                )
            )

            # Gas capacity glide: frozen while the driveline is open (also
            cap_target = max(new_max_accel or 0.1, 0.1)
            if (
                new_accel_capacity_glide is None
                or not math.isfinite(new_accel_capacity_glide)
            ):
                new_accel_capacity_glide = cap_target
            elif factor > 0.0:
                glide_tau = (
                    _CAPACITY_GLIDE_RISE_TAU_S
                    if cap_target > new_accel_capacity_glide
                    else _CAPACITY_GLIDE_FALL_TAU_S
                )
                new_accel_capacity_glide = self._ema_step(
                    new_accel_capacity_glide,
                    cap_target,
                    self._ema_alpha(dt, glide_tau),
                )
            max_a_use = max(new_accel_capacity_glide, 0.1)
            max_b_use = max(new_max_brake or 0.1, 0.1)

            # Pure-FF pedal for diagnostics (what the mapping would give with no trim)
            combined_ff_only = new_wanted_smooth + effective_road_load
            if combined_ff_only >= 0.0:
                ff = combined_ff_only / max_a_use
            else:
                ff = -self._brake_pedal_from_decel(-combined_ff_only, max_b_use)
            # Gas-side FF saturation = unsatisfiable bid (headroom cap).
            ff_saturated = combined_ff_only > max_a_use

            # Unified mapping: FF + fast trim through the same capacity scaling.
            combined = combined_ff_only + fast_trim_ms2
            combined = self._adaptive_output_ema_step(new_output_smooth_ms2, combined, dt)
            new_output_smooth_ms2 = combined
            if combined >= 0.0:
                unclamped_effort = combined / max_a_use
                capacity_used = max_a_use
            else:
                unclamped_effort = -self._brake_pedal_from_decel(-combined, max_b_use)
                capacity_used = max_b_use
            effort = _clamp(unclamped_effort, -1.0, 1.0)
            ff_saturated = ff_saturated or effort >= 1.0

            # Back-calc anti-windup: if FF saturates the pedal, snap fast_integral
            if (
                effort != unclamped_effort
                and not freeze_trim
                and not (cap_mode and effort >= 1.0)
                and factor > 0.0
            ):
                if effort >= 1.0:
                    combined_sat = max_a_use
                else:
                    combined_sat = -max_b_use
                desired_fast_sum = combined_sat - combined_ff_only
                new_fast_integral = _clamp(
                    desired_fast_sum - fast_p - fast_d,
                    -_FAST_I_CLAMP_MS2,
                    _FAST_I_CLAMP_MS2,
                )
                fast_i = new_fast_integral

            # Diagnostic: pedal-units delta contributed by fast trim (post-clamp).
            fast_out = effort - ff

            gas_cmd = _clamp(effort, 0.0, 1.0)
            brake_cmd = _clamp(-effort, 0.0, 1.0)

            # Rate limit on gas only (brake must be immediate)
            if new_prev_gas_cmd is not None:
                max_delta = _GAS_RATE_LIMIT_PER_S * dt
                gas_cmd = _clamp(
                    gas_cmd,
                    new_prev_gas_cmd - max_delta,
                    new_prev_gas_cmd + max_delta,
                )
            new_prev_gas_cmd = gas_cmd

            # No gas in neutral. prev_gas_cmd keeps the pre-cut trajectory so
            # gas returns to it immediately once a gear is engaged.
            if gear_dash == 0:
                gas_cmd = 0.0

        else:
            # Not commanding: drop output-trajectory state; integrals hold so
            # re-engagement doesn't overshoot
            new_prev_gas_cmd = None
            new_output_smooth_ms2 = None
            new_accel_capacity_glide = None

        pedal_state = _STATE_BRAKE if effort < 0.0 else _STATE_GAS

        # === Commit gating ===
        self._wanted_smooth = new_wanted_smooth
        self._fast_deriv_smooth = new_fast_deriv_smooth
        s.estimated_max_accel_ms2 = new_max_accel
        s.estimated_max_brake_ms2 = new_max_brake
        s.prev_mono = new_prev_mono
        s.raw_smooth = new_raw_smooth
        s.raw_smooth_live = new_raw_smooth_live
        s.prev_raw_smooth = new_raw_smooth_live
        s.road_load_smooth = new_road_load_smooth
        s.output_smooth_ms2 = new_output_smooth_ms2
        s.prev_gas_cmd = new_prev_gas_cmd
        s.prev_effort = effort
        s.accel_capacity_glide_ms2 = new_accel_capacity_glide

        # Learning: integrator state only. learn=False freezes adaptation
        # (accumulation and anti-windup snaps) while someone else drives.
        if learn:
            self._fast_integral = new_fast_integral
            s.slow_integral = new_slow_integral

        # Debug logging: only the commanding mapper logs
        if learn and cruise_commanding and math.isfinite(now):
            self._log_debug_step(
                now=now,
                speed_ms=speed,
                gear=gear_dash,
                gearshift_factor=factor,
                pedal_state=pedal_state,
                wanted_ms2=wanted,
                wanted_smooth=new_wanted_smooth,
                raw_ms2=raw,
                raw_smooth=new_raw_smooth,
                error_ms2=new_wanted_smooth - new_raw_smooth,
                road_load_ms2=road_load_accel,
                slow_integral_ms2=new_slow_integral,
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
                est_accel_ms2=new_max_accel or 0.0,
                est_brake_ms2=new_max_brake or 0.0,
            )

        # brake_ff diagnostic: magnitude of brake-side feedforward pedal
        brake_ff_diag = -ff if ff < 0.0 else 0.0

        return PedalTargets(
            gas=gas_cmd,
            brake=brake_cmd,
            command_gas=gas_cmd,
            command_brake=brake_cmd,
            slope_input_rad=grade_unc_rad,
            effective_slope_rad=grade_rad,
            measured_control_ms2=self._measured_control_accel_ms2(
                new_raw_smooth, road_load_accel
            ),
            road_load_ms2=road_load_accel,
            control_wanted_ms2=new_wanted_smooth + effective_road_load if cruise_commanding else 0.0,
            wanted_smooth=new_wanted_smooth,
            raw_smooth=new_raw_smooth,
            integral_correction=new_fast_integral,
            estimated_max_accel_ms2=new_max_accel,
            estimated_max_brake_ms2=new_max_brake,
            gas_p=fast_p,
            gas_i=fast_i,
            gas_d=fast_d,
            brake_ff=brake_ff_diag,
            brake_trim_p=new_slow_integral,
            brake_trim_i=fast_i,
            brake_multiplier=1.0,
            gain_scale=gain_scale,
            pedal_state=pedal_state,
            creep_ms2=creep_ms2,
            ff_saturated=ff_saturated,
        )

