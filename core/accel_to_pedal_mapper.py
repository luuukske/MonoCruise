"""
Maps a commanded longitudinal acceleration (m/s²) to normalized gas / brake
pedal targets (0–1). Used for ACC / cruise tuning; logs aggregate stats for
user reports and optional CSV samples when Settings.debug is True.

- Positive demand → smoothed gas (legacy weight + physics FF when cruise active).
- Negative demand → brake matches legacy MonoCruise ``cc_target_speed_thread_func``:
  normalized command ``temp_val`` (PID/ACC blend × weight + physics FF, **before**
  the unused asymmetric rewrite) maps to pedal via
  ``(min(abs(temp_val / divisor), cap) ** power)`` with tunable divisor/power.
  Optional integral trim on (raw − wanted) longitudinal accel; brake efficiency
  scales demand when grip is low.
- Mass vs a reference mass adjusts the demand (weight_var).
- Very low speed in gear: light brake cancels idle creep (gearDashboard; skipped in N
  or above ~5 m/s).
"""

from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.settings import Settings

logger = logging.getLogger(__name__)

# Diesel-ish kg per litre (telemetry fuel is litres).
FUEL_KG_PER_LITER: float = 0.85

# Summary line for users to copy into feedback (no paths or PII).
SUMMARY_INTERVAL_S: float = 45.0

# CSV lives next to the project root (same directory family as monocruise.log).
_DEBUG_CSV_NAME = "accel_mapper_debug.csv"


def compute_estimated_mass_kg(
    unit_mass_kg: float,
    cargo_mass_kg: float,
    fuel_litres: float,
    fuel_kg_per_liter: float = FUEL_KG_PER_LITER,
    trailer_count: int = 0,
) -> float:
    """Tractor + cargo + fuel mass from telemetry (kg)."""
    fuel_kg = max(0.0, float(fuel_litres)) * float(fuel_kg_per_liter)
    trailer_mass_kg = trailer_count * 1000.0
    if trailer_count == 0:
        cargo_mass_kg = 0.0
    return max(0.0, float(unit_mass_kg)) + cargo_mass_kg + fuel_kg + trailer_mass_kg


# Above this speed (m/s) idle-creep brake is not computed (saves work; curve is ~0 anyway).
_IDLE_CREEP_SPEED_SKIP_MS: float = 5.0

# Integral leak time constant (s) when not commanding cruise brake demand.
_BRAKE_I_LEAK_TAU_S: float = 0.35


def brake_drag_assist_ms2(speed_ms: float, coeff: float, v_ref_ms: float) -> float:
    """Equivalent extra decel (m/s²) from speed-dependent drag opposing motion."""
    if coeff <= 0.0 or not math.isfinite(coeff):
        return 0.0
    v_ref = max(float(v_ref_ms), 1e-3)
    s = abs(float(speed_ms))
    return float(coeff) * (s / v_ref) ** 2


def brake_pedal_from_decel_ms2(
    a_demand_ms2: float,
    *,
    speed_ms: float,
    speed_drag_coeff: float,
    speed_ref_ms: float,
    full_decel_ms2: float,
    efficiency: float,
) -> float:
    """
    Alternate normalized brake [0,1] from commanded decel magnitude (m/s²): drag
    assist, then linear map vs full-brake decel. Kept for experiments / tooling;
    the live :meth:`AccelToPedalMapper.step` path uses :func:`legacy_brake_pedal_normalized`.
    """
    a_d = max(0.0, float(a_demand_ms2))
    if not math.isfinite(a_d):
        return 0.0
    assist = brake_drag_assist_ms2(speed_ms, speed_drag_coeff, speed_ref_ms)
    a_net = max(0.0, a_d - assist)
    denom = max(float(full_decel_ms2), 0.5) * max(float(efficiency), 0.1)
    return max(0.0, min(1.0, a_net / denom))


def legacy_brake_pedal_normalized(
    abs_normalized_cmd: float,
    *,
    divisor: float,
    power: float,
    max_ratio: float = 2.0,
    efficiency: float = 1.0,
) -> float:
    """
    Legacy MonoCruise cruise brake: ``max((min(abs(temp_val/divisor), max_ratio)**power), 0)``
    then clip to [0, 1]. *abs_normalized_cmd* is ``abs(temp_val)`` in the same normalized
    units as ``wanted_accel_ms2 / mapper_accel_scale_ms2`` after linear weight + physics.
    Lower *efficiency* increases pedal demand to compensate for reduced decel per unit pedal.
    """
    d = max(float(divisor), 1e-6)
    cap = max(float(max_ratio), 1e-6)
    eff = max(float(efficiency), 0.1)
    a = abs(float(abs_normalized_cmd))
    if not math.isfinite(a):
        return 0.0
    r = min(a / d, cap)
    raw = r ** float(power)
    return max(0.0, min(1.0, raw / eff))

# Legacy CC used a constant m/s term (speed*0/3.6 + 100/3.6) in the slope-force expression.
_LEGACY_SLOPE_SPEED_MS: float = 100.0 / 3.6


def legacy_physics_adjustment_ff(
    speed_ms: float,
    slope: float,
    mass_kg: float,
    target_speed_kmh: float,
    dt: float,
    prev_slope: float,
    *,
    rolling_coeff: float,
    drag_coeff: float,
    slope_scalar: float,
    ff_horizon_s: float,
    ff_gain: float,
) -> float:
    """
    Match legacy cruise ``physics_adjustment`` + slope-rate feed-forward
    (``physics_adjustment_ff``): rolling + drag + slope force, scaled by
    ``slow_speed_adjustment``, plus predicted slope change over *ff_horizon_s*.
    """
    ts = max(float(target_speed_kmh), 30.0)
    slow_speed_adjustment = (-(2 ** (-(ts * 0.04) + 0.3)) + 1.0) * 1.3

    mass_eff_kg = (float(mass_kg) - 9000.0) / 3.0 + 9000.0
    tan_now = math.tan(float(slope) * 2 * math.pi)
    slope_force = (
        mass_eff_kg * 9.81 * tan_now * _LEGACY_SLOPE_SPEED_MS * float(slope_scalar)
    )
    s = float(speed_ms)
    physics_adjustment = (
        float(rolling_coeff) * s
        + float(drag_coeff) * (s / 3.6) ** 2
        + slope_force
    ) * slow_speed_adjustment

    if dt <= 0.0 or not math.isfinite(dt):
        return physics_adjustment

    slope_derivative = (float(slope) - float(prev_slope)) / dt
    predicted_slope_change = slope_derivative * float(ff_horizon_s)
    tan_pred = math.tan((float(slope) + predicted_slope_change) * 2 * math.pi)
    predicted_slope_force_change = (
        mass_eff_kg
        * 9.81
        * (tan_pred - tan_now)
        * _LEGACY_SLOPE_SPEED_MS
        * float(slope_scalar)
        * slow_speed_adjustment
    )
    return physics_adjustment + predicted_slope_force_change * float(ff_gain)


def idle_creep_brake(speed_ms: float, gear_dashboard: int) -> float:
    """
    Light brake at very low speed to cancel idle creep (normalized pedal space).

    This is mathematically equivalent to:
        max(((max( ((1 - ((abs(speed)+0.1) / 5.5) ** 2.22) * (-0.6/(abs(speed)+0.2)+1)/0.715 * 0.10 , 0))/7)**2.5, 0)
    """
    if int(gear_dashboard) == 0:
        return 0.0
    s = abs(float(speed_ms))
    if s > _IDLE_CREEP_SPEED_SKIP_MS:
        return 0.0
    # Implementation as per provided formula step by step for clarity
    value = (1.0 - ((s + 0.1) / 5.5) ** 2.22) * (-0.6 / (s + 0.2) + 1.0)
    num = max(value / 0.715 * 0.10, 0.0)
    denom = num / 7.0
    result = max(denom ** 2.5, 0.0)
    return result


@dataclass
class PedalTargets:
    gas: float
    brake: float
    weight_var: float
    temp_after_weight: float
    command_brake: float  # cruise-commanded brake only (excludes idle creep)
    physics_adjustment_ff: float = 0.0  # legacy slope / rolling feed-forward (normalized)


@dataclass
class _SummaryStats:
    n: int = 0
    sum_abs_wanted: float = 0.0
    sum_abs_raw: float = 0.0
    sum_gas: float = 0.0
    sum_brake: float = 0.0
    sum_mass: float = 0.0
    min_mass: float = field(default_factory=lambda: float("inf"))
    max_mass: float = 0.0

    def reset(self) -> None:
        self.n = 0
        self.sum_abs_wanted = 0.0
        self.sum_abs_raw = 0.0
        self.sum_gas = 0.0
        self.sum_brake = 0.0
        self.sum_mass = 0.0
        self.min_mass = float("inf")
        self.max_mass = 0.0

    def add(
        self,
        wanted: float,
        raw: float,
        gas: float,
        brake: float,
        mass: float,
    ) -> None:
        self.n += 1
        self.sum_abs_wanted += abs(wanted)
        self.sum_abs_raw += abs(raw)
        self.sum_gas += gas
        self.sum_brake += brake
        self.sum_mass += mass
        self.min_mass = min(self.min_mass, mass)
        self.max_mass = max(self.max_mass, mass)


class AccelToPedalMapper:
    """
    Stateful mapper (smoothed gas). Call :meth:`step` once per control tick.
    """

    def __init__(
        self,
        *,
        reference_mass_kg: float = 20_000.0,
        accel_scale_ms2: float = 3.5,
        brake_divisor: float = 7.0,
        brake_power: float = 2.5,
        weight_span_tons: float = 12.7,
        weight_strength: float = 0.27,
    ) -> None:
        self.reference_mass_kg = reference_mass_kg
        self.accel_scale_ms2 = accel_scale_ms2
        self.brake_divisor = brake_divisor
        self.brake_power = brake_power
        self.weight_span_tons = weight_span_tons
        self.weight_strength = weight_strength

        self._prev_cc_gas: float = 0.0
        self._prev_slope: float = 0.0
        self._prev_physics_mono: float | None = None
        self._prev_step_mono: float | None = None
        self._brake_integral: float = 0.0
        self._summary = _SummaryStats()
        self._last_summary_mono: float = 0.0
        self._csv_file = None
        self._csv_writer: csv.writer | None = None
        self._project_root = Path(__file__).resolve().parent.parent

    def close(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except OSError:
                pass
            self._csv_file = None
            self._csv_writer = None

    def reset_smoothing(self) -> None:
        self._prev_cc_gas = 0.0
        self._prev_slope = 0.0
        self._prev_physics_mono = None
        self._prev_step_mono = None
        self._brake_integral = 0.0

    def _weight_var(
        self,
        total_mass_kg: float,
        has_trailer: bool,
        ref_mass_kg: float,
        w_span_tons: float,
        w_strength: float,
    ) -> float:
        ref_t = ref_mass_kg / 1000.0
        cur_t = max(0.0, float(total_mass_kg)) / 1000.0
        w = w_strength * ((cur_t - ref_t) / max(w_span_tons, 0.1)) + 1.0
        w = max(0.5, min(2.0, w))
        if has_trailer:
            w = min(2.0, w * 1.02)
        return w

    def _ensure_csv(self) -> None:
        if self._csv_file is not None:
            return
        path = self._project_root / _DEBUG_CSV_NAME
        new_file = not path.exists()
        self._csv_file = path.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if new_file:
            self._csv_writer.writerow(
                [
                    "irl_iso_utc",
                    "monotonic_s",
                    "speed_ms",
                    "raw_accel_ms2",
                    "wanted_accel_ms2",
                    "gas_pedal",
                    "brake_pedal",
                    "weight_var",
                    "total_mass_kg",
                    "has_trailer",
                ]
            )
            self._csv_file.flush()

    def _write_csv_row(
        self,
        *,
        speed_ms: float,
        raw_accel_ms2: float,
        wanted_accel_ms2: float,
        gas: float,
        brake: float,
        weight_var: float,
        total_mass_kg: float,
        has_trailer: bool,
    ) -> None:
        self._ensure_csv()
        if self._csv_writer is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._csv_writer.writerow(
            [
                now,
                f"{time.monotonic():.6f}",
                f"{speed_ms:.6f}",
                f"{raw_accel_ms2:.6f}",
                f"{wanted_accel_ms2:.6f}",
                f"{gas:.6f}",
                f"{brake:.6f}",
                f"{weight_var:.6f}",
                f"{total_mass_kg:.3f}",
                int(has_trailer),
            ]
        )
        self._csv_file.flush()

    def _maybe_emit_summary(self) -> None:
        now = time.monotonic()
        if self._last_summary_mono == 0.0:
            self._last_summary_mono = now
            return
        if now - self._last_summary_mono < SUMMARY_INTERVAL_S:
            return
        self._last_summary_mono = now
        if self._summary.n <= 0:
            return
        n = self._summary.n
        logger.info(
            "accel_mapper tuning summary (paste for tuning): samples=%d "
            "mean_abs_wanted_ms2=%.4f mean_abs_raw_ms2=%.4f mean_gas=%.4f mean_brake=%.4f "
            "mass_kg_avg=%.0f mass_kg_min=%.0f mass_kg_max=%.0f ref_mass_kg=%.0f",
            n,
            self._summary.sum_abs_wanted / n,
            self._summary.sum_abs_raw / n,
            self._summary.sum_gas / n,
            self._summary.sum_brake / n,
            self._summary.sum_mass / n,
            self._summary.min_mass if self._summary.min_mass != float("inf") else 0.0,
            self._summary.max_mass,
            self.reference_mass_kg,
        )
        self._summary.reset()

    def step(
        self,
        wanted_accel_ms2: float,
        raw_accel_ms2: float,
        speed_ms: float,
        total_mass_kg: float,
        has_trailer: bool,
        *,
        slope_rad: float = 0.0,
        cruise_commanding: bool = False,
        target_speed_kmh: float | None = None,
        gear_dashboard: int = 0,
        brake_efficiency_ratio: float = 1.0,
        log_sample: bool = True,
    ) -> PedalTargets:
        """
        :param wanted_accel_ms2: longitudinal command from ACC / cruise or 0.
        :param raw_accel_ms2: measured longitudinal accel (SDK local axis).
        :param slope_rad: telemetry ``rotationY`` (legacy CC used this for grade).
        :param cruise_commanding: when True, apply legacy ``physics_adjustment_ff`` like old CC.
        :param target_speed_kmh: cruise target for ``slow_speed_adjustment`` in physics FF.
        :param gear_dashboard: gearDashboard from telemetry (0 = neutral on dash).
        :param brake_efficiency_ratio: 1.0 = nominal; <1.0 → brake demand scaled up
            proportionally to compensate for reduced grip (worn tires, snow, etc.).
        :param log_sample: if False, skip CSV and summary accumulation.

        Brake mapping uses ``mapper_brake_divisor`` / ``mapper_brake_power`` (legacy
        MonoCruise curve). Optional ``mapper_brake_integral_coeff`` /
        ``mapper_brake_integral_clamp`` add cruise-only tracking trim on (raw − wanted) accel.
        """
        # Snapshot Settings each tick for live hot-reload tuning support.
        ref_mass_kg = float(Settings.mapper_reference_mass_kg)
        accel_scale = max(float(Settings.mapper_accel_scale_ms2), 1e-6)
        w_span = max(float(Settings.mapper_weight_span_tons), 0.1)
        w_strength = float(Settings.mapper_weight_strength)
        efficiency = max(float(brake_efficiency_ratio), 0.1)
        brake_divisor = max(float(Settings.mapper_brake_divisor), 1e-6)
        brake_power = float(Settings.mapper_brake_power)
        ki_brake = float(Settings.mapper_brake_integral_coeff)
        i_clamp = max(float(Settings.mapper_brake_integral_clamp), 0.0)

        weight_var = self._weight_var(total_mass_kg, has_trailer, ref_mass_kg, w_span, w_strength)

        now_mono = time.monotonic()
        physics_ff = 0.0
        slope = float(slope_rad)
        if cruise_commanding and target_speed_kmh is not None:
            if self._prev_physics_mono is None:
                dt_ff = 0.1
            else:
                dt_ff = max(1e-4, min(now_mono - self._prev_physics_mono, 0.5))
            self._prev_physics_mono = now_mono
            physics_ff = legacy_physics_adjustment_ff(
                speed_ms,
                slope,
                total_mass_kg,
                float(target_speed_kmh),
                dt_ff,
                self._prev_slope,
                rolling_coeff=float(Settings.mapper_physics_rolling_coeff),
                drag_coeff=float(Settings.mapper_physics_drag_coeff),
                slope_scalar=float(Settings.mapper_physics_slope_scalar),
                ff_horizon_s=float(Settings.mapper_physics_ff_horizon_s),
                ff_gain=float(Settings.mapper_physics_ff_gain),
            )
            self._prev_slope = slope
        else:
            self._prev_slope = slope
            self._prev_physics_mono = now_mono

        t0 = float(wanted_accel_ms2) / accel_scale

        # Same linear blend as legacy ``temp_val`` before gas/brake (MonoCruise applies
        # ``-(abs(temp_val*1.3)**(1/weight_var))`` only after cc_gas/cc_brake — that
        # mutation is unused downstream, so pedals must not use the asymmetric form).
        t_cmd = t0 * weight_var + physics_ff

        cc_gas = (min(max(t_cmd, 0.0), 1.0) + self._prev_cc_gas) / 2.0
        self._prev_cc_gas = cc_gas

        if self._prev_step_mono is None:
            dt_step = 0.02
        else:
            dt_step = max(1e-4, min(now_mono - self._prev_step_mono, 0.5))
        self._prev_step_mono = now_mono

        integrate_brake_i = (
            cruise_commanding
            and t_cmd < 0.0
            and i_clamp > 0.0
            and ki_brake != 0.0
        )

        if t_cmd > 0.0:
            cc_brake = 0.0
        else:
            cc_brake = legacy_brake_pedal_normalized(
                abs(t_cmd),
                divisor=brake_divisor,
                power=brake_power,
                max_ratio=2.0,
                efficiency=efficiency,
            )
            if integrate_brake_i:
                e_track = float(raw_accel_ms2) - float(wanted_accel_ms2)
                if not math.isfinite(e_track):
                    e_track = 0.0
                self._brake_integral += e_track * dt_step
                self._brake_integral = max(
                    -i_clamp, min(i_clamp, self._brake_integral)
                )
                cc_brake = max(
                    0.0,
                    min(1.0, cc_brake + ki_brake * self._brake_integral),
                )

        if not integrate_brake_i and i_clamp > 0.0 and ki_brake != 0.0:
            leak = math.exp(-dt_step / max(_BRAKE_I_LEAK_TAU_S, 1e-3))
            self._brake_integral *= leak

        creep = idle_creep_brake(speed_ms, gear_dashboard)
        gas = min(1.0, cc_gas)
        brake = min(1.0, cc_brake + creep)

        if log_sample:
            self._summary.add(
                wanted_accel_ms2,
                raw_accel_ms2,
                gas,
                brake,
                total_mass_kg,
            )
            self._maybe_emit_summary()
            if Settings.debug:
                logger.debug(
                    "accel_mapper tick: wanted=%.3f raw=%.3f speed=%.1f "
                    "gas=%.3f brake=%.3f weight_var=%.3f t_cmd=%.3f phys_ff=%.4f "
                    "efficiency=%.3f mass=%.0f brake_i=%.4f brake_div=%.2f pow=%.2f",
                    wanted_accel_ms2,
                    raw_accel_ms2,
                    speed_ms,
                    gas,
                    brake,
                    weight_var,
                    t_cmd,
                    physics_ff,
                    efficiency,
                    total_mass_kg,
                    self._brake_integral,
                    brake_divisor,
                    brake_power,
                )
                self._write_csv_row(
                    speed_ms=speed_ms,
                    raw_accel_ms2=raw_accel_ms2,
                    wanted_accel_ms2=wanted_accel_ms2,
                    gas=gas,
                    brake=brake,
                    weight_var=weight_var,
                    total_mass_kg=total_mass_kg,
                    has_trailer=has_trailer,
                )

        return PedalTargets(
            gas=gas,
            brake=brake,
            weight_var=weight_var,
            temp_after_weight=t_cmd,
            command_brake=cc_brake,
            physics_adjustment_ff=physics_ff,
        )
