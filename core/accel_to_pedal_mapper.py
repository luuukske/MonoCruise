"""
Maps a commanded longitudinal acceleration (m/s²) to normalized gas / brake
pedal targets (0–1). Used for ACC / cruise tuning; logs aggregate stats for
user reports and optional CSV samples when Settings.debug is True.

Reference curve (from legacy cruise logic, adapted):
- Positive demand → smoothed gas; negative → brake from scaled magnitude ** 2.5.
- Mass vs a reference mass adjusts the demand (same spirit as main_pedal weight_var).
- Very low speed in gear: light brake cancels idle creep (skipped when dashboard shows
  neutral, or above 5 m/s). Uses gearDashboard, not SDK gear — the latter can stay
  non-zero while the dash reads N.
"""

from __future__ import annotations

import csv
import logging
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
        gear_dashboard: int = 0,
        brake_efficiency_ratio: float = 1.0,
        log_sample: bool = True,
    ) -> PedalTargets:
        """
        :param wanted_accel_ms2: longitudinal command from ACC / cruise or 0.
        :param raw_accel_ms2: measured longitudinal accel (SDK local axis).
        :param gear_dashboard: gearDashboard from telemetry (0 = neutral on dash).
        :param brake_efficiency_ratio: 1.0 = nominal; <1.0 → brake demand scaled up
            proportionally to compensate for reduced grip (worn tires, snow, etc.).
        :param log_sample: if False, skip CSV and summary accumulation.
        """
        # Snapshot Settings each tick for live hot-reload tuning support.
        ref_mass_kg = float(Settings.mapper_reference_mass_kg)
        accel_scale = max(float(Settings.mapper_accel_scale_ms2), 1e-6)
        brake_div = max(float(Settings.mapper_brake_divisor), 1e-6)
        brake_pow = float(Settings.mapper_brake_power)
        w_span = max(float(Settings.mapper_weight_span_tons), 0.1)
        w_strength = float(Settings.mapper_weight_strength)
        efficiency = max(float(brake_efficiency_ratio), 0.1)

        weight_var = self._weight_var(total_mass_kg, has_trailer, ref_mass_kg, w_span, w_strength)
        t0 = float(wanted_accel_ms2) / accel_scale

        if t0 >= 0.0:
            t = t0 * weight_var
        else:
            t = -(abs(t0 * 1.3) ** (1.0 / max(weight_var, 1e-6)))

        cc_gas = (min(max(t, 0.0), 1.0) + self._prev_cc_gas) / 2.0
        self._prev_cc_gas = cc_gas

        if t > 0.0:
            cc_brake = 0.0
        else:
            mag = abs(t)
            raw_brake = max(min(mag / brake_div, 2.0) ** brake_pow, 0.0)
            # Scale brake demand up when efficiency is degraded to maintain target decel.
            cc_brake = min(1.0, raw_brake / efficiency)

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
                    "gas=%.3f brake=%.3f weight_var=%.3f t=%.3f "
                    "efficiency=%.3f mass=%.0f",
                    wanted_accel_ms2,
                    raw_accel_ms2,
                    speed_ms,
                    gas,
                    brake,
                    weight_var,
                    t,
                    efficiency,
                    total_mass_kg,
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
            temp_after_weight=t,
            command_brake=cc_brake,
        )
