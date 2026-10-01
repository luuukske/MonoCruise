"""Gas ceiling for a clutch-slip launch. See core/sending_thread/README.md, "Launch governor"."""

from __future__ import annotations

from dataclasses import dataclass

IDLE: int = 0
SLIP: int = 1
RELEASE: int = 2

# Arms only from rest or a crawl, so an ordinary gearshift on the move never does.
ARM_SPEED_MS: float = 1.5
MAX_SPEED_MS: float = 7.0
# Pedal per m/s² of accel error at the reference mass, scaled by gain_scale.
# Sized on the logged launch plant: ~2.4 m/s² per unit pedal, 0.3 s dead time.
KP: float = 0.30
KI_RISE: float = 0.40
KI_FALL: float = 0.80
RISE_MAX_PER_S: float = 1.0
FALL_MAX_PER_S: float = 2.0
# Once the loop is closed again the cap opens at this rate until it stops binding.
RELEASE_PER_S: float = 1.0
GAIN_SCALE_MIN: float = 0.5
GAIN_SCALE_MAX: float = 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class LaunchGovernor:
    phase: int = IDLE
    cap_i: float = 0.0
    prev_wanted: float = 0.0


def launch_gas_cap(
    gov: LaunchGovernor,
    *,
    dt: float,
    enabled: bool,
    clutch_pressed: bool,
    speed_ms: float,
    factor: float,
    wanted_ms2: float,
    accel_ms2: float,
    hold_pedal: float,
    prev_gas: float,
    mapper_gas: float,
    gain_scale: float,
) -> tuple[LaunchGovernor, float | None]:
    """Advance one tick. Returns the new state and the gas cap, None while idle."""
    if not enabled:
        return LaunchGovernor(), None
    phase, cap_i = gov.phase, gov.cap_i
    launch_edge = wanted_ms2 > 0.0 and gov.prev_wanted <= 0.0
    if phase != SLIP and clutch_pressed and speed_ms < ARM_SPEED_MS:
        # Start from the gas actually sent, never from a trajectory run up in neutral.
        phase, cap_i = SLIP, _clamp(prev_gas, 0.0, 1.0)
        launch_edge = wanted_ms2 > 0.0
    if phase == IDLE:
        return LaunchGovernor(prev_wanted=wanted_ms2), None

    if phase == SLIP:
        if launch_edge:
            # A hill start must not wait for the servo to find the holding pedal.
            cap_i = max(cap_i, _clamp(hold_pedal, 0.0, 1.0))
        gs = _clamp(gain_scale, GAIN_SCALE_MIN, GAIN_SCALE_MAX)
        err = wanted_ms2 - accel_ms2
        ki = (KI_RISE if err > 0.0 else KI_FALL) / gs
        cap_i = _clamp(cap_i + _clamp(ki * err, -FALL_MAX_PER_S, RISE_MAX_PER_S) * dt, 0.0, 1.0)
        cap = _clamp(cap_i + KP / gs * err, 0.0, 1.0)
        if (not clutch_pressed and factor >= 1.0) or speed_ms > MAX_SPEED_MS:
            phase, cap_i = RELEASE, cap
    else:
        cap_i = min(1.0, cap_i + RELEASE_PER_S * dt)
        cap = cap_i

    if phase == RELEASE and mapper_gas <= cap:
        return LaunchGovernor(prev_wanted=wanted_ms2), None
    return LaunchGovernor(phase, cap_i, wanted_ms2), cap
