"""One-pedal drive mapping: combined gas and brake axis to outputs. See core/main_pedal_thread/README.md."""

from __future__ import annotations

from core.settings import Settings


def _opd_interpolate(y_lo: float, y_hi: float, x: float, x_lo: float, x_hi: float) -> float:
    """Linear map: x in [x_lo, x_hi] → y in [y_lo, y_hi]."""
    if abs(x_hi - x_lo) < 1e-12:
        return y_hi
    t = (x - x_lo) / (x_hi - x_lo)
    return y_lo + t * (y_hi - y_lo)


def onepedaldrive(
    gasval: float,
    brakeval: float,
    *,
    apply_opd_mapping: bool = True,
) -> tuple[float, float]:
    """One-pedal mapping: combined axis → gas/brake, with gas and brake exponents applied See `core/main_pedal_thread/README.md`."""
    offset = float(Settings.offset_variable or 0.0)
    be_full = float(Settings.brake_exponent_variable or 1.0)
    brake_exponent = be_full**0.5
    gas_exponent = float(Settings.gas_exponent_variable or 1.0)
    opd_mode = bool(Settings.opd_mode_variable) and apply_opd_mapping
    max_opd_brake = max(0.0, float(Settings.max_opd_brake_variable or 0.0))

    val1 = min(max(float(gasval), 0.0), 1.0)
    val2 = (min(max(float(brakeval), 0.0), 1.0) ** brake_exponent) * -1.0
    sum_values = val1 + val2

    if opd_mode:
        if sum_values <= offset:
            value = (1.0 / (1.0 + offset)) * sum_values - (offset / (1.0 + offset))
        else:
            value = (1.0 / (1.0 - offset)) * sum_values - (offset / (1.0 - offset))
    else:
        value = sum_values

    a = -(max_opd_brake ** (1.0 / brake_exponent))

    if sum_values >= 0.0 and sum_values <= offset and opd_mode:
        value = (a / (offset ** brake_exponent)) * ((-sum_values + offset) ** brake_exponent)
    if sum_values < 0.0 and opd_mode:
        value = _opd_interpolate(-1.0, a, sum_values, -1.0, 0.0)

    gas_out = max(0.0, value)
    brake_out = min(min(0.0, value), val2) * -1.0

    gas_out = gas_out ** gas_exponent
    brake_out = brake_out ** brake_exponent
    return gas_out, brake_out
