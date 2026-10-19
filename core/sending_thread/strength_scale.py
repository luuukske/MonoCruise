"""Live low-speed strength for the gas map. See core/sending_thread/README.md."""

from __future__ import annotations

import math
from dataclasses import dataclass

# October pulls below 12 m/s measured about twice the mapped slope. Cruise
# speed was close to the map, so the scale fades out with speed.
_MAX: float = 2.6
_RISE_S: float = 0.30
_FALL_S: float = 8.0
_FULL_BELOW_MS: float = 8.0
_OFF_ABOVE_MS: float = 20.0
_LEARN_BELOW_MS: float = 12.0
_MIN_GAS: float = 0.15
_GAS_EMA_S: float = 0.40
_MIN_AUTHORITY_MS2: float = 0.30


@dataclass(slots=True)
class StrengthScale:
    scale: float = 1.0
    gas_ema: float = 0.0


def speed_weight(speed_ms: float) -> float:
    """1 at a low-speed pull, 0 once cruise speed is reached."""
    span = _OFF_ABOVE_MS - _FULL_BELOW_MS
    return max(0.0, min(1.0, (_OFF_ABOVE_MS - speed_ms) / span))


def applied_scale(scale: float, speed_ms: float) -> float:
    return 1.0 + (scale - 1.0) * speed_weight(speed_ms)


def map_capacity(model_cap: float, scale: float, speed_ms: float) -> float:
    return model_cap * applied_scale(scale, speed_ms)


def gas_from_demand(
    demand: float, map_cap: float, offset: float, strength: float,
) -> float:
    """Pedal for a positive demand. The offset form is only used once the live scale is up."""
    if demand <= 0.0 or map_cap <= 0.1:
        return 0.0
    if strength <= 1.02 or offset <= 0.0:
        return demand / map_cap
    off = min(max(offset, 0.0), max(map_cap - 0.05, 0.0))
    slope = max(map_cap - off, 0.05)
    if demand <= off:
        return 0.0
    return (demand - off) / slope


def update(
    state: StrengthScale,
    dt: float,
    gas: float,
    raw_ms2: float,
    road_ms2: float,
    model_cap: float,
    speed_ms: float,
    gear: int,
    factor: float,
    clutch: float,
    offset: float,
) -> None:
    """Raise the scale when a low-speed pull outruns the model. Never on a sag."""
    if dt <= 0.0:
        return
    a = 1.0 - math.exp(-dt / _GAS_EMA_S)
    if state.gas_ema <= 0.0:
        state.gas_ema = gas
    else:
        state.gas_ema += a * (gas - state.gas_ema)
    if (
        factor < 0.99
        or clutch > 0.05
        or gear <= 0
        or speed_ms < 1.0
        or speed_ms > _LEARN_BELOW_MS
        or (gear <= 1 and speed_ms < 3.0)
        or model_cap < 0.3
        or state.gas_ema < _MIN_GAS
    ):
        return
    off = min(max(offset, 0.0), 1.5)
    tractive = raw_ms2 + road_ms2
    if tractive < off + _MIN_AUTHORITY_MS2:
        return
    obs_cap = off + (tractive - off) / state.gas_ema
    obs = max(1.0, min(obs_cap / model_cap, _MAX))
    tau = _RISE_S if obs > state.scale else _FALL_S
    b = 1.0 - math.exp(-dt / tau)
    state.scale += b * (obs - state.scale)
    state.scale = max(1.0, min(state.scale, _MAX))
