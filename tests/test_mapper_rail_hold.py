"""A bid above what the truck can do must not leave the gas sagging afterwards.

The fast trim used to be back-calculated to the rail, which cancelled the excess
request and pinned the integrator at -1.5 m/s2. Once the bid came back into range
the pedal sagged and unwound at 0.25 m/s2 per second. See core/sending_thread/README.md.
"""
from __future__ import annotations

import math
from collections import deque

import pytest

from core.sending_thread.accel_to_pedals import AccelToPedals

DT = 0.02
OFFSET = 0.50
MASS = 40_000.0


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    clk = _FakeClock()
    monkeypatch.setattr("core.sending_thread.accel_to_pedals.time.monotonic", clk)
    return clk


def _pull(clock, bid_cap, *, true_slope=0.9, model_slope=0.9, seconds=30.0):
    """Cruise P bid from 50 to 75 km/h through the real mapper and a lagged affine plant."""
    mapper = AccelToPedals()
    mapper._shared.slow_integral = 0.0
    delay: deque[float] = deque([0.0] * 16, maxlen=16)
    meas, v, target = 0.0, 50.0 / 3.6, 75.0 / 3.6
    last = None
    trace = []
    for i in range(int(seconds / DT)):
        clock.t += DT
        road = last.road_load_ms2 if last is not None else 0.7
        accel = OFFSET + true_slope * delay[0] - road
        meas += (1.0 - math.exp(-DT / 0.30)) * (accel - meas)
        v += accel * DT
        bid = min(bid_cap, 0.5 * (target - v) / 1.2)
        last = mapper.step(
            bid, meas, v, MASS, True,
            max_accel_ms2=model_slope, accel_offset_ms2=OFFSET,
            cruise_commanding=True, gear_dashboard=10, learn=True,
        )
        delay.appendleft(last.gas)
        trace.append((i * DT, v * 3.6, last.gas, mapper._fast_integral, bid, meas))
    return trace


def _seconds_to(trace, kmh):
    return next(t for t, v, *_ in trace if v >= kmh)


def test_the_trim_is_held_while_railed_and_short(clock):
    trace = _pull(clock, 12.0)
    railed = [fi for _, _, gas, fi, bid, meas in trace if gas >= 0.999 and bid > meas + 0.1]
    assert railed, "the fixture must rail the pedal"
    assert min(railed) > -0.2, "the trim wound down against an unsatisfiable bid"


@pytest.mark.parametrize("model_slope", [0.6, 0.9, 1.4], ids=["under", "exact", "over"])
def test_asking_for_more_than_the_truck_has_is_never_slower(clock, model_slope):
    """Sport, or Normal on a heavy truck: the pull must finish as fast as a bid it can meet."""
    met = _seconds_to(_pull(clock, 0.6, model_slope=model_slope), 72.5)
    over = _seconds_to(_pull(clock, 12.0, model_slope=model_slope), 72.5)
    assert over <= met + 0.25


def test_over_delivery_at_the_rail_still_backs_the_pedal_off(clock):
    """A map that under-reads rails a pedal the truck does not need; the trim must still pull it back."""
    trace = _pull(clock, 0.5, true_slope=1.6, model_slope=0.6, seconds=6.0)
    tail = trace[len(trace) // 2:]
    assert max(gas for _, _, gas, *_ in tail) < 0.999
    assert sum(meas for *_, meas in tail) / len(tail) == pytest.approx(0.5, abs=0.15)
