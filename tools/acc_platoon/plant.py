"""Truck, pedal mapper and standstill hold for one client. See tools/acc_platoon/README.md.

The mapper is a closed loop on measured accel, so it is modelled as tracking the
command through a dead time and a first-order lag, inside what the engine and the
brakes can deliver. The standstill hold is the shipped `HoldController`.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from core.sending_thread.hold_controller import (
    STATE_HOLDING, STATE_LAUNCHING, STATE_STOPPING, HoldController, HoldOutput,
)

# Rig geometry, front bumper first. TMP draws a remote truck at the tractor's middle.
TRACTOR_LEN_M: float = 6.0
RIG_LEN_M: float = 16.5
# Telemetry position sits this far behind the bumper; the ACC controller assumes the same.
EGO_FRONT_OFFSET_M: float = 2.5
# Flat-road floor the hold brake keeps while it owns the truck.
HOLD_FLOOR_MS2: float = 0.3


@dataclass(frozen=True)
class TruckSpec:
    """Brake capacity and lag come from the fitted rigs in tests/aeb/test_stop_distance_envelope.py."""

    label: str
    mass_t: float
    power_kw: float
    brake_ms2: float
    tau_brake_s: float
    launch_ms2: float = 1.6
    dead_time_s: float = 0.12
    tau_gas_s: float = 0.35

    def gas_limit_ms2(self, v: float) -> float:
        """Net accel at full gas on a flat road: power or traction, less rolling and air drag."""
        mass = self.mass_t * 1000.0
        drive = min(self.launch_ms2, self.power_kw * 1000.0 / (mass * max(v, 2.0)))
        load = 0.07 + 3.0 * v * v / mass
        return max(0.0, drive - load)


HEAVY = TruckSpec("heavy 40 t", mass_t=40.0, power_kw=430.0, brake_ms2=10.85, tau_brake_s=0.31)
MEDIUM = TruckSpec("medium 28 t", mass_t=28.0, power_kw=410.0, brake_ms2=12.0, tau_brake_s=0.25,
                   tau_gas_s=0.33)
LIGHT = TruckSpec("light 18 t", mass_t=18.0, power_kw=380.0, brake_ms2=12.58, tau_brake_s=0.19,
                  tau_gas_s=0.30)


class TruckPlant:
    """Longitudinal state of one truck, advanced one physics step at a time."""

    def __init__(self, spec: TruckSpec, s0: float, v0: float, dt: float) -> None:
        self.spec = spec
        self.s = s0
        self.v = v0
        self.a = 0.0
        self._lag_a = 0.0
        n = max(1, int(round(spec.dead_time_s / dt)))
        self._queue: deque[float] = deque([0.0] * n)
        self.hold = HoldController(lambda d: min(1.0, max(0.0, d) / spec.brake_ms2))
        self.hold_out = HoldOutput()

    def step(self, cmd_ms2: float, dt: float, crawl_follow: bool = False) -> None:
        spec = self.spec
        self.hold_out = self.hold.update(
            speed_kmh=self.v * 3.6, gear=1, pitch_norm=0.0, commanded_accel_ms2=cmd_ms2,
            gasval=0.0, opdgasval=0.0, offset=0.0, park_brake=False, aeb_active=False, dt=dt,
            crawl_follow=crawl_follow,
        )
        self._queue.append(cmd_ms2)
        target = self._queue.popleft()
        target = max(-spec.brake_ms2, min(spec.gas_limit_ms2(self.v), target))
        tau = spec.tau_brake_s if target < self._lag_a else spec.tau_gas_s
        self._lag_a += (target - self._lag_a) * (1.0 - math.exp(-dt / tau))
        a = self._lag_a
        state = self.hold_out.state
        if state in (STATE_STOPPING, STATE_HOLDING):
            a = min(a, -HOLD_FLOOR_MS2)
        elif state == STATE_LAUNCHING:
            ease = self.hold_out.launch_ease
            a = a * ease - HOLD_FLOOR_MS2 * (1.0 - ease)
        v_new = max(0.0, self.v + a * dt)
        self.a = (v_new - self.v) / dt
        self.s += 0.5 * (self.v + v_new) * dt
        self.v = v_new
