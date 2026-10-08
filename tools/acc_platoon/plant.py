"""Truck, pedal mapper, gearbox and standstill hold for one client. See tools/acc_platoon/README.md.

The mapper is a closed loop on measured accel, so it is modelled as tracking the
command through a dead time and a first-order lag, inside what the engine and the
brakes can deliver. Gear shifts cut the engine's drive on the measured profile. The
standstill hold is the shipped `HoldController`.
"""
from __future__ import annotations

import math
import random
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
# A box kicks down below `kick_kmh` once the command asks for this share of full drive.
KICKDOWN_SHARE: float = 0.6
# Upshift speed at no drive and at full drive, as a share of `up_kmh` (median over all throttles).
UP_LIGHT: float = 0.97
UP_FULL: float = 1.03


@dataclass(frozen=True)
class Gearbox:
    """Shift schedule and drive cut, fitted on the mapper debug log. See the README, "Gear shifts".

    Speeds are km/h: entry k is the boundary between gear k and gear k + 1 of the box.
    """

    label: str
    up_kmh: tuple[float, ...]
    down_kmh: tuple[float, ...]
    kick_kmh: tuple[float, ...]
    ramp_down_s: float
    cut_s: float
    cut_sd_s: float
    ramp_up_s: float
    floor: float = 0.0
    # The logged gear number changes this long after the drive starts to fall.
    label_s: float = 0.8

    def gear_for(self, kmh: float) -> int:
        return sum(1 for up in self.up_kmh if kmh > up)

    def drive_share(self, since_s: float, cut_s: float) -> float:
        """Share of engine drive left `since_s` into a shift whose full cut lasts `cut_s`."""
        if since_s < self.ramp_down_s:
            return 1.0 - (1.0 - self.floor) * since_s / self.ramp_down_s
        x = (since_s - self.ramp_down_s - cut_s) / self.ramp_up_s
        if x <= 0.0:
            return self.floor
        if x >= 1.0:
            return 1.0
        return self.floor + (1.0 - self.floor) * x * x * (3.0 - 2.0 * x)

    def duration_s(self, cut_s: float) -> float:
        return self.ramp_down_s + cut_s + self.ramp_up_s


# 14-speed AMT on its usual path 4-6-8-10-11-12-13-14; 12- and 13-speed boxes shift the same.
AMT = Gearbox("AMT", up_kmh=(11.8, 19.7, 27.6, 43.0, 55.3, 69.8, 88.7),
              down_kmh=(7.6, 12.1, 21.9, 31.4, 41.6, 56.2, 74.2),
              kick_kmh=(9.9, 15.7, 24.8, 40.4, 52.7, 67.0, 85.6),
              ramp_down_s=0.6, cut_s=0.3, cut_sd_s=0.25, ramp_up_s=1.2)
# Torque-converter automatic: a shallow dip instead of a cut.
POWERSHIFT = Gearbox("6-speed automatic", up_kmh=(17.0, 28.0, 43.5, 61.6, 90.3),
                     down_kmh=(12.0, 20.0, 31.0, 45.0, 68.0),
                     kick_kmh=(15.0, 25.0, 39.0, 55.0, 81.0),
                     ramp_down_s=0.15, cut_s=0.1, cut_sd_s=0.05, ramp_up_s=0.35, floor=0.65,
                     label_s=0.15)


@dataclass(frozen=True)
class TruckSpec:
    """Brake capacity from the fitted rigs in tests/aeb/test_stop_distance_envelope.py. Cruise
    braking lags `tau_brake_s` (gentle-braking fit); an AEB slam builds in `tau_slam_s`."""

    label: str
    mass_t: float
    power_kw: float
    brake_ms2: float
    tau_brake_s: float
    launch_ms2: float = 1.6
    dead_time_s: float = 0.12
    tau_gas_s: float = 0.35
    tau_slam_s: float = 0.15
    gearbox: Gearbox | None = AMT

    def drive_ms2(self, v: float) -> float:
        """Engine drive at full gas: power or traction."""
        return min(self.launch_ms2, self.power_kw * 1000.0 / (self.mass_t * 1000.0 * max(v, 2.0)))

    def load_ms2(self, v: float) -> float:
        """Rolling and air drag on a flat road."""
        return 0.07 + 3.0 * v * v / (self.mass_t * 1000.0)

    def gas_limit_ms2(self, v: float) -> float:
        """Net accel at full gas on a flat road."""
        return max(0.0, self.drive_ms2(v) - self.load_ms2(v))


HEAVY = TruckSpec("heavy 40 t", mass_t=40.0, power_kw=430.0, brake_ms2=10.85, tau_brake_s=0.31)
MEDIUM = TruckSpec("medium 28 t", mass_t=28.0, power_kw=410.0, brake_ms2=12.0, tau_brake_s=0.25,
                   tau_gas_s=0.33)
LIGHT = TruckSpec("light 18 t", mass_t=18.0, power_kw=380.0, brake_ms2=12.58, tau_brake_s=0.19,
                  tau_gas_s=0.30)


class TruckPlant:
    """Longitudinal state of one truck, advanced one physics step at a time.

    `rng` draws each shift's cut length; without one every shift takes the median.
    """

    def __init__(self, spec: TruckSpec, s0: float, v0: float, dt: float,
                 rng: random.Random | None = None) -> None:
        self.spec = spec
        self.s = s0
        self.v = v0
        self.a = 0.0
        self._lag_a = 0.0
        n = max(1, int(round(spec.dead_time_s / dt)))
        self._queue: deque[float] = deque([0.0] * n)
        self.hold = HoldController(lambda d: min(1.0, max(0.0, d) / spec.brake_ms2))
        self.hold_out = HoldOutput()
        self._rng = rng
        self.gear = spec.gearbox.gear_for(v0 * 3.6) if spec.gearbox is not None else 0
        self.shifts = 0
        self.drive_share = 1.0
        self._shift_since: float | None = None
        self._shift_cut = 0.0
        self._shift_to = self.gear

    def _gearbox(self, cmd: float, dt: float) -> float:
        """Advance the gearbox; returns the share of engine drive available this step."""
        box = self.spec.gearbox
        if box is None:
            return 1.0
        if self._shift_since is not None:
            self._shift_since += dt
            if self._shift_since >= box.label_s:
                self.gear = self._shift_to
            if self._shift_since >= box.duration_s(self._shift_cut):
                self._shift_since = None
                return 1.0
            return box.drive_share(self._shift_since, self._shift_cut)
        kmh = self.v * 3.6
        demand = min(1.0, max(0.0, cmd / max(self.spec.gas_limit_ms2(self.v), 1e-6)))
        g = to = self.gear
        if g < len(box.up_kmh) and kmh > box.up_kmh[g] * (UP_LIGHT + (UP_FULL - UP_LIGHT) * demand):
            to = g + 1
        elif g > 0 and (kmh < box.down_kmh[g - 1]
                        or (kmh < box.kick_kmh[g - 1] and demand > KICKDOWN_SHARE)):
            to = g - 1
        if to == g:
            return 1.0
        cut = box.cut_s if self._rng is None else self._rng.gauss(box.cut_s, box.cut_sd_s)
        self._shift_cut = min(box.cut_s + 3.0 * box.cut_sd_s, max(0.0, cut))
        self._shift_since = 0.0
        self._shift_to = to
        self.shifts += 1
        return 1.0

    def step(self, cmd_ms2: float, dt: float, crawl_follow: bool = False,
             slam: bool = False) -> None:
        spec = self.spec
        self.hold_out = self.hold.update(
            speed_kmh=self.v * 3.6, gear=1, pitch_norm=0.0, commanded_accel_ms2=cmd_ms2,
            gasval=0.0, opdgasval=0.0, offset=0.0, park_brake=False, aeb_active=False, dt=dt,
            crawl_follow=crawl_follow,
        )
        self._queue.append(cmd_ms2)
        target = self._queue.popleft()
        self.drive_share = self._gearbox(target, dt)
        target = max(-spec.brake_ms2, min(spec.gas_limit_ms2(self.v), target))
        brake_tau = spec.tau_slam_s if slam else spec.tau_brake_s
        tau = brake_tau if target < self._lag_a else spec.tau_gas_s
        self._lag_a += (target - self._lag_a) * (1.0 - math.exp(-dt / tau))
        a = self._lag_a
        load = spec.load_ms2(self.v)
        if self.drive_share < 1.0 and a > -load:
            # The box ramps the engine's torque out and back in, whatever the throttle.
            a = self.drive_share * (a + load) - load
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
