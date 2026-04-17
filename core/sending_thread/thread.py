from __future__ import annotations

"""
Sending Thread — owns SCSController and pushes inputs to the game.

Responsibilities:
- Open and manage the SCS shared-memory controller.
- Apply pedal shaping and adaptive accel-to-pedal mapping, then write aforward/abackward to the game.
- Expose toggle_bool() for timed boolean presses (False → True → False).
- Expose set_bool() for persistent boolean overrides.
- Expose change_hazards() for verified hazard toggling with retrigger (max 3).
"""

import csv
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from .accel_to_pedals import AccelToPedals, baseline_accel_ms2, baseline_brake_ms2
from .pedal_capacity import PedalCapacityTracker
from .scscontroller import SCSController
from .visualization_bar import VisualizationBar

logger = logging.getLogger(__name__)


def create_visualization_bar() -> VisualizationBar:
    """
    Create and return the pedal visualization bar (gas/brake + em_stop).
    Must be called from the Qt main thread.
    """
    return VisualizationBar()

BOOL_PRESS_DURATION: float = 0.1
HAZARD_PRESS_DURATION: float = 0.4
HAZARD_VERIFY_DELAY: float = 0.1
HAZARD_MAX_RETRIGGERS: int = 3


@dataclass
class SendingThreadData(ThreadData):
    aforward: float = 0.0
    abackward: float = 0.0
    hazardsActive: bool = False
    horn_active: bool = False
    airhorn_active: bool = False
    decel_measured_ms2: float = 0.0
    mapper_commanded_ms2: float = 0.0
    mapper_control_wanted_ms2: float = 0.0
    mapper_raw_accel_ms2: float = 0.0
    mapper_measured_control_ms2: float = 0.0
    mapper_slope_input_rad: float = 0.0
    mapper_effective_slope_rad: float = 0.0
    mapper_wanted_smooth_ms2: float = 0.0
    mapper_raw_smooth_ms2: float = 0.0
    mapper_road_load_ms2: float = 0.0
    mapper_integral: float = 0.0
    mapper_est_max_accel_ms2: float = 0.0
    mapper_est_max_brake_ms2: float = 0.0
    mapper_command_gas: float = 0.0
    mapper_command_brake: float = 0.0
    mapper_cruise_active: bool = False
    mapper_gas_p: float = 0.0
    mapper_gas_i: float = 0.0
    mapper_gas_d: float = 0.0
    mapper_brake_ff: float = 0.0
    mapper_brake_trim_p: float = 0.0
    mapper_brake_trim_i: float = 0.0
    mapper_brake_multiplier: float = 1.0
    mapper_gain_scale: float = 1.0
    mapper_pedal_state: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


class SendingThread(BaseThread):
    loop_interval = 1.0 / Settings.polling_rate
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="sending_thread")
        self.data = SendingThreadData()
        self._controller: SCSController | None = None
        self._lock = threading.Lock()

        self._bool_presses: dict[str, float] = {}
        self._bool_overrides: dict[str, bool] = {}

        self._hazard_wanted: bool | None = None
        self._hazard_duration: float = HAZARD_PRESS_DURATION
        self._hazard_press_until: float = 0.0
        self._hazard_verify_until: float = 0.0
        self._hazard_retriggers: int = 0
        self._hazard_phase: str = "idle"

        self._last_should_force: bool = False
        self._hazard_user_override: bool = False
        self._prev_tel_hazards: bool = False
        self._accel_mapper = AccelToPedals()
        self._capacity_tracker = PedalCapacityTracker()
        self._key_listener = None
        self._spd_smooth: float | None = None
        self._prev_spd_mono: float | None = None

        # Brake-curve refit logger: every 1 s while braking, append
        # (median_decel, median_pedal) to brake_fit_samples.csv. Temporary.
        self._brake_fit_pedal_samples: list[float] = []
        self._brake_fit_decel_samples: list[float] = []
        self._brake_fit_roadload_samples: list[float] = []
        self._brake_fit_window_start: float = 0.0
        # Absolute path anchored to project root (same as mapper debug log).
        self._brake_fit_csv_path: Path = (
            Path(__file__).resolve().parents[2] / "brake_fit_samples.csv"
        )
        self._brake_fit_csv_initialised: bool = False
        logger.info("brake_fit_logger path: %s", self._brake_fit_csv_path)

    def toggle_bool(self, name: str, duration: float = BOOL_PRESS_DURATION) -> None:
        """One-shot timed press: set *name* True for *duration* seconds, then False."""
        with self._lock:
            self._bool_presses[name] = time.monotonic() + max(duration, 0.0)
        logger.debug("toggle_bool: %s for %.3fs", name, duration)

    def set_bool(self, name: str, value: bool = True) -> None:
        """Persistently hold *name* at *value* on the controller every tick."""
        with self._lock:
            self._bool_overrides[name] = value
        logger.debug("set_bool: %s = %s", name, value)

    def change_hazards(self, wanted: bool, duration: float = HAZARD_PRESS_DURATION) -> None:
        """Request hazards ON or OFF with verification and up to 3 retriggers."""
        with self._lock:
            self._hazard_wanted = wanted
            self._hazard_duration = max(duration, 0.0)
            self._hazard_retriggers = 0
            self._hazard_phase = "idle"
        logger.debug("change_hazards: wanted=%s duration=%.3fs", wanted, duration)

    def reset_accel_mapper_smoothing(self) -> None:
        """Clear mapper smoothing/correction when cruise stops commanding."""
        self._accel_mapper.reset_smoothing()

    def _read_speed(self) -> float:
        try:
            tel = registry.get_thread("telemetry_thread")
            with tel.data._lock:
                return float(tel.data.speed)
        except (KeyError, AttributeError):
            return 0.0

    def _reset_controller(self, controller: SCSController | None) -> None:
        if controller is None:
            return
        try:
            controller.aforward = 0.0
            controller.abackward = 0.0
            controller.flasher4way = False
        except (AttributeError, OSError, TypeError):
            pass
        with self._lock:
            for name in list(self._bool_presses) + list(self._bool_overrides):
                try:
                    setattr(controller, name, False)
                except (AttributeError, OSError, TypeError):
                    pass
        logger.debug("controller outputs reset to defaults")

    def setup(self) -> None:
        logger.info("initialising SCSController shared-memory interface")
        try:
            self._controller = SCSController()
            self._reset_controller(self._controller)
        except Exception:
            logger.exception("SCSController init failed; thread will run with zero output")
            self._controller = None
        self._hazard_phase = "idle"
        self._hazard_wanted = None
        self._last_should_force = False
        self._hazard_user_override = False
        self._prev_tel_hazards = False

        self._capacity_tracker.load_persisted(
            baseline_brake=baseline_brake_ms2(0.0, False),
            baseline_accel=baseline_accel_ms2(0.0, False),
        )

        if self._controller is not None:
            logger.debug("SCSController initialised")

    def loop(self) -> None:
        if not self.running:
            return

        self.loop_interval = 1.0 / max(Settings.polling_rate, 10)

        controller = self._controller
        if controller is None:
            return

        try:
            self._loop_body(controller)
        except Exception:
            logger.exception("unexpected error in loop; zeroing outputs")
            try:
                controller.aforward = 0.0
                controller.abackward = 0.0
            except (AttributeError, OSError, TypeError):
                pass
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0

    def _loop_body(self, controller: SCSController) -> None:
        try:
            pedal_thread = registry.get_thread("main_pedal_thread")
        except KeyError:
            pedal_thread = None
            logger.warning("main sending code limited;\nno pedal thread found")

        pedal_alive = pedal_thread is not None and pedal_thread.is_alive()

        em_stop = False
        AEB_brake = False
        AEB_warn = False
        if pedal_thread is not None and pedal_alive:
            try:
                with pedal_thread.data._lock:
                    em_stop = bool(pedal_thread.data.em_stop)
            except Exception as e:
                logger.debug("em_stop read failed: %s", e)
            try:
                aeb_thread = registry.get_thread("aeb_thread")
                if aeb_thread is not None and aeb_thread.is_alive():
                    with aeb_thread.data._lock:
                        AEB_brake = bool(aeb_thread.data.AEB_brake)
                        AEB_warn = bool(aeb_thread.data.AEB_warn)
            except (KeyError, Exception):
                pass

            em_stop = em_stop or AEB_brake

        connected = False
        gear = 0
        tel_hazards = False
        speed_ms = 0.0

        try:
            tel_thread = registry.get_thread("telemetry_thread")
        except KeyError:
            tel_thread = None
            logger.warning("sending thread limited;\nno telemetry thread found")

        if tel_thread is not None and tel_thread.is_alive():
            try:
                with tel_thread.data._lock:
                    connected = tel_thread.data.is_connected
                    gear = tel_thread.data.gear_dashboard
                    tel_hazards = bool(tel_thread.data.hazardsActive)
                    speed_ms = tel_thread.data.speed
            except Exception as e:
                logger.debug("telemetry read failed: %s", e)

        # Resolve cruise_active before mapper so tracker can gate on it.
        cruise_active = False
        try:
            cruise_t = registry.get_thread("cruise_control_thread")
            if cruise_t is not None and cruise_t.is_alive():
                with cruise_t.data._lock:
                    cruise_active = bool(cruise_t.data.active)
        except (KeyError, AttributeError):
            pass

        mapper_gas = 0.0
        mapper_brake = 0.0
        mapper_command_brake = 0.0
        mapper_command_gas = 0.0
        mapper_control_wanted_ms2 = 0.0
        mapper_wanted_smooth_ms2 = 0.0
        mapper_raw_smooth_ms2 = 0.0
        mapper_measured_control_ms2 = 0.0
        mapper_slope_input_rad = 0.0
        mapper_effective_slope_rad = 0.0
        mapper_road_load_ms2 = 0.0
        mapper_integral = 0.0
        mapper_est_max_accel_ms2 = 0.0
        mapper_est_max_brake_ms2 = 0.0
        mapper_gas_p = 0.0
        mapper_gas_i = 0.0
        mapper_gas_d = 0.0
        mapper_brake_ff = 0.0
        mapper_brake_trim_p = 0.0
        mapper_brake_trim_i = 0.0
        mapper_brake_multiplier = 1.0
        mapper_gain_scale = 1.0
        mapper_pedal_state = 0
        wanted_a = 0.0
        raw_a = 0.0
        measured_decel_ms2 = 0.0
        mass_kg = 0.0
        has_t = False
        brake_grade_rad = 0.0
        game_clutch = 0.0
        if connected and tel_thread is not None and tel_thread.is_alive():
            try:
                with tel_thread.data._lock:
                    wanted_a = float(tel_thread.data.commanded_accel_ms2)
                    mass_kg = float(tel_thread.data.estimated_total_mass_kg)
                    spd_ms = float(tel_thread.data.speed)
                    has_t = bool(tel_thread.data.ego_has_trailer)
                    wheels_on_ground = int(tel_thread.data.wheels_on_ground)
                    road_pitch = float(tel_thread.data.rotationY)
                    tel_gear_dashboard = int(tel_thread.data.gear_dashboard)
                    game_throttle = float(tel_thread.data.gameThrottle)
                    game_clutch = float(tel_thread.data.gameClutch)
                now_spd = time.monotonic()
                if self._spd_smooth is None:
                    self._spd_smooth = spd_ms
                    raw_a = 0.0
                else:
                    # Tracking differentiator: acceleration = (speed - smoothed_speed) / tau.
                    # Differentiating the EMA residual gives a clean longitudinal signal that
                    # is immune to lateral/centripetal contamination and insensitive to the
                    # game's discrete physics-tick speed steps.
                    _TAU = 0.30
                    raw_a = (spd_ms - self._spd_smooth) / _TAU
                    dt_spd = now_spd - self._prev_spd_mono if self._prev_spd_mono else 0.02
                    alpha = 1.0 - math.exp(-max(dt_spd, 1e-4) / _TAU)
                    self._spd_smooth += alpha * (spd_ms - self._spd_smooth)
                self._prev_spd_mono = now_spd
                measured_decel_ms2 = max(0.0, -raw_a)
                targets = self._accel_mapper.step(
                    wanted_a,
                    raw_a,
                    spd_ms,
                    mass_kg,
                    has_t,
                    max_accel_ms2=self._capacity_tracker.max_accel_ms2,
                    max_brake_ms2=self._capacity_tracker.max_brake_ms2,
                    road_pitch=road_pitch,
                    cruise_commanding=cruise_active,
                    gear_dashboard=tel_gear_dashboard,
                    game_throttle=game_throttle,
                    game_clutch=game_clutch,
                )
                mapper_gas = float(targets.gas)
                mapper_brake = float(targets.brake)
                mapper_command_gas = float(targets.command_gas)
                mapper_command_brake = float(targets.command_brake)
                mapper_control_wanted_ms2 = float(targets.control_wanted_ms2)
                mapper_measured_control_ms2 = float(targets.measured_control_ms2)
                mapper_slope_input_rad = float(targets.slope_input_rad)
                mapper_effective_slope_rad = float(targets.effective_slope_rad)
                mapper_wanted_smooth_ms2 = float(targets.wanted_smooth)
                mapper_raw_smooth_ms2 = float(targets.raw_smooth)
                mapper_road_load_ms2 = float(targets.road_load_ms2)
                mapper_integral = float(targets.integral_correction)
                mapper_est_max_accel_ms2 = float(targets.estimated_max_accel_ms2)
                mapper_est_max_brake_ms2 = float(targets.estimated_max_brake_ms2)
                mapper_gas_p = float(targets.gas_p)
                mapper_gas_i = float(targets.gas_i)
                mapper_gas_d = float(targets.gas_d)
                mapper_brake_ff = float(targets.brake_ff)
                mapper_brake_trim_p = float(targets.brake_trim_p)
                mapper_brake_trim_i = float(targets.brake_trim_i)
                mapper_brake_multiplier = float(targets.brake_multiplier)
                mapper_gain_scale = float(targets.gain_scale)
                mapper_pedal_state = int(targets.pedal_state)
            except Exception as e:
                logger.debug("accel_mapper step failed: %s", e)
                brake_grade_rad = 0.0
            else:
                brake_grade_rad = float(mapper_slope_input_rad)

        speed_kmh = speed_ms * 3.6
        if connected:
            if tel_hazards and not self._prev_tel_hazards:
                with self._lock:
                    wanted_on = self._hazard_wanted is True
                if not wanted_on:
                    self._hazard_user_override = True
            if not tel_hazards:
                self._hazard_user_override = False
            elif speed_kmh <= 12.0:
                self._hazard_user_override = False
            self._prev_tel_hazards = tel_hazards

        if Settings.autodisable_hazards and pedal_thread is not None and pedal_alive:
            try:
                with pedal_thread.data._lock:
                    gas_pct = pedal_thread.data.gasval
                    brake_pct = pedal_thread.data.brakeval
                if (
                    speed_kmh > 12.0
                    and gas_pct >= 0.60
                    and brake_pct < 0.05
                    and not AEB_warn
                    and not self._hazard_user_override
                ):
                    with self._lock:
                        self._hazard_wanted = False
                        if self._hazard_phase == "idle":
                            self._hazard_retriggers = 0
            except Exception as e:
                logger.debug("autodisable_hazards read failed: %s", e)

        should_force = not pedal_alive or em_stop
        if should_force and not self._last_should_force:
            self.change_hazards(True)
        self._last_should_force = should_force

        self._tick_hazards(controller, tel_hazards)
        self._tick_bool_overrides(controller)


        if not connected:
            self._spd_smooth = None
            self._prev_spd_mono = None
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
                self.data.mapper_commanded_ms2 = 0.0
                self.data.mapper_control_wanted_ms2 = 0.0
                self.data.mapper_raw_accel_ms2 = 0.0
                self.data.mapper_measured_control_ms2 = 0.0
                self.data.mapper_slope_input_rad = 0.0
                self.data.mapper_effective_slope_rad = 0.0
                self.data.mapper_wanted_smooth_ms2 = 0.0
                self.data.mapper_raw_smooth_ms2 = 0.0
                self.data.mapper_road_load_ms2 = 0.0
                self.data.mapper_integral = 0.0
                self.data.mapper_est_max_accel_ms2 = 0.0
                self.data.mapper_est_max_brake_ms2 = 0.0
                self.data.mapper_command_gas = 0.0
                self.data.mapper_command_brake = 0.0
                self.data.mapper_cruise_active = False
                self.data.mapper_gas_p = 0.0
                self.data.mapper_gas_i = 0.0
                self.data.mapper_gas_d = 0.0
                self.data.mapper_brake_ff = 0.0
                self.data.mapper_brake_trim_p = 0.0
                self.data.mapper_brake_trim_i = 0.0
                self.data.mapper_brake_multiplier = 1.0
                self.data.mapper_gain_scale = 1.0
                self.data.mapper_pedal_state = 0
            return

        if not pedal_alive:
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
                self.data.mapper_commanded_ms2 = 0.0
                self.data.mapper_control_wanted_ms2 = 0.0
                self.data.mapper_raw_accel_ms2 = 0.0
                self.data.mapper_measured_control_ms2 = 0.0
                self.data.mapper_slope_input_rad = 0.0
                self.data.mapper_effective_slope_rad = 0.0
                self.data.mapper_wanted_smooth_ms2 = 0.0
                self.data.mapper_raw_smooth_ms2 = 0.0
                self.data.mapper_road_load_ms2 = 0.0
                self.data.mapper_integral = 0.0
                self.data.mapper_est_max_accel_ms2 = 0.0
                self.data.mapper_est_max_brake_ms2 = 0.0
                self.data.mapper_command_gas = 0.0
                self.data.mapper_command_brake = 0.0
                self.data.mapper_cruise_active = False
                self.data.mapper_gas_p = 0.0
                self.data.mapper_gas_i = 0.0
                self.data.mapper_gas_d = 0.0
                self.data.mapper_brake_ff = 0.0
                self.data.mapper_brake_trim_p = 0.0
                self.data.mapper_brake_trim_i = 0.0
                self.data.mapper_brake_multiplier = 1.0
                self.data.mapper_gain_scale = 1.0
                self.data.mapper_pedal_state = 0
            return

        try:
            with pedal_thread.data._lock:
                gas_output = pedal_thread.data.gas_output
                brake_output = pedal_thread.data.brake_output
                gasval = pedal_thread.data.gasval
                brakeval = pedal_thread.data.brakeval
        except Exception as e:
            logger.debug("pedal read failed: %s", e)
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
                self.data.mapper_commanded_ms2 = 0.0
                self.data.mapper_control_wanted_ms2 = 0.0
                self.data.mapper_raw_accel_ms2 = 0.0
                self.data.mapper_measured_control_ms2 = 0.0
                self.data.mapper_slope_input_rad = 0.0
                self.data.mapper_effective_slope_rad = 0.0
                self.data.mapper_wanted_smooth_ms2 = 0.0
                self.data.mapper_raw_smooth_ms2 = 0.0
                self.data.mapper_road_load_ms2 = 0.0
                self.data.mapper_integral = 0.0
                self.data.mapper_est_max_accel_ms2 = 0.0
                self.data.mapper_est_max_brake_ms2 = 0.0
                self.data.mapper_command_gas = 0.0
                self.data.mapper_command_brake = 0.0
                self.data.mapper_cruise_active = False
                self.data.mapper_gas_p = 0.0
                self.data.mapper_gas_i = 0.0
                self.data.mapper_gas_d = 0.0
                self.data.mapper_brake_ff = 0.0
                self.data.mapper_brake_trim_p = 0.0
                self.data.mapper_brake_trim_i = 0.0
                self.data.mapper_brake_multiplier = 1.0
                self.data.mapper_gain_scale = 1.0
                self.data.mapper_pedal_state = 0
            return

        gas_exp = Settings.gas_exponent_variable or 1.0
        brake_exp = Settings.brake_exponent_variable or 1.0

        a = complex(gas_output).real
        b = complex(brake_output).real

        try:
            if gear != 0:
                a = float(gasval) ** float(gas_exp)
        except Exception:
            a = float(gasval)

        try:
            b = max(b, max(float(brakeval), 0.0) ** float(brake_exp))
        except Exception:
            b = max(b, max(float(brakeval), 0.0))

        b = b ** 0.91 # from going from 110% braking to 100% braking intensity
        a = float(complex(a).real)
        b = float(complex(b).real)

        # Cruise / ACC: mapper gas when active; mapper brake merged whenever connected.
        if cruise_active:
            a = max(a, mapper_gas)
        b = max(b, mapper_brake)

        controller.aforward = a
        controller.abackward = b

        # Temporary brake-curve refit logger. Samples whenever *any* brake
        # pedal is applied (manual or cruise), so data collection works
        # even when cruise is off.
        self._tick_brake_fit_logger(
            braking_active=(b > 0.00),
            brake_pedal=b,
            measured_decel_ms2=measured_decel_ms2,
            road_load_ms2=mapper_road_load_ms2,
            speed_ms=speed_ms,
        )

        # Update pedal capacity estimates from actual pedal values sent to the game.
        _base_brake = baseline_brake_ms2(0.0, False)
        _base_accel = baseline_accel_ms2(mass_kg, has_t)
        if b > 0.01:
            self._capacity_tracker.update_brake(
                b, measured_decel_ms2, speed_ms, brake_grade_rad, _base_brake,
                road_load_ms2=mapper_road_load_ms2,
            )
        if a > 0.01:
            self._capacity_tracker.update_accel(
                a, max(0.0, raw_a), speed_ms, brake_grade_rad, _base_accel, game_clutch,
                road_load_ms2=mapper_road_load_ms2,
            )

        self._tick_bool_presses(controller)

        with self.data._lock:
            self.data.aforward = a
            self.data.abackward = b
            self.data.hazardsActive = tel_hazards
            self.data.horn_active = bool(getattr(controller, "horn", False))
            self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
            self.data.decel_active = False
            self.data.decel_brake_output = 0.0
            self.data.decel_measured_ms2 = measured_decel_ms2
            self.data.mapper_commanded_ms2 = wanted_a
            self.data.mapper_control_wanted_ms2 = mapper_control_wanted_ms2
            self.data.mapper_raw_accel_ms2 = raw_a
            self.data.mapper_measured_control_ms2 = mapper_measured_control_ms2
            self.data.mapper_slope_input_rad = mapper_slope_input_rad
            self.data.mapper_effective_slope_rad = mapper_effective_slope_rad
            self.data.mapper_wanted_smooth_ms2 = mapper_wanted_smooth_ms2
            self.data.mapper_raw_smooth_ms2 = mapper_raw_smooth_ms2
            self.data.mapper_road_load_ms2 = mapper_road_load_ms2
            self.data.mapper_integral = mapper_integral
            self.data.mapper_est_max_accel_ms2 = mapper_est_max_accel_ms2
            self.data.mapper_est_max_brake_ms2 = mapper_est_max_brake_ms2
            self.data.mapper_command_gas = mapper_command_gas
            self.data.mapper_command_brake = mapper_command_brake
            self.data.mapper_cruise_active = cruise_active
            self.data.mapper_gas_p = mapper_gas_p
            self.data.mapper_gas_i = mapper_gas_i
            self.data.mapper_gas_d = mapper_gas_d
            self.data.mapper_brake_ff = mapper_brake_ff
            self.data.mapper_brake_trim_p = mapper_brake_trim_p
            self.data.mapper_brake_trim_i = mapper_brake_trim_i
            self.data.mapper_brake_multiplier = mapper_brake_multiplier
            self.data.mapper_gain_scale = mapper_gain_scale
            self.data.mapper_pedal_state = mapper_pedal_state

    def _tick_brake_fit_logger(
        self,
        braking_active: bool,
        brake_pedal: float,
        measured_decel_ms2: float,
        road_load_ms2: float,
        speed_ms: float,
    ) -> None:
        """Temporary logger for refitting the brake curve.

        Every 1 s of active braking, writes one row to brake_fit_samples.csv
        containing the median brake pedal, the median measured deceleration,
        and the median brake-only deceleration (measured − road_load).

        Using the median over a 1 s window rejects physics-tick noise and
        transients; one point per second gives enough coverage to refit the
        y = A·(1 − exp(−k·xⁿ)) curve afterwards.

        Samples are only collected when:
          - mapper is in BRAKE state (braking_active=True)
          - speed ≥ 5 m/s (avoid stop/creep noise)
          - measured decel > 0 (actually slowing down)
        """
        now = time.monotonic()

        if not braking_active or speed_ms < 5.0 or measured_decel_ms2 <= 0.0:
            # Flush an in-progress window if we had enough samples, then reset.
            if self._brake_fit_pedal_samples:
                self._flush_brake_fit_window()
            self._brake_fit_window_start = now
            return

        if not self._brake_fit_pedal_samples:
            self._brake_fit_window_start = now

        self._brake_fit_pedal_samples.append(float(brake_pedal))
        self._brake_fit_decel_samples.append(float(measured_decel_ms2))
        self._brake_fit_roadload_samples.append(float(road_load_ms2))

        if now - self._brake_fit_window_start >= 1.0:
            self._flush_brake_fit_window()

    def _flush_brake_fit_window(self) -> None:
        """Write one row summarising the current 1 s window, then reset buffers."""
        try:
            if len(self._brake_fit_pedal_samples) < 3:
                return

            median_pedal = statistics.median(self._brake_fit_pedal_samples)
            median_decel = statistics.median(self._brake_fit_decel_samples)
            median_road_load = statistics.median(self._brake_fit_roadload_samples)
            # brake_only = measured decel minus road_load contribution (rolling + slope).
            # road_load positive resists forward motion, so it *adds* to decel.
            brake_only_decel = median_decel - median_road_load
            sample_count = len(self._brake_fit_pedal_samples)

            header = [
                "utc",
                "median_brake_pedal",
                "median_measured_decel_ms2",
                "median_road_load_ms2",
                "brake_only_decel_ms2",
                "sample_count",
            ]

            new_file = not self._brake_fit_csv_path.exists()
            with self._brake_fit_csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if new_file or not self._brake_fit_csv_initialised:
                    writer.writerow(header)
                    self._brake_fit_csv_initialised = True
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    f"{median_pedal:.4f}",
                    f"{median_decel:.4f}",
                    f"{median_road_load:.4f}",
                    f"{brake_only_decel:.4f}",
                    sample_count,
                ])
                fh.flush()
        except Exception as e:
            logger.warning("brake_fit_logger write failed: %s", e)
        finally:
            self._brake_fit_pedal_samples.clear()
            self._brake_fit_decel_samples.clear()
            self._brake_fit_roadload_samples.clear()

    def teardown(self) -> None:
        if self._key_listener is not None:
            try:
                self._key_listener.stop()
            except Exception:
                pass
            self._key_listener = None

        if self._controller is not None:
            try:
                self._reset_controller(self._controller)
                self._controller.close()
            except Exception:
                logger.exception("error closing SCSController (suppressed)")
            self._controller = None
        with self.data._lock:
            self.data.aforward = 0.0
            self.data.abackward = 0.0
            self.data.hazardsActive = False
            self.data.horn_active = False
            self.data.airhorn_active = False
            self.data.decel_active = False
            self.data.decel_brake_output = 0.0
            self.data.decel_measured_ms2 = 0.0
            self.data.mapper_commanded_ms2 = 0.0
            self.data.mapper_control_wanted_ms2 = 0.0
            self.data.mapper_raw_accel_ms2 = 0.0
            self.data.mapper_measured_control_ms2 = 0.0
            self.data.mapper_slope_input_rad = 0.0
            self.data.mapper_effective_slope_rad = 0.0
            self.data.mapper_wanted_smooth_ms2 = 0.0
            self.data.mapper_raw_smooth_ms2 = 0.0
            self.data.mapper_road_load_ms2 = 0.0
            self.data.mapper_integral = 0.0
            self.data.mapper_est_max_accel_ms2 = 0.0
            self.data.mapper_est_max_brake_ms2 = 0.0
            self.data.mapper_command_gas = 0.0
            self.data.mapper_command_brake = 0.0
            self.data.mapper_cruise_active = False
            self.data.mapper_gas_p = 0.0
            self.data.mapper_gas_i = 0.0
            self.data.mapper_gas_d = 0.0
            self.data.mapper_brake_ff = 0.0
            self.data.mapper_brake_trim_p = 0.0
            self.data.mapper_brake_trim_i = 0.0
            self.data.mapper_brake_multiplier = 1.0
            self.data.mapper_gain_scale = 1.0
            self.data.mapper_pedal_state = 0
        self._accel_mapper.close()
        logger.debug("teardown complete")

    def _tick_bool_overrides(self, controller: SCSController) -> None:
        with self._lock:
            overrides = dict(self._bool_overrides)
        for name, value in overrides.items():
            try:
                setattr(controller, name, value)
            except (AttributeError, OSError, TypeError):
                pass

    def _tick_bool_presses(self, controller: SCSController) -> None:
        now = time.monotonic()
        with self._lock:
            items = list(self._bool_presses.items())

        for name, release_at in items:
            if now < release_at:
                try:
                    setattr(controller, name, True)
                except (AttributeError, OSError, TypeError):
                    pass
            else:
                try:
                    setattr(controller, name, False)
                except (AttributeError, OSError, TypeError):
                    pass
                with self._lock:
                    self._bool_presses.pop(name, None)

    def _tick_hazards(self, controller: SCSController, tel_hazards: bool) -> None:
        """
        Hazard state machine with 3 phases: idle → pressing → verifying.

        - idle: if wanted != telemetry, start pressing.
        - pressing: hold flasher4way=True for the hold duration, then verify.
        - verifying: wait 0.1s, check telemetry. If matched → done. If not → retrigger (up to 3).
        """
        now = time.monotonic()

        with self._lock:
            wanted = self._hazard_wanted

        if wanted is None:
            try:
                controller.flasher4way = False
            except (AttributeError, OSError, TypeError):
                pass
            return

        if wanted and not Settings.hazards_variable:
            try:
                controller.flasher4way = False
            except (AttributeError, OSError, TypeError):
                pass
            return

        if self._hazard_phase == "idle":
            if tel_hazards != wanted:
                logger.debug("hazard: starting press (telemetry=%s, wanted=%s)", tel_hazards, wanted)
                self._hazard_phase = "pressing"
                self._hazard_press_until = now + self._hazard_duration
            else:
                with self._lock:
                    self._hazard_wanted = None
                try:
                    controller.flasher4way = False
                except (AttributeError, OSError, TypeError):
                    pass
                return

        if self._hazard_phase == "pressing":
            if now < self._hazard_press_until:
                try:
                    controller.flasher4way = True
                except (AttributeError, OSError, TypeError):
                    pass
            else:
                try:
                    controller.flasher4way = False
                except (AttributeError, OSError, TypeError):
                    pass
                self._hazard_phase = "verifying"
                self._hazard_verify_until = now + HAZARD_VERIFY_DELAY
                logger.debug("hazard: press released, verifying (wanted=%s)", wanted)

        elif self._hazard_phase == "verifying":
            try:
                controller.flasher4way = False
            except (AttributeError, OSError, TypeError):
                pass

            if tel_hazards == wanted:
                logger.debug("hazard: confirmed (telemetry=%s)", tel_hazards)
                self._hazard_phase = "idle"
                with self._lock:
                    self._hazard_wanted = None
                    self._hazard_retriggers = 0
            elif now >= self._hazard_verify_until:
                if self._hazard_retriggers >= HAZARD_MAX_RETRIGGERS:
                    logger.warning(
                        "hazard: retrigger limit (%d) reached, giving up (wanted=%s)",
                        HAZARD_MAX_RETRIGGERS, wanted,
                    )
                    self._hazard_phase = "idle"
                    with self._lock:
                        self._hazard_wanted = None
                        self._hazard_retriggers = 0
                else:
                    self._hazard_retriggers += 1
                    logger.debug(
                        "hazard: verify timeout, retrigger %d/%d (telemetry=%s, wanted=%s)",
                        self._hazard_retriggers, HAZARD_MAX_RETRIGGERS, tel_hazards, wanted,
                    )
                    self._hazard_phase = "pressing"
                    self._hazard_press_until = now + self._hazard_duration
