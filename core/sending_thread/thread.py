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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from core.aeb.calibration import DEFAULT as _AEB_CAL

from .accel_to_pedals import AccelToPedals, MapperSharedState, baseline_accel_ms2, baseline_brake_ms2
from .pedal_capacity import PedalCapacityTracker
from .scscontroller import SCSController
from .visualization_bar import VisualizationBar

logger = logging.getLogger(__name__)

# Coast-down logger — captures raw decel vs speed with pedals fully released,
# so rolling resistance + aerodynamic drag can be fitted offline.
_COAST_LOG_NAME: str = "coast_debug.csv"
_COAST_LOG_INTERVAL_S: float = 0.10  # 10 Hz
_COAST_LOG_MIN_SPEED_MS: float = 1.0
_COAST_LOG_HEADER_ROW: list[str] = [
    "t_s",
    "utc",
    "speed_ms",
    "raw_accel_ms2",
    "slope_rad",
    "slope_accel_ms2",
    "drag_accel_ms2",  # raw_accel - slope_accel — this is what you plot vs speed
    "gear",
    "game_clutch",
    "game_throttle",
    "game_brake",
    "mass_kg",
    "has_trailer",
    "aforward",
    "abackward",
]


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

# Closed-loop decel controller — feedforward via the inverse brake curve plus
# a small PI on lead-compensated measured decel. Conservative gains; tuned to
# correct residual model error without fighting the FF.
_AEB_KP: float = 0.06
_AEB_KI: float = 0.04
_AEB_LEAD_CLAMP_MS2: float = 3.0


class AEBDecelController:
    """Closed-loop decel controller.

    Consumes ``AEB_target_decel_ms2`` (already rate-limited and deadbanded by
    the AEB thread). Active whenever ``AEB_brake`` is true. The bulk of the
    pedal comes from the inverse brake curve (FF); a small PI corrects the
    residual error against lead-compensated measured decel. Anti-windup
    freezes the integrator on pedal saturation.
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._integral: float = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def update_active(self, aeb_brake: bool) -> bool:
        was_active = self._active
        self._active = aeb_brake
        if not aeb_brake:
            self._integral = 0.0
        return aeb_brake and not was_active

    def step(
        self,
        target_decel_ms2: float,
        measured_lead_decel_ms2: float,
        max_brake_ms2: float,
        ff_pedal_fn,
        dt: float,
    ) -> float:
        """Compute brake pedal [0, 1] for the current tick."""
        if not self._active or target_decel_ms2 <= 0.0 or max_brake_ms2 <= 0.1:
            self._integral = 0.0
            return 0.0

        ff_pedal = ff_pedal_fn(target_decel_ms2, max_brake_ms2)
        error = target_decel_ms2 - measured_lead_decel_ms2
        p_term = _AEB_KP * error
        i_candidate = self._integral + _AEB_KI * error * max(dt, 1e-4)

        unclamped = ff_pedal + p_term + i_candidate
        clamped = max(0.0, min(1.0, unclamped))

        saturated_pushing = (
            (clamped >= 1.0 and error > 0.0) or (clamped <= 0.0 and error < 0.0)
        )
        if not saturated_pushing:
            self._integral = i_candidate
        return clamped


@dataclass
class SendingThreadData(ThreadData):
    aforward: float = 0.0
    abackward: float = 0.0
    hazardsActive: bool = False
    horn_active: bool = False
    airhorn_active: bool = False
    decel_measured_ms2: float = 0.0
    decel_measured_lead_ms2: float = 0.0
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
    max_brake_ms2: float = 0.0         # live PedalCapacityTracker estimate (m/s²)
    # Holding-brake FSM (relocated from main_pedal_thread). Owned here so the
    # hold floor applies to every brake source — user pedal, mapper, AEB.
    stopped: bool = False
    # Most recent brake values written to the game (last N ticks). Used by
    # cruise_control_thread to distinguish a user's in-game brake press from
    # the game echoing back our own command (which lags by a few ticks).
    recent_brake_outputs: tuple[float, ...] = (0.0, 0.0, 0.0)
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
        # Single mapper. CruiseControlThread publishes one m/s² bid covering
        # both CC and limiter modes. This avoids the per-instance state drift
        # the dual-mapper design suffered at the limit boundary.
        self._mapper_shared = MapperSharedState()
        self._accel_mapper = AccelToPedals(self._mapper_shared)
        self._capacity_tracker = PedalCapacityTracker()
        self._aeb_controller = AEBDecelController()
        self._key_listener = None
        self._spd_smooth: float | None = None
        self._prev_spd_mono: float | None = None
        self._prev_measured_decel_ms2: float = 0.0
        self._prev_mapper_gas: float = 0.0
        self._prev_user_gas: float = 0.0
        self._prev_user_opd_gas: float = 0.0
        self._prev_aeb_loop_mono: float | None = None
        self._brake_active: bool = False
        self._brake_last_active_at: float = 0.0
        # Holding-brake FSM state. _prev_brake_snapshot is the merged brake
        # captured at the False→True edge so the 0.5 s blend ramps from
        # "merge at the moment we became stopped" up to "merge + hold".
        self._stopped: bool = False
        self._prev_brake_snapshot: float = 0.0
        self._prev_stop_mono: float = 0.0
        # Ring buffer of the last 3 brake outputs sent to the game (oldest first).
        self._recent_brake_outputs: list[float] = [0.0, 0.0, 0.0]

        # Coast-down logger state
        self._coast_log_file = None
        self._coast_log_writer = None
        self._last_coast_log_mono: float = 0.0
        self._coast_log_start_mono: float | None = None
        try:
            self._project_root = Path(__file__).resolve().parents[2]
        except Exception:
            self._project_root = Path(".").resolve()

    # Coast-down CSV logger 

    def _ensure_coast_log(self) -> None:
        if self._coast_log_file is not None:
            return
        path = self._project_root / _COAST_LOG_NAME
        write_header = not path.exists() or path.stat().st_size == 0
        try:
            self._coast_log_file = path.open("a", newline="", encoding="utf-8")
            self._coast_log_writer = csv.writer(self._coast_log_file)
            if write_header:
                self._coast_log_writer.writerow(_COAST_LOG_HEADER_ROW)
                self._coast_log_file.flush()
        except OSError:
            self._coast_log_file = None
            self._coast_log_writer = None
            logger.debug("coast_debug log unavailable", exc_info=True)

    def _log_coast_step(
        self,
        *,
        speed_ms: float,
        raw_accel_ms2: float,
        slope_rad: float,
        gear: int,
        game_clutch: float,
        game_throttle: float,
        game_brake: float,
        mass_kg: float,
        has_trailer: bool,
        aforward: float,
        abackward: float,
    ) -> None:
        if abs(aforward) > 1e-4 or abs(abackward) > 1e-4:
            return
        if abs(game_throttle) > 0.02 or abs(game_brake) > 0.02:
            return
        if speed_ms < _COAST_LOG_MIN_SPEED_MS:
            return

        now = time.monotonic()
        if now - self._last_coast_log_mono < _COAST_LOG_INTERVAL_S:
            return
        self._last_coast_log_mono = now
        if self._coast_log_start_mono is None:
            self._coast_log_start_mono = now
        t_s = now - self._coast_log_start_mono

        self._ensure_coast_log()
        if self._coast_log_writer is None:
            return

        slope_accel = 9.81 * math.sin(slope_rad)
        drag_accel = raw_accel_ms2 - slope_accel
        try:
            self._coast_log_writer.writerow([
                f"{t_s:.3f}",
                datetime.now(timezone.utc).isoformat(),
                f"{speed_ms:.3f}",
                f"{raw_accel_ms2:+.4f}",
                f"{slope_rad:+.5f}",
                f"{slope_accel:+.4f}",
                f"{drag_accel:+.4f}",
                gear,
                f"{game_clutch:.3f}",
                f"{game_throttle:.3f}",
                f"{game_brake:.3f}",
                f"{mass_kg:.1f}",
                int(bool(has_trailer)),
                f"{aforward:.4f}",
                f"{abackward:.4f}",
            ])
            self._coast_log_file.flush()
        except OSError:
            logger.debug("coast_debug log write failed", exc_info=True)

    def _close_coast_log(self) -> None:
        if self._coast_log_file is not None:
            try:
                self._coast_log_file.close()
            except OSError:
                pass
            self._coast_log_file = None
            self._coast_log_writer = None

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
        self._brake_active = False
        self._brake_last_active_at = 0.0
        self._stopped = False
        self._prev_brake_snapshot = 0.0
        self._prev_stop_mono = 0.0

        self._capacity_tracker.load_persisted(
            baseline_brake=baseline_brake_ms2(0.0, False),
            baseline_accel=baseline_accel_ms2(0.0, False),
        )

        if self._controller is not None:
            logger.debug("SCSController initialised")

    def loop(self) -> None:
        if not self.running:
            return

        # Idle throttle: when telemetry is disconnected the disconnected branch
        # in _loop_body already zeroes outputs, so running at full polling_rate
        # is wasted CPU. Drop to 1 Hz; resumes on reconnect.
        try:
            tel = registry.get_thread("telemetry_thread")
            is_connected = tel.is_alive() and bool(tel.data.is_connected)
        except (KeyError, AttributeError):
            is_connected = False
        if is_connected:
            self.loop_interval = 1.0 / max(Settings.polling_rate, 10)
        else:
            self.loop_interval = 1.0

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
        AEB_target_decel = 0.0
        AEB_ff_decel = 0.0
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
                        AEB_target_decel = float(
                            getattr(aeb_thread.data, "AEB_target_decel_ms2", 0.0)
                        )
                        AEB_ff_decel = float(
                            getattr(aeb_thread.data, "AEB_ff_decel_ms2", 0.0)
                        )
            except (KeyError, Exception):
                pass

            em_stop = em_stop or AEB_brake

        _just_entered_aeb = self._aeb_controller.update_active(AEB_brake)
        if _just_entered_aeb:
            self._accel_mapper.reset_smoothing()
        _aeb_active = self._aeb_controller.active

        connected = False
        gear = 0
        tel_hazards = False
        speed_ms = 0.0
        park_brake = False

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
                    park_brake = bool(tel_thread.data.parkBrake)
            except Exception as e:
                logger.debug("telemetry read failed: %s", e)

        # Resolve cruise_active + active_controller before mapper so tracker
        # can gate on it and the user-pedal merge can pick max vs min.
        cruise_active = False
        cruise_active_controller = "none"
        try:
            cruise_t = registry.get_thread("cruise_control_thread")
            if cruise_t is not None and cruise_t.is_alive():
                with cruise_t.data._lock:
                    cruise_active = bool(cruise_t.data.active)
                    cruise_active_controller = str(cruise_t.data.active_controller)
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
        measured_decel_lead_ms2 = 0.0
        dt_aeb = self.loop_interval
        mass_kg = 0.0
        has_t = False
        brake_grade_rad = 0.0
        game_clutch = 0.0
        game_throttle = 0.0
        game_brake = 0.0
        user_clutch = 0.0
        road_pitch = 0.0
        tel_gear_dashboard = 0
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
                    game_brake = float(getattr(tel_thread.data, "gameBrake", 0.0))
                    user_clutch = float(getattr(tel_thread.data, "userClutch", 0.0))
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

                now_aeb = time.monotonic()
                if self._prev_aeb_loop_mono is None:
                    dt_aeb = self.loop_interval
                else:
                    dt_aeb = max(1e-3, min(0.5, now_aeb - self._prev_aeb_loop_mono))
                self._prev_aeb_loop_mono = now_aeb
                d_decel_dt = (
                    measured_decel_ms2 - self._prev_measured_decel_ms2
                ) / max(dt_aeb, 1e-4)
                lead_term = max(
                    -_AEB_LEAD_CLAMP_MS2,
                    min(_AEB_LEAD_CLAMP_MS2, d_decel_dt * _AEB_CAL.brake_actuator_lag_s),
                )
                measured_decel_lead_ms2 = max(0.0, measured_decel_ms2 + lead_term)
                self._prev_measured_decel_ms2 = measured_decel_ms2

                # Mapper engages whenever the orchestrator (CruiseControlThread) is
                # bidding. The orchestrator's single bid covers both CC and limiter
                # modes — there is no separate limiter call here.
                mapper_engaged = cruise_active or _aeb_active

                # Learn-gate by who owned the previous tick:
                # - Limiter active: only learn when user was pushing above the
                #   cap (user gas > mapper_gas). Within the cap there's nothing
                #   to learn — letting the integrator run would inflate mapper_gas
                #   and loosen the cap as ego approaches the limit.
                # - CC/ACC active: freeze learning when the user was overriding
                #   with OPD gas (opdgasval > mapper_gas). Mapper output isn't
                #   what's reaching the truck during override, so the measured
                #   accel doesn't reflect the mapper's command — learning from it
                #   would corrupt the gain/integral state.
                # One-tick lag is negligible at 60–100 Hz.
                if cruise_active_controller == "limiter":
                    mapper_learn = mapper_engaged and (self._prev_user_gas > self._prev_mapper_gas)
                else:
                    mapper_learn = mapper_engaged and (self._prev_user_opd_gas <= self._prev_mapper_gas)

                targets = self._accel_mapper.step(
                    wanted_a,
                    raw_a,
                    spd_ms,
                    mass_kg,
                    has_t,
                    max_accel_ms2=self._capacity_tracker.max_accel_ms2,
                    max_brake_ms2=self._capacity_tracker.max_brake_ms2,
                    road_pitch=road_pitch,
                    cruise_commanding=mapper_engaged,
                    gear_dashboard=tel_gear_dashboard,
                    game_throttle=game_throttle,
                    game_clutch=game_clutch,
                    freeze_trim=_aeb_active,
                    learn=mapper_learn,
                )
                mapper_gas = float(targets.gas)
                self._prev_mapper_gas = mapper_gas
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
            self._prev_measured_decel_ms2 = 0.0
            self._prev_aeb_loop_mono = None
            self._stopped = False
            self._prev_stop_mono = 0.0
            self._prev_brake_snapshot = 0.0
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.stopped = False
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
                self.data.stopped = self._stopped
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

        opdgasval = 0.0
        try:
            with pedal_thread.data._lock:
                gas_output = pedal_thread.data.gas_output
                brake_output = pedal_thread.data.brake_output
                gasval = pedal_thread.data.gasval
                brakeval = pedal_thread.data.brakeval
                opdgasval = float(getattr(pedal_thread.data, "opdgasval", 0.0))
        except Exception as e:
            logger.debug("pedal read failed: %s", e)
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.stopped = self._stopped
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

        # Manual clutch gate — when the driver physically presses the clutch
        # (manual transmission), suppress all mapper gas so the user's pedal
        # commands the truck directly during the shift. Brake commands and AEB
        # are unaffected. gameClutch is excluded because automatic transmissions
        # also raise it during their own gear changes.
        manual_clutch = user_clutch > 0.1

        # User-pedal merge with the mapper's output. Keyed on which
        # controller is actually bidding (published by CruiseControlThread),
        # not cc_mode alone — in cruise mode the global limiter can be the
        # sole bidder when CC is disabled, and it must cap the user pedal.
        #   active_controller "cc"      → CC drives  → max(opd_gas, mapper_gas)
        #   active_controller "limiter" → cap        → min(user, mapper_gas)
        # CC override uses opdgasval (always OPD-mapped, regardless of CC
        # state) so the driver overrides CC through the same one-pedal feel
        # they get with CC off: below the OPD offset the gas portion is zero
        # and CC keeps commanding; above it the driver's intent wins.
        # Note: the OPD *brake* side (coast-down) is suppressed in
        # main_pedal_thread when CC is commanding — the truck's brake comes
        # only from user brake pedal + mapper_brake here, so OPD doesn't
        # fight CC.
        self._prev_user_gas = a
        self._prev_user_opd_gas = opdgasval

        cc_overridden_by_opd = False
        if not manual_clutch:
            if cruise_active:
                if cruise_active_controller == "limiter":
                    a = min(a, mapper_gas)
                else:
                    cc_overridden_by_opd = opdgasval > mapper_gas
                    a = max(opdgasval, mapper_gas)

        # Brake merge: when the user is overriding CC/ACC with OPD gas, drop
        # the mapper's brake bid too — otherwise the truck fights itself with
        # simultaneous gas and brake. Limiter brake always passes (hard cap,
        # not overridable). AEB brake is applied below independently.
        if not cc_overridden_by_opd:
            b = max(b, mapper_brake)

        # Holding brake — final floor on the merged brake so the truck does
        # not creep on slopes when any controller (user, mapper, AEB) is
        # driving. The FSM mirrors the one previously in main_pedal_thread;
        # transitions are gated by user gas (gasval / opdgasval) so the user
        # still owns whether to enter/leave the stopped state.
        offset = Settings.offset_variable or 0.0
        effective_gas = max(gasval, opdgasval)
        slope_val = road_pitch

        if self._stopped:
            hold_add = 0.0
            if gear > 0 and speed_kmh < 3 and 0 < effective_gas <= (0.7 + offset * 0.7):
                hold_add = min(
                    0.03 * (((-round(speed_kmh + 0.8, 1) + 4) ** 5) / (4 ** 5)) + slope_val * 2,
                    0.3,
                )
            elif gear < 0 and speed_kmh > -3 and 0 < effective_gas <= (0.7 + offset * 0.7):
                hold_add = min(
                    0.03 * (((round(speed_kmh + 0.8, 1) + 4) ** 5) / (4 ** 5)) - slope_val * 2,
                    0.3,
                )
            elif effective_gas == 0 and gear != 0:
                hold_add = 0.06

            b_with_hold = b + hold_add
            delta_time = time.monotonic() - self._prev_stop_mono
            t_blend = 0.5
            if self._prev_stop_mono != 0.0 and delta_time < t_blend:
                b = (
                    b_with_hold * (delta_time / t_blend)
                    + self._prev_brake_snapshot * (1.0 - delta_time / t_blend)
                )
            else:
                b = b_with_hold
                self._prev_stop_mono = 0.0

        prev_stopped = self._stopped
        if (
            -0.1 <= speed_kmh <= 0.1
            and gasval == 0
            and gear != 0
            and not self._stopped
        ):
            self._stopped = True
        elif self._stopped and (
            (speed_kmh >= 4 and gear > 0) or (speed_kmh <= -4 and gear < 0)
        ):
            self._stopped = False
            self._prev_stop_mono = 0.0
        elif self._stopped and opdgasval > 0.75:
            self._stopped = False
            self._prev_stop_mono = 0.0
        elif (
            self._stopped
            and cruise_active
            and cruise_active_controller == "cc"
            and mapper_gas > 0.02
            and not _aeb_active
        ):
            # CC has decided to launch (mapper is bidding gas) — release the
            # hold so its gas command isn't trapped by the brake floor. Without
            # this, mapper_gas and hold-brake fight at standstill and the truck
            # never rolls out from behind a lead.
            self._stopped = False
            self._prev_stop_mono = 0.0
        if park_brake and -2 <= speed_kmh <= 2 and not self._stopped:
            self._stopped = True

        if self._stopped and not prev_stopped:
            # False→True edge: the hold-brake block did not run this tick (it
            # is gated on the previous-tick value of _stopped), so b is the
            # merged brake without any hold addition. Snapshot it so the
            # 0.5 s blend can ramp from that level up to "merge + hold".
            self._prev_brake_snapshot = b
            self._prev_stop_mono = time.monotonic()

        # AEB additive FF pedal — always-on whenever AEB sees a real threat,
        # independent of engagement latch. Dropped on full-gas user override.
        if AEB_ff_decel > 0.0 and gasval < 0.8:
            aeb_ff_pedal = self._accel_mapper._brake_pedal_from_decel(
                AEB_ff_decel,
                max(self._capacity_tracker.max_brake_ms2, 0.1),
            )
            b = max(b, aeb_ff_pedal)

        # AEB active — closed-loop decel controller writes the brake pedal
        # directly from AEB_target_decel_ms2 (FF + small PI on lead-compensated
        # decel) and gas is suppressed.
        if _aeb_active:
            aeb_pedal = self._aeb_controller.step(
                target_decel_ms2=AEB_target_decel,
                measured_lead_decel_ms2=measured_decel_lead_ms2,
                max_brake_ms2=max(self._capacity_tracker.max_brake_ms2, 0.1),
                ff_pedal_fn=self._accel_mapper._brake_pedal_from_decel,
                dt=dt_aeb,
            )
            b = max(b, aeb_pedal)
            a = 0.0

        # Brake threshold hysteresis: suppress flicker from rapid OPD/CC transitions.
        _now = time.monotonic()
        if b > 0.006:
            self._brake_active = True
            self._brake_last_active_at = _now
        elif self._brake_active:
            if a >= 0.2:
                self._brake_active = False
            elif _now - self._brake_last_active_at >= 0.15 / (a + 0.075):
                self._brake_active = False
            b = max(0.0001, b)
        if not self._brake_active:
            b = 0.0

        controller.aforward = a
        controller.abackward = b

        # Push the actual brake value sent this tick into the ring buffer so
        # cruise_control_thread can compare gameBrake (lagged readback) against
        # recent commands when deciding whether to disengage on user brake.
        self._recent_brake_outputs.append(b)
        if len(self._recent_brake_outputs) > 3:
            self._recent_brake_outputs = self._recent_brake_outputs[-3:]

        # Coast-down sample: only logs when final outputs and driver inputs are zero.
        self._log_coast_step(
            speed_ms=speed_ms,
            raw_accel_ms2=raw_a,
            slope_rad=brake_grade_rad,
            gear=tel_gear_dashboard,
            game_clutch=game_clutch,
            game_throttle=game_throttle,
            game_brake=game_brake,
            mass_kg=mass_kg,
            has_trailer=has_t,
            aforward=a,
            abackward=b,
        )

        # Update pedal capacity estimates from actual pedal values sent to the game.
        _base_brake = baseline_brake_ms2(0.0, False)
        _base_accel = baseline_accel_ms2(mass_kg, has_t)
        if b > 0.01:
            self._capacity_tracker.update_brake(
                b, measured_decel_lead_ms2, speed_ms, brake_grade_rad, _base_brake,
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
            self.data.stopped = self._stopped
            self.data.recent_brake_outputs = tuple(self._recent_brake_outputs)
            self.data.hazardsActive = tel_hazards
            self.data.horn_active = bool(getattr(controller, "horn", False))
            self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
            self.data.decel_active = False
            self.data.decel_brake_output = 0.0
            self.data.decel_measured_ms2 = measured_decel_ms2
            self.data.decel_measured_lead_ms2 = measured_decel_lead_ms2
            self.data.max_brake_ms2 = self._capacity_tracker.max_brake_ms2
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
            self.data.stopped = False
            self.data.hazardsActive = False
            self.data.horn_active = False
            self.data.airhorn_active = False
            self.data.decel_active = False
            self.data.decel_brake_output = 0.0
            self.data.decel_measured_ms2 = 0.0
            self.data.decel_measured_lead_ms2 = 0.0
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
        self._prev_measured_decel_ms2 = 0.0
        self._prev_aeb_loop_mono = None
        self._accel_mapper.close()
        self._close_coast_log()
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
