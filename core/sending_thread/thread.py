from __future__ import annotations

"""
Sending Thread: owns SCSController and pushes inputs to the game.

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
from .hold_controller import (
    HoldController,
    STATE_HOLDING,
    STATE_LAUNCHING,
    STATE_ROLLING,
    STATE_STOPPING,
)
from .pedal_capacity import PedalCapacityTracker
from .scscontroller import SCSController
from .visualization_bar import VisualizationBar

logger = logging.getLogger(__name__)

# Coast-down logger: captures raw decel vs speed with pedals fully released,
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
    "drag_accel_ms2",  # raw_accel - slope_accel: this is what you plot vs speed
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

# Closed-loop decel controller: feedforward via the inverse brake curve plus
# a small PI on lead-compensated measured decel. Conservative gains; tuned to
# correct residual model error without fighting the FF.
_AEB_KP: float = 0.06
_AEB_KI: float = 0.04
_AEB_LEAD_CLAMP_MS2: float = 3.0

# Idle-creep compensation for the user brake path. In gear 1 below the creep
# idle-match speed the game's idle governor pushes the truck forward (up to
# ~2.3 m/s² at 20 t). The mapper's brake FF already cancels it when CC/ACC is
# commanding, but a manual brake press had to overcome it raw, so stopping in
# gear 1 needed a far harder press than gear 2. The user pedal is re-expressed
# in decel space, the live creep estimate added, and the sum mapped back
# through the inverse brake curve so the same pedal gives the same net decel
# in every gear. Ramped in over the first few % of pedal so a fully released
# brake still creeps (docking manoeuvres) with no step at the touch point;
# full compensation is reached below the OPD coast-down brake level, so
# one-pedal coast in the creep band glides to a stop instead of fighting
# the idle governor.
_CREEP_COMP_FULL_AT_PEDAL: float = 0.04
# In OPD mode the compensation additionally fades with pedal position over
# the last fraction of the coast band before the coast point, so pressing
# toward the coast point releases the compensated brake progressively
# instead of it vanishing in one step exactly at the offset.
_CREEP_COMP_GAS_FADE_FRAC: float = 0.5

# Auto neutral (Settings.auto_neutral, default off). Below 5 km/h, whenever
# the brake is on and the gas is off, shift to neutral so gear-1 idle creep
# is removed at the source; any user gas (or an ACC launch bid) shifts
# straight back to drive. Brake-on is the user pedal (OPD/two-pedal) OR an
# ACC/CC stop (tracking decel / standstill publish). Mapper/merged gas must
# not look like the user pressed gas. The creep compensation still covers
# every in-gear moment (mapper_creep_ms2 reads zero in neutral). Shifts are
# verified against telemetry gear and re-pressed a bounded number of times
# if ignored.
_AUTONEUTRAL_MAX_SPEED_KMH: float = 7.0
# Pedal deadbands. Any brake-side OPD output or physical brake above this is
# "brake pressed"; any OPD-mapped gas above this is "gas pressed".
_AUTONEUTRAL_BRAKE_ON: float = 0.01
_AUTONEUTRAL_GAS_ON: float = 0.02
# ACC/CC is "braking" for auto-neutral when the tracking bid is at least
# this negative (gentle gap-matching decel still counts). Standstill uses
# the separate speed gate below: ACC publishes 0 m/s2 at a stop, which
# would miss a pure "wanted < 0" check.
_AUTONEUTRAL_ACC_DECEL_MS2: float = -0.05
# Below this speed, a non-positive tracking bid counts as ACC/CC holding
# the stop (ACC's standstill publish is exactly 0 m/s2).
_AUTONEUTRAL_ACC_STANDSTILL_KMH: float = 2.0
_AUTONEUTRAL_WANTED_ACCEL_ON_MS2: float = 0.25  # CC/ACC launch bid: back to drive
_AUTONEUTRAL_ROLLING_EXIT_KMH: float = 8.0  # must sit above MAX_SPEED or engage exits immediately
_AUTONEUTRAL_COOLDOWN_S: float = 0.8        # min gap between neutral engagements
_AUTONEUTRAL_PRESS_S: float = 0.15          # gear button press duration
_AUTONEUTRAL_VERIFY_S: float = 0.7          # telemetry gear must confirm within this
_AUTONEUTRAL_MAX_RETRIES: int = 2
# Re-press window for the drive shift. Generous: leaving neutral is the
# safety-critical direction (a truck stuck in neutral ignores gas), so the
# FSM alternates the drive selector with a sequential shift-up (which leaves
# neutral in every transmission mode) for this long before warning.
_AUTONEUTRAL_DRIVE_WINDOW_S: float = 4.0
# Grace on the published "auto-neutral owns the gearbox" flag after the game
# confirms drive. cruise_control_thread snapshots telemetry near the top of
# its tick and evaluates its gear gate further down, so the tick that first
# reads a forward gear can still hold a gear-0 snapshot while the flag has
# already dropped. Without the grace that one tick disengages CC.
_AUTONEUTRAL_FLAG_LINGER_S: float = 0.3
# After the game refuses a neutral shift (retries exhausted), stop trying for
# this long instead of hammering the input, and warn the user once per
# session so a missing in-game binding is diagnosable.
_AUTONEUTRAL_FAIL_LOCKOUT_S: float = 30.0

# Soft creep-cancel brake while D/R is engaged. Opt-in via OPD or
# auto-neutral only (two-pedal traditionalists keep stock creep).
# Uses the mapper's creep_ms2 through the same inverse brake curve as the
# manual creep-comp path: 100% cancel when the user is on the brake side,
# faded toward zero as applied gas rises through the band below. The fade
# scales the creep DECEL target before the inverse curve, not the returned
# pedal: the brake curve is concave, so a pedal-space fade gives back decel
# slower than the driver gains thrust authority (50% pedal keeps ~60% of
# the decel). The band ends low in the gas region: any gas past the coast
# point is a thrust request, and holding residual brake deeper into the
# pedal pinned weak engines at standstill (brakes clamp with static
# authority at ~0 speed, so the truck never leaves the max-creep shape).
# Clutch fade lives in creep_ms2 itself (1 - max(user, game) clutch).
# Does NOT shift to neutral in reverse (that killed reverse lights / beeper).
_GEAR_ENGAGE_GAS_FADE_START: float = 0.02
_GEAR_ENGAGE_GAS_FADE_FULL: float = 0.12


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
    # Hold FSM (single authority for "we want zero motion"). Replaces the old
    # `stopped` bool: `hold_active` is the new equivalent flag (true in
    # STOPPING/HOLDING/LAUNCHING); `hold_state` exposes the full state name for
    # diagnostics / CSV tuning. `stopped` is kept for compatibility with
    # consumers that haven't migrated yet.
    stopped: bool = False
    hold_active: bool = False
    hold_state: str = "ROLLING"
    # True while auto-neutral owns the gearbox: it has commanded neutral, or
    # it has commanded drive and the game has not confirmed a forward gear
    # yet. cruise_control_thread reads this to exempt the shift from its
    # "can only engage in drive" disengage/engage gates. It must cover the
    # exit shift as well as the hold: telemetry keeps reporting gear 0 for
    # the length of the button press plus the game's shift, so a flag that
    # cleared the moment drive was commanded would disengage CC on every
    # ACC launch out of an auto-neutral stop.
    auto_neutral_holding: bool = False
    # Most recent brake values written to the game (last N ticks). Used by
    # cruise_control_thread to distinguish a user's in-game brake press from
    # the game echoing back our own command (which lags by a few ticks).
    recent_brake_outputs: tuple[float, ...] = (0.0, 0.0, 0.0)
    # True when the user's OPD gas exceeded the mapper's gas this tick while
    # any controller was engaged: CC/ACC bidding (max-merge) or the sole-
    # bidder limiter cap binding (min-merge). cruise_control_thread latches
    # on this to hand gas-cap authority to the global limiter for the
    # duration of the override. The limiter-branch case matters at CC/ACC
    # engagement under a binding cap: it lets the latch engage on the
    # enable tick itself, before the winner label ever flips to "cc".
    user_gas_above_mapper: bool = False
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
        # Hill-hold / stopping FSM. Reuses the mapper's brake inverse curve so
        # the slope-balance pedal stays consistent with the mapper's own brake
        # bid at standstill.
        self._hold = HoldController(self._accel_mapper.brake_pedal_from_decel)
        self._capacity_tracker = PedalCapacityTracker()
        self._aeb_controller = AEBDecelController()
        self._key_listener = None
        self._spd_smooth: float | None = None
        self._prev_spd_mono: float | None = None
        self._prev_measured_decel_ms2: float = 0.0
        # True when the mapper's gas bid won the user-pedal merge on the
        # previous tick (min in limiter mode, max in CC mode, ties to the
        # mapper). False on clutch/idle/AEB and on disconnect/pedal loss, so
        # an unknown or user-driven pedal freezes learning.
        self._prev_mapper_owned_gas: bool = False
        # Mapper reported an unsatisfiable gas-side FF on the previous tick
        # (headroom cap bid, not a trackable setpoint).
        self._prev_mapper_ff_saturated: bool = False
        # Final gas value sent to the game last tick: bumpless re-seat source
        # for commander handovers.
        self._prev_applied_gas: float = 0.0
        # Last tick's active_controller string, to detect commander handover.
        self._prev_active_controller: str = "none"
        self._prev_aeb_loop_mono: float | None = None
        self._brake_active: bool = False
        self._brake_last_active_at: float = 0.0
        # Ring buffer of the last 3 brake outputs sent to the game (oldest first).
        self._recent_brake_outputs: list[float] = [0.0, 0.0, 0.0]
        # Auto-neutral state: True while neutral has been commanded and the
        # transmission is expected to be (or become) neutral. Press
        # verification retries against telemetry gear are bounded.
        self._autoneutral_neutral: bool = False
        self._autoneutral_last_press_mono: float = 0.0
        self._autoneutral_last_engage_mono: float = 0.0
        self._autoneutral_retries: int = 0
        self._autoneutral_drive_until: float = 0.0
        self._autoneutral_flag_linger_until: float = 0.0
        self._autoneutral_lockout_until: float = 0.0
        self._autoneutral_warned: bool = False
        self._autoneutral_drive_presses: int = 0

        # Coast-down logger state
        self._coast_log_file = None
        self._coast_log_writer = None
        self._last_coast_log_mono: float = 0.0
        self._coast_log_start_mono: float | None = None
        # Last live pedal outputs held across game pause so the viz bar (and
        # SCS write) do not drop when CC nulls its bid in the menu/map.
        self._pause_held_aforward: float = 0.0
        self._pause_held_abackward: float = 0.0
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

    def _auto_neutral_owns_gearbox(self) -> bool:
        """Value published as `auto_neutral_holding`. See that field's note.

        True while neutral is commanded, while the drive shift out of it is
        still unconfirmed, and for a short grace after the confirming gear
        arrives. Pure: the linger deadline is armed by the FSM, not here.
        """
        return (
            self._autoneutral_neutral
            or self._autoneutral_drive_until > 0.0
            or time.monotonic() < self._autoneutral_flag_linger_until
        )

    def _tick_auto_neutral(
        self,
        *,
        enabled: bool,
        gear: int,
        speed_kmh: float,
        user_gas: float,
        pedal_brake: float,
        brakeval: float,
        wanted_accel_ms2: float,
        mapper_brake: float,
        cc_tracking: bool,
        manual_clutch: bool,
        aeb_active: bool,
        paused: bool,
        dt: float,
    ) -> None:
        """Advance the auto-neutral shift logic by one tick.

        Rule (below _AUTONEUTRAL_MAX_SPEED_KMH): if the brake is on and the
        gas is off, shift to neutral. Brake-on is the user pedal (OPD
        brake-side or physical brake) OR an ACC/CC stop (tracking
        commander decelerating / holding standstill — ACC publishes 0 m/s2
        at a stop and suppresses OPD coast, so the user-pedal path alone
        never sees it). Mapper/merged gas must not look like the user
        pressed gas. Returns to drive on any user gas, an ACC launch bid,
        rolling out of band, or when the setting is turned off. A driver
        reverse shift abandons without touching gears. If the game refuses
        a shift, warns once per session and locks out for
        _AUTONEUTRAL_FAIL_LOCKOUT_S.

        `wanted_accel_ms2` must be the TRACKING-commander bid only (caller
        zeroes the limiter's headroom cap). `cc_tracking` is True only when
        active_controller == "cc" so limiter-only ticks cannot look like an
        ACC standstill.
        """
        now = time.monotonic()

        # Game paused (menus, map): shifts cannot be processed, so freeze the
        # whole FSM. Deadlines slide forward by dt so verify/retry windows do
        # not expire against a wall the game never saw (which fired the
        # "shift not accepted" popup on unpause), and nothing is pressed.
        if paused:
            slide = max(dt, 0.0)
            self._autoneutral_last_press_mono += slide
            self._autoneutral_last_engage_mono += slide
            if self._autoneutral_drive_until > 0.0:
                self._autoneutral_drive_until += slide
            if self._autoneutral_flag_linger_until > 0.0:
                self._autoneutral_flag_linger_until += slide
            if self._autoneutral_lockout_until > 0.0:
                self._autoneutral_lockout_until += slide
            return

        # User pedals only. Mapper/merged gas is not "gas pressed".
        gas_pressed = user_gas > _AUTONEUTRAL_GAS_ON
        # ACC/CC stop: OPD coast is suppressed while cruise commands, and at
        # standstill ACC publishes 0 m/s2 so mapper_brake drops to 0. Treat
        # tracking-commander decel, any mapper brake, or a non-positive bid
        # while essentially stopped as brake-on. Limiter-only is excluded
        # via cc_tracking (its wanted is zeroed by the caller anyway).
        acc_stopping = cc_tracking and (
            mapper_brake > _AUTONEUTRAL_BRAKE_ON
            or wanted_accel_ms2 <= _AUTONEUTRAL_ACC_DECEL_MS2
            or (
                wanted_accel_ms2 <= 0.0
                and abs(speed_kmh) < _AUTONEUTRAL_ACC_STANDSTILL_KMH
            )
        )
        brake_pressed = (
            pedal_brake > _AUTONEUTRAL_BRAKE_ON
            or brakeval > _AUTONEUTRAL_BRAKE_ON
            or acc_stopping
        )
        # ACC/CC commanding a launch must leave neutral even with foot off.
        launch_intent = wanted_accel_ms2 > _AUTONEUTRAL_WANTED_ACCEL_ON_MS2

        if self._autoneutral_neutral:
            if gear < 0:
                self._autoneutral_neutral = False
                self._autoneutral_retries = 0
                logger.debug("auto-neutral: driver shifted to reverse, abandoning")
                return
            if (
                not enabled
                or gas_pressed
                or launch_intent
                or abs(speed_kmh) > _AUTONEUTRAL_ROLLING_EXIT_KMH
            ):
                self.toggle_bool("geardrive", _AUTONEUTRAL_PRESS_S)
                self._autoneutral_neutral = False
                self._autoneutral_retries = 0
                self._autoneutral_last_press_mono = now
                self._autoneutral_drive_until = now + _AUTONEUTRAL_DRIVE_WINDOW_S
                self._autoneutral_drive_presses = 0
                logger.debug(
                    "auto-neutral: back to drive (user_gas=%.3f wanted=%.2f "
                    "v=%.1f km/h)",
                    user_gas, wanted_accel_ms2, speed_kmh,
                )
                return
            # Verify the neutral shift took; re-press a bounded number of
            # times if telemetry still reports a forward gear.
            if (
                gear >= 1
                and now - self._autoneutral_last_press_mono >= _AUTONEUTRAL_VERIFY_S
            ):
                if self._autoneutral_retries >= _AUTONEUTRAL_MAX_RETRIES:
                    self._autoneutral_neutral = False
                    self._autoneutral_retries = 0
                    self._autoneutral_lockout_until = (
                        now + _AUTONEUTRAL_FAIL_LOCKOUT_S
                    )
                    if not self._autoneutral_warned:
                        self._autoneutral_warned = True
                        logger.warning(
                            "Auto neutral: the game did not accept the "
                            "neutral shift; feature paused",
                            extra={"popup": True},
                        )
                else:
                    self._autoneutral_retries += 1
                    self._autoneutral_last_press_mono = now
                    self.toggle_bool("gear0", _AUTONEUTRAL_PRESS_S)
            return

        # Drive re-press window after leaving neutral: if the game dropped
        # the drive shift the truck would rev in neutral on gas, so keep
        # pressing until a forward gear confirms or the window closes.
        # Presses alternate the drive selector with a sequential shift-up:
        # which input the game honours depends on the transmission mode, and
        # shift-up from neutral engages first/Drive in every mode.
        if self._autoneutral_drive_until > 0.0 and gear == 0:
            if now < self._autoneutral_drive_until:
                if now - self._autoneutral_last_press_mono >= _AUTONEUTRAL_VERIFY_S:
                    self._autoneutral_last_press_mono = now
                    self._autoneutral_drive_presses += 1
                    name = (
                        "gearup"
                        if self._autoneutral_drive_presses % 2
                        else "geardrive"
                    )
                    self.toggle_bool(name, _AUTONEUTRAL_PRESS_S)
                return
            # Window closed with the box still in neutral: warn and pause
            # instead of silently leaving the truck out of gear.
            self._autoneutral_drive_until = 0.0
            self._autoneutral_drive_presses = 0
            self._autoneutral_lockout_until = now + _AUTONEUTRAL_FAIL_LOCKOUT_S
            if not self._autoneutral_warned:
                self._autoneutral_warned = True
                logger.warning(
                    "Auto neutral: the game did not accept the drive shift; "
                    "shift back manually. Feature paused",
                    extra={"popup": True},
                )
            return
        if gear != 0:
            if self._autoneutral_drive_until > 0.0:
                # Drive confirmed. Arm the flag grace so the cruise thread
                # cannot evaluate its gear gate against a stale gear-0
                # telemetry snapshot on the very tick the flag drops.
                self._autoneutral_flag_linger_until = (
                    now + _AUTONEUTRAL_FLAG_LINGER_S
                )
            self._autoneutral_drive_until = 0.0
            self._autoneutral_drive_presses = 0

        if (
            enabled
            and not manual_clutch
            and not aeb_active
            and brake_pressed
            and not gas_pressed
            and not launch_intent
            and gear >= 1
            and -0.5 < speed_kmh < _AUTONEUTRAL_MAX_SPEED_KMH
            and now >= self._autoneutral_lockout_until
            and now - self._autoneutral_last_engage_mono >= _AUTONEUTRAL_COOLDOWN_S
        ):
            self.toggle_bool("gear0", _AUTONEUTRAL_PRESS_S)
            self._autoneutral_neutral = True
            self._autoneutral_retries = 0
            self._autoneutral_last_press_mono = now
            self._autoneutral_last_engage_mono = now
            logger.debug(
                "auto-neutral: shifting to neutral (v=%.1f km/h "
                "pedal_b=%.2f brakeval=%.2f mapper_b=%.2f wanted=%.2f "
                "acc_stop=%s)",
                speed_kmh, pedal_brake, brakeval, mapper_brake,
                wanted_accel_ms2, acc_stopping,
            )

    def _tick_gear_engage_cushion(
        self,
        *,
        enabled: bool,
        gear: int,
        pedal_brake: float,
        applied_gas: float,
        mapper_creep_ms2: float,
    ) -> float:
        """Mapper creep-cancel brake while D/R is engaged (OPD / auto-neutral).

        Maps the faded creep decel through the inverse brake curve (same as
        the manual creep-comp path). ``mapper_creep_ms2`` already includes
        clutch fade (1 - max(user, game) clutch: half clutch → half cancel).
        Scale:
          - user brake side (``pedal_brake``): 100% of that clutch-scaled
            cancel
          - otherwise: fade 100%→0% as ``applied_gas`` rises through the
            fade band, so gas progressively re-admits creep for launch.
            ``applied_gas`` is the merged pedal actually sent to the game,
            so a mapper/ACC launch releases the cushion the same way a user
            launch does (opdgasval alone left ACC fighting the full cushion
            through the creep band).
        The fade scales the decel target, not the returned pedal (concave
        curve: see the fade-band constants). Returns a pedal in [0, 1] to
        max into the output brake.
        """
        if not enabled:
            return 0.0
        if gear == 0:
            return 0.0
        if mapper_creep_ms2 <= 1e-4:
            return 0.0

        cap = float(self._capacity_tracker.max_brake_ms2)
        if cap <= 1.0:
            return 0.0

        # User braking (OPD brake side or two-pedal): keep full creep removal.
        if pedal_brake > _AUTONEUTRAL_BRAKE_ON:
            scale = 1.0
        else:
            # Progressive release with applied gas.
            span = _GEAR_ENGAGE_GAS_FADE_FULL - _GEAR_ENGAGE_GAS_FADE_START
            if span <= 1e-9:
                gas_frac = 1.0 if applied_gas > _GEAR_ENGAGE_GAS_FADE_START else 0.0
            else:
                gas_frac = min(
                    max(
                        (float(applied_gas) - _GEAR_ENGAGE_GAS_FADE_START) / span,
                        0.0,
                    ),
                    1.0,
                )
            scale = 1.0 - gas_frac
        if scale <= 0.0:
            return 0.0

        return self._accel_mapper.brake_pedal_from_decel(
            mapper_creep_ms2 * scale, cap
        )

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

    @staticmethod
    def safe_state() -> None:
        """Release the throttle command during a watchdog restart.

        Opens a fresh SCSController: the frozen thread's own handle may be
        stuck mid-flush: and zeroes ``aforward`` so a stuck non-zero throttle
        cannot keep the truck accelerating while the thread is being replaced.
        The brake command is left untouched: a stuck brake decelerates the
        truck, which is safe, and the replacement thread re-establishes full
        control within a few ticks. Never raises.
        """
        controller = None
        try:
            controller = SCSController()
            controller.aforward = 0.0
        except Exception:
            logger.exception("safe_state: failed to release throttle (suppressed)")
        finally:
            if controller is not None:
                try:
                    controller.close()
                except Exception:
                    pass

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
        self._hold.reset()

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

        tel_paused = False
        if tel_thread is not None and tel_thread.is_alive():
            try:
                with tel_thread.data._lock:
                    connected = tel_thread.data.is_connected
                    gear = tel_thread.data.gear_dashboard
                    tel_hazards = bool(tel_thread.data.hazardsActive)
                    speed_ms = tel_thread.data.speed
                    park_brake = bool(tel_thread.data.parkBrake)
                    tel_paused = bool(getattr(tel_thread.data, "paused", False))
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
        mapper_creep_ms2 = 0.0
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
        tel_gear = 0
        engine_rpm = 0.0
        if connected and tel_thread is not None and tel_thread.is_alive() and not tel_paused:
            try:
                with tel_thread.data._lock:
                    wanted_a = float(tel_thread.data.commanded_accel_ms2)
                    mass_kg = float(tel_thread.data.estimated_total_mass_kg)
                    spd_ms = float(tel_thread.data.speed)
                    has_t = bool(tel_thread.data.ego_has_trailer)
                    wheels_on_ground = int(tel_thread.data.wheels_on_ground)
                    road_pitch = float(tel_thread.data.rotationY)
                    tel_gear_dashboard = int(tel_thread.data.gear_dashboard)
                    tel_gear = int(tel_thread.data.gear)
                    engine_rpm = float(getattr(tel_thread.data, "engine_rpm", 0.0))
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
                # modes: there is no separate limiter call here.
                mapper_engaged = cruise_active or _aeb_active

                # Learn gate, integrators only. Output trajectory and the
                # mapper's physics view commit every tick inside step(), so
                # learn=False freezes adaptation without staling the mapper.
                # Learn when the mapper's bid won the previous tick's pedal
                # merge (its command is what the truck followed), except:
                # - AEB active: AEB owns the pedals.
                # - Limiter with a saturated FF: the headroom bid (allow max
                #   accel below the limit) is not a tracking setpoint, and
                #   learning toward it slams the anti-windup back-calc into
                #   the fast-I clamp. A governing limiter (holding ego at the
                #   limit, gas or brake side) has a sane FF and does learn.
                mapper_learn = (
                    mapper_engaged
                    and not _aeb_active
                    and self._prev_mapper_owned_gas
                    and not (
                        cruise_active_controller == "limiter"
                        and self._prev_mapper_ff_saturated
                    )
                )

                # Commander handover (cc <-> limiter <-> none): one-shot
                # bumpless re-seat. Wanted EMA seeds to the new bid instead
                # of slewing across the commanders' unrelated setpoints. The
                # gas anchor re-seat is directional: an incoming limiter cap
                # starts at the pedal the game actually received (no dip
                # inheriting CC's low gas mid-override), while an incoming
                # tracking commander starts fresh at its own output (anchor
                # to the user's abandoned pedal forces a rate-limited
                # throttle surge the controller never commanded).
                # The handover tick itself never learns: the saturation flag
                # lags one tick and the snapped wanted may already saturate.
                if cruise_active_controller != self._prev_active_controller:
                    self._accel_mapper.handover_reseed(
                        self._prev_applied_gas
                        if cruise_active_controller == "limiter"
                        else None
                    )
                    mapper_learn = False
                self._prev_active_controller = cruise_active_controller

                # Freeze the mapper's slow integral when the hold FSM owns
                # standstill: based on the FSM's state from the previous tick
                # (1-tick lag, negligible at >=60 Hz). Prevents the slow-I from
                # winding negative against the persistent zero-raw-accel at
                # rest and then fighting the eventual launch.
                hold_freeze_slow_i = self._hold.state in (
                    STATE_STOPPING, STATE_HOLDING, STATE_LAUNCHING
                )

                targets = self._accel_mapper.step(
                    wanted_a,
                    raw_a,
                    spd_ms,
                    mass_kg,
                    has_t,
                    max_accel_ms2=self._capacity_tracker.accel_gain_for_gear(tel_gear),
                    max_brake_ms2=self._capacity_tracker.max_brake_ms2,
                    road_pitch=road_pitch,
                    cruise_commanding=mapper_engaged,
                    gear_dashboard=tel_gear_dashboard,
                    game_throttle=game_throttle,
                    # Stronger of user vs game clutch: creep cancel and
                    # gearshift freeze must fade when either opens the
                    # driveline (0.5 clutch → 0.5 creep, 1.0 → none).
                    game_clutch=max(game_clutch, user_clutch),
                    freeze_trim=_aeb_active,
                    learn=mapper_learn,
                    freeze_slow_i=hold_freeze_slow_i,
                    cap_mode=(cruise_active_controller == "limiter"),
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
                mapper_creep_ms2 = float(targets.creep_ms2)
                self._prev_mapper_ff_saturated = bool(targets.ff_saturated)
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
            self._prev_mapper_owned_gas = False
            self._prev_applied_gas = 0.0
            self._hold.reset()
            # Game gone: any commanded neutral state is stale.
            self._autoneutral_neutral = False
            self._autoneutral_retries = 0
            self._autoneutral_drive_until = 0.0
            self._autoneutral_flag_linger_until = 0.0
            self._pause_held_aforward = 0.0
            self._pause_held_abackward = 0.0
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.stopped = False
                self.data.hold_active = False
                self.data.hold_state = STATE_ROLLING
                self.data.auto_neutral_holding = False
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
                self.data.user_gas_above_mapper = False
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
            self._prev_mapper_owned_gas = False
            self._prev_applied_gas = 0.0
            self._pause_held_aforward = 0.0
            self._pause_held_abackward = 0.0
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.stopped = self._hold.state != STATE_ROLLING
                self.data.hold_active = self._hold.state in (
                    STATE_STOPPING, STATE_HOLDING, STATE_LAUNCHING
                )
                self.data.hold_state = self._hold.state
                # Reflect the true internal state: a transient pedal glitch
                # must not read as "neutral released" to cruise_control.
                self.data.auto_neutral_holding = self._auto_neutral_owns_gearbox()
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
                self.data.user_gas_above_mapper = False
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

        if tel_paused:
            # Game paused (menu/map): freeze last pedal outputs so the live
            # visualization bar does not drop with the nulled CC bid. Mapper,
            # hold FSM, and capacity learning are left untouched above; cruise
            # also skips mapper reset on this edge.
            a = self._pause_held_aforward
            b = self._pause_held_abackward
            controller.aforward = a
            controller.abackward = b
            self._tick_bool_presses(controller)
            with self.data._lock:
                self.data.aforward = a
                self.data.abackward = b
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
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
            self._prev_mapper_owned_gas = False
            self._prev_applied_gas = 0.0
            self._pause_held_aforward = 0.0
            self._pause_held_abackward = 0.0
            controller.aforward = 0.0
            controller.abackward = 0.0
            with self.data._lock:
                self.data.aforward = 0.0
                self.data.abackward = 0.0
                self.data.stopped = self._hold.state != STATE_ROLLING
                self.data.hold_active = self._hold.state in (
                    STATE_STOPPING, STATE_HOLDING, STATE_LAUNCHING
                )
                self.data.hold_state = self._hold.state
                # Reflect the true internal state: a transient pedal glitch
                # must not read as "neutral released" to cruise_control.
                self.data.auto_neutral_holding = self._auto_neutral_owns_gearbox()
                self.data.hazardsActive = tel_hazards
                self.data.horn_active = bool(getattr(controller, "horn", False))
                self.data.airhorn_active = bool(getattr(controller, "airhorn", False))
                self.data.decel_active = False
                self.data.decel_brake_output = 0.0
                self.data.user_gas_above_mapper = False
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

        brake_exp = Settings.brake_exponent_variable or 1.0

        # Use OPD-mapped gas from main_pedal_thread (gas_output / opdgasval).
        # Raw gasval bypasses the offset deadband and must not drive the merge.
        a = float(complex(gas_output).real)
        b = complex(brake_output).real

        try:
            b = max(b, max(float(brakeval), 0.0) ** float(brake_exp))
        except Exception:
            b = max(b, max(float(brakeval), 0.0))

        b = b ** 0.91 # from going from 110% braking to 100% braking intensity
        a = float(complex(a).real)
        b = float(complex(b).real)

        # Idle-creep compensation (user pedal only). mapper_creep_ms2 is the
        # mapper's live gear-1 creep estimate: zero outside the creep band,
        # in other gears, in neutral, and once the game declutches near
        # standstill, so the term self-limits and hands off to the hold FSM
        # at the stop (and is inherently inactive while auto-neutral holds
        # the transmission open). Applied BEFORE the mapper merge: the
        # mapper's own brake bid already compensates internally (creep
        # subtracted from effective road load in its FF), so compensating
        # after the merge would double-count. The capacity guard skips the
        # remap when the brake estimate is nonsensical rather than
        # commanding a near-full pedal. In OPD mode the compensation fades
        # over the top of the coast band so approaching the coast point
        # releases the compensated brake progressively.
        # Auto-neutral keys on this pre-comp pedal (OPD/two-pedal brake-on).
        pedal_brake = b
        if mapper_creep_ms2 > 0.0 and b > 1e-4:
            cap = self._capacity_tracker.max_brake_ms2
            if cap > 1.0:
                comp_scale = min(b / _CREEP_COMP_FULL_AT_PEDAL, 1.0)
                if Settings.opd_mode_variable:
                    offset = float(Settings.offset_variable or 0.0)
                    if offset > 1e-6:
                        opd_sum = min(max(float(gasval), 0.0), 1.0) - (
                            min(max(float(brakeval), 0.0), 1.0)
                            ** (float(brake_exp) ** 0.5)
                        )
                        fade_span = max(offset * _CREEP_COMP_GAS_FADE_FRAC, 1e-6)
                        comp_scale *= min(
                            max((offset - opd_sum) / fade_span, 0.0), 1.0
                        )
                user_decel = self._accel_mapper.brake_decel_from_pedal(b, cap)
                b_comp = self._accel_mapper.brake_pedal_from_decel(
                    user_decel + mapper_creep_ms2, cap
                )
                b += (max(b_comp, b) - b) * comp_scale

        # Manual clutch gate: when the driver physically presses the clutch
        # (manual transmission), suppress all mapper gas so the user's pedal
        # commands the truck directly during the shift. Brake commands and AEB
        # are unaffected. gameClutch is excluded because automatic transmissions
        # also raise it during their own gear changes.
        manual_clutch = user_clutch > 0.1

        # User-pedal merge with the mapper's output. Keyed on which
        # controller is actually bidding (published by CruiseControlThread),
        # not cc_mode alone: in cruise mode the global limiter can be the
        # sole bidder when CC is disabled or user-overridden, and it must
        # cap the user pedal.
        #   active_controller "cc"      → CC/ACC bidding      → max(opd_gas, mapper_gas)
        #   active_controller "limiter" → limiter sole bidder → min(opd_gas, mapper_gas)
        # CC override uses opdgasval (always OPD-mapped, regardless of CC
        # state) so the driver overrides CC through the same one-pedal feel
        # they get with CC off: below the OPD offset the gas portion is zero
        # and CC keeps commanding; above it the driver's intent wins.
        # Note: the OPD *brake* side (coast-down) is suppressed in
        # main_pedal_thread when CC is commanding: the truck's brake comes
        # only from user brake pedal + mapper_brake here, so OPD doesn't
        # fight CC.
        # The one comparison all gas arbitration keys on: is the user's OPD
        # pedal higher than the mapper's gas bid? The mapper owns the merged
        # pedal when its bid wins its mode's merge (min for the limiter cap,
        # max for CC/ACC); ties resolve to the mapper in both, so a foot-off
        # 0.0 tie under a braking mapper still learns. The poisonous rail tie
        # (floored pedal against a saturated headroom cap) is excluded by the
        # ff_saturated learn gate, not here.
        pedals_higher = opdgasval > mapper_gas

        cc_overridden_by_opd = False
        mapper_owned_gas = False
        if not manual_clutch:
            if cruise_active:
                if cruise_active_controller == "limiter":
                    mapper_owned_gas = opdgasval >= mapper_gas
                    a = min(opdgasval, mapper_gas)
                else:
                    cc_overridden_by_opd = pedals_higher
                    mapper_owned_gas = not pedals_higher
                    a = max(opdgasval, mapper_gas)

        # Brake merge: when the user is overriding CC/ACC with OPD gas, drop
        # the mapper's brake bid too: otherwise the truck fights itself with
        # simultaneous gas and brake. Limiter brake always passes (hard cap,
        # not overridable). AEB brake is applied below independently.
        if not cc_overridden_by_opd:
            b = max(b, mapper_brake)

        # Commanded accel scoped to TRACKING commanders (CC/ACC) only. The
        # limiter's sole-bidder wanted_a is a positive headroom cap during
        # ordinary manual driving (allow accel up to the limit), not a
        # launch/decel intent; feeding it to the hold FSM blocked STOPPING
        # capture on every manual stop, and feeding it to auto-neutral's
        # gas_intent blocked every manual engage path. The mapper still
        # receives the raw bid: the cap is exactly what it must track.
        tracking_wanted_a = wanted_a if cruise_active_controller == "cc" else 0.0

        # Auto neutral: decided before the hold FSM so its flag can inform
        # the FSM's capture gate on this same tick (a neutral truck must
        # still get a hold floor).
        self._tick_auto_neutral(
            enabled=bool(Settings.auto_neutral),
            gear=gear,
            speed_kmh=speed_kmh,
            user_gas=opdgasval,
            pedal_brake=pedal_brake,
            brakeval=max(float(brakeval), 0.0),
            wanted_accel_ms2=tracking_wanted_a,
            mapper_brake=mapper_brake,
            cc_tracking=(cruise_active_controller == "cc"),
            manual_clutch=manual_clutch,
            aeb_active=_aeb_active,
            paused=tel_paused,
            dt=dt_aeb,
        )

        # Hold FSM: single authority for keeping the truck stationary. The
        # mapper's brake bid is left in `b` (user-confirmed choice: mapper
        # brake passes through and the FSM brake is a floor on top, so the
        # mapper's road-load FF and the FSM's slope-balance agree when both
        # are correct, and the FSM guarantees a safe minimum if the mapper's
        # estimates are off).
        offset = Settings.offset_variable or 0.0
        hold_out = self._hold.update(
            speed_kmh=speed_kmh,
            gear=gear,
            pitch_norm=road_pitch,
            commanded_accel_ms2=tracking_wanted_a,
            gasval=gasval,
            opdgasval=opdgasval,
            offset=offset,
            park_brake=park_brake,
            aeb_active=_aeb_active,
            dt=dt_aeb,
            game_clutch=game_clutch,
            auto_neutral_active=self._autoneutral_neutral,
        )
        b = max(b, hold_out.brake_pedal)

        # Mapper creep-cancel cushion (OPD or auto-neutral only). Full cancel
        # on user brake, faded out with the merged applied gas so any launch
        # (user pedal or mapper/ACC bid) re-admits creep.
        cushion = self._tick_gear_engage_cushion(
            enabled=bool(Settings.opd_mode_variable) or bool(Settings.auto_neutral),
            gear=gear,
            pedal_brake=pedal_brake,
            applied_gas=a,
            mapper_creep_ms2=mapper_creep_ms2,
        )
        if cushion > 0.0:
            b = max(b, cushion)

        # AEB additive FF pedal: sub-engagement assist that boosts active
        # user braking. Gated on brakeval so it does not phantom-brake during
        # normal manual cruising behind a slower lead (required_decel from
        # routine lead-following is non-zero but the driver has not opted into
        # automatic longitudinal control). Engagement slam path is unaffected
        # and still fires regardless of brake input.
        user_braking = brakeval > _AEB_CAL.user_brake_latch
        if AEB_ff_decel > 0.0 and gasval < 0.8 and user_braking:
            aeb_ff_pedal = self._accel_mapper._brake_pedal_from_decel(
                AEB_ff_decel,
                max(self._capacity_tracker.max_brake_ms2, 0.1),
            )
            b = max(b, aeb_ff_pedal)

        # AEB active: closed-loop decel controller writes the brake pedal
        # directly from AEB_target_decel_ms2 (FF + small PI on lead-compensated
        # decel) and gas is suppressed.
        #
        # Gated by `gasval < 0.8` to honor the same full-gas user-authority
        # rule as the engagement slam in main_pedal_thread and the FF
        # additive above. Without this gate the closed-loop keeps forcing
        # brake and zeroing gas regardless of pedal input, leaving the
        # other two AEB layers' gas overrides meaningless.
        if _aeb_active and gasval < 0.8:
            aeb_pedal = self._aeb_controller.step(
                target_decel_ms2=AEB_target_decel,
                measured_lead_decel_ms2=measured_decel_lead_ms2,
                max_brake_ms2=max(self._capacity_tracker.max_brake_ms2, 0.1),
                ff_pedal_fn=self._accel_mapper._brake_pedal_from_decel,
                dt=dt_aeb,
            )
            b = max(b, aeb_pedal)
            a = 0.0

        # Bookkeeping for next tick's learn gate and handover re-seat.
        self._prev_mapper_owned_gas = mapper_owned_gas and not _aeb_active
        self._prev_applied_gas = a

        # Brake threshold hysteresis: suppress flicker from rapid OPD/CC transitions.
        # `a` is the final gas command sent to the game, after all controller
        # arbitration and limiter caps, so brake-light release follows actual
        # game gas rather than the raw user pedal.
        
        # human made: DO NOT TOUCH
        _now = time.monotonic()
        if b > 0.006: # brake threshold
            self._brake_active = True
            self._brake_last_active_at = _now
        elif self._brake_active: # brake latching
            if a >= 0.2:
                self._brake_active = False
            elif _now - self._brake_last_active_at >= 0.15 / (a + 0.075): # more gas = faster brake release
                self._brake_active = False
            b = max(0.0001, b)
        if not self._brake_active:
            b = 0.0
        # end human made

        # Brake-light bump: when the truck is stationary in gear with the
        # engine running, guarantee a sub-perceptible pedal value so the in-
        # game brake lights stay on. Applied AFTER the hysteresis above so it
        # is not zeroed by the inactive-brake gate. Below any real braking
        # threshold; only signals intent to other drivers.
        if (
            abs(speed_kmh) < 0.5
            and gear != 0
            and engine_rpm > 100.0
            and b < 0.001
        ):
            b = 0.001

        controller.aforward = a
        controller.abackward = b
        self._pause_held_aforward = a
        self._pause_held_abackward = b

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
        # Brake learning runs every tick, not just while braking, so the
        # tracker's settle histories see releases and gaps: a new press must
        # never inherit a stale "settled" window from the previous one. It
        # gets the plain filtered decel (the lead-compensated variant exists
        # for the AEB controller's phase margin; its derivative term
        # amplifies telemetry cadence ripple, which the tracker's asymmetric
        # alpha would rectify into downward drift) and a mass-aware baseline
        # so the plausibility guards track the truck actually being driven.
        _base_brake = baseline_brake_ms2(mass_kg, has_t)
        self._capacity_tracker.update_brake(
            max(float(b), 0.0), measured_decel_ms2, speed_ms, brake_grade_rad,
            _base_brake,
            road_load_ms2=mapper_road_load_ms2,
            aeb_active=_aeb_active,
        )
        if a > 0.01:
            # Creep subtracted so gear-1 samples learn the throttle-commanded
            # gain only; leaving it in double-counts against the mapper's
            # creep FF (which already provides that accel for free).
            self._capacity_tracker.update_accel(
                a, max(0.0, raw_a), speed_ms, brake_grade_rad, game_clutch, tel_gear,
                mass_kg, has_t, road_load_ms2=mapper_road_load_ms2 - mapper_creep_ms2,
            )

        self._tick_bool_presses(controller)

        with self.data._lock:
            self.data.aforward = a
            self.data.abackward = b
            # `stopped` retained for legacy consumers: True whenever the FSM is
            # in a non-ROLLING state. `hold_active` is the new equivalent
            # (true in STOPPING/HOLDING/LAUNCHING); `hold_state` exposes the
            # state name for tuning CSV.
            self.data.stopped = hold_out.active
            self.data.hold_active = hold_out.active
            self.data.hold_state = hold_out.state
            self.data.auto_neutral_holding = self._auto_neutral_owns_gearbox()
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
            # Branch-independent: also true while the sole-bidder limiter cap
            # binds against the user pedal, so the orchestrator's override
            # latch can engage before a CC/ACC enable ever flips the winner.
            self.data.user_gas_above_mapper = (
                cruise_active and not manual_clutch and pedals_higher
            )

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
            self.data.hold_active = False
            self.data.hold_state = STATE_ROLLING
            self.data.auto_neutral_holding = False
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
        self._pause_held_aforward = 0.0
        self._pause_held_abackward = 0.0
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

