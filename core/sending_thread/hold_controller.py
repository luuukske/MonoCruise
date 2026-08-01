from __future__ import annotations

"""Standstill / rollback hold FSM. See core/sending_thread/README.md."""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GRAVITY_MS2: float = 9.81

# FSM thresholds and ramp timing
_HOLD_CAPTURE_SPEED_KMH: float = 2.0         # |speed| below this enters STOPPING (with decel intent)
_HOLD_HOLDING_SPEED_KMH: float = 0.3         # |speed| below this transitions STOPPING → HOLDING
_HOLD_RELEASE_ACCEL_MS2: float = 0.25        # commanded accel above this advances LAUNCHING `t`
_HOLD_RELEASE_DWELL_S: float = 0.15          # accel must stay above release for this long before `t` advances
_HOLD_LAUNCH_RAMP_S: float = 0.6             # full release at t=T
_HOLD_ROLLING_EXIT_SPEED_KMH: float = 1.0    # forward-gear speed needed to settle into ROLLING after ramp
# Bulletproof safety net: if the truck is unambiguously moving in the gear
_HOLD_DEFINITELY_ROLLING_KMH: float = 5.0

# Slope-balance pedal
_HOLD_SAFETY_MARGIN_MS2: float = 0.3         # extra decel beyond slope balance (was 0.15: bumped for rollback margin)
_HOLD_MAX_PEDAL: float = 0.7                 # clamp hold pedal
_MAX_ROAD_GRADE_RAD: float = 0.35            # matches mapper clamp
# Guaranteed minimum brake pedal while STOPPING/HOLDING so the truck has a
_HOLD_STATIONARY_MIN_PEDAL: float = 0.025

# Pitch smoothing
_HOLD_PITCH_EMA_TAU_S: float = 0.5

# Rollback integrator: closed-loop term on top of slope-FF.
_HOLD_ROLLBACK_DEADBAND_MS: float = 0.02     # 0.02 m/s ≈ 0.072 km/h: sensor-noise floor
_HOLD_ROLLBACK_GROW_GAIN_MS2_PER_MS: float = 8.0  # m/s² of decel added per (m/s · s) of rollback
_HOLD_ROLLBACK_MAX_MS2: float = 6.0          # hard ceiling on extra decel from rollback integrator
_HOLD_ROLLBACK_LEAK_TAU_S: float = 8.0       # slow bleed while held with no rollback (prevents stale growth)
# Fast drain once the truck is confirmed moving in the gear-intent direction:
_HOLD_ROLLBACK_FAST_LEAK_TAU_S: float = 1.0
# Rollback within this window of now counts as "active": retreats the launch
_HOLD_ROLLBACK_ACTIVE_WINDOW_S: float = 0.25

# Park brake debounce (gear changes can flicker parkBrake briefly)
_HOLD_PARK_BRAKE_DEBOUNCE_S: float = 0.2

# Clutch fully-released threshold for STOPPING → HOLDING capture. On a hill,
_HOLD_CLUTCH_RELEASED_THRESHOLD: float = 0.05

# Manual-launch shortcuts (driver intent obvious: skip ramp)
_HOLD_MANUAL_OPD_OVERRIDE: float = 0.75
_HOLD_MANUAL_GAS_OVERRIDE: float = 0.7       # added on top of `offset`
# `offset + _HOLD_MANUAL_GAS_OVERRIDE` can exceed 1.0 (offset 0.4 gives 1.1),
_HOLD_MANUAL_GAS_FLOOR_MAX: float = 0.95

# Proportional manual release. During manual driving `commanded_accel_ms2`
_HOLD_GAS_RELEASE_START: float = 0.02
_HOLD_GAS_RELEASE_FULL: float = 0.30

STATE_ROLLING: str = "ROLLING"
STATE_STOPPING: str = "STOPPING"
STATE_HOLDING: str = "HOLDING"
STATE_LAUNCHING: str = "LAUNCHING"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _finite_or_zero(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _road_grade_rad_from_norm(pitch: float) -> float:
    """Convert telemetry rotationY (normalized full-circle) to clamped radians. See `core/sending_thread/README.md`."""
    val = _finite_or_zero(pitch)
    val = (val + 0.5) % 1.0 - 0.5
    theta = val * 2.0 * math.pi
    return _clamp(theta, -_MAX_ROAD_GRADE_RAD, _MAX_ROAD_GRADE_RAD)


@dataclass(slots=True)
class HoldOutput:
    state: str = STATE_ROLLING
    brake_pedal: float = 0.0          # FSM's brake floor (max'd over mapper brake by caller)
    launch_ease: float = 0.0          # 0.0 fully held, 1.0 fully released (LAUNCHING)
    slope_rad: float = 0.0            # smoothed pitch for diagnostics
    rollback_decel_ms2: float = 0.0   # current rollback-integrator contribution (diagnostics)
    rollback_v_ms: float = 0.0        # detected rollback velocity (diagnostics)
    active: bool = False              # True in STOPPING/HOLDING/LAUNCHING (controls freeze_slow_i, CC integral)


class HoldController:
    """Standstill / hill-hold finite-state machine with rollback prevention. See `core/sending_thread/README.md`."""

    def __init__(self, brake_pedal_from_decel) -> None:
        self._state: str = STATE_ROLLING
        self._release_dwell_acc_s: float = 0.0
        self._launch_t: float = 0.0           # 0..ramp_s; advances when commanded_accel above threshold
        self._pitch_smooth_rad: float = 0.0
        self._pitch_seeded: bool = False
        self._park_brake_high_s: float = 0.0  # debounce accumulator
        self._rollback_decel_ms2: float = 0.0  # closed-loop integrator (extra decel beyond slope FF)
        self._since_rollback_s: float = 3600.0  # time since rollback_v was last nonzero
        self._brake_pedal_from_decel = brake_pedal_from_decel

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        """Drop all FSM state. Called on disconnect / pedal-thread loss."""
        self._state = STATE_ROLLING
        self._release_dwell_acc_s = 0.0
        self._launch_t = 0.0
        self._pitch_smooth_rad = 0.0
        self._pitch_seeded = False
        self._park_brake_high_s = 0.0
        self._rollback_decel_ms2 = 0.0
        self._since_rollback_s = 3600.0

    @staticmethod
    def _idle_creep_offset_ms2(pitch_rad: float, gear: int, speed_kmh: float) -> float:
        """Placeholder for the next plan (idle/creep thrust). See `core/sending_thread/README.md`."""
        return 0.0

    def _smooth_pitch(self, raw_pitch_norm: float, dt: float) -> float:
        theta = _road_grade_rad_from_norm(raw_pitch_norm)
        if not self._pitch_seeded:
            self._pitch_smooth_rad = theta
            self._pitch_seeded = True
            return self._pitch_smooth_rad
        tau = max(_HOLD_PITCH_EMA_TAU_S, 1e-6)
        alpha = 1.0 - math.exp(-max(dt, 1e-6) / tau)
        self._pitch_smooth_rad += alpha * (theta - self._pitch_smooth_rad)
        return self._pitch_smooth_rad

    @staticmethod
    def _rollback_velocity_ms(speed_kmh: float, gear: int) -> float:
        """Detected rollback velocity (m/s), always >= 0. See `core/sending_thread/README.md`."""
        speed_ms = speed_kmh / 3.6
        if gear > 0:
            v = max(0.0, -speed_ms)
        elif gear < 0:
            v = max(0.0, speed_ms)
        else:
            v = abs(speed_ms)
        if v < _HOLD_ROLLBACK_DEADBAND_MS:
            return 0.0
        return v

    def _update_rollback_integrator(
        self,
        rollback_v: float,
        dt: float,
        active_hold: bool,
        making_progress: bool,
    ) -> None:
        """Grow integrator on rollback, leak when held but not rolling back. See `core/sending_thread/README.md`."""
        if not active_hold:
            return
        if rollback_v > 0.0:
            self._rollback_decel_ms2 += (
                _HOLD_ROLLBACK_GROW_GAIN_MS2_PER_MS * rollback_v * dt
            )
        else:
            # No rollback this tick: bleed the integrator so a stale surface
            # (mud→pavement) doesn't keep over-braking forever.
            tau = (
                _HOLD_ROLLBACK_FAST_LEAK_TAU_S
                if making_progress
                else _HOLD_ROLLBACK_LEAK_TAU_S
            )
            leak = math.exp(-dt / max(tau, 1e-6))
            self._rollback_decel_ms2 *= leak
        self._rollback_decel_ms2 = _clamp(
            self._rollback_decel_ms2, 0.0, _HOLD_ROLLBACK_MAX_MS2
        )

    def _hold_pedal(
        self,
        pitch_rad: float,
        gear: int,
        speed_kmh: float,
        launch_ease: float,
    ) -> float:
        """Total brake pedal to keep the truck stationary. See `core/sending_thread/README.md`."""
        g_along = GRAVITY_MS2 * math.sin(pitch_rad)
        idle_offset = self._idle_creep_offset_ms2(pitch_rad, gear, speed_kmh)
        ff_required = max(0.0, abs(g_along) - idle_offset) + _HOLD_SAFETY_MARGIN_MS2
        ease_complement = 1.0 - _clamp(launch_ease, 0.0, 1.0)
        ff_required *= ease_complement

        total_required = ff_required + self._rollback_decel_ms2
        if total_required <= 0.0:
            pedal = 0.0
        else:
            try:
                pedal = float(self._brake_pedal_from_decel(total_required))
            except Exception:
                logger.debug("hold brake_pedal_from_decel failed", exc_info=True)
                pedal = 0.0

        # Guaranteed stationary pedal floor: gives the brake a perceptible
        min_pedal = _HOLD_STATIONARY_MIN_PEDAL * ease_complement
        return _clamp(max(pedal, min_pedal), 0.0, _HOLD_MAX_PEDAL)

    def _park_brake_debounced(self, park_brake: bool, dt: float) -> bool:
        if park_brake:
            self._park_brake_high_s = min(
                self._park_brake_high_s + max(dt, 0.0),
                _HOLD_PARK_BRAKE_DEBOUNCE_S * 4.0,
            )
        else:
            self._park_brake_high_s = 0.0
        return self._park_brake_high_s >= _HOLD_PARK_BRAKE_DEBOUNCE_S

    @staticmethod
    def _manual_launch_intent(
        gasval: float,
        opdgasval: float,
        offset: float,
    ) -> bool:
        if opdgasval >= _HOLD_MANUAL_OPD_OVERRIDE:
            return True
        gas_floor = min(
            offset + _HOLD_MANUAL_GAS_OVERRIDE, _HOLD_MANUAL_GAS_FLOOR_MAX
        )
        if max(gasval, opdgasval) > gas_floor:
            return True
        return False

    @staticmethod
    def _gas_release_fraction(opdgasval: float) -> float:
        """Proportional release fraction [0, 1] from the OPD-mapped gas."""
        span = _HOLD_GAS_RELEASE_FULL - _HOLD_GAS_RELEASE_START
        if span <= 1e-9:
            return 1.0 if opdgasval > _HOLD_GAS_RELEASE_START else 0.0
        return _clamp((opdgasval - _HOLD_GAS_RELEASE_START) / span, 0.0, 1.0)

    def update(
        self,
        *,
        speed_kmh: float,
        gear: int,
        pitch_norm: float,
        commanded_accel_ms2: float,
        gasval: float,
        opdgasval: float,
        offset: float,
        park_brake: bool,
        aeb_active: bool,
        dt: float,
        game_clutch: float = 0.0,
        auto_neutral_active: bool = False,
    ) -> HoldOutput:
        """Advance the FSM by one tick and produce a brake floor. See `core/sending_thread/README.md`."""
        dt = max(0.0, _finite_or_zero(dt))
        speed_kmh = _finite_or_zero(speed_kmh)
        gear = int(_finite_or_zero(gear))
        commanded_accel_ms2 = _finite_or_zero(commanded_accel_ms2)
        gasval = _finite_or_zero(gasval)
        opdgasval = _finite_or_zero(opdgasval)
        offset = _finite_or_zero(offset)
        game_clutch = _clamp(_finite_or_zero(game_clutch), 0.0, 1.0)

        pitch_rad = self._smooth_pitch(pitch_norm, dt)
        rollback_v = self._rollback_velocity_ms(speed_kmh, gear)
        gas_release = self._gas_release_fraction(opdgasval)

        # Active-rollback window: the ramp retreat and the ROLLING exit key on
        if rollback_v > 0.0:
            self._since_rollback_s = 0.0
        else:
            self._since_rollback_s = min(self._since_rollback_s + dt, 3600.0)
        rollback_active = self._since_rollback_s < _HOLD_ROLLBACK_ACTIVE_WINDOW_S

        park_held = self._park_brake_debounced(bool(park_brake), dt)

        forward_intent = gear > 0
        reverse_intent = gear < 0

        # Confirmed motion in the gear-intent direction (above the rollback
        # noise floor): drives the integrator's fast leak.
        speed_ms = speed_kmh / 3.6
        making_progress = (
            (forward_intent and speed_ms > _HOLD_ROLLBACK_DEADBAND_MS)
            or (reverse_intent and speed_ms < -_HOLD_ROLLBACK_DEADBAND_MS)
        )

        # Safety-net exit: if the truck is unambiguously moving in the gear
        if not aeb_active and not park_held:
            if (
                (forward_intent and speed_kmh > _HOLD_DEFINITELY_ROLLING_KMH)
                or (reverse_intent and speed_kmh < -_HOLD_DEFINITELY_ROLLING_KMH)
            ):
                if self._state != STATE_ROLLING:
                    self._state = STATE_ROLLING
                    self._launch_t = 0.0
                    self._release_dwell_acc_s = 0.0
                    self._rollback_decel_ms2 = 0.0

        # AEB always forces HOLDING (even from ROLLING). Brake stacking happens
        # downstream (AEB additive brake comes after the merge in sending_thread).
        if aeb_active:
            self._state = STATE_HOLDING
            self._launch_t = 0.0
            self._release_dwell_acc_s = 0.0

        # Park brake → HOLDING with debounce.
        if park_held and abs(speed_kmh) <= 2.0 and self._state != STATE_LAUNCHING:
            self._state = STATE_HOLDING
            self._launch_t = 0.0
            self._release_dwell_acc_s = 0.0

        # State transitions: STOPPING is engaged eagerly (no dwell) so the
        if self._state == STATE_ROLLING:
            below_capture = abs(speed_kmh) < _HOLD_CAPTURE_SPEED_KMH
            decel_intent = commanded_accel_ms2 <= 0.0
            if below_capture and decel_intent and (
                forward_intent or reverse_intent or auto_neutral_active
            ):
                self._state = STATE_STOPPING

        if self._state == STATE_STOPPING:
            # Re-emerge to ROLLING if upstream commands meaningful accel (or
            if (
                commanded_accel_ms2 > _HOLD_RELEASE_ACCEL_MS2
                or gas_release >= 1.0
            ):
                # Skip dwell entirely if gas is already being asked for:
                # straight to ROLLING (and reset rollback integrator).
                self._state = STATE_ROLLING
                self._rollback_decel_ms2 = 0.0
            elif (
                (forward_intent and speed_kmh >= _HOLD_CAPTURE_SPEED_KMH * 1.5)
                or (reverse_intent and speed_kmh <= -_HOLD_CAPTURE_SPEED_KMH * 1.5)
            ):
                self._state = STATE_ROLLING
                self._rollback_decel_ms2 = 0.0
            elif (
                abs(speed_kmh) < _HOLD_HOLDING_SPEED_KMH
                and game_clutch < _HOLD_CLUTCH_RELEASED_THRESHOLD
            ):
                # Truck is essentially stopped AND the game clutch has fully
                self._state = STATE_HOLDING
                self._release_dwell_acc_s = 0.0
                self._launch_t = 0.0

        if self._state == STATE_HOLDING:
            manual = self._manual_launch_intent(gasval, opdgasval, offset)
            if manual:
                # Driver explicitly asked to launch: start the ramp at the
                self._state = STATE_LAUNCHING
                self._launch_t = _HOLD_LAUNCH_RAMP_S
                self._release_dwell_acc_s = 0.0
            elif gas_release > 0.0:
                # Gentle manual launch: past the coast point but not flooring
                self._state = STATE_LAUNCHING
                self._launch_t = 0.0
                self._release_dwell_acc_s = 0.0
            else:
                if commanded_accel_ms2 > _HOLD_RELEASE_ACCEL_MS2:
                    self._release_dwell_acc_s += dt
                else:
                    self._release_dwell_acc_s = 0.0
                if self._release_dwell_acc_s >= _HOLD_RELEASE_DWELL_S:
                    self._state = STATE_LAUNCHING
                    self._launch_t = 0.0
                    self._release_dwell_acc_s = 0.0

        if self._state == STATE_LAUNCHING:
            manual = self._manual_launch_intent(gasval, opdgasval, offset)
            if manual:
                # Driver still pressing: pin `t` at the fully-released end.
                self._launch_t = _HOLD_LAUNCH_RAMP_S
            elif rollback_active:
                # Truck is slipping backward right now: gas is NOT enough to
                self._launch_t = max(self._launch_t - dt, 0.0)
            elif commanded_accel_ms2 >= _HOLD_RELEASE_ACCEL_MS2:
                self._launch_t = min(self._launch_t + dt, _HOLD_LAUNCH_RAMP_S)
            else:
                # Chase the gas-proportional target at the ramp rate: partial
                target_t = gas_release * _HOLD_LAUNCH_RAMP_S
                if self._launch_t < target_t:
                    self._launch_t = min(self._launch_t + dt, target_t)
                else:
                    self._launch_t = max(self._launch_t - dt, target_t)

            # Settle to ROLLING as soon as the truck is moving in the intended
            rolling_speed_ok = (
                (forward_intent and speed_kmh > _HOLD_ROLLING_EXIT_SPEED_KMH)
                or (reverse_intent and speed_kmh < -_HOLD_ROLLING_EXIT_SPEED_KMH)
            )
            if rolling_speed_ok and not rollback_active:
                self._state = STATE_ROLLING
                self._launch_t = 0.0
                self._rollback_decel_ms2 = 0.0
            elif self._launch_t <= 0.0:
                # `t` fully retreated AND the truck has not built speed:
                self._state = STATE_HOLDING
                self._release_dwell_acc_s = 0.0
                self._launch_t = 0.0

        # Closed-loop rollback integrator update: runs in every hold state.
        active_hold = self._state in (STATE_STOPPING, STATE_HOLDING, STATE_LAUNCHING)
        self._update_rollback_integrator(rollback_v, dt, active_hold, making_progress)

        # Compute brake floor for the current state. STOPPING, HOLDING and
        ease = 0.0
        brake_pedal = 0.0
        if self._state in (STATE_STOPPING, STATE_HOLDING):
            # STOPPING scales its floor with the gas-proportional release so
            if self._state == STATE_STOPPING:
                ease = gas_release
            brake_pedal = self._hold_pedal(pitch_rad, gear, speed_kmh, ease)
        elif self._state == STATE_LAUNCHING:
            t_norm = _clamp(self._launch_t / max(_HOLD_LAUNCH_RAMP_S, 1e-6), 0.0, 1.0)
            ease = 0.5 * (1.0 - math.cos(math.pi * t_norm))
            brake_pedal = self._hold_pedal(pitch_rad, gear, speed_kmh, ease)
        else:
            # ROLLING: keep integrator at zero for next entry.
            self._rollback_decel_ms2 = 0.0

        return HoldOutput(
            state=self._state,
            brake_pedal=brake_pedal,
            launch_ease=ease,
            slope_rad=pitch_rad,
            rollback_decel_ms2=self._rollback_decel_ms2,
            rollback_v_ms=rollback_v,
            active=active_hold,
        )

