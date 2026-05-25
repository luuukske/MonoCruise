from __future__ import annotations

"""
HoldController — single authority for keeping the vehicle stationary, with
closed-loop rollback prevention.

The primary purpose of this controller is to PREVENT ROLLBACK on hills. A pure
slope-feedforward brake is not enough: telemetry pitch can lag or be noisy, the
brake inverse curve depends on a learned capacity estimate, and rolling
resistance varies with surface. Any one of those being off by 10–20 % can lead
to a slow rearward creep on a hill.

Architecture:
- FSM with four states (ROLLING / STOPPING / HOLDING / LAUNCHING) that decides
  whether we are trying to hold and how aggressively to release.
- In every "hold" state (STOPPING, HOLDING, LAUNCHING) two brake terms run:
    1. Slope-balance feedforward via the mapper's inverse brake curve.
    2. A rollback integrator that grows the brake whenever the truck is moving
       against the intended direction — pure outer-loop correction for any
       residual error in the feedforward or pedal calibration. It only ever
       adds brake; it never reduces below the feedforward.
- The feedforward is applied as soon as STOPPING engages (not gated by the
  capture dwell), so the brake catches rollback as the truck crosses zero.
- LAUNCHING ramps the feedforward to zero, but the rollback integrator stays
  fully active and any significant rollback during the ramp aborts it (`t` is
  pushed back toward 0). This is what the user requested: the truck only
  drives off once gas is genuinely able to overcome the down-slope force.
- An idle/creep-thrust hook (`_idle_creep_offset_ms2`) returns 0.0 in v1; a
  later plan replaces the body without rewiring callers.

See the design plan at C:/Users/Lukas/.claude/plans/the-stopping-logic-for-floofy-kay.md
for the rationale behind each constant and edge case.
"""

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
# intent direction above this speed, force ROLLING regardless of FSM state.
# Mirrors the old `_stopped` code's `speed_kmh >= 4 → exit` line. Prevents the
# FSM from getting stuck in HOLDING (or partial LAUNCHING) at cruise speed
# when the upstream commanded_accel chatter around `_HOLD_RELEASE_ACCEL_MS2`
# prevents the normal ramp-complete exit (e.g. CC settled near setpoint where
# P-term is small).
_HOLD_DEFINITELY_ROLLING_KMH: float = 5.0

# Slope-balance pedal
_HOLD_SAFETY_MARGIN_MS2: float = 0.3         # extra decel beyond slope balance (was 0.15 — bumped for rollback margin)
_HOLD_MAX_PEDAL: float = 0.7                 # clamp hold pedal
_MAX_ROAD_GRADE_RAD: float = 0.35            # matches mapper clamp
# Guaranteed minimum brake pedal while STOPPING/HOLDING so the truck has a
# perceptible "parking bite" even on perfectly flat ground. The slope-FF on
# flat produces only ~0.008 pedal from safety_margin alone (below the
# user-perceivable threshold and barely above the brake hysteresis floor in
# sending_thread); a 0.025 floor (~0.75 m/s² decel via the inverse brake
# curve) prevents low-speed drift without feeling like a hard stop.
# Scaled by `(1 - launch_ease)` so the floor releases in lockstep with the
# slope FF during LAUNCHING and is fully gone in ROLLING.
_HOLD_STATIONARY_MIN_PEDAL: float = 0.025

# Pitch smoothing
_HOLD_PITCH_EMA_TAU_S: float = 0.5

# Rollback integrator — closed-loop term on top of slope-FF.
# `rollback_v` is the speed (m/s) moving against gear intent (or any motion in
# neutral). The integrator adds extra decel proportional to the time-integral
# of rollback_v above a small deadband. It is the safety net for any error in
# the slope FF or brake calibration.
_HOLD_ROLLBACK_DEADBAND_MS: float = 0.02     # 0.02 m/s ≈ 0.072 km/h: sensor-noise floor
_HOLD_ROLLBACK_GROW_GAIN_MS2_PER_MS: float = 8.0  # m/s² of decel added per (m/s · s) of rollback
_HOLD_ROLLBACK_MAX_MS2: float = 6.0          # hard ceiling on extra decel from rollback integrator
_HOLD_ROLLBACK_LEAK_TAU_S: float = 8.0       # slow bleed when no rollback detected (prevents stale growth)
_HOLD_ROLLBACK_ABORT_LAUNCH_MS2: float = 0.3  # integrator above this aborts an in-progress launch ramp

# Park brake debounce (gear changes can flicker parkBrake briefly)
_HOLD_PARK_BRAKE_DEBOUNCE_S: float = 0.2

# Clutch fully-released threshold for STOPPING → HOLDING capture. On a hill,
# committing to HOLDING while the (game) clutch is still partially engaged
# causes the truck to creep forward off the brake once the hold floor takes
# over — the residual clutch torque overpowers the small hold-brake. Waiting
# for full clutch release ensures the truck is genuinely coasting before we
# hand over to the FSM brake floor. Matches the mapper's gearshift gate
# (_GAME_CLUTCH_ACTIVE_THRESHOLD = 0.05).
_HOLD_CLUTCH_RELEASED_THRESHOLD: float = 0.05

# Manual-launch shortcuts (driver intent obvious — skip ramp)
_HOLD_MANUAL_OPD_OVERRIDE: float = 0.75
_HOLD_MANUAL_GAS_OVERRIDE: float = 0.7       # added on top of `offset`

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
    """Convert telemetry rotationY (normalized full-circle) to clamped radians.

    Matches the mapper's `_road_grade_from_norm` so the FSM and the mapper see
    the same slope.
    """
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
    """Standstill / hill-hold finite-state machine with rollback prevention.

    Caller (sending_thread) invokes `update()` once per tick with telemetry +
    upstream controller intent, and uses the returned `HoldOutput` to layer the
    hold brake and to gate downstream learning (mapper slow-I, CC integral).
    """

    def __init__(self, brake_pedal_from_decel) -> None:
        self._state: str = STATE_ROLLING
        self._release_dwell_acc_s: float = 0.0
        self._launch_t: float = 0.0           # 0..ramp_s; advances when commanded_accel above threshold
        self._pitch_smooth_rad: float = 0.0
        self._pitch_seeded: bool = False
        self._park_brake_high_s: float = 0.0  # debounce accumulator
        self._rollback_decel_ms2: float = 0.0  # closed-loop integrator (extra decel beyond slope FF)
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

    @staticmethod
    def _idle_creep_offset_ms2(pitch_rad: float, gear: int, speed_kmh: float) -> float:
        """Placeholder for the next plan (idle/creep thrust).

        Returns 0.0 in v1. When implemented, returns positive accel that the
        engine produces at idle on flat/downhill (the FSM brake can shrink),
        or negative on steep uphill (the FSM brake needs more).
        """
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
        """Detected rollback velocity (m/s), always >= 0.

        Rollback = motion against intended direction of travel:
          forward gear  → rollback = max(0, -speed)
          reverse gear  → rollback = max(0,  speed)
          neutral       → rollback = |speed|  (any motion is unintended)

        Below `_HOLD_ROLLBACK_DEADBAND_MS` the value is treated as zero so
        telemetry noise doesn't drive the integrator.
        """
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
    ) -> None:
        """Grow integrator on rollback, slowly leak when held but not rolling.

        Only runs while in a hold state (STOPPING/HOLDING/LAUNCHING). When
        ROLLING the integrator is held at its last value (reset on ROLLING
        entry) so it doesn't keep accumulating during normal driving.
        """
        if not active_hold:
            return
        if rollback_v > 0.0:
            self._rollback_decel_ms2 += (
                _HOLD_ROLLBACK_GROW_GAIN_MS2_PER_MS * rollback_v * dt
            )
        else:
            # No rollback this tick — slowly bleed the integrator so a stale
            # surface (mud→pavement) doesn't keep over-braking forever.
            leak = math.exp(-dt / max(_HOLD_ROLLBACK_LEAK_TAU_S, 1e-6))
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
        """Total brake pedal to keep the truck stationary.

        Two terms:
          1. Slope-balance feedforward (scaled by `1 - launch_ease` during a
             launch ramp so gas can take over smoothly).
          2. Rollback integrator (NOT scaled by `launch_ease` — if the truck
             starts rolling backward mid-ramp, we want full authority to brake
             regardless of how far the ramp has progressed).
        Pedal floor is computed in m/s² space, then mapped through the same
        inverse brake curve the mapper uses so units stay consistent.
        """
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

        # Guaranteed stationary pedal floor — gives the brake a perceptible
        # bite while held even on perfectly flat ground (where slope FF +
        # safety margin alone produces a sub-perceptible pedal value). Scales
        # with the launch ramp so it does not fight the LAUNCHING release.
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
        gas_floor = offset + _HOLD_MANUAL_GAS_OVERRIDE
        if max(gasval, opdgasval) > gas_floor:
            return True
        return False

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
    ) -> HoldOutput:
        """Advance the FSM by one tick and produce a brake floor.

        Args mirror what sending_thread already has on-hand. `pitch_norm` is
        the raw telemetry rotationY (normalized full-circle); conversion to
        radians is done internally so the FSM uses the same slope as the
        mapper. `game_clutch` is the game-applied clutch (0..1); STOPPING →
        HOLDING capture waits for it to fall below
        `_HOLD_CLUTCH_RELEASED_THRESHOLD` so residual driveline torque does
        not creep the truck off the hold floor on a slope.
        """
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

        park_held = self._park_brake_debounced(bool(park_brake), dt)

        forward_intent = gear > 0
        reverse_intent = gear < 0

        # Safety-net exit: if the truck is unambiguously moving in the gear
        # intent direction above _HOLD_DEFINITELY_ROLLING_KMH, force ROLLING
        # regardless of FSM state. Without this, the FSM can get stuck in
        # HOLDING after a partial LAUNCHING that aborted when CC's
        # commanded_accel dropped below the release threshold (e.g. when CC
        # settles within ~1.8 km/h of setpoint, P-term * 0.5 < 0.25 m/s² and
        # `launch_t` retreats back to 0 → state → HOLDING at cruise speed →
        # brake pinned by slope-balance pedal forever).
        #
        # Gated on (not aeb_active) and (not park_held) so emergency-brake
        # situations and park-brake-set are not overridden. Resets rollback
        # integrator so the next stop starts clean.
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

        # State transitions — STOPPING is engaged eagerly (no dwell) so the
        # slope FF and rollback integrator start defending the truck as soon
        # as it dips below the capture speed with decel intent.
        if self._state == STATE_ROLLING:
            below_capture = abs(speed_kmh) < _HOLD_CAPTURE_SPEED_KMH
            decel_intent = commanded_accel_ms2 <= 0.0
            if below_capture and decel_intent and (forward_intent or reverse_intent):
                self._state = STATE_STOPPING

        if self._state == STATE_STOPPING:
            # Re-emerge to ROLLING if upstream commands meaningful accel before
            # we settle, OR if the truck has gained too much speed in the
            # intended direction (false trigger). Hysteresis: exit threshold is
            # 1.5x the capture threshold.
            if commanded_accel_ms2 > _HOLD_RELEASE_ACCEL_MS2:
                # Skip dwell entirely if upstream is asking for gas already —
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
                # released — transition into HOLDING. The clutch gate matters
                # on slopes: handing over to the hold floor while the clutch
                # is still partially engaged lets residual driveline torque
                # creep the truck off the brake when the user lets go.
                # Staying in STOPPING keeps the slope-FF + rollback integrator
                # active until the driveline is genuinely disengaged.
                self._state = STATE_HOLDING
                self._release_dwell_acc_s = 0.0
                self._launch_t = 0.0

        if self._state == STATE_HOLDING:
            manual = self._manual_launch_intent(gasval, opdgasval, offset)
            if manual:
                # Driver explicitly asked to launch — start the ramp at the
                # fully-released end; the rollback integrator will still
                # immediately catch any backward motion.
                self._state = STATE_LAUNCHING
                self._launch_t = _HOLD_LAUNCH_RAMP_S
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
                # Driver still pressing — pin `t` at the fully-released end.
                self._launch_t = _HOLD_LAUNCH_RAMP_S
            elif self._rollback_decel_ms2 >= _HOLD_ROLLBACK_ABORT_LAUNCH_MS2:
                # Significant rollback while trying to launch — gas is NOT
                # enough to overcome the down-slope force. Push `t` back to 0
                # over the ramp interval so the brake returns smoothly.
                self._launch_t = max(self._launch_t - dt, 0.0)
            elif commanded_accel_ms2 >= _HOLD_RELEASE_ACCEL_MS2:
                self._launch_t = min(self._launch_t + dt, _HOLD_LAUNCH_RAMP_S)
            else:
                self._launch_t = max(self._launch_t - dt, 0.0)

            # Settle to ROLLING as soon as the truck is moving in the intended
            # direction with no active rollback — the launch is genuinely
            # underway. Snap `launch_t` to `ramp_s` on exit so the brake floor
            # collapses cleanly (cosine ease at t=ramp_s is 1.0 → FF*0 = 0).
            #
            # Dropping the prior `launch_t >= ramp_s` requirement fixes a
            # stuck-in-HOLDING bug: when CC's `commanded_accel_ms2` chatters
            # around `_HOLD_RELEASE_ACCEL_MS2` (which it does near setpoint,
            # because P-term = kp * error scales below 0.25 m/s² within
            # ~1.8 km/h of target), `launch_t` walks back and forth and can
            # hit 0 before reaching `ramp_s`. The truck would then re-enter
            # HOLDING at cruise speed and brake would be pinned at the
            # slope-balance pedal indefinitely, with CC fighting it with gas
            # — the "constantly-pressed brake at 20 km/h on hilly road"
            # symptom users reported.
            rolling_speed_ok = (
                (forward_intent and speed_kmh > _HOLD_ROLLING_EXIT_SPEED_KMH)
                or (reverse_intent and speed_kmh < -_HOLD_ROLLING_EXIT_SPEED_KMH)
            )
            if (
                rolling_speed_ok
                and self._rollback_decel_ms2 < _HOLD_ROLLBACK_ABORT_LAUNCH_MS2
            ):
                self._state = STATE_ROLLING
                self._launch_t = 0.0
                self._rollback_decel_ms2 = 0.0
            elif self._launch_t <= 0.0:
                # `t` fully retreated AND the truck has not built speed —
                # drop back to HOLDING and re-enter the release-dwell
                # process. Integrator is preserved so we don't forget what
                # we learned about this slope.
                self._state = STATE_HOLDING
                self._release_dwell_acc_s = 0.0
                self._launch_t = 0.0

        # Closed-loop rollback integrator update — runs in every hold state.
        # In LAUNCHING the integrator is what aborts an unsafe ramp; in
        # STOPPING / HOLDING it backs up any error in the slope FF.
        active_hold = self._state in (STATE_STOPPING, STATE_HOLDING, STATE_LAUNCHING)
        self._update_rollback_integrator(rollback_v, dt, active_hold)

        # Compute brake floor for the current state. STOPPING, HOLDING and
        # LAUNCHING all share the same combined FF + integrator math; only the
        # `launch_ease` differs.
        ease = 0.0
        brake_pedal = 0.0
        if self._state in (STATE_STOPPING, STATE_HOLDING):
            brake_pedal = self._hold_pedal(pitch_rad, gear, speed_kmh, 0.0)
        elif self._state == STATE_LAUNCHING:
            t_norm = _clamp(self._launch_t / max(_HOLD_LAUNCH_RAMP_S, 1e-6), 0.0, 1.0)
            ease = 0.5 * (1.0 - math.cos(math.pi * t_norm))
            brake_pedal = self._hold_pedal(pitch_rad, gear, speed_kmh, ease)
        else:
            # ROLLING — keep integrator at zero for next entry.
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
