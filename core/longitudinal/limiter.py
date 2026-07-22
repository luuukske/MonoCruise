"""
Speed limiter: LongitudinalController subclass.

Lifecycle is driven by the orchestrator (CruiseControlThread):
  enable() / disable() / set_target_kmh(v) / reset()

No disengage logic lives here. The orchestrator decides when to enable or
disable the limiter; this class only runs the PID.

Gains: Settings.limiter_kp/ki/kd/integral_clamp/accel_min_ms2: independent
of the CC gains so each controller can be tuned separately.
"""

from __future__ import annotations

import logging
import math

from core.settings import Settings

from .base import LongCtx, LongitudinalController, LongOutput

logger = logging.getLogger(__name__)

_TARGET_SPEED_EMA_TAU_S = 0.5

# Overshoot kp boost: multiplier reached once overshoot exceeds the blend band.
_OVERSHOOT_BOOST = 2.0
_OVERSHOOT_BOOST_BAND_MS = 0.3

# Overshoot recovery envelope.
#
# The kp term alone already asks for far more decel than any overshoot needs
# (2 km/h over bids -1.16 m/s²), so the thing that actually sets recovery speed
# is the floor the bid is clamped to, not the gain above it. A fixed floor ties
# recovery time to overshoot size: at -1.0 m/s² a 10 km/h overshoot needs 2.8 s
# even if the mapper delivers the bid perfectly, and longer whenever the
# mapper's learned road load is off. So the floor itself scales with overshoot.
#
# Cubic, so the extra authority is negligible at the boundary (0.04 m/s² at
# 2 km/h over, 0.67 at 5 km/h) and only real once the overshoot is real. A
# linear ramp would have a non-zero slope at zero error and would tug at the
# pedal every time ego drifts across the limit.
#
# The cubic term is capped at the configured PID accel limit, so the envelope
# can at most double the floor the user tuned: -2.0 m/s² at the default -1.0.
# It saturates there at 5.7 km/h over.
_OVERSHOOT_CUBIC_K = 0.25

# Engagement gate. The envelope exists for a standing overshoot the mapper is
# failing to clear, not for one that is already coming down: piling extra brake
# onto a truck that is visibly slowing is how you get rear-ended. So the cubic
# is scaled by how far ego's deceleration falls short of clearing the current
# overshoot inside _OVERSHOOT_CLEAR_S.
#
# Crucially the shortfall is measured against EXTERNAL deceleration only: the
# envelope's own contribution is subtracted back out before the comparison.
# Gating on total measured decel would be self-defeating, the envelope brakes,
# ego slows, the gate sees the slowing and leaks the envelope out, ego stops
# slowing. Subtracting its own share means the envelope holds while it is the
# thing doing the work, and only backs off when a hill, the driver, or the
# mapper is already decelerating ego on its own. That is the case the safety
# rule is about: something else is handling the overshoot, so let ego sit over
# the limit past the clear window rather than stack more brake behind it.
#
# Leak faster than it builds: the bias is toward not adding brake. The build
# constant also keeps a brief overshoot the mapper is already handling from
# ever spiking the term.
_OVERSHOOT_CLEAR_S = 1.0
_CUBIC_ENGAGE_TAU_S = 0.35
_CUBIC_LEAK_TAU_S = 0.25

# Measured accel for the gate: tracking differentiator on speed. Shorter tau
# than the mapper's own 0.30 s signal because this one must register external
# deceleration quickly: a slow differentiator lets the term build for a beat
# before it notices the driver or a hill is already braking, exactly when it
# should stay out. The engage smoothing below filters what little extra noise
# the shorter window admits.
_ACCEL_TRACK_TAU_S = 0.15


class SpeedLimiter(LongitudinalController):
    """Set-speed PID for the speed-limiter mode."""

    name = "limiter"

    def __init__(self) -> None:
        self._enabled: bool = False
        self._target_kmh: float | None = None

        self._integral_error: float = 0.0
        self._prev_shaped_error: float | None = None
        self._last_target_for_integral: float | None = None
        self._target_speed_ema_ms: float | None = None

        # Overshoot-envelope gate state.
        # Tracking differentiator on ego speed, giving the measured longitudinal
        # accel the gate reads (positive = accelerating). Same construction as
        # the mapper's own accel signal so both agree on whether ego is slowing.
        self._accel_track_smooth_ms: float | None = None
        # Envelope engagement scalar in [0, 1], separately smoothed so the cubic
        # builds slowly and leaks fast rather than tracking the noisy per-tick
        # shortfall directly.
        self._cubic_engage: float = 0.0

    @property
    def active(self) -> bool:
        return self._enabled and self._target_kmh is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def target_speed_kmh(self) -> float | None:
        return self._target_kmh

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def set_target_kmh(self, v: float) -> None:
        self._target_kmh = v

    def reset(self) -> None:
        self._integral_error = 0.0
        self._prev_shaped_error = None
        self._last_target_for_integral = None
        self._target_speed_ema_ms = None
        self._accel_track_smooth_ms = None
        self._cubic_engage = 0.0

    def step(self, ctx: LongCtx) -> LongOutput:
        if not self.active:
            self.reset()
            return LongOutput(None, False)

        target_kmh = float(self._target_kmh)

        if self._last_target_for_integral != target_kmh:
            self._integral_error = 0.0
            self._last_target_for_integral = target_kmh

        target_ms = target_kmh / 3.6
        target_ms = self._smooth_target_ema(target_ms, ctx.dt)

        kp = float(Settings.limiter_kp)
        ki = float(Settings.limiter_ki)
        kd = float(Settings.limiter_kd)
        clamp = float(Settings.limiter_integral_clamp)

        error_ms = target_ms - ctx.speed_ms
        # Power-shaped error: small errors amplified, large errors compressed.
        shaped_error = math.copysign(abs(error_ms) ** 0.9, error_ms)
        self._integral_error += shaped_error * ctx.dt
        self._integral_error = max(-clamp, min(clamp, self._integral_error))

        d_term = kd * (shaped_error - self._prev_shaped_error) / ctx.dt if self._prev_shaped_error is not None else 0.0
        self._prev_shaped_error = shaped_error

        # Boost kp when overshooting so ego decelerates faster back to the limit.
        # Blended over the first 0.3 m/s of overshoot rather than a hard step at
        # zero error: a gain discontinuity there combines with speed noise and
        # the power-shaped error (infinite slope at 0) to chatter the gas cap.
        overshoot_frac = min(1.0, max(0.0, -error_ms) / _OVERSHOOT_BOOST_BAND_MS)
        effective_kp = kp * (1.0 + (_OVERSHOOT_BOOST - 1.0) * overshoot_frac)
        wanted = effective_kp * shaped_error + ki * self._integral_error + d_term

        accel_min = float(Settings.limiter_accel_min_ms2)
        accel_min = self._overshoot_floor(accel_min, -error_ms, ctx)
        # Asymmetric clamp: only bound the lower side. Positive bids are left
        # uncapped so the mapper engages and the gas pedal cap tightens smoothly
        # as ego approaches the limit (continuous-tracker invariant, AGENTS.md).
        wanted = max(accel_min, wanted)

        # Return active=True every tick while enabled: continuous-tracker invariant
        # (AGENTS.md): the PID must run even when below the limit so the gas pedal
        # cap tightens progressively rather than snapping on at the boundary.
        return LongOutput(wanted, True)

    def _overshoot_floor(self, accel_min: float, overshoot_ms: float, ctx: LongCtx) -> float:
        """Deepen the decel floor for a standing overshoot the mapper isn't clearing.

        The extra authority comes from how far over ego is (cubic in the
        overshoot), not from whatever road load the mapper has learned, so a
        poisoned road-load bias can't slow the recovery. It is capped at the
        configured PID accel limit, so the envelope at most doubles the floor
        the user tuned.

        Gated on EXTERNAL deceleration: the envelope's own share is subtracted
        before the comparison, so it holds while it is the thing slowing ego and
        only leaks when a hill, the driver, or the mapper is already shedding the
        overshoot fast enough on its own. A truck that is visibly slowing for
        some other reason is allowed to sit over the limit past 2 s rather than
        have more brake stacked onto it, which is what would risk a rear-end.
        The gate is what confines the envelope to a constant offset the mapper
        is not dealing with.
        """
        dt = max(float(ctx.dt), 1e-4)
        accel_meas = self._update_accel_track(float(ctx.speed_ms), dt)

        cubic_ms2 = min(abs(accel_min), _OVERSHOOT_CUBIC_K * overshoot_ms ** 3) if overshoot_ms > 0.0 else 0.0

        if overshoot_ms <= 0.0:
            # Below the limit: no envelope, and let any built-up scale drain so
            # a fresh overshoot starts from zero rather than a stale value.
            self._cubic_engage += (dt / (_CUBIC_LEAK_TAU_S + dt)) * (0.0 - self._cubic_engage)
            return accel_min

        # Decel that would clear the current overshoot within the target window.
        required_decel = overshoot_ms / _OVERSHOOT_CLEAR_S
        decel_meas = max(0.0, -accel_meas)
        # External decel = whatever ego is shedding BEYOND what the limiter floor
        # is already commanding (base floor plus the cubic at last tick's
        # engagement). Subtracting the whole commanded floor, not just the cubic,
        # is what lets the envelope hold a standing overshoot: while the mapper
        # tracks or under-delivers the floor, external is ~0 and the term stays
        # engaged; only decel that overshoots the command, a hill, engine
        # braking, the driver, reads as external and leaks it out.
        commanded_floor_decel = abs(accel_min) + cubic_ms2 * self._cubic_engage
        decel_external = max(0.0, decel_meas - commanded_floor_decel)
        # 1 when nothing else is slowing ego (mapper failing), 0 once external
        # decel alone would clear the overshoot in time. Linear between.
        shortfall = 1.0 - decel_external / required_decel if required_decel > 0.0 else 0.0
        target = min(1.0, max(0.0, shortfall))

        # Asymmetric smoothing: build slowly, leak fast. Biases toward not
        # adding brake and keeps a momentary overshoot the mapper is handling
        # from spiking the term.
        tau = _CUBIC_ENGAGE_TAU_S if target > self._cubic_engage else _CUBIC_LEAK_TAU_S
        self._cubic_engage += (dt / (tau + dt)) * (target - self._cubic_engage)

        return accel_min - cubic_ms2 * self._cubic_engage

    def _update_accel_track(self, speed_ms: float, dt: float) -> float:
        """Tracking-differentiator longitudinal accel (m/s²), positive = speeding up."""
        tau = _ACCEL_TRACK_TAU_S
        if self._accel_track_smooth_ms is None:
            self._accel_track_smooth_ms = speed_ms
            return 0.0
        accel = (speed_ms - self._accel_track_smooth_ms) / tau
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
        self._accel_track_smooth_ms += alpha * (speed_ms - self._accel_track_smooth_ms)
        return accel

    def _smooth_target_ema(self, target_ms: float, dt: float) -> float:
        tau = _TARGET_SPEED_EMA_TAU_S
        alpha = dt / (tau + dt) if tau > 0.0 else 1.0
        if self._target_speed_ema_ms is None:
            self._target_speed_ema_ms = target_ms
        else:
            self._target_speed_ema_ms = (
                alpha * target_ms + (1.0 - alpha) * self._target_speed_ema_ms
            )
        return self._target_speed_ema_ms

