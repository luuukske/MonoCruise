"""Steer-led ego path curvature with a learned gain and a grip cap.

Model, gates and measured constants: see core/radar/README.md section 11."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# Steer gain (1/m per unit gameSteer). Fleet median is 0.19 over ~2100 corpus
# clips, but per-vehicle values run 0.09 to 0.24: the learner is what matters.
DEFAULT_GAIN: float = 0.19

# Learned gain never leaves this band: outside it the samples are glitches.
GAIN_MIN: float = 0.07
GAIN_MAX: float = 0.30


@dataclass(frozen=True)
class EgoPathParams:
    """Tunables for EgoPathModel. AEB passes values from AEBCalibration."""

    gain_prior: float = DEFAULT_GAIN
    learn_enabled: bool = True
    cap_enabled: bool = True

    # Measurement window on the simulated clock (seconds of pose history).
    meas_window_s: float = 0.13
    # Learner reads a longer window: lag does not matter, precision does.
    learn_window_s: float = 0.25

    # Learner gates.
    learn_min_speed_ms: float = 5.5
    # Low enough to accept a motorway bend: at 0.02 steer and 100 km/h the yaw
    # delta over the window is still 1.5 degrees, far above the pose noise.
    learn_min_steer: float = 0.012
    learn_max_steer: float = 0.40
    learn_max_steer_span: float = 0.03
    # Above this lateral accel an underperforming steer is grip, not geometry.
    learn_max_lat_ms2: float = 3.5
    learn_tau_s: float = 8.0
    learn_ref_steer: float = 0.10
    learn_ratio_band: float = 2.0
    # Frozen after a crash-sized speed step, for this long.
    crash_freeze_s: float = 2.0
    crash_speed_step_ms: float = 1.4

    # Saturation detector.
    sat_min_speed_ms: float = 5.0
    sat_ratio: float = 0.75
    sat_release_ratio: float = 0.85
    sat_min_lat_ms2: float = 4.0
    # Evidence needed before the cap may touch the path at all, then how long
    # it takes to arrive. Confirm first: a bare ramp capped on turn-in transients.
    sat_enter_s: float = 0.15
    sat_ramp_s: float = 0.15
    sat_exit_s: float = 0.20
    # Cap sits this far above the measured curvature so a genuinely tightening
    # line is not under-predicted.
    sat_cap_margin: float = 0.10
    # The cap is an EMA of the measured line: a raw per-frame cap put the yaw
    # measurement's own jitter straight into the drawn corridor.
    sat_cap_tau_s: float = 0.12


@dataclass
class EgoPathState:
    """One frame of model output. ``kappa_path`` is what the ego arc uses."""

    kappa_path: float = 0.0
    kappa_steer: float = 0.0
    kappa_meas: float | None = None
    gain: float = DEFAULT_GAIN
    saturated: bool = False
    # Grip ceiling on |kappa| while saturated; None when the tires have room.
    kappa_cap: float | None = None
    # How much of the path the cap owns, 0 to 1. Continuous by design: a latch
    # stepped the corridor radius by half in one frame.
    sat_weight: float = 0.0


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class EgoPathModel:
    """Ego curvature: steer-led, gain learned from the driven line, grip capped.

    History may only lower the magnitude while saturation is confirmed. It never
    raises it, never flips its sign, and never replaces the steer term in the
    linear regime, which is what keeps the path as responsive as the wheel.
    """

    params: EgoPathParams = field(default_factory=EgoPathParams)
    gain: float = 0.0
    # (t_kin, yaw_rad, speed, steer) newest last.
    _hist: list[tuple[float, float, float, float]] = field(default_factory=list)
    _sat_w: float = 0.0
    _armed: bool = False
    _confirm_run_s: float = 0.0
    _cap_ema: float | None = None
    _freeze_until: float = -1.0
    _vehicle_key: str | None = None
    state: EgoPathState = field(default_factory=EgoPathState)

    def __post_init__(self) -> None:
        if self.gain <= 0.0:
            self.gain = self.params.gain_prior

    def reset(self, *, keep_gain: bool = False) -> None:
        """Drop history and saturation state; optionally keep the learned gain."""
        self._hist.clear()
        self._sat_w = 0.0
        self._armed = False
        self._confirm_run_s = 0.0
        self._cap_ema = None
        self._freeze_until = -1.0
        if not keep_gain:
            self.gain = self.params.gain_prior
        self.state = EgoPathState(gain=self.gain)

    def note_vehicle(self, key: str | None) -> None:
        """Reset the learned gain when the driven vehicle changes."""
        if key == self._vehicle_key:
            return
        self._vehicle_key = key
        self.reset()

    def step(
        self,
        t_kin: float,
        yaw_rad: float,
        speed: float,
        steer: float,
    ) -> EgoPathState:
        """Advance one radar frame on the simulated clock; returns the new state."""
        p = self.params
        prev = self._hist[-1] if self._hist else None
        dt = (t_kin - prev[0]) if prev is not None else 0.0
        if prev is not None and dt <= 0.0:
            # Frozen simulated clock (hitch, pause, repeated frame): no new info.
            return self.state
        if prev is not None and abs(speed - prev[2]) > p.crash_speed_step_ms:
            self._freeze_until = t_kin + p.crash_freeze_s
            self._sat_w = 0.0
            self._armed = False
            self._confirm_run_s = 0.0
            self._cap_ema = None

        self._hist.append((t_kin, yaw_rad, speed, steer))
        cutoff = t_kin - max(p.meas_window_s, p.learn_window_s) - 0.5
        while len(self._hist) > 2 and self._hist[0][0] < cutoff:
            self._hist.pop(0)

        kappa_meas = self._measured_kappa(p.meas_window_s)
        if p.learn_enabled and t_kin >= self._freeze_until:
            self._learn(t_kin)

        kappa_steer = self.gain * steer if speed > 0.5 else 0.0
        kappa_path = kappa_steer
        cap: float | None = None
        # After an impact the yaw rate describes the wreck, not the driven line,
        # so neither the gain nor the cap may read from it.
        if p.cap_enabled and speed > 0.5 and t_kin >= self._freeze_until:
            cap = self._update_saturation(
                dt, kappa_steer, self._kappa_cmd_ref(p.meas_window_s), kappa_meas, speed,
            )
            if cap is not None and self._sat_w > 0.0:
                capped = math.copysign(min(abs(kappa_steer), cap), kappa_steer)
                kappa_path = (1.0 - self._sat_w) * kappa_steer + self._sat_w * capped
        else:
            self._sat_w = 0.0
            self._armed = False
            self._confirm_run_s = 0.0
            self._cap_ema = None

        self.state = EgoPathState(
            kappa_path=kappa_path,
            kappa_steer=kappa_steer,
            kappa_meas=kappa_meas,
            gain=self.gain,
            saturated=self._sat_w > 0.5,
            kappa_cap=cap,
            sat_weight=self._sat_w,
        )
        return self.state

    def _measured_kappa(self, window_s: float) -> float | None:
        """Yaw rate over the window divided by speed; None when unusable.

        Yaw deltas, not a position fit: over a tenth of a second the circle fit
        through three poses is dominated by its own noise.
        """
        if len(self._hist) < 2:
            return None
        t_now, yaw_now, speed_now, _ = self._hist[-1]
        if abs(speed_now) < 1.0:
            return None
        oldest = None
        for sample in reversed(self._hist):
            oldest = sample
            if t_now - sample[0] >= window_s:
                break
        if oldest is None:
            return None
        dt = t_now - oldest[0]
        if dt <= 1e-3:
            return None
        v_mean = 0.5 * (speed_now + oldest[2])
        if abs(v_mean) < 1.0:
            return None
        return _wrap(yaw_now - oldest[1]) / dt / v_mean

    def _learn(self, t_kin: float) -> None:
        """Slow log-space EMA of measured/commanded curvature ratio."""
        p = self.params
        window = [s for s in self._hist if t_kin - s[0] <= p.learn_window_s]
        if len(window) < 3:
            return
        span = window[-1][0] - window[0][0]
        if span < p.learn_window_s * 0.6:
            return
        speeds = [s[2] for s in window]
        steers = [s[3] for s in window]
        if min(speeds) < p.learn_min_speed_ms:
            return
        steer_mean = sum(steers) / len(steers)
        if not (p.learn_min_steer <= abs(steer_mean) <= p.learn_max_steer):
            return
        if max(steers) - min(steers) > p.learn_max_steer_span:
            return
        v_mean = sum(speeds) / len(speeds)
        kappa = _wrap(window[-1][1] - window[0][1]) / span / v_mean
        if kappa * steer_mean <= 0.0:
            return
        if v_mean * v_mean * abs(kappa) > p.learn_max_lat_ms2:
            return
        ratio = kappa / steer_mean
        if not (self.gain / p.learn_ratio_band <= ratio <= self.gain * p.learn_ratio_band):
            return
        dt = window[-1][0] - window[-2][0]
        weight = min(1.0, abs(steer_mean) / p.learn_ref_steer)
        alpha = min(1.0, (dt / max(p.learn_tau_s, 1e-3)) * weight)
        log_gain = math.log(self.gain) + alpha * (math.log(ratio) - math.log(self.gain))
        self.gain = min(GAIN_MAX, max(GAIN_MIN, math.exp(log_gain)))

    def _kappa_cmd_ref(self, window_s: float) -> float:
        """Commanded curvature averaged over the measurement window.

        The ratio has to compare like with like. Measured curvature is a yaw
        delta across the window, so testing it against the instantaneous steer
        reads a fast wind-on as understeer: the wheel has already moved on.
        """
        if not self._hist:
            return 0.0
        t_now = self._hist[-1][0]
        window = [s for s in self._hist if t_now - s[0] <= window_s]
        if not window:
            window = self._hist[-1:]
        return self.gain * (sum(s[3] for s in window) / len(window))

    def _update_saturation(
        self,
        dt: float,
        kappa_steer: float,
        kappa_cmd_ref: float,
        kappa_meas: float | None,
        speed: float,
    ) -> float | None:
        """Confirm the evidence, then ramp the cap in; returns the smoothed cap.

        Two stages on purpose. The confirm window keeps a turn-in transient, where
        the measurement window still trails the wheel, from touching the path at
        all. The ramp is what stops the confirmed cap arriving as a step: latched,
        it moved the corridor radius by half in one frame.
        """
        p = self.params
        if (
            kappa_meas is None
            or speed < p.sat_min_speed_ms
            or kappa_meas * kappa_steer <= 0.0
            or kappa_meas * kappa_cmd_ref <= 0.0
        ):
            # Sign flip or no measurement: the old cap describes another turn.
            self._sat_w = 0.0
            self._armed = False
            self._confirm_run_s = 0.0
            self._cap_ema = None
            return None

        cap_raw = abs(kappa_meas) * (1.0 + p.sat_cap_margin)
        if self._cap_ema is None:
            self._cap_ema = cap_raw
        else:
            alpha = min(1.0, dt / max(p.sat_cap_tau_s, 1e-3))
            self._cap_ema += alpha * (cap_raw - self._cap_ema)
            # Smooth downward only: into a tightening corner both the window
            # and the EMA trail the real line (README §1, lag traps).
            self._cap_ema = max(self._cap_ema, cap_raw)

        lat = speed * speed * abs(kappa_meas)
        ratio = (
            abs(kappa_meas) / abs(kappa_cmd_ref) if abs(kappa_cmd_ref) > 1e-9 else 1e9
        )
        under = ratio < p.sat_ratio and lat >= p.sat_min_lat_ms2
        if under:
            self._confirm_run_s += dt
            if self._confirm_run_s >= p.sat_enter_s:
                self._armed = True
        else:
            self._confirm_run_s = 0.0
            if ratio >= p.sat_release_ratio or lat < p.sat_min_lat_ms2:
                self._armed = False

        # Armed is all or nothing: scaling the weight across the ratio band was
        # measured and rejected, it capped mild under-turn the latch never took.
        target = 1.0 if (self._armed and abs(kappa_steer) > 1e-9) else 0.0

        rate_s = p.sat_ramp_s if target > self._sat_w else p.sat_exit_s
        step = dt / max(rate_s, 1e-3)
        if target > self._sat_w:
            self._sat_w = min(target, self._sat_w + step)
        else:
            self._sat_w = max(target, self._sat_w - step)
        return self._cap_ema


def warm_gain(
    samples: list[tuple[float, float, float, float]],
    params: EgoPathParams | None = None,
) -> float:
    """Gain a live session would already hold, estimated from a whole clip.

    Replay seeds the model with this instead of the prior, because a live truck
    has been learning for far longer than an 11 second window. Deterministic and
    non-causal on purpose; clips never record the learned value (adding a clip
    field bumps CONSENT_VERSION).
    """
    p = params or EgoPathParams()
    probe = EgoPathModel(params=p)
    for t_kin, yaw_rad, speed, steer in samples:
        probe.step(t_kin, yaw_rad, speed, steer)
    return probe.gain
