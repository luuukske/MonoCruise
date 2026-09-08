"""Driver intervention detection and the no-intervention counterfactual.

The corpus question is never "did they collide", it is "would they have collided
if the driver had done nothing". A swerve leaves a recorded geometry that reads as
a clean pass, so judging a clip on what actually happened systematically labels
driver-rescued threats as true negatives.

This module finds the moment the driver intervened, then flies a ghost ego from
that moment on the heading rate and speed it was already holding, and tests it
against the traffic's own recorded future. Everything it reports is about the
ghost, never about what actually happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.aeb.calibration import DEFAULT as _CAL

# Steering. userSteer sits under ~0.6/s of rate in ordinary driving across the
# corpus; a swerve runs 2 to 4/s.
SWERVE_RATE = 1.2
SWERVE_DELTA = 0.09
BASELINE_S = 1.0
# Pedal. A dab is not an intervention; a stamp is.
BRAKE_ONSET = 0.25
BRAKE_QUIET = 0.08
LIFT_DELTA = 0.45
# Only interventions this close to the action are evasive. Ordinary cornering
# early in a clip is not the driver dodging the thing the clip is about.
SEARCH_BEFORE_S = 4.0
SEARCH_AFTER_S = 1.5
# Counterfactual horizon and resolution. Past ~1.5 s the ghost's own drift is
# wider than a lane, so a verdict there is not worth stating as fact.
HORIZON_S = 2.5
GHOST_STEP_S = 0.05
# p90 ghost drift per replacement mode, measured on no-intervention clips. This
# is the error bar that makes the verdict graded; see the README for the table.
_DRIFT_P90 = {
    "heading": ((0.0, 0.0), (0.5, 0.66), (1.0, 1.28), (1.5, 2.74),
                (2.0, 6.48), (2.5, 11.07)),
    "speed": ((0.0, 0.0), (0.5, 0.66), (1.0, 1.09), (1.5, 2.13),
              (2.0, 3.20), (2.5, 4.42)),
    "both": ((0.0, 0.0), (0.5, 0.68), (1.0, 1.35), (1.5, 3.38),
             (2.0, 6.68), (2.5, 11.84)),
}
CREDIBLE_S = 1.5
# Bodies already overlapping at the fork make every projection "collide". That is
# a spawn pile or a decode artefact, not a threat the driver steered out of.
FORK_CLEAR_M = 0.0
MIN_IMPACT_DT_S = 0.25
# A body the ghost is not closing on is not a threat: it is being overtaken by,
# or driven away from. Only a shrinking gap counts toward the verdict.
BEHIND_M = -1.0
CLOSING_EPS = 0.02
# Held yaw above this is a yard or junction manoeuvre. Extrapolating it for the
# horizon draws a spiral, so the projection is refused rather than believed.
MAX_HELD_OMEGA = 0.35
# Below this a contact is a yard or parking scrape, not the emergency-braking
# scenario this corpus arbitrates.
MIN_GHOST_MS = 4.0
# Body samples along each vehicle's long axis, as fractions of half length.
_AXIS = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass
class Intervention:
    """When and how the driver acted, if they did."""

    found: bool = False
    kind: str = ""              # swerve | brake | lift | swerve+brake
    t: float = 0.0
    steer_before: float = 0.0
    steer_peak: float = 0.0
    steer_rate_peak: float = 0.0
    brake_peak: float = 0.0
    gas_before: float = 0.0
    speed_ms: float = 0.0
    yaw_rate_before: float = 0.0
    blinker: str = ""

    def describe(self) -> str:
        if not self.found:
            return "no driver intervention detected"
        bits = [f"{self.kind} at t={self.t:.2f}s"]
        if "swerve" in self.kind:
            bits.append(f"steer {self.steer_before:+.3f} -> {self.steer_peak:+.3f} "
                        f"(peak rate {self.steer_rate_peak:.2f}/s)")
        if "brake" in self.kind:
            bits.append(f"brake to {self.brake_peak:.2f}")
        if "lift" in self.kind:
            bits.append(f"gas dropped from {self.gas_before:.2f}")
        return ", ".join(bits)


def drift_p90(dt: float, mode: str = "both") -> float:
    """Expected p90 positional error of the ghost this far past the fork."""
    if dt <= 0.0:
        return 0.0
    table = _DRIFT_P90.get(mode or "both", _DRIFT_P90["both"])
    prev_t, prev_v = table[0]
    for t, v in table[1:]:
        if dt <= t:
            span = t - prev_t
            frac = (dt - prev_t) / span if span > 1e-9 else 0.0
            return prev_v + frac * (v - prev_v)
        prev_t, prev_v = t, v
    return prev_v


@dataclass
class Counterfactual:
    """What the ghost ego would have run into, had the driver held course.

    `verdict` is graded against the ghost's own measured drift rather than stated
    as a fact, because at the separations that matter the model error and the
    answer are the same size.
    """

    ran: bool = False
    reason: str = ""
    mode: str = ""            # heading | speed | both, what the ghost replaced
    fork_t: float = 0.0
    horizon_s: float = 0.0
    min_separation_m: float = 1e9
    dt_at_min_s: float = 0.0
    drift_at_min_m: float = 0.0
    verdict: str = "unknown"     # collides | likely | close | clear | degenerate
    fork_separation_m: float = 1e9
    collides: bool = False
    credible: bool = False
    t_impact: float | None = None
    target_vid: int | None = None
    target_speed_kmh: float = 0.0
    ghost_speed_kmh: float = 0.0
    lateral_shift_m: float = 0.0
    per_target: list = field(default_factory=list)

    def grade(self) -> None:
        """Set verdict from separation against the drift band at that moment."""
        sigma = drift_p90(self.dt_at_min_s, self.mode)
        self.drift_at_min_m = round(sigma, 2)
        if self.fork_separation_m <= FORK_CLEAR_M:
            self.verdict = "degenerate"
            self.collides = False
            self.credible = False
            return
        if self.min_separation_m <= 0.0 and self.dt_at_min_s < MIN_IMPACT_DT_S:
            self.verdict = "degenerate"
            self.collides = False
            self.credible = False
            return
        self.credible = self.dt_at_min_s <= CREDIBLE_S
        gap = self.min_separation_m
        if gap <= -sigma:
            self.verdict = "collides"
        elif gap <= 0.0:
            self.verdict = "likely"
        elif gap <= sigma:
            self.verdict = "close"
        else:
            self.verdict = "clear"
        self.collides = self.verdict == "collides"

    def describe(self) -> str:
        if not self.ran:
            return f"not run ({self.reason})"
        if self.verdict == "degenerate":
            return (f"DEGENERATE: bodies already overlap at the fork "
                    f"({self.fork_separation_m:+.1f} m), so holding course proves "
                    f"nothing. Likely a spawn pile or a decode artefact.")
        base = (f"{self.verdict.upper()}: holding course, nearest approach to vid "
                f"{self.target_vid} is {self.min_separation_m:+.1f} m at "
                f"{self.dt_at_min_s:.2f} s past the fork "
                f"(ghost drift band +/-{self.drift_at_min_m:.1f} m)")
        if not self.credible:
            base += "  [beyond the 1.5 s the ghost is trustworthy over]"
        return base


def _yaw(ego) -> float:
    return ego.rotationX * 2.0 * math.pi


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _blinker(ego) -> str:
    left = bool(getattr(ego, "blinkerLeft", False))
    right = bool(getattr(ego, "blinkerRight", False))
    if left and right:
        return "hazards"
    if left:
        return "left"
    if right:
        return "right"
    return ""


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _yaw_rate(frames, idx: int, span_s: float = 0.5) -> float:
    """Mean heading rate over the span ending at idx, unwrapped."""
    t_end = frames[idx].t_mono
    start = idx
    while start > 0 and t_end - frames[start].t_mono < span_s:
        start -= 1
    if start >= idx:
        return 0.0
    dt = frames[idx].t_mono - frames[start].t_mono
    if dt <= 1e-6:
        return 0.0
    total = 0.0
    for a, b in zip(frames[start:idx], frames[start + 1:idx + 1]):
        total += _wrap(_yaw(b.ego) - _yaw(a.ego))
    return total / dt


def detect(frames, t0: float, action_t: float = 0.0) -> Intervention:
    """Earliest driver action near the action that could have changed the outcome.

    Scoped to a window around `action_t`: a lane change two seconds into a clip
    is not the driver dodging the threat the clip was recorded for. Pedal state
    must already be attached to the frames as `brakeval` and `gasval`.
    """
    if len(frames) < 8:
        return Intervention()
    lo = action_t - SEARCH_BEFORE_S
    hi = action_t + SEARCH_AFTER_S
    out = Intervention()
    kinds: list[str] = []
    best_idx: int | None = None

    for i in range(1, len(frames)):
        t_rel = frames[i].t_mono - t0
        if not lo <= t_rel <= hi:
            continue
        window = [f for f in frames[:i]
                  if frames[i].t_mono - f.t_mono <= BASELINE_S]
        if len(window) < 4:
            continue
        base_steer = _median([f.ego.userSteer for f in window])
        dt = frames[i].t_mono - frames[i - 1].t_mono
        rate = (abs(frames[i].ego.userSteer - frames[i - 1].ego.userSteer) / dt
                if dt > 1e-4 else 0.0)
        delta = abs(frames[i].ego.userSteer - base_steer)
        swerved = rate >= SWERVE_RATE or delta >= SWERVE_DELTA
        brake_now = getattr(frames[i], "brakeval", 0.0)
        brake_base = _median([getattr(f, "brakeval", 0.0) for f in window])
        stamped = brake_now >= BRAKE_ONSET and brake_base <= BRAKE_QUIET
        gas_base = _median([getattr(f, "gasval", 0.0) for f in window])
        lifted = gas_base - getattr(frames[i], "gasval", 0.0) >= LIFT_DELTA
        if not (swerved or stamped or lifted):
            continue
        if swerved:
            kinds.append("swerve")
        if stamped:
            kinds.append("brake")
        if lifted:
            kinds.append("lift")
        best_idx = i
        out.found = True
        out.t = t_rel
        out.steer_before = base_steer
        out.steer_rate_peak = rate
        out.gas_before = gas_base
        break

    if best_idx is None:
        return out
    out.kind = "+".join(dict.fromkeys(kinds))
    tail = frames[best_idx:]
    out.steer_peak = max((f.ego.userSteer for f in tail),
                         key=lambda s: abs(s - out.steer_before), default=0.0)
    out.steer_rate_peak = max(
        (abs(b.ego.userSteer - a.ego.userSteer) / max(b.t_mono - a.t_mono, 1e-4)
         for a, b in zip(tail, tail[1:])), default=out.steer_rate_peak)
    out.brake_peak = max((getattr(f, "brakeval", 0.0) for f in tail), default=0.0)
    out.speed_ms = frames[best_idx].ego.speed
    out.yaw_rate_before = _yaw_rate(frames, best_idx)
    out.blinker = _blinker(frames[best_idx].ego)
    return out


def _samples(x: float, z: float, yaw: float, half_len: float) -> list:
    fwd_x = -math.sin(yaw)
    fwd_z = -math.cos(yaw)
    return [(x + f * half_len * fwd_x, z + f * half_len * fwd_z) for f in _AXIS]


def _bodies(veh: dict) -> list[tuple[list, float]]:
    """(axis samples, radius) for a vehicle and each of its trailers."""
    out = [(_samples(veh["x"], veh["z"], veh["yaw"], veh["length"] / 2.0),
            veh["half_w"])]
    for tr in veh.get("trailers", []) or []:
        out.append((_samples(tr["x"], tr["z"], tr["yaw"], tr["length"] / 2.0),
                    tr["half_w"]))
    return out


def _separation(ghost: tuple, targets: list[tuple[list, float]]) -> float:
    """Closest body gap, counting only bodies not wholly behind the ghost.

    A vehicle overtaking into the ghost's tail is not a threat AEB addresses:
    braking would deepen that impact, not avoid it.
    """
    gx, gz, gyaw = ghost
    fwd_x = -math.sin(gyaw)
    fwd_z = -math.cos(gyaw)
    ego_pts = _samples(gx, gz, gyaw, _CAL.ego_half_length)
    best = 1e9
    for pts, radius in targets:
        ahead = max((tx - gx) * fwd_x + (tz - gz) * fwd_z for tx, tz in pts)
        if ahead < BEHIND_M - _CAL.ego_half_length:
            continue
        for ex, ez in ego_pts:
            for tx, tz in pts:
                gap = math.hypot(ex - tx, ez - tz) - _CAL.ego_half_width - radius
                if gap < best:
                    best = gap
    return best


def run(intervention: Intervention, frames, veh_at_t, t0: float,
        horizon_s: float = HORIZON_S) -> Counterfactual:
    """Fly the ghost from the fork and test it against recorded traffic.

    `veh_at_t` maps a frame t_mono to the decoded vehicle dicts at that moment.
    Traffic keeps its own recorded motion, which is the honest limit of this
    method: a target that reacted to the real swerve is not re-simulated.
    """
    cf = Counterfactual()
    if not intervention.found:
        cf.reason = "no intervention, the recorded path is the no-action path"
        return cf
    fork_mono = t0 + intervention.t
    future = [f for f in frames if f.t_mono >= fork_mono]
    if len(future) < 4:
        cf.reason = "fewer than four frames after the fork"
        return cf

    start = future[0]
    x, z = start.ego.coordinateX, start.ego.coordinateZ
    yaw = _yaw(start.ego)
    speed = max(intervention.speed_ms, 0.0)
    if speed < MIN_GHOST_MS:
        cf.reason = (f"ego at {speed * 3.6:.0f} km/h at the fork: any contact is a "
                     f"parking scrape, not an AEB scenario")
        return cf
    omega = intervention.yaw_rate_before

    kind = intervention.kind
    hold_yaw = "swerve" in kind
    hold_speed = "brake" in kind or "lift" in kind
    if hold_yaw and abs(omega) > MAX_HELD_OMEGA:
        cf.reason = (f"held yaw rate {omega:+.2f} rad/s is a manoeuvre, not a "
                     f"course to extrapolate")
        return cf
    cf.ran = True
    cf.mode = ("both" if hold_yaw and hold_speed
               else "heading" if hold_yaw else "speed")
    cf.fork_t = intervention.t
    cf.ghost_speed_kmh = speed * 3.6
    per: dict[int, float] = {}
    t_end = min(future[-1].t_mono, fork_mono + horizon_s)
    cf.horizon_s = t_end - fork_mono

    gx, gz, gyaw = x, z, yaw
    prev_mono = fork_mono
    real_x, real_z = x, z
    at_fork = veh_at_t(start.t_mono)
    cf.fork_separation_m = min(
        (_separation((gx, gz, gyaw), _bodies(veh)) for veh in at_fork),
        default=1e9)
    for frame in future:
        if frame.t_mono > t_end:
            break
        step = frame.t_mono - prev_mono
        prev_mono = frame.t_mono
        substeps = max(1, int(step / GHOST_STEP_S))
        h = step / substeps
        # Replace only what the driver changed. A swerve leaves the speed profile
        # alone, so reusing the recorded speed removes most of the ghost's drift.
        v = speed if hold_speed else frame.ego.speed
        for _ in range(substeps):
            gx += -math.sin(gyaw) * v * h
            gz += -math.cos(gyaw) * v * h
            gyaw += omega * h if hold_yaw else 0.0
        if not hold_yaw:
            gyaw = _yaw(frame.ego)
        vehicles = veh_at_t(frame.t_mono)
        if not vehicles:
            continue
        for veh in vehicles:
            gap = _separation((gx, gz, gyaw), _bodies(veh))
            vid = int(veh["vid"])
            previous = per.get(vid)
            closing = previous is None or gap < previous - CLOSING_EPS
            if previous is None or gap < previous:
                per[vid] = gap
            if not closing:
                continue
            if gap < cf.min_separation_m:
                cf.min_separation_m = gap
                cf.dt_at_min_s = frame.t_mono - fork_mono
                cf.target_vid = vid
                cf.target_speed_kmh = veh.get("speed_kmh", 0.0)
                if gap <= 0.0 and cf.t_impact is None:
                    cf.t_impact = frame.t_mono - t0
        real_x, real_z = frame.ego.coordinateX, frame.ego.coordinateZ

    cf.grade()
    cf.lateral_shift_m = math.hypot(gx - real_x, gz - real_z)
    cf.per_target = sorted(
        ({"vid": vid, "min_separation_m": round(gap, 2)}
         for vid, gap in per.items()), key=lambda d: d["min_separation_m"])[:6]
    return cf
