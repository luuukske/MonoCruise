"""Lane keeping assist: a short steer back when the lane departure detector catches a drift out of the lane.

Pure logic. It starts on the detector's own warning or approach cue and ends once the truck is back inside its
line and parallel, the driver acts, or it times out; the LA core (core/la) does the steering. The detector
itself is not changed by it. See core/lane_departure/README.md, "Lane keeping assist".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from . import edges as edge_kinds
from .detector import LEVEL_CLOSE, SIDE_LEFT, SIDE_NONE, SIDE_RIGHT, Decision, LaneDetector, project

class _Seen(NamedTuple):
    gap: float
    margin: float
    drift: float
    hard: str


# Detector mutes that are the driver meaning to go there; any of them ends a pulse.
_DRIVER = frozenset({"blinker", "deliberate", "intent", "evade", "latched"})
# Detector mutes a pulse's own wheel can raise on the other side: the catch just after a pulse looks past them.
_OWN = frozenset({"", "deliberate", "intent", "latched"})
# Detector silences a running pulse rides through: the lane is still there, only the road ahead forks or joins.
_RIDE = frozenset({"branch", "merge", "short_route", "route_kind"})


@dataclass(frozen=True)
class LkaTuning:
    max_s: float = 3.0
    back_margin_m: float = 0.15
    catch_s: float = 2.5
    out_far_m: float = 1.5
    evade_hold_s: float = 5.0
    crossing_s: float = 0.5
    back_drift_ms: float = 0.05
    start_drift_ms: float = 0.2
    inner_drift_ms: float = 0.8
    own_steer: float = 0.01


@dataclass(frozen=True)
class Pulse:
    """``side`` is the edge being steered away from while a pulse runs; ``event`` names a start or an end.

    A departure the detector caught but no pulse took is ``skipped``, with ``line`` its side and ``reason`` why.
    """

    side: int = SIDE_NONE
    event: str = ""
    beyond: str = ""
    elapsed_s: float = 0.0
    line: int = SIDE_NONE
    reason: str = ""


class LaneKeeping:
    def __init__(self, tuning: LkaTuning | None = None) -> None:
        self.tune = tuning or LkaTuning()
        self._side = SIDE_NONE
        self._since = 0.0
        self._beyond = ""
        self._evade_seen = -1e9
        self._ended: tuple[int, float] = (SIDE_NONE, -1e9)
        self._caught = False
        self._ref: tuple[int, float] = (SIDE_NONE, math.nan)

    @property
    def side(self) -> int:
        return self._side

    def reset(self) -> None:
        self._side = SIDE_NONE
        self._beyond = ""

    def update(self, t: float, decision: Decision, detector: LaneDetector, frame, steered) -> Pulse:
        """This tick's pulse. ``steered`` is the LA core's last output: where it sees the truck, its correction and
        whether it yielded.
        """
        yielded = getattr(steered, "event", "") == "override" or getattr(steered, "reason", "") == "override"
        tn = self.tune
        if detector.evade_until > t:
            self._evade_seen = t
        if self._side != SIDE_NONE:
            end = self._end(t, decision, detector, frame, yielded, steered)
            if not end:
                return Pulse(self._side, "", self._beyond, t - self._since, self._side)
            pulse = Pulse(SIDE_NONE, end, self._beyond, t - self._since, self._side)
            self._ended = (self._side, t)
            self._side = SIDE_NONE
            return pulse
        side = decision.side if decision.started else self._early(detector, steered)
        caught = side == SIDE_NONE and not decision.started
        side = side or (self._catch(t, detector, steered) if caught else SIDE_NONE)
        if side == SIDE_NONE:
            return Pulse()
        cue = decision.started and decision.level == LEVEL_CLOSE
        why = ("override" if yielded else "swerve" if t - self._evade_seen < tn.evade_hold_s
               else "drift" if cue and not self._heading_out(side, steered) else "")
        if why:
            return Pulse(event="skipped", line=side, reason=why) if decision.started else Pulse()
        last, at = self._ended
        if not (side == last == self._ref[0] and t - at <= tn.catch_s):
            self._ref = (side, getattr(steered, "steer", math.nan))
        self._side = side
        self._since = t
        self._caught = caught and side != last
        self._beyond = self._class(frame, side)
        return Pulse(self._side, "start", self._beyond, 0.0, side)

    def _early(self, detector: LaneDetector, steered) -> int:
        """The detector's approach cue test where it mutes the cue in bends: on a straight or a bend's outer edge, and
        on its inner edge only for a run at ``inner_drift_ms`` or more; a slow corner cut gets the line warning alone.
        """
        tn = detector.tune
        for side, e in detector.edges.items():
            if e.hard or not e.armed_close or not e.close or (e.bend > 0.0 and e.drift < self.tune.inner_drift_ms):
                continue
            if not self._heading_out(side, steered):
                continue
            if 0.0 < e.margin <= tn.approach_range_m and e.tlc <= tn.approach_tlc_s:
                return side
        return SIDE_NONE

    def _catch(self, t: float, detector: LaneDetector, steered) -> int:
        """Just after a pulse, the other line on the detector's own tests even where it reads the wheel the pulse turned
        as the driver's intent: a pulse can only push away from its line, so the swing home is the other line's to stop.

        Its own line too while the truck still drifts out at the cue's speed: the detector keeps that line disarmed
        after its warning and reads the turned wheel coming back as intent, neither the driver while ``_unwinding``.
        """
        tn, ldw = self.tune, detector.tune
        last, at = self._ended
        if last == SIDE_NONE or t - at > tn.catch_s:
            return SIDE_NONE
        for side in (SIDE_LEFT if last == SIDE_RIGHT else SIDE_RIGHT, last):
            e = detector.edges.get(side)
            if e is None or e.hard not in _OWN or e.gap <= -tn.out_far_m:
                continue
            if side == last and not (e.close and self._heading_out(side, steered)
                                     and (not e.hard or self._unwinding(side, steered))):
                continue
            near = e.close and 0.0 < e.margin <= ldw.approach_range_m and e.tlc <= ldw.approach_tlc_s
            if near or (e.margin <= 0.0 and e.drift > 0.0):
                return side
        return SIDE_NONE

    def _unwinding(self, side: int, steered) -> bool:
        """The wheel has not turned toward that line past where it was as the line's first pulse began: a steering mute
        there is the pulse's own wheel coming back, which the detector read as intent and latched (2026-11-02 model).
        """
        ref_side, ref = self._ref
        steer = getattr(steered, "steer", math.nan)
        if ref_side != side or not (math.isfinite(ref) and math.isfinite(steer)):
            return False
        return (ref - steer if side == SIDE_RIGHT else steer - ref) <= self.tune.own_steer

    def _end(self, t: float, decision: Decision, detector: LaneDetector, frame, yielded: bool, steered) -> str:
        tn = self.tune
        e = detector.edges.get(self._side)
        if e is None:
            if decision.reason not in _RIDE:
                return "gate"
            e = self._seen(detector, frame, steered)
            if e is None:
                return "gate"
        if e.hard in _DRIVER and not (e.hard in _OWN and (self._caught or self._unwinding(self._side, steered))):
            return "driver"
        if yielded:
            return "override"
        if e.gap <= -tn.out_far_m:
            return "out"
        if e.margin >= tn.back_margin_m and e.drift <= 0.0 and self._parallel(steered):
            return "back"
        if t - self._since >= tn.max_s:
            return "timeout"
        return ""

    def _heading_out(self, side: int, steered) -> bool:
        """The truck itself drifts toward that line at ``start_drift_ms`` or more by the LA core's lane fit (none: the
        detector decides). The detector's heading comes off the route polyline, which in fast bends read up to 0.8 m/s
        toward the outer line while the truck headed home: 17 of 47 pulse starts on 2026-11-02, the 20:59:55 override
        and every stutter among them.
        """
        drift = getattr(steered, "drift_ms", math.nan)
        if not math.isfinite(drift) or not math.isfinite(getattr(steered, "offset_m", math.nan)):
            return True
        return (drift if side == SIDE_RIGHT else -drift) >= self.tune.start_drift_ms

    def _parallel(self, steered) -> bool:
        """The truck itself no longer heads toward the line, by the LA core's lane fit (none: the detector decides).

        The detector's polyline drift flickers in fast bends: pulses let go 0.2 s in, the truck still pointing out.
        """
        drift = getattr(steered, "drift_ms", math.nan)
        if not math.isfinite(drift) or not math.isfinite(getattr(steered, "offset_m", math.nan)):
            return True
        return (drift if self._side == SIDE_RIGHT else -drift) <= self.tune.back_drift_ms

    def _seen(self, detector: LaneDetector, frame, steered) -> _Seen | None:
        """The pulse's line as the LA core sees it, while the detector keeps quiet for a fork or a join ahead."""
        offset, drift = getattr(steered, "offset_m", math.nan), getattr(steered, "drift_ms", math.nan)
        if frame is None or not (math.isfinite(offset) and math.isfinite(drift)):
            return None
        sign = 1.0 if self._side == SIDE_RIGHT else -1.0
        lamp = frame.blinker_right if self._side == SIDE_RIGHT else frame.blinker_left
        gap = detector.tune.half_lane_m - frame.half_width_m - sign * offset
        return _Seen(gap, gap + detector.tune.warn_overlap_m, sign * drift, "blinker" if lamp or frame.hazards else "")

    def _class(self, frame, side: int) -> str:
        """What lies past that line where the truck would cross it, half a second ahead of the bumper."""
        pts = getattr(getattr(frame, "route", None), "pts", ())
        hit = project(pts, frame.x, frame.z) if len(pts) >= 2 else None
        if hit is None:
            return edge_kinds.UNKNOWN
        ahead = hit[0] + frame.front_m + self.tune.crossing_s * frame.speed_ms
        return edge_kinds.beyond(getattr(frame, "lanes", None), pts, ahead, 1.0 if side == SIDE_RIGHT else -1.0)
