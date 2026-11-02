"""Lane keeping assist: a short steer back when the lane departure detector catches a drift out of the lane.

Pure logic. It starts on the detector's own warning or approach cue and ends once the truck is back inside its
line and parallel, the driver acts, or it times out; the LA core (core/la) does the steering. The detector
itself is not changed by it. See core/lane_departure/README.md, "Lane keeping assist".
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

from . import edges as edge_kinds
from .detector import SIDE_LEFT, SIDE_NONE, SIDE_RIGHT, Decision, LaneDetector, project

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
    per_min: int = 8
    refund_s: float = 0.5
    crossing_s: float = 0.5


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
        self._starts: deque[float] = deque()
        self._evade_seen = -1e9
        self._ended: tuple[int, float] = (SIDE_NONE, -1e9)
        self._caught = False

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
        yielded = getattr(steered, "event", "") == "override"
        tn = self.tune
        if detector.evade_until > t:
            self._evade_seen = t
        if self._side != SIDE_NONE:
            end = self._end(t, decision, detector, frame, yielded, steered)
            if not end:
                return Pulse(self._side, "", self._beyond, t - self._since, self._side)
            pulse = Pulse(SIDE_NONE, end, self._beyond, t - self._since, self._side)
            if end == "gate" and pulse.elapsed_s < tn.refund_s and self._starts and self._starts[-1] == self._since:
                self._starts.pop()
            self._ended = (self._side, t)
            self._side = SIDE_NONE
            return pulse
        side = decision.side if decision.started else self._early(detector)
        caught = side == SIDE_NONE and not decision.started
        side = side or (self._catch(t, detector) if caught else SIDE_NONE)
        if side == SIDE_NONE:
            return Pulse()
        while self._starts and t - self._starts[0] > 60.0:
            self._starts.popleft()
        why = ("override" if yielded else "swerve" if t - self._evade_seen < tn.evade_hold_s
               else "rate" if len(self._starts) >= tn.per_min else "")
        if why:
            return Pulse(event="skipped", line=side, reason=why) if decision.started else Pulse()
        self._starts.append(t)
        self._side = side
        self._since = t
        self._caught = caught
        self._beyond = self._class(frame, side)
        return Pulse(self._side, "start", self._beyond, 0.0, side)

    @staticmethod
    def _early(detector: LaneDetector) -> int:
        """The detector's approach cue test where it mutes the cue in bends, on a straight or a bend's outer edge only.

        Running wide is caught early; cutting the inside of a bend gets the line warning alone, as in LDW.
        """
        tn = detector.tune
        for side, e in detector.edges.items():
            if e.hard or e.bend > 0.0 or not e.armed_close or not e.close:
                continue
            if 0.0 < e.margin <= tn.approach_range_m and e.tlc <= tn.approach_tlc_s:
                return side
        return SIDE_NONE

    def _catch(self, t: float, detector: LaneDetector) -> int:
        """Just after a pulse, the other line on the detector's own tests even where it reads the wheel the pulse turned
        as the driver's intent: a pulse can only push away from its line, so the swing home is the other line's to stop.
        """
        tn, ldw = self.tune, detector.tune
        last, at = self._ended
        if last == SIDE_NONE or t - at > tn.catch_s:
            return SIDE_NONE
        side = SIDE_LEFT if last == SIDE_RIGHT else SIDE_RIGHT
        e = detector.edges.get(side)
        if e is None or e.hard not in _OWN or e.gap <= -tn.out_far_m:
            return SIDE_NONE
        near = e.close and 0.0 < e.margin <= ldw.approach_range_m and e.tlc <= ldw.approach_tlc_s
        return side if near or (e.margin <= 0.0 and e.drift > 0.0) else SIDE_NONE

    def _end(self, t: float, decision: Decision, detector: LaneDetector, frame, yielded: bool, steered) -> str:
        tn = self.tune
        e = detector.edges.get(self._side)
        if e is None:
            if decision.reason not in _RIDE:
                return "gate"
            e = self._seen(detector, frame, steered)
            if e is None:
                return "gate"
        if e.hard in _DRIVER and not (self._caught and e.hard in _OWN):
            return "driver"
        if yielded:
            return "override"
        if e.gap <= -tn.out_far_m:
            return "out"
        if e.margin >= tn.back_margin_m and e.drift <= 0.0:
            return "back"
        if t - self._since >= tn.max_s:
            return "timeout"
        return ""

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
