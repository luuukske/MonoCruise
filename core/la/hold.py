"""Parks the servo once the wheel stands on its target: the plugin's push is frozen where it is, so it does not hunt.

Pure logic, run at MonoCruise's 20 Hz on the plugin's own reading. A wheel whose reading stays within ``band`` of a
still target for ``still_s``, or rattles over no more than ``rattle`` round it for ``rattle_s`` (backlash: the motor
bouncing in its play while the rim stands), has its servo's gains and friction pushes faded to nothing over
``fade_in_s``. The plugin's integral then stands still and keeps the force that parked the wheel, so the output neither
jitters nor flips sign. A wheel standing just short of the target has its integral sped up by ``boost`` so it finishes
the move instead of creeping. The wheel or the target leaving the band brings the servo back over ``fade_out_s``. It
only ever lowers the servo's gains (the boost only the integral's rate), so every cap and guard stays as it was.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

STILL_EPS = 2e-5


@dataclass(frozen=True)
class HoldTuning:
    """``floor`` is how near the target a fine encoder must stand (lock); ``slack`` what a coarse one may add."""

    enabled: bool = True
    floor: float = 0.00025
    slack: float = 0.00025
    still_s: float = 0.15
    rattle: float = 0.003
    rattle_s: float = 0.5
    release: float = 0.001
    fade_in_s: float = 0.15
    fade_out_s: float = 0.05
    target_rate: float = 0.005
    boost: float = 1.0
    boost_from_s: float = 0.3
    boost_reach: float = 0.004
    wide: float = 0.0015
    wide_after_s: float = 3.0
    wide_still_s: float = 0.5
    hunt_push: float = 30.0
    hunt_flips: int = 4
    hunt_s: float = 2.0


@dataclass(frozen=True)
class HoldOut:
    """``gain`` scales kp, kd and the friction pushes, ``integral`` scales ki."""

    gain: float = 1.0
    integral: float = 1.0
    held: bool = False
    hunted: bool = False


class TargetHold:
    def __init__(self, tuning: HoldTuning | None = None) -> None:
        self.tune = tuning or HoldTuning()
        self.reset()

    def band(self, quantum: float) -> float:
        """How far the reading may stand off the target and count as on it: half a reading step plus ``slack``."""
        return max(self.tune.floor, 0.5 * max(0.0, quantum) + self.tune.slack)

    def reset(self) -> None:
        self.factor, self.held, self._t = 1.0, False, 0.0
        self._seen: deque[tuple[float, float]] = deque()
        self._target: float | None = None
        self._last: float | None = None
        self._still = 0.0
        self._side = 0
        self._band = 0.0
        self._flips: deque[float] = deque()
        self._hunted = False

    def update(self, dt: float, target: float | None, rate: float, reading: float, quantum: float = 0.0,
               push: float = 0.0) -> HoldOut:
        """``push`` is the servo's part now; ``hunted`` is set once per still target the servo flipped round
        ``hunt_flips`` times within ``hunt_s`` without parking (the learner steps the design down for it)."""
        tn = self.tune
        if not tn.enabled or target is None or dt <= 0.0 or abs(rate) > tn.target_rate:
            self.reset()
            return HoldOut()
        if self._target is not None and abs(target - self._target) > self.band(quantum):
            self.reset()
        self._t += dt
        self._target = target if self._target is None else self._target
        moved = self._last is not None and abs(reading - self._last) > max(STILL_EPS, 0.25 * quantum)
        self._last = reading
        self._still = 0.0 if moved else self._still + dt
        self._seen.append((self._t, reading))
        while self._seen and self._t - self._seen[0][0] > tn.rattle_s:
            self._seen.popleft()
        band = self.band(quantum)
        late = max(band, tn.wide) if self._t >= tn.wide_after_s else band
        if self._still >= tn.wide_still_s:
            band = late
        err = reading - target
        if self.held:
            if abs(err) > self._band + max(tn.release, quantum):
                self.held = False
        else:
            recent = [r for t, r in self._seen if self._t - t <= tn.still_s]
            steady = self._t >= tn.still_s and all(abs(r - target) <= band for r in recent)
            span = [r for _, r in self._seen]
            rattles = (self._t >= tn.rattle_s and max(span) - min(span) <= tn.rattle
                       and abs(0.5 * (max(span) + min(span)) - target) <= late)
            self.held = steady or rattles
            self._band = band if steady else max(late, 0.5 * (max(span) - min(span)))
        step = dt / (tn.fade_in_s if self.held else tn.fade_out_s)
        self.factor = max(0.0, self.factor - step) if self.held else min(1.0, self.factor + step)
        short = (not self.held and self._still >= tn.boost_from_s and band < abs(err) <= tn.boost_reach)
        return HoldOut(self.factor, self.factor * (tn.boost if short else 1.0), self.held, self._hunt(push))

    def _hunt(self, push: float) -> bool:
        tn = self.tune
        side = 1 if push >= tn.hunt_push else (-1 if push <= -tn.hunt_push else 0)
        if side and self._side and side != self._side and not self.held:
            self._flips.append(self._t)
        self._side = side or self._side
        while self._flips and self._t - self._flips[0] > tn.hunt_s:
            self._flips.popleft()
        if self.held or self._hunted or len(self._flips) < tn.hunt_flips:
            return False
        self._hunted = True
        return True
