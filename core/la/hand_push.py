"""The driver's own torque on the rim, estimated from what the FFB plugin reports: what the wheel's motion needs, less
what the game and the assist push. A lane keeping pulse yields to it; see core/la/README.md, "The driver's push".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PushTuning:
    """``inertia`` is the rim with hands (``WheelTuning.inertia``); ``game_lag_s`` the plugin's low-pass on the game's
    force (``FFB_FORCE_TAU_S``), undone here. ``base_s`` averages the push before a pulse, its baseline.
    """

    inertia: float = 650.0
    game_lag_s: float = 0.10
    smooth_s: float = 0.10
    base_s: float = 0.3
    max_dt_s: float = 0.2


@dataclass(frozen=True)
class HandPush:
    """``hand`` is the driver's torque, counterclockwise positive like the wheel angle, in the plugin's units out of
    10000; ``accel`` the wheel's angular acceleration in lock per s^2.

    While a one-sided servo pushes (a lane keeping pulse), ``push`` is its push away from its line and ``against`` how
    much more the driver pushes toward that line than just before it began. Passive hands, friction and damping can
    at most cancel the push; only a driver steering toward the line goes past it (``beyond`` above zero).
    """

    hand: float = math.nan
    accel: float = math.nan
    push: float = math.nan
    against: float = math.nan

    @property
    def beyond(self) -> float:
        return self.against - max(0.0, self.push)


class HandPushEstimator:
    def __init__(self, tuning: PushTuning | None = None) -> None:
        self.tune = tuning or PushTuning()
        self.reset()

    def reset(self) -> None:
        self._rate: float | None = None
        self._game: float | None = None
        self._hand: float | None = None
        self._base: float | None = None
        self._held: float | None = None
        self._only = 0

    def update(self, dt: float, rate: float, game: float, applied: float, cancel: float, direction: int,
               only: int = 0) -> HandPush:
        """One reading: the wheel's rate (lock/s), the game's force as the plugin reports it, the assist applied and
        any cancelled game force. ``direction`` is the force sign that turns the wheel counterclockwise (0: unknown).
        """
        tn = self.tune
        values = (dt, rate, game, applied, cancel)
        if not direction or not all(math.isfinite(v) for v in values):
            self.reset()
            return HandPush()
        last_rate, last_game = self._rate, self._game
        self._rate, self._game = rate, game
        if last_rate is None or last_game is None or not 0.0 < dt <= tn.max_dt_s:
            self._hand = self._base = self._held = None
            self._only = 0
            return HandPush()
        accel = (rate - last_rate) / dt
        game_now = game + tn.game_lag_s * (game - last_game) / dt
        raw = tn.inertia * accel - direction * (game_now + applied + cancel)
        blend = 1.0 - math.exp(-dt / tn.smooth_s)
        self._hand = raw if self._hand is None else self._hand + blend * (raw - self._hand)
        only = (only > 0) - (only < 0)
        if not only:
            self._held = None
            blend = 1.0 - math.exp(-dt / tn.base_s)
            self._base = self._hand if self._base is None else self._base + blend * (self._hand - self._base)
            self._only = 0
            return HandPush(self._hand, accel)
        if not self._only:
            self._held = self._base if self._base is not None else self._hand
        self._only = only
        push = only * direction * applied
        return HandPush(self._hand, accel, push, -only * (self._hand - self._held))
