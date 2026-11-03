"""Lane lines beside the ACC distance lines: LDW/LKA state for the CC panel. Qt-free; see ui/cc_panel/README.md."""

from __future__ import annotations

import math

OFF = "off"
DISABLED = "disabled"
STANDBY = "standby"
WARN = "warn"
STYLES = (DISABLED, STANDBY, WARN)

LEFT = -1
RIGHT = 1

# Detector reasons (core/lane_departure/detector.py). No lane known: no lines.
_OFF_REASONS = frozenset({"off", "no_data", "no_map", "paused", "stale", "off_route"})
# Lane known, but the system stands down for a while (speed gate, bends, forks, settling, braking).
_DISABLED_REASONS = frozenset({
    "speed", "reverse", "aeb", "brake", "bend_sharp", "route_kind", "lane_settle", "short_route", "branch", "merge",
})
# Armed. The mutes of a single close call (blinker, steering, returning) stay armed so a lane change does not flicker.
_STANDBY_REASONS = frozenset({
    "clear", "warn", "approach", "blinker", "deliberate", "intent", "evade", "latched", "returning",
    "bend_change", "bend_tight", "bend_inner", "unarmed",
})

HOLD_S = 1.0
SETTLE_S = 0.4
FADE_S = 0.25


def base_state(reason: str, plugin_ok: bool) -> str:
    """Lines without a warning. An unknown reason reads as disabled: never show armed on a guess."""
    if not plugin_ok or reason in _OFF_REASONS:
        return OFF
    if reason in _STANDBY_REASONS:
        return STANDBY
    return DISABLED


def panel_side(detector_side: int) -> int:
    """Detector side (1 left, 2 right) to the panel's -1 / +1; anything else is no side."""
    return LEFT if detector_side == 1 else RIGHT if detector_side == 2 else 0


class LaneLines:
    """Per-side style weights. A warning shows at once and is held 1 s after it stops; other changes settle, then fade."""

    def __init__(self) -> None:
        self._base = OFF
        self._pending = OFF
        self._pending_since = 0.0
        self._firing = 0
        self._warn_side = 0
        self._warn_until = -math.inf
        self._t: float | None = None
        self._fading = False
        self._w ={side: dict.fromkeys(STYLES, 0.0) for side in (LEFT, RIGHT)}

    def set(self, base: str, firing_side: int, now: float) -> None:
        self.step(now)
        if base != self._pending:
            self._pending = base
            self._pending_since = now
        if firing_side:
            self._warn_side = firing_side
            self._warn_until = math.inf
            self._w[firing_side] = {s: (1.0 if s == WARN else 0.0) for s in STYLES}
        elif self._firing:
            self._warn_until = now + HOLD_S
        self._firing = firing_side
        self.step(now)

    def target(self, side: int, now: float) -> str:
        if side == self._warn_side and now < self._warn_until:
            return WARN
        return self._base

    def step(self, now: float) -> None:
        # A fade starts from where it was asked for, not from the last time anything moved.
        dt = min(max(now - self._t, 0.0), 0.1) if self._fading and self._t is not None else 0.0
        self._t = now
        if self._pending != self._base and now - self._pending_since >= SETTLE_S:
            self._base = self._pending
        if self._warn_side and now >= self._warn_until:
            self._warn_side = 0
        for side, weights in self._w.items():
            goal = self.target(side, now)
            for style, w in weights.items():
                want = 1.0 if style == goal else 0.0
                if w < want:
                    weights[style] = min(want, w + dt / FADE_S)
                elif w > want:
                    weights[style] = max(want, w - dt / FADE_S)
        self._fading = self._unsettled(now)

    def _unsettled(self, now: float) -> bool:
        for side, weights in self._w.items():
            goal = self.target(side, now)
            if any(w != (1.0 if s == goal else 0.0) for s, w in weights.items()):
                return True
        return False

    def weights(self, side: int) -> dict[str, float]:
        return self._w[side]

    def visible(self) -> bool:
        return any(w > 0.0 for weights in self._w.values() for w in weights.values())

    def wake_in(self, now: float) -> float | None:
        """Seconds until the lines change by themselves, 0 while fading, None when settled."""
        if self._unsettled(now):
            return 0.0
        waits = []
        if self._pending != self._base:
            waits.append(self._pending_since + SETTLE_S - now)
        if self._warn_side and math.isfinite(self._warn_until):
            waits.append(self._warn_until - now)
        return max(0.0, min(waits)) if waits else None
