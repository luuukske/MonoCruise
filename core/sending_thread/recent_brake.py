"""Brake pedal values recently written to the game. See core/sending_thread/README.md."""

from __future__ import annotations

from collections import deque

# The game's own brake reading trails the sent pedal by 50 to 110 ms (measured).
RECENT_BRAKE_WINDOW_S: float = 0.5


class RecentBrakeOutputs:
    """Sent brake values of the last *window_s* seconds, by time and not by tick count."""

    def __init__(self, window_s: float = RECENT_BRAKE_WINDOW_S) -> None:
        self._window_s = window_s
        self._samples: deque[tuple[float, float]] = deque()

    def push(self, now: float, sent_brake: float) -> None:
        self._samples.append((now, float(sent_brake)))
        cutoff = now - self._window_s
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def values(self) -> tuple[float, ...]:
        return tuple(v for _, v in self._samples)
