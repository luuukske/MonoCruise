"""Occupancy confirm for engage/warn/risk streaks. See core/aeb/README.md."""

from __future__ import annotations

from collections import deque


class OccupancyConfirm:
    """Trailing-window qualified fraction; `window_s` may change each `observe`."""

    __slots__ = ("occupancy", "max_gap", "window_s",
                 "_samples", "_first_mono", "_gap_run")

    def __init__(self, occupancy: float, max_gap: int, window_s: float) -> None:
        self.occupancy = occupancy
        self.max_gap = max_gap
        self.window_s = window_s
        self._samples: deque[tuple[float, bool]] = deque()
        self._first_mono: float | None = None
        self._gap_run: int = 0

    def observe(self, now_mono: float, qualified: bool) -> None:
        """Record one frame's qualification. A gap longer than `max_gap`
        consecutive unqualified frames drops the streak."""
        if qualified:
            self._gap_run = 0
        else:
            self._gap_run += 1
            if self._gap_run > self.max_gap:
                self.reset()
                return
        if self._first_mono is None:
            self._first_mono = now_mono
        self._samples.append((now_mono, qualified))
        cutoff = now_mono - self.window_s
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def confirmed(self, now_mono: float) -> bool:
        """True when the window has elapsed and the qualified fraction over
        the trailing `window_s` meets `occupancy`."""
        if self._first_mono is None:
            return False
        if now_mono - self._first_mono < self.window_s:
            return False
        cutoff = now_mono - self.window_s
        qual = 0
        total = 0
        for t, q in self._samples:
            if t < cutoff:
                continue
            total += 1
            if q:
                qual += 1
        if total == 0:
            return False
        return qual >= self.occupancy * total

    def reset(self) -> None:
        self._samples.clear()
        self._first_mono = None
        self._gap_run = 0

    @property
    def tracking(self) -> bool:
        """False once the streak has been dropped (gap exceeded or reset).
        Used to prune per-target confirm state that has lapsed."""
        return self._first_mono is not None
