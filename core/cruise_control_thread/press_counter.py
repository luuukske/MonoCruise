"""Turns the pedal thread's press counters into short-press events.

The cruise thread samples the pedal thread's published button level on its own
clock. At a low polling rate a tap can begin and end between two of those
samples, so the level alone loses presses. main_pedal_thread counts every rising
edge; this consumes that count instead. See core/cruise_control_thread/README.md.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_AUDIT_INTERVAL_S = 5.0


class PressCounter:
    """Consumes rising-edge counts so no press is lost to sampling."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._consumed: dict[str, int] = {}
        self._observed: dict[str, int] = {}
        self._prev_held: dict[str, bool] = {}
        self._last_audit_mono: float = 0.0

    def sync(self, counts: dict | None) -> None:
        """Adopt this tick's counters from the pedal thread."""
        counts = dict(counts or {})
        for name, count in counts.items():
            seen = self._consumed.get(name)
            # First sight ignores presses predating this thread; a lower count
            # means the pedal thread restarted and its counter reset.
            if seen is None or count < seen:
                self._consumed[name] = count
        self._counts = counts

    def discard(self, names: tuple[str, ...] | None = None) -> None:
        """Drop counted presses without acting: input was gated when they arrived."""
        for name, count in self._counts.items():
            if names is None or name in names:
                self._consumed[name] = count

    def take_short(self, name: str, held: bool) -> int:
        """Completed presses still owing a short action, excluding one in progress."""
        total = self._counts.get(name, 0)
        consumed = self._consumed.get(name, total)
        pending = total - consumed
        if held:
            pending -= 1
        if pending <= 0:
            return 0
        self._consumed[name] = consumed + pending
        return pending

    def consume_one(self, name: str) -> None:
        """Mark the in-progress press as already serviced by a long press."""
        self._consumed[name] = self._consumed.get(name, 0) + 1

    def audit(self, now: float, held_now: dict, pedal_hz: float, cruise_hz: float) -> None:
        """Debug: compare presses the pedal thread counted with those seen here."""
        for name, held in held_now.items():
            if held and not self._prev_held.get(name, False):
                self._observed[name] = self._observed.get(name, 0) + 1
            self._prev_held[name] = held

        if not self._counts or now - self._last_audit_mono < _AUDIT_INTERVAL_S:
            return
        if not any(self._counts.values()):
            return
        self._last_audit_mono = now
        unseen = {
            name: count - self._observed.get(name, 0)
            for name, count in self._counts.items()
            if count != self._observed.get(name, 0)
        }
        logger.debug(
            "button press audit: counted=%s seen_as_level=%s missed_by_level=%s "
            "(pedal %.0f Hz, cruise %.0f Hz)",
            self._counts, self._observed, unseen or "none", pedal_hz, cruise_hz,
        )
