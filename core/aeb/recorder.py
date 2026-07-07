"""AEB clip recorder: two ring buffers + trigger logic + retroactive freeze.

Owns two rolling ring buffers (one per producer thread) and the capture state
machine. The radar thread pushes ``RadarFrameRecord``s, the AEB thread pushes
``AEBTickRecord``s plus its current discrete warm state, both at their own
cadence. A trigger freezes the same ``[trigger - pre_s, trigger + post_s]``
window across both streams; once the post-roll has elapsed the window is copied,
assembled into a ``Clip`` and handed to the ``AsyncClipWriter`` (plan 3, 4, 5).

Nothing here blocks: pushes are O(1); the only heavier step (copying ~330 small
records at freeze) runs under a short lock, and serialization happens on the
writer's background thread. Capture is inert until ``enabled`` is set by the
caller (gated on ``Settings.debug`` at wiring time); this module imports no
settings so it stays unit-testable.

Freeze and cooldown run on the *stream* clock (record ``t_mono``), not wall
time, so replay windows line up with recorded timestamps and tests are
deterministic. In production ``t_mono`` is ``time.monotonic``, so a trigger
default of "now" and the recorded stream share one clock.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field

from core.aeb.clip_schema import (
    AEBTickRecord,
    AEBWarmState,
    Clip,
    ClipMetadata,
    RadarFrameRecord,
)
from core.aeb.clip_store import AsyncClipWriter

logger = logging.getLogger(__name__)

# Highest-priority trigger wins ``trigger_source``; the rest go to
# ``also_triggered`` (plan 3.1).
TRIGGER_PRIORITY: dict[str, int] = {
    "manual": 4,
    "auto_crash": 3,
    "auto_engagement": 2,
    "shadow_near": 1,
    "random": 0,
}


def calibration_fingerprint(cal: object) -> tuple[str, dict]:
    """(stable hash, full field dict) for an ``AEBCalibration`` dataclass.

    The hash keys the server index; the full dict is stored too because a hash
    alone goes stale when fields are added (plan 4.3).
    """
    if cal is None or not dataclasses.is_dataclass(cal):
        return "", {}
    fields = dataclasses.asdict(cal)
    blob = json.dumps(fields, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest(), fields


class _TimestampRing:
    """Deque of ``(t_mono, payload)`` evicted by age. Push and evict are O(1)."""

    def __init__(self, span_s: float) -> None:
        self._span = span_s
        self._items: deque[tuple[float, object]] = deque()

    def push(self, t_mono: float, payload: object) -> None:
        self._items.append((t_mono, payload))
        newest = t_mono
        items = self._items
        while items and newest - items[0][0] > self._span:
            items.popleft()

    def window(self, t_lo: float, t_hi: float) -> list[object]:
        return [p for (t, p) in self._items if t_lo <= t <= t_hi]

    def clear(self) -> None:
        self._items.clear()


@dataclass
class _Pending:
    """A capture in flight: fired, waiting for its post-roll to complete."""

    trigger_t: float
    freeze_at: float
    source: str
    also_triggered: list[str] = field(default_factory=list)
    brake_reached: bool = False
    session_kind: str = "SP"
    calibration_id: str = ""
    calibration: dict = field(default_factory=dict)


@dataclass
class _Bundle:
    """Frozen window handed off from under the lock for off-lock assembly."""

    pending: _Pending
    radar_frames: list[RadarFrameRecord]
    aeb_ticks: list[AEBTickRecord]
    warm_state: AEBWarmState


class AEBClipRecorder:
    """Ring buffers + trigger state machine feeding an ``AsyncClipWriter``."""

    def __init__(
        self,
        writer: AsyncClipWriter,
        *,
        pre_s: float = 8.0,
        post_s: float = 3.0,
        burn_in_s: float = 2.0,
        cooldown_s: float = 5.0,
        ring_margin_s: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self._writer = writer
        self.pre_s = pre_s
        self.post_s = post_s
        self.burn_in_s = burn_in_s
        self.cooldown_s = cooldown_s
        self.enabled = enabled

        span = pre_s + post_s + ring_margin_s
        self._radar_ring = _TimestampRing(span)
        self._aeb_ring = _TimestampRing(span)

        self._lock = threading.Lock()
        self._pending: _Pending | None = None
        self._cooldown_until: float = float("-inf")
        self._last_t: float = 0.0

    def push_radar_frame(self, frame: RadarFrameRecord) -> None:
        """Called from the radar loop. O(1); may finalize a pending capture."""
        if not self.enabled:
            return
        bundle = None
        with self._lock:
            self._radar_ring.push(frame.t_mono, frame)
            self._last_t = max(self._last_t, frame.t_mono)
            bundle = self._take_bundle_locked(frame.t_mono)
        if bundle is not None:
            self._emit(bundle)

    def push_aeb_tick(self, tick: AEBTickRecord, warm_state: AEBWarmState | None = None) -> None:
        """Called from the AEB loop. Stores the tick with the discrete warm state
        current at that tick; the earliest in-window warm state becomes the
        clip's window-start snapshot (plan 5). O(1)."""
        if not self.enabled:
            return
        ws = warm_state if warm_state is not None else AEBWarmState()
        bundle = None
        with self._lock:
            self._aeb_ring.push(tick.t_mono, (tick, ws))
            self._last_t = max(self._last_t, tick.t_mono)
            bundle = self._take_bundle_locked(tick.t_mono)
        if bundle is not None:
            self._emit(bundle)

    def trigger(
        self,
        source: str,
        *,
        session_kind: str = "SP",
        brake_reached: bool = False,
        calibration: object | None = None,
        at: float | None = None,
    ) -> str:
        """Request a capture. Returns 'started' | 'folded' | 'ignored' | 'disabled'.

        A trigger while a capture is already in flight folds into it: the window
        is unchanged, ``trigger_source`` upgrades to the higher priority and the
        rest accumulate in ``also_triggered``. Auto / shadow / random triggers
        respect the post-capture cooldown; ``manual`` bypasses it.
        """
        if not self.enabled:
            return "disabled"
        if source not in TRIGGER_PRIORITY:
            logger.warning("unknown AEB trigger source %r", source)
            return "ignored"

        with self._lock:
            t = at if at is not None else self._last_t
            if self._pending is not None:
                self._fold_locked(source, brake_reached)
                return "folded"
            if source != "manual" and t < self._cooldown_until:
                return "ignored"
            cid, cfields = calibration_fingerprint(calibration)
            self._pending = _Pending(
                trigger_t=t,
                freeze_at=t + self.post_s,
                source=source,
                brake_reached=brake_reached,
                session_kind=session_kind,
                calibration_id=cid,
                calibration=cfields,
            )
            return "started"

    def _fold_locked(self, source: str, brake_reached: bool) -> None:
        p = self._pending
        assert p is not None
        p.brake_reached = p.brake_reached or brake_reached
        if source == p.source or source in p.also_triggered:
            return
        if TRIGGER_PRIORITY[source] > TRIGGER_PRIORITY[p.source]:
            p.also_triggered.append(p.source)
            p.source = source
        else:
            p.also_triggered.append(source)

    def _take_bundle_locked(self, newest_t: float) -> _Bundle | None:
        p = self._pending
        if p is None or newest_t < p.freeze_at:
            return None
        t_lo = p.trigger_t - self.pre_s
        t_hi = p.trigger_t + self.post_s
        radar_frames = [f for f in self._radar_ring.window(t_lo, t_hi)]
        aeb_entries = self._aeb_ring.window(t_lo, t_hi)
        aeb_ticks = [tick for (tick, _ws) in aeb_entries]
        warm_state = aeb_entries[0][1] if aeb_entries else AEBWarmState()

        self._pending = None
        self._cooldown_until = newest_t + self.cooldown_s
        return _Bundle(p, radar_frames, aeb_ticks, warm_state)

    def _emit(self, bundle: _Bundle) -> None:
        p = bundle.pending
        meta = ClipMetadata.create(
            trigger_source=p.source,
            also_triggered=list(p.also_triggered),
            brake_reached=p.brake_reached,
            session_kind=p.session_kind,
            calibration_id=p.calibration_id,
            calibration=p.calibration,
            pre_s=self.pre_s,
            post_s=self.post_s,
            burn_in_s=self.burn_in_s,
            frame_count=len(bundle.radar_frames),
            tick_count=len(bundle.aeb_ticks),
            aeb_warm_state=bundle.warm_state,
        )
        clip = Clip(metadata=meta, radar_frames=bundle.radar_frames, aeb_ticks=bundle.aeb_ticks)
        self._writer.submit(clip)
        logger.debug(
            "AEB clip queued: source=%s frames=%d ticks=%d also=%s",
            p.source, len(bundle.radar_frames), len(bundle.aeb_ticks), p.also_triggered,
        )

    def reset(self) -> None:
        """Drop buffered data and any pending capture (e.g. on teardown)."""
        with self._lock:
            self._radar_ring.clear()
            self._aeb_ring.clear()
            self._pending = None
            self._last_t = 0.0
            self._cooldown_until = float("-inf")
