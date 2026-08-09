"""Ring buffers + trigger freeze for debug AEB clips (``t_mono`` clock)."""

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
    Label,
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

# Auto-tagged TN corpus sources: heavy throttle, no thumbnail (see recorder trigger docs).
_TN_SOURCES: frozenset[str] = frozenset({"shadow_near", "random"})


def _auto_label_for(source: str) -> Label | None:
    """Auto-tag TN sources; human may override in review tool."""
    if source in _TN_SOURCES:
        return Label(
            class_="tn",
            should_trigger=None,
            notes=f"auto-tagged true negative ({source})",
        )
    return None


def calibration_fingerprint(cal: object) -> tuple[str, dict]:
    """Stable hash + full cal dict for clip metadata fingerprint."""
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
    thumbnail: str | None = None
    thumb_thread: object = None


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
        tn_cooldown_s: float = 600.0,
        ring_margin_s: float = 1.0,
        enabled: bool = True,
        capture_tn: bool = True,
        screenshot_provider=None,
    ) -> None:
        self._writer = writer
        self.pre_s = pre_s
        self.post_s = post_s
        self.burn_in_s = burn_in_s
        self.cooldown_s = cooldown_s
        # TN captures throttled separately so they never crowd real events.
        self.tn_cooldown_s = tn_cooldown_s
        self.enabled = enabled
        # Background negatives are cheap to generate locally and the least
        # informative from a stranger's machine, so contributors skip them.
        self.capture_tn = capture_tn
        # Optional () -> base64 JPEG | None, grabbed off-thread at trigger time.
        self._screenshot_provider = screenshot_provider

        span = pre_s + post_s + ring_margin_s
        self._radar_ring = _TimestampRing(span)
        self._aeb_ring = _TimestampRing(span)

        self._lock = threading.Lock()
        self._pending: _Pending | None = None
        self._cooldown_until: float = float("-inf")
        self._tn_cooldown_until: float = float("-inf")
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
        """Capture trigger: started/folded/ignored/disabled; manual bypasses cooldown."""
        if not self.enabled:
            return "disabled"
        if source not in TRIGGER_PRIORITY:
            logger.warning("unknown AEB trigger source %r", source)
            return "ignored"
        # Refused before the fold too, so a background negative never even
        # reaches also_triggered on a contributor's clip.
        if source in _TN_SOURCES and not self.capture_tn:
            return "ignored"

        with self._lock:
            t = at if at is not None else self._last_t
            if self._pending is not None:
                self._fold_locked(source, brake_reached)
                return "folded"
            if source != "manual" and t < self._cooldown_until:
                return "ignored"
            if source in _TN_SOURCES and t < self._tn_cooldown_until:
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
            # No thumbnail for background negatives (see _TN_SOURCES).
            if source not in _TN_SOURCES:
                self._start_thumbnail_grab(self._pending)
            return "started"

    def _start_thumbnail_grab(self, pending: _Pending) -> None:
        """Grab the screen thumbnail off-thread so the 30 Hz loop never blocks."""
        provider = self._screenshot_provider
        if provider is None:
            return

        def _run() -> None:
            try:
                pending.thumbnail = provider()
            except Exception:
                logger.debug("AEB thumbnail provider failed", exc_info=True)

        th = threading.Thread(target=_run, name="aeb_thumbnail", daemon=True)
        pending.thumb_thread = th
        th.start()

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
        # Throttle the next background negative off the winning source, so a
        # capture that folded up into a real event does not spend the TN budget.
        if p.source in _TN_SOURCES:
            self._tn_cooldown_until = newest_t + self.tn_cooldown_s
        return _Bundle(p, radar_frames, aeb_ticks, warm_state)

    def _emit(self, bundle: _Bundle) -> None:
        p = bundle.pending
        if p.thumb_thread is not None:
            try:
                p.thumb_thread.join(timeout=1.0)
            except Exception:
                pass
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
            thumbnail_jpeg=p.thumbnail,
            label=_auto_label_for(p.source),
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
            self._tn_cooldown_until = float("-inf")
