"""Custom widgets and the background decoder behind tools/aeb_review.py. Not shipped."""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from core.aeb.clip_replay import ReviewFrame, replay_clip
from core.aeb.clip_schema import Clip, ClipMetadata
from core.aeb.clip_store import ClipInfo, ClipStore
from core.aeb.debug_window import AEBDebugWindow

TTC_INF = 100.0         # recorded sentinel is 1e9; past this there is no tracked threat
_DECEL_AXIS_MIN = 8.0   # m/s2 floor; the axis grows to the clip's own brake capacity
_TTC_AXIS_MAX = 8.0     # s

_THUMB_MIN_W = 200      # right panel is 320px wide; clamp floor
# Ceiling matters as much as the floor: an unshown/not-yet-laid-out label
# reports width() as 640 in this Qt version, not a small stale value.
_THUMB_MAX_W = 300      # clamp ceiling, also the panel's real content width


@dataclass
class Loaded:
    """One decoded clip plus everything derived from it once, at decode time."""

    clip: Clip
    frames: list[ReviewFrame]
    proposal: tuple[float, float] | None
    action_idx: int


def recorded_band(frames: list[ReviewFrame]) -> tuple[float, float] | None:
    """Span the live AEB reacted over: warn/brake if any, else tracked-threat ticks."""
    hits = [f.t_rel for f in frames if f.live_aeb.aeb_warn or f.live_aeb.aeb_brake]
    if not hits:
        hits = [f.t_rel for f in frames if f.live_aeb.time_to_collision < TTC_INF]
    return (hits[0], hits[-1]) if hits else None


def action_index(frames: list[ReviewFrame]) -> int:
    """Frame to land on: first warn/brake, else closest approach, else the start."""
    for i, f in enumerate(frames):
        if f.live_aeb.aeb_warn or f.live_aeb.aeb_brake:
            return i
    best_i, best_ttc = 0, TTC_INF
    for i, f in enumerate(frames):
        if f.live_aeb.time_to_collision < best_ttc:
            best_i, best_ttc = i, f.live_aeb.time_to_collision
    return best_i


class ThumbnailView(QLabel):
    """Clip screenshot: holds the decoded original and rescales into the panel width.

    Every rescale, including its own resize, starts from the cached original;
    an already-scaled pixmap is never re-scaled, which would destroy detail.
    """

    def __init__(self) -> None:
        super().__init__("(no screenshot)")
        self.setMinimumHeight(100)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111; color:#666; border:1px solid #333;")
        self._orig: QPixmap | None = None

    def set_jpeg(self, b64: str | None) -> None:
        self._orig = None
        self.setPixmap(QPixmap())
        if not b64:
            self.setText("(no screenshot)")
            return
        pix = QPixmap()
        if not pix.loadFromData(base64.b64decode(b64), "JPEG"):
            self.setText("(thumbnail decode failed)")
            return
        self.setText("")
        self._orig = pix
        self._rescale()

    def source_size(self) -> tuple[int, int] | None:
        return (self._orig.width(), self._orig.height()) if self._orig is not None else None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._orig is None or self._orig.isNull():
            return
        w = min(max(self.width(), _THUMB_MIN_W), _THUMB_MAX_W)
        self.setPixmap(self._orig.scaledToWidth(w, Qt.SmoothTransformation))


class ClipLoader(QObject):
    """Store reads off the GUI thread: load plus replay costs ~0.5 s per clip."""

    loaded = Signal(str, object, object)   # path, Clip | None, list[ReviewFrame]
    scanned = Signal(object)               # list[tuple[ClipInfo, ClipMetadata | None]]

    def __init__(self, store: ClipStore) -> None:
        super().__init__()
        self._store = store

    @Slot(str)
    def load(self, path: str) -> None:
        clip = self._store.load(path)
        frames = replay_clip(clip) if clip is not None else []
        self.loaded.emit(path, clip, frames)

    @Slot(object)
    def scan(self, known: dict) -> None:
        """Rescan the store, decoding metadata only for new or changed clips.

        ``known`` maps path to (mtime, size, metadata) from the caller's cache, so a
        refresh only pays for clips that actually appeared or were relabelled.
        """
        entries: list[tuple[ClipInfo, ClipMetadata | None]] = []
        for info in self._store.list_clips():
            cached = known.get(str(info.path))
            if cached is not None and cached[0] == info.mtime and cached[1] == info.size_bytes:
                meta = cached[2]
            else:
                meta = self._store.peek_metadata(info.path)
            entries.append((info, meta))
        self.scanned.emit(entries)


class SceneWidget(AEBDebugWindow):
    """Debug renderer fed replayed snapshots; click picks the nearest vehicle."""

    vehicle_picked = Signal(int)

    def __init__(self) -> None:
        super().__init__(snapshot_provider=lambda: self._provide(),
                         acc_provider=lambda: None, auto_refresh=False)
        self._snap = None
        self.pick_mode = False
        self.show_vehicle_paths = True

    def _provide(self):
        """Snapshot as drawn: vehicle corridors stripped when the toggle is off."""
        if self._snap is None or self.show_vehicle_paths:
            return self._snap
        return replace(self._snap, vehicle_arcs={})

    def set_snapshot(self, snap) -> None:
        self._snap = snap
        self.update()

    def set_vehicle_paths(self, on: bool) -> None:
        self.show_vehicle_paths = bool(on)
        self.update()

    def mousePressEvent(self, event) -> None:
        if not self.pick_mode or self._snap is None:
            return
        px, py = event.position().x(), event.position().y()
        best_vid, best_d2 = None, 24.0 ** 2
        for v in self._snap.vehicles:
            sx, sy = self._ws(v["x"], v["z"], self._snap.ego_x, self._snap.ego_z,
                              self._snap.ego_yaw)
            d2 = (sx - px) ** 2 + (sy - py) ** 2
            if d2 < best_d2:
                best_d2, best_vid = d2, v["vid"]
        if best_vid is not None:
            self.vehicle_picked.emit(int(best_vid))


class DecisionStrip(QWidget):
    """Three lanes over clip time: recorded decision, decel demand, time-to-collision.

    Everything except the cursor is cached into a pixmap, so scrubbing and playback
    repaint one line instead of a few thousand.
    """

    seeked = Signal(float)   # t_rel in seconds

    _LANE_DECISION = 22
    _LANE_DECEL = 62
    _LANE_TTC = 38
    _GAP = 5
    _PAD = 6

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(
            self._LANE_DECISION + self._LANE_DECEL + self._LANE_TTC + 3 * self._GAP + 6
        )
        self.setMouseTracking(True)
        self._frames: list[ReviewFrame] = []
        self._duration = 1.0
        self._cursor = 0.0
        self._window: tuple[float, float] | None = None
        self._proposal: tuple[float, float] | None = None
        self._decel_axis = _DECEL_AXIS_MIN
        self._font = QFont("Segoe UI", 7)
        self._bg: QPixmap | None = None

    def set_frames(self, frames: list[ReviewFrame], duration: float) -> None:
        self._frames = frames
        self._duration = max(duration, 1e-3)
        # Required decel is excluded from the axis on purpose: it runs to five figures
        # as the gap closes, and would flatten every other trace against the floor.
        peak = max(
            (max(f.raw_target_ms2, f.live_aeb.effective_max_decel_ms2) for f in frames),
            default=_DECEL_AXIS_MIN,
        )
        self._decel_axis = max(_DECEL_AXIS_MIN, peak)
        self._invalidate()

    def set_cursor(self, t_rel: float) -> None:
        self._cursor = t_rel
        self.update()

    def set_window(self, window: tuple[float, float] | None) -> None:
        self._window = window
        self._invalidate()

    def set_proposal(self, proposal: tuple[float, float] | None) -> None:
        self._proposal = proposal
        self._invalidate()

    def _invalidate(self) -> None:
        self._bg = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._bg = None
        super().resizeEvent(event)

    def _x(self, t: float) -> float:
        return self._PAD + (t / self._duration) * (self.width() - 2 * self._PAD)

    def mousePressEvent(self, event) -> None:
        span = max(self.width() - 2 * self._PAD, 1)
        frac = (event.position().x() - self._PAD) / span
        self.seeked.emit(max(0.0, min(1.0, frac)) * self._duration)

    def paintEvent(self, event) -> None:
        if self._bg is None:
            self._bg = self._render_lanes()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._bg)
        cx = self._x(self._cursor)
        p.setPen(QPen(QColor(255, 255, 255), 1.6))
        p.drawLine(int(cx), 0, int(cx), self.height())
        p.end()

    def _render_lanes(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        pm = QPixmap(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(QColor(24, 24, 30))

        p = QPainter(pm)
        p.setFont(self._font)
        # Full-height band first: the window has to be readable against the traces,
        # not just against the decision lane it was set in.
        if self._window is not None:
            xa, xb = self._x(self._window[0]), self._x(self._window[1])
            p.fillRect(int(xa), 0, max(1, int(xb - xa)), self.height(),
                       QColor(80, 210, 130, 26))
        y = 3
        self._draw_decision(p, y, self._LANE_DECISION)
        y += self._LANE_DECISION + self._GAP
        self._draw_decel(p, y, self._LANE_DECEL)
        y += self._LANE_DECEL + self._GAP
        self._draw_ttc(p, y, self._LANE_TTC)
        p.end()
        return pm

    def _draw_decision(self, p: QPainter, y0: int, h: int) -> None:
        if self._proposal is not None and self._window != self._proposal:
            a, b = self._proposal
            xa, xb = self._x(a), self._x(b)
            p.setPen(QPen(QColor(80, 210, 130, 170), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(xa), y0, max(1, int(xb - xa)), h)
            p.drawText(int(xa) + 4, y0 + h - 6, "proposed (W)")

        if self._window is not None:
            a, b = self._window
            xa, xb = self._x(a), self._x(b)
            p.fillRect(int(xa), y0, max(1, int(xb - xa)), h, QColor(80, 210, 130, 70))

        for f in self._frames:
            x = self._x(f.t_rel)
            if f.live_aeb.aeb_brake:
                c = QColor(240, 55, 55)
            elif f.live_aeb.aeb_warn:
                c = QColor(245, 185, 40)
            else:
                c = QColor(60, 70, 80)
            p.setPen(QPen(c, 1.4))
            p.drawLine(int(x), y0, int(x), y0 + h)

    def _draw_decel(self, p: QPainter, y0: int, h: int) -> None:
        """Pre-slew demand and raw required decel against the clip's brake capacity."""
        self._draw_baseline(p, y0 + h)
        axis = self._decel_axis

        def yv(v: float) -> float:
            return y0 + h - min(max(v, 0.0), axis) / axis * h

        for pen, pick in (
            (QPen(QColor(120, 120, 140, 120), 1.0, Qt.DashLine),
             lambda f: f.live_aeb.effective_max_decel_ms2),
            (QPen(QColor(235, 150, 60, 190), 1.2),
             lambda f: f.live_aeb.required_decel_ms2),
            (QPen(QColor(240, 70, 70), 1.9), lambda f: f.raw_target_ms2),
        ):
            run = [(self._x(f.t_rel), yv(pick(f))) for f in self._frames]
            self._draw_runs(p, pen, [run])

        self._draw_legend(p, y0, (
            (f"decel  0..{axis:.0f} m/s2", QColor(120, 120, 135)),
            ("demand (desmoothed)", QColor(240, 70, 70)),
            ("required", QColor(235, 150, 60, 190)),
            ("capacity", QColor(120, 120, 140)),
        ))

    def _draw_ttc(self, p: QPainter, y0: int, h: int) -> None:
        self._draw_baseline(p, y0 + h)
        runs: list[list[tuple[float, float]]] = [[]]
        for f in self._frames:
            ttc = f.live_aeb.time_to_collision
            if ttc >= TTC_INF:
                runs.append([])
                continue
            v = min(ttc, _TTC_AXIS_MAX)
            runs[-1].append((self._x(f.t_rel), y0 + h - v / _TTC_AXIS_MAX * h))
        self._draw_runs(p, QPen(QColor(90, 200, 235), 1.6), runs)
        self._draw_legend(p, y0, ((f"ttc  {_TTC_AXIS_MAX:.0f}..0 s",
                                   QColor(120, 120, 135)),))

    def _draw_baseline(self, p: QPainter, y: int) -> None:
        p.setPen(QPen(QColor(45, 45, 55), 1))
        p.drawLine(self._PAD, y, self.width() - self._PAD, y)

    def _draw_legend(self, p: QPainter, y0: int, items) -> None:
        """Lane key on a backing strip, so it stays readable over any trace."""
        fm = p.fontMetrics()
        gap = 10
        widths = [fm.horizontalAdvance(t) for t, _c in items]
        p.fillRect(self._PAD, y0, sum(widths) + gap * len(items) + 4,
                   fm.height() + 2, QColor(24, 24, 30, 225))
        x = self._PAD + 3
        for (text, clr), w in zip(items, widths):
            p.setPen(QPen(clr))
            p.drawText(x, y0 + fm.ascent(), text)
            x += w + gap

    @staticmethod
    def _draw_runs(p: QPainter, pen: QPen, runs: list[list[tuple[float, float]]]) -> None:
        """Polylines with breaks between runs, so gaps in a signal stay gaps."""
        p.setPen(pen)
        for run in runs:
            for (x0, y0), (x1, y1) in zip(run, run[1:]):
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
