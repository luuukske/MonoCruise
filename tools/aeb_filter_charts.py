"""Filter-tuning chart window behind tools/aeb_review.py. Dev only, never shipped.

Lanes, thresholds and what each signal is for: tools/README.md. The numbers come
from tools/aeb_filter_trace.py, which reads them off the replayed vehicles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget,
)

from core.radar import traffic as T
from tools.aeb_filter_trace import ClipTrace, VehicleTrace

_BG = QColor(22, 22, 28)
_GRID = QColor(44, 44, 54)
_TEXT = QColor(170, 170, 180)
_CURSOR = QColor(255, 255, 255)
_HOVER = QColor(120, 200, 255, 150)

_PAD_L = 78          # room for the axis labels and the state-row names
_RANGE_PCTL = 0.99   # auto axis percentile, see FilterChart._range
_LOG_SPAN = 3.0      # lag lane clip: 1/8x to 8x the threshold
_MAX_GAP_S = 0.30    # break a trace only past this hole, see FilterChart._draw_lane
_HOLD_EPS = 1e-9     # a repeat this close is a carried value, see _thin
_HOLD_MIN_S = 0.12   # a repeat run longer than this is a real hold, see _thin
_SETTLE_S = 0.9      # ignored by the auto range, see FilterChart._range
_PAD_R = 8
_LANE_GAP = 4
_READOUT_W = 236

# Lag gates are drawn as a fraction of their own threshold, so one line at 1.0
# reads as pass/fail for all of them at once.
_LAG_THRESHOLDS = {
    "lag_disp_ratio": T._LAG_DISP_RATIO,
    "lag_rot_rate": T._LAG_ROT_LIVE_DEG_S,
    "lag_raw_recent": T._LAG_ENTRY_RAW_SPEED_MS,
    "lag_raw_decay": T._LAG_ENTRY_DECAY_MIN,
}


@dataclass
class Curve:
    """One trace drawn in a lane."""

    key: str
    label: str
    color: QColor
    width: float = 1.4
    dashed: bool = False
    scale: float = 1.0


@dataclass
class Lane:
    """A stacked band with its own vertical axis and its own set of signals."""

    title: str
    signals: list[Curve]
    height: int = 120
    mode: str = "auto_sym"          # auto_sym | auto_pos | fixed
    lo: float = 0.0
    hi: float = 1.0
    unit: str = ""
    rules: list[tuple[float, str]] = field(default_factory=list)
    floor_span: float = 1.0
    span_cap: float = 1e9
    short: str = ""


def _c(r: int, g: int, b: int, a: int = 255) -> QColor:
    return QColor(r, g, b, a)


def _axis_label(value: float) -> str:
    if abs(value) >= 10.0:
        return f"{value:.0f}"
    return f"{value:.2f}" if abs(value) < 1.0 else f"{value:.1f}"


def _thin(times: list[float], values: list[float]):
    """Collapse carried repeats. Yields ``(t, value, connected)``.

    Half the radar frames are sub-frames, which copy the filter output forward
    untouched, so drawing every sample produces a staircase whose treads are an
    artefact of the radar rate rather than of the vehicle. A run of equal values
    contributes its first sample, and its last as well when the run outlasted
    ``_HOLD_MIN_S``: short runs become a line straight from update to update,
    while a genuine hold still reads flat and changes out of it stay steep.
    ``connected`` is False where the source had a real hole.
    """
    pts = [(t, v) for t, v in zip(times, values) if v == v and abs(v) < 1e6]
    out: list[tuple[float, float, bool]] = []
    i, n = 0, len(pts)
    while i < n:
        j = i
        while j + 1 < n and abs(pts[j + 1][1] - pts[i][1]) <= _HOLD_EPS:
            j += 1
        joined = i == 0 or pts[i][0] - pts[i - 1][0] <= _MAX_GAP_S
        out.append((pts[i][0], pts[i][1], joined))
        if j > i and pts[j][0] - pts[i][0] > _HOLD_MIN_S:
            out.append((pts[j][0], pts[j][1], True))
        i = j + 1
    return out


def _log_ratio(ratio: float) -> float:
    """log2 of a gate-to-threshold ratio, clipped. 0 is the threshold itself."""
    if ratio != ratio:
        return float("nan")
    if ratio <= 0.0:
        return -_LOG_SPAN
    return max(-_LOG_SPAN, min(_LOG_SPAN, math.log2(ratio)))


def _speed_lane() -> Lane:
    return Lane(
        title="speed chain",
        short="speed chain",
        unit="m/s",
        mode="auto_pos",
        height=150,
        floor_span=5.0,
        signals=[
            Curve("raw_long", "raw long-window (ACC in)", _c(110, 110, 125), 1.0, dashed=True),
            Curve("raw_sel", "raw selected (AEB in)", _c(200, 120, 220), 1.1),
            Curve("speed_ema", "1 speed_ema", _c(90, 170, 235), 1.2),
            Curve("speed", "3 speed_corr = AEB speed", _c(80, 225, 140), 1.9),
            Curve("acc_corr", "3 acc_corr (ACC step-4 in)", _c(235, 175, 70), 1.2, dashed=True),
            Curve("acc_speed", "4 acc_speed = ACC speed", _c(240, 95, 80), 1.9),
            Curve("ego_speed", "ego", _c(120, 120, 140), 1.0, dashed=True),
        ],
    )


def _accel_lane() -> Lane:
    return Lane(
        title="accel chain",
        short="accel chain",
        unit="m/s2",
        mode="auto_sym",
        height=130,
        floor_span=2.0,
        span_cap=12.0,
        rules=[
            (T._ACC_SPEED_FF_GATE_LO_MS2, "ff gate lo"),
            (T._ACC_SPEED_ACCEL_LO_MS2, "trend lo"),
            (T._ACC_SPEED_ACCEL_HI_MS2, "trend hi"),
        ],
        signals=[
            Curve("accel", "2 acceleration (AEB)", _c(80, 225, 140), 1.7),
            Curve("acc_accel", "2 acc_accel (ACC)", _c(240, 95, 80), 1.7),
            Curve("accel_trend", f"trend {T._ACC_SPEED_ACCEL_WINDOW_S:g}s", _c(235, 175, 70), 1.2),
            Curve("accel_long", f"long {T._ACC_SPEED_CONSIST_WINDOW_S:g}s", _c(90, 170, 235), 1.2),
            Curve("brake_floor", "hard-brake floor", _c(200, 120, 220), 1.6, dashed=True),
        ],
    )


def _gate_lane() -> Lane:
    return Lane(
        title="step 4 gates",
        short="step 4 gates",
        unit="0..1, tau x0.5",
        mode="fixed",
        lo=0.0,
        hi=1.05,
        height=110,
        rules=[(1.0, ""), (T._ACC_SPEED_TAU_SLOW_S * 0.5, "tau slow")],
        signals=[
            Curve("ramp", "ramp (fast tau)", _c(240, 95, 80), 1.4),
            Curve("ff_gate", "ff_gate", _c(235, 175, 70), 1.4),
            Curve("consistency", "consistency", _c(90, 170, 235), 1.4),
            Curve("accel_factor", "accel_factor", _c(80, 225, 140), 1.2, dashed=True),
            Curve("speed_factor", "speed_factor", _c(110, 110, 125), 1.0, dashed=True),
            Curve("tau", "tau (s, x0.5)", _c(200, 120, 220), 1.7, scale=0.5),
        ],
    )


def _lag_lane() -> Lane:
    return Lane(
        title="lag entry gates, log2 of the gate over its own threshold",
        short="lag gates (log2)",
        unit="0 = at threshold",
        mode="log_ratio",
        lo=-_LOG_SPAN,
        hi=_LOG_SPAN,
        height=126,
        rules=[(0.0, "threshold")],
        signals=[
            Curve("lag_disp_ratio",
                  f"raw disp / expected  (< {T._LAG_DISP_RATIO:g} flags lag)",
                  _c(240, 95, 80), 1.6),
            Curve("lag_rot_rate",
                  f"rotation  (>= {T._LAG_ROT_LIVE_DEG_S:g} deg/s blocks)",
                  _c(90, 170, 235), 1.3),
            Curve("lag_raw_recent",
                  f"raw recent  (< {T._LAG_ENTRY_RAW_SPEED_MS:g} m/s blocks)",
                  _c(80, 225, 140), 1.3),
            Curve("lag_raw_decay",
                  f"raw decay  (< {T._LAG_ENTRY_DECAY_MIN:g} blocks)",
                  _c(235, 175, 70), 1.3),
            Curve("lag_frac", "freeze elapsed / duration", _c(200, 120, 220), 1.6),
        ],
    )


_STATE_ROWS = [
    ("st_frozen", "lag freeze", _c(240, 95, 80)),
    ("st_lag_confirmed", "lag confirmed", _c(240, 150, 60)),
    ("st_raw_brake", "short window", _c(200, 120, 220)),
    ("st_pos_mismatch", "pos mismatch", _c(90, 170, 235)),
    ("st_crash", "crash", _c(235, 90, 200)),
    ("st_standstill", "acc standstill", _c(80, 225, 140)),
    ("st_stale", "paused frame", _c(110, 110, 125)),
    ("st_subframe", "sub-frame", _c(70, 70, 85)),
    ("st_bypassed", "chain bypassed", _c(200, 200, 70)),
]

_LANES = (_speed_lane, _accel_lane, _gate_lane, _lag_lane)
_STATE_ROW_H = 11
_DECISION_ROW_H = 13
_DECISION_H = 2 * _DECISION_ROW_H + 6
_AXIS_H = 22
_MIN_CHART_H = sum(build().height + _LANE_GAP for build in _LANES) + (
    _DECISION_H + len(_STATE_ROWS) * _STATE_ROW_H + _AXIS_H)


class FilterChart(QWidget):
    """Stacked lanes over clip time. Everything but the two cursors is cached."""

    seeked = Signal(float)
    hovered = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(_MIN_CHART_H)
        self.setMouseTracking(True)
        self._trace: ClipTrace | None = None
        self._veh: VehicleTrace | None = None
        self._duration = 1.0
        self._cursor = 0.0
        self._hover: float | None = None
        self._window: tuple[float, float] | None = None
        self._decision: list[tuple[float, int]] = []
        self._replayed: list[tuple[float, int]] | None = None
        self._enabled = {lane().title: True for lane in _LANES}
        self._font = QFont("Consolas", 8)
        self._bg: QPixmap | None = None

    def set_trace(self, trace: ClipTrace | None, veh: VehicleTrace | None) -> None:
        self._trace = trace
        self._veh = veh
        self._duration = max(trace.duration if trace else 1.0, 1e-3)
        self._invalidate()

    def set_decision(self, decision: list[tuple[float, int]],
                     window: tuple[float, float] | None) -> None:
        """Recorded AEB state per tick (0 standby, 1 warn, 2 brake) and the truth window."""
        self._decision = decision
        self._window = window
        self._invalidate()

    def set_replayed(self, replayed: list[tuple[float, int]] | None) -> None:
        """The same decision re-run at the working tree's constants, or None while pending."""
        self._replayed = replayed
        self._invalidate()

    def set_lane_enabled(self, title: str, on: bool) -> None:
        self._enabled[title] = bool(on)
        self._invalidate()

    def set_cursor(self, t_rel: float) -> None:
        self._cursor = t_rel
        self.update()

    def _invalidate(self) -> None:
        self._bg = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._bg = None
        super().resizeEvent(event)

    def _plot_w(self) -> int:
        return max(self.width() - _PAD_L - _PAD_R, 1)

    def _x(self, t: float) -> float:
        return _PAD_L + (t / self._duration) * self._plot_w()

    def _t(self, x: float) -> float:
        return max(0.0, min(1.0, (x - _PAD_L) / self._plot_w())) * self._duration

    def mousePressEvent(self, event) -> None:
        self.seeked.emit(self._t(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            self.seeked.emit(self._t(event.position().x()))
        self._hover = self._t(event.position().x())
        self.hovered.emit(self._hover)
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.hovered.emit(-1.0)
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        if self._bg is None:
            self._bg = self._render()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._bg)
        if self._hover is not None:
            hx = int(self._x(self._hover))
            p.setPen(QPen(_HOVER, 1.0, Qt.DashLine))
            p.drawLine(hx, 0, hx, self.height())
        p.setRenderHint(QPainter.Antialiasing, True)
        cx = self._x(self._cursor)
        p.setPen(QPen(_CURSOR, 1.6))
        p.drawLine(QPointF(cx, 0.0), QPointF(cx, float(self.height())))
        p.end()

    def _lane_stack(self) -> list[Lane]:
        """Enabled lanes, grown proportionally so the stack fills the window."""
        lanes = [lane for build in _LANES if self._enabled.get((lane := build()).title, True)]
        if not lanes:
            return lanes
        fixed = _DECISION_H + len(_STATE_ROWS) * _STATE_ROW_H + _AXIS_H
        spare = self.height() - fixed - _LANE_GAP * (len(lanes) + 1)
        asked = sum(lane.height for lane in lanes)
        if spare > asked:
            grow = spare / asked
            for lane in lanes:
                lane.height = int(lane.height * grow)
        return lanes

    def _render(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        pm = QPixmap(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(_BG)
        p = QPainter(pm)
        p.setFont(self._font)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        y = self._draw_decision(p, 2)
        if self._veh is not None:
            for lane in self._lane_stack():
                self._draw_lane(p, lane, y)
                y += lane.height + _LANE_GAP
            self._draw_states(p, y)
        else:
            p.setPen(QPen(_TEXT))
            p.drawText(_PAD_L, y + 30, "no vehicle traced for this clip")
        self._draw_time_axis(p)
        p.end()
        return pm

    def _draw_decision(self, p: QPainter, y0: int) -> int:
        """Two bands: what the clip recorded, and what this working tree decides now.

        The curves below are always the current filter, so a recorded-only band could
        not be compared against them. The replayed row closes that: same clip, same
        constants as everything else on the page.
        """
        if self._window is not None:
            xa, xb = self._x(self._window[0]), self._x(self._window[1])
            p.fillRect(int(xa), 0, max(1, int(xb - xa)), self.height(), _c(80, 210, 130, 22))
        self._draw_decision_row(p, y0, "rec", self._decision)
        y1 = y0 + _DECISION_ROW_H + 2
        if self._replayed is None:
            p.setPen(QPen(_c(120, 120, 140)))
            p.drawText(4, y1 + _DECISION_ROW_H - 3, "now  recomputing...")
        else:
            self._draw_decision_row(p, y1, "now", self._replayed)
            self._draw_divergence(p, y1)
        return y0 + _DECISION_H

    def _draw_decision_row(self, p: QPainter, y0: int, label: str,
                           track: list[tuple[float, int]]) -> None:
        for t_rel, state in track:
            if state == 0:
                continue
            x = int(self._x(t_rel))
            p.setPen(QPen(_c(240, 55, 55) if state == 2 else _c(245, 185, 40), 1.4))
            p.drawLine(x, y0, x, y0 + _DECISION_ROW_H)
        p.setPen(QPen(_TEXT))
        p.drawText(4, y0 + _DECISION_ROW_H - 3, label)

    def _draw_divergence(self, p: QPainter, y0: int) -> None:
        """Tick every moment the two bands disagree: the reason for showing both."""
        recorded = dict(self._decision)
        for t_rel, state in self._replayed or ():
            was = recorded.get(t_rel)
            if was is None or was == state:
                continue
            x = int(self._x(t_rel))
            p.setPen(QPen(_c(120, 200, 255), 1.2))
            p.drawLine(x, y0 + _DECISION_ROW_H - 3, x, y0 + _DECISION_ROW_H)

    def _series(self, key: str) -> list[float] | None:
        if self._veh is None:
            return None
        if key == "ego_speed":
            return self._trace.ego_speed if self._trace else None
        if key == "lag_frac":
            elapsed = self._veh.series.get("lag_elapsed")
            dur = self._veh.series.get("lag_freeze_dur")
            if elapsed is None or dur is None:
                return None
            return [e / d if (e == e and d and d > 1e-9) else float("nan")
                    for e, d in zip(elapsed, dur)]
        return self._veh.series.get(key)

    def _times(self, key: str) -> list[float]:
        if key == "ego_speed" and self._trace is not None:
            return self._trace.ego_t
        return self._veh.t if self._veh else []

    def _values(self, lane: Lane, sig: Curve) -> tuple[list[float], list[float]] | None:
        col = self._series(sig.key)
        if not col:
            return None
        if lane.mode == "log_ratio":
            div = _LAG_THRESHOLDS.get(sig.key, 1.0)
            return self._times(sig.key), [_log_ratio(v / div) for v in col]
        return self._times(sig.key), [v * sig.scale for v in col]

    def _range(self, lane: Lane) -> tuple[float, float]:
        """Range on the settled part of the trace, by percentile.

        A track's first samples fit a slope through two or three points, which reaches
        tens of m/s2 and is an artefact of the fit, not of the vehicle. Ranging on the
        max, or on the whole trace, buries the working signal against the axis.
        """
        if lane.mode in ("fixed", "log_ratio"):
            return lane.lo, lane.hi
        settled = (self._veh.t[0] + _SETTLE_S) if (self._veh and self._veh.t) else 0.0
        magnitudes = []
        for sig in lane.signals:
            got = self._values(lane, sig)
            if got is None:
                continue
            magnitudes += [abs(v) for t_rel, v in zip(*got)
                           if v == v and abs(v) < 1e6 and t_rel >= settled]
        magnitudes.sort()
        peak = magnitudes[int(len(magnitudes) * _RANGE_PCTL)] if magnitudes else 0.0
        peak = min(max(peak * 1.12, lane.floor_span), lane.span_cap)
        return (-peak, peak) if lane.mode == "auto_sym" else (0.0, peak)

    def _draw_lane(self, p: QPainter, lane: Lane, y0: int) -> None:
        h = lane.height
        lo, hi = self._range(lane)
        span = max(hi - lo, 1e-9)

        def yv(v: float) -> float:
            return y0 + h - (min(max(v, lo), hi) - lo) / span * h

        p.setPen(QPen(_GRID, 1))
        p.drawRect(_PAD_L, y0, self._plot_w(), h)
        for value, nudge in ((hi, 9), ((lo + hi) * 0.5, 3), (lo, -2)):
            y = yv(value)
            p.setPen(QPen(_GRID, 1, Qt.DotLine))
            p.drawLine(_PAD_L, int(y), _PAD_L + self._plot_w(), int(y))
            p.setPen(QPen(_TEXT))
            p.drawText(2, int(y) + nudge, _axis_label(value))
        fm = p.fontMetrics()
        for i, (value, name) in enumerate(lane.rules):
            for signed in ({value, -value} if lane.mode == "auto_sym" else {value}):
                if not lo <= signed <= hi:
                    continue
                p.setPen(QPen(_c(90, 90, 110), 1, Qt.DashLine))
                p.drawLine(_PAD_L, int(yv(signed)), _PAD_L + self._plot_w(), int(yv(signed)))
            if name and lo <= value <= hi:
                p.setPen(QPen(_c(120, 120, 140)))
                x = _PAD_L + self._plot_w() - fm.horizontalAdvance(name) - 4
                p.drawText(x, int(yv(value)) - 2 - i * (fm.height() - 2), name)

        p.setRenderHint(QPainter.Antialiasing, True)
        for sig in lane.signals:
            got = self._values(lane, sig)
            if got is None:
                continue
            times, values = got
            p.setPen(QPen(sig.color, sig.width,
                          Qt.DashLine if sig.dashed else Qt.SolidLine))
            prev: QPointF | None = None
            for t_rel, value, joined in _thin(times, values):
                cur = QPointF(self._x(t_rel), yv(value))
                if prev is not None and joined:
                    p.drawLine(prev, cur)
                prev = cur

        self._draw_key(p, lane, y0)

    def _draw_key(self, p: QPainter, lane: Lane, y0: int) -> None:
        fm = p.fontMetrics()
        items = [(f"{lane.title}  [{lane.unit}]", _TEXT)]
        items += [(s.label, s.color) for s in lane.signals if self._series(s.key)]
        x = _PAD_L + 3
        width = sum(fm.horizontalAdvance(t) + 12 for t, _ in items)
        p.fillRect(_PAD_L + 1, y0 + 1, min(width, self._plot_w() - 2), fm.height() + 2,
                   _c(22, 22, 28, 220))
        for text, color in items:
            p.setPen(QPen(color))
            p.drawText(x, y0 + fm.ascent() + 1, text)
            x += fm.horizontalAdvance(text) + 12
        p.setRenderHint(QPainter.Antialiasing, False)

    def _draw_states(self, p: QPainter, y0: int) -> None:
        fm = p.fontMetrics()
        p.setPen(QPen(_TEXT))
        p.drawText(_PAD_L + 3, y0 + fm.ascent(), "filter state")
        y = y0 + fm.height() + 2
        for key, name, color in _STATE_ROWS:
            col = self._veh.series.get(key)
            p.setPen(QPen(_c(80, 80, 95)))
            p.drawText(2, y + _STATE_ROW_H - 3, name)
            p.setPen(QPen(_GRID, 1))
            p.drawLine(_PAD_L, y + _STATE_ROW_H - 1, _PAD_L + self._plot_w(), y + _STATE_ROW_H - 1)
            if col:
                p.setPen(QPen(color, 1.4))
                for t_rel, value in zip(self._veh.t, col):
                    if value >= 0.5:
                        x = int(self._x(t_rel))
                        p.drawLine(x, y, x, y + _STATE_ROW_H - 2)
            y += _STATE_ROW_H

    def _draw_time_axis(self, p: QPainter) -> None:
        y = self.height() - 2
        step = 1.0 if self._duration <= 16 else 2.0
        p.setPen(QPen(_TEXT))
        t_rel = 0.0
        while t_rel <= self._duration:
            p.drawText(int(self._x(t_rel)) - 8, y, f"{t_rel:.0f}s")
            t_rel += step
        p.drawText(2, y, "t")


class FilterChartWindow(QMainWindow):
    """Separate top-level window: vehicle picker, lane toggles, synced cursor."""

    seeked = Signal(float)
    key_forwarded = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("AEB filter tuning")
        self.resize(1320, 840)
        self._trace: ClipTrace | None = None
        self._vid: int | None = None
        self._follow = True

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Vehicle"))
        self._picker = QComboBox()
        self._picker.setMinimumWidth(260)
        self._picker.setFocusPolicy(Qt.NoFocus)
        self._picker.currentIndexChanged.connect(self._on_pick)
        bar.addWidget(self._picker)
        self._follow_box = QCheckBox("follow target")
        self._follow_box.setChecked(True)
        self._follow_box.setFocusPolicy(Qt.NoFocus)
        self._follow_box.toggled.connect(self._on_follow)
        bar.addWidget(self._follow_box)
        bar.addSpacing(16)
        for build in _LANES:
            title = build().title
            box = QCheckBox(title)
            box.setChecked(True)
            box.setFocusPolicy(Qt.NoFocus)
            box.toggled.connect(lambda on, t=title: self._chart.set_lane_enabled(t, on))
            bar.addWidget(box)
        bar.addStretch(1)
        self._clip_lbl = QLabel("")
        self._clip_lbl.setStyleSheet("color:#888;")
        bar.addWidget(self._clip_lbl)
        outer.addLayout(bar)

        body = QHBoxLayout()
        self._chart = FilterChart()
        self._chart.seeked.connect(self.seeked)
        self._chart.hovered.connect(self._on_hover)
        body.addWidget(self._chart, 1)
        self._readout = QLabel("")
        self._readout.setFixedWidth(_READOUT_W)
        self._readout.setAlignment(Qt.AlignTop)
        self._readout.setFont(QFont("Consolas", 8))
        self._readout.setStyleSheet(
            "background:#16161c; color:#bbb; border:1px solid #333; padding:6px;")
        self._readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.addWidget(self._readout)
        outer.addLayout(body, 1)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_clip(self, trace: ClipTrace | None, clip_id: str,
                 decision: list[tuple[float, int]],
                 window: tuple[float, float] | None,
                 target_vid: int | None, replayed=None) -> None:
        self._trace = trace
        self._clip_lbl.setText(clip_id)
        self._chart.set_decision(decision, window)
        self._chart.set_replayed(replayed)
        self._rebuild_picker(target_vid)

    @classmethod
    def attached_to(cls, review) -> "FilterChartWindow":
        """Build the window already wired to the review window's clock and keymap."""
        win = cls(review)
        win.seeked.connect(review._seek_time)
        win.key_forwarded.connect(review.keyPressEvent)
        return win

    def show_clip(self, loaded, frames, window, target_vid, cursor_t: float) -> None:
        """Everything the chart needs for one clip, packed from the review window's state."""
        self.set_clip(loaded.trace, loaded.clip.metadata.clip_id,
                      decision_track(frames), window, target_vid, loaded.evaluated)
        self.set_cursor(cursor_t)

    def set_window(self, window: tuple[float, float] | None) -> None:
        self._chart.set_decision(self._chart._decision, window)

    def set_replayed(self, replayed) -> None:
        """Hand over the re-run decision band, or None to show it as pending."""
        self._chart.set_replayed(replayed)

    def set_target(self, target_vid: int | None) -> None:
        """Follow the review window's target pick unless the user took the picker over."""
        if self._follow and target_vid is not None:
            self._select_vid(target_vid)

    def set_cursor(self, t_rel: float) -> None:
        self._chart.set_cursor(t_rel)
        if self._chart._hover is None:
            self._render_readout(t_rel, live=True)

    def _rebuild_picker(self, target_vid: int | None) -> None:
        self._picker.blockSignals(True)
        self._picker.clear()
        if self._trace is not None:
            for vid in self._trace.order:
                self._picker.addItem(self._trace.vehicles[vid].label(), vid)
        self._picker.blockSignals(False)
        want = self._trace.pick_default(target_vid) if self._trace else None
        self._vid = None
        self._select_vid(want)

    def _select_vid(self, vid: int | None) -> None:
        if vid is None or self._trace is None or vid not in self._trace.vehicles:
            if self._trace is None or not self._trace.order:
                self._vid = None
                self._chart.set_trace(self._trace, None)
            return
        if vid == self._vid:
            return
        self._vid = vid
        index = self._picker.findData(vid)
        if index >= 0 and index != self._picker.currentIndex():
            self._picker.blockSignals(True)
            self._picker.setCurrentIndex(index)
            self._picker.blockSignals(False)
        self._chart.set_trace(self._trace, self._trace.vehicles[vid])
        self._render_readout(self._chart._cursor, live=True)

    def _on_pick(self, index: int) -> None:
        vid = self._picker.itemData(index)
        if vid is not None:
            self._follow = False
            self._follow_box.setChecked(False)
            self._select_vid(int(vid))

    def _on_follow(self, on: bool) -> None:
        self._follow = bool(on)

    def _on_hover(self, t_rel: float) -> None:
        if t_rel < 0.0:
            self._render_readout(self._chart._cursor, live=True)
        else:
            self._render_readout(t_rel, live=False)

    def _render_readout(self, t_rel: float, *, live: bool) -> None:
        if self._trace is None or self._vid is None:
            self._readout.setText("")
            return
        veh = self._trace.vehicles[self._vid]
        row = veh.at(t_rel)
        if not row:
            self._readout.setText("")
            return
        head = f"{'cursor' if live else 'hover'}  t={t_rel:5.2f}s   #{veh.vid}"
        lines = [head, ""]
        for build in _LANES:
            lane = build()
            lines.append(lane.short or lane.title)
            for sig in lane.signals:
                raw = row.get(sig.key)
                if sig.key == "lag_frac":
                    elapsed, dur = row.get("lag_elapsed"), row.get("lag_freeze_dur")
                    raw = (elapsed / dur) if (elapsed == elapsed and dur) else None
                if raw is None or raw != raw:
                    continue
                lines.append(f"  {sig.key:<15}{raw:9.3f}")
            lines.append("")
        active = [name for key, name, _clr in _STATE_ROWS if row.get(key, 0.0) >= 0.5]
        lines.append("state: " + (", ".join(active) if active else "none"))
        self._readout.setText("\n".join(lines))

    def keyPressEvent(self, event) -> None:
        """Every review binding keeps working while this window has focus."""
        self.key_forwarded.emit(event)

    def closeEvent(self, event) -> None:
        self.hide()
        event.ignore()


def decision_track(frames) -> list[tuple[float, int]]:
    """Recorded AEB state per replayed tick, for the alignment band at the top."""
    out = []
    for f in frames:
        state = 2 if f.live_aeb.aeb_brake else (1 if f.live_aeb.aeb_warn else 0)
        out.append((f.t_rel, state))
    return out

