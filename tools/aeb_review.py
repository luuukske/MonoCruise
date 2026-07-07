"""AEB clip review + labelling tool (repo-only, never shipped).

Standalone PySide6 app for tagging captured AEB clips. Lists clips from the
local store, replays each one through the real radar smoothing, renders the
scene with the existing AEB debug renderer, and lets you scrub, mark a
should-trigger window, pick the threat vehicle, assign class + severity, and
persist a label back into the clip (plan section 9).

It is never packaged into shipped builds (PyInstaller specs exclude ``tools/``);
testers only capture and submit. Run from the repo root:

    python -m tools.aeb_review
"""

from __future__ import annotations

import os
import sys

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QPlainTextEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from core.aeb.clip_replay import ReviewFrame, replay_clip
from core.aeb.clip_schema import Label
from core.aeb.clip_score import class_window_warning
from core.aeb.clip_store import ClipStore
from core.aeb.debug_window import AEBDebugWindow

_CLASSES = ["tp", "good_intervention", "fp", "fn", "ignore"]


class SceneWidget(AEBDebugWindow):
    """Debug renderer fed replayed snapshots; click picks the nearest vehicle."""

    vehicle_picked = Signal(int)

    def __init__(self) -> None:
        super().__init__(snapshot_provider=lambda: self._snap,
                         acc_provider=lambda: None, auto_refresh=False)
        self._snap = None
        self.pick_mode = False

    def set_snapshot(self, snap) -> None:
        self._snap = snap
        self.update()

    def mousePressEvent(self, event) -> None:
        if not self.pick_mode or self._snap is None:
            return
        px, py = event.position().x(), event.position().y()
        best_vid, best_d2 = None, 24.0 ** 2
        for v in self._snap.vehicles:
            sx, sy = self._ws(v["x"], v["z"], self._snap.ego_x, self._snap.ego_z, self._snap.ego_yaw)
            d2 = (sx - px) ** 2 + (sy - py) ** 2
            if d2 < best_d2:
                best_d2, best_vid = d2, v["vid"]
        if best_vid is not None:
            self.vehicle_picked.emit(int(best_vid))


class DecisionStrip(QWidget):
    """Timeline strip: recorded warn/brake bands + should-trigger window + cursor."""

    seeked = Signal(float)   # t_rel in seconds

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(46)
        self.setMouseTracking(True)
        self._frames: list[ReviewFrame] = []
        self._duration = 1.0
        self._cursor = 0.0
        self._window: tuple[float, float] | None = None

    def set_frames(self, frames: list[ReviewFrame], duration: float) -> None:
        self._frames = frames
        self._duration = max(duration, 1e-3)
        self.update()

    def set_cursor(self, t_rel: float) -> None:
        self._cursor = t_rel
        self.update()

    def set_window(self, window: tuple[float, float] | None) -> None:
        self._window = window
        self.update()

    def _x(self, t: float) -> float:
        return 6 + (t / self._duration) * (self.width() - 12)

    def mousePressEvent(self, event) -> None:
        frac = (event.position().x() - 6) / max(self.width() - 12, 1)
        self.seeked.emit(max(0.0, min(1.0, frac)) * self._duration)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 30))
        h = self.height()
        y0, bh = 8, h - 16

        # should-trigger window band
        if self._window is not None:
            a, b = self._window
            xa, xb = self._x(a), self._x(b)
            p.fillRect(int(xa), y0, max(1, int(xb - xa)), bh, QColor(80, 210, 130, 70))

        # recorded decision, one thin bar per frame
        for f in self._frames:
            x = self._x(f.t_rel)
            if f.live_aeb.aeb_brake:
                c = QColor(240, 55, 55)
            elif f.live_aeb.aeb_warn:
                c = QColor(245, 185, 40)
            else:
                c = QColor(60, 70, 80)
            p.setPen(QPen(c, 1.4))
            p.drawLine(int(x), y0, int(x), y0 + bh)

        # cursor
        cx = self._x(self._cursor)
        p.setPen(QPen(QColor(255, 255, 255), 1.6))
        p.drawLine(int(cx), 2, int(cx), h - 2)
        p.end()


class ReviewWindow(QMainWindow):

    def __init__(self, store: ClipStore) -> None:
        super().__init__()
        self.setWindowTitle("AEB Clip Review")
        self.resize(1650, 820)
        self._store = store
        self._clip = None
        self._path = None
        self._frames: list[ReviewFrame] = []
        self._idx = 0
        self._target_vid: int | None = None
        self._window: tuple[float, float] | None = None

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance)

        self._build_ui()
        self._reload_list()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        row = QHBoxLayout(root)

        # Left: clip list
        left = QVBoxLayout()
        self._untagged_only = QCheckBox("Untagged only")
        self._untagged_only.stateChanged.connect(self._reload_list)
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_select)
        self._count_lbl = QLabel("")
        left.addWidget(QLabel("Clips"))
        left.addWidget(self._untagged_only)
        left.addWidget(self._list, 1)
        left.addWidget(self._count_lbl)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(300)
        row.addWidget(lw)

        # Center: scene + strip + transport
        center = QVBoxLayout()
        self._scene = SceneWidget()
        self._scene.vehicle_picked.connect(self._on_vehicle_picked)
        center.addWidget(self._scene, 1)

        self._strip = DecisionStrip()
        self._strip.seeked.connect(self._seek_time)
        center.addWidget(self._strip)

        transport = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_play)
        for text, fn in [("|<", self._first), ("<", self._prev), (">", self._next), (">|", self._last)]:
            b = QPushButton(text)
            b.setFixedWidth(38)
            b.clicked.connect(fn)
            transport.addWidget(b)
        transport.addWidget(self._play_btn)
        self._time_lbl = QLabel("t=0.00s")
        transport.addWidget(self._time_lbl)
        transport.addStretch(1)
        self._decision_lbl = QLabel("")
        transport.addWidget(self._decision_lbl)
        center.addLayout(transport)
        row.addLayout(center, 1)

        # Right: annotation
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Label</b>"))

        right.addWidget(QLabel("Class"))
        self._class = QComboBox()
        self._class.addItems(_CLASSES)
        self._class.currentTextChanged.connect(lambda _t: self._sync_label_widgets())
        right.addWidget(self._class)
        self._warn_lbl = QLabel("")
        self._warn_lbl.setWordWrap(True)
        self._warn_lbl.setStyleSheet("color:#e0a020;")
        right.addWidget(self._warn_lbl)

        right.addWidget(QLabel("Severity (1-5)"))
        self._severity = QSpinBox()
        self._severity.setRange(1, 5)
        self._severity.setValue(3)
        right.addWidget(self._severity)

        right.addWidget(_hline())
        right.addWidget(QLabel("Should-trigger window"))
        self._window_lbl = QLabel("none (must NOT trigger)")
        right.addWidget(self._window_lbl)
        wb = QHBoxLayout()
        for text, fn in [("Set start", self._win_start), ("Set end", self._win_end), ("Clear", self._win_clear)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            wb.addWidget(b)
        right.addLayout(wb)

        right.addWidget(_hline())
        right.addWidget(QLabel("Target vehicle"))
        tb = QHBoxLayout()
        self._target_lbl = QLabel("none")
        self._pick_btn = QPushButton("Pick on scene")
        self._pick_btn.setCheckable(True)
        self._pick_btn.toggled.connect(self._toggle_pick)
        tb.addWidget(self._target_lbl, 1)
        tb.addWidget(self._pick_btn)
        right.addLayout(tb)

        right.addWidget(QLabel("Desired peak decel (m/s2, 0 = unset)"))
        self._desired = QDoubleSpinBox()
        self._desired.setRange(0.0, 12.0)
        self._desired.setSingleStep(0.5)
        right.addWidget(self._desired)

        right.addWidget(QLabel("Notes"))
        self._notes = QPlainTextEdit()
        self._notes.setFixedHeight(90)
        right.addWidget(self._notes)

        self._save_btn = QPushButton("Save label")
        self._save_btn.clicked.connect(self._save)
        right.addWidget(self._save_btn)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        right.addWidget(self._status)
        right.addStretch(1)

        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(320)
        row.addWidget(rw)

    # Clip list

    def _reload_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        clips = self._store.list_clips()
        untagged_only = self._untagged_only.isChecked()
        shown = 0
        for info in clips:
            meta = self._store.peek_metadata(info.path)
            tagged = meta is not None and meta.label is not None
            if untagged_only and tagged:
                continue
            cls = meta.label.class_ if tagged else None
            trig = meta.trigger_source if meta else "?"
            badge = f"● {cls}" if tagged else "○ untagged"
            kb = info.size_bytes // 1024
            item = QListWidgetItem(f"{badge}\n{trig}  {kb} KB  {info.name[:22]}")
            item.setData(Qt.UserRole, str(info.path))
            self._list.addItem(item)
            shown += 1
        self._count_lbl.setText(f"{shown} shown / {len(clips)} total")
        self._list.blockSignals(False)

    def _on_select(self, current, _prev) -> None:
        if current is None:
            return
        self._play_timer.stop()
        self._play_btn.setText("Play")
        path = current.data(Qt.UserRole)
        clip = self._store.load(path)
        if clip is None:
            self._status.setText("failed to load clip")
            return
        self._path = path
        self._clip = clip
        self._frames = replay_clip(clip)
        self._idx = 0
        dur = self._frames[-1].t_rel if self._frames else 1.0
        self._strip.set_frames(self._frames, dur)
        self._load_label_into_form(clip)
        self._refresh()

    # Label form

    def _load_label_into_form(self, clip) -> None:
        lbl = clip.metadata.label
        if lbl is None:
            self._class.setCurrentText("fp")
            self._severity.setValue(3)
            self._window = None
            self._target_vid = None
            self._desired.setValue(0.0)
            self._notes.setPlainText("")
        else:
            self._class.setCurrentText(lbl.class_ if lbl.class_ in _CLASSES else "ignore")
            self._severity.setValue(int(lbl.severity) if lbl.severity else 3)
            if lbl.should_trigger:
                self._window = (float(lbl.should_trigger["from_t"]), float(lbl.should_trigger["to_t"]))
            else:
                self._window = None
            self._target_vid = lbl.target_vid
            self._desired.setValue(lbl.desired_peak_decel_ms2 or 0.0)
            self._notes.setPlainText(lbl.notes or "")
        self._sync_label_widgets()

    def _sync_label_widgets(self) -> None:
        if self._window is None:
            self._window_lbl.setText("none (must NOT trigger)")
        else:
            self._window_lbl.setText(f"{self._window[0]:.2f} .. {self._window[1]:.2f} s")
        self._target_lbl.setText("none" if self._target_vid is None else f"#{self._target_vid}")
        self._strip.set_window(self._window)
        warn = class_window_warning(self._class.currentText(), self._window is not None)
        self._warn_lbl.setText(f"⚠ {warn}" if warn else "")

    def _win_start(self) -> None:
        t = self._cur_t()
        end = self._window[1] if self._window else t
        self._window = (t, max(end, t))
        self._sync_label_widgets()

    def _win_end(self) -> None:
        t = self._cur_t()
        start = self._window[0] if self._window else 0.0
        self._window = (min(start, t), t)
        self._sync_label_widgets()

    def _win_clear(self) -> None:
        self._window = None
        self._sync_label_widgets()

    def _toggle_pick(self, on: bool) -> None:
        self._scene.pick_mode = on

    def _on_vehicle_picked(self, vid: int) -> None:
        self._target_vid = int(vid)
        self._pick_btn.setChecked(False)
        self._sync_label_widgets()

    def _save(self) -> None:
        if self._path is None:
            return
        st = None
        if self._window is not None:
            st = {"from_t": round(self._window[0], 3), "to_t": round(self._window[1], 3)}
        lbl = Label(
            class_=self._class.currentText(),
            severity=int(self._severity.value()),
            should_trigger=st,
            target_vid=self._target_vid,
            desired_peak_decel_ms2=(self._desired.value() or None),
            notes=self._notes.toPlainText().strip(),
        )
        ok = self._store.write_label(self._path, lbl)
        self._status.setText("saved" if ok else "save failed")
        if ok:
            row = self._list.currentRow()
            self._reload_list()
            if 0 <= row < self._list.count():
                self._list.setCurrentRow(row)

    # Transport

    def _cur_t(self) -> float:
        return self._frames[self._idx].t_rel if self._frames else 0.0

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setText("Play")
        elif self._frames:
            self._play_timer.start(33)
            self._play_btn.setText("Pause")

    def _advance(self) -> None:
        if self._idx >= len(self._frames) - 1:
            self._play_timer.stop()
            self._play_btn.setText("Play")
            return
        self._idx += 1
        self._refresh()

    def _first(self) -> None:
        self._idx = 0
        self._refresh()

    def _last(self) -> None:
        self._idx = max(0, len(self._frames) - 1)
        self._refresh()

    def _prev(self) -> None:
        self._idx = max(0, self._idx - 1)
        self._refresh()

    def _next(self) -> None:
        self._idx = min(len(self._frames) - 1, self._idx + 1)
        self._refresh()

    def _seek_time(self, t_rel: float) -> None:
        if not self._frames:
            return
        self._idx = min(range(len(self._frames)), key=lambda i: abs(self._frames[i].t_rel - t_rel))
        self._refresh()

    def _refresh(self) -> None:
        if not self._frames:
            self._scene.set_snapshot(None)
            return
        f = self._frames[self._idx]
        self._scene.set_snapshot(f.snapshot)
        self._strip.set_cursor(f.t_rel)
        self._time_lbl.setText(f"t={f.t_rel:.2f}s  ({self._idx + 1}/{len(self._frames)})")
        la = f.live_aeb
        state = "BRAKE" if la.aeb_brake else ("WARN" if la.aeb_warn else "standby")
        in_win = self._window is not None and self._window[0] <= f.t_rel <= self._window[1]
        truth = "should-trigger" if in_win else ("must-not" if self._window is None else "outside")
        self._decision_lbl.setText(
            f"recorded: {state}  target={la.target_decel_ms2:.1f}  ttc={_fmt(la.time_to_collision)}"
            f"  |  truth: {truth}"
        )


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color:#333;")
    return f


def _fmt(v: float) -> str:
    return f"{v:.2f}" if v < 100 else "inf"


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    store = ClipStore()
    win = ReviewWindow(store)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
