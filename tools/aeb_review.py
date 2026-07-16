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

import base64
from dataclasses import replace

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPlainTextEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from core.aeb.clip_replay import ReviewFrame, replay_clip
from core.aeb.clip_schema import ClipMetadata, Label
from core.aeb.clip_score import class_window_warning
from core.aeb.clip_store import ClipInfo, ClipStore
from core.aeb.debug_window import AEBDebugWindow

_CLASSES = ["tp", "good_intervention", "fp", "fn", "tn", "ignore"]


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

        # Directory scan, cached across reloads. peek_metadata is decoded once
        # per clip and reused (keyed on mtime+size) so filtering / searching /
        # re-selecting never re-reads the store.
        self._entries: list[tuple[ClipInfo, ClipMetadata | None]] = []
        self._meta_cache: dict[str, tuple[float, int, ClipMetadata | None]] = {}

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance)

        self._build_ui()
        self._refresh_clips()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        row = QHBoxLayout(root)

        # Left: clip list
        left = QVBoxLayout()
        header = QHBoxLayout()
        header.addWidget(QLabel("Clips"))
        header.addStretch(1)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedWidth(70)
        self._refresh_btn.clicked.connect(self._refresh_clips)
        header.addWidget(self._refresh_btn)
        left.addLayout(header)

        self._search = QLineEdit()
        self._search.setPlaceholderText("search id / trigger / notes")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        left.addWidget(self._search)

        self._class_filter = QComboBox()
        self._class_filter.addItems(["all", "untagged", "tagged"] + _CLASSES)
        self._class_filter.currentTextChanged.connect(lambda _t: self._apply_filter())
        left.addWidget(self._class_filter)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_select)
        left.addWidget(self._list, 1)
        self._count_lbl = QLabel("")
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
        transport.addSpacing(12)
        self._veh_paths = QCheckBox("Vehicle paths")
        self._veh_paths.setChecked(True)
        self._veh_paths.toggled.connect(self._scene.set_vehicle_paths)
        transport.addWidget(self._veh_paths)
        transport.addStretch(1)
        self._decision_lbl = QLabel("")
        transport.addWidget(self._decision_lbl)
        center.addLayout(transport)
        row.addLayout(center, 1)

        # Right: clip identity + annotation
        right = QVBoxLayout()
        self._clip_name_lbl = QLabel("(no clip selected)")
        self._clip_name_lbl.setStyleSheet("font-weight:bold;")
        self._clip_name_lbl.setWordWrap(True)
        self._clip_name_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right.addWidget(self._clip_name_lbl)
        self._clip_meta_lbl = QLabel("")
        self._clip_meta_lbl.setStyleSheet("color:#999;")
        self._clip_meta_lbl.setWordWrap(True)
        right.addWidget(self._clip_meta_lbl)

        # Context screenshot grabbed at the trigger moment (above the labeling).
        self._thumb_lbl = QLabel("(no screenshot)")
        self._thumb_lbl.setMinimumHeight(150)
        self._thumb_lbl.setAlignment(Qt.AlignCenter)
        self._thumb_lbl.setStyleSheet("background:#111; color:#666; border:1px solid #333;")
        right.addWidget(self._thumb_lbl)
        right.addWidget(_hline())

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

    def _refresh_clips(self) -> None:
        """Rescan the store for new/changed clips, then re-render the list."""
        self._rescan()
        self._apply_filter()

    def _rescan(self) -> None:
        """Scan the store, decoding metadata only for new or changed clips.

        Cached entries are reused whenever the file's mtime and size are
        unchanged, so a manual refresh only pays for clips that actually
        appeared or were relabelled.
        """
        infos = self._store.list_clips()
        live: set[str] = set()
        entries: list[tuple[ClipInfo, ClipMetadata | None]] = []
        for info in infos:
            key = str(info.path)
            live.add(key)
            cached = self._meta_cache.get(key)
            if cached is not None and cached[0] == info.mtime and cached[1] == info.size_bytes:
                meta = cached[2]
            else:
                meta = self._store.peek_metadata(info.path)
                self._meta_cache[key] = (info.mtime, info.size_bytes, meta)
            entries.append((info, meta))
        for stale in [k for k in self._meta_cache if k not in live]:
            del self._meta_cache[stale]
        self._entries = entries

    def _apply_filter(self) -> None:
        """Render the cached scan through the current search + class filter."""
        search = self._search.text().strip().lower()
        cls_filter = self._class_filter.currentText()
        self._list.blockSignals(True)
        self._list.clear()
        shown = 0
        for info, meta in self._entries:
            if not _entry_visible(meta, search, cls_filter):
                continue
            self._list.addItem(_clip_item(info, meta))
            shown += 1
        self._count_lbl.setText(f"{shown} shown / {len(self._entries)} total")
        self._reselect(self._path)
        self._list.blockSignals(False)

    def _reselect(self, path) -> None:
        """Restore the selection to *path* after a rebuild, if still visible."""
        if not path:
            return
        target = str(path)
        for i in range(self._list.count()):
            if self._list.item(i).data(Qt.UserRole) == target:
                self._list.setCurrentRow(i)
                return

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
        m = clip.metadata
        self._clip_name_lbl.setText(m.clip_id)
        self._clip_meta_lbl.setText(
            f"{m.trigger_source} · {m.session_kind} · {m.captured_at}\n"
            f"{m.frame_count} frames / {m.tick_count} ticks · v{m.client_version}"
        )
        self._set_thumbnail(m.thumbnail_jpeg)
        self._frames = replay_clip(clip)
        self._idx = 0
        dur = self._frames[-1].t_rel if self._frames else 1.0
        self._strip.set_frames(self._frames, dur)
        self._load_label_into_form(clip)
        self._refresh()

    def _set_thumbnail(self, b64: str | None) -> None:
        self._thumb_lbl.setPixmap(QPixmap())
        if not b64:
            self._thumb_lbl.setText("(no screenshot)")
            return
        pix = QPixmap()
        if not pix.loadFromData(base64.b64decode(b64), "JPEG"):
            self._thumb_lbl.setText("(thumbnail decode failed)")
            return
        w = max(self._thumb_lbl.width(), 200)
        self._thumb_lbl.setText("")
        self._thumb_lbl.setPixmap(
            pix.scaled(w, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

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
            # Label write rewrote the file: drop its cache entry so the rescan
            # re-decodes just this clip (badge, filter state) and keeps it selected.
            self._meta_cache.pop(str(self._path), None)
            self._refresh_clips()

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


def _entry_visible(meta: ClipMetadata | None, search: str, cls_filter: str) -> bool:
    """Whether a scanned clip passes the class filter and search text."""
    label = meta.label if meta is not None else None
    cls = label.class_ if label is not None else None
    if cls_filter == "untagged" and cls is not None:
        return False
    if cls_filter == "tagged" and cls is None:
        return False
    if cls_filter not in ("all", "untagged", "tagged") and cls != cls_filter:
        return False
    if search:
        hay = " ".join(part for part in (
            (meta.clip_id if meta else ""),
            (meta.trigger_source if meta else ""),
            (cls or ""),
            (label.notes if label is not None else ""),
        ) if part).lower()
        if search not in hay:
            return False
    return True


def _clip_item(info: ClipInfo, meta: ClipMetadata | None) -> QListWidgetItem:
    """One list row: clip id, tag badge, trigger source, and size."""
    tagged = meta is not None and meta.label is not None
    cls = meta.label.class_ if tagged else None
    trig = meta.trigger_source if meta else "?"
    badge = f"● {cls}" if tagged else "○ untagged"
    kb = info.size_bytes // 1024
    cid = meta.clip_id[:8] if meta else "????????"
    item = QListWidgetItem(f"{cid}   {badge}\n{trig}  {kb} KB")
    item.setData(Qt.UserRole, str(info.path))
    return item


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
