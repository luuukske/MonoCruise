"""Dev-only PySide6 AEB clip labeling UI (repo root: python -m tools.aeb_review). Not shipped."""
from __future__ import annotations

import os
import sys

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from collections import OrderedDict

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPlainTextEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from core.aeb.clip_replay import ReviewFrame
from core.aeb.clip_schema import ClipMetadata, Label
from core.aeb.clip_score import class_window_warning
from core.aeb.clip_store import ClipInfo, ClipStore, contributed_clip_root
from tools.aeb_review_widgets import (
    ClipLoader, DecisionStrip, Loaded, SceneWidget, ThumbnailView, action_index, recorded_band,
)

_CLASSES = ["tp", "good_intervention", "fp", "fn", "tn", "ignore"]

# Digit row picks a class; order matches _CLASSES.
_CLASS_KEYS = {
    Qt.Key_1: "tp", Qt.Key_2: "good_intervention", Qt.Key_3: "fp",
    Qt.Key_4: "fn", Qt.Key_5: "tn", Qt.Key_6: "ignore",
}

_STEP_COARSE = 10       # frames per Shift+arrow
_CACHE_MAX = 4          # replayed clips held in RAM, ~14 MB each
_PREFETCH_AHEAD = 2

_KEYMAP_TEXT = """\
CLIPS      N / P             next / prev clip
           Ctrl+N            next untagged

TRANSPORT  Left / Right      step 1 frame
           Shift+Left/Right  step 10 frames
           Home / End        first / last frame
           Space             play / pause

LABEL      1 tp    2 good_intervention   3 fp
           4 fn    5 tn                  6 ignore
           PageUp / PageDown  severity +/-

WINDOW     [  set start at cursor
           ]  set end at cursor
           \\  clear window
           W  accept the proposed window

TARGET     V  pick-on-scene, then click the vehicle

SAVE       Enter  save and go to next untagged

NOTES      Tab into notes, Esc back out
F1         hide this panel"""


class ReviewWindow(QMainWindow):

    load_requested = Signal(str)
    scan_requested = Signal(object)

    def __init__(self, store: ClipStore) -> None:
        super().__init__()
        self.setWindowTitle("AEB Clip Review")
        self.resize(1650, 900)
        self._store = store
        self._clip = None
        self._path = None
        self._frames: list[ReviewFrame] = []
        self._idx = 0
        self._target_vid: int | None = None
        self._window: tuple[float, float] | None = None
        self._proposal: tuple[float, float] | None = None

        # Clip list cache: peek_metadata once per mtime+size; reload skips store re-reads.
        self._entries: list[tuple[ClipInfo, ClipMetadata | None]] = []
        self._meta_cache: dict[str, tuple[float, int, ClipMetadata | None]] = {}
        self._visible: list[str] = []

        # Decoded-clip LRU plus the request pipeline that fills it.
        self._cache: OrderedDict[str, Loaded] = OrderedDict()
        self._inflight: str | None = None
        self._queue: list[str] = []
        self._awaiting: str | None = None

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance)

        self._thread = QThread(self)
        self._loader = ClipLoader(store)
        self._loader.moveToThread(self._thread)
        self.load_requested.connect(self._loader.load)
        self.scan_requested.connect(self._loader.scan)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.scanned.connect(self._on_scanned)
        self._thread.start()

        self._build_ui()
        self._refresh_clips()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        row = QHBoxLayout(root)
        row.addWidget(self._build_left())
        row.addLayout(self._build_center(), 1)
        row.addWidget(self._build_right())
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _build_left(self) -> QWidget:
        left = QVBoxLayout()
        header = QHBoxLayout()
        header.addWidget(QLabel("Clips"))
        header.addStretch(1)
        self._refresh_btn = _button("Refresh", self._refresh_clips, width=70)
        header.addWidget(self._refresh_btn)
        left.addLayout(header)

        self._search = QLineEdit()
        self._search.setPlaceholderText("search id / trigger / notes")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        self._search.installEventFilter(self)
        left.addWidget(self._search)

        self._class_filter = QComboBox()
        self._class_filter.addItems(["all", "untagged", "tagged"] + _CLASSES)
        self._class_filter.setFocusPolicy(Qt.NoFocus)
        self._class_filter.currentTextChanged.connect(lambda _t: self._apply_filter())
        left.addWidget(self._class_filter)

        self._list = QListWidget()
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.currentItemChanged.connect(self._on_select)
        left.addWidget(self._list, 1)
        self._count_lbl = QLabel("scanning...")
        left.addWidget(self._count_lbl)

        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(300)
        return lw

    def _build_center(self) -> QVBoxLayout:
        center = QVBoxLayout()
        self._scene = SceneWidget()
        self._scene.vehicle_picked.connect(self._on_vehicle_picked)
        center.addWidget(self._scene, 1)

        self._keys_lbl = QLabel(_KEYMAP_TEXT, self._scene)
        self._keys_lbl.setFont(QFont("Consolas", 9))
        self._keys_lbl.setStyleSheet(
            "background:rgba(10,10,14,230); color:#bbb; border:1px solid #444; padding:10px;"
        )
        self._keys_lbl.move(14, 14)
        self._keys_lbl.adjustSize()

        self._strip = DecisionStrip()
        self._strip.seeked.connect(self._seek_time)
        center.addWidget(self._strip)

        transport = QHBoxLayout()
        for text, fn in [("|<", self._first), ("<", self._prev),
                         (">", self._next), (">|", self._last)]:
            transport.addWidget(_button(text, fn, width=38))
        self._play_btn = _button("Play", self._toggle_play)
        transport.addWidget(self._play_btn)
        self._time_lbl = QLabel("t=0.00s")
        transport.addWidget(self._time_lbl)
        transport.addSpacing(12)
        self._veh_paths = QCheckBox("Vehicle paths")
        self._veh_paths.setChecked(True)
        self._veh_paths.setFocusPolicy(Qt.NoFocus)
        self._veh_paths.toggled.connect(self._scene.set_vehicle_paths)
        transport.addWidget(self._veh_paths)
        transport.addSpacing(12)
        transport.addWidget(QLabel("F1 keys"))
        transport.addStretch(1)
        self._decision_lbl = QLabel("")
        transport.addWidget(self._decision_lbl)
        center.addLayout(transport)
        return center

    def _build_right(self) -> QWidget:
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
        self._thumb = ThumbnailView()
        right.addWidget(self._thumb)
        right.addWidget(_hline())

        right.addWidget(QLabel("<b>Label</b>"))
        right.addWidget(QLabel("Class  (1-6)"))
        self._class = QComboBox()
        self._class.addItems(_CLASSES)
        self._class.setFocusPolicy(Qt.NoFocus)
        self._class.currentTextChanged.connect(lambda _t: self._sync_label_widgets())
        right.addWidget(self._class)
        self._warn_lbl = QLabel("")
        self._warn_lbl.setWordWrap(True)
        self._warn_lbl.setStyleSheet("color:#e0a020;")
        right.addWidget(self._warn_lbl)

        right.addWidget(QLabel("Severity 1-5  (PgUp/PgDn)"))
        self._severity = QSpinBox()
        self._severity.setRange(1, 5)
        self._severity.setValue(3)
        self._severity.setFocusPolicy(Qt.NoFocus)
        right.addWidget(self._severity)

        right.addWidget(_hline())
        right.addWidget(QLabel("Should-trigger window  ( [ ] \\ W )"))
        self._window_lbl = QLabel("none (must NOT trigger)")
        right.addWidget(self._window_lbl)
        self._proposal_lbl = QLabel("")
        self._proposal_lbl.setStyleSheet("color:#50d282;")
        self._proposal_lbl.setWordWrap(True)
        right.addWidget(self._proposal_lbl)
        wb = QHBoxLayout()
        for text, fn in [("Start", self._win_start), ("End", self._win_end),
                         ("Clear", self._win_clear), ("Accept", self._win_accept)]:
            wb.addWidget(_button(text, fn))
        right.addLayout(wb)

        right.addWidget(_hline())
        right.addWidget(QLabel("Target vehicle  (V)"))
        tb = QHBoxLayout()
        self._target_lbl = QLabel("none")
        self._pick_btn = QPushButton("Pick on scene")
        self._pick_btn.setCheckable(True)
        self._pick_btn.setFocusPolicy(Qt.NoFocus)
        self._pick_btn.toggled.connect(self._toggle_pick)
        tb.addWidget(self._target_lbl, 1)
        tb.addWidget(self._pick_btn)
        right.addLayout(tb)

        right.addWidget(QLabel("Desired peak decel (m/s2, 0 = unset)"))
        self._desired = QDoubleSpinBox()
        self._desired.setRange(0.0, 12.0)
        self._desired.setSingleStep(0.5)
        self._desired.setFocusPolicy(Qt.NoFocus)
        right.addWidget(self._desired)

        right.addWidget(QLabel("Notes  (Tab in, Esc out)"))
        self._notes = QPlainTextEdit()
        self._notes.setFixedHeight(70)
        self._notes.setTabChangesFocus(True)   # Tab is a binding, not a note character
        self._notes.installEventFilter(self)
        right.addWidget(self._notes)

        self._save_btn = _button("Save label  (Enter)", self._save_and_advance)
        right.addWidget(self._save_btn)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        right.addWidget(self._status)
        right.addStretch(1)

        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(320)
        return rw

    # Keyboard

    def eventFilter(self, obj, event) -> bool:
        """Esc leaves a text field so the single-key bindings work again."""
        if (event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key_Escape
                and obj in (self._notes, self._search)):
            self.setFocus()
            return True
        return super().eventFilter(obj, event)

    def focusNextPrevChild(self, next_: bool) -> bool:
        """Tab is bound to the notes field, so it must not walk the focus chain."""
        return False

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mods = event.modifiers()
        shift = bool(mods & Qt.ShiftModifier)
        ctrl = bool(mods & Qt.ControlModifier)

        if key in _CLASS_KEYS:
            self._class.setCurrentText(_CLASS_KEYS[key])
        elif key == Qt.Key_PageUp:
            self._severity.setValue(min(5, self._severity.value() + 1))
        elif key == Qt.Key_PageDown:
            self._severity.setValue(max(1, self._severity.value() - 1))
        elif key == Qt.Key_Left:
            self._step(-_STEP_COARSE if shift else -1)
        elif key == Qt.Key_Right:
            self._step(_STEP_COARSE if shift else 1)
        elif key == Qt.Key_Home:
            self._first()
        elif key == Qt.Key_End:
            self._last()
        elif key == Qt.Key_Space:
            self._toggle_play()
        elif key == Qt.Key_BracketLeft:
            self._win_start()
        elif key == Qt.Key_BracketRight:
            self._win_end()
        elif key == Qt.Key_Backslash:
            self._win_clear()
        elif key == Qt.Key_W:
            self._win_accept()
        elif key == Qt.Key_V:
            self._pick_btn.setChecked(not self._pick_btn.isChecked())
        elif key == Qt.Key_N and ctrl:
            self._advance_to_untagged()
        elif key == Qt.Key_N:
            self._step_clip(1)
        elif key == Qt.Key_P:
            self._step_clip(-1)
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._save_and_advance()
        elif key == Qt.Key_Tab:
            self._notes.setFocus()
        elif key == Qt.Key_F1:
            self._keys_lbl.setVisible(not self._keys_lbl.isVisible())
        else:
            super().keyPressEvent(event)

    # Clip list

    def _refresh_clips(self) -> None:
        """Kick a background rescan; the list re-renders when it lands."""
        self._count_lbl.setText("scanning...")
        self.scan_requested.emit(dict(self._meta_cache))

    @Slot(object)
    def _on_scanned(self, entries) -> None:
        self._entries = entries
        self._meta_cache = {
            str(info.path): (info.mtime, info.size_bytes, meta) for info, meta in entries
        }
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Render the cached scan through the current search + class filter."""
        search = self._search.text().strip().lower()
        cls_filter = self._class_filter.currentText()
        self._list.blockSignals(True)
        self._list.clear()
        self._visible = []
        untagged = 0
        for info, meta in self._entries:
            if meta is None or meta.label is None:
                untagged += 1
            if not _entry_visible(meta, search, cls_filter):
                continue
            self._list.addItem(_clip_item(info, meta))
            self._visible.append(str(info.path))
        self._count_lbl.setText(
            f"{len(self._visible)} shown / {len(self._entries)} total / {untagged} untagged"
        )
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

    def _row_of(self, path: str) -> int:
        for i in range(self._list.count()):
            if self._list.item(i).data(Qt.UserRole) == path:
                return i
        return -1

    def _select_path(self, path: str) -> None:
        i = self._row_of(path)
        if i >= 0:
            self._list.setCurrentRow(i)

    def _is_untagged(self, path: str) -> bool:
        cached = self._meta_cache.get(path)
        meta = cached[2] if cached else None
        return meta is None or meta.label is None

    def _order_after_current(self) -> list[str]:
        """Visible paths starting after the current clip, wrapping once."""
        try:
            i = self._visible.index(str(self._path))
        except ValueError:
            return list(self._visible)
        return self._visible[i + 1:] + self._visible[:i]

    def _step_clip(self, delta: int) -> None:
        if not self._visible:
            return
        try:
            i = self._visible.index(str(self._path))
        except ValueError:
            self._select_path(self._visible[0 if delta > 0 else -1])
            return
        j = max(0, min(len(self._visible) - 1, i + delta))
        if j != i:
            self._select_path(self._visible[j])

    def _advance_to_untagged(self) -> None:
        for path in self._order_after_current():
            if self._is_untagged(path):
                self._select_path(path)
                return
        self._status.setText("no untagged clips left in this filter")

    def _on_select(self, current, _prev) -> None:
        if current is None:
            return
        self._play_timer.stop()
        self._play_btn.setText("Play")
        path = current.data(Qt.UserRole)
        self._path = path
        self._awaiting = path

        hit = self._cache.get(path)
        if hit is not None:
            self._cache.move_to_end(path)
            self._show(path, hit)
        else:
            self._clip_name_lbl.setText("loading...")
            self._request(path, urgent=True)

    # Background decode pipeline

    def _request(self, path: str, *, urgent: bool = False) -> None:
        if path in self._cache or path == self._inflight:
            return
        if path in self._queue:
            self._queue.remove(path)
        if urgent:
            self._queue.insert(0, path)
        else:
            self._queue.append(path)
        self._pump()

    def _pump(self) -> None:
        if self._inflight is not None or not self._queue:
            return
        self._inflight = self._queue.pop(0)
        self.load_requested.emit(self._inflight)

    @Slot(str, object, object)
    def _on_loaded(self, path: str, clip, frames) -> None:
        self._inflight = None
        if clip is not None:
            self._cache[path] = Loaded(
                clip=clip, frames=frames,
                proposal=recorded_band(frames), action_idx=action_index(frames),
            )
            self._cache.move_to_end(path)
            while len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)
        if path == self._awaiting:
            if clip is None:
                self._status.setText("failed to load clip")
                self._clip_name_lbl.setText("(load failed)")
            else:
                self._show(path, self._cache[path])
        self._pump()

    def _prefetch(self) -> None:
        """Queue the next few rows so the following selections land instantly."""
        i = self._row_of(str(self._path))
        if i < 0:
            return
        for path in self._visible[i + 1:i + 1 + _PREFETCH_AHEAD]:
            self._request(path)

    def _show(self, path: str, loaded: Loaded) -> None:
        self._clip = loaded.clip
        self._frames = loaded.frames
        self._proposal = loaded.proposal
        m = loaded.clip.metadata
        self._clip_name_lbl.setText(m.clip_id)
        self._thumb.set_jpeg(m.thumbnail_jpeg)
        meta = (
            f"{m.trigger_source} · {m.session_kind} · {m.captured_at}\n"
            f"{m.frame_count} frames / {m.tick_count} ticks · v{m.client_version}"
        )
        size = self._thumb.source_size()
        if size is not None:
            meta += f" · thumb {size[0]}x{size[1]}"
        self._clip_meta_lbl.setText(meta)
        dur = self._frames[-1].t_rel if self._frames else 1.0
        self._strip.set_frames(self._frames, dur)
        self._strip.set_proposal(self._proposal)
        self._idx = loaded.action_idx
        self._load_label_into_form(loaded.clip)
        self._refresh()
        self._prefetch()

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
                self._window = (float(lbl.should_trigger["from_t"]),
                                float(lbl.should_trigger["to_t"]))
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
        if self._proposal is None:
            self._proposal_lbl.setText("no proposal (nothing recorded)")
        elif self._window == self._proposal:
            self._proposal_lbl.setText("proposal accepted")
        else:
            self._proposal_lbl.setText(
                f"W accepts {self._proposal[0]:.2f} .. {self._proposal[1]:.2f} s"
            )
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

    def _win_accept(self) -> None:
        """Commit the proposed window. Deliberate: never applied on load."""
        if self._proposal is None:
            self._status.setText("no proposal for this clip")
            return
        self._window = self._proposal
        self._sync_label_widgets()

    def _toggle_pick(self, on: bool) -> None:
        self._scene.pick_mode = on

    def _on_vehicle_picked(self, vid: int) -> None:
        self._target_vid = int(vid)
        self._pick_btn.setChecked(False)
        self._sync_label_widgets()

    def _save(self) -> bool:
        if self._path is None:
            return False
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
            # The rewrite invalidates one clip only: re-peek that one instead of
            # rescanning, so a save stays in milliseconds at 600+ clips.
            self._cache.pop(str(self._path), None)
            self._refresh_entry(str(self._path))
        return ok

    def _refresh_entry(self, path: str) -> None:
        meta = self._store.peek_metadata(path)
        for i, (info, _old) in enumerate(self._entries):
            if str(info.path) != path:
                continue
            try:
                st = info.path.stat()
                info = ClipInfo(path=info.path, name=info.name,
                                size_bytes=st.st_size, mtime=st.st_mtime)
            except OSError:
                pass
            self._entries[i] = (info, meta)
            self._meta_cache[path] = (info.mtime, info.size_bytes, meta)
            break
        self._apply_filter()

    def _save_and_advance(self) -> None:
        candidates = self._order_after_current()
        if not self._save():
            return
        for path in candidates:
            if path in self._visible and self._is_untagged(path):
                self._select_path(path)
                return
        for path in candidates:
            if path in self._visible:
                self._select_path(path)
                self._status.setText("saved; no untagged clips left in this filter")
                return

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

    def _step(self, delta: int) -> None:
        if not self._frames:
            return
        self._idx = max(0, min(len(self._frames) - 1, self._idx + delta))
        self._refresh()

    def _first(self) -> None:
        self._step(-len(self._frames))

    def _last(self) -> None:
        self._step(len(self._frames))

    def _prev(self) -> None:
        self._step(-1)

    def _next(self) -> None:
        self._step(1)

    def _seek_time(self, t_rel: float) -> None:
        if not self._frames:
            return
        self._idx = min(range(len(self._frames)),
                        key=lambda i: abs(self._frames[i].t_rel - t_rel))
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
            f"recorded: {state}  demand={f.raw_target_ms2:.1f}"
            f" (sent {la.target_decel_ms2:.1f})  ttc={_fmt(la.time_to_collision)}"
            f"  |  truth: {truth}"
        )

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)


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


def _button(text: str, fn, *, width: int | None = None) -> QPushButton:
    """Keyboard-transparent button: focus stays on the window so bindings keep working."""
    b = QPushButton(text)
    b.setFocusPolicy(Qt.NoFocus)
    b.clicked.connect(fn)
    if width is not None:
        b.setFixedWidth(width)
    return b


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color:#333;")
    return f


def _fmt(v: float) -> str:
    return f"{v:.2f}" if v < 100 else "inf"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AEB clip review.")
    parser.add_argument("--root", default=None,
                        help="clip store to open (default: the local capture store)")
    parser.add_argument("--contributed", action="store_true",
                        help="open the pulled-in contributed store instead")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    root = args.root or (contributed_clip_root() if args.contributed else None)
    store = ClipStore(root=root)
    app = QApplication.instance() or QApplication(sys.argv)
    win = ReviewWindow(store)
    # Two stores now exist, so the window has to say which one is open.
    win.setWindowTitle(f"AEB Clip Review  [{store.root}]")
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
