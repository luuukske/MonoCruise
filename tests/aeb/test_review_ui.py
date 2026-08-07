"""Tests for the labelling UI: window proposal, landing frame, and the key bindings."""

from __future__ import annotations

import time

import pytest

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from core.aeb.clip_replay import ReviewFrame, replay_clip
from core.aeb.clip_schema import Label, LiveAEB
from core.aeb.clip_store import ClipStore
from tools.aeb_review import ReviewWindow
from tools.aeb_review_widgets import action_index, recorded_band

from tests.aeb.test_clip_review import _build_replayable_clip


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _frame(t_rel: float, *, warn: bool = False, brake: bool = False,
           ttc: float = 1e9) -> ReviewFrame:
    return ReviewFrame(
        t_rel=t_rel, t_mono=t_rel, snapshot=None,
        live_aeb=LiveAEB(aeb_warn=warn, aeb_brake=brake, time_to_collision=ttc),
        consumed=None,
    )


def test_recorded_band_spans_the_live_reaction():
    frames = [
        _frame(0.0), _frame(0.1, ttc=3.0), _frame(0.2, warn=True, ttc=2.0),
        _frame(0.3, brake=True, ttc=1.0), _frame(0.4, ttc=5.0), _frame(0.5),
    ]
    assert recorded_band(frames) == (0.2, 0.3)


def test_recorded_band_falls_back_to_tracked_threat_then_gives_up():
    tracked = [_frame(0.0), _frame(0.1, ttc=4.0), _frame(0.2, ttc=2.0), _frame(0.3)]
    assert recorded_band(tracked) == (0.1, 0.2)
    assert recorded_band([_frame(0.0), _frame(0.1)]) is None


def test_action_index_lands_on_the_reaction_or_closest_approach():
    reacted = [_frame(0.0), _frame(0.1, ttc=2.0), _frame(0.2, warn=True), _frame(0.3, brake=True)]
    assert action_index(reacted) == 2

    silent = [_frame(0.0, ttc=9.0), _frame(0.1, ttc=1.4), _frame(0.2, ttc=6.0)]
    assert action_index(silent) == 1

    assert action_index([_frame(0.0), _frame(0.1)]) == 0


def _seeded_window(tmp_path, qapp):
    """A ReviewWindow holding one decoded clip, with no background work pending."""
    store = ClipStore(root=tmp_path)
    clip = _build_replayable_clip()
    path = store.write(clip)
    win = ReviewWindow(store)
    win._on_scanned([(i, store.peek_metadata(i.path)) for i in store.list_clips()])

    frames = replay_clip(clip)
    win._path = str(path)
    win._frames = frames
    win._proposal = recorded_band(frames)
    win._idx = 0
    win._strip.set_frames(frames, frames[-1].t_rel)
    win._load_label_into_form(clip)
    win._refresh()
    return win, store, path


def _press(win, key, mods=Qt.NoModifier):
    win.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, mods))


def test_keys_set_class_and_severity(tmp_path, qapp):
    win, _store, _path = _seeded_window(tmp_path, qapp)
    try:
        _press(win, Qt.Key_4)
        assert win._class.currentText() == "fn"
        _press(win, Qt.Key_1)
        assert win._class.currentText() == "tp"

        win._severity.setValue(3)
        _press(win, Qt.Key_PageUp)
        _press(win, Qt.Key_PageUp)
        assert win._severity.value() == 5
        _press(win, Qt.Key_PageUp)      # clamps at the top of the range
        assert win._severity.value() == 5
        for _ in range(6):
            _press(win, Qt.Key_PageDown)
        assert win._severity.value() == 1
    finally:
        win.close()


def test_transport_keys_move_the_cursor(tmp_path, qapp):
    win, _store, _path = _seeded_window(tmp_path, qapp)
    try:
        _press(win, Qt.Key_End)
        assert win._idx == len(win._frames) - 1
        _press(win, Qt.Key_Left)
        assert win._idx == len(win._frames) - 2
        _press(win, Qt.Key_Home)
        assert win._idx == 0
        _press(win, Qt.Key_Right, Qt.ShiftModifier)   # coarse step past the end clamps
        assert win._idx == len(win._frames) - 1
    finally:
        win.close()


def test_window_keys_set_clear_and_accept_the_proposal(tmp_path, qapp):
    win, _store, _path = _seeded_window(tmp_path, qapp)
    try:
        assert win._proposal is not None

        win._idx = 1
        _press(win, Qt.Key_BracketLeft)
        win._idx = 4
        _press(win, Qt.Key_BracketRight)
        assert win._window == (win._frames[1].t_rel, win._frames[4].t_rel)

        _press(win, Qt.Key_Backslash)
        assert win._window is None

        # The proposal is never applied on load, only by W.
        _press(win, Qt.Key_W)
        assert win._window == win._proposal
    finally:
        win.close()


def test_proposal_is_not_written_unless_accepted(tmp_path, qapp):
    win, store, path = _seeded_window(tmp_path, qapp)
    try:
        assert win._window is None          # untagged clip opens with no window
        win._class.setCurrentText("fp")
        assert win._save() is True
        assert store.peek_metadata(path).label.should_trigger is None

        _press(win, Qt.Key_W)
        assert win._save() is True
        saved = store.peek_metadata(path).label.should_trigger
        assert saved is not None
        assert saved["from_t"] == pytest.approx(win._proposal[0], abs=1e-3)
        assert saved["to_t"] == pytest.approx(win._proposal[1], abs=1e-3)
    finally:
        win.close()


def _spin_until(qapp, pred, timeout_s: float = 20.0) -> bool:
    """Pump the event loop until *pred* holds, so worker results get delivered."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if pred():
            return True
    return False


def test_selecting_a_clip_decodes_off_thread_and_prefetches_the_next(tmp_path, qapp):
    store = ClipStore(root=tmp_path)
    for n in range(3):
        clip = _build_replayable_clip()
        # The filename keeps only the first 8 characters of the id, so vary those.
        clip.metadata.clip_id = f"prefetc{n}"
        store.write(clip)

    win = ReviewWindow(store)
    try:
        win._on_scanned([(i, store.peek_metadata(i.path)) for i in store.list_clips()])
        win._list.setCurrentRow(0)
        assert _spin_until(qapp, lambda: bool(win._frames)), "clip never decoded"

        first = win._visible[0]
        assert win._path == first
        assert first in win._cache
        assert win._proposal is not None

        # Selecting row 0 queues the following rows, so the next pick is a cache hit.
        assert _spin_until(qapp, lambda: win._visible[1] in win._cache), "no prefetch"
        assert len(win._cache) <= 4
    finally:
        win.close()


def test_save_advances_to_the_next_untagged_clip(tmp_path, qapp):
    store = ClipStore(root=tmp_path)
    paths = []
    for n in range(3):
        clip = _build_replayable_clip()
        clip.metadata.clip_id = f"advance{n}"   # the filename is built from the id
        paths.append(store.write(clip))
    store.write_label(paths[1], Label(class_="tn", severity=1))

    win = ReviewWindow(store)
    try:
        win._on_scanned([(i, store.peek_metadata(i.path)) for i in store.list_clips()])
        order = list(win._visible)
        assert len(order) == 3

        win._path = order[0]
        win._save_and_advance()
        # The already-tagged clip is skipped, whichever order the store listed them in.
        expected = next(p for p in order[1:] if store.peek_metadata(p).label is None)
        assert win._path == expected
    finally:
        win.close()
