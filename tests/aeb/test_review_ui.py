"""Tests for the labelling UI: window proposal, landing frame, and the key bindings."""

from __future__ import annotations

import base64
import time

import pytest

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QKeyEvent, QResizeEvent
from PySide6.QtWidgets import QApplication

from core.aeb.clip_replay import ReviewFrame, replay_clip
from core.aeb.clip_schema import Label, LiveAEB
from core.aeb.clip_store import ClipStore
from tools import aeb_fetch
from tools.aeb_fetch import PullResult
from tools.aeb_review import ReviewWindow
from tools.aeb_review_widgets import ThumbnailView, action_index, recorded_band

from tests.aeb.test_clip_review import _build_replayable_clip
from tests.aeb.test_fetch_tool import _FakeSession, _clip_blob


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


def _thumbnail_b64(size: tuple[int, int]) -> str:
    Image = pytest.importorskip("PIL.Image")
    from core.aeb.screenshot import encode_thumbnail

    img = Image.new("RGB", size, (30, 60, 90))
    return encode_thumbnail(img, max_px=max(size), quality=70)


@pytest.mark.parametrize("size", [(480, 270), (240, 135)])
def test_thumbnail_scales_aspect_correct_and_fits_the_panel(qapp, size):
    view = ThumbnailView()
    view.set_jpeg(_thumbnail_b64(size))
    pix = view.pixmap()
    assert pix is not None and not pix.isNull()
    assert pix.width() <= 320                     # never wider than the right panel
    src_w, src_h = size
    assert abs((pix.width() / pix.height()) - (src_w / src_h)) < 0.01
    assert view.source_size() == size


def test_thumbnail_rescales_from_the_cached_original(qapp):
    """Every rescale starts from the decoded original, never from a previously scaled copy."""
    view = ThumbnailView()
    view.set_jpeg(_thumbnail_b64((240, 135)))
    assert view.source_size() == (240, 135)

    # An unshown widget does not get resizeEvent delivered, so drive it directly.
    view.resize(150, 90)
    view.resizeEvent(QResizeEvent(QSize(150, 90), QSize(0, 0)))
    small = view.pixmap().width()

    view.resize(300, 180)
    view.resizeEvent(QResizeEvent(QSize(300, 180), QSize(150, 90)))
    large = view.pixmap().width()

    assert large > small
    # The tell for re-scaling a scaled copy: the source would have shrunk with it.
    assert view.source_size() == (240, 135)


def test_no_screenshot_state_still_works(qapp):
    view = ThumbnailView()
    view.set_jpeg(_thumbnail_b64((240, 135)))
    view.set_jpeg(None)
    assert view.text() == "(no screenshot)"
    assert view.source_size() is None


def test_thumbnail_decode_failure_state_still_works(qapp):
    view = ThumbnailView()
    view.set_jpeg(base64.b64encode(b"not a jpeg").decode("ascii"))
    assert view.text() == "(thumbnail decode failed)"
    assert view.source_size() is None


def test_show_appends_thumbnail_resolution_to_the_metadata_line(tmp_path, qapp):
    store = ClipStore(root=tmp_path)
    clip = _build_replayable_clip()
    clip.metadata.thumbnail_jpeg = _thumbnail_b64((240, 135))
    path = store.write(clip)

    win = ReviewWindow(store)
    try:
        win._on_scanned([(i, store.peek_metadata(i.path)) for i in store.list_clips()])
        win._list.setCurrentRow(0)
        assert _spin_until(qapp, lambda: bool(win._frames)), "clip never decoded"
        assert "240x135" in win._clip_meta_lbl.text()
    finally:
        win.close()


def test_update_from_server_button_pulls_and_refreshes(tmp_path, qapp, monkeypatch):
    """The button runs the pull off-thread, then rescans when clips landed."""
    monkeypatch.setenv("MONOCRUISE_PULL_TOKEN", "a-token")
    clip_id = "aaaaaaaa-1111-4111-8111-111111111111"
    rows = [{"clip_id": clip_id, "received_at": "2026-08-09T10:00:00Z",
             "trigger_source": "auto_engagement", "session_kind": "SP",
             "bytes": 100, "client_version": "1.1.0"}]
    session = _FakeSession(rows, {clip_id: _clip_blob(clip_id)})
    monkeypatch.setattr(aeb_fetch, "_session", lambda token: session)

    store = ClipStore(root=tmp_path)
    win = ReviewWindow(store)
    try:
        win._update_btn.click()
        assert _spin_until(qapp, lambda: not win._pulling), "pull never finished"
        assert len(store.list_clips()) == 1
        assert "saved 1" in win._status.text()
        assert win._update_btn.isEnabled()
        assert _spin_until(qapp, lambda: len(win._entries) == 1), "list never refreshed"
    finally:
        win.close()


def test_update_from_server_reports_a_missing_token(tmp_path, qapp, monkeypatch):
    monkeypatch.delenv("MONOCRUISE_PULL_TOKEN", raising=False)
    store = ClipStore(root=tmp_path)
    win = ReviewWindow(store)
    try:
        win._update_btn.click()
        assert _spin_until(qapp, lambda: not win._pulling), "pull never finished"
        assert "MONOCRUISE_PULL_TOKEN" in win._status.text()
        assert win._update_btn.isEnabled()
        assert store.list_clips() == []
    finally:
        win.close()


def test_update_from_server_finished_handler_summarises(tmp_path, qapp):
    store = ClipStore(root=tmp_path)
    win = ReviewWindow(store)
    try:
        win._on_pull_finished(PullResult(
            listed=3, already=2, saved=0, failed=0, landed=0, root=str(tmp_path),
        ))
        assert "up to date" in win._status.text()
    finally:
        win.close()
