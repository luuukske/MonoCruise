"""Map overlay for the AEB clip review window. Not shipped.

The Map checkbox matches the clip against this machine's game map and replays
the arcs the way AEB would, so the scene can draw lane edges and the road path.
"""
from __future__ import annotations

import logging
from collections import OrderedDict

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from core.aeb.clip_replay import replay_clip

logger = logging.getLogger(__name__)

_MAP_CACHE_MAX = 4


class MapReplayWorker(QObject):
    """Lane match plus the road-bent replay. Runs off the GUI thread."""

    mapped = Signal(str, object)  # path, None | ("nomatch",) | (game, frames, window)

    @Slot(str, object, object)
    def replay(self, path: str, clip, stream) -> None:
        from tools.map_replay import ClipMap

        try:
            clip_map = ClipMap(clip)
        except Exception:
            logger.exception("map match failed")
            self.mapped.emit(path, None)
            return
        if clip_map.window is None:
            self.mapped.emit(path, ("nomatch",))
            return
        try:
            frames = replay_clip(clip, stream=stream, map_provider=clip_map)
        except Exception:
            logger.exception("map replay failed")
            self.mapped.emit(path, None)
            return
        self.mapped.emit(path, (clip_map.game, frames, clip_map.window))


class _MapBridge(QObject):
    """GUI-thread landing for the worker. A plain function would run on the worker."""

    def __init__(self, win) -> None:
        super().__init__(win)
        self._win = win

    @Slot(str, object)
    def mapped(self, path: str, result) -> None:
        on_mapped(self._win, path, result)


def map_toggle(win) -> QWidget:
    """Checkbox plus status, and the worker thread behind them."""
    win._map_on = False
    win._map_cache = OrderedDict()
    win._map_inflight = None
    win._map_wanted = None
    win._note_map = lambda: note_shown(win)
    win._stop_map = lambda: stop_map(win)
    win._kick_map = lambda path: kick_map(win, path)
    win._on_mapped = lambda path, result: on_mapped(win, path, result)

    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    chk = QCheckBox("Map")
    chk.setChecked(False)
    chk.setFocusPolicy(Qt.NoFocus)
    chk.toggled.connect(lambda on: toggle_map(win, on))
    lbl = QLabel("")
    lbl.setStyleSheet("color:#999;")
    row.addWidget(chk)
    row.addWidget(lbl)
    win._map_chk = chk
    win._map_lbl = lbl

    win._map_thread = QThread(win)
    win._mapper = MapReplayWorker()
    win._mapper.moveToThread(win._map_thread)
    win._map_bridge = _MapBridge(win)
    win.map_requested.connect(win._mapper.replay)
    win._mapper.mapped.connect(win._map_bridge.mapped)
    win._map_thread.start()
    return box


def stop_map(win) -> None:
    thread = getattr(win, "_map_thread", None)
    if thread is None:
        return
    thread.quit()
    thread.wait(2000)


def note_shown(win) -> None:
    if win._map_on:
        ensure_map(win)


def toggle_map(win, on: bool) -> None:
    win._map_on = bool(on)
    if not win._map_on:
        win._scene.map_window = None
        win._map_lbl.setText("")
        if win._loaded is not None:
            swap_frames(win, win._loaded.frames)
        return
    ensure_map(win)


def ensure_map(win) -> None:
    path = win._path
    if path is None or win._loaded is None:
        return
    win._map_wanted = path
    if path in win._map_cache:
        apply_cached(win, path)
        return
    win._map_lbl.setText("matching...")
    win._kick_map(path)


def kick_map(win, path: str) -> None:
    if win._map_inflight is not None:
        return
    loaded = win._cache.get(path)
    if loaded is None and win._path == path:
        loaded = win._loaded
    if loaded is None:
        return
    win._map_inflight = path
    win.map_requested.emit(path, loaded.clip, loaded.stream)


def swap_frames(win, frames: list) -> None:
    t = win._frames[win._idx].t_rel if win._frames else 0.0
    win._frames = frames
    if frames:
        win._idx = min(range(len(frames)), key=lambda i: abs(frames[i].t_rel - t))
    win._refresh()


def apply_cached(win, path: str) -> None:
    hit = win._map_cache[path]
    win._map_cache.move_to_end(path)
    if not hit or hit[0] == "nomatch":
        win._scene.map_window = None
        win._map_lbl.setText("map failed" if hit is None else "no map match")
        if win._loaded is not None:
            swap_frames(win, win._loaded.frames)
        return
    game, frames, window = hit
    if not frames:
        win._scene.map_window = None
        win._map_lbl.setText("no map frames")
        return
    win._scene.map_window = window
    win._map_lbl.setText(game or "map")
    swap_frames(win, frames)


def _trim(win) -> None:
    while len(win._map_cache) > _MAP_CACHE_MAX:
        for key in list(win._map_cache):
            if key != win._path:
                del win._map_cache[key]
                break
        else:
            break


def on_mapped(win, path: str, result) -> None:
    if win._map_inflight == path:
        win._map_inflight = None
    win._map_cache[path] = result
    _trim(win)
    wanted = win._map_wanted
    if win._map_on and wanted and wanted != path and wanted not in win._map_cache:
        win._kick_map(wanted)
    if win._map_on and path == win._path:
        apply_cached(win, path)
