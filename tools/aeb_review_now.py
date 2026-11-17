"""Current-code view for the AEB clip review window. Not shipped.

The scene and the decision strip normally show what the build that recorded the
clip decided. The Now checkbox re-runs this working tree's AEB over the clip, on
the local game map while Map is on, and shows that instead. See core/aeb/README.md
section 11, "Current decisions".
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import replace

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from core.aeb.clip_replay import ReviewFrame
from core.aeb.clip_schema import LiveAEB

logger = logging.getLogger(__name__)

_NOW_CACHE_MAX = 6


def _reason_names(reasons: dict) -> dict[str, list[str]]:
    """Stage names per vehicle, the way a clip records them."""
    out: dict[str, list[str]] = {}
    for vid, results in (reasons or {}).items():
        names = [getattr(r, "reason", r) for r in results]
        names = [str(n) for n in names if n]
        if names:
            out[str(vid)] = names
    return out


def now_frames(recorded: list[ReviewFrame], ticks) -> list[ReviewFrame]:
    """Recorded frames with the re-run's snapshot and decision in place of the recorded ones.

    A frame the re-run has no tick for keeps its recorded content.
    """
    by_t = {tk.t_mono: tk for tk in ticks if getattr(tk, "snapshot", None) is not None}
    out: list[ReviewFrame] = []
    for f in recorded:
        tk = by_t.get(f.t_mono)
        if tk is None:
            out.append(f)
            continue
        snap = tk.snapshot
        live = LiveAEB(
            aeb_warn=tk.aeb_warn,
            aeb_brake=tk.aeb_brake,
            engaged=tk.engaged,
            target_decel_ms2=tk.target_decel_ms2,
            required_decel_ms2=tk.required_decel_ms2,
            effective_max_decel_ms2=f.live_aeb.effective_max_decel_ms2,
            time_to_brake=tk.time_to_brake,
            time_to_collision=tk.time_to_collision,
            colliding_ids=sorted(tk.colliding_ids),
            suppressed_ids=sorted(tk.suppressed_ids),
            braking_worsens_ids=sorted(tk.braking_worsens_ids),
            suppression_reasons=_reason_names(getattr(snap, "suppression_reasons", {})),
        )
        out.append(replace(f, snapshot=snap, live_aeb=live, raw_target_ms2=tk.target_decel_ms2))
    return out


def decision_track(frames: list[ReviewFrame]) -> list[tuple[float, int]]:
    return [(f.t_rel, 2 if f.live_aeb.aeb_brake else (1 if f.live_aeb.aeb_warn else 0))
            for f in frames]


class NowReplayWorker(QObject):
    """Working-tree AEB over one clip, with or without the map. Runs off the GUI thread."""

    # path, on the map or not, None | (status, frames)
    replayed = Signal(str, bool, object)

    @Slot(str, object, object, bool)
    def rerun(self, path: str, clip, recorded, use_map: bool) -> None:
        from core.aeb.clip_eval import run_headless
        from core.aeb.clip_replay import decode_radar_stream

        provider = None
        status = "now"
        if use_map:
            try:
                from tools.map_replay import ClipMap

                clip_map = ClipMap(clip)
            except Exception:
                logger.exception("map match failed")
                clip_map = None
            if clip_map is not None and clip_map.window is not None:
                provider = clip_map
                status = f"now, {clip_map.game or 'map'}"
            else:
                status = "now, no map match"
        try:
            # The review stream skips the elevation gate, which headless AEB reads.
            ticks = run_headless(clip, stream=decode_radar_stream(clip),
                                 map_provider=provider, keep_snapshots=True)
            frames = now_frames(recorded, ticks)
        except Exception:
            logger.exception("current-code replay failed")
            self.replayed.emit(path, use_map, None)
            return
        self.replayed.emit(path, use_map, (status, frames))


class _NowBridge(QObject):
    """GUI-thread landing for the worker."""

    def __init__(self, win) -> None:
        super().__init__(win)
        self._win = win

    @Slot(str, bool, object)
    def replayed(self, path: str, use_map: bool, result) -> None:
        on_replayed(self._win, path, use_map, result)


def now_toggle(win) -> QWidget:
    """Checkbox plus status, and the worker thread behind them."""
    win._now_on = False
    win._now_cache = OrderedDict()
    win._now_inflight = None
    win._decision_source = "recorded"
    win._note_now = lambda: note_shown(win)
    win._stop_now = lambda: stop_now(win)
    win._kick_now = lambda path, use_map: kick_now(win, path, use_map)
    win._on_now = lambda path, use_map, result: on_replayed(win, path, use_map, result)
    win._now_frames = lambda frames: overlay(win, frames)
    win._now_track = lambda: current_track(win)

    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    chk = QCheckBox("Now")
    chk.setToolTip("Show this working tree's AEB decisions instead of the recorded ones"
                   " (on the map while Map is on)")
    chk.setChecked(False)
    chk.setFocusPolicy(Qt.NoFocus)
    chk.toggled.connect(lambda on: toggle_now(win, on))
    lbl = QLabel("")
    lbl.setStyleSheet("color:#999;")
    row.addWidget(chk)
    row.addWidget(lbl)
    win._now_chk = chk
    win._now_lbl = lbl

    win._now_thread = QThread(win)
    win._now_worker = NowReplayWorker()
    win._now_worker.moveToThread(win._now_thread)
    win._now_bridge = _NowBridge(win)
    win.now_requested.connect(win._now_worker.rerun)
    win._now_worker.replayed.connect(win._now_bridge.replayed)
    win._now_thread.start()
    return box


def stop_now(win) -> None:
    thread = getattr(win, "_now_thread", None)
    if thread is None:
        return
    thread.quit()
    thread.wait(2000)


def _key(win, path: str) -> tuple[str, bool]:
    return path, bool(getattr(win, "_map_on", False))


def toggle_now(win, on: bool) -> None:
    win._now_on = bool(on)
    if not win._now_on:
        win._now_lbl.setText("")
    ensure_now(win)
    _reframe(win)


def ensure_now(win) -> None:
    """Ask for the re-run this clip and map setting need, unless it is cached or running."""
    path = win._path
    if not win._now_on or path is None or win._loaded is None:
        return
    key = _key(win, str(path))
    hit = win._now_cache.get(key)
    if hit is not None:
        win._now_cache.move_to_end(key)
        win._now_lbl.setText(hit[0] if hit else "now failed")
        return
    win._now_lbl.setText("running...")
    win._kick_now(*key)


def kick_now(win, path: str, use_map: bool) -> None:
    if win._now_inflight is not None:
        return
    loaded = win._cache.get(path)
    if loaded is None and str(win._path) == path:
        loaded = win._loaded
    if loaded is None:
        return
    win._now_inflight = (path, use_map)
    win.now_requested.emit(path, loaded.clip, loaded.frames, use_map)


def on_replayed(win, path: str, use_map: bool, result) -> None:
    if win._now_inflight == (path, use_map):
        win._now_inflight = None
    win._now_cache[(path, use_map)] = result
    while len(win._now_cache) > _NOW_CACHE_MAX:
        for key in list(win._now_cache):
            if key[0] != str(win._path):
                del win._now_cache[key]
                break
        else:
            break
    if win._now_on and win._path is not None:
        wanted = _key(win, str(win._path))
        if wanted not in win._now_cache:
            win._kick_now(*wanted)
        elif wanted == (path, use_map):
            win._now_lbl.setText(result[0] if result else "now failed")
            _reframe(win)


def _ready(win):
    """The re-run frames for the clip and map setting on screen, or None."""
    if not getattr(win, "_now_on", False) or win._path is None:
        return None
    hit = win._now_cache.get(_key(win, str(win._path)))
    return hit[1] if hit else None


def overlay(win, frames: list[ReviewFrame]) -> list[ReviewFrame]:
    ready = _ready(win)
    win._decision_source = "now" if ready is not None else "recorded"
    return ready if ready is not None else frames


def current_track(win) -> list[tuple[float, int]] | None:
    ready = _ready(win)
    return decision_track(ready) if ready is not None else None


def note_shown(win) -> None:
    """A clip was put on screen: with Now on, show or fetch its re-run."""
    if win._now_on:
        ensure_now(win)
        _reframe(win)


def _reframe(win) -> None:
    """Put the frames the current toggles ask for on screen, and the chart's now row with them."""
    if win._loaded is None:
        return
    win._refresh_frames()
    if win._charts is not None:
        track = current_track(win)
        win._charts.set_replayed(track if track is not None else win._loaded.evaluated)
