"""Always-on-top overlays must not fight z-order or restore a minimised window."""
from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.sending_thread.visualization_bar import VisualizationBar
from core.settings import Settings
from ui.cc_panel.main import cc_panel
from ui.main_window.window import MonoCruiseWindow
from ui.overlay_topmost import OVERLAY_KEEP_MS, reassert_topmost
from ui.popup.popup_window import PopupWindow, State
import ui.overlay_topmost as topmost

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fn(tree: ast.Module, class_name: str, fn_name: str) -> ast.FunctionDef:
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                    return node
    raise AssertionError(f"{class_name}.{fn_name} not found")


def _calls_raise(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "raise_"
        ):
            return True
    return False


def test_visualization_bar_hot_path_does_not_raise():
    path = REPO / "core" / "sending_thread" / "visualization_bar.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    assert not _calls_raise(_fn(tree, "VisualizationBar", "_animate_inner"))
    assert not _calls_raise(_fn(tree, "VisualizationBar", "_animate"))


def test_cc_panel_does_not_raise_on_update_or_show():
    path = REPO / "ui" / "cc_panel" / "main.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    assert not _calls_raise(_fn(tree, "_PanelWidget", "_on_update"))
    assert not _calls_raise(_fn(tree, "_PanelWidget", "_on_show"))


def test_cc_panel_show_is_gated_on_visibility():
    path = REPO / "ui" / "main_window" / "window.py"
    src = path.read_text(encoding="utf-8-sig")
    assert "if not self._cc_panel.is_visible():" in src
    assert "self._cc_panel.show()" in src


def test_visualization_bar_does_not_activate(qapp):
    bar = VisualizationBar()
    try:
        assert bar.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        flags = bar.windowFlags()
        assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
        assert flags & Qt.WindowType.WindowStaysOnTopHint
        assert flags & Qt.WindowType.Tool
    finally:
        bar.timer.stop()
        bar.close()
        bar.deleteLater()
        qapp.processEvents()


def test_cc_panel_does_not_activate_on_show(qapp):
    panel = cc_panel("-- km/h", scale_mult=0.5)
    try:
        w = panel._widget
        assert w.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        assert w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        assert w.windowFlags() & Qt.WindowType.Tool
        assert not panel.is_visible()
        panel.show()
        qapp.processEvents()
        assert panel.is_visible()
    finally:
        panel.stop()
        qapp.processEvents()


def test_overlay_keep_interval_is_five_seconds():
    assert OVERLAY_KEEP_MS == 5000
    src = (REPO / "ui" / "main_window" / "window.py").read_text(encoding="utf-8-sig")
    assert "OVERLAY_KEEP_MS" in src
    assert "_reassert_overlays" in src


def test_animate_path_does_not_reassert_topmost():
    path = REPO / "core" / "sending_thread" / "visualization_bar.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    body = _fn(tree, "VisualizationBar", "_animate_inner")
    for node in ast.walk(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"ensure_present", "reassert_topmost", "raise_"}


def test_reassert_topmost_uses_set_window_pos_without_activating(qapp, monkeypatch):
    bar = VisualizationBar()
    calls: list[tuple[int, int, int]] = []

    class _User32:
        def SetWindowPos(self, hwnd, insert_after, x, y, cx, cy, flags):
            calls.append((int(hwnd), int(insert_after), int(flags)))
            return 1

    try:
        monkeypatch.setattr(topmost.sys, "platform", "win32")
        monkeypatch.setattr(topmost, "_load_user32", lambda: _User32())
        assert reassert_topmost(bar) is True
        assert len(calls) == 1
        _hwnd, insert_after, flags = calls[0]
        assert insert_after == -1
        assert flags & 0x0010  # SWP_NOACTIVATE
        assert flags & 0x0040  # SWP_SHOWWINDOW
        assert flags & 0x0001  # SWP_NOSIZE
        assert flags & 0x0002  # SWP_NOMOVE
        assert flags & 0x0200  # SWP_NOOWNERZORDER
        bar.hide()
        qapp.processEvents()
        assert reassert_topmost(bar) is False
        assert len(calls) == 1
    finally:
        bar.timer.stop()
        bar.close()
        bar.deleteLater()
        qapp.processEvents()


def test_reassert_topmost_is_a_no_op_off_windows(qapp, monkeypatch):
    bar = VisualizationBar()
    try:
        monkeypatch.setattr(topmost.sys, "platform", "linux")
        assert reassert_topmost(bar) is False
    finally:
        bar.timer.stop()
        bar.close()
        bar.deleteLater()
        qapp.processEvents()


def test_enabled_bar_comes_back_after_its_timer_stops(qapp):
    bar = VisualizationBar()
    settings = Settings.instance()
    previous = settings.bar_variable
    try:
        settings.bar_variable = True
        bar.timer.stop()
        bar.hide()
        qapp.processEvents()
        assert not bar.isVisible()
        assert not bar.timer.isActive()
        bar.ensure_present()
        qapp.processEvents()
        assert bar.isVisible()
        assert bar.timer.isActive()
    finally:
        settings.bar_variable = previous
        bar.timer.stop()
        bar.close()
        bar.deleteLater()
        qapp.processEvents()


def test_disabled_bar_stays_closed(qapp):
    bar = VisualizationBar()
    settings = Settings.instance()
    previous = settings.bar_variable
    try:
        settings.bar_variable = False
        bar.timer.stop()
        bar.hide()
        qapp.processEvents()
        bar.ensure_present()
        qapp.processEvents()
        assert not bar.isVisible()
        assert not bar.timer.isActive()
    finally:
        settings.bar_variable = previous
        bar.timer.stop()
        bar.close()
        bar.deleteLater()
        qapp.processEvents()


def test_cruise_panel_reassert_follows_the_setting():
    panel = SimpleNamespace(visible=True, shown=0, reasserted=0)

    def is_visible() -> bool:
        return panel.visible

    def show() -> None:
        panel.shown += 1
        panel.visible = True

    def reassert() -> None:
        panel.reasserted += 1

    panel.is_visible = is_visible
    panel.show = show
    panel.reassert_topmost = reassert
    host = SimpleNamespace(
        _cc_panel=panel,
        _settings=SimpleNamespace(show_cc_ui=False, _state_lock=threading.RLock()),
        _closing=False,
    )
    MonoCruiseWindow._reassert_cc_panel(host)
    assert panel.shown == 0
    assert panel.reasserted == 0

    host._settings.show_cc_ui = True
    panel.visible = False
    MonoCruiseWindow._reassert_cc_panel(host)
    assert panel.shown == 1
    assert panel.reasserted == 1


def test_popup_reassert_skips_idle(qapp, monkeypatch):
    popup = PopupWindow()
    calls: list[object] = []
    monkeypatch.setattr(topmost, "reassert_topmost", lambda w: calls.append(w) or True)
    try:
        popup.show()
        qapp.processEvents()
        popup.reassert_if_showing()
        assert calls == []
        popup._state = State.DISPLAYING
        popup.reassert_if_showing()
        assert calls == [popup]
    finally:
        popup.close()
        popup.deleteLater()
        PopupWindow._instance = None
        qapp.processEvents()


def test_popup_window_does_not_activate(qapp):
    popup = PopupWindow()
    try:
        assert popup.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        flags = popup.windowFlags()
        assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
        assert flags & Qt.WindowType.Tool
    finally:
        popup.close()
        popup.deleteLater()
        PopupWindow._instance = None
        qapp.processEvents()
