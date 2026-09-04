"""Always-on-top overlays must not fight z-order or restore a minimised window."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.sending_thread.visualization_bar import VisualizationBar
from ui.cc_panel.main import cc_panel
from ui.popup.popup_window import PopupWindow

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
