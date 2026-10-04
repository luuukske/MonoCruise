"""Main-window layout: the body owns everything under the banner."""
from __future__ import annotations

import pytest

from PySide6.QtCore import QSize
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QApplication

from core.settings import Settings


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    from ui.main_window.window import MonoCruiseWindow

    w = MonoCruiseWindow(Settings.instance(), version="v0.0.0-test")
    w._poll_timer.stop()
    w._overlay_keep_timer.stop()
    yield w
    if w._cc_panel is not None:
        w._cc_panel.stop()
        w._cc_panel = None
    w.deleteLater()


def test_the_body_runs_down_to_the_bottom_margin(window):
    central = window.centralWidget()
    layout = central.layout()
    central.resize(900, 700)
    layout.activate()

    body = layout.itemAt(layout.count() - 1).widget()
    assert body.geometry().bottom() + 1 + layout.contentsMargins().bottom() == central.height()


def test_the_version_label_sits_in_the_bottom_right_corner(window):
    central = window.centralWidget()
    central.resize(900, 700)
    window.resizeEvent(QResizeEvent(QSize(900, 700), QSize(700, 500)))

    geo = window._version_label.geometry()
    assert central.rect().contains(geo)
    assert central.width() - geo.right() - 1 == 8
    assert central.height() - geo.bottom() - 1 == 4
