"""Bottom bar lane corners: shape, fade, and when the bar lights them."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from core.lane_departure import display as ll
from core.lane_departure.display import LEFT, RIGHT
from core.sending_thread import lane_corners as lc
from core.sending_thread.lane_corners import LaneCorner
from core.settings import Settings
from core.thread_management.registry import registry


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _render(corner: LaneCorner) -> QImage:
    img = QImage(corner.width(), corner.height(), QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    pa = QPainter(img)
    corner.render(pa, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    pa.end()
    return img


def _red_alpha(img: QImage, x: int, y: int) -> int:
    c = QColor.fromRgba(img.pixel(x, y))
    return c.alpha() if c.red() > c.green() + 100 or c.alpha() == 0 else -1


@pytest.mark.parametrize("side", [LEFT, RIGHT])
def test_a_corner_is_solid_near_the_corner_and_fades_along_both_legs(qapp, side):
    corner = LaneCorner(side)
    corner.place(0, 0, 1600, 800)
    corner.set_weights(0.0, 1.0)
    e = corner.extent
    assert e == round(800 * lc.EXTENT_OF_HEIGHT)
    img = _render(corner)
    edge = 1 if side == LEFT else e - 2

    def along_bottom(f):
        return int(f * (e - 1)) if side == LEFT else e - 1 - int(f * (e - 1))

    y_bottom = e - 2
    assert _red_alpha(img, along_bottom(0.0), y_bottom) == 255
    assert _red_alpha(img, along_bottom(lc.SOLID_SHARE * 0.9), y_bottom) == 255
    assert 0 < _red_alpha(img, along_bottom(0.7), y_bottom) < 255
    assert _red_alpha(img, along_bottom(0.995), y_bottom) < 20
    assert _red_alpha(img, edge, int((1.0 - lc.SOLID_SHARE * 0.9) * e)) == 255
    assert 0 < _red_alpha(img, edge, int(0.3 * e)) < 255
    assert _red_alpha(img, e // 2, e // 2) == 0
    assert _red_alpha(img, along_bottom(0.3), e - 1 - lc.THICKNESS_PX - 1) == 0
    corner.deleteLater()


@pytest.mark.parametrize("side", [LEFT, RIGHT])
def test_the_legs_do_not_overlap_while_fading(qapp, side):
    corner = LaneCorner(side)
    corner.place(0, 0, 1600, 800)
    corner.set_weights(0.0, 0.5)
    e, t = corner.extent, lc.THICKNESS_PX
    img = _render(corner)
    edge = 2 if side == LEFT else e - 3
    inward = t + 2 if side == LEFT else e - 3 - t
    in_corner = QColor.fromRgba(img.pixel(edge, e - 3)).alpha()
    on_bottom_leg = QColor.fromRgba(img.pixel(inward, e - 3)).alpha()
    on_side_leg = QColor.fromRgba(img.pixel(edge, e - 3 - t)).alpha()
    assert 100 < in_corner < 155
    assert in_corner == on_bottom_leg == on_side_leg
    corner.deleteLater()


def test_an_idle_corner_paints_nothing(qapp):
    corner = LaneCorner(LEFT)
    corner.place(0, 0, 1600, 800)
    img = _render(corner)
    assert all(QColor.fromRgba(img.pixel(x, corner.extent - 2)).alpha() == 0 for x in range(corner.extent))
    corner.deleteLater()


class _Threads:
    def __init__(self, monkeypatch):
        plugin = SimpleNamespace(present=True, hooked=True, dead=False)
        self.lane = SimpleNamespace(
            alive=True, data=SimpleNamespace(_lock=threading.Lock(), firing_side=0, firing_alert=False, plugin=plugin),
        )
        self.lane.is_alive = lambda: self.lane.alive
        self.aeb = SimpleNamespace(data=SimpleNamespace(_lock=threading.Lock(), AEB_cue=False))
        self.threads = {"lane_departure_thread": self.lane, "aeb_thread": self.aeb}
        monkeypatch.setattr(registry, "get_thread", self._get)
        monkeypatch.setattr(Settings.instance(), "bar_variable", True, raising=False)

    def _get(self, name):
        if name not in self.threads:
            raise KeyError(name)
        return self.threads[name]


@pytest.fixture
def bar(qapp, monkeypatch):
    from core.sending_thread.visualization_bar import VisualizationBar

    threads = _Threads(monkeypatch)
    b = VisualizationBar()
    b.timer.stop()
    b.threads = threads
    yield b
    b.timer.stop()
    b.close()
    b.deleteLater()
    qapp.processEvents()


def _weights(bar):
    return [c.warn + c.alert for c in bar.corners]


def test_the_bar_lights_the_corner_on_the_side_acted_on(bar):
    bar.threads.lane.data.firing_side = 1
    bar._animate_inner()
    assert _weights(bar) == [1.0, 0.0]
    bar.threads.lane.data.firing_side = 2
    bar._animate_inner()
    assert _weights(bar)[1] == 1.0


def test_the_corner_stays_lit_after_firing_stops(bar):
    bar.threads.lane.data.firing_side = 2
    bar._animate_inner()
    bar.threads.lane.data.firing_side = 0
    bar._animate_inner()
    assert _weights(bar) == [0.0, 1.0]


def test_aeb_clears_the_corner(bar):
    bar.threads.lane.data.firing_side = 1
    bar.threads.aeb.data.AEB_cue = True
    bar._animate_inner()
    assert _weights(bar) == [0.0, 0.0]


@pytest.mark.parametrize("broken", ["unhooked", "dead_plugin", "thread_down", "thread_missing"])
def test_no_corner_when_lane_departure_cannot_act(bar, broken):
    lane = bar.threads.lane
    lane.data.firing_side = 1
    if broken == "unhooked":
        lane.data.plugin.hooked = False
    elif broken == "dead_plugin":
        lane.data.plugin.dead = True
    elif broken == "thread_down":
        lane.alive = False
    else:
        del bar.threads.threads["lane_departure_thread"]
    bar._animate_inner()
    assert _weights(bar) == [0.0, 0.0]


def test_switching_the_bar_off_takes_the_corners_with_it(bar, monkeypatch):
    bar.threads.lane.data.firing_side = 1
    bar._animate_inner()
    monkeypatch.setattr(Settings.instance(), "bar_variable", False, raising=False)
    bar._animate_inner()
    assert _weights(bar) == [0.0, 0.0]
    assert not any(c.isVisible() for c in bar.corners)
    monkeypatch.setattr(Settings.instance(), "bar_variable", True, raising=False)
    bar.ensure_present()
    assert all(c.isVisible() for c in bar.corners)


def test_corners_sit_in_the_bottom_corners_of_the_bar_screen(bar):
    geo = QApplication.primaryScreen().geometry()
    left, right = bar.corners
    assert left.geometry().bottomLeft() == geo.bottomLeft()
    assert right.geometry().bottomRight() == geo.bottomRight()
    assert left.width() == round(geo.height() * lc.EXTENT_OF_HEIGHT)


def test_a_warning_is_orange_and_an_alert_red(qapp):
    corner = LaneCorner(LEFT)
    corner.place(0, 0, 1600, 800)
    y = corner.extent - 2
    corner.set_weights(1.0, 0.0)
    warn = QColor.fromRgba(_render(corner).pixel(1, y))
    assert warn.name() == lc.WARN_COLOR.lower()
    corner.set_weights(0.0, 1.0)
    alert = QColor.fromRgba(_render(corner).pixel(1, y))
    assert (alert.red(), alert.green(), alert.blue()) == (255, 0, 0)
    corner.deleteLater()


def test_the_bar_shows_an_alert_and_holds_it_like_a_warning(bar, monkeypatch):
    monkeypatch.setattr(ll, "BLINK_S", 10.0)
    data = bar.threads.lane.data
    data.firing_side, data.firing_alert = 2, True
    bar._animate_inner()
    assert [(c.warn, c.alert) for c in bar.corners] == [(0.0, 0.0), (0.0, 1.0)]
    data.firing_side, data.firing_alert = 0, False
    bar._animate_inner()
    assert bar.corners[1].alert == 1.0, "held 1 s, still an alert"
