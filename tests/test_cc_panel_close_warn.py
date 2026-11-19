"""CC panel close-follow warning: vehicle in the closest slot over the wanted-gap lines, an orange triangle."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPixmap, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from core.lane_departure.display import WARN_COLOR


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    from ui.cc_panel.main import cc_panel

    p = cc_panel("80 km/h", "Cruise control", True, scale_mult=1.0)
    yield p
    p.stop()


def _render(p, **fields) -> QImage:
    state = dict(
        text_content="80 km/h", cc_mode="Cruise control", cc_enabled=True, acc_enabled=True, acc_locked=True,
        acc_truck=False, distance_to_lead=3, AEB_warn=False, lead_vehicle_speed=None,
        lane_state="off", lane_side=0, close_warn=False, _complete_update=True,
    )
    state.update(fields)
    p._widget._on_update(state)
    p._widget._lane_timer.stop()
    p._vehicle_anim_layout_key = ()
    img = QImage(p._panel_w, p._panel_h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#000000"))
    pa = QPainter(img)
    p._widget.render(pa, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    pa.end()
    p._widget._vehicle_anim_timer.stop()
    return img


def _pixels(img: QImage, test) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(img.height())
        for x in range(img.width())
        if test(QColor(img.pixel(x, y)))
    ]


def _orange(c: QColor) -> bool:
    return c.name() == WARN_COLOR.lower()


def _blue(c: QColor) -> bool:
    return c.blue() > 200 and c.red() < 120


def test_no_warning_looks_like_today(panel):
    assert _pixels(_render(panel), _orange) == []


def _white(c: QColor) -> bool:
    return min(c.red(), c.green(), c.blue()) > 200


def _icon_x(p) -> int:
    icon_sz = int(QFontMetrics(p._font).height() * 1.8 * 0.85)
    return p._panel_w - icon_sz - int(20 * p._scale_mult)


def _icon(p, img: QImage, test) -> list[tuple[int, int]]:
    """Pixels right of the set speed: the vehicle, the distance lines and the triangle."""
    return [q for q in _pixels(img, test) if q[0] >= _icon_x(p)]


def _lit_lines(p, img: QImage) -> int:
    """Runs of rows holding lit lines; the faint unlit ones are not blue enough to count."""
    rows = sorted({y for _, y in _icon(p, img, _blue)})
    return sum(1 for i, y in enumerate(rows) if i == 0 or y != rows[i - 1] + 1)


def _vehicle_slot(p) -> tuple[float, float]:
    return p._vehicle_anim_target_y, p._vehicle_anim_target_sh


@pytest.fixture
def no_triangle(panel, monkeypatch):
    monkeypatch.setattr(panel._widget, "_paint_warn_triangle", lambda *a, **k: None)
    return panel


@pytest.fixture
def lines_only(no_triangle, monkeypatch):
    """The vehicle drawn fully transparent: nothing drawn, nothing cut out of the lines."""

    def clear(pm: QPixmap, _color: QColor) -> QPixmap:
        out = QPixmap(pm.size())
        out.setDevicePixelRatio(pm.devicePixelRatio())
        out.fill(Qt.GlobalColor.transparent)
        return out

    monkeypatch.setattr(no_triangle._widget, "_tinted", clear)
    return no_triangle


@pytest.mark.parametrize("truck", [False, True])
def test_the_vehicle_keeps_its_colour(no_triangle, truck):
    # Lukas, 2026-11-19: an orange vehicle read as a big error; the triangle alone says "watch out, you're close".
    assert _pixels(_render(no_triangle, close_warn=True, acc_truck=truck), _orange) == []


@pytest.mark.parametrize("truck", [False, True])
def test_the_vehicle_takes_the_closest_slot_at_every_wanted_gap(panel, truck):
    _render(panel, distance_to_lead=1, acc_truck=truck)
    closest = _vehicle_slot(panel)
    for gap in (1, 2, 3, 4):
        _render(panel, distance_to_lead=gap, close_warn=True, acc_truck=truck)
        assert _vehicle_slot(panel) == closest, "the slot where gap 1 draws its vehicle"


@pytest.mark.parametrize("truck", [False, True])
def test_the_lines_keep_showing_the_wanted_gap_under_the_vehicle(lines_only, truck):
    p = lines_only
    for gap in (1, 2, 3, 4):
        for warn in (False, True):
            # A fresh lines layer per render: the cache would hand the warning the plain render's lines.
            p._widget._acc_lines_layer_cache.clear()
            assert _lit_lines(p, _render(p, distance_to_lead=gap, close_warn=warn, acc_truck=truck)) == gap


def test_the_vehicle_jumps_in_and_out_and_a_gap_change_leaves_it_alone(panel):
    w = panel._widget
    _render(panel, distance_to_lead=4)
    far = _vehicle_slot(panel)
    w._on_update({"close_warn": True})
    w.grab()
    closest = _vehicle_slot(panel)
    assert closest != far and not w._vehicle_anim_timer.isActive(), "jumps to the front, no glide"
    for gap in (1, 3, 2):
        w._on_update({"distance_to_lead": gap})
        w.grab()
        assert _vehicle_slot(panel) == closest and not w._vehicle_anim_timer.isActive()
    w._on_update({"close_warn": False})
    w.grab()
    assert _vehicle_slot(panel) != closest and not w._vehicle_anim_timer.isActive(), "jumps back to the set gap"
    w._on_update({"distance_to_lead": 4})
    w.grab()
    assert w._vehicle_anim_timer.isActive(), "without the warning a gap change still glides"
    w._vehicle_anim_timer.stop()


@pytest.mark.parametrize("truck", [False, True])
@pytest.mark.parametrize("cc_enabled, line_test", [(True, _blue), (False, _white)])
def test_an_orange_triangle_stands_on_the_closest_line_clear_of_it(panel, truck, cc_enabled, line_test):
    img = _render(panel, close_warn=True, acc_truck=truck, cc_enabled=cc_enabled)
    line_bottom = max(y for _, y in _icon(panel, img, line_test))
    y_mid = line_bottom - int(2 * panel._scale_mult)
    row = [(x, QColor(img.pixel(x, y_mid))) for x in range(img.width())]
    orange = [x for x, c in row if _orange(c)]
    line = [x for x, c in row if line_test(c)]
    assert orange, "the triangle crosses the line"
    left, right = max(x for x in line if x < orange[0]), min(x for x in line if x > orange[-1])
    assert orange[0] - left > 3 and right - orange[-1] > 3, "a clear ring between triangle and line"
    tri_bottom = max(y for x, y in _pixels(img, _orange) if orange[0] <= x <= orange[-1])
    assert tri_bottom >= line_bottom, "it stands on the line, not above it"


def test_it_needs_a_tracked_vehicle_while_the_lines_show(panel):
    assert _pixels(_render(panel, close_warn=True, acc_locked=False), _orange) == []


@pytest.mark.parametrize("fields", [dict(acc_enabled=False), dict(cc_mode="Speed limiter")])
def test_it_shows_only_with_the_acc_distance_lines(panel, fields):
    assert _pixels(_render(panel, close_warn=True, **fields), _orange) == []


def test_aeb_outranks_it(panel):
    assert _pixels(_render(panel, close_warn=True, AEB_warn=True), _orange) == []


def test_it_builds_no_cutout_masks_of_its_own(panel):
    _render(panel)
    masks = dict(panel._widget._vehicle_cutout_mask_cache)
    assert masks
    panel._widget._on_update({"close_warn": True})
    panel._widget.grab()
    panel._widget._on_update({"close_warn": False})
    panel._widget.grab()
    assert panel._widget._vehicle_cutout_mask_cache == masks
