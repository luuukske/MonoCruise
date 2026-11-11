"""CC panel close-follow warning: orange vehicle in the closest slot, lines keep the wanted gap."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage, QPainter, QRegion
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


def _box(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def test_no_warning_looks_like_today(panel):
    assert _pixels(_render(panel), _orange) == []


@pytest.mark.parametrize("truck", [False, True])
def test_the_vehicle_turns_orange_and_the_lines_keep_their_colour(panel, truck):
    img = _render(panel, close_warn=True, acc_truck=truck)
    assert len(_pixels(img, _orange)) > 100
    bottom = _box(_pixels(img, _orange))[3]
    assert [p for p in _pixels(img, _blue) if p[1] > bottom], "the wanted-gap lines stay in the mode colour"


def _icon_blue(img: QImage, x0: int) -> list[tuple[int, int]]:
    return [p for p in _pixels(img, _blue) if p[0] >= x0]


def test_the_vehicle_takes_the_closest_slot_at_every_wanted_gap(panel):
    closest = _box(_pixels(_render(panel, distance_to_lead=1, close_warn=True), _orange))
    for gap in (2, 3, 4):
        assert _box(_pixels(_render(panel, distance_to_lead=gap, close_warn=True), _orange)) == closest
    gap_one = _box([p for p in _icon_blue(_render(panel, distance_to_lead=1), closest[0] - 10) if p[1] <= closest[3]])
    # One pixel of slack: the two colour tests cut the antialiased edges differently.
    assert all(abs(a - b) <= 1 for a, b in zip(closest, gap_one)), "the slot where gap 1 draws its vehicle"


def test_the_lines_still_show_the_wanted_gap(panel):
    x0 = _box(_pixels(_render(panel, close_warn=True), _orange))[0] - 10

    def lit(gap: int) -> int:
        return len(_icon_blue(_render(panel, distance_to_lead=gap, close_warn=True), x0))

    assert lit(4) > lit(3) > lit(2) > lit(1)


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
