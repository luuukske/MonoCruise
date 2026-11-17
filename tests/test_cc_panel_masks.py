"""CC panel vehicle cutout masks: built with Qt blits, the same pixels as the old per-pixel loops, and kept
across ACC lock and lead changes (each rebuild held the GUI thread 70 to 115 ms at 175 % scaling)."""
from __future__ import annotations

import os
import random

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QRegion
from PySide6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    from ui.cc_panel.main import cc_panel

    p = cc_panel("80 km/h", "Cruise control", True, scale_mult=0.5, acc_enabled=True)
    yield p
    p.stop()


def _old_dilate(src: QImage, radius: int) -> QImage:
    w, h = src.width(), src.height()
    tmp = QImage(w, h, QImage.Format.Format_ARGB32)
    out = QImage(w, h, QImage.Format.Format_ARGB32)
    tmp.fill(0)
    out.fill(0)
    for y in range(h):
        for x in range(w):
            amax = max(src.pixelColor(sx, y).alpha() for sx in range(max(0, x - radius), min(w, x + radius + 1)))
            if amax > 0:
                tmp.setPixelColor(x, y, QColor(255, 255, 255, amax))
    for y in range(h):
        for x in range(w):
            amax = max(tmp.pixelColor(x, sy).alpha() for sy in range(max(0, y - radius), min(h, y + radius + 1)))
            if amax > 0:
                out.setPixelColor(x, y, QColor(255, 255, 255, amax))
    return out


def _old_cutout(pm: QPixmap, r: int) -> QPixmap:
    img = pm.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    padded = QImage(w + 2 * r, h + 2 * r, QImage.Format.Format_ARGB32)
    padded.fill(0)
    for y in range(h):
        for x in range(w):
            padded.setPixelColor(x + r, y + r, img.pixelColor(x, y))
    return QPixmap.fromImage(_old_dilate(padded, r))


def _old_exact(pm: QPixmap) -> QPixmap:
    img = pm.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    out = QImage(img.width(), img.height(), QImage.Format.Format_ARGB32)
    out.fill(0)
    for y in range(img.height()):
        for x in range(img.width()):
            a = img.pixelColor(x, y).alpha()
            if a > 0:
                out.setPixelColor(x, y, QColor(255, 255, 255, a))
    return QPixmap.fromImage(out)


def _pixels(pm: QPixmap) -> tuple[int, int, bytes]:
    img = pm.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    return img.width(), img.height(), bytes(img.constBits())


def _sprites(qapp) -> list[QPixmap]:
    assets = os.path.join(os.path.dirname(__file__), "..", "ui", "cc_panel", "assets")
    out = []
    for name, size in (("car1.png", 23), ("truck1.png", 31)):
        pm = QPixmap(os.path.join(assets, name))
        out.append(pm.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    rng = random.Random(7)
    noisy = QImage(17, 11, QImage.Format.Format_ARGB32)
    for y in range(11):
        for x in range(17):
            noisy.setPixelColor(x, y, QColor(rng.randrange(256), 0, 0, rng.choice((0, 0, 1, 37, 128, 254, 255))))
    out.append(QPixmap.fromImage(noisy))
    return out


@pytest.mark.parametrize("radius", [1, 2, 4])
def test_cutout_mask_matches_the_pixel_loop(panel, qapp, radius):
    w = panel._widget
    for i, pm in enumerate(_sprites(qapp)):
        new = w._line_cutout_mask_from_vehicle_pixmap(pm, 1.0, ("t", i), radius)
        assert _pixels(new) == _pixels(_old_cutout(pm, radius))


def test_exact_mask_matches_the_pixel_loop(panel, qapp):
    w = panel._widget
    for i, pm in enumerate(_sprites(qapp)):
        assert _pixels(w._exact_vehicle_alpha_mask(pm, 1.0, ("t", i))) == _pixels(_old_exact(pm))


def _render(p, **fields) -> QImage:
    state = dict(
        text_content="80 km/h", cc_mode="Cruise control", cc_enabled=True, acc_enabled=True, acc_locked=True,
        acc_truck=False, distance_to_lead=3, AEB_warn=False, lead_vehicle_speed=None,
        lane_state="off", lane_side=0, close_warn=False,
    )
    state.update(fields)
    p._widget._on_update(state)
    p._widget._lane_timer.stop()
    img = QImage(p._panel_w, p._panel_h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#000000"))
    pa = QPainter(img)
    p._widget.render(pa, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    pa.end()
    p._widget._vehicle_anim_timer.stop()
    return img


_FLIPS = [
    dict(acc_locked=False),
    dict(acc_locked=True, acc_truck=False),
    dict(acc_locked=True, acc_truck=True),
    dict(acc_locked=False, acc_truck=False),
    dict(acc_locked=True, acc_truck=False),
]


def test_lock_and_lead_changes_reuse_the_masks(panel, monkeypatch):
    for fields in _FLIPS:
        _render(panel, **fields)
    w = panel._widget

    def rebuilt(*_a, **_k):
        raise AssertionError("mask rebuilt on a lock or lead change")

    monkeypatch.setattr(w, "_dilated_alpha", rebuilt)
    monkeypatch.setattr(w, "_white_with_alpha", rebuilt)
    for fields in _FLIPS:
        _render(panel, **fields)


@pytest.mark.parametrize("final", _FLIPS[1:4], ids=["car", "truck", "lost"])
def test_kept_caches_draw_what_a_fresh_panel_draws(qapp, final):
    from ui.cc_panel.main import cc_panel

    used = cc_panel("80 km/h", "Cruise control", True, scale_mult=0.5, acc_enabled=True)
    fresh = cc_panel("80 km/h", "Cruise control", True, scale_mult=0.5, acc_enabled=True)
    try:
        for fields in _FLIPS + [dict(AEB_warn=True), dict(AEB_warn=False)]:
            _render(used, **fields)
        # End the AEB flash's 2 s red cooldown, as the blink timer would.
        used._AEB_warn_off_time = 0.0
        used._widget._stop_blinking()
        used._vehicle_anim_layout_key = ()
        assert _render(used, **final) == _render(fresh, **final)
    finally:
        used.stop()
        fresh.stop()
