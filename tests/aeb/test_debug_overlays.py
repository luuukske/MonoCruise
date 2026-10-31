"""Debug window drawing: the world transform, the per-window lane path cache, ground dot sprites."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.aeb import debug_overlays as ov
from core.aeb.debug_window import _PPM, _e2s, _w2e


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _lane(kind: str, x0: float, z0: float, x1: float, z1: float) -> SimpleNamespace:
    pts = tuple((x0 + (x1 - x0) * k / 10.0, z0 + (z1 - z0) * k / 10.0) for k in range(11))
    return SimpleNamespace(
        kind=kind, pts=pts,
        min_x=min(x0, x1), max_x=max(x0, x1), min_z=min(z0, z1), max_z=max(z0, z1),
    )


@pytest.mark.parametrize("ey", [0.0, 0.7, -2.4, math.pi])
def test_world_transform_matches_the_window_mapping(ey):
    ex, ez, cx, cy = 41250.3, -18733.9, 600.0, 525.0
    xform = ov.world_transform(ex, ez, ey, cx, cy, _PPM)
    for wx, wz in [(ex, ez), (ex + 30.0, ez - 12.0), (ex - 85.5, ez + 140.25)]:
        want = _e2s(*_w2e(wx, wz, ex, ez, ey), cx, cy)
        got = xform.map(wx, wz)
        assert got == pytest.approx(want, abs=1e-6)


def test_lanes_out_of_view_are_never_built_and_the_cull_follows_ego():
    from PySide6.QtGui import QImage, QPainter

    app = _app()
    near = _lane("road", -20.0, 0.0, 20.0, 0.0)
    junction = _lane("prefab", 0.0, -20.0, 0.0, 20.0)
    far = _lane("road", 500.0, 0.0, 540.0, 0.0)
    window = SimpleNamespace(lanes=(near, junction, far))
    img = QImage(400, 300, QImage.Format_ARGB32_Premultiplied)

    def draw(ex: float, ez: float) -> None:
        p = QPainter(img)
        ov.draw_lanes(p, window, ov.world_transform(ex, ez, 0.0, 200.0, 225.0, _PPM), ex, ez, 60.0)
        p.end()

    draw(0.0, 0.0)
    shapes = ov._cache["shapes"]
    assert shapes[0] is not None and shapes[1] is not None
    assert shapes[2] is None
    edges, centres, junctions = ov._cache["paths"]
    assert not edges.isEmpty() and not centres.isEmpty() and not junctions.isEmpty()

    draw(ov._VIEW_STEP_M * 0.5, 0.0)
    assert ov._cache["paths"][0] is edges

    draw(520.0, 0.0)
    assert ov._cache["shapes"][2] is not None
    assert ov._cache["paths"][1].isEmpty() is False
    assert ov._cache["paths"][2].isEmpty()

    other = SimpleNamespace(lanes=(far,))
    p = QPainter(img)
    ov.draw_lanes(p, other, ov.world_transform(0.0, 0.0, 0.0, 200.0, 225.0, _PPM), 0.0, 0.0, 60.0)
    p.end()
    assert ov._cache["window"] is other
    assert ov._cache["shapes"] == [None]
    assert app is not None


def test_map_path_paints_on_screen():
    from PySide6.QtGui import QColor, QImage, QPainter

    app = _app()
    img = QImage(400, 300, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(0, 0, 0))
    pts = [(0.0, -2.5 * k) for k in range(20)]
    trust = [k / 19.0 for k in range(20)]
    xform = ov.world_transform(0.0, 0.0, 0.0, 200.0, 225.0, _PPM)
    p = QPainter(img)
    ov.draw_map_path(p, pts, trust, xform)
    p.end()
    sx, sy = xform.map(0.0, -10.0)
    assert QColor(img.pixel(round(sx), round(sy))) != QColor(0, 0, 0)
    assert app is not None


@pytest.mark.parametrize("dpr", [1.0, 1.75])
@pytest.mark.parametrize("fx, fy", [(0.0, 0.0), (0.3, 0.8), (0.55, 0.1), (0.95, 0.45)])
def test_ground_dot_sprite_lands_where_the_projection_puts_it(dpr, fx, fy):
    from PySide6.QtGui import QImage, QPainter

    from core.aeb.debug_grid import draw_ground_markers

    app = _app()
    size = 40
    img = QImage(round(size * dpr), round(size * dpr), QImage.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    img.fill(0)
    ox, oy = 20.0 + fx, 20.0 + fy
    p = QPainter(img)
    draw_ground_markers(
        p,
        lambda wx, wz: (ox + wx * 10.0, oy + wz * 10.0),
        lambda sx, sy: ((sx - ox) / 10.0, (sy - oy) / 10.0),
        float(size), float(size),
    )
    p.end()
    sx = sy = sw = 0.0
    for y in range(img.height()):
        for x in range(img.width()):
            a = (img.pixel(x, y) >> 24) & 255
            if a and abs(x - ox * dpr) < 6 * dpr and abs(y - oy * dpr) < 6 * dpr:
                sx += a * (x + 0.5)
                sy += a * (y + 0.5)
                sw += a
    assert sw > 0
    assert math.hypot(sx / sw - ox * dpr, sy / sw - oy * dpr) < 0.25
    assert app is not None
