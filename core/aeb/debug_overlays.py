"""Overlays in the AEB debug window that are not AEB's own: map lanes, ACC's map path and trails.

Road lanes draw as outlines (edges half a lane each side of the centreline);
junction curves draw as a dashed centreline, since their edges overlap into a
blur. The map path is coloured by its trust: green where AEB may use it,
fading to orange where a branch or ego's offset takes the trust away.

Lanes are kept as world-space paths, built once per lane per window and merged
per pen for the lanes in view, then drawn through the painter's world transform
with cosmetic pens. Per-point Python work each frame held the GUI thread about
11 ms on a dense window; the merged paths are re-culled only every ``_VIEW_STEP_M``.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF, QTransform

from core.road_layout.looks import LANE_WIDTH_M

_EDGE_CLR = QColor(110, 115, 140, 170)
_CENTRE_CLR = QColor(85, 90, 110, 120)
_JUNCTION_CLR = QColor(90, 140, 190, 150)
_PATH_HI = QColor(150, 235, 120)
_ROAD_PATH_CLR = QColor(235, 215, 90, 170)
_PATH_LO = QColor(245, 150, 60)
_VIEW_STEP_M = 10.0
_TRUST_LEVELS = 32

_cache: dict = {"window": None, "shapes": [], "view": None, "paths": ()}
_pens: dict = {}


def overlay_legend() -> list[tuple[QColor, str]]:
    return [(_EDGE_CLR, "Map lane edge"), (_PATH_HI, "ACC map path (trust)"),
            (_ROAD_PATH_CLR, "AEB road path")]


def draw_road_path(p: QPainter, arc, to_screen) -> None:
    """AEB's road-bent ego path as a dashed centreline (README road-bent trajectories)."""
    if arc is None:
        return
    pts = [to_screen(*arc.position_at_dist(arc.arc_length * i / 40.0)) for i in range(41)]
    p.setPen(QPen(_ROAD_PATH_CLR, 1.5, Qt.DashLine))
    p.setBrush(Qt.NoBrush)
    p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))


def fetch_map_label() -> str:
    """Which map the worker is serving, for the debug legend."""
    from core.map_thread.thread import map_snapshot

    snap = map_snapshot(1.0)
    if snap is None:
        return "Map: none"
    return "Map: game + TMP" if snap.tmp else "Map: game"


def fetch_map_window():
    """Lane window from the map worker, or None. Lanes stay drawn for a second of staleness."""
    from core.map_thread.thread import map_snapshot

    snap = map_snapshot(1.0)
    return None if snap is None else snap.window


def world_transform(ex: float, ez: float, ey: float, cx: float, cy: float, ppm: float) -> QTransform:
    """World XZ to pixels for the ego-locked view: ego at ``(cx, cy)``, forward up."""
    c = math.cos(-ey)
    s = math.sin(-ey)
    m11, m21 = ppm * c, ppm * s
    m12, m22 = -ppm * s, ppm * c
    return QTransform(m11, m12, m21, m22, cx - m11 * ex - m21 * ez, cy - m12 * ex - m22 * ez)


def draw_lanes(p: QPainter, window, xform: QTransform, ex: float, ez: float, reach_m: float) -> None:
    """Outlines of the lanes within ``reach_m`` of ego. ``xform`` maps world XZ to pixels."""
    if window is None:
        return
    paths = _paths_in_view(window, ex, ez, reach_m)
    dpr = _dpr(p)
    p.save()
    p.setWorldTransform(xform)
    p.setBrush(Qt.NoBrush)
    for key, path in zip(("edge", "centre", "junction"), paths):
        if not path.isEmpty():
            p.setPen(_pen(key, dpr))
            p.drawPath(path)
    p.restore()


def draw_map_path(p: QPainter, pts, trust, xform: QTransform) -> None:
    """ACC's fused map path this frame, one segment per sample pair."""
    if len(pts) < 2:
        return
    dpr = _dpr(p)
    p.save()
    p.setWorldTransform(xform)
    pts = [QPointF(x, z) for x, z in pts]
    level = -1
    for i in range(len(pts) - 1):
        w = trust[i] if i < len(trust) else 0.0
        lv = min(max(round(w * _TRUST_LEVELS), 0), _TRUST_LEVELS)
        if lv != level:
            level = lv
            p.setPen(_pen(("path", lv), dpr))
        p.drawLine(pts[i], pts[i + 1])
    p.restore()


def _dpr(p: QPainter) -> float:
    """Device pixels per logical pixel of the painter's target, render redirection included."""
    dev = p.worldTransform().inverted()[0] * p.deviceTransform()
    return math.hypot(dev.m11(), dev.m12())


def _pen(key, dpr: float) -> QPen:
    """Cosmetic pens, so width and dashes stay in pixels. Lanes keep a one device pixel hairline
    (a wider AA stroke cost ~3x on HiDPI) and take ``dpr`` as alpha instead."""
    pen = _pens.get((key, dpr))
    if pen is not None:
        return pen
    if key in ("edge", "centre", "junction"):
        clr = QColor({"edge": _EDGE_CLR, "centre": _CENTRE_CLR, "junction": _JUNCTION_CLR}[key])
        clr.setAlpha(min(255, round(clr.alpha() * max(dpr, 1.0))))
        style = {"edge": Qt.SolidLine, "centre": Qt.DotLine, "junction": Qt.DashLine}[key]
        pen = QPen(clr, 1.0, style)
    else:
        w = key[1] / _TRUST_LEVELS
        pen = QPen(QColor(
            int(_PATH_LO.red() + w * (_PATH_HI.red() - _PATH_LO.red())),
            int(_PATH_LO.green() + w * (_PATH_HI.green() - _PATH_LO.green())),
            int(_PATH_LO.blue() + w * (_PATH_HI.blue() - _PATH_LO.blue())),
            int(90 + 140 * w),
        ), 2.2 * dpr)
    pen.setCosmetic(True)
    _pens[(key, dpr)] = pen
    return pen


def _paths_in_view(window, ex: float, ez: float, reach_m: float) -> tuple:
    """Merged edge, centre and junction paths of the lanes near ego, re-culled every ``_VIEW_STEP_M``."""
    if _cache["window"] is not window:
        _cache["window"] = window
        _cache["shapes"] = [None] * len(getattr(window, "lanes", ()))
        _cache["view"] = None
    view = _cache["view"]
    if (
        view is not None
        and abs(ex - view[0]) <= _VIEW_STEP_M
        and abs(ez - view[1]) <= _VIEW_STEP_M
        and reach_m <= view[2]
    ):
        return _cache["paths"]
    pad = reach_m + _VIEW_STEP_M
    edges, centres, junctions = QPainterPath(), QPainterPath(), QPainterPath()
    shapes = _cache["shapes"]
    for i, lane in enumerate(window.lanes):
        if lane.max_x < ex - pad or lane.min_x > ex + pad or lane.max_z < ez - pad or lane.min_z > ez + pad:
            continue
        shape = shapes[i]
        if shape is None:
            shape = shapes[i] = _lane_shape(lane)
        if lane.kind == "road":
            edges.addPath(shape[0])
            centres.addPath(shape[1])
        else:
            junctions.addPath(shape[0])
    _cache["view"] = (ex, ez, reach_m)
    _cache["paths"] = (edges, centres, junctions)
    return _cache["paths"]


def _lane_shape(lane) -> tuple:
    """World-space paths of one lane: (both edges, centreline) for a road, (centreline,) for a junction."""
    if lane.kind != "road":
        return (_path(lane.pts),)
    half = LANE_WIDTH_M / 2.0
    edges = _path(_offset(lane.pts, -half))
    edges.addPath(_path(_offset(lane.pts, half)))
    return edges, _path(lane.pts)


def _path(pts) -> QPainterPath:
    path = QPainterPath()
    path.addPolygon(QPolygonF([QPointF(x, z) for x, z in pts]))
    return path


def _offset(pts, right: float):
    out = []
    n = len(pts)
    for i in range(n):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, n - 1)]
        hx, hz = b[0] - a[0], b[1] - a[1]
        length = math.hypot(hx, hz)
        if length < 1e-9:
            out.append(pts[i])
            continue
        out.append((pts[i][0] - right * hz / length, pts[i][1] + right * hx / length))
    return out
