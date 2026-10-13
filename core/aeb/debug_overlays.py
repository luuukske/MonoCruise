"""Overlays in the AEB debug window that are not AEB's own: map lanes, ACC's map path and trails.

Road lanes draw as outlines (edges half a lane each side of the centreline);
junction curves draw as a dashed centreline, since their edges overlap into a
blur. The map path is coloured by its trust: green where AEB may use it,
fading to orange where a branch or ego's offset takes the trust away.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF

from core.road_layout.looks import LANE_WIDTH_M

_EDGE_CLR = QColor(110, 115, 140, 170)
_CENTRE_CLR = QColor(85, 90, 110, 120)
_JUNCTION_CLR = QColor(90, 140, 190, 150)
_PATH_HI = QColor(150, 235, 120)
_ROAD_PATH_CLR = QColor(235, 215, 90, 170)
_PATH_LO = QColor(245, 150, 60)
_REACH_M = 260.0
_TRAIL_ARC_CLR = QColor(130, 130, 200, 90)
_TRAIL_CROSS_CLR = QColor(255, 200, 80, 180)
_TRAIL_ARC_HALF_SPAN_M = 50.0
_TRAIL_ARC_SAMPLES = 32

_cache: dict = {"window": None, "outlines": ()}


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


def draw_trail_arc(p: QPainter, comp: dict | None, to_screen) -> None:
    """Faint dashed ACC trail arc around a target, plus its ego-row crossing marker."""
    if comp is None or not comp.get("trail_valid", False):
        return
    p.setPen(QPen(_TRAIL_ARC_CLR, 1.0, Qt.DashLine))
    p.setBrush(Qt.NoBrush)
    if comp.get("trail_is_straight", False):
        px, pz = comp["trail_point_x"], comp["trail_point_z"]
        dx, dz = comp["trail_dir_x"], comp["trail_dir_z"]
        span = _TRAIL_ARC_HALF_SPAN_M
        ax, ay = to_screen(px - span * dx, pz - span * dz)
        bx, by = to_screen(px + span * dx, pz + span * dz)
        p.drawLine(QPointF(ax, ay), QPointF(bx, by))
    else:
        cx, cz, radius = comp["trail_cx"], comp["trail_cz"], comp["trail_R"]
        if radius < 1e-3:
            return
        target = math.atan2(comp["trail_point_z"] - cz, comp["trail_point_x"] - cx)
        half_sweep = min(_TRAIL_ARC_HALF_SPAN_M / radius, math.pi)
        poly = QPolygonF()
        n = _TRAIL_ARC_SAMPLES
        for i in range(n):
            ang = target + (i / (n - 1) - 0.5) * 2.0 * half_sweep
            sx, sy = to_screen(cx + radius * math.cos(ang), cz + radius * math.sin(ang))
            poly.append(QPointF(sx, sy))
        p.drawPolyline(poly)
    if comp.get("trail_crossing_valid", False):
        mx, my = to_screen(comp["trail_crossing_x"], comp["trail_crossing_z"])
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(_TRAIL_CROSS_CLR))
        p.drawEllipse(QPointF(mx, my), 3.0, 3.0)


def draw_lanes(p: QPainter, window, to_screen, ex: float, ez: float) -> None:
    """Outlines of the lanes within reach of ego. ``to_screen(wx, wz)`` returns pixels."""
    if window is None:
        return
    for kind, polylines, box in _outlines(window):
        if box[1] < ex - _REACH_M or box[0] > ex + _REACH_M or box[3] < ez - _REACH_M or box[2] > ez + _REACH_M:
            continue
        if kind == "road":
            left, centre, right = polylines
            p.setPen(QPen(_EDGE_CLR, 1.0))
            p.drawPolyline(_poly(left, to_screen))
            p.drawPolyline(_poly(right, to_screen))
            p.setPen(QPen(_CENTRE_CLR, 1.0, Qt.DotLine))
            p.drawPolyline(_poly(centre, to_screen))
        else:
            p.setPen(QPen(_JUNCTION_CLR, 1.0, Qt.DashLine))
            p.drawPolyline(_poly(polylines[0], to_screen))


def draw_map_path(p: QPainter, pts, trust, to_screen) -> None:
    """ACC's fused map path this frame, one segment per sample pair."""
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        w = trust[i] if i < len(trust) else 0.0
        clr = QColor(
            int(_PATH_LO.red() + w * (_PATH_HI.red() - _PATH_LO.red())),
            int(_PATH_LO.green() + w * (_PATH_HI.green() - _PATH_LO.green())),
            int(_PATH_LO.blue() + w * (_PATH_HI.blue() - _PATH_LO.blue())),
            int(90 + 140 * w),
        )
        p.setPen(QPen(clr, 2.2))
        ax, ay = to_screen(*pts[i])
        bx, by = to_screen(*pts[i + 1])
        p.drawLine(QPointF(ax, ay), QPointF(bx, by))


def _poly(pts, to_screen) -> QPolygonF:
    poly = QPolygonF()
    for wx, wz in pts:
        sx, sy = to_screen(wx, wz)
        poly.append(QPointF(sx, sy))
    return poly


def _outlines(window):
    """Edges per lane, built once per window."""
    if _cache["window"] is window:
        return _cache["outlines"]
    half = LANE_WIDTH_M / 2.0
    out = []
    for lane in getattr(window, "lanes", ()):
        box = (lane.min_x, lane.max_x, lane.min_z, lane.max_z)
        if lane.kind == "road":
            out.append(("road", (_offset(lane.pts, -half), lane.pts, _offset(lane.pts, half)), box))
        else:
            out.append(("prefab", (lane.pts,), box))
    _cache["window"] = window
    _cache["outlines"] = tuple(out)
    return _cache["outlines"]


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
