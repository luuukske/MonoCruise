"""Showcase painter: one flat frame per `FrameState`, drawn into a QImage.

Flat by design: the one gradient is the fade along a predicted path. What the debug
view draws and this one drops is listed in `tools/clip_export/README.md`.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QPen, QPolygonF,
    QTransform,
)

from core.aeb.debug_grid import draw_ground_markers
from tools.clip_export.camera import Camera
from tools.clip_export.timeline import Body, Corridor, FrameState

BG = QColor(9, 11, 16)
DOT_MINOR = QColor(78, 80, 100, 170)
DOT_MAJOR = QColor(125, 127, 160, 205)
EGO = QColor(64, 168, 255)
EGO_EDGE = QColor(160, 212, 255)
EGO_TRAILER = QColor(58, 96, 142)
EGO_TRAILER_EDGE = QColor(128, 172, 222)
TRAFFIC = QColor(54, 62, 78)
TRAFFIC_EDGE = QColor(92, 102, 122)
TRAFFIC_TRAILER = QColor(44, 51, 65)
PATH_GRAY = QColor(150, 158, 174)
WARN = QColor(255, 184, 48)
WARN_EDGE = QColor(255, 216, 140)
BRAKE = QColor(255, 60, 60)
BRAKE_EDGE = QColor(255, 150, 150)
STANDBY_BG = QColor(38, 44, 56)
STANDBY_TEXT = QColor(176, 184, 200)
TEXT = QColor(236, 240, 246)
MUTED = QColor(146, 154, 172)
DARK_TEXT = QColor(22, 16, 6)

# (fill, edge) alpha at the near end of each predicted path; both fade to clear.
EGO_PATH_ALPHA = (0.14, 0.55)
THREAT_PATH_ALPHA = (0.22, 0.55)
TRAFFIC_PATH_ALPHA = (0.08, 0.30)

# State index (timeline.aeb_state) to label, badge fill, text colour.
STATE_LABELS = (
    ("STANDBY", STANDBY_BG, STANDBY_TEXT),
    ("AEB WARN", WARN, DARK_TEXT),
    ("AEB BRAKE", BRAKE, TEXT),
)

_FONT_FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "Inter", "Helvetica Neue",
                  "Arial", "DejaVu Sans"]
_CULL_PAD_M = 30.0


def with_alpha(c: QColor, a: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(min(max(a, 0.0), 1.0))
    return out


def world_transform(cam: Camera, width: int, height: int) -> QTransform:
    """Timeline metres to pixels: ``cam`` centred, ego heading up. Same axes as `AEBDebugWindow`."""
    c, s = math.cos(cam.yaw) * cam.ppm, math.sin(cam.yaw) * cam.ppm
    return QTransform(c, s, -s, c,
                      width / 2.0 - (c * cam.x - s * cam.z),
                      height / 2.0 - (s * cam.x + c * cam.z))


def _font(px: float, weight: QFont.Weight, spacing: float = 0.0) -> QFont:
    f = QFont()
    f.setFamilies(_FONT_FAMILIES)
    f.setPixelSize(max(1, round(px)))
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return f


class ShowcaseRenderer:
    """Stateless per frame; holds fonts and the badge geometry so they are built once."""

    def __init__(self, width: int, height: int, *, origin: tuple[float, float],
                 hud: bool = True, credit: str | None = None) -> None:
        self.w, self.h = width, height
        self.s = min(width, height) / 1080.0
        self.origin = origin
        self.hud = hud
        self.credit = credit
        s = self.s
        self._f_state = _font(22 * s, QFont.Bold, 2.0 * s)
        self._f_credit = _font(22 * s, QFont.Medium)
        # Top-left like the debug view's state title, one width so it never resizes.
        fm = QFontMetricsF(self._f_state)
        text_w = max(fm.horizontalAdvance(label) for label, _bg, _ink in STATE_LABELS)
        badge_w, badge_h = text_w + 60.0 * s, 54.0 * s
        self._badge = QRectF(48.0 * s, 44.0 * s, badge_w, badge_h)

    def render(self, st: FrameState, cam: Camera) -> QImage:
        img = QImage(self.w, self.h, QImage.Format_ARGB32_Premultiplied)
        img.fill(BG)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)
        world = world_transform(cam, self.w, self.h)
        self._dots(p, world)

        threat, threat_edge = (BRAKE, BRAKE_EDGE) if st.state == 2 else (WARN, WARN_EDGE)
        p.setTransform(world)
        screen = QRectF(0.0, 0.0, self.w, self.h)
        threats = {v.vid for v in st.vehicles if v.threat}
        for vid, corridors in st.corridors.items():
            if vid not in threats:
                for corridor in corridors:
                    self._corridor(p, world, screen, corridor, PATH_GRAY, TRAFFIC_PATH_ALPHA)
        if st.ego_corridor is not None:
            self._corridor(p, world, screen, st.ego_corridor, EGO, EGO_PATH_ALPHA)
        for vid in threats:
            for corridor in st.corridors.get(vid, ()):
                self._corridor(p, world, screen, corridor, threat, THREAT_PATH_ALPHA)

        for view in sorted(st.vehicles, key=lambda v: v.threat):
            if not self._on_screen(world, view.body):
                continue
            if view.threat:
                fill, trailer_fill, edge = threat, threat, threat_edge
            else:
                fill, trailer_fill, edge = TRAFFIC, TRAFFIC_TRAILER, TRAFFIC_EDGE
            for tr in view.trailers:
                self._body(p, world, tr, trailer_fill, edge)
            self._body(p, world, view.body, fill, edge)

        if st.ego_trailer is not None:
            self._body(p, world, st.ego_trailer, EGO_TRAILER, EGO_TRAILER_EDGE)
        self._body(p, world, st.ego, EGO, EGO_EDGE)

        p.resetTransform()
        if self.hud:
            self._state_badge(p, st.state)
        if self.credit:
            self._credit(p)
        p.end()
        return img

    def _on_screen(self, world: QTransform, body: Body) -> bool:
        sx, sy = world.map(body.x, body.z)
        pad = (_CULL_PAD_M + body.length) * math.hypot(world.m11(), world.m12())
        return -pad <= sx <= self.w + pad and -pad <= sy <= self.h + pad

    def _dots(self, p: QPainter, world: QTransform) -> None:
        inv, ok = world.inverted()
        if not ok:
            return
        ox, oz = self.origin

        def project(wx: float, wz: float) -> tuple[float, float]:
            return world.map(wx - ox, wz - oz)

        def unproject(sx: float, sy: float) -> tuple[float, float]:
            x, z = inv.map(sx, sy)
            return x + ox, z + oz

        draw_ground_markers(p, project, unproject, float(self.w), float(self.h),
                            minor_clr=DOT_MINOR, major_clr=DOT_MAJOR,
                            minor_r=1.6 * self.s, major_r=3.6 * self.s)

    def _corridor(self, p: QPainter, world: QTransform, screen: QRectF, corridor: Corridor,
                  color: QColor, alpha: tuple[float, float]) -> None:
        left, right = corridor
        if len(left) < 2:
            return
        poly = QPolygonF([QPointF(x, z) for x, z in left] + [QPointF(x, z) for x, z in reversed(right)])
        if not world.mapRect(poly.boundingRect()).intersects(screen):
            return
        # Fades with distance along the path: near end at full alpha, far end clear.
        a = QPointF((left[0][0] + right[0][0]) / 2.0, (left[0][1] + right[0][1]) / 2.0)
        b = QPointF((left[-1][0] + right[-1][0]) / 2.0, (left[-1][1] + right[-1][1]) / 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._fade(a, b, color, alpha[0])))
        p.drawPolygon(poly)
        pen = QPen(QBrush(self._fade(a, b, color, alpha[1])), 1.5 * self.s)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF([QPointF(x, z) for x, z in left]))
        p.drawPolyline(QPolygonF([QPointF(x, z) for x, z in right]))

    @staticmethod
    def _fade(a: QPointF, b: QPointF, color: QColor, alpha: float) -> QLinearGradient:
        g = QLinearGradient(a, b)
        g.setColorAt(0.0, with_alpha(color, alpha))
        g.setColorAt(1.0, with_alpha(color, 0.0))
        return g

    def _body(self, p: QPainter, world: QTransform, b: Body, fill: QColor, edge: QColor) -> None:
        local = QTransform().rotateRadians(-b.yaw) * QTransform.fromTranslate(b.x, b.z) * world
        p.setTransform(local)
        rect = QRectF(-b.half_w, -b.length / 2.0, 2.0 * b.half_w, b.length)
        radius = min(0.55, b.half_w * 0.45)
        pen = QPen(edge, 1.5 * self.s)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(rect, radius, radius)
        p.setTransform(world)

    def _state_badge(self, p: QPainter, state: int) -> None:
        label, bg, ink = STATE_LABELS[state]
        rect = self._badge
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(rect, rect.height() / 2.0, rect.height() / 2.0)
        p.setPen(ink)
        p.setFont(self._f_state)
        p.drawText(rect, Qt.AlignCenter, label)

    def _credit(self, p: QPainter) -> None:
        s = self.s
        p.setFont(self._f_credit)
        p.setPen(MUTED)
        rect = QRectF(0, 0, self.w - 48.0 * s, self.h - 40.0 * s)
        p.drawText(rect, Qt.AlignRight | Qt.AlignBottom, self.credit)
