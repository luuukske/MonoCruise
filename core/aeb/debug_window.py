"""
AEB debug visualisation — PySide6 top-down radar view.

Ego-locked: ego is centred, always points up.  World rotates around it.

Coordinate transform (Reference sec. 3):
    dx = wx - ego_x;  dz = wz - ego_z
    rx = (-dx)*cos(-yaw) - dz*sin(-yaw)
    rz = (-dx)*sin(-yaw) + dz*cos(-yaw)

Create from the Qt main thread::

    from core.aeb.debug_window import AEBDebugWindow
    win = AEBDebugWindow()
    win.show()
"""

from __future__ import annotations

import math
import logging

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF, QPainterPath, QFont,
)
from PySide6.QtWidgets import QWidget

from core.thread_management.registry import registry
from .thread import AEBState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

_WIN_W = 600
_WIN_H = 600
_BG = QColor(20, 20, 25)
_GRID = QColor(45, 45, 55)
_GRID_STEP = 10.0

_EGO_CLR = QColor(60, 160, 255)
_EGO_PATH = QColor(60, 160, 255, 100)
_SAFE_CLR = QColor(80, 200, 120)
_DANGER_CLR = QColor(230, 60, 60)
_WARN_CLR = QColor(240, 180, 40)
_SUPPRESSED_CLR = QColor(100, 100, 110)
_SAFE_PATH = QColor(80, 200, 120, 80)
_DANGER_PATH = QColor(230, 60, 60, 100)
_WARN_PATH = QColor(240, 180, 40, 80)
_SUPPRESSED_PATH = QColor(100, 100, 110, 60)
_TRAILER_CLR = QColor(180, 140, 80)
_TEXT = QColor(200, 200, 210)
_HUD_BG = QColor(0, 0, 0, 140)

_REFRESH_MS = 33
PPM = 8.0
MIN_SPEED_KMH = 35.0


# ---------------------------------------------------------------------------
# Coordinate helpers  (Reference sec. 3)
# ---------------------------------------------------------------------------

def _world_to_ego(
    wx: float, wz: float, ego_x: float, ego_z: float, ego_yaw: float,
) -> tuple[float, float]:
    """World XZ -> ego-space.  rx>0 = right, rz>0 = forward."""
    dx = wx - ego_x
    dz = wz - ego_z
    neg_dx = -dx
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    rx = neg_dx * c - dz * s
    rz = neg_dx * s + dz * c
    return rx, rz


def _ego_to_screen(rx: float, rz: float, cx: float, cy: float) -> tuple[float, float]:
    return cx + rx * PPM, cy - rz * PPM


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class AEBDebugWindow(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AEB Debug Radar")
        self.setMinimumSize(_WIN_W, _WIN_H)
        self.resize(_WIN_W, _WIN_H)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(_REFRESH_MS)

    @staticmethod
    def _fetch_snapshot():
        try:
            aeb = registry.get_thread("aeb_thread")
        except KeyError:
            return None
        if aeb is None or not aeb.is_alive():
            return None
        return aeb.data.snapshot

    def _w2s(self, wx: float, wz: float, ex: float, ez: float, ey: float) -> tuple[float, float]:
        rx, rz = _world_to_ego(wx, wz, ex, ez, ey)
        return _ego_to_screen(rx, rz, self.width() / 2.0, self.height() / 2.0)

    # -- paint -------------------------------------------------------------

    def paintEvent(self, event) -> None:
        snap = self._fetch_snapshot()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), _BG)

        if snap is None:
            p.setPen(QPen(_TEXT))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "AEB thread not running")
            p.end()
            return

        ex, ez, ey = snap.ego_x, snap.ego_z, snap.ego_yaw

        self._draw_grid(p)

        for v in snap.vehicles:
            is_suppressed = v.get("rear_suppressed", False)
            danger = v["vid"] in snap.colliding_ids

            if is_suppressed:
                path_clr = _SUPPRESSED_PATH
                veh_clr = _SUPPRESSED_CLR
            elif danger:
                path_clr = _DANGER_PATH
                veh_clr = _DANGER_CLR
            else:
                path_clr = _SAFE_PATH
                veh_clr = _SAFE_CLR

            self._draw_path(
                p, v["path"], ex, ez, ey, v["half_w"], path_clr,
            )
            self._draw_vehicle(
                p, v["x"], v["z"], v["yaw"],
                v["half_w"], v["length"], v["is_tmp"],
                ex, ez, ey, veh_clr,
            )

            for tr in v.get("trailers", []):
                self._draw_vehicle(
                    p, tr["x"], tr["z"], tr["yaw"],
                    tr["half_w"], tr["length"], tr["is_tmp"],
                    ex, ez, ey, _TRAILER_CLR,
                )

        self._draw_path(p, snap.ego_path, ex, ez, ey, snap.ego_half_w, _EGO_PATH)
        self._draw_vehicle_ego(
            p, ex, ez, ey, snap.ego_half_w, snap.ego_half_l, ex, ez, ey, _EGO_CLR,
        )

        self._draw_hud(p, snap)
        p.end()

    # -- grid --------------------------------------------------------------

    def _draw_grid(self, p: QPainter) -> None:
        p.setPen(QPen(_GRID, 1, Qt.DotLine))
        p.setBrush(Qt.NoBrush)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        for d in range(int(_GRID_STEP), 200, int(_GRID_STEP)):
            r = d * PPM
            p.drawEllipse(QPointF(cx, cy), r, r)

    # -- traffic vehicle ---------------------------------------------------

    def _draw_vehicle(
        self, p: QPainter,
        wx: float, wz: float, yaw: float,
        half_w: float, length: float, is_tmp: bool,
        ego_x: float, ego_z: float, ego_yaw: float,
        color: QColor,
    ) -> None:
        if is_tmp:
            corners = self._corners_tmp(wx, wz, yaw, half_w, length)
        else:
            corners = self._corners_ai(wx, wz, yaw, half_w, length)

        poly = QPolygonF()
        for (cw, cz) in corners:
            sx, sy = self._w2s(cw, cz, ego_x, ego_z, ego_yaw)
            poly.append(QPointF(sx, sy))

        fill = QColor(color);  fill.setAlpha(160)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(color, 1.5))
        p.drawPolygon(poly)

    @staticmethod
    def _corners_tmp(wx, wz, yaw, hw, length):
        """TMP: centre = position, rotate by -yaw (traffic convention)."""
        hl = length / 2.0
        s = math.sin(-yaw);  c = math.cos(-yaw)
        local = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        return [(wx + lx*c - lz*s, wz + lx*s + lz*c) for lx, lz in local]

    @staticmethod
    def _corners_ai(wx, wz, yaw, hw, length):
        """AI: pivot near front axle, 0.18 front / 0.82 back."""
        front = length * 0.18;  back = length * 0.82
        s = math.sin(-yaw);  c = math.cos(-yaw)
        local = [(-hw, -front), (hw, -front), (hw, back), (-hw, back)]
        return [(wx + lx*c - lz*s, wz + lx*s + lz*c) for lx, lz in local]

    # -- ego vehicle -------------------------------------------------------

    def _draw_vehicle_ego(
        self, p: QPainter,
        wx: float, wz: float, yaw: float,
        half_w: float, half_l: float,
        ego_x: float, ego_z: float, ego_yaw: float,
        color: QColor,
    ) -> None:
        """Ego: forward = (+sin(yaw), +cos(yaw)).  Build corners directly."""
        fx = math.sin(yaw)
        fz = math.cos(yaw)
        rx = fz      # right = cos(yaw)
        rz = -fx     # right_z = -sin(yaw)

        corners = [
            (wx - rx*half_w - fx*half_l, wz - rz*half_w - fz*half_l),  # rear-left
            (wx + rx*half_w - fx*half_l, wz + rz*half_w - fz*half_l),  # rear-right
            (wx + rx*half_w + fx*half_l, wz + rz*half_w + fz*half_l),  # front-right
            (wx - rx*half_w + fx*half_l, wz - rz*half_w + fz*half_l),  # front-left
        ]

        poly = QPolygonF()
        for (cw, cz) in corners:
            sx, sy = self._w2s(cw, cz, ego_x, ego_z, ego_yaw)
            poly.append(QPointF(sx, sy))

        fill = QColor(color);  fill.setAlpha(160)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(color, 1.5))
        p.drawPolygon(poly)

    # -- path corridor -----------------------------------------------------

    def _draw_path(
        self, p: QPainter,
        path, ego_x: float, ego_z: float, ego_yaw: float,
        half_w: float, color: QColor,
    ) -> None:
        if len(path) < 2:
            return

        pts: list[tuple[float, float]] = []
        for entry in path:
            sx, sy = self._w2s(entry[0], entry[1], ego_x, ego_z, ego_yaw)
            pts.append((sx, sy))

        cpx = half_w * PPM
        left: list[QPointF] = []
        right: list[QPointF] = []

        for i in range(len(pts)):
            if i == 0:
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
            elif i == len(pts) - 1:
                dx = pts[-1][0] - pts[-2][0]
                dy = pts[-1][1] - pts[-2][1]
            else:
                dx = pts[i+1][0] - pts[i-1][0]
                dy = pts[i+1][1] - pts[i-1][1]

            ln = math.sqrt(dx*dx + dy*dy)
            if ln < 1e-6:
                nx, ny = 0.0, 0.0
            else:
                nx = -dy / ln * cpx
                ny = dx / ln * cpx

            sx, sy = pts[i]
            left.append(QPointF(sx + nx, sy + ny))
            right.append(QPointF(sx - nx, sy - ny))

        base_alpha = color.alpha()
        min_alpha = max(int(base_alpha * 0.15), 0)
        n_segments = len(left) - 1

        p.setBrush(Qt.NoBrush)

        for i in range(n_segments):
            t = (i + 1) / float(n_segments)
            fade = 1.0 - t
            seg_alpha = int(base_alpha * fade)
            if seg_alpha < min_alpha:
                seg_alpha = min_alpha

            seg_color = QColor(color)
            seg_color.setAlpha(seg_alpha)

            pen = QPen(seg_color, 1.5, Qt.SolidLine)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)

            p.drawLine(left[i], left[i + 1])
            p.drawLine(right[i], right[i + 1])

    # -- HUD ---------------------------------------------------------------

    def _draw_hud(self, p: QPainter, snap) -> None:
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(_HUD_BG))
        p.drawRoundedRect(8, 8, 280, 130, 6, 6)

        p.setFont(QFont("Segoe UI", 10))
        p.setPen(QPen(_TEXT))

        kmh = snap.ego_speed * 3.6
        y = 24
        p.drawText(16, y, f"Speed: {kmh:.0f} km/h")
        y += 20
        nv = len(snap.vehicles)
        nc = len(snap.colliding_ids)
        ns = len(snap.suppressed_ids)
        p.drawText(16, y, f"Traffic: {nv}   Threats: {nc}   Rear: {ns}")
        y += 20

        state = snap.aeb_state

        if state == AEBState.BRAKE:
            p.setPen(QPen(_DANGER_CLR))
            ttc_str = f"{snap.time_to_collision:.2f}" if snap.time_to_collision < 100 else "\u221E"
            ttb_str = f"{snap.time_to_brake:.2f}" if snap.time_to_brake < 100 else "\u221E"
            p.drawText(16, y, f"\u26A0 AEB BRAKE  TTC: {ttc_str}s  TTB: {ttb_str}s")
        elif state == AEBState.WARN:
            p.setPen(QPen(_WARN_CLR))
            ttc_str = f"{snap.time_to_collision:.2f}" if snap.time_to_collision < 100 else "\u221E"
            p.drawText(16, y, f"\u26A0 AEB WARN  TTC: {ttc_str}s")
        else:
            p.setPen(QPen(_SAFE_CLR))
            p.drawText(16, y, "AEB standby")

        y += 20
        if kmh < MIN_SPEED_KMH:
            p.setPen(QPen(QColor(160, 160, 170)))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(16, y, f"AEB inactive below {MIN_SPEED_KMH:.0f} km/h")

        y += 20
        if ns > 0:
            p.setPen(QPen(_SUPPRESSED_CLR))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(16, y, f"{ns} rear-approach vehicle(s) suppressed")