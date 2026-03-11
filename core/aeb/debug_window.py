"""
AEB debug visualisation — PySide6 top-down radar view.

Rebuilt to display arc-based corridors, evasion paths, collision markers,
and per-vehicle speed labels.

Ego-locked: ego is centred, always points up.  World rotates around it.

Coordinate transform:
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

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF, QPainterPath, QFont,
    QRadialGradient, QLinearGradient, QPaintEvent,
)
from PySide6.QtWidgets import QWidget

from core.thread_management.registry import registry
from .thread import AEBState, AEBSnapshot
from .traffic import ArcPath

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_WIN_W = 700
_WIN_H = 700
_BG = QColor(15, 15, 20)
_GRID_MAJOR = QColor(40, 40, 50)
_GRID_MINOR = QColor(28, 28, 35)
_GRID_STEP_MAJOR = 25.0  # metres
_GRID_STEP_MINOR = 5.0

_EGO_CLR = QColor(70, 170, 255)
_EGO_CORRIDOR = QColor(70, 170, 255, 50)
_SAFE_CLR = QColor(80, 210, 130)
_DANGER_CLR = QColor(240, 55, 55)
_WARN_CLR = QColor(245, 185, 40)
_SUPPRESSED_CLR = QColor(100, 100, 115)
_BRAKE_SUPP_CLR = QColor(255, 140, 40)
_EVASION_CLR = QColor(140, 80, 255)
_EVASION_CORRIDOR = QColor(140, 80, 255, 40)
_TRAILER_CLR = QColor(180, 140, 80)
_EGO_TRAILER_CLR = QColor(55, 130, 215)
_HIT_CLR = QColor(255, 30, 30)
_TEXT = QColor(200, 200, 215)
_HUD_BG = QColor(0, 0, 0, 160)
_HUD_BORDER = QColor(60, 60, 70)

_EGO_TRAILER_HALF_W = 1.25
_EGO_TRAILER_HALF_L = 6.8

_REFRESH_MS = 33
_PPM = 7.0          # pixels per metre
_MIN_SPEED_KMH = 35.0
_ARC_SAMPLES = 20   # samples for drawing corridors
_CORRIDOR_FADE_SEGMENTS = 16


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _w2e(wx: float, wz: float, ex: float, ez: float, ey: float) -> tuple[float, float]:
    """World XZ → ego-space.  rx>0 right, rz>0 forward."""
    dx = wx - ex
    dz = wz - ez
    c = math.cos(-ey)
    s = math.sin(-ey)
    return (-dx) * c - dz * s, (-dx) * s + dz * c


def _e2s(rx: float, rz: float, cx: float, cy: float) -> tuple[float, float]:
    """Ego-space → screen pixels."""
    return cx + rx * _PPM, cy - rz * _PPM


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class AEBDebugWindow(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AEB Debug Radar")
        self.setMinimumSize(500, 500)
        self.resize(_WIN_W, _WIN_H)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(_REFRESH_MS)

        # Fonts
        self._font_main = QFont("Segoe UI", 10)
        self._font_small = QFont("Segoe UI", 8)
        self._font_hud_title = QFont("Segoe UI Semibold", 11)
        self._font_label = QFont("Segoe UI", 7)

    # ---- data access -----------------------------------------------------

    @staticmethod
    def _fetch() -> AEBSnapshot | None:
        try:
            aeb = registry.get_thread("aeb_thread")
        except KeyError:
            return None
        if aeb is None or not aeb.is_alive():
            return None
        return aeb.data.snapshot

    def _ws(self, wx: float, wz: float, ex: float, ez: float, ey: float) -> tuple[float, float]:
        """World → screen shorthand."""
        rx, rz = _w2e(wx, wz, ex, ez, ey)
        return _e2s(rx, rz, self.width() / 2.0, self.height() / 2.0)

    # ---- main paint ------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        snap = self._fetch()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), _BG)

        if snap is None:
            p.setPen(QPen(_TEXT))
            p.setFont(self._font_main)
            p.drawText(self.rect(), Qt.AlignCenter, "AEB thread not running")
            p.end()
            return

        ex, ez, ey = snap.ego_x, snap.ego_z, snap.ego_yaw
        cx, cy = self.width() / 2.0, self.height() / 2.0

        self._draw_grid(p, cx, cy)

        # --- Vehicle corridors + boxes ------------------------------------
        for v in snap.vehicles:
            vid = v["vid"]
            is_supp = v.get("rear_suppressed", False)
            is_danger = vid in snap.colliding_ids
            is_brake_supp = vid in snap.braking_suppressed_ids

            if is_supp:
                body_clr, corr_clr = _SUPPRESSED_CLR, QColor(_SUPPRESSED_CLR)
            elif is_danger and is_brake_supp:
                body_clr, corr_clr = _BRAKE_SUPP_CLR, QColor(_BRAKE_SUPP_CLR)
            elif is_danger:
                body_clr, corr_clr = _DANGER_CLR, QColor(_DANGER_CLR)
            else:
                body_clr, corr_clr = _SAFE_CLR, QColor(_SAFE_CLR)

            # Draw arc corridor
            arc = snap.vehicle_arcs.get(vid)
            if arc is not None:
                corr_clr.setAlpha(45)
                self._draw_arc_corridor(p, arc, ex, ez, ey, corr_clr)

            # Draw vehicle box
            self._draw_vehicle_box(
                p, v["x"], v["z"], v["yaw"],
                v["half_w"], v["length"], v["is_tmp"],
                ex, ez, ey, body_clr,
            )

            # Speed label
            sx, sy = self._ws(v["x"], v["z"], ex, ez, ey)
            spd = v.get("speed_kmh", 0.0)
            self._draw_label(p, sx, sy - 14, f"{spd:.0f}", body_clr)

            # Trailers
            for tr in v.get("trailers", []):
                self._draw_vehicle_box(
                    p, tr["x"], tr["z"], tr["yaw"],
                    tr["half_w"], tr["length"], tr["is_tmp"],
                    ex, ez, ey, _TRAILER_CLR,
                )

        # --- Ego corridor -------------------------------------------------
        if snap.ego_arc is not None:
            self._draw_arc_corridor(p, snap.ego_arc, ex, ez, ey, _EGO_CORRIDOR)

        # --- Evasion corridor (purple, only in WARN/BRAKE) ----------------
        if snap.evasion_arc is not None and snap.aeb_state >= AEBState.WARN:
            self._draw_arc_corridor(
                p, snap.evasion_arc, ex, ez, ey, _EVASION_CORRIDOR,
                edge_color=_EVASION_CLR, edge_width=2.0,
            )

        # --- Ego trailer --------------------------------------------------
        if snap.ego_has_trailer:
            fx = -math.sin(ey)
            fz = -math.cos(ey)
            reach = snap.ego_half_l + _EGO_TRAILER_HALF_L
            self._draw_ego_box(
                p, ex - fx * reach, ez - fz * reach, ey,
                _EGO_TRAILER_HALF_W, _EGO_TRAILER_HALF_L,
                ex, ez, ey, _EGO_TRAILER_CLR,
            )

        # --- Ego box (drawn last, on top) ---------------------------------
        self._draw_ego_box(
            p, ex, ez, ey, snap.ego_half_w, snap.ego_half_l,
            ex, ez, ey, _EGO_CLR,
        )

        # --- Collision marker ---------------------------------------------
        if snap.aeb_state >= AEBState.WARN and snap.time_to_collision < 100:
            self._draw_hit_marker(p, snap.hit_x, snap.hit_z, ex, ez, ey)

        # --- HUD ----------------------------------------------------------
        self._draw_hud(p, snap)

        p.end()

    # ---- grid ------------------------------------------------------------

    def _draw_grid(self, p: QPainter, cx: float, cy: float) -> None:
        max_r = max(self.width(), self.height()) / _PPM + _GRID_STEP_MAJOR

        # Minor rings
        p.setPen(QPen(_GRID_MINOR, 0.5, Qt.DotLine))
        p.setBrush(Qt.NoBrush)
        d = _GRID_STEP_MINOR
        while d < max_r:
            # Skip if it coincides with a major ring
            if abs(d % _GRID_STEP_MAJOR) > 0.01:
                r = d * _PPM
                p.drawEllipse(QPointF(cx, cy), r, r)
            d += _GRID_STEP_MINOR

        # Major rings with distance labels
        p.setBrush(Qt.NoBrush)
        d = _GRID_STEP_MAJOR
        while d < max_r:
            r = d * _PPM
            p.setPen(QPen(_GRID_MAJOR, 1, Qt.DotLine))
            p.drawEllipse(QPointF(cx, cy), r, r)
            # Label
            p.setPen(QPen(QColor(70, 70, 85)))
            p.setFont(self._font_label)
            p.drawText(QPointF(cx + 3, cy - r + 11), f"{d:.0f}m")
            d += _GRID_STEP_MAJOR

        # Cross-hair
        p.setPen(QPen(_GRID_MAJOR, 0.5))
        p.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))
        p.drawLine(QPointF(0, cy), QPointF(self.width(), cy))

    # ---- arc corridor drawing --------------------------------------------

    def _draw_arc_corridor(
        self, p: QPainter,
        arc: ArcPath,
        ex: float, ez: float, ey: float,
        fill_color: QColor,
        edge_color: QColor | None = None,
        edge_width: float = 1.0,
    ) -> None:
        """Draw an arc corridor as a filled polygon with fading edges."""
        left, right = arc.sample_corridor(_ARC_SAMPLES)
        if len(left) < 2:
            return

        # Convert to screen
        s_left = [self._ws(x, z, ex, ez, ey) for x, z in left]
        s_right = [self._ws(x, z, ex, ez, ey) for x, z in right]

        # Build polygon: left forward + right backward
        poly = QPolygonF()
        for sx, sy in s_left:
            poly.append(QPointF(sx, sy))
        for sx, sy in reversed(s_right):
            poly.append(QPointF(sx, sy))

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(fill_color))
        p.drawPolygon(poly)

        # Draw edges with fade
        if edge_color is None:
            ec = QColor(fill_color)
            ec.setAlpha(min(fill_color.alpha() + 60, 200))
        else:
            ec = QColor(edge_color)

        n = len(s_left)
        for i in range(n - 1):
            fade = 1.0 - (i + 1) / n
            alpha = int(ec.alpha() * fade)
            c = QColor(ec)
            c.setAlpha(max(alpha, 8))
            pen = QPen(c, edge_width)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(*s_left[i]), QPointF(*s_left[i + 1]))
            p.drawLine(QPointF(*s_right[i]), QPointF(*s_right[i + 1]))

    # ---- vehicle box (traffic) -------------------------------------------

    def _draw_vehicle_box(
        self, p: QPainter,
        wx: float, wz: float, yaw: float,
        hw: float, length: float, is_tmp: bool,
        ex: float, ez: float, ey: float,
        color: QColor,
    ) -> None:
        if is_tmp:
            hl = length / 2.0
            corners_local = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        else:
            front = length * 0.18
            back = length * 0.82
            corners_local = [(-hw, -front), (hw, -front), (hw, back), (-hw, back)]

        s = math.sin(-yaw)
        c = math.cos(-yaw)
        world_corners = [
            (wx + lx * c - lz * s, wz + lx * s + lz * c)
            for lx, lz in corners_local
        ]

        poly = QPolygonF()
        for cw, cz in world_corners:
            sx, sy = self._ws(cw, cz, ex, ez, ey)
            poly.append(QPointF(sx, sy))

        fill = QColor(color)
        fill.setAlpha(140)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(color, 1.5))
        p.drawPolygon(poly)

        # Direction indicator (small line from center forward)
        cx_w, cz_w = wx, wz
        flen = min(length * 0.4, 4.0)
        tip_x = wx - flen * math.sin(yaw)
        tip_z = wz - flen * math.cos(yaw)
        scx, scy = self._ws(cx_w, cz_w, ex, ez, ey)
        stx, sty = self._ws(tip_x, tip_z, ex, ez, ey)
        p.setPen(QPen(QColor(255, 255, 255, 160), 1.5))
        p.drawLine(QPointF(scx, scy), QPointF(stx, sty))

    # ---- ego box ---------------------------------------------------------

    def _draw_ego_box(
        self, p: QPainter,
        wx: float, wz: float, yaw: float,
        hw: float, hl: float,
        ex: float, ez: float, ey: float,
        color: QColor,
    ) -> None:
        fx = -math.sin(yaw)
        fz = -math.cos(yaw)
        rx_d = fz
        rz_d = -fx

        corners = [
            (wx - rx_d * hw - fx * hl, wz - rz_d * hw - fz * hl),
            (wx + rx_d * hw - fx * hl, wz + rz_d * hw - fz * hl),
            (wx + rx_d * hw + fx * hl, wz + rz_d * hw + fz * hl),
            (wx - rx_d * hw + fx * hl, wz - rz_d * hw + fz * hl),
        ]

        poly = QPolygonF()
        for cw, cz in corners:
            sx, sy = self._ws(cw, cz, ex, ez, ey)
            poly.append(QPointF(sx, sy))

        fill = QColor(color)
        fill.setAlpha(170)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(color, 2.0))
        p.drawPolygon(poly)

    # ---- collision hit marker --------------------------------------------

    def _draw_hit_marker(
        self, p: QPainter,
        hx: float, hz: float,
        ex: float, ez: float, ey: float,
    ) -> None:
        sx, sy = self._ws(hx, hz, ex, ez, ey)

        # Pulsing glow
        grad = QRadialGradient(QPointF(sx, sy), 18)
        grad.setColorAt(0.0, QColor(255, 50, 50, 180))
        grad.setColorAt(0.5, QColor(255, 50, 50, 60))
        grad.setColorAt(1.0, QColor(255, 50, 50, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPointF(sx, sy), 18, 18)

        # Cross
        r = 7
        p.setPen(QPen(_HIT_CLR, 2.5))
        p.drawLine(QPointF(sx - r, sy - r), QPointF(sx + r, sy + r))
        p.drawLine(QPointF(sx - r, sy + r), QPointF(sx + r, sy - r))

        # Inner dot
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(_HIT_CLR))
        p.drawEllipse(QPointF(sx, sy), 3, 3)

    # ---- text label ------------------------------------------------------

    def _draw_label(
        self, p: QPainter, sx: float, sy: float, text: str, color: QColor,
    ) -> None:
        p.setFont(self._font_label)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        # Background pill
        bg = QColor(0, 0, 0, 130)
        rect = QRectF(sx - tw / 2 - 3, sy - th + 2, tw + 6, th + 1)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(rect, 3, 3)
        # Text
        tc = QColor(color)
        tc.setAlpha(220)
        p.setPen(QPen(tc))
        p.drawText(QPointF(sx - tw / 2, sy), text)

    # ---- HUD overlay -----------------------------------------------------

    def _draw_hud(self, p: QPainter, snap: AEBSnapshot) -> None:
        hud_w = 310
        hud_h = 175
        hud_x = 10
        hud_y = 10

        # Background
        p.setPen(QPen(_HUD_BORDER, 1))
        p.setBrush(QBrush(_HUD_BG))
        p.drawRoundedRect(QRectF(hud_x, hud_y, hud_w, hud_h), 8, 8)

        x = hud_x + 14
        y = hud_y + 22

        # Title bar colored by state
        if snap.aeb_state == AEBState.BRAKE:
            title_clr = _DANGER_CLR
            title = "⚠ AEB EMERGENCY BRAKE"
        elif snap.aeb_state == AEBState.WARN:
            title_clr = _WARN_CLR
            title = "⚠ AEB WARNING"
        else:
            title_clr = _SAFE_CLR
            title = "✓ AEB STANDBY"

        # State indicator strip
        strip_clr = QColor(title_clr)
        strip_clr.setAlpha(80)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(strip_clr))
        p.drawRoundedRect(QRectF(hud_x + 2, hud_y + 2, hud_w - 4, 24), 6, 6)

        p.setFont(self._font_hud_title)
        p.setPen(QPen(title_clr))
        p.drawText(QPointF(x, y), title)

        # Metrics
        y += 26
        p.setFont(self._font_main)
        p.setPen(QPen(_TEXT))

        kmh = snap.ego_speed * 3.6
        p.drawText(QPointF(x, y), f"Speed: {kmh:.0f} km/h")

        y += 18
        nv = len(snap.vehicles)
        nc = len(snap.colliding_ids)
        ns = len(snap.suppressed_ids)
        nb = len(snap.braking_suppressed_ids)
        p.drawText(QPointF(x, y), f"Tracked: {nv}   Threats: {nc}   Suppressed: {ns}")

        y += 18
        if snap.aeb_state >= AEBState.WARN:
            p.setPen(QPen(title_clr))
            ttc = f"{snap.time_to_collision:.2f}" if snap.time_to_collision < 100 else "∞"
            ttb = f"{snap.time_to_brake:.2f}" if snap.time_to_brake < 100 else "∞"
            p.drawText(QPointF(x, y), f"TTC: {ttc}s    TTB: {ttb}s")
        else:
            p.setPen(QPen(QColor(120, 120, 135)))
            p.drawText(QPointF(x, y), "TTC: —    TTB: —")

        y += 18
        p.setFont(self._font_small)

        if kmh < _MIN_SPEED_KMH:
            p.setPen(QPen(QColor(140, 140, 155)))
            p.drawText(QPointF(x, y), f"AEB inactive below {_MIN_SPEED_KMH:.0f} km/h")
            y += 15

        if snap.evasion_arc is not None and snap.aeb_state >= AEBState.WARN:
            p.setPen(QPen(_EVASION_CLR))
            kappa = snap.evasion_curvature
            if abs(kappa) < 1e-5:
                direction = "STRAIGHT"
            elif kappa > 0:
                direction = f"LEFT κ={kappa:.4f}"
            else:
                direction = f"RIGHT κ={abs(kappa):.4f}"
            p.drawText(QPointF(x, y), f"Evasion: {direction}")
            y += 15

        if ns > 0:
            p.setPen(QPen(_SUPPRESSED_CLR))
            p.drawText(QPointF(x, y), f"{ns} rear-approach vehicle(s) suppressed")
            y += 14

        if nb > 0:
            p.setPen(QPen(_BRAKE_SUPP_CLR))
            p.drawText(QPointF(x, y), f"{nb} threat(s): braking would worsen TTC")

        # --- Legend (bottom-left) -----------------------------------------
        self._draw_legend(p)

    # ---- legend ----------------------------------------------------------

    def _draw_legend(self, p: QPainter) -> None:
        lx = 10
        ly = self.height() - 110
        lw = 145
        lh = 105

        p.setPen(QPen(_HUD_BORDER, 1))
        p.setBrush(QBrush(_HUD_BG))
        p.drawRoundedRect(QRectF(lx, ly, lw, lh), 6, 6)

        p.setFont(self._font_label)
        items = [
            (_EGO_CLR, "Ego corridor"),
            (_SAFE_CLR, "Safe vehicle"),
            (_DANGER_CLR, "Threat"),
            (_WARN_CLR, "Warning"),
            (_BRAKE_SUPP_CLR, "Brake-suppressed"),
            (_SUPPRESSED_CLR, "Rear-suppressed"),
            (_EVASION_CLR, "Evasion path"),
        ]
        y = ly + 13
        for clr, label in items:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(clr))
            p.drawRoundedRect(QRectF(lx + 8, y - 5, 8, 8), 2, 2)
            p.setPen(QPen(_TEXT))
            p.drawText(QPointF(lx + 22, y + 2), label)
            y += 13