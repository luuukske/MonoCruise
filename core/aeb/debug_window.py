"""AEB debug visualisation — PySide6 top-down radar view (ego-locked)."""

from __future__ import annotations

import math
import logging

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF, QFont,
    QRadialGradient, QPaintEvent,
)
from PySide6.QtWidgets import QWidget

from core.thread_management.registry import registry
from .thread import AEBState, AEBSnapshot
from core.radar.traffic import ArcPath

logger = logging.getLogger(__name__)

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
_EVASION_FILTER_CLR = QColor(0, 200, 200)
_EVASION_FILTER_CORRIDOR = QColor(0, 200, 200, 25)
_EVASION_FILTERED_CLR = QColor(0, 200, 200, 140)
_TRAILER_CLR = QColor(180, 140, 80)
_EGO_TRAILER_CLR = QColor(55, 130, 215)
_ACC_LEAD_CLR = QColor(255, 80, 230)           # primary ACC lead — magenta
_ACC_CANDIDATE_CLR = QColor(180, 110, 220)     # top-3 non-primary
_ACC_CORRIDOR = QColor(255, 80, 230, 35)
_HIT_CLR = QColor(255, 30, 30)
_TEXT = QColor(200, 200, 215)
_HUD_BG = QColor(0, 0, 0, 160)
_HUD_BORDER = QColor(60, 60, 70)

_EGO_TRAILER_HALF_W = 1.25
_EGO_TRAILER_HALF_L = 6.8

_REFRESH_MS = 33
_PPM = 5.5
_ARC_SAMPLES = 20
_CORRIDOR_FADE_SEGMENTS = 16


def _w2e(wx: float, wz: float, ex: float, ez: float, ey: float) -> tuple[float, float]:
    dx = wx - ex
    dz = wz - ez
    c = math.cos(-ey)
    s = math.sin(-ey)
    return (-dx) * c - dz * s, (-dx) * s + dz * c


def _e2s(rx: float, rz: float, cx: float, cy: float) -> tuple[float, float]:
    return cx + rx * _PPM, cy - rz * _PPM


class AEBDebugWindow(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AEB Debug Radar")
        self.setMinimumSize(500, 500)
        self.resize(_WIN_W, _WIN_H)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(_REFRESH_MS)

        self._font_main = QFont("Segoe UI", 10)
        self._font_small = QFont("Segoe UI", 8)
        self._font_hud_title = QFont("Segoe UI Semibold", 11)
        self._font_label = QFont("Segoe UI", 7)

    @staticmethod
    def _fetch() -> AEBSnapshot | None:
        try:
            aeb = registry.get_thread("aeb_thread")
        except KeyError:
            return None
        if aeb is None or not aeb.is_alive():
            return None
        return aeb.data.snapshot

    @staticmethod
    def _fetch_acc() -> dict | None:
        """ACC snapshot: {enabled, lead_id, top_ids, top_leads, dist, rel, score,
        components, blinker, ego_kappa, corridor_half}."""
        try:
            acc = registry.get_thread("acc_thread")
        except KeyError:
            return None
        if acc is None or not acc.is_alive():
            return None
        try:
            with acc.data._lock:
                leads = list(acc.data.leads)
                top = [(l.vehicle.id, l.score, l.dist_m) for l in leads]
                components = dict(acc.data.debug_components)
                return {
                    "enabled": bool(acc.data.enabled),
                    "lead_id": int(acc.data.lead_id),
                    "top_ids": {int(t[0]) for t in top},
                    "top_leads": top,
                    "dist": float(acc.data.lead_dist_m),
                    "rel": float(acc.data.lead_rel_speed_ms),
                    "score": float(acc.data.lead_score),
                    "components": components,
                    "blinker": float(acc.data.debug_blinker),
                    "ego_kappa": float(acc.data.debug_ego_kappa),
                    "corridor_half": float(acc.data.debug_corridor_half),
                }
        except AttributeError:
            return None

    def _ws(self, wx: float, wz: float, ex: float, ez: float, ey: float) -> tuple[float, float]:
        rx, rz = _w2e(wx, wz, ex, ez, ey)
        return _e2s(rx, rz, self.width() / 2.0, self.height() / 2.0)

    def paintEvent(self, event: QPaintEvent) -> None:
        snap = self._fetch()
        acc = self._fetch_acc()
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

        for v in snap.vehicles:
            vid = v["vid"]
            is_supp = v.get("rear_suppressed", False)
            is_danger = vid in snap.colliding_ids
            is_brake_supp = vid in snap.braking_suppressed_ids
            is_evasion_filtered = vid in snap.evasion_filtered_ids

            is_acc_lead = acc is not None and acc["lead_id"] == vid
            is_acc_top = acc is not None and vid in acc["top_ids"] and not is_acc_lead

            if is_supp:
                body_clr, corr_clr = _SUPPRESSED_CLR, QColor(_SUPPRESSED_CLR)
            elif is_evasion_filtered:
                body_clr, corr_clr = _EVASION_FILTERED_CLR, QColor(_EVASION_FILTERED_CLR)
            elif is_danger and is_brake_supp:
                body_clr, corr_clr = _BRAKE_SUPP_CLR, QColor(_BRAKE_SUPP_CLR)
            elif is_danger:
                body_clr, corr_clr = _DANGER_CLR, QColor(_DANGER_CLR)
            elif is_acc_lead:
                body_clr, corr_clr = _ACC_LEAD_CLR, QColor(_ACC_LEAD_CLR)
            elif is_acc_top:
                body_clr, corr_clr = _ACC_CANDIDATE_CLR, QColor(_ACC_CANDIDATE_CLR)
            else:
                body_clr, corr_clr = _SAFE_CLR, QColor(_SAFE_CLR)

            arcs = snap.vehicle_arcs.get(vid)
            if arcs is not None:
                arc_list = arcs if isinstance(arcs, list) else [arcs]
                corr_clr.setAlpha(20)
                for arc in arc_list:
                    self._draw_arc_corridor(p, arc, ex, ez, ey, corr_clr)

            self._draw_vehicle_box(
                p, v["x"], v["z"], v["yaw"],
                v["half_w"], v["length"], v["is_tmp"],
                ex, ez, ey, body_clr,
            )

            sx, sy = self._ws(v["x"], v["z"], ex, ez, ey)
            spd = v.get("speed_kmh", 0.0)
            is_trailer_veh = v.get("is_trailer", False)
            kin_swapped = v.get("kinematics_swapped", False)
            is_tmp_veh = v.get("is_tmp", False)
            tag = "TR" if is_trailer_veh else ""
            tag += ("/TMP" if is_tmp_veh else "/AI") if tag else ("TMP" if is_tmp_veh else "AI")
            self._draw_label(p, sx, sy - 14, f"{tag} {spd:.0f}", body_clr)
            if is_trailer_veh:
                if kin_swapped:
                    kin_text, kin_clr = "kin:tractor", _SAFE_CLR
                elif not is_tmp_veh:
                    kin_text, kin_clr = "kin:raw (not TMP)", _DANGER_CLR
                else:
                    kin_text, kin_clr = "kin:raw (no tractor<30m)", _DANGER_CLR
                self._draw_label(p, sx, sy - 26, kin_text, kin_clr)
            self._draw_acc_components(p, sx, sy, vid, acc, body_clr)

            for tr in v.get("trailers", []):
                yaw = tr["yaw"]
                self._draw_vehicle_box(
                    p, tr["x"], tr["z"], yaw,
                    tr["half_w"], tr["length"], tr["is_tmp"],
                    ex, ez, ey, _TRAILER_CLR,
                )
                trsx, trsy = self._ws(tr["x"], tr["z"], ex, ez, ey)
                tr_spd = tr.get("speed_kmh", 0.0)
                self._draw_label(p, trsx, trsy - 14, f"TR {tr_spd:.0f}", _TRAILER_CLR)

        if snap.ego_arc is not None:
            self._draw_arc_corridor(p, snap.ego_arc, ex, ez, ey, _EGO_CORRIDOR)

        if snap.evasion_left_arc is not None:
            self._draw_arc_corridor(
                p, snap.evasion_left_arc, ex, ez, ey,
                _EVASION_FILTER_CORRIDOR,
                edge_color=_EVASION_FILTER_CLR, edge_width=1.0,
            )
        if snap.evasion_right_arc is not None:
            self._draw_arc_corridor(
                p, snap.evasion_right_arc, ex, ez, ey,
                _EVASION_FILTER_CORRIDOR,
                edge_color=_EVASION_FILTER_CLR, edge_width=1.0,
            )

        if snap.ego_has_trailer:
            fx = -math.sin(ey)
            fz = -math.cos(ey)
            reach = snap.ego_half_l + _EGO_TRAILER_HALF_L
            self._draw_ego_box(
                p, ex - fx * reach, ez - fz * reach, ey,
                _EGO_TRAILER_HALF_W, _EGO_TRAILER_HALF_L,
                ex, ez, ey, _EGO_TRAILER_CLR,
            )

        self._draw_ego_box(
            p, ex, ez, ey, snap.ego_half_w, snap.ego_half_l,
            ex, ez, ey, _EGO_CLR,
        )

        if snap.aeb_state >= AEBState.WARN and snap.time_to_collision < 100:
            self._draw_hit_marker(p, snap.hit_x, snap.hit_z, ex, ez, ey)

        self._draw_hud(p, snap)
        self._draw_acc_hud(p, acc)

        p.end()

    def _draw_acc_hud(self, p: QPainter, acc: dict | None) -> None:
        hud_w = 310
        hud_h = 122
        hud_x = 10
        hud_y = 195  # below main AEB HUD (175+10+10)

        p.setPen(QPen(_HUD_BORDER, 1))
        p.setBrush(QBrush(_HUD_BG))
        p.drawRoundedRect(QRectF(hud_x, hud_y, hud_w, hud_h), 8, 8)

        x = hud_x + 14
        y = hud_y + 20

        if acc is None:
            p.setFont(self._font_hud_title)
            p.setPen(QPen(QColor(120, 120, 135)))
            p.drawText(QPointF(x, y), "ACC thread not running")
            return

        title_clr = _ACC_LEAD_CLR if acc["enabled"] else QColor(120, 120, 135)
        p.setFont(self._font_hud_title)
        p.setPen(QPen(title_clr))
        state = "ACC tracking" if acc["enabled"] else "ACC disabled"
        p.drawText(QPointF(x, y), state)

        y += 20
        p.setFont(self._font_main)
        p.setPen(QPen(_TEXT))
        if acc["lead_id"] >= 0:
            p.drawText(
                QPointF(x, y),
                f"Lead #{acc['lead_id']}  d={acc['dist']:.1f} m  "
                f"rel={acc['rel']*3.6:+.0f} km/h  s={acc['score']:+.1f}",
            )
        else:
            p.drawText(QPointF(x, y), "Lead: —")

        y += 18
        p.setFont(self._font_small)
        n_top = len(acc["top_leads"])
        n_tracked = len(acc.get("components", {}))
        p.drawText(QPointF(x, y), f"in-lane: {n_top}   scored tracks: {n_tracked}")

        kappa = acc.get("ego_kappa", 0.0)
        if abs(kappa) > 1e-6:
            radius = 1.0 / abs(kappa)
            radius_str = f"{radius:.0f}m"
        else:
            radius_str = "∞"
        ch = acc.get("corridor_half", 0.0)
        kappa_clr = _WARN_CLR if (abs(kappa) > 1e-6 and 1.0 / abs(kappa) < 400.0) else _TEXT
        y += 14
        p.setPen(QPen(kappa_clr))
        p.drawText(
            QPointF(x, y),
            f"ego κ={kappa:+.4f} (R={radius_str})   ½lane={ch:.2f}m",
        )

        y += 14
        p.setPen(QPen(_TEXT))
        p.drawText(QPointF(x, y), f"blinker scalar: {acc.get('blinker', 0.0):+.2f}")

    def _draw_acc_components(
        self,
        p: QPainter,
        sx: float,
        sy: float,
        vid: int,
        acc: dict | None,
        body_clr: QColor,
    ) -> None:
        """Render the per-vehicle scoring breakdown next to the vehicle.

        Shows everything the ACC scorer saw for this id this frame so the
        operator can see *why* a vehicle is or is not being tracked: the
        four components, current accumulated score, in-path flag, lat/dist
        in the arc frame, and the lateral gate margin.
        """
        if acc is None:
            return
        comp = acc.get("components", {}).get(vid)
        if comp is None:
            return

        score = comp["score"]
        in_path = comp["in_path"]
        lat_margin = comp.get("lat_margin", 0.0)
        seen = comp.get("seen", True)
        lat = comp["lat"]

        score_clr = _SAFE_CLR if score > 0 else (_WARN_CLR if score > -2 else _DANGER_CLR)
        self._draw_label(p, sx, sy + 14, f"s{score:+.1f}", score_clr)

        flags = []
        flags.append("IN" if in_path else "out")
        flags.append(f"m{lat_margin:+.2f}")   # gate margin: positive = inside
        if not seen:
            flags.append("decay")
        flags_clr = _SAFE_CLR if (in_path and seen) else _WARN_CLR
        self._draw_label(p, sx, sy + 26, " ".join(flags), flags_clr)

        offset = comp["offset"]
        yaw_c = comp["yaw"]
        path = comp["path"]
        comp_clr = _TEXT if seen else _SUPPRESSED_CLR
        self._draw_label(
            p, sx, sy + 38,
            f"o{offset:+.2f} y{yaw_c:+.2f} p{path:+.2f}",
            comp_clr,
        )

        ofs = comp["offset_for_score"]
        yd = comp["yaw_diff_deg"]
        self._draw_label(
            p, sx, sy + 50,
            f"lat={lat:+.2f} ofs={ofs:+.2f} Δyaw={yd:+.0f}°",
            comp_clr,
        )

    def _draw_grid(self, p: QPainter, cx: float, cy: float) -> None:
        max_r = max(self.width(), self.height()) / _PPM + _GRID_STEP_MAJOR

        p.setPen(QPen(_GRID_MINOR, 0.5, Qt.DotLine))
        p.setBrush(Qt.NoBrush)
        d = _GRID_STEP_MINOR
        while d < max_r:
            if abs(d % _GRID_STEP_MAJOR) > 0.01:
                r = d * _PPM
                p.drawEllipse(QPointF(cx, cy), r, r)
            d += _GRID_STEP_MINOR

        p.setBrush(Qt.NoBrush)
        d = _GRID_STEP_MAJOR
        while d < max_r:
            r = d * _PPM
            p.setPen(QPen(_GRID_MAJOR, 1, Qt.DotLine))
            p.drawEllipse(QPointF(cx, cy), r, r)
            p.setPen(QPen(QColor(70, 70, 85)))
            p.setFont(self._font_label)
            p.drawText(QPointF(cx + 3, cy - r + 11), f"{d:.0f}m")
            d += _GRID_STEP_MAJOR

        p.setPen(QPen(_GRID_MAJOR, 0.5))
        p.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))
        p.drawLine(QPointF(0, cy), QPointF(self.width(), cy))

    def _draw_arc_corridor(
        self, p: QPainter,
        arc: ArcPath,
        ex: float, ez: float, ey: float,
        fill_color: QColor,
        edge_color: QColor | None = None,
        edge_width: float = 1.0,
    ) -> None:
        left, right = arc.sample_corridor(_ARC_SAMPLES)
        if len(left) < 2:
            return

        s_left = [self._ws(x, z, ex, ez, ey) for x, z in left]
        s_right = [self._ws(x, z, ex, ez, ey) for x, z in right]

        poly = QPolygonF()
        for sx, sy in s_left:
            poly.append(QPointF(sx, sy))
        for sx, sy in reversed(s_right):
            poly.append(QPointF(sx, sy))

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(fill_color))
        p.drawPolygon(poly)

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

    def _draw_vehicle_box(
        self, p: QPainter,
        wx: float, wz: float, yaw: float,
        hw: float, length: float, is_tmp: bool,
        ex: float, ez: float, ey: float,
        color: QColor,
    ) -> None:
        hl = length / 2.0
        corners_local = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]

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

        cx_w, cz_w = wx, wz
        flen = min(length * 0.4, 4.0)
        tip_x = wx - flen * math.sin(yaw)
        tip_z = wz - flen * math.cos(yaw)
        scx, scy = self._ws(cx_w, cz_w, ex, ez, ey)
        stx, sty = self._ws(tip_x, tip_z, ex, ez, ey)
        p.setPen(QPen(QColor(255, 255, 255, 160), 1.5))
        p.drawLine(QPointF(scx, scy), QPointF(stx, sty))

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

    def _draw_hit_marker(
        self, p: QPainter,
        hx: float, hz: float,
        ex: float, ez: float, ey: float,
    ) -> None:
        sx, sy = self._ws(hx, hz, ex, ez, ey)

        grad = QRadialGradient(QPointF(sx, sy), 18)
        grad.setColorAt(0.0, QColor(255, 50, 50, 180))
        grad.setColorAt(0.5, QColor(255, 50, 50, 60))
        grad.setColorAt(1.0, QColor(255, 50, 50, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPointF(sx, sy), 18, 18)

        r = 7
        p.setPen(QPen(_HIT_CLR, 2.5))
        p.drawLine(QPointF(sx - r, sy - r), QPointF(sx + r, sy + r))
        p.drawLine(QPointF(sx - r, sy + r), QPointF(sx + r, sy - r))

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(_HIT_CLR))
        p.drawEllipse(QPointF(sx, sy), 3, 3)

    def _draw_label(
        self, p: QPainter, sx: float, sy: float, text: str, color: QColor,
    ) -> None:
        p.setFont(self._font_label)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        bg = QColor(0, 0, 0, 130)
        rect = QRectF(sx - tw / 2 - 3, sy - th + 2, tw + 6, th + 1)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(rect, 3, 3)
        tc = QColor(color)
        tc.setAlpha(220)
        p.setPen(QPen(tc))
        p.drawText(QPointF(sx - tw / 2, sy), text)

    def _draw_hud(self, p: QPainter, snap: AEBSnapshot) -> None:
        hud_w = 310
        hud_h = 175
        hud_x = 10
        hud_y = 10

        p.setPen(QPen(_HUD_BORDER, 1))
        p.setBrush(QBrush(_HUD_BG))
        p.drawRoundedRect(QRectF(hud_x, hud_y, hud_w, hud_h), 8, 8)

        x = hud_x + 14
        y = hud_y + 22

        if snap.aeb_state == AEBState.BRAKE:
            title_clr = _DANGER_CLR
            title = "⚠ AEB EMERGENCY BRAKE"
        elif snap.aeb_state == AEBState.WARN:
            title_clr = _WARN_CLR
            title = "⚠ AEB WARNING"
        else:
            title_clr = _SAFE_CLR
            title = "✓ AEB STANDBY"

        strip_clr = QColor(title_clr)
        strip_clr.setAlpha(80)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(strip_clr))
        p.drawRoundedRect(QRectF(hud_x + 2, hud_y + 2, hud_w - 4, 24), 6, 6)

        p.setFont(self._font_hud_title)
        p.setPen(QPen(title_clr))
        p.drawText(QPointF(x, y), title)

        y += 26
        p.setFont(self._font_main)
        p.setPen(QPen(_TEXT))

        kmh = snap.ego_speed * 3.6
        tmp_sess = getattr(snap, "tmp_traffic_session", False)
        sess_txt = "TMP" if tmp_sess else "SP"
        p.drawText(QPointF(x, y), f"Speed: {kmh:.0f} km/h   Session: {sess_txt}")

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

        if ns > 0:
            p.setPen(QPen(_SUPPRESSED_CLR))
            p.drawText(QPointF(x, y), f"{ns} rear-approach vehicle(s) suppressed")
            y += 14

        if nb > 0:
            p.setPen(QPen(_BRAKE_SUPP_CLR))
            p.drawText(QPointF(x, y), f"{nb} threat(s): braking would worsen TTC")
            y += 14

        nef = len(snap.evasion_filtered_ids)
        if nef > 0:
            p.setPen(QPen(_EVASION_FILTER_CLR))
            p.drawText(QPointF(x, y), f"{nef} vehicle(s) evasion-filtered (corner/roadside)")

        self._draw_legend(p)

    def _draw_legend(self, p: QPainter) -> None:
        lx = 10
        ly = self.height() - 135
        lw = 145
        lh = 130

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
            (_EVASION_FILTER_CLR, "Evasion-filtered"),
            (_ACC_LEAD_CLR, "ACC lead"),
            (_ACC_CANDIDATE_CLR, "ACC candidate"),
        ]
        y = ly + 13
        for clr, label in items:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(clr))
            p.drawRoundedRect(QRectF(lx + 8, y - 5, 8, 8), 2, 2)
            p.setPen(QPen(_TEXT))
            p.drawText(QPointF(lx + 22, y + 2), label)
            y += 13
