"""
Cruise control display panel using PySide6.

Single translucent window rendered entirely via QPainter.
Thread-safe: update() can be called from any thread without dropping changes.
High-frequency calls coalesce: pending fields merge under a lock and at most
one queued flush runs per GUI burst, avoiding Qt event-queue overload.

The QWidget must be created and driven from the Qt main thread (QApplication
owner). A separate Python thread cannot host this window; worker threads
only push state via cc_panel.update(), which marshals to the GUI via signals.
"""

import logging
import math
import os
import sys
import time
import threading
from typing import Callable

from PySide6.QtWidgets import QWidget, QApplication
if not __name__ == "__main__":
    from core.settings import Settings
from PySide6.QtCore import Qt, QTimer, Signal, QByteArray
from PySide6.QtGui import (
    QPainter,
    QColor,
    QFont,
    QFontMetrics,
    QPixmap,
    QImage,
    QPainterPath,
    QCursor,
    QGuiApplication,
    qAlpha,
    qBlue,
    qGreen,
    qRed,
    qRgba,
)

SPEEDLIMITER_COLOR = "#008B00"
CRUISECONTROL_COLOR = "#4876FF"
DISABLED_COLOR = "#F1F1F1"
AEB_COLOR = "#FF0000"

# Faded “last locked” vehicle hint when ACC lines show but no target is locked.
_LAST_LOCKED_VEHICLE_OPACITY = 0.30
# Logical px: expand vehicle alpha by this radius to cut holes in distance lines (shape outline).
_LAST_LOCKED_LINE_GAP_PX = 1.5

logger = logging.getLogger(__name__)

def _cursor_pos() -> tuple[int, int]:
    pos = QCursor.pos()
    return pos.x(), pos.y()

# ---------------------------------------------------------------------------
# Internal rendering widget – every method runs on the Qt main thread.
# ---------------------------------------------------------------------------

class _PanelWidget(QWidget):
    _flush_coalesced_sig = Signal()
    _show_sig = Signal()
    _hide_sig = Signal()
    _stop_sig = Signal()
    _move_sig = Signal(int, int)
    _scale_sig = Signal(float)
    _bg_opacity_sig = Signal(float)

    def __init__(self, panel: "cc_panel"):
        super().__init__()
        self._p = panel
        self._drag_offset: tuple[int, int] | None = None
        self._tint_cache: dict[tuple, QPixmap] = {}
        self._scaled_icon_lines_cache: dict[tuple, QPixmap] = {}
        self._ghost_raw_cache: dict[tuple, QPixmap] = {}
        self._ghost_scaled_cache: dict[tuple, QPixmap] = {}
        self._ghost_exact_alpha_cache: dict[tuple, QPixmap] = {}
        self._vehicle_cutout_mask_cache: dict[tuple, QPixmap] = {}
        # Final ACC line layer (after vehicle cutout); bounded, evict oldest.
        self._acc_lines_layer_cache: dict[tuple, QPixmap] = {}
        self._acc_lines_layer_cache_max = 8

        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_tick)

        self._setup_window()

        # QueuedConnection avoids deadlock when update() runs on the GUI thread
        # while holding _pending_lock (AutoConnection would run the slot inline).
        self._flush_coalesced_sig.connect(
            self._on_flush_coalesced, Qt.ConnectionType.QueuedConnection
        )
        self._show_sig.connect(self._on_show)
        self._hide_sig.connect(self._on_hide)
        self._stop_sig.connect(self._on_stop)
        self._move_sig.connect(self._on_move)
        self._scale_sig.connect(self._on_scale)
        self._bg_opacity_sig.connect(self._on_bg_opacity)

    # -- window --

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        p = self._p
        self.setFixedSize(p._panel_w, p._panel_h)
        if p._start_x is not None and p._start_y is not None:
            self.move(p._start_x, p._start_y)

    # -- signal slots --

    def _on_show(self):
        self.show()
        self.raise_()

    def _on_hide(self):
        self.hide()

    def _on_stop(self):
        self._p.running = False
        self._p._blink_running = False
        self._blink_timer.stop()
        self.close()

    def _on_move(self, x: int, y: int):
        self.move(x, y)

    def _on_scale(self, s: float):
        p = self._p
        p._scale_mult = s
        p._panel_w = int(315 * s)
        p._panel_h = int(100 * s)
        p._radius = int(30 * s)
        p._icon_spacing = int(20 * s)
        f = QFont("Arial")
        f.setBold(True)
        f.setPixelSize(max(1, int(40 * s)))
        p._font = f
        self.setFixedSize(p._panel_w, p._panel_h)
        p._icon_cache.clear()
        self._tint_cache.clear()
        self._scaled_icon_lines_cache.clear()
        self._ghost_raw_cache.clear()
        self._ghost_scaled_cache.clear()
        self._ghost_exact_alpha_cache.clear()
        self._vehicle_cutout_mask_cache.clear()
        self._acc_lines_layer_cache.clear()
        p._current_icon = self._load_icon()
        self.update()

    def _on_bg_opacity(self, v: float):
        self._p._bg_opacity = v
        self.update()

    def _on_flush_coalesced(self):
        """Apply merged pending updates (runs on Qt GUI thread)."""
        p = self._p
        with p._pending_lock:
            merged = p._pending
            p._pending = {}
            p._coalesce_armed = False
        if merged:
            self._on_update(merged)

    def _on_update(self, changes: dict):
        p = self._p
        complete = changes.pop("_complete_update", False)

        old = {
            "cc_mode": p._cc_mode,
            "cc_enabled": p._cc_enabled,
            "AEB_warn": p._AEB_warn,
            "acc_locked": p._acc_locked,
            "acc_enabled": p._acc_enabled,
            "acc_truck": p._acc_truck,
            "distance_to_lead": p._distance_to_lead,
            "text_content": p._text_content,
        }

        for key, val in changes.items():
            setattr(p, f"_{key}", val)

        if p._acc_locked:
            p._last_locked_acc_truck = bool(p._acc_truck)

        needs_icon = complete or any(
            getattr(p, f"_{k}") != old[k]
            for k in ("cc_mode", "AEB_warn", "acc_locked", "acc_enabled", "acc_truck")
        ) or (p._acc_locked and p._distance_to_lead != old["distance_to_lead"])

        needs_color = (
            complete
            or p._cc_enabled != old["cc_enabled"]
            or p._AEB_warn != old["AEB_warn"]
        )

        # AEB state transitions
        if "AEB_warn" in changes or complete:
            now = time.time()
            if p._AEB_warn:
                p._last_AEB_warn_true = now
                p._AEB_warn_off_time = 0.0
            elif old["AEB_warn"] and not p._AEB_warn:
                p._AEB_warn_off_time = now

            if self._is_aeb_active() and not p._blink_running:
                self._start_blinking()

        if needs_icon:
            p._current_icon = self._load_icon()
            self._tint_cache.clear()
            self._scaled_icon_lines_cache.clear()
            self._ghost_raw_cache.clear()
            self._ghost_scaled_cache.clear()
            self._ghost_exact_alpha_cache.clear()
            self._vehicle_cutout_mask_cache.clear()
            self._acc_lines_layer_cache.clear()

        if needs_color or needs_icon:
            p._text_color = cc_panel._color_for_mode(p._cc_mode, p._cc_enabled)

        self.update()
        self.raise_()

    # -- AEB blink --

    def _is_aeb_active(self) -> bool:
        p = self._p
        if p._AEB_warn:
            return True
        if p._AEB_warn_off_time > 0:
            return (time.time() - p._AEB_warn_off_time) < p._time_after_AEB_warn
        return False

    def _start_blinking(self):
        p = self._p
        p._blink_running = True
        p._hide_icon = False
        self._blink_timer.start(p._blinker_t_on_ms)

    def _stop_blinking(self):
        p = self._p
        p._blink_running = False
        p._hide_icon = False
        p._icon_cache.clear()
        self._tint_cache.clear()
        self._scaled_icon_lines_cache.clear()
        self._ghost_raw_cache.clear()
        self._ghost_scaled_cache.clear()
        self._ghost_exact_alpha_cache.clear()
        self._vehicle_cutout_mask_cache.clear()
        self._acc_lines_layer_cache.clear()
        p._current_icon = self._load_icon()
        self._blink_timer.stop()
        self.update()

    def _blink_tick(self):
        if not self._is_aeb_active():
            self._stop_blinking()
            return
        p = self._p
        p._hide_icon = not p._hide_icon
        self._blink_timer.setInterval(
            p._blinker_t_off_ms if p._hide_icon else p._blinker_t_on_ms
        )
        self.update()

    # -- icon loading --

    def _load_icon(self) -> QPixmap:
        p = self._p
        is_aeb = p._blink_running or p._AEB_warn
        dpr = self.devicePixelRatio()

        key = (
            p._cc_mode, p._scale_mult,
            is_aeb, p._acc_locked, p._distance_to_lead,
            p._acc_enabled, p._acc_truck, dpr,
        )
        cached = p._icon_cache.get(key)
        if cached is not None:
            return cached

        # TODO(ACC): choose car1 vs truck1 from lead vehicle type when ACC is implemented.
        if is_aeb or (
            p._acc_locked and not p._acc_truck
            and p._cc_mode == "Cruise control" and p._acc_enabled
        ):
            fname = "car1.png"
        elif (
            p._acc_locked and p._acc_truck
            and p._cc_mode == "Cruise control" and p._acc_enabled
        ):
            fname = "truck1.png"
        elif p._cc_mode == "Speed limiter":
            fname = "speed limiter.png"
        elif p._cc_mode == "Cruise control":
            fname = "cruise control.png"
        else:
            fname = None

        fm = QFontMetrics(p._font)
        icon_sz = int(fm.height() * 1.8 * 0.85)
        phys = int(icon_sz * dpr)

        if fname:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "assets", fname
            )
            pm = QPixmap(path)
            if pm.isNull():
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    img = QImage()
                    if img.loadFromData(QByteArray(data)):
                        pm = QPixmap.fromImage(img)
                except Exception:
                    logger.exception(
                        "cc_panel.main._PanelWidget._load_icon: failed to load asset from data: %s",
                        path,
                    )
            if pm.isNull():
                logger.warning(
                    "cc_panel.main._PanelWidget._load_icon: using placeholder for missing/invalid asset: %s",
                    path,
                )
                pm = self._placeholder(icon_sz, dpr)
            else:
                pm = pm.scaled(
                    phys, phys,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                pm.setDevicePixelRatio(dpr)
        else:
            pm = self._placeholder(icon_sz, dpr)

        p._icon_cache[key] = pm
        return pm

    @staticmethod
    def _placeholder(sz: int, dpr: float = 1.0) -> QPixmap:
        try:
            phys = int(sz * dpr)
            pm = QPixmap(phys, phys)
            pm.fill(Qt.GlobalColor.transparent)
            pa = QPainter(pm)
            pa.setPen(QColor(255, 255, 255))
            m = max(2, phys // 8)
            pa.drawEllipse(m, m, phys - 2 * m, phys - 2 * m)
            pa.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "?")
            pa.end()
            pm.setDevicePixelRatio(dpr)
            return pm
        except Exception:
            logger.exception("cc_panel.main._PanelWidget._placeholder: failed to create placeholder pixmap")
            raise

    @staticmethod
    def _dilate_alpha_channel_sep(src: QImage, radius: int) -> QImage:
        """Separable max-filter dilation on alpha (fast O(n·r) vs O(n·r²))."""
        if radius <= 0 or src.isNull():
            return src
        w, h = src.width(), src.height()
        tmp = QImage(w, h, QImage.Format.Format_ARGB32)
        out = QImage(w, h, QImage.Format.Format_ARGB32)
        tmp.fill(0)
        out.fill(0)
        for y in range(h):
            for x in range(w):
                amax = 0
                for dx in range(-radius, radius + 1):
                    sx = x + dx
                    if 0 <= sx < w:
                        a = src.pixelColor(sx, y).alpha()
                        if a > amax:
                            amax = a
                if amax > 0:
                    tmp.setPixelColor(x, y, QColor(255, 255, 255, amax))
        for y in range(h):
            for x in range(w):
                amax = 0
                for dy in range(-radius, radius + 1):
                    sy = y + dy
                    if 0 <= sy < h:
                        a = tmp.pixelColor(x, sy).alpha()
                        if a > amax:
                            amax = a
                if amax > 0:
                    out.setPixelColor(x, y, QColor(255, 255, 255, amax))
        return out

    def _load_vehicle_asset_scaled(self, truck: bool, sh: int, dpr: float) -> QPixmap:
        key = (truck, sh, dpr)
        hit = self._ghost_raw_cache.get(key)
        if hit is not None:
            return hit
        fname = "truck1.png" if truck else "car1.png"
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", fname
        )
        pm = QPixmap(path)
        if pm.isNull():
            try:
                with open(path, "rb") as f:
                    data = f.read()
                img = QImage()
                if img.loadFromData(QByteArray(data)):
                    pm = QPixmap.fromImage(img)
            except Exception:
                logger.exception(
                    "cc_panel.main._PanelWidget._load_vehicle_asset_scaled: load failed: %s",
                    path,
                )
        phys = max(1, int(sh * dpr))
        if pm.isNull():
            pm = self._placeholder(sh, dpr)
        else:
            pm = pm.scaled(
                phys, phys,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            pm.setDevicePixelRatio(dpr)
        self._ghost_raw_cache[key] = pm
        return pm

    def _load_vehicle_asset_scaled_tinted(
        self,
        truck: bool,
        sh: int,
        dpr: float,
        line_color: QColor,
    ) -> QPixmap:
        tkey = (truck, sh, dpr, line_color.name())
        hit = self._ghost_scaled_cache.get(tkey)
        if hit is not None:
            return hit
        raw = self._load_vehicle_asset_scaled(truck, sh, dpr)
        tinted = self._tinted(raw, line_color)
        if tinted is not None and not tinted.isNull():
            self._ghost_scaled_cache[tkey] = tinted
        return tinted if tinted is not None else raw

    def _line_cutout_mask_from_vehicle_pixmap(
        self,
        scaled_vehicle_raw: QPixmap,
        dpr: float,
        cache_key: tuple,
        r_phys: int,
    ) -> QPixmap:
        """
        Expanded vehicle alpha from untinted pixmap (stable silhouette).
        White + alpha for DestinationOut. Draw at (vx - pad_log, vy - pad_log), pad_log = r_phys/dpr.
        """
        if scaled_vehicle_raw.isNull():
            return scaled_vehicle_raw
        hit = self._vehicle_cutout_mask_cache.get(cache_key + (r_phys,))
        if hit is not None:
            return hit

        img = scaled_vehicle_raw.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        if img.isNull():
            return QPixmap()

        w, h = img.width(), img.height()
        padded = QImage(w + 2 * r_phys, h + 2 * r_phys, QImage.Format.Format_ARGB32)
        padded.fill(0)
        for yy in range(h):
            for xx in range(w):
                padded.setPixelColor(xx + r_phys, yy + r_phys, img.pixelColor(xx, yy))

        dil = self._dilate_alpha_channel_sep(padded, r_phys)
        out = QPixmap.fromImage(dil)
        out.setDevicePixelRatio(dpr)
        self._vehicle_cutout_mask_cache[cache_key + (r_phys,)] = out
        return out

    def _exact_vehicle_alpha_mask(
        self, scaled_vehicle_raw: QPixmap, dpr: float, cache_pk: tuple
    ) -> QPixmap:
        """White with source alpha only — punches lines under the glyph exactly."""
        hit = self._ghost_exact_alpha_cache.get(cache_pk)
        if hit is not None:
            return hit
        if scaled_vehicle_raw.isNull():
            return scaled_vehicle_raw
        img = scaled_vehicle_raw.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        if img.isNull():
            return QPixmap()
        w, h = img.width(), img.height()
        out_img = QImage(w, h, QImage.Format.Format_ARGB32)
        out_img.fill(0)
        for yy in range(h):
            for xx in range(w):
                a = img.pixelColor(xx, yy).alpha()
                if a > 0:
                    out_img.setPixelColor(xx, yy, QColor(255, 255, 255, a))
        pm = QPixmap.fromImage(out_img)
        pm.setDevicePixelRatio(dpr)
        self._ghost_exact_alpha_cache[cache_pk] = pm
        return pm

    @staticmethod
    def _stamp_alpha_max(dst: QImage, src: QImage, x0: int, y0: int) -> None:
        """Merge src alpha into dst (white, A=max) at offset (x0,y0), clipped."""
        if dst.isNull() or src.isNull():
            return
        dw, dh = dst.width(), dst.height()
        sw, sh = src.width(), src.height()
        y_start = max(0, -y0)
        y_end = min(sh, dh - y0)
        x_start = max(0, -x0)
        x_end = min(sw, dw - x0)
        for y in range(y_start, y_end):
            ly = y0 + y
            for x in range(x_start, x_end):
                lx = x0 + x
                sa = qAlpha(src.pixel(x, y))
                if sa <= 0:
                    continue
                da = qAlpha(dst.pixel(lx, ly))
                if sa > da:
                    dst.setPixel(lx, ly, qRgba(255, 255, 255, sa))

    @staticmethod
    def _cutout_roi(
        lw_img: int,
        lh_img: int,
        vx_px: int,
        vy_px: int,
        ex_w: int,
        ex_h: int,
        mx_px: int,
        my_px: int,
        di_w: int,
        di_h: int,
        margin: int = 3,
    ) -> tuple[int, int, int, int]:
        """Inclusive pixel bounds to process for vehicle + dilated cutout (+ AA margin)."""
        if lw_img <= 0 or lh_img <= 0:
            return 0, 0, 0, 0
        x0 = min(vx_px, mx_px) - margin
        y0 = min(vy_px, my_px) - margin
        x1 = max(vx_px + ex_w, mx_px + di_w) + margin
        y1 = max(vy_px + ex_h, my_px + di_h) + margin
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(lw_img - 1, x1)
        y1 = min(lh_img - 1, y1)
        if x0 > x1 or y0 > y1:
            return 0, 0, max(0, lw_img - 1), max(0, lh_img - 1)
        return x0, y0, x1, y1

    @staticmethod
    def _multiply_line_alpha_by_mask(
        lines: QImage,
        mask: QImage,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> None:
        """
        For each pixel in [x0,x1]×[y0,y1]: lines_alpha *= (255 - mask_alpha) / 255.
        Uses QRgb + setPixel (faster than pixelColor).
        """
        if lines.isNull() or mask.isNull():
            return
        w, h = lines.width(), lines.height()
        if mask.width() != w or mask.height() != h:
            return
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0, min(x1, w - 1))
        y1 = max(y0, min(y1, h - 1))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                m = qAlpha(mask.pixel(x, y))
                if m == 0:
                    continue
                p = lines.pixel(x, y)
                ca = qAlpha(p)
                if ca <= 0:
                    continue
                na = (ca * (255 - m) + 127) // 255
                if na <= 0:
                    lines.setPixel(x, y, qRgba(0, 0, 0, 0))
                else:
                    nr = (qRed(p) * na + ca // 2) // ca
                    ng = (qGreen(p) * na + ca // 2) // ca
                    nb = (qBlue(p) * na + ca // 2) // ca
                    lines.setPixel(x, y, qRgba(nr, ng, nb, na))

    @staticmethod
    def _distance_lines_bounds(
        ax: int,
        ay: int,
        area_sz: int,
        num_lines: int,
        lw: int,
        ls: int,
        sc: float,
        car_up_lines: int,
        lines_start_rel: int,
    ) -> tuple[float, float, float, float]:
        lmin_x = float("inf")
        lmax_x = float("-inf")
        lmin_y = float("inf")
        lmax_y = float("-inf")
        for i in range(num_lines):
            ly = lines_start_rel + i * (lw + ls)
            indent = -2 * (i - (num_lines - 1)) * sc
            x0 = ax + indent
            x1 = ax + area_sz + 2 * (i - (num_lines - 1)) * sc - 1
            y0 = ay + ly + 1 - car_up_lines
            y1 = y0 + lw
            lmin_x = min(lmin_x, x0)
            lmax_x = max(lmax_x, x1)
            lmin_y = min(lmin_y, y0)
            lmax_y = max(lmax_y, y1)
        return lmin_x, lmax_x, lmin_y, lmax_y

    def _acc_lines_layer_cache_put(self, key: tuple, pm: QPixmap) -> None:
        c = self._acc_lines_layer_cache
        if len(c) >= self._acc_lines_layer_cache_max:
            c.pop(next(iter(c)))
        c[key] = pm

    def _composite_acc_lines_with_vehicle_cutout(
        self,
        pa: QPainter,
        ax: int,
        ay: int,
        area_sz: int,
        sc: float,
        line_color: QColor,
        car_up_lines: int,
        lines_start_rel: int,
        n: int,
        lw: int,
        ls: int,
        num_lines: int,
        vx_i: int,
        vy_i: int,
        cutout_shape_pm: QPixmap,
        mask_cache_pk: tuple,
        vehicle_draw_pm: QPixmap,
        vehicle_opacity: float,
        draw_all_lines: Callable[[QPainter, int], None],
    ) -> None:
        """
        Draw distance lines, subtract alpha under vehicle silhouette + gap, then draw vehicle.
        cutout_shape_pm supplies alpha (tinted is fine; SourceIn preserves silhouette).
        """
        if cutout_shape_pm.isNull() or vehicle_draw_pm.isNull():
            draw_all_lines(pa, car_up_lines)
            if not vehicle_draw_pm.isNull():
                pa.save()
                pa.setOpacity(vehicle_opacity)
                pa.drawPixmap(vx_i, vy_i, vehicle_draw_pm)
                pa.restore()
            return

        dpr = self.devicePixelRatio()
        gap = max(1, int(round(_LAST_LOCKED_LINE_GAP_PX * sc)))
        r_phys = max(1, int(math.ceil(float(gap) * float(dpr))))
        pad_log = r_phys / float(dpr)

        mx = float(vx_i) - pad_log
        my = float(vy_i) - pad_log
        cw = cutout_shape_pm.width()
        ch = cutout_shape_pm.height()
        mask_lw = int(math.ceil(cw + 2.0 * pad_log))
        mask_lh = int(math.ceil(ch + 2.0 * pad_log))

        lmin_x, lmax_x, lmin_y, lmax_y = self._distance_lines_bounds(
            ax,
            ay,
            area_sz,
            num_lines,
            lw,
            ls,
            sc,
            car_up_lines,
            lines_start_rel,
        )
        gmin_x = min(lmin_x, mx)
        gmax_x = max(lmax_x, mx + mask_lw)
        gmin_y = min(lmin_y, my)
        gmax_y = max(lmax_y, my + mask_lh)

        origin_x = int(math.floor(gmin_x))
        origin_y = int(math.floor(gmin_y))
        layer_w = max(1, int(math.ceil(gmax_x - origin_x)) + 1)
        layer_h = max(1, int(math.ceil(gmax_y - origin_y)) + 1)

        shape_ck = cutout_shape_pm.cacheKey()
        layer_key = (
            origin_x,
            origin_y,
            layer_w,
            layer_h,
            int(round(dpr * 4096)),
            ax,
            ay,
            area_sz,
            int(round(sc * 8192)),
            car_up_lines,
            lines_start_rel,
            n,
            lw,
            ls,
            num_lines,
            line_color.rgba(),
            vx_i,
            vy_i,
            r_phys,
            shape_ck,
        )
        hit_layer = self._acc_lines_layer_cache.get(layer_key)
        if hit_layer is not None and not hit_layer.isNull():
            lines_pm = hit_layer
        else:
            mask_pm = self._line_cutout_mask_from_vehicle_pixmap(
                cutout_shape_pm, dpr, mask_cache_pk, r_phys
            )
            exact_pm = self._exact_vehicle_alpha_mask(
                cutout_shape_pm, dpr, mask_cache_pk
            )
            lines_pm = QPixmap(
                max(1, int(layer_w * dpr)),
                max(1, int(layer_h * dpr)),
            )
            lines_pm.fill(Qt.GlobalColor.transparent)
            lines_pm.setDevicePixelRatio(dpr)

            lp = QPainter(lines_pm)
            lp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            lp.translate(-origin_x, -origin_y)
            draw_all_lines(lp, car_up_lines)
            lp.end()

            layer_img = lines_pm.toImage().convertToFormat(
                QImage.Format.Format_ARGB32
            )
            if not layer_img.isNull():
                comb = QImage(
                    layer_img.width(),
                    layer_img.height(),
                    QImage.Format.Format_ARGB32,
                )
                comb.fill(0)
                vx_px = int(round((vx_i - origin_x) * dpr))
                vy_px = int(round((vy_i - origin_y) * dpr))
                mx_px = vx_px - r_phys
                my_px = vy_px - r_phys
                ex_img = exact_pm.toImage().convertToFormat(
                    QImage.Format.Format_ARGB32
                )
                di_img = mask_pm.toImage().convertToFormat(
                    QImage.Format.Format_ARGB32
                )
                ex_ok = not ex_img.isNull()
                di_ok = not di_img.isNull()
                if ex_ok:
                    self._stamp_alpha_max(comb, ex_img, vx_px, vy_px)
                if di_ok:
                    self._stamp_alpha_max(comb, di_img, mx_px, my_px)
                if ex_ok or di_ok:
                    rx0, ry0, rx1, ry1 = self._cutout_roi(
                        layer_img.width(),
                        layer_img.height(),
                        vx_px,
                        vy_px,
                        ex_img.width() if ex_ok else 0,
                        ex_img.height() if ex_ok else 0,
                        mx_px,
                        my_px,
                        di_img.width() if di_ok else 0,
                        di_img.height() if di_ok else 0,
                    )
                    self._multiply_line_alpha_by_mask(
                        layer_img, comb, rx0, ry0, rx1, ry1
                    )
                lines_pm = QPixmap.fromImage(layer_img)
                lines_pm.setDevicePixelRatio(dpr)
                self._acc_lines_layer_cache_put(layer_key, lines_pm)

        pa.drawPixmap(origin_x, origin_y, lines_pm)

        pa.save()
        pa.setOpacity(vehicle_opacity)
        pa.drawPixmap(vx_i, vy_i, vehicle_draw_pm)
        pa.restore()

    # -- tinting --

    def _tinted(self, pm: QPixmap, color: QColor) -> QPixmap:
        if pm is None or pm.isNull():
            return pm
        key = (id(pm), color.name())
        hit = self._tint_cache.get(key)
        if hit is not None:
            return hit
        try:
            out = QPixmap(pm.size())
            out.setDevicePixelRatio(pm.devicePixelRatio())
            out.fill(Qt.GlobalColor.transparent)
            pa = QPainter(out)
            pa.drawPixmap(0, 0, pm)
            pa.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            pa.fillRect(out.rect(), color)
            pa.end()
            self._tint_cache[key] = out
            return out
        except Exception:
            logger.exception("cc_panel.main._PanelWidget._tinted: tint composition failed")
            raise

    # -- painting --

    def paintEvent(self, _event):
        try:
            self._paint()
        except Exception:
            logger.exception("cc_panel.main._PanelWidget.paintEvent: paint failed")

    def _paint(self):
        p = self._p
        pa = QPainter(self)
        pa.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = p._panel_w, p._panel_h
        sc = p._scale_mult

        # background
        bg = QPainterPath()
        bg.addRoundedRect(0.0, 0.0, float(w), float(h),
                          float(p._radius), float(p._radius))
        pa.fillPath(bg, QColor(0, 0, 0, int(255 * p._bg_opacity)))

        # layout metrics
        fm = QFontMetrics(p._font)
        tw = fm.horizontalAdvance(p._text_content)
        icon_sz = int(fm.height() * 1.8 * 0.85)
        right_m = int(20 * sc)

        icon_x = w - icon_sz - right_m
        now = time.time()
        base_acc_active = (
            p._acc_enabled
            and p._cc_mode == "Cruise control"
            and not p._AEB_warn
            and not p._blink_running
        )
        acc_locked_now = p._acc_locked
        # Show distance lines whenever ACC is enabled (including pre-lock).
        # Keep them suppressed during AEB blink/warn for clarity.
        show_lines = base_acc_active
        icon_y = (h - icon_sz) // 2
        ghost_truck_slot = (
            show_lines
            and not acc_locked_now
            and (p._last_locked_acc_truck is True)
        )
        if show_lines and not (acc_locked_now and p._acc_truck) and not ghost_truck_slot:
            icon_y -= int(4 * sc)

        text_x = icon_x - p._icon_spacing - tw
        text_y_baseline = (h - fm.height()) // 2 + fm.ascent()

        # text
        pa.setFont(p._font)
        pa.setPen(p._text_color)
        pa.drawText(int(text_x), int(text_y_baseline), p._text_content)

        # icon
        icon = p._current_icon
        if icon and not icon.isNull() and not p._hide_icon:
            ic = QColor(AEB_COLOR) if self._is_aeb_active() else p._text_color
            tinted = self._tinted(icon, ic)

            if show_lines:
                # TODO(ACC): gap / lead visualization — fed by ACC when implemented.
                # While ACC is locked, show both vehicle and lines. Otherwise,
                # show only lines.
                if acc_locked_now:
                    lines_distance = p._distance_to_lead
                    lines_truck = p._acc_truck
                    draw_vehicle = True
                else:
                    # When awaiting lock (or in hold), keep reflecting the current
                    # distance setting so user changes are immediately visible.
                    lines_distance = p._distance_to_lead
                    lines_truck = p._acc_truck
                    draw_vehicle = False
                self._paint_icon_lines(
                    pa, tinted, icon_x, icon_y, icon_sz, sc, ic,
                    lines_distance, lines_truck, draw_vehicle,
                    p._last_locked_acc_truck,
                )
            else:
                pa.drawPixmap(int(icon_x), int(icon_y), tinted)

        pa.end()

    def _paint_icon_lines(
        self,
        pa: QPainter,
        icon_pm: QPixmap,
        ax: int,
        ay: int,
        area_sz: int,
        sc: float,
        line_color: QColor,
        distance_to_lead: int,
        acc_truck: bool,
        draw_vehicle: bool = True,
        last_locked_truck: bool | None = None,
    ):
        n = max(1, min(4, distance_to_lead))
        lw = int(4 * sc)
        ls = int(3 * sc)
        num_lines = 4
        lines_h_block = num_lines * lw + (num_lines - 1) * ls
        lines_h_icon = n * lw + (n - 1) * ls

        need_ghost = not draw_vehicle
        ghost_is_truck = need_ghost and (last_locked_truck is True)
        if need_ghost:
            car_up_lines = int(6 * sc) if not ghost_is_truck else 0
        else:
            car_up_lines = int(6 * sc) if not acc_truck or not draw_vehicle else 0

        sh = area_sz - lines_h_icon - ls - int((3 + 7 * (not acc_truck)) * sc)
        if sh % 2 == area_sz % 2:
            sh += 1
        sh = max(1, sh)

        lines_start_rel = area_sz - lines_h_block - 1
        center_y_icon = (area_sz - lines_h_icon - 1) / 2
        iy = int(center_y_icon - sh / 2) + int((1.5 + 8 * (not acc_truck)) * sc)
        ix = (area_sz - sh) // 2

        if need_ghost:
            sh_v = area_sz - lines_h_icon - ls - int(
                (3 + 7 * (not ghost_is_truck)) * sc
            )
            if sh_v % 2 == area_sz % 2:
                sh_v += 1
            sh_v = max(1, sh_v)
            iy_v = int(center_y_icon - sh_v / 2) + int(
                (1.5 + 8 * (not ghost_is_truck)) * sc
            )
            ix_v = (area_sz - sh_v) // 2
            car_up_v = int(6 * sc) if not ghost_is_truck else 0
        else:
            sh_v = ix_v = iy_v = car_up_v = 0

        def draw_line_at(
            target_pa: QPainter, car_up_off: int, i: int, color: QColor
        ) -> None:
            ly = lines_start_rel + i * (lw + ls)
            indent = -2 * (i - (num_lines - 1)) * sc
            x0 = ax + indent
            x1 = ax + area_sz + 2 * (i - (num_lines - 1)) * sc - 1
            lp = QPainterPath()
            lp.addRoundedRect(
                float(x0), float(ay + ly + 1 - car_up_off),
                float(x1 - x0), float(lw),
                lw / 2.0, lw / 2.0,
            )
            target_pa.fillPath(lp, color)

        def draw_all_lines(target_pa: QPainter, car_up_off: int) -> None:
            target_pa.setPen(Qt.PenStyle.NoPen)
            for i in range(num_lines - n, num_lines):
                draw_line_at(target_pa, car_up_off, i, line_color)
            for i in range(0, num_lines - n):
                opacity = 0.35 * 0.6 ** (num_lines - n - i) + 0.05
                inactive_color = QColor(line_color)
                inactive_color.setAlphaF(opacity)
                draw_line_at(target_pa, car_up_off, i, inactive_color)

        if need_ghost:
            dpr_g = self.devicePixelRatio()
            scaled_raw = self._load_vehicle_asset_scaled(ghost_is_truck, sh_v, dpr_g)
            scaled_v = self._load_vehicle_asset_scaled_tinted(
                ghost_is_truck, sh_v, dpr_g, line_color
            )
            vx_i = int(ax + ix_v)
            vy_i = int(ay + iy_v - car_up_v)
            self._composite_acc_lines_with_vehicle_cutout(
                pa,
                ax,
                ay,
                area_sz,
                sc,
                line_color,
                car_up_lines,
                lines_start_rel,
                n,
                lw,
                ls,
                num_lines,
                vx_i,
                vy_i,
                scaled_raw,
                (ghost_is_truck, sh_v, dpr_g),
                scaled_v,
                _LAST_LOCKED_VEHICLE_OPACITY,
                draw_all_lines,
            )
        else:
            dpr = self.devicePixelRatio()
            cache_key = (id(icon_pm), n, area_sz, dpr, acc_truck, sc)
            scaled = self._scaled_icon_lines_cache.get(cache_key)
            if scaled is None:
                phys_sh = int(sh * dpr)
                scaled = icon_pm.scaled(
                    phys_sh, phys_sh,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                scaled.setDevicePixelRatio(dpr)
                self._scaled_icon_lines_cache[cache_key] = scaled
            car_up_v_draw = int(6 * sc) if not acc_truck else 0
            vx_i = int(ax + ix)
            vy_i = int(ay + iy - car_up_v_draw)
            self._composite_acc_lines_with_vehicle_cutout(
                pa,
                ax,
                ay,
                area_sz,
                sc,
                line_color,
                car_up_lines,
                lines_start_rel,
                n,
                lw,
                ls,
                num_lines,
                vx_i,
                vy_i,
                scaled,
                ("tracked", n, area_sz, dpr, acc_truck, sc),
                scaled,
                1.0,
                draw_all_lines,
            )

    # -- drag --
    def moveEvent(self, event):
        super().moveEvent(event)
        new_dpr = self.devicePixelRatio()
        if new_dpr != getattr(self, "_last_dpr", new_dpr):
            self._last_dpr = new_dpr
            self._p._icon_cache.clear()
            self._tint_cache.clear()
            self._scaled_icon_lines_cache.clear()
            self._ghost_raw_cache.clear()
            self._ghost_scaled_cache.clear()
            self._ghost_exact_alpha_cache.clear()
            self._vehicle_cutout_mask_cache.clear()
            self._acc_lines_layer_cache.clear()
            self._p._current_icon = self._load_icon()
            self.update()
        else:
            self._last_dpr = new_dpr

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            cx, cy = _cursor_pos()
            pos = self.pos()
            self._drag_offset = (cx - pos.x(), cy - pos.y())

    def mouseMoveEvent(self, event):
        if self._drag_offset:
            cx, cy = _cursor_pos()
            self.move(cx - self._drag_offset[0], cy - self._drag_offset[1])

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_offset:
            self._drag_offset = None
            try:
                pos = self.pos()
                Settings.save(values={"panel_x": pos.x(), "panel_y": pos.y()})
            except Exception:
                logger.exception("cc_panel: failed to persist panel position")


# ---------------------------------------------------------------------------
# Public API – thin thread-safe facade over _PanelWidget.
# ---------------------------------------------------------------------------

class cc_panel:
    """
    Thread-safe cruise control display panel (PySide6).

    Create on the Qt main thread.  Every public method is safe to call from
    any thread – state mutations are forwarded to the Qt event loop via
    signals, so no update is ever dropped.

    update() coalesces: rapid calls merge into one pending dict (last write
    per field wins) and schedule a single queued flush, so the GUI thread is
    not flooded with one signal per call.
    """

    def __init__(
        self,
        text_content: str,
        cc_mode: str = "Cruise control",
        cc_enabled: bool = True,
        x_co: int = 100,
        y_co: int = 100,
        acc_enabled: bool = False,
        scale_mult: float = 1,
    ):
        s = scale_mult
        self._scale_mult = s
        self._panel_w = int(315 * s)
        self._panel_h = int(100 * s)
        self._radius = int(30 * s)
        self._icon_spacing = int(20 * s)

        self._text_content = text_content
        self._cc_mode = cc_mode
        self._cc_enabled = cc_enabled
        self._acc_enabled = acc_enabled
        self._acc_truck = False
        self._acc_locked = False
        self._last_locked_acc_truck: bool | None = None
        self._distance_to_lead = 2
        self._AEB_warn = False
        self._AEB_warn_off_time = 0.0
        self._last_AEB_warn_true = 0.0

        self._bg_opacity = 0.6
        self._text_color: QColor = self._color_for_mode(cc_mode, cc_enabled)

        self._blink_running = False
        self._hide_icon = False
        # AEB blink timing: express as whole frames at 60 Hz to avoid
        # beat-frequency jitter (consistent cadence across cycles).
        self._blink_fps = 60.0
        self._blinker_on_frames = 9   # 150ms @ 60 Hz
        self._blinker_off_frames = 6  # 100ms @ 60 Hz
        self._blinker_t_on_ms = max(1, int(round(1000.0 * (self._blinker_on_frames / self._blink_fps))))
        self._blinker_t_off_ms = max(1, int(round(1000.0 * (self._blinker_off_frames / self._blink_fps))))
        self._time_after_AEB_warn = 2.0

        self._font = QFont("Arial")
        self._font.setBold(True)
        self._font.setPixelSize(max(1, int(40 * s)))

        self._icon_cache: dict[tuple, QPixmap] = {}
        self._current_icon: QPixmap | None = None

        self._start_x = x_co
        self._start_y = y_co
        self.running = True

        self._widget = _PanelWidget(self)
        self._current_icon = self._widget._load_icon()

        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        self._coalesce_armed = False

    # -- properties --

    @property
    def blink_running(self) -> bool:
        """Whether the AEB blink animation is currently active."""
        return self._blink_running

    # -- public thread-safe API --

    def update(
        self,
        new_text: str | None = None,
        cc_mode: str | None = None,
        cc_enabled: bool | None = None,
        acc_locked: bool | None = None,
        distance_to_lead: int | None = None,
        AEB_warn: bool | None = None,
        complete_update: bool = False,
        acc_enabled: bool | None = None,
        acc_truck: bool | None = None,
    ):
        """
        Update the display.  Thread-safe, never drops changes.

        Safe to call at high frequency from worker loops: successive calls
        merge into pending state and coalesce to one GUI flush per event-loop
        burst (cheap lock + dict update on callers; one repaint batch).

        Args:
            new_text: Speed text to display (e.g. "100 km/h").
            cc_mode: "Speed limiter" or "Cruise control".
            cc_enabled: Whether CC is enabled.
            acc_locked: Whether ACC has locked onto a vehicle.
            distance_to_lead: Distance lines (1-3) shown under the icon.
            AEB_warn: Whether AEB warning is active (triggers blinking).
            complete_update: Force a full repaint / recalculation.
            acc_enabled: Whether ACC is enabled.
            acc_truck: Whether a truck is being tracked.
        """
        d: dict = {}
        if new_text is not None:
            d["text_content"] = new_text
        if cc_mode is not None:
            d["cc_mode"] = cc_mode
        if cc_enabled is not None:
            d["cc_enabled"] = cc_enabled
        if acc_locked is not None:
            d["acc_locked"] = acc_locked
        if distance_to_lead is not None:
            d["distance_to_lead"] = distance_to_lead
        if AEB_warn is not None:
            d["AEB_warn"] = AEB_warn
        if acc_enabled is not None:
            d["acc_enabled"] = acc_enabled
        if acc_truck is not None:
            d["acc_truck"] = acc_truck
        if complete_update:
            d["_complete_update"] = True
        if not d:
            return
        with self._pending_lock:
            self._pending.update(d)
            if d.get("_complete_update"):
                self._pending["_complete_update"] = True
            if not self._coalesce_armed:
                self._coalesce_armed = True
                self._widget._flush_coalesced_sig.emit()

    def show(self):
        """Show the panel and bring it to the front."""
        self._widget._show_sig.emit()

    def ensure_on_screen(self) -> None:
        """
        If the frame does not intersect any monitor's work area, move to the
        primary screen center and persist. Call from the Qt main thread.
        Fixes invisible panels when config has coordinates from another setup.
        """
        w = self._widget
        screens = QGuiApplication.screens()
        if not screens:
            return
        fg = w.frameGeometry()
        union = screens[0].availableGeometry()
        for s in screens[1:]:
            union = union.united(s.availableGeometry())
        if union.intersects(fg):
            return
        ps = QGuiApplication.primaryScreen()
        if ps is None:
            return
        ag = ps.availableGeometry()
        x = ag.x() + max(0, (ag.width() - fg.width()) // 2)
        y = ag.y() + max(0, (ag.height() - fg.height()) // 2)
        w.move(x, y)
        try:
            Settings.save(values={"panel_x": x, "panel_y": y})
        except Exception:
            logger.exception("cc_panel: failed to persist clamped position")

    def hide(self):
        """Hide the panel (keeps state; use show() to reveal again)."""
        self._widget._hide_sig.emit()

    def stop(self):
        """Stop the panel and close the window permanently."""
        self._widget._stop_sig.emit()

    def move(self, x: int, y: int):
        """Reposition the panel on screen."""
        self._widget._move_sig.emit(int(x), int(y))

    def update_scaling(self, scale_mult: float):
        """Apply a new scale multiplier and resize everything."""
        self._widget._scale_sig.emit(float(scale_mult))

    def set_background_opacity(self, opacity: float):
        """Set background-only opacity (0.0–1.0).  Text/icon stay fully opaque."""
        self._widget._bg_opacity_sig.emit(max(0.0, min(1.0, float(opacity))))

    @staticmethod
    def _color_for_mode(mode: str, enabled: bool) -> QColor:
        if not enabled:
            return QColor(DISABLED_COLOR)
        if mode == "Speed limiter":
            return QColor(SPEEDLIMITER_COLOR)
        return QColor(CRUISECONTROL_COLOR)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qapp = QApplication.instance() or QApplication(sys.argv)

    panel = cc_panel("80 km/h", "Cruise control", True, scale_mult=2)
    panel.show()

    MODES = ["Cruise control", "Speed limiter", "Other"]
    SPEEDS = ["-- km/h", "80 km/h", "120 km/h"]

    state = {
        "new_text": "80 km/h",
        "cc_mode": "Cruise control",
        "cc_enabled": True,
        "acc_locked": False,
        "acc_enabled": False,
        "acc_truck": False,
        "distance_to_lead": 3,
        "AEB_warn": False,
    }

    def update_scaling():
        time.sleep(3)
        panel.update_scaling(0.5)
        time.sleep(3)
        panel.update_scaling(1)

    def apply():
        panel.update(**state)

    def print_help():
        print("\nKeys (then Enter) – each toggles or cycles one value:")
        print("  m = cc_mode (cycle)     e = cc_enabled    s = speed text (cycle)")
        print("  l = acc_locked          a = acc_enabled   t = acc_truck")
        print("  1/2/3/4 = distance_to_lead (lines)   b = AEB_warn")
        print("  h = help   q = quit")
        print(f"\n  Now: mode={state['cc_mode']!r} enabled={state['cc_enabled']} "
              f"locked={state['acc_locked']} acc={state['acc_enabled']} truck={state['acc_truck']} "
              f"lines={state['distance_to_lead']} AEB={state['AEB_warn']} text={state['new_text']!r}\n")

    def keyboard_test():
        print_help()
        while panel.running:
            try:
                line = input("Key: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            key = line[0]
            if key == "q":
                panel.stop()
                break
            if key == "h":
                print_help()
                continue
            if key == "m":
                i = (MODES.index(state["cc_mode"]) + 1) % len(MODES) if state["cc_mode"] in MODES else 0
                state["cc_mode"] = MODES[i]
                print(f"  cc_mode = {state['cc_mode']!r}")
            elif key == "e":
                state["cc_enabled"] = not state["cc_enabled"]
                print(f"  cc_enabled = {state['cc_enabled']}")
            elif key == "s":
                i = (SPEEDS.index(state["new_text"]) + 1) % len(SPEEDS) if state["new_text"] in SPEEDS else 0
                state["new_text"] = SPEEDS[i]
                print(f"  new_text = {state['new_text']!r}")
            elif key == "l":
                state["acc_locked"] = not state["acc_locked"]
                print(f"  acc_locked = {state['acc_locked']}")
            elif key == "a":
                state["acc_enabled"] = not state["acc_enabled"]
                print(f"  acc_enabled = {state['acc_enabled']}")
            elif key == "t":
                state["acc_truck"] = not state["acc_truck"]
                print(f"  acc_truck = {state['acc_truck']}")
            elif key in "1234":
                state["distance_to_lead"] = int(key)
                print(f"  distance_to_lead = {state['distance_to_lead']}")
            elif key == "b":
                state["AEB_warn"] = not state["AEB_warn"]
                print(f"  AEB_warn = {state['AEB_warn']}")
            else:
                print("  Unknown key. Press h for help.")
                continue
            apply()

    threading.Thread(target=keyboard_test, daemon=True).start()
    threading.Thread(target=update_scaling, daemon=True).start()

    exit_timer = QTimer()
    exit_timer.timeout.connect(lambda: qapp.quit() if not panel.running else None)
    exit_timer.start(100)

    sys.exit(qapp.exec())
