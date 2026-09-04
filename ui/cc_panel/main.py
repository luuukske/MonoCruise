"""Cruise control panel (PySide6 QPainter). Thread-safe facade; see ui/cc_panel/README.md."""

import logging
import math
import os
import sys
import time
import threading
from typing import Callable

from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsBlurEffect,
)
if not __name__ == "__main__":
    from core.settings import Settings
from PySide6.QtCore import Qt, QTimer, Signal, QByteArray, QRectF, QPointF
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
    QLinearGradient,
    QRadialGradient,
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

# Lead-vehicle speed indicator (rendered above set speed when ACC tracks a lead).
# Values tuned in ui/cc_panel/prototype_lead_speed.html.
_LEAD_FONT_RATIO = 0.5
# _LEAD_BRIGHTNESS keeps hue; perceived dimming is mostly _LEAD_OPACITY on dark backdrops.
_LEAD_BRIGHTNESS = 0.95
_LEAD_OPACITY = 0.75
_LEAD_SHIFT_PX_BASE = 5.0
_LEAD_GAP_PX_BASE = 0.0
_LEAD_MASK_BLUR_BASE = 13.0
_LEAD_MASK_PAD_BASE = 2.0
_LEAD_ANIM_FORWARD_MS = 750
_LEAD_ANIM_RETRACT_MS = 2000
_LEAD_TICK_MS = 16

_VEHICLE_ANIM_MS = 350
_VEHICLE_TICK_MS = 16

# Sentinel for update(lead_vehicle_speed=...): distinguishes "not provided" from None ("no lead").
_LEAD_UNSET = object()


def _smoothstep(t: float) -> float:
    """Hermite smoothstep (flat ends); pre-warps lead/vehicle easing curves."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _ease_out_cubic(t: float) -> float:
    t = _smoothstep(t)
    return 1.0 - (1.0 - t) ** 3


def _ease_out_quad(t: float) -> float:
    t = _smoothstep(t)
    return 1.0 - (1.0 - t) ** 2


logger = logging.getLogger(__name__)

def _cursor_pos() -> tuple[int, int]:
    pos = QCursor.pos()
    return pos.x(), pos.y()

# Internal rendering widget – every method runs on the Qt main thread.

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

        self._lead_anim_timer = QTimer(self)
        self._lead_anim_timer.timeout.connect(self._lead_anim_tick)
        # Reused per-frame buffer for lead text + soft-mask composite (avoids alloc churn).
        self._lead_buf: QPixmap | None = None
        self._lead_eraser_cache: dict[tuple, QPixmap] = {}
        self._lead_eraser_cache_max = 16

        # Set-speed pixmap for sub-pixel slide; use drawPixmap(QRectF,...) not integer overload.
        self._set_text_pm_cache: dict[tuple, QPixmap] = {}
        self._set_text_pm_cache_max = 16
        # Lead-speed text rasterized for the same reason as set-speed.
        self._lead_text_pm_cache: dict[tuple, QPixmap] = {}
        self._lead_text_pm_cache_max = 16

        self._vehicle_anim_timer = QTimer(self)
        self._vehicle_anim_timer.timeout.connect(self._vehicle_anim_tick)
        # Reused offscreen buffer for lines+cutout composite (avoids per-frame alloc).
        self._lines_composite_buf: QPixmap | None = None

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

    # window

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        p = self._p
        self.setFixedSize(p._panel_w, p._panel_h)
        if p._start_x is not None and p._start_y is not None:
            self.move(p._start_x, p._start_y)

    # signal slots

    def _on_show(self):
        self.show()

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
        # PreferNoHinting so QPointF text slide is not snapped to the pixel grid.
        f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
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
        self._lead_eraser_cache.clear()
        self._lead_buf = None
        self._lines_composite_buf = None
        self._set_text_pm_cache.clear()
        self._lead_text_pm_cache.clear()
        p._vehicle_anim_layout_key = ()
        p._vehicle_visual_y = None
        p._vehicle_visual_sh = None
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

        # Lead speed: keep last non-None value so the retract animation can still
        # render the text while it fades behind set speed.
        if "lead_vehicle_speed" in changes:
            lead_val = changes.pop("lead_vehicle_speed")
            p._lead_vehicle_speed = lead_val
            if lead_val is not None:
                p._lead_displayed_speed = lead_val

        for key, val in changes.items():
            setattr(p, f"_{key}", val)

        if p._acc_locked:
            p._last_locked_acc_truck = bool(p._acc_truck)

        needs_icon = complete or any(
            getattr(p, f"_{k}") != old[k]
            for k in ("cc_mode", "AEB_warn", "acc_locked", "acc_enabled", "acc_truck")
        )

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

        # Drive lead-visible animation from current state.
        new_lead_target = 1.0 if self._should_show_lead() else 0.0
        if new_lead_target != p._lead_anim_target:
            self._start_lead_anim(new_lead_target)

        self.update()

    # AEB blink

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

    # icon loading

    def _load_icon(self) -> QPixmap:
        p = self._p
        is_aeb = p._blink_running or p._AEB_warn
        dpr = self.devicePixelRatio()

        key = (
            p._cc_mode, p._scale_mult,
            is_aeb, p._acc_locked,
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
        """Expanded vehicle alpha for DestinationOut cutout; draw at (vx - pad_log, vy - pad_log).
        Expanded vehicle alpha for DestinationOut cutout; draw at (vx - pad_log, vy - pad_log)."""
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
        """White with source alpha only: punches lines under the glyph exactly."""
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
        """Multiply lines alpha by (255 - mask_alpha) in ROI; QRgb fast path."""
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
        vx: float,
        vy: float,
        target_vy: float,
        target_sh: float,
        sh_visual: float,
        sh_max: float,
        cutout_shape_pm: QPixmap,
        mask_cache_pk: tuple,
        vehicle_draw_pm: QPixmap,
        vehicle_opacity: float,
        draw_all_lines: Callable[[QPainter, int], None],
    ) -> None:
        """Cached lines, per-frame DestinationOut cutout, vehicle draw (ui/cc_panel/README.md)."""
        if cutout_shape_pm.isNull() or vehicle_draw_pm.isNull():
            draw_all_lines(pa, car_up_lines)
            if not vehicle_draw_pm.isNull():
                pa.save()
                pa.setOpacity(vehicle_opacity)
                pa.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                pa.drawPixmap(QPointF(vx, vy), vehicle_draw_pm)
                pa.restore()
            return

        dpr = self.devicePixelRatio()
        gap = max(1, int(round(_LAST_LOCKED_LINE_GAP_PX * sc)))
        r_phys = max(1, int(math.ceil(float(gap) * float(dpr))))
        pad_log = r_phys / float(dpr)

        # Mask dimensions at max size (physical pixels of the pixmap).
        cw_max = cutout_shape_pm.width()
        ch_max = cutout_shape_pm.height()
        # Logical dimensions of the dilated mask at max size.
        mask_lw_max = cw_max / dpr + 2.0 * pad_log
        mask_lh_max = ch_max / dpr + 2.0 * pad_log

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
        # Logical dimensions of the vehicle pixmap at max size.
        w_max = cw_max / dpr
        h_max = ch_max / dpr

        # Fixed horizontal centre at max vehicle size; mask left edge from vc_x_fixed.
        vc_x_fixed = vx + w_max / 2.0

        # Lines layer bbox from target_vy at max sh so cache key stays stable during animation.
        vc_y_target = target_vy + h_max / 2.0

        tmx = vc_x_fixed - mask_lw_max / 2.0
        tmy = vc_y_target - mask_lh_max / 2.0
        gmin_x = min(lmin_x, tmx)
        gmax_x = max(lmax_x, tmx + mask_lw_max)
        gmin_y = min(lmin_y, tmy)
        gmax_y = max(lmax_y, tmy + mask_lh_max)

        origin_x = int(math.floor(gmin_x))
        origin_y = int(math.floor(gmin_y))
        layer_w = max(1, int(math.ceil(gmax_x - origin_x)) + 1)
        layer_h = max(1, int(math.ceil(gmax_y - origin_y)) + 1)

        # Cache the lines-only layer (no cutout: cutout is applied per-frame
        # so the vehicle can move and resize without invalidating this cache).
        lines_layer_key = (
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
        )
        hit_lines = self._acc_lines_layer_cache.get(lines_layer_key)
        if hit_lines is not None and not hit_lines.isNull():
            lines_only_pm = hit_lines
        else:
            lines_only_pm = QPixmap(
                max(1, int(layer_w * dpr)),
                max(1, int(layer_h * dpr)),
            )
            lines_only_pm.fill(Qt.GlobalColor.transparent)
            lines_only_pm.setDevicePixelRatio(dpr)
            lp = QPainter(lines_only_pm)
            lp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            lp.translate(-origin_x, -origin_y)
            draw_all_lines(lp, car_up_lines)
            lp.end()
            self._acc_lines_layer_cache_put(lines_layer_key, lines_only_pm)

        mask_pm = self._line_cutout_mask_from_vehicle_pixmap(
            cutout_shape_pm, dpr, mask_cache_pk, r_phys
        )
        exact_pm = self._exact_vehicle_alpha_mask(
            cutout_shape_pm, dpr, mask_cache_pk
        )

        # Scale factor: how much smaller than max-sh the vehicle is drawn this frame.
        scale = sh_visual / sh_max if sh_max > 0 else 1.0

        w_visual = w_max * scale
        h_visual = h_max * scale

        # vc_x fixed at max-sh centre; vc_y follows animated top-left vy.
        vc_x = vc_x_fixed
        vc_y = vy + h_visual / 2.0

        # Dilated mask logical dimensions at visual scale.
        m_w = mask_lw_max * scale
        m_h = mask_lh_max * scale
        mx = vc_x - m_w / 2.0
        my = vc_y - m_h / 2.0

        # Vehicle draw rect (exact mask = same bounds as vehicle).
        ex_w = w_visual
        ex_h = h_visual
        ex_x = vc_x - ex_w / 2.0
        ex_y = vc_y - ex_h / 2.0

        buf = self._ensure_lines_composite_buf(layer_w, layer_h, dpr)
        buf.fill(Qt.GlobalColor.transparent)

        bp = QPainter(buf)
        try:
            bp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            bp.drawPixmap(0, 0, lines_only_pm)
            bp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            if not mask_pm.isNull():
                # Source rect is in pixmap device pixels, not logical: Qt's
                # drawPixmap ignores devicePixelRatio for the source rectangle.
                src_mask_rect = QRectF(0.0, 0.0, float(mask_pm.width()), float(mask_pm.height()))
                dst_mask_rect = QRectF(mx - origin_x, my - origin_y, m_w, m_h)
                bp.drawPixmap(dst_mask_rect, mask_pm, src_mask_rect)
            if not exact_pm.isNull():
                src_exact_rect = QRectF(0.0, 0.0, float(exact_pm.width()), float(exact_pm.height()))
                dst_exact_rect = QRectF(ex_x - origin_x, ex_y - origin_y, ex_w, ex_h)
                bp.drawPixmap(dst_exact_rect, exact_pm, src_exact_rect)
        finally:
            bp.end()

        pa.drawPixmap(origin_x, origin_y, buf)

        pa.save()
        pa.setOpacity(vehicle_opacity)
        pa.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        src_veh_rect = QRectF(0.0, 0.0, float(vehicle_draw_pm.width()), float(vehicle_draw_pm.height()))
        dst_veh_rect = QRectF(ex_x, ex_y, ex_w, ex_h)
        pa.drawPixmap(dst_veh_rect, vehicle_draw_pm, src_veh_rect)
        pa.restore()

    # lead vehicle speed indicator

    def _should_show_lead(self) -> bool:
        p = self._p
        # Show lead when lead_vehicle_speed is set (not acc_locked); None means no lead.
        return (
            p._acc_enabled
            and p._cc_mode == "Cruise control"
            and p._lead_vehicle_speed is not None
        )

    def _start_lead_anim(self, target: float):
        """Start a new lead-visible animation segment from current visual to target. Each segment
        locks its easing + duration at start so target changes mid-flight never produce a position
        jump (animation lerps from current _lead_visual under the new curve/duration)."""
        p = self._p
        p._lead_anim_start = p._lead_visual
        p._lead_anim_target = float(target)
        p._lead_anim_t0 = time.monotonic()
        if p._lead_anim_target > p._lead_anim_start:
            p._lead_anim_dur_s = _LEAD_ANIM_FORWARD_MS / 1000.0
            p._lead_anim_easing = _ease_out_cubic
        else:
            p._lead_anim_dur_s = _LEAD_ANIM_RETRACT_MS / 1000.0
            p._lead_anim_easing = _ease_out_quad
        if p._lead_anim_dur_s <= 0.0:
            p._lead_visual = p._lead_anim_target
            self._lead_anim_timer.stop()
            return
        if not self._lead_anim_timer.isActive():
            self._lead_anim_timer.start(_LEAD_TICK_MS)

    def _lead_anim_tick(self):
        p = self._p
        if p._lead_anim_dur_s <= 0.0:
            p._lead_visual = p._lead_anim_target
        else:
            t = (time.monotonic() - p._lead_anim_t0) / p._lead_anim_dur_s
            if t >= 1.0:
                p._lead_visual = p._lead_anim_target
            else:
                eased = p._lead_anim_easing(t)
                p._lead_visual = p._lead_anim_start + (p._lead_anim_target - p._lead_anim_start) * eased
        if p._lead_visual == p._lead_anim_target:
            self._lead_anim_timer.stop()
        self.update()

    def _start_vehicle_anim(self, target_y: float, target_sh: float) -> None:
        """Glide vehicle y/sh from current visuals; first paint snaps without timer."""
        p = self._p
        cur_y = p._vehicle_visual_y
        cur_sh = p._vehicle_visual_sh
        if cur_y is None or cur_sh is None:
            p._vehicle_visual_y = target_y
            p._vehicle_anim_target_y = target_y
            p._vehicle_visual_sh = target_sh
            p._vehicle_anim_target_sh = target_sh
            self._vehicle_anim_timer.stop()
            return
        p._vehicle_anim_start_y = cur_y
        p._vehicle_anim_target_y = target_y
        p._vehicle_anim_start_sh = cur_sh
        p._vehicle_anim_target_sh = target_sh
        p._vehicle_anim_t0 = time.monotonic()
        p._vehicle_anim_dur_s = _VEHICLE_ANIM_MS / 1000.0
        p._vehicle_anim_easing = _ease_out_cubic
        if p._vehicle_anim_dur_s <= 0.0 or (cur_y == target_y and cur_sh == target_sh):
            p._vehicle_visual_y = target_y
            p._vehicle_visual_sh = target_sh
            self._vehicle_anim_timer.stop()
            return
        if not self._vehicle_anim_timer.isActive():
            self._vehicle_anim_timer.start(_VEHICLE_TICK_MS)

    def _vehicle_anim_tick(self) -> None:
        p = self._p
        if p._vehicle_anim_dur_s <= 0.0:
            p._vehicle_visual_y = p._vehicle_anim_target_y
            p._vehicle_visual_sh = p._vehicle_anim_target_sh
        else:
            t = (time.monotonic() - p._vehicle_anim_t0) / p._vehicle_anim_dur_s
            if t >= 1.0:
                p._vehicle_visual_y = p._vehicle_anim_target_y
                p._vehicle_visual_sh = p._vehicle_anim_target_sh
            else:
                eased = p._vehicle_anim_easing(t)
                p._vehicle_visual_y = (
                    p._vehicle_anim_start_y
                    + (p._vehicle_anim_target_y - p._vehicle_anim_start_y) * eased
                )
                p._vehicle_visual_sh = (
                    p._vehicle_anim_start_sh
                    + (p._vehicle_anim_target_sh - p._vehicle_anim_start_sh) * eased
                )
        if (
            p._vehicle_visual_y == p._vehicle_anim_target_y
            and p._vehicle_visual_sh == p._vehicle_anim_target_sh
        ):
            self._vehicle_anim_timer.stop()
        self.update()

    def _ensure_lines_composite_buf(self, w: int, h: int, dpr: float) -> QPixmap:
        tw = max(1, int(w * dpr))
        th = max(1, int(h * dpr))
        buf = self._lines_composite_buf
        if buf is None or buf.width() < tw or buf.height() < th:
            buf = QPixmap(tw, th)
            buf.setDevicePixelRatio(dpr)
            self._lines_composite_buf = buf
        return buf

    @staticmethod
    def _dimmed_color(base: QColor, brightness: float) -> QColor:
        b = max(0.0, min(1.0, brightness))
        return QColor(
            max(0, min(255, int(round(base.red() * b)))),
            max(0, min(255, int(round(base.green() * b)))),
            max(0, min(255, int(round(base.blue() * b)))),
        )

    @staticmethod
    def _format_lead_text(speed: float) -> str:
        return f"{int(round(speed))} km/h"

    def _make_feathered_rect_pixmap(
        self, inner_w: int, inner_h: int, blur_r: int, dpr: float
    ) -> QPixmap:
        """Gaussian-blurred white rect eraser; size inner + 2*margin (margin=2*blur_r)."""
        inner_w = max(1, int(inner_w))
        inner_h = max(1, int(inner_h))
        blur_r = max(0, int(blur_r))

        if blur_r == 0:
            # No blur → simple solid rect, no margin needed.
            pm = QPixmap(int(inner_w * dpr), int(inner_h * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            pa = QPainter(pm)
            try:
                pa.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pa.setPen(Qt.PenStyle.NoPen)
                pa.setBrush(QColor(255, 255, 255, 255))
                pa.drawRect(QRectF(0.0, 0.0, float(inner_w), float(inner_h)))
            finally:
                pa.end()
            return pm

        # 2*blur_r margin for gaussian tail (~2 sigma at QualityHint sigma ~= blur_r).
        margin = max(blur_r * 2, 1)
        total_w = inner_w + 2 * margin
        total_h = inner_h + 2 * margin

        pm = QPixmap(int(total_w * dpr), int(total_h * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)

        # QGraphicsBlurEffect renders through a scene. Quality hint forces a
        # real (separable) gaussian rather than a fast-but-blocky box blur.
        scene = QGraphicsScene()
        try:
            item = QGraphicsRectItem(0.0, 0.0, float(inner_w), float(inner_h))
            item.setBrush(QColor(255, 255, 255, 255))
            item.setPen(Qt.PenStyle.NoPen)
            item.setPos(float(margin), float(margin))
            blur = QGraphicsBlurEffect()
            blur.setBlurRadius(float(blur_r))
            blur.setBlurHints(QGraphicsBlurEffect.BlurHint.QualityHint)
            item.setGraphicsEffect(blur)
            scene.addItem(item)

            pa = QPainter(pm)
            try:
                pa.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pa.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                # Render the scene in its own coords (no scaling) into the
                # pixmap. devicePixelRatio on the pixmap handles HiDPI.
                target = QRectF(0.0, 0.0, float(total_w), float(total_h))
                source = QRectF(0.0, 0.0, float(total_w), float(total_h))
                scene.render(pa, target, source)
            finally:
                pa.end()
        finally:
            # Drop scene/item references so Qt can collect them; effect is
            # owned by the item so it goes with it.
            scene.clear()

        return pm

    def _get_lead_eraser(
        self, set_w: int, set_h: int, pad_px: int, blur_px: int, dpr: float
    ) -> QPixmap:
        key = (set_w, set_h, pad_px, blur_px, dpr)
        hit = self._lead_eraser_cache.get(key)
        if hit is not None:
            return hit
        pm = self._make_feathered_rect_pixmap(
            set_w + 2 * pad_px, set_h + 2 * pad_px, blur_px, dpr
        )
        if len(self._lead_eraser_cache) >= self._lead_eraser_cache_max:
            self._lead_eraser_cache.pop(next(iter(self._lead_eraser_cache)))
        self._lead_eraser_cache[key] = pm
        return pm

    def _set_text_pixmap(
        self, text: str, color: QColor, font: QFont, fm: QFontMetrics
    ) -> QPixmap:
        """Raster set-speed text for fractional pixmap blit (avoids Windows baseline snap)."""
        try:
            dpr = float(self.devicePixelRatioF() or 1.0)
        except Exception:
            dpr = 1.0
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        # rgba() encodes color+alpha. Pixel size + dpr re-key on rescale.
        key = (text, int(color.rgba()), font.pixelSize(),
               int(round(dpr * 100)), tw, th)
        hit = self._set_text_pm_cache.get(key)
        if hit is not None:
            return hit
        # +2px padding so antialiased edges don't get clipped.
        px_w = max(1, int(tw * dpr) + 2)
        px_h = max(1, int(th * dpr) + 2)
        pm = QPixmap(px_w, px_h)
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        tp = QPainter(pm)
        try:
            tp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            tp.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            tp.setFont(font)
            tp.setPen(color)
            tp.drawText(QPointF(0.0, float(fm.ascent())), text)
        finally:
            tp.end()
        if len(self._set_text_pm_cache) >= self._set_text_pm_cache_max:
            # Pop oldest insertion to bound memory (dicts preserve order).
            self._set_text_pm_cache.pop(next(iter(self._set_text_pm_cache)))
        self._set_text_pm_cache[key] = pm
        return pm

    def _lead_text_pixmap(
        self, text: str, color: QColor, font: QFont, fm: QFontMetrics
    ) -> QPixmap:
        """Rasterize lead-speed text to a pixmap. Same rationale as ``_set_text_pixmap``:..."""
        try:
            dpr = float(self.devicePixelRatioF() or 1.0)
        except Exception:
            dpr = 1.0
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        key = (text, int(color.rgba()), font.pixelSize(),
               int(round(dpr * 100)), tw, th)
        hit = self._lead_text_pm_cache.get(key)
        if hit is not None:
            return hit
        px_w = max(1, int(tw * dpr) + 2)
        px_h = max(1, int(th * dpr) + 2)
        pm = QPixmap(px_w, px_h)
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        tp = QPainter(pm)
        try:
            tp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            tp.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            tp.setFont(font)
            tp.setPen(color)
            tp.drawText(QPointF(0.0, float(fm.ascent())), text)
        finally:
            tp.end()
        if len(self._lead_text_pm_cache) >= self._lead_text_pm_cache_max:
            self._lead_text_pm_cache.pop(next(iter(self._lead_text_pm_cache)))
        self._lead_text_pm_cache[key] = pm
        return pm

    @staticmethod
    def _draw_pixmap_subpixel(pa: QPainter, pm: QPixmap, x: float, y: float) -> None:
        """Draw ``pm`` at fractional ``(x, y)`` with sub-pixel precision. Qt's raster engine..."""
        if pm.isNull():
            return
        dpr = pm.devicePixelRatio() or 1.0
        log_w = pm.width() / dpr
        log_h = pm.height() / dpr
        pa.save()
        try:
            pa.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            pa.translate(x, y)
            pa.scale(1.0001, 1.0001)
            pa.drawPixmap(
                QRectF(0.0, 0.0, log_w, log_h),
                pm,
                QRectF(0.0, 0.0, float(pm.width()), float(pm.height())),
            )
        finally:
            pa.restore()

    def _ensure_lead_buf(self, w: int, h: int, dpr: float) -> QPixmap:
        tw = max(1, int(w * dpr))
        th = max(1, int(h * dpr))
        buf = self._lead_buf
        if buf is None or buf.width() != tw or buf.height() != th:
            buf = QPixmap(tw, th)
            buf.setDevicePixelRatio(dpr)
            self._lead_buf = buf
        return buf

    def _paint_lead(
        self,
        pa: QPainter,
        w: int,
        h: int,
        sc: float,
        set_x: float,
        set_top_y: float,
        set_w: int,
        set_h: int,
        set_baseline_default_y: float,
    ) -> None:
        p = self._p
        visual = p._lead_visual
        if visual <= 0.001:
            return
        if p._lead_displayed_speed is None:
            return

        set_font_px = p._font.pixelSize()
        lead_font_px = max(1, int(round(set_font_px * _LEAD_FONT_RATIO)))
        lead_font = QFont(p._font)
        lead_font.setPixelSize(lead_font_px)
        fm_lead = QFontMetrics(lead_font)
        lead_text = self._format_lead_text(p._lead_displayed_speed)
        lead_w_px = fm_lead.horizontalAdvance(lead_text)
        lead_descent = fm_lead.descent()

        # Right-anchor: lead's right edge aligns with set text's right edge.
        set_right = set_x + set_w
        lead_anchor_x = set_right - lead_w_px
        lead_hidden_x = lead_anchor_x  # no horizontal travel

        gap_px = _LEAD_GAP_PX_BASE * sc
        lead_anchor_baseline_y = set_top_y - gap_px - lead_descent
        lead_hidden_baseline_y = float(set_baseline_default_y)

        lead_x = lead_hidden_x + (lead_anchor_x - lead_hidden_x) * visual
        lead_baseline_y = lead_hidden_baseline_y + (lead_anchor_baseline_y - lead_hidden_baseline_y) * visual

        # Lead alpha ramps after visual>0.5; capped at _LEAD_OPACITY (dimmer than set speed).
        lead_alpha = max(0.0, (visual - 0.5) / 0.5) * _LEAD_OPACITY
        if lead_alpha <= 0.001:
            return

        lead_color = self._dimmed_color(p._text_color, _LEAD_BRIGHTNESS)

        dpr = self.devicePixelRatio()
        buf = self._ensure_lead_buf(w, h, dpr)
        buf.fill(Qt.GlobalColor.transparent)

        lead_pm = self._lead_text_pixmap(lead_text, lead_color, lead_font, fm_lead)
        lead_top_y = lead_baseline_y - fm_lead.ascent()

        bp = QPainter(buf)
        try:
            bp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            bp.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            bp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            bp.setOpacity(lead_alpha)
            # Pre-rasterized lead text; _draw_pixmap_subpixel keeps fractional translate.
            self._draw_pixmap_subpixel(bp, lead_pm, lead_x, lead_top_y)

            # Gaussian soft eraser at full opacity (prototype blur+fillRect); see cc_panel README.
            pad_px = max(0, int(round(_LEAD_MASK_PAD_BASE * sc)))
            blur_px = max(0, int(round(_LEAD_MASK_BLUR_BASE * sc)))
            eraser_pm = self._get_lead_eraser(set_w, set_h, pad_px, blur_px, dpr)
            eraser_margin = 2 * blur_px if blur_px > 0 else 0
            erase_x = set_x - pad_px - eraser_margin
            erase_y = set_top_y - pad_px - eraser_margin
            bp.setOpacity(1.0)
            bp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            bp.drawPixmap(QPointF(erase_x, erase_y), eraser_pm)
        finally:
            bp.end()

        pa.drawPixmap(0, 0, buf)

    # tinting

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

    # painting

    def paintEvent(self, _event):
        try:
            self._paint()
        except Exception:
            logger.exception("cc_panel.main._PanelWidget.paintEvent: paint failed")

    def _paint(self):
        p = self._p
        pa = QPainter(self)
        pa.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Required (alongside font's PreferNoHinting) so the set-speed text
        # animates sub-pixel without staircasing during the lead-speed slide.
        pa.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        pa.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

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

        text_x = float(icon_x - p._icon_spacing - tw)
        text_y_baseline_default = float((h - fm.height()) // 2 + fm.ascent())
        # Set speed slides down for lead speed above; float baseline avoids mid-animation snap.
        set_y_shift = _LEAD_SHIFT_PX_BASE * sc * p._lead_visual
        text_y_baseline = text_y_baseline_default + set_y_shift

        # Set-speed pixmap blit for sub-pixel slide (_draw_pixmap_subpixel).
        text_pm = self._set_text_pixmap(p._text_content, p._text_color, p._font, fm)
        text_top_y = text_y_baseline - fm.ascent()
        self._draw_pixmap_subpixel(pa, text_pm, text_x, text_top_y)

        # lead vehicle speed (above set, soft-masked under set text)
        set_text_top_y = text_y_baseline - fm.ascent()
        self._paint_lead(
            pa, w, h, sc,
            text_x, set_text_top_y, int(tw), int(fm.height()),
            text_y_baseline_default,
        )

        # icon
        icon = p._current_icon
        if icon and not icon.isNull() and not p._hide_icon:
            ic = QColor(AEB_COLOR) if self._is_aeb_active() else p._text_color
            tinted = self._tinted(icon, ic)

            if show_lines:
                # TODO(ACC): gap/lead viz from ACC; locked shows vehicle+lines.
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

    @staticmethod
    def _compute_sh(area_sz: int, n: int, lw: int, ls: int, sc: float, truck: bool) -> int:
        """Logical sprite height/width for vehicle at distance slot n (1=closest/largest)."""
        lines_h_icon = n * lw + (n - 1) * ls
        sh = area_sz - lines_h_icon - ls - int((3 + 7 * (not truck)) * sc)
        if sh % 2 == area_sz % 2:
            sh += 1
        return max(1, sh)

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

        sh = self._compute_sh(area_sz, n, lw, ls, sc, acc_truck)

        lines_start_rel = area_sz - lines_h_block - 1
        center_y_icon = (area_sz - lines_h_icon - 1) / 2
        iy = int(center_y_icon - sh / 2) + int((1.5 + 8 * (not acc_truck)) * sc)
        ix = (area_sz - sh) // 2

        if need_ghost:
            sh_v = self._compute_sh(area_sz, n, lw, ls, sc, ghost_is_truck)
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
            # Pre-render at max sh (n=1 gives largest sprite) for smooth scale-down.
            sh_max = self._compute_sh(area_sz, 1, lw, ls, sc, ghost_is_truck)
            scaled_raw_max = self._load_vehicle_asset_scaled(ghost_is_truck, sh_max, dpr_g)
            scaled_v_max = self._load_vehicle_asset_scaled_tinted(
                ghost_is_truck, sh_max, dpr_g, line_color
            )
            target_sh = float(sh_v)
            # vx is the left edge of the sprite at max sh so the horizontal
            # centre (vx + sh_max/2) stays fixed while size animates.
            ix_v_max = (area_sz - sh_max) // 2
            vx = float(ax + ix_v_max)
            target_vy = float(ay + iy_v - car_up_v)
            layout_key = ("ghost", ghost_is_truck, dpr_g)
            p = self._p
            if p._vehicle_anim_layout_key != layout_key:
                p._vehicle_anim_layout_key = layout_key
                p._vehicle_visual_y = target_vy
                p._vehicle_anim_target_y = target_vy
                p._vehicle_visual_sh = target_sh
                p._vehicle_anim_target_sh = target_sh
                self._vehicle_anim_timer.stop()
            elif target_vy != p._vehicle_anim_target_y or target_sh != p._vehicle_anim_target_sh:
                self._start_vehicle_anim(target_vy, target_sh)
            vy = p._vehicle_visual_y if p._vehicle_visual_y is not None else target_vy
            sh_visual = p._vehicle_visual_sh if p._vehicle_visual_sh is not None else target_sh
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
                vx,
                vy,
                target_vy,
                target_sh,
                sh_visual,
                float(sh_max),
                scaled_raw_max,
                (ghost_is_truck, sh_max, dpr_g),
                scaled_v_max,
                _LAST_LOCKED_VEHICLE_OPACITY,
                draw_all_lines,
            )
        else:
            dpr = self.devicePixelRatio()
            # Pre-render at max sh (n=1 gives largest sprite) for smooth scale-down.
            sh_max = self._compute_sh(area_sz, 1, lw, ls, sc, acc_truck)
            cache_key = (id(icon_pm), area_sz, dpr, acc_truck, sc)
            scaled_max = self._scaled_icon_lines_cache.get(cache_key)
            if scaled_max is None:
                phys_sh_max = int(sh_max * dpr)
                scaled_max = icon_pm.scaled(
                    phys_sh_max, phys_sh_max,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                scaled_max.setDevicePixelRatio(dpr)
                self._scaled_icon_lines_cache[cache_key] = scaled_max
            target_sh = float(sh)
            car_up_v_draw = int(6 * sc) if not acc_truck else 0
            # vx is the left edge of the sprite at max sh so the horizontal
            # centre (vx + sh_max/2) stays fixed while size animates.
            ix_max = (area_sz - sh_max) // 2
            vx = float(ax + ix_max)
            target_vy = float(ay + iy - car_up_v_draw)
            layout_key = ("tracked", acc_truck, dpr)
            p = self._p
            if p._vehicle_anim_layout_key != layout_key:
                p._vehicle_anim_layout_key = layout_key
                p._vehicle_visual_y = target_vy
                p._vehicle_anim_target_y = target_vy
                p._vehicle_visual_sh = target_sh
                p._vehicle_anim_target_sh = target_sh
                self._vehicle_anim_timer.stop()
            elif target_vy != p._vehicle_anim_target_y or target_sh != p._vehicle_anim_target_sh:
                self._start_vehicle_anim(target_vy, target_sh)
            vy = p._vehicle_visual_y if p._vehicle_visual_y is not None else target_vy
            sh_visual = p._vehicle_visual_sh if p._vehicle_visual_sh is not None else target_sh
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
                vx,
                vy,
                target_vy,
                target_sh,
                sh_visual,
                float(sh_max),
                scaled_max,
                ("tracked", sh_max, dpr, acc_truck, sc),
                scaled_max,
                1.0,
                draw_all_lines,
            )

    # drag
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
            self._lead_eraser_cache.clear()
            self._lead_buf = None
            self._lines_composite_buf = None
            self._p._vehicle_anim_layout_key = ()
            self._p._vehicle_visual_y = None
            self._p._vehicle_visual_sh = None
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


# Public API – thin thread-safe facade over _PanelWidget.

class cc_panel:
    """Thread-safe cruise control display panel (PySide6). Create on the Qt main thread...."""

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
        # See update_scaling: PreferNoHinting is required so QPointF baselines
        # are honored sub-pixel during the lead-speed slide animation.
        self._font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)

        self._icon_cache: dict[tuple, QPixmap] = {}
        self._current_icon: QPixmap | None = None

        # Lead speed: latest in _lead_vehicle_speed; _lead_displayed_speed kept during retract.
        self._lead_vehicle_speed: float | None = None
        self._lead_displayed_speed: float | None = None
        self._lead_visual = 0.0
        self._lead_anim_start = 0.0
        self._lead_anim_target = 0.0
        self._lead_anim_t0 = 0.0
        self._lead_anim_dur_s = 0.0
        self._lead_anim_easing = _ease_out_cubic

        # Vehicle vertical position + size animation state.
        # None = uninitialised; snap to target on first paint instead of animating.
        self._vehicle_visual_y: float | None = None
        self._vehicle_anim_start_y: float = 0.0
        self._vehicle_anim_target_y: float = 0.0
        self._vehicle_visual_sh: float | None = None
        self._vehicle_anim_start_sh: float = 0.0
        self._vehicle_anim_target_sh: float = 0.0
        self._vehicle_anim_t0: float = 0.0
        self._vehicle_anim_dur_s: float = 0.0
        self._vehicle_anim_easing = _ease_out_cubic
        # Tracks sprite identity (truck-ness, ghost-ness, dpr) so that
        # sprite swaps snap instead of animating across mismatched glyphs.
        self._vehicle_anim_layout_key: tuple = ()

        self._start_x = x_co
        self._start_y = y_co
        self.running = True

        self._widget = _PanelWidget(self)
        self._current_icon = self._widget._load_icon()

        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        self._coalesce_armed = False

    # properties

    @property
    def blink_running(self) -> bool:
        """Whether the AEB blink animation is currently active."""
        return self._blink_running

    # public thread-safe API

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
        lead_vehicle_speed=_LEAD_UNSET,
    ):
        """Update the display. Thread-safe, never drops changes. Safe to call at high..."""
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
        if lead_vehicle_speed is not _LEAD_UNSET:
            d["lead_vehicle_speed"] = lead_vehicle_speed
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

    def is_visible(self) -> bool:
        """True if the panel widget is currently shown. Qt main thread only."""
        return self._widget.isVisible()

    def show(self):
        """Show the panel. WindowStaysOnTopHint keeps it above other apps."""
        self._widget._show_sig.emit()

    def ensure_on_screen(self) -> None:
        """If the frame does not intersect any monitor's work area, move to the primary..."""
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


# Standalone test

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
        "lead_vehicle_speed": None,
    }

    LEAD_SPEEDS = [None, 110.0]

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
        print("  v = lead_vehicle_speed (cycle: None/60/80/96/110)")
        print("  h = help   q = quit")
        print(f"\n  Now: mode={state['cc_mode']!r} enabled={state['cc_enabled']} "
              f"locked={state['acc_locked']} acc={state['acc_enabled']} truck={state['acc_truck']} "
              f"lines={state['distance_to_lead']} AEB={state['AEB_warn']} text={state['new_text']!r} "
              f"lead={state['lead_vehicle_speed']!r}\n")

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
            elif key == "v":
                i = (LEAD_SPEEDS.index(state["lead_vehicle_speed"]) + 1) % len(LEAD_SPEEDS) \
                    if state["lead_vehicle_speed"] in LEAD_SPEEDS else 0
                state["lead_vehicle_speed"] = LEAD_SPEEDS[i]
                print(f"  lead_vehicle_speed = {state['lead_vehicle_speed']!r}")
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

