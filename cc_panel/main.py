"""
Cruise control display panel using PySide6.

Single translucent window rendered entirely via QPainter.
Thread-safe: update() can be called from any thread without dropping changes.
"""

import os
import sys
import time
import ctypes
import threading

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QPainter, QColor, QFont, QFontMetrics, QPixmap,
    QPainterPath, QCursor,
)

SPEEDLIMITER_COLOR = "#008B00"
CRUISECONTROL_COLOR = "#4876FF"
DISABLED_COLOR = "#F1F1F1"
AEB_COLOR = "#FF0000"

_HAS_WINDLL = hasattr(ctypes, "windll")


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _cursor_pos() -> tuple[int, int]:
    """Screen cursor position, DPI-aware on Windows."""
    if _HAS_WINDLL:
        try:
            pt = _POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return pt.x, pt.y
        except Exception:
            pass
    pos = QCursor.pos()
    return pos.x(), pos.y()


# ---------------------------------------------------------------------------
# Internal rendering widget – every method runs on the Qt main thread.
# ---------------------------------------------------------------------------

class _PanelWidget(QWidget):
    _update_sig = Signal(dict)
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

        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_tick)

        self._setup_window()

        self._update_sig.connect(self._on_update)
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
        p._panel_w = int(300 * s)
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
        p._current_icon = self._load_icon()
        self.update()

    def _on_bg_opacity(self, v: float):
        self._p._bg_opacity = v
        self.update()

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

        key = (
            p._cc_mode, p._text_content, p._scale_mult,
            is_aeb, p._acc_locked, p._distance_to_lead,
            p._acc_enabled, p._acc_truck,
        )
        cached = p._icon_cache.get(key)
        if cached is not None:
            return cached

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
        icon_sz = int(fm.height() * 2)

        if fname:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "assets", fname
            )
            pm = QPixmap(path)
            if pm.isNull():
                pm = self._placeholder(icon_sz)
            else:
                pm = pm.scaled(
                    icon_sz, icon_sz,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        else:
            pm = self._placeholder(icon_sz)

        p._icon_cache[key] = pm
        return pm

    @staticmethod
    def _placeholder(sz: int) -> QPixmap:
        pm = QPixmap(sz, sz)
        pm.fill(Qt.GlobalColor.transparent)
        pa = QPainter(pm)
        pa.setPen(QColor(255, 255, 255))
        m = max(2, sz // 8)
        pa.drawEllipse(m, m, sz - 2 * m, sz - 2 * m)
        pa.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "?")
        pa.end()
        return pm

    # -- tinting --

    def _tinted(self, pm: QPixmap, color: QColor) -> QPixmap:
        """Return a copy of *pm* with all opaque pixels recoloured to *color*."""
        if pm is None or pm.isNull():
            return pm
        key = (id(pm), color.name())
        hit = self._tint_cache.get(key)
        if hit is not None:
            return hit
        out = QPixmap(pm.size())
        out.fill(Qt.GlobalColor.transparent)
        pa = QPainter(out)
        pa.drawPixmap(0, 0, pm)
        pa.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        pa.fillRect(out.rect(), color)
        pa.end()
        self._tint_cache[key] = out
        return out

    # -- painting --

    def paintEvent(self, _event):
        try:
            self._paint()
        except Exception:
            pass

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
        icon_sz = int(fm.height() * 2)
        right_m = int(20 * sc)

        icon_x = w - icon_sz - right_m
        show_lines = (
            p._acc_enabled
            and p._cc_mode == "Cruise control"
            and not p._AEB_warn
            and not p._blink_running
            and p._acc_locked
        )
        acc_adj = int(6 * (not p._blink_running and p._acc_enabled) * sc)
        icon_y = (h - icon_sz) // 2 - 2 - acc_adj

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
                self._paint_icon_lines(pa, tinted, icon_x, icon_y, icon_sz, sc, ic)
            else:
                pa.drawPixmap(int(icon_x), int(icon_y), tinted)

        pa.end()

    def _paint_icon_lines(self, pa: QPainter, icon_pm: QPixmap,
                          ax: int, ay: int, area_sz: int,
                          sc: float, line_color: QColor):
        """Draw the icon scaled down with ACC distance lines underneath."""
        p = self._p
        n = max(1, min(3, p._distance_to_lead))
        lw = int(4 * sc)
        ls = int(3 * sc)
        lines_h = n * lw + (n - 1) * ls

        sh = area_sz - lines_h - ls - int((3 + 7 * (not p._acc_truck)) * sc)
        if sh % 2 == area_sz % 2:
            sh += 1
        sh = max(1, sh)

        scaled = icon_pm.scaled(
            sh, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        lines_start_rel = area_sz - lines_h - 1
        center_y = lines_start_rel / 2
        iy = int(center_y - sh / 2) + int((1.5 + 8 * (not p._acc_truck)) * sc)
        ix = (area_sz - sh) // 2

        pa.drawPixmap(int(ax + ix), int(ay + iy), scaled)

        pa.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            ly = lines_start_rel + i * (lw + ls)
            indent = -2 * (i - n) * sc
            x0 = ax + indent
            x1 = ax + area_sz + 2 * (i - n) * sc - 1
            lp = QPainterPath()
            lp.addRoundedRect(
                float(x0), float(ay + ly + 1),
                float(x1 - x0), float(lw),
                lw / 2.0, lw / 2.0,
            )
            pa.fillPath(lp, line_color)

    # -- drag --

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
                save_variables(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "saves.json"),
                    panel_x=self.pos().x(),
                    panel_y=self.pos().y(),
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public API – thin thread-safe facade over _PanelWidget.
# ---------------------------------------------------------------------------

class cc_panel:
    """
    Thread-safe cruise control display panel (PySide6).

    Create on the Qt main thread.  Every public method is safe to call from
    any thread – state mutations are forwarded to the Qt event loop via
    signals, so no update is ever dropped.
    """

    def __init__(
        self,
        text_content: str,
        cc_mode: str = "Cruise control",
        cc_enabled: bool = True,
        x_co: int = 100,
        y_co: int = 100,
        acc_enabled: bool = False,
        scale_mult: float = 0.5,
    ):
        s = scale_mult
        self._scale_mult = s
        self._panel_w = int(300 * s)
        self._panel_h = int(100 * s)
        self._radius = int(30 * s)
        self._icon_spacing = int(20 * s)

        self._text_content = text_content
        self._cc_mode = cc_mode
        self._cc_enabled = cc_enabled
        self._acc_enabled = acc_enabled
        self._acc_truck = False
        self._acc_locked = False
        self._distance_to_lead = 2
        self._AEB_warn = False
        self._AEB_warn_off_time = 0.0
        self._last_AEB_warn_true = 0.0

        self._bg_opacity = 0.6
        self._text_color: QColor = self._color_for_mode(cc_mode, cc_enabled)

        self._blink_running = False
        self._hide_icon = False
        self._blinker_t_off_ms = 100
        self._blinker_t_on_ms = 150
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
        if d:
            self._widget._update_sig.emit(d)

    def show(self):
        """Show the panel and bring it to the front."""
        self._widget._show_sig.emit()

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

    panel = cc_panel("-- km/h", "Cruise control", False, scale_mult=1)
    panel.show()

    # All icon configurations: each triggers a different icon in _load_icon().
    ICON_CONFIGS = [
        # 1. Placeholder (unknown/other mode)
        {
            "name": "Placeholder (other mode)",
            "kwargs": {"new_text": "---", "cc_mode": "Other", "cc_enabled": False},
        },
        # 2. Cruise control icon only (no ACC)
        {
            "name": "Cruise control icon",
            "kwargs": {
                "new_text": "80 km/h",
                "cc_mode": "Cruise control",
                "cc_enabled": True,
                "acc_locked": False,
                "acc_enabled": False,
            },
        },
        # 3. Speed limiter
        {
            "name": "Speed limiter icon",
            "kwargs": {
                "new_text": "120 km/h",
                "cc_mode": "Speed limiter",
                "cc_enabled": True,
                "acc_locked": False,
                "acc_enabled": False,
            },
        },
        # 4. Car ACC, 3 lines
        {
            "name": "Car ACC, 3 lines",
            "kwargs": {
                "new_text": "80 km/h",
                "cc_mode": "Cruise control",
                "cc_enabled": True,
                "acc_locked": True,
                "acc_truck": False,
                "acc_enabled": True,
                "distance_to_lead": 3,
            },
        },
        # 5. Car ACC, 2 lines
        {
            "name": "Car ACC, 2 lines",
            "kwargs": {
                "new_text": "80 km/h",
                "cc_mode": "Cruise control",
                "cc_enabled": True,
                "acc_locked": True,
                "acc_truck": False,
                "acc_enabled": True,
                "distance_to_lead": 2,
            },
        },
        # 6. Car ACC, 1 line
        {
            "name": "Car ACC, 1 line",
            "kwargs": {
                "new_text": "80 km/h",
                "cc_mode": "Cruise control",
                "cc_enabled": True,
                "acc_locked": True,
                "acc_truck": False,
                "acc_enabled": True,
                "distance_to_lead": 1,
            },
        },
        # 7. Truck ACC, 3 lines
        {
            "name": "Truck ACC, 3 lines",
            "kwargs": {
                "new_text": "80 km/h",
                "cc_mode": "Cruise control",
                "cc_enabled": True,
                "acc_locked": True,
                "acc_truck": True,
                "acc_enabled": True,
                "distance_to_lead": 3,
            },
        },
        # 8. Truck ACC, 2 lines
        {
            "name": "Truck ACC, 2 lines",
            "kwargs": {
                "new_text": "80 km/h",
                "cc_mode": "Cruise control",
                "cc_enabled": True,
                "acc_locked": True,
                "acc_truck": True,
                "acc_enabled": True,
                "distance_to_lead": 2,
            },
        },
        # 9. Truck ACC, 1 line
        {
            "name": "Truck ACC, 1 line",
            "kwargs": {
                "new_text": "80 km/h",
                "cc_mode": "Cruise control",
                "cc_enabled": True,
                "acc_locked": True,
                "acc_truck": True,
                "acc_enabled": True,
                "distance_to_lead": 1,
            },
        },
        # 10. AEB warning (car icon, red tint + blink)
        {
            "name": "AEB warning (car, red)",
            "kwargs": {
                "new_text": "80 km/h",
                "cc_mode": "Cruise control",
                "cc_enabled": True,
                "acc_locked": True,
                "acc_truck": False,
                "acc_enabled": True,
                "distance_to_lead": 2,
                "AEB_warn": True,
            },
        },
        # 11. Speed limiter disabled (grey text, same icon)
        {
            "name": "Speed limiter disabled",
            "kwargs": {
                "new_text": "120 km/h",
                "cc_mode": "Speed limiter",
                "cc_enabled": False,
                "acc_locked": False,
                "acc_enabled": False,
            },
        },
    ]

    def run_tests():
        print("Icon configuration test (1 s between each).")
        time.sleep(1)

        for cfg in ICON_CONFIGS:
            panel.update(**cfg["kwargs"])
            print(f"  {cfg['name']}")
            time.sleep(1)

        # Stop AEB so later tests don't keep blinking
        panel.update(AEB_warn=False)
        time.sleep(2)

        print("\nBaseline set to defaults.")
        time.sleep(1)

        flow = [
            {"acc_locked": True, "cc_enabled": True, "acc_truck": True,
             "acc_enabled": True, "distance_to_lead": 3},
            {"distance_to_lead": 2},
            {"distance_to_lead": 1},
            {"acc_truck": False, "distance_to_lead": 3, "AEB_warn": True},
            {"AEB_warn": True},
            {"AEB_warn": False},
        ]

        for kw in flow:
            panel.update(**kw)
            desc = ", ".join(f"{k}={v!r}" for k, v in kw.items())
            print(f"Update: {desc}")
            time.sleep(2)

        rapid_update_test()

    def rapid_update_test():
        print("\n" + "=" * 60)
        print("RAPID UPDATE TEST (50 updates/sec)")
        print("=" * 60 + "\n")

        print("Test 1: AEB_warn=True for 1 s")
        t0 = time.time()
        while time.time() - t0 < 1.0:
            panel.update(
                new_text="80 km/h", cc_enabled=True,
                acc_locked=True, acc_enabled=True, AEB_warn=True,
            )
            time.sleep(0.02)
        print(f"  blink_running = {panel.blink_running}")

        aeb_off = time.time()
        print("Test 2: AEB_warn=False (cooldown)")
        t0 = time.time()
        while time.time() - t0 < 1.5:
            panel.update(
                new_text="80 km/h", cc_enabled=True,
                acc_locked=True, acc_enabled=True, AEB_warn=False,
            )
            time.sleep(0.02)
        print(f"  elapsed = {time.time() - aeb_off:.2f} s  blink = {panel.blink_running}")

        print("Test 3: Waiting for cooldown to expire …")
        t0 = time.time()
        stopped = None
        while time.time() - t0 < 1.5:
            if not panel.blink_running and stopped is None:
                stopped = time.time() - aeb_off
                print(f"  stopped at {stopped:.2f} s (expected ~2.0 s)")
            panel.update(
                new_text="80 km/h", cc_enabled=True,
                acc_locked=True, acc_enabled=True, AEB_warn=False,
            )
            time.sleep(0.02)
        if stopped is None and not panel.blink_running:
            stopped = time.time() - aeb_off
            print(f"  stopped at {stopped:.2f} s")

        ok = 1.8 <= (stopped or 0) <= 2.5
        print(f"\nCooldown correct: {ok}")
        print("=" * 60 + "\n")

    def stop_later():
        time.sleep(20)
        print("Stopping from background thread …")
        panel.stop()

    threading.Thread(target=run_tests, daemon=True).start()
    threading.Thread(target=stop_later, daemon=True).start()

    exit_timer = QTimer()
    exit_timer.timeout.connect(lambda: qapp.quit() if not panel.running else None)
    exit_timer.start(100)

    sys.exit(qapp.exec())
