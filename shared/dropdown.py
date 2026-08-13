"""Custom animated dropdown (app + updater). Port of MonoCruise Dropdown.dc.html; see
shared/README.md."""
from __future__ import annotations

from PySide6.QtCore import (
    Qt, Signal, Property, QPropertyAnimation, QPoint, QPointF, QRectF, QEasingCurve,
    QEvent, QObject,
)
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QPen, QFont, QFontMetricsF, QCursor,
)
from PySide6.QtWidgets import QWidget, QApplication

# QFont.setFamilies list (QPainter ignores CSS comma stacks); Inter may fall back.
FONT_FALLBACKS = ["Inter", "Segoe UI", "sans-serif"]

# Palette (monochrome, no accent colour)
DROPDOWN_FIELD_BG = "#242424"       # field + popup background
DROPDOWN_BORDER = "#3a3a3a"         # 1px border
DROPDOWN_BORDER_HOVER = "#555555"   # field border on hover (#555)
DROPDOWN_DIVIDER = "#383838"        # inset divider under the field
DROPDOWN_TEXT = "#e8e8e8"           # field + row text
DROPDOWN_CHEVRON = "#8a8a8a"        # chevron stroke
DROPDOWN_SELECTION_BAR = "#d8d8d8"  # 3x15 selection marker bar
DROPDOWN_ROW_HOVER_RGBA = (255, 255, 255, 13)  # rgba(255,255,255,.05) -> a=round(.05*255)=13
DROPDOWN_SHADOW_RGBA = (0, 0, 0, 115)          # box-shadow rgba(0,0,0,.45) -> a=round(.45*255)=115

# Box model (logical px, matching the CSS)
DROPDOWN_RADIUS = 8                 # field/popup corner radius
DROPDOWN_ROW_RADIUS = 6             # row hover/selection radius
DROPDOWN_ROW_MARGIN_Y = 3           # vertical inset of the highlight box within
                                    # each row -> a 2x gap between adjacent boxes
                                    # (row pitch/text position unchanged)
DROPDOWN_BORDER_W = 1
DROPDOWN_FIELD_PAD_X = 13           # field padding: 9px 13px
DROPDOWN_FIELD_PAD_Y = 9
DROPDOWN_FIELD_GAP = 10             # gap between text and chevron
DROPDOWN_CHEVRON_SIZE = 14          # svg 14x14 (viewBox 24, stroke-width 2.5)
DROPDOWN_CHEVRON_STROKE = 2.5
DROPDOWN_POPUP_PAD = 5              # popup padding when open
DROPDOWN_DIVIDER_INSET = 8          # divider margin: 0 8px 6px
DROPDOWN_DIVIDER_GAP = 6
DROPDOWN_ROW_PAD_X = 10             # row padding: 9px 10px
DROPDOWN_ROW_PAD_Y = 9
DROPDOWN_BAR_W = 3                  # selection bar 3x15, radius 2
DROPDOWN_BAR_H = 15
DROPDOWN_BAR_RADIUS = 2
DROPDOWN_BAR_GAP = 10               # gap between bar and text
DROPDOWN_FONT_PX = 14               # font: 500 14px/1 Inter
DROPDOWN_FONT_WEIGHT = 500          # QFont.Weight.Medium

# Open-height length limit: the popup shows at most this many rows, longer lists
# scroll. A thin thumb appears on the right edge when the list overflows.
DROPDOWN_MAX_VISIBLE_ROWS = 6
DROPDOWN_SCROLLBAR_W = 3
DROPDOWN_SCROLLBAR_MARGIN = 3       # inset from the card's right/top/bottom edges
DROPDOWN_SCROLLBAR_GAP = 2          # clearance between the highlight box and the thumb
DROPDOWN_SCROLLBAR_RGBA = (255, 255, 255, 46)

# Animation (reference data-props defaults: dur=250ms, stagger=40ms)
DROPDOWN_DURATION_MS = 250
DROPDOWN_STAGGER_MS = 30
DROPDOWN_CASCADE_DELAY_MS = 100     # rows wait this long before sliding in, so the
                                    # slide is visible after the popup has expanded
DROPDOWN_OPACITY_MS = 175           # round(dur * 0.7)
DROPDOWN_SLIDE_PX = 4               # popup translateY(-4px) -> translateY(0)
DROPDOWN_ROW_OFFSET_PX = 7          # row translateY(-7px) on enter
# Easing: field/popup/chevron use cubic-bezier(.22,1,.36,1); rows + opacity use
# the CSS default "ease" = cubic-bezier(.25,.1,.25,1).
DROPDOWN_EASE_SNAP = (0.22, 1.0, 0.36, 1.0)
DROPDOWN_EASE_STD = (0.25, 0.1, 0.25, 1.0)


def _curve(points) -> QEasingCurve:
    """Build a QEasingCurve from CSS cubic-bezier control points (x1,y1,x2,y2). The spline
    implicitly starts at (0,0) and must end at (1,1); the two control     points are passed as
    absolute [0,1] points, exactly like CSS cubic-bezier."""
    c = QEasingCurve(QEasingCurve.Type.BezierSpline)
    x1, y1, x2, y2 = points
    c.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return c


EASE_SNAP = _curve(DROPDOWN_EASE_SNAP)   # cubic-bezier(.22,1,.36,1)
EASE_STD = _curve(DROPDOWN_EASE_STD)     # CSS "ease"


def dropdown_font(px: int = DROPDOWN_FONT_PX) -> QFont:
    """Field/row font: 500 weight, Inter with system fallback (14px default)."""
    f = QFont()
    f.setFamilies(FONT_FALLBACKS)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight.Medium)  # 500
    f.setStyleStrategy(QFont.StyleStrategy.PreferMatch)
    return f


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
        round(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


def _rounded_path(rect: QRectF, tl: float, tr: float, br: float, bl: float) -> QPainterPath:
    """QPainterPath rounded rectangle with independent per-corner radii."""
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    # Clamp radii so they never exceed half the shortest side.
    m = min(w, h) / 2.0
    tl, tr, br, bl = (max(0.0, min(r, m)) for r in (tl, tr, br, bl))
    p = QPainterPath()
    p.moveTo(x + tl, y)
    p.lineTo(x + w - tr, y)
    if tr:
        p.arcTo(x + w - 2 * tr, y, 2 * tr, 2 * tr, 90, -90)
    p.lineTo(x + w, y + h - br)
    if br:
        p.arcTo(x + w - 2 * br, y + h - 2 * br, 2 * br, 2 * br, 0, -90)
    p.lineTo(x + bl, y + h)
    if bl:
        p.arcTo(x, y + h - 2 * bl, 2 * bl, 2 * bl, 270, -90)
    p.lineTo(x, y + tl)
    if tl:
        p.arcTo(x, y, 2 * tl, 2 * tl, 180, -90)
    p.closeSubpath()
    return p



def _field_border_path(rect: QRectF, top_r: float, bot_r: float) -> QPainterPath:
    """Field border without the bottom segment (bottom-left up and over the top to bottom-right).

    Bottom segment is painted separately in its faded colour; stroking here would leave AA fringes.
    """
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    m = min(w, h) / 2.0
    top_r, bot_r = (max(0.0, min(r, m)) for r in (top_r, bot_r))
    p = QPainterPath()
    p.moveTo(x + bot_r, y + h)
    if bot_r:
        p.arcTo(x, y + h - 2 * bot_r, 2 * bot_r, 2 * bot_r, 270, -90)
    p.lineTo(x, y + top_r)
    if top_r:
        p.arcTo(x, y, 2 * top_r, 2 * top_r, 180, -90)
        p.lineTo(x + w - top_r, y)
        p.arcTo(x + w - 2 * top_r, y, 2 * top_r, 2 * top_r, 90, -90)
    else:
        p.lineTo(x + w, y)
    p.lineTo(x + w, y + h - bot_r)
    if bot_r:
        p.arcTo(x + w - 2 * bot_r, y + h - 2 * bot_r, 2 * bot_r, 2 * bot_r, 0, -90)
    return p


# Popup imports helpers from this module, so this import cannot sit at the top.
from shared.dropdown_popup import _PopupCard, _SHADOW_HALO



class Dropdown(QWidget):
    """Custom-painted field that owns an animated popup. QComboBox-compatible."""

    currentTextChanged = Signal(str)
    currentIndexChanged = Signal(int)

    # State machine for robust rapid open/close.
    _CLOSED, _OPENING, _OPEN, _CLOSING = range(4)

    def __init__(self, min_width: int, parent=None, *,
                 field_bg: QColor | str | None = None,
                 border_w: float | None = None,
                 border_color: QColor | str | None = None,
                 border_hover: QColor | str | None = None,
                 radius: float | None = None,
                 text_color: QColor | str | None = None,
                 font_px: int | None = None,
                 pad_y: float | None = None):
        super().__init__(parent)
        self._min_width = min_width
        self._field_bg = QColor(field_bg) if field_bg is not None else QColor(DROPDOWN_FIELD_BG)
        self._border_w = float(border_w) if border_w is not None else float(DROPDOWN_BORDER_W)
        self._border_color = QColor(border_color) if border_color is not None else QColor(DROPDOWN_BORDER)
        self._border_hover = QColor(border_hover) if border_hover is not None else QColor(DROPDOWN_BORDER_HOVER)
        self._radius = float(radius) if radius is not None else float(DROPDOWN_RADIUS)
        self._text_color = QColor(text_color) if text_color is not None else QColor(DROPDOWN_TEXT)
        self._font_px = int(font_px) if font_px is not None else DROPDOWN_FONT_PX
        self._pad_y = float(pad_y) if pad_y is not None else float(DROPDOWN_FIELD_PAD_Y)
        self._items: list[tuple[str, object]] = []  # (text, userData)
        self._index = -1
        self._hover = False
        self._open_progress = 0.0  # 0 closed .. 1 open (field corners/chevron)
        self._state = self._CLOSED

        self._font = dropdown_font(self._font_px)
        self._fm = QFontMetricsF(self._font)
        self.setFont(self._font)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFixedHeight(self._field_height())
        self._apply_width()

        # Popup child of field top-level window (shared DPR); hidden on field until window exists.
        self._card = _PopupCard(self, field_bg=self._field_bg,
                                border_w=self._border_w,
                                border_color=self._border_color,
                                radius=self._radius,
                                text_color=self._text_color,
                                font_px=self._font_px)
        self._card.hide()
        self._card.activated.connect(self._on_row_activated)

        self._anim_field = self._mk_anim(self, b"openProgress")
        self._anim_reveal = self._mk_anim(self._card, b"reveal")
        self._anim_slide = self._mk_anim(self._card, b"slide")
        self._anim_cascade = self._mk_anim(self._card, b"cascade")
        self._anim_opacity = self._mk_anim(self._card, b"alpha")
        self._anim_reveal.finished.connect(self._on_reveal_finished)

        self._filter_installed = False

    def _mk_anim(self, target, prop) -> QPropertyAnimation:
        a = QPropertyAnimation(target, prop, self)
        return a

    # -- QComboBox-compatible API ------------------------------------------
    def addItem(self, text: str, userData=None):
        first = self._index < 0
        self._items.append((str(text), userData))
        self._apply_width()
        if first:
            # Mirror QComboBox: adding the first item moves the current index
            # from -1 to 0 and emits, which callers rely on to populate state.
            self._index = 0
            self.update()
            self.currentIndexChanged.emit(0)
            self.currentTextChanged.emit(self.itemText(0))
        else:
            self.update()

    def addItems(self, texts):
        for t in texts:
            self.addItem(t)

    def clear(self):
        had_current = self._index != -1
        self._items.clear()
        self._index = -1
        self._apply_width()
        self.update()
        if had_current:
            self.currentIndexChanged.emit(-1)
            self.currentTextChanged.emit("")

    def count(self) -> int:
        return len(self._items)

    def itemText(self, i: int) -> str:
        return self._items[i][0] if 0 <= i < len(self._items) else ""

    def itemData(self, i: int):
        return self._items[i][1] if 0 <= i < len(self._items) else None

    def currentIndex(self) -> int:
        return self._index

    def currentText(self) -> str:
        return self.itemText(self._index)

    def setCurrentIndex(self, i: int, *, emit: bool = True):
        if i == self._index or not (0 <= i < len(self._items)):
            if 0 <= i < len(self._items):
                self._index = i
                self.update()
            return
        self._index = i
        self.update()
        if emit:
            self.currentIndexChanged.emit(i)
            self.currentTextChanged.emit(self.currentText())

    def setCurrentText(self, text: str):
        for i, (t, _d) in enumerate(self._items):
            if t == text:
                self.setCurrentIndex(i)
                return

    def setSizeAdjustPolicy(self, *_a, **_k):
        # No-op: this widget always sizes to the widest item.
        pass

    def setMinimumWidth(self, w: int):
        self._min_width = max(self._min_width, int(w))
        self._apply_width()

    # -- sizing -------------------------------------------------------------
    def _field_height(self) -> int:
        # vertical padding + line box (font: .../1) + top/bottom border.
        content = max(self._font_px, DROPDOWN_CHEVRON_SIZE)
        return int(round(2 * self._pad_y + content + 2 * self._border_w))

    def _widest_text(self) -> float:
        widths = [self._fm.horizontalAdvance(t) for t, _ in self._items]
        return max(widths) if widths else 0.0

    def _field_width(self) -> int:
        need = (2 * self._border_w + 2 * DROPDOWN_FIELD_PAD_X
                + self._widest_text() + DROPDOWN_FIELD_GAP + DROPDOWN_CHEVRON_SIZE)
        return int(round(max(self._min_width, need)))

    def _apply_width(self):
        self.setFixedWidth(self._field_width())

    # -- field animatable property -----------------------------------------
    def _get_open(self) -> float:
        return self._open_progress

    def _set_open(self, v: float):
        self._open_progress = v
        self.update()

    openProgress = Property(float, _get_open, _set_open)

    # -- open / close -------------------------------------------------------
    def _configure(self, anim: QPropertyAnimation, end, duration, curve):
        anim.stop()
        name = anim.propertyName().data().decode()
        anim.setStartValue(anim.targetObject().property(name))
        anim.setEndValue(end)
        anim.setDuration(int(duration))
        anim.setEasingCurve(curve)
        anim.start()

    def _overlay_cap(self) -> float:
        """Max popup height that fits below the field within the window (0=none)."""
        win = self.window()
        if win is None or win is self:
            return 0.0
        field_bottom = self.mapTo(win, self.rect().bottomLeft()).y()
        return max(0.0, win.height() - field_bottom - 8)

    def open(self):
        if self._state in (self._OPEN, self._OPENING) or not self._items:
            return
        self._state = self._OPENING
        texts = [t for t, _ in self._items]
        self._card.set_items(texts, self._index, float(self.width()),
                             self._overlay_cap(), field_h=float(self.height()))
        self._reposition()
        self._card.cascade = 0.0   # fresh cascade on every open
        self._card.alpha = 0.0     # fade in from transparent
        self._card.show()
        self._card.raise_()

        self._configure(self._anim_field, 1.0, DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_reveal, self._card.full_height(),
                        DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_slide, 0.0, DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_opacity, 1.0, DROPDOWN_OPACITY_MS, EASE_STD)
        self._configure(self._anim_cascade, self._card.cascade_total(),
                        self._card.cascade_total(), QEasingCurve(QEasingCurve.Type.Linear))
        self._state = self._OPEN
        self._install_filter(True)

    def close(self):
        if self._state in (self._CLOSED, self._CLOSING):
            return
        self._state = self._CLOSING
        self._anim_cascade.stop()  # rows stay put on close (no re-stagger)
        self._configure(self._anim_field, 0.0, DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_reveal, 0.0, DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_slide, -float(DROPDOWN_SLIDE_PX),
                        DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_opacity, 0.0, DROPDOWN_OPACITY_MS, EASE_STD)
        self._install_filter(False)

    def _on_reveal_finished(self):
        if self._state == self._CLOSING:
            self._state = self._CLOSED
            self._card.hide()

    def _reposition(self):
        # Map field bottom-left to card content origin (_SHADOW_HALO offset).
        win = self.window()
        if self._card.parent() is not win:
            self._card.setParent(win)
        p = self.mapTo(win, self.rect().bottomLeft())
        self._card.move(p.x() - _SHADOW_HALO, p.y() - _SHADOW_HALO)

    def _on_row_activated(self, i: int):
        self.setCurrentIndex(i)
        self.close()

    # -- outside-click / move dismissal ------------------------------------
    def _install_filter(self, on: bool):
        app = QApplication.instance()
        if on and not self._filter_installed:
            app.installEventFilter(self)
            self._filter_installed = True
        elif not on and self._filter_installed:
            app.removeEventFilter(self)
            self._filter_installed = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        et = event.type()
        if et == QEvent.Type.MouseButtonPress and self._state in (self._OPEN, self._OPENING):
            gpos = event.globalPosition().toPoint()
            # Inside the visible popup content: let the card handle the row click.
            c0 = self._card.mapToGlobal(QPoint(_SHADOW_HALO, _SHADOW_HALO))
            content = QRectF(c0.x(), c0.y(), self.width(), self._card.full_height())
            if content.contains(gpos):
                return False
            # Anywhere else (including the field, which the overlay covers) closes
            # and swallows the click, mirroring the reference's dismiss overlay.
            self.close()
            return True
        # Close if the window moves/resizes or loses focus while open.
        if obj is self.window() and et in (
            QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.WindowDeactivate,
        ):
            if self._state in (self._OPEN, self._OPENING):
                self.close()
        return False

    # -- interaction --------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self._state in (self._OPEN, self._OPENING):
                self.close()
            else:
                self.open()
            e.accept()
            return
        super().mousePressEvent(e)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def hideEvent(self, e):
        # Never leave an orphaned popup if the field is hidden mid-animation.
        self._install_filter(False)
        self._card.hide()
        self._state = self._CLOSED
        super().hideEvent(e)

    # -- painting -----------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        op = self._open_progress
        w = float(self.width())
        h = float(self.height())
        hb = self._border_w / 2.0
        rect = QRectF(hb, hb, w - 2 * hb, h - 2 * hb)
        top_r = self._radius
        bot_r = self._radius * (1.0 - op)  # bottom corners square as it opens
        path = _rounded_path(rect, top_r, top_r, bot_r, bot_r)

        # Background.
        p.fillPath(path, self._field_bg)

        # Border (hover when hovered). Open: fade bottom segment to field bg to hide popup seam.
        # Paint that segment once in final colour; sides/top stroke after for full-width corners.
        border = self._border_hover if self._hover else self._border_color
        p.setBrush(Qt.BrushStyle.NoBrush)
        if op > 0:
            seam_pen = QPen(_lerp_color(border, self._field_bg, op), self._border_w)
            seam_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(seam_pen)
            p.drawLine(QPointF(bot_r + hb, h - hb), QPointF(w - bot_r - hb, h - hb))
            p.setPen(QPen(border, self._border_w))
            p.drawPath(_field_border_path(rect, top_r, bot_r))
        else:
            p.setPen(QPen(border, self._border_w))
            p.drawPath(path)

        # Field text.
        p.setFont(self._font)
        p.setPen(self._text_color)
        text_rect = QRectF(
            self._border_w + DROPDOWN_FIELD_PAD_X, 0,
            w - 2 * (self._border_w + DROPDOWN_FIELD_PAD_X)
            - DROPDOWN_FIELD_GAP - DROPDOWN_CHEVRON_SIZE,
            h,
        )
        p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self.currentText())

        self._paint_chevron(p, w, h, op)
        p.end()

    def _paint_chevron(self, p: QPainter, w: float, h: float, op: float):
        size = DROPDOWN_CHEVRON_SIZE
        cx = w - self._border_w - DROPDOWN_FIELD_PAD_X - size / 2.0
        cy = h / 2.0
        p.save()
        p.translate(cx, cy)
        p.rotate(180.0 * op)
        p.scale(size / 24.0, size / 24.0)   # svg viewBox is 24x24
        p.translate(-12, -12)
        pen = QPen(QColor(DROPDOWN_CHEVRON), DROPDOWN_CHEVRON_STROKE)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        path = QPainterPath()
        path.moveTo(6, 9)      # polyline points "6 9 12 15 18 9"
        path.lineTo(12, 15)
        path.lineTo(18, 9)
        p.strokePath(path, pen)
        p.restore()
