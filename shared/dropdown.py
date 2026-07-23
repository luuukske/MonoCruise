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
from PySide6.QtWidgets import QWidget, QGraphicsDropShadowEffect, QApplication

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


def _open_border_path(rect: QRectF, br: float) -> QPainterPath:
    """Border outline with square top corners and NO top edge, so the popup fuses with the field
    (CSS ``border-top: none``). Traversed top-right -> down     the right edge -> across the
    bottom (rounded) -> up the left edge."""
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    br = max(0.0, min(br, min(w, h) / 2.0))
    p = QPainterPath()
    p.moveTo(x + w, y)
    p.lineTo(x + w, y + h - br)
    if br:
        p.arcTo(x + w - 2 * br, y + h - 2 * br, 2 * br, 2 * br, 0, -90)
    p.lineTo(x + br, y + h)
    if br:
        p.arcTo(x, y + h - 2 * br, 2 * br, 2 * br, 270, -90)
    p.lineTo(x, y)
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


# Transparent margin so the drop shadow fits inside the popup window.
_SHADOW_HALO = 48


class _SeamClippedShadow(QGraphicsDropShadowEffect):
    """Shadow clipped below content top; avoids a dark seam over the field (shared/README.md)."""

    def draw(self, painter) -> None:
        painter.save()
        # Logical widget coords: content top sits at y == _SHADOW_HALO.
        r = self.boundingRect()
        painter.setClipRect(QRectF(r.x(), _SHADOW_HALO,
                                   r.width(), r.height() - _SHADOW_HALO))
        super().draw(painter)
        painter.restore()


class _PopupCard(QWidget):
    """Popup card paint and reveal/slide/cascade motion. See shared/README.md."""

    activated = Signal(int)

    def __init__(self, parent=None, field_bg: QColor | str | None = None,
                 border_w: float | None = None,
                 border_color: QColor | str | None = None,
                 radius: float | None = None,
                 text_color: QColor | str | None = None,
                 font_px: int | None = None):
        super().__init__(parent)
        self._field_bg = QColor(field_bg) if field_bg is not None else QColor(DROPDOWN_FIELD_BG)
        self._border_w = float(border_w) if border_w is not None else float(DROPDOWN_BORDER_W)
        self._border_color = QColor(border_color) if border_color is not None else QColor(DROPDOWN_BORDER)
        self._radius = float(radius) if radius is not None else float(DROPDOWN_RADIUS)
        self._text_color = QColor(text_color) if text_color is not None else QColor(DROPDOWN_TEXT)
        self.setMouseTracking(True)
        self._font = dropdown_font(font_px if font_px is not None else DROPDOWN_FONT_PX)
        self._fm = QFontMetricsF(self._font)
        self._items: list[str] = []
        self._selected = -1
        self._hover = -1
        self._card_w = 0.0
        self._field_h = 0.0     # height of the field this popup drops from
        self._natural_h = 0.0   # height the full list wants
        self._full_h = 0.0      # actual (possibly screen-clamped) open height
        self._scroll = 0.0      # row scroll offset when clamped
        self._reveal = 0.0
        self._slide = -float(DROPDOWN_SLIDE_PX)
        self._cascade = 0.0
        self._alpha = 1.0  # whole-popup fade (content + shadow)
        self._anchor = 0             # rows above this index don't cascade (static)

        color = QColor(0, 0, 0)
        color.setAlpha(DROPDOWN_SHADOW_RGBA[3])
        shadow = _SeamClippedShadow(self)
        shadow.setOffset(0, 16)          # box-shadow: 0 16px ...
        shadow.setBlurRadius(30)         # ... 30px
        shadow.setColor(color)           # rgba(0,0,0,.45)
        self.setGraphicsEffect(shadow)

    # -- geometry -----------------------------------------------------------
    def row_height(self) -> float:
        # Row height: 2*pad + max(14px line box, bar); matches reference 33px rows.
        return 2 * DROPDOWN_ROW_PAD_Y + max(DROPDOWN_FONT_PX, DROPDOWN_BAR_H)

    def _content_top(self) -> float:
        # popup padding + divider line + divider bottom margin
        return DROPDOWN_POPUP_PAD + DROPDOWN_BORDER_W + DROPDOWN_DIVIDER_GAP

    def _rows_origin(self) -> float:
        # Row origin; ROW_MARGIN_Y inset cancels first/last pad (gaps between rows only).
        return self._content_top() - DROPDOWN_ROW_MARGIN_Y

    def _bottom_pad(self) -> float:
        return DROPDOWN_POPUP_PAD - DROPDOWN_ROW_MARGIN_Y

    def _field_rect_local(self) -> QRectF:
        # Field band above content top (shadow halo overlaps field for hand cursor).
        return QRectF(_SHADOW_HALO, _SHADOW_HALO - self._field_h,
                      self._card_w, self._field_h)

    def set_items(self, items: list[str], selected: int, card_w: float,
                  cap: float = 0.0, field_h: float = 0.0):
        self._items = list(items)
        self._selected = selected
        self._hover = -1
        self._scroll = 0.0
        self._card_w = card_w
        self._field_h = field_h
        self._natural_h = (self._rows_origin() + len(items) * self.row_height()
                           + self._bottom_pad())
        # Open height: min(natural, row cap, window cap); overflow scrolls with thumb.
        min_h = self._rows_origin() + self.row_height() + self._bottom_pad()
        row_cap = (self._rows_origin() + DROPDOWN_MAX_VISIBLE_ROWS * self.row_height()
                   + self._bottom_pad())
        limit = row_cap if cap <= 0 else min(cap, row_cap)
        self._full_h = max(min_h, min(self._natural_h, limit))
        # When overflow, scroll open so selected row is centred in the viewport.
        smax = self._scroll_max()
        if smax > 0 and 0 <= selected < len(items):
            rh = self.row_height()
            vp_h = self._full_h - self._rows_origin() - self._bottom_pad()
            target = selected * rh + rh / 2.0 - vp_h / 2.0
            self._scroll = max(0.0, min(smax, target))
        # _anchor = first visible row; rows above skip cascade when scrolled open.
        self._anchor = int(self._scroll // self.row_height())
        self.setFixedSize(int(round(card_w + 2 * _SHADOW_HALO)),
                          int(round(self._full_h + 2 * _SHADOW_HALO)))

    def full_height(self) -> float:
        return self._full_h

    def _scroll_max(self) -> float:
        return max(0.0, self._natural_h - self._full_h)

    def _rows_bottom(self) -> float:
        # Bottom y (card-local) of the scrollable rows viewport.
        return _SHADOW_HALO + self._slide + self._full_h - self._bottom_pad()

    def wheelEvent(self, e):
        if self._scroll_max() <= 0:
            return
        self._scroll = max(0.0, min(self._scroll_max(),
                                    self._scroll - e.angleDelta().y() / 2.0))
        # Re-evaluate hover under the new scroll position.
        self._hover = self._row_at(e.position().y())
        self.update()

    def cascade_total(self) -> float:
        # Only rows from the anchor down cascade; count from there so the clock
        # still runs long enough for every animating row to reach full opacity.
        n = len(self._items)
        return (DROPDOWN_CASCADE_DELAY_MS
                + max(0, n - 1 - self._anchor) * DROPDOWN_STAGGER_MS
                + DROPDOWN_DURATION_MS)

    # -- animatable properties ---------------------------------------------
    def _get_reveal(self) -> float:
        return self._reveal

    def _set_reveal(self, v: float):
        self._reveal = v
        # Re-render so the drop shadow follows the growing card silhouette.
        if self.graphicsEffect():
            self.graphicsEffect().update()
        self.update()

    reveal = Property(float, _get_reveal, _set_reveal)

    def _get_slide(self) -> float:
        return self._slide

    def _set_slide(self, v: float):
        self._slide = v
        self.update()

    slide = Property(float, _get_slide, _set_slide)

    def _get_cascade(self) -> float:
        return self._cascade

    def _set_cascade(self, v: float):
        self._cascade = v
        self.update()

    cascade = Property(float, _get_cascade, _set_cascade)

    def _get_alpha(self) -> float:
        return self._alpha

    def _set_alpha(self, v: float):
        self._alpha = v
        # The drop shadow is derived from the (now alpha'd) content, so it fades
        # in step -- no separate shadow animation needed.
        if self.graphicsEffect():
            self.graphicsEffect().update()
        self.update()

    alpha = Property(float, _get_alpha, _set_alpha)

    # -- hit testing --------------------------------------------------------
    def _row_at(self, y: float) -> int:
        # Ignore hits outside the rows viewport (e.g. over the divider).
        if not (_SHADOW_HALO + self._slide + self._rows_origin() <= y < self._rows_bottom()):
            return -1
        top = _SHADOW_HALO + self._slide + self._rows_origin() - self._scroll
        rh = self.row_height()
        for i in range(len(self._items)):
            if top + i * rh <= y < top + (i + 1) * rh:
                return i
        return -1

    def mouseMoveEvent(self, e):
        pos = e.position()
        idx = self._row_at(pos.y())
        # Only within the card's horizontal band.
        if not (_SHADOW_HALO <= pos.x() <= _SHADOW_HALO + self._card_w):
            idx = -1
        if idx != self._hover:
            self._hover = idx
            self.update()
        # Hand cursor on row or overlapping field band; arrow elsewhere.
        over_field = self._field_h > 0 and self._field_rect_local().contains(pos)
        self.setCursor(Qt.CursorShape.PointingHandCursor if (idx >= 0 or over_field)
                       else Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, e):
        if self._hover != -1:
            self._hover = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            idx = self._row_at(e.position().y())
            x = e.position().x()
            if idx >= 0 and _SHADOW_HALO <= x <= _SHADOW_HALO + self._card_w:
                self.activated.emit(idx)
                return
        super().mousePressEvent(e)

    # -- painting -----------------------------------------------------------
    def paintEvent(self, event):
        if self._reveal <= 0.5:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        p.setOpacity(self._alpha)  # whole-popup fade (rows multiply on top)

        ox = _SHADOW_HALO
        oy = _SHADOW_HALO + self._slide
        w = self._card_w
        h = self._reveal
        br = min(self._radius, h / 2.0)

        # Fused top (square corners), rounded bottom; top row opaque to hide shadow seam.
        hb = self._border_w / 2.0
        outer = QRectF(ox + hb, oy, w - 2 * hb, h - hb)
        path = _rounded_path(outer, 0, 0, br, br)

        p.save()
        p.setClipPath(path)
        p.fillRect(QRectF(ox, oy, w, h), self._field_bg)

        # Inset divider (1px, #383838, 8px left/right inset), just under the top.
        dy = oy + DROPDOWN_POPUP_PAD
        p.setPen(Qt.PenStyle.NoPen)
        p.fillRect(
            QRectF(ox + DROPDOWN_POPUP_PAD + DROPDOWN_DIVIDER_INSET, dy,
                   w - 2 * (DROPDOWN_POPUP_PAD + DROPDOWN_DIVIDER_INSET),
                   DROPDOWN_BORDER_W),
            QColor(DROPDOWN_DIVIDER),
        )

        self._paint_rows(p, ox, oy)
        self._paint_scrollbar(p, ox, oy)
        p.restore()

        # Border: left/bottom/right only (no top edge -> fuses with the
        # field, matching CSS border-top:none).
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(self._border_color, self._border_w))
        p.drawPath(_open_border_path(outer, br))
        p.end()

    def _paint_rows(self, p: QPainter, ox: float, oy: float):
        rh = self.row_height()
        pad = DROPDOWN_POPUP_PAD
        top = oy + self._rows_origin() - self._scroll
        row_w = self._card_w - 2 * pad
        # Shrink row highlight width when scrollbar visible (clear the thumb).
        if self._scroll_max() > 0:
            row_w = (self._card_w - DROPDOWN_SCROLLBAR_MARGIN - DROPDOWN_SCROLLBAR_W
                     - DROPDOWN_SCROLLBAR_GAP - pad)
        dur = DROPDOWN_DURATION_MS
        p.setFont(self._font)

        # Clip rows to their viewport so scrolled content never overlaps the
        # divider or the bottom padding. (No-op when the list fits.)
        if self._scroll_max() > 0:
            p.setClipRect(QRectF(ox, oy + self._rows_origin(), self._card_w,
                                 self._full_h - self._rows_origin() - self._bottom_pad()),
                          Qt.ClipOperation.IntersectClip)

        for i, text in enumerate(self._items):
            # Rows before anchor static; from anchor down use normal stagger cascade.
            ai = i - self._anchor
            if ai < 0:
                prog = 1.0
            else:
                local = self._cascade - DROPDOWN_CASCADE_DELAY_MS - ai * DROPDOWN_STAGGER_MS
                t = 0.0 if local <= 0 else (1.0 if local >= dur else local / dur)
                prog = EASE_STD.valueForProgress(t)
            row_op = prog
            row_dy = -DROPDOWN_ROW_OFFSET_PX * (1.0 - prog)
            if row_op <= 0.01:
                continue

            ry = top + i * rh + row_dy
            p.save()
            p.setOpacity(row_op * self._alpha)

            # Highlight inset by ROW_MARGIN_Y (inter-row gap; row pitch unchanged).
            my = DROPDOWN_ROW_MARGIN_Y
            hl_rect = QRectF(ox + pad, ry + my, row_w, rh - 2 * my)
            selected = (i == self._selected)
            if selected or i == self._hover:
                hov = QColor(*DROPDOWN_ROW_HOVER_RGBA)
                rp = QPainterPath()
                rp.addRoundedRect(hl_rect, DROPDOWN_ROW_RADIUS, DROPDOWN_ROW_RADIUS)
                p.fillPath(rp, hov)

            # Selection bar (3x15, radius 2) at the row's left padding.
            bar_x = ox + pad + DROPDOWN_ROW_PAD_X
            bar_y = ry + (rh - DROPDOWN_BAR_H) / 2.0
            if selected:
                bp = QPainterPath()
                bp.addRoundedRect(
                    QRectF(bar_x, bar_y, DROPDOWN_BAR_W, DROPDOWN_BAR_H),
                    DROPDOWN_BAR_RADIUS, DROPDOWN_BAR_RADIUS,
                )
                p.fillPath(bp, QColor(DROPDOWN_SELECTION_BAR))

            # Row text.
            text_x = bar_x + DROPDOWN_BAR_W + DROPDOWN_BAR_GAP
            p.setPen(self._text_color)
            p.drawText(QRectF(text_x, ry, row_w, rh),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            p.restore()

    def _paint_scrollbar(self, p: QPainter, ox: float, oy: float):
        # Thin overflow thumb on the right; size and position follow scroll ratio.
        smax = self._scroll_max()
        content_h = self._natural_h - self._rows_origin() - self._bottom_pad()
        if smax <= 0 or content_h <= 0:
            return
        inset = DROPDOWN_SCROLLBAR_MARGIN
        sb_w = DROPDOWN_SCROLLBAR_W
        vp_top = oy + self._rows_origin()
        vp_h = self._full_h - self._rows_origin() - self._bottom_pad()
        track_top = vp_top + inset
        track_h = vp_h - 2 * inset
        if track_h <= 0:
            return
        thumb_h = max(sb_w * 2.0, track_h * vp_h / content_h)
        thumb_y = track_top + (self._scroll / smax) * (track_h - thumb_h)
        sb_x = ox + self._card_w - inset - sb_w
        p.save()
        p.setOpacity(self._alpha)
        tp = QPainterPath()
        tp.addRoundedRect(QRectF(sb_x, thumb_y, sb_w, thumb_h), sb_w / 2.0, sb_w / 2.0)
        p.fillPath(tp, QColor(*DROPDOWN_SCROLLBAR_RGBA))
        p.restore()


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
