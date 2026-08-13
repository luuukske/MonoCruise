"""Popup card for shared.dropdown.Dropdown. See shared/README.md."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Property, QRectF
from PySide6.QtGui import QFontMetricsF, QPainter, QPainterPath, QColor, QPen
from PySide6.QtWidgets import QWidget, QGraphicsDropShadowEffect

from shared.dropdown import (
    DROPDOWN_BAR_GAP,
    DROPDOWN_BAR_H,
    DROPDOWN_BAR_RADIUS,
    DROPDOWN_BAR_W,
    DROPDOWN_BORDER,
    DROPDOWN_BORDER_W,
    DROPDOWN_CASCADE_DELAY_MS,
    DROPDOWN_DIVIDER,
    DROPDOWN_DIVIDER_GAP,
    DROPDOWN_DIVIDER_INSET,
    DROPDOWN_DURATION_MS,
    DROPDOWN_FIELD_BG,
    DROPDOWN_FONT_PX,
    DROPDOWN_MAX_VISIBLE_ROWS,
    DROPDOWN_POPUP_PAD,
    DROPDOWN_RADIUS,
    DROPDOWN_ROW_HOVER_RGBA,
    DROPDOWN_ROW_MARGIN_Y,
    DROPDOWN_ROW_OFFSET_PX,
    DROPDOWN_ROW_PAD_X,
    DROPDOWN_ROW_PAD_Y,
    DROPDOWN_ROW_RADIUS,
    DROPDOWN_SCROLLBAR_GAP,
    DROPDOWN_SCROLLBAR_MARGIN,
    DROPDOWN_SCROLLBAR_RGBA,
    DROPDOWN_SCROLLBAR_W,
    DROPDOWN_SELECTION_BAR,
    DROPDOWN_SHADOW_RGBA,
    DROPDOWN_SLIDE_PX,
    DROPDOWN_STAGGER_MS,
    DROPDOWN_TEXT,
    EASE_STD,
    _rounded_path,
    dropdown_font,
)


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

