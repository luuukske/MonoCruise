"""Custom animated channel/version dropdown for the MonoCruise updater.

This is a pixel-faithful port of the reference HTML/CSS/JS dropdown
("MonoCruise Dropdown.dc.html"). It is a fully custom-painted widget rather
than a styled ``QComboBox`` because Qt Style Sheets cannot reproduce the
reference's box-shadow, CSS transitions, ``transform: rotate`` chevron or the
animated field ``border-radius``. Everything is drawn in ``paintEvent`` and
animated with ``QPropertyAnimation`` so it renders identically under the native
Windows style and Fusion.

Behaviour (mirrors the reference):
- Field with the current value + a chevron that rotates 180 deg on open.
- Popup extends *down* from the field, sharing its background so the two fuse
  into one rounded rectangle (field bottom corners square when open).
- On open: height + opacity grow together and rows cascade in with a per-row
  stagger. On close: a clean roll-up (rows do NOT re-stagger; the clip hides
  them). Chevron and corners animate in sync.
- Selection is marked by a 3x15 bar at the row's left edge (no checkmark).
- Dismisses on an outside click; closes if the owning window moves/resizes.

``Dropdown`` exposes the subset of the ``QComboBox`` API the updater uses
(addItem/addItems/clear/count/itemData/currentText/currentIndex/setCurrentIndex
/setCurrentText + currentTextChanged/currentIndexChanged) so it is a drop-in
replacement.
"""

from __future__ import annotations

from PySide6.QtCore import (
    Qt, Signal, Property, QPropertyAnimation, QPoint, QPointF, QRectF, QEasingCurve,
    QEvent, QObject,
)
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QPen, QFont, QFontMetricsF, QCursor,
)
from PySide6.QtWidgets import QWidget, QGraphicsDropShadowEffect, QApplication

import styles as S


def _curve(points) -> QEasingCurve:
    """Build a QEasingCurve from CSS cubic-bezier control points (x1,y1,x2,y2).

    The spline implicitly starts at (0,0) and must end at (1,1); the two control
    points are passed as absolute [0,1] points, exactly like CSS cubic-bezier.
    """
    c = QEasingCurve(QEasingCurve.Type.BezierSpline)
    x1, y1, x2, y2 = points
    c.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return c


EASE_SNAP = _curve(S.DROPDOWN_EASE_SNAP)   # cubic-bezier(.22,1,.36,1)
EASE_STD = _curve(S.DROPDOWN_EASE_STD)     # CSS "ease"


def dropdown_font() -> QFont:
    """Field/row font: 500 weight, 14px, Inter with system fallback."""
    f = QFont()
    f.setFamilies(S.FONT_FALLBACKS)
    f.setPixelSize(S.DROPDOWN_FONT_PX)
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
    """Border outline with square top corners and NO top edge, so the popup
    fuses with the field (CSS ``border-top: none``). Traversed top-right -> down
    the right edge -> across the bottom (rounded) -> up the left edge."""
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


# Shadow halo (transparent margin) around the popup card so the drop shadow has
# room to render inside the top-level popup window.
_SHADOW_HALO = 48


class _PopupCard(QWidget):
    """The visible popup card: background, border, inset divider and rows.

    Painted at an internal (halo, halo) offset so the drop shadow (installed on
    this widget) has transparent room on every side. All motion is driven by the
    animatable ``reveal`` (clip height), ``slide`` (translateY) and ``cascade``
    (row stagger clock) properties.
    """

    activated = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._font = dropdown_font()
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
        self._slide = -float(S.DROPDOWN_SLIDE_PX)
        self._cascade = 0.0
        self._alpha = 1.0  # whole-popup fade (content + shadow)
        self._anchor = 0             # rows above this index don't cascade (static)

        color = QColor(0, 0, 0)
        color.setAlpha(S.DROPDOWN_SHADOW_RGBA[3])
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setOffset(0, 16)          # box-shadow: 0 16px ...
        shadow.setBlurRadius(30)         # ... 30px
        shadow.setColor(color)           # rgba(0,0,0,.45)
        self.setGraphicsEffect(shadow)

    # -- geometry -----------------------------------------------------------
    def row_height(self) -> float:
        # Match the reference's row box: padding + max(line box, selection bar).
        # Use the nominal 14px line box (CSS line-height:1), like the field does,
        # not QFontMetricsF.height() -- that keeps rows at the reference's 33px
        # and integer-aligned so the bottom border stays crisp.
        return 2 * S.DROPDOWN_ROW_PAD_Y + max(S.DROPDOWN_FONT_PX, S.DROPDOWN_BAR_H)

    def _content_top(self) -> float:
        # popup padding + divider line + divider bottom margin
        return S.DROPDOWN_POPUP_PAD + S.DROPDOWN_BORDER_W + S.DROPDOWN_DIVIDER_GAP

    def _rows_origin(self) -> float:
        # Origin of the row rects. Each highlight box is inset ROW_MARGIN_Y within
        # its rh-tall row to open the inter-row gap; that inset would also push the
        # first box down and the last box up, inflating the section's top/bottom
        # margins. Shifting the origin up by the inset (and trimming the bottom pad
        # by the same below) cancels that, so the box block's outer spacing matches
        # the left/right padding -- only the gaps *between* rows remain.
        return self._content_top() - S.DROPDOWN_ROW_MARGIN_Y

    def _bottom_pad(self) -> float:
        return S.DROPDOWN_POPUP_PAD - S.DROPDOWN_ROW_MARGIN_Y

    def _field_rect_local(self) -> QRectF:
        # The field this popup drops from sits directly above the card's content
        # origin (content top-left == field bottom-left). The card's transparent
        # shadow halo overlaps the whole field, so expose that band to keep the
        # field's hand cursor when hovered while the popup is open.
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
        # Clamp the open height to the shortest of: the natural height, the
        # length limit (DROPDOWN_MAX_VISIBLE_ROWS), and the window cap (so a long
        # list never runs off screen). Overflow becomes wheel-scrollable with a
        # thumb. When the list fits within the limit, behaviour matches the
        # reference (no scroll).
        min_h = self._rows_origin() + self.row_height() + self._bottom_pad()
        row_cap = (self._rows_origin() + S.DROPDOWN_MAX_VISIBLE_ROWS * self.row_height()
                   + self._bottom_pad())
        limit = row_cap if cap <= 0 else min(cap, row_cap)
        self._full_h = max(min_h, min(self._natural_h, limit))
        # When the list overflows, open scrolled so the selected row is centred in
        # the viewport (clamped to the scroll range) instead of always starting at
        # the top -- the current version is visible immediately.
        smax = self._scroll_max()
        if smax > 0 and 0 <= selected < len(items):
            rh = self.row_height()
            vp_h = self._full_h - self._rows_origin() - self._bottom_pad()
            target = selected * rh + rh / 2.0 - vp_h / 2.0
            self._scroll = max(0.0, min(smax, target))
        # Cascade anchor: the first on-screen row. A scrolled-open list cascades
        # only its visible window; rows scrolled off the top render static, so a
        # mid/end-of-list open doesn't wait for them to tick through. Derived from
        # the scroll offset and row height (which follow the max visible length),
        # so nothing is hard-coded -- it adapts if the length limit changes. A
        # top-of-list open has scroll == 0, so anchor 0: identical to before.
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
        return (S.DROPDOWN_CASCADE_DELAY_MS
                + max(0, n - 1 - self._anchor) * S.DROPDOWN_STAGGER_MS
                + S.DROPDOWN_DURATION_MS)

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
        # Show the clickable hand cursor over a row OR the field the card's halo
        # overlaps (so the field keeps its cursor while open); arrow otherwise
        # (divider / padding / empty halo). Set every move: crossing between the
        # field and the empty halo doesn't change the hovered row.
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
        radius = S.DROPDOWN_RADIUS
        br = min(radius, h / 2.0)

        # Card background: top corners square (fused with the field), bottom
        # rounded. Inset by half the border on every side so the full 1px stroke
        # lands inside the widget and the bottom edge stays as crisp as the sides.
        hb = S.DROPDOWN_BORDER_W / 2.0
        outer = QRectF(ox + hb, oy + hb, w - 2 * hb, h - 2 * hb)
        path = _rounded_path(outer, 0, 0, br, br)

        p.save()
        p.setClipPath(path)
        p.fillRect(QRectF(ox, oy, w, h), QColor(S.DROPDOWN_FIELD_BG))

        # Inset divider (1px, #383838, 8px left/right inset), just under the top.
        dy = oy + S.DROPDOWN_POPUP_PAD
        p.setPen(Qt.PenStyle.NoPen)
        p.fillRect(
            QRectF(ox + S.DROPDOWN_POPUP_PAD + S.DROPDOWN_DIVIDER_INSET, dy,
                   w - 2 * (S.DROPDOWN_POPUP_PAD + S.DROPDOWN_DIVIDER_INSET),
                   S.DROPDOWN_BORDER_W),
            QColor(S.DROPDOWN_DIVIDER),
        )

        self._paint_rows(p, ox, oy)
        self._paint_scrollbar(p, ox, oy)
        p.restore()

        # Border: 1px #3a3a3a, left/bottom/right only (no top edge -> fuses with
        # the field, matching CSS border-top:none).
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(S.DROPDOWN_BORDER), S.DROPDOWN_BORDER_W))
        p.drawPath(_open_border_path(outer, br))
        p.end()

    def _paint_rows(self, p: QPainter, ox: float, oy: float):
        rh = self.row_height()
        pad = S.DROPDOWN_POPUP_PAD
        top = oy + self._rows_origin() - self._scroll
        row_w = self._card_w - 2 * pad
        # When the scrollbar is visible, pull the highlight boxes in on the right
        # so they clear the thumb instead of sliding under it. (Left-aligned text
        # is unaffected.)
        if self._scroll_max() > 0:
            row_w = (self._card_w - S.DROPDOWN_SCROLLBAR_MARGIN - S.DROPDOWN_SCROLLBAR_W
                     - S.DROPDOWN_SCROLLBAR_GAP - pad)
        dur = S.DROPDOWN_DURATION_MS
        p.setFont(self._font)

        # Clip rows to their viewport so scrolled content never overlaps the
        # divider or the bottom padding. (No-op when the list fits.)
        if self._scroll_max() > 0:
            p.setClipRect(QRectF(ox, oy + self._rows_origin(), self._card_w,
                                 self._full_h - self._rows_origin() - self._bottom_pad()),
                          Qt.ClipOperation.IntersectClip)

        for i, text in enumerate(self._items):
            # Rows above the anchor render static (no slide) so a mid/end open
            # doesn't wait for them. From the anchor down it's the same cascade as
            # before: row (i-anchor) starts at that many staggers past the delay.
            ai = i - self._anchor
            if ai < 0:
                prog = 1.0
            else:
                local = self._cascade - S.DROPDOWN_CASCADE_DELAY_MS - ai * S.DROPDOWN_STAGGER_MS
                t = 0.0 if local <= 0 else (1.0 if local >= dur else local / dur)
                prog = EASE_STD.valueForProgress(t)
            row_op = prog
            row_dy = -S.DROPDOWN_ROW_OFFSET_PX * (1.0 - prog)
            if row_op <= 0.01:
                continue

            ry = top + i * rh + row_dy
            p.save()
            p.setOpacity(row_op * self._alpha)

            # Highlight box is inset vertically by ROW_MARGIN_Y top and bottom so
            # a symmetric gap opens between adjacent boxes; the row pitch (rh) and
            # text/bar positions are untouched, only the painted box shrinks.
            my = S.DROPDOWN_ROW_MARGIN_Y
            hl_rect = QRectF(ox + pad, ry + my, row_w, rh - 2 * my)
            selected = (i == self._selected)
            if selected or i == self._hover:
                hov = QColor(*S.DROPDOWN_ROW_HOVER_RGBA)
                rp = QPainterPath()
                rp.addRoundedRect(hl_rect, S.DROPDOWN_ROW_RADIUS, S.DROPDOWN_ROW_RADIUS)
                p.fillPath(rp, hov)

            # Selection bar (3x15, radius 2) at the row's left padding.
            bar_x = ox + pad + S.DROPDOWN_ROW_PAD_X
            bar_y = ry + (rh - S.DROPDOWN_BAR_H) / 2.0
            if selected:
                bp = QPainterPath()
                bp.addRoundedRect(
                    QRectF(bar_x, bar_y, S.DROPDOWN_BAR_W, S.DROPDOWN_BAR_H),
                    S.DROPDOWN_BAR_RADIUS, S.DROPDOWN_BAR_RADIUS,
                )
                p.fillPath(bp, QColor(S.DROPDOWN_SELECTION_BAR))

            # Row text.
            text_x = bar_x + S.DROPDOWN_BAR_W + S.DROPDOWN_BAR_GAP
            p.setPen(QColor(S.DROPDOWN_TEXT))
            p.drawText(QRectF(text_x, ry, row_w, rh),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            p.restore()

    def _paint_scrollbar(self, p: QPainter, ox: float, oy: float):
        # Thin rounded thumb on the right edge, shown only when the list overflows
        # the length limit. Sized to the visible/total ratio and positioned by the
        # current scroll offset; inset from the edges so it clears the corners.
        smax = self._scroll_max()
        content_h = self._natural_h - self._rows_origin() - self._bottom_pad()
        if smax <= 0 or content_h <= 0:
            return
        inset = S.DROPDOWN_SCROLLBAR_MARGIN
        sb_w = S.DROPDOWN_SCROLLBAR_W
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
        p.fillPath(tp, QColor(*S.DROPDOWN_SCROLLBAR_RGBA))
        p.restore()


class Dropdown(QWidget):
    """Custom-painted field that owns an animated popup. QComboBox-compatible."""

    currentTextChanged = Signal(str)
    currentIndexChanged = Signal(int)

    # State machine for robust rapid open/close.
    _CLOSED, _OPENING, _OPEN, _CLOSING = range(4)

    def __init__(self, min_width: int, parent=None):
        super().__init__(parent)
        self._min_width = min_width
        self._items: list[tuple[str, object]] = []  # (text, userData)
        self._index = -1
        self._hover = False
        self._open_progress = 0.0  # 0 closed .. 1 open (field corners/chevron)
        self._state = self._CLOSED

        self._font = dropdown_font()
        self._fm = QFontMetricsF(self._font)
        self.setFont(self._font)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFixedHeight(self._field_height())
        self._apply_width()

        # The popup is a child overlay of the field's TOP-LEVEL WINDOW (not a
        # separate top-level). Sharing one window means one coordinate space and
        # one device-pixel ratio, so the popup aligns to the field exactly even
        # on fractional-DPI (4K) displays -- a separate top-level can only be
        # positioned in integer logical px and lands ~1 device px off. It is
        # reparented to the window lazily in open() (the window isn't known yet).
        self._card = _PopupCard(None)
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
        # padding 9px top/bottom + 14px line box (font: .../1) + 1px top/bottom border.
        content = max(S.DROPDOWN_FONT_PX, S.DROPDOWN_CHEVRON_SIZE)
        return int(round(2 * S.DROPDOWN_FIELD_PAD_Y + content + 2 * S.DROPDOWN_BORDER_W))

    def _widest_text(self) -> float:
        widths = [self._fm.horizontalAdvance(t) for t, _ in self._items]
        return max(widths) if widths else 0.0

    def _field_width(self) -> int:
        need = (2 * S.DROPDOWN_BORDER_W + 2 * S.DROPDOWN_FIELD_PAD_X
                + self._widest_text() + S.DROPDOWN_FIELD_GAP + S.DROPDOWN_CHEVRON_SIZE)
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

        self._configure(self._anim_field, 1.0, S.DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_reveal, self._card.full_height(),
                        S.DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_slide, 0.0, S.DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_opacity, 1.0, S.DROPDOWN_OPACITY_MS, EASE_STD)
        self._configure(self._anim_cascade, self._card.cascade_total(),
                        self._card.cascade_total(), QEasingCurve(QEasingCurve.Type.Linear))
        self._state = self._OPEN
        self._install_filter(True)

    def close(self):
        if self._state in (self._CLOSED, self._CLOSING):
            return
        self._state = self._CLOSING
        self._anim_cascade.stop()  # rows stay put on close (no re-stagger)
        self._configure(self._anim_field, 0.0, S.DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_reveal, 0.0, S.DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_slide, -float(S.DROPDOWN_SLIDE_PX),
                        S.DROPDOWN_DURATION_MS, EASE_SNAP)
        self._configure(self._anim_opacity, 0.0, S.DROPDOWN_OPACITY_MS, EASE_STD)
        self._install_filter(False)

    def _on_reveal_finished(self):
        if self._state == self._CLOSING:
            self._state = self._CLOSED
            self._card.hide()

    def _reposition(self):
        # Card content is painted at (_SHADOW_HALO, _SHADOW_HALO) inside the card
        # widget. Place the card so that content top-left lands on the field's
        # bottom-left (rect().bottomLeft() is the last row -> a 1px overlap) in
        # the shared window coordinate space, so it aligns to the exact device
        # pixel at any DPI.
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
        hb = S.DROPDOWN_BORDER_W / 2.0
        rect = QRectF(hb, hb, w - 2 * hb, h - 2 * hb)
        top_r = S.DROPDOWN_RADIUS
        bot_r = S.DROPDOWN_RADIUS * (1.0 - op)  # bottom corners square as it opens
        path = _rounded_path(rect, top_r, top_r, bot_r, bot_r)

        # Background.
        p.fillPath(path, QColor(S.DROPDOWN_FIELD_BG))

        # Border. On hover: #555; otherwise #3a3a3a.
        border = QColor(S.DROPDOWN_BORDER_HOVER if self._hover else S.DROPDOWN_BORDER)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(border, S.DROPDOWN_BORDER_W))
        p.drawPath(path)
        # Bottom border fades to the background as it opens so the seam with the
        # popup disappears (CSS: border-bottom-color -> #242424).
        if op > 0:
            seam = _lerp_color(border, QColor(S.DROPDOWN_FIELD_BG), op)
            p.setPen(QPen(seam, S.DROPDOWN_BORDER_W))
            p.drawLine(QPointF(bot_r + hb, h - hb), QPointF(w - bot_r - hb, h - hb))

        # Field text.
        p.setFont(self._font)
        p.setPen(QColor(S.DROPDOWN_TEXT))
        text_rect = QRectF(
            S.DROPDOWN_BORDER_W + S.DROPDOWN_FIELD_PAD_X, 0,
            w - 2 * (S.DROPDOWN_BORDER_W + S.DROPDOWN_FIELD_PAD_X)
            - S.DROPDOWN_FIELD_GAP - S.DROPDOWN_CHEVRON_SIZE,
            h,
        )
        p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self.currentText())

        self._paint_chevron(p, w, h, op)
        p.end()

    def _paint_chevron(self, p: QPainter, w: float, h: float, op: float):
        size = S.DROPDOWN_CHEVRON_SIZE
        cx = w - S.DROPDOWN_BORDER_W - S.DROPDOWN_FIELD_PAD_X - size / 2.0
        cy = h / 2.0
        p.save()
        p.translate(cx, cy)
        p.rotate(180.0 * op)
        p.scale(size / 24.0, size / 24.0)   # svg viewBox is 24x24
        p.translate(-12, -12)
        pen = QPen(QColor(S.DROPDOWN_CHEVRON), S.DROPDOWN_CHEVRON_STROKE)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        path = QPainterPath()
        path.moveTo(6, 9)      # polyline points "6 9 12 15 18 9"
        path.lineTo(12, 15)
        path.lineTo(18, 9)
        p.strokePath(path, pen)
        p.restore()
