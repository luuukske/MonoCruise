"""Shared full-window overlay chrome: parent sync and a shrink-to-fit centred card."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QSize
from PySide6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

# Keep the card off the window edge when the host is only barely larger.
CARD_EDGE_PAD = 16


class OverlayCard(QFrame):
    """Prefers ``preferred_width`` but shrinks when the host is narrower."""

    def __init__(self, preferred_width: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._preferred_width = preferred_width
        self.setObjectName("dialogCard")
        self.setMaximumWidth(preferred_width)
        # Preferred vertical so wrapped labels keep their height-for-width; the
        # card still shrinks when the host is short (down to minimumSizeHint).
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

    def hasHeightForWidth(self) -> bool:
        lay = self.layout()
        return bool(lay is not None and lay.hasHeightForWidth())

    def heightForWidth(self, width: int) -> int:
        lay = self.layout()
        if lay is not None and lay.hasHeightForWidth():
            return lay.heightForWidth(width)
        return super().heightForWidth(width)

    def sizeHint(self) -> QSize:
        width = self._preferred_width
        if self.hasHeightForWidth():
            return QSize(width, self.heightForWidth(width))
        return QSize(width, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())


def attach_overlay(overlay: QWidget, parent: QWidget) -> None:
    """Fill *parent* and keep matching its size on resize."""
    overlay.setGeometry(parent.rect())
    overlay.setAutoFillBackground(True)
    parent.installEventFilter(overlay)


def sync_overlay_to_parent(overlay: QWidget, obj, event) -> bool:
    """``eventFilter`` helper: resize the overlay when its parent resizes."""
    if obj is overlay.parent() and event.type() == QEvent.Type.Resize:
        overlay.setGeometry(obj.rect())
    return False


def begin_centered_outer(overlay: QWidget) -> QVBoxLayout:
    """Outer column with edge pad and a leading stretch for vertical centring."""
    outer = QVBoxLayout(overlay)
    outer.setContentsMargins(CARD_EDGE_PAD, CARD_EDGE_PAD, CARD_EDGE_PAD, CARD_EDGE_PAD)
    outer.setSpacing(0)
    outer.addStretch(1)
    return outer


def finish_centered_outer(outer: QVBoxLayout, card: QWidget) -> None:
    """Centre *card* with side stretches (AlignHCenter clips height-for-width text)."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    row.addStretch(1)
    row.addWidget(card, 0)
    row.addStretch(1)
    outer.addLayout(row, 0)
    outer.addStretch(1)
