"""
MonoCruise – In‑window modal confirmation overlay.

Non‑blocking: uses signals/slots, never ``QDialog.exec()``.

Signature::

    show_confirmation(parent, title, message, on_confirm, on_cancel=None)
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ConfirmationOverlay(QWidget):
    """Full‑window semi‑transparent overlay with a centred dialog card."""

    def __init__(
        self,
        parent: QWidget,
        title: str,
        message: str,
        on_confirm: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("overlayBg")
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel

        # Fill entire parent
        self.setGeometry(parent.rect())
        self.setAutoFillBackground(True)

        # Centred card
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("dialogCard")
        card.setFixedWidth(380)
        card_lay = QVBoxLayout(card)
        card_lay.setSpacing(14)
        card_lay.setContentsMargins(24, 20, 24, 20)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 15px; font-weight: bold; background: transparent;")
        title_lbl.setWordWrap(True)
        card_lay.addWidget(title_lbl)

        msg_lbl = QLabel(message)
        msg_lbl.setStyleSheet("font-size: 13px; background: transparent;")
        msg_lbl.setWordWrap(True)
        card_lay.addWidget(msg_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("dangerButton")
        cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(cancel_btn)

        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self._confirm)
        btn_row.addWidget(confirm_btn)

        card_lay.addLayout(btn_row)
        outer.addWidget(card)

        self.show()
        self.raise_()

    def _confirm(self) -> None:
        self._on_confirm()
        self._close()

    def _cancel(self) -> None:
        if self._on_cancel:
            self._on_cancel()
        self._close()

    def _close(self) -> None:
        self.hide()
        self.deleteLater()

    def resizeEvent(self, event) -> None:
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)


# Convenience function

def show_confirmation(
    parent: QWidget,
    title: str,
    message: str,
    on_confirm: Callable[[], None],
    on_cancel: Callable[[], None] | None = None,
) -> ConfirmationOverlay:
    """Show a modal‑style overlay inside *parent*.  Returns the overlay widget."""
    return ConfirmationOverlay(parent, title, message, on_confirm, on_cancel)
