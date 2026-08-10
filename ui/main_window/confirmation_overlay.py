"""In-window modal confirmation overlay.

Non-blocking: signals and slots, never ``QDialog.exec()``.
Signature:: show_confirmation(parent, title, message, on_confirm, on_cancel=None)
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.main_window.overlay_chrome import (
    attach_overlay,
    begin_centered_outer,
    finish_centered_outer,
    OverlayCard,
    sync_overlay_to_parent,
)

_CARD_WIDTH = 380


class ConfirmationOverlay(QWidget):
    """Full-window semi-transparent overlay with a centred dialog card."""

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

        attach_overlay(self, parent)

        outer = begin_centered_outer(self)

        card = OverlayCard(_CARD_WIDTH)
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
        finish_centered_outer(outer, card)
        self._card = card

        self.show()
        self.raise_()

    def eventFilter(self, obj, event) -> bool:
        sync_overlay_to_parent(self, obj, event)
        return super().eventFilter(obj, event)

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


def show_confirmation(
    parent: QWidget,
    title: str,
    message: str,
    on_confirm: Callable[[], None],
    on_cancel: Callable[[], None] | None = None,
) -> ConfirmationOverlay:
    """Show a modal-style overlay inside *parent*. Returns the overlay widget."""
    return ConfirmationOverlay(parent, title, message, on_confirm, on_cancel)
