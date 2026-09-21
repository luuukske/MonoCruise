"""In-window support prompt: share links, a GitHub star and a Patreon line.

Non-blocking like the other overlays: signals and slots, never ``QDialog.exec()``.
Shown by the main window on the schedule in core/usage_hours.py.
Signature:: show_support(parent, on_dismiss=None)
"""
from __future__ import annotations

import logging
import os
import webbrowser
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.main_window.constants import (
    GITHUB_URL,
    PATREON_URL,
    PROJECT_URL,
    YOUTUBE_URL,
)
from ui.main_window.overlay_chrome import (
    attach_overlay,
    begin_centered_outer,
    finish_centered_outer,
    OverlayCard,
    sync_overlay_to_parent,
)

logger = logging.getLogger(__name__)

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Five share buttons with brand marks land near 510 px at the shipped font.
# The slack over that is what absorbs a wider font or a DPI change.
_CARD_WIDTH = 600

TITLE = "Hey, Lukas here \N{WAVING HAND SIGN}"
BODY = (
    "Glad to see you're getting some kilometres in with MonoCruise. It's just "
    "me behind it, so every share or star helps more than you'd think."
)
SHARE_LABEL = "Share on:"
SHARE_HINT = "Opens the site and copies the link you can paste."
COPIED_HINT = "Link copied, paste it into your post."
COPY_FAILED_HINT = "Could not copy the link, it is on the MonoCruise site."
DISMISS_LABEL = "Maybe later"

# The bare link and nothing else. People word their own post; a canned blurb
# pasted verbatim across forums reads as spam.
SHARE_LINK = PROJECT_URL


@dataclass(frozen=True)
class ShareTarget:
    """One share destination. *icon* is a file in assets/, used when present."""

    label: str
    url: str
    icon: str


# Steam app 227300 is Euro Truck Simulator 2. Every URL here is a landing page
# on purpose: none of these boards has a share intent to deep link into.
SHARE_TARGETS: tuple[ShareTarget, ...] = (
    ShareTarget("TruckersMP", "https://forum.truckersmp.com/", "truckersmp.png"),
    ShareTarget("SCS Forum", "https://forum.scssoft.com/", "scsforum.png"),
    ShareTarget("Reddit", "https://www.reddit.com/r/trucksim/", "reddit.png"),
    ShareTarget("YouTube", YOUTUBE_URL, "youtube.png"),
    ShareTarget(
        "Steam discussions",
        "https://steamcommunity.com/app/227300/discussions/",
        "steam.png",
    ),
)

STAR_LABEL = "GitHub"
PATREON_LABEL = "Patreon"

# Brand marks are square and sit beside 11px text, so they follow the cap height
# rather than Qt's default 16px, which reads as oversized next to a small label.
_ICON_PX = 14

# QPushButton has no icon-to-text spacing, so the label carries it. Same trick
# the settings panel button bar uses; ``ShareTarget.label`` stays clean.
_ICON_GAP = "  "


def _apply_icon(button: QPushButton, icon_name: str) -> None:
    """Best effort: a missing logo leaves the button as its text label, ungapped."""
    path = os.path.join(_ASSET_DIR, icon_name)
    if not os.path.exists(path):
        return
    button.setIcon(QIcon(QPixmap(path)))
    button.setIconSize(QSize(_ICON_PX, _ICON_PX))
    button.setText(_ICON_GAP + button.text())


def _open(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        logger.exception("could not open a support link in the browser")


def _copy(text: str) -> bool:
    """Put *text* on the clipboard. False when there is no clipboard to write to."""
    try:
        clipboard = QGuiApplication.clipboard()
    except Exception:
        clipboard = None
    if clipboard is None:
        logger.debug("no clipboard available for the share text")
        return False
    try:
        clipboard.setText(text)
    except Exception:
        logger.exception("could not copy the share text to the clipboard")
        return False
    return True


class SupportOverlay(QWidget):
    """Full-window overlay asking for a share or a star. Every exit dismisses."""

    def __init__(
        self,
        parent: QWidget,
        *,
        on_dismiss: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("overlayBg")
        self._on_dismiss = on_dismiss
        self._dismissed = False

        attach_overlay(self, parent)

        outer = begin_centered_outer(self)

        card = OverlayCard(_CARD_WIDTH)
        # Margins and spacing match confirmation_overlay so every prompt in the
        # window reads as the same component.
        card_lay = QVBoxLayout(card)
        card_lay.setSpacing(14)
        card_lay.setContentsMargins(24, 20, 24, 20)

        title_lbl = QLabel(TITLE)
        title_lbl.setStyleSheet(
            "font-size: 15px; font-weight: bold; background: transparent;"
        )
        title_lbl.setWordWrap(True)
        title_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        card_lay.addWidget(title_lbl)

        body_lbl = QLabel(BODY)
        body_lbl.setStyleSheet("font-size: 13px; background: transparent;")
        body_lbl.setWordWrap(True)
        body_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        card_lay.addWidget(body_lbl)

        share_lbl = QLabel(SHARE_LABEL)
        share_lbl.setStyleSheet("font-size: 13px; background: transparent;")
        card_lay.addWidget(share_lbl)

        share_row = QHBoxLayout()
        share_row.setSpacing(6)
        share_row.setContentsMargins(0, 0, 0, 0)
        self._share_buttons: list[QPushButton] = []
        for target in SHARE_TARGETS:
            btn = QPushButton(target.label)
            btn.setObjectName("shareButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            _apply_icon(btn, target.icon)
            btn.clicked.connect(lambda _checked=False, t=target: self._share(t))
            share_row.addWidget(btn, 1)
            self._share_buttons.append(btn)
        card_lay.addLayout(share_row)

        self._hint = QLabel(SHARE_HINT)
        self._hint.setStyleSheet(
            "font-size: 11px; color: #9a9a9a; background: transparent;"
        )
        self._hint.setWordWrap(True)
        card_lay.addWidget(self._hint)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        foot.setContentsMargins(0, 0, 0, 0)

        self._star_btn = QPushButton(STAR_LABEL)
        self._star_btn.setObjectName("shareButton")
        self._star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _apply_icon(self._star_btn, "github.png")
        self._star_btn.clicked.connect(lambda: _open(GITHUB_URL))
        foot.addWidget(self._star_btn)

        dot = QLabel("\N{MIDDLE DOT}")
        dot.setStyleSheet("color: #9a9a9a; background: transparent;")
        foot.addWidget(dot)

        self._patreon_btn = QPushButton(PATREON_LABEL)
        self._patreon_btn.setObjectName("shareButton")
        self._patreon_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _apply_icon(self._patreon_btn, "patreon.png")
        self._patreon_btn.clicked.connect(lambda: _open(PATREON_URL))
        foot.addWidget(self._patreon_btn)

        foot.addStretch(1)

        self._dismiss_btn = QPushButton(DISMISS_LABEL)
        self._dismiss_btn.setObjectName("quietButton")
        self._dismiss_btn.clicked.connect(self._dismiss)
        foot.addWidget(self._dismiss_btn)
        card_lay.addLayout(foot)

        finish_centered_outer(outer, card)
        self._card = card

        self.show()
        self.raise_()

    def eventFilter(self, obj, event) -> bool:
        sync_overlay_to_parent(self, obj, event)
        return super().eventFilter(obj, event)

    def _share(self, target: ShareTarget) -> None:
        copied = _copy(SHARE_LINK)
        self._hint.setText(COPIED_HINT if copied else COPY_FAILED_HINT)
        _open(target.url)
        logger.info("support prompt: opened the %s share link", target.label)

    def _dismiss(self) -> None:
        """Idempotent, so the caller counts exactly one dismissal per prompt."""
        if self._dismissed:
            return
        self._dismissed = True
        if self._on_dismiss:
            try:
                self._on_dismiss()
            except Exception:
                logger.exception("support prompt dismissal callback failed")
        self._close()

    def _close(self) -> None:
        self.hide()
        self.deleteLater()

    def resizeEvent(self, event) -> None:
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)


def show_support(
    parent: QWidget,
    on_dismiss: Callable[[], None] | None = None,
) -> SupportOverlay:
    """Show the support prompt inside *parent*. Returns the overlay widget."""
    return SupportOverlay(parent, on_dismiss=on_dismiss)
