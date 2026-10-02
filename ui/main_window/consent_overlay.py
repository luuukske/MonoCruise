"""In-window consent overlay: a markdown document behind a scroll-to-the-end gate.

Non-blocking like ``confirmation_overlay``: signals and slots, never ``QDialog.exec()``.
Signature:: show_consent(parent, markdown_path, on_accept, on_decline=None)
"""
from __future__ import annotations

import logging
import os
from typing import Callable

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# Lives in core so nothing under core/ has to import this UI module. Re-exported
# here because the settings panel reads it alongside the overlay.
from core.settings import CONSENT_VERSION  # noqa: F401
from ui.main_window.constants import SETTINGS_COLOR
from ui.main_window.overlay_chrome import (
    attach_overlay,
    begin_centered_outer,
    finish_centered_outer,
    OverlayCard,
    sync_overlay_to_parent,
)
from ui.main_window.widgets import CheckBox

logger = logging.getLogger(__name__)

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CONSENT_MARKDOWN = os.path.join(_ASSET_DIR, "clip_contribution.md")

_CARD_WIDTH = 620
_BODY_HEIGHT = 380
# Must match QFrame#dialogCard in constants.STYLESHEET.
_DIALOG_CARD_BG = "#252525"
# Qt leaves the scrollbar a pixel or two short of maximum at the bottom.
_SCROLL_EPSILON_PX = 4

# Transparent QTextBrowser bg; scrollbar track matches the card so the groove
# does not read as a separate box (same trick settings uses with #333333).
_BODY_STYLE = (
    "QTextBrowser{background:transparent; border:none;}"
    f"QScrollBar:vertical{{background-color:{_DIALOG_CARD_BG}; width:8px; margin:0;}}"
    f"QScrollBar::handle:vertical{{background-color:{SETTINGS_COLOR};"
    " min-height:30px; border-radius:4px;}}"
    "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical{height:0;}"
    "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical{background:none;}"
)


class _ConsentBody(QTextBrowser):
    """Prefers ``_BODY_HEIGHT`` but yields space so the card footer never overlaps."""

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.width(), _BODY_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(hint.width(), 0)


class ConsentOverlay(QWidget):
    """Full-window overlay whose accept button unlocks once the text has been read."""

    def __init__(
        self,
        parent: QWidget,
        *,
        markdown_path: str = CONSENT_MARKDOWN,
        on_accept: Callable[[bool], None],
        on_decline: Callable[[], None] | None = None,
        screenshot_default: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("overlayBg")
        self._on_accept = on_accept
        self._on_decline = on_decline
        self._unlocked = False

        attach_overlay(self, parent)

        # Stretches centre the card when tall; when short the body shrinks first
        # so the checkbox and buttons stay visible instead of being painted over.
        outer = begin_centered_outer(self)

        card = OverlayCard(_CARD_WIDTH)
        # Margins and spacing match confirmation_overlay so both prompts read
        # as the same component.
        card_lay = QVBoxLayout(card)
        card_lay.setSpacing(14)
        card_lay.setContentsMargins(24, 20, 24, 20)

        title, body_html = _render_markdown(markdown_path)
        title_lbl = QLabel(title or "Help improve AEB and ACC")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: bold; background: transparent;")
        title_lbl.setWordWrap(True)
        title_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        card_lay.addWidget(title_lbl)

        self._body = _ConsentBody()
        self._body.setOpenExternalLinks(True)
        self._body.setMaximumHeight(_BODY_HEIGHT)
        self._body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._body.setStyleSheet(_BODY_STYLE)
        self._body.setHtml(body_html)
        card_lay.addWidget(self._body, 1)

        # Same CheckBox glyph as the settings panel; label + box both right-aligned.
        shot_row = QHBoxLayout()
        shot_row.setSpacing(8)
        shot_row.setContentsMargins(0, 0, 0, 0)
        self._screenshot = CheckBox()
        self._screenshot.setFixedSize(24, 24)
        self._screenshot.setChecked(screenshot_default)
        shot_lbl = QLabel("Include the screenshot")
        shot_lbl.setStyleSheet("background: transparent;")
        shot_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        shot_row.addStretch(1)
        shot_row.addWidget(shot_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        shot_row.addWidget(self._screenshot, 0, Qt.AlignmentFlag.AlignVCenter)
        card_lay.addLayout(shot_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        # Inline with the buttons rather than its own row: it is one small line
        # and a full row of its own left a large hole once it clears.
        self._hint = QLabel("Scroll to the end to continue")
        self._hint.setStyleSheet("font-size: 11px; color: #9a9a9a; background: transparent;")
        btn_row.addWidget(self._hint)
        btn_row.addStretch(1)
        decline_btn = QPushButton("Not now")
        decline_btn.setObjectName("dangerButton")
        decline_btn.clicked.connect(self._decline)
        btn_row.addWidget(decline_btn)
        self._accept_btn = QPushButton("Turn it on")
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self._accept)
        btn_row.addWidget(self._accept_btn)
        card_lay.addLayout(btn_row)

        finish_centered_outer(outer, card)
        self._card = card

        self._body.verticalScrollBar().valueChanged.connect(self._check_read)
        self._body.verticalScrollBar().rangeChanged.connect(self._check_read)
        # Layout has not run yet, so the scroll range is not final until the
        # event loop turns once. Without this a short document never unlocks.
        QTimer.singleShot(0, self._check_read)

        self.show()
        self.raise_()

    @property
    def include_screenshot(self) -> bool:
        return bool(self._screenshot.isChecked())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._check_read()

    def eventFilter(self, obj, event) -> bool:
        sync_overlay_to_parent(self, obj, event)
        return super().eventFilter(obj, event)

    def _check_read(self, *_args) -> None:
        """Unlock at the bottom, or at once when the text does not scroll at all."""
        if self._unlocked:
            return
        # A widget that has never been laid out reports no scroll range, which
        # would unlock the gate before the text has been on screen at all.
        if not self._body.isVisible():
            return
        bar = self._body.verticalScrollBar()
        at_end = bar.maximum() <= 0 or bar.value() >= bar.maximum() - _SCROLL_EPSILON_PX
        if not at_end:
            return
        self._unlocked = True
        self._accept_btn.setEnabled(True)
        self._hint.setText("")

    def _accept(self) -> None:
        self._on_accept(self.include_screenshot)
        self._close()

    def _decline(self) -> None:
        if self._on_decline:
            self._on_decline()
        self._close()

    def _close(self) -> None:
        self.hide()
        self.deleteLater()

    def resizeEvent(self, event) -> None:
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)
        self._check_read()


def _split_title(text: str) -> tuple[str, str]:
    """Lift a leading '# ' heading out so it can sit above the scroll area."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return line[2:].strip(), "\n".join(lines[i + 1:])
        if line.strip():
            break
    return "", text


def _render_markdown(path: str) -> tuple[str, str]:
    """Consent document as (title, HTML), with fallbacks that never render blank."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        logger.exception("consent document could not be read")
        return "", "<p>The consent document could not be loaded, so this cannot be enabled.</p>"
    title, rest = _split_title(text)
    try:
        from shared.markdown_renderer import GitHubMarkdownRenderer
        from shared.theme import Theme

        html = GitHubMarkdownRenderer(Theme(), image_base=os.path.dirname(path)).render(rest)
    except Exception:
        logger.exception("consent document could not be rendered")
        html = f"<pre>{rest}</pre>"
    return title, html


def show_consent(
    parent: QWidget,
    on_accept: Callable[[bool], None],
    on_decline: Callable[[], None] | None = None,
    *,
    markdown_path: str = CONSENT_MARKDOWN,
    screenshot_default: bool = True,
) -> ConsentOverlay:
    """Show the consent overlay inside *parent*. Returns the overlay widget."""
    return ConsentOverlay(
        parent,
        markdown_path=markdown_path,
        on_accept=on_accept,
        on_decline=on_decline,
        screenshot_default=screenshot_default,
    )
