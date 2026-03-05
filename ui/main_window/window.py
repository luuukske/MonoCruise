"""
MonoCruise – Main application window.

Assembles banner, settings panel, gear button, command bar, and version
label.  Integrates with ``core.settings.Settings`` for persistence and
reads thread state from the ``Registry`` via periodic QTimer polling.

Lives on the main thread alongside the QApplication event loop.  Created
by ``create_main_window()`` in ``ui/main_window/main.py``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QEvent,
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QGraphicsOpacityEffect,
)

from core.thread_management.registry import registry
from ui.main_window.banner import BannerState, BannerWidget
from ui.main_window.confirmation_overlay import show_confirmation
from ui.main_window.constants import (
    APP_NAME,
    SETTINGS_PANEL_WIDTH,
    STYLESHEET,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from ui.main_window.settings_panel import SettingsPanel

if TYPE_CHECKING:
    from core.settings import Settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_BANNER_STATE_NAMES: dict[BannerState, str] = {
    BannerState.WAITING: "waiting",
    BannerState.CONNECTED: "connected",
    BannerState.LOST: "lost",
}


class MonoCruiseWindow(QMainWindow):
    """Top‑level MonoCruise window (700×500, dark theme).

    Emits ``window_closed`` when the user closes the window so the root
    main.py can trigger a clean shutdown of all threads.
    """

    # Signal emitted when the window is closed by the user
    window_closed = Signal()

    def __init__(self, settings: "Settings", version: str = "v2.0.0") -> None:
        super().__init__()
        self._settings = settings
        self._version = version
        self._closing = False
        self._open_on_taskbar = False

        # -- Window properties ------------------------------------------------
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(STYLESHEET)

        icon_path = os.path.join(_PROJECT_ROOT, "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # -- Central widget ---------------------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(0)

        # -- Banner -----------------------------------------------------------
        self._banner = BannerWidget()
        root.addWidget(self._banner)

        # -- Body (settings panel + right area) --------------------------------
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 5, 0, 0)
        body_lay.setSpacing(0)

        self._settings_panel = SettingsPanel(
            self,
            self._settings,
            on_save=self._save,
            on_reset=self._reset_settings,
            show_confirm=self._show_confirmation,
        )
        body_lay.addWidget(self._settings_panel)

        # Right area (gear icon at top‑left, rest is empty space) --------------
        right = QWidget()
        right.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self._gear_btn = QPushButton()
        self._gear_btn.setObjectName("gearButton")
        self._gear_btn.setFixedSize(32, 32)
        gear_path = os.path.join(_PROJECT_ROOT, "ui/main_window/assets/gear.png")
        if os.path.exists(gear_path):
            self._gear_btn.setIcon(QIcon(QPixmap(gear_path)))
            self._gear_btn.setIconSize(self._gear_btn.size())
        else:
            self._gear_btn.setText("⚙")
        self._gear_btn.clicked.connect(lambda: self.toggle_settings())
        right_lay.addWidget(
            self._gear_btn,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        right_lay.addStretch()

        body_lay.addWidget(right)
        root.addWidget(body, 1)

        # -- Command bar (bottom strip) ----------------------------------------
        self._cmd_label = QLabel("Starting MonoCruise...")
        self._cmd_label.setObjectName("cmdLabel")
        self._cmd_label.setFixedHeight(22)
        root.addWidget(self._cmd_label)

        # -- Version label (bottom‑right, absolute position) -------------------
        self._version_label = QLabel(self._version, central)
        self._version_label.setObjectName("versionLabel")
        self._version_label.adjustSize()

        # -- Settings panel slide animation ------------------------------------
        self._panel_anim = QPropertyAnimation(
            self._settings_panel, b"maximumWidth"
        )
        self._panel_anim.setDuration(300)
        self._panel_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._panel_opacity_effect = QGraphicsOpacityEffect(self._settings_panel)
        self._settings_panel.setGraphicsEffect(self._panel_opacity_effect)
        self._panel_opacity_effect.setOpacity(1.0)
        self._panel_fade_anim = QPropertyAnimation(
            self._panel_opacity_effect, b"opacity"
        )
        self._panel_fade_anim.setDuration(300)
        self._panel_fade_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._panel_anim_group = QParallelAnimationGroup(self)
        self._panel_anim_group.addAnimation(self._panel_anim)
        self._panel_anim_group.addAnimation(self._panel_fade_anim)
        self._panel_open = True

        # -- Registry polling timer (reads thread state → updates UI) ----------
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_threads)
        self._poll_timer.start()

        # -- Apply loaded settings to widgets ----------------------------------
        self._settings_panel.apply_settings(self._settings)

    # ==================================================================
    # Properties for external reads
    # ==================================================================

    @property
    def banner_state_name(self) -> str:
        return _BANNER_STATE_NAMES.get(self._banner.state, "waiting")

    @property
    def is_settings_open(self) -> bool:
        return self._panel_open

    @property
    def is_open_on_taskbar(self) -> bool:
        return self._open_on_taskbar

    # ==================================================================
    # Settings panel slide
    # ==================================================================

    def toggle_settings(self, *, force_close: bool = False) -> None:
        target_open = not self._panel_open
        if force_close:
            target_open = False
        if target_open == self._panel_open:
            return

        self._panel_anim_group.stop()
        start = self._settings_panel.maximumWidth()
        end = SETTINGS_PANEL_WIDTH if target_open else 0
        self._panel_anim.setStartValue(start)
        self._panel_anim.setEndValue(end)
        opacity_start = self._panel_opacity_effect.opacity()
        opacity_end = 1.0 if target_open else 0.0
        self._panel_fade_anim.setStartValue(opacity_start)
        self._panel_fade_anim.setEndValue(opacity_end)
        self._panel_open = target_open
        self._panel_anim_group.start()

    # ==================================================================
    # Persistence
    # ==================================================================

    def _save(self) -> None:
        try:
            self._settings.save()
        except Exception:
            logger.exception("Failed to save settings")

    def _reset_settings(self) -> None:
        from core.settings import Settings

        fresh = Settings()
        for k in fresh.__dataclass_fields__:
            if not k.startswith("_"):
                setattr(self._settings, k, getattr(fresh, k))
        self._settings.save()
        self._settings_panel.apply_settings(self._settings)
        self.set_cmd("All settings reset to defaults.")

    # ==================================================================
    # Confirmation overlay
    # ==================================================================

    def _show_confirmation(self, title, message, on_confirm, on_cancel=None):
        show_confirmation(
            self.centralWidget(), title, message, on_confirm, on_cancel
        )

    # ==================================================================
    # Command bar
    # ==================================================================

    def set_cmd(self, text: str) -> None:
        self._cmd_label.setText(text)

    # ==================================================================
    # Banner convenience
    # ==================================================================

    def set_banner_state(self, state: BannerState) -> None:
        self._banner.set_state(state)

    # ==================================================================
    # Startup visibility
    # ==================================================================

    def apply_startup_visibility(self) -> None:
        """
        Show or hide the main window at startup:
        - hide when telemetry is connected and autostart is enabled
        - otherwise show normally
        """
        is_connected = False
        try:
            telemetry = registry.get_thread("telemetry_thread")
            is_connected = bool(getattr(telemetry.data, "is_connected", False))
        except (KeyError, AttributeError):
            is_connected = False

        if self._settings.autostart_variable and is_connected:
            self.hide()
            logger.info("Main window hidden on startup (autostart + game connected)")
        else:
            self.show()
            logger.info("Main window shown on startup")

    # ==================================================================
    # Thread‑state polling
    # ==================================================================

    def _poll_threads(self) -> None:
        """
        Called every 100 ms by QTimer.  Reads data from registered threads
        via the Registry and updates the UI accordingly.
        """
        # TODO: read from registry to update banner, pedal labels, etc.
        #
        # Example:
        #   from core.thread_management.registry import registry
        #
        #   telem = registry.get_thread("telemetry_thread")
        #   if telem and telem.data.game_connected:
        #       self.set_banner_state(BannerState.CONNECTED)
        #   elif telem and not telem.data.game_connected:
        #       self.set_banner_state(BannerState.LOST)
        pass

    # ==================================================================
    # Overrides
    # ==================================================================

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_version_label"):
            cw = self.centralWidget()
            self._version_label.adjustSize()
            self._version_label.move(
                cw.width() - self._version_label.width() - 8,
                cw.height() - self._version_label.height() - 4,
            )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._open_on_taskbar = True

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._open_on_taskbar = False

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._open_on_taskbar = self.isVisible() and not self.isMinimized()

    def closeEvent(self, event) -> None:
        """Handle window close — save settings, stop polling, signal shutdown."""
        if self._closing:
            event.accept()
            return
        self._closing = True
        self._open_on_taskbar = False

        self._poll_timer.stop()
        try:
            self._settings.save()
        except Exception:
            logger.exception("Failed to save settings on close")

        logger.info("Window closeEvent — emitting window_closed signal")
        self.window_closed.emit()
        event.accept()