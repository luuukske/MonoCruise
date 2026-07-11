"""
MonoCruise – Animated top banner widget.

Displays status text with animated trailing dots.  Background colour
transitions between WAITING → CONNECTED → LOST using QVariantAnimation
with an InOutCubic easing curve, exactly replicating the original
``start_color_transition`` / ``_perform_step`` code in MonoCruise.py.
"""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import QEasingCurve, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

from core.thread_management.registry import registry
from core.settings import Settings

from ui.main_window.constants import (
    CONNECTED_COLOR,
    FONT_FAMILY,
    LOST_COLOR,
    RADIUS_BANNER,
    TEXT_COLOR,
    WAITING_COLOR,
)


class BannerState(Enum):
    WAITING   = auto()
    CONNECTED = auto()
    LOST      = auto()



class BannerWidget(QFrame):
    """Colour‑animated status banner drawn at the top of the main window."""

    _game: int = 1               # 1 = ETS2, 2 = ATS

    _STATE_COLORS: dict[BannerState, str] = {
        BannerState.WAITING:   WAITING_COLOR,
        BannerState.CONNECTED: CONNECTED_COLOR,
        BannerState.LOST:      LOST_COLOR,
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(32)

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        self._label = QLabel(self._state_text(BannerState.WAITING))
        self._label.setStyleSheet(
            f"color: {TEXT_COLOR}; font-family: '{FONT_FAMILY}'; "
            f"font-weight: bold; font-size: 13px; background: transparent;"
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignCenter)

        # State
        self._state = BannerState.WAITING
        self._current_color = QColor(WAITING_COLOR)
        self._apply_bg(self._current_color)

        # Dot animation (≈ LoadingDots from the original)
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.setInterval(500)
        self._dot_timer.timeout.connect(self._tick_dots)
        self._dot_timer.start()

        # State polling timer
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(500)
        self._state_timer.timeout.connect(self._update_state_from_telemetry)
        self._state_timer.start()

        # Colour transition animation
        self._color_anim = QVariantAnimation(self)
        self._color_anim.setDuration(1000)   # TRANSITION_DURATION = 1 s
        self._color_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._color_anim.valueChanged.connect(self._on_color_tick)

    # Public API

    def set_state(self, state: BannerState) -> None:
        if state == self._state:
            return
        self._state = state
        # Reset text immediately on state change; only WAITING gets animated dots.
        if self._state != BannerState.WAITING:
            self._dot_count = 0
            self._label.setText(self._state_text(self._state))
        target = QColor(self._STATE_COLORS[state])
        self._color_anim.stop()
        self._color_anim.setStartValue(QColor(self._current_color))
        self._color_anim.setEndValue(target)
        self._color_anim.start()

    def set_text(self, text: str) -> None:
        """Override the default state‑based text (e.g. 'finish setup in settings')."""
        self._label.setText(text)

    @property
    def state(self) -> BannerState:
        return self._state

    # Internal

    def _update_state_from_telemetry(self) -> None:
        """
        Auto-update banner state from TelemetryThread and MainPedalThread.

        Logic:

            if ets2_detected.is_set() and device_lost == False:
                current_mode = "running"  -> CONNECTED
            elif device_lost == True:
                current_mode = "lost"     -> LOST
            else:
                current_mode = "waiting"  -> WAITING
        """
        device_lost = self._pedals_lost()

        try:
            telemetry_thread = registry.get_thread("telemetry_thread")
            ets2_running = getattr(telemetry_thread, "sdk_initialized", False)
            old_game = self._game
            self._game = getattr(Settings, "last_game", 1) # 1 = ETS2, 2 = ATS
            if self._game != old_game:
                self._refresh_label_text()
        except KeyError:
            ets2_running = False

        if ets2_running and not device_lost:
            self.set_state(BannerState.CONNECTED)
        elif device_lost:
            self.set_state(BannerState.LOST)
        else:
            self.set_state(BannerState.WAITING)

    def _pedals_lost(self) -> bool:
        """
        True when configured pedals are currently disconnected.

        device_lost defaults to True in MainPedalThreadData and never clears
        when no pedal device is configured, so an unconfigured install must
        read as "not lost" (banner stays WAITING/CONNECTED).
        """
        if not getattr(Settings, "device", None):
            return False
        try:
            pedal_thread = registry.get_thread("main_pedal_thread")
            with pedal_thread.data._lock:
                return bool(pedal_thread.data.device_lost)
        except (KeyError, AttributeError):
            return False

    def _tick_dots(self) -> None:
        if self._state != BannerState.WAITING:
            self._label.setText(self._state_text(self._state))
            return
        self._dot_count = (self._dot_count % 3) + 1
        base = self._state_text(self._state)
        self._label.setText(base + "." * self._dot_count)

    def _game_name(self) -> str:
        return "ATS" if self._game == 2 else "ETS2"

    def _state_text(self, state: BannerState) -> str:
        if state == BannerState.WAITING:
            return f"Waiting for {self._game_name()}"
        if state == BannerState.CONNECTED:
            return f"Connected to {self._game_name()}"
        return "Connection lost to pedals"

    def _refresh_label_text(self) -> None:
        if self._state == BannerState.WAITING and self._dot_count > 0:
            self._label.setText(self._state_text(self._state) + "." * self._dot_count)
        else:
            self._label.setText(self._state_text(self._state))

    def _on_color_tick(self, color: QColor) -> None:
        self._current_color = color
        self._apply_bg(color)

    def _apply_bg(self, color: QColor) -> None:
        self.setStyleSheet(
            f"BannerWidget {{ background-color: {color.name()}; border: none; border-radius: {RADIUS_BANNER}px; }}"
        )