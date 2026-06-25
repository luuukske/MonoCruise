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
import time
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
from ui.cc_panel.main import cc_panel as CcPanel
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

_CC_LEAD_SPEED_MIN_INTERVAL_S = 0.5

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
        # Startup visibility is decided lazily once telemetry has produced a
        # first result; until then the window stays hidden.
        self._startup_visibility_applied = False
        self._open_on_taskbar = False

        # Window properties
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(STYLESHEET)

        icon_path = os.path.join(_PROJECT_ROOT, "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(0)

        # Banner
        self._banner = BannerWidget()
        root.addWidget(self._banner)

        # Body (settings panel + right area)
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

        # Right area (gear icon at top‑left, rest is empty space)
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

        # Command bar (bottom strip)
        self._cmd_label = QLabel("Starting MonoCruise...")
        self._cmd_label.setObjectName("cmdLabel")
        self._cmd_label.setFixedHeight(22)
        root.addWidget(self._cmd_label)

        # Version label (bottom‑right, absolute position)
        self._version_label = QLabel(self._version, central)
        self._version_label.setObjectName("versionLabel")
        self._version_label.adjustSize()

        # Settings panel slide animation
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

        # Registry polling timer (reads thread state → updates UI)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_threads)
        self._poll_timer.start()

        # Cruise control floater (Qt main thread only; see CcPanel docstring)
        self._cc_panel: CcPanel | None = None
        self._cc_panel_scale_snap: float | None = None
        self._cc_panel_update_snap: tuple | None = None
        # Lead speed throttle: only push a new int km/h every _CC_LEAD_SPEED_MIN_INTERVAL_S
        # so rapid telemetry jitter doesn't flicker the displayed value faster
        # than a human can read it. None pass-through is immediate (retract).
        self._cc_lead_speed_emit_val: int | None = None
        self._cc_lead_speed_emit_ts: float = 0.0
        self._init_cc_panel()

        # Apply loaded settings to widgets
        self._settings_panel.apply_settings(self._settings)

    # Properties for external reads

    @property
    def banner_state_name(self) -> str:
        return _BANNER_STATE_NAMES.get(self._banner.state, "waiting")

    @property
    def is_settings_open(self) -> bool:
        return self._panel_open

    @property
    def is_open_on_taskbar(self) -> bool:
        return self._open_on_taskbar

    # Settings panel slide

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

    # Persistence

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

    # Confirmation overlay

    def _show_confirmation(self, title, message, on_confirm, on_cancel=None):
        show_confirmation(
            self.centralWidget(), title, message, on_confirm, on_cancel
        )

    # Command bar

    def set_cmd(self, text: str) -> None:
        self._cmd_label.setText(text)

    # Banner convenience

    def set_banner_state(self, state: BannerState) -> None:
        self._banner.set_state(state)

    # Startup visibility

    def apply_startup_visibility(self) -> None:
        """
        Initial startup behaviour.

        The window is created and shown in the taskbar (minimised) so it is
        fully loaded and ready, but does not steal focus. The actual decision
        about whether to pop it up on screen is deferred until the telemetry
        thread has produced a first result. That decision is handled in
        ``_poll_threads()``.
        """
        self.showMinimized()
        self._startup_visibility_applied = False
        logger.info(
            "Main window minimised on startup (waiting for telemetry result)"
        )

    # Thread‑state polling

    def _poll_threads(self) -> None:
        """
        Called every 100 ms by QTimer.  Reads data from registered threads
        via the Registry and updates the UI accordingly.
        """
        try:
            self._sync_cc_panel()
        except Exception:
            logger.exception("main window poll: CC panel sync failed")

    @staticmethod
    def _cc_scale_mult(raw: object) -> float:
        """Map settings value (e.g. 1.5 or '150%') to cc_panel scale multiplier."""
        if raw is None:
            return 1.0
        if isinstance(raw, (int, float)):
            x = float(raw)
            return max(0.25, min(4.0, x / 100.0 if x > 4.0 else x))
        s = str(raw).strip()
        if s.endswith("%"):
            try:
                return max(0.25, min(4.0, float(s[:-1].strip()) / 100.0))
            except ValueError:
                return 1.0
        try:
            x = float(s)
            return max(0.25, min(4.0, x / 100.0 if x > 4.0 else x))
        except ValueError:
            return 1.0

    @staticmethod
    def _cc_panel_display_mode(raw: str | None) -> str:
        if raw == "Speed limiter":
            return "Speed limiter"
        return "Cruise control"

    # Truck detection for the CC-panel icon swap. The ACC tracker keeps
    # `LeadInfo.vehicle` pointing at the trailer record even after the
    # tractor-kinematics swap, so `is_trailer` alone identifies an
    # articulated rig. trailer_count and length cover unattached tractors
    # and long rigid trucks (no trailer, but clearly not a car).
    _TRUCK_LENGTH_M: float = 6.0

    @classmethod
    def _classify_lead_as_truck(cls, vehicle) -> bool:
        if getattr(vehicle, "is_trailer", False):
            return True
        if int(getattr(vehicle, "trailer_count", 0) or 0) >= 1:
            return True
        size = getattr(vehicle, "size", None)
        length = float(getattr(size, "length", 0.0) or 0.0) if size is not None else 0.0
        return length >= cls._TRUCK_LENGTH_M

    def _init_cc_panel(self) -> None:
        try:
            scale = self._cc_scale_mult(self._settings.cc_panel_scaling)
            self._cc_panel_scale_snap = scale
            px = int(self._settings.panel_x) if self._settings.panel_x is not None else 100
            py = int(self._settings.panel_y) if self._settings.panel_y is not None else 100
            mode = self._cc_panel_display_mode(self._settings.cc_mode)
            self._cc_panel = CcPanel(
                "-- km/h",
                cc_mode=mode,
                cc_enabled=False,
                x_co=px,
                y_co=py,
                acc_enabled=bool(self._settings.acc_enabled),
                scale_mult=scale,
            )
            # Saved panel_x/y may be from another resolution: pull onto a visible desktop.
            self._cc_panel.ensure_on_screen()
        except Exception:
            logger.exception("failed to create cruise control panel")
            self._cc_panel = None

    def _sync_cc_panel(self) -> None:
        """Drive CcPanel from registry + Settings (AEB blink, speed limiter colours, etc.)."""
        if self._cc_panel is None:
            return

        cruise_enabled = False
        target_kmh: float | None = None
        try:
            cr = registry.get_thread("cruise_control_thread")
            if cr is not None and cr.is_alive():
                with cr.data._lock:
                    cruise_enabled = bool(cr.data.cc_enabled)
                    t = cr.data.target_speed_kmh
                    target_kmh = float(t) if t is not None else None
        except (KeyError, AttributeError):
            pass

        aeb_warn = False
        try:
            aeb = registry.get_thread("aeb_thread")
            if aeb is not None and aeb.is_alive():
                with aeb.data._lock:
                    aeb_warn = bool(aeb.data.AEB_warn)
        except (KeyError, AttributeError):
            pass

        acc_locked = False
        acc_truck = False
        lead_speed_kmh: int | None = None
        try:
            acc = registry.get_thread("acc_thread")
            if acc is not None and acc.is_alive():
                with acc.data._lock:
                    has_lead = bool(acc.data.has_lead)
                    primary = acc.data.leads[0] if (has_lead and acc.data.leads) else None
                if primary is not None:
                    acc_locked = True
                    acc_truck = self._classify_lead_as_truck(primary.vehicle)
                    # Quantize the lead's smoothed absolute speed to int km/h so
                    # the snap below only fires panel updates when the displayed
                    # value would actually change (panel renders to int anyway).
                    lead_speed_kmh = int(round(primary.effective_speed_ms * 3.6))
        except (KeyError, AttributeError):
            pass

        # Throttle lead-speed pass-through so rapid changes don't flicker the
        # text faster than a human can read. Pass-through is immediate when
        # the lead first appears (last emit was None) or when retracting (new
        # value is None); only value-to-value changes are rate-limited.
        now_mono = time.monotonic()
        last_val = self._cc_lead_speed_emit_val
        if lead_speed_kmh is None or last_val is None:
            emit_now = True
        else:
            emit_now = (now_mono - self._cc_lead_speed_emit_ts) >= _CC_LEAD_SPEED_MIN_INTERVAL_S
        if emit_now:
            self._cc_lead_speed_emit_val = lead_speed_kmh
            self._cc_lead_speed_emit_ts = now_mono
        lead_speed_kmh = self._cc_lead_speed_emit_val

        s = self._settings
        with s._state_lock:
            show_ui = bool(s.show_cc_ui)
            cc_mode_raw = s.cc_mode
            scaling_raw = s.cc_panel_scaling
            acc_on = bool(s.acc_enabled)
            try:
                gap_level = max(1, min(4, int(s.acc_gap_level)))
            except (TypeError, ValueError):
                gap_level = 2

        # Show whenever "Show CC UI" is on (including game pause / menu).
        should_show = show_ui

        new_scale = self._cc_scale_mult(scaling_raw)
        if self._cc_panel_scale_snap != new_scale:
            self._cc_panel_scale_snap = new_scale
            self._cc_panel.update_scaling(new_scale)

        display_mode = self._cc_panel_display_mode(cc_mode_raw)
        if target_kmh is None:
            text = "-- km/h"
        else:
            text = f"{int(round(target_kmh))} km/h"

        update_snap = (
            text, display_mode, cruise_enabled, aeb_warn, acc_on,
            acc_locked, acc_truck, gap_level, lead_speed_kmh,
        )
        if self._cc_panel_update_snap != update_snap:
            self._cc_panel_update_snap = update_snap
            self._cc_panel.update(
                new_text=text,
                cc_mode=display_mode,
                cc_enabled=cruise_enabled,
                AEB_warn=aeb_warn,
                acc_enabled=acc_on,
                acc_locked=acc_locked,
                distance_to_lead=gap_level,
                acc_truck=acc_truck,
                lead_vehicle_speed=lead_speed_kmh,
            )

        if should_show:
            self._cc_panel.ensure_on_screen()
            self._cc_panel.show()
        else:
            self._cc_panel.hide()

    # Overrides

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
        """Handle window close: save settings, stop polling, signal shutdown."""
        if self._closing:
            event.accept()
            return
        self._closing = True
        self._open_on_taskbar = False

        self._poll_timer.stop()
        if self._cc_panel is not None:
            try:
                self._cc_panel.stop()
            except Exception:
                logger.exception("failed to stop cruise control panel")
            self._cc_panel = None
        try:
            self._settings.save()
        except Exception:
            logger.exception("Failed to save settings on close")

        logger.info("Window closeEvent: emitting window_closed signal")
        self.window_closed.emit()
        event.accept()
