import logging
import time

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QPainter, QColor

from core.lane_departure.display import ALERT, LEFT, RIGHT, STANDBY, WARN, LaneLines, panel_side, plugin_ok
from core.settings import Settings
from core.thread_management.registry import registry

from .lane_corners import LaneCorner

logger = logging.getLogger(__name__)

_TRANSPARENT = QColor(0, 0, 0, 0)
_BLUE = QColor("blue")
_RED = QColor("red")

# Delay before re-reading screen geometry after a display change event.
# Windows display drivers may report a transient incorrect resolution
# (e.g. FHD fallback) immediately after wake-from-sleep; waiting briefly
# lets the driver settle to the real geometry.
_SCREEN_SETTLE_MS = 500


class VisualizationBar(QWidget):
    def __init__(self):
        super().__init__()
        self.temp_gasval = 0.0
        self.temp_brakeval = 0.0
        self.bar_width = 3
        self.flicker_state = True
        self.gas_rect = QRect()
        self.brake_rect = QRect()
        self.flicker_color = _TRANSPARENT
        self._painted_key: tuple | None = None
        self._tracked_screen = None

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowOpacity(0.8)

        # Lane departure corners: own windows, shown, hidden and kept on top with the bar.
        self._lane = LaneLines()
        self._lane_side = 0
        self._lane_alert = False
        self.corners = (LaneCorner(LEFT, self), LaneCorner(RIGHT, self))
        for corner in self.corners:
            corner.setWindowOpacity(0.8)

        self._update_screen_geometry()
        self._connect_screen_signals()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(10)
        self.show()
        for corner in self.corners:
            corner.show()

    def _connect_screen_signals(self):
        """Connect to QApplication screen signals for event-driven geometry updates."""
        app = QApplication.instance()
        if app is None:
            return
        app.screenAdded.connect(self._schedule_geometry_update)
        app.screenRemoved.connect(self._schedule_geometry_update)
        app.primaryScreenChanged.connect(self._on_primary_screen_changed)
        self._track_primary_screen()

    def _track_primary_screen(self):
        """Connect geometryChanged on the current primary screen, disconnecting from any previous one."""
        screen = QApplication.primaryScreen()
        if self._tracked_screen is not None:
            try:
                self._tracked_screen.geometryChanged.disconnect(self._schedule_geometry_update)
            except RuntimeError:
                pass  # already disconnected or deleted
        self._tracked_screen = screen
        if screen is not None:
            screen.geometryChanged.connect(self._schedule_geometry_update)

    def _on_primary_screen_changed(self, _screen=None):
        self._track_primary_screen()
        self._schedule_geometry_update()

    def _schedule_geometry_update(self, *_args):
        """Schedule a geometry re-read after a short settling delay.

        The delay gives the Windows display driver time to report the real
        resolution instead of the transient FHD fallback that can appear
        immediately after wake-from-sleep.
        """
        QTimer.singleShot(_SCREEN_SETTLE_MS, self._update_screen_geometry)

    def _update_screen_geometry(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.geometry()
        self.screen_width = geo.width()
        self.screen_height = geo.height()
        self.center = self.screen_width // 2
        self.setGeometry(0, self.screen_height - self.bar_width, self.screen_width, self.bar_width)
        for corner in self.corners:
            corner.place(geo.x(), geo.y(), self.screen_width, self.screen_height)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), _TRANSPARENT)
        if self.flicker_color is not None:
            # Emergency mode
            painter.fillRect(self.gas_rect, self.flicker_color)
            painter.fillRect(self.brake_rect, self.flicker_color)
        else:
            if not self.gas_rect.isEmpty():
                painter.fillRect(self.gas_rect, _BLUE)
            if not self.brake_rect.isEmpty():
                painter.fillRect(self.brake_rect, _RED)
        painter.end()

    def is_settled_at_center(self) -> bool:
        """True when the live bar has decayed to the screen center (idle).

        Used by auto-close: wait for the rundown animation before quitting so
        the bar does not vanish mid-extension. Hidden / disabled bars count as
        settled immediately.
        """
        if not self.isVisible() or not self.timer.isActive():
            return True
        return max(self.temp_gasval, self.temp_brakeval) < 0.0005

    def _animate(self):
        try:
            self._animate_inner()
        except Exception as e:
            # Never let a runtime error in the bar crash the main program
            logger.warning("animate error: %s", e, exc_info=False)

    def _animate_inner(self):
        # Safe read of bar visibility flag from settings
        try:
            # config.json / Settings exposes this as `bar_variable`
            bar_active = bool(Settings.bar_variable)
        except Exception:
            bar_active = False

        # Read live values from worker threads
        gas_output = 0.0
        brake_output = 0.0
        em_stop = False
        AEB_cue = False

        # SendingThread: aforward / abackward are logical pedals (pre intensity remap).
        try:
            sending_thread = registry.get_thread("sending_thread")
        except KeyError:
            sending_thread = None

        if sending_thread is not None and hasattr(sending_thread, "data"):
            try:
                with sending_thread.data._lock:
                    gas_output = float(getattr(sending_thread.data, "aforward", 0.0))
                    brake_output = float(getattr(sending_thread.data, "abackward", 0.0))
            except Exception:
                gas_output = 0.0
                brake_output = 0.0

        # MainPedalThread: em_stop flag for emergency flashing
        try:
            pedal_thread = registry.get_thread("main_pedal_thread")
        except KeyError:
            pedal_thread = None

        if pedal_thread is not None and hasattr(pedal_thread, "data"):
            try:
                with pedal_thread.data._lock:
                    em_stop = bool(getattr(pedal_thread.data, "em_stop", False))
            except Exception:
                em_stop = False

        # AEB HMI cue (warn or brake, as the sound) for flicker
        try:
            aeb_thread = registry.get_thread("aeb_thread")
        except KeyError:
            aeb_thread = None

        if aeb_thread is not None and hasattr(aeb_thread, "data"):
            try:
                with aeb_thread.data._lock:
                    AEB_cue = bool(getattr(aeb_thread.data, "AEB_cue", False))
            except Exception:
                AEB_cue = False

        # LDW rumbling or an LKA pulse, only while the FFB plugin can act on it (as the CC panel's lines).
        lane_side, lane_alert = 0, False
        try:
            lane_thread = registry.get_thread("lane_departure_thread")
        except KeyError:
            lane_thread = None

        if lane_thread is not None and hasattr(lane_thread, "data") and lane_thread.is_alive():
            try:
                with lane_thread.data._lock:
                    firing = int(getattr(lane_thread.data, "firing_side", 0))
                    alert = bool(getattr(lane_thread.data, "firing_alert", False))
                    plugin = getattr(lane_thread.data, "plugin", None)
                lane_side = panel_side(firing) if plugin_ok(plugin) else 0
                lane_alert = alert and bool(lane_side)
            except Exception:
                lane_side, lane_alert = 0, False

        # ACC/cruise control could increase smoothness here; kept commented out.
        # average = 20 if (cc_enabled and cc_locked) else 10

        close_bar = not bar_active

        self._step_corners(lane_side, lane_alert, close_bar or em_stop or AEB_cue)

        if close_bar and max(self.temp_gasval, self.temp_brakeval) < 0.0005:
            self.timer.stop()
            self.close()
            for corner in self.corners:
                corner.close()
            return
        elif close_bar:
            gas_output = 0.0
            brake_output = 0.0

        average = 10  # fixed; ACC could use higher value for extra smoothing (see comment above)
        self.temp_gasval = (gas_output + self.temp_gasval * average) / (average + 1)
        self.temp_brakeval = (brake_output + self.temp_brakeval * average) / (average + 1)

        value = self.temp_gasval - self.temp_brakeval

        if em_stop or AEB_cue:
            self.flicker_state = not self.flicker_state
            self.gas_rect = QRect(self.center, 0, self.screen_width - self.center, self.bar_width)
            self.brake_rect = QRect(0, 0, self.center, self.bar_width)
            self.flicker_color = _RED if self.flicker_state else _TRANSPARENT
            self.temp_gasval = 0.0
            self.temp_brakeval = 1.0
            self.timer.setInterval(125)
        else:
            self.flicker_state = True
            self.flicker_color = None

            gas_ext = int((self.screen_width - self.center) * min(max(value, 0), 1))
            self.gas_rect = QRect(self.center, 0, gas_ext, self.bar_width) if gas_ext > 0 else QRect()

            brake_ext = int(self.center * min(max(value, -1), 0) * -1)
            self.brake_rect = QRect(self.center - brake_ext, 0, brake_ext, self.bar_width) if brake_ext > 0 else QRect()

            self.timer.setInterval(10)

        # Idle ticks leave the rects as they were; repainting the full-width window is wasted.
        key = (
            self.gas_rect.getRect(),
            self.brake_rect.getRect(),
            self.flicker_color.rgba() if self.flicker_color is not None else None,
        )
        if key != self._painted_key:
            self._painted_key = key
            self.update()

    def _step_corners(self, lane_side: int, alert: bool, silenced: bool) -> None:
        """A warning or an alert shows at once and is held 1 s after it stops, the alert blinking (display.LaneLines).

        AEB, an emergency stop or a closing bar clear it.
        """
        now = time.monotonic()
        if (lane_side, alert) != (self._lane_side, self._lane_alert):
            self._lane_side, self._lane_alert = lane_side, alert
            self._lane.set(STANDBY, lane_side, now, alert)
        self._lane.step(now)
        for corner in self.corners:
            shown = self._lane.shown(corner.side, now)
            if silenced:
                corner.set_weights(0.0, 0.0)
            else:
                corner.set_weights(shown[WARN], shown[ALERT])

    def ensure_present(self) -> None:
        """GUI thread. Re-open an enabled bar whose timer was stopped."""
        try:
            bar_active = bool(Settings.bar_variable)
        except Exception:
            return
        if not bar_active:
            return
        if not self.timer.isActive():
            self.timer.start(10)
        if not self.isVisible():
            self.show()
        for corner in self.corners:
            if not corner.isVisible():
                corner.show()

