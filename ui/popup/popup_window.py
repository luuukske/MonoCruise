"""Main popup window with queue-based message handling."""
import logging
import os
from typing import ClassVar, Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QGraphicsOpacityEffect, QApplication
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QElapsedTimer, QRectF
from PySide6.QtGui import QFont, QPainter, QColor, QBrush, QPen, QPainterPath, QCursor
from PySide6.QtSvgWidgets import QSvgWidget

from ui.popup.popup_animator import PopupAnimator
from ui.popup.message_queue import MessageQueue
from ui.popup.message_types import PopupMessage, MessageStyle, StyleConfig, STYLE_CONFIGS, MESSAGE_TYPE_MAP


class State(Enum):
    IDLE = auto()
    ANIMATING_IN = auto()
    DISPLAYING = auto()
    ANIMATING_OUT = auto()


class PopupContainer(QWidget):
    """Custom widget that paints its own background - no palette interference."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        self._bg_color = QColor(50, 50, 50, 220)
        self._border_color = QColor(255, 255, 255)
        self._border_width = 2
        self._border_radius = 10
    
    def set_border_color(self, color: str):
        self._border_color = QColor(color)
        self.update()
    
    def set_border_width(self, width: int):
        self._border_width = width
        self.update()
    
    def set_border_radius(self, radius: int):
        self._border_radius = radius
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        path = QPainterPath()
        rect = QRectF(self.rect().adjusted(
            self._border_width // 2,
            self._border_width // 2,
            -self._border_width // 2,
            -self._border_width // 2
        ))
        path.addRoundedRect(rect, self._border_radius, self._border_radius)
        
        painter.fillPath(path, QBrush(self._bg_color))
        
        pen = QPen(self._border_color)
        pen.setWidth(self._border_width)
        painter.setPen(pen)
        painter.drawPath(path)


class PopupWindow(QWidget):
    """Priority popup queue with slide/scale animations; thread-safe via PopupWindow.emit()."""

    _instance: ClassVar[Optional["PopupWindow"]] = None
    _new_message_signal = Signal(object)
    
    # Design constants in pixels (reference: 4K display at 175% DPI scaling)
    # Reference logical screen height: 2160 / 1.75 ≈ 1234
    _REF_SCREEN_HEIGHT = 1234
    
    # All sizes are defined at 1:1 for the reference screen.
    # The ScalableContainer graphics transform scales the whole popup uniformly.
    _DESIGN_PANEL_WIDTH = 450
    _DESIGN_PANEL_HEIGHT = 100
    _DESIGN_MARGIN = 8
    _DESIGN_PADDING = 12
    _DESIGN_GAP = 22
    _DESIGN_FONT_SIZE = 13
    _DESIGN_TIMER_FONT_SIZE = 10
    _DESIGN_TIMER_W = 40
    _DESIGN_TIMER_H = 20
    _DESIGN_BORDER_WIDTH = 2
    _DESIGN_BORDER_RADIUS = 10
    
    BACKGROUND_COLOR = "rgba(50, 50, 50, 220)"
    PRIORITY_CHECK_MS = 100
    
    def __init__(self):
        super().__init__()
        PopupWindow._instance = self

        self._screen = QApplication.primaryScreen().availableGeometry()
        screen_full = QApplication.primaryScreen().geometry()
        
        # Calculate initial scale from screen resolution
        self._scale = screen_full.height() / self._REF_SCREEN_HEIGHT
        
        self._state = State.IDLE
        self._current: Optional[PopupMessage] = None
        self._queue = MessageQueue()
        
        self._display_timer: Optional[QTimer] = None
        self._check_timer: Optional[QTimer] = None
        self._update_timer_label_timer: Optional[QTimer] = None
        self._elapsed: Optional[QElapsedTimer] = None
        self._is_hovering = False
        self._last_displayed_seconds = -1
        self._is_finishing = False  # Track if message is finishing (not pushed back)
        
        self._setup_window()
        self._setup_content()
        self._setup_animator()
        
        # Apply initial resolution scale via the animator's transform
        self._animator.set_base_scale(self._scale)
        
        self._new_message_signal.connect(self._on_new_message)
    
    def _scaled_window_width(self) -> int:
        return int(self._DESIGN_PANEL_WIDTH * self._scale)
    
    def _scaled_window_height(self) -> int:
        return int(self._DESIGN_PANEL_HEIGHT * self._scale)
    
    def _scaled_margin(self) -> int:
        return int(self._DESIGN_MARGIN * self._scale)
    
    def set_scale(self, scale: float):
        """Override auto DPI scale; scales via graphics transform (design-pixel layout)."""
        self._scale = scale
        
        sw = self._scaled_window_width()
        sh = self._scaled_window_height()
        margin = self._scaled_margin()
        
        # Resize the outer window to the scaled dimensions
        self.setFixedSize(sw, sh)
        
        # Reposition on screen
        x = (self._screen.width() - sw) // 2
        self.move(x, self.pos().y())
        
        # Let the animator's scalable container handle the visual transform
        self._animator.set_base_scale(scale)
        
        # Update target / hidden Y for animations
        target_y = self._screen.height() - sh - margin
        hidden_y = self._screen.height()
        self._animator.update_geometry(target_y, hidden_y)
    
    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        sw = self._scaled_window_width()
        sh = self._scaled_window_height()
        self.setFixedSize(sw, sh)
        
        x = (self._screen.width() - sw) // 2
        y = self._screen.height()
        self.move(x, y)
        
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)
    
    def _setup_content(self):
        # Container is always built at design-pixel size.
        # The ScalableContainer graphics transform handles resolution scaling.
        pw = self._DESIGN_PANEL_WIDTH
        ph = self._DESIGN_PANEL_HEIGHT
        padding = self._DESIGN_PADDING
        gap = self._DESIGN_GAP
        bw = self._DESIGN_BORDER_WIDTH
        
        self._container = PopupContainer()
        self._container.setFixedSize(pw, ph)
        self._container.set_border_width(bw)
        self._container.set_border_radius(self._DESIGN_BORDER_RADIUS)
        
        # Timer label in top right (added first for lower z-order)
        self._timer_label = QLabel(self._container)
        self._timer_label.setFont(QFont("Arial", self._DESIGN_TIMER_FONT_SIZE))
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self._timer_label.setStyleSheet("""
            QLabel {
                color: rgba(100, 100, 100, 220);
                background-color: transparent;
                border: none;
            }
        """)
        self._timer_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._timer_label.lower()  # Ensure it's on bottom-most z-layer above background
        self._timer_label.hide()  # Initially hidden
        
        h_layout = QHBoxLayout(self._container)
        h_layout.setContentsMargins(padding, padding, padding, padding)
        h_layout.setSpacing(gap)
        
        self._icon = QSvgWidget()
        self._icon.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        icon_size = ph - (2 * padding) - (2 * bw)
        self._icon.setFixedSize(QSize(icon_size, icon_size))
        h_layout.addWidget(self._icon, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Text area widget with manual positioning for title + message
        self._text_area = QWidget()
        self._text_area.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        h_layout.addWidget(self._text_area, stretch=1)
        
        # Title label
        self._title_label = QLabel(self._text_area)
        title_font = QFont("Arial", self._DESIGN_FONT_SIZE)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._title_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        # Message label
        self._label = QLabel(self._text_area)
        self._label.setFont(QFont("Arial", self._DESIGN_FONT_SIZE))
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        self._apply_style(STYLE_CONFIGS[MessageStyle.NOTICE])
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_text_labels()
        self._position_timer_label()
    
    def _position_text_labels(self):
        """Position title and message evenly spaced over the available vertical space."""
        if not hasattr(self, '_text_area'):
            return
        
        area_w = self._text_area.width()
        area_h = self._text_area.height()
        
        title_h = self._title_label.sizeHint().height()
        msg_h = self._label.sizeHint().height()
        
        # Split remaining height into three vertical padding bands (title / message / bottom).
        total_content_h = title_h + msg_h
        available_padding = area_h - total_content_h
        section_padding = available_padding // 3
        
        # Position title in first section
        title_y = section_padding
        self._title_label.setGeometry(0, title_y, area_w, title_h)
        
        # Position message in second section
        msg_y = title_y + title_h + section_padding
        self._label.setGeometry(0, msg_y, area_w, msg_h)
    
    def _position_timer_label(self):
        """Position timer label in top right corner."""
        if not hasattr(self, '_timer_label'):
            return
        
        padding = self._DESIGN_PADDING
        tw = self._DESIGN_TIMER_W
        th = self._DESIGN_TIMER_H
        
        x = self._container.width() - tw - padding
        y = padding
        
        self._timer_label.setGeometry(x, y, tw, th)
    
    def _setup_animator(self):
        sh = self._scaled_window_height()
        margin = self._scaled_margin()
        target_y = self._screen.height() - sh - margin
        hidden_y = self._screen.height()
        
        self._animator = PopupAnimator(
            widget=self,
            content_widget=self._container,
            opacity_effect=self._opacity,
            target_y=target_y,
            hidden_y=hidden_y
        )
        self._animator.animation_finished.connect(self._on_animation_done)
    
    def _apply_style(self, config: StyleConfig):
        self._container.set_border_color(config.border_color)
        label_style = f"""
            QLabel {{
                color: {config.text_color};
                background-color: transparent;
                border: none;
            }}
        """
        self._title_label.setStyleSheet(label_style)
        self._label.setStyleSheet(label_style)
        self._load_icon(config.icon_name, config.icon_color)
    
    def _load_icon(self, name: str, color: str):
        path = os.path.join(os.path.dirname(__file__), "icons", name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                svg = f.read().replace("currentColor", color)
            self._icon.load(svg.encode("utf-8"))
        except FileNotFoundError:
            svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="{color}"/>
                <text x="50" y="65" font-size="50" text-anchor="middle" fill="white">!</text>
            </svg>'''
            self._icon.load(svg.encode("utf-8"))
    
    def enterEvent(self, event):
        """Handle mouse entering the widget."""
        self._is_hovering = True
        self._pause_display_timers()
        # Show cross immediately when hovering, regardless of state
        if self._state == State.DISPLAYING:
            self._update_timer_label()
        elif self._state in (State.ANIMATING_IN, State.ANIMATING_OUT):
            self._timer_label.setText("✕")
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leaving the widget."""
        self._is_hovering = False
        self._last_displayed_seconds = -1  # Force update when leaving hover
        self._resume_display_timers()
        # Restore appropriate timer state
        if self._state == State.DISPLAYING:
            self._update_timer_label()
        elif self._state == State.ANIMATING_OUT:
            # Only show 0s if message is finishing, not if pushed back
            if self._is_finishing:
                self._timer_label.setText("0s")
            # Otherwise keep the current timer value
        elif self._state == State.ANIMATING_IN and self._current:
            total_seconds = int(self._current.remaining_ms / 1000)
            self._timer_label.setText(f"{total_seconds}s")
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Dismiss popup on click."""
        if event.button() == Qt.MouseButton.LeftButton and self._state == State.DISPLAYING:
            self._finish_current()
        super().mousePressEvent(event)
    
    def emit_message(
        self,
        title: str,
        message: str,
        message_type: str,
        duration_ms: int = 5000,
        priority: int = 0
    ):
        """Enqueue a popup (thread-safe on GUI thread via signal). Types: e/w/c/n."""
        style = MESSAGE_TYPE_MAP.get(message_type.lower(), MessageStyle.NOTICE)
        popup_message = PopupMessage(
            priority=priority,
            title=title,
            text=message,
            style=style,
            duration_ms=duration_ms
        )
        self._new_message_signal.emit(popup_message)

    @classmethod
    def emit(
        cls,
        title: str,
        message: str,
        message_type: str,
        duration_ms: int = 5000,
        priority: int = 0,
    ) -> None:
        """Post from any thread; no-op with warning if singleton not created yet."""
        if cls._instance is None:
            logger.warning("PopupWindow.emit called before window was created; dropping message: %s", title)
            return
        cls._instance.emit_message(title, message, message_type, duration_ms, priority)
        logger.info(f"PopupWindow.emit: {title}, {message}, {message_type}, {duration_ms}, {priority}")

    def _on_new_message(self, message: PopupMessage):
        """Handle new message arrival; identical messages are dropped."""

        if self._current is not None and self._current.dedup_key == message.dedup_key:
            logger.debug("Duplicate popup dropped (currently on screen): %s", message.title)
            return

        if not self._queue.push(message):
            logger.debug("Duplicate popup dropped (already queued): %s", message.title)
            return

        if self._state == State.IDLE:
            self._show_next(from_queue=False)
        elif self._state == State.DISPLAYING:
            if self._queue.has_higher_priority_than(self._current):
                self._push_back_current()
    
    def _show_next(self, from_queue: bool = False):
        """Show the next message from the queue."""
        message = self._queue.pop()
        if message is None:
            self._state = State.IDLE
            self._timer_label.hide()
            return

        self._current = message
        self._apply_style(message.get_style_config())
        self._title_label.setText(message.title)
        self._label.setText(message.text)
        self._position_text_labels()
        self._timer_label.show()
        
        # Show total duration during animation in
        total_seconds = int(self._current.remaining_ms / 1000)
        self._timer_label.setText(f"{total_seconds}s")
        self._last_displayed_seconds = total_seconds
        self._is_finishing = False  # Reset for new message
        
        self.show()
        
        self._state = State.ANIMATING_IN
        if from_queue:
            self._animator.scale_in()
        else:
            self._animator.slide_in()
    
    def _push_back_current(self):
        """Push current message back to queue for higher priority."""
        if self._current is None:
            return
        
        self._stop_timers()
        
        elapsed = self._elapsed.elapsed() if self._elapsed else 0
        remaining = self._current.remaining_ms - elapsed
        self._current.remaining_ms = max(remaining, self._current.duration_ms // 2)
        
        self._queue.push(self._current)
        
        self._current = None
        self._is_finishing = False  # Not finishing, being pushed back
        self._state = State.ANIMATING_OUT
        self._animator.scale_out()
    
    def _finish_current(self):
        """Current message display time complete."""
        self._stop_timers()
        
        # Show 0s during animation out for finished messages
        self._timer_label.setText("0s")
        self._last_displayed_seconds = 0
        self._is_finishing = True  # Message is finishing, not pushed back
        
        self._state = State.ANIMATING_OUT
        
        if self._queue.is_empty():
            self._animator.slide_out()
        else:
            self._animator.slide_out()
    
    def _start_timers(self):
        self._elapsed = QElapsedTimer()
        self._elapsed.start()
        
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.timeout.connect(self._finish_current)
        self._display_timer.start(self._current.remaining_ms)
        
        self._check_timer = QTimer(self)
        self._check_timer.timeout.connect(self._check_priority)
        self._check_timer.start(self.PRIORITY_CHECK_MS)
        
        self._update_timer_label_timer = QTimer(self)
        self._update_timer_label_timer.timeout.connect(self._update_timer_label)
        self._update_timer_label_timer.start(50)  # Check every 50ms
        self._last_displayed_seconds = -1  # Reset for new message
        self._update_timer_label()  # Initial update
        self._check_cursor_position()
    
    def _stop_timers(self):
        if self._display_timer:
            self._display_timer.stop()
            self._display_timer = None
        if self._check_timer:
            self._check_timer.stop()
            self._check_timer = None
        if self._update_timer_label_timer:
            self._update_timer_label_timer.stop()
            self._update_timer_label_timer = None
        self._elapsed = None
    
    def _pause_display_timers(self):
        """Pause display-related timers while hovering."""
        if (
            self._state == State.DISPLAYING
            and self._current
            and self._elapsed
        ):
            elapsed = self._elapsed.elapsed()
            remaining = max(0, self._current.remaining_ms - elapsed)
            self._current.remaining_ms = remaining
            self._stop_timers()
    
    def _resume_display_timers(self):
        """Resume display-related timers after hover."""
        if (
            self._state == State.DISPLAYING
            and self._current
            and not self._display_timer
            and self._current.remaining_ms > 0
        ):
            self._start_timers()
    
    def _update_timer_label(self):
        """Update the timer label to show remaining time or cross when hovering."""
        # Only update timer during DISPLAYING state
        if self._state != State.DISPLAYING:
            return
        
        if self._is_hovering:
            self._timer_label.setText("✕")
            self._last_displayed_seconds = -1  # Reset when hovering
        elif self._current and self._elapsed:
            elapsed = self._elapsed.elapsed()
            remaining_ms = max(0, self._current.remaining_ms - elapsed) + 500
            remaining_seconds = int(remaining_ms / 1000)
            
            # Only update if seconds value changed to prevent jumping
            if remaining_seconds != self._last_displayed_seconds:
                self._last_displayed_seconds = remaining_seconds
                self._timer_label.setText(f"{remaining_seconds}s")
        else:
            self._timer_label.setText("")
            self._last_displayed_seconds = -1
    
    def _check_priority(self):
        """Check if higher priority message arrived."""
        if self._state == State.DISPLAYING and self._current:
            if self._queue.has_higher_priority_than(self._current):
                self._push_back_current()
    
    def _check_cursor_position(self):
        """Check if cursor is over the widget and update hover state."""
        cursor_pos = QCursor.pos()
        widget_rect = self.geometry()
        
        # Check if cursor is within widget bounds
        is_under_cursor = widget_rect.contains(cursor_pos)
        
        if is_under_cursor and not self._is_hovering:
            self._is_hovering = True
            self._pause_display_timers()
            self._update_timer_label()
        elif not is_under_cursor and self._is_hovering:
            self._is_hovering = False
            self._last_displayed_seconds = -1
            self._resume_display_timers()
            self._update_timer_label()
    
    def _on_animation_done(self, anim_type: str):
        if anim_type in ("slide_in", "scale_in"):
            self._state = State.DISPLAYING
            self._start_timers()
            # After scale-in, defer cursor-over check until geometry settles (singleShot 0).
            QTimer.singleShot(0, self._check_cursor_position)
        
        elif anim_type == "scale_out":
            self.hide()
            self._timer_label.hide()
            self._animator.reset()
            self._current = None
            self._state = State.IDLE
            self._show_next(from_queue=False)
        
        elif anim_type == "slide_out":
            self.hide()
            self._timer_label.hide()
            self._animator.reset()
            self._current = None
            self._state = State.IDLE
            if not self._queue.is_empty():
                self._show_next(from_queue=True)
