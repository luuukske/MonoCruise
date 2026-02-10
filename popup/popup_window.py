"""
Main popup window with clean queue-based message handling.
"""
import os
from typing import Optional
from enum import Enum, auto

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QGraphicsOpacityEffect, QApplication
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QElapsedTimer, QRectF
from PySide6.QtGui import QFont, QPainter, QColor, QBrush, QPen, QPainterPath, QCursor
from PySide6.QtSvgWidgets import QSvgWidget

from popup_animator import PopupAnimator
from message_queue import MessageQueue
from message_types import PopupMessage, MessageStyle, StyleConfig, STYLE_CONFIGS, MESSAGE_TYPE_MAP


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
    """
    Popup notification window with priority queue.
    
    Animation rules:
    - New message (first in queue): slide_in
    - Message from queue (was waiting): scale_in (previous slides out)
    - Higher priority interrupts: current scales out, new slides in
    """
    
    _new_message_signal = Signal(object)
    
    BORDER_WIDTH = 2
    BORDER_RADIUS = 10
    BACKGROUND_COLOR = "rgba(50, 50, 50, 220)"
    FONT_SIZE = 13
    PRIORITY_CHECK_MS = 100
    
    def __init__(self):
        super().__init__()
        
        self._screen = QApplication.primaryScreen().availableGeometry()
        self._panel_width = int(self._screen.width() * 0.20)
        self._panel_height = int(self._screen.height() * 0.08)
        self._margin = int(self._screen.height() * 0.015)
        
        self._state = State.IDLE
        self._current: Optional[PopupMessage] = None
        self._queue = MessageQueue()
        
        self._display_timer: Optional[QTimer] = None
        self._check_timer: Optional[QTimer] = None
        self._elapsed: Optional[QElapsedTimer] = None
        
        self._setup_window()
        self._setup_content()
        self._setup_animator()
        
        self._new_message_signal.connect(self._on_new_message)
    
    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(self._panel_width, self._panel_height)
        
        x = (self._screen.width() - self._panel_width) // 2
        y = self._screen.height()
        self.move(x, y)
        
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)
    
    TITLE_TOP_MARGIN = 5
    
    def _setup_content(self):
        self._container = PopupContainer()
        self._container.setFixedSize(self._panel_width, self._panel_height)
        
        padding = int(self._screen.height() * 0.01)
        gap = int(self._panel_width * 0.05)
        
        h_layout = QHBoxLayout(self._container)
        h_layout.setContentsMargins(padding, padding, padding, padding)
        h_layout.setSpacing(gap)
        
        self._icon = QSvgWidget()
        self._icon.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        icon_size = self._panel_height - (2 * padding) - (2 * self.BORDER_WIDTH)
        self._icon.setFixedSize(QSize(icon_size, icon_size))
        h_layout.addWidget(self._icon, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Text area widget with manual positioning for title + message
        self._text_area = QWidget()
        self._text_area.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        h_layout.addWidget(self._text_area, stretch=1)
        
        # Title: fixed 15px from top of the text area
        self._title_label = QLabel(self._text_area)
        title_font = QFont("Arial", self.FONT_SIZE)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._title_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        # Message: will be centered in remaining space below title
        self._label = QLabel(self._text_area)
        self._label.setFont(QFont("Arial", self.FONT_SIZE))
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        self._apply_style(STYLE_CONFIGS[MessageStyle.NOTICE])
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_text_labels()
    
    def _position_text_labels(self):
        """Position title at fixed top margin, message centered in remaining space."""
        if not hasattr(self, '_text_area'):
            return
        
        area_w = self._text_area.width()
        area_h = self._text_area.height()
        
        # Title at fixed offset from top
        title_h = self._title_label.sizeHint().height()
        self._title_label.setGeometry(0, self.TITLE_TOP_MARGIN, area_w, title_h)
        
        # Message centered in the space below the title
        msg_top = self.TITLE_TOP_MARGIN + title_h
        remaining_h = area_h - msg_top
        msg_h = self._label.sizeHint().height()
        msg_y = msg_top + (remaining_h - msg_h) // 2
        self._label.setGeometry(0, msg_y, area_w, msg_h)
    
    def _setup_animator(self):
        target_y = self._screen.height() - self._panel_height - self._margin
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
        """
        Emit a popup message (thread-safe).
        
        Args:
            title: Bold title shown at the top of the popup.
            message: Body text shown below the title.
            message_type: Single character for the type of message:
                'e' = error, 'w' = warning, 'c' = check, 'n' = notice.
            duration_ms: How long to display the message (optional, default 5000).
            priority: Message priority for queue ordering (optional, default 0).
        """
        style = MESSAGE_TYPE_MAP.get(message_type.lower(), MessageStyle.NOTICE)
        popup_message = PopupMessage(
            priority=priority,
            title=title,
            text=message,
            style=style,
            duration_ms=duration_ms
        )
        self._new_message_signal.emit(popup_message)
    
    def _on_new_message(self, message: PopupMessage):
        """Handle new message arrival."""
        
        self._queue.push(message)
        
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
            return

        self._current = message
        self._apply_style(message.get_style_config())
        self._title_label.setText(message.title)
        self._label.setText(message.text)
        self._position_text_labels()
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
        self._state = State.ANIMATING_OUT
        self._animator.scale_out()
    
    def _finish_current(self):
        """Current message display time complete."""
        self._stop_timers()
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
    
    def _stop_timers(self):
        if self._display_timer:
            self._display_timer.stop()
            self._display_timer = None
        if self._check_timer:
            self._check_timer.stop()
            self._check_timer = None
        self._elapsed = None
    
    def _check_priority(self):
        """Check if higher priority message arrived."""
        if self._state == State.DISPLAYING and self._current:
            if self._queue.has_higher_priority_than(self._current):
                self._push_back_current()
    
    def _on_animation_done(self, anim_type: str):        
        if anim_type in ("slide_in", "scale_in"):
            self._state = State.DISPLAYING
            self._start_timers()
        
        elif anim_type == "scale_out":
            self.hide()
            self._animator.reset()
            self._current = None
            self._state = State.IDLE
            self._show_next(from_queue=False)
        
        elif anim_type == "slide_out":
            self.hide()
            self._animator.reset()
            self._current = None
            self._state = State.IDLE
            if not self._queue.is_empty():
                self._show_next(from_queue=True)