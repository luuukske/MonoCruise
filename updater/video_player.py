"""
Simple video player widget using PyQt6 Multimedia.
Minimalistic overlay controls that fade on hover.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton,
    QSlider, QLabel
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, pyqtProperty
from PyQt6.QtGui import QCursor, QPainter, QColor, QPolygon
from PyQt6.QtCore import QPoint


class PlayPauseButton(QPushButton):
    """Custom play/pause button with white icons."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_playing = False
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet("background: transparent; border: none;")
    
    def set_playing(self, is_playing: bool):
        self._is_playing = is_playing
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw hover background
        if self.underMouse():
            painter.setBrush(QColor(255, 255, 255, 40))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self.rect(), 4, 4)
        
        # Draw icon
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)
        
        w = self.width()
        h = self.height()
        
        if self._is_playing:
            # Pause icon (two vertical bars)
            bar_width = w // 5
            bar_height = h // 2
            gap = bar_width
            left_x = (w - (2 * bar_width + gap)) // 2
            top_y = (h - bar_height) // 2
            
            painter.drawRect(left_x, top_y, bar_width, bar_height)
            painter.drawRect(left_x + bar_width + gap, top_y, bar_width, bar_height)
        else:
            # Play icon (triangle)
            triangle_width = w // 3
            triangle_height = h // 2
            left_x = (w - triangle_width) // 2 + 2
            top_y = (h - triangle_height) // 2
            
            triangle = QPolygon([
                QPoint(left_x, top_y),
                QPoint(left_x, top_y + triangle_height),
                QPoint(left_x + triangle_width, top_y + triangle_height // 2)
            ])
            painter.drawPolygon(triangle)
        
        painter.end()


class VideoOverlayControls(QWidget):
    """Overlay controls for video player."""
    
    play_clicked = pyqtSignal()
    slider_moved = pyqtSignal(int)
    slider_pressed = pyqtSignal()
    slider_released = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._control_height = 40
        self._opacity = 1.0  # Start visible
        self._setup_ui()
        self._setup_animation()
    
    def _get_opacity(self):
        return self._opacity
    
    def _set_opacity(self, value):
        self._opacity = value
        self.update()
        # Update child widgets opacity effect
        for child in self.findChildren(QWidget):
            child.update()
    
    opacity_prop = pyqtProperty(float, _get_opacity, _set_opacity)
    
    def _setup_ui(self):
        """Setup the control UI."""
        font_size = max(10, self._control_height // 4)
        slider_height = max(4, self._control_height // 10)
        slider_radius = max(2, self._control_height // 20)
        handle_size = max(10, self._control_height // 4)
        
        self.setStyleSheet(f"""
            QLabel#timeLabel {{
                background: transparent;
                color: white;
                font-size: {font_size}px;
            }}
            QSlider {{
                background: transparent;
            }}
            QSlider::groove:horizontal {{
                background: rgba(255, 255, 255, 0.3);
                height: {slider_height}px;
                border-radius: {slider_radius}px;
            }}
            QSlider::handle:horizontal {{
                background: white;
                width: {handle_size}px;
                height: {handle_size}px;
                margin: -{handle_size // 3}px 0;
                border-radius: {handle_size // 2}px;
            }}
            QSlider::sub-page:horizontal {{
                background: rgba(255, 255, 255, 0.8);
                border-radius: {slider_radius}px;
            }}
        """)
        
        # Clear existing layout if rebuilding
        if self.layout():
            while self.layout().count():
                item = self.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            QWidget().setLayout(self.layout())
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)
        
        # Play/Pause button
        self.play_button = PlayPauseButton()
        self.play_button.setFixedSize(max(28, self._control_height - 12), max(28, self._control_height - 12))
        self.play_button.clicked.connect(self.play_clicked.emit)
        layout.addWidget(self.play_button)
        
        # Current time label
        self.current_time_label = QLabel("0:00")
        self.current_time_label.setObjectName("timeLabel")
        layout.addWidget(self.current_time_label)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.progress_slider.sliderMoved.connect(self.slider_moved.emit)
        self.progress_slider.sliderPressed.connect(self.slider_pressed.emit)
        self.progress_slider.sliderReleased.connect(self.slider_released.emit)
        layout.addWidget(self.progress_slider, stretch=1)
        
        # Duration label
        self.duration_label = QLabel("0:00")
        self.duration_label.setObjectName("timeLabel")
        layout.addWidget(self.duration_label)
    
    def _setup_animation(self):
        """Setup fade animation."""
        self._fade_animation = QPropertyAnimation(self, b"opacity_prop")
        self._fade_animation.setDuration(300)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
    
    def fade_in(self):
        """Fade in the controls."""
        self._fade_animation.stop()
        self._fade_animation.setStartValue(self._opacity)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()
    
    def fade_out(self):
        """Fade out the controls."""
        self._fade_animation.stop()
        self._fade_animation.setStartValue(self._opacity)
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()
    
    def paintEvent(self, event):
        """Custom paint to draw background with opacity."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw semi-transparent background
        alpha = int(140 * self._opacity)
        bg_color = QColor(0, 0, 0, alpha)
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 8, 8)
        
        painter.end()
    
    def set_control_height(self, height: int):
        """Update control sizing based on player dimensions."""
        self._control_height = height
        self._setup_ui()
    
    def set_playing(self, is_playing: bool):
        """Update play button icon."""
        self.play_button.set_playing(is_playing)
    
    def set_position(self, position: int):
        """Update slider position."""
        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(position)
        self.current_time_label.setText(self._format_time(position))
    
    def set_duration(self, duration: int):
        """Update slider range and duration label."""
        self.progress_slider.setRange(0, duration)
        self.duration_label.setText(self._format_time(duration))
    
    def get_opacity(self) -> float:
        """Get current opacity."""
        return self._opacity
    
    @staticmethod
    def _format_time(ms: int) -> str:
        """Format milliseconds as M:SS or H:MM:SS."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        hours = minutes // 60
        minutes = minutes % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


class VideoPlayer(QWidget):
    """Minimalistic video player with overlay controls."""
    
    DEFAULT_HEIGHT = 300
    DEFAULT_ASPECT_RATIO = 16 / 9
    CONTROL_HEIGHT_PERCENT = 0.15
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._aspect_ratio = self.DEFAULT_ASPECT_RATIO
        self._video_url = None
        self._is_slider_pressed = False
        self._metadata_checked = False
        self._mouse_inside = False
        
        self._setup_media()
        self._setup_ui()
        self._update_size()
        
        # Timer for hover detection
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(50)
        self._hover_timer.timeout.connect(self._check_hover)
        
        # Hide by default
        self.hide()
    
    def _setup_media(self):
        """Setup media player components."""
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # Connect signals
        self.media_player.positionChanged.connect(self._position_changed)
        self.media_player.durationChanged.connect(self._duration_changed)
        self.media_player.playbackStateChanged.connect(self._state_changed)
        self.media_player.errorOccurred.connect(self._handle_error)
        self.media_player.mediaStatusChanged.connect(self._media_status_changed)
    
    def _setup_ui(self):
        """Setup the player UI."""
        # Video widget - child of this widget
        self.video_widget = QVideoWidget(self)
        self.video_widget.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.video_widget.setStyleSheet("background-color: #000000;")
        self.media_player.setVideoOutput(self.video_widget)
        
        # Overlay controls - child of this widget, on top of video
        self.controls = VideoOverlayControls(self)
        self.controls.play_clicked.connect(self._toggle_play)
        self.controls.slider_moved.connect(self._seek)
        self.controls.slider_pressed.connect(self._slider_pressed)
        self.controls.slider_released.connect(self._slider_released)
    
    def _update_size(self):
        """Update player size based on aspect ratio."""
        height = self.DEFAULT_HEIGHT
        width = int(height * self._aspect_ratio)
        
        self.setFixedHeight(height)
        self.setFixedWidth(width)
        
        # Video fills entire widget
        self.video_widget.setGeometry(0, 0, width, height)
        
        # Control sizing
        control_height = int(min(width, height) * self.CONTROL_HEIGHT_PERCENT)
        self.controls.set_control_height(control_height)
        
        # Position controls at bottom with padding
        control_width = width - 20
        self.controls.setGeometry(
            10,
            height - control_height - 10,
            control_width,
            control_height
        )
        
        # Ensure controls are on top
        self.controls.raise_()
    
    def load_video(self, url: str):
        """Load a video from URL."""
        if not url:
            self.hide()
            return
        
        self._video_url = url
        self._aspect_ratio = self.DEFAULT_ASPECT_RATIO
        self._metadata_checked = False
        self._update_size()
        
        self.media_player.setSource(QUrl(url))
        self.controls.set_playing(False)
        self.controls.set_position(0)
        self.controls.set_duration(0)
        
        # Start with controls visible
        self.controls._opacity = 1.0
        self.controls.update()
        
        self.show()
        
        # Start hover timer
        self._hover_timer.start()
        
        # Play briefly then pause to get thumbnail
        QTimer.singleShot(100, self._generate_thumbnail)
    
    def _generate_thumbnail(self):
        """Play briefly and pause to generate a thumbnail frame."""
        if self.media_player.mediaStatus() in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia
        ):
            self.media_player.play()
            QTimer.singleShot(50, self._pause_for_thumbnail)
        else:
            # Media not ready yet, try again
            QTimer.singleShot(100, self._generate_thumbnail)
    
    def _pause_for_thumbnail(self):
        """Pause after brief play to show thumbnail."""
        self.media_player.pause()
        self.media_player.setPosition(0)
    
    def clear(self):
        """Clear the current video."""
        self._hover_timer.stop()
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        self._video_url = None
        self._aspect_ratio = self.DEFAULT_ASPECT_RATIO
        self._metadata_checked = False
        self.controls.set_position(0)
        self.controls.set_duration(0)
        self.hide()
    
    def _toggle_play(self):
        """Toggle between play and pause."""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
    
    def _seek(self, position: int):
        """Seek to position in video."""
        self.media_player.setPosition(position)
    
    def _slider_pressed(self):
        """Track when slider is being dragged."""
        self._is_slider_pressed = True
    
    def _slider_released(self):
        """Track when slider drag ends."""
        self._is_slider_pressed = False
        self._seek(self.controls.progress_slider.value())
    
    def _position_changed(self, position: int):
        """Update UI when playback position changes."""
        if not self._is_slider_pressed:
            self.controls.set_position(position)
    
    def _duration_changed(self, duration: int):
        """Update UI when video duration is known."""
        if duration > 0:
            self.controls.set_duration(duration)
    
    def _state_changed(self, state: QMediaPlayer.PlaybackState):
        """Update play button based on state."""
        is_playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.controls.set_playing(is_playing)
    
    def _media_status_changed(self, status: QMediaPlayer.MediaStatus):
        """Handle media status changes."""
        if status in (QMediaPlayer.MediaStatus.LoadedMedia,
                      QMediaPlayer.MediaStatus.BufferedMedia):
            self._check_video_size()
    
    def _check_video_size(self):
        """Check video size from video sink when available."""
        if self._metadata_checked:
            return
            
        video_sink = self.video_widget.videoSink()
        if video_sink:
            video_size = video_sink.videoSize()
            if video_size.isValid() and video_size.width() > 0 and video_size.height() > 0:
                new_ratio = video_size.width() / video_size.height()
                if abs(new_ratio - self._aspect_ratio) > 0.01:
                    self._aspect_ratio = new_ratio
                    self._update_size()
                self._metadata_checked = True
    
    def _handle_error(self, error, error_string: str):
        """Handle media player errors."""
        print(f"Video player error: {error_string}")
    
    def _check_hover(self):
        """Check if mouse is inside the widget."""
        if not self.isVisible():
            return
            
        global_pos = QCursor.pos()
        widget_rect = self.rect()
        local_pos = self.mapFromGlobal(global_pos)
        is_inside = widget_rect.contains(local_pos)
        
        if is_inside and not self._mouse_inside:
            self._mouse_inside = True
            self.controls.fade_in()
        elif not is_inside and self._mouse_inside:
            self._mouse_inside = False
            self.controls.fade_out()
    
    def showEvent(self, event):
        """Handle show event."""
        super().showEvent(event)
        self.controls.raise_()
        self._hover_timer.start()
    
    def hideEvent(self, event):
        """Handle hide event."""
        super().hideEvent(event)
        self._hover_timer.stop()
    
    def mousePressEvent(self, event):
        """Handle click on video area to toggle play."""
        controls_rect = self.controls.geometry()
        if not controls_rect.contains(event.pos()):
            self._toggle_play()
        super().mousePressEvent(event)
    
    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        self._update_size()