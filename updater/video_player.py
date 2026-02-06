"""
Simple video player widget using PyQt6 Multimedia.
Supports play/pause and progress bar.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QSlider, QLabel, QFrame, QSizePolicy
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QCursor

from styles import VIDEO_PLAYER_STYLE, BG_SECTION, COLOR_INACTIVE


class VideoPlayer(QFrame):
    """Simple video player with play/pause and progress bar."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoPlayer")
        self.setStyleSheet(VIDEO_PLAYER_STYLE)
        self.setMinimumHeight(200)
        self.setMaximumHeight(300)
        
        # Media components
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #000000; border-radius: 5px;")
        self.media_player.setVideoOutput(self.video_widget)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Video display
        layout.addWidget(self.video_widget, stretch=1)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Play/Pause button
        self.play_button = QPushButton("▶")
        self.play_button.setObjectName("playButton")
        self.play_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.play_button.setFixedWidth(50)
        self.play_button.clicked.connect(self._toggle_play)
        controls_layout.addWidget(self.play_button)
        
        # Current time label
        self.current_time_label = QLabel("0:00")
        self.current_time_label.setObjectName("timeLabel")
        self.current_time_label.setFixedWidth(45)
        controls_layout.addWidget(self.current_time_label)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.progress_slider.sliderMoved.connect(self._seek)
        self.progress_slider.sliderPressed.connect(self._slider_pressed)
        self.progress_slider.sliderReleased.connect(self._slider_released)
        controls_layout.addWidget(self.progress_slider, stretch=1)
        
        # Duration label
        self.duration_label = QLabel("0:00")
        self.duration_label.setObjectName("timeLabel")
        self.duration_label.setFixedWidth(45)
        controls_layout.addWidget(self.duration_label)
        
        layout.addLayout(controls_layout)
        
        # Connect signals
        self.media_player.positionChanged.connect(self._position_changed)
        self.media_player.durationChanged.connect(self._duration_changed)
        self.media_player.playbackStateChanged.connect(self._state_changed)
        self.media_player.errorOccurred.connect(self._handle_error)
        
        # State tracking
        self._is_slider_pressed = False
        self._video_url = None
        
        # Hide by default
        self.hide()
    
    def load_video(self, url: str):
        """Load a video from URL."""
        if not url:
            self.hide()
            return
        
        self._video_url = url
        self.media_player.setSource(QUrl(url))
        self.show()
        # Don't auto-play, let user click play
        self.play_button.setText("▶")
    
    def clear(self):
        """Clear the current video."""
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        self._video_url = None
        self.progress_slider.setValue(0)
        self.current_time_label.setText("0:00")
        self.duration_label.setText("0:00")
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
    
    def _position_changed(self, position: int):
        """Update UI when playback position changes."""
        if not self._is_slider_pressed:
            self.progress_slider.setValue(position)
        self.current_time_label.setText(self._format_time(position))
    
    def _duration_changed(self, duration: int):
        """Update UI when video duration is known."""
        self.progress_slider.setRange(0, duration)
        self.duration_label.setText(self._format_time(duration))
    
    def _state_changed(self, state: QMediaPlayer.PlaybackState):
        """Update play button based on state."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("⏸")
        else:
            self.play_button.setText("▶")
    
    def _handle_error(self, error, error_string: str):
        """Handle media player errors."""
        print(f"Video player error: {error_string}")
        # Could show error in UI if needed
    
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