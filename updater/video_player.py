"""
Video player widget using PySide6 Multimedia with QVideoSink.
Renders frames manually for rounded corners and transparent background.
Shows thumbnail with big play button when paused or not started.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink, QVideoFrame
from PySide6.QtCore import Qt, QUrl, QTimer, Signal, QRect, QRectF, QPoint
from PySide6.QtGui import QCursor, QPainter, QColor, QPolygon, QImage, QPainterPath


class _ControlBar(QWidget):
    """Bottom overlay control bar: play/pause, time labels, progress slider."""

    play_clicked = Signal()
    slider_moved = Signal(int)
    slider_pressed = Signal()
    slider_released = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._opacity = 0.0
        self._is_playing = False
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        # Play/Pause button (invisible - icon drawn in paintEvent)
        self.play_btn = QPushButton()
        self.play_btn.setFixedSize(24, 24)
        self.play_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.play_btn.setStyleSheet("background: transparent; border: none;")
        self.play_btn.clicked.connect(self.play_clicked.emit)
        layout.addWidget(self.play_btn)

        # Current time
        self.time_label = QLabel("0:00")
        layout.addWidget(self.time_label)

        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.progress_slider.sliderMoved.connect(self.slider_moved.emit)
        self.progress_slider.sliderPressed.connect(self.slider_pressed.emit)
        self.progress_slider.sliderReleased.connect(self.slider_released.emit)
        layout.addWidget(self.progress_slider, stretch=1)

        # Duration
        self.duration_label = QLabel("0:00")
        layout.addWidget(self.duration_label)

        self._apply_style()

    def _apply_style(self):
        a = int(255 * self._opacity)
        a_low = int(77 * self._opacity)
        a_high = int(204 * self._opacity)

        self.setStyleSheet(f"""
            QWidget {{
                background: transparent;
            }}
            QLabel {{
                color: rgba(255, 255, 255, {a});
                font-size: 11px;
                background: transparent;
            }}
            QPushButton {{
                background: transparent;
                border: none;
            }}
            QSlider {{
                background: transparent;
            }}
            QSlider::groove:horizontal {{
                background: rgba(255, 255, 255, {a_low});
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: rgba(255, 255, 255, {a});
                width: 10px;
                height: 10px;
                margin: -3px 0;
                border-radius: 5px;
            }}
            QSlider::sub-page:horizontal {{
                background: rgba(255, 255, 255, {a_high});
                border-radius: 2px;
            }}
        """)

    def set_opacity(self, opacity: float):
        self._opacity = opacity
        # Hide interactive child widgets at very low opacity to prevent
        # the slider/labels from flashing while nearly transparent.
        visible = opacity > 0.15
        self.progress_slider.setVisible(visible)
        self.time_label.setVisible(visible)
        self.duration_label.setVisible(visible)
        self.play_btn.setVisible(visible)
        self._apply_style()
        self.update()

    def paintEvent(self, event):
        if self._opacity <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent dark background
        alpha = int(160 * self._opacity)
        painter.setBrush(QColor(0, 0, 0, alpha))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 6, 6)

        # Draw play/pause icon on the button area
        btn = self.play_btn.geometry()
        painter.setOpacity(self._opacity)
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)
        cx, cy = btn.center().x(), btn.center().y()

        if self._is_playing:
            # Pause bars
            bw, bh, gap = 3, 10, 4
            painter.drawRect(cx - gap // 2 - bw, cy - bh // 2, bw, bh)
            painter.drawRect(cx + gap // 2, cy - bh // 2, bw, bh)
        else:
            # Play triangle
            tw, th = 8, 10
            lx = cx - tw // 2 + 1
            ty = cy - th // 2
            painter.drawPolygon(QPolygon([
                QPoint(lx, ty),
                QPoint(lx, ty + th),
                QPoint(lx + tw, ty + th // 2),
            ]))

        painter.end()

    def mousePressEvent(self, event):
        """Accept all clicks so they don't propagate to the VideoPlayer."""
        event.accept()

    def set_playing(self, playing: bool):
        self._is_playing = playing
        self.update()

    def set_position(self, ms: int):
        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(ms)
        self.time_label.setText(self._format_time(ms))

    def set_duration(self, ms: int):
        self.progress_slider.setRange(0, ms)
        self.duration_label.setText(self._format_time(ms))

    @staticmethod
    def _format_time(ms: int) -> str:
        seconds = ms // 1000
        minutes = seconds // 60
        seconds %= 60
        hours = minutes // 60
        minutes %= 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


class VideoPlayer(QWidget):
    """
    Video player with rounded corners and transparent surroundings.

    Uses QVideoSink to capture frames and paint them manually, enabling
    rounded-corner clipping and a transparent widget background. When
    paused or not yet started the last frame is shown as a thumbnail
    with a large centered play button overlay.
    """

    DEFAULT_HEIGHT = 300
    DEFAULT_ASPECT_RATIO = 16 / 9
    CORNER_RADIUS = 10

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self._aspect_ratio = self.DEFAULT_ASPECT_RATIO
        self._video_url = None
        self._is_slider_pressed = False
        self._metadata_checked = False
        self._mouse_inside = False
        self._is_playing = False
        self._has_started = False
        self._generating_thumbnail = False

        # Frame storage
        self._current_frame: QImage | None = None
        self._thumbnail: QImage | None = None
        self._first_frame: QImage | None = None  # cached for reuse at end-of-media

        # Transparent background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        # Media pipeline
        self._setup_media()

        # Overlay controls (child widget)
        self.controls = _ControlBar(self)
        self.controls.play_clicked.connect(self._toggle_play)
        self.controls.slider_moved.connect(self._seek)
        self.controls.slider_pressed.connect(self._on_slider_pressed)
        self.controls.slider_released.connect(self._on_slider_released)

        self._update_size()

        # --- Timers ---

        # Hover polling
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(50)
        self._hover_timer.timeout.connect(self._check_hover)

        # Delayed control hide
        self._hide_timer = QTimer(self)
        self._hide_timer.setInterval(2000)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._delayed_hide_controls)

        # Controls fade animation
        self._ctrl_opacity = 0.0
        self._ctrl_target = 0.0
        self._ctrl_fade = QTimer(self)
        self._ctrl_fade.setInterval(16)
        self._ctrl_fade.timeout.connect(self._animate_ctrl_fade)

        # Big play-button fade animation
        self._play_btn_opacity = 1.0
        self._play_btn_target = 1.0
        self._play_btn_fade = QTimer(self)
        self._play_btn_fade.setInterval(16)
        self._play_btn_fade.timeout.connect(self._animate_play_btn_fade)

        self._play_btn_hovered = False

        self.hide()

    # ------------------------------------------------------------------
    #  Media setup
    # ------------------------------------------------------------------

    def _setup_media(self):
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        self.video_sink = QVideoSink()
        self.video_sink.videoFrameChanged.connect(self._on_video_frame)
        self.media_player.setVideoOutput(self.video_sink)

        self.media_player.positionChanged.connect(self._position_changed)
        self.media_player.durationChanged.connect(self._duration_changed)
        self.media_player.playbackStateChanged.connect(self._state_changed)
        self.media_player.errorOccurred.connect(self._handle_error)
        self.media_player.mediaStatusChanged.connect(self._media_status_changed)

    # ------------------------------------------------------------------
    #  Sizing
    # ------------------------------------------------------------------

    def _update_size(self):
        height = self.DEFAULT_HEIGHT
        width = int(height * self._aspect_ratio)
        self.setFixedSize(width, height)

        ctrl_h = max(36, int(height * 0.12))
        pad = 8
        self.controls.setGeometry(QRect(
            pad, height - ctrl_h - pad,
            width - 2 * pad, ctrl_h,
        ))

    # ------------------------------------------------------------------
    #  Frame capture
    # ------------------------------------------------------------------

    def _on_video_frame(self, frame: QVideoFrame):
        if not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        self._current_frame = image.convertToFormat(QImage.Format.Format_ARGB32)
        # While paused, keep the thumbnail in sync with the current frame
        # (e.g. the user seeks while paused)
        if not self._is_playing:
            self._thumbnail = self._current_frame
        self.update()

    # ------------------------------------------------------------------
    #  Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Clip everything to rounded rect
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, w, h), self.CORNER_RADIUS, self.CORNER_RADIUS)
        painter.setClipPath(clip)

        # Choose which image to draw
        frame = self._current_frame if self._is_playing else self._thumbnail

        if frame and not frame.isNull():
            # Scale to device-pixel dimensions so the image stays sharp
            # on high-DPI / 4K screens.
            dpr = self.devicePixelRatioF()
            scaled = frame.scaled(
                int(w * dpr), int(h * dpr),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            scaled.setDevicePixelRatio(dpr)
            x = int((w - scaled.width() / dpr) / 2)
            y = int((h - scaled.height() / dpr) / 2)
            painter.drawImage(x, y, scaled)
        else:
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0))

        # Show big play button only when video hasn't been started yet
        if not self._has_started and self._play_btn_opacity > 0:
            # Subtle dark overlay so the play button pops
            painter.setOpacity(self._play_btn_opacity * 0.15)
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0))
            painter.setOpacity(1.0)

            self._paint_big_play_button(painter, w, h)

        painter.end()

    def _paint_big_play_button(self, painter: QPainter, w: int, h: int):
        """Draw the large centered play-button circle with triangle."""
        painter.save()
        painter.setOpacity(self._play_btn_opacity)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = w // 2, h // 2
        radius = min(w, h) // 6
        radius = max(28, min(radius, 45))

        # Dark circle
        bg_alpha = 190 if self._play_btn_hovered else 150
        painter.setBrush(QColor(0, 0, 0, bg_alpha))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # White play triangle (offset a bit right for visual centering)
        tw = int(radius * 0.75)
        th = int(radius * 0.9)
        offset = int(tw * 0.12)
        lx = cx - tw // 2 + offset
        ty = cy - th // 2

        painter.setBrush(QColor(255, 255, 255))
        painter.drawPolygon(QPolygon([
            QPoint(lx, ty),
            QPoint(lx, ty + th),
            QPoint(lx + tw, ty + th // 2),
        ]))

        painter.restore()

    # ------------------------------------------------------------------
    #  Fade animations
    # ------------------------------------------------------------------

    def _animate_ctrl_fade(self):
        diff = self._ctrl_target - self._ctrl_opacity
        if abs(diff) < 0.02:
            self._ctrl_opacity = self._ctrl_target
            self._ctrl_fade.stop()
            if self._ctrl_opacity <= 0:
                self.controls.hide()
        else:
            self._ctrl_opacity += diff * 0.18
        self.controls.set_opacity(self._ctrl_opacity)

    def _show_controls(self):
        self.controls.show()
        self._ctrl_target = 1.0
        self._ctrl_fade.start()

    def _hide_controls_anim(self):
        self._ctrl_target = 0.0
        self._ctrl_fade.start()

    def _animate_play_btn_fade(self):
        diff = self._play_btn_target - self._play_btn_opacity
        if abs(diff) < 0.02:
            self._play_btn_opacity = self._play_btn_target
            self._play_btn_fade.stop()
        else:
            self._play_btn_opacity += diff * 0.18
        self.update()

    def _fade_out_play_btn(self):
        self._play_btn_target = 0.0
        self._play_btn_fade.start()

    def _fade_in_play_btn(self):
        self._play_btn_target = 1.0
        if self._play_btn_opacity < 0.01:
            self._play_btn_opacity = 0.0
        self._play_btn_fade.start()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def load_video(self, url: str):
        """Load a video from *url*. Shows thumbnail + play button once ready."""
        if not url:
            self.hide()
            return

        self._video_url = url
        self._aspect_ratio = self.DEFAULT_ASPECT_RATIO
        self._metadata_checked = False
        self._has_started = False
        self._generating_thumbnail = True
        self._current_frame = None
        self._thumbnail = None
        self._first_frame = None
        self._play_btn_opacity = 1.0
        self._play_btn_target = 1.0
        self._ctrl_opacity = 0.0
        self._ctrl_target = 0.0
        self.controls.hide()

        self._update_size()

        self.media_player.setSource(QUrl(url))
        self.controls.set_position(0)
        self.controls.set_duration(0)
        self.controls.set_playing(False)

        self.show()
        self._hover_timer.start()

        # Play briefly to grab a thumbnail frame
        QTimer.singleShot(100, self._generate_thumbnail)

    def _generate_thumbnail(self):
        status = self.media_player.mediaStatus()
        if status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
        ):
            # Mute so the brief play is silent
            self.audio_output.setMuted(True)
            self.media_player.play()
            QTimer.singleShot(80, self._pause_for_thumbnail)
        else:
            QTimer.singleShot(100, self._generate_thumbnail)

    def _pause_for_thumbnail(self):
        self.media_player.pause()
        if self._current_frame and not self._current_frame.isNull():
            self._thumbnail = self._current_frame
            self._first_frame = self._current_frame
        self.media_player.setPosition(0)
        self._has_started = False
        self._is_playing = False
        self._generating_thumbnail = False
        self.audio_output.setMuted(False)
        self.update()

    def clear(self):
        """Stop playback and hide the widget."""
        self._hover_timer.stop()
        self._hide_timer.stop()
        self._ctrl_fade.stop()
        self._play_btn_fade.stop()
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        self._video_url = None
        self._current_frame = None
        self._thumbnail = None
        self._first_frame = None
        self._aspect_ratio = self.DEFAULT_ASPECT_RATIO
        self._metadata_checked = False
        self._has_started = False
        self._generating_thumbnail = False
        self.controls.set_position(0)
        self.controls.set_duration(0)
        self.hide()

    # ------------------------------------------------------------------
    #  Playback controls
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self._is_playing:
            self.media_player.pause()
        else:
            if not self._has_started:
                self._has_started = True
            # If at end, restart
            dur = self.media_player.duration()
            if dur > 0 and self.media_player.position() >= dur - 100:
                self.media_player.setPosition(0)
            self.media_player.play()

    def _seek(self, position: int):
        self.media_player.setPosition(position)

    def _on_slider_pressed(self):
        self._is_slider_pressed = True

    def _on_slider_released(self):
        self._is_slider_pressed = False
        self._seek(self.controls.progress_slider.value())

    # ------------------------------------------------------------------
    #  Media-player signal handlers
    # ------------------------------------------------------------------

    def _position_changed(self, position: int):
        if not self._is_slider_pressed:
            self.controls.set_position(position)

    def _duration_changed(self, duration: int):
        if duration > 0:
            self.controls.set_duration(duration)

    def _state_changed(self, state: QMediaPlayer.PlaybackState):
        if self._generating_thumbnail:
            return

        was_playing = self._is_playing
        self._is_playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.controls.set_playing(self._is_playing)

        if self._is_playing:
            self._fade_out_play_btn()
            # Show controls when playback starts (e.g. user clicked video to start)
            # so the overlay appears even if the mouse was already over the widget.
            if self._has_started:
                self._show_controls()
                self._hide_timer.start()
        elif state == QMediaPlayer.PlaybackState.PausedState and was_playing:
            # Capture thumbnail of where the player paused
            if self._current_frame and not self._current_frame.isNull():
                self._thumbnail = self._current_frame

        self.update()

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus):
        if self._generating_thumbnail:
            return

        if status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            self._check_video_size()

        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            # Loop back to the start using the cached first frame —
            # no play needed so there's no audio blip or frame flicker.
            self._has_started = False
            self._is_playing = False
            self._play_btn_opacity = 1.0
            self._play_btn_target = 1.0
            if self._first_frame and not self._first_frame.isNull():
                self._thumbnail = self._first_frame
                # Also reset the \"current\" frame so starting playback again
                # does not briefly show the last frame from the previous run.
                self._current_frame = self._first_frame
            self.controls.set_position(0)
            self.controls.set_playing(False)
            self.media_player.setPosition(0)
            self.update()

    def _check_video_size(self):
        if self._metadata_checked:
            return
        size = self.video_sink.videoSize()
        if size.isValid() and size.width() > 0 and size.height() > 0:
            new_ratio = size.width() / size.height()
            if abs(new_ratio - self._aspect_ratio) > 0.01:
                self._aspect_ratio = new_ratio
                self._update_size()
            self._metadata_checked = True

    def _handle_error(self, error, error_string: str):
        print(f"Video player error: {error_string}")

    # ------------------------------------------------------------------
    #  Hover / mouse
    # ------------------------------------------------------------------

    def _check_hover(self):
        if not self.isVisible():
            return

        local = self.mapFromGlobal(QCursor.pos())
        inside = self.rect().contains(local)

        # Track play-button hover for visual feedback
        if not self._has_started:
            cx, cy = self.width() // 2, self.height() // 2
            radius = min(self.width(), self.height()) // 6
            radius = max(28, min(radius, 45))
            dx, dy = local.x() - cx, local.y() - cy
            was = self._play_btn_hovered
            self._play_btn_hovered = inside and (dx * dx + dy * dy <= radius * radius)
            if was != self._play_btn_hovered:
                self.update()

        if inside and not self._mouse_inside:
            self._mouse_inside = True
            self._hide_timer.stop()
            if self._has_started:
                self._show_controls()
        elif not inside and self._mouse_inside:
            self._mouse_inside = False
            if self._has_started:
                self._hide_timer.start()

    def _delayed_hide_controls(self):
        if not self._mouse_inside and self._is_playing:
            self._hide_controls_anim()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        # Let the control bar handle its own clicks
        if self.controls.isVisible() and self.controls.geometry().contains(event.pos()):
            return super().mousePressEvent(event)

        # Click on the video area → toggle play
        if not self._has_started:
            self._has_started = True
        self._toggle_play()

    def showEvent(self, event):
        super().showEvent(event)
        self._hover_timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._hover_timer.stop()
        self._hide_timer.stop()
