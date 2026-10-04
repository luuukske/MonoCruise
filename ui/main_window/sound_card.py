"""Compact AEB audio settings with local import and a one-shot preview."""
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout

from core.aeb.sound_preferences import STYLES, import_sound, load_sound, volume
from core.thread_management.registry import registry


class SoundCard(QFrame):
    def __init__(self, settings, save):
        super().__init__()
        self.setObjectName("soundCard")
        self.settings = settings
        self.save = save
        self._preview = None
        self._channel = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(QLabel("AEB warning sound"))
        self.style = QComboBox()
        self.style.addItems(STYLES)
        self.style.setCurrentText(settings.aeb_sound_style)
        self.style.currentTextChanged.connect(self._change_style)
        layout.addWidget(self.style)
        self.level = QLabel()
        layout.addWidget(self.level)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(round(volume() * 100))
        self.slider.valueChanged.connect(self._volume)
        self.slider.valueChanged.connect(lambda _: save())
        layout.addWidget(self.slider)
        row = QHBoxLayout()
        for text, callback in (("Choose file", self._choose), ("Listen", self._listen), ("Reset", self._reset)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.hint = QLabel("WAV, OGG or MP3 • 0.2–5 seconds")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self._volume(self.slider.value())
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._preview_tick)

    def sync(self):
        self.style.blockSignals(True)
        self.style.setCurrentText(self.settings.aeb_sound_style)
        self.style.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setValue(round(volume() * 100))
        self.slider.blockSignals(False)
        self._volume(self.slider.value())
        self._apply()

    def _handler(self):
        try:
            return registry.get_thread("aeb_thread")._sound_handler
        except (KeyError, AttributeError):
            return None

    def _apply(self):
        try:
            sound = load_sound()
            handler = self._handler()
            if handler is not None and not handler.replace_sound(sound):
                self.hint.setText("Wait until the active warning finishes, then try again.")
                return False
            self.save()
            self.hint.setText("Sound saved. Use Listen to preview.")
            return True
        except Exception:
            self.hint.setText("Audio could not be loaded. Check your audio device.")
            return False

    def _change_style(self, style):
        previous = self.settings.aeb_sound_style
        if style == "Custom" and not self.settings.aeb_sound_file:
            self.style.blockSignals(True)
            self.style.setCurrentText(previous)
            self.style.blockSignals(False)
            self._choose()
            return
        self.settings.aeb_sound_style = style
        if not self._apply():
            self.settings.aeb_sound_style = previous
            self.style.blockSignals(True)
            self.style.setCurrentText(previous)
            self.style.blockSignals(False)

    def _choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose AEB sound", "", "Audio (*.wav *.ogg *.mp3)")
        if not path:
            return
        previous = (self.settings.aeb_sound_file, self.settings.aeb_sound_style)
        try:
            self.settings.aeb_sound_file = import_sound(Path(path))
            self.settings.aeb_sound_style = "Custom"
            if not self._apply():
                self.settings.aeb_sound_file, self.settings.aeb_sound_style = previous
                return
            self.style.blockSignals(True)
            self.style.setCurrentText("Custom")
            self.style.blockSignals(False)
        except Exception:
            self.settings.aeb_sound_file, self.settings.aeb_sound_style = previous
            self.hint.setText("Choose valid audio, 0.2–5 seconds, under 10 MB.")

    def _volume(self, value):
        self.settings.aeb_sound_volume = value
        self.level.setText(f"Volume: {value}%" + (" (muted)" if value == 0 else ""))
        handler = self._handler()
        if handler is not None and handler._sound is not None:
            handler._sound.set_volume(volume())
        if self._preview is not None:
            self._preview.set_volume(volume())

    def _listen(self):
        if self._channel is not None:
            self._channel.stop()
            self._channel = None
            return
        try:
            handler = self._handler()
            if handler is not None and handler._state != 0:
                self.hint.setText("Preview is unavailable during an active warning.")
                return
            self._preview = load_sound()
            self._channel = self._preview.play()
            self._timer.start(50)
        except Exception:
            self.hint.setText("Preview unavailable. Check your audio device.")

    def _preview_tick(self):
        handler = self._handler()
        if self._channel is not None and handler is not None and handler._state != 0:
            self._channel.stop()
        if self._channel is None or not self._channel.get_busy():
            self._channel = None
            self._timer.stop()

    def _reset(self):
        self.style.setCurrentText("Chime")
        self.slider.setValue(50)
        self.save()
