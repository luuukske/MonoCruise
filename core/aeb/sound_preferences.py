"""Offline warning presets and portable custom audio imports."""
from __future__ import annotations

import hashlib
import io
import logging
import math
from pathlib import Path
import struct
import wave

from core import settings as settings_module
from core.settings import Settings

STYLES = ("Chime", "Pulse", "Original", "Custom")


def volume() -> float:
    try:
        return max(0, min(100, int(Settings.aeb_sound_volume))) / 100
    except (ValueError, TypeError, OverflowError):
        return 0.5


def custom_path() -> Path:
    name = Settings.aeb_sound_file
    if not isinstance(name, str) or Path(name).name != name or not name:
        raise ValueError("Invalid sound selection")
    return settings_module.CONFIG_PATH.parent / "sounds" / name


def preset_bytes(style: str) -> bytes:
    rate = 44100
    samples = bytearray()
    for i in range(int(rate * 0.9)):
        t = i / rate
        local = t % 0.3
        active = local < 0.17 and t < 0.77
        envelope = min(1, local / 0.012, max(0, (0.17 - local) / 0.035)) if active else 0
        frequency = 880 if style == "Chime" else 1100
        tone = math.sin(2 * math.pi * frequency * t)
        samples.extend(struct.pack("<h", int(11000 * envelope * tone)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples)
    return buffer.getvalue()


def load_sound():
    import pygame

    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
    style = Settings.aeb_sound_style
    try:
        if style == "Custom":
            sound = pygame.mixer.Sound(str(custom_path()))
        elif style == "Original":
            sound = pygame.mixer.Sound(str(Path(__file__).with_name("AEB_warning.wav")))
        else:
            sound = pygame.mixer.Sound(file=io.BytesIO(preset_bytes(style)))
        if not 0.2 <= sound.get_length() <= 5:
            raise ValueError("Warning audio must be 0.2 to 5 seconds")
    except Exception:
        logging.getLogger(__name__).warning("Warning audio unavailable; using built-in chime")
        sound = pygame.mixer.Sound(file=io.BytesIO(preset_bytes("Chime")))
    sound.set_volume(volume())
    return sound


def import_sound(path: Path) -> str:
    import pygame

    if path.suffix.lower() not in {".wav", ".ogg", ".mp3"}:
        raise ValueError("Choose WAV, OGG or MP3 audio")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Audio must be smaller than 10 MB")
    data = path.read_bytes()
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    sound = pygame.mixer.Sound(file=io.BytesIO(data))
    if not 0.2 <= sound.get_length() <= 5:
        raise ValueError("Choose a clip between 0.2 and 5 seconds")
    name = hashlib.sha256(data).hexdigest() + path.suffix.lower()
    directory = settings_module.CONFIG_PATH.parent / "sounds"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(data)
    return name
