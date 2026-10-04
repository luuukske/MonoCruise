"""Portable audio imports, preset decoding and fallback behavior."""
import io
import wave

import pytest

from core.settings import Settings
from core.aeb import sound_preferences as audio


@pytest.fixture()
def mixer(monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame
    pygame.mixer.init()
    yield pygame
    pygame.mixer.quit()


def test_presets_decode_with_a_bounded_duration(mixer):
    for style in ("Chime", "Pulse"):
        with wave.open(io.BytesIO(audio.preset_bytes(style))) as sound:
            assert sound.getnframes() / sound.getframerate() == pytest.approx(0.9)
        assert mixer.mixer.Sound(file=io.BytesIO(audio.preset_bytes(style))).get_length() > 0.2


def test_import_is_portable_and_survives_source_removal(mixer, tmp_path, monkeypatch):
    monkeypatch.setattr(audio.settings_module, "CONFIG_PATH", tmp_path / "app" / "config.json")
    source = tmp_path / "source.wav"
    source.write_bytes(audio.preset_bytes("Pulse"))
    name = audio.import_sound(source)
    assert "/" not in name and "\\" not in name
    source.unlink()
    Settings.save({"aeb_sound_style": "Custom", "aeb_sound_file": name, "aeb_sound_volume": 25})
    assert audio.load_sound().get_volume() == pytest.approx(0.25, abs=0.01)


def test_bad_custom_file_falls_back_to_audible_chime(mixer):
    Settings.save({"aeb_sound_style": "Custom", "aeb_sound_file": "missing.wav", "aeb_sound_volume": 50})
    assert audio.load_sound().get_length() == pytest.approx(0.9)


def test_traversal_is_rejected():
    Settings.save({"aeb_sound_file": "../outside.wav"})
    with pytest.raises(ValueError):
        audio.custom_path()


def test_invalid_audio_does_not_get_copied(mixer, tmp_path):
    source = tmp_path / "bad.wav"
    source.write_bytes(b"invalid")
    with pytest.raises(mixer.error):
        audio.import_sound(source)
