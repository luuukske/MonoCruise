"""Tests for the AEB clip screenshot thumbnail path."""

from __future__ import annotations

import base64
import importlib
import io
import sys

import pytest

import core.aeb.screenshot as screenshot_mod
from core.aeb.recorder import AEBClipRecorder
from core.aeb.clip_store import deserialize_clip, serialize_clip

from tests.aeb.test_clip_capture import _StubWriter, _drive, _make_clip


def test_encode_thumbnail_downscales_and_roundtrips():
    Image = pytest.importorskip("PIL.Image")
    from core.aeb.screenshot import encode_thumbnail

    img = Image.new("RGB", (1920, 1080), (10, 20, 30))
    b64 = encode_thumbnail(img, max_px=480, quality=50)
    back = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert back.format == "JPEG"
    assert max(back.size) <= 480          # long side downscaled


def test_grab_thumbnail_never_raises():
    # No display / no Pillow in CI must yield None, not an exception.
    from core.aeb.screenshot import grab_thumbnail

    result = grab_thumbnail()
    assert result is None or isinstance(result, str)


class _FakeUser32:
    """Stand-in for ctypes.windll.user32: no real window, no real Windows call."""

    def __init__(
        self, *, found_title=None, found_class=None, hwnd=4321,
        rect=(100, 60, 740, 330), valid=True,
    ):
        self.found_title = found_title
        self.found_class = found_class
        self.hwnd = hwnd
        self.rect = rect
        self.valid = valid
        self.found_titles: list[str] = []
        self.found_classes: list[str] = []

    def FindWindowW(self, class_name, title):
        if title is not None:
            self.found_titles.append(title)
            return self.hwnd if title == self.found_title else 0
        self.found_classes.append(class_name)
        return self.hwnd if class_name == self.found_class else 0

    def IsWindow(self, hwnd):
        return bool(self.valid and hwnd == self.hwnd)

    def GetWindowRect(self, hwnd, rect_ptr):
        r = rect_ptr.contents
        r.left, r.top, r.right, r.bottom = self.rect
        return 1


@pytest.fixture(autouse=True)
def _reset_screenshot_module_state():
    screenshot_mod._cached_hwnd = None
    screenshot_mod._user32 = None
    yield
    screenshot_mod._cached_hwnd = None
    screenshot_mod._user32 = None


def test_module_import_is_safe_on_non_windows(monkeypatch):
    # ctypes.windll must never be touched at import time: CI runs on Ubuntu.
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    importlib.reload(screenshot_mod)


def test_grab_thumbnail_none_when_no_game_window(monkeypatch):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: _FakeUser32(found_title=None))
    assert screenshot_mod.grab_thumbnail() is None


def test_grab_thumbnail_never_falls_back_to_full_grab(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image_grab_mod = pytest.importorskip("PIL.ImageGrab")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: _FakeUser32(found_title=None))
    calls = []

    def _fake_grab(**kw):
        calls.append(kw)
        return Image.new("RGB", (10, 10))

    monkeypatch.setattr(image_grab_mod, "grab", _fake_grab)
    result = screenshot_mod.grab_thumbnail()
    assert result is None
    assert calls == []          # ImageGrab.grab is never invoked without a real window


def test_grab_thumbnail_bbox_matches_window_rect(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image_grab_mod = pytest.importorskip("PIL.ImageGrab")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="Euro Truck Simulator 2", rect=(50, 40, 1330, 760))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    calls = []

    def _fake_grab(**kw):
        calls.append(kw.get("bbox"))
        return Image.new("RGB", (1280, 720))

    monkeypatch.setattr(image_grab_mod, "grab", _fake_grab)
    result = screenshot_mod.grab_thumbnail()
    assert result is not None
    assert calls == [(50, 40, 1330, 760)]


def test_grab_thumbnail_finds_ats_when_ets2_absent(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image_grab_mod = pytest.importorskip("PIL.ImageGrab")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="American Truck Simulator", rect=(0, 0, 800, 600))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    monkeypatch.setattr(image_grab_mod, "grab", lambda **kw: Image.new("RGB", (800, 600)))
    assert screenshot_mod.grab_thumbnail() is not None
    assert fake.found_titles == [
        "Euro Truck Simulator 2",
        "Euro Truck Simulator 2 Multiplayer",
        "American Truck Simulator",
    ]
    assert fake.found_classes == []


@pytest.mark.parametrize("title", [
    "Euro Truck Simulator 2 Multiplayer",
    "American Truck Simulator Multiplayer",
])
def test_grab_thumbnail_finds_truckersmp_titles(monkeypatch, title):
    Image = pytest.importorskip("PIL.Image")
    image_grab_mod = pytest.importorskip("PIL.ImageGrab")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title=title, rect=(0, 0, 800, 600))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    monkeypatch.setattr(image_grab_mod, "grab", lambda **kw: Image.new("RGB", (800, 600)))
    assert screenshot_mod.grab_thumbnail() is not None
    assert title in fake.found_titles
    assert fake.found_classes == []


def test_grab_thumbnail_falls_back_to_prism3d_class(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image_grab_mod = pytest.importorskip("PIL.ImageGrab")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_class="prism3d", rect=(0, 0, 800, 600))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    monkeypatch.setattr(image_grab_mod, "grab", lambda **kw: Image.new("RGB", (800, 600)))
    assert screenshot_mod.grab_thumbnail() is not None
    assert fake.found_titles == list(screenshot_mod._GAME_WINDOW_TITLES)
    assert fake.found_classes == ["prism3d"]


def test_grab_thumbnail_output_long_side_is_240(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image_grab_mod = pytest.importorskip("PIL.ImageGrab")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="American Truck Simulator", rect=(0, 0, 1920, 1080))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    monkeypatch.setattr(image_grab_mod, "grab", lambda **kw: Image.new("RGB", (1920, 1080)))

    b64 = screenshot_mod.grab_thumbnail()
    assert b64 is not None
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(img.size) == 240


def test_grab_thumbnail_none_on_non_windows(monkeypatch):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "linux")

    def _must_not_be_called():
        raise AssertionError("must not touch user32 on a non-Windows platform")

    monkeypatch.setattr(screenshot_mod, "_get_user32", _must_not_be_called)
    assert screenshot_mod.grab_thumbnail() is None


def test_thumbnail_survives_clip_serialization():
    clip = _make_clip()
    clip.metadata.thumbnail_jpeg = base64.b64encode(b"fake-jpeg").decode("ascii")
    back = deserialize_clip(serialize_clip(clip))
    assert back.metadata.thumbnail_jpeg == clip.metadata.thumbnail_jpeg


def test_recorder_attaches_thumbnail_from_provider():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0,
                          screenshot_provider=lambda: "THUMB64")
    _drive(rec, 0.0, 3.0)
    rec.trigger("manual")
    _drive(rec, 3.0 + 1 / 30, 4.2)
    assert writer.clips
    assert writer.clips[0].metadata.thumbnail_jpeg == "THUMB64"


def test_recorder_without_provider_has_no_thumbnail():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0)
    _drive(rec, 0.0, 3.0)
    rec.trigger("manual")
    _drive(rec, 3.0 + 1 / 30, 4.2)
    assert writer.clips[0].metadata.thumbnail_jpeg is None


def test_recorder_thumbnail_provider_failure_is_swallowed():
    def _boom():
        raise RuntimeError("grab failed")

    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0, screenshot_provider=_boom)
    _drive(rec, 0.0, 3.0)
    rec.trigger("manual")
    _drive(rec, 3.0 + 1 / 30, 4.2)
    assert writer.clips[0].metadata.thumbnail_jpeg is None
