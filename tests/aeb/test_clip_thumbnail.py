"""Tests for the AEB clip screenshot thumbnail path."""

from __future__ import annotations

import base64
import ctypes
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


def _bgrx(size):
    w, h = size
    return b"\x30\x20\x10\x00" * (w * h)


@pytest.fixture
def capture(monkeypatch):
    """Stands in for the GDI copy; records (rect, size) of every call."""
    calls: list[tuple] = []

    def _fake(rect, size):
        calls.append((rect, size))
        return _bgrx(size)

    monkeypatch.setattr(screenshot_mod, "_stretch_capture", _fake)
    return calls


def test_grab_thumbnail_never_falls_back_to_full_grab(monkeypatch, capture):
    image_grab_mod = pytest.importorskip("PIL.ImageGrab")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: _FakeUser32(found_title=None))
    grabs = []
    monkeypatch.setattr(image_grab_mod, "grab", lambda **kw: grabs.append(kw))
    result = screenshot_mod.grab_thumbnail()
    assert result is None
    assert capture == []        # no screen copy without a real window
    assert grabs == []


def test_grab_thumbnail_never_uses_pillow_grab(monkeypatch, capture):
    """Pillow's grab copies the whole monitor holding the GIL: 0.1-2.2 s of every thread stopped."""
    image_grab_mod = pytest.importorskip("PIL.ImageGrab")

    def _gil_holder(**kw):
        raise AssertionError("ImageGrab.grab must not be used")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="Euro Truck Simulator 2", rect=(0, 0, 3840, 2160))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    monkeypatch.setattr(image_grab_mod, "grab", _gil_holder)
    assert screenshot_mod.grab_thumbnail() is not None


def test_grab_thumbnail_copies_exactly_the_window_rect(monkeypatch, capture):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="Euro Truck Simulator 2", rect=(50, 40, 1330, 760))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    result = screenshot_mod.grab_thumbnail()
    assert result is not None
    assert capture == [((50, 40, 1330, 760), (240, 135))]


def test_grab_thumbnail_finds_ats_when_ets2_absent(monkeypatch, capture):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="American Truck Simulator", rect=(0, 0, 800, 600))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
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
def test_grab_thumbnail_finds_truckersmp_titles(monkeypatch, capture, title):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title=title, rect=(0, 0, 800, 600))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    assert screenshot_mod.grab_thumbnail() is not None
    assert title in fake.found_titles
    assert fake.found_classes == []


def test_grab_thumbnail_falls_back_to_prism3d_class(monkeypatch, capture):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_class="prism3d", rect=(0, 0, 800, 600))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    assert screenshot_mod.grab_thumbnail() is not None
    assert fake.found_titles == list(screenshot_mod._GAME_WINDOW_TITLES)
    assert fake.found_classes == ["prism3d"]


class _ScaledUser32(_FakeUser32):
    """175% 4K: virtualized rect unless the thread is per-monitor aware."""

    LOGICAL = (676, 293, 1518, 894)
    PHYSICAL = (1183, 512, 2657, 1564)

    def __init__(self):
        super().__init__(
            found_title="Euro Truck Simulator 2", rect=self.LOGICAL,
        )
        self.ctx = None
        self.rect_ctx = None

    def SetThreadDpiAwarenessContext(self, ctx):
        prev = 1 if self.ctx is None else self.ctx
        self.ctx = ctx
        return prev

    def GetWindowRect(self, hwnd, rect_ptr):
        self.rect_ctx = self.ctx
        vals = self.PHYSICAL if self.ctx == screenshot_mod._DPI_CONTEXT_PER_MONITOR_V2 else self.LOGICAL
        rect_ptr.contents.left, rect_ptr.contents.top, rect_ptr.contents.right, rect_ptr.contents.bottom = vals
        return 1


def test_grab_thumbnail_uses_physical_rect_on_scaled_display(monkeypatch):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _ScaledUser32()
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    calls = []

    def _fake_capture(rect, size):
        calls.append((rect, fake.ctx))
        return _bgrx(size)

    monkeypatch.setattr(screenshot_mod, "_stretch_capture", _fake_capture)
    result = screenshot_mod.grab_thumbnail()
    assert result is not None
    # The copy runs inside the per-monitor context too, not only the rect query.
    assert calls == [(_ScaledUser32.PHYSICAL, screenshot_mod._DPI_CONTEXT_PER_MONITOR_V2)]
    assert fake.rect_ctx == screenshot_mod._DPI_CONTEXT_PER_MONITOR_V2
    assert fake.ctx == 1


def test_dpi_context_restored_when_grab_fails(monkeypatch):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _ScaledUser32()
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)

    def _boom(rect, size):
        raise OSError("grab failed")

    monkeypatch.setattr(screenshot_mod, "_stretch_capture", _boom)
    assert screenshot_mod.grab_thumbnail() is None
    assert fake.ctx == 1


def test_failed_screen_copy_gives_no_thumbnail(monkeypatch):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="Euro Truck Simulator 2", rect=(0, 0, 800, 600))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    monkeypatch.setattr(screenshot_mod, "_stretch_capture", lambda rect, size: None)
    assert screenshot_mod.grab_thumbnail() is None


def test_window_without_area_is_never_copied(monkeypatch, capture):
    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="Euro Truck Simulator 2", rect=(100, 100, 100, 400))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)
    assert screenshot_mod.grab_thumbnail() is None
    assert capture == []


@pytest.mark.parametrize("w, h, want", [
    (3840, 2160, (240, 135)),
    (1920, 1080, (240, 135)),
    (1474, 1052, (240, 171)),
    (1080, 1920, (135, 240)),
    (200, 100, (200, 100)),       # never upscaled
    (0, 600, None),
    (800, -5, None),
])
def test_thumb_size(w, h, want):
    assert screenshot_mod._thumb_size(w, h, 240) == want


class _FakeGdi32:
    """Tracks GDI object state so the capture's call order and cleanup can be checked."""

    def __init__(self):
        self.selected = None
        self.deleted: list[str] = []
        self.stretch = None
        self.dib_while_selected = None
        self.rows = None

    def CreateCompatibleDC(self, dc):
        return "mem"

    def CreateCompatibleBitmap(self, dc, w, h):
        return "bmp"

    def SelectObject(self, dc, obj):
        prev, self.selected = self.selected, obj
        return prev or "stock"

    def SetStretchBltMode(self, dc, mode):
        return 1

    def SetBrushOrgEx(self, dc, x, y, pt):
        return 1

    def StretchBlt(self, dst, dx, dy, dw, dh, src, sx, sy, sw, sh, rop):
        self.stretch = (dst, dx, dy, dw, dh, src, sx, sy, sw, sh, rop)
        return 1

    def GetDIBits(self, dc, bmp, start, lines, buf, info, usage):
        self.dib_while_selected = self.selected == "bmp"
        hdr = info._obj.bmiHeader
        data = b"\x30\x20\x10\x00" * (hdr.biWidth * -hdr.biHeight)
        ctypes.memmove(buf, data, len(data))
        return lines if self.rows is None else self.rows

    def DeleteObject(self, obj):
        self.deleted.append(obj)
        return 1

    def DeleteDC(self, dc):
        self.deleted.append(dc)
        return 1


class _DcUser32:
    def __init__(self):
        self.released = []

    def GetDC(self, hwnd):
        return "screen"

    def ReleaseDC(self, hwnd, dc):
        self.released.append(dc)
        return 1


def test_stretch_capture_scales_the_rect_and_frees_every_gdi_object(monkeypatch):
    gdi, user = _FakeGdi32(), _DcUser32()
    monkeypatch.setattr(screenshot_mod, "_get_gdi32", lambda: gdi)
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: user)

    data = screenshot_mod._stretch_capture((50, 40, 1330, 760), (240, 135))
    assert data == b"\x30\x20\x10\x00" * (240 * 135)
    assert gdi.stretch == (
        "mem", 0, 0, 240, 135, "screen", 50, 40, 1280, 720, screenshot_mod._SRCCOPY)
    assert gdi.dib_while_selected is False      # GetDIBits rejects a selected bitmap
    assert gdi.deleted == ["bmp", "mem"]
    assert user.released == ["screen"]


def test_stretch_capture_short_read_is_none_and_still_cleans_up(monkeypatch):
    gdi, user = _FakeGdi32(), _DcUser32()
    gdi.rows = 0
    monkeypatch.setattr(screenshot_mod, "_get_gdi32", lambda: gdi)
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: user)

    assert screenshot_mod._stretch_capture((0, 0, 800, 600), (240, 180)) is None
    assert gdi.deleted == ["bmp", "mem"]
    assert user.released == ["screen"]


def test_screenshot_does_not_set_process_dpi():
    import pathlib
    src = pathlib.Path(screenshot_mod.__file__).read_text(encoding="utf-8")
    assert "SetProcessDpiAwareness" not in src


def test_grab_thumbnail_output_long_side_is_240(monkeypatch, capture):
    Image = pytest.importorskip("PIL.Image")

    monkeypatch.setattr(screenshot_mod.sys, "platform", "win32")
    fake = _FakeUser32(found_title="American Truck Simulator", rect=(0, 0, 1920, 1080))
    monkeypatch.setattr(screenshot_mod, "_get_user32", lambda: fake)

    b64 = screenshot_mod.grab_thumbnail()
    assert b64 is not None
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.format == "JPEG"
    assert max(img.size) == 240
    r, g, b = img.convert("RGB").getpixel((120, 67))
    assert abs(r - 0x10) <= 6 and abs(g - 0x20) <= 6 and abs(b - 0x30) <= 6   # BGRX read as RGB


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
