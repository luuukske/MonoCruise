"""Best-effort game-window JPEG thumbnail for debug AEB clips (core/aeb/capture.py). Never raises."""

from __future__ import annotations

import base64
import ctypes
import io
import logging
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 240x135 keeps in-game text illegible, which a user-facing claim depends on.
# Do not raise this; see core/aeb/README.md section 12.
_MAX_PX = 240
_QUALITY = 50
# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2. Thread-local only, Win10 1703+.
_DPI_CONTEXT_PER_MONITOR_V2 = -4

# TruckersMP appends " Multiplayer"; FindWindowW is an exact-title match.
_GAME_WINDOW_TITLES = (
    "Euro Truck Simulator 2",
    "Euro Truck Simulator 2 Multiplayer",
    "American Truck Simulator",
    "American Truck Simulator Multiplayer",
)
# SCS Prism3D class, last resort if a future title variant appears.
_GAME_WINDOW_CLASS = "prism3d"

_SRCCOPY = 0x00CC0020
_HALFTONE = 4
_BI_RGB = 0
_DIB_RGB_COLORS = 0

_cached_hwnd = None
_user32 = None
_gdi32 = None


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32), ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32), ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [("bmiHeader", _BitmapInfoHeader), ("bmiColors", ctypes.c_uint32 * 3)]


def encode_thumbnail(image, max_px: int = _MAX_PX, quality: int = _QUALITY) -> str:
    """PIL Image -> base64 JPEG, downscaled so its long side is <= max_px."""
    img = image.convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _get_user32():
    """Bind user32 with HWND-sized argtypes so a 64-bit handle can't truncate."""
    global _user32
    if _user32 is not None:
        return _user32
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetDC.restype = ctypes.c_void_p
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.ReleaseDC.argtypes = [wintypes.HWND, ctypes.c_void_p]
    if hasattr(user32, "SetThreadDpiAwarenessContext"):
        user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    _user32 = user32
    return user32


def _get_gdi32():
    """Bind the gdi32 calls the capture uses, handles as pointers."""
    global _gdi32
    if _gdi32 is not None:
        return _gdi32
    vp, i = ctypes.c_void_p, ctypes.c_int
    gdi32 = ctypes.windll.gdi32
    for name, restype, argtypes in (
        ("CreateCompatibleDC", vp, [vp]),
        ("CreateCompatibleBitmap", vp, [vp, i, i]),
        ("SelectObject", vp, [vp, vp]),
        ("DeleteObject", i, [vp]),
        ("DeleteDC", i, [vp]),
        ("SetStretchBltMode", i, [vp, i]),
        ("SetBrushOrgEx", i, [vp, i, i, vp]),
        ("StretchBlt", i, [vp, i, i, i, i, vp, i, i, i, i, ctypes.c_uint32]),
        ("GetDIBits", i, [vp, vp, ctypes.c_uint, ctypes.c_uint, vp,
                          ctypes.POINTER(_BitmapInfo), ctypes.c_uint]),
    ):
        fn = getattr(gdi32, name)
        fn.restype = restype
        fn.argtypes = argtypes
    _gdi32 = gdi32
    return gdi32


def _find_game_window():
    """Cached game window handle, revalidated with IsWindow since the game can restart."""
    global _cached_hwnd
    user32 = _get_user32()
    if _cached_hwnd is not None and user32.IsWindow(_cached_hwnd):
        return _cached_hwnd
    _cached_hwnd = None
    for title in _GAME_WINDOW_TITLES:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            _cached_hwnd = hwnd
            return _cached_hwnd
    hwnd = user32.FindWindowW(_GAME_WINDOW_CLASS, None)
    if hwnd:
        _cached_hwnd = hwnd
    return _cached_hwnd


@contextmanager
def _physical_pixels():
    """Per-monitor DPI on this thread so the window rect matches the grab."""
    if sys.platform != "win32":
        yield
        return
    user32 = _get_user32()
    set_ctx = getattr(user32, "SetThreadDpiAwarenessContext", None)
    if set_ctx is None:
        yield
        return
    prev = set_ctx(_DPI_CONTEXT_PER_MONITOR_V2)
    if not prev:
        logger.debug("AEB screenshot: thread DPI context unchanged, scaled crop may be wrong")
        yield
        return
    try:
        yield
    finally:
        set_ctx(prev)


def _game_window_rect():
    """(left, top, right, bottom) of the game window, or None. Physical only inside _physical_pixels."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    user32 = _get_user32()
    hwnd = _find_game_window()
    if not hwnd:
        return None
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.pointer(rect)):
        logger.debug("AEB screenshot: GetWindowRect failed")
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def _thumb_size(width: int, height: int, max_px: int) -> tuple[int, int] | None:
    """Aspect-kept size whose long side is at most max_px; never upscales."""
    if width <= 0 or height <= 0:
        return None
    scale = min(1.0, max_px / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _stretch_capture(rect, size: tuple[int, int]) -> bytes | None:
    """Screen pixels inside rect, scaled by GDI to size, as top-down BGRX rows.

    Plain ctypes calls drop the GIL; Pillow's grab held it for a whole-monitor copy (README section 12).
    """
    user32 = _get_user32()
    gdi32 = _get_gdi32()
    left, top, right, bottom = rect
    w, h = size
    screen = user32.GetDC(None)
    if not screen:
        return None
    mem = bmp = None
    try:
        mem = gdi32.CreateCompatibleDC(screen)
        bmp = gdi32.CreateCompatibleBitmap(screen, w, h)
        if not mem or not bmp:
            return None
        old = gdi32.SelectObject(mem, bmp)
        gdi32.SetStretchBltMode(mem, _HALFTONE)
        gdi32.SetBrushOrgEx(mem, 0, 0, None)
        copied = gdi32.StretchBlt(
            mem, 0, 0, w, h, screen, left, top, right - left, bottom - top, _SRCCOPY,
        )
        # GetDIBits needs the bitmap deselected first.
        gdi32.SelectObject(mem, old)
        if not copied:
            return None
        info = _BitmapInfo()
        hdr = info.bmiHeader
        hdr.biSize = ctypes.sizeof(_BitmapInfoHeader)
        hdr.biWidth, hdr.biHeight = w, -h
        hdr.biPlanes, hdr.biBitCount, hdr.biCompression = 1, 32, _BI_RGB
        buf = ctypes.create_string_buffer(w * h * 4)
        if gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), _DIB_RGB_COLORS) != h:
            return None
        return buf.raw
    finally:
        if bmp:
            gdi32.DeleteObject(bmp)
        if mem:
            gdi32.DeleteDC(mem)
        user32.ReleaseDC(None, screen)


def grab_thumbnail(max_px: int = _MAX_PX, quality: int = _QUALITY) -> str | None:
    """Grab the game window only; return a base64 JPEG thumbnail, or None.

    Never falls back to a full-screen grab: a game window on a second
    monitor must not leak whatever is on the primary display.
    """
    try:
        from PIL import Image
    except Exception:
        logger.debug("Pillow unavailable; AEB screenshot skipped")
        return None
    try:
        with _physical_pixels():
            try:
                rect = _game_window_rect()
            except Exception:
                logger.debug("AEB screenshot: game window lookup failed", exc_info=True)
                return None
            if rect is None:
                logger.debug("AEB screenshot: no game window found")
                return None
            size = _thumb_size(rect[2] - rect[0], rect[3] - rect[1], max_px)
            if size is None:
                logger.debug("AEB screenshot: game window has no area")
                return None
            pixels = _stretch_capture(rect, size)
        if pixels is None:
            logger.debug("AEB screenshot: screen copy failed")
            return None
        image = Image.frombytes("RGB", size, pixels, "raw", "BGRX")
        return encode_thumbnail(image, max_px, quality)
    except Exception:
        logger.debug("AEB screenshot grab failed", exc_info=True)
        return None


def decode_thumbnail(b64: str) -> bytes:
    """base64 JPEG string -> raw JPEG bytes (for a viewer)."""
    return base64.b64decode(b64)
