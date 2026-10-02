"""Best-effort game-window JPEG thumbnail for debug AEB clips (core/aeb/capture.py). Never raises."""

from __future__ import annotations

import base64
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

_cached_hwnd = None
_user32 = None


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
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    if hasattr(user32, "SetThreadDpiAwarenessContext"):
        user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    _user32 = user32
    return user32


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
    import ctypes
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


def grab_thumbnail(max_px: int = _MAX_PX, quality: int = _QUALITY) -> str | None:
    """Grab the game window only; return a base64 JPEG thumbnail, or None.

    Never falls back to a full-screen grab: a game window on a second
    monitor must not leak whatever is on the primary display.
    """
    try:
        from PIL import ImageGrab
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
            image = ImageGrab.grab(bbox=rect)
        return encode_thumbnail(image, max_px, quality)
    except Exception:
        logger.debug("AEB screenshot grab failed", exc_info=True)
        return None


def decode_thumbnail(b64: str) -> bytes:
    """base64 JPEG string -> raw JPEG bytes (for a viewer)."""
    return base64.b64decode(b64)
