"""Best-effort game-window JPEG thumbnail for debug AEB clips (core/aeb/capture.py). Never raises."""

from __future__ import annotations

import base64
import io
import logging
import sys

logger = logging.getLogger(__name__)

# 240x135 keeps in-game text illegible, which a user-facing claim depends on.
# Do not raise this; see core/aeb/README.md section 12.
_MAX_PX = 240
_QUALITY = 50

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


def _dpi_mismatch_note(user32, hwnd) -> None:
    """Debug-only flag for a scaled-display crop risk; detection, not correction."""
    try:
        dpi = user32.GetDpiForWindow(hwnd)
    except (AttributeError, OSError):
        return
    if dpi and dpi != 96:
        logger.debug(
            "AEB screenshot: game window DPI %s, crop is unverified on scaled displays", dpi
        )


def _game_window_rect():
    """Physical-pixel (left, top, right, bottom) of the game window, or None."""
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
    _dpi_mismatch_note(user32, hwnd)
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
        rect = _game_window_rect()
    except Exception:
        logger.debug("AEB screenshot: game window lookup failed", exc_info=True)
        return None
    if rect is None:
        logger.debug("AEB screenshot: no game window found")
        return None
    try:
        return encode_thumbnail(ImageGrab.grab(bbox=rect), max_px, quality)
    except Exception:
        logger.debug("AEB screenshot grab failed", exc_info=True)
        return None


def decode_thumbnail(b64: str) -> bytes:
    """base64 JPEG string -> raw JPEG bytes (for a viewer)."""
    return base64.b64decode(b64)
