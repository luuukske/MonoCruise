"""Put an already-shown tool window back in the topmost band.

The NVIDIA overlay plus alt-tab can clear HWND_TOPMOST while Qt still reports
the widget visible, so the usual ``show()`` path never runs again. ``raise_()``
stays banned on the animation timers: per frame it fights the other overlay and
can freeze Qt on Windows while the main window is minimised.

``SetWindowPos`` here does not activate the window and does not move or resize it.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

OVERLAY_KEEP_MS = 5000

_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOACTIVATE = 0x0010
_SWP_SHOWWINDOW = 0x0040
_SWP_NOOWNERZORDER = 0x0200
_HWND_TOPMOST = -1
_SWP_FLAGS = (
    _SWP_NOSIZE
    | _SWP_NOMOVE
    | _SWP_NOACTIVATE
    | _SWP_SHOWWINDOW
    | _SWP_NOOWNERZORDER
)

_user32 = None


def _load_user32():
    global _user32
    if _user32 is None:
        _user32 = ctypes.WinDLL("user32", use_last_error=True)
        _user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        _user32.SetWindowPos.restype = wintypes.BOOL
    return _user32


def reassert_topmost(widget) -> bool:
    """GUI thread. No-op when the widget is hidden or this is not Windows."""
    if widget is None or not widget.isVisible():
        return False
    if sys.platform != "win32":
        return False
    hwnd = int(widget.winId())
    if hwnd == 0:
        return False
    user32 = _load_user32()
    return bool(user32.SetWindowPos(hwnd, _HWND_TOPMOST, 0, 0, 0, 0, _SWP_FLAGS))
