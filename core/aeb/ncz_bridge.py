"""Read-only local NCZ state; unknown or stale data leaves AEB enabled."""
from __future__ import annotations

import ctypes
import struct
import sys
import time

MAGIC = 0x4E435A31


def decode_state(raw: bytes, now_ms: int) -> bool | None:
    if len(raw) != 32:
        return None
    sequence, magic, zone, connected, tick = struct.unpack_from("<IIIIQ", raw)
    if sequence & 1 or magic != MAGIC or connected != 1 or zone not in (0, 1):
        return None
    if not 0 <= now_ms - tick <= 500:
        return None
    return zone == 1


class NCZReader:
    def __init__(self):
        self._handle = None
        self._view = None
        self._retry = 0.0
        self._api = None
        if sys.platform == "win32":
            from ctypes import wintypes
            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
            api.OpenFileMappingW.restype = wintypes.HANDLE
            api.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
            api.MapViewOfFile.restype = ctypes.c_void_p
            api.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
            api.CloseHandle.argtypes = [wintypes.HANDLE]
            api.GetTickCount64.restype = ctypes.c_ulonglong
            self._api = api

    def read(self) -> bool | None:
        if self._api is None:
            return None
        if self._view is None:
            now = time.monotonic()
            if now < self._retry:
                return None
            self._retry = now + 1
            self._handle = self._api.OpenFileMappingW(4, False, "Local\\MonoCruise_TMP_NCZ_v1")
            if not self._handle:
                return None
            self._view = self._api.MapViewOfFile(self._handle, 4, 0, 0, 32)
            if not self._view:
                self.close()
                return None
        raw = ctypes.string_at(self._view, 32)
        after = ctypes.string_at(self._view, 4)
        if raw[:4] != after:
            return None
        return decode_state(raw, self._api.GetTickCount64())

    def close(self):
        if self._api is not None:
            if self._view:
                self._api.UnmapViewOfFile(self._view)
            if self._handle:
                self._api.CloseHandle(self._handle)
        self._view = None
        self._handle = None
