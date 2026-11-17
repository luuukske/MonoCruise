"""Debug-only GUI stall probe: where the GUI thread was and which threads used the CPU meanwhile.

See core/thread_management/README.md, "GUI stall probe".
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.thread_management.watchdog import frozen_stack

logger = logging.getLogger("gui_stall")

# A GUI heartbeat gap this long is a stall the driver sees.
STALL_S = 0.25
# Per-thread CPU summary while nothing stalls, so a drive shows what grows.
REPORT_S = 10.0
# At most one stall line per this long; the rest are counted into the next one.
LOG_MIN_GAP_S = 1.0
_TOP_THREADS = 6
_STACK_KEY_FRAMES = 3


@dataclass
class GuiStallProbeData(ThreadData):
    stalls: int = 0
    worst_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class GuiStallProbe(BaseThread):
    loop_interval = 0.02
    watched = False

    def __init__(self, gui_ident: int | None = None) -> None:
        super().__init__(name="gui_stall_probe", daemon=True)
        self.data = GuiStallProbeData()
        self._gui_ident = gui_ident if gui_ident is not None else threading.main_thread().ident
        self._beat = time.monotonic()
        self._cpu_reader = None
        self._stall_t0: float | None = None
        self._stall_cpu: dict[int, float] = {}
        self._stall_stacks: Counter = Counter()
        self._stall_first: str | None = None
        self._last_log = -LOG_MIN_GAP_S
        self._unlogged = 0
        self._report_at = 0.0
        self._report_cpu: dict[int, float] = {}
        self._report_worst = 0.0

    def beat(self) -> None:
        """GUI thread, from a QTimer: the event loop got to run."""
        self._beat = time.monotonic()

    def setup(self) -> None:
        self._cpu_reader = _thread_cpu
        self._report_cpu = self._cpu()
        self._report_at = time.monotonic() + REPORT_S
        logger.info("GUI stall probe on: logs GUI gaps of %d ms or more", round(STALL_S * 1000))

    def loop(self) -> None:
        if not self.running:
            return
        now = time.monotonic()
        gap = now - self._beat
        self._report_worst = max(self._report_worst, gap)
        if gap >= STALL_S:
            self._sample_stall()
        elif self._stall_t0 is not None:
            self._end_stall(now)
        if now >= self._report_at:
            self._report(now)

    def teardown(self) -> None:
        logger.debug("GUI stall probe off: %d stalls, worst %.0f ms", self.data.stalls, self.data.worst_ms)

    def _sample_stall(self) -> None:
        if self._stall_t0 is None:
            self._stall_t0 = self._beat
            self._stall_cpu = self._cpu()
            self._stall_stacks = Counter()
            self._stall_first = None
        stack = frozen_stack(self._gui_ident)
        if stack:
            self._stall_stacks[" | ".join(stack.split(" | ")[-_STACK_KEY_FRAMES:])] += 1
            if self._stall_first is None:
                self._stall_first = stack

    def _end_stall(self, now: float) -> None:
        span = now - self._stall_t0
        cpu = self._shares(self._stall_cpu, self._cpu(), span)
        stacks = self._stall_stacks.most_common(2)
        first = self._stall_first
        self._stall_t0 = None
        with self.data._lock:
            self.data.stalls += 1
            self.data.worst_ms = max(self.data.worst_ms, span * 1000.0)
        if now - self._last_log < LOG_MIN_GAP_S:
            self._unlogged += 1
            return
        where = "; ".join(f"{n}x {s}" for s, n in stacks) or "no Python frame"
        more = f" (+{self._unlogged} shorter-spaced stalls not logged)" if self._unlogged else ""
        logger.warning("GUI stalled %.0f ms%s. GUI thread at: %s. %s", span * 1000.0, more, where, cpu)
        if first:
            logger.debug("GUI stall first sample, full stack: %s", first)
        self._last_log = now
        self._unlogged = 0

    def _report(self, now: float) -> None:
        cpu = self._cpu()
        logger.debug(
            "GUI longest gap %.0f ms in the last %.0f s. %s",
            self._report_worst * 1000.0, REPORT_S, self._shares(self._report_cpu, cpu, REPORT_S),
        )
        self._report_cpu = cpu
        self._report_worst = 0.0
        self._report_at = now + REPORT_S

    def _cpu(self) -> dict[int, float]:
        """CPU seconds so far per native thread id of the Python threads; empty when unreadable."""
        if self._cpu_reader is None:
            return {}
        try:
            return self._cpu_reader()
        except Exception:
            logger.debug("GUI stall probe: thread CPU read failed", exc_info=True)
            return {}

    def _shares(self, before: dict[int, float], after: dict[int, float], span_s: float) -> str:
        """Busiest threads between two CPU reads, by name, with their share of the span."""
        if not after:
            return "thread CPU unknown"
        names = _thread_names()
        used: Counter = Counter()
        for tid, cpu in after.items():
            delta = cpu - before.get(tid, 0.0)
            if delta > 0.0:
                used[names.get(tid, "other")] += delta
        span = max(span_s, 1e-3)
        top = ", ".join(f"{name} {sec * 1000:.0f} ms ({100 * sec / span:.0f}%)" for name, sec in used.most_common(_TOP_THREADS))
        return f"CPU {sum(used.values()) * 1000:.0f} ms over {span * 1000:.0f} ms: {top or 'idle'}"


def _thread_cpu() -> dict[int, float]:
    """CPU seconds per native id of each Python thread. Never psutil: its thread list walks
    every process on the machine, 116 ms a call on Windows, and would deepen the stall it measures."""
    out: dict[int, float] = {}
    for t in threading.enumerate():
        if t.native_id is None:
            continue
        sec = _one_thread_cpu(t)
        if sec is not None:
            out[t.native_id] = sec
    return out


if sys.platform == "win32":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenThread.restype = ctypes.c_void_p
    _k32.OpenThread.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    _k32.GetThreadTimes.restype = ctypes.c_int
    _k32.GetThreadTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    _k32.CloseHandle.argtypes = [ctypes.c_void_p]
    _THREAD_QUERY_LIMITED_INFORMATION = 0x0800

    def _one_thread_cpu(t: threading.Thread) -> float | None:
        handle = _k32.OpenThread(_THREAD_QUERY_LIMITED_INFORMATION, 0, t.native_id)
        if not handle:
            return None
        try:
            created, exited, kernel, user = (ctypes.c_uint64() for _ in range(4))
            if not _k32.GetThreadTimes(handle, created, exited, kernel, user):
                return None
            return (kernel.value + user.value) * 1e-7
        finally:
            _k32.CloseHandle(handle)
else:
    def _one_thread_cpu(t: threading.Thread) -> float | None:
        try:
            return time.clock_gettime(time.pthread_getcpuclockid(t.ident))
        except (AttributeError, OSError, TypeError, ValueError):
            return None


def _thread_names() -> dict[int, str]:
    """Native thread id to a readable name: the registry name for workers, ``gui`` for the main thread."""
    by_ident: dict[int, str] = {}
    try:
        for t in registry.all_threads():
            if t.ident is not None:
                by_ident[t.ident] = t.name
    except Exception:
        logger.debug("GUI stall probe: registry unreadable", exc_info=True)
    main = threading.main_thread()
    out: dict[int, str] = {}
    for t in threading.enumerate():
        if t.native_id is None:
            continue
        out[t.native_id] = "gui" if t is main else by_ident.get(t.ident, t.name)
    return out
