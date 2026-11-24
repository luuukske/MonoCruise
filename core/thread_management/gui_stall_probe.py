"""Debug-only GUI stall probe: where the GUI thread was, which threads used the CPU, how long
their loops took and where the truck was. See core/thread_management/README.md, "GUI stall probe".
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

from core.thread_management.base_thread import BaseThread, LoopTimes, ThreadData
from core.thread_management.registry import registry
from core.thread_management.watchdog import frozen_stack

logger = logging.getLogger("gui_stall")

# A GUI heartbeat gap this long is a stall the driver sees.
STALL_S = 0.25
# Per-thread CPU and loop summary while nothing stalls, so a drive shows what grows.
REPORT_S = 10.0
# At most one stall line per this long; the rest are counted into the next one.
LOG_MIN_GAP_S = 1.0
# Where the truck is and how much traffic and map it carries, read this often.
CONTEXT_S = 1.0
_TOP_THREADS = 6
_STACK_KEY_FRAMES = 3


@dataclass
class GuiStallProbeData(ThreadData):
    stalls: int = 0
    worst_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


@dataclass
class Context:
    """Ego's last place and the most traffic and map seen since the last report."""

    x: float | None = None
    z: float | None = None
    speed_kmh: float = 0.0
    tmp: bool = False
    paused: bool = False
    vehicles: int = 0
    trailers: int = 0
    map_status: str = "unknown"
    lanes: int = 0
    routed: int = 0

    def merge(self, new: Context) -> None:
        if new.x is not None:
            self.x, self.z = new.x, new.z
            self.speed_kmh, self.tmp, self.paused = new.speed_kmh, new.tmp, new.paused
        if new.map_status != "unknown":
            self.map_status = new.map_status
        self.vehicles = max(self.vehicles, new.vehicles)
        self.trailers = max(self.trailers, new.trailers)
        self.lanes = max(self.lanes, new.lanes)
        self.routed = max(self.routed, new.routed)

    def describe(self) -> str:
        if self.x is None:
            parts = ["position unknown"]
        else:
            parts = [f"x {round(self.x, -1):.0f} z {round(self.z, -1):.0f}", f"{self.speed_kmh:.0f} km/h", "TMP" if self.tmp else "SP"]
            if self.paused:
                parts.append("paused")
        parts.append(f"{self.vehicles} vehicles + {self.trailers} trailers")
        parts.append(f"map {self.map_status} {self.lanes} lanes {self.routed} routed")
        return ", ".join(parts)


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
        self._stall_cpu_at = 0.0
        self._stall_stacks: Counter = Counter()
        self._stall_first: str | None = None
        self._last_log = -LOG_MIN_GAP_S
        self._unlogged = 0
        self._report_at = 0.0
        self._report_t0 = 0.0
        self._report_cpu: dict[int, float] = {}
        self._report_worst = 0.0
        self._context = Context()
        self._context_at = 0.0
        self._context_failed: set[str] = set()
        self._timer_logged = False

    def beat(self) -> None:
        """GUI thread, from a QTimer: the event loop got to run."""
        self._beat = time.monotonic()

    def setup(self) -> None:
        self._cpu_reader = _thread_cpu
        self._report_cpu = self._cpu()
        self._report_t0 = time.monotonic()
        self._report_at = self._report_t0 + REPORT_S
        LoopTimes.on = True
        logger.info("GUI stall probe on: logs GUI gaps of %d ms or more", round(STALL_S * 1000))

    def loop(self) -> None:
        if not self.running:
            return
        now = time.monotonic()
        gap = now - self._beat
        self._report_worst = max(self._report_worst, gap)
        if now >= self._context_at:
            self._sample_context(now)
        if gap >= STALL_S:
            self._sample_stall()
        elif self._stall_t0 is not None:
            self._end_stall(now)
        if now >= self._report_at:
            self._report(now)

    def teardown(self) -> None:
        LoopTimes.on = False
        logger.debug("GUI stall probe off: %d stalls, worst %.0f ms", self.data.stalls, self.data.worst_ms)

    def _sample_context(self, now: float) -> None:
        self._context.merge(self._read_context())
        self._context_at = now + CONTEXT_S

    def _read_context(self) -> Context:
        """Ego pose and traffic from radar, lanes from the map worker; each part unknown when unreadable."""
        ctx = Context()
        try:
            radar = registry.get_thread("radar_thread").data
            with radar._lock:
                if radar.t_mono > 0.0:
                    ctx.x, ctx.z = float(radar.ego_x), float(radar.ego_z)
                    ctx.speed_kmh = float(radar.ego_speed) * 3.6
                    ctx.tmp, ctx.paused = bool(radar.tmp_session), bool(radar.paused)
                ctx.vehicles = len(radar.vehicles)
                ctx.trailers = len(radar.trailer_vehicles)
        except Exception:
            self._note_context_failure("radar_thread")
        try:
            map_data = registry.get_thread("map_thread").data
            with map_data._lock:
                snap = map_data.snapshot
            ctx.map_status = str(snap.status)
            ctx.lanes = len(snap.window.lines) if snap.window is not None else 0
            ctx.routed = len(snap.target_routes)
        except Exception:
            self._note_context_failure("map_thread")
        return ctx

    def _note_context_failure(self, name: str) -> None:
        if name not in self._context_failed:
            self._context_failed.add(name)
            logger.debug("GUI stall probe: %s unreadable, its context stays unknown", name, exc_info=True)

    def _sample_stall(self) -> None:
        if self._stall_t0 is None:
            self._stall_t0 = self._beat
            self._stall_cpu, self._stall_cpu_at = self._timed_cpu()
            self._stall_stacks = Counter()
            self._stall_first = None
        stack = frozen_stack(self._gui_ident)
        if stack:
            self._stall_stacks[" | ".join(stack.split(" | ")[-_STACK_KEY_FRAMES:])] += 1
            if self._stall_first is None:
                self._stall_first = stack

    def _end_stall(self, now: float) -> None:
        span = now - self._stall_t0
        # The first CPU read comes only once the gap is seen, so the share covers the stall from there.
        after, after_at = self._timed_cpu()
        cpu = self._shares(self._stall_cpu, after, after_at - self._stall_cpu_at)
        seen_ms = (self._stall_cpu_at - self._stall_t0) * 1000.0
        stacks = self._stall_stacks.most_common(2)
        first = self._stall_first
        self._stall_t0 = None
        with self.data._lock:
            self.data.stalls += 1
            self.data.worst_ms = max(self.data.worst_ms, span * 1000.0)
        if now - self._last_log < LOG_MIN_GAP_S:
            self._unlogged += 1
            return
        self._sample_context(now)
        where = "; ".join(f"{n}x {s}" for s, n in stacks) or "no Python frame"
        more = f" (+{self._unlogged} shorter-spaced stalls not logged)" if self._unlogged else ""
        logger.warning(
            "GUI stalled %.0f ms%s. GUI thread at: %s. From %.0f ms in: %s. At %s",
            span * 1000.0, more, where, seen_ms, cpu, self._context.describe(),
        )
        if first:
            logger.debug("GUI stall first sample, full stack: %s", first)
        self._last_log = now
        self._unlogged = 0

    def _report(self, now: float) -> None:
        cpu = self._cpu()
        span = max(now - self._report_t0, 1e-3)
        self._sample_context(now)
        logger.debug(
            "GUI longest gap %.0f ms in the last %.0f s. %s. At %s",
            self._report_worst * 1000.0, REPORT_S, self._shares(self._report_cpu, cpu, REPORT_S),
            self._context.describe(),
        )
        logger.debug("Loops in the last %.0f s: %s", REPORT_S, self._loops(span, self._report_cpu, cpu))
        if not self._timer_logged:
            self._timer_logged = True
            logger.info(
                "GIL switch interval %.1f ms; a 1 ms timed wait lasts %.1f ms",
                sys.getswitchinterval() * 1000.0, timed_wait_ms(),
            )
        self._report_cpu = cpu
        self._report_worst = 0.0
        self._report_t0 = now
        self._report_at = now + REPORT_S
        self._context = Context()

    def _loops(self, span_s: float, before: dict[int, float], after: dict[int, float]) -> str:
        """Each worker's loop rate against its target and its loop wall time against its budget,
        worst p99 share of the budget first. Wall time beyond the loop's own CPU is waiting, mostly for the GIL."""
        try:
            threads = registry.all_threads()
        except Exception:
            logger.debug("GUI stall probe: registry unreadable", exc_info=True)
            return "threads unknown"
        natives = {t.ident: t.native_id for t in threading.enumerate()}
        rows: list[tuple[float, str]] = []
        for t in threads:
            times_of = getattr(t, "loop_times", None)
            times = times_of.take() if times_of is not None else []
            budget = float(getattr(t, "loop_interval", 0.0) or 0.0)
            if not times or budget <= 0.0:
                continue
            times.sort()
            n = len(times)
            p50, p99, worst = times[n // 2], times[min(n - 1, int(0.99 * n))], times[-1]
            row = (
                f"{t.name} {n / span_s:.0f}/{1.0 / budget:.0f} Hz, "
                f"p50 {p50 * 1000:.1f} p99 {p99 * 1000:.1f} max {worst * 1000:.0f} ms of {budget * 1000:.0f}"
            )
            native = natives.get(getattr(t, "ident", None))
            if native is not None and native in after:
                row += f", cpu {1000.0 * (after[native] - before.get(native, 0.0)) / n:.1f} ms"
            rows.append((p99 / budget, row))
        rows.sort(key=lambda r: r[0], reverse=True)
        return "; ".join(row for _, row in rows) or "none"

    def _cpu(self) -> dict[int, float]:
        """CPU seconds so far per native thread id of the Python threads; empty when unreadable."""
        if self._cpu_reader is None:
            return {}
        try:
            return self._cpu_reader()
        except Exception:
            logger.debug("GUI stall probe: thread CPU read failed", exc_info=True)
            return {}

    def _timed_cpu(self) -> tuple[dict[int, float], float]:
        """Thread CPU and the middle of the read: under GIL contention one read takes 100 ms or more."""
        t0 = time.monotonic()
        cpu = self._cpu()
        return cpu, (t0 + time.monotonic()) / 2.0

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


def timed_wait_ms(samples: int = 5) -> float:
    """Median length of a 1 ms timed lock wait. Windows rounds it up to the process timer tick
    (15.6 ms unless something raised it), and the GIL's own wait rounds the same way."""
    lock = threading.Lock()
    lock.acquire()
    waits = []
    for _ in range(samples):
        t0 = time.perf_counter()
        lock.acquire(timeout=0.001)
        waits.append((time.perf_counter() - t0) * 1000.0)
    waits.sort()
    return waits[len(waits) // 2]


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
