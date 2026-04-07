"""
Monitor — interactive CLI control panel.

Only active when settings.debug is True.
The table is never auto-printed; call `status` whenever you want a snapshot.
Logging output is never interrupted or overwritten.

Commands:
  status            — print thread table once
  stop <name>       — stop a thread
  restart <name>    — stop + restart a thread via watchdog factory
  quit              — stop all threads and exit
  help              — show this list
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import TYPE_CHECKING

from core.thread_management.base_thread import BaseThread
from core.thread_management.registry    import registry

if TYPE_CHECKING:
    from core.thread_management.watchdog import Watchdog

logger = logging.getLogger("monitor")

_COL = 56   # table width


class Monitor(BaseThread):
    loop_interval = 0.5
    watched       = False

    def __init__(self, watchdog: "Watchdog") -> None:
        super().__init__(name="monitor", daemon=True)
        self._watchdog   = watchdog
        self._input_thread: threading.Thread | None = None
        self._stop_input  = threading.Event()
        self._mapper_live = False
        self._last_mapper_print = 0.0

    # lifecycle

    def setup(self) -> None:
        self._input_thread = threading.Thread(
            target=self._input_loop,
            name="monitor.input",
            daemon=True,
        )
        self._input_thread.start()
        print("[monitor] debug shell active — type 'help' for commands", flush=True)

    def loop(self) -> None:
        if self._mapper_live:
            self._print_mapper_status(throttle_s=0.25)

    def teardown(self) -> None:
        self._stop_input.set()

    # rendering (on demand)

    def _print_status(self) -> None:
        header = f"{'NAME':<20} {'ALIVE':<6} {'OK':<4} {'RST':<5} {'AVG FPS':>8}"
        sep    = "─" * _COL
        print(sep)
        print(header)
        print(sep)
        for t in registry.all_threads():
            snap = t.snapshot()
            fps = f"{snap['avg_framerate']:>8.2f}" if snap["avg_framerate"] > 0 else f"{'n/a':>8}"
            print(
                f"{snap['name']:<20} "
                f"{'yes' if t.is_alive() else 'no':<6} "
                f"{'yes' if snap['healthy'] else 'NO':<4} "
                f"{snap['restart_count']:<5} "
                f"{fps}"
            )
        print(sep, flush=True)

    def _print_mapper_status(self, throttle_s: float = 0.0) -> None:
        import time

        now = time.monotonic()
        if throttle_s > 0.0 and now - self._last_mapper_print < throttle_s:
            return

        try:
            sending = registry.get_thread("sending_thread")
            with sending.data._lock:
                line = (
                    "[mapper] "
                    f"cc={'on' if sending.data.mapper_cruise_active else 'off'} "
                    f"lim={'yes' if sending.data.mapper_accel_limited else 'no'} "
                    f"cmd={sending.data.mapper_commanded_ms2:+.2f} "
                    f"ctrl={sending.data.mapper_control_wanted_ms2:+.2f} "
                    f"raw={sending.data.mapper_raw_accel_ms2:+.2f} "
                    f"measCtrl={sending.data.mapper_measured_control_ms2:+.2f} "
                    f"slope={sending.data.mapper_slope_input_rad:+.3f} "
                    f"effSlope={sending.data.mapper_effective_slope_rad:+.3f} "
                    f"ws={sending.data.mapper_wanted_smooth_ms2:+.2f} "
                    f"rs={sending.data.mapper_raw_smooth_ms2:+.2f} "
                    f"load={sending.data.mapper_road_load_ms2:+.2f} "
                    f"i={sending.data.mapper_integral:+.3f} "
                    f"estA={sending.data.mapper_est_max_accel_ms2:.2f} "
                    f"estB={sending.data.mapper_est_max_brake_ms2:.2f} "
                    f"gas={sending.data.mapper_command_gas:.3f} "
                    f"brk={sending.data.mapper_command_brake:.3f} "
                    f"measB={sending.data.decel_measured_ms2:.2f}"
                )
        except (KeyError, AttributeError):
            line = "[mapper] sending_thread unavailable"

        print(line, flush=True)
        self._last_mapper_print = now

    # CLI input

    def _input_loop(self) -> None:
        if os.name == "nt":
            self._input_loop_windows()
        else:
            self._input_loop_posix()

    def _input_loop_posix(self) -> None:
        import select

        buf = ""
        while not self._stop_input.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if not ch:          # EOF
                break
            if ch == "\n":
                self._handle_command(buf)
                buf = ""
            else:
                buf += ch

    def _input_loop_windows(self) -> None:
        import msvcrt

        buf = ""
        while not self._stop_input.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    self._handle_command(buf)
                    buf = ""
                elif ch in ("\x08", "\x7f"):  # backspace (BS or DEL)
                    if buf:
                        buf = buf[:-1]
                        sys.stdout.write("\x08 \x08")  # erase last char on screen
                        sys.stdout.flush()
                else:
                    buf += ch
                    sys.stdout.write(ch)
                    sys.stdout.flush()
            else:
                self._stop_input.wait(0.05)

    # command dispatch

    def _handle_command(self, raw: str) -> None:
        parts = raw.strip().split()
        if not parts:
            return
        cmd  = parts[0].lower()
        args = parts[1:]

        match cmd:
            case "help":
                print(__doc__, flush=True)
            case "status":
                self._print_status()
            case "mapper":
                if args and args[0].lower() == "off":
                    self._mapper_live = False
                    print("[monitor] mapper live view disabled", flush=True)
                else:
                    self._mapper_live = not self._mapper_live
                    print(
                        f"[monitor] mapper live view {'enabled' if self._mapper_live else 'disabled'}",
                        flush=True,
                    )
                    if self._mapper_live:
                        self._print_mapper_status()
            case "stop" if args:
                self._cmd_stop(args[0])
            case "restart" if args:
                self._cmd_restart(args[0])
            case "quit":
                logger.info("quit requested via monitor")
                for t in registry.all_threads():
                    t.stop()
            case _:
                logger.error(f"unknown command: {raw!r} — type 'help'")

    def _cmd_stop(self, name: str) -> None:
        try:    
            registry.get_thread(name).stop()
            logger.info(f"stopping '{name}'")
        except KeyError:
            logger.error(f"unknown thread '{name}'")

    def _cmd_restart(self, name: str) -> None:
        try:
            t = registry.get_thread(name)
        except KeyError:
            logger.error(f"unknown thread '{name}'")
            return

        factory = self._watchdog._factories.get(name)
        if factory is None:
            logger.error(f"no factory for '{name}'")
            return

        t.stop()
        t.join(timeout=2.0)
        new_t               = factory()
        new_t.restart_count = t.restart_count   # manual restart doesn't count
        new_t.name          = name
        registry.replace(new_t)
        new_t.start()
        logger.info(f"restarted '{name}'")
