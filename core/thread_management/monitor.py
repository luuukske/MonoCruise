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

_COL = 54   # table width


class Monitor(BaseThread):
    loop_interval = 0.5
    watched       = False

    def __init__(self, watchdog: "Watchdog") -> None:
        super().__init__(name="monitor", daemon=True)
        self._watchdog   = watchdog
        self._input_thread: threading.Thread | None = None
        self._stop_input  = threading.Event()

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
        pass   # heartbeat only; no auto-render

    def teardown(self) -> None:
        self._stop_input.set()

    # rendering (on demand)

    def _print_status(self) -> None:
        header = f"{'NAME':<20} {'ALIVE':<6} {'OK':<4} {'RST':<5} {'HB AGE':>8}"
        sep    = "─" * _COL
        print(sep)
        print(header)
        print(sep)
        for t in registry.all_threads():
            snap = t.snapshot()
            print(
                f"{snap['name']:<20} "
                f"{'yes' if t.is_alive() else 'no':<6} "
                f"{'yes' if snap['healthy'] else 'NO':<4} "
                f"{snap['restart_count']:<5} "
                f"{snap['heartbeat_age']:>7.2f}s"
            )
        print(sep, flush=True)

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