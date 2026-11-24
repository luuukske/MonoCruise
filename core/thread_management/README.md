# Thread management and app wiring

> Every long-running feature is a `BaseThread` subclass supervised by the watchdog.
> Agent rules (loop timing, sibling failures, popup discipline) are in root `AGENTS.md`.

## Entry point (`monocruise.py`)

Repo root, not `main.py`. Loads settings, configures logging (file log plus the popup
handler), instantiates and registers every worker thread, registers their restart
factories with the watchdog, starts the monitor in debug mode, then runs the Qt event
loop and coordinates shutdown.

## Thread model

- `base_thread.py`: `BaseThread` owns the lifecycle (`setup`, `loop`, `teardown`),
  heartbeat, loop pacing, FPS tracking, restart counting, and force-stop. Backends are
  `threading.Thread` or `QThread`, chosen per thread.
- `registry.py`: named singleton registry. Threads register themselves; `replace()` is
  the atomic swap the watchdog and monitor use. `get_thread()` raises `KeyError`, and
  every caller is expected to catch it and carry on.
- Threads publish state through a typed `ThreadData` subclass so siblings read a
  consistent snapshot under its lock rather than touching internals.

## Watchdog (`watchdog.py`)

Polls every registered, watched, healthy thread:

- `running == False` and `is_alive() is False` means crashed: restart immediately.
- Heartbeat older than `HEARTBEAT_TIMEOUT` for `FREEZE_STREAK_RESTART` consecutive polls
  means frozen. The streak is a debounce; a single stale poll is a hiccup.
- On a freeze, before restarting (or giving up), it logs where the thread is stuck:
  the innermost `STACK_FRAMES` frames from `sys._current_frames()`, one warning line,
  never a popup. A freeze leaves no trace of its own; the 2026-10-12 sending-thread
  freezes took a log-and-replay hunt to pin on a 382 MB file read that this line would
  have named. Paths are repo-relative, or their last two parts outside the repo, so
  the log stays shareable.
- At `restart_count >= max_restarts` the thread is marked unhealthy instead of restarted.
- `telemetry_thread` with `data.request_quit` is never restarted: shutdown is the entry
  point's business.
- `safe_state()` runs on a throwaway thread (it may block on I/O) the moment a freeze or
  crash is confirmed, before any log line and also at the restart ceiling: on 2026-11-20
  the watchdog's own log lines waited 2-4 s on a stalled disk with the gas held.
- Restart order: brief
  `stop(force=True)` plus `join(STOP_GRACE)`, then build the replacement from the
  registered factory, carry `restart_count + 1` over, and `registry.replace()` it.
- Separately, threads delivering under `LOW_FPS_THRESHOLD` of their target for
  `LOW_FPS_STREAK_WARN` polls are collected for `LOW_FPS_WARN_WINDOW` and warned about
  once, batched, so a system-wide stall produces one message and not one per thread.

Behaviour is pinned by `tests/thread_management/`.

## Logging (`log_queue.py`)

The file and console handlers run on one `log_writer` daemon thread. Every logger call
only formats the message and puts the record on a bounded queue (50 000); a full queue
drops the record, it never waits. A `FileHandler` flushes under its lock on every record,
so before this a stalled disk held every thread that logged, the GUI for 3.5 s on
2026-11-20. The popup handler stays on the calling thread (it only emits a Qt signal).
`QueuedLogging.stop` drains at exit with a timeout.

## Monitor (`monitor.py`)

Interactive CLI, only when `Settings.debug` is true: `status`, `stop <name>`,
`restart <name>`, `quit`.

## GUI stall probe (`gui_stall_probe.py`)

Debug only, like the monitor. The watchdog watches workers, never the Qt GUI thread, so a
frozen cruise panel or pedal bar left no trace. A 20 ms `QTimer` on the GUI thread beats;
`GuiStallProbe` (a `BaseThread`, not watched) logs one warning per gap of 250 ms or more,
once the GUI beats again: how long, where the GUI thread was (its innermost frames, sampled
every 20 ms during the gap, repo-relative paths) and which Python threads used the CPU
meanwhile. Every 10 s a debug line gives the longest gap and the busiest threads, so a drive
shows what grows. Stall lines closer than 1 s are counted into the next one.

Where and who: the 10 s line and every stall line end with where the truck was (`At x .. z ..`,
10 m, speed, SP or TMP, paused) and the most vehicles, trailers, map lanes and routed vehicles
seen in the window, read once a second from radar and the map worker (unknown while either is
down). A second 10 s line, `Loops in the last 10 s`, gives every worker's loop rate against its
target, its loop wall time p50, p99 and max against its `loop_interval`, and its CPU per loop,
worst p99 share of the budget first; wall time beyond the CPU is mostly waiting for the GIL.
`BaseThread` times each `loop()` on `perf_counter` (`monotonic` ticks at 15.6 ms on Windows)
into its `LoopTimes` only while the probe runs (`LoopTimes.on`), so without `debug` a loop pays
one clock read and one flag test. `tools/lag_report.py` turns a drive's log into a table: map
cells ranked by process CPU with their stalls, traffic, lanes and busiest thread, each thread's
late windows and how its CPU follows traffic and lanes. It splits stalls into busy (threads used
the CPU) and quiet (under half a core: OS scheduling, disk or a blocking native call). A stall's
CPU is read only once the gap is seen (`From 260 ms in:`) and shared over the time between the two
reads; lines before 2026-11-23 divided it by the whole stall, which read busy stalls as quiet, so
the report leaves those unclassified. For the
functions behind a hot place, record the drive with `tools/profile_drive.py` (py-spy, GIL holder
only) and read it with `tools/profile_report.py`.

Why the CPU column matters: almost every PySide6 call releases the GIL and must win it
back, so a GUI thread making many Qt calls slows down in proportion to how busy the
workers are (2000 `drawLine` calls: 7 ms alone, 775 ms beside one busy thread). Thread CPU
comes from `GetThreadTimes` (Windows, 15.6 ms ticks) or the thread CPU clock (Linux), never
psutil: its thread list walks every process, 116 ms a call on Windows.

## GIL switch interval (`gil_switch.py`)

One thread runs Python at a time. A thread that wants the lock while a busy one holds it
waits up to the switch interval, then asks it to let go; a thread that gives the lock up
often (the GUI on almost every Qt call, every worker between loops) pays that wait each time
it wants it back. In busy TruckersMP traffic the process sits at a full core and the GUI
froze for 400 ms while the 100 Hz loops fell to 69-76 Hz (2026-11-23). `Settings.gil_switch_ms`
(config.json only) sets the interval at start: 1 ms by default, Python's own is 5. The start
logs it, and an unusable value keeps Python's interval with a warning.

Windows counts that wait in whole milliseconds, rounded down, and ends it on the process
timer tick: 15.6 ms unless something raised the timer (SDL, under pygame, normally sets
1 ms). So 1 to 5 ms all behave as one tick on a default timer, and anything under 1 ms is
no wait at all: the lock is handed over at once. A bench of a GUI-like thread (200
lock-releasing calls) beside 3 busy threads, one 3 s run each, 1 ms timer: 619 ms at 5 ms,
296 at 2, 149 at 1, 0.6 under 1 ms, but the busy threads then did 15 to 20 % less work. In
a process already at a full core that loss lands on AEB and the map worker, so the
default stays at 1 ms. With `debug` on, the stall probe logs once, at its first 10 s
report, how long a 1 ms timed wait really lasts: about 1 ms means the interval works as
set; about 15 ms means the timer was never raised and 1 ms buys nothing. Not yet measured
in game: compare a busy drive at 1 ms against `gil_switch_ms: 5` with `tools/lag_report.py`
(GUI gap above 95 % CPU, sending and cruise loop rates, AEB p99).

## Popup UI

`ui/popup/popup_window.py` is a process-wide singleton created by the entry point, with
`PopupAnimator` for animation and `MessageQueue` / `PopupMessage` for queueing and
styling. `popup_log_handler.py` forwards log records carrying `extra={"popup": True}` to
it. Core modules import `PopupWindow` lazily inside the calling function so the control
path stays importable without Qt.

## Adding a worker thread

1. Copy `core/example_thread/thread.py`, the canonical template, and rename the class
   and its `ThreadData`. `core/test_thread/thread.py` is a smaller reference.
2. Put only fields siblings actually need on the `ThreadData`, and update related fields
   together under the lock.
3. `setup` opens resources and validates config, `loop` does the smallest useful unit of
   work and returns so `BaseThread` can pace it, `teardown` releases everything and stays
   safe to call after an error. Do not remove any of the three.
4. Register the instance in the entry point and register a restart factory with the
   watchdog, or the thread cannot be recovered.
5. Read siblings through the registry and their typed `ThreadData`, never their
   internals.

## Settings (`core/settings.py`)

`Settings.save` holds the state lock only to apply values and copy them; the file is
written outside it, serialised by `_IO_LOCK` (taken before the state lock, never after).
Every `Settings.x` read takes the state lock, so a write under it held every thread for the
length of an fsync. Worker threads and the GUI call `Settings.save_async`: the values apply
at once and the settings writer thread writes the file, coalescing requests.
`Settings.flush_pending()` waits for it (tests, exit). With `debug` on, the config refresh
reloads only hand edits: it skips while a write is pending and skips the mtime of our own
last write, which would otherwise undo values applied since.

One instance for the process. Read with `Settings.field`, write with
`Settings.save(values={"field": value})`. Valid keys are the dataclass fields, mirrored
in `config.json` at the install root. Load, atomic write, backup rotation and corruption
recovery are covered by `tests/test_settings_persistence.py`.
