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
- At `restart_count >= max_restarts` the thread is marked unhealthy instead of restarted.
- `telemetry_thread` with `data.request_quit` is never restarted: shutdown is the entry
  point's business.
- Restart order: `safe_state()` on a throwaway thread (it may block on I/O), brief
  `stop(force=True)` plus `join(STOP_GRACE)`, then build the replacement from the
  registered factory, carry `restart_count + 1` over, and `registry.replace()` it.
- Separately, threads delivering under `LOW_FPS_THRESHOLD` of their target for
  `LOW_FPS_STREAK_WARN` polls are collected for `LOW_FPS_WARN_WINDOW` and warned about
  once, batched, so a system-wide stall produces one message and not one per thread.

Behaviour is pinned by `tests/thread_management/`.

## Monitor (`monitor.py`)

Interactive CLI, only when `Settings.debug` is true: `status`, `stop <name>`,
`restart <name>`, `quit`.

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

One instance for the process. Read with `Settings.field`, write with
`Settings.save(values={"field": value})`. Valid keys are the dataclass fields, mirrored
in `config.json` at the install root. Load, atomic write, backup rotation and corruption
recovery are covered by `tests/test_settings_persistence.py`.
