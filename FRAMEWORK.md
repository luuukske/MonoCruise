## MonoCruise threading framework

This document explains the small threading framework used by MonoCruise.
It is intended to be a **one-stop reference** – you should not need to
read the source to implement a new worker.

The framework targets **Python 3.13+** and runs on both Linux and
Windows.  All workers use `threading.Thread` (not `multiprocessing`)
because MonoCruise is primarily I/O-bound (game telemetry, network,
UI).

---

### Overview

- **Entry point**: `main.py`
- **Core framework**:
  - `base_thread.py` – `BaseThread` + `SnapshotMixin`
  - `registry.py` – `Registry`, `Event`, pub/sub, thread lifecycle
  - `watchdog.py` – `WatchdogThread`
  - `monitor.py` – `MonitorThread`, `MonitorData`, `ThreadMetrics`
  - `settings.py` – process-wide `Settings` dataclass
- **Example/template**:
  - `template_thread.py` – ready-to-copy example worker
- **MonoCruise workers**:
  - `telemetry_thread.py` – sole reader of game telemetry
  - `cruise_thread.py` – cruise-control logic
  - `updater_thread.py` – update checker, fully isolated
  - `ui_thread.py` – owns HUD/popup and reacts to events
- **UI surfaces (no threading logic)**:
  - `ui/hud.py` – `HUD` class
  - `ui/popup.py` – `PopupManager` class

All threads log via a **shared logger** configured in `main.py`.  The
log format includes the **thread name** automatically.

---

## BaseThread and data dataclasses

Every worker is a subclass of `BaseThread[DataT]` where `DataT` is a
typed dataclass describing the public state that other threads may
read.

```python
from dataclasses import dataclass
from base_thread import BaseThread, SnapshotMixin
from registry import Registry


@dataclass
class MyData(SnapshotMixin):
    speed_kmh: float = 0.0
    engaged: bool = False


class MyThread(BaseThread[MyData]):
    def __init__(self, registry: Registry, *, name: str = "my", loop_interval: float = 0.05):
        self._registry = registry
        super().__init__(name=name, loop_interval=loop_interval)

    def create_initial_data(self) -> MyData:
        return MyData()

    def on_startup(self) -> None:
        # Optional – one-time setup.
        ...

    def loop(self) -> None:
        # Called roughly every loop_interval seconds.
        # Do NOT call time.sleep() here for rate limiting.
        ...

    def on_shutdown(self) -> None:
        # Optional – best-effort cleanup.
        ...
```

Key rules:

- **Do not use `time.sleep()` in `loop()`** for rate limiting.
  - Use `loop_interval` when constructing the thread.
  - The base class uses `Event.wait(loop_interval)` so shutdown remains responsive.
- **Public data lives in a typed dataclass on each thread**:
  - Owner thread mutates `self.data` (ideally via `self.data.update(...)`).
  - Other threads read via `registry.get_thread("name").data`.
- **Primitive reads are safe bare**:
  - Reading a single `float`, `int`, `bool` field from another thread is safe.
  - Python’s GIL guarantees atomic reads of these primitives.
- **Consistent multi-field reads use `snapshot()`**:
  - Dataclasses should inherit from `SnapshotMixin`.
  - Other threads call `snapshot()` when multiple fields must be read together:

    ```python
    tele = registry.get_thread("telemetry")
    snap = tele.data.snapshot()
    speed = snap.speed_kmh
    brake = snap.brake_pressed
    ```

  - `SnapshotMixin` guards `snapshot()` and `update()` with a `threading.Lock`.

`BaseThread` exposes metrics used by the monitor and watchdog:

- `iteration_count`
- `last_loop_started`
- `last_loop_duration`
- `last_heartbeat`
- `last_exception`
- `heartbeat_timeout` (default `1.0`, can be overridden per subclass)

Worker exceptions **never propagate past the thread boundary** – they
are caught, logged, and recorded in `last_exception`.  The watchdog may
restart failed threads.

---

## Registry and inter-thread communication

`Registry` is the central coordination point:

- Thread lifecycle:
  - `register_thread(name, factory, auto_start=True, auto_restart=True, heartbeat_timeout=None)`
  - `start_thread(name)` / `stop_thread(name)` / `restart_thread(name)`
  - `start_all()` / `stop_all()`
  - `get_thread(name)` – returns the `BaseThread` instance or `None`
  - `status_snapshot()` – returns a list of dicts describing each thread
- Pub/sub event bus:
  - `subscribe(event_name, subscriber_name)`
  - `unsubscribe(event_name, subscriber_name)`
  - `publish(event_name, payload=None, sender=None)`
  - `poll_events(subscriber_name, max_events=None)` – non-blocking

### Data sharing (typed dataclasses, no key-value store)

Threads **do not** use a generic string key-value store.  Instead:

- Each worker defines its own typed dataclass (e.g. `TelemetryData`,
  `CruiseData`, `UiData`, etc.).
- Other threads obtain a reference to the owning thread via
  `registry.get_thread("name")` and read its `.data` attribute.
- For multi-field consistency, they call `data.snapshot()`.

This keeps cross-thread contracts explicit and type-safe while remaining
simple to use.

### Events (pub/sub)

Events are used for **discrete notifications**, e.g.:

- `"cruise:engaged"` / `"cruise:disengaged"`
- `"updater:update_available"`
- `"ui:popup"` (if you define it – currently `UiThread` calls
  `PopupManager` directly)

Pattern:

```python
# Publisher (inside a worker loop)
self._registry.publish(
    "cruise:engaged",
    payload={"target_speed_kmh": target},
    sender=self.name,
)

# Subscriber (on startup)
self._registry.subscribe("cruise:engaged", self.name)

# Subscriber loop()
events = self._registry.poll_events(self.name, max_events=20)
for ev in events:
    if ev.name == "cruise:engaged":
        ...
```

Under the hood each subscriber has its own `Queue[Event]`.  `publish()`
fan-outs to all current subscribers.

---

## Watchdog and monitor

### WatchdogThread

`WatchdogThread` lives in `watchdog.py` and is registered as the
`"watchdog"` thread.  It:

- Iterates over `Registry.iter_specs()`.
- For each thread:
  - If the thread is not alive → logs a warning and optionally restarts
    it if `ThreadSpec.auto_restart` is `True`.
  - If the heartbeat is older than the configured timeout → logs an
    error and optionally restarts it.
- Writes a human-readable list of recent problems into `WatchdogData`.

The watchdog itself is **not auto-restarted** (by design).

### MonitorThread

`MonitorThread` lives in `monitor.py` and is registered as the
`"monitor"` thread.  It:

- Periodically calls `registry.status_snapshot()`.
- Projects that into a `MonitorData` dataclass containing:
  - `last_refresh_ts`
  - `threads: dict[str, ThreadMetrics]`
- The CLI uses this snapshot to display live performance metrics.

---

## Settings

`settings.py` contains a frozen dataclass `Settings` and a tiny loader:

```python
from settings import Settings

settings = Settings.load("settings.json")
```

- If `settings.json` does not exist or cannot be parsed, safe defaults
  are used.
- Only `main.py` touches the config file; threads receive
  configuration via constructor arguments.

Important fields:

- `telemetry_loop_interval`, `cruise_loop_interval`, `updater_loop_interval`,
  `ui_loop_interval`, `watchdog_loop_interval`, `monitor_loop_interval`
- `heartbeat_timeout`
- `auto_restart_workers`

---

## Main entrypoint and CLI

`main.py` wires everything together:

1. Configures logging (shared root logger with thread names).
2. Loads `Settings` from `settings.json`.
3. Builds the `Registry` and registers:
   - `telemetry` (`TelemetryThread`)
   - `cruise` (`CruiseThread`)
   - `updater` (`UpdaterThread`)
   - `ui` (`UiThread`)
   - `watchdog` (`WatchdogThread`)
   - `monitor` (`MonitorThread`)
4. Starts all threads marked `auto_start=True`.
5. Enters a **CLI loop** on the main thread:

   - `help` / `h` – show commands
   - `threads` / `ls` – list all registered threads
   - `status` – basic status (alive, last heartbeat age, last exception, etc.)
   - `perf` – performance snapshot (iteration counts, loop durations)
   - `stop <name>` – stop a specific thread
   - `start <name>` – start (or re-start) a thread
   - `restart <name>` – restart a thread
   - `quit` / `exit` – stop all threads and exit

The main thread never blocks on workers except during shutdown when
`registry.stop_all()` is called.

---

## MonoCruise-specific workers

### telemetry_thread

- **File**: `telemetry_thread.py`
- **Class**: `TelemetryThread(BaseThread[TelemetryData])`
- **Responsibility**:
  - Sole reader of game telemetry.
  - Writes into `TelemetryData` only.
  - Other threads read via `registry.get_thread("telemetry").data`.
- **Current implementation**:
  - Simulates speed and brake state so that the framework can be tested.
  - Replace `loop()` with your actual game integration.

### cruise_thread

- **File**: `cruise_thread.py`
- **Class**: `CruiseThread(BaseThread[CruiseData])`
- **Responsibility**:
  - Reads `telemetry.data`.
  - Runs cruise-control logic.
  - Emits events:
    - `"cruise:engaged"` with `{"target_speed_kmh": ...}`
    - `"cruise:disengaged"` with `{"speed_kmh": ..., "brake_pressed": ...}`
- This is where real control logic should live.

### updater_thread

- **File**: `updater_thread.py`
- **Class**: `UpdaterThread(BaseThread[UpdaterData])`
- **Responsibility**:
  - Fully isolated – does not depend on other threads.
  - Checks for updates (stubbed out for now).
  - Publishes `"updater:update_available"` when appropriate.
  - Then idles between checks.

### ui_thread

- **File**: `ui_thread.py`
- **Class**: `UiThread(BaseThread[UiData])`
- **Responsibility**:
  - Owns the HUD and popup manager.
  - Subscribes to:
    - `"cruise:engaged"`
    - `"cruise:disengaged"`
    - `"updater:update_available"`
  - Updates:
    - `HUD` via `ui/hud.py`
    - `PopupManager` via `ui/popup.py`
  - No other thread touches rendering objects.

Currently the HUD and popup are logging-only facades; they can later be
backed by PySide6 (`core/cc_panel`, `core/popup`) from within
`UiThread` only.

---

## Template for new threads

To add a new worker:

1. Copy `template_thread.py` to e.g. `foo_thread.py`.
2. Rename the dataclass and thread class.
3. Define your typed fields on the dataclass.
4. Implement `loop()` using `loop_interval` for rate limiting.
5. If needed, use:
   - `self._registry.get_thread("other")` and read `.data`
   - `self._registry.publish("topic", payload, sender=self.name)`
6. Register the thread in `main.py` via `Registry.register_thread(...)`.

Guidelines:

- Prefer **events** for cross-thread signalling.
- Use **typed dataclasses** instead of generic key-value stores.
- If you ever need a CPU-bound worker (e.g. CV, FFT), consider a
  separate `multiprocessing.Process` that communicates via
  `Queue` – do not rewrite the main framework.

---

## Thread safety summary

- Each worker owns **exactly one** dataclass instance.
- Owner thread:
  - May freely mutate fields.
  - Should prefer `data.update(...)` when touching multiple fields.
- Other threads:
  - May read **single primitive fields** directly.
  - Must use `snapshot()` for multi-field consistency.
- No exceptions leave a worker thread – everything is logged and the
  watchdog may auto-restart failed workers.

With this structure in place, new workers can be added with minimal
boilerplate while staying within clear, documented concurrency rules.

