## Monocruise – Agent Guide

This document explains how this program is structured and where to look for usages and examples when making changes as an AI agent.

**Project details:**

MonoCruise is a third-party software that sits in between ETS2/ATS and your pedals. MonoCruise has a ton of quality of life features, like a better Adaptive Cruise Controll or a One-Pedal Driving system for heavy traffic. every feature (including the ACC) works in TruckersMP and singleplayer ETS2/ATS.

### High‑level architecture

- **Entry point**
  - `main.py` is the application entry point.
  - Responsibilities:
    - Load settings.
    - Configure logging (including file logging to `monocruise.log` and popup logging).
    - Instantiate and register all worker threads.
    - Start the watchdog (and the monitor in debug mode).
    - Run the Qt event loop and coordinate shutdown.

- **Thread model**
  - All long‑running work is implemented as subclasses of `BaseThread` in `core/thread_management/base_thread.py`.
  - Threads register themselves in the central `Registry` singleton defined in `core/thread_management/registry.py`.
  - Threads expose state via typed `ThreadData` subclasses so other threads can safely read their data.

- **Watchdog**
  - Implemented in `core/thread_management/watchdog.py`.
  - Periodically checks all registered threads for:
    - `running == False` and `is_alive() is False` → crashed.
    - Heartbeat age greater than `HEARTBEAT_TIMEOUT` → frozen.
  - Optionally restarts dead or frozen threads using per‑thread factory callables that are registered in `main.py`.

- **Monitor (debug shell)**
  - Implemented in `core/thread_management/monitor.py`.
  - Only active when `settings.debug` is `True`.
  - Provides an interactive CLI for:
    - Inspecting thread status (`status`).
    - Stopping or restarting threads (`stop <name>`, `restart <name>`).
    - Quitting the whole application (`quit`).

- **Popup UI**
  - The main notification popup window is implemented in `ui/popup/popup_window.py`.
  - It uses:
    - `PopupAnimator` for animations.
    - `MessageQueue`, `PopupMessage`, and related types in `ui/popup` for message queuing and styling.
  - `main.py` creates a single `PopupWindow` instance and attaches a `PopupLogHandler` (from `core/thread_management/popup_log_handler.py`) so logging messages can appear as on‑screen notifications.

### Where to look for examples and usages

- **How threads are wired together**
  - See `main.py`:
    - Creation of worker threads.
    - Registration with `registry`. 
    - Registration of restart factories with the `Watchdog`.
    - Optional registration of the `Monitor` in debug mode.

- **Thread base class and lifecycle**
  - `core/thread_management/base_thread.py`:
    - Base class API (`setup`, `loop`, `teardown`, heartbeat handling, restart limits, etc.).
    - Shared behaviour such as pacing the loop and tracking health.

- **Thread registry**
  - `core/thread_management/registry.py`:
    - How threads are stored and looked up.
    - How `registry.replace(...)` is used by the `Watchdog` and `Monitor` to swap threads.

- **Example worker template**
  - `core/example_thread/thread.py`:
    - Canonical template for a new worker thread.
    - Shows how to:
      - Define a typed `ThreadData` dataclass.
      - Implement `setup`, `loop`, and `teardown`.
      - Read from other threads via `registry`.
      - Update internal state safely (including use of locks for multi‑field updates).

- **Minimal test worker**
  - `core/test_thread/thread.py`:
    - Simple worker used for debugging and for exercising error handling and the popup system.
    - Good reference for a very small `BaseThread` subclass.

- **Popup and message system**
  - `ui/popup/popup_window.py`:
    - How popup messages are displayed and how priorities, durations, and hover behaviour work.
  - Other files in `ui/popup`:
    - Message types, style configuration, and animation details.

- **Sending pedal mapping**
  - `core/sending_thread/accel_to_pedals.py`:
    - Maps commanded longitudinal acceleration to gas/brake.
    - Smooths commanded/measured accel, applies the leaky integral correction, includes slope plus rolling-resistance road-load feed-forward, and adapts the estimated full-pedal accel/brake capability.
    - Hill compensation uses telemetry `rotationY` as road pitch in degrees (same convention as AEB: `math.radians(rotationY)`), plus gravity along grade and rolling resistance via configurable `mapper_rolling_resistance`.
    - Adaptive accel/brake estimates are learned from load-compensated accel (`raw + road_load`) so slopes do not bias the learned full-pedal capability.
    - Also contains the shared telemetry mass estimate helper used by `telemetry_thread`.
    - Appends concise tuning rows to `accel_to_pedals_tuning.csv` when high-demand accel/brake estimates clearly underperform, including slope and computed road load.

- **Settings**
  - `core/settings.py`:
    - How configuration for MonoCruise is loaded and exposed using only one instance.
    - If you are just reading or writing values, all you need to know is to just import the Settings function from settings.py, and write `Settings.save(values={"a_valid_value": self.data.a_valid_value})` for saving and `Settings.a_valid_value` for reading values. all valid values are found in `config.json` in the root folder.

### Creating a new worker thread

- **Goal**
  - Every long‑running feature should live in its own `BaseThread` subclass so that the watchdog can monitor it and the rest of the system stays responsive.

- **Step‑by‑step basics**
  - **1. Start from the template**
    - Copy `core/example_thread/thread.py` to a new module (for example, `core/my_feature_thread/thread.py`).
    - Keep the overall structure (class layout, `ThreadData` dataclass, `setup`, `loop`, `teardown`, and heartbeat updates) and rename the class and data types to match your feature.
  - **2. Define your thread data**
    - Edit the `ThreadData` dataclass in your new file to include only the fields you actually need to expose to other threads.
    - Use locks when updating multiple related fields so readers always see a consistent snapshot.
  - **3. Implement `setup`, `loop`, and `teardown`**
    - `setup`: open any required resources (devices, sockets, files) and validate configuration.
    - `loop`: perform the smallest useful unit of work, update `self.data`, send a heartbeat, and then return to let `BaseThread` pace the loop.
    - `teardown`: cleanly release all resources, making sure it is safe to call even after errors.
  - **4. Register the thread**
    - In `main.py`, create an instance of your new thread, register it in the `registry`, and register a restart factory with the watchdog so it can be recreated if it crashes.
  - **5. Read from other threads**
    - Use the `registry` to look up other threads and read their typed `ThreadData` instead of accessing internal attributes directly.

Refer back to `core/example_thread/thread.py` whenever you are unsure about the correct structure or lifecycle; it is the canonical example.

### Privacy and safety requirements for agents

- **Do not store absolute paths**
  - Never write or persist absolute filesystem paths into source files, documentation, logs, or configuration that will be committed or logged.
  - This includes any path that contains user or machine names (for example, home directories).
  - When referencing files or directories in this repository, always use relative paths from the project root (for example, `core/thread_management/registry.py`).

- **Be careful with logging**
  - Avoid introducing log messages that include personally identifying information or machine‑specific details.
  - Prefer generic wording that does not reveal usernames, hostnames, or full filesystem layouts.

- **Logging and resilience**
  - Use the standard `logging` module for all diagnostic output. Log unexpected behaviour (e.g. exceptions, missing or down threads, invalid state) so that failures are diagnosable; avoid using `print` for errors or warnings. if the user must see these logs, add `, extra={"popup": True}`. if you need to log detailed contexts, make two logs with one being the popup and the other a detailed context of the log (optional).
  - When a thread reads from the registry or another thread’s data, it must handle missing or down sibling threads gracefully: catch `KeyError` and attribute or lock failures, use safe defaults, log at debug or warning level as appropriate, and continue. A thread must never crash or exit its loop because another thread is missing or has crashed.

- **Independent threads**
  - Any error or looping code CAN NOT impact other critical systems. use the example thread code `core/example_thread/thread.py`.
  - Do NOT diviate from the template by removing key methods like `teardown()` or `setup()`.
  - The main thread is dedicated to program critical code. The main thread has to be as stable as possible for both supported OS; Windows and Linux.
  - Always check self.running at the top of loop() and in any inner loop.

- **Loop timing and blocking calls**
  - Never block a thread’s `loop` method or any inner loop for longer than **0.5 seconds** at a time.
  - **Do not call `time.sleep` inside `loop` or any method it calls.** Use the pacing and sleep facilities provided by `BaseThread` instead so the watchdog and health checks stay accurate.

- **Testing and validation**
  - Do not add or update tests when changing thread behaviour that could affect stability or watchdog/monitor behaviour.
  - Do not disable existing safety checks (watchdog, health checks, restart limits, etc.) to “fix” failing tests or crashes.

- **Physical safety**
  - When an error uccurs or certain parts of the code fail, the user must ALWAYS be able to stop the vehicle being controlled in ETS2 or ATS without causing an accident.
