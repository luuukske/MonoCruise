## Monocruise – Agent Guide

This document explains how this program is structured and where to look for usages and examples when making changes as an AI agent.

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

- **Settings**
  - `core/settings.py`:
    - How configuration for MonoCruise is loaded and exposed.

### Privacy and safety requirements for agents

- **Do not store absolute paths**
  - Never write or persist absolute filesystem paths into source files, documentation, logs, or configuration that will be committed or logged.
  - This includes any path that contains user or machine names (for example, home directories).
  - When referencing files or directories in this repository, always use relative paths from the project root (for example, `core/thread_management/registry.py`).

- **Be careful with logging**
  - Avoid introducing log messages that include personally identifying information or machine‑specific details.
  - Prefer generic wording that does not reveal usernames, hostnames, or full filesystem layouts.

- **Independent threads**
  - Any error or looping code CAN NOT impact other critical systems. use the example thread code `core/example_thread/thread.py`.
  - Do NOT diviate from the template by removing key methods like `teardown()` or `setup()`.
  - The main thread is dedicated to program critical code. The main thread has to be as stable as possible for both supported OS; Windows and Linux.
  - Always check self.running at the top of loop() and in any inner loop.

- **Testing and validation**
  - Do not add or update tests when changing thread behaviour that could affect stability or watchdog/monitor behaviour.
  - Do not disable existing safety checks (watchdog, health checks, restart limits, etc.) to “fix” failing tests or crashes.