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
  - Use the standard `logging` module for all diagnostic output. Log unexpected behaviour (e.g. exceptions, missing or down threads, invalid state) so that failures are diagnosable; avoid using `print` for errors or warnings.
  - **`extra={"popup": True}` must be used sparingly.** Only add it when the message is genuinely useful information for the end-user (e.g. "cruise control engaged", "target vehicle lost"). Never add it to error messages, exception traces, debug output, or internal state logs: those belong in the log file only. Overuse causes popup spam; the popup threshold was lowered and this surfaced many inappropriate popups as a regression. If you need to surface a user-facing summary alongside a detailed log, make two separate log calls: one with `extra={"popup": True}` for the short user message, and one without for the full context.
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

### Longitudinal control invariants

- **CC and Limiter are mutually exclusive sibling controllers.**
  - `Settings.cc_mode` selects which controller steps each tick: `"Cruise control"` → `CruiseController` (+ `AdaptiveCruiseController`); `"Speed limiter"` → `SpeedLimiter`. Both live in `CruiseControlThread`.
  - The CC FSM (`enable`/`disable`/`set_target_kmh`) drives `_cc_ctrl` in both modes. In limiter mode the CC's enabled state and target are forwarded to the limiter as its cap.
  - On mode flip, the now-inactive controller's PID state is reset to avoid stale integrators on re-entry.

- **Disengage conditions (brake, park, neutral/reverse, disarm-on-stop) are CC-only.**
  - `CruiseControlThread._handle_cc_disengage_conditions()` is called inside `if mode == "Cruise control"` only. The limiter never sees these events.
  - This preserves the original always-on limiter behaviour: the limiter remains active through brake presses, gear changes, and crash events.

- **`global_speed_limit_kmh` has a dual role.**
  - In CC mode: clamps the CC set-speed so the user can never command above this value (even via stale state). `CruiseController._clamp_target_kmh` enforces this every tick inside `CruiseController.step()`.
  - In limiter mode: activates `SpeedLimiter` unconditionally regardless of CC FSM state. The truck is always capped even without pressing any button. Setting it to `None` deactivates this always-on path.
  - In limiter mode `CruiseController.step()` never runs, so `CruiseControlThread` re-applies the clamp itself each tick before forwarding the button target to the limiter. Without this, tightening the global limit mid-session leaves a stale higher button target capping the truck above the new global limit.

- **Speed limiter is a continuous tracker, not an over-limit reactor.**
  - The `SpeedLimiter` (`core/longitudinal/limiter.py`) PID runs every tick while active and returns `LongOutput(wanted, True)` (active=True) unconditionally. The mapper engages and the user-gas cap fires even when ego is below the limit.
  - Why: a limiter that only wakes on overshoot overshoots: by the time the PID engages, ego is already past the cap. The continuous tracker tightens the gas cap progressively as ego approaches the limit.
  - Do not reintroduce an "only when over the limit" gate (e.g. `if wanted_ms2 < 0: ...` on the limiter). This was tried twice; both times it caused overshoot or fight-with-cruise behaviour at the limit boundary.
  - The asymmetric clamp in `SpeedLimiter.step()` bounds only the lower side (`max(accel_min, wanted)`). The upper side is left open so positive bids propagate and the mapper can shape the gas pedal while below the cap.

- **Winner label and user gas override.**
  - `CruiseControlThread` publishes `active_controller`: `"cc"` whenever CC or ACC is bidding (max-merge in `SendingThread`: user OPD gas may override the mapper's gas), `"limiter"` only when the limiter is the sole bidder (min-merge: hard cap on the user pedal). Do not label a tick `"limiter"` just because the limiter's bid owns the arbitration min while CC is active: the min-merge zeroes CC's gas with the foot off the pedal and the output jitters whenever the bids cross (e.g. set speed == global limit).
  - User gas override of CC: `SendingThread` publishes `user_gas_above_mapper` (user OPD gas exceeded the mapper's gas while any controller was engaged: CC/ACC bidding, or the sole-bidder limiter cap binding). `CruiseControlThread` latches on it in cruise mode while the limiter is active and excludes the CC/ACC bids, making the limiter the sole bidder so the global limit caps the user pedal during the override. The latch exits when the user lifts off the OPD gas region or ego falls below the CC target. Without this handover, a user gas override bypasses the global speed limit entirely.
  - The flag must stay branch-independent (also set while the limiter cap binds). If it only fired in the CC/ACC max-merge branch, enabling CC/ACC at a binding global-limit cap with the foot on the gas would round-trip the winner label "limiter" -> "cc" -> "limiter": one tick of uncapped user pedal reaches the game, then the handover re-seats the gas cap at the user's full pedal and walks it down at the gas rate limit (a throttle surge past the limit). With the branch-independent flag the latch engages on the enable tick itself and the winner never flips.

- **One mapper, one published bid.**
  - There is exactly one `AccelToPedals` instance in the running system (in `SendingThread`). `CruiseControlThread` publishes a single m/s² bid covering whichever controller is active (CC or limiter). `SendingThread` reads that bid from `telemetry_thread.commanded_accel_ms2` and feeds it straight to the mapper.
  - Do not give the limiter (or any other longitudinal child) its own `AccelToPedals` instance. Two parallel mappers diverge in `wanted_smooth` / fast PID / output EMA per-instance state, which broke commander handover at the limit boundary.

- **New `limiter_*` settings.**
  - `limiter_kp`, `limiter_ki`, `limiter_kd`, `limiter_integral_clamp`, `limiter_accel_min_ms2` in `core/settings.py`. Independent of the CC gains so each can be tuned separately. Defaults match the original CC defaults so behaviour is identical until the user tunes them.

