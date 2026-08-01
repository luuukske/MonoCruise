## Monocruise – Agent Guide

This document explains how this program is structured and where to look for usages and examples when making changes as an AI agent.

**Project details:**

MonoCruise is a third-party software that sits in between ETS2/ATS and your pedals. MonoCruise has a ton of quality of life features, like a better Adaptive Cruise Controll or a One-Pedal Driving system for heavy traffic. every feature (including the ACC) works in TruckersMP and singleplayer ETS2/ATS.

### Module docs (human-facing)

Long domain / tuning / coordinate docs live in module `README.md` files, not here. **Read a directory's README.md before modifying code in it** (and parent-directory READMEs, if present) — they hold architectural constraints and conventions that blind edits will miss:

- `core/radar/README.md`: coordinates, traffic buffer, Vehicle smoothing, ArcPath
- `core/aeb/README.md`: AEB pipeline, filters, engagement, calibration
- `core/acc/README.md`: in-lane tracker, scoring, blinker bias
- `core/sending_thread/README.md`: mapper, pedal capacity, hold, brake efficiency
- `core/longitudinal/README.md`: CC/limiter/ACC children
- `core/cruise_control_thread/README.md`: orchestrator and ACC anticipation
- `core/main_pedal_thread/README.md`: joystick, OPD, capture APIs
- `core/sdk_installer/README.md`, `core/update_check/README.md`, `updater/README.md`, `shared/README.md`
- `checker/README.md`: background game-launch checker, mutex-based detection, AV-safe design

Do-not-break rules for radar/AEB/ACC/longitudinal are summarized under **Domain invariants** below; full rationale stays in the module README.

### Comment budget

Root `AGENTS.md` is present, so inline comments/docstrings that act as comments are capped at **2 consecutive lines** (absolute max **3** only when truly necessary). Put durable explanation in the nearest module `README.md` (human) or this file (agent-only). No decorative `# ---` / `# ====` dividers. No em dash in comments, docstrings, commit-adjacent text, or log strings.

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
  - When a thread reads from the registry or another thread’s data, it must handle missing or down sibling threads gracefully: catch `KeyError` and attribute or lock failures, use safe defaults, log at debug or warning level as appropriate, and continue. A thread must never crash or exit its loop because another thread is missing or has crashed. Enforced by `tests/invariants/test_source_invariants.py`.

- **Independent threads**
  - Any error or looping code CAN NOT impact other critical systems. use the example thread code `core/example_thread/thread.py`.
  - Do NOT diviate from the template by removing key methods like `teardown()` or `setup()`.
  - The main thread is dedicated to program critical code. The main thread has to be as stable as possible for both supported OS; Windows and Linux.
  - Always check self.running at the top of loop() and in any inner loop.

- **Loop timing and blocking calls**
  - Never block a thread’s `loop` method or any inner loop for longer than **0.5 seconds** at a time.
  - **Do not call `time.sleep` inside `loop` or any method it calls.** Use the pacing and sleep facilities provided by `BaseThread` instead so the watchdog and health checks stay accurate. Enforced by `tests/invariants/test_source_invariants.py`.

- **Testing and validation**
  - Never weaken, disable, or delete an existing safety check (watchdog, health check, restart limit) or an existing test assertion to make a test or build pass. Adding new tests for thread behaviour is encouraged; changing an existing safety assertion needs explicit approval.
  - `pytest` is safe to run: the root `conftest.py` redirects `core.settings` at a throwaway directory for the whole session, so no test can touch the live `config.json`. A scratch script that imports `core.settings` outside pytest still can, so snapshot `config.json` and `config.json.bak` before running one.
  - CI (`.github/workflows/tests.yml`) runs `pytest -m "not needs_clips"` on every PR with `MONOCRUISE_STRICT_SKIPS=1`: any test that skips fails the build. A test needing an AEB clip from the local clip store must carry `@pytest.mark.needs_clips`.

- **Physical safety**
  - When an error uccurs or certain parts of the code fail, the user must ALWAYS be able to stop the vehicle being controlled in ETS2 or ATS without causing an accident.

### Domain invariants (radar / AEB / ACC / longitudinal)

Do not reintroduce these without reading the linked README section first:

- **Radar** (`core/radar/README.md` §14): quaternion x/y swap intentional; `rotationX` is yaw; radar render uses `+0.5` yaw offset, arc geometry does not; bodies are symmetric `± length/2`; use `_smooth_yaw` for arcs; `acc_speed` is ACC-only; consumers must not open the traffic shared-memory buffer or mutate `Vehicle` instances.
- **AEB** (`core/aeb/README.md` §9): yaw-rate ego curvature only (never `RadarData.ego_curvature`); target κ via `_vehicle_curvature_blend` stepped once per vehicle per frame; all tunables in `AEBCalibration`; `lane_frame.project_to_ego_arc` is the lane primitive; TMP trailer kinematics via shallow-copy swap; two-layer pedal authority (FF assist + engagement slam), gas ≥ 0.8 is the only AEB override.
- **ACC** (`core/acc/README.md` §8): never mutate `Vehicle`; history-fit ego curvature (not AEB's yaw-rate proxy); no control law in this module (cruise owns accel); scoring stays meter-native.
- **Longitudinal telemetry** (`core/longitudinal/README.md`): `lv_accelerationX` is the truck-local **lateral** axis (right/left), not longitudinal, despite its name and the SDK's own labeling. Using it as an accel/decel feedback signal causes phantom braking on right turns and phantom acceleration on left turns, worsening the longer CC stays in the turn (the integral term amplifies it). Use a tracking differentiator on `speed` instead (`_spd_smooth` in `core/sending_thread/thread.py`, `_ACCEL_TRACK_TAU_S` in `core/longitudinal/limiter.py`) — not raw tick-to-tick `d(speed)/dt`, which spikes because ETS2 physics runs at ~20Hz against a faster control loop. Enforced by `tests/invariants/test_source_invariants.py`.

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
  - In limiter mode `CruiseController.step()` never runs, so `CruiseControlThread` re-applies the clamp itself each tick before forwarding the button target to the limiter. Without this, tightening the global limit mid-session leaves a stale higher button target capping the truck above the new global limit. Enforced by `tests/longitudinal/test_orchestrator_limiter_mode.py`.

- **Speed limiter is a continuous tracker, not an over-limit reactor.**
  - Why: a limiter that only wakes on overshoot overshoots: by the time the PID engages, ego is already past the cap. The continuous tracker tightens the gas cap progressively as ego approaches the limit, and its asymmetric clamp bounds only the lower side so positive bids still shape the gas pedal below the cap.
  - Do not reintroduce an "only when over the limit" gate (e.g. `if wanted_ms2 < 0: ...` on the limiter). This was tried twice; both times it caused overshoot or fight-with-cruise behaviour at the limit boundary. Enforced by `tests/longitudinal/test_limiter_continuous.py`.

- **Winner label and user gas override.**
  - `CruiseControlThread` publishes `active_controller`: `"cc"` whenever CC or ACC is bidding (max-merge in `SendingThread`: user OPD gas may override the mapper's gas), `"limiter"` only when the limiter is the sole bidder (min-merge: hard cap on the user pedal). Do not label a tick `"limiter"` just because the limiter's bid owns the arbitration min while CC is active: the min-merge zeroes CC's gas with the foot off the pedal and the output jitters whenever the bids cross (e.g. set speed == global limit). Enforced by `tests/longitudinal/test_arbitration_winner.py`.
  - User gas override of CC: `SendingThread` publishes `user_gas_above_mapper` (user OPD gas exceeded the mapper's gas while any controller was engaged: CC/ACC bidding, or the sole-bidder limiter cap binding). `CruiseControlThread` latches on it in cruise mode while the limiter is active and excludes the CC/ACC bids, making the limiter the sole bidder so the global limit caps the user pedal during the override. The latch exits when the user lifts off the OPD gas region or ego falls below the CC target. Without this handover, a user gas override bypasses the global speed limit entirely.
  - The flag must stay branch-independent (also set while the limiter cap binds), or the winner label round-trips "limiter" -> "cc" -> "limiter" on the enable tick and one tick of uncapped user pedal reaches the game, followed by a throttle surge past the limit. Enforced by `tests/invariants/test_source_invariants.py`.

- **One mapper, one published bid.**
  - There is exactly one `AccelToPedals` instance in the running system (in `SendingThread`). `CruiseControlThread` publishes a single m/s² bid covering whichever controller is active (CC or limiter). `SendingThread` reads that bid from `telemetry_thread.commanded_accel_ms2` and feeds it straight to the mapper.
  - Do not give the limiter (or any other longitudinal child) its own `AccelToPedals` instance. Two parallel mappers diverge in `wanted_smooth` / fast PID / output EMA per-instance state, which broke commander handover at the limit boundary. Enforced by `tests/invariants/test_source_invariants.py`.

- **New `limiter_*` settings.**
  - `limiter_kp`, `limiter_ki`, `limiter_kd`, `limiter_integral_clamp`, `limiter_accel_min_ms2` in `core/settings.py`. Independent of the CC gains so each can be tuned separately. Defaults match the original CC defaults so behaviour is identical until the user tunes them.

---

### Workflow and communication conventions

- **Keep responses and commit messages terse.** Drop filler, pleasantries, and unneeded articles/words while keeping technical precision (referred to internally as "Caveman Lite" style). Applies to chat responses and to git commit messages alike.
- **Never use an em dash ("—") in code or code-adjacent text**: comments, docstrings, strings, commit messages, log messages. Use a comma, a colon, or start a new sentence instead.
- **Do not commit immediately after writing code.** Implement the change, then wait for the user to test it and explicitly confirm it works before running `git commit`.
- **No decorative divider comments.** Do not write banner-style separators like `# --- Section ---` or `# ====` in source files. Section structure should come from function/class boundaries; a short prose lead-in comment is fine (for example: `# Headway time follows the user's gap-level setting.`). Module docstrings may use markdown headings since they render as documentation; source-code comments may not.
- **When unsure, ask.** If an instruction or the surrounding context is ambiguous (multiple plausible interpretations, missing key parameter, unclear scope), ask a clarifying question instead of guessing or silently picking a default. Don't overuse this for trivial ambiguity resolvable from context, codebase conventions, or existing docs.
- **State confidence.** Flag it explicitly when unsure (below roughly 80% confidence) so the user knows to verify. Above roughly 90% confidence, only call it out for larger architectural changes, not routine fixes.
- **Pushback is welcome.** Higher-reasoning models are encouraged to suggest a better approach or architecture instead of executing a request mechanically, and to briefly state the tradeoff so the user can decide.

### Changelog conventions

- `CHANGELOG.md` entries are user-facing, not developer-facing. Format: a bold title naming the symptom or feature as the user experiences it, then one short sentence (two at most, only if genuinely needed) describing what they noticed and what changed. No internal constant names, thresholds, taus, or control-theory jargon. Releases `1.1.0-preview.1` through `.5` are the style reference for length and tone.
- Example of the right size: "**Anti-creep too strong at launch**: weak engines couldn't overcome the creep-cancel brake; it now releases much earlier on the gas pedal."

### Reputation and community-trust guardrails

MonoCruise's community signed up for offline operation and no data collection. Changes that touch that trust need a higher bar than ordinary code review. Two failure modes this guards against: shipping something that damages community trust (privacy, monetization, tone), and acting as an echo chamber that validates a weak feature idea just because it was pitched with confidence.

- **Tier 1, hard gate (halts implementation):** selling, sharing, or monetizing user data, hardware stats, telemetry, or usage info; adding telemetry, phone-home behavior, or logging beyond the stated offline / no-data stance; license changes or paywalling previously free functionality; reusing third-party or copyrighted mod assets; anything else that directly contradicts the offline-operation / no-data-collection / privacy stance the community expects. State the specific risk and who would object (max 5 sentences), then halt implementation until the user replies with the exact phrase "I see the risk and I acknowledge the risk." Discussion, planning, and red-teaming stay open; only implementation halts.
- **Tier 2, inline flag (no gate):** monetization presentation (donation prompts, placement, frequency, wording); public-facing comms (Discord, README, release notes, replies to critics) that read as defensive, dismissive, condescending, or that overcommit on features or timelines; any change the community could plausibly read as values drift even if it technically isn't one. Flag inline (max 3 sentences), then keep helping. Raise during planning and design discussion, not only at commit time.
- **Tier 3, minor optics:** "this could read as weird or off to users." One sentence, then proceed; no follow-up unless asked.
- **Feature merit check** (separate axis, anti-echo-chamber): before writing implementation code for a new user-facing feature or UX change, give an independent verdict: ship / ship with changes / don't ship, plus the strongest case against it from the driver's seat (distraction while driving, visual clutter, config burden, surprising behavior, whether any trucker actually asked for this). Confidence or enthusiasm in the request is not evidence the feature is good; a merit check that always returns "ship" is a failed check. Skip only for bug fixes, refactors, perf work, internal tooling, or features already vetted earlier in the same conversation. Placement/styling questions ("where do I put it", "how should it look") presuppose the feature ships — treat that framing as a trigger, not a pass.
- **Noise control:** one flag per issue per conversation. After acknowledgment or override, drop it; don't re-raise or append warnings to later replies.
- **Limits:** this is a soft prior, not a guarantee — it catches recognizable categories, not every community-specific landmine. Mechanizable risks (telemetry/phone-home, leaked secrets, license incompatibility) belong in CI where possible; this section is the backstop for what CI can't check (optics, tone, feature merit, community vibe).

### Packaging and antivirus false positives

MonoCruise ships unsigned (no code-signing certificate) and will remain unsigned until there is monetization to justify one. Unsigned binaries already draw more antivirus scrutiny.

- When touching the updater (`updater/`), installer (`installer/MonoCruise.iss`), the background checker (`checker/`), or the release/build pipeline (`.github/workflows/release.yml`), prefer conventional, well-documented approaches: standard HTTPS downloads, standard file replacement. Avoid patterns that read as malware to AV heuristics: packing/UPX, self-modifying or self-updating executable tricks, process/memory-injection-like behavior, `shell=True` process enumeration, or writing Run/startup registry keys from anywhere but the (visible, consent-based) installer.
- If a risky-looking pattern is genuinely necessary, flag it explicitly rather than shipping it silently.
- **PyInstaller 6 resolves relative paths in `.spec` files against the spec file's own directory, not the invocation cwd.** A spec under a subdirectory (e.g. `updater/updater.spec`) must build paths from the `SPECPATH` global, not repo-root-relative strings, or CI builds silently break on PyInstaller ≥6. Specs at repo root (`monocruise.spec`) are unaffected.

