## Monocruise – Agent Guide

This document explains how this program is structured and where to look for usages and examples when making changes as an AI agent.

**Project details:**

MonoCruise is a third-party software that sits in between ETS2/ATS and your pedals. MonoCruise has a ton of quality of life features, like a better Adaptive Cruise Controll or a One-Pedal Driving system for heavy traffic. every feature (including the ACC) works in TruckersMP and singleplayer ETS2/ATS.

### Module docs (human-facing)

Long domain / tuning / coordinate docs live in module `README.md` files, not here. **Read a directory's README.md before modifying code in it** (and parent-directory READMEs, if present) — they hold architectural constraints and conventions that blind edits will miss:

- `core/thread_management/README.md`: entry point, thread model, watchdog policy, adding a worker
- `core/radar/README.md`: coordinates, traffic buffer, Vehicle smoothing, ArcPath
- `core/aeb/README.md`: AEB pipeline, filters, engagement; `core/aeb/TUNING.md`: calibration reference
- `core/acc/README.md`: in-lane tracker, scoring, blinker bias
- `core/sending_thread/README.md`: mapper, pedal capacity, hold, brake efficiency
- `core/longitudinal/README.md`: CC/limiter/ACC children
- `core/cruise_control_thread/README.md`: orchestrator and ACC anticipation
- `core/main_pedal_thread/README.md`: joystick, OPD, capture APIs
- `core/button_device_thread/README.md`: HID button reading, report drain, debounce, capture scan
- `core/sdk_installer/README.md`, `core/update_check/README.md`, `updater/README.md`, `shared/README.md`
- `checker/README.md`: background game-launch checker, mutex-based detection, AV-safe design
- `tools/README.md`: offline probes. `acc_response_map.py` answers "what does the ACC gap law command here, and what did my change do to it" without the game running, and has a text/JSON mode built for agents.

Do-not-break rules for radar/AEB/ACC/longitudinal are summarized under **Domain invariants** below; full rationale stays in the module README.

### Comment budget

Inline comments/docstrings that act as comments are capped at **2 consecutive lines**. Put durable explanation in the nearest module `README.md` (human) or this file (agent-only). No decorative `# ---` / `# ====` dividers. No em dash in comments, docstrings, commit-adjacent text, or log strings. Comment budget and em dashes are enforced by `tests/invariants/test_repo_hygiene.py`.

### How the app is wired

Entry point is `monocruise.py` at the repo root. It loads settings, configures logging,
registers every worker thread plus its watchdog restart factory, and runs the Qt event loop.
Thread model, watchdog policy, popup wiring, and the step-by-step for adding a worker thread
live in `core/thread_management/README.md`. Start from `core/example_thread/thread.py`.

### Privacy and safety requirements for agents

- **Do not store absolute paths**
  - Never write or persist absolute filesystem paths (or anything containing a user or machine name) into files that will be committed or logged. Reference repository files by their path from the project root, for example `core/thread_management/registry.py`. Enforced by `tests/invariants/test_repo_hygiene.py`.

- **Be careful with logging**
  - Avoid introducing log messages that include personally identifying information or machine‑specific details.
  - Prefer generic wording that does not reveal usernames, hostnames, or full filesystem layouts.

- **Logging and resilience**
  - Use the standard `logging` module for all diagnostic output. Log unexpected behaviour (e.g. exceptions, missing or down threads, invalid state) so that failures are diagnosable; avoid using `print` for errors or warnings.
  - **`extra={"popup": True}` must be used sparingly.** Only for information the driver can act on (e.g. "cruise control engaged", "target vehicle lost"); never on traces, debug output, or internal state. Overuse already shipped as a popup-spam regression. To surface a user-facing summary alongside detail, make two log calls: a short one with the flag, a full one without. Enforced by `tests/invariants/test_repo_hygiene.py` (hard block on traces/debug, ratchet on error/critical).
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
  - CI (`.github/workflows/tests.yml`) runs `pytest -m "not needs_clips"` on every PR with `MONOCRUISE_STRICT_SKIPS=1`: any test that skips fails the build. A test needing an AEB clip from the local clip store must carry `@pytest.mark.needs_clips`. CI then runs `ruff check .`; the rule set in `ruff.toml` is deliberately narrow, so a ruff failure is always something you just introduced.
  - `tests/invariants/` holds the mechanised AGENTS.md rules. Some are absolute, some carry a recorded baseline the tree does not yet meet. Lower a baseline when you clean up; never raise one to go green.

- **Physical safety**
  - When an error uccurs or certain parts of the code fail, the user must ALWAYS be able to stop the vehicle being controlled in ETS2 or ATS without causing an accident.

### Domain invariants (radar / AEB / ACC / longitudinal)

Do not reintroduce these without reading the linked README section first:

- **Radar** (`core/radar/README.md` §14): quaternion x/y swap intentional; `rotationX` is yaw; radar render uses `+0.5` yaw offset, arc geometry does not; bodies are symmetric `± length/2`; use `_smooth_yaw` for arcs; `acc_speed` is ACC-only; consumers must not open the traffic shared-memory buffer or mutate `Vehicle` instances.
- **Elevation** (`core/radar/README.md` §15): one shared gate in `core/radar/elevation.py`, stepped by `RadarThread` and published as `off_surface_ids`; AEB and ACC consume it and must never re-derive a pitch-window test of their own. Traffic `position.y` is a body datum (`0.58 * size.height` above its road), not the road itself, and traffic euler pitch is the negated road grade with only its ego-axis component usable; roll is not a cross-grade signal. The gate fails open on every uncertain input (unusable rotation, ego pose, body height, ego history), marginal failures need repeated frames, and latched AEB threats bypass it entirely. `cal.elevation_margin` is dead: kept for clip-metadata compatibility, not a tunable.
- **AEB** (`core/aeb/README.md` §9): yaw-rate ego curvature only (never `RadarData.ego_curvature`); target κ via `_vehicle_curvature_blend` stepped once per vehicle per frame; all tunables in `AEBCalibration`; `lane_frame.project_to_ego_arc` is the lane primitive; TMP trailer kinematics via shallow-copy swap; two-layer pedal authority (FF assist + closed-loop `AEBDecelController` tracking, no engagement slam: it pinned the pedal at 1.0 and made the controller dead code), gas ≥ 0.8 is the only AEB override; the brake build-up pad (`stop_buffer_response_s`, trailer variant) is what pays for actuation lag now that AEB tracks its target, so it must never go back to 0; engagement-entry vetoes (`_los_veto_bar`, `_extrapolation_veto`, lane-confidence range) feed `engage_vetoed_ids` and must never reach FF, disarm, or the holds, and may reach warn only as the `aeb_warn_confirm_vetoed_s` persistence window (delay for a fully-vetoed out-of-lane set, never removal from the warn aggregate), and a measured CBDR miss outranks arc-projected lane membership in both directions.
- **ACC** (`core/acc/README.md` §8): never mutate `Vehicle`; history-fit ego curvature (not AEB's yaw-rate proxy); no control law in this module (cruise owns accel); scoring stays meter-native; the geometric lead failsafe (§10) reads raw ego-arc geometry only and floors the published score rather than `st.score`, so do not couple it to the road model, the blended lateral, or the score integrator it exists to survive.
- **ACC gap law** (`core/acc/ACC_ARCHITECTURE.md` §8.5, §8.6): the §8.3 blend is one-directional and discards `a_cah` whenever `a_iidm >= a_cah`, which left `a_lead` no path to the command at all. Closed by the bounded feedforward in §8.6, which keys on `cah(a_lead) - cah(0)` so it is identically zero for a lead that is not braking. Do not re-fix it by making CAH bind downward: that exit was costed and rejected in §8.5 because it trades string stability, a first-class requirement (§5). Do not re-add the claim that CAH covers the matched-speed hard-brake case. Do not give the feedforward a distance ramp: `cah` already folds in gap and both speeds, and a second range term double-counts. Keep `lead_brake_ff_max_ms2` at or below `b_comfort`, or laggy TMP `a_lead` can drive a slam.
- **ACC transition smoothness** (`core/acc/ACC_ARCHITECTURE.md` §8.7, §12.2): hard clamps within ~1 m/s^2 of `a_lead = 0` rectify telemetry jitter into a one-sided mean brake, so the soft corner, the soft floor and the ramped `a_lead` tau all exist to remove steps in `d(cap)/d(a_lead)`. The `max(a_acc, a_cah)` inside the floor stays hard on purpose: a C1 max can only approach from above, which relaxes the kinematic floor. `lead_law_floor_soft_ms2` is a measured optimum and is not monotone, 0.18 is worse than no softening at all, so re-measure with `tools/acc_transition_probe.py` before moving it and average over seeds.
- **ACC closing-speed relief** (`core/acc/ACC_ARCHITECTURE.md` §8.11): gap-error braking is halved while ego is level with the lead or falling back, ramping to normal by 2 km/h of closing. Driver-requested. Above 2 km/h the command is bit identical, which is the whole point: every approach and lead-braking measurement in §8.6 to §8.10 lives above that line, so do not widen the ramp without re-running them. It multiplies the comfort term only and the §8.4 floor `max(a_acc, a_cah)` still binds under it. That floor is **not** `a_cah` alone: beyond ~30 m the blend has already relaxed CAH, so `a_acc` is the higher of the two. Costs 0.2 to 0.3 m of minimum gap in hard-braking cases under 25 m; the overlay trips at those gaps are pre-existing, not caused by the relief.
- **ACC accel-side nudge** (`core/acc/ACC_ARCHITECTURE.md` §8.8, §8.9): the `tanh` blend is cubic in `a_cah` at `a_iidm = 0`, so a gently accelerating lead used to command almost the same as a coasting one. §8.9 adds a gated nudge for responsiveness, share 0.50, bound 0.75 m/s^2. The bound cannot bind (`demand` is already capped at `a_max`), so ego's own authority is the limiter and the **closing-speed gate is the load-bearing guard**: it fades the nudge to zero by 2 m/s of closing, which is what stops a phantom `a_lead` adding pull toward a lead ego is catching. Do not drop that gate, and do not remove either soft corner (zero and `a_max`): without the first, jitter rectifies into standing pull; without the second, raising the share doubles the gain step at `a_lead = 1.5`. The share was 0.25 until the gap cost was re-priced: the original 15.65 m figure came from a test holding `v_lead` constant while reporting sustained `+1.0`, which is self-contradictory, and the honest cost is 0.2 m. **The real open risk is that the nudge takes no distance argument** and can invert a small brake into a small pull at close range (6 m, 2 m/s, lead `+1.0`: -0.096 becomes +0.451). Add the range term before raising the share again.
- **ACC feedforward input** (`core/acc/ACC_ARCHITECTURE.md` §8.10): both feedforwards read `a_lead` through their own slower symmetric filter (`tau_alead_ff_s`, carried on `_LeadSnapshot.a_lead_ff_ms2`); CAH keeps the fast asymmetric one because it sizes a stopping requirement. Do not collapse them back to one estimate: a static gain on the raw `a_lead` put commanded peak jerk on the 2.5 m/s^3 limiter in steady following. The virtual lead must build its feedforward input the same way or the anticipation delta carries the noise back. Any A/B of these features must zero every knob in `tools/acc_transition_probe.BASELINE`, pinned by `test_every_feature_knob_disables_its_feature_at_zero`: an earlier version omitted the two shares and reported an improvement where there was a 24x regression.
- **Longitudinal telemetry** (`core/longitudinal/README.md`): `lv_accelerationX` is the truck-local **lateral** axis (right/left), not longitudinal, despite its name and the SDK's own labeling. Using it as an accel/decel feedback signal causes phantom braking on right turns and phantom acceleration on left turns, worsening the longer CC stays in the turn (the integral term amplifies it). Use a tracking differentiator on `speed` instead (`_spd_smooth` in `core/sending_thread/thread.py`, `_ACCEL_TRACK_TAU_S` in `core/longitudinal/limiter.py`) — not raw tick-to-tick `d(speed)/dt`, which spikes because ETS2 physics runs at ~20Hz against a faster control loop. Enforced by `tests/invariants/test_source_invariants.py`.

### Longitudinal control invariants

- **CC and Limiter are mutually exclusive sibling controllers.**
  - `Settings.cc_mode` selects which controller steps each tick: `"Cruise control"` → `CruiseController` (+ `AdaptiveCruiseController`); `"Speed limiter"` → `SpeedLimiter`. Both live in `CruiseControlThread`.
  - The CC FSM (`enable`/`disable`/`set_target_kmh`) drives `_cc_ctrl` in both modes. In limiter mode the CC's enabled state and target are forwarded to the limiter as its cap.
  - On mode flip, the now-inactive controller's PID state is reset to avoid stale integrators on re-entry.

- **Disengage conditions (brake, park, reverse, disarm-on-stop) are CC-only.**
  - `CruiseControlThread._handle_cc_disengage_conditions()` is called inside `if mode == "Cruise control"` only. The limiter never sees these events.
  - Neutral does not disengage CC (simulated manuals flash N while shifting). While `gear_dashboard == 0`, CC/ACC positive m/s² bids are clamped to 0; after 2 s continuous N with gas cut, a rate-limited popup explains that CC can't accelerate in neutral. Exception: while `sending_thread` publishes `auto_neutral_holding`, the clamp is skipped so the launch bid can shift back to drive.
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

- **Slope compensation is mapper feed-forward, not the cruise PID.**
  - `AccelToPedals` adds `g sin(θ)` (plus rolling and aero) to the bid. CC and the limiter only track speed error.
  - Do not add pitch or grade terms to `CruiseController` or `SpeedLimiter`. A second copy in the PID fights the mapper FF and still waits on speed error if the FF is wrong. Enforced by `tests/invariants/test_source_invariants.py`.

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
- **Limits:** this is a soft prior, not a guarantee — it catches recognizable categories, not every community-specific landmine. Mechanizable risks belong in CI, and the ones that already are live in `tests/invariants/`; this section is the backstop for what CI can't check (optics, tone, feature merit, community vibe).

### Packaging and antivirus false positives

MonoCruise ships unsigned (no code-signing certificate) and will remain unsigned until there is monetization to justify one. Unsigned binaries already draw more antivirus scrutiny.

- When touching the updater (`updater/`), installer (`installer/MonoCruise.iss`), the background checker (`checker/`), or the release/build pipeline (`.github/workflows/release.yml`), prefer conventional, well-documented approaches: standard HTTPS downloads, standard file replacement. Avoid patterns that read as malware to AV heuristics: packing/UPX, self-modifying or self-updating executable tricks, process/memory-injection-like behavior, `shell=True` process enumeration, or writing Run/startup registry keys from anywhere but the (visible, consent-based) installer.
- If a risky-looking pattern is genuinely necessary, flag it explicitly rather than shipping it silently.
- **PyInstaller 6 resolves relative paths in `.spec` files against the spec file's own directory, not the invocation cwd.** A spec under a subdirectory (e.g. `updater/updater.spec`) must build paths from the `SPECPATH` global, not repo-root-relative strings, or CI builds silently break on PyInstaller ≥6. Specs at repo root (`monocruise.spec`) are unaffected.

