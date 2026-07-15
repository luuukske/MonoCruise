# Changelog

All notable changes to MonoCruise are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Sections under each release: `Added`, `Changed`, `Fixed`, `Removed`, `Security`.

The `[Unreleased]` block accumulates changes between releases. `tools/release.py bump`
renames it to the released version and starts a fresh `[Unreleased]` above it.

## [Unreleased]
### Added
- **Pedals lost banner**: banner now shows your pedals disconnected, just like v1.0.4.
### Fixed
- **Connect pedals actually work**: wired up the connect button to be more reliable and actually work.
- **AEB general improvements**: capsule bodies + ego-arc steering wiggle were beeping on adjacent-lane overtakes and passes; parallel-margin scaling, rear-overtaker filter suppress, and braking-worsens for cleared rear overtakers kill the class-A/B phantoms (clips f0b2ace6 etc.).
- **ACC oscilate at full throttle**: ACC would be switching between 95% throttle and 100% caused by the downshifting.
### Changed
- **ACC follow distance**: made the distance more realistic. the original values were placeholders.
- **AEB less sensitive**: changed back the AEB brake latch from 70% max brake to 80% max brake.
- **ACC smoother in SP**: SP now uses the same filtering from TMP as it showed so much success, i wanted to move the two systems together. only crash detection and lag detection stays for TMP only.
- **Less AEB shadow_near**: changed the shadow near to only clip about 6/h.
- **Limiters more responsive**: limiters react faster to speed changes and hold their speed more accuratly. 

## [1.1.0-preview.4] - 2026-07-11
### Fixed
- **AEB corner filtering**: filtering for AEB improved **MASIVELY** using clips gathered from AEB clipper (testers only), mostly focusing on corners, but general improvements can be seen.
- **AEB improved brake capacity**: improved the stability of the brake capacity estimation for correct AEB triggering. this prevents late reaction from the AEB system previously seen in one of the clips.
- **Highway slow queue FN**: fixed a (really really bad) bug in the AEB filtering causing slow moving traffic to be ignored when above certain high speed. also found thanks to the clipping.
- AEB clipping irregularities
- **OPD known issues**: reverted to legacy code. sending_thread caps opdgasval now, not just raw gas input. opd offset actually effects the gas output now. hard to get moving at slow speeds.

## [1.1.0-preview.3] - 2026-07-09
### Added

- **AEB clip capture** (debug mode only): when AEB triggers, MonoCruise saves a short replay clip plus a screenshot thumbnail to `%LocalAppData%/MonoCruise/`. Intended for testers to report false positives or missed detections. Send clips manually (no automatic upload). You'll get an on-screen notification when a clip is saved.

### Changed

- **Debug mode off by default**: developer tools (AEB radar view, clip capture) now require enabling debug in settings. Preview builds previously had this on.
- **Updater hands off after installing**: once an update completes, the updater shows the finished state for a moment, starts MonoCruise and closes itself.

### Fixed

- **AEB false triggers in corners**: cross-traffic that sweeps clear at intersections no longer fools the threat filter, and AEB won't engage when the target's movement already shows it will pass beside you. (thanks to eary AEB clips captured by me)
- **Updater self-updates now actually apply**: the updater's own new version used to be staged but never swapped in. A file the running updater keeps open blocks the swap, and the staged update was silently discarded afterwards. MonoCruise now applies the staged updater files once the updater has closed. This also removes the broken in-place swap that could leave an old updater unable to reach GitHub.

## [1.1.0-preview.2] - 2026-07-06

### Added

- **Updater closes MonoCruise for you**: clicking Update while MonoCruise is running now asks the app to shut down cleanly (settings saved, pedals released) instead of showing an error. If it will not close within 15 seconds, the old "Close MonoCruise before updating" message still appears.

### Changed

- **Failed updates show in red**: when an update fails, the stage it failed on (download/install) turns red in the updater's progress column.

### Fixed

- **Updater window icon**: the updater now shows the MonoCruise icon in its title bar and on the taskbar instead of the default icon.
- **AEB debug view no longer opens on every start**: the developer radar view now only appears in debug mode.

## [1.1.0-preview.1] - 2026-07-06

A ground-up rewrite focused on **stability** and **performance**, with a more reliable take on every existing feature: plus a built-in updater and on-screen notifications.

### Added

- **In-app updater**: installs new releases from GitHub without reinstalling; your config and logs are kept.
- **Stable / Preview update channels**: pick your channel in settings — Preview builds are released earlier and may contain bugs.
- **On-screen notifications**: always-on-top popups for updates, errors, and onboarding tips.
- **Lead-vehicle speed readout**: the lead truck's speed now shows on the CC panel above your set speed.
- **Multi-device button assignment**: cruise-control buttons can be assigned to any joystick button, hat direction, keyboard key, or USB button device (e.g. a button stalk) — click a configure button in settings and press the input. Assigned buttons light up while pressed, and an Unassign button clears a single binding.
- **ETS2 v1.60 support** (thanks to the automatic SDK fetcher).
- **Rewritten auto-start checker**: now simpler and antivirus-friendly — uses telemetry for game detection (no process or registry scanning), with installer-based startup opt-in and a plain-language log. Details in `checker/README.md`.
- **Global speed limiter**: a highly accurate global limiter so your truck never exceeds that set speed (useful for Trucky, for example).

### Changed

- **Keeps running when something breaks**: rewritten to one independent thread per subsystem, with a watchdog that detects a crashed or frozen part and restarts it automatically.
- **Faster, smoother UI**: switched from CustomTkinter to GPU-accelerated PySide6, lowering CPU usage and clearing a class of visual bugs.
- **More reliable ACC**: reworked lead-vehicle selection (arc-based in-lane scoring) sharply reduces the brake-checking the old ACC was prone to, and now accounts for road-trains (trailers-of-trailers).
- **Reworked AEB**: arc-trajectory geometry with staged braking (warning brake, then full brake) in place of the old straight-line check.
- **Self-tuning pedals**: gas/brake output now calibrates to your hardware over time (with per-gear learning), plus road-load/hill feedforward and a gearshift hold for smoother, more consistent control.
- **Automatic SDK installer**: automatically fetches the latest SDK for the latest ETS2/ATS version.
- **Thread-safe settings & logging**: settings save atomically with no global variables (faster and race-free), and errors can still surface via popup even after a worker thread crashes.
- **Much smaller downloads**: the installer and update packages no longer bundle unused Qt components (roughly 75% smaller).

### Fixed

- **Hazard flickering** during AEB braking.
- **CC vs. brake conflicts**: game braking now reliably disengages CC, and CC / limiter / user-pedal priority no longer fight each other.
- **CC panel on every display**: scale changes apply live (no restart), and the panel stays put on 4K and across mixed-DPI monitors.
- **Pedal bar misplacement** after waking from sleep.
- **One-Pedal braking** under speed-limiter mode.
- **Hazards** sometimes not switching off on acceleration.
- **Popup crash** on early `getattr()` calls.
- **Single-instance check**: MonoCruise now uses a named mutex instead of a process-name scan (which matched the app's own process in packaged builds); a second copy exits cleanly.
- **Release builds on PyInstaller 6**: the updater spec's script path stopped resolving under PyInstaller 6's spec-relative path rule.

**Known issues**

- ACC gap level can't be changed while actively following a lead vehicle: runtime adjustment is coming in a future update.
- AEB is experimental and can false-trigger in corners and during lane changes, so it is **disabled by default** — enable it in settings. **Use with care**.

