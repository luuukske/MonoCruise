# Changelog

All notable changes to MonoCruise are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Sections under each release: `Added`, `Changed`, `Fixed`, `Removed`, `Security`.

The `[Unreleased]` block accumulates changes between releases. `tools/release.py bump`
renames it to the released version and starts a fresh `[Unreleased]` above it.

## [Unreleased]

A ground-up rewrite focused on **stability** and **performance**, with a more reliable take on every existing feature: plus a built-in updater and on-screen notifications.

### Added
- **In-app updater**: installs new releases from GitHub without reinstalling; your config and logs are kept.
- **Stable / Preview update channels**: pick your channel in settings — Preview builds are released earlier and may contain bugs.
- **On-screen notifications**: always-on-top popups for updates, errors, and onboarding tips.
- **Lead-vehicle speed readout**: the lead truck's speed now shows on the CC panel above your set speed.
- **ETS2 v1.60 support.**

### Changed
- **Keeps running when something breaks**: rewritten to one independent thread per subsystem, with a watchdog that detects a crashed or frozen part and restarts it automatically.
- **Faster, smoother UI**: switched from CustomTkinter to GPU-accelerated PySide6, lowering CPU usage and clearing a class of visual bugs.
- **More reliable ACC**: reworked lead-vehicle selection (arc-based in-lane scoring) sharply reduces the brake-checking the old ACC was prone to, and now accounts for road-trains (trailers-of-trailers).
- **Reworked AEB**: arc-trajectory geometry with staged braking (warning brake, then full brake) in place of the old straight-line check.
- **Self-tuning pedals**: gas/brake output now calibrates to your hardware over time (with per-gear learning), plus road-load/hill feedforward and a gearshift hold for smoother, more consistent control.
- **Thread-safe settings & logging**: settings save atomically with no global variables (faster and race-free), and errors can still surface via popup even after a worker thread crashes.

### Fixed
- **Hazard flickering** during AEB braking.
- **CC vs. brake conflicts**: game braking now reliably disengages CC, and CC / limiter / user-pedal priority no longer fight each other.
- **CC panel on every display**: scale changes apply live (no restart), and the panel stays put on 4K and across mixed-DPI monitors.
- **Pedal bar misplacement** after waking from sleep.
- **One-Pedal braking** under speed-limiter mode.
- **Hazards** sometimes not switching off on acceleration.
- **Popup crash** on early `getattr()` calls.

**Known issues**
- ACC gap level can't be changed while actively following a lead vehicle: runtime adjustment is coming in a future update.
- AEB is experimental and can false-trigger in corners and during lane changes. **Use with care**.
