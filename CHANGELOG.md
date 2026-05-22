# Changelog

All notable changes to MonoCruise are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Sections under each release: `Added`, `Changed`, `Fixed`, `Removed`, `Security`.

The `[Unreleased]` block accumulates changes between releases. `tools/release.py bump`
renames it to the released version and starts a fresh `[Unreleased]` above it.

## [Unreleased]

## [1.1.0] - 2025-07-02

### Added
- Adaptive Cruise Control (ACC) with intelligent driver model and 4 selectable gap levels.
- Automatic Emergency Braking (AEB) with arc-trajectory geometry and staged braking.
- One-Pedal Driving system with configurable exponential pedal curves.
- Auto-horn when braking hard; auto hazard lights when braking for traffic.
- Live braking/accelerating visualization bar.

### Changed
- Major architecture rewrite: thread-per-subsystem design with a watchdog and registry.
