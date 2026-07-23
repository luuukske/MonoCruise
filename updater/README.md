# MonoCruise updater

Separate executable; cannot import the main app. Reads install-root state from files the running app writes: `config.json` (update channel) and `installed_version.txt`.

## Self-update layout (install root)

| Path | Role |
|------|------|
| `updater/` | This program and support files |
| `updater_pending/` | New updater files staged by an update (sentinel written last) |
| `updater_old/` | Previous updater files parked by a swap |
| `update_staging.tmp/` | Scratch extract dir before live files are touched |

This exe only **stages** its own update into `updater_pending/`. The swap into place runs in MonoCruise after the updater exits (`shared/updater_swap.py`: in-process swap is blocked while zipimport holds `base_library.zip`). No helper script or watcher: unsigned exe rewriting exes is an AV false-positive pattern.

Directory and sentinel names must match `shared/updater_swap.py` (tests assert this).

## Shared imports

Markdown renderer and dropdown live at repo root (`shared/`), bundled via `updater.spec`. Repo root is added to `sys.path` for source and frozen builds.

## Update safety

Paths under install root in `_PRESERVE_PREFIXES` are never overwritten or deleted. Root-level updater files from the old flat zip layout are skipped during extract; current zips ship `updater/` and stage via `_move_into_place`.
