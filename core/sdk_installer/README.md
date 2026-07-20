# SDK installer (backend)

Detects, fetches and installs the ETS2 / ATS SDK DLLs that MonoCruise needs to
talk to the game. Backend only - it shows no UI. A front-end drives it and
presents the result (the popup).

## Managed files

Pulled from the ETS2LA repository, folder
`Assets/SDKs/<game version>/Windows`
(https://github.com/ETS2LA/ETS2LA - client-side download is permitted by the
maintainers and the license):

| file | role |
| --- | --- |
| `scs-telemetry.dll` | telemetry out of the game (speed, position, ...) |
| `scs_sdk_controller.dll` | input into the game (gas, brake, ...) |
| `ets2la_plugin.dll` | AI / MP vehicle data for ACC and AEB (version-specific) |
| `ets2la_<version>` | marker ETS2LA uses to detect the installed SDK version (version is in the filename) |
| `sources.txt` | courtesy: documents where each DLL comes from |

They are installed into each game's `bin/win_x64/plugins` folder. ETS2LA installs
the same files, so the two coexist - matching files are simply left in place.

## Boot policy

1. Find every ETS2 / ATS install and check which managed files are present.
   This is local and fast; nothing hits the network.
2. The repository is consulted (one JSON request to the GitHub contents API)
   only when a file is missing, or when `FORCE_REFETCH` is set and this build
   has not refetched yet. **When the DLLs are already installed on a normal
   build, no network call happens.**
3. Update detection compares the git-blob SHA the API reports against the SHA of
   the installed file - no binary is downloaded just to check for an update.

## Knobs (`manager.py`)

- `SUPPORTED_GAME_VERSION` - game engine version the plugin targets (a matching
  `Assets/SDKs/<version>` folder must exist upstream). Bump per plugin release.
- `FORCE_REFETCH` - set `True` in a build that must re-pull the plugin even
  though the DLLs look installed (e.g. shipping a fix for a bad plugin build).
  Runs **once**: after the SDK is confirmed up to date the running MonoCruise
  version is recorded in `sdk_state.json` and later boots skip the network
  again.

## AV note

MonoCruise is unsigned, so "download a DLL and place it in another app's folder"
is exactly the shape heuristic scanners dislike. Mitigations kept deliberately:

- HTTPS only, from `api.github.com` / `raw.githubusercontent.com`, with a
  descriptive User-Agent and the shared `requests` dependency (same as the
  updater).
- Every download is integrity-checked: the git-blob SHA of the bytes must match
  the SHA the API reported before anything is written.
- Downloads land in a cache first, then move into place atomically; a partial or
  mismatched file never reaches the game folder.
- No process injection, no obfuscation, no executing downloaded content, no
  deleting files we did not create.

## Public API

`check_sdk()` / `start_boot_check(cb)` for detection; `SdkManager.apply(games,
close_running=..., on_progress=...)` to install once the user agrees. See
`__init__.py` for the exported names.

`apply(..., allow_running_missing=True)` is the boot auto-install mode: a
running game is not skipped, but only files it never loaded (absent on disk)
are written, so a live game is never touched in place. A present-but-outdated
DLL it holds loaded is returned in `GameApplyResult.deferred_running` for a
later close + reinstall. The startup path (`monocruise.py`) uses this to install
a missing plugin automatically and notify the user; replacing a loaded DLL still
goes through the consent-gated "reinstall SDK" action (`close_running=True`).
