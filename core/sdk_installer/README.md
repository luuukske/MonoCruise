# SDK installer (backend)

Detects, fetches and installs the ETS2 / ATS SDK DLLs that MonoCruise needs to
talk to the game. Backend only - it shows no UI. A front-end drives it and
presents the result (the popup).

## Managed files

Pulled from the ETS2LA repository
(https://github.com/ETS2LA/ETS2LA - client-side download is permitted by the
maintainers and the license), folder `Assets/SDKs/<game version>/Windows`, where
`<game version>` is the engine version the installed game reports (see **Game
version detection**):

| file | role |
| --- | --- |
| `scs-telemetry.dll` | telemetry out of the game (speed, position, ...) |
| `scs_sdk_controller.dll` | input into the game (gas, brake, ...) |
| `ets2la_plugin.dll` | AI / MP vehicle data for ACC and AEB (version-specific) |
| `ets2la_<version>` | marker ETS2LA uses to detect the installed SDK version (version is in the filename) |
| `sources.txt` | courtesy: documents where each DLL comes from |

They are installed into each game's `bin/win_x64/plugins` folder. ETS2LA installs
the same files, so the two coexist - matching files are simply left in place.

## Superseded plugins (`LEGACY_FILES`)

MonoCruise 1.0 shipped its own copies of two plugins under different filenames:

| old file | superseded by |
| --- | --- |
| `input_semantical.dll` | `scs_sdk_controller.dll` |
| `ets2_la_plugin.dll` | `ets2la_plugin.dll` |

Because the names differ, upgrading leaves the old copy in the plugins folder and
the game loads both. For the input plugin that is fatal: both builds register the
same SCS input device id (`laneassist`, shown as "ETS2 Lane Assist"), so the
second registration fails with `Unable to register device` in the game log, and
plugin load order is alphabetical, so the **old** copy wins. It then reads
`Local\SCSControls` with the 1.0 layout (62 bytes, `aforward` at offset 4) while
MonoCruise writes the current one (342 bytes, `aforward` at offset 122). Gas and
brake land on bytes nothing reads. Telemetry is a separate plugin and keeps
working, so the app looks perfectly healthy while the pedals do nothing.

`check()` reports these in `GameSdkState.conflicting` (a local file test, no
network) and `needs_action` covers them, so a stale install is repaired on the
next boot. `apply()` renames each one to `<name>.monocruise-disabled` and reports
it in `GameApplyResult.disabled`.

Renamed, never deleted: the file came from an older MonoCruise, the user can put
it back, and it keeps the AV story below intact. Windows allows renaming a DLL
the game currently holds loaded, so this needs no running-game special case; the
old plugin stays resident until the game restarts, which is what the popup asks
for.

## Boot policy

1. Find every ETS2 / ATS install and check which managed files are present, plus
   any superseded plugin left behind. This is local and fast; nothing hits the
   network.
2. The repository is consulted (one JSON request to the GitHub contents API per
   distinct game version, so normally one) only when a file is missing, or when
   `FORCE_REFETCH` is set and this build has not refetched yet. **When the DLLs
   are already installed on a normal build, no network call happens.** A game
   update alone already shows up here: the version marker carries the version in
   its filename, so `ets2la_1.60` is missing the moment the game becomes 1.61.
3. Update detection compares the git-blob SHA the API reports against the SHA of
   the installed file - no binary is downloaded just to check for an update.

## Game version detection

`ets2la_plugin.dll` is built against one engine version and finds the game's AI
and MP vehicle lists by scanning for byte patterns. Against any other version it
still loads, resolves nothing, and reports no vehicles: ACC and AEB simply never
see traffic, with no error anywhere. So the version is never assumed.

`detect_game_version()` reads the four-part file version out of the game
executable's Windows version resource (`eurotrucks2.exe` / `amtrucks.exe`) and
keeps `major.minor`, which is exactly how the upstream folders are named
(`1.60.1.7` -> `1.60`). It is read per install, so ETS2 on the open beta and ATS
on the stable branch each get their own plugin, and each `GameSdkState` /
`GameApplyResult` carries the `game_version` it was resolved against.

`DEFAULT_GAME_VERSION` is the fallback for the one case where the resource
cannot be read at all (`version_detected` is then False). It is not a supported
version list, and nothing else should treat it as one.

When the resolved version has no folder upstream, the install is reported as
`version_unsupported` with a plain `unsupported_reason`, and **no files are
installed for it**. Installing the neighbouring version instead is the exact bug
this replaced. The reason distinguishes both directions, since ETS2LA prunes old
SDK folders (1.57 and 1.58 went when 1.60 landed) as well as adding new ones:
a version above everything published reads as "newer than any published game
plugin", one below as "no longer published". The direction costs one extra API
call listing `Assets/SDKs`, made only on the failure path.

## Local cache (`sdk_cache/<version>/`)

Every verified download is kept, and `manifest.json` beside it records the
git-blob SHA each file was verified against. When the source has nothing for a
version - pruned upstream, or GitHub unreachable or rate limited - `apply()`
falls back to that cache and installs from it (`GameApplyResult.from_cache`),
provided it holds the complete tracked set for that version. This is what stops
an upstream prune from stranding a user whose plugin folder needs a repair;
`check()` reports it as `cache_available` and does not warn about the version.

Cache entries carry no download URL on purpose: a cached file is usable as it
stands or not at all, so a fallback can never turn into a fetch of something
that was never verified.

## Knobs (`manager.py`)

- `DEFAULT_GAME_VERSION` - only used when the game's own version cannot be read.
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
  deleting files we did not create. Superseded plugins we did install are
  renamed aside, not removed.

## Game paths (`game_paths.py`)

Windows-only install discovery (registry, Steam libraries, common paths). Non-Windows
imports succeed but return no games.


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
