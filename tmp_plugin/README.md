# MonoCruise TruckersMP plugin (`monocruise_tmp.dll`)

A TruckersMP Client SDK plugin that MonoCruise builds and ships itself. It
publishes the no-collision zone state AEB needs (`core/radar/README.md` §18).
ETS2LA's `ets2la_plugin.dll` stays stock: AI and TMP vehicle data still come from
there, and nothing here touches it.

It reads no game memory, so it does not break on ETS2 / ATS updates. It only has
to be rebuilt when the TruckersMP SDK changes, or when we want new data from it.

## State file

`Local\MonoCruiseTmpState`, 16 bytes, written on every rendered frame
(`Render().OnPreRender`), decoded by `core/radar/tmp_state.py`:

| offset | field | type | meaning |
|---|---|---|---|
| 0 | version | u32 | layout version, 1; 0 = no writer |
| 4 | heartbeat | u32 | +1 per write, skips 0 |
| 8 | connected | u8 | connected to a TruckersMP server |
| 9 | in_no_collision_zone | u8 | `Gameplay().OnNoCollisionZone`; cleared on disconnect and shutdown, not on connect (see below) |
| 10 | players_streamed | u16 | stream-in minus stream-out events, diagnostics only |
| 12 | players_collidable | u16 | always 0 (not measured) |
| 14 | reserved | u16 | 0 |

`truckersmp_shutdown` zeroes the file, so MonoCruise reads "no data" at once
rather than waiting for the heartbeat to go stale.

The SDK has no "am I in a zone" getter, only the enter and leave event. Spawning
inside a zone left the flag at 0 in 1.0.0 (2026-10-10), which cleared it on
`OnConnected`; the event may arrive before that. Since 1.0.1 only a disconnect
clears it, and every connect, disconnect and zone event is written to the
TruckersMP client log (`[MonoCruise] ...`), including "Left a no-collision zone
that was never reported entered", the mark of a spawn TruckersMP did not report.

## Rules

- **No player identity.** Never call `GetSteamID`, `GetAccountID`, `GetUsername`,
  `GetTagText` or the account module, and never write them anywhere. The plugin
  tells TruckersMP users it reads none, and MonoCruise clips can be shared.
- **No SDK calls inside `Player().OnUpdate`.** The SDK header says it is raised at
  network rate and must not call back into the SDK. Poll from `OnPreRender`.
- **Fail closed.** Anything that cannot run (no render module, no shared memory)
  must leave the zone flag at 0, which keeps AEB watching every player.
- **Layout changes move together.** Bump `kStateVersion` here and
  `_STATE_VERSION` in `core/radar/tmp_state.py` in the same change;
  `tests/test_sdk_own_plugins.py` checks that name, size and version agree.
- `truckersmp_init` returning false unloads the DLL without
  `truckersmp_shutdown`, so it cleans up first.

## Build and ship

From a VS 2022 x64 developer prompt (CMake and Ninja on `PATH`):

```powershell
cmake -S tmp_plugin -B <short build dir> -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build <short build dir>
```

Keep the build directory path short: MSVC fails with C1041 on very long ones.
The build is reproducible (`/Brepro`, static CRT, only `KERNEL32.dll` imported),
so the same source and toolchain give the same bytes.

To ship a new build:

1. Copy `monocruise_tmp.dll` to `core/sdk_installer/own/`.
2. Put its git-blob SHA in `OWN_PLUGINS` (`core/sdk_installer/own.py`):
   `git hash-object core/sdk_installer/own/monocruise_tmp.dll`.
3. Run `pytest tests/test_sdk_own_plugins.py tests/test_tmp_ncz.py`.

Installed copies whose SHA differs are replaced on the next boot, or deferred
until the game is closed (`core/sdk_installer/README.md`).

The SDK is vendored at `vendor/truckersmp_sdk` (v1.1.0, MIT, header-only). A
plugin built against a newer SDK than the player's client is refused, so update
it only when a feature needs it.

## Candidate SDK data for AEB (v1.1.3)

What the v1.1.0 SDK exposes that AEB could use. None of it is wired up yet. Each
item needs measuring against `Local\ETS2LATraffic` on real drives before AEB
trusts it, while ETS2LA's stock plugin still publishes TMP vehicles to compare
with.

| data | SDK | what it could fix | open question |
|---|---|---|---|
| World velocity | `Vehicle` / `Trailer::GetLinearVelocity` | TMP speed is an LS fit over ~1.3 s of positions (radar §7): late on a braking lead, rippled by netcode. Direction of travel for reversing targets (`travel_sign`) | Physics value, or the network-interpolated one? |
| Yaw rate | `Vehicle` / `Trailer::GetAngularVelocity` | Target curvature without differentiating poses; crash rotation gates | Same |
| Truck to trailer link | `Vehicle::GetTrailer`, `Player::GetTrailer` | Replaces the geometric pairing in ACC `trailer_lock`, AEB `_find_tractor_for_trailer` and the exit hold | Map trailer handles to `ETS2LATraffic` records by pose |
| Exact body | `GetBoundingBox` + `GetPlacement` | Body size and centre offset; the TMP trailer pivot shift (`Trailer.correct_position`) | Box includes mirrors? |
| Network sample arrival | `Player().OnUpdate` (timestamp only) | Tells a fresh network sample from client interpolation: lag freeze, pose jumps, the rel-speed floor | Rate and jitter per server |
| Per-player ghost flag | `Player::CanCollideWith` | Per-player suppression beyond the exit hold | Read "cannot" for real trucks on 2026-10-06; re-measure with poses side by side |
| Own link quality | `Network().GetLastPing` | Ego-side lag for TMP confidence | Correlation with remote lag |
| Track birth and death | `OnStreamIn` / `OnStreamOut`, `Vehicle().OnSpawned` / `OnDespawned` | Cold starts and id reuse without guessing | None |

Excluded: names, tags, Steam and account ids, staff and patron flags, mounted
packages, input, and per-player latency (identity-adjacent; needs an explicit
decision). `UserInterface().ShowNotification` could show AEB messages in game, but
that is a driver-distraction question, not AEB data.
