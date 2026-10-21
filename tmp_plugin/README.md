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
| 0 | version | u32 | layout version, 2 (1 up to plugin 1.0.4, same fields with offset 14 unused); 0 = no writer |
| 4 | heartbeat | u32 | +1 per write, skips 0 |
| 8 | connected | u8 | connected to a TruckersMP server |
| 9 | in_no_collision_zone | u8 | `Gameplay().OnNoCollisionZone`; cleared on disconnect and shutdown, not on connect (see below) |
| 10 | players_streamed | u16 | stream-in minus stream-out events, diagnostics only |
| 12 | players_collidable | u16 | always 0 (not measured) |
| 14 | zone_events | u16 | +1 per `OnNoCollisionZone` event, wraps; zeroed on shutdown only. Lets the reader's teleport guard see that TruckersMP judged the zone after a jump (`core/radar/README.md` §18) |

`truckersmp_shutdown` zeroes the file, so MonoCruise reads "no data" at once
rather than waiting for the heartbeat to go stale.

The SDK has no "am I in a zone" getter, only the enter and leave event. Spawning
inside a zone left the flag at 0 in 1.0.0 (2026-10-10), which cleared it on
`OnConnected`; the event may arrive before that. Since 1.0.1 only a disconnect
clears it, and every connect, disconnect and zone event is written to the
TruckersMP client log (`[MonoCruise] ...`), including "Left a no-collision zone
that was never reported entered", the mark of a spawn TruckersMP did not report.

## Experimental vehicle block (1.0.3)

`Local\MonoCruiseTmpVehicles`, written beside the state file at 20 Hz (1.0.4; it was
every rendered frame before, but samples arrive at ~4.5 Hz so nothing is lost).
`RadarThread` reads it for TMP speed and the lag / crash split
(`core/radar/README.md` section 19), `tools/tmp_sdk_probe.py` records it. Failing to
open it leaves the zone state untouched. Its heartbeat is the state file's heartbeat
at the time of the write, so it advances in steps of about three.

Header (24 bytes): version (1), heartbeat, write time (QPC, microseconds), own
ping (always `0xFFFF`, see below), slot count, connected.
Then up to 96 slots of 80 bytes, nearest rig first within 250 m, the local rig
included: placement (double position, float quaternion), `GetLinearVelocity`,
`GetAngularVelocity`, age of the player's last network sample in ms (`-1` none
seen), samples since stream-in, distance, kind (tractor, trailer, local tractor,
local trailer), flags and a per-write rig index pairing a trailer with its tractor.

Sample arrival comes from `Player().OnUpdate`, which only stores a timestamp per
opaque handle; the handle is never published. The heartbeat field reads 0 while
the slots are written, so a reader that sees the same non-zero value before and
after its copy has one consistent write. No ids, names or per-player latency.

**Never call `Network().GetLastPing()`.** Off a server, client 0.7.7.11 dereferences
null inside it: 1.0.2 crashed the game at the main menu on its first frame
(2026-10-11). Player queries run only while connected. Faults are handled by the
guard described under "Fault handling" below.

## Fault handling (1.0.4)

The plugin runs inside the game, so a client bug must cost MonoCruise its TMP data and
nothing else. Every SDK call and every callback runs on the game's main thread, and the
docs say nothing about exceptions in callbacks or shutdown order, so the plugin assumes
anything can fault.

- **One guard.** `guard()` (SEH plus `try`/`catch`) wraps every SDK call in
  `truckersmp_init`, every event handler (connect, disconnect, zone, stream in and out,
  update, pre-render) and `truckersmp_shutdown`'s listener removal. The first fault sets
  `g_dead`: the shared memory is zeroed (readers see "no writer", which keeps the zone
  flag at 0 and AEB watching every player), one line goes to the TruckersMP log
  (`[MonoCruise] A TruckersMP call faulted (0x... in <where>); plugin off until restart.`),
  and from then on every entry point returns at once without an SDK call. A new
  `truckersmp_init` starts clean. Debugger and thread-name exceptions are not faults.
- **Nothing is destroyed at detach.** The session and the sample table are leaked
  pointers, so a client that unloads the DLL without `truckersmp_shutdown`, or a process
  exit, never runs `~Session`, which calls `RemoveListener` into a client that may be
  gone. `truckersmp_shutdown` unregisters the listeners (guarded; skipped when dead) so a
  client that unloads the DLL afterwards holds no stale callback, but never deletes the
  session. `DllMain` only zeroes the shared memory at detach.
- **Shared memory is never unmapped.** Shutdown and detach zero the contents; the
  mappings live until process exit, so a late callback or a reader never meets freed
  memory.
- **Vehicle block cost.** At most one publish per 50 ms. Collecting players gets 0.8 ms
  and reading rigs 1.2 ms (nearest first); past either budget the farthest rigs are left
  out and one warning is logged. Connect and disconnect publish at the next frame.
- **Client minimum.** The vehicle block runs only when `Core().ClientVersion()` parses as up
  to four dotted numbers (a leading `v` is ignored) at or above `kMinClient` in `main.cpp`,
  today `0.7.7.11`. An older client, an unreadable string (`0.7.7.11-beta`, `banana`) or no
  reported version runs zone-state-only and logs `client <version> is not at or above the
  tested 0.7.7.11 or is unreadable; zone state only` once. Newer clients get the block
  untested (Lukas chose a minimum over an exact list); raise `kMinClient` if a release
  changes SDK behaviour. The format of the string `ClientVersion()` returns was not seen
  on a real client when this was written, so check the log line on the first drive.

What the guard cannot cover, so do not read "crash-proof" into it:

- **Heap corruption.** A client that scribbles on the heap crashes later, somewhere else,
  and no handler here sees it.
- **Stack overflow.** The filter would run on the exhausted stack; treat it as fatal.
- **Faults on client threads.** The guard catches a fault only inside our own callback
  frame. A client thread that dies before or after calling us is not ours to catch, and
  a wild pointer inside `OnUpdate` (which may run off the render thread) is caught but
  proves nothing about the thread's other state.
- **A client lock left held.** A swallowed fault inside the client can leave one of its
  locks taken or its state half-updated. We then stop calling it, apart from the single
  log line, which is itself a client call and could block on such a lock.
- **Our own lock.** A fault while holding the sample table lock on another thread leaves
  it held; the render path checks `g_dead` first, and shutdown only try-locks it, but a
  render call already waiting on it would hang.
- **After a fault, listeners stay registered.** Making no SDK call includes
  `RemoveListener`. The callbacks stay valid as long as the DLL stays loaded and return at
  once, but a client that unloads a dead plugin relies on its own listener cleanup.

Not built: a crash-loop breaker (a sentinel file written before the first vehicle publish
and removed on clean shutdown, so a session that died with the block live starts the next
one zone-only). It makes the plugin write a file, which is an antivirus and optics
question (`AGENTS.md`, "Packaging and antivirus false positives"), so it waits for an
explicit decision.

## Fault-injection harness

`tmp_plugin/tests/fault_host.cpp` is a fake TruckersMP client. It loads the DLL that
MonoCruise ships (`core/sdk_installer/own/`) and hands it function tables that fault
(access violation or C++ exception) in each SDK call the plugin makes, return junk (a
list of 4 billion entries over a bad pointer, a null list with a count, a version string
at an unreadable address), return failure, fire callbacks with null, short and unreadable
payloads, from another thread, after `truckersmp_shutdown`, and after the client is gone
(`FreeLibrary` without shutdown). Any call to a player identity, account, latency,
collision or `GetLastPing` function ends the run. A scenario passes when the process
survives, the plugin makes no SDK call after its first fault, and the shared memory ends
zeroed. `tests/test_tmp_plugin_faults.py` builds the host with MSVC and runs every
scenario in its own process.

The host patches the DLL's `CreateFileMappingW` import so the plugin writes to private
mapping names: a test must never write into a running game's real shared memory. Run by
hand:

```powershell
cl /nologo /EHsc /MT /I tmp_plugin/vendor/truckersmp_sdk/include tmp_plugin/tests/fault_host.cpp /Fe:fault_host.exe
.\fault_host.exe --list
.\fault_host.exe core\sdk_installer\own\monocruise_tmp.dll fault.Vehicle.GetPlacement.crash
```

Exit 0 is a pass; 91 a forbidden call, 92 a `RemoveListener` after the client was gone,
93 a log after the client was gone, 94 a mapping name the host does not know, 95 any other
client call after the client was gone, 98 a hang, 99 a fault that escaped the plugin. The
test is marked `needs_msvc` and CI deselects it (CI is Linux). Run against 1.0.3, 45 of
the 56 scenarios fail, so the harness does have teeth.

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
so the same source and toolchain give the same bytes. 1.0.5 was built without
CMake, by `cl` / `rc` / `link` with the flags above and CMake's Release defaults;
the code section matched a CMake build of 1.0.4 byte for byte but the file did not,
so a CMake rebuild of 1.0.5 needs the SHA updated.

To ship a new build:

1. Copy `monocruise_tmp.dll` to `core/sdk_installer/own/`.
2. Put its git-blob SHA in `OWN_PLUGINS` (`core/sdk_installer/own.py`):
   `git hash-object core/sdk_installer/own/monocruise_tmp.dll`.
3. Run `pytest tests/test_sdk_own_plugins.py tests/test_tmp_ncz.py
   tests/test_tmp_sdk_motion.py tests/test_tmp_plugin_faults.py`.

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
| World velocity | `Vehicle` / `Trailer::GetLinearVelocity` | TMP speed is an LS fit over ~1.3 s of positions (radar §7): late on a braking lead, rippled by netcode. Direction of travel for reversing targets (`travel_sign`) | Measured 2026-10-11 (below): the sender's own velocity, carried in each packet |
| Yaw rate | `Vehicle` / `Trailer::GetAngularVelocity` | Target curvature without differentiating poses; crash rotation gates | Same |
| Truck to trailer link | `Vehicle::GetTrailer`, `Player::GetTrailer` | Replaces the geometric pairing in ACC `trailer_lock`, AEB `_find_tractor_for_trailer` and the exit hold | Map trailer handles to `ETS2LATraffic` records by pose |
| Exact body | `GetBoundingBox` + `GetPlacement` | Body size and centre offset; the TMP trailer pivot shift (`Trailer.correct_position`) | Box includes mirrors? |
| Network sample arrival | `Player().OnUpdate` (timestamp only) | Tells a fresh network sample from client interpolation: lag freeze, pose jumps | Measured 2026-10-11 (below): ~4.5 Hz |
| Per-player ghost flag | `Player::CanCollideWith` | Per-player suppression beyond the exit hold | Read "cannot" for real trucks on 2026-10-06; re-measure with poses side by side |
| Own link quality | `Network().GetLastPing` | Ego-side lag for TMP confidence | Correlation with remote lag |
| Track birth and death | `OnStreamIn` / `OnStreamOut`, `Vehicle().OnSpawned` / `OnDespawned` | Cold starts and id reuse without guessing | None |

### Measured (2026-10-11, 1.0.3, one 4.5 min ETS2 drive, 22 TMP trucks)

`tools/tmp_sdk_analyze.py`; a short drive, so the braking numbers are thin.

- **Velocity is the sender's.** It changed on 0 of 30,738 frame pairs without a new
  packet. For the local truck it matches SCS telemetry speed to 0.02 m/s (p90), so the
  axes and units are right. Body positions match `ETS2LATraffic` to 0.2 m.
- **Samples arrive at ~4.5 Hz** (interval p50 217 ms, p99 434 ms). The client
  extrapolates a late player for ~300 ms, then stops it: positions still move at
  250-300 ms of age, 25% move at 400-450 ms, none past 450 ms. All 10 position stalls
  of a moving truck had a sample gap over 400 ms; none happened with fresh samples.
- **Radar speed collapses on those gaps.** Of 47 gaps over 300 ms on moving trucks, the
  radar speed fell more than 2 m/s in 22 and more than 4 m/s in 9 (worst 22 to 5 m/s
  and back up to 28) while the SDK velocity held. The lag freeze opened on 2 of them:
  `_raw_brake_active` was latched on 37% of moving TMP updates by position jitter, and
  its hard-brake veto blocked 31 of 49 lag entries.
- **Gating the freeze on sample age is not enough** (offline A/B): drops over 4 m/s go
  from 9 to 6, because the dip comes after the stall, from the position fit spanning
  it. Moving the stalled body is wrong too: the client shows it frozen, which is
  where ego can hit it. The fix is the speed source: SDK velocity for matched TMP trucks.
- **Braking lead** of SDK speed over radar speed: p50 0.22 s, but on 5 events only.

Excluded: names, tags, Steam and account ids, staff and patron flags, mounted
packages, input, and per-player latency (identity-adjacent; needs an explicit
decision). `UserInterface().ShowNotification` could show AEB messages in game, but
that is a driver-distraction question, not AEB data.
