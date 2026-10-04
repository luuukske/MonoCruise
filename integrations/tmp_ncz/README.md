# MonoCruise NCZ Bridge for TruckersMP

A Windows x64 plugin that reports TruckersMP no-collision-zone (NCZ) entry and
exit events to a compatible MonoCruise client through local shared memory.
It uses the TruckersMP Game Client SDK; it is a third-party project, not an
official TruckersMP product.

## Download

1. Open this repository's **Releases** page on GitHub.
2. Open the release you want and expand **Assets**.
3. Download **MonoCruise-NCZ-GitHub.zip**, if the maintainer has attached it.
4. Right-click the downloaded ZIP and choose **Extract All**.
5. Open the extracted folder. Do not run the installer from inside the ZIP.

If no release asset is available, choose **Code > Download ZIP** on the repository
page and extract it. The installer needs `MonoCruiseNCZ.dll`, `Install-NCZ.cmd`
and `Install-NCZ.ps1` together in the same folder. If the DLL is not included,
build it using the instructions below.

## Requirements

- Windows x64 and an ETS2 or ATS installation used with TruckersMP.
- A MonoCruise version with the NCZ shared-memory reader and AEB suppression
  integration. The original MonoCruise executable does not automatically gain
  this feature by installing the DLL.

The bridge only reports NCZ state. A compatible client must act on that state.

## Installation

1. Close ETS2, ATS and MonoCruise.
2. Double-click `Install-NCZ.cmd`. It searches Steam libraries for ETS2 and ATS;
   if neither is found, select the game folder containing `bin`.
3. If access is denied, right-click `Install-NCZ.cmd` and select
   **Run as administrator**. An existing DLL is backed up as `.bak`.
4. Alternatively, copy `MonoCruiseNCZ.dll` into the game's
   `bin/win_x64/plugins` folder manually. You do not need both installation methods.
5. Start the compatible MonoCruise version, then launch the game through TruckersMP.
6. Accept the SDK/plugin prompt if the game displays one.
7. Enable **Settings > Pause AEB in TMP NCZ** in MonoCruise.
8. Enter and leave an NCZ, checking the status below the setting.

The automatic installer copies the DLL to every detected supported game.
Use manual installation if you want to install it into only one game.

## Behavior

The plugin subscribes to `Session::Gameplay().OnNoCollisionZone` and reads
`GetEntered()`. It publishes a 32-byte block named
`Local\MonoCruise_TMP_NCZ_v1`, including zone state, connection status,
a timestamp and a sequence counter. The reader rejects incomplete writes,
invalid data, disconnected sessions and timestamps older than 500 milliseconds.
No information is sent over the internet.

In the compatible personal MonoCruise integration, confirmed NCZ entry pauses
AEB warnings, automatic braking and AEB brake assistance, and resets temporary
intervention holds. Radar and collision calculations continue. Manual pedals,
ACC and OPD retain their existing behavior; ACC can still react to vehicles.
Normal AEB behavior resumes outside the zone. Unknown or stale NCZ data leaves
AEB enabled.

## Limitations and troubleshooting

- This implementation receives entry/exit events; it does not query the current
  zone at startup. If loaded inside an NCZ, leave and enter again to establish
  the state. Reconnecting also resets the state to unknown.
- If the status remains unknown, check the DLL installation, the game/plugin
  prompt and whether your MonoCruise version includes the reader.
- The supplied DLL was built for Windows x64 and its exports/dependencies were
  inspected. Windows loading and real TruckersMP events were not verified in
  the development environment. Compatibility with a particular game/TMP build
  is not established by this package.

## Uninstall

Close the game and remove `MonoCruiseNCZ.dll` from `bin/win_x64/plugins`.
Restore the `.bak` file if you need the previous plugin version.

## Build from source

Use CMake and Visual Studio's x64 C++ build tools in this directory:

```powershell
cmake -B build -A x64
cmake --build build --config Release
```

The DLL is generated in `build/Release`. The included prebuilt DLL was compiled
with MinGW for Windows x64.

## License and attribution

The unmodified SDK headers come from
[TruckersMP/GameClientSDK](https://github.com/TruckersMP/GameClientSDK).
Their MIT license is preserved in `SDK_LICENSE`. The publication package also
includes MonoCruise's MIT `LICENSE`, preserving the original attribution.

## Integration in the complete source package

The reader is `core/aeb/ncz_bridge.py`; its AEB integration is in
`core/aeb/thread.py`. Publishing the complete MonoCruise source retains both
sides of the bridge. Follow the repository-root DOWNLOAD_AND_INSTALL.txt for
the full application package. The DLL alone does not add support to an older
MonoCruise executable.
