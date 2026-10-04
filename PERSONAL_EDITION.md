# MonoCruise Personal Edition

This package contains the complete modified MonoCruise source, not a prebuilt
Windows application installer. It is based on
[luuukske/MonoCruise](https://github.com/luuukske/MonoCruise) and preserves the
original MIT license and third-party license notices.

## Download and launch

Read `DOWNLOAD_AND_INSTALL.txt` for the complete Windows setup instructions.
Extract the entire ZIP and double-click `Start-MonoCruise.cmd` in the project
root. The launcher reuses an existing virtual environment or prepares Python
3.12.10 and the required packages. Initial setup needs internet access. Python
is downloaded from python.org and the installer signature is checked.

Physical pedals and the game's telemetry SDK setup are still required. Follow
the pedal and SDK sections in `README.md`. Automatic dependency installation
does not replace game configuration. Windows installation and actual game/pedal
operation were not verified in the development environment.

Before updating an existing installation, back up `config.json` and
`config.json.bak`. Keep your `maps`, `sounds` and `.venv` directories. Copy the
new application files over the existing project folder. The upstream updater
does not preserve these personal modifications.

## Interface and warning sounds

The settings screen uses a dark navy theme, clearer secondary text, rounded
controls and section navigation. Interface labels and added documentation are
in English. The right-hand driving view remains visible when settings close.

AEB supports Chime, Pulse, Original and imported WAV, OGG or MP3 sounds. Imported
clips must be 0.2-5 seconds long and below 10 MB. Files are copied to `sounds`
beside the settings file. Volume is adjustable; Listen previews the sound and
Reset restores Chime at 50%. Missing or invalid files fall back to Chime.

## Experimental hazard flashing

With Hazards enabled, enable Fast hazard trial in Settings. Test / Stop (5s)
starts or stops a five-second trial while the game is connected. Hard braking
can also start the trial when enabled. The post-confirmation toggle dwell is
0.10 seconds; the existing 80 ms key press and command verification are retained.
The previous hazard state is restored afterward. This does not guarantee a
particular flash rate or visibility to other TruckersMP players. Disable the
trial if the result is unsuitable. Actual TMP operation was not verified here.

## Optional TruckersMP NCZ bridge

See `integrations/tmp_ncz/README.md`. Close the game, then run
`integrations/tmp_ncz/Install-NCZ.cmd`, or copy `MonoCruiseNCZ.dll` into the game's
`bin/win_x64/plugins` folder. Enable Pause AEB in TMP NCZ in this source version.
Confirmed NCZ state pauses AEB warnings, braking and brake assistance. ACC and
OPD retain their behavior. Missing or stale data leaves AEB enabled. If the
plugin starts inside an NCZ, leave and enter again. Actual Windows/TMP loading
and events were not verified in the development environment.

## ETS2 road map preparation

Run `Prepare-Road-Map.cmd` or `Prepare-TruckLib-Map.cmd`, then select the ETS2
installation folder containing `base.scs`. The current preparation path uses
the standalone `integrations/trucklib_reader` and official TruckLib libraries.
It prepares missing Python and .NET 10 tooling automatically. Node.js, Git and
Windows C++ build tools are not prerequisites for this path.

Exporting can take time. Game files are read, not modified; the resulting map
is stored as `maps/roads.json`. The exporter preserves the previous map on
failure. The live view's Prepare map validation data button invokes the same
exporter and reloads the new map on success. Otherwise use Reload road map.

Only ordinary road centre curves are exported by the current official reader.
Prefab junction paths, verified driving lanes and shoulder boundaries are not
provided. Older junction-only SQLite exports cannot be treated as complete maps.
ProMods is not established as a supported source. Actual ETS2 1.61 placement
and Windows export behavior require validation on local game files.

## Live radar and map display

The view is a telemetry-based top-down display, not a live camera feed. It
refreshes every 0.2 seconds and clears stale radar data. Radar works without a
map. Blue lines show road centres; dashed yellow lines show explicit definition
width extents. Purple dotted lines show local driven traces. Orange dotted
lines show eligible but unverified model surface candidates.

A separate preview index displays up to 2048 nearby road segments within
120 metres. This is independent of the AEB matching cap. Elevated roads appear
muted; the AEB elevation gate remains unchanged. The view reports segment
counts, world X/Z coordinates and actual AEB warning/brake state.

## Road widths and model evidence

Explicit legacy road definitions can provide left/right width extents. Modern
PMD/PMG models export parts, variants, materials and measured sections. Only
restricted flat, symmetric, consistent single-collision-piece profiles produce
candidate envelopes. Unknown or unsupported models stay unknown; no default
lane width is invented.

Automatic checks examine the selected variant's collision-part visibility,
explicit zero placement offset and supported road-instance overrides. Older
exports lacking variant-part or placement evidence stay pending. Preparing the
map with the current reader supplies the additional evidence where supported.

A surface_geometry_checked result validates those structural checks only. It
does not prove physical alignment, driving-lane membership or shoulder edges.
Roadside assessments consider the full target body, neighboring roads, height
and road ends. They remain advisory and never suppress, delay or reduce AEB
braking. Roadside false braking has not been solved by the map display.

## Local driven routes and feedback

Record my driven routes automatically (local) stores ETS2 positions roughly
three metres apart in `maps/driven-routes.jsonl`. Existing routes load after
restart. Pauses, stale data, teleports and disconnects break the trace. Recording
stops at 32 MiB or 200000 segments while preserving existing traces. Uncheck the
option to stop recording. A driven trace is not evidence of either road edge
and is never used as a brake veto or uploaded by this feature.

Unnecessary brake and Missed hazard save local review snapshots in
`maps/aeb-feedback.jsonl`. They do not replace full clips or video and do not
automatically train AEB or change thresholds. Existing steering-path adaptation
remains separate.

## AEB clip playback

Double-click `Watch-AEB-Clips.cmd`, select a clip and press Play or Space.
Missing Python dependencies are prepared automatically. `maps/roads.json` loads
in the background; Choose map JSON can select another local map for that
session. Show roads toggles the road overlay. Large files may take time to load.
Replay map layers are advisory and do not change recorded labels or decisions.

## Crossing-traffic brake correction

The TMP turning filter previously discarded a collision occurring before a
target's predicted endpoint cleared the lane. It now keeps a centre encounter
eligible when measured miss is within the combined body clearance and the miss
is non-opening. Unknown measurements or a clear pass retain the previous test.
This correction affects actual AEB decisions, independently of map diagnostics.

Recorded-input replay of ceadfde4 moved the first brake from 4.500 s to 3.594 s,
0.906 s earlier. Seven other supplied clips retained their brake timing and tick
counts. Replay does not simulate the altered ego trajectory and therefore does
not prove that the new brake prevents a crash. Live screen-based road recognition
has not been implemented.

## Validation and limitations

Python regression tests, headless Qt checks, audio decoding and lint were run
in the Linux development environment. The .NET reader built with no errors or
warnings; synthetic export and model-evidence checks passed. These checks do
not establish real Windows audio, physical pedal, TruckersMP or game-version
compatibility. Source installation is separate from the upstream Windows EXE
release workflow.

## GitHub publication

Upload the contents of this package's `MonoCruise` folder to your repository
root, retaining the MIT license and attribution. Create a release and attach
`MonoCruise-GitHub-English.zip`. Users can then follow DOWNLOAD_AND_INSTALL.txt.
Do not publish local settings, pedal profiles, driven routes, captured clips,
recordings or exported game assets. These personal files are excluded from the
provided publication package. No repository commit or release is created by
preparing this ZIP.
