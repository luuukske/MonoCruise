# ETS2 road map connection

Stage one is advisory: road data is read and matched to ego telemetry, and the
match is published with AEB diagnostics. It does not change collision filters,
steer-led curvature, pedal authority or braking thresholds.

Prepare-Road-Map.cmd and Prepare-TruckLib-Map.cmd now use the TruckLib exporter.
The bootstrap downloads missing Python and .NET 8 SDK locally and retrieves a
pinned exporter ZIP. Git, Node and Windows C++ tools are no longer prerequisites.
The older tools/import_road_map.py JSON converter is retained for existing exports.
Parser node x/y are game x/z, parser z is game elevation.

Road curves use standard cubic Bezier interpolation with endpoint tangents.
This first pass does not model lane boundaries, road-look offsets or prefab
junction paths. It never labels a target harmless based on map position.
ProMods is not supported by the external parser, so it must not be selected as
a verified source. Map files are loaded and indexed on a background thread.
Live lookups are capped to 512 segments and skip overloaded grid cells.

Missing, invalid, unmatched or ambiguous map data returns no match; AEB remains
unchanged. Actual ETS2 1.61 compatibility and positioning must be verified using
a local export and driving clips before adding map-based braking behavior.

## TruckLib export adapter and live diagnostics

Prepare-TruckLib-Map.cmd runs a separate pinned LXMedia1/MapExporter using
TruckLib.HashFs, Models and Sii. Its bootstrap installs the .NET 8 SDK if missing; Git is not required. The Python
adapter reads its SQLite lane_points and nav_route_points in read-only mode;
world x/y/z becomes x/z/elevation. It never copies external GPL source into the
app. Nav route data in the inspected exporter contains endpoint chords, not
complete curved junction paths. Resampling does not restore missing curvature.
The adapter must remain advisory until geometry is validated on actual game data.

Junction routes within 40 metres are reported independent of ego heading, so
side roads appear even when perpendicular. Elevation mismatches are excluded;
overloaded cells return unknown. The UI updates every 500 ms while the settings
panel is visible, showing matching, junction counts and actual warning/brake outputs.
No map route is used to discard radar threats or enable stronger braking.

Unnecessary brake / Missed hazard buttons append user-labelled snapshots to
maps/aeb-feedback.jsonl on a worker thread. No feedback is uploaded by this
feature, and labels are not treated as verified ground truth. These events are
review inputs, not a self-training model or an automatic tuning mechanism.
The existing steer-path gain adaptation remains unchanged.

The main-window radar preview uses bounded road segments and copied traffic positions. It refreshes on the Qt thread every 200 ms, never writes control data, and clears radar snapshots older than one second.

Road rendering uses a separate simplified preview grid, up to 1024 segments per
cell, nine neighboring cells and 2048 nearest segments within 120 metres. It
does not share the AEB matching overflow gate. Elevated roads remain visible
in a muted colour; this does not relax AEB's elevation or ambiguity gates.
Live diagnostics report drawn segments and ego world coordinates when no
nearby geometry is found.

TruckLib conversion now imports ordinary roads directly from roads plus their
joined start/end nodes. Node tangents generate sampled cubic road centre lines;
lanes and prefab endpoint routes remain supplemental. Unknown lane count is zero.
A junction-only TruckLib export is rejected with Repair-Road-Map.cmd guidance.
Repair reuses the existing local SQLite export and preserves a backup of the old
JSON; conversion with zero valid normal roads never replaces the destination.

The current preparation path supersedes the SQLite repair above. It uses the
standalone integrations/trucklib_reader tool and official TruckLib.ScsMap with
.NET 10, installed automatically. Legacy MapExporter JSON is rejected during
service loading regardless of its normal-road count. Old DB data must not be
relabelled as official. Only ordinary road centre lines are currently exported
by the official reader; prefab paths require a later validated implementation.

## Width extents and AEB shadow diagnostics

Optional boundary objects come only from explicit sii-road-size definitions.
Finite left/right values in 0..40 metres, with a total of at least 2 metres,
produce offset polylines following the sampled centre curve. Unknown widths
produce no edges. Their preview has its own bounded grid; old centre-only JSON
remains compatible. Dashed yellow lines denote definition extents, not verified
physical pavement, lane borders or driving permissions.

MapService.assess_roadside checks all nearby ordinary roads, without a heading
filter, with elevation gating, endpoint caution and a conservative bounding
circle covering the entire target body plus one metre clearance. Moving targets
have a separate moving_outside state; unknown width, overloaded cells and road
ends remain unknown. This is a shadow classification only. Live AEB publishes
road_edge_context for at most eight current threats, and the live panel shows
the stationary outside candidate count. It never suppresses, delays or reduces
braking. Replay decisions and clip schema/consent remain unchanged. Actual
width-bearing exports and annotated roadside/cut-in/lead-stop clips are needed
to validate a future braking filter, particularly modern template roads.

## Local driven traces

DrivenRoutes records ETS2 ego positions as local, deduplicated segments, sampled
by distance at three metres. The Qt polling timer also samples while the live
panel is hidden. Stale radar, pauses, disconnected telemetry, clock reversal,
stationary jitter and implausible jumps break the trace; no straight connection
is invented across a teleport. ATS samples are excluded. File I/O and loading
run on a bounded worker queue, independent of AEB. The local JSONL file has a
32 MiB limit and 200000 unique segment cap; previews read nine bounded grid cells
and draw at most 1024 nearby segments with a five-metre elevation gate.

The live panel provides an enabled-by-default, persistent recording checkbox
for the user-requested local feature and shows purple dotted driven traces.
Routes survive restart, stay on the device, and are never uploaded or included
in AEB clips. They are observations of where the truck travelled, including any
off-road driving, not evidence of road centre, road width or safe driving space.
They never enter map matching, road extent classification or brake vetoes.

## Automatic model surface candidates

When an official TruckLib JSON is loaded, template profiles are resolved once
on the map loader worker. Eligible profiles have only a right template, one
coll part with one measured piece, a centred symmetric flat envelope, finite
consistent transverse extents across longitudinal sections, and a known selected
right variant name. Missing, asymmetric, tapered, non-flat, multi-piece and
unsupported models remain unknown. No default lane width is invented.

The candidate width comes from that collision piece, never the bounds of the
whole model or a driven trace. Orange dotted offsets show its envelope along
road centre curves on the live display and replay background. Candidate grids
have the same bounded preview caps as definition edges; live previews gate
height. Replay lacks ego height and shows all heights explicitly for inspection.
Explicit definition widths keep their separate yellow layer and take precedence.

These are unverified driving-space envelopes. Earlier exports do not retain PMD
part-attribute ranges, so their selected variant membership is not proven. New
exports retain this evidence for the automatic checks below. Active lane markings,
banks, prefab geometry and legal driving space remain unresolved. A model collision envelope can include shoulders and
cannot prove whether a target is in a driving lane.

Live road_edge_context includes a nested model_surface assessment, with
validated=false and advisory_only=true, using cached combined width dictionaries.
The original top-level classification remains unchanged when model width alone
is available. All nearby roads, target bounding radius, height, unknown widths,
and road-end caution still apply. Model diagnostics are calculated after the
brake decision, never change control, and are not added to captured clip schema.

## Automatic surface evidence checks

Map loading now checks every eligible model road automatically, caching results
per type, selected variant and road-instance eligibility. Earlier exports stay
needs_variant_evidence regardless of how many times the driver passes. This is
not an online learner and no traffic motion is treated as proof of pavement.

New exports retain PMD v4 part-attribute ranges and explicit road-look placement
fields. Ranges must cover every attribute exactly once, reference the exported
part order, and contain exactly one supported visible attribute for coll in the
selected variant. An inactive collision part receives no model preview edges.
Missing, unsupported, overlapping and malformed evidence stays pending.

The restricted placement check requires explicitly present finite zero road_offset
and no unresolved offset keys. The road instance must have zero right-side variant
overrides, additional parts and height offset. Nonzero or missing values remain
pending instead of assuming defaults. surface_geometry_checked records only these
structural checks; drivable_boundary_verified and braking_authority always remain
false. It does not prove geometry alignment, a driving lane or banket boundary.

The live panel shows the current road's check and global checked/pending counts.
Prepare map validation data starts the existing PowerShell exporter in a child
process on Windows, with the ETS2 folder picker and automatic dependency setup.
The GUI stays responsive; successful completion reloads and checks the new JSON.
The existing temporary-file export and backup policy preserve the previous map
on parser failure. Process output is drained without logging machine-specific
paths. Missing exporter, process errors and Linux show an unavailable/failed action.
