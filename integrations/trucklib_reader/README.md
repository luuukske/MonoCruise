# Standalone TruckLib reader

This isolated GPL-2.0 tool reads map sectors using the official TruckLib.ScsMap
reader, not LXMedia1's custom sector parser. It downloads pinned upstream source
and builds with .NET 10 separately from the Python application. No game assets
are distributed. It emits ordinary road centre lines in game coordinates.

The old SQLite export is not a trustworthy source: the supplied database has
empty lane tables, malformed node coordinates and no nearby road nodes at the
reported driving position. A new export from installed game archives is needed.

Base/DLC HashFS archives are mounted read-only with later archives overriding
identical file paths. ProMods and external seasonal maps are not supported by
this first reader. Full prefab paths and lane boundaries are not exported.
Writes use a temporary file; no output replaces the old JSON on parser failure.

The cached upstream project has SourceLink's build-only package reference removed
before building; map reader code is unchanged. This avoids running unneeded Git
build metadata tasks. The reader was compiled and checked with a generated
HashFS map archive, then matched and rendered by the Python map service. This
is a format round-trip test, not a real ETS2 1.61 game export test.

## Definition extents

The reader also mounts def.scs and base_share.scs, so road_look SII files can be
resolved with their includes. Ordinary roads retain their RoadType token.
Explicit finite road_size_left/right with zero road_offset is emitted as a
boundary object, source sii-road-size, validated false. Template models,
lane-offset configurations and missing dimensions remain null. No nominal
lane width is assumed. boundary_road_count and boundary_status report coverage
and definition parse failures. These are definition extents, not verified
pavement or drivable shoulder edges. Physical alignment and side orientation
must be checked in the game before using them to veto a brake. Modern template
models need a separate validated geometry reader; this version does not infer
width from lane count or whole-model bounding boxes.

## Template model measurements

TemplateProfiles uses the existing TruckLib.Models dependency to read PMD/PMG
models referenced by road_look definitions. It caches each model once and adds
per-type template_profiles to the same roads JSON, with part names, material
slots, locators, variants and up to sixteen longitudinal groups of measured
vertex extents per mesh piece. Road instances also retain their side variants.
This is a compact measurement report, not a copy of full meshes or game assets.

Missing, unsupported or unreadable models have explicit statuses. Model read,
missing and unsupported counts appear in JSON and console output. Existing
centre lines and explicit legacy extents remain compatible. Template reports
have drivable_boundary false and never create boundary objects or influence AEB.
Parts must be interpreted and their local coordinates, variant selection, left
side orientation and alignment validated against real game data before a
pavement polygon can be derived. Whole-model or part extents alone do not prove
a drivable shoulder edge. Prefab junction paths remain a separate missing layer.

## Surface validation evidence

New profiles include part_attributes read directly from the PMD v4 range table,
which upstream Model does not retain. Counts, offsets, half-open index ranges,
non-overlap and complete coverage are checked; a failed table has an explicit
status while ordinary road export remains usable. Placement retains explicit
road_offset presence/value and unresolved offset keys, without assigning an
unproven default. Each road also carries right-side variant override count,
additional part count and height offset in instance_placement.

Python uses these fields for restricted automatic structural checks of collision
surfaces. Passing them is not a driving-space or physical alignment certificate
and does not enable a brake veto. Older JSON remains readable but lacks this
proof and must be exported again to advance structural validation. Neither
existing boundary objects nor captured AEB clip schemas are changed.
