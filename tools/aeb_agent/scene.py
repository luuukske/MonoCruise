"""Top-down ASCII render of one replayed tick, for agents with no GUI.

Ego frame, forward up the page, right to the right. Same axes as the debug
window (`_w2e` / `_e2s` in core/aeb/debug_window.py), so a scene printed here and
the same tick opened in the review UI agree on which side a target is on.
"""

from __future__ import annotations

import math

from core.aeb.calibration import DEFAULT as _CAL

from tools.aeb_agent.features import world_to_ego

_EGO_CHAR = "E"
# One glyph per id, never reused: a repeated letter reads as one vehicle in two
# places, which is the exact confusion these maps exist to prevent.
_GLYPHS = "abcdfghijklmnopqrstuvwxyz0123456789"
# Drawn lane edges sit this far either side of the map lane centre.
LANE_EDGE_M = 2.0


def glyph_map(vids) -> dict[int, str]:
    """Stable glyph per vehicle id. Ids past the alphabet get `?` rather than a clash."""
    out: dict[int, str] = {}
    for i, vid in enumerate(sorted(int(v) for v in vids)):
        out[vid] = _GLYPHS[i] if i < len(_GLYPHS) else "?"
    return out


def _cells(fwd_m: float, right_m: float, span_fwd: float, span_side: float,
           rows: int, cols: int) -> tuple[int, int] | None:
    if not (-span_fwd * 0.25 <= fwd_m <= span_fwd):
        return None
    if abs(right_m) > span_side:
        return None
    row = int(round((span_fwd - fwd_m) / (span_fwd * 1.25) * (rows - 1)))
    col = int(round((right_m + span_side) / (2.0 * span_side) * (cols - 1)))
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    return row, col


def _densify(pts, step_m: float = 0.5):
    """Points every `step_m` along an ego-frame polyline, so a sparse route draws solid."""
    out = []
    for (f0, r0), (f1, r1) in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(f1 - f0, r1 - r0) / step_m))
        for k in range(n):
            a = k / n
            out.append((f0 + a * (f1 - f0), r0 + a * (r1 - r0)))
    if pts:
        out.append(tuple(pts[-1]))
    return out


def _offset(pts, right_m: float):
    """An ego-frame polyline moved `right_m` to the right of its own travel."""
    out = []
    for i, (f, r) in enumerate(pts):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, len(pts) - 1)]
        df, dr = b[0] - a[0], b[1] - a[1]
        n = math.hypot(df, dr)
        if n < 1e-9:
            continue
        out.append((f - dr / n * right_m, r + df / n * right_m))
    return out


def _draw_overlay(grid, overlay, span_fwd, span_side, rows, cols) -> None:
    route = [tuple(p) for p in overlay.get("route") or []]
    path = [tuple(p) for p in overlay.get("path") or []]
    if len(route) >= 2:
        dense = _densify(route)
        for side in (-LANE_EDGE_M, LANE_EDGE_M):
            for f, r in _offset(dense, side):
                cell = _cells(f, r, span_fwd, span_side, rows, cols)
                if cell is not None:
                    grid[cell[0]][cell[1]] = ":"
        for f, r in dense:
            cell = _cells(f, r, span_fwd, span_side, rows, cols)
            if cell is not None:
                grid[cell[0]][cell[1]] = "~"
    if len(path) >= 2:
        for f, r in _densify(path):
            cell = _cells(f, r, span_fwd, span_side, rows, cols)
            if cell is not None:
                grid[cell[0]][cell[1]] = "+"


def render(frame, *, span_fwd: float = 80.0, span_side: float = 20.0,
           rows: int = 24, cols: int = 61,
           glyphs: dict[int, str] | None = None, overlay: dict | None = None) -> str:
    """One ReviewFrame as an ASCII map plus a legend line per drawn vehicle.

    With an `overlay` (from the scene context) the map shows ego's real later path and
    its lane from the game map instead of straight reference lines.
    """
    snap = frame.snapshot
    grid = [[" "] * cols for _ in range(rows)]
    if overlay is not None:
        _draw_overlay(grid, overlay, span_fwd, span_side, rows, cols)
    else:
        for r in range(rows):
            grid[r][cols // 2] = "."
        for edge in (-_CAL.lane_separation, _CAL.lane_separation):
            col = int(round((edge + span_side) / (2.0 * span_side) * (cols - 1)))
            if 0 <= col < cols:
                for r in range(rows):
                    grid[r][col] = ":"

    if glyphs is None:
        glyphs = glyph_map(v["vid"] for v in snap.vehicles)

    ego_cell = _cells(0.0, 0.0, span_fwd, span_side, rows, cols)
    entries: list[tuple[float, str]] = []
    hidden = 0
    for veh in sorted(snap.vehicles, key=lambda v: v["vid"]):
        fwd, right = world_to_ego(veh["x"], veh["z"], snap.ego_x, snap.ego_z,
                                  snap.ego_yaw)
        vid = int(veh["vid"])
        ch = glyphs.get(vid, "?")
        flagged = vid in snap.colliding_ids or vid in snap.suppressed_ids
        state = ("COLL" if vid in snap.colliding_ids
                 else "supp" if vid in snap.suppressed_ids else "----")
        rel_yaw = math.degrees(_norm(veh["yaw"] - snap.ego_yaw))
        cell = _cells(fwd, right, span_fwd, span_side, rows, cols)
        if cell is not None:
            r, c = cell
            grid[r][c] = ch.upper() if vid in snap.colliding_ids else ch
        rng = math.hypot(fwd, right)
        if cell is None and not flagged and rng > span_fwd * 1.5:
            hidden += 1
            continue
        suffix = "" if cell is not None else "  (off map)"
        entries.append((0.0 if flagged else rng,
                        f"  {ch} vid {vid:<6} {state}  fwd {fwd:+7.1f} "
                        f"lat {right:+6.1f}  {veh['speed_kmh']:5.1f} km/h "
                        f"yaw {rel_yaw:+6.0f}{suffix}"))
    if ego_cell is not None:
        grid[ego_cell[0]][ego_cell[1]] = _EGO_CHAR
    entries.sort(key=lambda e: e[0])
    legend = [text for _, text in entries[:12]]
    hidden += max(0, len(entries) - 12)
    if hidden:
        legend.append(f"  ({hidden} more vehicles beyond the map)")

    header = (f"t={frame.t_rel:6.2f}s  ego {snap.ego_speed * 3.6:5.1f} km/h  "
              f"state {snap.aeb_state.name}  ttc "
              f"{_fmt_inf(snap.time_to_collision)}  ttb "
              f"{_fmt_inf(snap.time_to_brake)}")
    if overlay is None:
        key = "':' = straight reference lines, NOT the road"
    elif overlay.get("route"):
        key = "'+' = where ego really drove next, '~' = ego's lane centre per map, ':' = its edges"
    else:
        key = "'+' = where ego really drove next, no map lane here"
    scale = (f"  [forward {span_fwd:.0f} m up, +/-{span_side:.0f} m across; "
             f"{key}; uppercase = AEB called it colliding]")
    body = "\n".join("  |" + "".join(row) + "|" for row in grid)
    return "\n".join([header, scale, body] + legend)


def _norm(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _fmt_inf(value: float) -> str:
    return "inf" if value >= 1e8 else f"{value:.2f}"


def render_series(frames, times, **kwargs) -> str:
    """Several scenes with one shared glyph map so ids keep their letter."""
    vids = {int(v["vid"]) for f in frames for v in f.snapshot.vehicles}
    glyphs = glyph_map(vids)
    picked = []
    for t in times:
        best = min(frames, key=lambda f: abs(f.t_rel - t))
        if best not in picked:
            picked.append(best)
    return "\n\n".join(render(f, glyphs=glyphs, **kwargs) for f in picked)
