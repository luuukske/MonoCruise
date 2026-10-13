"""Roadside objects for the standalone lane view (``tools/road_layout_view.py``).

The runtime map reads lanes only. This plugs the obstacle probe into
``LiveMap`` so the dev window can still draw guardrails, signs and posts.
"""

from __future__ import annotations

import colorsys
import hashlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.road_layout.archive import HashFs
from core.road_layout.live import MapDot, MapLine
from tools.map_obstacle_parse import _load_sector
from tools.map_obstacle_probe import geometry_for, is_clutter, load_catalog

log = logging.getLogger("road_layout")


class ObstacleProps:
    """``LiveMap`` props hook: full sector parse plus drawable object geometry."""

    def __init__(self, archives: Sequence[HashFs]) -> None:
        try:
            self._books, _note = load_catalog(list(archives))
        except (OSError, ValueError):
            log.warning("obstacle catalog failed")
            self._books = {}

    def load_sector(self, archives, map_name: str, ix: int, iz: int):
        mesh, cap, _times, note = _load_sector(archives, map_name, ix, iz)
        if "failed" in note:
            log.warning("obstacle sector %+d %+d: %s", ix, iz, note)
        return mesh, cap

    def geometry(self, nodes: dict, caps: list) -> tuple[tuple[MapLine, ...], tuple[MapDot, ...]]:
        if not caps or not self._books:
            return (), ()
        try:
            lines, dots = geometry_for(nodes, caps, self._books)
        except Exception:
            log.exception("obstacle geometry failed")
            return (), ()
        return _drawable(lines, dots)


def _drawable(lines, dots) -> tuple[tuple[MapLine, ...], tuple[MapDot, ...]]:
    kept_lines: list[tuple[str, list, str]] = []
    kept_dots: list[tuple[str, float, float, str]] = []
    labels: list[str] = []
    for row in lines:
        kind, pts, label = row[0], row[1], row[2] if len(row) > 2 else ""
        if kind == "road" or len(pts) < 2 or is_clutter(label):
            continue
        kept_lines.append((kind, pts, label))
        labels.append(label or kind)
    for row in dots:
        kind, x, z = row[0], row[1], row[2]
        label = row[4] if len(row) > 4 else ""
        if is_clutter(label):
            continue
        kept_dots.append((kind, x, z, label))
        labels.append(label or kind)
    colors = _assign_colors(labels)
    props: list[MapLine] = []
    marks: list[MapDot] = []
    for kind, pts, label in kept_lines:
        key = label or kind
        packed = _pack_xz(kind, pts, key, colors[key])
        if packed is not None:
            props.append(packed)
    for kind, x, z, label in kept_dots:
        key = label or kind
        marks.append(MapDot(kind, x, z, key, colors[key]))
    return tuple(props), tuple(marks)


def _assign_colors(labels: list[str]) -> dict[str, str]:
    """One bright hex per model path. The same path keeps its color."""
    taken: set[str] = set()
    out: dict[str, str] = {}
    for label in sorted(set(labels)):
        digest = hashlib.sha1(label.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:2], "big") / 65535.0
        sat = 0.55 + 0.4 * (digest[2] / 255.0)
        val = 0.62 + 0.35 * (digest[3] / 255.0)
        red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
        number = (int(red * 255) << 16) | (int(green * 255) << 8) | int(blue * 255)
        color = f"#{number:06X}"
        while color in taken or (number & 255) + ((number >> 8) & 255) + ((number >> 16) & 255) < 280:
            number = (number + 0x1F3A5B) & 0xFFFFFF
            color = f"#{number:06X}"
        taken.add(color)
        out[label] = color
    return out


def _pack_xz(kind: str, pts: list[tuple[float, float]], label: str, color: str) -> MapLine | None:
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    return MapLine(kind, tuple(pts), min(xs), max(xs), min(zs), max(zs), None, label, color)
