"""Road-look lane layout from ``def/world/road_look*.sii``.

A look lists the lanes on each side of the spline and, when the template
needs it, a lateral shift for each of those lanes. The AI lane is 4.5 m
wide: that is the width the game's own connection check uses.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from core.road_layout.archive import HashFs

LANE_WIDTH_M = 4.5

_UNIT = re.compile(r"(\w+)\s*:\s*(\S+)\s*\{")
_HEADER = re.compile(r"^(\w+)\s*:\s*(\S+)\s*$")
_OFFSET = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,")


@dataclass
class RoadLook:
    """Lanes of one road look, in the order the def file lists them."""

    token: str
    lanes_left: list[str] = field(default_factory=list)
    lanes_right: list[str] = field(default_factory=list)
    offset_m: float = 0.0
    shift_left: list[float] = field(default_factory=list)
    shift_right: list[float] = field(default_factory=list)

    def laterals(self, *, apply_shift: bool = True) -> list[float]:
        """Signed metres, positive to the right of travel from start to end."""
        right = _shifts(self.lanes_right, self.shift_right, apply_shift)
        left = _shifts(self.lanes_left, self.shift_left, apply_shift)
        if right and left:
            return (
                _from_median(right, self.offset_m, 1.0)
                + _from_median(left, self.offset_m, -1.0)
            )
        if right:
            return _one_way(right)
        return [-value for value in _one_way(left)]


def _shifts(lanes: list[str], shifts: list[float], apply_shift: bool) -> list[float]:
    out: list[float] = []
    for i in range(len(lanes)):
        out.append(shifts[i] if apply_shift and i < len(shifts) else 0.0)
    return out


def _from_median(shifts: list[float], offset_m: float, sign: float) -> list[float]:
    return [
        sign * (offset_m + LANE_WIDTH_M * (i + 0.5) + shift)
        for i, shift in enumerate(shifts)
    ]


def _one_way(shifts: list[float]) -> list[float]:
    """Even counts are centered on the spline. Odd counts sit half a lane left."""
    count = len(shifts)
    mid = count // 2
    extra = LANE_WIDTH_M / 2.0 if count % 2 == 0 else -LANE_WIDTH_M / 2.0
    return [(i - mid) * LANE_WIDTH_M + extra + shift for i, shift in enumerate(shifts)]


def load_road_looks(archives: Sequence[HashFs]) -> dict[str, RoadLook]:
    """Every road look under ``def/world``. A later file replaces the same token."""
    looks: dict[str, RoadLook] = {}
    for text in _def_texts(archives, "road_look"):
        for kind, name, body in _units(text):
            if kind != "road_look":
                continue
            look = _look_from_body(name, body)
            looks[name] = look
            alias = name.split(".", 1)[1] if "." in name else ""
            if alias:
                looks[alias] = look
    return looks


def _look_from_body(token: str, body: str) -> RoadLook:
    look = RoadLook(token=token)
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key == "lanes_left[]":
            look.lanes_left.append(value)
        elif key == "lanes_right[]":
            look.lanes_right.append(value)
        elif key == "road_offset":
            look.offset_m = _float(value)
        elif key == "lane_offsets_left[]":
            look.shift_left.append(_first_number(value))
        elif key == "lane_offsets_right[]":
            look.shift_right.append(_first_number(value))
    return look


def _def_texts(archives: Sequence[HashFs], prefix: str) -> Iterable[str]:
    for archive in archives:
        listing = archive.listdir("def/world")
        if listing is None:
            continue
        _subs, files = listing
        for name in sorted(files):
            if not name.startswith(prefix) or not name.endswith(".sii"):
                continue
            blob = archive.get(f"def/world/{name}")
            if blob:
                yield blob.decode("utf-8", "replace")


def _units(text: str):
    """Yield ``(kind, name, body)`` for each SII unit. The opening brace may be on the next line."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        match = _HEADER.match(line) or _UNIT.match(line.strip())
        if match is None:
            continue
        kind, name = match.group(1), match.group(2)
        if "{" not in line:
            while index < len(lines) and lines[index].strip() == "":
                index += 1
            if index >= len(lines) or "{" not in lines[index]:
                continue
            index += 1
        depth = 1
        buf: list[str] = []
        while index < len(lines) and depth > 0:
            raw = lines[index]
            index += 1
            depth += raw.count("{") - raw.count("}")
            if depth > 0:
                buf.append(raw)
        yield kind, name, "\n".join(buf)


def _float(value: str) -> float:
    try:
        return float(value.split()[0])
    except ValueError:
        return 0.0


def _first_number(value: str) -> float:
    match = _OFFSET.search(value)
    if match is None:
        return _float(value)
    return float(match.group(1))
