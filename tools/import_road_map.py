"""Convert locally exported TruckSim Maps roads into advisory game coordinates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import os


def convert(source: Path, target: Path, version: str = "1.61") -> int:
    nodes = {str(n["uid"]): n for n in json.loads((source / "europe-nodes.json").read_text())}
    roads = json.loads((source / "europe-roads.json").read_text())
    looks = {str(n["token"]): n for n in json.loads((source / "europe-roadLooks.json").read_text())}
    output = []
    for road in roads:
        if road.get("hidden"):
            continue
        a, b = nodes.get(str(road["startNodeUid"])), nodes.get(str(road["endNodeUid"]))
        if not a or not b:
            continue
        look = looks.get(road.get("roadLookToken"), {})
        lane_count = len(look.get("lanesLeft", [])) + len(look.get("lanesRight", []))
        if lane_count == 0:
            continue
        dx, dz = b["x"] - a["x"], b["y"] - a["y"]
        distance = math.hypot(dx, dz)
        if not 0.1 < distance < 5000:
            continue
        handles = (
            (a["x"], a["y"]),
            (a["x"] + math.cos(a["rotation"]) * distance / 3,
             a["y"] + math.sin(a["rotation"]) * distance / 3),
            (b["x"] - math.cos(b["rotation"]) * distance / 3,
             b["y"] - math.sin(b["rotation"]) * distance / 3),
            (b["x"], b["y"]),
        )
        count = max(4, math.ceil(distance / 5))
        points = []
        for i in range(count + 1):
            t = i / count
            weights = ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)
            x = sum(w * h[0] for w, h in zip(weights, handles))
            z = sum(w * h[1] for w, h in zip(weights, handles))
            elevation = a["z"] * (1-t) + b["z"] * t
            points.append([x, z, elevation])
        output.append({"id": str(road["uid"]), "lanes": lane_count, "points": points})
    if not output:
        raise ValueError("No usable roads in the exported map")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "game": "ETS2", "version": version, "source": "local-base-game", "roads": output}
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, separators=(",", ":"))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--version", default="1.61")
    args = parser.parse_args()
    print(f"Imported {convert(args.source, args.target, args.version)} roads")
