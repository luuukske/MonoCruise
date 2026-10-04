"""Import world-coordinate lane and junction curves from a TruckLib exporter."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile


def resample(points):
    output = []
    for a, b in zip(points, points[1:]):
        if not all(math.isfinite(float(v)) for v in a + b):
            raise ValueError("Nonfinite map coordinate")
        count = max(1, math.ceil(math.dist(a, b) / 5))
        if count > 1000:
            raise ValueError("Implausible map segment length")
        for i in range(count):
            output.append([a[j] + (b[j]-a[j])*i/count for j in range(3)])
    if output:
        output.append(points[-1])
    return output


def convert(source: Path, target: Path, version: str):
    roads = []
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
        query = """SELECT r.uid, r.lanes_left, r.lanes_right,
                   a.x, a.y, a.z, a.rotation, b.x, b.y, b.z, b.rotation
                   FROM roads r JOIN nodes a ON a.uid=r.start_node_uid
                   JOIN nodes b ON b.uid=r.end_node_uid"""
        for uid, left, right, ax, ay, az, ar, bx, by, bz, br in db.execute(query):
            values = (ax, ay, az, ar, bx, by, bz, br)
            if any(v is None or not math.isfinite(float(v)) for v in values):
                continue
            distance = math.hypot(bx-ax, bz-az)
            if not 0.1 < distance < 5000:
                continue
            handles = ((ax, az), (ax+math.cos(ar)*distance/3, az+math.sin(ar)*distance/3),
                       (bx-math.cos(br)*distance/3, bz-math.sin(br)*distance/3), (bx, bz))
            count = max(4, math.ceil(distance/5))
            points = []
            for i in range(count+1):
                t = i/count
                weights = ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)
                points.append([sum(w*h[0] for w,h in zip(weights,handles)),
                               sum(w*h[1] for w,h in zip(weights,handles)), ay*(1-t)+by*t])
            roads.append({"id": f"road:{uid}", "lanes": max(0, (left or 0)+(right or 0)),
                          "kind": "road", "points": resample(points)})
        normal_count = len(roads)
        if normal_count == 0:
            raise ValueError("No normal roads with valid nodes. Export is incomplete; old map kept.")
        for table, key, parent, kind in (
            ("lane_points", "lane_id", "lanes", "lane"),
            ("nav_route_points", "nav_route_id", "nav_routes", "junction"),
        ):
            groups = {}
            for identity, x, y, z in db.execute(
                f"SELECT {key}, x, y, z FROM {table} ORDER BY {key}, point_index"
            ):
                groups.setdefault(identity, []).append([x, z, y])
            valid_ids = {row[0] for row in db.execute(f"SELECT id FROM {parent}")}
            for identity, points in groups.items():
                if identity not in valid_ids:
                    raise ValueError("Orphan map curve")
                sampled = resample(points)
                if len(sampled) >= 2:
                    roads.append({"id": f"{kind}:{identity}", "lanes": 1,
                                  "kind": kind, "points": sampled})
    if not roads:
        raise ValueError("No lane or junction geometry exported")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"schema": 1, "game": "ETS2", "version": version,
                       "source": "trucklib-mapexporter", "normal_road_count": normal_count, "roads": roads}, stream)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(roads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(f"Imported {convert(args.source, args.target, args.version)} road/lane/junction curves")
