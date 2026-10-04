import json
import math

import pytest

from core.road_map.service import MapService, RoadMap
from tools.import_road_map import convert


def payload(roads):
    return {"schema": 1, "game": "ETS2", "version": "1.61", "roads": roads}


def road(identity, height=0, x=0):
    return {"id": identity, "lanes": 2, "points": [[x, -20, height], [x, 20, height]]}


def test_direction_elevation_and_ambiguity():
    geometry = RoadMap(payload([road("lower"), road("bridge", 12)]))
    assert geometry.match(1, 0, 0, 0)["road_id"] == "lower"
    assert geometry.match(1, 0, 12, math.pi)["road_id"] == "bridge"
    assert geometry.match(0, 0, 0, math.pi / 2) is None
    assert geometry.match(0, 0, 6, 0) is None
    assert geometry.match(float("nan"), 0, 0, 0) is None
    assert RoadMap(payload([road("a"), road("b", x=1)])).match(0, 0, 0, 0) is None


def test_overloaded_cell_skips_match():
    geometry = RoadMap(payload([road(str(i)) for i in range(513)]))
    assert geometry.match(0, 0, 0, 0) is None


def test_missing_or_invalid_map(tmp_path):
    service = MapService()
    service._load(tmp_path / "absent.json")
    assert service.match(0, 0, 0, 0) is None
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema":1}')
    service._load(bad)
    assert service.match(0, 0, 0, 0) is None
    with pytest.raises(ValueError):
        RoadMap(payload([]))


def test_convert_coordinates_and_curve(tmp_path):
    data = {
        "nodes": [{"uid": "a", "x": 0, "y": 0, "z": 10, "rotation": 0},
                  {"uid": "b", "x": 30, "y": 30, "z": 12, "rotation": math.pi/2}],
        "roads": [{"uid": "r", "startNodeUid": "a", "endNodeUid": "b", "roadLookToken": "l"}],
        "roadLooks": [{"token": "l", "lanesLeft": [1], "lanesRight": [1]}],
    }
    for key, value in data.items():
        (tmp_path / f"europe-{key}.json").write_text(json.dumps(value))
    target = tmp_path / "map.json"
    assert convert(tmp_path, target) == 1
    geometry = json.loads(target.read_text())
    points = geometry["roads"][0]["points"]
    assert points[0] == [0, 0, 10]
    assert points[-1] == [30, 30, 12]
    assert points[1][0] > points[1][1]
    assert RoadMap(geometry).count > 0


def test_side_road_context_ignores_heading_but_checks_height():
    side = {"id": "side", "lanes": 1, "kind": "junction",
            "points": [[-20, 0, 0], [20, 0, 0]]}
    bridge = {"id": "above", "lanes": 1, "kind": "junction",
              "points": [[-20, 0, 12], [20, 0, 12]]}
    geometry = RoadMap(payload([side, bridge]))
    assert geometry.match(0, 0, 0, 0) is None
    assert geometry.nearby_junctions(0, 0, 0) == [{"route_id": "side", "distance_m": 0}]
    assert geometry.nearby_junctions(0, 0, 12) == [{"route_id": "above", "distance_m": 0}]


def test_trucklib_database_import(tmp_path):
    import sqlite3
    from tools.import_trucklib_map import convert as import_db

    source = tmp_path / "europe.db"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE nodes (uid INTEGER, x REAL, y REAL, z REAL, rotation REAL)")
        db.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
                       [(10,0,12,0,math.pi/2), (11,0,12,60,math.pi/2)])
        db.execute("CREATE TABLE roads (uid INTEGER, lanes_left INTEGER, lanes_right INTEGER, start_node_uid INTEGER, end_node_uid INTEGER)")
        db.execute("INSERT INTO roads VALUES (99,1,1,10,11)")
        for table in ("lanes", "nav_routes"):
            db.execute(f"CREATE TABLE {table} (id INTEGER)")
            db.execute(f"INSERT INTO {table} VALUES (1)")
        for table, key in (("lane_points", "lane_id"), ("nav_route_points", "nav_route_id")):
            db.execute(f"CREATE TABLE {table} ({key} INTEGER, point_index INTEGER, x REAL, y REAL, z REAL)")
            db.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?)",
                           [(1, 0, 0, 12, 0), (1, 1, 0, 12, 60)])
    target = tmp_path / "map.json"
    assert import_db(source, target, "1.61") == 3
    exported = json.loads(target.read_text())
    assert exported["roads"][0]["points"][0] == [0, 0, 12]
    assert {r["kind"] for r in exported["roads"]} == {"road", "lane", "junction"}
    assert RoadMap(exported).count >= 36
    original = target.read_bytes()
    with sqlite3.connect(source) as db:
        db.execute("DELETE FROM roads")
    with pytest.raises(ValueError, match="No normal roads"):
        import_db(source, target, "1.61")
    assert target.read_bytes() == original


def test_feedback_is_local_and_does_not_change_settings(tmp_path):
    from core.road_map.feedback import FeedbackJournal
    from core.settings import Settings

    path = tmp_path / "feedback.jsonl"
    journal = FeedbackJournal(path)
    enabled = Settings.AEB_enabled
    assert journal.submit("false_brake", {"brake": True})
    journal._queue.join()
    event = json.loads(path.read_text())
    assert event["label"] == "false_brake"
    assert event["context"]["brake"] is True
    assert Settings.AEB_enabled == enabled
    with pytest.raises(ValueError):
        journal.submit("unknown", {})


def test_preview_survives_overloaded_matching_cells_and_elevation():
    service = MapService()
    service._map = RoadMap(payload([road(str(i), height=20, x=i/100) for i in range(600)]))
    assert service.match(0, 0, 20, 0) is None
    assert service.preview_segments(0, 0, 0)
    assert len(service.preview_segments(0, 0, 0)) <= 2048


def test_preview_reads_adjacent_cell_and_rejects_distant_map():
    service = MapService()
    service._map = RoadMap(payload([road('adjacent', x=130)]))
    assert service.preview_segments(127, 0, 0)
    assert service.preview_segments(1000, 1000, 0) == []
    assert service.preview_segments(float('nan'), 0, 0) == []


def test_junction_only_export_is_not_a_complete_road_map():
    incomplete = payload([dict(road("junction"), kind="junction")])
    incomplete["source"] = "trucklib-mapexporter"
    with pytest.raises(ValueError, match="Repair-Road-Map"):
        RoadMap(incomplete)


def test_legacy_export_requires_new_reader(tmp_path):
    data = payload([dict(road('ordinary'), kind='road')])
    data['source'] = 'trucklib-mapexporter'
    target = tmp_path / 'legacy.json'
    target.write_text(json.dumps(data))
    service = MapService()
    service._load(target)
    assert service.match(0,0,0,0) is None
    assert 'Prepare-TruckLib-Map.cmd' in service.status
