import json
import math

from core.road_map.service import MapService, RoadMap
from core.road_map.boundaries import dimensions, offset_edges


def road(identity='road', boundary=None, points=None):
    return {'id': identity, 'kind': 'road', 'lanes': 0,
            'boundary': boundary if boundary is not None else
            {'left_m': 3.5, 'right_m': 5.5, 'source': 'sii-road-size', 'validated': False},
            'points': points or [[0, 0, 0], [0, 50, 0], [0, 100, 0], [0, 150, 0]]}


def service(tmp_path, roads):
    path = tmp_path / 'roads.json'
    path.write_text(json.dumps({'schema': 1, 'game': 'ETS2', 'roads': roads}))
    result = MapService()
    result._load(path)
    return result


def test_asymmetric_offsets_follow_curve_and_keep_height():
    edges = offset_edges([[0, 0, 12], [0, 20, 13], [20, 20, 14]], (3.5, 5.5))
    assert edges[0][0] == (3.5, 0, 12)
    assert edges[1][0] == (-5.5, 0, 12)
    assert edges[0][-1] == (20, 16.5, 14)
    assert edges[1][-1] == (20, 25.5, 14)
    assert math.isclose(math.dist(edges[0][1][:2], (0, 20)), 3.5)


def test_outside_requires_entire_body_and_clearance(tmp_path):
    s = service(tmp_path, [road()])
    assert s.preview_boundaries(0, 70, 0)
    assert s.assess_roadside(6, 70, 0, 2, 0)['state'] == 'within_or_near_extent'
    result = s.assess_roadside(9, 70, 0, 2, 0)
    assert result['state'] == 'stationary_outside_candidate'
    assert result['advisory_only'] is True
    assert s.assess_roadside(9, 70, 0, 2, 1)['state'] == 'moving_outside'
    assert s.assess_roadside(9, 70, 0, 6, 0)['state'] == 'within_or_near_extent'
    assert s.assess_roadside(-8, 70, 0, 2, 0)['state'] == 'within_or_near_extent'


def test_missing_sizes_endpoints_elevation_and_crossing_stay_unknown(tmp_path):
    s = service(tmp_path, [road()])
    assert s.assess_roadside(9, 2, 0, 2, 0)['state'] == 'unknown'
    assert s.assess_roadside(9, 70, 12, 2, 0)['state'] == 'unknown'
    assert s.assess_roadside(float('nan'), 70, 0, 2, 0)['state'] == 'unknown'
    unknown = road('side', boundary={})
    s = service(tmp_path, [road(), unknown])
    assert s.assess_roadside(9, 70, 0, 2, 0)['state'] == 'unknown'
    crossing = road('crossing', points=[[-60, 70, 0], [-10, 70, 0], [40, 70, 0], [90, 70, 0]])
    s = service(tmp_path, [road(), crossing])
    assert s.assess_roadside(9, 70, 0, 2, 0)['state'] == 'within_or_near_extent'


def test_invalid_sizes_and_legacy_map_do_not_invent_edges(tmp_path):
    for bad in [float('nan'), -1, 41, 'not_a_number']:
        assert dimensions(road(boundary={'left_m': bad, 'right_m': 4, 'source': 'sii-road-size'})) is None
    r = road(boundary={})
    geometry = RoadMap({'schema': 1, 'game': 'ETS2', 'roads': [r]})
    assert not geometry.boundary_widths
    s = service(tmp_path, [r])
    assert not s.preview_boundaries(0, 70, 0)
    assert s.preview_segments(0, 70, 0)
    assert s.assess_roadside(9, 70, 0, 2, 0)['state'] == 'unknown'
