import copy
import json

import pytest

from core.road_map.service import MapService, RoadMap
from core.road_map.template_extents import resolve_profiles
from core.road_map.validation import profile_checks, road_check


def fixture():
    geometry = dict(status='measured', vertex_count=4, min_x=-5.25, max_x=5.25,
                    min_y=0, max_y=0, min_z=-15, max_z=0)
    geometry['sections'] = [dict(geometry, min_z=-15, max_z=-15),
                            dict(geometry, min_z=0, max_z=0)]
    model = dict(status='read', parts=[dict(name='coll', pieces=[dict(geometry=geometry)])],
                 variants=[dict(name='single')])
    return dict(schema=1, game='ETS2', source='trucklib-official',
                template_profiles={'ger1': dict(source='trucklib-model-measurements',
                                                templates={'right': model})},
                roads=[dict(id='r', kind='road', lanes=0, road_type='ger1',
                            variant_left='', variant_right='single', boundary=None,
                            points=[[0, z, 12] for z in range(0, 201, 10)])])


def test_model_surface_loads_automatically_without_becoming_a_boundary(tmp_path):
    target = tmp_path / 'roads.json'
    target.write_text(json.dumps(fixture()))
    service = MapService()
    service._load(target)
    assert service._map.model_widths == {'r': (5.25, 5.25)}
    assert not service._map.boundary_widths
    assert service.preview_model_extents(0, 100, 12)
    assert not service.preview_boundaries(0, 100, 12)
    assert not service.preview_model_extents(0, 100, 0)
    assert service.preview_model_extents(0, 100, 0, all_heights=True)
    assert 'model surface candidates: 1 roads' in service.status
    result = service.assess_roadside(10, 100, 12, 2, 0)
    assert result['state'] == 'unknown'
    assert result['model_surface']['state'] == 'stationary_outside_candidate'
    assert result['model_surface']['validated'] is False
    assert result['model_surface']['advisory_only'] is True
    assert service.assess_roadside(10, 100, 12, 6, 0)['model_surface']['state'] == 'within_or_near_extent'
    assert service.assess_roadside(10, 100, 12, 2, 1)['model_surface']['state'] == 'moving_outside'
    assert service.assess_roadside(10, 2, 12, 2, 0)['model_surface']['state'] == 'unknown'


@pytest.mark.parametrize('failure', ['source', 'two_sides', 'asymmetric', 'tapered', 'elevation',
                                     'nonfinite', 'empty_sections', 'short_coverage', 'malformed',
                                     'unsupported', 'multiple_pieces'])
def test_ambiguous_and_invalid_models_produce_no_candidate(failure):
    payload = fixture()
    profile = payload['template_profiles']['ger1']
    model = profile['templates']['right']
    part = model['parts'][0]
    geometry = part['pieces'][0]['geometry']
    if failure == 'source':
        payload['source'] = 'trucklib-mapexporter'
    elif failure == 'two_sides':
        profile['templates']['left'] = copy.deepcopy(model)
    elif failure == 'asymmetric':
        geometry['min_x'] = -3
    elif failure == 'tapered':
        geometry['sections'][0]['min_x'] = -3
    elif failure == 'elevation':
        geometry['max_y'] = 1
    elif failure == 'nonfinite':
        geometry['max_z'] = float('inf')
    elif failure == 'empty_sections':
        geometry['sections'] = []
    elif failure == 'short_coverage':
        geometry['sections'][0]['min_z'] = -12
        geometry['sections'][0]['max_z'] = -12
    elif failure == 'malformed':
        payload['template_profiles']['ger1'] = None
    elif failure == 'unsupported':
        model['status'] = 'unsupported_model_version'
    elif failure == 'multiple_pieces':
        part['pieces'].append(copy.deepcopy(part['pieces'][0]))
    assert resolve_profiles(payload) == {}
    assert not RoadMap(payload).model_widths


def test_unknown_variant_and_nonroad_do_not_receive_model_edges():
    payload = fixture()
    for update in [dict(variant_right='unknown'), dict(variant_left='single'), dict(kind='junction')]:
        candidate = copy.deepcopy(payload)
        candidate['roads'][0].update(update)
        assert not RoadMap(candidate).model_widths


def test_unknown_side_road_prevents_an_outside_classification():
    payload = fixture()
    crossing = copy.deepcopy(payload['roads'][0])
    crossing.update(id='side', road_type='unknown', points=[[-50, 100, 12], [0, 100, 12], [50, 100, 12]])
    payload['roads'].append(crossing)
    service = MapService()
    service._map = RoadMap(payload)
    assert service.assess_roadside(10, 100, 12, 2, 0)['model_surface']['state'] == 'unknown'


def add_evidence(payload):
    payload['roads'][0]['instance_placement'] = dict(source='trucklib-road-instance',
        variant_override_count=0, additional_part_count=0, right_height_offset_m=0)
    profile = payload['template_profiles']['ger1']
    profile['placement'] = dict(source='sii-road-look', road_offset_present=True,
                                road_offset_m=0, unresolved_offset_keys=[])
    model = profile['templates']['right']
    model['part_attributes'] = dict(status='read', pmd_version=4, attribute_count=1,
                                    ranges=[dict(part_index=0, start=0, end=1)])
    model['variants'][0]['attributes'] = [dict(tag='visible', Type=0, Value=1)]
    return profile, model


def test_old_export_stays_pending_and_new_evidence_checks_only_surface_geometry():
    payload = fixture()
    service = MapService()
    service._map = RoadMap(payload)
    for _ in range(20):
        check = service.validation_at(0, 100, 12, 0)
        assert check['state'] == 'needs_variant_evidence'
        assert check['surface_geometry_checked'] is False
    add_evidence(payload)
    service._map = RoadMap(payload)
    check = service.validation_at(0, 100, 12, 0)
    assert check['state'] == 'surface_geometry_checked'
    assert check['variant_checked'] and check['placement_checked']
    assert check['drivable_boundary_verified'] is False
    assert check['braking_authority'] is False
    assert service._map.validation_counts['surface_geometry_checked'] == 1
    assert service.assess_roadside(10, 100, 12, 2, 0)['state'] == 'unknown'
    assert service.validation_at(0, 100, 0, 0) is None


def test_attribute_ranges_are_used_instead_of_assuming_part_order():
    payload = fixture()
    _, model = add_evidence(payload)
    model['parts'].insert(0, dict(name='vis', pieces=[]))
    model['part_attributes'].update(attribute_count=2, ranges=[
        dict(part_index=0, start=1, end=2), dict(part_index=1, start=0, end=1)])
    model['variants'][0]['attributes'].append(dict(tag='visible', Type=0, Value=0))
    geometry = RoadMap(payload)
    assert geometry.model_validation['r']['surface_geometry_checked'] is True
    model['variants'][0]['attributes'][0]['Value'] = 0
    geometry = RoadMap(payload)
    assert geometry.model_validation['r']['state'] == 'inactive_collision_part'
    assert not geometry.model_widths
    assert not geometry.model_edge_grid


@pytest.mark.parametrize('failure', ['missing_range', 'out_of_bounds', 'wrong_part', 'wrong_version',
                                     'bool_count', 'wrong_tag', 'wrong_type', 'bool_value', 'short_attributes'])
def test_invalid_variant_evidence_cannot_approve_geometry(failure):
    payload = fixture()
    _, model = add_evidence(payload)
    evidence = model['part_attributes']
    attribute = model['variants'][0]['attributes'][0]
    if failure == 'missing_range':
        evidence['ranges'] = []
    elif failure == 'out_of_bounds':
        evidence['ranges'][0]['end'] = 2
    elif failure == 'wrong_part':
        evidence['ranges'][0]['part_index'] = 1
    elif failure == 'wrong_version':
        evidence['pmd_version'] = 5
    elif failure == 'bool_count':
        evidence['attribute_count'] = True
    elif failure == 'wrong_tag':
        attribute['tag'] = 'unknown'
    elif failure == 'wrong_type':
        attribute['Type'] = 1
    elif failure == 'bool_value':
        attribute['Value'] = True
    elif failure == 'short_attributes':
        model['variants'][0]['attributes'] = []
    checks = profile_checks(payload, resolve_profiles(payload))
    check = road_check(payload['roads'][0], checks)
    assert check['state'] == 'needs_variant_evidence'
    assert check['braking_authority'] is False


@pytest.mark.parametrize('update', [dict(road_offset_m=1), dict(road_offset_m=float('nan')),
                                  dict(road_offset_m=None), dict(road_offset_present=False),
                                  dict(road_offset_present='true'), dict(road_offset_m=True),
                                  dict(unresolved_offset_keys=['lane_offsets_right'])])
def test_missing_or_unsupported_placement_cannot_approve_geometry(update):
    payload = fixture()
    profile, _ = add_evidence(payload)
    profile['placement'].update(update)
    geometry = RoadMap(payload)
    check = geometry.model_validation['r']
    assert check['state'] == 'needs_placement_evidence'
    assert check['variant_checked'] is True
    assert check['surface_geometry_checked'] is False


@pytest.mark.parametrize('update', [None, dict(variant_override_count=1), dict(additional_part_count=1),
                                  dict(right_height_offset_m=1), dict(variant_override_count=False)])
def test_instance_overrides_cannot_reuse_checked_base_variant(update):
    payload = fixture()
    add_evidence(payload)
    if update is None:
        payload['roads'][0].pop('instance_placement')
    else:
        payload['roads'][0]['instance_placement'].update(update)
    assert RoadMap(payload).model_validation['r']['state'] == 'needs_instance_evidence'


def test_checked_and_overridden_roads_do_not_share_cached_validation():
    payload = fixture()
    add_evidence(payload)
    overridden = copy.deepcopy(payload['roads'][0])
    overridden['id'] = 'overridden'
    overridden['instance_placement']['variant_override_count'] = 1
    payload['roads'].append(overridden)
    geometry = RoadMap(payload)
    assert geometry.model_validation['r']['surface_geometry_checked'] is True
    assert geometry.model_validation['overridden']['surface_geometry_checked'] is False


def test_duplicate_variant_names_do_not_prove_visibility():
    payload = fixture()
    _, model = add_evidence(payload)
    model['variants'].append(copy.deepcopy(model['variants'][0]))
    assert RoadMap(payload).model_validation['r']['state'] == 'needs_variant_evidence'


def test_noninteger_pmd_version_does_not_prove_visibility():
    payload = fixture()
    _, model = add_evidence(payload)
    model['part_attributes']['pmd_version'] = 4.0
    assert RoadMap(payload).model_validation['r']['state'] == 'needs_variant_evidence'
