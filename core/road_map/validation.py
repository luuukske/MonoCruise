"""Automatic evidence checks for model surfaces, never driving-space approval."""
from __future__ import annotations

import math


def profile_checks(payload, resolved):
    checks = {}
    for token in resolved:
        profile = payload['template_profiles'][token]
        model = profile['templates']['right']
        try:
            variants = _variant_checks(model)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            variants = {}
        placement = _placement_check(profile.get('placement'))
        checks[token] = variants, placement
    return checks


def _variant_checks(model):
    evidence = model.get('part_attributes')
    if (not isinstance(evidence, dict) or evidence.get('status') != 'read' or
            type(evidence.get('pmd_version')) is not int or evidence['pmd_version'] != 4):
        return {}
    count = evidence.get('attribute_count')
    if type(count) is not int or not 0 < count <= 65536:
        return {}
    parts, ranges = model['parts'], evidence['ranges']
    if len(ranges) != len(parts) or len(parts) > 4096:
        return {}
    assigned = set()
    collision_range = None
    for index, entry in enumerate(ranges):
        start, end = entry['start'], entry['end']
        if (type(entry.get('part_index')) is not int or entry['part_index'] != index or
                type(start) is not int or type(end) is not int or not 0 <= start <= end <= count):
            return {}
        indices = set(range(start, end))
        if indices & assigned:
            return {}
        assigned.update(indices)
        if parts[index]['name'] == 'coll':
            collision_range = start, end
    if len(assigned) != count or collision_range is None:
        return {}
    results = {}
    names = [variant['name'] for variant in model['variants']]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        return {}
    for variant in model['variants']:
        attrs = variant['attributes']
        if len(attrs) != count:
            continue
        selected = attrs[collision_range[0]:collision_range[1]]
        if len(selected) != 1:
            continue
        attribute = selected[0]
        if (attribute.get('tag') != 'visible' or type(attribute.get('Type')) is not int or
                attribute['Type'] != 0 or type(attribute.get('Value')) is not int or attribute['Value'] not in (0, 1)):
            continue
        results[variant['name']] = bool(attribute['Value'])
    return results


def _placement_check(placement):
    if not isinstance(placement, dict) or placement.get('source') != 'sii-road-look':
        return False
    offset = placement.get('road_offset_m')
    return (placement.get('road_offset_present') is True and type(offset) in (int, float) and
            abs(offset) <= 0.001 and math.isfinite(offset) and
            placement.get('unresolved_offset_keys') == [])


def instance_check(road):
    instance = road.get('instance_placement')
    if not isinstance(instance, dict) or instance.get('source') != 'trucklib-road-instance':
        return False
    for key in ('variant_override_count', 'additional_part_count'):
        if type(instance.get(key)) is not int or instance[key] != 0:
            return False
    height = instance.get('right_height_offset_m')
    return type(height) in (int, float) and abs(height) <= 0.001 and math.isfinite(height)


def road_check(road, checks):
    variants, placement = checks.get(road.get('road_type'), ({}, False))
    visibility = variants.get(road.get('variant_right'))
    if visibility is False:
        state = 'inactive_collision_part'
    elif visibility is None:
        state = 'needs_variant_evidence'
    elif not placement:
        state = 'needs_placement_evidence'
    elif not instance_check(road):
        state = 'needs_instance_evidence'
    else:
        state = 'surface_geometry_checked'
    return dict(state=state, variant_checked=visibility is not None, placement_checked=placement,
                instance_checked=instance_check(road),
                surface_geometry_checked=state == 'surface_geometry_checked',
                drivable_boundary_verified=False, braking_authority=False)


def description(check):
    if check is None:
        return 'No eligible model or unambiguous road match at this position.'
    messages = {
        'inactive_collision_part': 'The surface part is inactive in the selected variant; its width is not used.',
        'needs_variant_evidence': 'Variant-part evidence is missing. Prepare the map again with the latest package.',
        'needs_placement_evidence': 'Variant checked; placement/offset evidence is missing or unsupported.',
        'needs_instance_evidence': 'Road-instance variant, additional-part or height evidence is missing or unsupported.',
        'surface_geometry_checked': 'Model surface: variant, zero offset and road instance checked. Lane/shoulder boundaries are unverified.',
    }
    return messages.get(check['state'], 'Model check could not be completed.')
