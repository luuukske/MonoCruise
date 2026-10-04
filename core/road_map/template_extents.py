"""Unverified model surface envelopes, separate from driving boundaries."""
from __future__ import annotations

import math


def resolve_profiles(payload):
    if payload.get('source') != 'trucklib-official':
        return {}
    profiles = payload.get('template_profiles', {})
    if not isinstance(profiles, dict):
        return {}
    resolved = {}
    for token, profile in profiles.items():
        try:
            candidate = _resolve(profile)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            candidate = None
        if candidate is not None:
            resolved[token] = candidate
    return resolved


def _resolve(profile):
    if profile.get('source') != 'trucklib-model-measurements':
        return None
    templates = profile['templates']
    if set(templates) != {'right'}:
        return None
    model = templates['right']
    if model.get('status') != 'read':
        return None
    collision = [part for part in model['parts'] if part['name'] == 'coll']
    if len(collision) != 1 or len(collision[0]['pieces']) != 1:
        return None
    geometry = collision[0]['pieces'][0]['geometry']
    if geometry.get('status') != 'measured' or geometry['vertex_count'] < 4:
        return None
    low, high = float(geometry['min_x']), float(geometry['max_x'])
    min_z, max_z = float(geometry['min_z']), float(geometry['max_z'])
    if not all(math.isfinite(v) for v in (low, high, min_z, max_z)):
        return None
    if not low < 0 < high or not 2 <= high-low <= 40 or abs(low+high) > 0.05:
        return None
    if not 0.1 <= max_z-min_z <= 200:
        return None
    sections = geometry['sections']
    if len(sections) < 2:
        return None
    for section in [geometry, *sections]:
        values = [float(section[key]) for key in ('min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z')]
        if not all(math.isfinite(v) for v in values):
            return None
        sx, ex, sy, ey, sz, ez = values
        if (sx > ex or sy > ey or sz > ez or abs(sy) > 0.1 or abs(ey) > 0.1 or
                abs(sx-low) > 0.05 or abs(ex-high) > 0.05 or sz < min_z-0.01 or ez > max_z+0.01):
            return None
    if (abs(min(float(s['min_z']) for s in sections)-min_z) > 0.01 or
            abs(max(float(s['max_z']) for s in sections)-max_z) > 0.01):
        return None
    variants = frozenset(v['name'] for v in model['variants'] if isinstance(v.get('name'), str))
    return ((high-low)/2, (high-low)/2), variants


def road_extent(road, profiles):
    if road.get('kind') != 'road' or road.get('variant_left'):
        return None
    candidate = profiles.get(road.get('road_type'))
    if candidate is None or road.get('variant_right') not in candidate[1]:
        return None
    return candidate[0]
