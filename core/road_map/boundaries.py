"""Definition extents for display and bounded, read-only roadside diagnostics."""
from __future__ import annotations

import math


def dimensions(road):
    boundary = road.get('boundary')
    if not isinstance(boundary, dict) or boundary.get('source') != 'sii-road-size':
        return None
    try:
        left, right = float(boundary['left_m']), float(boundary['right_m'])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) and 0 <= v <= 40 for v in (left, right)) or left+right < 2:
        return None
    return left, right


def offset_edges(points, widths):
    edges = [[], []]
    for i, point in enumerate(points):
        a, b = points[max(0, i-1)], points[min(len(points)-1, i+1)]
        dx, dz = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dz)
        if length <= 0.01:
            return []
        nx, nz = dz/length, -dx/length
        for edge, offset in zip(edges, (widths[0], -widths[1])):
            edge.append((point[0]+nx*offset, point[1]+nz*offset, point[2]))
    return edges


def assess(geometry, x, z, elevation, radius, speed, cell, limit, *, widths=None, ends=None):
    widths = geometry.boundary_widths if widths is None else widths
    ends = geometry.boundary_ends if ends is None else ends
    values = (x, z, elevation, radius, speed)
    if not all(math.isfinite(v) for v in values) or not 0 < radius <= 25:
        return {'state': 'unknown'}
    candidates = geometry.grid.get((math.floor(x/cell), math.floor(z/cell)), [])
    if not candidates or len(candidates) > limit:
        return {'state': 'unknown'}
    nearest = {}
    for identity, _, kind, a, b, length in candidates:
        if kind != 'road':
            continue
        dx, dz = b[0]-a[0], b[1]-a[1]
        t = max(0, min(1, ((x-a[0])*dx+(z-a[1])*dz)/(length*length)))
        if abs(elevation-a[2]-t*(b[2]-a[2])) > 5:
            continue
        px, pz = x-a[0]-t*dx, z-a[1]-t*dz
        distance = math.hypot(px, pz)
        if distance <= 40 and (identity not in nearest or distance < nearest[identity][0]):
            nearest[identity] = distance, (px*dz-pz*dx)/length
    if not nearest:
        return {'state': 'unknown'}
    clearances = []
    for identity, (distance, lateral) in nearest.items():
        extent = widths.get(identity)
        if extent is None:
            return {'state': 'unknown', 'reason': 'missing_width'}
        if any(math.hypot(x-p[0], z-p[1]) < radius+10 for p in ends[identity]):
            return {'state': 'unknown', 'reason': 'road_end_or_junction'}
        clearance = max(lateral-extent[0], -lateral-extent[1])-radius
        if clearance <= 1:
            return {'state': 'within_or_near_extent', 'road_id': identity}
        clearances.append((clearance, identity, distance))
    clearance, identity, _ = min(clearances)
    return {'state': 'stationary_outside_candidate' if abs(speed) <= 0.3 else 'moving_outside',
            'road_id': identity, 'clearance_m': round(clearance, 2), 'advisory_only': True}
