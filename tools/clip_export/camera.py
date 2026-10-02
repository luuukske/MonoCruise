"""Auto-framing: keep ego and the threat in frame, centred, with no dead space.

Ego always points straight up, as in the debug view; only pan and zoom move. The
threat is framed from its entry into the widest shot, so an export that starts there
never zooms out to find it. Every smoothing pass is zero-phase. See
`tools/clip_export/README.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tools.clip_export.timeline import FrameState, local_linear, smoothstep


@dataclass(frozen=True)
class Camera:
    x: float
    z: float
    yaw: float
    ppm: float


@dataclass(frozen=True)
class FramingConfig:
    margin_x: float = 0.14
    margin_y: float = 0.10
    min_span_m: float = 34.0
    max_span_m: float = 170.0
    lookahead_s: float = 1.6
    lookahead_min_m: float = 16.0
    threat_lead_in_s: float = 1.5
    threat_release_s: float = 1.5
    pan_sigma_s: float = 0.45
    zoom_sigma_s: float = 0.55
    zoom_hold_s: float = 0.8


def to_view(x: float, z: float, cx: float, cz: float, yaw: float) -> tuple[float, float]:
    """World point to camera metres: +u screen right, +v screen down, ego forward up."""
    dx, dz = x - cx, z - cz
    c, s = math.cos(yaw), math.sin(yaw)
    return dx * c - dz * s, dx * s + dz * c


def from_view(u: float, v: float, cx: float, cz: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return cx + u * c + v * s, cz - u * s + v * c


def threat_weight(t: float, span: tuple[float, float] | None, cfg: FramingConfig,
                  ramp_from: float | None = None) -> float:
    """How much the primary threat counts toward the frame at ``t``.

    Ramps in from ``ramp_from`` (default: a lead-in before the span), out after the span.
    """
    if span is None:
        return 0.0
    a, b = span
    if t > b:
        return smoothstep(1.0 - (t - b) / cfg.threat_release_s)
    start = a - cfg.threat_lead_in_s if ramp_from is None else ramp_from
    return smoothstep((t - start) / cfg.threat_lead_in_s)


def entry_time(states: list[FrameState], width: int, height: int,
               span: tuple[float, float] | None, cfg: FramingConfig | None = None) -> float | None:
    """First time ego and the primary threat both fit the widest allowed shot, by the span end."""
    if span is None:
        return None
    cfg = cfg if cfg is not None else FramingConfig()
    ppm_lo = min(width, height) / cfg.max_span_m
    avail_w = width * (1.0 - 2.0 * cfg.margin_x)
    avail_h = height * (1.0 - 2.0 * cfg.margin_y)
    for st in states:
        if st.t > span[1]:
            break
        threat = _threat_points(st)
        if not threat:
            continue
        box = _bbox(_ego_points(st, cfg) + threat, st.ego.x, st.ego.z, st.ego.yaw)
        if (box[2] - box[0]) * ppm_lo <= avail_w and (box[3] - box[1]) * ppm_lo <= avail_h:
            return st.t
    return None


def _ego_points(st: FrameState, cfg: FramingConfig) -> list[tuple[float, float]]:
    pts = st.ego.corners()
    if st.ego_trailer is not None:
        pts += st.ego_trailer.corners()
    reach = max(cfg.lookahead_min_m, abs(st.ego_speed_ms) * cfg.lookahead_s)
    pts.append((st.ego.x - math.sin(st.ego.yaw) * reach, st.ego.z - math.cos(st.ego.yaw) * reach))
    return pts


def _threat_points(st: FrameState, pull: float = 1.0) -> list[tuple[float, float]]:
    """Primary threat corners, drawn toward ego by ``1 - pull`` so containment has no step."""
    view = st.vehicle(st.primary_vid)
    if view is None or pull <= 0.0:
        return []
    pts = view.body.corners()
    for tr in view.trailers:
        pts += tr.corners()
    ex, ez = st.ego.x, st.ego.z
    return [(ex + (x - ex) * pull, ez + (z - ez) * pull) for x, z in pts]


def _bbox(pts, cx, cz, yaw) -> tuple[float, float, float, float]:
    uv = [to_view(x, z, cx, cz, yaw) for x, z in pts]
    us = [u for u, _ in uv]
    vs = [v for _, v in uv]
    return min(us), min(vs), max(us), max(vs)


def _smooth(ts: list[float], vs: list[float], sigma: float) -> list[float]:
    return [local_linear(ts, vs, t, sigma) for t in ts]


def _sliding_min(ts: list[float], vs: list[float], half_s: float) -> list[float]:
    out = []
    lo = hi = 0
    for t in ts:
        while ts[lo] < t - half_s:
            lo += 1
        while hi + 1 < len(ts) and ts[hi + 1] <= t + half_s:
            hi += 1
        out.append(min(vs[lo:hi + 1]))
    return out


def solve(states: list[FrameState], width: int, height: int,
          span: tuple[float, float] | None, cfg: FramingConfig | None = None) -> list[Camera]:
    """One camera per state. ``span`` is when AEB tracked the primary threat."""
    if not states:
        return []
    cfg = cfg if cfg is not None else FramingConfig()
    ts = [st.t for st in states]
    entry = entry_time(states, width, height, span, cfg)
    ramp_from = None if entry is None else (float("-inf") if entry <= ts[0] else entry)
    short = float(min(width, height))
    ppm_lo, ppm_hi = short / cfg.max_span_m, short / cfg.min_span_m
    avail_w = width * (1.0 - 2.0 * cfg.margin_x)
    avail_h = height * (1.0 - 2.0 * cfg.margin_y)

    # Camera yaw is ego's own heading: the centre offset is smoothed in that frame, so a
    # turn swings the world around ego instead of sliding ego across the screen.
    yaws = [st.ego.yaw for st in states]
    weights, off_u, off_v, log_ppm = [], [], [], []
    for st, yaw in zip(states, yaws):
        view = st.vehicle(st.primary_vid)
        w = threat_weight(st.t, span, cfg, ramp_from) if view is not None else 0.0
        threat = _threat_points(st)
        ex, ez = st.ego.x, st.ego.z
        box = _bbox(_ego_points(st, cfg), ex, ez, yaw)
        if threat and w > 0.0:
            tb = _bbox(threat, ex, ez, yaw)
            union = (min(box[0], tb[0]), min(box[1], tb[1]), max(box[2], tb[2]), max(box[3], tb[3]))
            box = tuple(a + (b - a) * w for a, b in zip(box, union))
        bw, bh = max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)
        weights.append(w)
        off_u.append((box[0] + box[2]) / 2.0)
        off_v.append((box[1] + box[3]) / 2.0)
        ppm = min(avail_w / bw, avail_h / bh, ppm_hi)
        log_ppm.append(math.log(max(ppm, ppm_lo)))

    off_u = _smooth(ts, off_u, cfg.pan_sigma_s)
    off_v = _smooth(ts, off_v, cfg.pan_sigma_s)
    log_ppm = _smooth(ts, _sliding_min(ts, log_ppm, cfg.zoom_hold_s), cfg.zoom_sigma_s)
    centres = [from_view(u, v, st.ego.x, st.ego.z, yaw)
               for st, yaw, u, v in zip(states, yaws, off_u, off_v)]

    # Containment pass: the smoothed centre may have drifted off a fast subject.
    contain = []
    for st, yaw, w, (cx, cz) in zip(states, yaws, weights, centres):
        pts = _ego_points(st, cfg) + _threat_points(st, w)
        limit = ppm_hi
        for x, z in pts:
            u, v = to_view(x, z, cx, cz, yaw)
            if abs(u) > 1e-6:
                limit = min(limit, width * (0.5 - cfg.margin_x / 2.0) / abs(u))
            if abs(v) > 1e-6:
                limit = min(limit, height * (0.5 - cfg.margin_y / 2.0) / abs(v))
        contain.append(math.log(max(limit, ppm_lo)))
    tightened = [min(a, b) for a, b in zip(log_ppm, contain)]
    tightened = _smooth(ts, _sliding_min(ts, tightened, cfg.zoom_hold_s / 2.0), cfg.zoom_sigma_s / 2.0)

    cams = []
    for yaw, (cx, cz), lp, lc in zip(yaws, centres, tightened, contain):
        ppm = min(math.exp(min(lp, lc)), ppm_hi)
        cams.append(Camera(cx, cz, yaw, max(ppm, ppm_lo)))
    return cams
