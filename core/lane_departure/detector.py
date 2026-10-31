"""Lane departure detector: decides when a drift out of the lane earns a rumble.

Pure logic with no threads and no telemetry reads. Design and tuning: core/lane_departure/README.md.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from core.radar.ego_geometry import REF_HALF_LENGTH_M, REF_HALF_WIDTH_M, REF_PATH_ORIGIN_M

SIDE_NONE = 0
SIDE_LEFT = 1
SIDE_RIGHT = 2
LEVEL_NONE = 0
LEVEL_CLOSE = 1
LEVEL_WARN = 2
_SIDES = (SIDE_LEFT, SIDE_RIGHT)
_SIGN = {SIDE_LEFT: -1.0, SIDE_RIGHT: 1.0}
_INF = math.inf
_RETURNING_MS = 0.1
_KAPPA_SPAN_M = 25.0
_BEND_STEP_M = 10.0
_EXTEND_M = 10.0
# Silences after which the lane has to be re-learned; a fork or a fast bend ahead keeps it.
_FRESH_START = frozenset({"no_data", "no_map", "route_kind", "off_route", "stale"})


@dataclass(frozen=True)
class Tuning:
    half_lane_m: float = 2.25
    arm_kmh: float = 60.0
    disarm_kmh: float = 50.0
    warn_overlap_m: float = -0.1
    approach_tlc_s: float = 0.8
    approach_hold: float = 1.5
    approach_min_ms: float = 0.4
    approach_range_m: float = 0.4
    approach_scale: float = 0.6
    return_ms: float = 0.2
    horizon_s: float = 2.0
    accel_cap_ms2: float = 1.5
    accel_weight: float = 0.5
    history_s: float = 0.6
    min_history_s: float = 0.3
    bend_gain_m2: float = 80.0
    bend_cap_m: float = 0.3
    bend_behind_m: float = 10.0
    bend_ahead_min_m: float = 30.0
    bend_ahead_s: float = 1.5
    kappa_sharp: float = 1.0 / 90.0
    cue_tight_kappa: float = 1.0 / 400.0
    cue_change_kappa: float = 1.0 / 1000.0
    cue_change_ahead_s: float = 3.0
    cue_hold_s: float = 2.0
    attentive_s: float = 1.5
    attentive_net: float = 0.008
    attentive_peak: float = 0.03
    prefab_max_kappa: float = 1.0 / 400.0
    rearm_m: float = 0.15
    rearm_hold_s: float = 0.3
    cooldown_s: float = 1.0
    max_per_min: int = 12
    min_rumble_s: float = 1.0
    max_rumble_s: float = 4.0
    out_far_m: float = 1.5
    grace_s: float = 0.3
    max_age_s: float = 0.25
    blinker_hold_s: float = 2.0
    deliberate_hold_s: float = 1.0
    steer_window_s: float = 2.0
    steer_net: float = 0.06
    steer_peak: float = 0.3
    evade_net: float = 0.15
    evade_peak: float = 0.6
    evade_hold_s: float = 2.0
    steer_bend_kappa: float = 1.0 / 500.0
    intent_psi_rad: float = math.radians(1.2)
    intent_net: float = 0.015
    intent_peak: float = 0.06
    intent_hold_s: float = 2.0
    steer_peak_dt_s: float = 0.3
    steer_min_s: float = 0.6
    steady_s: float = 1.0
    lane_jump_m: float = 1.5
    brake_gate: float = 0.7
    lookahead_min_m: float = 40.0
    lookahead_s: float = 2.0
    jump_m: float = 80.0
    test_mode: bool = False


@dataclass(frozen=True)
class Frame:
    """One tick of inputs. ``steer_right`` is positive when the wheel is turned right.

    ``path_m`` is the rear axle group ahead of the origin (negative: behind), the point that rolls along the heading.
    """

    t: float
    age_s: float
    x: float
    z: float
    yaw_rad: float
    speed_ms: float
    steer_right: float
    gear: int
    brake: float
    blinker_left: bool
    blinker_right: bool
    hazards: bool
    paused: bool
    aeb_active: bool
    half_width_m: float
    front_m: float
    rear_m: float
    path_m: float
    route: object | None


@dataclass(frozen=True)
class Decision:
    """``active`` is the rumble request, ``level`` its kind. ``near_miss`` is what muted a close call."""

    active: bool = False
    side: int = SIDE_NONE
    level: int = LEVEL_NONE
    reason: str = "off"
    near_miss: str = ""
    started: bool = False
    gap_m: float = _INF
    allow_m: float = 0.0
    tlc_s: float = _INF
    closing_ms: float = 0.0
    psi_deg: float = 0.0
    kappa: float = 0.0
    speed_kmh: float = 0.0
    steer_net: float = 0.0
    steer_peak: float = 0.0
    steer_away: float = 0.0
    kappa_change: float = 0.0


@dataclass
class _Side:
    blink_until: float = -_INF
    delib_until: float = -_INF
    intent_until: float = -_INF
    latched: bool = False
    armed_warn: bool = True
    armed_close: bool = True
    inside_since: float | None = None
    last_warn: float = -_INF
    last_close: float = -_INF


@dataclass(frozen=True)
class _Edge:
    """``gap`` is the body's distance inside the lane edge, ``allow`` how far past it the warning line sits."""

    gap: float
    allow: float
    bend: float
    drift: float
    tlc: float
    returning: bool
    close: bool
    hard: str
    armed_warn: bool
    armed_close: bool
    steer_net: float = 0.0
    steer_peak: float = 0.0
    cue_mute: str = ""
    steer_away: float = 0.0

    @property
    def margin(self) -> float:
        return self.gap + self.allow


def body_from_geometry(geometry) -> tuple[float, float, float, float]:
    """Ego's half width, bumper ahead of and rear end behind the placement origin, and its path origin."""
    if geometry is None:
        return REF_HALF_WIDTH_M, REF_HALF_LENGTH_M, REF_HALF_LENGTH_M, REF_PATH_ORIGIN_M
    return (
        float(geometry.half_width_m), float(geometry.front_m), float(geometry.rear_m),
        float(geometry.path_origin_m),
    )


def project(pts, x: float, z: float):
    """``(arc length, metres right of the polyline, tangent x, tangent z)`` at the nearest span.

    The end spans extend ``_EXTEND_M`` past the polyline: the route starts at ego, and ego's rear
    sits behind it.
    """
    best = None
    walked = 0.0
    last = len(pts) - 2
    for i in range(len(pts) - 1):
        ax, az = pts[i]
        sx, sz = pts[i + 1][0] - ax, pts[i + 1][1] - az
        seg2 = sx * sx + sz * sz
        if seg2 < 1e-9:
            continue
        seg = math.sqrt(seg2)
        ext = _EXTEND_M / seg
        u = ((x - ax) * sx + (z - az) * sz) / seg2
        u = min(1.0 + ext if i == last else 1.0, max(-ext if i == 0 else 0.0, u))
        dist2 = (x - ax - u * sx) ** 2 + (z - az - u * sz) ** 2
        if best is None or dist2 < best[0]:
            tx, tz = sx / seg, sz / seg
            best = (dist2, walked + u * seg, (x - ax) * -tz + (z - az) * tx, tx, tz)
        walked += seg
    return None if best is None else best[1:]


def _span_mids(pts) -> list[tuple[float, float]]:
    """``(arc length, unwrapped tangent angle)`` at the middle of every span."""
    mids = []
    walked = 0.0
    prev = None
    for i in range(len(pts) - 1):
        dx, dz = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        hop = math.hypot(dx, dz)
        if hop < 1e-9:
            continue
        angle = math.atan2(dz, dx)
        if prev is not None:
            angle = prev + math.atan2(math.sin(angle - prev), math.cos(angle - prev))
        mids.append((walked + 0.5 * hop, angle))
        prev = angle
        walked += hop
    return mids


def _heading_at(mids, s: float):
    if not mids or s <= mids[0][0]:
        return mids[0][1] if mids else None
    for (s0, a0), (s1, a1) in zip(mids, mids[1:]):
        if s <= s1:
            return a0 + (s - s0) / (s1 - s0) * (a1 - a0)
    return mids[-1][1]


def _kappa_at(mids, s: float, span: float = _KAPPA_SPAN_M) -> float:
    a0, a1 = _heading_at(mids, s), _heading_at(mids, s + span)
    if a0 is None or a1 is None:
        return 0.0
    return (a1 - a0) / span


def smooth_heading(pts, s: float):
    """Tangent angle ``atan2(dz, dx)`` ``s`` metres along the polyline, or None without a span.

    Interpolated between span midpoints: a span's chord is the tangent at its middle, so the
    raw span direction would wobble by half a span over the radius on every bend.
    """
    return _heading_at(_span_mids(pts), s)


def lane_kappa_right(pts, s: float, span: float = _KAPPA_SPAN_M) -> float:
    """Curvature of the lane over ``span`` metres from ``s``; positive turns right."""
    return _kappa_at(_span_mids(pts), s, span)


def bend_allowance(kappa_toward: float, tn: Tuning) -> float:
    """Metres past an edge allowed for cutting a bend that turns toward it, capped."""
    return min(tn.bend_cap_m, tn.bend_gain_m2 * max(0.0, kappa_toward))


def bend_peaks(pts, s0: float, speed_ms: float, tn: Tuning) -> tuple[float, float]:
    """Peak lane curvature toward ``(left, right)`` from just behind ego to the road ahead.

    Looking ahead covers turning in before the bend, where the lane under ego is still straight.
    """
    mids = _span_mids(pts)
    left = right = 0.0
    s, end = s0 - tn.bend_behind_m, s0 + max(tn.bend_ahead_min_m, tn.bend_ahead_s * speed_ms)
    while s <= end:
        k = _kappa_at(mids, s)
        right, left = max(right, k), max(left, -k)
        s += _BEND_STEP_M
    return left, right


def curvature_span(pts, s0: float, end: float) -> tuple[float, float]:
    """Lowest and highest signed lane curvature from ``s0`` to ``end``; positive turns right.

    Starts at ego: the route has no road behind its foot point, and a clipped span reads as a bend ending.
    """
    mids = _span_mids(pts)
    low = high = _kappa_at(mids, s0)
    s = s0 + _BEND_STEP_M
    while s <= end:
        k = _kappa_at(mids, s)
        low, high = min(low, k), max(high, k)
        s += _BEND_STEP_M
    return low, high


def time_to_cross(gap: float, closing: float, accel: float, horizon: float) -> float:
    """Seconds until a gap closes at constant lateral acceleration; 0 when already over."""
    if gap <= 0.0:
        return 0.0 if closing >= -_RETURNING_MS else _INF
    if abs(accel) < 1e-6:
        t = gap / closing if closing > 1e-6 else _INF
    else:
        disc = closing * closing + 2.0 * accel * gap
        if disc < 0.0:
            return _INF
        root = math.sqrt(disc)
        t = min((r for r in ((-closing + root) / accel, (-closing - root) / accel) if r > 0.0), default=_INF)
    return t if t <= horizon else _INF


def _slope(samples) -> float:
    n = len(samples)
    mean_t = sum(t for t, _ in samples) / n
    mean_y = sum(y for _, y in samples) / n
    den = sum((t - mean_t) ** 2 for t, _ in samples)
    if den < 1e-9:
        return 0.0
    return sum((t - mean_t) * (y - mean_y) for t, y in samples) / den


def _steer_features(hist, tn: Tuning) -> tuple[float, float, float]:
    """``(net change, peak rightward rate, peak leftward rate)`` of the wheel over the window.

    Rates use pairs ``steer_peak_dt_s`` apart. Too little history reads as a wheel that did nothing.
    """
    if len(hist) < 3 or hist[-1][0] - hist[0][0] < tn.steer_min_s:
        return 0.0, 0.0, 0.0
    right = left = 0.0
    j = 0
    for i in range(len(hist)):
        while hist[i][0] - hist[j][0] > tn.steer_peak_dt_s:
            j += 1
        dt = hist[i][0] - hist[j][0]
        if dt >= 0.6 * tn.steer_peak_dt_s:
            rate = (hist[i][1] - hist[j][1]) / dt
            right = max(right, rate)
            left = max(left, -rate)
    return hist[-1][1] - hist[0][1], right, left


class LaneDetector:
    def __init__(self, tuning: Tuning | None = None) -> None:
        self.tune = tuning or Tuning()
        self._fast = False
        self._last_xz: tuple[float, float] | None = None
        self._last_t = 0.0
        self._sides = {SIDE_LEFT: _Side(), SIDE_RIGHT: _Side()}
        self._warns: deque[float] = deque()
        self._active = SIDE_NONE
        self._level = LEVEL_NONE
        self._level_since = 0.0
        self._lost_since: float | None = None
        self._evade_until = -_INF
        self._tight_until = -_INF
        self._change_until = -_INF
        self._steer: deque[tuple[float, float]] = deque()
        self._clear_track()

    def _clear_track(self) -> None:
        self._hist: deque[tuple[float, float]] = deque()
        self._lane_since: float | None = None
        self._last_right0: float | None = None

    def _clear_active(self) -> None:
        self._active = SIDE_NONE
        self._level = LEVEL_NONE
        self._lost_since = None

    def _restart(self) -> None:
        self._clear_track()
        self._steer.clear()
        self._evade_until = -_INF
        self._tight_until = self._change_until = -_INF
        self._clear_active()
        for st in self._sides.values():
            st.latched = False
            st.blink_until = st.delib_until = st.intent_until = -_INF

    def _off(self, reason: str, kmh: float) -> Decision:
        self._hist.clear()
        if reason in _FRESH_START:
            self._lane_since = None
            self._last_right0 = None
        self._clear_active()
        return Decision(reason=reason, speed_kmh=kmh)

    def update(self, f: Frame | None) -> Decision:
        if f is None:
            return self._off("no_data", 0.0)
        tn = self.tune
        if self._last_xz is not None and (
            math.hypot(f.x - self._last_xz[0], f.z - self._last_xz[1]) > tn.jump_m
            or f.t - self._last_t > 1.0
        ):
            self._restart()
        self._last_xz = (f.x, f.z)
        self._last_t = f.t
        self._steer.append((f.t, f.steer_right))
        while f.t - self._steer[0][0] > tn.steer_window_s:
            self._steer.popleft()
        kmh = f.speed_ms * 3.6
        reason = self._gate(f, kmh)
        if reason:
            return self._off(reason, kmh)
        return self._evaluate(f, kmh)

    def _gate(self, f: Frame, kmh: float) -> str:
        tn = self.tune
        if f.route is None:
            return "no_map"
        if f.paused:
            return "paused"
        if f.age_s > tn.max_age_s:
            return "stale"
        if f.gear < 0:
            return "reverse"
        if kmh >= tn.arm_kmh:
            self._fast = True
        elif kmh < tn.disarm_kmh:
            self._fast = False
        if not self._fast:
            return "speed"
        if f.aeb_active:
            return "aeb"
        if f.brake >= tn.brake_gate:
            return "brake"
        return ""

    def _evaluate(self, f: Frame, kmh: float) -> Decision:
        tn = self.tune
        t = f.t
        pts = getattr(f.route, "pts", ())
        fx, fz = -math.sin(f.yaw_rad), -math.cos(f.yaw_rad)
        rx, rz = -fz, fx
        hit = project(pts, f.x, f.z)
        if hit is None or abs(hit[1]) > 3.0 * tn.half_lane_m:
            return self._off("off_route", kmh)
        rows = {}
        for side in _SIDES:
            rows[side] = []
            for along in (f.front_m, -f.rear_m):
                corner = project(
                    pts,
                    f.x + fx * along + _SIGN[side] * rx * f.half_width_m,
                    f.z + fz * along + _SIGN[side] * rz * f.half_width_m,
                )
                if corner is None:
                    return self._off("off_route", kmh)
                rows[side].append((tn.half_lane_m - _SIGN[side] * corner[1], along - f.path_m))
        s0, right0 = hit[0], hit[1]
        reason = self._route_reason(f, s0)
        if reason:
            return self._off(reason, kmh)
        kappa = lane_kappa_right(pts, s0)
        if abs(kappa) >= tn.kappa_sharp:
            return self._off("bend_sharp", kmh)
        peaks = dict(zip(_SIDES, bend_peaks(pts, s0, f.speed_ms, tn)))
        kind = getattr(f.route, "kind", "")
        if kind != "road" and not (kind == "prefab" and max(peaks.values()) <= tn.prefab_max_kappa):
            return self._off("route_kind", kmh)
        # Tight corners never cue; where the curvature changes (entry, exit, wavy roads) only a still wheel does.
        ahead = max(tn.bend_ahead_min_m, tn.cue_change_ahead_s * f.speed_ms)
        low, high = curvature_span(pts, s0, min(s0 + ahead, getattr(f.route, "reach_m", 0.0) - _KAPPA_SPAN_M))
        if max(-low, high, *peaks.values()) >= tn.cue_tight_kappa:
            self._tight_until = t + tn.cue_hold_s
        if high - low >= tn.cue_change_kappa:
            self._change_until = t + tn.cue_hold_s
        cue_zone = "bend_tight" if self._tight_until > t else "bend_change" if self._change_until > t else ""

        if self._lane_since is None:
            self._lane_since = t
        if self._last_right0 is not None and abs(right0 - self._last_right0) > tn.lane_jump_m:
            self._clear_track()
            self._lane_since = t
            self._clear_active()
            for st in self._sides.values():
                st.latched = False
        self._last_right0 = right0
        # Heading against the lane at the rear axle: from the origin a steady bend reads as kappa * d of drift outward.
        tangent = smooth_heading(pts, project(pts, f.x + fx * f.path_m, f.z + fz * f.path_m)[0])
        ego_angle = math.atan2(fz, fx)
        psi = math.atan2(math.sin(ego_angle - tangent), math.cos(ego_angle - tangent))
        self._hist.append((t, psi))
        while t - self._hist[0][0] > tn.history_s:
            self._hist.popleft()
        ready = len(self._hist) >= 4 and t - self._hist[0][0] >= tn.min_history_s
        psi_rate = _slope(list(self._hist)) if ready else 0.0
        acc_right = tn.accel_weight * max(-tn.accel_cap_ms2, min(tn.accel_cap_ms2, f.speed_ms * math.cos(psi) * psi_rate))
        motion = (f.speed_ms * math.sin(psi), psi_rate, acc_right, psi, ready)
        steer = self._steer_input(t, peaks)
        att_net, att_right, att_left = _steer_features([p for p in self._steer if t - p[0] <= tn.attentive_s], tn)

        edges = {}
        for side in _SIDES:
            lamp = f.blinker_left if side == SIDE_LEFT else f.blinker_right
            toward_peak = steer[1] if side == SIDE_RIGHT else steer[2]
            away = (att_net, att_right) if side == SIDE_LEFT else (-att_net, att_left)
            edges[side] = self._edge(
                side, f, rows[side], peaks[side], (_SIGN[side] * steer[0], toward_peak), lamp, motion,
                cue_zone, away,
            )
        if t - self._lane_since < tn.steady_s:
            return Decision(reason="lane_settle", speed_kmh=kmh, kappa=kappa)
        return self._decide(t, kmh, kappa, psi, edges, high - low)

    def _steer_input(self, t: float, peaks) -> tuple[float, float, float]:
        """Wheel movement as intent evidence: steering the way a tight bend turns is the bend, not intent.

        A hard move in either direction (a swerve and its counter-steer) mutes both sides for a while.
        """
        tn = self.tune
        net, right, left = _steer_features(self._steer, tn)
        follows = {side: peaks[side] >= tn.steer_bend_kappa for side in _SIDES}
        if follows[SIDE_RIGHT]:
            right, net = 0.0, min(net, 0.0)
        if follows[SIDE_LEFT]:
            left, net = 0.0, max(net, 0.0)
        if abs(net) >= tn.evade_net or max(right, left) >= tn.evade_peak:
            self._evade_until = t + tn.evade_hold_s
        return net, right, left

    def _route_reason(self, f: Frame, s0: float) -> str:
        tn = self.tune
        route = f.route
        look = max(tn.lookahead_min_m, tn.lookahead_s * f.speed_ms)
        if getattr(route, "reach_m", 0.0) - s0 < look:
            return "short_route"
        for name in ("branch", "merge"):
            s = getattr(route, name + "_s", None)
            if s is not None and s - s0 < look:
                return name
        return ""

    def _edge(self, side, f, rows, peak, steer_toward, lamp, motion, cue_zone, away) -> _Edge:
        """``rows`` holds ``(gap, distance ahead of the rear axle)`` per outer corner, ``peak`` the bend toward this edge.

        ``away`` is the wheel's recent ``(net, peak rate)`` away from this edge, the driver correcting.
        """
        tn = self.tune
        t = f.t
        sign = _SIGN[side]
        drift_right, psi_rate, acc_right, psi, ready = motion
        st = self._sides[side]
        bend = bend_allowance(peak, tn)
        allow = tn.warn_overlap_m + bend
        gap = min(g for g, _ in rows)
        margin = gap + allow
        if lamp or f.hazards:
            st.blink_until = t + tn.blinker_hold_s
        if steer_toward[0] >= tn.steer_net or steer_toward[1] >= tn.steer_peak:
            st.delib_until = t + tn.deliberate_hold_s
        # Heading toward the edge that the wheel itself produced is a lane change; a still wheel is a drift.
        if sign * psi >= tn.intent_psi_rad and (steer_toward[0] >= tn.intent_net or steer_toward[1] >= tn.intent_peak):
            st.intent_until = t + tn.intent_hold_s
        evading = self._evade_until > t
        if margin <= 0.0 and (st.blink_until > t or st.delib_until > t or st.intent_until > t or evading):
            st.latched = True
        # Back just inside the warning line re-arms the side; it never has to recentre.
        if margin >= tn.rearm_m:
            st.latched = False
            if st.inside_since is None:
                st.inside_since = t
            if t - st.inside_since >= tn.rearm_hold_s:
                if t - st.last_warn >= tn.cooldown_s:
                    st.armed_warn = True
                if t - st.last_close >= tn.cooldown_s:
                    st.armed_close = True
        else:
            st.inside_since = None
        hard = ""
        if st.blink_until > t:
            hard = "blinker"
        elif st.delib_until > t:
            hard = "deliberate"
        elif st.intent_until > t:
            hard = "intent"
        elif evading:
            hard = "evade"
        elif st.latched:
            hard = "latched"
        if tn.test_mode:
            hard = ""
        drift = sign * drift_right
        tlc = min(
            time_to_cross(g + allow, sign * (drift_right + along * psi_rate), sign * acc_right, tn.horizon_s)
            for g, along in rows
        )
        close = ready and drift >= tn.approach_min_ms
        correcting = away[0] >= tn.attentive_net or away[1] >= tn.attentive_peak
        cue_mute = cue_zone if cue_zone == "bend_tight" or (cue_zone and correcting) else ""
        return _Edge(
            gap, allow, bend, drift, tlc, drift <= -tn.return_ms, close, hard,
            st.armed_warn, st.armed_close, steer_toward[0], steer_toward[1],
            "" if tn.test_mode else cue_mute, away[0],
        )

    def _want(self, e: _Edge) -> int:
        """The level this edge asks for: at or past the warning line, or closing on it fast enough."""
        tn = self.tune
        if e.margin <= 0.0:
            return LEVEL_WARN if e.gap > -tn.out_far_m else LEVEL_NONE
        near = e.margin <= tn.approach_range_m
        return LEVEL_CLOSE if e.close and near and e.tlc <= tn.approach_tlc_s else LEVEL_NONE

    @staticmethod
    def _mute(e: _Edge, level: int) -> str:
        if e.hard:
            return e.hard
        if level == LEVEL_CLOSE and e.cue_mute:
            return e.cue_mute
        if level == LEVEL_WARN and e.returning:
            return "returning"
        if not (e.armed_warn if level == LEVEL_WARN else e.armed_close):
            return "unarmed"
        return ""

    def _hold(self, t: float, e: _Edge) -> None:
        """Keep or end the running rumble. Nothing but a gate ends it before ``min_rumble_s``."""
        tn = self.tune
        age = t - self._level_since
        if self._level == LEVEL_WARN:
            keep = e.margin <= 0.0 and not e.hard and not e.returning
        else:
            keep = (
                e.margin > 0.0 and e.tlc <= tn.approach_tlc_s * tn.approach_hold and not e.hard
                and e.cue_mute != "bend_tight"
            )
        over = age >= tn.max_rumble_s or (self._level == LEVEL_WARN and e.gap <= -tn.out_far_m)
        if over:
            due = True
        elif keep:
            self._lost_since = None
            due = False
        else:
            if self._lost_since is None:
                self._lost_since = t
            due = t - self._lost_since >= tn.grace_s
        if due and age >= tn.min_rumble_s:
            self._stop(t)

    def _decide(self, t, kmh, kappa, psi, edges, kappa_change) -> Decision:
        tn = self.tune
        started = False
        if self._active != SIDE_NONE:
            e = edges[self._active]
            if self._level == LEVEL_CLOSE and self._want(e) == LEVEL_WARN and not self._mute(e, LEVEL_WARN):
                self._begin(self._active, LEVEL_WARN, t)
                started = True
            else:
                self._hold(t, e)
        near_miss = ""
        if self._active == SIDE_NONE:
            while self._warns and t - self._warns[0] > 60.0:
                self._warns.popleft()
            for side in sorted(_SIDES, key=lambda s: edges[s].margin):
                e = edges[side]
                if 0.0 < e.margin <= e.bend:
                    near_miss = near_miss or "bend_inner"
                level = self._want(e)
                if not level:
                    continue
                muted = self._mute(e, level) or ("rate" if len(self._warns) >= tn.max_per_min else "")
                if muted:
                    near_miss = near_miss or muted
                    continue
                self._begin(side, level, t)
                started = True
                break
        active = self._active != SIDE_NONE
        focus = self._active if active else min(_SIDES, key=lambda s: edges[s].margin)
        e = edges[focus]
        return Decision(
            active=active, side=focus if active else SIDE_NONE, level=self._level,
            reason=("warn" if self._level == LEVEL_WARN else "approach") if active else (near_miss or "clear"),
            near_miss=near_miss, started=started, gap_m=e.gap, allow_m=e.allow, tlc_s=e.tlc, closing_ms=e.drift,
            psi_deg=math.degrees(psi), kappa=kappa, speed_kmh=kmh, steer_net=e.steer_net, steer_peak=e.steer_peak,
            steer_away=e.steer_away, kappa_change=kappa_change,
        )

    def _begin(self, side: int, level: int, t: float) -> None:
        st = self._sides[side]
        st.armed_close = False
        st.last_close = t
        if level == LEVEL_WARN:
            st.armed_warn = False
            st.last_warn = t
        if self._active == SIDE_NONE:
            self._warns.append(t)
        self._active = side
        self._level = level
        self._level_since = t
        self._lost_since = None

    def _stop(self, t: float) -> None:
        st = self._sides[self._active]
        st.armed_close = False
        st.last_close = t
        if self._level == LEVEL_WARN:
            st.armed_warn = False
            st.last_warn = t
        self._clear_active()
