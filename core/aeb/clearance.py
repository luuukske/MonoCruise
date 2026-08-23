"""Clearance-based required decel: what it takes to arrive behind a target.

See `core/aeb/README.md` section 5 (continuous-decel logic) for the model and
for why the co-directional limit reproduces `v_rel^2 / (2 * gap)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from core.radar.traffic import ArcPath
from core.aeb.calibration import AEBCalibration
from core.aeb.lane_frame import project_to_ego_arc

_INF = float("inf")

# Capsule fractions (rear to front) sampled for corridor occupancy. Interior
# points are load-bearing; see core/aeb/README.md (clearance-based demand).
_BODY_SAMPLES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class ClearanceResult:
    """Minimum constant ego decel that keeps ego's front behind the target."""
    required_ms2: float
    t_bind_s: float
    s_bind_m: float
    v_pass_ms: float
    pad_rate_ms: float
    clears: bool
    n_samples: int


def extend_horizon(arc: ArcPath, horizon_s: float) -> ArcPath:
    """Same curve, longer reach. Only the `position_at_dist` clamp moves."""
    if horizon_s <= arc.horizon:
        return arc
    return replace(arc, horizon=horizon_s).build()


def sample_times(
    near_horizon_s: float, far_horizon_s: float, n_near: int, n_far: int,
) -> list[float]:
    """Dense grid where the hit lives, sparse tail for a distant co-dir peak."""
    n_near = max(int(n_near), 1)
    out = [near_horizon_s * i / n_near for i in range(n_near + 1)]
    if n_far > 0 and far_horizon_s > near_horizon_s:
        span = far_horizon_s - near_horizon_s
        out.extend(near_horizon_s + span * i / n_far for i in range(1, n_far + 1))
    return out


def occupancy_profile(
    ego_arc: ArcPath,
    target_arcs: list[ArcPath],
    times: list[float],
    cal: AEBCalibration,
) -> tuple[list[tuple[float, float]], bool]:
    """[(t, s_near)] where a target body sits in ego's corridor, plus `clears`.

    `s_near` is the smallest ego arc-length of any in-corridor body point. The
    lateral test runs before the arc-length one so `project_to_ego_arc` and its
    `atan2` are only paid for points already inside the corridor.
    """
    straight = ego_arc.is_straight
    e_cx, e_cz, e_r = ego_arc.center_x, ego_arc.center_z, ego_arc.radius
    e_fx, e_fz = ego_arc.fwd_x, ego_arc.fwd_z
    e_sx, e_sz = ego_arc.start_x, ego_arc.start_z
    e_hw = ego_arc.half_width
    e_pms = ego_arc.parallel_margin_scale
    margin = cal.corridor_margin

    out: list[tuple[float, float]] = []
    last_idx = -1
    for idx, t in enumerate(times):
        s_near: float | None = None
        for arc in target_arcs:
            dist = arc._dist_at_time(t)
            ax, az = arc.position_at_dist(dist)
            head = arc.heading_at_dist(dist)
            t_fx = -math.sin(head)
            t_fz = -math.cos(head)
            back = -arc._cap_back
            span = arc._cap_fwd + arc._cap_back
            hw_sum = e_hw + arc.half_width

            mid = back + span * 0.5
            mx = ax + mid * t_fx
            mz = az + mid * t_fz
            m_du = mx - e_sx
            m_dv = mz - e_sz
            m_s = m_du * e_fx + m_dv * e_fz
            if straight:
                m_lat = abs(m_dv * e_fx - m_du * e_fz)
            else:
                m_lat = abs(e_r - math.hypot(mx - e_cx, mz - e_cz))

            # Near-parallel capsules use a reduced margin, same sine blend as
            # `_sampled_collision`, so both layers agree on "in the corridor".
            pms = e_pms if e_pms < arc.parallel_margin_scale else arc.parallel_margin_scale
            if pms < 1.0 and margin > 0.0:
                e_head = ego_arc.heading_at_dist(m_s if m_s > 0.0 else 0.0)
                cosd = abs(-math.sin(e_head) * t_fx + -math.cos(e_head) * t_fz)
                sind_sq = 1.0 - cosd * cosd
                sind = math.sqrt(sind_sq) if sind_sq > 0.0 else 0.0
                thr = hw_sum + margin * (pms + (1.0 - pms) * sind)
            else:
                thr = hw_sum + margin

            if m_lat > thr + span * 0.5:
                continue

            for frac in _BODY_SAMPLES:
                off = back + span * frac
                px = ax + off * t_fx
                pz = az + off * t_fz
                if straight:
                    p_du = px - e_sx
                    p_dv = pz - e_sz
                    p_s = p_du * e_fx + p_dv * e_fz
                    p_lat = abs(p_dv * e_fx - p_du * e_fz)
                else:
                    p_lat = abs(e_r - math.hypot(px - e_cx, pz - e_cz))
                    if p_lat > thr:
                        continue
                    p_s, p_lat_arc = project_to_ego_arc(ego_arc, px, pz)
                    if p_lat_arc > p_lat:
                        p_lat = p_lat_arc
                if p_lat > thr or p_s <= 0.0:
                    continue
                # Capsule segment ends are inset by half_width; the cap radius
                # is what reaches ego, so the near surface sits that much closer.
                p_surf = p_s - arc.half_width
                if s_near is None or p_surf < s_near:
                    s_near = p_surf
        if s_near is not None:
            out.append((t, s_near))
            last_idx = idx

    clears = bool(out) and last_idx < len(times) - 1
    if clears and cal.clearance_clear_margin_s > 0.0:
        # Do not shave the tail of a crosser: hold the last conflict position
        # for a margin past the frame it vacates on.
        t_last, s_last = out[-1]
        out.append((t_last + cal.clearance_clear_margin_s, s_last))
    return out, clears


def required_at(v0: float, t: float, d_avail: float, lag_s: float) -> float:
    """Min constant decel keeping ego's front behind `d_avail` at time `t`."""
    if d_avail <= 0.0:
        return _INF
    lag_dist = v0 * lag_s
    t_brake = t - lag_s
    if t_brake <= 0.0:
        return _INF if v0 * t > d_avail else 0.0
    if lag_dist >= d_avail:
        return _INF
    a_roll = 2.0 * (v0 * t - d_avail) / (t_brake * t_brake)
    if a_roll <= 0.0:
        return 0.0
    if a_roll * t_brake <= v0:
        return a_roll
    return (v0 * v0) / (2.0 * (d_avail - lag_dist))


def _interp_s(profile: list[tuple[float, float]], idx: int, t: float) -> float:
    """Linear `s_near` across segment `idx`; `s_near(t)` is smooth inside a run."""
    lo = min(max(idx, 0), len(profile) - 1)
    hi = min(lo + 1, len(profile) - 1)
    t0, s0 = profile[lo]
    t1, s1 = profile[hi]
    if t1 <= t0:
        return s0
    return s0 + (s1 - s0) * (t - t0) / (t1 - t0)


def _extend_tail(
    profile: list[tuple[float, float]], v0: float, offset: float, lag_s: float,
) -> list[tuple[float, float]]:
    """Linear-extrapolation samples past a profile whose peak is off the window.

    A target still occupying the corridor at the last sample has not cleared, so
    holding its measured rate is the same constant-velocity assumption the
    relative-frame formula always made. With `s_near` linear in `t` the peak sits
    at `t* = lag + 2 * (C - dv * lag) / dv`, and evaluating there recovers
    `dv^2 / (2 * (gap - dv * lag))` exactly.
    """
    if len(profile) < 2:
        return []
    t_last, s_last = profile[-1]
    t_prev, s_prev = profile[-2]
    if t_last <= t_prev:
        return []
    rate = (s_last - s_prev) / (t_last - t_prev)
    dv = v0 - rate
    if dv <= 1e-3:
        return []
    c_term = (s_last - offset) - rate * t_last
    denom = c_term - dv * lag_s
    if denom <= 1e-3:
        return []
    t_star = lag_s + 2.0 * denom / dv
    if t_star <= t_last:
        return []
    t_star = min(t_star, t_last + 60.0)
    out = []
    for i in range(1, 9):
        t = t_last + (t_star - t_last) * i / 8.0
        out.append((t, s_last + rate * (t - t_last)))
    return out


def min_decel_to_clear(
    v0: float,
    profile: list[tuple[float, float]],
    clears: bool,
    cal: AEBCalibration,
    *,
    lag_s: float,
    pad_m: float,
    front_to_surface: float,
) -> ClearanceResult | None:
    """Max over the occupancy profile of the per-sample decel requirement."""
    if not profile or v0 <= 1e-3:
        return None

    offset = front_to_surface + cal.stop_buffer + pad_m
    n_measured = len(profile)

    def scan(seq, start, best, best_i, best_t, best_d):
        for i, (t, s) in enumerate(seq, start):
            req = required_at(v0, t, s - offset, lag_s)
            if req > best:
                best, best_i, best_t, best_d = req, i, t, s - offset
                if req == _INF:
                    break
        return best, best_i, best_t, best_d

    best, best_i, best_t, best_d = scan(profile, 0, -_INF, 0, profile[0][0], 0.0)

    if not clears and best < _INF:
        # Still in the corridor at the window edge, so the demand peak may sit
        # past it. Extrapolating restores the old formula for exactly that case.
        tail = _extend_tail(profile, v0, offset, lag_s)
        if tail:
            start = len(profile)
            profile = profile + tail
            best, best_i, best_t, best_d = scan(
                tail, start, best, best_i, best_t, best_d,
            )

    steps = int(cal.clearance_refine_steps)
    if steps > 0 and 0.0 < best < _INF and len(profile) > 1:
        lo_i = max(best_i - 1, 0)
        hi_i = min(best_i + 1, len(profile) - 1)
        t_mid = profile[best_i][0]
        t_lo = profile[lo_i][0]
        t_hi = profile[hi_i][0]
        for _ in range(steps):
            if t_hi - t_lo <= 1e-6:
                break
            t_a = t_lo + (t_hi - t_lo) * 0.382
            t_b = t_lo + (t_hi - t_lo) * 0.618
            d_a = _interp_s(profile, lo_i if t_a < t_mid else best_i, t_a) - offset
            d_b = _interp_s(profile, lo_i if t_b < t_mid else best_i, t_b) - offset
            r_a = required_at(v0, t_a, d_a, lag_s)
            r_b = required_at(v0, t_b, d_b, lag_s)
            if r_a >= r_b:
                t_hi = t_b
                if r_a > best:
                    best, best_t, best_d = r_a, t_a, d_a
            else:
                t_lo = t_a
                if r_b > best:
                    best, best_t, best_d = r_b, t_b, d_b

    required = 0.0 if best < 0.0 else best

    if required == _INF or required <= 0.0 or best_d <= 0.0:
        v_pass = max(v0, 0.0) if required <= 0.0 else 0.0
    else:
        brake_dist = best_d - v0 * lag_s
        v_sq = v0 * v0 - 2.0 * required * max(brake_dist, 0.0)
        v_pass = math.sqrt(v_sq) if v_sq > 0.0 else 0.0

    # Constraint closing rate at the binding sample: the frame-correct analogue
    # of the old `v_closing`, and what seeds the latched build-up reserve.
    ds_dt = 0.0
    if len(profile) > 1:
        lo_i = max(best_i - 1, 0)
        hi_i = min(best_i + 1, len(profile) - 1)
        if hi_i > lo_i and profile[hi_i][0] > profile[lo_i][0]:
            ds_dt = ((profile[hi_i][1] - profile[lo_i][1])
                     / (profile[hi_i][0] - profile[lo_i][0]))
    pad_rate = v0 - ds_dt
    if pad_rate < 0.0:
        pad_rate = 0.0

    return ClearanceResult(
        required_ms2=required,
        t_bind_s=best_t,
        s_bind_m=best_d + offset,
        v_pass_ms=v_pass,
        pad_rate_ms=pad_rate,
        clears=clears,
        n_samples=n_measured,
    )


def clearance_required(
    ego_arc: ArcPath,
    target_arcs: list[ArcPath],
    ego_speed: float,
    cal: AEBCalibration,
    *,
    lag_s: float,
    pad_m: float,
    front_to_surface: float,
    near_horizon_s: float,
) -> ClearanceResult | None:
    """Clearance demand for one target. None means the caller must fall back."""
    if not target_arcs or ego_speed <= 1e-3:
        return None
    far_h = max(cal.clearance_horizon_s, near_horizon_s)
    n_far = int(cal.clearance_far_samples)
    arcs = target_arcs
    if n_far > 0 and far_h > near_horizon_s:
        arcs = [extend_horizon(a, far_h) for a in target_arcs]
    times = sample_times(near_horizon_s, far_h, cal.clearance_samples, n_far)
    profile, clears = occupancy_profile(ego_arc, arcs, times, cal)
    if not profile:
        return None
    return min_decel_to_clear(
        ego_speed, profile, clears, cal,
        lag_s=lag_s, pad_m=pad_m, front_to_surface=front_to_surface,
    )
