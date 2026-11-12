"""Lane departure detector on routes built from straights and arcs, with the truck rolling on its rear axle."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.lane_departure.detector import SIDE_LEFT, SIDE_NONE, SIDE_RIGHT, LaneDetector, smooth_heading
from core.radar.ego_geometry import REF_PATH_ORIGIN_M
from tests.lane_departure.test_detector import DT, HZ, SPEED, TN, cues, frame_at, warns


def composite_route(*pieces, step=5.0):
    """Lane centre from the origin heading north, from ``(length, radius)`` pieces; radius > 0 bends right, None is straight."""
    pts, x, z, heading = [(0.0, 0.0)], 0.0, 0.0, -0.5 * math.pi
    for length, radius in pieces:
        for _ in range(int(length / step)):
            turn = 0.0 if radius is None else step / radius
            x, z = x + step * math.cos(heading + 0.5 * turn), z + step * math.sin(heading + 0.5 * turn)
            heading += turn
            pts.append((x, z))
    return SimpleNamespace(
        pts=tuple(pts), kind="road", divergence=(), branch_s=None, merge_s=None, reach_m=step * (len(pts) - 1),
    )


def pose_on(pts, s, lateral, psi, lead_m=0.0):
    """``(x, z, yaw)`` of an origin ``lead_m`` ahead of a point ``lateral`` right of the lane, ``psi`` right of its tangent."""
    walked = 0.0
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        hop = math.hypot(bx - ax, bz - az)
        if walked + hop >= s:
            break
        walked += hop
    u = (s - walked) / hop
    theta = smooth_heading(pts, s)
    angle = theta + psi
    x = ax + u * (bx - ax) - lateral * math.sin(theta) + lead_m * math.cos(angle)
    z = az + u * (bz - az) + lateral * math.cos(theta) + lead_m * math.sin(angle)
    return x, z, math.atan2(-math.cos(angle), -math.sin(angle))


def drive_on(route, s_start, plan, seconds, lead_m=0.0, **fixed):
    """``drive`` along any polyline; the planned point rolls along its heading ``lead_m`` behind the origin."""
    detector = LaneDetector()
    out = []
    for i in range(int(seconds * HZ)):
        t = i * DT
        lateral, psi, extras = plan(t)
        x, z, yaw = pose_on(route.pts, s_start + SPEED * t, lateral, psi, lead_m)
        frame = frame_at(t, route, None, 0.0, 0.0, **{**fixed, **extras})
        out.append(detector.update(replace(frame, x=x, z=z, yaw_rad=yaw)))
    return out


def test_a_steady_bend_reads_no_lateral_speed_at_the_rear_axle():
    """Rolling on a concentric line: read at the rear axle nothing moves, read at the origin it drifts outward by v kappa d."""
    route = composite_route((600.0, 150.0))
    lead = -REF_PATH_ORIGIN_M
    rear = drive_on(route, 40.0, lambda t: (-0.5, 0.0, {}), 6.0, lead_m=lead, path_m=REF_PATH_ORIGIN_M)
    origin = drive_on(route, 40.0, lambda t: (-0.5, 0.0, {}), 6.0, lead_m=lead, path_m=0.0)
    settled = slice(int(2.0 * HZ), None)
    assert all(d.side == SIDE_NONE and d.reason != "lane_settle" for d in rear[settled])
    assert max(abs(d.closing_ms) for d in rear[settled]) < 0.03
    assert min(d.closing_ms for d in origin[settled]) == pytest.approx(SPEED * lead / 150.0, abs=0.05)


def test_holding_a_line_past_the_inside_allowance_of_a_steady_bend_warns():
    """Read at the origin, the inside edge looked like a truck moving back to the centre, and that muted the warning."""
    route = composite_route((600.0, 200.0))
    lead = -REF_PATH_ORIGIN_M
    rear = drive_on(route, 40.0, lambda t: (1.3, 0.0, {}), 6.0, lead_m=lead, path_m=REF_PATH_ORIGIN_M)
    origin = drive_on(route, 40.0, lambda t: (1.3, 0.0, {}), 6.0, lead_m=lead, path_m=0.0)
    assert warns(rear) and warns(rear)[0].side == SIDE_RIGHT
    assert not warns(origin) and any(d.near_miss == "returning" for d in origin)


def drift_left_at(route, s_cue, correction):
    """A drift to the left that would cue at ``s_cue``; ``correction`` turns the wheel right, away from that edge."""
    t0, rate = 2.0, 0.6

    def plan(t):
        steer = {"steer_right": correction * max(0.0, t - t0 + 0.5)}
        if t < t0:
            return -0.3, 0.0, steer
        return -0.3 - rate * (t - t0), -math.atan2(rate, SPEED), steer

    return drive_on(route, s_cue - SPEED * (t0 + 0.2), plan, 4.0)


@pytest.mark.parametrize("s_cue,changing", [
    (150.0, False), (250.0, True), (470.0, False), (650.0, True), (760.0, False),
])
def test_a_wheel_turned_home_holds_the_cue_only_where_the_lane_s_curvature_is_steady(s_cue, changing):
    """Straight to 300 m, a slight right bend R 700 to 700 m, straight again; here the truck never answers the wheel.

    Where the curvature changes ahead a wheel that moves is mostly the bend, and the truck itself has to curve in.
    """
    route = composite_route((300.0, None), (400.0, 700.0), (400.0, None))
    still = drift_left_at(route, s_cue, 0.0)
    correcting = drift_left_at(route, s_cue, 0.02)
    assert cues(still) and cues(still)[0].side == SIDE_LEFT
    assert bool(cues(correcting)) == changing
    warn = warns(correcting)[0]
    assert warn.side == SIDE_LEFT and -TN.back_reach_m - 0.1 < warn.gap_m + warn.allow_m <= 0.0
    assert changing or any(d.near_miss == "correcting" for d in correcting)


@pytest.mark.parametrize("s_cue", [250.0, 360.0, 430.0])
def test_running_wide_of_a_tight_corner_cues_before_the_line(s_cue):
    """A right corner R 300 from 300 to 420 m: approaching, inside it, just out of it."""
    route = composite_route((300.0, None), (120.0, 300.0), (400.0, None))
    decisions = drift_left_at(route, s_cue, 0.0)
    cue, warn = cues(decisions)[0], warns(decisions)[0]
    assert cue.side == warn.side == SIDE_LEFT and decisions.index(cue) < decisions.index(warn)


def test_following_a_bend_with_the_wheel_is_not_steering_back_from_its_outer_line():
    """The same turn of the wheel: on a straight it is the driver correcting, entering a right corner it is the bend."""
    route = composite_route((300.0, None), (120.0, 300.0), (400.0, None))
    assert not cues(drift_left_at(route, 150.0, 0.012))
    assert cues(drift_left_at(route, 305.0, 0.012))[0].side == SIDE_LEFT


def into_the_bend(drifts_from, extra_turn=0.0, seconds=7.0):
    """A right corner R 300 from 300 m: the wheel turns in as the lane does (0.0185 of lock), ``extra_turn`` more for a
    cut; from ``drifts_from`` seconds the truck runs to the inside at 0.9 m/s. Starts 100 m before the corner."""
    route = composite_route((300.0, None), (600.0, 300.0))
    rate = 0.9

    def plan(t):
        s = 200.0 + SPEED * t
        turned = min(1.0, max(0.0, (s - 275.0) / 25.0))
        steer = {"steer_right": (1.0 / 300.0 / TN.steer_kappa + extra_turn) * turned}
        if t < drifts_from:
            return 0.0, 0.0, steer
        return rate * (t - drifts_from), math.atan2(rate, SPEED), steer

    return drive_on(route, 200.0, plan, seconds)


def test_following_a_bend_in_and_then_drifting_to_its_inside_is_a_drift_not_a_cut():
    """2026-11-12: the wheel a driver turned in with the bend counted as intent for 4 s and the inside line latched;
    trucks ran 1 to 2 m past it with no rumble and no pulse."""
    drifted = into_the_bend(drifts_from=4.6)
    assert warns(drifted) and warns(drifted)[0].side == SIDE_RIGHT
    assert not any(d.near_miss in {"intent", "latched"} for d in drifted)
    cut = into_the_bend(drifts_from=4.6, extra_turn=0.03)
    assert not any(d.active for d in cut) and any(d.near_miss in {"intent", "latched"} for d in cut)

