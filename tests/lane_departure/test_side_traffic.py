"""Side traffic experiment: what makes moving over a lane unsafe, per side, in ego's lane frame."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.lane_departure import side_traffic as st
from core.lane_departure.side_traffic import AHEAD, BEHIND, BESIDE, ONCOMING, Ego, SideTraffic, box
from core.radar.ego_geometry import TrailerBody, trailer_bodies_from_sdk
from core.road_layout.ego_lane import LineIndex
from core.road_layout.ego_route import route_behind

SPEED = 25.0
EGO_S = 100.0
HALF_W, FRONT, REAR = 1.265, 3.333, 3.333


def lane_point(s: float, d: float = 0.0, radius: float | None = None):
    """``(x, z, yaw)`` ``s`` metres along a lane running -z from the origin, ``d`` right of it; ``radius`` > 0 bends right."""
    if radius is None:
        return d, -s, 0.0
    phi = s / radius
    x, z = radius * (1.0 - math.cos(phi)), -radius * math.sin(phi)
    return x + d * math.cos(phi), z + d * math.sin(phi), -phi


def make_route(radius=None, start=EGO_S, length=400.0, step=10.0):
    pts = tuple(lane_point(start + i * step, 0.0, radius)[:2] for i in range(int(length / step) + 1))
    return SimpleNamespace(pts=pts, lane_index=-1)


class Car:
    def __init__(self, s, d, radius=None, speed=SPEED, oncoming=False, length=4.5, width=1.8, vid=1, **kw):
        self.x, self.z, yaw = lane_point(s, d, radius)
        self.yaw = yaw + (math.pi if oncoming else 0.0)
        self.speed, self.length, self.width, self.id = speed, length, width, vid
        self.is_tmp = kw.get("tmp", False)
        self.is_parked = kw.get("parked", False)
        self.trailers = ()

    def get_corners(self):
        half = 0.5 * self.length
        return box(self.x, self.z, self.yaw, half, half, 0.5 * self.width)


def ego(radius=None, speed=SPEED, trailers=()):
    x, z, yaw = lane_point(EGO_S, 0.0, radius)
    return Ego(x, z, yaw, speed, HALF_W, FRONT, REAR, tuple(trailers))


def judge(*cars, radius=None, route=None, window=None, skip=frozenset(), traffic=None, **ego_kw):
    traffic = traffic or SideTraffic()
    return traffic.update(0.0, ego(radius, **ego_kw), cars, skip, route or make_route(radius), window)


def kinds(state):
    return tuple(h.kind if h is not None else None for h in (state.left, state.right))


def test_a_car_beside_the_cab_flags_its_side_only():
    assert kinds(judge(Car(EGO_S, 4.5))) == (None, BESIDE)
    assert kinds(judge(Car(EGO_S + 1.0, -4.5))) == (BESIDE, None)


def test_traffic_in_our_own_lane_or_two_lanes_over_is_not_side_traffic():
    assert kinds(judge(Car(EGO_S + 30.0, 0.3), Car(EGO_S - 12.0, -0.5), Car(EGO_S, 9.0))) == (None, None)


def test_a_car_squeezed_beside_us_in_our_lane_counts_but_an_overlapping_ghost_does_not():
    assert kinds(judge(Car(EGO_S, 2.4))) == (None, BESIDE)
    assert kinds(judge(Car(EGO_S, 0.5))) == (None, None)


def test_a_car_closing_fast_from_behind_counts_until_it_is_too_far_or_too_slow():
    gap = 25.0
    behind = EGO_S - REAR - 2.25 - gap
    hit = judge(Car(behind, 4.5, speed=SPEED + 10.0)).right
    assert hit.kind == BEHIND
    assert hit.gap_m == pytest.approx(gap, abs=0.01)
    assert hit.time_s == pytest.approx(2.5, abs=0.01)
    assert judge(Car(behind - 20.0, 4.5, speed=SPEED + 10.0)).right is None, "4.5 s away"
    assert judge(Car(behind, 4.5, speed=SPEED - 3.0)).right is None


def test_only_the_time_lookahead_counts_not_the_distance():
    assert judge(Car(EGO_S - REAR - 2.25 - 5.0, -4.5)).left is None, "close behind at our speed: never reaches us"
    assert judge(Car(EGO_S + FRONT + 2.25 + 5.0, -4.5)).left is None, "close ahead at our speed"


def test_a_slower_car_ahead_we_are_catching_counts_inside_three_seconds():
    gap = 20.0
    ahead = EGO_S + FRONT + 2.25 + gap
    hit = judge(Car(ahead, 4.5, speed=SPEED - 8.0)).right
    assert (hit.kind, hit.time_s) == (AHEAD, pytest.approx(2.5, abs=0.01))
    assert judge(Car(ahead, 4.5, speed=SPEED - 5.0)).right is None, "4 s away"
    assert judge(Car(ahead, 4.5, speed=SPEED + 2.0)).right is None, "pulling away"


def test_oncoming_traffic_counts_inside_two_and_a_half_seconds():
    near = judge(Car(EGO_S + 80.0, -4.5, oncoming=True, speed=20.0)).left
    assert near.kind == ONCOMING
    assert near.closing_ms == pytest.approx(SPEED + 20.0)
    assert near.time_s == pytest.approx((80.0 - FRONT - 2.25) / 45.0, abs=0.01)
    assert judge(Car(EGO_S + 130.0, -4.5, oncoming=True, speed=20.0)).left is None, "2.8 s away"
    assert judge(Car(EGO_S - 30.0, -4.5, oncoming=True, speed=20.0)).left is None, "already past"


def test_an_oncoming_car_passing_beside_us_is_beside():
    assert judge(Car(EGO_S, -4.5, oncoming=True)).left.kind == BESIDE


def test_parked_skipped_and_crossing_traffic_is_ignored():
    assert kinds(judge(Car(EGO_S, 4.5, parked=True))) == (None, None)
    assert kinds(judge(Car(EGO_S, 4.5, vid=7), skip=frozenset({7}))) == (None, None)
    crossing = Car(EGO_S + 20.0, 4.5)
    crossing.yaw += 0.5 * math.pi
    assert kinds(judge(crossing)) == (None, None)


def test_the_trailer_makes_a_car_beside_it_beside_us():
    car = Car(EGO_S - 14.0, 4.5)
    assert judge(car).right is None, "behind the cab at our speed"
    trailer = TrailerBody(ahead_m=-10.0, right_m=0.0, yaw_rel_rad=0.0, half_length_m=6.8)
    traffic = SideTraffic()
    assert judge(car, traffic=traffic, trailers=[trailer]).right.kind == BESIDE
    assert traffic.rig_length_m == pytest.approx(FRONT + 10.0 + 6.8, abs=0.01)


def test_in_a_bend_the_lane_frame_puts_oncoming_traffic_in_the_left_lane():
    # 80 m into an R 300 right bend the left lane sits about 6 m right of ego's heading line.
    state = judge(Car(EGO_S + 80.0, -4.5, radius=300.0, oncoming=True, speed=20.0), radius=300.0)
    assert kinds(state) == (ONCOMING, None)
    assert kinds(judge(Car(EGO_S + 2.0, 4.5, radius=300.0), radius=300.0)) == (None, BESIDE)


def _lane(x: float, travel: int):
    pts = tuple((x, -z) for z in range(-300, 801, 10))
    return SimpleNamespace(pts=pts, travel=travel, min_x=x, max_x=x, min_z=-800.0, max_z=300.0, prefab_id=None)


def _window(*lanes):
    return SimpleNamespace(lanes=lanes, index=LineIndex(lanes))


def test_no_lane_on_the_map_past_the_line_means_no_side_traffic_there():
    car = Car(EGO_S + 60.0, -5.0, oncoming=True)
    assert judge(car, window=_window(_lane(0.0, 1))).left is None, "a median or a shoulder"
    hit = judge(car, window=_window(_lane(0.0, 1), _lane(-4.5, -1))).left
    assert hit.kind == ONCOMING
    assert hit.lane == "oncoming"


def test_the_lane_behind_comes_from_the_map():
    window = _window(_lane(0.0, 1))
    behind = route_behind(window, 0, 0.0, -100.0, 0.0, -1.0, 120.0)
    assert behind[0] == pytest.approx((0.0, -100.0))
    assert behind[-1] == pytest.approx((0.0, 20.0))
    assert route_behind(None, 0, 0.0, -100.0, 0.0, -1.0) == ()
    route = SimpleNamespace(pts=make_route().pts, lane_index=0)
    car = Car(EGO_S - REAR - 2.25 - 10.0, 4.5, speed=SPEED + 10.0)
    assert judge(car, route=route, window=_window(_lane(0.0, 1), _lane(4.5, 1))).right.kind == BEHIND


def test_a_flag_is_held_briefly_and_a_gate_drops_it_at_once():
    traffic = SideTraffic()
    route = make_route()
    traffic.update(0.0, ego(), [Car(EGO_S, 4.5)], frozenset(), route, None)
    assert traffic.update(0.2, ego(), [], frozenset(), route, None).right.kind == BESIDE
    assert traffic.update(0.4, ego(), [], frozenset(), route, None).right is None
    traffic.update(1.0, ego(), [Car(EGO_S, 4.5)], frozenset(), route, None)
    slow = traffic.update(1.1, ego(speed=5.0), [Car(EGO_S, 4.5)], frozenset(), route, None)
    assert (slow.reason, slow.right) == ("speed", None)
    assert traffic.update(1.2, None, [], frozenset(), route, None).reason == "no_data"


def test_beside_outranks_a_closer_call_behind():
    state = judge(Car(EGO_S - REAR - 2.25 - 5.0, 4.5, speed=SPEED + 10.0, vid=1), Car(EGO_S, 4.5, vid=2))
    assert (state.right.kind, state.right.vid) == (BESIDE, 2)


def _sdk(trailers, hook_z=1.0):
    return {"coordinateX": 0.0, "coordinateZ": 0.0, "rotationX": 0.0, "truckHookPositionX": 0.0,
            "truckHookPositionZ": hook_z, "trailer": trailers}


def _unit(z=6.5, yaw=0.0, attached=True, wheels=(3.0, 4.3, 5.6)):
    return {"attached": attached, "wheelCount": len(wheels), "wheelPositionZ": list(wheels),
            "worldX": 0.0, "worldZ": z, "rotationX": yaw, "hookPositionX": 0.0, "hookPositionZ": -5.5}


def test_a_trailer_straight_behind_sits_behind_the_tractor():
    (body,) = trailer_bodies_from_sdk(_sdk([_unit()]))
    # Kingpin 5.5 m ahead of the trailer origin, last axle 5.6 m behind it, 3.5 m overhang, 1.3 m setback.
    assert body.half_length_m == pytest.approx(0.5 * (6.8 + 9.1))
    assert body.ahead_m == pytest.approx(-(6.5 - 0.5 * (6.8 - 9.1)))
    assert body.right_m == pytest.approx(0.0)
    assert body.hitch_gap_m == pytest.approx(0.0, abs=1e-6)


def test_an_articulated_trailer_carries_its_angle_and_an_unreadable_one_is_left_out():
    (body,) = trailer_bodies_from_sdk(_sdk([_unit(yaw=0.02)]))
    assert body.yaw_rel_rad == pytest.approx(0.02 * 2.0 * math.pi)
    assert trailer_bodies_from_sdk(_sdk([_unit(attached=False), _unit()])) == ()
    assert trailer_bodies_from_sdk(_sdk([_unit(wheels=())])) == ()
    assert trailer_bodies_from_sdk({}) == ()


def test_a_trailer_far_from_the_fifth_wheel_shows_in_its_hitch_gap():
    (body,) = trailer_bodies_from_sdk(_sdk([_unit()], hook_z=2.5))
    assert body.hitch_gap_m == pytest.approx(1.5)


def test_the_rig_boxes_follow_the_trailer_angle():
    trailer = TrailerBody(ahead_m=-10.0, right_m=0.5, yaw_rel_rad=0.1, half_length_m=6.8)
    boxes = st.rig_boxes(Ego(0.0, 0.0, 0.0, SPEED, HALF_W, FRONT, REAR, (trailer,)))
    assert len(boxes) == 2
    cx = sum(c[0] for c in boxes[1]) / 4.0
    cz = sum(c[1] for c in boxes[1]) / 4.0
    assert (cx, cz) == pytest.approx((0.5, 10.0))
