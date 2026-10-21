"""Bodies deep inside ego are ghosts for AEB until they separate. See core/aeb/README.md §4."""

from __future__ import annotations

from core.aeb.calibration import DEFAULT
from core.aeb.clip_eval import _make_headless, _snapshot_tuple
from core.aeb.clip_schema import EgoTelemetry
from core.aeb.overlap_ghosts import OverlapGhostHold
from core.radar.tmp_state import ego_box
from core.radar.traffic import Position, Quaternion, Size, Trailer
from tests.aeb.harness import make_vehicle

_EGO = ego_box(0.0, 0.0, 0.0, None)        # reference rig at the origin, facing -z
_DEPTH = DEFAULT.overlap_ghost_depth_m
_LINGER = 0.5 * (DEFAULT.overlap_ghost_contact_m + _DEPTH)   # clearly inside the slack, short of the depth
_FRONT_Z = -_EGO.half_length               # ego's front bumper


_clock = [0.0]


def _now(dt: float = 0.033) -> float:
    """A steady 30 Hz step clock shared by the hold tests."""
    _clock[0] += dt
    return _clock[0]


def _body(vid: int, inside_m: float, **kw):
    """A stopped 10 m body straight ahead whose rear sits ``inside_m`` into ego's front."""
    return make_vehicle(vid, 0.0, _FRONT_Z - 5.0 + inside_m, 0.0, 0.0, **kw)


def test_a_body_deep_inside_ego_stays_a_ghost_until_it_separates():
    hold = OverlapGhostHold()
    assert hold.step(_EGO, [_body(5, _DEPTH + 0.3)], _now(), DEFAULT) == {5}
    # Backing out to a touch keeps it: it has not separated yet.
    assert hold.step(_EGO, [_body(5, 0.1)], _now(), DEFAULT) == {5}
    assert hold.step(_EGO, [_body(5, -0.5)], _now(), DEFAULT) == frozenset()
    assert hold.step(_EGO, [_body(5, 0.1)], _now(), DEFAULT) == frozenset()
    # Unlike the zone-exit hold, a new deep overlap is a ghost again.
    assert hold.step(_EGO, [_body(5, _DEPTH + 0.3)], _now(), DEFAULT) == {5}


def test_contact_short_of_the_depth_is_not_a_ghost_at_once():
    hold = OverlapGhostHold()
    bumper = _body(5, _LINGER, is_tmp=True)
    neighbour = make_vehicle(6, 3.6, 0.0, 0.0, 0.0, is_tmp=True)
    assert hold.step(_EGO, [bumper, neighbour], _now(), DEFAULT) == frozenset()


def test_a_body_that_stays_inside_ego_becomes_a_ghost():
    # Real contact lasts a frame before AEB latches; 0.08 s at most on the labelled corpus.
    hold = OverlapGhostHold()
    body = _body(5, _LINGER)
    persist = DEFAULT.overlap_ghost_persist_s
    assert hold.step(_EGO, [body], 0.0, DEFAULT) == frozenset()
    assert hold.step(_EGO, [body], persist - 0.03, DEFAULT) == frozenset()
    assert hold.step(_EGO, [body], persist, DEFAULT) == {5}


def test_box_slack_and_broken_contact_never_add_up_to_a_ghost():
    hold = OverlapGhostHold()
    slack = _body(5, DEFAULT.overlap_ghost_contact_m - 0.1)
    assert all(hold.step(_EGO, [slack], _now(), DEFAULT) == frozenset() for _ in range(60))
    inside = _body(6, _LINGER)
    hold.step(_EGO, [inside], _now(), DEFAULT)
    # A pause longer than the persistence window starts the count over.
    assert hold.step(_EGO, [inside], _now(1.0), DEFAULT) == frozenset()


def test_ai_traffic_deep_inside_ego_is_a_ghost_too():
    hold = OverlapGhostHold()
    assert hold.step(_EGO, [_body(8, _DEPTH + 0.3)], _now(), DEFAULT) == {8}


def test_a_truckersmp_trailer_inside_ego_takes_its_tractor():
    hold = OverlapGhostHold()
    trailer = make_vehicle(11, 0.0, -6.0, 0.0, 0.0, is_tmp=True, is_trailer=True)
    tractor = make_vehicle(10, 0.0, -17.0, 0.0, 0.0, is_tmp=True, length=6.0)
    other = make_vehicle(12, 0.0, -40.0, 0.0, 0.0, is_tmp=True)
    assert {10, 11} <= hold.step(_EGO, [trailer, tractor, other], _now(), DEFAULT)
    assert 12 not in hold.ids


def test_a_nested_trailer_inside_ego_takes_the_vehicle_that_tows_it():
    hold = OverlapGhostHold()
    tractor = make_vehicle(20, 0.0, -20.0, 0.0, 0.0)
    tractor.trailers = [Trailer(Position(0.0, 0.0, -6.0), Quaternion(1.0, 0.0, 0.0, 0.0),
                                Size(2.5, 1.5, 10.0), slot=0)]
    assert 20 in hold.step(_EGO, [tractor], _now(), DEFAULT)


def _aeb_thread(vehicles, latched=frozenset()):
    t = _make_headless(DEFAULT)
    t._latched_threat_ids = set(latched)
    ego = EgoTelemetry(coordinateX=0.0, coordinateZ=0.0, rotationX=0.0, speed=10.0)
    snap = _snapshot_tuple(ego, vehicles, 10.0, frozenset(), 10.0, frozenset())
    t._read_radar_snapshot = lambda: snap
    t._read_ego_geometry = lambda: None
    t._read_max_brake_ms2 = lambda: 7.8
    t._read_user_braking = lambda: False
    t._read_addressing_brake = lambda: False
    t._read_vehicle_key = lambda: None
    t._now = lambda: 10.0
    t.loop()
    return t


def _aeb_frame(vehicles, latched=frozenset()):
    return _aeb_thread(vehicles, latched).data.snapshot


def test_aeb_draws_a_deep_overlap_as_a_ghost_and_never_evaluates_it():
    snap = _aeb_frame([_body(7, _DEPTH + 0.5, is_tmp=True)])
    assert snap.ghost_ids == {7}
    assert 7 not in snap.colliding_ids
    assert next(v for v in snap.vehicles if v["vid"] == 7)["ghost"] is True


def test_aeb_keeps_a_body_at_the_bumper():
    snap = _aeb_frame([_body(7, 0.2, is_tmp=True)])
    assert snap.ghost_ids == set() and 7 in snap.colliding_ids


def test_a_latched_threat_deep_inside_ego_is_a_ghost_and_loses_its_latch():
    # Lukas 2026-10-20: past the depth bar a body is ignored no matter what, latched included.
    t = _aeb_thread([_body(7, _DEPTH + 0.5, is_tmp=True)], latched={7})
    assert t.data.snapshot.ghost_ids == {7} and 7 not in t.data.snapshot.colliding_ids
    assert 7 not in t._latched_threat_ids
