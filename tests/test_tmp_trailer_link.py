"""TMP trailer to tractor: the SDK rig link first, the geometric lock as fallback.

See core/acc/README.md §4 and core/aeb/README.md, TMP trailer swap. Vehicles face -z at yaw 0,
so a tractor sits at a more negative z than its trailer.
"""
from __future__ import annotations

from core.acc.trailer_lock import resolve_tractor, sdk_tractor
from core.aeb.thread import _swap_trailer_kinematics
from core.radar.tmp_sdk import FLAG_LINEAR, FLAG_PLACEMENT, KIND_TRACTOR, KIND_TRAILER, SdkBody
from core.radar.tmp_state import rigs_of
from tests.aeb.harness import make_vehicle


def _sdk(v, kind: int, rig: int):
    v.sdk = SdkBody(v.position.x, 1.0, v.position.z, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, -1.0, 0, 0.0, kind, FLAG_PLACEMENT | FLAG_LINEAR, rig)
    return v


def _trailer(vid=1, speed=0.0):
    return make_vehicle(vid, 0.0, 0.0, 0.0, speed, length=13.6, is_tmp=True, is_trailer=True)


def _tractor(vid, x, z, yaw_deg=0.0, speed=0.0):
    return make_vehicle(vid, x, z, yaw_deg, speed, length=6.0, is_tmp=True)


def test_the_sdk_link_beats_a_tidier_geometric_candidate():
    trailer = _sdk(_trailer(), KIND_TRAILER, 3)
    # Real tractor mid-bend: 30 deg and 2 m off the trailer axis, so the strict gate rejects it.
    real = _sdk(_tractor(2, 2.0, -10.0, yaw_deg=30.0), KIND_TRACTOR, 3)
    decoy = _sdk(_tractor(3, 0.0, -8.0), KIND_TRACTOR, 4)
    cache: dict[int, int] = {}
    assert resolve_tractor(trailer, [trailer, real, decoy], cache) is real
    assert cache == {1: 2}
    assert resolve_tractor(trailer, [trailer, real, decoy], {}, use_sdk=False) is decoy


def test_a_tractor_the_sdk_puts_on_another_rig_is_never_picked():
    trailer = _sdk(_trailer(), KIND_TRAILER, 3)
    other_rig = _sdk(_tractor(3, 0.0, -8.0), KIND_TRACTOR, 4)
    assert resolve_tractor(trailer, [trailer, other_rig], {}) is None
    # Not matched to any SDK body: geometry still decides.
    unmatched = _tractor(4, 0.0, -8.0)
    assert resolve_tractor(trailer, [trailer, unmatched], {}) is unmatched


def test_a_cached_lock_on_another_rig_is_dropped():
    trailer = _sdk(_trailer(), KIND_TRAILER, 3)
    wrong = _sdk(_tractor(3, 0.0, -8.0), KIND_TRACTOR, 4)
    cache = {1: 3}
    assert resolve_tractor(trailer, [trailer, wrong], cache) is None
    assert cache == {}


def test_the_lock_outlives_a_lost_link():
    trailer = _sdk(_trailer(), KIND_TRAILER, 3)
    bend = _sdk(_tractor(2, 2.0, -10.0, yaw_deg=30.0), KIND_TRACTOR, 3)
    cache: dict[int, int] = {}
    assert resolve_tractor(trailer, [trailer, bend], cache) is bend
    trailer.sdk = None
    bend.sdk = None
    # The loose revalidation holds what the link acquired; a cold strict acquire would not.
    assert resolve_tractor(trailer, [trailer, bend], cache) is bend
    assert resolve_tractor(trailer, [trailer, bend], {}) is None


def test_a_link_to_a_distant_record_is_refused():
    trailer = _sdk(_trailer(), KIND_TRAILER, 3)
    far = _sdk(_tractor(2, 0.0, -40.0), KIND_TRACTOR, 3)
    assert sdk_tractor(trailer, [trailer, far]) is None


def test_aeb_swap_takes_the_linked_tractors_speed_not_a_passing_truck():
    trailer = _sdk(_trailer(speed=0.0), KIND_TRAILER, 3)
    tractor = _sdk(_tractor(2, 0.0, -9.5, speed=0.0), KIND_TRACTOR, 3)
    # Next lane, level with the trailer and nearer to it than its own tractor: the old nearest pick.
    passing = _tractor(3, 3.5, -2.0, speed=20.0)
    for link in (True, False):
        if not link:
            trailer.sdk = tractor.sdk = None
        out = _swap_trailer_kinematics([trailer, tractor, passing], {})
        assert out[0].speed == 0.0
        assert out[0]._debug_kinematics_swapped


def test_aeb_swap_hands_a_stalled_trailer_its_tractors_speed():
    trailer = _sdk(_trailer(speed=0.0), KIND_TRAILER, 3)
    tractor = _sdk(_tractor(2, 0.0, -9.5, speed=15.0), KIND_TRACTOR, 3)
    out = _swap_trailer_kinematics([trailer, tractor], {})
    assert out[0].speed == 15.0
    assert out[0] is not trailer and trailer.speed == 0.0


def test_aeb_swap_leaves_parked_trailers_alone():
    parked = make_vehicle(1, 0.0, 0.0, 0.0, 0.0, length=13.6, is_trailer=True)
    parked.is_parked = True
    passing = make_vehicle(2, 0.0, -9.5, 0.0, 20.0, length=6.0)
    out = _swap_trailer_kinematics([parked, passing], {})
    assert out[0] is parked and out[0].speed == 0.0


def test_aeb_swap_forgets_trailers_that_left():
    trailer = _sdk(_trailer(), KIND_TRAILER, 3)
    tractor = _sdk(_tractor(2, 0.0, -9.5), KIND_TRACTOR, 3)
    cache: dict[int, int] = {}
    _swap_trailer_kinematics([trailer, tractor], cache)
    assert cache == {1: 2}
    _swap_trailer_kinematics([tractor], cache)
    assert cache == {}


def test_whole_rigs_follow_the_sdk_link():
    trailer = _sdk(_trailer(), KIND_TRAILER, 3)
    bend = _sdk(_tractor(2, 2.0, -10.0, yaw_deg=30.0), KIND_TRACTOR, 3)
    vehicles = [trailer, bend]
    boxes = {v.id: None for v in vehicles}
    assert rigs_of({1}, vehicles, boxes) == [frozenset({1, 2})]
    trailer.sdk = bend.sdk = None
    assert rigs_of({1}, vehicles, boxes) == [frozenset({1})]
