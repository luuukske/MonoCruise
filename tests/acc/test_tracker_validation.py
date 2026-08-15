"""Tracker-level behaviour on synthetic traffic: no clips, no radar thread.

Covers the stationary-target validation latch and the lock/release ordering the
evidence gating is meant to produce."""
from __future__ import annotations

import math

import pytest

from core.acc.scoring import IN_PATH_THRESHOLD
from core.acc.tracker import ACCTracker

from .harness import make_vehicle


_DT = 1.0 / 30.0
_EGO_YAW = 0.0            # forward = (0, -1) in the radar frame


def _run(tracker: ACCTracker, vehicles, frames: int, ego_speed: float = 20.0,
         start_t: float = 100.0):
    """Tick the tracker ``frames`` times with a static scene; returns final tracks."""
    for i in range(frames):
        tracker.update(
            now_mono=start_t + i * _DT, dt=_DT, vehicles=vehicles,
            ego_x=0.0, ego_z=0.0,
            ego_yaw_rad=_EGO_YAW,
            ego_speed_ms=ego_speed, ego_steer=0.0,
            ego_history_kappa=0.0,
            blinker_left=False, blinker_right=False,
        )
    return tracker.tracks


def _ahead(vid, dist_m, speed, **kw):
    """Vehicle ``dist_m`` directly ahead of ego (forward is -Z at yaw 0)."""
    return make_vehicle(vid, 0.0, -dist_m, speed, yaw_rad=_EGO_YAW, **kw)


def test_moving_in_lane_target_locks():
    tracker = ACCTracker()
    tracks = _run(tracker, [_ahead(1, 40.0, 22.0)], frames=60)
    assert tracks[1].score > IN_PATH_THRESHOLD
    assert tracks[1].in_path
    assert tracks[1].last_evidence > 0.9


def test_never_moved_stationary_target_is_not_validated():
    """A target stationary the whole time earns no trail credit, only the floor.

    The floor is the ego arc, which is a measurement in its own right; without it
    the offset term vanished and such a target scored on ``path`` alone, which is
    the same absence-as-evidence inversion the evidence scaling exists to stop."""
    from core.acc.tracker import ARC_EVIDENCE_FLOOR, VALIDATED_STATIONARY_EVIDENCE

    tracker = ACCTracker()
    stopped = _ahead(2, 40.0, 0.0, history_speed=0.0)
    tracks = _run(tracker, [stopped], frames=60)
    assert not tracks[2].moving_validated
    assert tracks[2].last_evidence == pytest.approx(ARC_EVIDENCE_FLOOR)
    # Ordering invariant: watched-then-stopped must outrank never-moved.
    assert ARC_EVIDENCE_FLOOR < VALIDATED_STATIONARY_EVIDENCE


def test_stationary_target_still_reachable_through_corridor_geometry():
    """Queues that were already stopped must remain trackable, just less eagerly."""
    tracker = ACCTracker()
    stopped = _ahead(3, 25.0, 0.0, history_speed=0.0)
    tracks = _run(tracker, [stopped], frames=90)
    assert tracks[3].in_path
    assert tracks[3].score > IN_PATH_THRESHOLD


def test_target_that_drove_then_stopped_keeps_its_lateral_evidence():
    """Watched it drive this line, so its lateral still counts once it stops."""
    tracker = ACCTracker()
    moving = _ahead(4, 45.0, 20.0)
    _run(tracker, [moving], frames=30)
    assert tracker.tracks[4].moving_validated

    stopped = _ahead(4, 45.0, 0.0, history_speed=0.0)
    tracks = _run(tracker, [stopped], frames=30, start_t=102.0)
    assert tracks[4].moving_validated
    assert tracks[4].last_evidence > 0.0


def test_validated_stationary_target_gains_more_offset_than_a_never_moved_one():
    """The latch separates the two cases through the offset term, not the clamp.

    Both saturate the score ceiling eventually when they sit dead ahead; what
    the latch changes is how much lateral evidence each one contributes."""
    seen_driving = ACCTracker()
    _run(seen_driving, [_ahead(5, 45.0, 20.0)], frames=30)
    _run(seen_driving, [_ahead(5, 45.0, 0.0, history_speed=0.0)], frames=30,
         start_t=102.0)

    never_moved = ACCTracker()
    _run(never_moved, [_ahead(6, 45.0, 0.0, history_speed=0.0)], frames=60)

    validated = seen_driving.tracks[5]
    unseen = never_moved.tracks[6]
    assert validated.last_offset > unseen.last_offset
    assert validated.last_evidence > unseen.last_evidence
    assert unseen.last_offset == 0.0


def test_adjacent_lane_target_does_not_lock():
    tracker = ACCTracker()
    beside = make_vehicle(7, 3.6, -40.0, 22.0, yaw_rad=_EGO_YAW)
    tracks = _run(tracker, [beside], frames=60)
    assert not tracks[7].in_path
    assert tracks[7].score <= IN_PATH_THRESHOLD


def test_oncoming_target_does_not_lock():
    tracker = ACCTracker()
    oncoming = make_vehicle(8, 0.0, -40.0, 22.0, yaw_rad=math.pi)
    tracks = _run(tracker, [oncoming], frames=60)
    assert tracks[8].score <= IN_PATH_THRESHOLD


def test_unconfirmed_stationary_target_is_not_locked_from_far_away():
    """The complaint this gate exists for: shoulder pickup at long range.

    At 100 m the lateral estimate cannot resolve a lane width, so a target with
    no trajectory evidence must not be called in-lane at all."""
    tracker = ACCTracker()
    far = _ahead(10, 100.0, 0.0, history_speed=0.0)
    tracks = _run(tracker, [far], frames=60)
    assert not tracks[10].in_path
    assert tracks[10].last_lat_uncertainty > 2.0


def test_moving_target_is_still_locked_at_the_same_range():
    """The gate must cost nothing where trajectory evidence exists."""
    tracker = ACCTracker()
    tracks = _run(tracker, [_ahead(11, 100.0, 22.0)], frames=60)
    assert tracks[11].in_path
    assert tracks[11].last_lat_uncertainty < 0.5


def test_validated_stationary_target_survives_the_gate_at_range():
    """A queue you watched form stays tracked after it stops."""
    tracker = ACCTracker()
    _run(tracker, [_ahead(12, 90.0, 20.0)], frames=30)
    assert tracker.tracks[12].moving_validated
    tracks = _run(tracker, [_ahead(12, 90.0, 0.0, history_speed=0.0)],
                  frames=30, start_t=102.0)
    assert tracks[12].in_path


def test_unconfirmed_stationary_target_is_reachable_once_close():
    """Nothing is permanently hidden: the gate opens as ego closes in."""
    tracker = ACCTracker()
    tracks = _run(tracker, [_ahead(13, 25.0, 0.0, history_speed=0.0)], frames=90)
    assert tracks[13].in_path
    assert tracks[13].score > IN_PATH_THRESHOLD


def test_uncertainty_does_not_reject_a_vehicle_centred_in_its_own_lane():
    """Body width is already in the corner lateral; do not add sigma on top."""
    tracker = ACCTracker()
    tracks = _run(tracker, [_ahead(14, 30.0, 22.0, width=2.5)], frames=45)
    assert tracks[14].in_path


def test_road_model_is_built_each_frame():
    tracker = ACCTracker()
    _run(tracker, [_ahead(15, 50.0, 22.0), _ahead(16, 90.0, 22.0)], frames=45)
    assert tracker.last_road_model is not None
    assert tracker.last_road_model.support_s_m > 0.0


def test_parked_vehicles_are_dropped_from_tracking():
    tracker = ACCTracker()
    parked = _ahead(9, 30.0, 0.0, history_speed=0.0)
    parked.is_parked = True
    tracks = _run(tracker, [parked], frames=30)
    assert 9 not in tracks
