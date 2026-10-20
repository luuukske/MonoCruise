"""A stopped ego lets go of an engagement the geometry still holds. See core/aeb/README.md, standstill release.

Log of 2026-10-20: AEB stayed engaged for 26 s at 0 km/h among TruckersMP trucks that
overlapped ego. A latched body deep inside ego is now a ghost and loses its latch
(test_overlap_ghosts.py); one touching ego, short of the ghost bars, still holds it.
"""
from __future__ import annotations

import dataclasses

from core.aeb.calibration import DEFAULT
from core.aeb.clip_eval import _make_headless, _snapshot_tuple
from core.aeb.clip_schema import EgoTelemetry
from core.radar.tmp_state import ego_box
from tests.aeb.harness import make_vehicle

_HZ = 30.0
_ENGAGE_FRAME = 12      # the risk confirm needs a few frames of hits before the hold counts
_VEHICLE_ID = 7
_LENGTH = 13.8
# The truck's rear sits this far into ego's front bumper: contact, short of every ghost bar.
_INSIDE_M = 0.5 * DEFAULT.overlap_ghost_contact_m
_TRUCK_Z = -ego_box(0.0, 0.0, 0.0, None).half_length - 0.5 * _LENGTH + _INSIDE_M


def _run(speeds: list[float], cal=DEFAULT):
    """(engaged, latched ids, ghost ids) per frame; the thread is engaged on a latched truck touching ego."""
    t = _make_headless(cal)
    t._latched_threat_ids = {_VEHICLE_ID}
    out = []
    for k, speed in enumerate(speeds):
        now = 10.0 + k / _HZ
        ego = EgoTelemetry(coordinateX=0.0, coordinateZ=0.0, rotationX=0.0, speed=speed)
        truck = make_vehicle(_VEHICLE_ID, 0.0, _TRUCK_Z, 0.0, 1.0, is_tmp=True, length=_LENGTH, width=2.5)
        snap = _snapshot_tuple(ego, [truck], now, frozenset(), now, frozenset())
        t._read_radar_snapshot = lambda s=snap: s
        t._read_ego_geometry = lambda: None
        t._read_max_brake_ms2 = lambda: 7.8
        t._read_user_braking = lambda: False
        t._read_addressing_brake = lambda: False
        t._read_vehicle_key = lambda: None
        t._now = lambda now=now: now
        if k == _ENGAGE_FRAME:
            t._engaged = True
        t.loop()
        out.append((t._engaged, set(t._latched_threat_ids), set(t.data.snapshot.ghost_ids)))
    return out


def _frames(seconds: float) -> int:
    return int(round(seconds * _HZ))


def test_the_hold_would_last_forever_without_the_release():
    off = dataclasses.replace(DEFAULT, aeb_standstill_release_s=1e9)
    states = _run([0.0] * _frames(4.0), off)
    assert all(engaged for engaged, _, _ in states[_ENGAGE_FRAME + 2:])


def test_a_stopped_ego_lets_go_of_a_truck_at_its_bumper():
    states = _run([0.0] * _frames(4.0))
    release_at = _ENGAGE_FRAME + _frames(DEFAULT.aeb_standstill_release_s)
    assert states[release_at - 3][0], "the dwell must hold the engagement first"
    engaged, latched, ghosts = states[release_at + 3]
    assert not engaged
    assert not latched
    assert not any(engaged for engaged, _, _ in states[release_at + 3:])
    assert not ghosts, "contact short of the ghost bars is released, not ghosted"


def test_a_moving_ego_keeps_its_engagement():
    states = _run([2.0] * _frames(4.0))
    assert all(engaged for engaged, _, _ in states[_ENGAGE_FRAME + 2:])


def test_a_stop_shorter_than_the_dwell_does_not_release():
    dwell = DEFAULT.aeb_standstill_release_s
    speeds = ([2.0] * _ENGAGE_FRAME + [0.0] * _frames(dwell * 0.6)
              + [2.0] * _frames(1.0) + [0.0] * _frames(dwell * 0.6) + [2.0] * _frames(1.0))
    states = _run(speeds)
    assert all(engaged for engaged, _, _ in states[_ENGAGE_FRAME + 2:])
