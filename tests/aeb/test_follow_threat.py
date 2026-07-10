"""Follow-threat flag: behavioral detection of a genuinely braking in-lane lead.

The flag requires BOTH sustained closing range and sustained target
deceleration while co-directional in Lane.EGO (see AEBCalibration
follow_threat_*). These tests drive _update_follow_threats directly over
synthetic multi-frame histories; the single-frame scenario harness cannot
exercise a temporal tracker.
"""

from __future__ import annotations

import math

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.thread import AEBThread
from core.radar.traffic import build_arc
from tests.aeb.harness import make_vehicle

EGO_SPEED = 20.0          # m/s, heading -Z (yaw 0)
DT = 1.0 / 30.0


def _tracker() -> AEBThread:
    t = AEBThread()
    t._cal = CAL
    return t


def _ego_arc(speed: float = EGO_SPEED):
    return build_arc(0.0, 0.0, 0.0, speed, 0.0, CAL.ego_half_width, 3.0)


def _drive(thread, frames):
    """frames: list of (now, vehicles). Returns final flag set."""
    arc = _ego_arc()
    for now, vehicles in frames:
        thread._update_follow_threats(
            vehicles, arc, 0.0, 0.0, 0.0, -1.0, now, CAL,
        )
    return thread._follow_threat_ids


def _lead_frames(vid=7, ticks=30, start_dist=40.0, closing=6.0,
                 start_speed=14.0, decel=3.0, lateral=0.0):
    """Simulate a lead ahead of ego (-Z), closing and decelerating."""
    frames = []
    for i in range(ticks):
        now = 100.0 + i * DT
        dist = start_dist - closing * i * DT
        speed = max(0.0, start_speed - decel * i * DT)
        v = make_vehicle(vid, lateral, -dist, 0.0, speed)
        frames.append((now, [v]))
    return frames


def test_braking_closing_lead_flagged():
    flags = _drive(_tracker(), _lead_frames())
    assert 7 in flags


def test_steady_speed_lead_not_flagged():
    flags = _drive(_tracker(), _lead_frames(decel=0.0))
    assert 7 not in flags


def test_non_closing_braking_lead_not_flagged():
    flags = _drive(_tracker(), _lead_frames(closing=-2.0))
    assert 7 not in flags


def test_out_of_lane_lead_not_flagged():
    flags = _drive(_tracker(), _lead_frames(lateral=5.0))
    assert 7 not in flags


def test_oncoming_not_flagged():
    frames = []
    for i in range(30):
        now = 100.0 + i * DT
        dist = 40.0 - 8.0 * i * DT
        speed = max(0.0, 14.0 - 3.0 * i * DT)
        v = make_vehicle(7, 0.0, -dist, 180.0, speed)
        frames.append((now, [v]))
    flags = _drive(_tracker(), frames)
    assert 7 not in flags


def test_flag_holds_after_brake_release():
    t = _tracker()
    frames = _lead_frames(ticks=30)
    _drive(t, frames)
    assert 7 in t._follow_threat_ids
    last_now = frames[-1][0]
    # Lead releases the brake and holds speed; flag must survive the hold
    # window (the 0fe85c88 lead let off ~1 s before impact).
    for i in range(1, 20):
        now = last_now + i * DT
        v = make_vehicle(7, 0.0, -30.0, 0.0, 8.0)
        t._update_follow_threats([v], _ego_arc(), 0.0, 0.0, 0.0, -1.0, now, CAL)
    assert 7 in t._follow_threat_ids
    # And expire after follow_threat_hold_s with no re-qualification.
    for i in range(int(CAL.follow_threat_hold_s / DT) + 5):
        now = last_now + (20 + i) * DT
        v = make_vehicle(7, 0.0, -30.0, 0.0, 8.0)
        t._update_follow_threats([v], _ego_arc(), 0.0, 0.0, 0.0, -1.0, now, CAL)
    assert 7 not in t._follow_threat_ids


def test_jitter_does_not_flag():
    """Range/speed jitter with no net trend must not earn the flag."""
    frames = []
    for i in range(30):
        now = 100.0 + i * DT
        wob = 1.5 * math.sin(i * 1.7)
        v = make_vehicle(7, 0.0, -(35.0 + wob), 0.0, 10.0 + 0.8 * math.sin(i * 2.3))
        frames.append((now, [v]))
    flags = _drive(_tracker(), frames)
    assert 7 not in flags
