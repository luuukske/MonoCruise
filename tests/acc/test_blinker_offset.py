"""Tracker-side blinker fixtures: candidacy rules and the intent model.

Controller arbitration lives in `test_blinker_arbitration.py`. Corpus clips
carry no blinker state; harness defaults keep b_eff at 0 so the baseline stays
bit-identical. These fixtures drive synthetic scenes only."""
from __future__ import annotations

import math

import pytest

from core.acc.blinker import (
    BLINKER_COMMIT_LAT_M,
    BLINKER_LAMP_GAP_S,
    BLINKER_MIN_HOLD_S,
    BLINKER_PEAK_S,
    BLINKER_RELEASE_S,
    BLINKER_SUSTAIN,
)
from core.acc.scoring import IN_PATH_THRESHOLD
from core.acc.tracker import ACCTracker
from core.cruise_control_thread.blinker_arbitration import (
    BLINKER_RELEASE_HOLD_MIN_M,
)

from .harness import make_vehicle


_DT = 1.0 / 30.0
_EGO_YAW = 0.0
_HW_KMH = 90.0
_HW_MS = _HW_KMH / 3.6


def _tick(
    tracker: ACCTracker,
    vehicles,
    *,
    t: float,
    ego_speed: float = _HW_MS,
    blinker_left: bool = False,
    blinker_right: bool = False,
    ego_x: float = 0.0,
    ego_z: float = 0.0,
):
    return tracker.update(
        now_mono=t, dt=_DT, vehicles=vehicles,
        ego_x=ego_x, ego_z=ego_z,
        ego_yaw_rad=_EGO_YAW,
        ego_speed_ms=ego_speed, ego_steer=0.0,
        ego_history_kappa=0.0,
        blinker_left=blinker_left, blinker_right=blinker_right,
    )


def _ahead(vid, dist_m, speed, lat: float = 0.0, **kw):
    return make_vehicle(vid, lat, -dist_m, speed, yaw_rad=_EGO_YAW, **kw)


def _run_locked_lead(tracker: ACCTracker, lead, *, frames: int = 60, t0: float = 100.0):
    for i in range(frames):
        _tick(tracker, [lead], t=t0 + i * _DT, blinker_right=False)
    assert tracker.tracks[lead.id].score > IN_PATH_THRESHOLD


def test_below_gate_blinker_no_bias():
    """Below the R0 gate, blinker must not publish an indicated lead."""
    tracker = ACCTracker()
    lead = _ahead(1, 40.0, 10.0)
    adj = _ahead(2, 35.0, 10.0, lat=4.5)
    for i in range(45):
        _tick(
            tracker, [lead, adj], t=100.0 + i * _DT,
            ego_speed=30.0 / 3.6, blinker_right=True,
        )
    assert abs(tracker.last_b_eff) < 1e-6
    assert tracker.last_indicated_lead is None


def test_same_speed_merge_target_is_candidate():
    """R3 fix: v_rel ≈ 0 at 30 m must still be adoptable."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    merge = _ahead(2, 30.0, _HW_MS, lat=4.5)
    _run_locked_lead(tracker, lead, frames=45)
    # Sample during the peak of the rising-edge pulse.
    for i in range(12):
        _tick(tracker, [lead, merge], t=102.0 + i * _DT, blinker_right=True)
    assert tracker.last_b_eff > 0.5
    ind = tracker.last_indicated_lead
    assert ind is not None
    assert ind.vehicle.id == 2


def test_oncoming_in_indicated_lane_never_candidate():
    tracker = ACCTracker()
    lead = _ahead(1, 40.0, _HW_MS - 4.0)
    oncoming = make_vehicle(2, 4.5, -40.0, 22.0, yaw_rad=math.pi)
    _run_locked_lead(tracker, lead, frames=45)
    for i in range(45):
        _tick(tracker, [lead, oncoming], t=102.0 + i * _DT, blinker_right=True)
    assert tracker.last_indicated_lead is None
    assert not tracker.tracks.get(2, type("T", (), {"indicated_candidate": False})()).indicated_candidate


def test_overtaker_never_indicated_lead():
    from core.acc.tracker import TrackState

    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    passer = _ahead(2, 25.0, _HW_MS + 5.0, lat=4.5)
    _run_locked_lead(tracker, lead, frames=30)
    # Seed behind-history: passer was behind recently (R3).
    tracker.tracks[2] = TrackState(last_behind_mono=101.5)
    for i in range(45):
        _tick(tracker, [lead, passer], t=102.0 + i * _DT, blinker_right=True)
        if 2 in tracker.tracks:
            tracker.tracks[2].last_behind_mono = 101.5
    ind = tracker.last_indicated_lead
    assert ind is None or ind.vehicle.id != 2


def test_alongside_never_indicated_lead():
    """R2: front-over-front margin rejects a vehicle that is only alongside."""
    tracker = ACCTracker()
    lead = _ahead(1, 40.0, _HW_MS - 3.0)
    # Center ~0.5 m ahead: front-over-front gap stays under the 1.5 m margin.
    beside = _ahead(2, 0.5, _HW_MS, lat=4.5)
    _run_locked_lead(tracker, lead, frames=45)
    for i in range(12):
        _tick(tracker, [lead, beside], t=102.0 + i * _DT, blinker_right=True)
    ind = tracker.last_indicated_lead
    assert ind is None or ind.vehicle.id != 2


def test_alongside_truck_trailer_never_indicated_lead():
    """Long adjacent body: nose clears R2 but parallel overlap must not brake."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    # ~16 m truck/trailer beside ego; front is well ahead, rear still overlaps.
    beside = _ahead(2, 4.0, _HW_MS, lat=4.5, length=16.0)
    _run_locked_lead(tracker, lead, frames=45)
    for i in range(12):
        _tick(tracker, [lead, beside], t=102.0 + i * _DT, blinker_right=True)
    assert tracker.last_b_eff > 0.5
    ind = tracker.last_indicated_lead
    assert ind is None or ind.vehicle.id != 2
    assert not tracker.tracks.get(2, type("T", (), {"indicated_candidate": False})()).indicated_candidate


def test_alongside_tmp_tractor_trailer_never_indicated_lead():
    """TMP: trailer is a separate vehicle; cab alone must not clear the parallel gate."""
    from core.radar.traffic import Position, Quaternion, Size, Trailer

    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    # Cab fully ahead of ego front; trailer still overlaps ego longitudinally.
    tractor = _ahead(2, 8.0, _HW_MS, lat=4.5, length=8.0)
    tractor.is_tmp = True
    trailer = _ahead(3, -1.0, _HW_MS, lat=4.5, length=12.0)
    trailer.is_tmp = True
    trailer.is_trailer = True
    # Cab-only rear is ~4 m (> ego front); without trailer link this wrongly passes.
    assert 8.0 - 4.0 > 2.5
    _run_locked_lead(tracker, lead, frames=45)
    for i in range(12):
        _tick(
            tracker, [lead, tractor, trailer],
            t=102.0 + i * _DT, blinker_right=True,
        )
    assert tracker.last_b_eff > 0.5
    ind = tracker.last_indicated_lead
    assert ind is None or ind.vehicle.id not in (2, 3)
    assert not tracker.tracks.get(2, type("T", (), {"indicated_candidate": False})()).indicated_candidate
    assert not tracker.tracks.get(3, type("T", (), {"indicated_candidate": False})()).indicated_candidate

    # Nested AI trailer on a short cab: same geometry, same rejection.
    tracker2 = ACCTracker()
    lead2 = _ahead(1, 50.0, _HW_MS - 5.0)
    cab = _ahead(2, 8.0, _HW_MS, lat=4.5, length=8.0)
    cab.trailers = [
        Trailer(
            Position(4.5, 0.0, 1.0),
            Quaternion(1.0, 0.0, 0.0, 0.0),
            Size(2.4, 3.0, 12.0),
            is_tmp=False,
            slot=0,
        )
    ]
    _run_locked_lead(tracker2, lead2, frames=45)
    for i in range(12):
        _tick(tracker2, [lead2, cab], t=102.0 + i * _DT, blinker_right=True)
    assert not tracker2.tracks.get(2, type("T", (), {"indicated_candidate": False})()).indicated_candidate


def test_r4_headway_uses_train_rear_not_nose():
    """R4 2 s window is to the rear bumper, including a nested trailer."""
    from core.acc.blinker import BlinkerBias, BLINKER_HEADWAY_MARGIN_S
    from core.radar.traffic import Position, Quaternion, Size, Trailer

    ego_ms = _HW_MS
    desired = 1.5
    limit = desired + BLINKER_HEADWAY_MARGIN_S
    # Nose outside the window; train rear inside (cab length 8 m).
    train_rear_m = (limit - 0.4) * ego_ms
    nose_m = train_rear_m + 16.0
    assert nose_m / ego_ms >= limit
    assert train_rear_m / ego_ms < limit
    ok, th = BlinkerBias.is_candidate(
        last_behind_mono=float("-inf"),
        now_mono=100.0,
        b_eff=1.0,
        dist_m=nose_m,
        body_rear_m=train_rear_m,
        road_lat=4.5,
        yaw_diff_deg=0.0,
        ego_speed_ms=ego_ms,
        v_speed_ms=ego_ms,
        desired_th_s=desired,
    )
    assert ok
    assert th == pytest.approx(train_rear_m / ego_ms)
    # Closing does not bypass the window: far slower traffic stays out.
    far_rear_m = limit * ego_ms + 5.0
    ok_far, _ = BlinkerBias.is_candidate(
        last_behind_mono=float("-inf"),
        now_mono=100.0,
        b_eff=1.0,
        dist_m=far_rear_m + 10.0,
        body_rear_m=far_rear_m,
        road_lat=4.5,
        yaw_diff_deg=0.0,
        ego_speed_ms=ego_ms,
        v_speed_ms=ego_ms - 5.0,
        desired_th_s=desired,
    )
    assert not ok_far

    from core.acc.blinker import desired_time_headway_s

    tracker = ACCTracker()
    live_limit = desired_time_headway_s() + BLINKER_HEADWAY_MARGIN_S
    live_rear_m = (live_limit - 0.4) * ego_ms
    lead = _ahead(1, live_rear_m + 40.0, _HW_MS - 5.0)
    # Cab + nested trailer: nose outside live window, train rear inside.
    cab_half = 4.0
    trailer_len = 12.0
    cab_center = live_rear_m + trailer_len + cab_half
    trailer_center = live_rear_m + trailer_len * 0.5
    truck = _ahead(2, cab_center, _HW_MS, lat=4.5, length=8.0)
    truck.trailers = [
        Trailer(
            Position(4.5, 0.0, -trailer_center),
            Quaternion(1.0, 0.0, 0.0, 0.0),
            Size(2.4, 3.0, trailer_len),
            is_tmp=False,
            slot=0,
        )
    ]
    _run_locked_lead(tracker, lead, frames=45)
    saw = False
    for i in range(20):
        _tick(tracker, [lead, truck], t=102.0 + i * _DT, blinker_right=True)
        st = tracker.tracks.get(2)
        if st is not None and st.indicated_candidate:
            saw = True
            assert st.last_time_headway_s < live_limit
            # Must be measuring the trailer rear, not the cab nose.
            assert st.last_time_headway_s < (cab_center + cab_half) / _HW_MS
    assert saw


def test_r4_window_closes_as_ego_draws_level():
    """R4 lower bound: the window has a bottom, not just a 3.5 s top."""
    from core.acc.blinker import (
        BLINKER_PASS_CLEAR_S,
        EGO_FRONT_OFFSET_M,
        BlinkerBias,
    )

    def cand(dist_m, body_rear_m, v_speed_ms):
        return BlinkerBias.is_candidate(
            last_behind_mono=float("-inf"), now_mono=100.0, b_eff=1.0,
            dist_m=dist_m, body_rear_m=body_rear_m, road_lat=4.5,
            yaw_diff_deg=0.0, ego_speed_ms=_HW_MS, v_speed_ms=v_speed_ms,
            desired_th_s=1.5,
        )[0]

    v_close = 5.0
    slower = _HW_MS - v_close
    inside = EGO_FRONT_OFFSET_M + (BLINKER_PASS_CLEAR_S + 0.2) * v_close
    outside = EGO_FRONT_OFFSET_M + (BLINKER_PASS_CLEAR_S - 0.2) * v_close
    assert cand(inside, inside, slower)
    assert not cand(outside, outside, slower)
    # Not closing: ego never draws level, so the window stays open.
    assert cand(outside, outside, _HW_MS)
    # A rear bumper beside ego is alongside whatever the closing speed is.
    assert not cand(20.0, EGO_FRONT_OFFSET_M - 0.5, _HW_MS)
    assert not cand(20.0, EGO_FRONT_OFFSET_M - 0.5, slower)


def test_vehicle_being_passed_never_becomes_the_indicated_lead():
    """Indicating right mid-overtake must not adopt the vehicle being passed.

    Reported from the seat as the ACC slamming on the brakes: the candidate
    landed 2 m off ego's bumper, where the gap law demands everything it has."""
    tracker = ACCTracker()
    passed_ms = 70.0 / 3.6
    lead = _ahead(1, 120.0, _HW_MS)
    # Rear 2 m ahead of ego's front bumper, ego 20 km/h faster.
    beside = _ahead(2, 7.0, passed_ms, lat=4.5)
    _run_locked_lead(tracker, lead, frames=45)
    for i in range(30):
        _tick(tracker, [lead, beside], t=102.0 + i * _DT, blinker_right=True)
    assert tracker.last_b_eff > 0.5
    st = tracker.tracks.get(2)
    assert st is None or not st.indicated_candidate
    ind = tracker.last_indicated_lead
    assert ind is None or ind.vehicle.id != 2


def test_three_flash_bias_survives_decay():
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    merge = _ahead(2, 30.0, _HW_MS - 1.0, lat=4.5)
    _run_locked_lead(tracker, lead, frames=45)
    t = 102.0
    for i in range(9):
        _tick(tracker, [lead, merge], t=t, blinker_right=True)
        t += _DT
    # Short flash then release: the intent covers the merge across the gap.
    for i in range(20):
        _tick(tracker, [lead, merge], t=t, blinker_right=False)
        t += _DT
    assert abs(tracker.last_b_eff) > 0.2 or tracker.last_indicated_lead is not None


def test_stalk_hold_does_not_pin_offset_at_max():
    """Drivers blink and turn immediately; a long hold must not keep b=1.

    The hold also has to sustain past BLINKER_MIN_HOLD_S: the coast that ends a
    tap belongs to a dark lamp only, never to a driver still asking."""
    from core.acc.blinker import BLINKER_DECAY_S

    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    merge = _ahead(2, 30.0, _HW_MS - 1.0, lat=4.5)
    _run_locked_lead(tracker, lead, frames=45)
    t = 102.0
    settled = BLINKER_PEAK_S + BLINKER_DECAY_S + 0.5
    seen = []
    for i in range(int((BLINKER_MIN_HOLD_S + 2.0) / _DT)):
        _tick(tracker, [lead, merge], t=t, blinker_right=True)
        if (t - 102.0) >= settled:
            seen.append(tracker.last_b_eff)
        t += _DT
    # Still indicating, so the envelope sits at its sustain level and stays.
    assert seen[0] == pytest.approx(BLINKER_SUSTAIN, abs=1e-3)
    assert seen[-1] == pytest.approx(BLINKER_SUSTAIN, abs=1e-3)


def test_lamp_toggle_never_drops_intent():
    """Telemetry gives the lamp, not the stalk. Blinking must not gap b_eff."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    _run_locked_lead(tracker, lead, frames=45)
    t = 102.0
    seen = []
    # 4 s of lamp at ~1.5 Hz: 10 frames on, 10 frames off, repeatedly.
    for i in range(120):
        lamp_on = (i // 10) % 2 == 0
        _tick(tracker, [lead], t=t, blinker_right=lamp_on)
        seen.append(tracker.last_b_eff)
        t += _DT
    assert min(seen) > 0.0
    assert seen[-1] == pytest.approx(BLINKER_SUSTAIN, abs=1e-3)


def test_lamp_cancel_ends_intent():
    """A dark gap longer than the debounce is the driver cancelling.

    The dark window has to outlast the minimum envelope too, or the floor is
    still serving the tap that started this intent."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    _run_locked_lead(tracker, lead, frames=45)
    t = 102.0
    for i in range(30):
        _tick(tracker, [lead], t=t, blinker_right=True)
        t += _DT
    assert tracker.last_b_eff > 0.0
    dark_s = BLINKER_MIN_HOLD_S + BLINKER_RELEASE_S + 0.3
    for i in range(int(dark_s / _DT)):
        _tick(tracker, [lead], t=t, blinker_right=False)
        t += _DT
    assert abs(tracker.last_b_eff) < 1e-6


def _tap_and_change(tracker, lat_ms: float = 0.7, frames: int = 150):
    """One tap, then move over at lat_ms. Returns the (age, b_eff, offset) trace."""
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    _run_locked_lead(tracker, lead, frames=45)
    t0 = 102.0
    t = t0
    ego_x = 0.0
    trace = []
    for _ in range(frames):
        age = t - t0
        # Tap, react for 0.6 s, then move over.
        if age > 0.6:
            ego_x += lat_ms * _DT
        moved = _ahead(1, 50.0 + ego_x, _HW_MS - 5.0)
        _tick(tracker, [moved], t=t, blinker_right=age < 0.3, ego_x=ego_x)
        trace.append((age, tracker.last_b_eff, tracker.last_lane_offset_m))
        t += _DT
    return trace


def test_one_tap_carries_a_whole_lane_change():
    """A single tap must carry the intent through the whole manoeuvre.

    Without the floor the intent died a second after the lamp went dark, which
    wiped the commit latch and the offset stage 2 needs to let the old lane go."""
    trace = _tap_and_change(ACCTracker())
    live = [row for row in trace if abs(row[1]) > 1e-6]
    assert live[-1][0] > 3.0
    assert max(row[2] for row in live) >= BLINKER_RELEASE_HOLD_MIN_M


def test_one_tap_coasts_to_zero_instead_of_holding_the_sustain():
    """The minimum envelope decays the whole way down, it does not sit and drop.

    A flat sustain to the end of the floor means a step off it, and the point of
    the floor is only that the offset does not return to zero on a dark lamp."""
    trace = _tap_and_change(ACCTracker())
    peak_end = [row for row in trace if row[0] >= BLINKER_PEAK_S]
    prev_b = None
    for _, b, _off in peak_end:
        if prev_b is not None:
            assert b <= prev_b + 1e-9, "envelope must never rise after the peak"
            assert prev_b - b < 0.1, "envelope must never step down"
        prev_b = b
    # No plateau: strictly falling across the dark phase, not held at sustain.
    def at(age: float) -> float:
        return min(trace, key=lambda row: abs(row[0] - age))[1]

    assert at(1.5) - at(2.5) > 0.05
    assert at(2.5) - at(3.5) > 0.05
    assert at(BLINKER_MIN_HOLD_S + 0.2) == pytest.approx(0.0, abs=1e-6)


def test_min_hold_does_not_extend_a_cancelled_long_hold():
    """The floor is a minimum, not an extension: a real cancel still releases."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    _run_locked_lead(tracker, lead, frames=45)
    t = 102.0
    for i in range(int((BLINKER_MIN_HOLD_S + 1.0) / _DT)):
        _tick(tracker, [lead], t=t, blinker_right=True)
        t += _DT
    assert tracker.last_b_eff > 0.0
    # Nothing left of the floor to serve: only the debounce and the fade.
    dark_s = BLINKER_LAMP_GAP_S + BLINKER_RELEASE_S + 0.2
    for i in range(int(dark_s / _DT)):
        _tick(tracker, [lead], t=t, blinker_right=False)
        t += _DT
    assert abs(tracker.last_b_eff) < 1e-6


def test_min_hold_yields_to_the_r0_gate():
    """Slowing into a junction ends the intent whatever the floor says.

    Holding it would resurrect a stale intent on the way back up, with no
    rising edge left to re-seat the commit reference or the floor-lift."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    _run_locked_lead(tracker, lead, frames=45)
    t = 102.0
    for i in range(9):
        _tick(tracker, [lead], t=t, blinker_right=True)
        t += _DT
    assert tracker.last_b_eff > 0.5
    # Lamp still lit, but ego drops well under the gate.
    for i in range(15):
        _tick(tracker, [lead], t=t, ego_speed=20.0 / 3.6, blinker_right=True)
        t += _DT
    assert abs(tracker.last_b_eff) < 1e-6
    assert tracker._blinker.intent_sign == 0.0


def test_commit_ignores_common_mode_lateral_motion():
    """A curve moves ego and its lead together; only ego moving is a lane change."""
    def run(*, move_lead: bool) -> float:
        tracker = ACCTracker()
        lead = _ahead(1, 50.0, _HW_MS - 5.0)
        _run_locked_lead(tracker, lead, frames=45)
        t = 102.0
        ego_x = 0.0
        for i in range(60):
            ego_x += 0.02          # 0.6 m/s to ego's right
            lat = ego_x if move_lead else 0.0
            moved = _ahead(1, 50.0 + ego_x, _HW_MS - 5.0, lat=lat)
            _tick(tracker, [moved], t=t, blinker_right=True, ego_x=ego_x)
            t += _DT
        return tracker.last_lane_offset_m

    ego_only = run(move_lead=False)
    common_mode = run(move_lead=True)
    assert ego_only >= BLINKER_COMMIT_LAT_M
    assert common_mode < BLINKER_COMMIT_LAT_M
    assert common_mode < ego_only * 0.5


def test_commit_latches_and_survives_a_bad_frame():
    """The latch holds for the intent: a lane change does not un-happen."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 5.0)
    _run_locked_lead(tracker, lead, frames=45)
    t = 102.0
    ego_x = 0.0
    for i in range(40):
        ego_x += 0.03
        _tick(tracker, [lead], t=t, blinker_right=True, ego_x=ego_x)
        t += _DT
    assert tracker.last_blinker_committed
    # Snap ego back: the latch must not drop out under one bad measurement.
    _tick(tracker, [lead], t=t, blinker_right=True, ego_x=0.0)
    assert tracker.last_blinker_committed


def test_r0_boundary_no_flicker():
    """Ego oscillating 48-52 km/h must not toggle the R0 gate on/off."""
    tracker = ACCTracker()
    lead = _ahead(1, 40.0, 12.0)
    adj = _ahead(2, 30.0, 12.0, lat=4.5)
    gate = []
    t = 100.0
    for i in range(90):
        kmh = 48.0 + (4.0 if (i // 5) % 2 == 0 else 0.0)
        _tick(
            tracker, [lead, adj], t=t,
            ego_speed=kmh / 3.6, blinker_right=True,
        )
        gate.append(1 if tracker._blinker.gate_on else 0)
        t += _DT
    if any(gate):
        first_on = gate.index(1)
        assert all(g == 1 for g in gate[first_on:])


def test_candidate_only_in_indicated_lead_slot():
    """R15: candidate must not appear in leads[]."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 6.0)
    merge = _ahead(2, 30.0, _HW_MS - 1.0, lat=4.5)
    _run_locked_lead(tracker, lead, frames=45)
    saw_indicated = False
    for i in range(20):
        leads = _tick(tracker, [lead, merge], t=102.0 + i * _DT, blinker_right=True)
        ind = tracker.last_indicated_lead
        if ind is not None:
            saw_indicated = True
            assert ind.vehicle.id == 2
            assert all(L.vehicle.id != 2 for L in leads)
    assert saw_indicated


def test_next_lane_over_never_adopted():
    """Empty-lane pass: traffic two lanes over must not become the candidate."""
    tracker = ACCTracker()
    lead = _ahead(1, 50.0, _HW_MS - 6.0)
    far = _ahead(3, 30.0, _HW_MS, lat=9.0)
    _run_locked_lead(tracker, lead, frames=45)
    for i in range(60):
        _tick(tracker, [lead, far], t=102.0 + i * _DT, blinker_right=True)
    ind = tracker.last_indicated_lead
    assert ind is None or ind.vehicle.id != 3


def test_handover_collapses_b_eff_and_carries_score():
    """R11: when the candidate enters the unshifted corridor, b_eff collapses."""
    tracker = ACCTracker()
    lead = _ahead(1, 60.0, _HW_MS - 8.0)
    # Start in the indicated lane, then walk into ego's corridor.
    merge = _ahead(2, 35.0, _HW_MS - 1.0, lat=4.5)
    _run_locked_lead(tracker, lead, frames=45)
    t = 102.0
    for i in range(30):
        _tick(tracker, [lead, merge], t=t, blinker_right=True)
        t += _DT
    assert tracker.last_indicated_lead is not None
    # Slide the candidate into the ego lane.
    for i in range(40):
        lat = 4.5 * max(0.0, 1.0 - i / 25.0)
        merge = _ahead(2, 35.0, _HW_MS - 1.0, lat=lat)
        leads = _tick(tracker, [lead, merge], t=t, blinker_right=True)
        t += _DT
    assert abs(tracker.last_b_eff) < 1e-6
    # Score carried: id 2 should be publishable once in-corridor.
    assert 2 in tracker.tracks and tracker.tracks[2].score > 0.0


def test_locked_lead_present_every_frame_on_blink(monkeypatch):
    """R10: floor-lift must not drop a locked published lead for a blink."""
    tracker = ACCTracker()
    lead = _ahead(1, 40.0, _HW_MS - 4.0)
    _run_locked_lead(tracker, lead, frames=60)
    lead_ids = []
    for i in range(45):
        leads = _tick(tracker, [lead], t=102.0 + i * _DT, blinker_right=True)
        lead_ids.append(leads[0].vehicle.id if leads else None)
    assert all(vid == 1 for vid in lead_ids)
