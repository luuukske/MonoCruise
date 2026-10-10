"""Ego body and path origin from the SDK wheel layout (core/radar/README.md §17)."""
from __future__ import annotations

import math

import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clip_schema import EgoTelemetry
from core.radar import ego_geometry as eg
from core.radar.ego_geometry import (
    EgoGeometry, calibration_geometry, estimate_ego_geometry, geometry_from_sdk,
)
from core.radar.traffic import build_arc

# vehicle.volvo.fh_2024 6x4 as the SDK reported it on 2026-10-04 (float32 values).
REF_X = [-1.04, 1.04, -0.932, 0.932, -0.932, 0.932]
REF_Z = [-1.792468786239624, -1.792468786239624, 1.4181249141693115,
         1.4181249141693115, 2.767826557159424, 2.767826557159424]
REF_STEER = [True, True, False, False, False, False]
REF_REAR_MEAN = -(REF_Z[2] + REF_Z[4]) / 2.0


def _ref() -> EgoGeometry:
    g = estimate_ego_geometry(REF_X, REF_Z, REF_STEER)
    assert g is not None
    return g


def test_reference_truck_reproduces_the_calibrated_body():
    g = _ref()
    assert g.front_m == pytest.approx(CAL.ego_half_length, abs=1e-3)
    assert g.rear_m == pytest.approx(CAL.ego_half_length, abs=1e-3)
    assert g.half_width_m == pytest.approx(CAL.ego_half_width, abs=1e-6)
    assert g.front_delta_m == pytest.approx(0.0, abs=1e-3)
    assert g.from_wheels


def test_reference_constants_track_the_aeb_calibration():
    # The estimator carries the fitted body over to other trucks; a retune of
    # the calibration body must move the reference with it.
    assert eg.REF_HALF_LENGTH_M == CAL.ego_half_length
    assert eg.REF_HALF_WIDTH_M == CAL.ego_half_width


def test_path_origin_is_the_rear_wheel_mean():
    g = _ref()
    assert g.path_origin_m == pytest.approx(REF_REAR_MEAN)
    # The 20 % rule it replaces sat within 0.1 m of it on this truck.
    old = (CAL.arc_start_pctg - 0.5) * 2.0 * CAL.ego_half_length
    assert abs(g.path_origin_m - old) < 0.1


def test_calibration_geometry_is_the_pre_wheel_body():
    g = calibration_geometry(CAL.ego_half_length, CAL.ego_half_width, CAL.arc_start_pctg)
    assert not g.from_wheels
    assert g.front_m == g.rear_m == CAL.ego_half_length
    assert g.path_origin_m == pytest.approx((CAL.arc_start_pctg - 0.5) * 2.0 * CAL.ego_half_length)


def test_longer_wheelbase_moves_the_front_and_the_path_origin():
    z = [v - 0.5 if s else v + 0.4 for v, s in zip(REF_Z, REF_STEER)]
    g = estimate_ego_geometry(REF_X, z, REF_STEER)
    ref = _ref()
    assert g.front_m == pytest.approx(ref.front_m + 0.5)
    assert g.rear_m == pytest.approx(ref.rear_m + 0.4)
    assert g.path_origin_m == pytest.approx(ref.path_origin_m - 0.4)


def test_lifted_axle_leaves_the_path_origin_on_the_grounded_one():
    lift = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    g = estimate_ego_geometry(REF_X, REF_Z, REF_STEER, lift)
    assert g.path_origin_m == pytest.approx(-REF_Z[2])
    # The axle still exists, so the body keeps its length.
    assert g.rear_m == pytest.approx(_ref().rear_m)


def test_missing_steer_flags_fall_back_to_the_front_axle_band():
    g = estimate_ego_geometry(REF_X, REF_Z, [False] * 6)
    assert g.path_origin_m == pytest.approx(REF_REAR_MEAN)


@pytest.mark.parametrize("xs, zs, steer", [
    (REF_X[:3], REF_Z[:3], REF_STEER[:3]),                        # too few wheels
    (REF_X, [0.0] * 6, REF_STEER),                                # no wheelbase
    (REF_X, REF_Z[:5] + [math.nan], REF_STEER),                   # unreadable
    ([0.1] * 6, REF_Z, REF_STEER),                                # no track
    (REF_X, REF_Z, [True] * 6),                                   # nothing to roll on
])
def test_implausible_layouts_are_rejected(xs, zs, steer):
    assert estimate_ego_geometry(xs, zs, steer) is None


def test_sdk_dict_reader():
    raw = {
        "truckWheelCount": 6,
        "truckWheelPositionX": REF_X + [0.0] * 10,
        "truckWheelPositionZ": REF_Z + [0.0] * 10,
        "truckWheelSteerable": REF_STEER + [False] * 10,
        "truck_wheelLift": [0.0] * 16,
    }
    assert geometry_from_sdk(raw) == _ref()
    assert geometry_from_sdk({}) is None
    assert geometry_from_sdk({**raw, "truckWheelCount": 0}) is None


def test_clip_frames_carry_the_geometry_and_old_clips_carry_none():
    ego = EgoTelemetry()
    ego.set_geometry(_ref())
    back = EgoTelemetry.from_json(ego.to_json())
    assert back.geometry() == EgoGeometry(
        _ref().half_width_m, _ref().front_m, _ref().rear_m, _ref().path_origin_m,
    )
    old = EgoTelemetry().to_json()
    for k in ("ego_front_m", "ego_rear_m", "ego_half_width_m", "ego_path_origin_m"):
        old.pop(k)
    assert EgoTelemetry.from_json(old).geometry() is None


def _aeb_tick(geometry: EgoGeometry | None):
    from core.aeb.clip_eval import _make_headless, _snapshot_tuple

    t = _make_headless(CAL)
    ego = EgoTelemetry(coordinateX=100.0, coordinateZ=200.0, rotationX=0.1, speed=20.0)
    snap = _snapshot_tuple(ego, [], 10.0, frozenset(), 10.0)
    t._read_radar_snapshot = lambda: snap
    t._read_ego_geometry = lambda: geometry
    t._read_max_brake_ms2 = lambda: 7.8
    t._read_user_braking = lambda: False
    t._read_addressing_brake = lambda: False
    t._read_vehicle_key = lambda: None
    t._now = lambda: 10.0
    t.loop()
    return ego, t.data.snapshot


def test_aeb_without_wheels_keeps_the_calibration_body():
    ego, snap = _aeb_tick(None)
    yaw = ego.rotationX * 2.0 * math.pi
    off = (CAL.arc_start_pctg - 0.5) * 2.0 * CAL.ego_half_length
    arc = snap.ego_arc
    assert arc.start_x == pytest.approx(ego.coordinateX - off * math.sin(yaw))
    assert arc.start_z == pytest.approx(ego.coordinateZ - off * math.cos(yaw))
    assert arc.fwd_len == pytest.approx(CAL.ego_half_length - off)
    assert arc.back_len == pytest.approx(CAL.ego_half_length + off)
    assert snap.ego_half_w == CAL.ego_half_width


def test_aeb_starts_the_arc_at_the_rear_wheels_and_keeps_the_bumper():
    g = EgoGeometry(half_width_m=1.3, front_m=3.8, rear_m=3.1, path_origin_m=-2.4)
    ego, snap = _aeb_tick(g)
    yaw = ego.rotationX * 2.0 * math.pi
    arc = snap.ego_arc
    assert arc.start_x == pytest.approx(ego.coordinateX + 2.4 * math.sin(yaw))
    assert arc.start_z == pytest.approx(ego.coordinateZ + 2.4 * math.cos(yaw))
    # Front of the capsule lands on the estimated bumper, not on the arc start.
    assert arc.fwd_len == pytest.approx(3.8 + 2.4)
    assert arc.back_len == pytest.approx(3.1 - 2.4)
    assert arc.half_width == 1.3
    assert (snap.ego_front_m, snap.ego_rear_m) == (3.8, 3.1)


def _acc_lat_and_dist(geometry: EgoGeometry | None, lead_xz, lead_yaw, kappa, frames=30):
    from core.acc.tracker import ACCTracker
    from tests.acc.harness import make_vehicle

    tracker = ACCTracker()
    lead = make_vehicle(7, lead_xz[0], lead_xz[1], 15.0, yaw_rad=lead_yaw)
    for i in range(frames):
        tracker.update(
            now_mono=100.0 + i / 30.0, dt=1.0 / 30.0, vehicles=[lead],
            ego_x=0.0, ego_z=0.0, ego_yaw_rad=0.0, ego_speed_ms=15.0,
            ego_steer=kappa / 0.17, ego_history_kappa=kappa,
            blinker_left=False, blinker_right=False, ego_geometry=geometry,
        )
    st = tracker.tracks[7]
    return st.last_lat, st.dist_m


def test_acc_arc_from_the_rear_wheels_follows_the_true_path_in_a_bend():
    kappa = 0.02
    g = _ref()
    # The path ego actually drives: the circle the rear-wheel mean rolls on.
    rear = (0.0, -g.path_origin_m)
    true_path = build_arc(rear[0], rear[1], 0.0, 15.0, kappa, 1.25, 5.0)
    s = 30.0 - g.path_origin_m
    lead_xz = true_path.position_at_dist(s)
    lead_yaw = true_path.heading_at_dist(s)
    lat_new, _ = _acc_lat_and_dist(g, lead_xz, lead_yaw, kappa)
    lat_old, _ = _acc_lat_and_dist(None, lead_xz, lead_yaw, kappa)
    assert abs(lat_new) < 0.05
    # The origin-launched arc lags the bend by about kappa * d * s.
    assert abs(lat_old) > 0.8


def test_acc_distance_is_unchanged_for_the_reference_truck_on_a_straight():
    _, d_new = _acc_lat_and_dist(_ref(), (0.0, -40.0), 0.0, 0.0)
    _, d_old = _acc_lat_and_dist(None, (0.0, -40.0), 0.0, 0.0)
    assert d_new == pytest.approx(d_old, abs=1e-3)


def test_acc_gap_shrinks_with_a_longer_nose():
    g = _ref()
    longer = EgoGeometry(g.half_width_m, g.front_m + 0.6, g.rear_m, g.path_origin_m)
    _, d_ref = _acc_lat_and_dist(g, (0.0, -40.0), 0.0, 0.0)
    _, d_long = _acc_lat_and_dist(longer, (0.0, -40.0), 0.0, 0.0)
    assert d_long == pytest.approx(d_ref - 0.6, abs=1e-6)
