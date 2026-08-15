"""Shared road-surface elevation gate: datum, profile band, grade test, failsafes.

Scenario numbers come from the clip-corpus fit recorded in core/radar/README.md §15."""
from __future__ import annotations

import math

from core.radar import elevation as EL
from core.radar.elevation import (
    EgoElevationTrack,
    ElevationGate,
    RoadSurface,
    build_surface,
    evaluate_vehicle,
    max_curvature,
    profile_band,
    required_curvature,
    road_y_offset,
    target_grade,
)
from core.radar.traffic import Position, Quaternion, Size, Vehicle


def _quat(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> Quaternion:
    """Quaternion whose euler() returns the requested (pitch, yaw, roll)."""
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    q._euler_cache = (pitch_deg, yaw_deg, roll_deg)
    return q


def _veh(
    vid: int,
    s: float,
    y: float,
    *,
    height: float = 1.5,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
    speed: float = 20.0,
) -> Vehicle:
    """Vehicle ``s`` metres ahead of an ego at the origin facing yaw 0 (-Z)."""
    v = Vehicle(
        position=Position(0.0, y, -s),
        rotation=_quat(pitch_deg, yaw_deg, roll_deg),
        size=Size(2.5, height, 5.0),
        speed=speed,
        acceleration=0.0,
        trailer_count=0,
        trailers=[],
        id=vid,
        is_tmp=False,
        is_trailer=False,
    )
    v._smooth_yaw = math.radians(yaw_deg)
    return v


def _surface(grade: float = 0.0, curvature: float | None = None) -> RoadSurface:
    return RoadSurface(
        ego_y=0.0, grade=grade,
        curvature=0.0 if curvature is None else curvature,
        curvature_ok=curvature is not None,
    )


# Ego at the origin facing -Z: forward range of a vehicle at z = -s.
_EGO_YAW = 0.0


def _s_of(v: Vehicle) -> float:
    return -v.position.z


def _old_gate(v: Vehicle, ego_pitch_rad: float) -> bool:
    """The fixed +-5 m pitch-tangent window this module replaced."""
    s = _s_of(v)
    return abs(v.position.y - s * math.tan(ego_pitch_rad)) > 5.0


class TestDatum:
    def test_body_height_bias_is_removed(self):
        """A car and a truck on ego's own road both read ~0 offset."""
        car = _veh(1, 40.0, EL.BODY_DATUM_FRAC * 1.5, height=1.5)
        truck = _veh(2, 40.0, EL.BODY_DATUM_FRAC * 3.9, height=3.9)
        for v in (car, truck):
            dy, h = road_y_offset(v, 0.0)
            assert h > 0.0
            assert abs(dy) < 0.01

    def test_missing_height_falls_open_with_slack(self):
        v = _veh(1, 40.0, 1.0, height=0.0)
        dy, h = road_y_offset(v, 0.0)
        assert h == 0.0
        assert dy == 1.0
        assert profile_band(40.0, 0.0) > profile_band(40.0, 1.5)

    def test_truck_datum_alone_would_not_trip_the_old_gate(self):
        """Sanity: the bias is real but under 5 m, so it only ate budget."""
        truck = _veh(1, 40.0, EL.BODY_DATUM_FRAC * 3.9, height=3.9)
        assert not _old_gate(truck, 0.0)
        assert 1.5 < truck.position.y < 2.5


class TestProfileBand:
    def test_band_grows_with_range(self):
        assert profile_band(20.0, 1.5) < profile_band(60.0, 1.5)
        assert profile_band(60.0, 1.5) < profile_band(120.0, 1.5)

    def test_band_is_tighter_than_the_old_window_up_close(self):
        assert profile_band(30.0, 1.5) < 5.0

    def test_band_is_wider_than_the_old_window_far_out(self):
        assert profile_band(110.0, 1.5) > 5.0

    def test_band_is_capped(self):
        assert profile_band(400.0, 0.0) == EL._BAND_MAX_M + EL._UNKNOWN_DATUM_SLACK_M

    def test_band_is_symmetric_behind_ego(self):
        assert profile_band(-45.0, 1.5) == profile_band(45.0, 1.5)


class TestCrestAndHill:
    def test_lead_over_a_crest_is_kept(self):
        """Ego pitched up 4%; the lead 120 m out is 4 m below the tangent line.

        The old fixed window lost this: it is the ACC high-speed hill dropout."""
        grade = 0.04
        s = 120.0
        dy = -4.0
        v = _veh(1, s, dy + EL.BODY_DATUM_FRAC * 1.5, pitch_deg=-math.degrees(math.atan(-0.03)))
        assert _old_gate(v, math.atan(grade))
        assert not evaluate_vehicle(v, _surface(grade), s, 0.0).off_surface

    def test_lead_down_a_long_descent_is_kept(self):
        grade = -0.06
        s = 100.0
        v = _veh(1, s, -6.5 + EL.BODY_DATUM_FRAC * 1.5,
                 pitch_deg=math.degrees(math.atan(0.06)))
        assert not evaluate_vehicle(v, _surface(grade), s, 0.0).off_surface

    def test_history_curvature_moves_the_prediction(self):
        """A sag ego has just entered lifts the predicted road ahead."""
        flat = _surface(-0.06)
        sag = _surface(-0.06, curvature=0.0008)
        assert sag.predict(60.0) > flat.predict(60.0)

    def test_curvature_term_is_clamped(self):
        wild = _surface(0.0, curvature=EL._MAX_VERT_CURVATURE)
        assert abs(wild.predict(200.0)) <= EL._CURV_TERM_CLAMP_M + 1e-9


class TestUnderBridge:
    def test_vehicle_under_a_bridge_deck_is_suppressed(self):
        """Ego level on the deck, traffic 5 m below at 30 m."""
        v = _veh(1, 30.0, -5.0 + EL.BODY_DATUM_FRAC * 1.5)
        verdict = evaluate_vehicle(v, _surface(0.0), 30.0, 0.0)
        assert verdict.off_surface
        assert verdict.gross

    def test_hill_into_a_bridge_is_caught_by_the_target_grade(self):
        """Ego descending 8% points straight at traffic under the bridge.

        The height residual is small (that is the reported false positive); the
        target sitting on a flat road is what gives it away."""
        grade = math.tan(math.radians(-8.0))
        s = 40.0
        dy = -5.0
        v = _veh(1, s, dy + EL.BODY_DATUM_FRAC * 1.5, pitch_deg=0.0)
        surface = _surface(grade)
        assert abs(dy - surface.predict(s)) < profile_band(s, 1.5)
        assert _old_gate(v, math.atan(grade)) is False
        assert evaluate_vehicle(v, surface, s, 0.0).off_surface

    def test_same_geometry_on_a_continuous_road_is_kept(self):
        """Identical height drop, but the target lies on ego's own grade."""
        grade = math.tan(math.radians(-8.0))
        s = 40.0
        v = _veh(1, s, -5.0 + EL.BODY_DATUM_FRAC * 1.5,
                 pitch_deg=-math.degrees(math.atan(grade)))
        assert not evaluate_vehicle(v, _surface(grade), s, 0.0).off_surface


class TestGradeSignal:
    def test_codirectional_pitch_is_negated(self):
        """Traffic euler pitch runs opposite to ego pitch (corpus fit)."""
        g = target_grade(_veh(1, 40.0, 0.0, pitch_deg=-5.0), 0.0, 0.0)
        assert g > 0.0
        assert math.isclose(g, math.tan(math.radians(5.0)), rel_tol=1e-6)

    def test_oncoming_pitch_flips_back(self):
        v = _veh(1, 40.0, 0.0, pitch_deg=5.0, yaw_deg=180.0)
        g = target_grade(v, 0.0, math.pi)
        assert math.isclose(g, math.tan(math.radians(5.0)), rel_tol=1e-6)

    def test_perpendicular_target_yields_no_grade_evidence(self):
        v = _veh(1, 40.0, 0.0, pitch_deg=-6.0, yaw_deg=90.0)
        assert math.isclose(target_grade(v, 0.123, math.pi / 2.0), 0.123, rel_tol=1e-9)

    def test_rolled_wreck_fails_open(self):
        v = _veh(1, 40.0, 0.0, pitch_deg=0.0, roll_deg=40.0)
        assert target_grade(v, 0.077, 0.0) == 0.077

    def test_extreme_pitch_fails_open(self):
        v = _veh(1, 40.0, 0.0, pitch_deg=55.0)
        assert target_grade(v, 0.077, 0.0) == 0.077

    def test_zero_quaternion_fails_open(self):
        v = _veh(1, 40.0, 0.0)
        v.rotation = Quaternion(0.0, 0.0, 0.0, 0.0)
        assert target_grade(v, 0.077, 0.0) == 0.077

    def test_wreck_keeps_its_pipeline_seat(self):
        """The bridge geometry with an unusable rotation must not suppress."""
        grade = math.tan(math.radians(-8.0))
        v = _veh(1, 40.0, -5.0 + EL.BODY_DATUM_FRAC * 1.5, roll_deg=40.0)
        assert not evaluate_vehicle(v, _surface(grade), 40.0, 0.0).off_surface


class TestCurvatureBudget:
    def test_required_curvature_is_zero_on_a_consistent_profile(self):
        assert required_curvature(50.0, 0.05 * 50.0, 0.05, 0.05) == 0.0

    def test_budget_shrinks_with_range(self):
        assert max_curvature(20.0) > max_curvature(40.0)
        assert max_curvature(200.0) == EL._K_FLOOR

    def test_short_range_is_left_to_the_band(self):
        s = EL._MIN_GRADE_RANGE_M - 1.0
        v = _veh(1, s, 0.0, pitch_deg=-15.0)
        assert evaluate_vehicle(v, _surface(0.0), s, 0.0).k_req == 0.0


class TestHardCap:
    def test_far_below_is_always_gross(self):
        v = _veh(1, 60.0, -(EL._HARD_CAP_M + 5.0))
        verdict = evaluate_vehicle(v, _surface(-0.3), 60.0, 0.0)
        assert verdict.off_surface and verdict.gross

    def test_non_finite_height_fails_open(self):
        v = _veh(1, 60.0, float("nan"))
        assert not evaluate_vehicle(v, _surface(0.0), 60.0, 0.0).off_surface


class TestPersistence:
    def _step(self, gate, vehicles, surface):
        return gate.step(vehicles, surface, 0.0, 0.0, _EGO_YAW)

    def test_marginal_failure_needs_repeat_frames(self):
        s = 40.0
        band = profile_band(s, 1.5)
        v = _veh(1, s, -(band * 1.2) + EL.BODY_DATUM_FRAC * 1.5)
        gate = ElevationGate()
        surface = _surface(0.0)
        for _ in range(EL.SUPPRESS_CONFIRM_FRAMES - 1):
            assert 1 not in self._step(gate, [v], surface)
        assert 1 in self._step(gate, [v], surface)

    def test_one_good_frame_clears_the_strike_count(self):
        s = 40.0
        band = profile_band(s, 1.5)
        bad = _veh(1, s, -(band * 1.2) + EL.BODY_DATUM_FRAC * 1.5)
        good = _veh(1, s, EL.BODY_DATUM_FRAC * 1.5)
        gate = ElevationGate()
        surface = _surface(0.0)
        self._step(gate, [bad], surface)
        self._step(gate, [good], surface)
        assert 1 not in self._step(gate, [bad], surface)

    def test_gross_failure_suppresses_on_the_first_frame(self):
        v = _veh(1, 30.0, -5.0 + EL.BODY_DATUM_FRAC * 1.5)
        assert 1 in self._step(ElevationGate(), [v], _surface(0.0))

    def test_strikes_do_not_leak_for_departed_ids(self):
        gate = ElevationGate()
        surface = _surface(0.0)
        band = profile_band(40.0, True)
        v = _veh(7, 40.0, -(band * 1.2) + EL.BODY_DATUM_FRAC * 1.5)
        self._step(gate, [v], surface)
        self._step(gate, [], surface)
        assert gate._strikes == {}


class TestEgoElevationTrack:
    def _drive(self, track, grade, curvature, n=120, step=0.5):
        for i in range(n):
            s = i * step
            track.push(0.0, -s, grade * s + 0.5 * curvature * s * s)

    def test_recovers_a_known_vertical_curvature(self):
        track = EgoElevationTrack()
        self._drive(track, -0.05, 0.0006)
        kappa, ok = track.curvature()
        assert ok
        assert abs(kappa - 0.0006) < 1e-4

    def test_short_span_is_unusable(self):
        track = EgoElevationTrack()
        self._drive(track, 0.0, 0.0, n=10, step=0.5)
        assert track.curvature() == (0.0, False)

    def test_teleport_restarts_the_window(self):
        track = EgoElevationTrack()
        self._drive(track, 0.0, 0.0008)
        assert track.curvature()[1]
        track.push(5000.0, 5000.0, 300.0)
        assert track.curvature() == (0.0, False)

    def test_build_surface_uses_pitch_for_grade(self):
        track = EgoElevationTrack()
        self._drive(track, -0.05, 0.0006)
        surface = build_surface(12.0, math.radians(-3.0), track)
        assert surface.ego_y == 12.0
        assert math.isclose(surface.grade, math.tan(math.radians(-3.0)), rel_tol=1e-9)
        assert surface.curvature_ok

    def test_build_surface_without_history_is_tangent_only(self):
        surface = build_surface(0.0, math.radians(2.0), None)
        assert not surface.curvature_ok
        assert surface.predict(50.0) == surface.grade * 50.0

    def test_absurd_ego_pitch_is_treated_as_level(self):
        assert build_surface(0.0, math.radians(80.0), None).grade == 0.0


class TestBehindEgo:
    def test_a_vehicle_behind_ego_uses_the_same_profile(self):
        """s < 0 is the road ego just drove, so the same test applies."""
        grade = 0.05
        v = _veh(1, -60.0, -grade * 60.0 + EL.BODY_DATUM_FRAC * 1.5)
        assert not evaluate_vehicle(v, _surface(grade), -60.0, 0.0).off_surface

    def test_a_vehicle_below_the_road_behind_ego_is_suppressed(self):
        v = _veh(1, -30.0, -6.0 + EL.BODY_DATUM_FRAC * 1.5)
        assert evaluate_vehicle(v, _surface(0.0), -30.0, 0.0).off_surface
