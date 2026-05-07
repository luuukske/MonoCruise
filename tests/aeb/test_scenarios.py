"""Pytest test cases for AEB scenarios."""

from __future__ import annotations

import importlib
import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.thread import AEBState, _INF
from tests.aeb.harness import evaluate_frame

_SCENARIO_MODULES = [
    ("tp_stopped_in_lane", "TP"),
    ("tp_slow_lead", "TP"),
    ("tp_lane_cutter", "TP"),
    ("tp_head_on_in_lane", "TP"),
    ("fp_oncoming_straight", "FP"),
    ("fp_oncoming_gentle_curve", "FP"),
    ("fp_oncoming_sharp_curve", "FP"),
    ("fp_corner_entry_stationary", "FP"),
    ("fp_side_road_uturn", "FP"),
    ("fp_overtaker", "FP"),
    ("fp_co_directional_outer_lane", "FP"),
    ("fp_parked_shoulder", "FP"),
    # New edge-case scenarios
    ("fp_tmp_side_road_right_turn", "FP"),
    ("fp_cross_traffic_completing_turn", "FP"),
    ("fp_adjacent_lateral_jump_tmp", "FP"),
    ("fp_ego_lateral_jump_sp", "FP"),
    ("fp_slow_oncoming_intersection", "FP"),
    ("fp_roadside_inside_curve", "FP"),
    ("fp_oncoming_with_trailer", "FP"),
    ("fp_intersection_waiting_to_turn", "FP"),
    ("fp_oncoming_in_curve_with_trailer", "FP"),
    ("tp_oncoming_overtake_into_lane", "TP"),
    ("tp_high_velocity_cut_off", "TP"),
    ("tp_lead_brake_check", "TP"),
    ("tp_wrong_way_driver", "TP"),
    ("tp_perpendicular_cross_traffic", "TP"),
    ("tp_close_range_lane_cut", "TP"),
    ("fp_far_behind_high_closure", "FP"),
]


def _load(name: str):
    return importlib.import_module(f"tests.aeb.scenarios.{name}")


@pytest.mark.parametrize("mod_name,kind", _SCENARIO_MODULES)
def test_scenario(mod_name: str, kind: str) -> None:
    mod = _load(mod_name)
    frames = mod.build()
    expected = mod.EXPECTED

    assert frames, f"{mod_name}: build() returned no frames"

    max_state = AEBState.STANDBY
    t_first_warn: float | None = None
    t_first_brake: float | None = None
    suppression_filter: str | None = None

    for frame in frames:
        result = evaluate_frame(frame, CAL)
        if result.aeb_state > max_state:
            max_state = result.aeb_state
        if result.aeb_state >= AEBState.WARN and t_first_warn is None:
            t_first_warn = frame.t
        if result.aeb_state >= AEBState.BRAKE and t_first_brake is None:
            t_first_brake = frame.t

        # Track which filter suppressed the target
        for vid, reasons in result.suppression_reasons.items():
            if reasons:
                suppression_filter = reasons[0].reason

    exp_state_name = expected.get("max_state", "STANDBY")
    exp_state = AEBState[exp_state_name]

    if kind == "FP":
        assert max_state == AEBState.STANDBY, (
            f"{mod_name}: expected STANDBY (no trigger) but got {max_state.name}. "
            f"t_warn={t_first_warn}"
        )
        must_suppress = expected.get("must_be_suppressed_by")
        if must_suppress is not None:
            assert suppression_filter == must_suppress, (
                f"{mod_name}: expected suppressed by {must_suppress!r} "
                f"but got {suppression_filter!r}"
            )
    else:
        # TP: should trigger
        assert max_state >= AEBState.WARN, (
            f"{mod_name}: expected at least WARN but got {max_state.name}"
        )
        t_warn_max = expected.get("t_warn_max")
        if t_warn_max is not None and t_first_warn is not None:
            assert t_first_warn <= t_warn_max, (
                f"{mod_name}: warn at {t_first_warn:.2f}s > limit {t_warn_max}s"
            )
