"""Standalone AEB scenario report runner. Usage: python -m tests.aeb.report"""

from __future__ import annotations

import importlib
import sys

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.thread import AEBState, _INF
from tests.aeb.harness import evaluate_frame

_SCENARIOS = [
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
    ("tp_fast_perpendicular_crosser", "TP"),
    ("fp_far_behind_high_closure", "FP"),
]


def _run_scenario(mod_name: str, kind: str) -> dict:
    mod = importlib.import_module(f"tests.aeb.scenarios.{mod_name}")
    frames = mod.build()
    expected = mod.EXPECTED

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
        for vid, reasons in result.suppression_reasons.items():
            if reasons:
                suppression_filter = reasons[0].reason

    exp_state = AEBState[expected.get("max_state", "STANDBY")]
    must_suppress = expected.get("must_be_suppressed_by")

    if kind == "FP":
        passed = max_state == AEBState.STANDBY
        if passed and must_suppress is not None:
            passed = suppression_filter == must_suppress
    else:
        passed = max_state >= AEBState.WARN

    return {
        "name": mod_name,
        "kind": kind,
        "passed": passed,
        "max_state": max_state,
        "t_warn": t_first_warn,
        "t_brake": t_first_brake,
        "suppressed_by": suppression_filter,
    }


def main() -> None:
    print("== AEB scenario suite ==")
    results = []
    for mod_name, kind in _SCENARIOS:
        r = _run_scenario(mod_name, kind)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        name_col = f"{kind} {mod_name:<35}"
        if kind == "TP":
            warn_str = f"warn@{r['t_warn']:.2f}s" if r["t_warn"] is not None else "no-warn"
            brake_str = f"brake@{r['t_brake']:.2f}s" if r["t_brake"] is not None else "brake@-"
            print(f"{name_col} {status}  {warn_str}  {brake_str}")
        else:
            supp = r["suppressed_by"] or "unknown"
            state_str = "no-trigger" if r["max_state"] == AEBState.STANDBY else f"TRIGGERED({r['max_state'].name})"
            print(f"{name_col} {status}  {state_str}  suppressed={supp}")

    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} passed")
    if n_pass < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
