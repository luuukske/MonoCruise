"""Full-authority braking must require the confidence the rest of the law uses.

The TTC overlay used to run on the nearest published lead regardless of tracker
score, so the least certain leads produced the most violent response. The
close-range emergency overlay stays ungated."""
from __future__ import annotations

from core.cruise_control_thread.acc_controller import (
    AdaptiveCruiseController, _LeadSnapshot,
)


def _chain(score: float, dist_m: float, v_lead: float = 0.0, a_lead: float = 0.0):
    lead = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead,
                         a_lead_ms2=a_lead, score=score)
    smooth = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead,
                           a_lead_ms2=a_lead, score=score, conf=_conf(score))
    return [lead], [smooth]


def _conf(score: float) -> float:
    return max(0.0, min(1.0, (score - 1.0) / 4.0))


def _command(score: float, dist_m: float, v_ego: float):
    controller = AdaptiveCruiseController()
    raw, smooth = _chain(score, dist_m)
    return controller._compute_command(raw, smooth, v_ego, 1.0 / 30.0)


def test_confident_lead_still_gets_full_braking_authority():
    accel, emergency = _command(score=6.0, dist_m=15.0, v_ego=20.0)
    assert emergency
    assert accel == AdaptiveCruiseController().config.max_decel_ms2


def test_zero_confidence_lead_does_not_trip_the_ttc_overlay():
    """Score at or below the confidence floor must not command max decel."""
    accel, _ = _command(score=0.5, dist_m=15.0, v_ego=20.0)
    assert accel > AdaptiveCruiseController().config.max_decel_ms2


def test_zero_confidence_lead_never_commands_acceleration():
    """Softening full authority must not become acceleration toward the lead."""
    accel, _ = _command(score=0.5, dist_m=15.0, v_ego=20.0)
    assert accel <= 0.0


def test_confidence_only_softens_braking():
    """The uncertain command must sit between full braking and coasting."""
    confident, _ = _command(score=6.0, dist_m=40.0, v_ego=20.0)
    uncertain, _ = _command(score=0.5, dist_m=40.0, v_ego=20.0)
    assert confident <= uncertain <= 0.0


def test_close_range_emergency_overlay_stays_ungated():
    """Inside the emergency distance the command is unconditional."""
    cfg = AdaptiveCruiseController().config
    accel, emergency = _command(score=0.1, dist_m=cfg.d_emergency_m * 0.5,
                                v_ego=5.0)
    assert emergency
    assert accel == cfg.emergency_decel_ms2


def test_confidence_ramp_bounds_match_the_tracker_contract():
    from core.acc.scoring import IN_PATH_THRESHOLD, SCORE_MAX
    from core.cruise_control_thread.acc_controller import (
        ANT_SCORE_FULL, ANT_SCORE_MIN,
    )

    assert IN_PATH_THRESHOLD < ANT_SCORE_MIN, (
        "leads are published below the confidence floor, so the gap must exist"
    )
    assert ANT_SCORE_FULL <= SCORE_MAX
