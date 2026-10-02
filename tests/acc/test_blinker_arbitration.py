"""Controller-side blinker arbitration (R5-R8) and indicated-lead authority.

Tracker candidacy and the intent model live in `test_blinker_offset.py`.
Corpus clips carry no blinker state, so these are synthetic scenes only."""
from __future__ import annotations

import math

import pytest

from core.cruise_control_thread.acc_controller import (
    AdaptiveCruiseController,
    BLINKER_TTC_FLOOR_S,
    _LeadSnapshot,
)
from core.cruise_control_thread.blinker_arbitration import (
    BLINKER_HYST_S,
    BLINKER_RELEASE_HOLD_S,
    BlinkerArbiter,
    release_fraction,
)


_DT = 1.0 / 30.0


def _arbiter_kw(**over):
    kw = dict(
        b_eff=1.0, committed=True, soft_ok=True, ind_vid=None, now=10.0,
        lead_ttc_s=math.inf, lead_gap_m=60.0, v_ego=22.0,
        lane_vid=1, a_free=1.5, lane_offset_m=2.5,
    )
    kw.update(over)
    return kw


def test_committed_pass_fully_releases_old_lane():
    """Clear intent to move over, realistic gap: the old lead is let go."""
    arb = BlinkerArbiter()
    out = arb.arbitrate(-1.0, 1.5, **_arbiter_kw())
    assert arb.mode == "pass"
    assert out == pytest.approx(1.5)


def test_uncommitted_intent_keeps_lane_policy():
    """Blinking but not yet moving over is stage 1: gap softens, lane holds."""
    arb = BlinkerArbiter()
    out = arb.arbitrate(-1.0, 1.5, **_arbiter_kw(committed=False))
    assert arb.mode == "soften"
    assert out == pytest.approx(-1.0)


def test_unrealistic_gap_holds_the_old_lead():
    """R8: committed or not, a gap this short keeps full lane authority."""
    arb = BlinkerArbiter()
    out = arb.arbitrate(-1.0, 1.5, **_arbiter_kw(lead_gap_m=6.0))
    assert arb.mode == "pass"
    assert out == pytest.approx(-1.0)


def test_release_is_graded_between_the_bounds():
    arb = BlinkerArbiter()
    out = arb.arbitrate(-1.0, 1.5, **_arbiter_kw(lead_gap_m=14.0, v_ego=14.0))
    assert -1.0 < out < 1.5


def test_release_fraction_bounds_and_monotonicity():
    assert release_fraction(math.inf, 60.0, 22.0) == pytest.approx(1.0)
    assert release_fraction(math.inf, 4.0, 22.0) == pytest.approx(0.0)
    assert release_fraction(1.0, 60.0, 22.0) == pytest.approx(0.0)
    grades = [release_fraction(math.inf, g, 14.0) for g in (8.0, 12.0, 16.0, 18.0)]
    assert grades == sorted(grades)


def test_collapse_does_not_hand_the_lane_back():
    """R11 fires before the vacated lead leaves leads[]; that gap was a brake blip."""
    arb = BlinkerArbiter()
    released = arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.0))
    assert arb.mode == "pass"
    assert arb.released_vid == 1
    # b_eff collapses, but the vehicle ego left is still chain[0].
    held = arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.1, b_eff=0.0))
    assert arb.mode == "lane"
    assert held == pytest.approx(released)


def test_release_hold_expires():
    arb = BlinkerArbiter()
    arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.0))
    late = 10.0 + BLINKER_RELEASE_HOLD_S + 0.05
    out = arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=late, b_eff=0.0))
    assert arb.released_vid is None
    assert out == pytest.approx(-1.0)


def test_release_hold_drops_on_a_new_primary():
    """A different vehicle ahead is a new constraint, not the one we left."""
    arb = BlinkerArbiter()
    arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.0))
    out = arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.1, b_eff=0.0, lane_vid=9))
    assert arb.released_vid is None
    assert out == pytest.approx(-1.0)


def test_release_hold_respects_the_distance_floor():
    """Held release is still gated on the gap being realistic."""
    arb = BlinkerArbiter()
    arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.0))
    out = arb.arbitrate(
        -1.0, 1.5, **_arbiter_kw(now=10.1, b_eff=0.0, lead_gap_m=6.0),
    )
    assert out == pytest.approx(-1.0)


def test_aborted_change_never_latches_the_hold():
    """Blink, edge over 0.5 m, think better of it: no release survives it."""
    arb = BlinkerArbiter()
    arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.0, lane_offset_m=0.5))
    assert arb.released_vid is None
    out = arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.1, b_eff=0.0))
    assert out == pytest.approx(-1.0)


def test_merge_drops_the_hold():
    """A tighter indicated lane reasserts a constraint; the hold must yield."""
    arb = BlinkerArbiter()
    arb.arbitrate(-1.0, 1.5, **_arbiter_kw(now=10.0))
    assert arb.released_vid == 1
    arb.arbitrate(1.0, -1.0, **_arbiter_kw(now=11.0, ind_vid=7))
    assert arb.mode == "merge"
    assert arb.released_vid is None


def test_no_blip_across_the_whole_collapse_window():
    """End to end: the command must not step down as ego clears the old lane."""
    ctrl = AdaptiveCruiseController()
    raw = [_LeadSnapshot(vid=1, dist_m=45.0, v_lead_ms=18.0, a_lead_ms2=0.0, score=6.0)]
    smooth = [_LeadSnapshot(
        vid=1, dist_m=45.0, v_lead_ms=18.0, a_lead_ms2=0.0, score=6.0, conf=1.0,
    )]
    cmds = []
    t = 0.0
    for i in range(60):
        ctrl._prev_mono = t
        # Committed pass for 20 frames, then R11 collapses b_eff to 0 while
        # the vacated lead is still published.
        collapsed = i >= 20
        a, _ = ctrl._compute_command(
            raw, smooth, 22.0, _DT,
            b_eff=0.0 if collapsed else 1.0,
            committed=not collapsed,
            lane_offset_m=0.0 if collapsed else 2.5,
        )
        cmds.append(a)
        t += _DT
    # No downward step anywhere in the window: a blip is a step.
    steps = [b - a for a, b in zip(cmds, cmds[1:])]
    assert min(steps) >= -1e-6, f"brake blip of {min(steps):.3f} m/s^2"


def test_mode_hysteresis_expires():
    """The dwell is BLINKER_HYST_S from the last change, not every frame."""
    arb = BlinkerArbiter()
    # Tighter indicated lane: merge.
    arb.arbitrate(1.0, -1.0, **_arbiter_kw(now=10.0, ind_vid=7))
    assert arb.mode == "merge"
    # Neither freer nor tighter now: wants lane, held by the dwell.
    for step in (0.1, 0.2, 0.3):
        arb.arbitrate(-1.0, -1.0, **_arbiter_kw(now=10.0 + step, ind_vid=7))
        assert arb.mode == "merge"
    arb.arbitrate(-1.0, -1.0, **_arbiter_kw(now=10.0 + BLINKER_HYST_S + 0.05, ind_vid=7))
    assert arb.mode == "lane"


def test_stage1_ttc_floor_restores_full_policy():
    """R8: without commitment, a closing lead through the TTC floor hardens."""
    ctrl = AdaptiveCruiseController()
    # Close, closing in-lane lead: TTC well below the floor.
    raw = [_LeadSnapshot(vid=1, dist_m=20.0, v_lead_ms=10.0, a_lead_ms2=0.0, score=6.0)]
    smooth = [_LeadSnapshot(
        vid=1, dist_m=20.0, v_lead_ms=10.0, a_lead_ms2=0.0, score=6.0, conf=1.0,
    )]
    indicated = _LeadSnapshot(
        vid=2, dist_m=40.0, v_lead_ms=25.0, a_lead_ms2=0.0, score=4.0, conf=1.0,
    )
    v_ego = 20.0
    ttc = 20.0 / max(v_ego - 10.0, 0.3)
    assert ttc < BLINKER_TTC_FLOOR_S
    a_full, _ = ctrl._compute_command(raw, smooth, v_ego, _DT)
    ctrl.reset()
    a_blink, _ = ctrl._compute_command(
        raw, smooth, v_ego, _DT,
        indicated_raw=indicated, indicated_smooth=indicated,
        b_eff=1.0, committed=False,
    )
    # Softening must not release: command stays at least as hard as full policy.
    assert a_blink <= a_full + 0.05


def test_committed_pass_reaches_the_command():
    """End to end: committed overtake with room lifts the cap off the old lead."""
    ctrl = AdaptiveCruiseController()
    raw = [_LeadSnapshot(vid=1, dist_m=60.0, v_lead_ms=21.0, a_lead_ms2=0.0, score=6.0)]
    smooth = [_LeadSnapshot(
        vid=1, dist_m=60.0, v_lead_ms=21.0, a_lead_ms2=0.0, score=6.0, conf=1.0,
    )]
    a_lane, _ = ctrl._compute_command(raw, smooth, 22.0, _DT)
    ctrl.reset()
    a_pass, _ = ctrl._compute_command(
        raw, smooth, 22.0, _DT, b_eff=1.0, committed=True,
    )
    assert ctrl._blinker.mode == "pass"
    assert a_pass > a_lane
    assert a_pass == pytest.approx(ctrl.config.no_lead_ceiling_ms2)


def test_committed_pass_does_not_reghost_the_lead_left_behind():
    """The vehicle we drove away from must not come back as a ghost."""
    ctrl = AdaptiveCruiseController()
    raw = [_LeadSnapshot(vid=1, dist_m=60.0, v_lead_ms=21.0, a_lead_ms2=0.0, score=6.0)]
    smooth = [_LeadSnapshot(
        vid=1, dist_m=60.0, v_lead_ms=21.0, a_lead_ms2=0.0, score=6.0, conf=1.0,
    )]
    ctrl._ghost_vid = 1
    ctrl._compute_command(raw, smooth, 22.0, _DT, b_eff=1.0, committed=True)
    assert ctrl._ghost_vid is None


def test_worse_candidate_is_merged_min():
    """R6: tighter indicated lane takes the min of both constraints."""
    ctrl = AdaptiveCruiseController()
    raw = [_LeadSnapshot(vid=1, dist_m=60.0, v_lead_ms=24.0, a_lead_ms2=0.0, score=6.0)]
    smooth = [_LeadSnapshot(
        vid=1, dist_m=60.0, v_lead_ms=24.0, a_lead_ms2=0.0, score=6.0, conf=1.0,
    )]
    indicated = _LeadSnapshot(
        vid=2, dist_m=25.0, v_lead_ms=10.0, a_lead_ms2=-1.0, score=5.0, conf=1.0,
    )
    a_lane, _ = ctrl._compute_command(raw, smooth, 25.0, _DT)
    ctrl.reset()
    a_merge, _ = ctrl._compute_command(
        raw, smooth, 25.0, _DT,
        indicated_raw=indicated, indicated_smooth=indicated,
        b_eff=1.0, committed=False,
    )
    assert a_merge < a_lane - 0.05


def test_indicated_only_path_never_commands_emergency():
    """The indicated lane is not a collision path: comfort bounds its authority.

    A candidate is published only from outside ego's corridor, so routing it
    through the overlays slammed the brakes for a vehicle beside ego."""
    ctrl = AdaptiveCruiseController()
    cfg = ctrl.config
    indicated = _LeadSnapshot(
        vid=2, dist_m=cfg.d_emergency_m * 0.5, v_lead_ms=0.0,
        a_lead_ms2=0.0, score=5.0, conf=1.0,
    )
    accel, emergency = ctrl._compute_command(
        [], [], 15.0, _DT,
        indicated_raw=indicated, indicated_smooth=indicated,
        b_eff=1.0, committed=True,
    )
    assert not emergency
    assert accel == pytest.approx(-cfg.b_comfort_ms2)
    assert ctrl._blinker.mode == "pass"


def test_indicated_lead_never_exceeds_comfort_braking():
    """R6 min-merge must not let the next lane demand more than comfort."""
    ctrl = AdaptiveCruiseController()
    cfg = ctrl.config
    raw = [_LeadSnapshot(vid=1, dist_m=120.0, v_lead_ms=25.0, a_lead_ms2=0.0, score=6.0)]
    smooth = [_LeadSnapshot(
        vid=1, dist_m=120.0, v_lead_ms=25.0, a_lead_ms2=0.0, score=6.0, conf=1.0,
    )]
    # Stopped vehicle in the indicated lane: worst case the gap law can see.
    indicated = _LeadSnapshot(
        vid=2, dist_m=8.0, v_lead_ms=0.0, a_lead_ms2=0.0, score=6.0, conf=1.0,
    )
    accel, emergency = ctrl._compute_command(
        raw, smooth, 25.0, _DT,
        indicated_raw=indicated, indicated_smooth=indicated,
        b_eff=1.0, committed=False,
    )
    assert not emergency
    assert accel >= -cfg.b_comfort_ms2 - 1e-6


def test_indicated_only_path_keeps_arbiter_in_pass():
    """Avoid stale merge hysteresis when leads[] returns after an empty-lane pass."""
    ctrl = AdaptiveCruiseController()
    ctrl._blinker.mode = "merge"
    indicated = _LeadSnapshot(
        vid=2, dist_m=40.0, v_lead_ms=20.0, a_lead_ms2=0.0, score=5.0, conf=1.0,
    )
    ctrl._compute_command(
        [], [], 22.0, _DT,
        indicated_raw=indicated, indicated_smooth=indicated,
        b_eff=1.0, committed=True,
    )
    assert ctrl._blinker.mode == "pass"
